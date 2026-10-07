//! Evaluating constants during translation.
//!
//! A constant expression is one whose value can be computed now, without
//! running the program: literals, `const` bindings whose own values could be
//! computed, arithmetic and comparison over those, structure, variant and
//! array values built from them, `if` and `match` over them, and calls of
//! *constant functions* (functions whose parameters are all `constexpr`)
//! with constant arguments. This module computes
//! such values by interpreting the typed IR directly. `core`'s operations on
//! `frac`, `cyclo` and `phase` are computed with [`crate::exact`] rather than
//! by running their Topiq code ([`super::exact`]).
//!
//! # Trying is the test
//!
//! There is no separate analysis deciding whether an expression is constant.
//! The evaluator simply tries, and if it meets something whose value only
//! exists at run time (an ordinary variable, a loop, an assignment, a
//! reference) it stops and says which subexpression that was. So the error
//! points at the exact part that is not constant, not merely at the
//! declaration that needed a constant.
//!
//! # A constant cannot abort
//!
//! Every operation goes through [`super::arith`], the same definition of
//! integer arithmetic that the compiled program obeys, and every index is
//! checked against its array's length. An operation that would abort at run
//! time is reported during translation instead: a constant that divides by
//! zero is a program that cannot be built, not one that crashes later.
//!
//! # What is deliberately not constant
//!
//! A constant expression may not read a binding or object whose type is not
//! `const`, which rules out state that changes, and without mutable state a loop could
//! never end, so loops, `break`, `continue` and assignment are not constant
//! either. Nor is a reference: it is an address, and addresses only exist
//! once the program runs. The one exception is a string literal, whose
//! characters are part of the program itself. Recursion is constant: a
//! constant function may call itself, up to a depth of 64, and the stack
//! grows as each level needs.

use crate::intern::Interner;
use crate::span::Span;
use crate::tir::{
    BinOp, Block, Callee, Expr, ExprKind, FnId, FnKind, GlobalId, GlobalKind, Intrinsic, LogicalOp, Pat,
    PatKind, Stmt, Ty, UnOp, Unit, Value,
};

use super::arith::{self, Abort};

/// Why an expression has no value during translation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalError {
    /// It depends on something only known when the program runs.
    NotConstant {
        /// The subexpression responsible.
        span: Span,
        /// Why, in a sentence about the reader's program.
        why: String,
    },
    /// Evaluating it would abort.
    Abort {
        /// Which abort.
        abort: Abort,
        /// The operation that would abort.
        span: Span,
    },
    /// Constant-function calls nested too deeply.
    TooDeep {
        /// The call that went one level too far.
        span: Span,
    },
    /// A unit-scope constant whose value depends on its own.
    Cycle {
        /// The object whose value is needed to compute itself.
        global: GlobalId,
        /// Where its value was needed.
        span: Span,
    },
}

/// The deepest constant-function calls may nest.
pub const MAX_DEPTH: u32 = 64;

/// How much stack must be left before a constant-function call runs on a
/// fresh segment, and how large that segment is.
const STACK_RED_ZONE: usize = 256 * 1024;
const STACK_GROWTH: usize = 4 * 1024 * 1024;

/// The most elements `[e; N]` is built with during translation.
const MAX_REPEAT: u64 = 1 << 24;

/// What evaluating a statement or expression did, other than fail.
enum Interrupt {
    /// A `return` unwinding to the enclosing call.
    Return(Value),
    /// A real failure.
    Error(EvalError),
}

impl From<EvalError> for Interrupt {
    fn from(e: EvalError) -> Interrupt {
        Interrupt::Error(e)
    }
}

type Eval<T> = Result<T, Interrupt>;

/// The bindings of one function activation.
#[derive(Clone, Debug)]
pub struct Frame {
    /// The function whose bindings these are.
    pub func: Option<FnId>,
    /// Each binding's value, where known.
    pub locals: Vec<Option<Value>>,
}

impl Frame {
    /// A frame with no bindings.
    /// A binding it makes.
    pub fn empty() -> Frame {
        Frame {
            func: None,
            locals: Vec::new(),
        }
    }

    /// Gives the binding `l` the value `v`.
    pub fn set(&mut self, l: crate::tir::LocalId, v: Option<Value>) {
        if self.locals.len() <= l.index() {
            self.locals.resize(l.index() + 1, None);
        }
        self.locals[l.index()] = v;
    }

    /// A frame for `func`.
    pub fn for_fn(unit: &Unit, func: FnId) -> Frame {
        Frame {
            func: Some(func),
            locals: vec![None; unit.func(func).locals.len()],
        }
    }
}

#[derive(Clone, Debug)]
enum GlobalState {
    Pending,
    InProgress,
    Done(Value),
    Failed,
}

/// Evaluates constants in one unit.
pub struct Evaluator<'a> {
    unit: &'a Unit,
    interner: &'a Interner,
    globals: Vec<GlobalState>,
    depth: u32,
}

impl<'a> Evaluator<'a> {
    /// An evaluator for `unit`.
    pub fn new(unit: &'a Unit, interner: &'a Interner) -> Evaluator<'a> {
        Evaluator {
            unit,
            interner,
            globals: vec![GlobalState::Pending; unit.globals.len()],
            depth: 0,
        }
    }

    fn name(&self, sym: crate::intern::Symbol) -> &'a str {
        self.interner.resolve(sym)
    }

    /// The initial value of a unit-scope, persistent or imported object.
    ///
    /// # Errors
    ///
    /// If its initialiser is not constant, would abort, or depends on the
    /// object itself. A failure is remembered, so an object read by several
    /// others is reported once.
    pub fn global(&mut self, id: GlobalId, needed_at: Span) -> Result<Value, EvalError> {
        let g = self.unit.global(id);
        if let Some(v) = &g.value {
            return Ok(v.clone());
        }
        match &self.globals[id.index()] {
            GlobalState::Done(v) => return Ok(v.clone()),
            GlobalState::InProgress => {
                return Err(EvalError::Cycle {
                    global: id,
                    span: needed_at,
                });
            }
            GlobalState::Failed => {
                // already reported; stand in with something harmless
                return Ok(placeholder(g.ty));
            }
            GlobalState::Pending => {}
        }
        self.globals[id.index()] = GlobalState::InProgress;
        let Some(init) = &g.init else {
            self.globals[id.index()] = GlobalState::Failed;
            return Ok(placeholder(g.ty));
        };
        let mut frame = match g.kind {
            GlobalKind::Persist(f) => Frame::for_fn(self.unit, f),
            GlobalKind::Unit | GlobalKind::Imported(_) => Frame::empty(),
        };
        match self.top(init, &mut frame) {
            Ok(v) => {
                self.globals[id.index()] = GlobalState::Done(v.clone());
                Ok(v)
            }
            Err(e) => {
                self.globals[id.index()] = GlobalState::Failed;
                Err(e)
            }
        }
    }

    /// Evaluates an expression outside any call, as a constant context does.
    ///
    /// # Errors
    ///
    /// As for [`EvalError`].
    pub fn top(&mut self, e: &Expr, frame: &mut Frame) -> Result<Value, EvalError> {
        match self.expr(e, frame) {
            Ok(v) => Ok(v),
            Err(Interrupt::Error(err)) => Err(err),
            Err(Interrupt::Return(_)) => Err(EvalError::NotConstant {
                span: e.span,
                why: "a `return` leaves the function the program is running, which a \
                      constant cannot do"
                    .to_owned(),
            }),
        }
    }

    /// Calls a function with argument values.
    ///
    /// # Errors
    ///
    /// If its body is not constant, would abort, or recurses too deeply.
    pub fn call(&mut self, callee: FnId, args: &[Value], span: Span) -> Result<Value, EvalError> {
        let f = self.unit.func(callee);
        if f.kind == FnKind::Runtime && !f.params.is_empty() {
            return Err(EvalError::NotConstant {
                span,
                why: format!(
                    "`{}` has parameters that are only known when the program runs, so a \
                     call to it is not a constant",
                    self.name(f.name)
                ),
            });
        }
        if self.depth >= MAX_DEPTH {
            return Err(EvalError::TooDeep { span });
        }
        self.depth += 1;
        let mut frame = Frame::for_fn(self.unit, callee);
        for (&p, v) in f.params.iter().zip(args) {
            frame.set(p, Some(v.clone()));
        }
        // each level of a recursion within the limit must evaluate, however
        // much stack the body's expressions take, so the stack grows here
        let body = stacker::maybe_grow(STACK_RED_ZONE, STACK_GROWTH, || self.block(&f.body, &mut frame));
        let result = match body {
            Ok(v) | Err(Interrupt::Return(v)) => Ok(v),
            Err(Interrupt::Error(e)) => Err(e),
        };
        self.depth -= 1;
        result
    }

    fn block(&mut self, b: &Block, frame: &mut Frame) -> Eval<Value> {
        for s in &b.stmts {
            match s {
                Stmt::LetPat { pat, init } => {
                    let v = self.expr(init, frame)?;
                    // the pattern matches every value of the type, so this
                    // only fills in the names it binds
                    matches(pat, &v, frame);
                }
                Stmt::Let { local, init } => {
                    let v = init.as_ref().map(|e| self.expr(e, frame)).transpose()?;
                    frame.set(*local, v);
                }
                Stmt::Expr(e) => {
                    self.expr(e, frame)?;
                }
            }
        }
        match &b.value {
            Some(v) => self.expr(v, frame),
            None => Ok(Value::Void),
        }
    }

    fn not_constant<T>(span: Span, why: impl Into<String>) -> Eval<T> {
        Err(Interrupt::Error(EvalError::NotConstant {
            span,
            why: why.into(),
        }))
    }

    fn abort<T>(abort: Abort, span: Span) -> Eval<T> {
        Err(Interrupt::Error(EvalError::Abort { abort, span }))
    }

    fn expr(&mut self, e: &Expr, frame: &mut Frame) -> Eval<Value> {
        let span = e.span;
        match &e.kind {
            ExprKind::Const(v) => Ok(v.clone()),
            ExprKind::Local(id) => {
                // a binding a unit-scope condition's own pattern made
                if let Some(Some(v)) = frame.locals.get(id.index()) {
                    return Ok(v.clone());
                }
                let Some(func) = frame.func else {
                    return Self::not_constant(span, "no binding is in scope here");
                };
                let l = self.unit.func(func).local(*id);
                // a binding given its value in this evaluation (a parameter, a
                // `let` or a pattern in a constant function) or a `const` one
                // folded already has it; any other gets one when the program runs
                match frame.locals.get(id.index()).unwrap_or(&None) {
                    Some(v) => Ok(v.clone()),
                    None if !l.constant => Self::not_constant(
                        span,
                        format!(
                            "`{}` is not a constant: its type is not `const`, so it may be \
                             assigned, and its value is only known when the program runs",
                            self.name(l.name)
                        ),
                    ),
                    None => Self::not_constant(
                        span,
                        format!(
                            "`{}` is `const`, but its value is only known when the program runs",
                            self.name(l.name)
                        ),
                    ),
                }
            }
            ExprKind::Global(id) => {
                let g = self.unit.global(*id);
                if !g.constant {
                    return Self::not_constant(
                        span,
                        format!(
                            "`{}` is not a constant: its type is not `const`, so it is an \
                             object whose value may change while the program runs",
                            self.name(g.name)
                        ),
                    );
                }
                Ok(self.global(*id, span)?)
            }
            ExprKind::FnRef(_) | ExprKind::ThunkRef(_) | ExprKind::Closure { .. } | ExprKind::FnAsClosure(_) => Self::not_constant(
                span,
                "a function value is an address in the running program, which translation does \
                 not have",
            ),
            // a growable array made from constant elements is a constant: its
            // buffer is placed in the program, and each use of it by value
            // copies it
            ExprKind::Grow(array) => self.expr(array, frame),
            ExprKind::Growable { .. } => Self::not_constant(
                span,
                "changing a growable array needs the program to be running",
            ),
            ExprKind::IndirectCall { .. } => Self::not_constant(
                span,
                "which function this calls is only known when the program runs",
            ),
            ExprKind::Call { callee, args } => {
                // an exact scalar's operation is computed here, exactly as
                // its code computes it; its operands, passed by reference,
                // are read as values
                if let Some(op) = super::exact::exact_op(self.unit, self.interner, *callee) {
                    let values = args
                        .iter()
                        .map(|a| {
                            // an operand lent as `*const`, which a `*T` was
                            // converted to, is read the same way
                            let a = match &a.kind {
                                ExprKind::Coerce(inner) => inner,
                                _ => a,
                            };
                            match &a.kind {
                                ExprKind::Ref(inner) => self.expr(inner, frame),
                                _ => self.expr(a, frame),
                            }
                        })
                        .collect::<Eval<Vec<_>>>()?;
                    match super::exact::eval_exact(&op, &values) {
                        Ok(v) => return Ok(v),
                        Err(super::exact::ExactError::Abort(a)) => return Self::abort(a, span),
                        Err(super::exact::ExactError::Unsupported) => {}
                    }
                }
                let Callee::Fn(callee) = callee else {
                    return Self::not_constant(
                        span,
                        "this function is compiled as part of another unit, so it only runs \
                         when the program does",
                    );
                };
                let values = self.exprs(args, frame)?;
                Ok(self.call(*callee, &values, span)?)
            }
            ExprKind::Intrinsic { which, args } => self.intrinsic(*which, args, span, frame),
            ExprKind::Unary { op, operand } => {
                let v = self.expr(operand, frame)?;
                match (op, v) {
                    (UnOp::Neg, Value::Int(x, t)) => match arith::neg(t, x) {
                        Ok(r) => Ok(Value::Int(r, t)),
                        Err(a) => Self::abort(a, span),
                    },
                    (UnOp::Neg, Value::Float(x, t)) => Ok(Value::Float(arith::float::neg(t, x), t)),
                    (UnOp::Not, Value::Int(x, t)) => Ok(Value::Int(arith::not(t, x), t)),
                    (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
                    _ => unreachable!("the checker admits no other operand"),
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.expr(lhs, frame)?;
                let r = self.expr(rhs, frame)?;
                binary(*op, l, r).or_else(|a| Self::abort(a, span))
            }
            ExprKind::Logical { op, lhs, rhs } => {
                let l = self.expr(lhs, frame)?.as_bool().expect("checked as bool");
                let decided = match op {
                    LogicalOp::And => !l,
                    LogicalOp::Or => l,
                };
                if decided {
                    return Ok(Value::Bool(l));
                }
                self.expr(rhs, frame)
            }
            ExprKind::Cast { expr, to } => {
                let v = self.expr(expr, frame)?;
                cast(&v, *to).or_else(|a| Self::abort(a, span))
            }
            ExprKind::Field { base, field } => match self.expr(base, frame)? {
                Value::Struct(mut fields) => Ok(fields.swap_remove(*field as usize)),
                other => unreachable!("a field of {other:?}"),
            },
            ExprKind::Index { base, index } => {
                let b = self.expr(base, frame)?;
                let i = self.expr(index, frame)?.as_int().expect("an index is a usize");
                match b {
                    Value::Array(mut items) => match usize::try_from(i).ok().filter(|&i| i < items.len()) {
                        Some(i) => Ok(items.swap_remove(i)),
                        None => Self::abort(Abort::Index, span),
                    },
                    Value::Str(s) => match usize::try_from(i).ok().and_then(|i| self.name(s).chars().nth(i)) {
                        Some(c) => Ok(Value::Char(c)),
                        None => Self::abort(Abort::Index, span),
                    },
                    _ => Self::not_constant(
                        span,
                        "the elements of this slice are only known when the program runs",
                    ),
                }
            }
            // `*&x` is `x` again, so it is as constant as `x` is; no address
            // is needed to see that
            ExprKind::Deref(r) => match &r.kind {
                ExprKind::Ref(x) => self.expr(x, frame),
                // a pattern binds a reference to a part of a value, the part
                // then copied from it; the binding holds the part itself
                ExprKind::Local(id) if frame.locals.get(id.index()).is_some_and(Option::is_some) => self.expr(r, frame),
                ExprKind::Coerce(c) if matches!(c.kind, ExprKind::Ref(_)) => {
                    let ExprKind::Ref(x) = &c.kind else { unreachable!() };
                    self.expr(x, frame)
                }
                _ => Self::not_constant(
                    span,
                    "reading through a reference reads memory, which only exists when the program runs",
                ),
            },
            ExprKind::Ref(_) => Self::not_constant(
                span,
                "a reference is an address, and addresses only exist when the program runs",
            ),
            ExprKind::StructLit { fields } => {
                let n = match e.ty {
                    Ty::Adt(id) => self.unit.types.adt(id).fields().len(),
                    _ => fields.len(),
                };
                Ok(Value::Struct(self.members(n, fields, frame)?))
            }
            ExprKind::Variant { variant, fields } => {
                let n = match e.ty {
                    Ty::Adt(id) => self.unit.types.adt(id).variants()[*variant as usize].fields.len(),
                    _ => fields.len(),
                };
                Ok(Value::Enum {
                    variant: *variant,
                    fields: self.members(n, fields, frame)?,
                })
            }
            ExprKind::ArrayLit(items) => Ok(Value::Array(self.exprs(items, frame)?)),
            ExprKind::ArrayRepeat { elem, len } => {
                let v = self.expr(elem, frame)?;
                if *len > MAX_REPEAT {
                    return Self::not_constant(
                        span,
                        format!("an array of {len} elements is too large to build during translation"),
                    );
                }
                Ok(Value::Array(vec![v; *len as usize]))
            }
            ExprKind::Coerce(x) => self.expr(x, frame),
            ExprKind::Block(b) => self.block(b, frame),
            ExprKind::If { cond, then, els } => {
                if self.expr(cond, frame)?.as_bool().expect("checked as bool") {
                    self.block(then, frame)
                } else {
                    match els {
                        Some(e) => self.expr(e, frame),
                        None => Ok(Value::Void),
                    }
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                let v = self.expr(scrutinee, frame)?;
                for arm in arms {
                    if matches(&arm.pat, &v, frame) {
                        return self.expr(&arm.body, frame);
                    }
                }
                Self::not_constant(span, "no arm of this `match` accepts the value")
            }
            ExprKind::Return(v) => {
                let v = match v {
                    Some(v) => self.expr(v, frame)?,
                    None => Value::Void,
                };
                Err(Interrupt::Return(v))
            }
            ExprKind::Assign { .. } => Self::not_constant(
                span,
                "an assignment changes a variable, and a constant cannot depend on anything \
                 that changes",
            ),
            ExprKind::Loop { .. }
            | ExprKind::While { .. }
            | ExprKind::ForRange { .. }
            | ExprKind::Break { .. }
            | ExprKind::Continue { .. } => Self::not_constant(
                span,
                "a loop is not constant: without variables that change, it could never \
                 finish; use recursion instead",
            ),
            ExprKind::Quantum { op, .. } => Self::not_constant(
                span,
                format!(
                    "`{}` acts on quantum state, which exists only while a circuit runs",
                    op.name()
                ),
            ),
        }
    }

    /// The values of `es`.
    fn exprs(&mut self, es: &[Expr], frame: &mut Frame) -> Eval<Vec<Value>> {
        es.iter().map(|e| self.expr(e, frame)).collect()
    }

    /// Evaluates a structure literal's or variant's fields in the order
    /// written, placing each in its declared position.
    fn members(&mut self, n: usize, fields: &[(u32, Expr)], frame: &mut Frame) -> Eval<Vec<Value>> {
        let mut out = vec![Value::Void; n];
        for (i, f) in fields {
            out[*i as usize] = self.expr(f, frame)?;
        }
        Ok(out)
    }

    fn intrinsic(&mut self, which: Intrinsic, args: &[Expr], span: Span, frame: &mut Frame) -> Eval<Value> {
        match which {
            Intrinsic::Len => match self.expr(&args[0], frame)? {
                Value::Array(items) => Ok(Value::Int(items.len() as i128, crate::tir::IntTy::USIZE)),
                Value::Str(s) => Ok(Value::Int(
                    self.name(s).chars().count() as i128,
                    crate::tir::IntTy::USIZE,
                )),
                _ => Self::not_constant(span, "the length of this slice is only known when the program runs"),
            },
            Intrinsic::Panic => {
                // the message is still evaluated, as the call would
                self.expr(&args[0], frame)?;
                Self::abort(Abort::Panic, span)
            }
            Intrinsic::Abort(n) => {
                self.exprs(args, frame)?;
                Self::abort(Abort::Called(n), span)
            }
            Intrinsic::Overflowing(_)
            | Intrinsic::OverflowingNeg
            | Intrinsic::Saturating(_)
            | Intrinsic::Truncate
            | Intrinsic::SaturatingAs => Self::not_constant(
                span,
                "the library's total arithmetic runs when the program does",
            ),
            Intrinsic::Print | Intrinsic::Eprint => Self::not_constant(
                span,
                "printing is something the program does when it runs, not a value",
            ),
            Intrinsic::Exit => Self::not_constant(
                span,
                "`exit` ends the program when it runs, which a constant cannot do",
            ),
            Intrinsic::Clone => Self::not_constant(
                span,
                "`clone` follows a reference, which only the running program has",
            ),
            Intrinsic::Exchange => Self::not_constant(
                span,
                "`swap` changes what two references refer to, which a constant cannot do",
            ),
            Intrinsic::TypeInfo(_)
            | Intrinsic::TypeInfoOf
            | Intrinsic::Downcast(_)
            | Intrinsic::Invoke(_)
            | Intrinsic::DynOf(_)
            | Intrinsic::DynAs(_)
            | Intrinsic::DynTake
            | Intrinsic::DynView
            | Intrinsic::FieldAt
            | Intrinsic::DynCall
            | Intrinsic::DynMake
            | Intrinsic::DynStore
            | Intrinsic::IsNull
            | Intrinsic::CircuitOf
            | Intrinsic::CircuitDoc
            | Intrinsic::RawParts
            | Intrinsic::FromRawParts
            | Intrinsic::GrowZeroed
            | Intrinsic::FloatDigits
            | Intrinsic::ParseFloat
            | Intrinsic::Sin
            | Intrinsic::Cos => Self::not_constant(
                span,
                "run-time type information is an address in the running program, which translation \
                 does not have",
            ),
            Intrinsic::FileOpen
            | Intrinsic::FileSize
            | Intrinsic::FileRead
            | Intrinsic::FileWrite
            | Intrinsic::FileClose => Self::not_constant(
                span,
                "files are read and written when the program runs; a document wanted as a constant \
                 is embedded with `@embed`",
            ),
            Intrinsic::ModuleOpen
            | Intrinsic::ModuleFind
            | Intrinsic::ModuleClose
            | Intrinsic::CallVoid
            | Intrinsic::HostDescriptor => Self::not_constant(
                span,
                "a module is loaded into the running program, and its code and addresses exist \
                 only then",
            ),
            Intrinsic::StdHandle | Intrinsic::IsConsole | Intrinsic::ConsoleRead => Self::not_constant(
                span,
                "the standard streams belong to the running program, and what is read from them \
                 can differ every run",
            ),
            Intrinsic::Clock => Self::not_constant(
                span,
                "the clock is read when the program runs, and a constant must be the same every run",
            ),
            Intrinsic::Derived(_) => Self::not_constant(
                span,
                "a judgment is derived once the unit's circuits are generated, after its constants are evaluated",
            ),
            Intrinsic::Gate(_) | Intrinsic::Apply | Intrinsic::Node | Intrinsic::Adjoint | Intrinsic::Controlled | Intrinsic::Then => Self::not_constant(
                span,
                "a gate acts on qubits, which exist only while a circuit runs",
            ),
        }
    }
}

/// Whether `v` matches `p`, binding what the pattern binds.
fn matches(p: &Pat, v: &Value, frame: &mut Frame) -> bool {
    match (&p.kind, v) {
        (PatKind::Wild, _) => true,
        // a reference to a part: a match through a reference, which is never
        // constant, or a part copied from where it is, which holds it
        (PatKind::Bind(l) | PatKind::BindRef(l), _) => {
            frame.set(*l, Some(v.clone()));
            true
        }
        (PatKind::Const(c), _) => c == v,
        (PatKind::Struct { fields }, Value::Struct(vs)) => fields.iter().all(|(i, f)| matches(f, &vs[*i as usize], frame)),
        (
            PatKind::Variant { variant, fields },
            Value::Enum {
                variant: w,
                fields: vs,
            },
        ) => variant == w && fields.iter().all(|(i, f)| matches(f, &vs[*i as usize], frame)),
        _ => false,
    }
}

/// `v as to`, exactly as the compiled program converts.
///
/// # Errors
///
/// The abort the conversion would raise.
///
/// # Panics
///
/// For a conversion the checker does not admit.
pub fn cast(v: &Value, to: Ty) -> Result<Value, Abort> {
    match (v, to) {
        (Value::Int(x, _), Ty::Int(t)) => arith::cast(t, *x).map(|r| Value::Int(r, t)),
        (Value::Char(c), Ty::Int(t)) => arith::cast(t, i128::from(u32::from(*c))).map(|r| Value::Int(r, t)),
        (Value::Int(x, _), Ty::Char) => arith::int_to_char(*x).map(Value::Char),
        (Value::Char(c), Ty::Char) => Ok(Value::Char(*c)),
        (Value::Int(x, _), Ty::Float(t)) => Ok(Value::Float(arith::int_to_float(t, *x), t)),
        (Value::Float(x, _), Ty::Float(t)) => Ok(Value::Float(arith::float_to_float(t, *x), t)),
        (Value::Float(x, _), Ty::Int(t)) => arith::float_to_int(t, *x).map(|r| Value::Int(r, t)),
        other => unreachable!("the checker does not convert {other:?}"),
    }
}

/// `l op r`, exactly as the compiled program computes it.
///
/// # Errors
///
/// The abort the operation would raise.
pub fn binary(op: BinOp, l: Value, r: Value) -> Result<Value, Abort> {
    let compared = match (&l, &r) {
        (Value::Int(a, _), Value::Int(b, _)) => compare(op, a, b),
        (Value::Float(a, _), Value::Float(b, _)) => compare(op, a, b),
        (Value::Bool(a), Value::Bool(b)) => compare(op, a, b),
        (Value::Char(a), Value::Char(b)) => compare(op, a, b),
        _ => None,
    };
    if let Some(b) = compared {
        return Ok(Value::Bool(b));
    }
    Ok(match (l, r) {
        (Value::Int(a, t), Value::Int(b, _)) => match op {
            BinOp::Add => Value::Int(arith::add(t, a, b)?, t),
            BinOp::Sub => Value::Int(arith::sub(t, a, b)?, t),
            BinOp::Mul => Value::Int(arith::mul(t, a, b)?, t),
            BinOp::Div => Value::Int(arith::div(t, a, b)?, t),
            BinOp::Rem => Value::Int(arith::rem(t, a, b)?, t),
            BinOp::Shl => Value::Int(arith::shl(t, a, b)?, t),
            BinOp::Shr => Value::Int(arith::shr(t, a, b)?, t),
            BinOp::BitAnd => Value::Int(arith::bit_and(a, b), t),
            BinOp::BitOr => Value::Int(arith::bit_or(a, b), t),
            BinOp::BitXor => Value::Int(arith::bit_xor(a, b), t),
            _ => unreachable!("a comparison, done above"),
        },
        (Value::Float(a, t), Value::Float(b, _)) => match op {
            BinOp::Add => Value::Float(arith::float::add(t, a, b), t),
            BinOp::Sub => Value::Float(arith::float::sub(t, a, b), t),
            BinOp::Mul => Value::Float(arith::float::mul(t, a, b), t),
            BinOp::Div => Value::Float(arith::float::div(t, a, b), t),
            BinOp::Rem => Value::Float(arith::float::rem(t, a, b), t),
            _ => unreachable!("the checker admits no other operator on floating values"),
        },
        (Value::Bool(a), Value::Bool(b)) => Value::Bool(match op {
            BinOp::BitAnd => a & b,
            BinOp::BitOr => a | b,
            BinOp::BitXor => a ^ b,
            _ => unreachable!("the checker admits no other operator on bools"),
        }),
        _ => unreachable!("the checker admits no other operands"),
    })
}

/// `a op b` for a comparison `op`; `None` for any other operator. Every
/// comparison with a floating NaN is false, `!=` excepted.
fn compare<T: PartialOrd>(op: BinOp, a: T, b: T) -> Option<bool> {
    Some(match op {
        BinOp::Eq => a == b,
        BinOp::Ne => a != b,
        BinOp::Lt => a < b,
        BinOp::Le => a <= b,
        BinOp::Gt => a > b,
        BinOp::Ge => a >= b,
        _ => return None,
    })
}

/// A value of the right type to stand in for one whose evaluation already
/// failed and was reported.
fn placeholder(t: Ty) -> Value {
    match t {
        Ty::Int(i) => Value::Int(0, i),
        Ty::Float(f) => Value::Float(0.0, f),
        Ty::Bool => Value::Bool(false),
        Ty::Char => Value::Char('\0'),
        _ => Value::Void,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::Code;
    use crate::sema::testing::analyzed;
    use crate::tir::IntTy;

    /// The folded value of the unit's first object.
    fn first_global(src: &str) -> Value {
        let (u, d) = analyzed(src);
        assert!(d.iter().all(|d| !d.is_error()), "{:?}", d.iter().map(|d| d.to_string()).collect::<Vec<_>>());
        u.globals[0].value.clone().expect("evaluated")
    }

    fn reported(src: &str) -> Vec<Code> {
        analyzed(src).1.iter().map(|d| d.code).collect()
    }

    #[test]
    fn arithmetic_on_constants_is_folded() {
        assert_eq!(first_global("let X: const i32 = 6 * 7;"), Value::Int(42, IntTy::I32));
    }

    #[test]
    fn constants_may_refer_to_later_constants() {
        assert_eq!(
            first_global("let A: const u8 = B + 1;\nlet B: const u8 = 2;"),
            Value::Int(3, IntTy::U8)
        );
    }

    #[test]
    fn a_constant_function_is_evaluated_at_its_call() {
        let src = "fn mask(n: constexpr u32) -> constexpr u64 { (1u64 << n) - 1 }\n\
                   let M: const u64 = mask(12);";
        assert_eq!(first_global(src), Value::Int(4095, IntTy::U64));
    }

    #[test]
    fn a_constant_function_may_recurse() {
        let src = "fn fact(n: constexpr u64) -> constexpr u64 { if n == 0 { 1 } else { n * fact(n - 1) } }\n\
                   let F: const u64 = fact(20);";
        assert_eq!(first_global(src), Value::Int(2_432_902_008_176_640_000, IntTy::U64));
    }

    #[test]
    fn short_circuiting_skips_what_would_abort() {
        assert_eq!(first_global("let X: const bool = false && (1 / 0 == 1);"), Value::Bool(false));
    }

    #[test]
    fn a_cast_that_does_not_fit_aborts() {
        assert!(reported("let X: const u8 = 300 as u8;").contains(&Code::Ec04));
    }

    #[test]
    fn aggregates_are_built_and_taken_apart() {
        let src = "struct P { x: i32, y: i32 }\n\
                   let O: const P = P { y: 2, x: 1 };\n\
                   let X: const i32 = O.x * 10 + O.y;";
        let (u, _) = analyzed(src);
        assert_eq!(
            u.globals[0].value,
            Some(Value::Struct(vec![Value::Int(1, IntTy::I32), Value::Int(2, IntTy::I32)]))
        );
        assert_eq!(u.globals[1].value, Some(Value::Int(12, IntTy::I32)));
    }

    #[test]
    fn arrays_index_and_measure_themselves() {
        let (u, _) = analyzed(
            "let T: const [u8; 4] = [1, 2, 3, 5];\n\
             let X: const u8 = T[3];\n\
             let N: const usize = \"hello\".len();\n\
             let C: const char = \"hello\"[1];",
        );
        assert_eq!(u.globals[1].value, Some(Value::Int(5, IntTy::U8)));
        assert_eq!(u.globals[2].value, Some(Value::Int(5, IntTy::USIZE)));
        assert_eq!(u.globals[3].value, Some(Value::Char('e')));
    }

    #[test]
    fn a_match_is_evaluated() {
        let src = "enum S { A, B(i32) }\n\
                   fn val(s: constexpr S) -> constexpr i32 { match s { S::A => 0, S::B(n) => n } }\n\
                   let X: const i32 = val(S::B(7));";
        let (u, d) = analyzed(src);
        assert!(d.is_empty(), "{d:?}");
        assert_eq!(u.globals[0].value, Some(Value::Int(7, IntTy::I32)));
    }

    #[test]
    fn a_constant_index_out_of_bounds_aborts_with_ra06() {
        let (_, d) = analyzed("let T: const [u8; 2] = [1, 2];\nfn f(i: constexpr usize) -> constexpr u8 { T[i] }\nlet X: const u8 = f(2);");
        assert!(d.iter().any(|d| d.code == Code::Ec04 && d.message.contains("RA06")), "{d:?}");
    }

    #[test]
    fn a_panic_in_a_constant_is_ec04_with_ra10() {
        let (_, d) = analyzed("fn f(n: constexpr i32) -> constexpr i32 { if n > 0 { n } else { panic(\"negative\") } }\nlet X: const i32 = f(-1);");
        assert!(d.iter().any(|d| d.code == Code::Ec04 && d.message.contains("RA10")), "{d:?}");
    }

    #[test]
    fn a_reference_is_not_constant() {
        let src = "fn f(n: constexpr i32) -> constexpr i32 { let r = &n; n }\nlet X: const i32 = f(1);";
        assert_eq!(reported(src), [Code::Ec01]);
    }
}
