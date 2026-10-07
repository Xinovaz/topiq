//! The two kinds of classical value in a quantum unit, and where each may go.
//!
//! Inside a quantum unit, every classical value is either
//!
//! - **generation-time:** known while the circuit is generated (constants,
//!   literals, loop indices, the operator's classical parameters), which
//!   shapes the circuit and is erased from it; or
//! - an **outcome:** the result of a `measure`, or anything computed from
//!   one, which exists only while the circuit runs.
//!
//! A circuit's structure is fixed before it runs, so an outcome cannot shape
//! it: it may not bound a loop, index data fixed at generation, be passed to a
//! function (which is generated for fixed arguments), be captured by a
//! closure, or be stored in a unit-scope object. Each of these is `EQ06`. An
//! outcome may control later operations (an `if` or `match` on it), be
//! combined with other values, flow into the operator's result, and be
//! `lift`ed, which makes it generation-time at the cost of a dynamic circuit.
//!
//! A value computed under an `if` or `match` that tests an outcome depends on
//! that outcome, so a binding assigned there becomes an outcome too.
//!
//! An array of outcomes is an outcome, but how long it is is not, unless it
//! grew or was assigned under a test of an outcome, or was made from such an
//! array: what `measure` gives has as many elements as the register it
//! measured, and an array pushed onto once a round has as many as there were
//! rounds. So the length of what `measure` gives may bound a loop.
//!
//! The walk is flow-insensitive within a binding's life: a binding that ever
//! holds an outcome is treated as one throughout. A loop's body is walked
//! twice, so that an outcome assigned late in it is known at its start.

use std::collections::HashSet;

use crate::diag::{Code, Diagnostic};
use crate::span::Span;
use crate::tir::{Block, Expr, ExprKind, Intrinsic, LocalId, Pat, PatKind, QuantumOp, Stmt, Unit};

/// Checks every function of a quantum unit.
pub fn check(unit: &Unit, diags: &mut Vec<Diagnostic>) {
    for f in &unit.fns {
        let mut k = Kinds {
            unit,
            func: f,
            diags,
            outcome: HashSet::new(),
            shaped: HashSet::new(),
            control: 0,
            reported: HashSet::new(),
        };
        k.block(&f.body);
    }
}

/// Where an outcome value went that it may not.
#[derive(Clone, Copy)]
enum Misuse {
    Bound,
    Index,
    Argument,
    Object,
    Capture,
}

struct Kinds<'a> {
    unit: &'a Unit,
    func: &'a crate::tir::Fn,
    diags: &'a mut Vec<Diagnostic>,
    /// The bindings that may hold an outcome.
    outcome: HashSet<LocalId>,
    /// The bindings whose shape (how long an array is) may depend on an
    /// outcome.
    shaped: HashSet<LocalId>,
    /// How many enclosing `if`s and `match`es test an outcome.
    control: u32,
    reported: HashSet<(u32, u32)>,
}

impl Kinds<'_> {
    fn misuse(&mut self, span: Span, how: Misuse) {
        if !self.reported.insert((span.start, span.end)) {
            return;
        }
        let what = match how {
            Misuse::Bound => "a measured value cannot bound a loop",
            Misuse::Index => "a measured value cannot index data fixed when the circuit is generated",
            Misuse::Argument => "a measured value cannot be passed to a function",
            Misuse::Object => "a measured value cannot be stored in a unit-scope object",
            Misuse::Capture => "a measured value cannot be captured by a closure",
        };
        self.diags.push(
            Diagnostic::new(Code::Eq06)
                .with_message(what)
                .at(span)
                .with_note(
                    "a measured value exists only while the circuit runs, after the circuit's \
                     structure is fixed: it may control later operations, be combined with other \
                     values and become the operator's result, but not shape the circuit",
                )
                .with_help(
                    "test it with `if` or `match` to control what follows, return it, or `lift` \
                     it where the target runs dynamic circuits",
                ),
        );
    }

    /// Records that a binding may hold an outcome. A qubit is a qubit still,
    /// whatever was measured to decide it, so only a classical binding is.
    fn mark(&mut self, l: LocalId) {
        if !self.unit.types.is_quantum(self.func.local(l).ty) {
            self.outcome.insert(l);
        }
    }

    /// Records that a binding bound by `p` may hold an outcome, of a shape
    /// that may depend on one.
    fn mark_pattern(&mut self, p: &Pat) {
        match &p.kind {
            PatKind::Bind(l) | PatKind::BindRef(l) => {
                self.mark(*l);
                self.shaped.insert(*l);
            }
            PatKind::Struct { fields } | PatKind::Variant { fields, .. } => {
                for (_, f) in fields {
                    self.mark_pattern(f);
                }
            }
            PatKind::Wild | PatKind::Const(_) => {}
        }
    }

    /// Walks a block, returning whether its value may be an outcome.
    fn block(&mut self, b: &Block) -> bool {
        for s in &b.stmts {
            match s {
                Stmt::Let { local, init } => {
                    if let Some(e) = init {
                        let o = self.expr(e);
                        if o || self.control > 0 {
                            self.mark(*local);
                        }
                        if self.control > 0 || (o && self.shaped(e)) {
                            self.shaped.insert(*local);
                        }
                    }
                }
                Stmt::LetPat { pat, init } => {
                    if self.expr(init) || self.control > 0 {
                        self.mark_pattern(pat);
                    }
                }
                Stmt::Expr(e) => {
                    self.expr(e);
                }
            }
        }
        b.value.as_ref().is_some_and(|v| self.expr(v))
    }

    /// Walks the expressions of a list, returning whether any may be an
    /// outcome.
    fn any(&mut self, es: &[Expr]) -> bool {
        es.iter().fold(false, |acc, e| self.expr(e) | acc)
    }

    /// Walks each of `es`, none of which may be an outcome, being used `how`.
    fn forbid<'e>(&mut self, es: impl IntoIterator<Item = &'e Expr>, how: Misuse) {
        for e in es {
            if self.expr(e) {
                self.misuse(e.span, how);
            }
        }
    }

    /// Walks an expression, returning whether its value may be an outcome.
    fn expr(&mut self, e: &Expr) -> bool {
        stacker::maybe_grow(64 * 1024, 1024 * 1024, || self.expr_inner(e))
    }

    fn expr_inner(&mut self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Const(_)
            | ExprKind::Global(_)
            | ExprKind::FnRef(_)
            | ExprKind::ThunkRef(_)
            | ExprKind::Continue { .. } => false,
            ExprKind::Local(l) => self.outcome.contains(l),
            ExprKind::Quantum { op, args } => {
                let any = self.any(args);
                match op {
                    QuantumOp::Measure | QuantumOp::Lookup(_) | QuantumOp::LookupAt => true,
                    QuantumOp::Lift => {
                        if !any {
                            self.lift_of_fixed(e.span);
                        }
                        false
                    }
                    _ => false,
                }
            }
            ExprKind::Call { args, .. } => {
                self.forbid(args, Misuse::Argument);
                false
            }
            ExprKind::IndirectCall { callee, args } => {
                self.expr(callee);
                self.forbid(args, Misuse::Argument);
                false
            }
            // an array of outcomes is itself an outcome; how long it is
            // depends on one only when it changes under a test of one
            ExprKind::Growable { op, array, args } => {
                let a = self.expr(array);
                let v = self.any(args) || self.control > 0;
                if v {
                    self.assigned(array);
                }
                if self.control > 0 && *op != crate::tir::GrowOp::Reserve {
                    self.reshaped(array);
                }
                a
            }
            ExprKind::Intrinsic { which: Intrinsic::Len, args } => {
                let o = self.any(args);
                o && args.iter().any(|a| self.shaped(a))
            }
            ExprKind::Closure { captures, .. } => {
                self.forbid(captures, Misuse::Capture);
                false
            }
            ExprKind::Intrinsic { which, args } => {
                if which.acts_on_environment() {
                    self.side_effect(*which, e.span);
                }
                self.any(args)
            }
            ExprKind::ArrayLit(args) => self.any(args),
            ExprKind::StructLit { fields } | ExprKind::Variant { fields, .. } => {
                fields.iter().fold(false, |acc, (_, f)| self.expr(f) | acc)
            }
            ExprKind::Grow(x)
            | ExprKind::Unary { operand: x, .. }
            | ExprKind::Cast { expr: x, .. }
            | ExprKind::Coerce(x)
            | ExprKind::FnAsClosure(x)
            | ExprKind::Field { base: x, .. }
            | ExprKind::Deref(x)
            | ExprKind::Ref(x)
            | ExprKind::ArrayRepeat { elem: x, .. } => self.expr(x),
            ExprKind::Binary { lhs, rhs, .. } | ExprKind::Logical { lhs, rhs, .. } => {
                let l = self.expr(lhs);
                self.expr(rhs) | l
            }
            ExprKind::Index { base, index } => {
                let b = self.expr(base);
                let i = self.expr(index);
                if i && !b {
                    self.misuse(index.span, Misuse::Index);
                }
                b || i
            }
            ExprKind::Assign { place, value } => {
                let o = self.expr(value);
                let v = o || self.control > 0;
                self.expr(place);
                if v {
                    self.assigned(place);
                }
                if self.control > 0 || (o && self.shaped(value)) {
                    self.reshaped(place);
                }
                false
            }
            ExprKind::Block(b) => self.block(b),
            ExprKind::If { cond, then, els } => {
                let c = self.expr(cond);
                self.control += u32::from(c);
                let t = self.block(then);
                let o = els.as_ref().is_some_and(|x| self.expr(x));
                self.control -= u32::from(c);
                c || t || o
            }
            ExprKind::Match { scrutinee, arms } => {
                let s = self.expr(scrutinee);
                self.control += u32::from(s);
                let mut any = s;
                for a in arms {
                    if s {
                        self.mark_pattern(&a.pat);
                    }
                    any |= self.expr(&a.body);
                }
                self.control -= u32::from(s);
                any
            }
            ExprKind::Loop { body, .. } => {
                self.block(body);
                self.block(body);
                false
            }
            ExprKind::While { cond, body, .. } => {
                for _ in 0..2 {
                    self.forbid([&**cond], Misuse::Bound);
                    self.block(body);
                }
                false
            }
            ExprKind::ForRange { start, end, body, .. } => {
                self.forbid([&**start, &**end], Misuse::Bound);
                self.block(body);
                self.block(body);
                false
            }
            ExprKind::Break { value, .. } | ExprKind::Return(value) => {
                if let Some(v) = value {
                    self.expr(v);
                }
                false
            }
        }
    }

    /// An outcome stored into `place`: a binding becomes an outcome, and a
    /// unit-scope object may not hold one.
    fn assigned(&mut self, place: &Expr) {
        match &place.kind {
            ExprKind::Local(l) => {
                self.mark(*l);
            }
            ExprKind::Global(_) => self.misuse(place.span, Misuse::Object),
            ExprKind::Field { base, .. } | ExprKind::Index { base, .. } => self.assigned(base),
            // through a reference, the object referred to is someone
            // else's; what it is cannot be told here
            _ => {}
        }
    }

    /// Whether the shape of `e`'s value may depend on an outcome. What is
    /// measured has the shape of what it measures, fixed when the circuit
    /// is generated or, for a register grown as it runs, by values that are
    /// not outcomes; a literal has the shape it is written with.
    fn shaped(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Quantum { op: QuantumOp::Measure, .. }
            | ExprKind::ArrayLit(_)
            | ExprKind::ArrayRepeat { .. }
            | ExprKind::StructLit { .. } => false,
            ExprKind::Local(l) => self.shaped.contains(l),
            ExprKind::Ref(x) | ExprKind::Deref(x) | ExprKind::Field { base: x, .. } | ExprKind::Coerce(x) | ExprKind::Grow(x) => {
                self.shaped(x)
            }
            _ => true,
        }
    }

    /// An array whose length may come to depend on an outcome, at `place`.
    fn reshaped(&mut self, place: &Expr) {
        match &place.kind {
            ExprKind::Local(l) => {
                self.shaped.insert(*l);
            }
            ExprKind::Field { base, .. } | ExprKind::Index { base, .. } => self.reshaped(base),
            _ => {}
        }
    }

    /// An operation on the world outside the program (printing, leaving,
    /// files, modules), which a quantum unit's code, run while circuits are
    /// generated, has no world to perform on.
    fn side_effect(&mut self, which: Intrinsic, span: Span) {
        if !self.reported.insert((span.start, span.end)) {
            return;
        }
        self.diags.push(
            Diagnostic::new(Code::Eq11)
                .with_message(format!(
                    "`{}` acts on the running program's surroundings, which a quantum unit's code has none of",
                    which.name()
                ))
                .at(span)
                .with_note(
                    "a quantum unit's code runs while its circuits are generated, during \
                     translation; what it computes shapes circuits, and nothing it does reaches \
                     the program that later runs them",
                )
                .with_help("do this in a classical unit, with what the circuit's result gives it"),
        );
    }

    /// `lift` of a value that is generation-time already.
    fn lift_of_fixed(&mut self, span: Span) {
        if !self.reported.insert((span.start, span.end)) {
            return;
        }
        self.diags.push(
            Diagnostic::new(Code::Es06)
                .with_message("`lift` takes a measured value, and this one is fixed when the circuit is generated")
                .at(span)
                .with_note(
                    "`lift` makes a value observed while the circuit runs available to the structure \
                     of the rest of it; a value known already needs no lifting",
                )
                .with_help("remove `lift`"),
        );
    }
}
