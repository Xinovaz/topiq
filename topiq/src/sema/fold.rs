//! Replacing every constant with its value.
//!
//! A constant has no run-time representation: a `const` binding whose value
//! is known during translation has no storage, and a constant function is
//! never compiled. So before code generation sees a unit, every place such a
//! constant is used must already hold its value. This pass computes those
//! values with the [`Evaluator`] and substitutes them:
//!
//! - every unit-scope and persistent object gets its initial value, which is
//!   placed directly in the program image;
//! - every `const` binding whose value can be computed now is folded: every
//!   read of it becomes its value, and its `let` disappears. One whose value
//!   depends on something only known when the program runs stays a binding,
//!   which is never assigned, unless it is `constexpr`, which promises a
//!   value now, and is reported;
//! - every call of a constant function becomes its result, after checking
//!   that each argument is itself constant, since a `constexpr` parameter has
//!   no run-time value to receive.
//!
//! Whether a `const` binding is folded is inferred, and nothing depends on
//! the answer but where its value lives: everything that needs a value known
//! during translation (an array's length, a generic argument, a `constexpr`
//! parameter) is written with unit-scope constants, constant functions and
//! `constexpr`, which are checked.
//!
//! It also runs the checks that need a value to decide: whether a constant's
//! initialiser really is constant, whether computing it would abort, and
//! whether a constant index is outside the array it indexes, which is
//! reported now rather than left to abort when the program runs.

use crate::diag::{Code, Diagnostic, Limit};
use crate::intern::Interner;
use crate::span::Span;
use crate::tir::visit::{self, VisitMut};
use crate::tir::{Block, Callee, Expr, ExprKind, FnId, FnKind, GlobalId, LocalId, Stmt, Unit, Value};

use super::consteval::{EvalError, Evaluator, Frame, MAX_DEPTH};

/// What a constant was needed for, which decides how a failure is described.
pub enum Need<'a> {
    /// The initial value of an object.
    Object(&'a str),
    /// The value of a `const` binding.
    Binding(&'a str),
    /// An argument for a `const` parameter.
    Argument {
        /// The function called.
        callee: &'a str,
        /// Which argument, from 0.
        index: usize,
        /// The parameter's name.
        param: &'a str,
    },
    /// The result of a call of a constant function.
    Result(&'a str),
    /// A constant the language requires in some position, such as "the length
    /// of an array".
    Constant(&'a str),
}

/// Folds every constant in the unit.
pub fn fold(unit: &mut Unit, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    // the evaluator reads a snapshot, so the unit can be rewritten while it
    // works. folding only ever replaces an expression with its own value, so
    // the snapshot never disagrees with the result
    let snapshot = unit.clone();
    let mut ev = Evaluator::new(&snapshot, interner);

    for (i, g) in snapshot.globals.iter().enumerate() {
        if g.value.is_some() {
            continue;
        }
        let Some(init) = &g.init else { continue };
        match ev.global(GlobalId(i as u32), init.span) {
            Ok(v) => unit.globals[i].value = Some(v),
            Err(e) => diags.push(describe(
                e,
                &Need::Object(interner.resolve(g.name)),
                &snapshot,
                interner,
            )),
        }
    }

    for (i, f) in snapshot.fns.iter().enumerate() {
        if f.kind != FnKind::Runtime {
            continue;
        }
        let mut folder = Folder {
            frame: Frame::for_fn(&snapshot, FnId(i as u32)),
            func: FnId(i as u32),
            ev: &mut ev,
            unit: &snapshot,
            interner,
            diags,
        };
        folder.block(&mut unit.fns[i].body);
    }
}

struct Folder<'e, 'a> {
    ev: &'e mut Evaluator<'a>,
    frame: Frame,
    func: FnId,
    unit: &'a Unit,
    interner: &'a Interner,
    diags: &'e mut Vec<Diagnostic>,
}

impl Folder<'_, '_> {
    fn is_const_local(&self, id: LocalId) -> bool {
        self.unit.func(self.func).local(id).constant
    }

    fn is_constexpr_local(&self, id: LocalId) -> bool {
        self.unit.func(self.func).local(id).constexpr
    }

    /// Evaluates a call of a constant function whose arguments have already
    /// been folded.
    fn call(&mut self, callee: FnId, args: &[Expr], span: Span) -> Option<Value> {
        let f = self.unit.func(callee);
        let callee_name = self.interner.resolve(f.name);
        let mut values = Vec::with_capacity(args.len());
        let mut ok = true;
        for (index, (a, &p)) in args.iter().zip(&f.params).enumerate() {
            match self.ev.top(a, &mut self.frame) {
                Ok(v) => values.push(v),
                Err(e) => {
                    ok = false;
                    let param = self.interner.resolve(f.local(p).name);
                    self.diags.push(describe(
                        e,
                        &Need::Argument {
                            callee: callee_name,
                            index,
                            param,
                        },
                        self.unit,
                        self.interner,
                    ));
                }
            }
        }
        if !ok {
            return None;
        }
        match self.ev.call(callee, &values, span) {
            Ok(v) => Some(v),
            Err(e) => {
                self.diags
                    .push(describe(e, &Need::Result(callee_name), self.unit, self.interner));
                None
            }
        }
    }

    /// Reports a constant index outside what it indexes.
    fn check_index(&mut self, base: &Expr, index: &Expr) {
        let ExprKind::Const(Value::Int(i, _)) = index.kind else {
            return;
        };
        let len = match &base.kind {
            ExprKind::Const(Value::Str(s)) => Some(self.interner.resolve(*s).chars().count() as u64),
            _ => self.unit.types.as_array(base.ty).map(|(_, n)| n),
        };
        if let Some(len) = len
            && u64::try_from(i).is_ok_and(|i| i >= len)
        {
            self.diags.push(
                Diagnostic::new(Code::Ec07)
                    .with_message(format!(
                        "index {i} is out of bounds: this has {len} element{}, so the last index is {}",
                        if len == 1 { "" } else { "s" },
                        len.saturating_sub(1)
                    ))
                    .at(index.span)
                    .with_note(
                        "both the index and the length are known during translation, so the \
                         access that would abort when the program runs is reported now",
                    ),
            );
        }
    }
}

impl VisitMut for Folder<'_, '_> {
    fn block(&mut self, b: &mut Block) {
        let mut kept = Vec::with_capacity(b.stmts.len());
        for s in b.stmts.drain(..) {
            match s {
                Stmt::Let {
                    local,
                    init: Some(mut init),
                } if self.is_const_local(local) => {
                    match self.ev.top(&init, &mut self.frame) {
                        // a constant has no storage, so its `let` has nothing
                        // left to do
                        Ok(v) => self.frame.locals[local.index()] = Some(v),
                        // a value only known when the program runs: an
                        // ordinary binding that is never assigned
                        Err(EvalError::NotConstant { .. }) if !self.is_constexpr_local(local) => {
                            self.expr(&mut init);
                            kept.push(Stmt::Let { local, init: Some(init) });
                        }
                        Err(e) => {
                            let name = self
                                .interner
                                .resolve(self.unit.func(self.func).local(local).name);
                            self.diags
                                .push(describe(e, &Need::Binding(name), self.unit, self.interner));
                        }
                    }
                }
                Stmt::Let { local, mut init } => {
                    if let Some(e) = &mut init {
                        self.expr(e);
                    }
                    kept.push(Stmt::Let { local, init });
                }
                Stmt::LetPat { mut pat, mut init } => {
                    self.pat(&mut pat);
                    self.expr(&mut init);
                    kept.push(Stmt::LetPat { pat, init });
                }
                Stmt::Expr(mut e) => {
                    self.expr(&mut e);
                    kept.push(Stmt::Expr(e));
                }
            }
        }
        b.stmts = kept;
        if let Some(v) = &mut b.value {
            self.expr(v);
        }
    }

    fn expr(&mut self, e: &mut Expr) {
        visit::walk_expr_mut(self, e);
        let span = e.span;
        let folded = match &e.kind {
            ExprKind::Local(id) if self.is_const_local(*id) => self.frame.locals[id.index()].clone(),
            ExprKind::Global(id) if self.unit.global(*id).constant => {
                // an object's failure was reported when its value was computed
                self.ev.global(*id, span).ok()
            }
            ExprKind::Call {
                callee: Callee::Fn(callee),
                args,
            } if self.unit.func(*callee).kind == FnKind::Constant => self.call(*callee, args, span),
            ExprKind::Index { base, index } => {
                self.check_index(base, index);
                None
            }
            _ => None,
        };
        if let Some(v) = folded {
            *e = Expr::constant(v, e.ty, span);
        }
    }
}

/// Turns an evaluation failure into a diagnostic, in terms of what the value
/// was needed for.
pub fn describe(e: EvalError, need: &Need<'_>, unit: &Unit, interner: &Interner) -> Diagnostic {
    match e {
        EvalError::NotConstant { span, why } => {
            let (code, message, note) = match need {
                Need::Object(name) => (
                    Code::Ec01,
                    format!("the initial value of `{name}` must be known before the program runs"),
                    "an object declared outside any function, or with `persist`, is placed in \
                     the program with its value already computed, so no code ever runs to \
                     compute it",
                ),
                Need::Binding(name) => (
                    Code::Ec01,
                    format!("`{name}` is declared `constexpr`, so its value must be known during translation"),
                    "a `constexpr` binding promises a value computed while the program is \
                     translated; write `const` for one that is only never assigned",
                ),
                Need::Argument {
                    callee,
                    index,
                    param,
                } => (
                    Code::Ec02,
                    format!(
                        "argument {} of `{callee}` must be known during translation, because the \
                         parameter `{param}` is `constexpr`",
                        index + 1
                    ),
                    "a function whose parameters are all `constexpr` is evaluated while the \
                     program is translated, so there is no run-time value for it to receive",
                ),
                Need::Result(callee) => (
                    Code::Ec01,
                    format!("`{callee}` cannot be evaluated during translation"),
                    "a function whose parameters are all `constexpr` is evaluated at every call, \
                     so its body may only use constants",
                ),
                Need::Constant(role) => (
                    Code::Ec01,
                    format!("{role} must be known during translation"),
                    "it decides the layout or meaning of what is being declared, so it cannot \
                     wait until the program runs",
                ),
            };
            Diagnostic::new(code)
                .with_message(message)
                .at_with(span, why)
                .with_note(note)
        }
        EvalError::Abort { abort, span } => {
            let code = abort.code();
            let what = match need {
                Need::Object(n) | Need::Binding(n) => format!("the value of `{n}`"),
                Need::Argument { callee, index, .. } => {
                    format!("argument {} of `{callee}`", index + 1)
                }
                Need::Result(c) => format!("the call of `{c}`"),
                Need::Constant(role) => (*role).to_owned(),
            };
            Diagnostic::new(Code::Ec04)
                .with_message(format!(
                    "computing {what} would abort the program with {code}: {}",
                    code.message()
                ))
                .at_with(span, format!("{code} here"))
                .with_note(
                    "a constant is computed while the program is translated, so an operation \
                     that would abort when the program runs is reported now instead",
                )
        }
        EvalError::TooDeep { span } => Limit::ConstRecursion
            .exceeded(span, MAX_DEPTH + 1)
            .with_note("constant functions may call each other, or themselves, at most 64 deep"),
        EvalError::Cycle { global, span } => {
            let g = unit.global(global);
            let name = interner.resolve(g.name);
            Diagnostic::new(Code::Ec01)
                .with_message(format!(
                    "the value of `{name}` depends on itself, so it cannot be computed"
                ))
                .at_with(span, format!("`{name}` is needed here"))
                .also(g.span, format!("while computing `{name}` itself"))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{analyzed, check, codes};
    use crate::tir::{ExprKind, IntTy, Stmt, Value};

    #[test]
    fn a_const_binding_is_folded_away() {
        let u = check("fn f() -> i32 { let k: const i32 = 6 * 7; k + 1 }");
        let body = &u.fns[0].body;
        assert!(body.stmts.is_empty(), "the `let` of a constant disappears");
        let ExprKind::Binary { lhs, .. } = &body.value.as_ref().unwrap().kind else {
            panic!()
        };
        assert!(matches!(lhs.kind, ExprKind::Const(Value::Int(42, IntTy::I32))));
    }

    #[test]
    fn an_ordinary_binding_is_kept() {
        let u = check("fn f() -> i32 { let k = 1; k }");
        assert!(matches!(u.fns[0].body.stmts[0], Stmt::Let { .. }));
    }

    #[test]
    fn a_call_of_a_constant_function_becomes_its_result() {
        let u = check("fn sq(n: constexpr i32) -> constexpr i32 { n * n }\nfn f() -> i32 { sq(7) }");
        let v = u.fns[1].body.value.as_ref().unwrap();
        assert!(matches!(v.kind, ExprKind::Const(Value::Int(49, IntTy::I32))), "{:?}", v.kind);
    }

    #[test]
    fn a_non_constant_argument_for_a_const_parameter_is_ec02() {
        let got = codes("fn sq(n: constexpr i32) -> constexpr i32 { n * n }\nfn f(x: i32) -> i32 { sq(x) }");
        assert_eq!(got, [Code::Ec02]);
    }

    #[test]
    fn a_const_binding_with_a_run_time_value_stays_a_binding() {
        let u = check("fn f(x: i32) -> i32 { let k: const i32 = x; let j: const i32 = 2; k + j }");
        let body = &u.fns[0].body;
        assert_eq!(body.stmts.len(), 1, "`j` is folded, `k` kept: {:?}", body.stmts);
        assert!(matches!(body.stmts[0], Stmt::Let { .. }));
    }

    #[test]
    fn a_constexpr_binding_with_a_run_time_value_is_ec01() {
        assert_eq!(codes("fn f(x: i32) { let k: constexpr i32 = x; }"), [Code::Ec01]);
        check("fn f() -> i32 { let k: constexpr i32 = 2 * 3; k }");
    }

    #[test]
    fn a_unit_scope_object_needs_a_constant_value() {
        assert_eq!(codes("fn g() -> i32 { let a = 1; a += 1; a }\nlet X: i32 = g();"), [Code::Ec01]);
        check("let X: i32 = 5;");
    }

    #[test]
    fn a_constant_that_would_abort_is_ec04() {
        let (_, d) = analyzed("let X: const u8 = 255 + 1;");
        assert_eq!(d.iter().map(|d| d.code).collect::<Vec<_>>(), [Code::Ec04]);
        assert!(d[0].message.contains("RA01"), "{}", d[0].message);
    }

    #[test]
    fn a_self_dependent_constant_is_refused() {
        assert_eq!(codes("let A: const i32 = B;\nlet B: const i32 = A;"), [Code::Ec01]);
    }

    #[test]
    fn unbounded_constant_recursion_hits_the_limit() {
        let src = "fn r(n: constexpr u32) -> constexpr u32 { r(n + 1) }\nlet X: const u32 = r(0);";
        assert_eq!(codes(src), [Code::Em05]);
    }

    #[test]
    fn reading_a_non_const_object_in_a_constant_is_refused() {
        assert_eq!(codes("let A: i32 = 1;\nlet B: const i32 = A;"), [Code::Ec01]);
    }

    #[test]
    fn a_constant_index_outside_an_array_is_ec07() {
        assert_eq!(codes("fn f(a: [u8; 3]) -> u8 { a[3] }"), [Code::Ec07]);
        assert_eq!(codes("let I: const usize = 5;\nfn f(a: [u8; 3]) -> u8 { a[I] }"), [Code::Ec07]);
        check("fn f(a: [u8; 3]) -> u8 { a[2] }");
        assert_eq!(codes("fn f() -> char { \"ab\"[2] }"), [Code::Ec07]);
    }
}
