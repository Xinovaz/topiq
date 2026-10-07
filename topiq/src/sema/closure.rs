//! Function values and closures.
//!
//! A function is a value of a function type, `fn(T…) -> U`: naming one
//! without calling it gives its code's address, and `&f` gives a `*fn` that
//! stays valid for the whole program. A value of a function type is called
//! like a function, directly or through a reference.
//!
//! A closure expression, `fn(x: i64) -> i64 with (n) { x + n }`, makes a
//! function together with what it captured. Nothing is captured implicitly:
//! inside the body, only the parameters, the names in the capture list and
//! the unit's own items are in scope. A capture written `n` moves `n` into the
//! closure; one written `&n` holds a reference to it. Either way the body
//! reads the capture by its own name and cannot change it, since a closure
//! may be called any number of times and each call must find what the last
//! one did.
//!
//! The closure's body becomes a function of its own whose first parameter is
//! a reference to the environment the captures are kept in. A closure that
//! captures nothing needs no environment, and the closure expression itself
//! converts to a `*fn` wherever one is wanted.

use crate::ast::{self, Capture};
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{self, Access, Callee, Expr, ExprKind, FnKind, Local, LocalId, Ty};

use super::body::{self, Checker};
use super::report;
use super::scope::Def;

impl Checker<'_, '_> {
    /// The `core` variant a one-part path names (`Some`, `None`, `Ok` or
    /// `Err`), with the generic enumeration it belongs to, unless something
    /// in scope has that name.
    pub(super) fn prelude_variant(&mut self, path: &ast::Path) -> Option<(super::items::GenericKey, Spanned<Symbol>)> {
        let [name] = path.segments.as_slice() else {
            return None;
        };
        if self.scopes.in_blocks(name.node).is_some() || self.cx.lookup_value(name.node).is_some() {
            return None;
        }
        self.cx.prelude_variant(name.node).map(|k| (k, *name))
    }

    /// A library function, such as `print`, as a value: a function of the
    /// unit that calls it, made the first time it is asked for. One generic
    /// over what it is given, such as `clone`, has no one type to be.
    pub(super) fn library_function_value(&mut self, which: crate::tir::Intrinsic, path: &ast::Path, span: Span) -> Expr {
        use crate::tir::Intrinsic;
        // made already, or one with a type of its own
        let n = path.segments.iter().map(|s| self.name(s.node)).collect::<Vec<_>>().join("::");
        if let Some(&id) = self.cx.library_values.get(&which) {
            return self.function_value(Callee::Fn(id), path, span);
        }
        if !matches!(which, Intrinsic::Print | Intrinsic::Eprint | Intrinsic::Exit | Intrinsic::Panic) {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`{n}` is not a function the program can hold as a value"))
                    .at(span)
                    .with_note(format!(
                        "`{n}` works on values of any type, or is part of another construct, so it has \
                         no one function type"
                    ))
                    .with_help(format!("call it where it is needed, or hold a closure that calls it, as in `|x| {n}(x)`")),
            );
            return Checker::error(span);
        }

        // a hidden function passing its parameters to the intrinsic
        let (params, ret) = super::core::signature(which, self.types_mut());
        let name = path.segments.last().map_or_else(|| self.interner.intern_late(&n), |s| s.node);
        let arg = self.interner.intern_late("argument");
        let locals: Vec<Local> = params
            .iter()
            .map(|&ty| Local {
                name: arg,
                ty,
                constant: false,
                constexpr: false,
                aux: false,
                span,
            })
            .collect();
        let args = (0..locals.len() as u32).map(|i| Expr::local(LocalId(i), params[i as usize], span)).collect();
        let call = Expr::intrinsic(which, args, ret, span);
        let id = self.cx.add_checked_fn(tir::Fn {
            params: (0..locals.len() as u32).map(LocalId).collect(),
            ret,
            locals,
            body: tir::Block {
                stmts: Vec::new(),
                value: Some(Box::new(call)),
                ty: ret,
                span,
            },
            attrs: tir::FnAttrs {
                hidden: true,
                ..tir::FnAttrs::default()
            },
            ..tir::Fn::empty(name, crate::ast::Linkage::Unit, span, span)
        });
        self.cx.library_values.insert(which, id);
        self.function_value(Callee::Fn(id), path, span)
    }

    pub(super) fn function_value(&mut self, callee: Callee, path: &ast::Path, span: Span) -> Expr {
        let (params, ret) = match callee {
            Callee::Fn(id) => {
                self.cx.fn_sig(id);
                let f = self.cx.unit.func(id);
                if f.kind == FnKind::Constant {
                    let n = path.segments.iter().map(|s| self.name(s.node)).collect::<Vec<_>>().join("::");
                    self.report(
                        Diagnostic::new(Code::Ec01)
                            .with_message(format!("`{n}` is a constant function, so it is not a value the program can hold"))
                            .at(span)
                            .with_note("a constant function runs during translation and has no code in the program")
                            .with_help(format!("call it, as in `{n}(…)`")),
                    );
                    return Checker::error(span);
                }
                (f.param_types().collect::<Vec<_>>(), f.ret)
            }
            Callee::Extern(id) => {
                let x = self.cx.unit.extern_fn(id);
                (x.params.clone(), x.ret)
            }
        };
        let ty = self.types_mut().function(params, ret);
        Expr {
            kind: ExprKind::FnRef(callee),
            ty,
            span,
        }
    }

    /// A closure's capture, named in its body: the place it is kept in the
    /// environment, or, for one captured by reference, the place that
    /// reference refers to.
    pub(super) fn captured(&mut self, index: u32, span: Span) -> Expr {
        let (env, by_ref) = self.env.clone().expect("a capture is only bound in a closure's body");
        let env_ty = self.locals[env.index()].ty;
        let tuple = self.types().as_ref(env_ty).expect("the environment is a reference").1;
        let field_ty = self.types().as_tuple(tuple).expect("the environment is a tuple")[index as usize];
        let base = Expr::deref(Expr::local(env, env_ty, span), tuple, span);
        let held = Expr {
            kind: ExprKind::Field {
                base: Box::new(base),
                field: index,
            },
            ty: field_ty,
            span,
        };
        if !by_ref[index as usize] {
            return held;
        }
        let target = self.types().as_ref(field_ty).expect("a reference capture").1;
        Expr::deref(held, target, span)
    }

    /// Calls a value: a function value, a closure, or a value whose type
    /// defines `$call`, directly or through a reference.
    pub(super) fn call_value(&mut self, callee: Expr, args: &[Spanned<ast::Expr>], span: Span) -> Option<Expr> {
        let callee = self.auto_deref(callee);
        let t = self.shallow(callee.ty);
        let (params, ret) = self.types().as_sig(t).map(|(p, r)| (p.to_vec(), r))?;
        let text = self.show(t);
        let mut checked = Vec::with_capacity(args.len());
        for a in args {
            checked.push(self.expr(a));
        }
        if checked.len() != params.len() {
            let plural = |k: usize| if k == 1 { "argument" } else { "arguments" };
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!(
                        "this `{text}` takes {} {}, but this call gives {}",
                        params.len(),
                        plural(params.len()),
                        checked.len()
                    ))
                    .at(span),
            );
            return Some(Checker::error(span));
        }
        let args = checked
            .into_iter()
            .zip(params)
            .enumerate()
            .map(|(i, (a, p))| self.coerce(a, p, &format!("argument {} of this call", i + 1)))
            .collect();
        Some(Expr {
            kind: ExprKind::IndirectCall {
                callee: Box::new(callee),
                args,
            },
            ty: ret,
            span,
        })
    }

    /// `fn(params) -> ret with (captures) { body }`.
    pub(super) fn closure(
        &mut self,
        params: &[ast::Param],
        ret: Option<&Spanned<ast::Type>>,
        captures: &[Capture],
        block: &ast::Block,
        span: Span,
    ) -> Expr {
        // what is captured, checked where the closure is written
        let mut held: Vec<Expr> = Vec::with_capacity(captures.len());
        let mut names: Vec<(Symbol, bool)> = Vec::with_capacity(captures.len());
        let mut env: Vec<Ty> = Vec::with_capacity(captures.len());
        for c in captures {
            let text = self.name(c.name.node);
            if names.iter().any(|&(n, _)| n == c.name.node) {
                self.report(
                    Diagnostic::new(Code::Es05)
                        .with_message(format!("`{text}` is captured twice"))
                        .at(c.name.span)
                        .with_help("capture each name once"),
                );
                continue;
            }
            let value = match self.scopes.in_blocks(c.name.node) {
                Some(Def::Local(id)) => Expr::local(id, self.locals[id.index()].ty, c.name.span),
                Some(Def::Capture(i)) => self.captured(i, c.name.span),
                _ => {
                    let d = if self.resolve(c.name.node).is_some() {
                        Diagnostic::new(Code::Es07)
                            .with_message(format!(
                                "`{text}` is declared at unit scope, so a closure sees it without capturing it"
                            ))
                            .at(c.name.span)
                            .with_help(format!("remove `{text}` from the capture list"))
                    } else {
                        report::not_found(c.name.span, "binding", text, self.visible())
                            .with_note("a closure captures bindings of the block it is written in")
                    };
                    self.report(d);
                    continue;
                }
            };
            if self.types().is_quantum(value.ty) {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("a closure cannot capture `{text}`, which holds qubits"))
                        .at(c.name.span)
                        .with_note(
                            "a closure's environment lives as long as the closure does, which nothing \
                             fixes, while a qubit must be measured, forgotten or handed on at a point \
                             the program states",
                        )
                        .with_help(format!("pass `{text}`, or a reference to it, as a parameter instead")),
                );
                continue;
            }
            let value = if c.by_ref {
                self.reference_to(value, c.name.span)
            } else {
                value
            };
            // a literal nothing has settled yet takes its default here: the
            // environment's layout must be known now
            let ty = self.settled(value.ty);
            env.push(ty);
            names.push((c.name.node, c.by_ref));
            held.push(value);
        }

        // the body's own function: the environment first, then the
        // parameters
        let mut locals: Vec<Local> = Vec::with_capacity(params.len() + 1);
        let env_tuple = self.types_mut().tuple(env);
        let env_ref = self.types_mut().reference(Access::Write, env_tuple);
        let name = self.func.map_or(c_name(captures, params), |f| self.cx.unit.func(f).name);
        locals.push(Local {
            name,
            ty: env_ref,
            constant: false,
            constexpr: false,
            aux: false,
            span,
        });
        let mut param_tys = Vec::with_capacity(params.len());
        for p in params {
            if p.is_self {
                self.report(
                    Diagnostic::new(Code::Es07)
                        .with_message("a closure has no `self`: it is not a method")
                        .at(p.name.span),
                );
            }
            let r = self.resolve_type(&p.ty);
            let t = r.ty;
            if t == Ty::Void {
                self.report(super::items::void_binding(p.ty.span));
            }
            if r.constexpr {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message("a closure's parameter cannot be `constexpr`")
                        .at(p.ty.span)
                        .with_note("a closure is called when the program runs, with values only known then")
                        .with_help("write `const` for a parameter that is never assigned"),
                );
            }
            if let Some(earlier) = locals[1..].iter().find(|l| l.name == p.name.node) {
                let d = report::duplicate(p.name.span, earlier.span, self.name(p.name.node), "in one parameter list");
                self.report(d);
            }
            param_tys.push(t);
            locals.push(Local {
                name: p.name.node,
                ty: t,
                constant: r.constant,
                constexpr: false,
                aux: false,
                span: p.name.span,
            });
        }
        let ret = ret.map_or(Ty::Void, |r| self.resolve_type(r).ty);
        let origin = self.func.and_then(|f| self.cx.unit.func(f).origin.clone());
        let empty = tir::Fn::empty(name, crate::ast::Linkage::Unit, span, span);
        let id = self.cx.add_checked_fn(tir::Fn {
            origin,
            closure: true,
            params: (0..locals.len() as u32).map(LocalId).collect(),
            ret,
            locals,
            body: tir::Block { ty: ret, ..empty.body },
            ..empty
        });
        body::check_body(self.cx, id, block, Some(&names));
        let ty = self.types_mut().closure(param_tys, ret);
        Expr {
            kind: ExprKind::Closure { code: id, captures: held },
            ty,
            span,
        }
    }
}

/// A name for a closure's body written outside any function, which only a
/// message could show.
fn c_name(captures: &[Capture], params: &[ast::Param]) -> Symbol {
    captures
        .first()
        .map(|c| c.name.node)
        .or_else(|| params.first().map(|p| p.name.node))
        .unwrap_or(Symbol::EMPTY)
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};

    #[test]
    fn nothing_is_captured_implicitly() {
        assert_eq!(codes("fn f() { let n = 1; let g = fn() -> i64 with () { n }; }"), [Code::Es04]);
        check("fn f() { let n = 1; let g = fn() -> i64 with (n) { n }; }");
    }

    #[test]
    fn unit_scope_is_seen_without_capturing() {
        check("let K: i64 = 3;\nfn f() { let g = fn() -> i64 with () { K }; }");
        assert_eq!(codes("let K: i64 = 3;\nfn f() { let g = fn() -> i64 with (K) { K }; }"), [Code::Es07]);
    }

    #[test]
    fn a_name_is_captured_once() {
        assert_eq!(codes("fn f() { let n = 1; let g = fn() with (n, &n) { }; }"), [Code::Es05]);
    }

    #[test]
    fn a_capture_by_reference_changes_what_it_refers_to() {
        check("fn f() { let n = 1; let g = fn() with (&n) { n = 2; }; }");
        assert_eq!(codes("fn f() { let n: const i32 = 1; let g = fn() with (&n) { n = 2; }; }"), [Code::Ec03]);
    }

    #[test]
    fn a_moved_capture_is_gone_from_its_block() {
        assert_eq!(
            codes("struct S { v: i64 }\nfn f() -> i64 { let s = S { v: 1 }; let g = fn() -> i64 with (s) { s.v }; s.v }"),
            [Code::Es10]
        );
    }

    #[test]
    fn a_closure_is_moved_not_copied() {
        assert_eq!(
            codes("fn f() { let g = fn() with () { }; let h = g; g(); }"),
            [Code::Es10]
        );
    }

    #[test]
    fn a_call_through_a_value_checks_its_arguments() {
        assert_eq!(codes("fn sq(x: i64) -> i64 { x * x }\nfn f() -> i64 { let g = sq; g(1, 2) }"), [Code::Es07]);
        assert_eq!(codes("fn sq(x: i64) -> i64 { x * x }\nfn f() -> i64 { let g = sq; g(true) }"), [Code::Es06]);
    }

    #[test]
    fn a_closure_that_captures_is_not_a_plain_function() {
        assert_eq!(
            codes("fn f() { let n = 1; let p: *fn() -> i64 = fn() -> i64 with (n) { n }; }"),
            [Code::Es06]
        );
    }

    #[test]
    fn a_constant_function_is_not_a_value() {
        assert_eq!(codes("fn c(n: constexpr i64) -> constexpr i64 { n }\nfn f() { let g = c; }"), [Code::Ec01]);
    }
}
