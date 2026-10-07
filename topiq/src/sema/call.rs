//! Calls: of functions, of another unit's functions, of the `core` library,
//! of tuple variants, of function values and closures, of `invoke`, of
//! generic functions whose arguments are worked out from the values given,
//! and of functions specialised for their `const` arguments, and `len` on
//! arrays. The builtin macros are in [`super::macros`], methods in
//! [`super::method`].
//!
//! # Specialised functions
//!
//! A function whose parameters mix `const` and ordinary ones is made once
//! for each set of values of its `const` parameters. A call computes those
//! arguments during translation and calls the function made for them with
//! the rest.
//!
//! # `len`
//!
//! `xs.len()` gives the number of elements of an array or slice, as a `usize`.
//! On a fixed array the answer is its declared length, known during
//! translation.

use crate::ast::{self, TArg};
use crate::diag::{Code, Diagnostic};
use crate::span::{Span, Spanned};
use crate::tir::{Arg, Callee, Expr, ExprKind, IntTy, Intrinsic, Ty, Value};

use super::body::Checker;
use super::core;
use super::items::GenericKey;
use super::path::Resolution;
use super::scope::Def;

impl Checker<'_, '_> {
    /// `callee(args)`.
    pub(super) fn call(&mut self, callee: &Spanned<ast::Expr>, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let ast::Expr::Path { path, args: targs } = &callee.node else {
            // a value is called through its `$call`
            let value = self.expr(callee);
            return self.value_call(value, args, span);
        };
        // `T::f(…)`: a type's associated function, or a method called with
        // its receiver written first
        if let Some((entry, label)) = self.associated(path) {
            let checked: Vec<Expr> = args.iter().map(|a| self.expr(a)).collect();
            return self.call_method(entry, None, targs, checked, &label, span);
        }
        // `circuit::controlled(f)`, whose type depends on `f`'s
        if let [unit, name] = path.segments.as_slice()
            && targs.is_empty()
            && !self.cx.quantum
            && self.name(unit.node) == "circuit"
            && self.name(name.node) == "controlled"
        {
            return self.controlled_call(args, span);
        }
        // `invoke<Sig>(addr, args…)`, unless the program names something
        // `invoke` itself
        if let [name] = path.segments.as_slice()
            && self.name(name.node) == "invoke"
            && self.resolve(name.node).is_none()
        {
            return self.invoke_call(targs, args, span);
        }
        // `qcopy(&x)`, unless the program names something `qcopy` itself
        if let [name] = path.segments.as_slice()
            && targs.is_empty()
            && self.name(name.node) == "qcopy"
            && self.resolve(name.node).is_none()
        {
            return self.qcopy_call(args, span);
        }
        // `w(k, N)`, where nothing else is called `w`: e^(2πik/N)
        if let [name] = path.segments.as_slice()
            && targs.is_empty()
            && self.name(name.node) == "w"
            && self.resolve(name.node).is_none()
            && let [k, n] = args
        {
            return self.exact_root_call(k, n, span);
        }
        // `Some(x)`, `Ok(x)` and `Err(e)`, written without their enumeration
        if targs.is_empty()
            && let Some((key, variant)) = self.prelude_variant(path)
        {
            let payload = super::aggregate::VariantArgs::Positional(args);
            return self.generic_variant(key, variant, payload, span);
        }
        // a generic function called without its arguments written out: they
        // are worked out from the arguments given
        if targs.is_empty()
            && let Some((key, used)) = self.generic_path(&path.segments)
        {
            match path.segments[used..] {
                // `wrapping_add` and the rest on a `quint` are the `gates`
                // library's, which work on qubits
                [] if self.cx.is_generic_fn(key)
                    && self.cx.quantum
                    && let [name] = path.segments.as_slice()
                    && super::qbits::QUINT_COUNTERPARTS.contains(&self.name(name.node))
                    && !self.cx.specialized.contains(&key) =>
                {
                    let name = self.name(name.node).to_owned();
                    let checked: Vec<Expr> = args.iter().map(|a| self.expr(a)).collect();
                    return match self.quint_call(&name, checked, span) {
                        Ok(done) => done,
                        Err(checked) => self.generic_call_checked(key, checked, span),
                    };
                }
                [] if self.cx.is_generic_fn(key) => return self.generic_call(key, args, span),
                // a variant of a generic enumeration, its arguments deduced
                // from the values it carries
                [variant] => {
                    let payload = super::aggregate::VariantArgs::Positional(args);
                    return self.generic_variant(key, variant, payload, span);
                }
                _ => {}
            }
        }
        let Some(r) = self.resolve_path(path, targs, callee.span) else {
            self.check_only(args);
            return Checker::error(span);
        };
        let written = path.segments.iter().map(|s| self.name(s.node)).collect::<Vec<_>>().join("::");
        match r {
            Resolution::Value(Def::Fn(id)) if self.cx.unit.init == Some(id) => {
                self.check_only(args);
                let d = self.initializer_used(&written, callee.span);
                self.report(d);
                Checker::error(span)
            }
            Resolution::Value(Def::Fn(id)) => {
                self.cx.fn_sig(id);
                let f = self.cx.unit.func(id);
                let (params, ret, decl): (Vec<Ty>, Ty, Span) = (f.param_types().collect(), f.ret, f.span);
                match self.arguments(&written, &params, args, span, Some(decl)) {
                    Some(args) => Expr::call(Callee::Fn(id), args, ret, span),
                    None => Checker::error(span),
                }
            }
            Resolution::Value(Def::Extern(id)) => {
                let x = self.cx.unit.extern_fn(id);
                let (params, ret) = (x.params.clone(), x.ret);
                match self.arguments(&written, &params, args, span, None) {
                    Some(args) => Expr::call(Callee::Extern(id), args, ret, span),
                    None => Checker::error(span),
                }
            }
            Resolution::Value(Def::Intrinsic(Intrinsic::Abort(0))) => self.abort_call(args, span),
            Resolution::Value(Def::Intrinsic(which)) if core::is_generic(which) => {
                self.library_operation(which, args, span)
            }
            Resolution::Value(Def::Intrinsic(which)) => {
                let (params, ret) = core::signature(which, self.types_mut());
                let Some(args) = self.arguments(&written, &params, args, span, None) else {
                    return Checker::error(span);
                };
                Expr::intrinsic(which, args, ret, span)
            }
            Resolution::Value(Def::Local(_) | Def::Global(_) | Def::Capture(_)) => {
                // a function value, a closure, or a value whose type defines
                // `$call`, called through a reference or not
                let value = self.path_value(path, targs, callee.span);
                let t = self.shallow(value.ty);
                let base = self.types().as_ref(t).map_or(t, |(_, x)| x);
                let base = self.shallow(base);
                if matches!(base, Ty::Circuit(_)) {
                    return self.circuit_call(value, args, span);
                }
                if matches!(base, Ty::Fn(_) | Ty::Closure(_) | Ty::Adt(_) | Ty::Never) {
                    return self.value_call(value, args, span);
                }
                self.check_only(args);
                self.report(
                    Diagnostic::new(Code::Es07)
                        .with_message(format!("`{written}` is a variable, not a function, so it cannot be called"))
                        .at(callee.span)
                        .with_note("a local binding hides a function of the same name for the rest of its block"),
                );
                Checker::error(span)
            }
            Resolution::Variant(id, v) => self.variant_call(id, v, args, span),
            Resolution::Adt(id) => {
                let name = self.types().adt_name(id, self.interner);
                let help = if self.adt(id).is_struct() {
                    format!("a value of it is written `{name} {{ field: value, … }}`")
                } else {
                    format!("a value of it is one of its variants, such as `{name}::…`")
                };
                self.report(
                    Diagnostic::new(Code::Es07)
                        .with_message(format!("`{name}` is a type, and a type cannot be called"))
                        .at(callee.span)
                        .with_help(help),
                );
                Checker::error(span)
            }
        }
    }

    /// `invoke<Sig>(addr, args…)`: a call through a `*void` function address,
    /// such as a method's in a table, after checking when the program runs
    /// that the signature the address carries is `Sig`. A mismatch aborts
    /// with `RA11`: no other call through an address exists, so none can
    /// call code with the wrong arguments.
    fn invoke_call(&mut self, targs: &[Spanned<TArg>], args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        // the signature, a function type
        let sig = match targs {
            [Spanned {
                node: TArg::Type(t),
                ..
            }] => self.resolve_type(t).ty,
            _ => {
                self.check_only(args);
                self.report(
                    Diagnostic::new(Code::Es07)
                        .with_message("`invoke` takes the signature it calls as its one generic argument")
                        .at(span)
                        .with_help("write it as `invoke::<fn(*any) -> f64>(addr, …)`"),
                );
                return Checker::error(span);
            }
        };
        let Some((params, ret)) = self.types().as_sig(sig).filter(|_| matches!(sig, Ty::Fn(_))).map(|(p, r)| (p.to_vec(), r)) else {
            if sig != Ty::Never {
                let what = self.describe(sig);
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`invoke`'s generic argument is a function type, but this is {what}"))
                        .at(span),
                );
            }
            return Checker::error(span);
        };

        // the address, then the arguments for the signature
        let Some((addr, rest)) = args.split_first() else {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message("`invoke` takes the address to call, then its arguments")
                    .at(span),
            );
            return Checker::error(span);
        };
        let a = self.expr(addr);
        let void = self.types_mut().reference(crate::tir::Access::Write, Ty::Void);
        let a = self.coerce(a, void, "the address `invoke` calls");
        if !self.cx.need_table(sig, span) {
            return Checker::error(span);
        }
        let Some(checked) = self.arguments("invoke", &params, rest, span, None) else {
            return Checker::error(span);
        };

        // the code at the address, once its signature is checked, called with them
        let code = Expr::intrinsic(Intrinsic::Invoke(sig), vec![a], sig, span);
        Expr {
            kind: ExprKind::IndirectCall {
                callee: Box::new(code),
                args: checked,
            },
            ty: ret,
            span,
        }
    }

    /// A call of a generic function whose arguments are not written: the
    /// arguments given are checked, what each parameter must be is deduced
    /// from their types, and the instance that makes is called.
    fn generic_call(&mut self, key: GenericKey, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        if self.cx.specialized.contains(&key) {
            let Some(decl) = self.cx.generic_fn_item(key) else {
                return Checker::error(span);
            };
            let text = self.cx.generic_label(key);
            return self.specialized_call(key, &text, decl, args, span);
        }

        // the arguments, and each generic parameter deduced from their types
        let checked: Vec<Expr> = args.iter().map(|a| self.expr(a)).collect();
        self.generic_call_checked(key, checked, span)
    }

    /// [`Self::generic_call`] with the arguments already checked.
    pub(super) fn generic_call_checked(&mut self, key: GenericKey, checked: Vec<Expr>, span: Span) -> Expr {
        let Some(params) = self.cx.generic_params(key).map(<[ast::GenericParam]>::to_vec) else {
            return Checker::error(span);
        };
        let Some(decl) = self.cx.generic_fn_item(key) else {
            return Checker::error(span);
        };
        let text = self.cx.generic_label(key);
        if checked.len() != decl.params.len() {
            self.report(Checker::wrong_arity(&text, decl.params.len(), checked.len(), span));
            return Checker::error(span);
        }
        // a literal that nothing has settled yet would deduce a parameter to
        // be an undecided type, so it takes the type it would end up with
        let found: Vec<Ty> = checked.iter().map(|e| self.settled(e.ty)).collect();
        let subst = super::generic::deduce(self.cx, &params, &decl.params, &found);
        let mut resolved = Vec::with_capacity(params.len());
        for p in &params {
            match super::generic::lookup(&subst, p.name.node) {
                Some(a) => resolved.push(a),
                None => {
                    let d = super::generic::undeduced(&text, self.name(p.name.node), span);
                    self.report(d);
                    return Checker::error(span);
                }
            }
        }

        // the instance, called with the arguments
        let Some(id) = self.cx.instantiate_fn(key, resolved, span) else {
            return Checker::error(span);
        };
        self.cx.use_callee(Callee::Fn(id), span);
        let f = self.cx.unit.func(id);
        let (ptys, ret): (Vec<Ty>, Ty) = (f.param_types().collect(), f.ret);
        let args = checked
            .into_iter()
            .zip(&ptys)
            .enumerate()
            .map(|(i, (a, &p))| self.coerce(a, p, &format!("argument {} of `{text}`", i + 1)))
            .collect();
        Expr::call(Callee::Fn(id), args, ret, span)
    }

    /// A call of a function whose parameters mix `const` and ordinary ones:
    /// the `const` arguments are computed now, the specialisation for their
    /// values is made, and it is called with the others.
    fn specialized_call(
        &mut self,
        key: GenericKey,
        text: &str,
        decl: &ast::FnItem,
        args: &[Spanned<ast::Expr>],
        span: Span,
    ) -> Expr {
        if args.len() != decl.params.len() {
            self.check_only(args);
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!(
                        "`{text}` takes {} arguments, but this call gives {}",
                        decl.params.len(),
                        args.len()
                    ))
                    .at(span),
            );
            return Checker::error(span);
        }

        // the `constexpr` arguments' values, and the others set aside
        let mut consts = Vec::new();
        let mut rest = Vec::new();
        for (p, a) in decl.params.iter().zip(args) {
            if p.ty.node.is_constexpr() {
                let ty = self.resolve_type(&p.ty).ty;
                let role = format!("the `constexpr` argument `{}` of `{text}`", self.name(p.name.node));
                match self.cx.const_value(a, ty, &role).and_then(|v| v.as_int()) {
                    Some(n) => consts.push(Arg::Const(n)),
                    None => return Checker::error(span),
                }
            } else {
                rest.push(a);
            }
        }

        // the specialisation for those values, called with the others
        let Some(id) = self.cx.instantiate_fn(key, consts, span) else {
            return Checker::error(span);
        };
        self.cx.use_callee(Callee::Fn(id), span);
        let f = self.cx.unit.func(id);
        let (ptys, ret): (Vec<Ty>, Ty) = (f.param_types().collect(), f.ret);
        let args = rest
            .into_iter()
            .zip(ptys)
            .enumerate()
            .map(|(i, (a, p))| {
                let e = self.expr_for(a, p);
                self.coerce(e, p, &format!("argument {} of `{text}`", i + 1))
            })
            .collect();
        Expr::call(Callee::Fn(id), args, ret, span)
    }

    fn arguments(
        &mut self,
        name: &str,
        params: &[Ty],
        args: &[Spanned<ast::Expr>],
        span: Span,
        decl: Option<Span>,
    ) -> Option<Vec<Expr>> {
        let checked: Vec<Expr> = args
            .iter()
            .enumerate()
            .map(|(i, a)| match params.get(i) {
                Some(&p) => self.expr_for(a, p),
                None => self.expr(a),
            })
            .collect();
        if checked.len() != params.len() {
            let mut d = Checker::wrong_arity(name, params.len(), checked.len(), span);
            if let Some(decl) = decl {
                d = d.also(decl, format!("`{name}` is declared here"));
            }
            self.report(d);
            return None;
        }
        Some(
            checked
                .into_iter()
                .zip(params)
                .enumerate()
                .map(|(i, (a, &p))| self.coerce(a, p, &format!("argument {} of `{name}`", i + 1)))
                .collect(),
        )
    }

    /// `receiver.len()` on a fixed array or a slice, which have no other
    /// method.
    pub(super) fn len_call(
        &mut self,
        receiver: Expr,
        targs: &[Spanned<TArg>],
        args: &[Spanned<ast::Expr>],
        span: Span,
    ) -> Expr {
        if !targs.is_empty() {
            self.check_only(args);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("`len` takes no generic arguments")
                    .at(span)
                    .with_note("the length of an array or slice is a `usize` whatever its elements are")
                    .with_help("write `.len()`"),
            );
            return Checker::error(span);
        }
        let base = self.auto_deref(receiver);
        if !args.is_empty() {
            self.check_only(args);
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message("`len` takes no arguments")
                    .at(span),
            );
            return Checker::error(span);
        }
        let bt = self.shallow(base.ty);
        match self.types().as_array(bt) {
            Some((_, n)) if is_pure_place(&base) => Expr::constant(Value::Int(i128::from(n), IntTy::USIZE), Ty::USIZE, span),
            _ => Expr::intrinsic(Intrinsic::Len, vec![base], Ty::USIZE, span),
        }
    }
}

/// Whether evaluating `e` has no effect beyond reading, so that it may be
/// skipped when only its type matters.
fn is_pure_place(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Local(_) | ExprKind::Global(_) | ExprKind::Const(_) => true,
        ExprKind::Field { base, .. } | ExprKind::Deref(base) => is_pure_place(base),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};
    use crate::tir::{ExprKind, Value};

    #[test]
    fn calls_check_arity_and_argument_types() {
        assert_eq!(codes("fn g(a: i32) { }\nfn f() { g(1, 2); }"), [Code::Es07]);
        assert_eq!(codes("fn g(a: i32) { }\nfn f() { g(true); }"), [Code::Es06]);
        check("fn g(a: i32) -> i32 { a }\nfn f() -> i32 { g(1) }");
    }

    #[test]
    fn a_function_may_be_called_before_its_declaration() {
        check("fn f() -> i32 { g() }\nfn g() -> i32 { 1 }");
    }

    #[test]
    fn calling_a_variable_is_refused() {
        assert_eq!(codes("fn f(x: i32) { x(); }"), [Code::Es07]);
    }

    #[test]
    fn an_unknown_name_suggests_a_close_one() {
        assert_eq!(codes("fn f() -> i32 { let total = 1; totl }"), [Code::Es04]);
    }

    #[test]
    fn an_argument_converts_to_a_weaker_reference() {
        check("fn g(s: *[u8]) { }\nfn f() { let a = [1u8, 2]; g(&a); }");
    }

    #[test]
    fn len_of_an_array_is_its_declared_length() {
        let u = check("fn f(a: [bool; 7]) -> usize { a.len() }");
        let v = u.fns[0].body.value.as_ref().unwrap();
        assert!(matches!(v.kind, ExprKind::Const(Value::Int(7, _))), "{:?}", v.kind);
        check("fn f(s: *[bool]) -> usize { s.len() }");
        check("fn f(a: *[bool; 2]) -> usize { a.len() }");
        assert_eq!(codes("fn f(x: i32) -> usize { x.len() }"), [Code::Es04]);
    }

    #[test]
    fn sizes_and_alignments_follow_the_layout_rules() {
        let u = check(
            "struct H { tag: u8, len: u32 }\n\
             let S: const usize = @sizeof(H);\n\
             let A: const usize = @alignof(H);",
        );
        assert_eq!(u.globals[0].value, Some(Value::Int(8, crate::tir::IntTy::USIZE)));
        assert_eq!(u.globals[1].value, Some(Value::Int(4, crate::tir::IntTy::USIZE)));
    }

    #[test]
    fn introspection_macros_answer_during_translation() {
        assert_eq!(codes("fn f() -> bool { @is_quantum(u8) }"), []);
        assert_eq!(codes("fn f() -> bool { @made_up(u8) }"), [Code::Es04]);
        assert_eq!(codes("fn f() { @static_assert(1 > 2, \"never\"); }"), [Code::Em01]);
        assert_eq!(codes("fn f() -> bool { @target_has(\"teleport\") }"), [Code::Em04]);
    }
}
