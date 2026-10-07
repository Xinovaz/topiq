//! Generic items, and the instances a program makes of them.
//!
//! A generic declaration is not itself a type or a function. `struct Pair<A,
//! B>` describes how to make a type once `A` and `B` are known, and
//! `fn sum<T>(…)` how to make a function. Each set of arguments produces one
//! **instance** (an ordinary structure or an ordinary function, with its
//! arguments recorded on it), and everything after this point works on
//! instances alone. Nothing downstream knows generics exist.
//!
//! # Instances
//!
//! There are no trait bounds in this language. A body may write `a + b` on a
//! parameter's type and a `where` clause may ask any structural question about
//! it, so what a generic body means is only decided once its arguments are
//! known. A definition is therefore checked once per instance, with the
//! parameters bound, and a body that is wrong for one set of arguments is
//! reported at the place that asked for it.
//!
//! # Working out the arguments
//!
//! `sum::<i32>(xs)` says them. `sum(xs)` does not, so they are deduced: the
//! written parameter types are matched against the types of the arguments
//! given, and a parameter is bound the first time it meets one. `xs: *[T]`
//! against a `*[i32]` binds `T` to `i32`; `r: [T; N]` against a `[u8; 4]`
//! binds `N` to 4 as well. A parameter no argument mentions cannot be
//! deduced, and the diagnostic asks for it to be written.

use crate::ast::{self, GenericParam, TArg};
use crate::diag::{Code, Diagnostic, Limit};
use crate::intern::{Interner, Symbol};
use crate::span::{Span, Spanned};
use crate::tir::{Arg, Ty, TypeTable};

use super::items::UnitCx;

/// What each generic parameter of one instance stands for.
pub type Subst = Vec<(Symbol, Arg)>;

/// The argument bound to `name`.
pub fn lookup(subst: &Subst, name: Symbol) -> Option<Arg> {
    subst.iter().find(|(n, _)| *n == name).map(|(_, a)| *a)
}

/// An argument as a program would write it: `i32`, `Point`, `4`.
pub fn arg_text(a: Arg, types: &TypeTable, interner: &Interner) -> String {
    match a {
        Arg::Type(t) => types.display(t, interner),
        Arg::Const(v) => v.to_string(),
    }
}

/// An instance's name as a program would write it: `sum<i32>`.
pub fn instance_name(base: &str, args: &[Arg], types: &TypeTable, interner: &Interner) -> String {
    if args.is_empty() {
        return base.to_owned();
    }
    let parts: Vec<String> = args.iter().map(|&a| arg_text(a, types, interner)).collect();
    format!("{base}<{}>", parts.join(", "))
}

impl UnitCx<'_> {
    /// Turns written generic arguments into the arguments of an instance,
    /// checking that there are as many as the declaration has parameters and
    /// that each is of the right sort.
    ///
    /// Returns `None` when something was wrong, which has been reported.
    pub fn generic_args(
        &mut self,
        what: &str,
        params: &[GenericParam],
        written: &[Spanned<TArg>],
        span: Span,
    ) -> Option<Vec<Arg>> {
        if written.len() != params.len() {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!(
                        "`{what}` has {} generic parameter{}, and {} {} given here",
                        params.len(),
                        if params.len() == 1 { "" } else { "s" },
                        written.len(),
                        if written.len() == 1 { "is" } else { "are" }
                    ))
                    .at(span),
            );
            return None;
        }
        let mut args = Vec::with_capacity(params.len());
        for (p, w) in params.iter().zip(written) {
            let wants_const = p.ty.is_some();
            let arg = match (&w.node, wants_const) {
                (TArg::Type(t), false) => Arg::Type(self.written_type(t)?),
                (TArg::Const(e), true) => Arg::Const(i128::from(self.const_usize(e, "a generic argument")?)),
                // `Pair<4>` where a type is wanted, or `Reg<u8>` where a number
                // is: an argument that parses as a path may be meant as a
                // constant, so the other reading is tried before reporting
                (TArg::Type(t), true) => {
                    let e = type_as_expr(t)?;
                    Arg::Const(i128::from(self.const_usize(&e, "a generic argument")?))
                }
                (TArg::Const(e), false) => {
                    let name = self.interner.resolve(p.name.node);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`{name}` is a type parameter, and this is a value"))
                            .at(e.span),
                    );
                    return None;
                }
            };
            args.push(arg);
        }
        Some(args)
    }

    /// Binds a declaration's parameters to an instance's arguments.
    pub fn substitution(&self, params: &[GenericParam], args: &[Arg]) -> Subst {
        params
            .iter()
            .zip(args)
            .map(|(p, &a)| (p.name.node, a))
            .collect()
    }

    /// Whether making one more instance would pass the limit on how many may
    /// be under way at once, which is what a generic that instantiates
    /// itself with a new argument does forever.
    pub fn too_deep(&mut self, span: Span) -> bool {
        let depth = self.generic_depth;
        if Limit::GenericDepth.permits(depth) {
            return false;
        }
        let d = Limit::GenericDepth.exceeded(span, depth).with_note(
            "a generic item that instantiates itself with a new argument each time never finishes",
        );
        self.report(d);
        true
    }
}

/// Works out what each generic parameter must be, by matching the written
/// parameter types against the types of the arguments given.
///
/// Every parameter must be settled; the caller reports the ones that are not.
pub fn deduce(
    cx: &mut UnitCx<'_>,
    generics: &[GenericParam],
    params: &[ast::Param],
    args: &[Ty],
) -> Subst {
    let names: Vec<Symbol> = generics.iter().map(|g| g.name.node).collect();
    let mut out: Subst = Vec::new();
    for (p, &a) in params.iter().zip(args) {
        match_type(cx, &p.ty.node, a, &names, &mut out);
    }
    out
}

/// Matches one written type against the type an argument turned out to have,
/// binding any parameter it names.
pub fn match_written(
    cx: &mut UnitCx<'_>,
    written: &ast::Type,
    found: Ty,
    names: &[Symbol],
    out: &mut Subst,
) {
    match_type(cx, written, found, names, out);
}

fn match_type(cx: &mut UnitCx<'_>, written: &ast::Type, found: Ty, names: &[Symbol], out: &mut Subst) {
    match written {
        ast::Type::Const(inner) | ast::Type::Constexpr(inner) => match_type(cx, &inner.node, found, names, out),
        ast::Type::Path { path, args } => {
            let Some(name) = path.is_simple().then(|| path.last()).flatten() else {
                return;
            };
            if args.is_empty() && names.contains(&name) {
                if lookup(out, name).is_none() {
                    out.push((name, Arg::Type(found)));
                }
                return;
            }
            // `Pair<A, B>` against an instance of `Pair`: match argument for
            // argument
            let Ty::Adt(id) = found else { return };
            let def_args = cx.unit.types.adt(id).args.clone();
            if def_args.len() != args.len() {
                return;
            }
            for (w, &a) in args.iter().zip(&def_args) {
                match (&w.node, a) {
                    (TArg::Type(t), Arg::Type(x)) => match_type(cx, &t.node, x, names, out),
                    (TArg::Type(t), Arg::Const(v)) => match_const(&t.node, v, names, out),
                    _ => {}
                }
            }
        }
        ast::Type::Ref { target, .. } => {
            let types = &cx.unit.types;
            if let Some((_, t)) = types.as_ref(found).or_else(|| types.as_slice(found)) {
                // a slice's element type is what stands inside the brackets;
                // a reference to a fixed array given for one gives its
                // elements', since it converts to a slice of them
                match &target.node {
                    ast::Type::Array { elem, len: None } => {
                        let e = match types.as_ref(found) {
                            Some(_) => types.as_array(t).map_or(t, |(e, _)| e),
                            None => t,
                        };
                        match_type(cx, &elem.node, e, names, out);
                    }
                    // a slice is a reference to a growable array's elements,
                    // so what `*T` refers to there is the array: `T` is
                    // `[X]` for a `*[X]`
                    other if types.as_slice(found).is_some() => {
                        let whole = cx.unit.types.growable(t);
                        match_type(cx, other, whole, names, out);
                    }
                    other => match_type(cx, other, t, names, out),
                }
            }
        }
        ast::Type::Array { elem, len: None } => {
            if let Some(e) = cx.unit.types.as_growable(found) {
                match_type(cx, &elem.node, e, names, out);
            }
        }
        ast::Type::Array { elem, len } => {
            let Some((e, n)) = cx.unit.types.as_array(found) else {
                return;
            };
            match_type(cx, &elem.node, e, names, out);
            if let Some(len) = len {
                match_length(&len.node, i128::from(n), names, out);
            }
        }
        // `fn(A) -> B` or `closure<fn(A) -> B>` against a function or a
        // closure: parameter for parameter, then the result. a function may
        // stand where a closure is wanted, so either matches either
        ast::Type::Fn { .. } | ast::Type::Closure(_) => {
            let Some((params, ret)) = written_sig(written) else {
                return;
            };
            let Some((ps, r)) = cx.unit.types.as_sig(found).map(|(p, r)| (p.to_vec(), r)) else {
                return;
            };
            if ps.len() != params.len() {
                return;
            }
            for (w, t) in params.iter().zip(ps) {
                match_type(cx, &w.node, t, names, out);
            }
            if let Some(w) = ret {
                match_type(cx, &w.node, r, names, out);
            }
        }
        // `circuit<S>` against a circuit handle: `S` is its operator's
        // function type; `circuit<fn(A) -> B>` matches as the signature does
        ast::Type::Circuit(inner) => {
            if !matches!(found, Ty::Circuit(_)) {
                return;
            }
            let Some((ps, r)) = cx.unit.types.as_sig(found).map(|(p, r)| (p.to_vec(), r)) else {
                return;
            };
            let sig = cx.unit.types.function(ps, r);
            match_type(cx, &inner.node, sig, names, out);
        }
        ast::Type::Tuple(ws) => {
            let Some(ts) = cx.unit.types.as_tuple(found).map(<[Ty]>::to_vec) else {
                return;
            };
            if ts.len() != ws.len() {
                return;
            }
            for (w, t) in ws.iter().zip(ts) {
                match_type(cx, &w.node, t, names, out);
            }
        }
        _ => {}
    }
}

/// The parameter and return types written in `fn(…) -> U` or
/// `closure<fn(…) -> U>`.
type WrittenSig<'t> = (&'t [Spanned<ast::Type>], Option<&'t Spanned<ast::Type>>);

fn written_sig(t: &ast::Type) -> Option<WrittenSig<'_>> {
    match t {
        ast::Type::Fn { params, ret } => Some((params, ret.as_deref())),
        ast::Type::Closure(inner) => written_sig(&inner.node),
        _ => None,
    }
}

/// Binds a `const` parameter written where a type argument stands.
fn match_const(written: &ast::Type, value: i128, names: &[Symbol], out: &mut Subst) {
    if let ast::Type::Path { path, args } = written
        && args.is_empty()
    {
        bind_const(path, value, names, out);
    }
}

/// Binds the `const` parameter a bare path names, if it names one not yet
/// bound.
fn bind_const(path: &ast::Path, value: i128, names: &[Symbol], out: &mut Subst) {
    if let Some(name) = path.is_simple().then(|| path.last()).flatten()
        && names.contains(&name)
        && lookup(out, name).is_none()
    {
        out.push((name, Arg::Const(value)));
    }
}

/// Binds a `const` parameter written as an array's length.
fn match_length(written: &ast::Expr, value: i128, names: &[Symbol], out: &mut Subst) {
    if let ast::Expr::Path { path, args } = written
        && args.is_empty()
    {
        bind_const(path, value, names, out);
    }
}

/// A type argument read as the constant expression it also parses as, for a
/// `const` parameter given a bare name or number.
fn type_as_expr(t: &Spanned<ast::Type>) -> Option<Spanned<ast::Expr>> {
    match &t.node {
        ast::Type::Path { path, args } if args.is_empty() => Some(Spanned::new(
            ast::Expr::Path {
                path: path.clone(),
                args: Vec::new(),
            },
            t.span,
        )),
        _ => None,
    }
}

/// A parameter that nothing settled.
pub fn undeduced(what: &str, name: &str, span: Span) -> Diagnostic {
    Diagnostic::new(Code::Es07)
        .with_message(format!(
            "the arguments of `{what}` do not say what `{name}` is"
        ))
        .at(span)
        .with_note("a generic parameter is worked out from the arguments, and none of them mentions this one")
        .with_help(format!("write it, as in `{what}::<…>(…)`"))
}

/// A parameter that none of a structure literal's fields settled.
pub fn undeduced_field(what: &str, name: &str, span: Span) -> Diagnostic {
    Diagnostic::new(Code::Es07)
        .with_message(format!(
            "the fields of this `{what}` do not say what `{name}` is"
        ))
        .at(span)
        .with_note(
            "a structure literal has nowhere to write generic arguments, so they are worked out \
             from the values given for the fields",
        )
        .with_help(format!("give a value whose type mentions `{name}`"))
}

/// A generic item named without its arguments.
pub fn needs_arguments(what: &str, kind: &str, params: usize, span: Span, in_expression: bool) -> Diagnostic {
    let placeholders = vec!["_"; params].join(", ");
    // in an expression, `<` is a comparison, so arguments follow `::`
    let written = if in_expression {
        format!("{what}::<{placeholders}>")
    } else {
        format!("{what}<{placeholders}>")
    };
    Diagnostic::new(Code::Es06)
        .with_message(format!("`{what}` is a generic {kind}, so it needs its arguments here"))
        .at(span)
        .with_note(format!(
            "a generic declaration describes how to make a {kind} once its arguments are known; it is not one itself"
        ))
        .with_help(format!("write them, as in `{written}`"))
}
