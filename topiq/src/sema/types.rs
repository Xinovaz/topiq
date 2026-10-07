//! Turning a written type into a [`Ty`].
//!
//! The primitive types (`i32`, `bool`, `char` and the rest) are not
//! keywords. They are names, found in the type name space like any other type
//! would be, which is why a misspelt one is reported as a name that was not
//! found rather than as a syntax error.
//!
//! # Where `const` may appear
//!
//! `const` and `constexpr` at the outside of a type are peeled off here and
//! reported separately, as [`Resolved::constant`] and
//! [`Resolved::constexpr`]: whether a value is ever assigned, and whether it
//! must be known during translation, are properties of the binding that holds
//! it. `const const T` is simply `const T`, and `constexpr T` is `const` too.
//!
//! Behind a reference `const` means something else: `*const T` refers to a
//! constant, which it may only read, and that becomes the reference's
//! [`Access::Const`]. A plain `*T` may read and write its referent. There is
//! no `*constexpr T`, since a reference is an address known only when the
//! program runs. Anywhere else (`[const u8; 4]`, a structure field of type
//! `const u8`), a `const` would describe part of a value, which nothing in the
//! language gives a meaning to. Those are refused, naming the position.
//!
//! # Arrays
//!
//! `[T; N]` needs `N`'s value, which may be written with constants declared
//! anywhere in the unit, constant functions, or `@sizeof`. It is evaluated on
//! the spot (see [`super::items`] for how that works when the constants it
//! names have not been looked at yet). `*[T]` is a slice: an address and a
//! count. `[T]` on its own is a growable array that owns its storage; a
//! reference to one is a slice of its elements, so `*[T]` means the same
//! whether what it views is fixed or growable.

use crate::ast::{Path, Type};
use crate::diag::{Code, Diagnostic};
use crate::span::{Span, Spanned};
use crate::tir::{Access, FloatTy, IntTy, Ty};

use super::items::UnitCx;
use super::report;

/// A written type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Resolved {
    /// The type.
    pub ty: Ty,
    /// Whether it was written `const T` or `constexpr T`: a value that is
    /// never assigned.
    pub constant: bool,
    /// Whether it was written `constexpr T`: a value that must be known
    /// during translation.
    pub constexpr: bool,
}

/// The names of every primitive type.
pub const PRIMITIVES: [&str; 14] = [
    "i8", "i16", "i32", "i64", "isize", "u8", "u16", "u32", "u64", "usize", "f32", "f64", "bool",
    "char",
];

type Found = Result<Ty, Box<Diagnostic>>;

/// Resolves a written type.
///
/// # Errors
///
/// A diagnostic when the name denotes no type (`ES04`), denotes quantum
/// storage in a classical unit (`EU02`), or names a quantum type the compiler does
/// not translate (`TQ003`).
pub fn resolve(cx: &mut UnitCx<'_>, t: &Spanned<Type>) -> Result<Resolved, Box<Diagnostic>> {
    let (mut constant, mut constexpr) = (false, false);
    let mut t = t;
    loop {
        match &t.node {
            Type::Const(inner) => t = inner,
            Type::Constexpr(inner) => {
                constexpr = true;
                t = inner;
            }
            _ => break,
        }
        constant = true;
    }
    let ty = inner(cx, t)?;
    if constant && cx.unit.types.is_quantum(ty) {
        return Err(Box::new(
            Diagnostic::new(Code::Ec01)
                .with_message(format!(
                    "`{}` holds qubits, so it cannot be `const`",
                    cx.unit.types.display(ty, cx.interner)
                ))
                .at(t.span)
                .with_note(
                    "a qubit's state changes as operations act on it, which a `const` binding \
                     would forbid, and exists only while its circuit runs",
                )
                .with_help("drop `const`"),
        ));
    }
    Ok(Resolved { ty, constant, constexpr })
}

/// Resolves a type in a position where `const` has no meaning.
fn inner(cx: &mut UnitCx<'_>, t: &Spanned<Type>) -> Found {
    let span = t.span;
    let unsupported = |what: &str| Err(Box::new(report::unsupported(span, what)));
    match &t.node {
        Type::Const(_) | Type::Constexpr(_) => Err(Box::new(nested_const(span))),
        Type::Void => Ok(Ty::Void),
        Type::Path { path, args } => named(cx, path, args, span),
        Type::Ref { target } => reference(cx, target),
        Type::Array { elem, len: Some(len) } => {
            let e = element(cx, elem)?;
            // a length that could not be computed has been reported; the
            // array then stands in as `never`, like any other ill-formed part
            let Some(n) = cx.const_usize(len, "the length of an array") else {
                return Ok(Ty::Never);
            };
            if e == Ty::Qubit
                && let Some(d) = crate::diag::Limit::RegisterQubits.check(u32::try_from(n).unwrap_or(u32::MAX), span)
            {
                return Err(Box::new(d));
            }
            Ok(cx.unit.types.array(e, n))
        }
        Type::Array { elem, len: None } => {
            let e = element(cx, elem)?;
            Ok(cx.unit.types.growable(e))
        }
        Type::Tuple(elems) => {
            let mut ts = Vec::with_capacity(elems.len());
            for e in elems {
                ts.push(element(cx, e)?);
            }
            Ok(cx.unit.types.tuple(ts))
        }
        Type::Fn { params, ret } => {
            let (params, ret) = signature(cx, params, ret.as_deref())?;
            Ok(cx.unit.types.function(params, ret))
        }
        Type::Closure(sig) => match &sig.node {
            Type::Fn { params, ret } => {
                let (params, ret) = signature(cx, params, ret.as_deref())?;
                Ok(cx.unit.types.closure(params, ret))
            }
            _ => Err(Box::new(
                Diagnostic::new(Code::Es06)
                    .with_message("a closure type names the function it can be called as")
                    .at(sig.span)
                    .with_help("write it as `closure<fn(T, …) -> U>`"),
            )),
        },
        Type::Macro { name, args } => cx.macro_type(*name, args, span),
        Type::Circuit(sig) => match &sig.node {
            Type::Fn { params, ret } => {
                cx.in_circuit += 1;
                let resolved = signature(cx, params, ret.as_deref());
                cx.in_circuit -= 1;
                let (params, ret) = resolved?;
                Ok(cx.unit.types.circuit(params, ret))
            }
            // a name standing for a function type, as a generic parameter
            // does once it is known
            _ => {
                let t = inner(cx, sig)?;
                match (t, cx.unit.types.as_sig(t)) {
                    (Ty::Fn(_), Some((params, ret))) => {
                        let params = params.to_vec();
                        Ok(cx.unit.types.circuit(params, ret))
                    }
                    _ => Err(Box::new(
                        Diagnostic::new(Code::Es06)
                            .with_message("a circuit handle's type names the signature of its operator")
                            .at(sig.span)
                            .with_help("write it as `circuit<fn(T, …) -> U>`"),
                    )),
                }
            }
        },
        Type::Qmap { key, value } => {
            if !cx.quantum {
                return Err(Box::new(
                    Diagnostic::new(Code::Eu02)
                        .with_message("a classical unit cannot hold a map locale")
                        .at(span)
                        .with_note("a map locale's entries are states and operators of a quantum unit's circuits")
                        .with_help("move this into a unit that begins with `#unit quantum`"),
                ));
            }
            let k = inner(cx, key)?;
            let width = match k {
                Ty::Qubit => 1,
                _ => match cx.unit.types.as_array(k) {
                    Some((Ty::Qubit, n)) => n,
                    _ => {
                        let what = report::describe(k, &cx.unit.types, cx.interner);
                        return Err(Box::new(
                            Diagnostic::new(Code::Es06)
                                .with_message(format!("a map locale's key is a register, `qubit` or `[qubit; N]`, and this is {what}"))
                                .at(key.span),
                        ));
                    }
                },
            };
            let v = inner(cx, value)?;
            if let Some(d) = super::qmap::entry_problem(&cx.unit.types, cx.interner, v, value.span) {
                return Err(Box::new(d));
            }
            Ok(cx.unit.types.qmap(width, v))
        }
        Type::Dyn => Ok(Ty::Dyn),
        Type::Restriction { .. } => Err(Box::new(not_a_value_type(span, "a restriction `A of B`"))),
        // `T?` is `Opt<T>`, the `core` library's
        Type::Optional(t) => {
            let inner = element(cx, t)?;
            let opt = cx.interner.get("Opt").and_then(|s| cx.generic(s));
            match opt {
                Some(key) => Ok(cx
                    .instantiate_adt(key, vec![crate::tir::Arg::Type(inner)], span)
                    .map_or(Ty::Never, Ty::Adt)),
                None => unsupported("optional types `T?` without the `core` library"),
            }
        }
    }
}

/// The parameter types and return type of `fn(T…) -> U`; `void` when no
/// return type is written, as for a function.
fn signature(
    cx: &mut UnitCx<'_>,
    params: &[Spanned<Type>],
    ret: Option<&Spanned<Type>>,
) -> Result<(Vec<Ty>, Ty), Box<Diagnostic>> {
    let mut ps = Vec::with_capacity(params.len());
    for p in params {
        let t = inner(cx, p)?;
        if t == Ty::Void {
            return Err(Box::new(super::items::void_binding(p.span)));
        }
        ps.push(t);
    }
    let r = match ret {
        Some(r) => inner(cx, r)?,
        None => Ty::Void,
    };
    Ok((ps, r))
}

/// The element type of an array or slice.
fn element(cx: &mut UnitCx<'_>, elem: &Spanned<Type>) -> Found {
    let e = inner(cx, elem)?;
    if e == Ty::Void {
        return Err(Box::new(
            Diagnostic::new(Code::Es06)
                .with_message("an array of `void` has nothing to hold")
                .at(elem.span)
                .with_note("`void` has no values, so it can only be a return type"),
        ));
    }
    Ok(e)
}

/// `*T` and `*const T`, and the slices `*[T]` and `*const [T]`.
fn reference(cx: &mut UnitCx<'_>, target: &Spanned<Type>) -> Found {
    if target.node.is_constexpr() {
        return Err(Box::new(
            Diagnostic::new(Code::Es06)
                .with_message("a reference cannot be `*constexpr`")
                .at(target.span)
                .with_note(
                    "a reference is an address, which exists only when the program runs; what \
                     it refers to may be constant, but not known during translation through it",
                )
                .with_help("write `*const T` for a reference that may only read"),
        ));
    }
    let (access, target) = match &target.node {
        Type::Const(inner) => {
            let mut t = inner.as_ref();
            while let Type::Const(i) = &t.node {
                t = i;
            }
            (Access::Const, t)
        }
        _ => (Access::Write, target),
    };
    match &target.node {
        // `*[T]`: a slice
        Type::Array { elem, len: None } => {
            let e = element(cx, elem)?;
            Ok(cx.unit.types.slice(access, e))
        }
        // `*void` and `*any`: a reference whose referent's type is known only
        // from the type information it carries. the two are one type here
        Type::Void => Ok(cx.unit.types.reference(access, Ty::Void)),
        Type::Path { path, args }
            if args.is_empty()
                && path.is_simple()
                && path.last().is_some_and(|s| cx.interner.resolve(s) == "any")
                && path.last().and_then(|s| cx.lookup_type(s)).is_none() =>
        {
            Ok(cx.unit.types.reference(access, Ty::Void))
        }
        _ => {
            let t = inner(cx, target)?;
            Ok(cx.unit.types.reference(access, t))
        }
    }
}

/// A type written as a name, `u8` or `Point` or `geometry::Point`, with its
/// generic arguments if it has any.
fn named(cx: &mut UnitCx<'_>, path: &Path, args: &[Spanned<crate::ast::TArg>], span: Span) -> Found {
    let t = named_as_written(cx, path, args, span)?;
    let parameter = matches!(path.segments.as_slice(), [name] if cx.param_arg(name.node).is_some());
    if let Ty::Adt(id) = t
        && !parameter
    {
        cx.use_adt(id, span);
    }
    Ok(t)
}

/// The type `path` names.
fn named_as_written(cx: &mut UnitCx<'_>, path: &Path, args: &[Spanned<crate::ast::TArg>], span: Span) -> Found {
    // a generic parameter of the instance being checked stands for whatever
    // that instance was made with
    if let [name] = path.segments.as_slice()
        && let Some(a) = cx.param_arg(name.node)
    {
        return match a {
            crate::tir::Arg::Type(t) => Ok(t),
            crate::tir::Arg::Const(_) => Err(Box::new(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`{}` is a value, not a type", cx.interner.resolve(name.node)))
                    .at(span),
            )),
        };
    }
    if let Some((key, used)) = cx.generic_at(&path.segments)
        && used == path.segments.len()
    {
        let text = cx.generic_label(key);
        let params: Vec<crate::ast::GenericParam> = cx.generic_params(key).unwrap_or_default().to_vec();
        if args.is_empty() {
            let kind = if cx.is_generic_fn(key) { "function" } else { "type" };
            return Err(Box::new(super::generic::needs_arguments(&text, kind, params.len(), span, false)));
        }
        let Some(resolved) = cx.generic_args(&text, &params, args, span) else {
            return Ok(Ty::Never);
        };
        if let Some(t) = cx.instantiate_alias(key, resolved.clone(), span) {
            return Ok(t);
        }
        return match cx.instantiate_adt(key, resolved, span) {
            Some(id) => Ok(Ty::Adt(id)),
            None => Ok(Ty::Never),
        };
    }
    match path.segments.as_slice() {
        [name] => {
            let sym = name.node;
            if !args.is_empty() {
                if let Some(what) = cx.scope.unsupported_type(sym) {
                    return Err(Box::new(report::unsupported(span, what)));
                }
                let text = cx.interner.resolve(sym);
                return Err(Box::new(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("`{text}` takes no generic arguments"))
                        .at(span),
                ));
            }
            let text = cx.interner.resolve(sym);
            if let Some(i) = IntTy::from_name(text) {
                return Ok(Ty::Int(i));
            }
            if let Some(f) = FloatTy::from_name(text) {
                return Ok(Ty::Float(f));
            }
            match text {
                "bool" => return Ok(Ty::Bool),
                "char" => return Ok(Ty::Char),
                "qubit" if cx.quantum || cx.in_circuit > 0 => return Ok(Ty::Qubit),
                "qubit" => return Err(Box::new(quantum_in_classical(span))),
                _ => {}
            }
            // a unit's own alias hides a structure or enumeration of `core`
            // of the same name, as its other declarations hide the library's
            let own_alias = cx.declares_alias(sym) && cx.local_type(sym).is_none();
            if !own_alias && let Some(id) = cx.lookup_type(sym) {
                return Ok(Ty::Adt(id));
            }
            // a type alias is another spelling of a type, resolved the first
            // time it is named
            if let Some(t) = cx.alias(sym, span) {
                return Ok(t);
            }
            if let Some(what) = cx.geometry_noun(sym) {
                return Err(Box::new(not_a_value_type(span, &format!("`{text}` is {what}"))));
            }
            if let Some(what) = cx.scope.unsupported_type(sym) {
                return Err(Box::new(report::unsupported(span, what)));
            }
            let interner = cx.interner;
            let candidates: Vec<&str> = PRIMITIVES
                .into_iter()
                .chain(cx.scope.type_names().map(|s| interner.resolve(s)))
                .collect();
            let mut d = report::not_found(span, "type", text, candidates);
            if cx.scope.get(sym).is_some() {
                d = d.with_note(format!("`{text}` is a value here, not a type"));
            }
            Err(Box::new(d))
        }
        [unit, name] => match cx.scope.get_unit(unit.node) {
            Some(u) => match super::imports::import_type(cx, u, name.node) {
                Some(_) if !args.is_empty() => Err(Box::new(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!(
                            "`{}::{}` takes no generic arguments",
                            cx.interner.resolve(unit.node),
                            cx.interner.resolve(name.node)
                        ))
                        .at(span),
                )),
                Some(id) => Ok(Ty::Adt(id)),
                None => match cx.imported_alias(u, name.node, span) {
                    Some(t) if args.is_empty() => Ok(t),
                    Some(_) => Err(Box::new(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!(
                                "`{}::{}` takes no generic arguments",
                                cx.interner.resolve(unit.node),
                                cx.interner.resolve(name.node)
                            ))
                            .at(span),
                    )),
                    None => Err(Box::new(super::imports::no_such_item(cx, u, name.node, span, "type"))),
                },
            },
            None => Err(Box::new(super::imports::unknown_unit(cx, unit.node, unit.span))),
        },
        _ => {
            let written = path.segments.iter().map(|s| cx.interner.resolve(s.node)).collect::<Vec<_>>().join("::");
            Err(Box::new(report::path_too_long(span, &written, 2)))
        }
    }
}

/// `const` somewhere other than the outside of a type or behind a reference.
fn nested_const(span: Span) -> Diagnostic {
    Diagnostic::new(Code::Es06)
        .with_message("`const` or `constexpr` cannot be part of another type here")
        .at(span)
        .with_note(
            "`const` and `constexpr` apply to a whole value (a binding, a parameter or a \
             result) or `const`, behind a reference, to the object referred to; part of a \
             value being constant has no meaning here",
        )
        .with_help("make the binding that holds the value `const` instead")
}

/// `qubit` written in a classical unit.
fn quantum_in_classical(span: Span) -> Diagnostic {
    Diagnostic::new(Code::Eu02)
        .with_message("a classical unit cannot hold qubits")
        .at(span)
        .with_note(
            "a unit is classical or quantum in its entirety; the `#unit` directive at \
             the top of the file decides which, and a classical unit may not declare \
             anything that stores quantum state",
        )
        .with_help("move this into a unit that begins with `#unit quantum`")
}

/// `ES06`: a name of the geometry, or a restriction, written where a type
/// of values is.
fn not_a_value_type(span: Span, what: &str) -> Diagnostic {
    Diagnostic::new(Code::Es06)
        .with_message(format!("{what}, which describes states, not a type of values"))
        .at(span)
        .with_note(
            "covers, gauges and base types describe the states of a register, and are named in \
             annotations, restrictions, locales and chains; a value's type is its register's, as \
             `[qubit; 2]`",
        )
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};
    use crate::tir::{Access, IntTy, Ty};

    #[test]
    fn every_primitive_resolves() {
        for name in super::PRIMITIVES {
            let u = check(&format!("fn f(x: {name}) {{ }}"));
            assert_eq!(u.types.display(u.fns[0].locals[0].ty, &crate::intern::Interner::new()), name);
        }
    }

    #[test]
    fn const_is_peeled_off_and_recorded() {
        let u = check("fn f(x: const const u32) { }");
        let l = &u.fns[0].locals[0];
        assert!(l.constant, "`const const T` is `const T`");
        assert_eq!(l.ty, Ty::Int(IntTy::U32));
    }

    #[test]
    fn references_record_what_they_permit() {
        let u = check("fn f(a: *u8, b: *u8, c: *const u8, d: *[u8], e: *[u8; 3]) { }");
        let t = &u.types;
        let ls = &u.fns[0].locals;
        assert_eq!(t.as_ref(ls[0].ty), Some((Access::Write, Ty::Int(IntTy::U8))));
        assert_eq!(t.as_ref(ls[1].ty).map(|r| r.0), Some(Access::Write));
        assert_eq!(t.as_ref(ls[2].ty).map(|r| r.0), Some(Access::Const));
        assert_eq!(t.as_slice(ls[3].ty), Some((Access::Write, Ty::Int(IntTy::U8))));
        let (a, arr) = t.as_ref(ls[4].ty).unwrap();
        assert_eq!(a, Access::Write);
        assert_eq!(t.as_array(arr), Some((Ty::Int(IntTy::U8), 3)));
    }

    #[test]
    fn a_reference_to_a_constant_is_written_const() {
        let u = check("fn f(a: *const u8) { }");
        let a = u.fns[0].params[0];
        assert_eq!(u.types.as_ref(u.fns[0].local(a).ty).map(|r| r.0), Some(Access::Const));
    }

    #[test]
    fn const_inside_another_type_is_refused() {
        assert_eq!(codes("fn f(a: [const u8; 2]) { }"), [Code::Es06]);
    }

    #[test]
    fn a_misspelt_type_suggests_the_intended_one() {
        let (_, d) = crate::sema::testing::analyzed("fn f(x: i23) { }");
        assert_eq!(d[0].code, Code::Es04);
        assert!(d[0].helps.iter().any(|h| h.contains("`i32`")), "{:?}", d[0].helps);
        let (_, d) = crate::sema::testing::analyzed("struct Point { }\nfn f(x: Pont) { }");
        assert!(d[0].helps.iter().any(|h| h.contains("`Point`")), "{:?}", d[0].helps);
    }

    #[test]
    fn a_qubit_in_a_classical_unit_is_refused() {
        assert_eq!(codes("fn f(q: qubit) { }"), [Code::Eu02]);
        assert_eq!(codes("fn f(q: *[qubit]) { }"), [Code::Eu02]);
        // a circuit handle's operator may take qubits the handle does not
        // hold
        assert_eq!(codes("fn f(c: circuit<fn(*qubit) -> bool>) { }"), []);
    }

    #[test]
    fn types_this_build_cannot_compile_say_so() {
        assert_eq!(codes("fn f(x: u8?) { }"), [Code::Tq003]);
    }

    #[test]
    fn a_growable_array_is_its_own_type_and_a_reference_to_one_a_slice() {
        let u = check("fn f(x: [u8], y: *[u8]) { }");
        let ps: Vec<Ty> = u.fns[0].param_types().collect();
        assert_eq!(u.types.as_growable(ps[0]), Some(Ty::Int(IntTy::U8)));
        assert!(ps[1].is_slice());
    }

    #[test]
    fn an_array_length_is_evaluated() {
        let u = check("let N: const usize = 3;\nfn f(x: [bool; N + 1]) { }");
        assert_eq!(u.types.as_array(u.fns[0].locals[0].ty), Some((Ty::Bool, 4)));
    }

    #[test]
    fn an_array_length_must_be_a_constant_usize() {
        assert_eq!(codes("fn f(x: [bool; true]) { }"), [Code::Es06]);
        assert_eq!(codes("let N: usize = 3;\nfn f(x: [bool; N]) { }"), [Code::Ec01]);
    }

    #[test]
    fn void_is_a_type_of_its_own() {
        let u = check("fn f() -> void { }");
        assert_eq!(u.fns[0].ret, Ty::Void);
    }
}
