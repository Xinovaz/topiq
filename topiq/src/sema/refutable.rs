//! Whether a pattern matches every value of its type.
//!
//! A `let` that takes a value apart has nowhere to go when the pattern does
//! not match, so only a pattern that always matches may be written there. A
//! name and `_` always match; a tuple or structure pattern matches when each
//! of its parts does; a variant pattern matches only when its enumeration has
//! that one variant; and a literal matches one value out of many, so it never
//! qualifies.
//!
//! This is deliberately simpler than the coverage check of [`super::exhaust`],
//! which decides whether a *set* of patterns covers a type. Here there is one
//! pattern, and the question is whether it alone leaves anything out.

use crate::tir::{Pat, PatKind, Ty, TypeTable};

/// Whether `p` matches every value of the type it was checked against.
pub fn is_irrefutable(p: &Pat, types: &TypeTable) -> bool {
    match &p.kind {
        PatKind::Wild | PatKind::Bind(_) | PatKind::BindRef(_) => true,
        PatKind::Const(_) => false,
        PatKind::Struct { fields } => fields.iter().all(|(_, f)| is_irrefutable(f, types)),
        PatKind::Variant { fields, .. } => {
            let one_variant = match p.ty {
                Ty::Adt(id) => types.adt(id).variants().len() == 1,
                _ => false,
            };
            one_variant && fields.iter().all(|(_, f)| is_irrefutable(f, types))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::sema::testing::{check, codes};
    use crate::diag::Code;

    #[test]
    fn a_tuple_of_names_matches_everything() {
        check("fn f(p: (i32, bool)) -> i32 { let (a, b) = p; if b { a } else { 0 } }");
        check("struct P { x: i32, y: i32 }\nfn f(p: P) -> i32 { let P { x, y } = p; x + y }");
        check("enum One { Only(i32) }\nfn f(o: One) -> i32 { let One::Only(n) = o; n }");
    }

    #[test]
    fn a_pattern_that_can_fail_is_refused() {
        assert_eq!(
            codes("fn f(p: (i32, bool)) -> i32 { let (a, true) = p; a }"),
            [Code::Es13]
        );
        assert_eq!(
            codes("enum E { A, B }\nfn f(e: E) { let E::A = e; }"),
            [Code::Es13]
        );
    }
}
