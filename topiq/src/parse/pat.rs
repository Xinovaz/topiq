//! The pattern grammar.
//!
//! # A bare identifier binds; a qualified path does not
//!
//! A single identifier in a pattern introduces a **binding**: it matches
//! anything and names it. A `::`-qualified name instead refers to something
//! that already exists, a unit variant or a constant. So `Guess::Balanced`
//! matches only that variant, while `x` matches anything at all and binds it.
//! A constant of any type named by a qualified path matches exactly its
//! value.
//!
//! # Negative literals
//!
//! A minus sign directly before an integer literal makes one negative
//! literal, as it does in an expression, so `-1 => …` matches minus one.
//! Without it, a `match` on a signed integer could name a negative case only
//! through a named constant.
//!
//! # Patterns take no generic arguments
//!
//! A pattern may name a path but never a path with generic arguments. That is
//! what makes `=>` safe to lex as a single token: a pattern can never end with
//! the `>` that closes a generic argument list, so a `=>` after one is never
//! two tokens that need pulling apart.

use chumsky::input::ValueInput;
use chumsky::prelude::*;

use crate::ast::pat::{FieldPat, Pattern};
use crate::ast::Expr;
use crate::lex::{Punct, Token};
use crate::span::{Span, Spanned};

use super::input::{Cx, Extra, ident, listed, punct};
use super::ty::path;

/// Builds the pattern parser.
pub fn grammar<'t, I, PP>(cx: Cx<'t>, pat: PP) -> impl Parser<'t, I, Spanned<Pattern>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
    PP: Parser<'t, I, Spanned<Pattern>, Extra<'t>> + Clone + 't,
{

    let literal = select! {
        Token::Int { raw, base, suffix } = e => Spanned::new(Expr::Int { raw, base, suffix }, e.span()),
        Token::Float { raw, suffix } = e => Spanned::new(Expr::Float { raw, suffix }, e.span()),
        Token::Char(c) = e => Spanned::new(Expr::Char(c), e.span()),
        Token::Bool(b) = e => Spanned::new(Expr::Bool(b), e.span()),
        Token::Str { value, kind } = e => Spanned::new(Expr::Str { value, kind }, e.span()),
    }
    .map_with(|c, e| Spanned::new(Pattern::Literal(Box::new(c)), e.span()));

    // `-1`: an integer literal written with a minus sign. the literal forms
    // alone would leave a signed value's negative cases unmatchable, so the
    // sign is accepted here as it is in `let x: i8 = -128;`
    let negative = punct(Punct::Minus)
        .ignore_then(select! {
            Token::Int { raw, base, suffix } = e => Spanned::new(Expr::Int { raw, base, suffix }, e.span()),
        })
        .map_with(|int, e| {
            let neg = Expr::Unary {
                op: crate::ast::UnOp::Neg,
                operand: Box::new(int),
            };
            Spanned::new(Pattern::Literal(Box::new(Spanned::new(neg, e.span()))), e.span())
        });

    // `(p1, p2, …)`
    let tuple = listed(pat.clone(), Punct::LParen, Punct::RParen)
        .map_with(|ps, e| Spanned::new(Pattern::Tuple(ps), e.span()));

    // `Path(p, …)`, `Path { f: p, … }`, a qualified `Path`, or a binding
    let field_pat = ident()
        .then(punct(Punct::Colon).ignore_then(pat.clone()).or_not())
        .map(|(name, pattern)| FieldPat { name, pattern });

    let path_forms = path(cx)
        .then(
            choice((
                listed(pat.clone(), Punct::LParen, Punct::RParen).map(Tail::Tuple),
                listed(field_pat, Punct::LBrace, Punct::RBrace).map(Tail::Struct),
            ))
            .or_not(),
        )
        .map_with(move |(p, tail), e| {
            let node = match tail {
                Some(Tail::Tuple(elements)) => Pattern::TupleStruct {
                    path: p,
                    elements,
                },
                Some(Tail::Struct(fields)) => Pattern::Struct { path: p, fields },
                // `_` matches anything and binds nothing. it reaches here as
                // an identifier, since an identifier may begin with `_`
                None if p.is_simple() && p.segments[0].node == cx.underscore => {
                    Pattern::Wildcard
                }
                // a single unqualified name binds; a qualified one names an
                // existing entity
                None if p.is_simple() => Pattern::Binding(p.segments[0]),
                None => Pattern::Path(p),
            };
            Spanned::new(node, e.span())
        });

    choice((literal, negative, tuple, path_forms))
}

enum Tail {
    Tuple(Vec<Spanned<Pattern>>),
    Struct(Vec<FieldPat>),
}

#[cfg(test)]
mod tests {
    use crate::ast::pat::Pattern;
    use crate::ast::{Expr, Stmt};
    use crate::intern::Interner;
    use crate::parse::testing::{errors, stmts};

    /// The pattern of the first `match` arm in `fn t() { match v { … } }`.
    fn arm(src: &str) -> (Pattern, Interner) {
        let (ss, i) = stmts(&format!("match v {{ {src} }};"));
        let e = match ss.into_iter().next().expect("one statement") {
            Stmt::Expr(e) | Stmt::BlockExpr(e) => e.node,
            other => panic!("expected an expression statement, got {other:?}"),
        };
        match e {
            Expr::Match { arms, .. } => (
                arms.into_iter().next().expect("one arm").pattern.node,
                i,
            ),
            other => panic!("expected a match, got {other:?}"),
        }
    }

    #[test]
    fn a_bare_identifier_binds() {
        // an unqualified name binds; a qualified one names something that exists
        let (p, i) = arm("x => 1");
        match p {
            Pattern::Binding(s) => assert_eq!(i.resolve(s.node), "x"),
            other => panic!("expected a binding, got {other:?}"),
        }
    }

    #[test]
    fn a_qualified_path_names_an_existing_variant() {
        let (p, _) = arm("Guess::Balanced => 1");
        match &p {
            Pattern::Path(path) => assert_eq!(path.segments.len(), 2),
            other => panic!("expected a path pattern, got {other:?}"),
        }
        assert!(p.bindings().is_empty(), "a path pattern binds nothing");
    }

    #[test]
    fn the_wildcard_matches_anything_and_binds_nothing() {
        // `_` arrives as an identifier, since an identifier may begin with it
        let (p, _) = arm("_ => 1");
        assert_eq!(p, Pattern::Wildcard);
        assert!(p.is_irrefutable());
        assert!(p.bindings().is_empty());
    }

    #[test]
    fn a_literal_pattern_parses() {
        for src in ["0 => 1", "true => 1", "'a' => 1"] {
            let (p, _) = arm(src);
            assert!(
                matches!(p, Pattern::Literal(_)),
                "{src} should be a literal pattern, got {p:?}"
            );
            assert!(!p.is_irrefutable());
        }
    }

    #[test]
    fn a_negative_integer_is_a_literal_pattern() {
        let (p, _) = arm("-1 => 1");
        let Pattern::Literal(e) = p else {
            panic!("expected a literal pattern, got {p:?}");
        };
        assert!(matches!(e.node, Expr::Unary { .. }), "{:?}", e.node);
    }

    #[test]
    fn a_tuple_variant_pattern_binds_its_elements() {
        // as in `match s as *Circle { Some(c) => Some(c.r), … }`
        let (p, i) = arm("Some(c) => 1");
        match &p {
            Pattern::TupleStruct { path, elements } => {
                assert!(path.is_simple());
                assert_eq!(elements.len(), 1);
            }
            other => panic!("expected a tuple-variant pattern, got {other:?}"),
        }
        let names: Vec<&str> = p.bindings().iter().map(|s| i.resolve(s.node)).collect();
        assert_eq!(names, vec!["c"]);
    }

    #[test]
    fn a_structure_pattern_parses_in_both_field_forms() {
        let (p, i) = arm("Point { x, y: inner } => 1");
        match &p {
            Pattern::Struct { fields, .. } => assert_eq!(fields.len(), 2),
            other => panic!("expected a structure pattern, got {other:?}"),
        }
        // the shorthand binds the field name; the long form binds its
        // sub-pattern's name instead
        let names: Vec<&str> = p.bindings().iter().map(|s| i.resolve(s.node)).collect();
        assert_eq!(names, vec!["x", "inner"]);
    }

    #[test]
    fn a_tuple_pattern_parses() {
        let (p, _) = arm("(a, b) => 1");
        match &p {
            Pattern::Tuple(ps) => assert_eq!(ps.len(), 2),
            other => panic!("expected a tuple pattern, got {other:?}"),
        }
        assert!(p.is_irrefutable(), "both elements are bindings");
        assert_eq!(p.bindings().len(), 2);
    }

    #[test]
    fn patterns_nest() {
        let (p, _) = arm("Some((a, _)) => 1");
        assert_eq!(p.bindings().len(), 1);
        assert!(!p.is_irrefutable());
    }

    #[test]
    fn a_for_loop_takes_a_pattern() {
        // the other place a pattern can appear. a trailing block form is the
        // block's value rather than a statement, so something must follow it
        // for it to count as one
        let (ss, _) = stmts("for (a, b) in xs { } let z = 1;");
        assert_eq!(ss.len(), 2);
        assert!(matches!(ss[0], Stmt::BlockExpr(_)));
    }

    #[test]
    fn a_path_segment_after_the_separator_may_be_a_keyword() {
        // `phase::<8>::of(k)` needs `of` as a path segment, and `of` is also
        // a keyword
        let (ss, _) = stmts("let p = phase::<8>::of(4);");
        assert_eq!(ss.len(), 1);
    }

    #[test]
    fn a_pattern_takes_no_generic_arguments() {
        // a pattern never ends with the `>` of generic arguments, which is what
        // makes `=>` safe to lex as one token
        assert!(!errors("fn t() { match v { Some<u8> => 1 }; }").is_empty());
    }

    #[test]
    fn malformed_patterns_are_rejected() {
        for src in ["Some( => 1", "Point { => 1", "(a, => 1"] {
            assert!(
                !errors(&format!("fn t() {{ match v {{ {src} }}; }}")).is_empty(),
                "{src:?} should not parse"
            );
        }
    }
}
