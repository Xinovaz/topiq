//! The type grammar.
//!
//! # Postfix and prefix formers
//!
//! `T of U` and `T?` extend a type on the right, so they are parsed as postfix
//! suffixes over an atom. `const T` and `*T` are prefixes, and a
//! former always applies to what stands to its **right**. So `*const T` is a
//! reference to a constant object, and `const *T` is a constant reference to a
//! mutable one (the same two words, in the other order, meaning something
//! else).
//!
//! # `<` is never a comparison here
//!
//! In type position `Foo<A, B>` is a generic application, and `<` can never be
//! a comparison, because there are no comparisons inside a type. That is why
//! this module reads angle brackets directly while [`super::expr`] requires
//! the turbofish: in an expression, `a < b` is ambiguous with a generic, and
//! requiring `::<` removes the ambiguity.
//!
//! # Generic arguments
//!
//! A generic argument is a type when a `,` or `>` follows it, and otherwise
//! a constant: a literal, a path or a parenthesised expression, joined by
//! `*`, `/` and `%`, then `+` and `-`, as in `mat<T, R * P, C * Q>`. A
//! constant argument holds no comparison, so a `>` always closes the list;
//! a comparison is written inside parentheses.

use chumsky::input::ValueInput;
use chumsky::prelude::*;

use crate::ast::expr::MacroArg;
use crate::ast::ty::{Path, TArg, Type};
use crate::ast::Expr;
use crate::lex::{Keyword, Punct, Token};
use crate::span::{Span, Spanned};

use super::input::{Cx, Extra, close_angle, halves, ident, kw, listed, name, punct};

/// Parses a `::`-separated path.
///
/// # A keyword may follow `::`
///
/// `phase::<8>::of(k)` names a member `of`, which is also the restriction
/// former's keyword. So a keyword is accepted as a non-initial segment,
/// where it can only be a member name. The leading segment stays an
/// identifier, which keeps `A of B` a restriction rather than a path.
pub fn path<'t, I>(cx: Cx<'t>) -> impl Parser<'t, I, Path, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    ident()
        .then(
            punct(Punct::ColonColon)
                .ignore_then(name(cx.kws))
                .repeated()
                .collect::<Vec<_>>(),
        )
        .map(|(head, tail): (Spanned<crate::intern::Symbol>, Vec<_>)| {
            let mut segments = vec![head];
            segments.extend(tail);
            Path { segments }
        })
}

/// One generic argument, `targ := type | const-expr`, in a type's `<…>` or an
/// expression's `::<…>`.
///
/// A type is one only if the argument ends after it, at `,` or `>`: `C * 2`
/// starts like the type `C` but is a constant expression. A constant
/// argument is literals, names and parenthesised expressions joined by
/// `* / %` and then `+ -`; comparisons are left out, since a `>` there closes
/// the arguments, and `3>()` must not read as `3 > ()`.
pub fn generic_arg<'t, I, PT, PE>(cx: Cx<'t>, ty: PT, expr: PE) -> impl Parser<'t, I, TArg, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
    PT: Parser<'t, I, Spanned<Type>, Extra<'t>> + Clone + 't,
    PE: Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone + 't,
{
    // a constant argument's operands, then its two levels of operator
    let operand = select! {
        Token::Int { raw, base, suffix } = e => Spanned::new(Expr::Int { raw, base, suffix }, e.span()),
    }
    .or(path(cx).map_with(|path, e| Spanned::new(Expr::Path { path, args: Vec::new() }, e.span())))
    .or(expr
        .delimited_by(punct(Punct::LParen), punct(Punct::RParen))
        .map_with(|x, e| Spanned::new(Expr::Paren(Box::new(x)), e.span())));
    let binary = |l: Spanned<Expr>, (op, r): (crate::ast::BinOp, Spanned<Expr>)| {
        let span = Span::new(l.span.source, l.span.start, r.span.end);
        Spanned::new(
            Expr::Binary {
                op,
                lhs: Box::new(l),
                rhs: Box::new(r),
            },
            span,
        )
    };
    let product = operand.clone().foldl(
        choice((
            punct(Punct::Star).to(crate::ast::BinOp::Mul),
            punct(Punct::Slash).to(crate::ast::BinOp::Div),
            punct(Punct::Percent).to(crate::ast::BinOp::Rem),
        ))
        .then(operand)
        .repeated(),
        binary,
    );
    let sum = product.clone().foldl(
        choice((punct(Punct::Plus).to(crate::ast::BinOp::Add), punct(Punct::Minus).to(crate::ast::BinOp::Sub)))
            .then(product)
            .repeated(),
        binary,
    );

    // a type if one ends there, else a constant
    ty.then_ignore(punct(Punct::Comma).ignored().or(close_angle().ignored()).rewind())
        .map(TArg::Type)
        .or(sum.map(TArg::Const))
}

/// Builds the type parser.
///
/// `ty` is the recursive handle for a nested type and `expr` parses the
/// constant expression in an array length.
pub fn grammar<'t, I, PT, PE>(
    cx: Cx<'t>,
    ty: PT,
    expr: PE,
) -> impl Parser<'t, I, Spanned<Type>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
    PT: Parser<'t, I, Spanned<Type>, Extra<'t>> + Clone + 't,
    PE: Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone + 't,
{
    // `targ := type | const-expr`. a type is tried first: a bare identifier is
    // far more often a type name than a constant, and a constant expression
    // that begins with an identifier is resolved at phase 7 anyway
    let targ = generic_arg(cx, ty.clone(), expr.clone())
        .map_with(|a, e| Spanned::new(a, e.span()));

    let targs = targ
        .separated_by(punct(Punct::Comma))
        .at_least(1)
        .collect::<Vec<_>>()
        .delimited_by(punct(Punct::Lt), close_angle())
        .or_not()
        .map(Option::unwrap_or_default);

    // `[T]` and `[T; N]`
    let array = ty
        .clone()
        .then(punct(Punct::Semi).ignore_then(expr.clone()).or_not())
        .delimited_by(punct(Punct::LBracket), punct(Punct::RBracket))
        .map(|(elem, len)| Type::Array {
            elem: Box::new(elem),
            len: len.map(Box::new),
        });

    // `(T1, T2, …)`. a one-element parenthesised type is that type, not a
    // one-tuple, so the grouping is transparent
    let tuple = listed(ty.clone(), Punct::LParen, Punct::RParen)
        .map(|mut ts: Vec<Spanned<Type>>| if ts.len() == 1 { ts.remove(0).node } else { Type::Tuple(ts) });

    // `fn(T…) -> U`
    let fn_ty = kw(Keyword::Fn)
        .ignore_then(listed(ty.clone(), Punct::LParen, Punct::RParen))
        .then(punct(Punct::Arrow).ignore_then(ty.clone()).or_not())
        .map(|(params, ret)| Type::Fn {
            params,
            ret: ret.map(Box::new),
        });

    // `qmap<K, V>`; `qmap` is a keyword, unlike `closure` and `circuit`
    let qmap_ty = kw(Keyword::Qmap)
        .ignore_then(
            ty.clone()
                .then_ignore(punct(Punct::Comma))
                .then(ty.clone())
                .delimited_by(punct(Punct::Lt), close_angle()),
        )
        .map(|(key, value)| Type::Qmap {
            key: Box::new(key),
            value: Box::new(value),
        });

    let void = kw(Keyword::Void).map(|_| Type::Void);

    // a path, with the three identifier-spelled formers recognised out of it
    let named = path(cx).then(targs).map(move |(p, args)| {
        let one = p.is_simple().then(|| p.last()).flatten();
        match one {
            Some(s) if s == cx.dyn_ && args.is_empty() => Type::Dyn,
            Some(s) if s == cx.closure && args.len() == 1 => {
                Type::Closure(Box::new(unwrap_type_arg(args)))
            }
            Some(s) if s == cx.circuit && args.len() == 1 => {
                Type::Circuit(Box::new(unwrap_type_arg(args)))
            }
            _ => Type::Path { path: p, args },
        }
    });

    // `@name(args)`: a macro yielding a type, such as `@field_type(T, "x")`
    // its arguments are types and strings
    let macro_arg = choice((
        select! { Token::Str { value, .. } = e => Spanned::new(MacroArg::Str(Spanned::new(value, e.span())), e.span()) },
        ty.clone().map_with(|t, e| Spanned::new(MacroArg::Type(t), e.span())),
    ));
    let macro_ty = select! { Token::MacroName(n) = e => Spanned::new(n, e.span()) }
        .then(listed(macro_arg, Punct::LParen, Punct::RParen))
        .map(|(name, args)| Type::Macro { name, args });

    let atom = choice((array, tuple, fn_ty, qmap_ty, void, macro_ty, named))
        .map_with(|t, e| Spanned::new(t, e.span()));

    // prefixes nest rightwards: `const` and `*`. `**` is one token (the
    // tensor operator) but in a type it can only be two references, so `**T`
    // is read as `* *T`
    enum Prefix {
        Const,
        Constexpr,
        Ref,
    }
    let prefix = choice((
        kw(Keyword::Const).map(|s| vec![(Prefix::Const, s)]),
        kw(Keyword::Constexpr).map(|s| vec![(Prefix::Constexpr, s)]),
        punct(Punct::Star).map(|s| vec![(Prefix::Ref, s)]),
        punct(Punct::StarStar).map(|s| {
            let (a, b) = halves(s);
            vec![(Prefix::Ref, a), (Prefix::Ref, b)]
        }),
    ));

    // postfixes fold leftwards: `?` and `of`
    enum Postfix {
        Optional(Span),
        Of(Spanned<Type>),
    }
    let postfix = choice((
        punct(Punct::Question).map(Postfix::Optional),
        kw(Keyword::Of).ignore_then(atom.clone()).map(Postfix::Of),
    ));

    let with_postfix = atom
        .clone()
        .foldl(postfix.repeated(), |lhs, suffix| match suffix {
            Postfix::Optional(s) => {
                let span = lhs.span.join(s);
                Spanned::new(Type::Optional(Box::new(lhs)), span)
            }
            // `A of B`: `lhs` is the restriction, contributing its predicate,
            // and `rhs` is the container, whose gauge the result carries. the
            // order is not symmetric and must not be normalised away
            Postfix::Of(rhs) => {
                let span = lhs.span.join(rhs.span);
                Spanned::new(
                    Type::Restriction {
                        restricted: Box::new(lhs),
                        container: Box::new(rhs),
                    },
                    span,
                )
            }
        });

    prefix
        .repeated()
        .collect::<Vec<_>>()
        .then(with_postfix)
        .map(|(prefixes, inner)| {
            prefixes.into_iter().flatten().rev().fold(inner, |inner, (p, at)| {
                let span = at.join(inner.span);
                let t = match p {
                    Prefix::Const => Type::Const(Box::new(inner)),
                    Prefix::Constexpr => Type::Constexpr(Box::new(inner)),
                    Prefix::Ref => Type::Ref { target: Box::new(inner) },
                };
                Spanned::new(t, span)
            })
        })
}

/// Pulls the single type out of a one-element generic argument list.
fn unwrap_type_arg(mut args: Vec<Spanned<TArg>>) -> Spanned<Type> {
    match args.remove(0) {
        Spanned {
            node: TArg::Type(t),
            ..
        } => t,
        // `closure<3>` is nonsense the grammar cannot rule out; phase 8
        // rejects it, and the span lets that diagnostic point at it
        Spanned { node: _, span } => Spanned::new(Type::Void, span),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::parse::input::{TokenStream, eoi_span, stream};
    use crate::source::Spliced;
    use crate::span::SourceId;

    /// Parses `src` as a type, with a minimal expression parser for array
    /// lengths and const generic arguments.
    ///
    /// The marking pass is not run: a bare type fragment looks like item
    /// position, so it would retag `[u8]` as an annotation. In a real unit a
    /// type is never in item position.
    fn parse_with(interner: &mut Interner, src: &str) -> Result<Spanned<Type>, Vec<String>> {
        // lex first: the parse context takes a shared borrow of the table, and
        // nothing is interned once parsing starts
        let spliced = Spliced::from_text(src);
        let lexed = crate::lex::lex(SourceId(0), &spliced, interner);
        assert!(lexed.diagnostics.is_empty(), "{:?}", lexed.diagnostics);
        let tokens = lexed.tokens;
        let eoi = eoi_span(SourceId(0), &tokens);
        let cx = Cx::new(interner);

        let mut ty = Recursive::declare();
        // only the literal and path forms an array length or const generic
        // needs; the full expression grammar is tested in `super::expr`
        let expr = select! {
            Token::Int { raw, base, suffix } = e => Spanned::new(Expr::Int { raw, base, suffix }, e.span()),
        }
        .or(ident().map(|s: Spanned<crate::intern::Symbol>| {
            Spanned::new(
                Expr::Path {
                    path: Path::single(s),
                    args: Vec::new(),
                },
                s.span,
            )
        }));
        ty.define(grammar::<TokenStream, _, _>(cx, ty.clone(), expr));

        ty.parse(stream(&tokens, eoi))
            .into_result()
            .map_err(|es| es.iter().map(ToString::to_string).collect())
    }

    fn parse(src: &str) -> (Result<Spanned<Type>, Vec<String>>, Interner) {
        let mut interner = Interner::new();
        let out = parse_with(&mut interner, src);
        (out, interner)
    }

    fn ok(src: &str) -> Type {
        let (r, _) = parse(src);
        r.unwrap_or_else(|e| panic!("{src:?} did not parse: {e:?}")).node
    }

    fn name_of(t: &Type) -> Option<String> {
        match t {
            Type::Path { path, .. } => Some(format!("{}", path.segments.len())),
            _ => None,
        }
    }

    #[test]
    fn a_plain_name_is_a_path_type() {
        let t = ok("u32");
        assert!(matches!(t, Type::Path { .. }));
        assert_eq!(name_of(&t).as_deref(), Some("1"));
    }

    #[test]
    fn a_qualified_name_keeps_its_segments() {
        let t = ok("qpu::Device");
        match t {
            Type::Path { path, .. } => assert_eq!(path.segments.len(), 2),
            other => panic!("expected a path, got {other:?}"),
        }
    }

    #[test]
    fn generic_application_needs_no_turbofish_in_type_position() {
        // in a type, `<` is never a comparison
        let t = ok("Pair<A, B>");
        match t {
            Type::Path { args, .. } => assert_eq!(args.len(), 2),
            other => panic!("expected a generic path, got {other:?}"),
        }
    }

    #[test]
    fn nested_generics_close_with_two_angles() {
        // two generic lists closing at once: the lexer emits two separate `>`
        // tokens, so there is nothing to split
        let t = ok("Vec<Vec<T>>");
        match t {
            Type::Path { args, .. } => {
                assert_eq!(args.len(), 1);
                match &args[0].node {
                    TArg::Type(inner) => {
                        assert!(matches!(inner.node, Type::Path { .. }));
                    }
                    other => panic!("expected a type argument, got {other:?}"),
                }
            }
            other => panic!("expected a generic path, got {other:?}"),
        }
    }

    #[test]
    fn references_and_const_compose_right_to_left() {
        // a former applies to what stands to its right
        // `*const T` is a reference to a constant object
        match ok("*const u32") {
            Type::Ref { target } => assert!(target.node.is_const()),
            other => panic!("expected a reference, got {other:?}"),
        }
        // `const *T` is a constant reference to a mutable object
        match ok("const *u32") {
            Type::Const(inner) => assert!(matches!(inner.node, Type::Ref { .. })),
            other => panic!("expected a const type, got {other:?}"),
        }
        // `const *const T` is both
        match ok("const *const u32") {
            Type::Const(inner) => match inner.node {
                Type::Ref { target, .. } => assert!(target.node.is_const()),
                other => panic!("expected a reference, got {other:?}"),
            },
            other => panic!("expected a const type, got {other:?}"),
        }
    }

    #[test]
    fn a_doubled_star_is_two_references() {
        // `**` lexes as the tensor operator, which cannot begin a type
        let depth = |src: &str| {
            let mut t = ok(src);
            let mut n = 0;
            while let Type::Ref { target } = t {
                n += 1;
                t = target.node;
            }
            n
        };
        assert_eq!(depth("**u32"), 2);
        assert_eq!(depth("**const u32"), 2);
        assert_eq!(depth("***u32"), 3);
        assert_eq!(depth("* *u32"), depth("**u32"));
    }

    #[test]
    fn arrays_come_in_growable_and_fixed_forms() {
        match ok("[u8]") {
            Type::Array { len, .. } => assert!(len.is_none()),
            other => panic!("expected an array, got {other:?}"),
        }
        match ok("[qubit; 2]") {
            Type::Array { len, .. } => assert!(len.is_some()),
            other => panic!("expected an array, got {other:?}"),
        }
        // a const generic as the length, as in `struct Reg<N: const usize>`
        match ok("[qubit; N]") {
            Type::Array { len, .. } => assert!(len.is_some()),
            other => panic!("expected an array, got {other:?}"),
        }
    }

    #[test]
    fn tuples_parse_and_a_single_parenthesized_type_is_transparent() {
        match ok("(u8, f64, bool)") {
            Type::Tuple(ts) => assert_eq!(ts.len(), 3),
            other => panic!("expected a tuple, got {other:?}"),
        }
        // `(T)` is `T`, not a one-tuple
        assert!(matches!(ok("(u8)"), Type::Path { .. }));
        // the empty tuple is a tuple
        match ok("()") {
            Type::Tuple(ts) => assert!(ts.is_empty()),
            other => panic!("expected the empty tuple, got {other:?}"),
        }
    }

    #[test]
    fn function_types_parse_with_and_without_a_return() {
        // as in the type of an oracle parameter, `fn(*qubit, *qubit)`
        match ok("fn(*qubit, *qubit)") {
            Type::Fn { params, ret } => {
                assert_eq!(params.len(), 2);
                assert!(ret.is_none());
            }
            other => panic!("expected a function type, got {other:?}"),
        }
        match ok("fn(u8) -> u32") {
            Type::Fn { params, ret } => {
                assert_eq!(params.len(), 1);
                assert!(ret.is_some());
            }
            other => panic!("expected a function type, got {other:?}"),
        }
        match ok("fn() -> void") {
            Type::Fn { ret, .. } => assert!(matches!(ret.unwrap().node, Type::Void)),
            other => panic!("expected a function type, got {other:?}"),
        }
    }

    #[test]
    fn the_identifier_spelled_formers_are_recognized() {
        // none of these is a reserved word, so each is recognised by name
        assert!(matches!(ok("dyn"), Type::Dyn));
        assert!(matches!(ok("closure<Sig>"), Type::Closure(_)));
        assert!(matches!(ok("circuit<Sig>"), Type::Circuit(_)));
        // but they are still ordinary names when used as such
        assert!(matches!(ok("dyn_thing"), Type::Path { .. }));
    }

    #[test]
    fn a_map_locale_type_parses() {
        match ok("qmap<[qubit; 2], Entry>") {
            Type::Qmap { key, value } => {
                assert!(matches!(key.node, Type::Array { .. }));
                assert!(matches!(value.node, Type::Path { .. }));
            }
            other => panic!("expected a qmap, got {other:?}"),
        }
    }

    #[test]
    fn the_optional_suffix_folds_left() {
        match ok("u32?") {
            Type::Optional(inner) => assert!(matches!(inner.node, Type::Path { .. })),
            other => panic!("expected an optional, got {other:?}"),
        }
        // as in `fn narrow(s: *Shape) -> f64?`
        assert!(matches!(ok("f64?"), Type::Optional(_)));
    }

    #[test]
    fn restriction_is_ordered() {
        // `A of B` and `B of A` are different types. both go through one
        // interner, or the first name in each would get the same symbol and
        // the trees would compare equal for the wrong reason
        let mut interner = Interner::new();
        let ab = parse_with(&mut interner, "A of B").unwrap().node;
        let ba = parse_with(&mut interner, "B of A").unwrap().node;
        assert_ne!(ab, ba);
        match ab {
            Type::Restriction {
                restricted,
                container,
            } => {
                // in `M of Search` the container is the second operand
                assert!(matches!(restricted.node, Type::Path { .. }));
                assert!(matches!(container.node, Type::Path { .. }));
            }
            other => panic!("expected a restriction, got {other:?}"),
        }
    }

    #[test]
    fn restriction_chains_fold_left() {
        // `A of B of C` is `(A of B) of C`, so the outermost container is `C`
        match ok("A of B of C") {
            Type::Restriction { restricted, .. } => {
                assert!(
                    matches!(restricted.node, Type::Restriction { .. }),
                    "the left operand should itself be a restriction"
                );
            }
            other => panic!("expected a restriction, got {other:?}"),
        }
    }

    #[test]
    fn a_span_covers_the_whole_type() {
        let (r, _) = parse("*[qubit; 2]");
        let t = r.unwrap();
        assert_eq!((t.span.start, t.span.end), (0, 11));
    }

    #[test]
    fn malformed_types_are_rejected() {
        for src in ["[", "fn(", "Vec<", "const", "*", "A of"] {
            let (r, _) = parse(src);
            assert!(r.is_err(), "{src:?} should not parse");
        }
    }

    #[test]
    fn a_deeply_derived_type_parses() {
        // a former applies to everything to its right, the `?` included, so
        // this is a reference to a constant optional array
        match ok("*const [Pair<u8, f64>; 4]?") {
            Type::Ref { target } => {
                match target.node {
                    Type::Const(inner) => assert!(matches!(inner.node, Type::Optional(_))),
                    other => panic!("expected a const type, got {other:?}"),
                }
            }
            other => panic!("expected a reference, got {other:?}"),
        }
    }

    #[test]
    fn a_bracketed_type_parses_when_marking_has_not_mistaken_it() {
        // the companion to the harness note: fed through the real pipeline, a
        // type after `:` is never in item position, so its `[` stays an array
        let interner = Interner::new();
        let spliced = Spliced::from_text("fn f(x: [u8; 4]) {}");
        let lexed = crate::lex::lex(SourceId(0), &spliced, &interner);
        let marked = crate::mark::mark(&lexed.tokens);
        assert_eq!(
            marked.annotation_count(),
            0,
            "`[u8; 4]` after a colon is an array type, not an annotation"
        );
    }
}
