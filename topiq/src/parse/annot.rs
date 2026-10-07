//! The annotation grammar.
//!
//! Whether the opening bracket begins an annotation at all has already been
//! settled by [`crate::mark`], which retags it [`Punct::LBracketAnnot`]. This
//! module never revisits that question.
//!
//! # Three annotations open a quantum context
//!
//! `cover`, `gauge` and `frame` take quantum forms as arguments, parsed by
//! [`crate::quon::parse`], where `|0>` is a ket. Every other annotation takes
//! ordinary constant expressions or types, where the same three characters are
//! a bitwise or, a zero and a comparison.
//!
//! So the grammar here is **name-directed**: the annotation's name is matched
//! first, and the name chooses which argument parser runs. There is no way to
//! decide it later, because by then the tokens have already been read one way
//! or the other.
//!
//! Two of those names, `cover` and `gauge`, are also reserved words. Annotation
//! names have a name space of their own, so they are accepted here as names;
//! see [`super::input::name`].
//!
//! # What a `;` separates
//!
//! Inside a group, `;` normally separates one annotation from the next and `,`
//! separates one argument from the next:
//!
//! ```text
//! [packed; align: 8]                      // two annotations
//! [iso: X, Z]                             // one annotation, two arguments
//! ```
//!
//! But a contract ascription is written with a `;` inside a single annotation:
//!
//! ```text
//! [expect: monic; contract = flat(0)]     // one annotation, two arguments
//! ```
//!
//! Here `contract = flat(0)` qualifies the expectation; it is not a second
//! annotation, and reading it as one would detach the contract from the claim
//! it belongs to.
//!
//! The two are told apart with one token of lookahead. An annotation always
//! begins with a name followed by `:`, `;` or `]`, so:
//!
//! - in `[packed; align: 8]`, `align` is followed by `:`, a new annotation;
//! - in `[expect: monic; contract = flat(0)]`, `contract` is followed by `=`,
//!   another argument of `expect`.

use chumsky::input::ValueInput;
use chumsky::prelude::*;

use crate::ast::annot::{AnnArg, Annotation, AnnotationGroup};
use crate::ast::{Expr, Type};
use crate::intern::Symbol;
use crate::lex::{Keyword, Punct, Token};
use crate::quon::ast::QForm;
use crate::quon::parse::{qcover, qgauge, qstate};
use crate::span::{Span, Spanned};

use super::input::{Cx, Extra, ident, name, punct};

/// Matches one specific name, spelled as an identifier or as a keyword.
fn exactly<'t, I>(cx: Cx<'t>, want: Symbol) -> impl Parser<'t, I, Spanned<Symbol>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    name(cx.kws).try_map(move |s, span| {
        if s.node == want {
            Ok(s)
        } else {
            Err(Rich::custom(span, "expected a different annotation name"))
        }
    })
}

/// Builds the annotation-group parser.
pub fn grammar<'t, I, PE, PT>(
    cx: Cx<'t>,
    expr: PE,
    ty: PT,
) -> impl Parser<'t, I, Spanned<AnnotationGroup>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
    PE: Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone + 't,
    PT: Parser<'t, I, Spanned<Type>, Extra<'t>> + Clone + 't,
{
    // `[cover: …]` and `[frame: …]` take a cover, `[gauge: …]` a gauge. a
    // frame tries a state of names first, so the first name of `a ** (b ** c)`
    // is not taken for a cover
    let cover_annot = exactly(cx, cx.kws.of(Keyword::Cover))
        .then_ignore(punct(Punct::Colon))
        .then(
            qcover(cx)
                .map(QForm::Cover)
                .or(qstate(cx).map(QForm::State)),
        )
        .or(exactly(cx, cx.frame).then_ignore(punct(Punct::Colon)).then(
            qstate(cx)
                .map(QForm::State)
                .or(qcover(cx).map(QForm::Cover)),
        ))
        .map_with(|(name, form), e| {
            Spanned::new(
                Annotation {
                    name,
                    args: vec![Spanned::new(AnnArg::Quon(Box::new(form)), e.span())],
                },
                e.span(),
            )
        });

    let gauge_annot = exactly(cx, cx.kws.of(Keyword::Gauge))
        .then_ignore(punct(Punct::Colon))
        .then(qgauge(cx).map(QForm::Gauge))
        .map_with(|(name, form), e| {
            Spanned::new(
                Annotation {
                    name,
                    args: vec![Spanned::new(AnnArg::Quon(Box::new(form)), e.span())],
                },
                e.span(),
            )
        });

    // every other annotation takes constant expressions, types, or named
    // arguments such as `[expect: contract = flat]`
    let plain_arg = recursive(|arg| {
        let named = ident()
            .then_ignore(punct(Punct::Eq))
            .then(arg)
            .map(|(name, value): (Spanned<Symbol>, Spanned<AnnArg>)| AnnArg::Named {
                name,
                value: Box::new(value),
            });
        choice((
            named,
            // an expression first, for `[expect: kernel.outcomes = Guess]`,
            // which no type can be; a type catches what no expression can be,
            // such as `*qubit` and `fn(T) -> U`
            expr.clone().map(|x| AnnArg::Expr(Box::new(x))),
            ty.clone().map(|t| AnnArg::Type(Box::new(t))),
        ))
        .map_with(|a, e| Spanned::new(a, e.span()))
    });

    // the shape that marks the start of a new annotation rather than another
    // argument: a name followed by `:`, `;` or `]`
    let annotation_head = name(cx.kws)
        .then(choice((
            punct(Punct::Colon),
            punct(Punct::Semi),
            punct(Punct::RBracket),
        )))
        .ignored();

    let more_args = choice((
        punct(Punct::Comma).ignore_then(plain_arg.clone()),
        punct(Punct::Semi).ignore_then(plain_arg.clone().and_is(annotation_head.not())),
    ))
    .repeated()
    .collect::<Vec<_>>();

    let plain_annot = name(cx.kws)
        .then(
            punct(Punct::Colon)
                .ignore_then(plain_arg.clone().then(more_args))
                .or_not()
                .map(|args| match args {
                    None => Vec::new(),
                    Some((head, tail)) => {
                        let mut all = vec![head];
                        all.extend(tail);
                        all
                    }
                }),
        )
        .map_with(|(name, args), e| Spanned::new(Annotation { name, args }, e.span()));

    let annotation = choice((cover_annot, gauge_annot, plain_annot));

    // several annotations may share one group, separated by `;`
    annotation
        .separated_by(punct(Punct::Semi))
        .at_least(1)
        .allow_trailing()
        .collect::<Vec<_>>()
        .delimited_by(punct(Punct::LBracketAnnot), punct(Punct::RBracket))
        .map_with(|annotations, e| Spanned::new(AnnotationGroup { annotations }, e.span()))
        .labelled("annotation group")
}

#[cfg(test)]
mod tests {
    use crate::ast::annot::AnnArg;
    use crate::intern::Interner;
    use crate::parse::testing::{errors, unit};

    /// The annotation groups on the first item.
    fn groups(src: &str) -> (Vec<crate::ast::AnnotationGroup>, Interner) {
        let (u, i) = unit(src);
        let item = u.items.into_iter().next().expect("one item").node;
        (
            item.annotations.into_iter().map(|g| g.node).collect(),
            i,
        )
    }

    /// The names of every annotation on the first item, in order.
    fn names(src: &str) -> Vec<String> {
        let (gs, i) = groups(src);
        gs.iter()
            .flat_map(|g| g.annotations.iter())
            .map(|a| i.resolve(a.node.name.node).to_owned())
            .collect()
    }

    #[test]
    fn a_bare_annotation_parses() {
        assert_eq!(names("[entry]\nfn f() { }"), vec!["entry"]);
    }

    #[test]
    fn groups_stack_and_keep_their_order() {
        // several groups may be stacked on one item
        let (gs, _) = groups("[entry]\n[inline]\nfn f() { }");
        assert_eq!(gs.len(), 2);
        assert_eq!(names("[entry]\n[inline]\nfn f() { }"), vec!["entry", "inline"]);
    }

    #[test]
    fn several_annotations_may_share_one_group() {
        // two annotations in one group: `[packed; align: 8]`
        let (gs, _) = groups("[packed; align: 8]\nstruct H { tag: u8 }");
        assert_eq!(gs.len(), 1, "one bracket");
        assert_eq!(gs[0].annotations.len(), 2, "two annotations in it");
    }

    #[test]
    fn an_annotation_takes_a_constant_expression_argument() {
        let (gs, _) = groups("[align: 16]\nstruct S { x: u8 }");
        let a = &gs[0].annotations[0].node;
        assert_eq!(a.args.len(), 1);
        assert!(matches!(a.args[0].node, AnnArg::Expr(_)));
    }

    #[test]
    fn an_annotation_takes_several_arguments() {
        // two arguments to one annotation: `[iso: X, Z]`
        let (gs, _) = groups("[iso: X, Z]\nenum QBit { Zero(qubit), One(qubit) }");
        assert_eq!(gs[0].annotations[0].node.args.len(), 2);
    }

    #[test]
    fn a_named_argument_parses() {
        // a named argument: `[expect: contract = flat]`
        let (gs, i) = groups("[expect: contract = flat]\nfn f() { }");
        match &gs[0].annotations[0].node.args[0].node {
            AnnArg::Named { name, .. } => assert_eq!(i.resolve(name.node), "contract"),
            other => panic!("expected a named argument, got {other:?}"),
        }
    }

    #[test]
    fn an_argument_may_be_a_dotted_assignment() {
        // `[expect: kernel.outcomes = Guess]`. the left operand is a field
        // access, which a plain `name = value` form cannot express, so the
        // whole argument is parsed as an expression
        let (gs, _) = groups("[expect: kernel.outcomes = Guess]\nfn f() { }");
        assert!(matches!(gs[0].annotations[0].node.args[0].node, AnnArg::Expr(_)));
    }

    #[test]
    fn an_annotation_name_may_be_spelled_like_a_keyword() {
        // annotation names have their own name space, and two of the
        // standard annotations are spelt with reserved words
        assert_eq!(names("[cover: B2]\nfn f() { }"), vec!["cover"]);
        assert_eq!(names("[gauge: GB]\nfn f() { }"), vec!["gauge"]);
    }

    ///////////////////////////////////////////////////////
    // THE THREE ANNOTATIONS THAT OPEN A QUANTUM CONTEXT //
    ///////////////////////////////////////////////////////

    #[test]
    fn a_cover_annotation_takes_a_quon_form() {
        // `[cover: fin{ |0>, |1> }]`. inside it, `|0>` is a ket
        let (gs, _) = groups("[cover: fin{ |0>, |1> }]\nfn f() { }");
        assert!(matches!(gs[0].annotations[0].node.args[0].node, AnnArg::Quon(_)));
    }

    #[test]
    fn a_gauge_annotation_takes_a_quon_form() {
        // `[gauge: fid((|00> + |11>) * isq2)]`: a fiducial gauge
        let (gs, _) = groups("[gauge: fid((|00> + |11>) * isq2)]\nfn f() { }");
        assert!(matches!(gs[0].annotations[0].node.args[0].node, AnnArg::Quon(_)));
    }

    #[test]
    fn a_named_cover_is_accepted_as_well_as_a_literal_one() {
        // a cover and gauge given by name: `[cover: Orb] [gauge: GOrb]`
        assert_eq!(
            names("[cover: Orb] [gauge: GOrb]\nfn f() { }"),
            vec!["cover", "gauge"]
        );
    }

    #[test]
    fn a_frame_annotation_takes_a_quon_form_too() {
        // `frame` takes a quantum form too, like `cover` and `gauge`
        assert_eq!(names("[frame: B2]\nfn f() { }"), vec!["frame"]);
    }

    #[test]
    fn an_ordinary_annotation_does_not_open_a_quon_context() {
        // `[expect: …]` does not open a quantum context, so a `|` inside it
        // is an ordinary bitwise or
        let (gs, _) = groups("[expect: a | b]\nfn f() { }");
        assert!(matches!(gs[0].annotations[0].node.args[0].node, AnnArg::Expr(_)));
    }

    #[test]
    fn a_phase_constant_parses_in_an_annotation() {
        // `[expect: stat(phase::<8>::of(4))]`, which needs both
        // a keyword as a path segment and a path continued after a turbofish
        assert_eq!(
            names("[expect: stat(phase::<8>::of(4))]\nfn f() { }"),
            vec!["expect"]
        );
    }

    #[test]
    fn annotations_attach_to_parameters_and_fields_too() {
        let (u, _) = unit("fn f([cover: B] r: u8) { }");
        match &u.items[0].node.kind {
            crate::ast::ItemKind::Fn(f) => assert_eq!(f.params[0].annotations.len(), 1),
            other => panic!("expected a function, got {}", other.describe()),
        }

        let (u, _) = unit("struct S { [align: 4] x: u8 }");
        match &u.items[0].node.kind {
            crate::ast::ItemKind::Struct { fields, .. } => {
                assert_eq!(fields[0].annotations.len(), 1);
            }
            other => panic!("expected a structure, got {}", other.describe()),
        }
    }

    #[test]
    fn a_semicolon_separates_annotations_when_a_new_one_follows() {
        // `[packed; align: 8]` is two annotations
        let (gs, _) = groups("[packed; align: 8]\nstruct S { x: u8 }");
        assert_eq!(gs.len(), 1, "one group");
        assert_eq!(gs[0].annotations.len(), 2, "two annotations in it");
        assert_eq!(
            names("[packed; align: 8]\nstruct S { x: u8 }"),
            vec!["packed", "align"]
        );
    }

    #[test]
    fn a_semicolon_continues_the_argument_list_when_no_annotation_follows() {
        // `[expect: monic; contract = flat(0)]`. here
        // `contract` is followed by `=`, so it qualifies the expectation
        // rather than starting a new annotation
        let (gs, _) = groups("[expect: monic; contract = flat(0)]\nfn f() { }");
        assert_eq!(gs[0].annotations.len(), 1, "one annotation, `expect`");
        assert_eq!(
            gs[0].annotations[0].node.args.len(),
            2,
            "with two arguments"
        );
        assert!(matches!(
            gs[0].annotations[0].node.args[1].node,
            AnnArg::Named { .. }
        ));
    }

    #[test]
    fn the_two_semicolon_readings_do_not_interfere() {
        // a group may use both in turn
        let (gs, _) = groups("[expect: monic; contract = flat; inline]\nfn f() { }");
        assert_eq!(gs[0].annotations.len(), 2);
        assert_eq!(gs[0].annotations[0].node.args.len(), 2);
        assert!(gs[0].annotations[1].node.args.is_empty());
    }

    #[test]
    fn malformed_annotations_are_rejected() {
        for src in ["[entry\nfn f() { }", "[: 1]\nfn f() { }", "[a:]\nfn f() { }"] {
            assert!(!errors(src).is_empty(), "{src:?} should not parse");
        }
    }
}
