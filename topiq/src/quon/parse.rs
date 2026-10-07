//! The grammar for quantum states, covers and gauges.
//!
//! # Entry points
//!
//! A quantum context opens in a handful of places: after `prep`; inside the
//! arguments of the `cover`, `gauge` and `frame` annotations; on the right of a
//! `cover`, `gauge` or `base` declaration; inside a macro argument documented
//! as taking a quantum form; and throughout a `.quon` file, which is one such
//! context from beginning to end.
//!
//! All of them reach the same productions, which live here rather than being
//! reimplemented at each entry point, so every context reads the same forms.
//!
//! # Scaling
//!
//! A term is parsed as a `*`- and `/`-separated sequence of units, which are
//! then sorted into the coefficient and the tensor factors. That allows the
//! scalar to be written before *or* after the state it scales, and allows
//! division:
//!
//! ```text
//! gauge GB   = fid((|00> + |11>) * isq2);
//! cover Orb  = fin{ (|00> + |01> + |10> + |11>) / 2, … };
//! ```
//!
//! A coefficient after the state is the usual way to write one, and a
//! normalisation reads as `/ 2` rather than a leading `1/2 *`.
//!
//! Dividing by a *state* is refused, since it has no meaning: only the
//! coefficient may be a divisor.

use chumsky::input::ValueInput;
use chumsky::prelude::*;

use crate::lex::{IntBase, Punct, Token};
use crate::parse::input::{Cx, Extra, close_angle, ident, listed, punct};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};

use super::ast::{
    DeclName, Exact, Ket, Pauli, QCover, QDoc, QExpr, QFactor, QGauge, QItem, QOp, QState, QTerm, QType,
    Sign,
};
use super::ket::reconstruct;

/// A ket.
///
/// # Refusals
///
/// In a quantum context, `|` … `>` can only be an attempted ket, so when the
/// pieces do not form one, the error says why. It is emitted rather than
/// returned: a returned failure competes with the other branches, and the
/// parser keeps whichever reached furthest, usually not this one. Parsing
/// goes on with a placeholder, so the rest of the state is still checked.
///
/// # Basis states are the only kets
///
/// Between the bars a ket holds only `0` and `1`, so `|+>` and `|->` are not
/// kets however natural they look. They are refused with a diagnostic that
/// says so and gives a spelling that works: `|+>` is `(|0> + |1>) * isq2`.
pub fn ket<'t, I>(cx: Cx<'t>) -> impl Parser<'t, I, Spanned<Ket>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    /// Stands in for a ket that was refused, so the rest of the state parses.
    fn placeholder(span: Span) -> Spanned<Ket> {
        Spanned::new(Ket { bits: "0".to_owned() }, span)
    }

    let basis = punct(Punct::Or)
        .then(select! { t @ Token::Int { .. } = e => (e.span(), t) })
        .then(punct(Punct::Gt))
        .validate(move |((bar, int), gt), _, emitter| {
            match reconstruct(bar, int, gt, cx.interner) {
                Ok(k) => Spanned::new(k, bar.join(gt)),
                Err(why) => {
                    emitter.emit(Rich::custom(bar.join(gt), why.message()));
                    placeholder(bar.join(gt))
                }
            }
        });

    // `|+>` and `|->` look like kets and are not
    let sign_ket = punct(Punct::Or)
        .then(choice((punct(Punct::Plus), punct(Punct::Minus))))
        .then(punct(Punct::Gt))
        .validate(|((bar, _), gt), _, emitter| {
            emitter.emit(Rich::custom(
                bar.join(gt),
                "`|+>` is not a ket: a ket names a computational basis state, so \
                 between the bars it holds only `0` and `1`. Write the superposition \
                 out instead: `|+>` is `(|0> + |1>) * isq2`, and `|->` is \
                 `(|0> - |1>) * isq2`",
            ));
            placeholder(bar.join(gt))
        });

    basis.or(sign_ket)
}

/// An exact-scalar constant: `i`, `isq2` or `w(k, N)`.
///
/// Recognised from ordinary identifiers, since the spellings are also
/// variable names: only in a quantum context is `i` the imaginary unit. See
/// [`crate::lex::token`].
pub fn exact<'t, I>(cx: Cx<'t>) -> impl Parser<'t, I, Spanned<Exact>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    let int = select! { Token::Int { raw, .. } = e => (raw, e.span()) };
    let w = named(cx.exact_w, "expected `w`")
        .ignore_then(
            int.then_ignore(punct(Punct::Comma))
                .then(int)
                .delimited_by(punct(Punct::LParen), punct(Punct::RParen)),
        )
        .try_map(move |((k, ks), (n, ns)), span| {
            let parse = |sym, at: Span| -> Result<u64, Rich<'t, Token, Span>> {
                cx.text(sym)
                    .replace('_', "")
                    .parse::<u64>()
                    .map_err(|_| Rich::custom(at, "the argument of `w` must be a whole number"))
            };
            Ok(Spanned::new(
                Exact::W {
                    k: parse(k, ks)?,
                    n: parse(n, ns)?,
                },
                span,
            ))
        });

    let named = ident().try_map(move |s, span| {
        if s.node == cx.exact_i {
            Ok(Spanned::new(Exact::I, s.span))
        } else if s.node == cx.exact_isq2 {
            Ok(Spanned::new(Exact::Isq2, s.span))
        } else {
            Err(Rich::custom(span, "not an exact-scalar constant"))
        }
    });

    w.or(named)
}

/// One unit of a term, before the coefficient and the factors are separated.
#[derive(Clone)]
enum Unit {
    Scalar(Spanned<QExpr>),
    Factor(Spanned<QFactor>),
}

/// A quantum state: a sum of scaled tensor products of basis states.
pub fn qstate<'t, I>(cx: Cx<'t>) -> impl Parser<'t, I, Spanned<QState>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    recursive(move |state| {
        // `qatom`: a ket, a parenthesised state, an exact constant, a numeric
        // literal, or a named state
        let literal = select! {
            Token::Int { raw, base, .. } = e => Spanned::new(QExpr::Int(raw, base), e.span()),
            Token::Float { raw, .. } = e => Spanned::new(QExpr::Float(raw), e.span()),
        };
        let atom = choice((
            ket(cx).map(|k| Unit::Factor(k.map(QFactor::Ket))),
            exact(cx).map(|x| Unit::Scalar(x.map(QExpr::Exact))),
            literal.map(Unit::Scalar),
            state
                .clone()
                .delimited_by(punct(Punct::LParen), punct(Punct::RParen))
                .map(|s: Spanned<QState>| match scalar(&s) {
                    // parentheses around coefficients alone group a
                    // coefficient, as in `(1 + i) / 2 * |0>`
                    Some(c) => Unit::Scalar(c),
                    None => {
                        let span = s.span;
                        Unit::Factor(Spanned::new(QFactor::Paren(Box::new(s)), span))
                    }
                }),
            ident().map(|s| Unit::Factor(s.map(QFactor::Name))),
        ));

        // `**` binds tighter than `*` and `/`. how a tensor product groups
        // is judgement-significant, so a tensor group is kept flat rather
        // than being re-associated
        let tensor = atom.separated_by(punct(Punct::StarStar)).at_least(1).collect::<Vec<_>>();

        // `unit (("*" | "/") unit)*`: the documented extension
        let term = tensor
            .clone()
            .then(
                choice((punct(Punct::Star).to(QOp::Mul), punct(Punct::Slash).to(QOp::Div)))
                    .then(tensor.clone())
                    .repeated()
                    .collect::<Vec<_>>(),
            )
            .try_map(move |(head, tail), span| assemble_term(cx, head, tail, span))
            .map_with(|t, e| Spanned::new(t, e.span()));

        punct(Punct::Minus)
            .or_not()
            .then(term.clone())
            .then(
                choice((punct(Punct::Plus).to(Sign::Plus), punct(Punct::Minus).to(Sign::Minus)))
                    .then(term)
                    .repeated()
                    .collect::<Vec<_>>(),
            )
            .map_with(move |((neg, head), tail), e| {
                let head = match neg {
                    // a leading `-` negates the first term's coefficient, as in
                    // `(-|00> - |01> - |10> + |11>) / 2`
                    Some(at) => head.map(|t| negate(cx, t, at)),
                    None => head,
                };
                Spanned::new(QState { head, tail }, e.span())
            })
    })
}

/// Sorts a term's units into its coefficient and its tensor factors.
fn assemble_term<'t>(
    cx: Cx<'t>,
    head: Vec<Unit>,
    tail: Vec<(QOp, Vec<Unit>)>,
    span: Span,
) -> Result<QTerm, Rich<'t, Token, Span>> {
    let mut coeff: Option<Spanned<QExpr>> = None;
    let mut factors: Vec<Spanned<QFactor>> = Vec::new();
    for (op, units) in std::iter::once((QOp::Mul, head)).chain(tail) {
        for u in units {
            match u {
                Unit::Scalar(s) => {
                    let at = s.span;
                    coeff = Some(match coeff.take() {
                        None if op == QOp::Mul => s,
                        // `x / 2` with no coefficient yet means `1 / 2`
                        None => binary(op, Spanned::new(QExpr::Int(cx.lit_one, IntBase::Decimal), at), s, at),
                        Some(acc) => {
                            let joined = acc.span.join(at);
                            binary(op, acc, s, joined)
                        }
                    });
                }
                Unit::Factor(f) if op == QOp::Div => {
                    return Err(Rich::custom(
                        f.span,
                        "a state cannot be a divisor; only the coefficient may be divided",
                    ));
                }
                Unit::Factor(f) => factors.push(f),
            }
        }
    }
    if factors.is_empty() && coeff.is_none() {
        return Err(Rich::custom(span, "a term needs at least one factor"));
    }
    Ok(QTerm { coeff, factors })
}

/// The coefficient a state of coefficients alone is, their signed sum; `None`
/// when a term has a factor.
fn scalar(s: &Spanned<QState>) -> Option<Spanned<QExpr>> {
    let head = &s.node.head.node;
    if !head.factors.is_empty() || s.node.tail.iter().any(|(_, t)| !t.node.factors.is_empty()) {
        return None;
    }
    let mut acc = head.coeff.clone()?;
    for (sign, t) in &s.node.tail {
        let rhs = t.node.coeff.clone()?;
        let op = match sign {
            Sign::Plus => QOp::Add,
            Sign::Minus => QOp::Sub,
        };
        let span = acc.span.join(rhs.span);
        acc = binary(op, acc, rhs, span);
    }
    Some(Spanned::new(acc.node, s.span))
}

/// `lhs op rhs`, spanning `span`.
fn binary(op: QOp, lhs: Spanned<QExpr>, rhs: Spanned<QExpr>, span: Span) -> Spanned<QExpr> {
    Spanned::new(
        QExpr::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        span,
    )
}

/// The identifier `want` alone; any other is refused with `refusal`.
fn named<'t, I>(want: Symbol, refusal: &'static str) -> impl Parser<'t, I, Spanned<Symbol>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    ident().try_map(move |s, span| if s.node == want { Ok(s) } else { Err(Rich::custom(span, refusal)) })
}

/// Negates a term's coefficient, introducing one if it had none.
fn negate(cx: Cx<'_>, mut t: QTerm, at: Span) -> QTerm {
    t.coeff = Some(match t.coeff.take() {
        Some(c) => {
            let span = at.join(c.span);
            Spanned::new(QExpr::Neg(Box::new(c)), span)
        }
        None => Spanned::new(
            QExpr::Neg(Box::new(Spanned::new(QExpr::Int(cx.lit_one, IntBase::Decimal), at))),
            at,
        ),
    });
    t
}

/// A Pauli string.
///
/// The letters arrive as a single identifier, because `XZ` lexes that way, so
/// the text is read here and checked letter by letter.
pub fn pauli<'t, I>(cx: Cx<'t>) -> impl Parser<'t, I, Spanned<Pauli>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    choice((
        punct(Punct::Plus).to(Sign::Plus),
        punct(Punct::Minus).to(Sign::Minus),
    ))
    .or_not()
    .then(ident())
    .try_map(move |(sign, letters), span| {
        let text = cx.text(letters.node);
        if text.is_empty() || !text.bytes().all(|b| matches!(b, b'I' | b'X' | b'Y' | b'Z')) {
            return Err(Rich::custom(
                span,
                "a Pauli string holds only the letters I, X, Y and Z",
            ));
        }
        Ok(Spanned::new(
            Pauli {
                sign,
                letters: text.to_owned(),
            },
            span,
        ))
    })
}

/// The name of a cover, gauge or base type: `T01`, or another unit's,
/// `gates::T01`.
pub fn decl_name<'t, I>() -> impl Parser<'t, I, Spanned<DeclName>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    ident()
        .then(punct(Punct::ColonColon).ignore_then(ident()).or_not())
        .map_with(|(first, second), e| {
            let name = match second {
                Some(n) => DeclName {
                    unit: Some(first.node),
                    name: n.node,
                },
                None => DeclName::local(first.node),
            };
            Spanned::new(name, e.span())
        })
}

/// A cover: the set of states a judgement is made over.
pub fn qcover<'t, I>(cx: Cx<'t>) -> impl Parser<'t, I, Spanned<QCover>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    let former = |want| named(want, "not this cover former");

    let pt = former(cx.pt)
        .ignore_then(
            qstate(cx).delimited_by(punct(Punct::LParen), punct(Punct::RParen)),
        )
        .map(|s| QCover::Pt(Box::new(s)));

    let fin = former(cx.fin)
        .ignore_then(listed(qstate(cx), Punct::LBrace, Punct::RBrace))
        .map(QCover::Fin);

    let span_cover = former(cx.span_)
        .ignore_then(listed(ket(cx), Punct::LBrace, Punct::RBrace))
        .map(QCover::Span);

    let code = former(cx.code)
        .ignore_then(
            pauli(cx)
                .separated_by(punct(Punct::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(punct(Punct::Lt), close_angle()),
        )
        .map(QCover::Code);

    choice((pt, fin, span_cover, code, decl_name().map(|s| QCover::Name(s.node))))
        .map_with(|c, e| Spanned::new(c, e.span()))
}

/// A gauge: the frame each point of a cover is measured against.
pub fn qgauge<'t, I>(cx: Cx<'t>) -> impl Parser<'t, I, Spanned<QGauge>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    recursive(move |gauge| {
        let former = |want| named(want, "not this gauge former");

        let fid = former(cx.fid)
            .ignore_then(qstate(cx).delimited_by(punct(Punct::LParen), punct(Punct::RParen)))
            .map(|s| QGauge::Fid(Box::new(s)));

        let atlas = former(cx.atlas)
            .ignore_then(listed(gauge.clone(), Punct::LBrace, Punct::RBrace))
            .map(QGauge::Atlas);

        let none = former(cx.none).to(QGauge::None);

        choice((fid, atlas, none, decl_name().map(|s| QGauge::Name(s.node))))
            .map_with(|g, e| Spanned::new(g, e.span()))
    })
}

/// A register type, as an item of a `.quon` document states it.
///
/// A register's length is a constant expression in ordinary source, but a
/// `.quon` file is a standalone document with no access to the expression
/// grammar. Only the two spellings such a document can actually carry are
/// accepted here: an integer literal, and a name resolved later.
pub fn qtype<'t, I>() -> impl Parser<'t, I, Spanned<QType>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    let length = select! {
        Token::Int { raw, .. } = e => Spanned::new(raw, e.span()),
    }
    .or(ident());

    let register = punct(Punct::LBracket)
        .ignore_then(ident())
        .ignore_then(punct(Punct::Semi))
        .ignore_then(length)
        .then_ignore(punct(Punct::RBracket))
        .map(QType::Register);

    let path_or_qubit = ident()
        .separated_by(punct(Punct::ColonColon))
        .at_least(1)
        .collect::<Vec<_>>()
        .map(|segs: Vec<Spanned<Symbol>>| {
            QType::Path(segs.into_iter().map(|s| s.node).collect())
        });

    register
        .or(path_or_qubit)
        .map_with(|t, e| Spanned::new(t, e.span()))
}

/// One item of a `.quon` document: a name, a register type, and a state.
pub fn qitem<'t, I>(cx: Cx<'t>) -> impl Parser<'t, I, QItem, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    ident()
        .then_ignore(punct(Punct::Colon))
        .then(qtype())
        .then_ignore(punct(Punct::Eq))
        .then(qstate(cx))
        .then_ignore(choice((punct(Punct::Comma), punct(Punct::Semi))))
        .map(|((name, ty), state)| QItem { name, ty, state })
}

/// A `.quon` document.
///
/// The whole file is a quantum context, which is why kets need no special
/// handling here: every `|0>` in such a file is a ket, with nothing else it
/// could be.
pub fn qdoc<'t, I>(cx: Cx<'t>) -> impl Parser<'t, I, QDoc, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    qitem(cx)
        .repeated()
        .collect::<Vec<_>>()
        .map(|items| QDoc { items })
}

/// Parsing a state alone, for other modules' tests.
#[cfg(test)]
pub mod testing {
    use chumsky::Parser;

    use super::qstate;
    use crate::intern::Interner;
    use crate::parse::input::{Cx, TokenStream, eoi_span, stream};
    use crate::quon::ast::QState;
    use crate::source::Spliced;
    use crate::span::{SourceId, Spanned};

    /// `src` parsed as a state, with the string table its names are in.
    ///
    /// # Panics
    ///
    /// When `src` is not a state.
    pub fn state_in(src: &str) -> (Spanned<QState>, Interner) {
        let interner = Interner::new();
        let spliced = Spliced::from_text(src);
        let lexed = crate::lex::lex(SourceId(0), &spliced, &interner);
        let tokens = lexed.tokens;
        let eoi = eoi_span(SourceId(0), &tokens);
        let s = qstate::<TokenStream>(Cx::new(&interner))
            .parse(stream(&tokens, eoi))
            .into_result()
            .unwrap_or_else(|e| panic!("{src:?} did not parse: {e:?}"));
        (s, interner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::parse::input::{TokenStream, eoi_span, stream};
    use crate::source::Spliced;
    use crate::span::SourceId;

    /// Runs one QUON production over `src`.
    macro_rules! run {
        ($src:expr, $make:expr) => {{
            let mut interner = Interner::new();
            let spliced = Spliced::from_text($src);
            let lexed = crate::lex::lex(SourceId(0), &spliced, &mut interner);
            assert!(lexed.diagnostics.is_empty(), "lex: {:?}", lexed.diagnostics);
            let tokens = lexed.tokens;
            let eoi = eoi_span(SourceId(0), &tokens);
            // the parse context borrows the string table, so the borrow has
            // to end before the table is handed back
            let out = {
                let cx = Cx::new(&mut interner);
                $make(cx)
                    .parse(stream(&tokens, eoi))
                    .into_result()
                    .map_err(|es| es.iter().map(ToString::to_string).collect::<Vec<_>>())
            };
            (out, interner)
        }};
    }

    fn state(src: &str) -> QState {
        let (r, _) = run!(src, qstate::<TokenStream>);
        r.unwrap_or_else(|e| panic!("{src:?} did not parse: {e:?}")).node
    }

    /// Everything reported, whether or not the parse still produced a tree.
    ///
    /// A refusal emitted by `validate` comes back alongside output, so
    /// `into_result` would hide it.
    fn state_err(src: &str) -> Vec<String> {
        let interner = Interner::new();
        let spliced = Spliced::from_text(src);
        let lexed = crate::lex::lex(SourceId(0), &spliced, &interner);
        let tokens = lexed.tokens;
        let eoi = eoi_span(SourceId(0), &tokens);
        let cx = Cx::new(&interner);
        let (_, errs) = qstate::<TokenStream>(cx)
            .parse(stream(&tokens, eoi))
            .into_output_errors();
        errs.iter().map(ToString::to_string).collect()
    }

    fn cover(src: &str) -> QCover {
        let (r, _) = run!(src, qcover::<TokenStream>);
        r.unwrap_or_else(|e| panic!("{src:?} did not parse: {e:?}")).node
    }

    fn gauge(src: &str) -> QGauge {
        let (r, _) = run!(src, qgauge::<TokenStream>);
        r.unwrap_or_else(|e| panic!("{src:?} did not parse: {e:?}")).node
    }

    //////////
    // KETS //
    //////////

    #[test]
    fn a_ket_is_rebuilt_from_three_tokens() {
        // outside a quantum context, `|0>` is three ordinary tokens
        let s = state("|0>");
        assert_eq!(s.term_count(), 1);
        match &s.head.node.factors[0].node {
            QFactor::Ket(k) => assert_eq!(k.bits, "0"),
            other => panic!("expected a ket, got {other:?}"),
        }
    }

    #[test]
    fn kets_of_several_widths_parse() {
        for (src, bits) in [("|0>", "0"), ("|11>", "11"), ("|0110>", "0110")] {
            match &state(src).head.node.factors[0].node {
                QFactor::Ket(k) => assert_eq!(k.bits, bits),
                other => panic!("expected a ket, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_spaced_ket_is_refused_with_its_reason() {
        let e = state_err("| 0 >");
        assert!(
            e.iter().any(|m| m.contains("no spaces inside it")),
            "expected the adjacency rule to be named, got {e:?}"
        );
    }

    #[test]
    fn a_based_or_suffixed_literal_is_not_a_ket() {
        assert!(!state_err("|0b1>").is_empty());
        assert!(!state_err("|01i8>").is_empty());
        assert!(!state_err("|02>").is_empty());
    }

    ////////////
    // STATES //
    ////////////

    #[test]
    fn a_sum_of_terms_parses() {
        // the members of `fin{ |00>, |11> }`
        let s = state("|00> + |11>");
        assert_eq!(s.term_count(), 2);
        assert_eq!(s.tail[0].0, Sign::Plus);
    }

    #[test]
    fn a_signed_sum_records_each_sign() {
        let s = state("|00> - |01> + |10>");
        assert_eq!(s.term_count(), 3);
        assert_eq!(s.tail[0].0, Sign::Minus);
        assert_eq!(s.tail[1].0, Sign::Plus);
    }

    #[test]
    fn a_leading_minus_negates_the_first_term() {
        // a signed superposition: `(-|00> - |01> - |10> + |11>) / 2`
        let s = state("-|00> - |01>");
        assert!(
            s.head.node.coeff.is_some(),
            "the leading sign should become a coefficient"
        );
        assert!(matches!(s.head.node.coeff.as_ref().unwrap().node, QExpr::Neg(_)));
    }

    #[test]
    fn a_coefficient_may_precede_a_state() {
        let s = state("isq2 * |00>");
        assert!(s.head.node.coeff.is_some());
        assert_eq!(s.head.node.factors.len(), 1);
    }

    #[test]
    fn a_coefficient_may_follow_a_state() {
        // `fid((|00> + |11>) * isq2)`: the scalar written after the state it
        // scales, which is how anyone actually writes a Bell gauge
        let s = state("(|00> + |11>) * isq2");
        assert!(s.head.node.coeff.is_some());
        assert_eq!(s.head.node.factors.len(), 1, "the parenthesised state");
    }

    #[test]
    fn a_state_may_be_divided_by_a_scalar() {
        // normalising by division: `(|00> + |01> + |10> + |11>) / 2`
        let s = state("(|00> + |01> + |10> + |11>) / 2");
        match s.head.node.coeff.as_ref().map(|c| &c.node) {
            Some(QExpr::Binary { op, .. }) => assert_eq!(*op, QOp::Div),
            other => panic!("expected a division, got {other:?}"),
        }
    }

    #[test]
    fn dividing_by_a_state_is_refused() {
        let e = state_err("2 / |00>");
        assert!(
            e.iter().any(|m| m.contains("divisor")),
            "expected the reason to be named, got {e:?}"
        );
    }

    #[test]
    fn the_exact_constants_are_recognized_from_identifiers() {
        // see `crate::lex::token` for why these are not a token class
        for (src, want) in [
            ("i * |0>", Exact::I),
            ("isq2 * |0>", Exact::Isq2),
            ("w(1, 8) * |0>", Exact::W { k: 1, n: 8 }),
        ] {
            let s = state(src);
            match s.head.node.coeff.as_ref().map(|c| &c.node) {
                Some(QExpr::Exact(e)) => assert_eq!(*e, want, "for {src}"),
                other => panic!("expected an exact constant for {src}, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_tensor_product_keeps_its_factors_in_order() {
        // a grouping is judgement-significant and must not be
        // re-associated, so the factors stay a flat sequence
        let s = state("|0> ** |1> ** |0>");
        assert_eq!(s.head.node.factors.len(), 3);
    }

    #[test]
    fn parentheses_around_coefficients_alone_group_a_coefficient() {
        let s = state("(1 + i) / 2 * |0>");
        assert_eq!(s.head.node.factors.len(), 1);
        assert!(matches!(
            s.head.node.coeff.as_ref().map(|c| &c.node),
            Some(QExpr::Binary { op: QOp::Div, .. })
        ));
        // parentheses around a state stay a factor
        let s = state("(|0> + |1>) / 2");
        assert!(matches!(s.head.node.factors[0].node, QFactor::Paren(_)));
    }

    #[test]
    fn a_named_state_parses() {
        let s = state("psi");
        assert!(matches!(s.head.node.factors[0].node, QFactor::Name(_)));
    }

    ///////////////////////
    // COVERS AND GAUGES //
    ///////////////////////

    #[test]
    fn the_four_cover_formers_parse() {
        assert!(matches!(cover("pt(|11>)"), QCover::Pt(_)));
        match cover("fin{ |00>, |11> }") {
            QCover::Fin(ss) => assert_eq!(ss.len(), 2),
            other => panic!("expected a fin cover, got {other:?}"),
        }
        match cover("span{ |00>, |01>, |10> }") {
            QCover::Span(ks) => assert_eq!(ks.len(), 3),
            other => panic!("expected a span cover, got {other:?}"),
        }
        match cover("code<XZ, ZX>") {
            QCover::Code(ps) => assert_eq!(ps.len(), 2),
            other => panic!("expected a code cover, got {other:?}"),
        }
    }

    #[test]
    fn a_named_cover_parses() {
        assert!(matches!(cover("Orb"), QCover::Name(DeclName { unit: None, .. })));
    }

    #[test]
    fn another_units_cover_parses() {
        assert!(matches!(cover("gates::T01"), QCover::Name(DeclName { unit: Some(_), .. })));
    }

    #[test]
    fn a_pauli_string_admits_only_ixyz() {
        match cover("code<-XYZI>") {
            QCover::Code(ps) => {
                assert_eq!(ps[0].node.letters, "XYZI");
                assert_eq!(ps[0].node.sign, Some(Sign::Minus));
            }
            other => panic!("expected a code cover, got {other:?}"),
        }
        let (r, _) = run!("code<ABC>", qcover::<TokenStream>);
        assert!(r.is_err(), "`ABC` is not a Pauli string");
    }

    #[test]
    fn a_sign_ket_is_refused_with_the_spelling_that_works() {
        // only `0` and `1` may appear in a ket, so `|+>` is refused, but it
        // is a natural thing to reach for, so the diagnostic should teach the
        // spelling that works rather than merely say no
        let e = state_err("|+>");
        assert!(
            e.iter().any(|m| m.contains("(|0> + |1>) * isq2")),
            "expected the working spelling to be suggested, got {e:?}"
        );
        assert!(!state_err("|->").is_empty());
    }

    #[test]
    fn the_three_gauge_formers_parse() {
        assert!(matches!(gauge("fid((|00> + |11>) * isq2)"), QGauge::Fid(_)));
        match gauge("atlas{ fid(|00>), fid(|11>) }") {
            QGauge::Atlas(gs) => assert_eq!(gs.len(), 2),
            other => panic!("expected an atlas, got {other:?}"),
        }
        assert_eq!(gauge("none"), QGauge::None);
        assert!(matches!(gauge("GOrb"), QGauge::Name(_)));
    }

    #[test]
    fn a_stationary_only_gauge_is_distinct_from_a_named_one() {
        // `none` is the form a subspace cover requires
        assert_ne!(gauge("none"), gauge("GNone"));
    }

    ///////////////////////
    // `.quon` DOCUMENTS //
    ///////////////////////

    #[test]
    fn a_quon_document_parses_its_items() {
        // a `.quon` file holds a sequence of named states
        let (r, i) = run!(
            "bell: [qubit; 2] = (|00> + |11>) * isq2;\n\
             plus: qubit = (|0> + |1>) * isq2;",
            qdoc::<TokenStream>
        );
        let doc = r.unwrap_or_else(|e| panic!("did not parse: {e:?}"));
        assert_eq!(doc.items.len(), 2);
        assert_eq!(i.resolve(doc.items[0].name.node), "bell");
        assert!(matches!(doc.items[0].ty.node, QType::Register(_)));
        assert!(matches!(doc.items[1].ty.node, QType::Path(_)));
    }

    #[test]
    fn a_quon_item_may_end_with_a_comma() {
        // an item is a name, a register type, a state, and a separator
        let (r, _) = run!("a: qubit = |0>,", qdoc::<TokenStream>);
        assert_eq!(r.unwrap().items.len(), 1);
    }

    #[test]
    fn an_empty_quon_document_parses() {
        let (r, _) = run!("", qdoc::<TokenStream>);
        assert!(r.unwrap().items.is_empty());
    }

    #[test]
    fn a_malformed_quon_document_is_rejected() {
        for src in ["a: qubit = |0>", "a = |0>;", "a: qubit |0>;"] {
            let (r, _) = run!(src, qdoc::<TokenStream>);
            assert!(r.is_err(), "{src:?} should not parse");
        }
    }
}
