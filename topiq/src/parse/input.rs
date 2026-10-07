//! The `chumsky` input and the shared token combinators.
//!
//! This is the one module that names the parser's input type, its error type
//! and its state, so that every other module in [`crate::parse`] can be written
//! against the aliases here.
//!
//! # Parsers are generic over the input
//!
//! `chumsky` 0.13 has no `Input::spanned`; a token slice becomes a parser input
//! through [`chumsky::input::Input::map`], which produces an opaque
//! `MappedInput` type. Naming that type in a signature is unpleasant, so every
//! parser here is generic over `I: ValueInput<'t, Token = Token, Span = Span>`
//! and [`stream`] does the wrapping at the single call site.
//!
//! # Rejoining `>`
//!
//! The lexer never glues `>` (see [`crate::lex::token`] for why), so the
//! parser rebuilds `>>`, `>=` and `>>=` here, requiring span adjacency. This is
//! the same rule [`crate::pp::eval`] applies to a `#if` condition, and both are
//! written against these helpers so the two cannot drift apart.

use chumsky::input::{Input, MappedInput, ValueInput};
use chumsky::prelude::*;

use crate::ast::Doc;
use crate::intern::Symbol;
use crate::lex::{DocComment, DocKind, Keyword, Punct, Token};
use crate::span::{SourceId, Span, Spanned};

/// The error type every parser in this module produces.
pub type Error<'t> = Rich<'t, Token, Span>;

/// Separates a custom error's message from its note and help; see
/// [`explained`].
pub const PART: char = '\u{1f}';

/// A custom error with a note saying why and a help saying what to write
/// instead, which the diagnostic built from it shows as such.
pub fn explained<'t>(span: Span, message: &str, note: &str, help: &str) -> Error<'t> {
    Rich::custom(span, format!("{message}{PART}{note}{PART}{help}"))
}

/// The `extra` bundle: rich errors, no state, no context.
pub type Extra<'t> = extra::Err<Error<'t>>;

/// The concrete input produced by [`stream`].
///
/// The type is unwieldy because `chumsky` 0.13's `Input::map` returns an opaque
/// `MappedInput` parameterised by the projection function. Naming it once here
/// is the point: every parser is written generically over
/// `I: ValueInput<'t, Token = Token, Span = Span>` and never has to.
#[allow(clippy::type_complexity, reason = "this alias exists to hold the complexity")]
pub type TokenStream<'t> =
    MappedInput<'t, Token, Span, &'t [(Token, Span)], fn(&'t (Token, Span)) -> (&'t Token, &'t Span)>;

/// Wraps a token slice as a `chumsky` input.
///
/// `eoi` is the span reported for an unexpected end of input; it should be an
/// empty span at the end of the file so a "this file ends too early" diagnostic
/// points somewhere real.
pub fn stream<'t>(tokens: &'t [(Token, Span)], eoi: Span) -> TokenStream<'t> {
    fn project(pair: &(Token, Span)) -> (&Token, &Span) {
        (&pair.0, &pair.1)
    }
    tokens.map(eoi, project as fn(&'t (Token, Span)) -> (&'t Token, &'t Span))
}

/// An empty span at the end of a file.
pub fn eoi_span(source: SourceId, tokens: &[(Token, Span)]) -> Span {
    match tokens.last() {
        Some((_, s)) => Span::at(s.source, s.end),
        None => Span::at(source, 0),
    }
}

/// Matches one punctuator.
pub fn punct<'t, I>(p: Punct) -> impl Parser<'t, I, Span, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    just(Token::Punct(p)).map_with(|_, e| e.span())
}

/// `p, p, …` between the punctuation `open` and `close`, a trailing comma
/// allowed.
pub fn listed<'t, I, O, P>(p: P, open: Punct, close: Punct) -> impl Parser<'t, I, Vec<O>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
    P: Parser<'t, I, O, Extra<'t>> + Clone,
{
    p.separated_by(punct(Punct::Comma))
        .allow_trailing()
        .collect::<Vec<_>>()
        .delimited_by(punct(open), punct(close))
}

/// Matches one keyword.
pub fn kw<'t, I>(k: Keyword) -> impl Parser<'t, I, Span, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    just(Token::Kw(k)).map_with(|_, e| e.span())
}

/// Matches an identifier.
pub fn ident<'t, I>() -> impl Parser<'t, I, Spanned<Symbol>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    select! { Token::Ident(s) = e => Spanned::new(s, e.span()) }
}

/// The interned spelling of every keyword.
///
/// A `chumsky` parser is an `Fn` built once and reused, so it cannot borrow the
/// interner. This table is `Copy`, is built before parsing, and is captured by
/// the parsers that need to turn a keyword token back into a name.
#[derive(Clone, Copy, Debug)]
pub struct Keywords {
    syms: [Symbol; 39],
}

impl Keywords {
    /// Interns every keyword's spelling.
    pub fn new(interner: &crate::intern::Interner) -> Keywords {
        let mut syms = [Symbol::EMPTY; 39];
        for (slot, kw) in syms.iter_mut().zip(Keyword::ALL) {
            *slot = interner.intern_late(kw.text());
        }
        Keywords { syms }
    }

    /// The symbol for a keyword's spelling.
    pub fn of(&self, k: Keyword) -> Symbol {
        let idx = Keyword::ALL
            .iter()
            .position(|x| *x == k)
            .expect("every keyword is in Keyword::ALL");
        self.syms[idx]
    }
}

/// Everything the grammar must recognise by name, plus the string table.
///
/// # Why the interner is here
///
/// A `chumsky` parser is an `Fn` built once and reused, so it cannot borrow
/// mutably; nothing is interned while parsing, so a shared borrow serves the
/// productions that read text:
///
/// - a ket reassembled from three tokens must hold `[01]+`, with no base
///   prefix, separator or suffix, so `|0110>` is one and `|0b1>` is not;
/// - the exact-scalar constants `i`, `isq2` and `w(k, N)` are recognised from
///   identifiers; see [`crate::lex::token`].
///
/// # Type formers that are not reserved words
///
/// `closure`, `circuit` and `dyn` form types, but none of them is a keyword, so
/// each arrives here as an ordinary identifier and must be recognised by name
/// in type position. `void` is not in this group (it *is* reserved).
#[derive(Clone, Copy, Debug)]
pub struct Cx<'i> {
    /// The string table.
    pub interner: &'i crate::intern::Interner,
    /// Every keyword's interned spelling.
    pub kws: Keywords,
    /// `closure`.
    pub closure: Symbol,
    /// `circuit`.
    pub circuit: Symbol,
    /// `dyn`.
    pub dyn_: Symbol,
    /// `self`, which is an ordinary identifier rather than a keyword.
    pub self_: Symbol,
    /// `_`, the wildcard pattern. An identifier may begin with `_`, so the
    /// wildcard arrives as an identifier rather than a punctuator.
    pub underscore: Symbol,
    /// `i`, the imaginary unit (in a quantum context only; everywhere else
    /// it is an ordinary name).
    pub exact_i: Symbol,
    /// `isq2`, one over the square root of two.
    pub exact_isq2: Symbol,
    /// `w`, the root-of-unity constant `w(k, N)`.
    pub exact_w: Symbol,
    /// `frame`, one of the three annotations whose argument is a quantum
    /// form.
    pub frame: Symbol,
    /// `pt`, which forms a cover from a single point.
    pub pt: Symbol,
    /// `fin`, a cover former.
    pub fin: Symbol,
    /// `span`, a cover former.
    pub span_: Symbol,
    /// `code`, a cover former.
    pub code: Symbol,
    /// `fid`, a gauge former.
    pub fid: Symbol,
    /// `atlas`, a gauge former.
    pub atlas: Symbol,
    /// `none`, the stationary-only gauge.
    pub none: Symbol,
    /// `1`, the implicit coefficient a QUON term gets when one is needed but
    /// none was written, as in `(|00> + |11>) / 2`.
    pub lit_one: Symbol,
    /// The unit's doc comments, in source order, which declarations take by
    /// their anchors.
    pub docs: &'i [DocComment],
}

impl<'i> Cx<'i> {
    /// Interns everything the grammar recognises by name, then keeps a shared
    /// borrow of the table for the productions that must read text.
    pub fn new(interner: &'i crate::intern::Interner) -> Cx<'i> {
        let kws = Keywords::new(interner);
        let closure = interner.intern_late("closure");
        let circuit = interner.intern_late("circuit");
        let dyn_ = interner.intern_late("dyn");
        let self_ = interner.intern_late("self");
        let underscore = interner.intern_late("_");
        let exact_i = interner.intern_late("i");
        let exact_isq2 = interner.intern_late("isq2");
        let exact_w = interner.intern_late("w");
        let frame = interner.intern_late("frame");
        let pt = interner.intern_late("pt");
        let fin = interner.intern_late("fin");
        let span_ = interner.intern_late("span");
        let code = interner.intern_late("code");
        let fid = interner.intern_late("fid");
        let atlas = interner.intern_late("atlas");
        let none = interner.intern_late("none");
        let lit_one = interner.intern_late("1");
        Cx {
            interner,
            kws,
            closure,
            circuit,
            dyn_,
            self_,
            underscore,
            exact_i,
            exact_isq2,
            exact_w,
            frame,
            pt,
            fin,
            span_,
            code,
            fid,
            atlas,
            none,
            lit_one,
            docs: &[],
        }
    }

    /// The same context, giving declarations the doc comments `docs`, which
    /// must be in source order.
    pub fn with_docs(self, docs: &'i [DocComment]) -> Cx<'i> {
        Cx { docs, ..self }
    }

    /// The text of a symbol.
    pub fn text(&self, s: Symbol) -> &'i str {
        self.interner.resolve(s)
    }

    /// The documentation of `///` comments anchored from `from` to `to`
    /// inclusive: those before a declaration, whose tokens start there.
    pub fn outer_docs(&self, from: u32, to: u32) -> Option<Doc> {
        // anchors rise with the comments, and an unanchored one ends the list
        let anchor = |d: &DocComment| d.anchor.unwrap_or(u32::MAX);
        let first = self.docs.partition_point(|d| anchor(d) < from);
        let last = self.docs.partition_point(|d| anchor(d) <= to);
        let docs = self.docs.get(first..last)?;
        Doc::from_comments(docs.iter().filter(|d| d.kind == DocKind::Outer))
    }
}

/// Matches an identifier *or* a keyword, returning the name.
///
/// Needed wherever the grammar writes `IDENT` for a name in a space of its own
/// that is also a reserved word. Annotation names have a name space of their
/// own (and two of the standard annotations are spelt `cover` and `gauge`),
/// as do field and variant names.
pub fn name<'t, I>(kws: Keywords) -> impl Parser<'t, I, Spanned<Symbol>, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    select! {
        Token::Ident(s) = e => Spanned::new(s, e.span()),
        Token::Kw(k) = e => Spanned::new(kws.of(k), e.span()),
    }
}

/// Matches `>` `>` written adjacently, which is the shift operator.
pub fn shr<'t, I>() -> impl Parser<'t, I, Span, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    punct(Punct::Gt)
        .then(punct(Punct::Gt))
        .try_map(join_adjacent(">>"))
}

/// Matches `>` `=` written adjacently, which is the `>=` comparison.
pub fn ge<'t, I>() -> impl Parser<'t, I, Span, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    punct(Punct::Gt)
        .then(punct(Punct::Eq))
        .try_map(join_adjacent(">="))
}

/// Matches `>` `>` `=` written adjacently, which is `>>=`.
pub fn shr_assign<'t, I>() -> impl Parser<'t, I, Span, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    punct(Punct::Gt)
        .then(punct(Punct::Gt))
        .try_map(join_adjacent(">>"))
        .then(punct(Punct::Eq))
        .try_map(join_adjacent(">>="))
}

/// Builds the adjacency check two `>`-derived tokens must pass.
///
/// Without it `a > > b` would read as a shift, and `Vec<Vec<T> >` would fail to
/// close. Adjacency is exactly what distinguishes the two.
fn join_adjacent<'t>(what: &'static str) -> impl Fn((Span, Span), Span) -> Result<Span, Error<'t>> + Clone {
    move |(a, b), _| {
        if a.adjacent_to(b) {
            Ok(a.join(b))
        } else {
            Err(Rich::custom(
                a.join(b),
                format!("`{what}` must be written without a space"),
            ))
        }
    }
}

/// The spans of the two characters of a doubled token, `**` or `&&`.
///
/// Where no binary operator can stand (in prefix position, or at the start
/// of a type), such a token is two operators, and each then points at its own
/// character. A token from a macro expansion may not span two characters of
/// source; both halves then take its whole span.
pub fn halves(s: Span) -> (Span, Span) {
    if s.len() == 2 {
        (Span::new(s.source, s.start, s.start + 1), Span::new(s.source, s.start + 1, s.end))
    } else {
        (s, s)
    }
}

/// A closing `>` of a generic argument list.
///
/// Just a `>`: because the lexer never glues them, `Vec<Vec<T>>` closes with
/// two ordinary `>` tokens, with nothing to split.
pub fn close_angle<'t, I>() -> impl Parser<'t, I, Span, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    punct(Punct::Gt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::source::Spliced;

    fn lex(src: &str) -> (Vec<(Token, Span)>, Interner) {
        let i = Interner::new();
        let spliced = Spliced::from_text(src);
        let out = crate::lex::lex(SourceId(0), &spliced, &i);
        assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
        (out.tokens, i)
    }

    /// Parses `src` with `p`, which is built fresh for the borrow.
    macro_rules! run {
        ($src:expr, $p:expr) => {{
            let (toks, _interner) = lex($src);
            let eoi = eoi_span(SourceId(0), &toks);
            $p.parse(stream(&toks, eoi))
                .into_result()
                .map_err(|es| es.iter().map(ToString::to_string).collect::<Vec<_>>())
        }};
    }

    #[test]
    fn a_token_slice_becomes_a_parser_input() {
        let got = run!("fn", kw::<TokenStream>(Keyword::Fn).map(|_| 1u8));
        assert_eq!(got, Ok(1));
    }

    #[test]
    fn punctuators_match() {
        assert!(run!(";", punct::<TokenStream>(Punct::Semi)).is_ok());
        assert!(run!(",", punct::<TokenStream>(Punct::Semi)).is_err());
    }

    #[test]
    fn identifiers_match_and_carry_a_span() {
        let (toks, i) = lex("alpha");
        let eoi = eoi_span(SourceId(0), &toks);
        let got = ident::<TokenStream>()
            .parse(stream(&toks, eoi))
            .into_result()
            .unwrap();
        assert_eq!(i.resolve(got.node), "alpha");
        assert_eq!((got.span.start, got.span.end), (0, 5));
    }

    #[test]
    fn a_keyword_can_serve_as_a_name() {
        // annotation names have a name space of their own, and one of the
        // standard annotations is spelt `cover`, which is also a keyword
        let i = Interner::new();
        let kws = Keywords::new(&i);
        let spliced = Spliced::from_text("cover");
        let out = crate::lex::lex(SourceId(0), &spliced, &i);
        let eoi = eoi_span(SourceId(0), &out.tokens);
        let got = name::<TokenStream>(kws)
            .parse(stream(&out.tokens, eoi))
            .into_result()
            .unwrap();
        assert_eq!(i.resolve(got.node), "cover");
    }

    #[test]
    fn the_keyword_table_covers_every_keyword() {
        let i = Interner::new();
        let kws = Keywords::new(&i);
        for &k in Keyword::ALL {
            assert_eq!(i.resolve(kws.of(k)), k.text());
        }
    }

    #[test]
    fn adjacent_gt_tokens_join_into_a_shift() {
        assert!(run!(">>", shr::<TokenStream>()).is_ok());
        assert!(run!(">=", ge::<TokenStream>()).is_ok());
        assert!(run!(">>=", shr_assign::<TokenStream>()).is_ok());
    }

    #[test]
    fn spaced_gt_tokens_do_not_join() {
        // this is the point of the adjacency test: `Vec<Vec<T> >` must close
        // two generic lists rather than read as a shift
        assert!(run!("> >", shr::<TokenStream>()).is_err());
        assert!(run!("> =", ge::<TokenStream>()).is_err());
    }

    #[test]
    fn a_joined_span_covers_both_tokens() {
        let span = run!(">>", shr::<TokenStream>()).unwrap();
        assert_eq!((span.start, span.end), (0, 2));
    }

    #[test]
    fn nested_generics_close_with_two_ordinary_angles() {
        // two nested generic lists closing at once, which needs no token
        // splitting given that the lexer never glues `>>` in the first place
        let got = run!(
            ">>",
            close_angle::<TokenStream>()
                .then(close_angle::<TokenStream>())
                .map(|_| ())
        );
        assert!(got.is_ok());
    }

    #[test]
    fn the_end_of_input_span_points_past_the_last_token() {
        let (toks, _) = lex("let x");
        let eoi = eoi_span(SourceId(0), &toks);
        assert_eq!(eoi.start, 5);
        assert!(eoi.is_empty());

        let (empty, _) = lex("");
        assert_eq!(eoi_span(SourceId(0), &empty), Span::at(SourceId(0), 0));
    }

    #[test]
    fn an_error_names_what_was_expected() {
        let err = run!(";", kw::<TokenStream>(Keyword::Fn).map(|_| ())).unwrap_err();
        assert!(!err.is_empty());
    }
}
