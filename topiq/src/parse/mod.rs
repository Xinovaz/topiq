//! Translation phase 5: turning tokens into a syntax tree.
//!
//! The grammar is closed. A token sequence is a Topiq unit if and only if it
//! derives from `unit`: there is no "accepted as an extension", and nothing is
//! tolerated because it looked close enough.
//!
//! # One module per syntactic category
//!
//! | module | what it parses |
//! |---|---|
//! | [`input`] | the token input itself, including rejoining split `>` tokens |
//! | [`ty`] | types, paths and generic arguments |
//! | [`pat`] | patterns |
//! | [`expr`] | expressions |
//! | [`stmt`] | blocks and statements |
//! | [`item`] | items, and everything one can declare |
//! | [`annot`] | annotation groups |
//!
//! The categories are mutually recursive (a type holds a constant expression,
//! an expression holds a block, a block holds items, an item holds types), so
//! [`grammar`] declares one recursive handle per category and defines them all
//! together. Each module's `grammar` function takes the handles it needs as
//! ordinary parser arguments, which keeps the parser-generator's recursive
//! types out of every signature.
//!
//! # What has already happened
//!
//! By the time this runs, [`crate::pp`] has executed the directives and
//! expanded the macros, and [`crate::mark`] has decided what every `[` means.
//! The parser reads a stream in which those questions are already settled, and
//! never revisits them.

pub mod annot;
pub mod expr;
pub mod input;
pub mod item;
pub mod pat;
pub mod stmt;
pub mod ty;

use chumsky::input::ValueInput;
use chumsky::prelude::*;

use crate::ast::{Doc, Item, Unit};
use crate::diag::{Code, Diagnostic};
use crate::lex::{DocKind, Keyword, Punct, Token};
use crate::pp::UnitKind;
use crate::span::{Span, Spanned};

use input::punct;
pub use input::{Cx, TokenStream, eoi_span, stream};

/// Builds the whole grammar. A unit is a sequence of items.
pub fn grammar<'t, I>(cx: Cx<'t>) -> impl Parser<'t, I, Vec<Spanned<Item>>, input::Extra<'t>>
where
    I: ValueInput<'t, Token = Token, Span = Span> + 't,
{
    let mut ty = Recursive::declare();
    let mut pattern = Recursive::declare();
    let mut expression = Recursive::declare();
    let mut expression_ns = Recursive::declare();
    let mut block = Recursive::declare();
    let mut item = Recursive::declare();

    // the quantum grammar has no back-edge into the rest of the grammar, so it
    // is built once, up front, rather than declared recursively
    let qstate = crate::quon::parse::qstate(cx);
    let annotation = annot::grammar(cx, expression.clone(), ty.clone());

    ty.define(ty::grammar(cx, ty.clone(), expression.clone()));
    pattern.define(pat::grammar(cx, pattern.clone()));
    expression.define(expr::grammar(
        cx,
        false,
        expression.clone(),
        expression_ns.clone(),
        ty.clone(),
        pattern.clone(),
        block.clone(),
        qstate.clone(),
    ));
    // the same chain with structure literals excluded, for the positions
    // where a `{` would otherwise be read as opening a body rather than a
    // literal
    expression_ns.define(expr::grammar(
        cx,
        true,
        expression.clone(),
        expression_ns.clone(),
        ty.clone(),
        pattern.clone(),
        block.clone(),
        qstate,
    ));
    block.define(stmt::grammar(
        cx,
        expression.clone(),
        expression_ns.clone(),
        ty.clone(),
        pattern.clone(),
        block.clone(),
        item.clone(),
    ));
    item.define(item::grammar(
        cx,
        expression,
        ty,
        block,
        annotation,
    ));

    // recovery: skip to the next token that could begin an item, so one error
    // does not hide the next. the first token is always skipped, since the
    // failing item usually starts with an item keyword itself
    let starts_item = choice((
        kw_any_item_start().ignored(),
        punct(Punct::LBracketAnnot).ignored(),
    ));
    let skip_to_next_item = any()
        .ignore_then(any().and_is(starts_item.not()).repeated())
        .ignored()
        .map(|()| None);

    item.map(Some)
        .recover_with(via_parser(skip_to_next_item))
        .repeated()
        .collect::<Vec<_>>()
        .map(|items: Vec<Option<Spanned<Item>>>| items.into_iter().flatten().collect())
}

/// Any keyword that can begin an item, plus the `static` linkage
/// specifier. Used only to resynchronise after a syntax error.
fn kw_any_item_start<'t, I>() -> impl Parser<'t, I, Span, input::Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
{
    one_of([
        Token::Kw(Keyword::Fn),
        Token::Kw(Keyword::Struct),
        Token::Kw(Keyword::Enum),
        Token::Kw(Keyword::Impl),
        Token::Kw(Keyword::Let),
        Token::Kw(Keyword::Type),
        Token::Kw(Keyword::Import),
        Token::Kw(Keyword::Cover),
        Token::Kw(Keyword::Gauge),
        Token::Kw(Keyword::Base),
        Token::Kw(Keyword::Locale),
        Token::Kw(Keyword::Chain),
        Token::Kw(Keyword::Qmap),
        Token::Kw(Keyword::Static),
    ])
    .map_with(|_, e| e.span())
}

/// The result of parsing one unit.
#[derive(Clone, Debug)]
pub struct Parsed {
    /// The tree. Partial when parsing failed.
    pub unit: Unit,
    /// Everything the parser could not derive from `unit`.
    pub diagnostics: Vec<Diagnostic>,
}

impl Parsed {
    /// Whether parsing produced any error.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }
}

/// Parses a marked token stream into a translation unit.
///
/// `kind` comes from the `#unit` directive, which [`crate::pp`] has read.
pub fn parse_unit(
    cx: Cx<'_>,
    kind: Option<UnitKind>,
    tokens: &[(Token, Span)],
    eoi: Span,
) -> Parsed {
    let (items, errors) = grammar(cx)
        .parse(stream(tokens, eoi))
        .into_output_errors();
    let items = items.unwrap_or_default();

    // the unit's documentation is its `//!` comments above the first item;
    // one further down is an ordinary comment
    let first = items.first().map_or(u32::MAX, |i| i.span.start);
    let doc = Doc::from_comments(
        cx.docs
            .iter()
            .filter(|d| d.kind == DocKind::Inner && d.anchor.unwrap_or(u32::MAX) <= first),
    );

    Parsed {
        unit: Unit { kind, items, doc },
        diagnostics: errors.iter().map(to_diagnostic).collect(),
    }
}

/// Turns a parser error into a `ES02` diagnostic.
fn to_diagnostic(e: &input::Error<'_>) -> Diagnostic {
    let span = *e.span();
    let text = e.to_string();
    let mut parts = text.split(input::PART);
    let mut d = Diagnostic::new(Code::Es02)
        .with_message(parts.next().unwrap_or_default())
        .at(span);
    if let (Some(note), Some(help)) = (parts.next(), parts.next()) {
        d = d.with_note(note).with_help(help);
    }
    // the error records which productions were in progress; naming them turns
    // "unexpected token" into something a reader can act on
    let contexts: Vec<String> = e
        .contexts()
        .map(|(label, _)| format!("{label}"))
        .collect();
    if !contexts.is_empty() {
        d = d.with_note(format!("while parsing {}", contexts.join(", then ")));
    }
    d
}

/// Helpers shared by the grammar submodules' tests.
///
/// The categories are mutually recursive, so no grammar module runs alone.
/// These run the real pipeline over a whole unit and then reach in.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::ast::{Block, Expr, Stmt};
    use crate::intern::Interner;
    use crate::source::Spliced;
    use crate::span::SourceId;

    /// Runs phases 3 to 5 over `src`.
    pub fn parse(src: &str) -> (Parsed, Interner) {
        let mut interner = Interner::new();
        let out = parse_in(src, &mut interner);
        (out, interner)
    }

    /// Runs phases 3 to 5 over `src`, interning into `interner`, so that
    /// several units parsed this way share names, as a program's units do.
    pub fn parse_in(src: &str, interner: &mut Interner) -> Parsed {
        let spliced = Spliced::from_text(src);
        let lexed = crate::lex::lex(SourceId(0), &spliced, interner);
        assert!(lexed.diagnostics.is_empty(), "lex: {:?}", lexed.diagnostics);
        let marked = crate::mark::mark(&lexed.tokens);
        let eoi = eoi_span(SourceId(0), &marked.tokens);
        let cx = Cx::new(interner).with_docs(&lexed.docs);
        parse_unit(cx, Some(UnitKind::Classical), &marked.tokens, eoi)
    }

    /// Parses a clean unit into a shared interner.
    pub fn unit_in(src: &str, interner: &mut Interner) -> Unit {
        let p = parse_in(src, interner);
        assert!(p.diagnostics.is_empty(), "{src:?} did not parse: {:?}", p.diagnostics);
        p.unit
    }

    /// Parses a unit that is expected to be clean.
    pub fn unit(src: &str) -> (Unit, Interner) {
        let (p, i) = parse(src);
        assert!(
            !p.has_errors(),
            "{src:?} did not parse: {:?}",
            p.diagnostics
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        );
        (p.unit, i)
    }

    /// The messages of everything reported.
    pub fn errors(src: &str) -> Vec<String> {
        let (p, _) = parse(src);
        p.diagnostics.iter().map(|d| d.message.clone()).collect()
    }

    /// The body of the first function in a unit.
    pub fn body(src: &str) -> (Block, Interner) {
        let (u, i) = unit(src);
        let item = u.items.first().expect("one item").node.clone();
        match item.kind {
            crate::ast::ItemKind::Fn(f) => (f.body, i),
            other => panic!("expected a function, got {}", other.describe()),
        }
    }

    /// Wraps `src` as the value of a function body and returns the expression.
    ///
    /// `expr("1 + 2")` parses `fn t() { 1 + 2 }` and hands back the `1 + 2`.
    pub fn expr(src: &str) -> (Expr, Interner) {
        let (b, i) = body(&format!("fn t() {{ {src} }}"));
        let e = b.value.expect("the body should end in an expression");
        (e.node, i)
    }

    /// Renders `expr(src)` as an indented tree, which is what most expression
    /// assertions want to look at.
    pub fn dump(src: &str) -> String {
        let (e, i) = expr(src);
        crate::ast::print::expr(&e, &i)
    }

    /// The statements of a function body.
    pub fn stmts(src: &str) -> (Vec<Stmt>, Interner) {
        let (b, i) = body(&format!("fn t() {{ {src} }}"));
        (b.stmts.into_iter().map(|s| s.node).collect(), i)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::parse;
    use super::*;
    use crate::intern::Interner;

    fn ok(src: &str) -> Unit {
        super::testing::unit(src).0
    }

    fn interner_of(src: &str) -> Interner {
        super::testing::unit(src).1
    }

    #[test]
    fn an_empty_unit_parses() {
        assert!(ok("").is_empty());
    }

    #[test]
    fn a_simple_function_parses() {
        let u = ok("fn main() -> i32 { 0 }");
        assert_eq!(u.len(), 1);
        assert_eq!(u.functions().count(), 1);
    }

    #[test]
    fn imports_parse_with_and_without_an_alias() {
        let u = ok("import geometry;\nimport qpu::gates as g;");
        assert_eq!(u.imports().count(), 2);
    }

    #[test]
    fn a_structure_parses() {
        let u = ok("struct Vec3 { x: f64, y: f64, z: f64 }");
        match &u.items[0].node.kind {
            crate::ast::ItemKind::Struct { fields, .. } => assert_eq!(fields.len(), 3),
            other => panic!("expected a structure, got {other:?}"),
        }
    }

    #[test]
    fn an_enumeration_with_payloads_parses() {
        let u = ok("enum Shape { Circle { r: f64 }, Rect { w: f64, h: f64 } }");
        match &u.items[0].node.kind {
            crate::ast::ItemKind::Enum { variants, .. } => assert_eq!(variants.len(), 2),
            other => panic!("expected an enumeration, got {other:?}"),
        }
    }

    #[test]
    fn a_unit_scope_binding_parses() {
        let u = ok("let VERSION: const u32 = 2;\nstatic let SALT: const u32 = 0xA5;");
        assert_eq!(u.len(), 2);
        assert!(u.items[0].node.is_exported());
        assert!(!u.items[1].node.is_exported(), "`static` is unit linkage");
    }

    #[test]
    fn an_annotated_item_parses() {
        let u = ok("[entry]\nfn f() { }");
        assert_eq!(u.items[0].node.annotations.len(), 1);
    }

    #[test]
    fn the_interner_survives_a_parse() {
        let i = interner_of("struct S { x: u8 }");
        assert!(i.get("S").is_some());
    }

    ////////////////////
    // ERROR RECOVERY //
    ////////////////////

    #[test]
    fn a_broken_item_does_not_hide_the_next_one() {
        // without recovery the first syntax error ends the parse and everything
        // after it is invisible, which makes a single typo look like a file of
        // one error
        let (p, _) = parse("fn a( { }\nstruct S { x: u8 }\nfn b() { }");
        assert!(p.has_errors());
        let names: Vec<Option<crate::intern::Symbol>> = p
            .unit
            .items
            .iter()
            .map(|i| i.node.kind.declared_name())
            .collect();
        assert!(
            names.len() >= 2,
            "the parser should resynchronise and keep going, got {} item(s)",
            names.len()
        );
    }

    #[test]
    fn recovery_resynchronizes_on_an_item_keyword() {
        let (p, i) = parse("fn broken( {\nstruct Good { x: u8 }");
        assert!(p.has_errors());
        let found: Vec<String> = p
            .unit
            .items
            .iter()
            .filter_map(|it| it.node.kind.declared_name())
            .map(|s| i.resolve(s).to_owned())
            .collect();
        assert!(found.iter().any(|n| n == "Good"), "found {found:?}");
    }

    #[test]
    fn the_units_documentation_is_above_its_first_item() {
        let u = ok("//! The unit.\n//!\n//! More.\nfn a() { }\n//! Not the unit's.\nfn b() { }");
        assert_eq!(u.doc.map(|d| d.text).as_deref(), Some("The unit.\n\nMore."));
        let u = ok("//! Nothing follows.");
        assert_eq!(u.doc.map(|d| d.text).as_deref(), Some("Nothing follows."));
        let u = ok("fn a() { }\n//! Trailing.");
        assert_eq!(u.doc, None);
    }

    #[test]
    fn a_clean_unit_recovers_nothing() {
        let (p, _) = parse("fn a() { }\nfn b() { }");
        assert!(!p.has_errors());
        assert_eq!(p.unit.len(), 2);
    }
}
