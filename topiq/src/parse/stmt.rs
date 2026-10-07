//! The statement and block grammar.
//!
//! # A trailing block form is the block's value
//!
//! Two rules meet here. A block is a sequence of statements that may end in an
//! expression, and that expression is the block's value. Separately, a
//! statement beginning with `if`, `match`, `loop`, `while` or `{` ends at its
//! closing brace.
//!
//! Together they mean the same shape reads two ways depending on what follows:
//!
//! ```text
//! { f(); if c { 1 } else { 2 } }    // the `if` is the block's value
//! { if c { f(); } g(); }            // the `if` is a statement: something follows
//! ```
//!
//! This is settled after the fact rather than by lookahead. Every element is
//! parsed, and a trailing block form with no semicolon after it becomes the
//! value. Writing a semicolon after a block form keeps it a statement, which is
//! how to say `{ … };` when its value is not wanted.
//!
//! # `{ … } * x;` is two statements
//!
//! Because the closing brace ends the statement, an operator after it begins a
//! new one: this is a block followed by `* x;`, which follows the reference
//! `x`, not a multiplication. It falls out of parsing a block form as a
//! complete element rather than as the left operand of an expression.
//! Parenthesise the block to get the multiplication.

use chumsky::input::ValueInput;
use chumsky::prelude::*;

use crate::ast::stmt::{Flow, LetStmt, Storage};
use crate::ast::{Block, Expr, Item, Pattern, Stmt, Type};
use crate::lex::{Keyword, Punct, Token};
use crate::span::{Span, Spanned};

use super::input::{Extra, Cx, kw, punct};

/// One element of a block, before the trailing-value rule is applied.
#[derive(Clone)]
enum Element {
    /// An ordinary statement.
    Stmt(Spanned<Stmt>),
    /// A block form and whether a semicolon followed it.
    Block(Spanned<Expr>, bool),
    /// An expression with its semicolon.
    ExprSemi(Spanned<Expr>),
}

/// Builds the block parser.
///
/// `expr` is the full expression handle and `expr_ns` the one excluding bare
/// structure literals, which the conditions of a statement-level `if`,
/// `while` or `for` need.
#[allow(clippy::too_many_arguments)]
pub fn grammar<'t, I, PE, PN, PT, PP, PB, PI>(
    cx: Cx<'t>,
    expr: PE,
    expr_ns: PN,
    ty: PT,
    pat: PP,
    block: PB,
    item: PI,
) -> impl Parser<'t, I, Block, Extra<'t>> + Clone
where
    I: ValueInput<'t, Token = Token, Span = Span>,
    PE: Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone + 't,
    PN: Parser<'t, I, Spanned<Expr>, Extra<'t>> + Clone + 't,
    PT: Parser<'t, I, Spanned<Type>, Extra<'t>> + Clone + 't,
    PP: Parser<'t, I, Spanned<Pattern>, Extra<'t>> + Clone + 't,
    PB: Parser<'t, I, Block, Extra<'t>> + Clone + 't,
    PI: Parser<'t, I, Spanned<Item>, Extra<'t>> + Clone + 't,
{
    let semi = punct(Punct::Semi);

    // `let storage? IDENT (":" type)? ("=" expr)? ";"`
    let storage = choice((
        kw(Keyword::Persist).to(Storage::Persist),
        kw(Keyword::Aux).to(Storage::Aux),
    ))
    .map_with(|s, e| Spanned::new(s, e.span()));

    // `aux` and `persist` precede `let`, as in `aux let a: qubit;` and
    // `persist let counter: u64 = 0;`
    let let_stmt = storage
        .or_not()
        .then_ignore(kw(Keyword::Let))
        .then(pat.clone())
        .then(punct(Punct::Colon).ignore_then(ty.clone()).or_not())
        .then(punct(Punct::Eq).ignore_then(expr.clone()).or_not())
        .then_ignore(semi.clone())
        .map_with(|(((storage, binding), ty), init), e| {
            Spanned::new(
                Stmt::Let(Box::new(LetStmt {
                    storage,
                    binding,
                    ty,
                    init,
                })),
                e.span(),
            )
        });

    // `return e?;`, `break e?;`, `continue;`
    let flow = choice((
        kw(Keyword::Return)
            .ignore_then(expr.clone().or_not())
            .map(Flow::Return),
        kw(Keyword::Break)
            .ignore_then(expr.clone().or_not())
            .map(Flow::Break),
        kw(Keyword::Continue).to(Flow::Continue),
    ))
    .then_ignore(semi.clone())
    .map_with(|f, e| Spanned::new(Stmt::Flow(f), e.span()));

    // `forget e;`: the only way a quantum value is discarded
    let forget = kw(Keyword::Forget)
        .ignore_then(expr.clone())
        .then_ignore(semi.clone())
        .map_with(|x, e| Spanned::new(Stmt::Forget(x), e.span()));

    let empty = semi
        .clone()
        .map_with(|_, e| Spanned::new(Stmt::Empty, e.span()));

    // in a block, `@static_assert(…);` is the expression statement it reads
    // as, not an item
    let item_stmt = item.clone().map(|i: Spanned<Item>| match i.node.kind {
        crate::ast::ItemKind::Assert { call } if i.node.annotations.is_empty() => Spanned::new(Stmt::Expr(call), i.span),
        _ => Spanned::new(Stmt::Item(Box::new(i.node)), i.span),
    });

    // a block form and nothing more. its closing brace ends it, so what
    // follows (an operator, a parenthesis) begins a new statement rather
    // than continuing this one
    let block_form = super::expr::block_form(expr.clone(), expr_ns, pat, block)
        .then(semi.clone().or_not())
        .map(|(x, s)| Element::Block(x, s.is_some()));

    // `@foreach_field(T, M)` and its kin expand to statements, so like a block
    // form they stand on their own, without a semicolon
    let foreach = expr
        .clone()
        .filter(move |x: &Spanned<Expr>| {
            matches!(&x.node, Expr::Macro { name, .. } if cx.interner.resolve(name.node).starts_with("foreach_"))
        })
        .then(semi.clone().or_not())
        .map(|(x, s)| Element::Block(x, s.is_some()));

    let element = choice((
        let_stmt.map(Element::Stmt),
        flow.map(Element::Stmt),
        forget.map(Element::Stmt),
        item_stmt.map(Element::Stmt),
        empty.map(Element::Stmt),
        block_form,
        foreach,
        expr.clone()
            .then_ignore(semi.clone())
            .map(Element::ExprSemi),
    ));

    // a final expression with no semicolon is the block's value
    element
        .repeated()
        .collect::<Vec<_>>()
        .then(expr.clone().or_not())
        .delimited_by(punct(Punct::LBrace), punct(Punct::RBrace))
        .map(|(elements, tail)| assemble(elements, tail))
        .labelled("block")
}

/// Decides which parsed element, if any, is the block's trailing value.
fn assemble(elements: Vec<Element>, tail: Option<Spanned<Expr>>) -> Block {
    let mut stmts: Vec<Spanned<Stmt>> = Vec::with_capacity(elements.len());
    let mut value = tail;

    let last = elements.len().saturating_sub(1);
    for (i, element) in elements.into_iter().enumerate() {
        match element {
            Element::Stmt(s) => stmts.push(s),
            Element::ExprSemi(x) => {
                let span = x.span;
                stmts.push(Spanned::new(Stmt::Expr(x), span));
            }
            // a block form is the block's value when it is last and no
            // semicolon followed it; otherwise it is a statement
            Element::Block(x, semi) => {
                if i == last && !semi && value.is_none() {
                    value = Some(x);
                } else {
                    let span = x.span;
                    stmts.push(Spanned::new(Stmt::BlockExpr(x), span));
                }
            }
        }
    }
    Block {
        annotations: Vec::new(),
        stmts,
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::testing::{body, errors};

    #[test]
    fn a_parenthesis_after_a_block_statement_starts_a_new_expression() {
        // not a call of the loop: the loop ends at its brace, and the
        // parenthesised expression is the block's value
        let (b, _) = body("fn t() -> i32 { for i in 0..3 { } (1 + 2) as i32 }");
        assert_eq!(b.stmts.len(), 1);
        assert!(matches!(b.stmts[0].node, Stmt::BlockExpr(_)));
        assert!(matches!(
            b.value.as_ref().map(|v| &v.node),
            Some(Expr::Cast { .. })
        ));
    }

    #[test]
    fn an_operator_after_a_block_statement_is_not_a_continuation() {
        // `- 1` is a new statement, a negation, not a subtraction from the
        // loop's value
        let (b, _) = body("fn t() { loop { break; } - 1; }");
        assert_eq!(b.stmts.len(), 2);
        assert!(matches!(b.stmts[1].node, Stmt::Expr(_)));
        // nor is `* p`: it is a statement of its own, a dereference
        let (b, _) = body("fn t() { { } * p; }");
        assert_eq!(b.stmts.len(), 2);
        let Stmt::Expr(e) = &b.stmts[1].node else {
            panic!("{:?}", b.stmts[1].node)
        };
        assert!(matches!(e.node, Expr::Unary { op: crate::ast::UnOp::Deref, .. }));
        // `/ 2` cannot begin a statement at all
        assert!(!errors("fn t() { { } / 2; }").is_empty());
    }

    #[test]
    fn every_block_form_ends_at_its_brace() {
        for form in [
            "if true { }",
            "match x { _ => 0 }",
            "while false { }",
            "loop { break; }",
            "for i in 0..1 { }",
            "{ }",
        ] {
            let (b, _) = body(&format!("fn t() {{ {form} (7); }}"));
            assert_eq!(b.stmts.len(), 2, "{form} should end before `(7)`");
        }
    }

    #[test]
    fn inside_an_expression_a_block_form_is_an_ordinary_operand() {
        let (b, _) = body("fn t() { let x = 1 + if c { 2 } else { 3 }; }");
        let Stmt::Let(l) = &b.stmts[0].node else {
            panic!("a let");
        };
        assert!(matches!(
            l.init.as_ref().map(|e| &e.node),
            Some(Expr::Binary { .. })
        ));
    }

    #[test]
    fn a_trailing_block_form_becomes_the_value() {
        // `{ if c { 1 } else { 2 } }`: the `if` is the block's value
        let at = Span::new(crate::span::SourceId(0), 0, 1);
        let x = Spanned::new(Expr::Bool(true), at);
        let b = assemble(vec![Element::Block(x.clone(), false)], None);
        assert!(b.value.is_some());
        assert!(b.stmts.is_empty());
    }

    #[test]
    fn a_semicolon_keeps_a_block_form_a_statement() {
        let at = Span::new(crate::span::SourceId(0), 0, 1);
        let x = Spanned::new(Expr::Bool(true), at);
        let b = assemble(vec![Element::Block(x, true)], None);
        assert!(b.value.is_none());
        assert_eq!(b.stmts.len(), 1);
        assert!(matches!(b.stmts[0].node, Stmt::BlockExpr(_)));
    }

    #[test]
    fn a_block_form_that_is_not_last_stays_a_statement() {
        // `{ if c { f(); } g(); }`
        let at = Span::new(crate::span::SourceId(0), 0, 1);
        let x = Spanned::new(Expr::Bool(true), at);
        let b = assemble(
            vec![
                Element::Block(x.clone(), false),
                Element::ExprSemi(x.clone()),
            ],
            None,
        );
        assert!(b.value.is_none());
        assert_eq!(b.stmts.len(), 2);
        assert!(matches!(b.stmts[0].node, Stmt::BlockExpr(_)));
        assert!(matches!(b.stmts[1].node, Stmt::Expr(_)));
    }

    #[test]
    fn an_explicit_tail_expression_wins_over_a_block_form() {
        let at = Span::new(crate::span::SourceId(0), 0, 1);
        let x = Spanned::new(Expr::Bool(true), at);
        let b = assemble(
            vec![Element::Block(x.clone(), false)],
            Some(Spanned::new(Expr::Bool(false), at)),
        );
        assert_eq!(b.value.map(|v| v.node), Some(Expr::Bool(false)));
        assert_eq!(b.stmts.len(), 1, "the block form fell back to a statement");
    }

    #[test]
    fn an_empty_block_assembles_to_nothing() {
        let b = assemble(Vec::new(), None);
        assert!(b.is_empty());
    }
}
