//! The tiny expression language a `#if` condition is written in.
//!
//! A `#if` condition is an expression over integer literals, `defined(NAME)`,
//! the arithmetic, comparison and logical operators, and grouping, nothing
//! else. So small a language gets a direct recursive-descent evaluator rather
//! than the parser generator [`crate::parse`] uses.
//!
//! Two rules shape it:
//!
//! - identifiers that are not defined macros evaluate to **0**, so a typo in a
//!   `#if` is silently false rather than an error, exactly as in C;
//! - `defined(NAME)` is evaluated *before* macro expansion of its operand, so
//!   the name is not expanded away before the question is asked.
//!
//! Arithmetic is over `i64`, saturating rather than wrapping, so an overflowing
//! `#if` cannot silently change a branch.
//!
//! # The `>` tokens
//!
//! Because the lexer never glues `>` (see [`crate::lex::token`]), this evaluator
//! joins `>` `>` into a shift and `>` `=` into a comparison itself, requiring
//! span adjacency so that `a > > b` is not read as a shift. The main parser
//! applies the same rule to the same tokens.

use crate::diag::{Code, Diagnostic};
use crate::intern::Interner;
use crate::lex::{Punct, Token};
use crate::span::Span;

use super::macros::{MacroTable, PpToken};

/// Evaluates a `#if` or `#elif` condition.
///
/// Returns whether the branch is taken. On a malformed expression a diagnostic
/// is pushed and the result is `false`, so translation continues down the
/// `#else` path rather than stopping.
pub fn eval(
    tokens: &[PpToken],
    table: &MacroTable,
    interner: &mut Interner,
    diagnostics: &mut Vec<Diagnostic>,
    at: Span,
) -> bool {
    let mut p = Eval {
        toks: tokens,
        pos: 0,
        table,
        interner,
        diagnostics,
        at,
        failed: false,
        quiet: 0,
    };
    let value = p.expr(0);
    if !p.failed && p.pos < p.toks.len() {
        let span = p.toks[p.pos].span;
        p.error_at("unexpected token after the condition", span);
    }
    if p.failed { false } else { value != 0 }
}

/// Binding powers, tightest last, following the language's own table.
const LEVELS: &[&[BinOp]] = &[
    &[BinOp::OrOr],
    &[BinOp::AndAnd],
    &[BinOp::Eq, BinOp::Ne, BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge],
    &[BinOp::Or],
    &[BinOp::Xor],
    &[BinOp::And],
    &[BinOp::Shl, BinOp::Shr],
    &[BinOp::Add, BinOp::Sub],
    &[BinOp::Mul, BinOp::Div, BinOp::Rem],
];

/// The level at which comparisons sit.
const CMP_LEVEL: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BinOp {
    OrOr,
    AndAnd,
    Or,
    Xor,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Shl,
    Shr,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

struct Eval<'a> {
    toks: &'a [PpToken],
    pos: usize,
    table: &'a MacroTable,
    interner: &'a mut Interner,
    diagnostics: &'a mut Vec<Diagnostic>,
    at: Span,
    failed: bool,
    /// Non-zero while parsing an operand that short-circuiting has skipped.
    ///
    /// The operand is still parsed, to find where the expression continues,
    /// but not evaluated, so `0 && (1 / 0)` is false and not an error.
    quiet: u32,
}

impl<'a> Eval<'a> {
    fn peek(&self) -> Option<Token> {
        self.toks.get(self.pos).map(|t| t.tok)
    }

    fn span(&self) -> Span {
        self.toks.get(self.pos).map_or(self.at, |t| t.span)
    }

    fn error_at(&mut self, msg: &str, span: Span) {
        // a skipped operand is parsed but not evaluated, so nothing inside it
        // is a diagnosable condition
        if self.failed || self.quiet > 0 {
            return;
        }
        self.failed = true;
        self.diagnostics.push(
            Diagnostic::new(Code::Es02)
                .with_message(format!("in a #if condition: {msg}"))
                .at(span),
        );
    }

    fn error(&mut self, msg: &str) {
        let span = self.span();
        self.error_at(msg, span);
    }

    /// Recognises a binary operator at the cursor, joining `>` tokens as
    /// the lexer's rule requires. Returns the operator and its token width.
    fn peek_binop(&self) -> Option<(BinOp, usize)> {
        let a = self.toks.get(self.pos)?;
        let b = self.toks.get(self.pos + 1);
        let joined = |b: Option<&PpToken>| -> bool {
            b.is_some_and(|b| a.span.adjacent_to(b.span))
        };
        let op = match a.tok {
            Token::Punct(Punct::Gt) => {
                // `>` `>` is a shift and `>` `=` a comparison, but only when
                // written adjacently
                return match b.map(|b| b.tok) {
                    Some(Token::Punct(Punct::Gt)) if joined(b) => Some((BinOp::Shr, 2)),
                    Some(Token::Punct(Punct::Eq)) if joined(b) => Some((BinOp::Ge, 2)),
                    _ => Some((BinOp::Gt, 1)),
                };
            }
            Token::Punct(Punct::OrOr) => BinOp::OrOr,
            Token::Punct(Punct::AndAnd) => BinOp::AndAnd,
            Token::Punct(Punct::Or) => BinOp::Or,
            Token::Punct(Punct::Caret) => BinOp::Xor,
            Token::Punct(Punct::And) => BinOp::And,
            Token::Punct(Punct::EqEq) => BinOp::Eq,
            Token::Punct(Punct::Ne) => BinOp::Ne,
            Token::Punct(Punct::Lt) => BinOp::Lt,
            Token::Punct(Punct::Le) => BinOp::Le,
            Token::Punct(Punct::Shl) => BinOp::Shl,
            Token::Punct(Punct::Plus) => BinOp::Add,
            Token::Punct(Punct::Minus) => BinOp::Sub,
            Token::Punct(Punct::Star) => BinOp::Mul,
            Token::Punct(Punct::Slash) => BinOp::Div,
            Token::Punct(Punct::Percent) => BinOp::Rem,
            _ => return None,
        };
        Some((op, 1))
    }

    fn expr(&mut self, level: usize) -> i64 {
        if level >= LEVELS.len() {
            return self.unary();
        }
        let mut lhs = self.expr(level + 1);
        let mut seen_cmp = false;
        loop {
            if self.failed {
                return 0;
            }
            let Some((op, width)) = self.peek_binop() else {
                return lhs;
            };
            if !LEVELS[level].contains(&op) {
                return lhs;
            }
            // comparisons are non-associative, so `a < b < c` is
            // ill-formed rather than left-folded
            if level == CMP_LEVEL {
                if seen_cmp {
                    self.error("comparison operators are non-associative");
                    return 0;
                }
                seen_cmp = true;
            }
            let op_span = self.span();
            self.pos += width;

            // `&&` and `||` short-circuit: the skipped operand is parsed, so
            // the cursor lands past it, but `quiet` stops its evaluation
            let short_circuits =
                (op == BinOp::AndAnd && lhs == 0) || (op == BinOp::OrOr && lhs != 0);
            if short_circuits {
                self.quiet += 1;
                let _ = self.expr(level + 1);
                self.quiet -= 1;
                lhs = i64::from(op == BinOp::OrOr);
                continue;
            }

            let rhs = self.expr(level + 1);
            lhs = self.apply(op, lhs, rhs, op_span);
        }
    }

    fn apply(&mut self, op: BinOp, a: i64, b: i64, span: Span) -> i64 {
        let logical = |v: bool| i64::from(v);
        match op {
            BinOp::OrOr => logical(a != 0 || b != 0),
            BinOp::AndAnd => logical(a != 0 && b != 0),
            BinOp::Or => a | b,
            BinOp::Xor => a ^ b,
            BinOp::And => a & b,
            BinOp::Eq => logical(a == b),
            BinOp::Ne => logical(a != b),
            BinOp::Lt => logical(a < b),
            BinOp::Le => logical(a <= b),
            BinOp::Gt => logical(a > b),
            BinOp::Ge => logical(a >= b),
            BinOp::Add => a.saturating_add(b),
            BinOp::Sub => a.saturating_sub(b),
            BinOp::Mul => a.saturating_mul(b),
            BinOp::Shl | BinOp::Shr if !(0..64).contains(&b) => {
                self.error_at("shift count is out of range", span);
                0
            }
            BinOp::Shl => a.wrapping_shl(b as u32),
            BinOp::Shr => a.wrapping_shr(b as u32),
            BinOp::Div | BinOp::Rem if b == 0 => {
                self.error_at("division by zero", span);
                0
            }
            BinOp::Div => a.wrapping_div(b),
            BinOp::Rem => a.wrapping_rem(b),
        }
    }

    fn unary(&mut self) -> i64 {
        match self.peek() {
            Some(Token::Punct(Punct::Minus)) => {
                self.pos += 1;
                self.unary().saturating_neg()
            }
            Some(Token::Punct(Punct::Plus)) => {
                self.pos += 1;
                self.unary()
            }
            Some(Token::Punct(Punct::Bang)) => {
                self.pos += 1;
                i64::from(self.unary() == 0)
            }
            _ => self.primary(),
        }
    }

    fn primary(&mut self) -> i64 {
        match self.peek() {
            None => {
                self.error("the condition ends unexpectedly");
                0
            }
            Some(Token::Punct(Punct::LParen)) => {
                self.pos += 1;
                let v = self.expr(0);
                if self.peek() == Some(Token::Punct(Punct::RParen)) {
                    self.pos += 1;
                } else {
                    self.error("expected `)`");
                }
                v
            }
            Some(Token::Int { raw, base, .. }) => {
                self.pos += 1;
                let text: String = self.interner.resolve(raw).replace('_', "");
                match i64::from_str_radix(&text, base.radix()) {
                    Ok(v) => v,
                    Err(_) => {
                        self.error("integer literal does not fit in i64");
                        0
                    }
                }
            }
            Some(Token::Bool(b)) => {
                self.pos += 1;
                i64::from(b)
            }
            Some(Token::Ident(sym)) => {
                self.pos += 1;
                if self.interner.resolve(sym) == "defined" {
                    return self.defined();
                }
                // an identifier that is not a defined macro is zero
                0
            }
            Some(other) => {
                let msg = format!("unexpected {} in a condition", other.describe());
                self.error(&msg);
                0
            }
        }
    }

    /// `defined(NAME)`, and the bare `defined NAME` form C also allows.
    fn defined(&mut self) -> i64 {
        let parenthesized = self.peek() == Some(Token::Punct(Punct::LParen));
        if parenthesized {
            self.pos += 1;
        }
        let Some(Token::Ident(name)) = self.peek() else {
            self.error("`defined` needs a macro name");
            return 0;
        };
        self.pos += 1;
        if parenthesized {
            if self.peek() == Some(Token::Punct(Punct::RParen)) {
                self.pos += 1;
            } else {
                self.error("expected `)` after the name");
                return 0;
            }
        }
        i64::from(self.table.is_defined(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pp::macros::{Linkage, MacroDef};
    use crate::source::Spliced;
    use crate::span::SourceId;

    struct Fixture {
        interner: Interner,
        table: MacroTable,
    }

    impl Fixture {
        fn new() -> Fixture {
            Fixture {
                interner: Interner::new(),
                table: MacroTable::new(),
            }
        }

        fn define(&mut self, name: &str) {
            let sym = self.interner.intern(name);
            self.table.define(MacroDef {
                name: sym,
                params: None,
                variadic: false,
                body: Vec::new(),
                linkage: Linkage::Program,
                span: Span::new(SourceId(0), 0, 1),
                predefined: false,
            });
        }

        fn tokens(&mut self, src: &str) -> Vec<PpToken> {
            let spliced = Spliced::from_text(src);
            let out = crate::lex::lex(SourceId(0), &spliced, &self.interner);
            out.tokens
                .into_iter()
                .map(|(t, s)| PpToken::new(t, s))
                .collect()
        }

        fn eval(&mut self, src: &str) -> bool {
            let toks = self.tokens(src);
            let mut diags = Vec::new();
            let v = eval(
                &toks,
                &self.table,
                &mut self.interner,
                &mut diags,
                Span::new(SourceId(0), 0, 1),
            );
            assert!(
                diags.is_empty(),
                "unexpected diagnostics for {src:?}: {:?}",
                diags.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
            );
            v
        }

        fn eval_err(&mut self, src: &str) -> Vec<Diagnostic> {
            let toks = self.tokens(src);
            let mut diags = Vec::new();
            eval(
                &toks,
                &self.table,
                &mut self.interner,
                &mut diags,
                Span::new(SourceId(0), 0, 1),
            );
            diags
        }
    }

    #[test]
    fn integer_literals_and_truthiness() {
        let mut f = Fixture::new();
        assert!(f.eval("1"));
        assert!(!f.eval("0"));
        assert!(f.eval("42"));
        assert!(f.eval("0xFF"));
        assert!(f.eval("0b1"));
        assert!(!f.eval("0o0"));
        assert!(f.eval("1_000"));
    }

    #[test]
    fn arithmetic() {
        let mut f = Fixture::new();
        assert!(f.eval("1 + 1 == 2"));
        assert!(f.eval("7 - 3 == 4"));
        assert!(f.eval("6 * 7 == 42"));
        assert!(f.eval("7 / 2 == 3"));
        assert!(f.eval("7 % 2 == 1"));
        assert!(f.eval("-3 + 3 == 0"));
    }

    #[test]
    fn precedence_follows_c() {
        let mut f = Fixture::new();
        assert!(f.eval("1 + 2 * 3 == 7"));
        assert!(f.eval("(1 + 2) * 3 == 9"));
        assert!(f.eval("1 << 3 == 8"));

        // `&` binds tighter than `^`: 3 & 6 is 2, and 1 ^ 2 is 3. folding the
        // other way would give (1 ^ 3) & 6, which is 2
        assert!(f.eval("(1 ^ 3 & 6) == 3"));
        // `^` binds tighter than `|`: 2 ^ 3 is 1, and 1 | 1 is 1. the other
        // way would give (1 | 2) ^ 3, which is 0
        assert!(f.eval("(1 | 2 ^ 3) == 1"));
    }

    #[test]
    fn comparisons_bind_looser_than_the_bitwise_operators() {
        // unlike C, the bitwise operators bind tighter than comparison, so
        // `4 | 2 == 6` is `(4 | 2) == 6`, which is 1; C would give 4
        let mut f = Fixture::new();
        assert!(f.eval("(4 | 2 == 6) == 1"));
        assert!(!f.eval("(4 | 2 == 6) == 4"));
    }

    #[test]
    fn shift_right_is_joined_from_two_gt_tokens() {
        // the lexer never glues `>`; this evaluator rejoins it, as the main
        // parser does
        let mut f = Fixture::new();
        assert!(f.eval("16 >> 2 == 4"));
        assert!(f.eval("1 >= 1"));
        assert!(f.eval("2 > 1"));
        assert!(!f.eval("1 > 2"));
    }

    #[test]
    fn a_spaced_pair_of_gt_is_not_a_shift() {
        // `16 > > 2` has no reading; adjacency is what makes `>>` a shift
        let mut f = Fixture::new();
        assert!(!f.eval_err("16 > > 2").is_empty());
    }

    #[test]
    fn logical_operators_short_circuit() {
        let mut f = Fixture::new();
        assert!(f.eval("1 || 0"));
        assert!(!f.eval("0 && 1"));
        assert!(f.eval("!0"));
        assert!(!f.eval("!1"));
        // the right operand of a short-circuited `&&` is parsed but not
        // evaluated, so a division by zero inside it does not fire. `eval`
        // asserts no diagnostics, which is the real check here
        assert!(!f.eval("0 && (1 / 0)"));
        assert!(f.eval("1 || (1 / 0)"));
        assert!(!f.eval("0 && 1 / 0"));
        assert!(f.eval("1 || 1 % 0"));
        // nested short-circuits stay quiet too
        assert!(!f.eval("0 && (1 || (1 / 0))"));
        // but an operand that *is* evaluated still reports
        assert!(!f.eval_err("1 && (1 / 0)").is_empty());
    }

    #[test]
    fn comparisons_are_non_associative() {
        // `a < b < c` is ill-formed, not left-folded
        let mut f = Fixture::new();
        let diags = f.eval_err("1 < 2 < 3");
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("non-associative"), "{}", diags[0].message);
    }

    #[test]
    fn defined_reports_the_macro_table() {
        let mut f = Fixture::new();
        f.define("FEATURE");
        assert!(f.eval("defined(FEATURE)"));
        assert!(!f.eval("defined(MISSING)"));
        assert!(f.eval("defined FEATURE"), "the unparenthesised form works too");
        assert!(f.eval("!defined(MISSING)"));
        assert!(f.eval("defined(FEATURE) && !defined(MISSING)"));
    }

    #[test]
    fn an_undefined_identifier_is_zero() {
        // an undefined name is zero, not an error
        let mut f = Fixture::new();
        assert!(!f.eval("NOT_A_MACRO"));
        assert!(f.eval("NOT_A_MACRO == 0"));
    }

    #[test]
    fn division_by_zero_is_reported_rather_than_panicking() {
        let mut f = Fixture::new();
        let diags = f.eval_err("1 / 0");
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("division by zero"));
        assert!(!f.eval_err("1 % 0").is_empty());
    }

    #[test]
    fn an_out_of_range_shift_is_reported() {
        let mut f = Fixture::new();
        assert!(!f.eval_err("1 << 64").is_empty());
        assert!(!f.eval_err("1 << -1").is_empty());
    }

    #[test]
    fn overflow_saturates_rather_than_wrapping_a_branch() {
        let mut f = Fixture::new();
        // a wrapping multiply could make this negative and flip the branch
        assert!(f.eval("9223372036854775807 * 2 > 0"));
    }

    #[test]
    fn malformed_conditions_are_reported() {
        let mut f = Fixture::new();
        assert!(!f.eval_err("1 +").is_empty());
        assert!(!f.eval_err("(1").is_empty());
        assert!(!f.eval_err("").is_empty());
        assert!(!f.eval_err("1 2").is_empty());
        assert!(!f.eval_err("defined()").is_empty());
    }

    #[test]
    fn a_failed_condition_is_false_so_the_else_branch_is_taken() {
        let mut f = Fixture::new();
        let toks = f.tokens("1 +");
        let mut diags = Vec::new();
        let taken = eval(
            &toks,
            &f.table,
            &mut f.interner,
            &mut diags,
            Span::new(SourceId(0), 0, 1),
        );
        assert!(!taken);
        assert!(!diags.is_empty());
    }

    #[test]
    fn only_one_diagnostic_is_reported_per_condition() {
        // a cascade of follow-on errors from one malformed condition would be
        // noise; the first is the useful one
        let mut f = Fixture::new();
        assert_eq!(f.eval_err("((((").len(), 1);
    }

    #[test]
    fn booleans_are_usable() {
        let mut f = Fixture::new();
        assert!(f.eval("true"));
        assert!(!f.eval("false"));
    }
}
