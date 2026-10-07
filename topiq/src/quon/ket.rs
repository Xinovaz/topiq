//! Reassembling a ket from ordinary tokens.
//!
//! `|0>` is a ket inside a quantum context. Anywhere else it is three separate
//! tokens (a bitwise or, a zero, and a greater-than) and that reading is
//! equally valid Topiq. The lexer cannot tell which it is looking at, so it
//! never produces a ket; this module rebuilds one wherever the quantum grammar
//! says a ket may stand.
//!
//! # What makes that sound
//!
//! Two things, together:
//!
//! - The places a quantum context can open are fixed by the grammar and known
//!   to the parser, so the set of positions where a ket may appear is closed
//!   rather than guessed at.
//! - Expressions never appear inside a quantum context: coefficients use a
//!   small closed arithmetic over exact constants. So no `|` there could be a
//!   bitwise or and no `>` could close a comparison.
//!
//! # The three checks
//!
//! A ket is a `|`, an integer literal, and a `>`. All three of these must hold
//! as well:
//!
//! 1. **Adjacency.** The tokens are written touching, with nothing at all
//!    between them, so `| 0 >` is not a ket. Comparing spans is the test.
//! 2. **Plain decimal digits, no suffix.** `|0b1>` and `|01i8>` also have an
//!    integer in the middle, but neither is a ket: between the bars are basis
//!    digits, not a number in some base or of some type.
//! 3. **Only `0` and `1`, and no digit separator.** Checked against the
//!    literal's raw text rather than a parsed value, because `|01>` and `|1>`
//!    name basis states of registers of *different width* (two qubits against
//!    one) and as numbers they are both just one.

use crate::intern::Interner;
use crate::lex::{IntBase, Token};
use crate::span::Span;

use super::ast::Ket;

/// Why a `|` INT `>` sequence is not a ket.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NotAKet {
    /// The three tokens were not written adjacently.
    NotAdjacent,
    /// The literal was written in a base other than decimal.
    NotDecimal,
    /// The literal carried a type suffix.
    Suffixed,
    /// The digits are not all `0` or `1`, or a separator was used.
    NotBits,
}

impl NotAKet {
    /// A message for a diagnostic.
    pub fn message(self) -> &'static str {
        match self {
            NotAKet::NotAdjacent => {
                "a ket is written as one piece, with no spaces inside it: `|0>`, `|01>`"
            }
            NotAKet::NotDecimal => {
                "a ket's digits are written plainly (`|01>`, not `|0b01>`); the digits \
                 are the basis state itself, not a number in some base"
            }
            NotAKet::Suffixed => {
                "a ket's digits carry no type suffix: write `|01>`, not `|01u8>`"
            }
            NotAKet::NotBits => {
                "a ket names a computational basis state, so it holds only the digits \
                 `0` and `1` (one per qubit)"
            }
        }
    }
}

/// Rebuilds a ket from the three tokens `|`, INT and `>`.
///
/// `bar`, `int` and `gt` are the spans of the three tokens and `raw` is the
/// literal's digits as written.
///
/// # Errors
///
/// Returns why the sequence is not a ket, so the caller can say so rather than
/// merely failing to parse.
pub fn reconstruct(
    bar: Span,
    int: (Span, Token),
    gt: Span,
    interner: &Interner,
) -> Result<Ket, NotAKet> {
    let (int_span, tok) = int;
    if !bar.adjacent_to(int_span) || !int_span.adjacent_to(gt) {
        return Err(NotAKet::NotAdjacent);
    }
    let Token::Int { raw, base, suffix } = tok else {
        return Err(NotAKet::NotBits);
    };
    if base != IntBase::Decimal {
        return Err(NotAKet::NotDecimal);
    }
    if suffix.is_some() {
        return Err(NotAKet::Suffixed);
    }
    let text = interner.resolve(raw);
    if !is_bitstring(text) {
        return Err(NotAKet::NotBits);
    }
    Ok(Ket {
        bits: text.to_owned(),
    })
}

/// Whether a bit string is well formed for a ket.
pub fn is_bitstring(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|b| b == b'0' || b == b'1')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::{IntSuffix, lex};
    use crate::source::Spliced;
    use crate::span::SourceId;

    /// Lexes `src` and tries to read a ket from its first three tokens.
    fn try_ket(src: &str) -> Result<Ket, NotAKet> {
        let interner = Interner::new();
        let spliced = Spliced::from_text(src);
        let out = lex(SourceId(0), &spliced, &interner);
        assert!(out.tokens.len() >= 3, "expected three tokens from {src:?}");
        reconstruct(
            out.tokens[0].1,
            (out.tokens[1].1, out.tokens[1].0),
            out.tokens[2].1,
            &interner,
        )
    }

    #[test]
    fn a_plain_ket_is_reconstructed() {
        assert_eq!(try_ket("|0>").unwrap().bits, "0");
        assert_eq!(try_ket("|1>").unwrap().bits, "1");
        assert_eq!(try_ket("|00>").unwrap().bits, "00");
        assert_eq!(try_ket("|0110>").unwrap().bits, "0110");
    }

    #[test]
    fn leading_zeros_are_preserved() {
        // `|01>` and `|1>` name basis states of registers of different width
        assert_eq!(try_ket("|01>").unwrap().width(), 2);
        assert_eq!(try_ket("|1>").unwrap().width(), 1);
        assert_ne!(try_ket("|01>").unwrap(), try_ket("|1>").unwrap());
    }

    #[test]
    fn a_wide_ket_is_reconstructed() {
        // twenty-four qubits: as decimal, as the lexer sees it, this overflows
        // `u64`, so the lexer must not range-check an integer literal. as the
        // binary index it is, it fits
        let bits = "1".repeat(24);
        let got = try_ket(&format!("|{bits}>")).unwrap();
        assert_eq!(got.width(), 24);
        assert_eq!(got.bits, bits);
        assert_eq!(got.index(), Some((1u64 << 24) - 1));
        assert!(bits.parse::<u64>().is_err(), "decimal would overflow");
    }

    #[test]
    fn a_register_wider_than_sixty_four_qubits_has_no_index() {
        // a fixed register may hold 4096 qubits, so a ket can name a
        // basis state no integer type can hold. the width is still exact
        let bits = "1".repeat(100);
        let got = try_ket(&format!("|{bits}>")).unwrap();
        assert_eq!(got.width(), 100);
        assert!(got.index().is_none());
    }

    #[test]
    fn spacing_defeats_a_ket() {
        // the adjacency requirement, which is also what stops a bitwise or
        // followed by a comparison from being mistaken for one
        assert_eq!(try_ket("| 0>"), Err(NotAKet::NotAdjacent));
        assert_eq!(try_ket("|0 >"), Err(NotAKet::NotAdjacent));
        assert_eq!(try_ket("| 0 >"), Err(NotAKet::NotAdjacent));
    }

    #[test]
    fn a_based_literal_is_not_a_ket() {
        assert_eq!(try_ket("|0b1>"), Err(NotAKet::NotDecimal));
        assert_eq!(try_ket("|0x1>"), Err(NotAKet::NotDecimal));
        assert_eq!(try_ket("|0o1>"), Err(NotAKet::NotDecimal));
    }

    #[test]
    fn a_suffixed_literal_is_not_a_ket() {
        assert_eq!(try_ket("|01i8>"), Err(NotAKet::Suffixed));
        assert_eq!(try_ket("|0u8>"), Err(NotAKet::Suffixed));
    }

    #[test]
    fn digits_other_than_zero_and_one_are_not_a_ket() {
        assert_eq!(try_ket("|02>"), Err(NotAKet::NotBits));
        assert_eq!(try_ket("|9>"), Err(NotAKet::NotBits));
    }

    #[test]
    fn a_digit_separator_is_not_a_ket() {
        // `|0_1>` lexes as one integer whose raw text holds an underscore
        assert_eq!(try_ket("|0_1>"), Err(NotAKet::NotBits));
    }

    #[test]
    fn a_non_integer_middle_is_not_a_ket() {
        // `|x>` is a bitwise-or of `x` with something, in an expression
        assert_eq!(try_ket("|x>"), Err(NotAKet::NotBits));
    }

    #[test]
    fn every_refusal_explains_itself() {
        for e in [
            NotAKet::NotAdjacent,
            NotAKet::NotDecimal,
            NotAKet::Suffixed,
            NotAKet::NotBits,
        ] {
            assert!(!e.message().is_empty());
        }
    }

    #[test]
    fn the_bitstring_predicate_matches_the_grammar() {
        // between the bars, only basis digits
        assert!(is_bitstring("0"));
        assert!(is_bitstring("0110"));
        assert!(!is_bitstring(""));
        assert!(!is_bitstring("02"));
        assert!(!is_bitstring("0_1"));
    }

    #[test]
    fn the_suffix_check_precedes_the_digit_check() {
        // `|01i8>` should say the suffix is the problem, not the digits, since
        // `01` on its own is a perfectly good bit string
        let mut interner = Interner::new();
        let raw = interner.intern("01");
        let at = |a, b| Span::new(SourceId(0), a, b);
        let got = reconstruct(
            at(0, 1),
            (
                at(1, 5),
                Token::Int {
                    raw,
                    base: IntBase::Decimal,
                    suffix: Some(IntSuffix::I8),
                },
            ),
            at(5, 6),
            &interner,
        );
        assert_eq!(got, Err(NotAKet::Suffixed));
    }
}
