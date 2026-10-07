//! What arithmetic means.
//!
//! Every integer operator is checked, always: there is no build in which
//! overflow wraps silently. This module is the one place that says precisely
//! what each operator computes and when it aborts instead. The constant
//! evaluator calls it directly; code generation implements the same rules in
//! machine code, and the program tests compare the two case by case, so a
//! disagreement between compile-time and run-time arithmetic is a test failure
//! rather than a surprise.
//!
//! Values are carried as `i128` together with their [`IntTy`]. Every Topiq
//! integer fits, so each operation is computed exactly and then checked
//! against the type's range, which is the definition of overflow.
//!
//! # The rules
//!
//! | operation | result | aborts with |
//! |---|---|---|
//! | `+` `-` `*` | the exact result | overflow if it does not fit |
//! | `/` | rounded toward zero | division by zero; overflow for `MIN / -1` |
//! | `%` | takes the sign of the dividend | division by zero only (`MIN % -1` is `0`) |
//! | `-x` | the exact negation | overflow if it does not fit, including any non-zero unsigned |
//! | `!x` | the bitwise complement | never |
//! | `x << n` | `x × 2ⁿ` | a bad shift count; overflow if bits are lost |
//! | `x >> n` | `x ÷ 2ⁿ` rounded down | a bad shift count |
//! | `&` `\|` `^` | bitwise, in two's complement | never |
//! | `x as T` | the same value | conversion if it does not fit `T` |
//!
//! A shift count is bad when it is negative or at least the width of the value
//! being shifted: shifting a `u8` by 8 has no sensible answer.
//!
//! `MIN % -1` is not an overflow: the mathematical remainder is `0`, which
//! fits every type, even though machine instructions that compute quotient
//! and remainder together fail on it. Code generation special-cases it so
//! that the program sees the mathematical answer.
//!
//! # Floating arithmetic
//!
//! The [`float`] submodule is the other half. Nothing there aborts: an
//! operation with no finite answer gives an infinity or a NaN, exactly as
//! IEEE 754 requires. The conversions between the two families are here too,
//! and those do abort, since not every floating value has an integer that
//! stands for it.

use crate::diag::Code;
use crate::tir::{FloatTy, IntTy};

/// Why an operation has no result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Abort {
    /// The exact result does not fit the type.
    Overflow,
    /// The divisor of `/` or `%` is zero.
    DivideByZero,
    /// An `as` conversion whose value does not fit the target type.
    Conversion,
    /// An `as` conversion of a floating value that no integer can stand for:
    /// a NaN, an infinity, or a value outside the target type's range.
    FloatConversion,
    /// A shift count that is negative or at least the width of the value.
    ShiftCount,
    /// An index not less than the length of what it indexes.
    Index,
    /// A deliberate abort: a call of `panic`.
    Panic,
    /// A deliberate abort with the identifier `RAnn` numbered here: a call
    /// of `abort`, or an `unwrap` that failed.
    Called(u8),
}

/// The abort identifier `RAnn` numbered `n`, from 1; `RA10` for a number no
/// identifier has.
pub fn runtime_code(n: u8) -> Code {
    const CODES: [Code; 11] = [
        Code::Ra01,
        Code::Ra02,
        Code::Ra03,
        Code::Ra04,
        Code::Ra05,
        Code::Ra06,
        Code::Ra07,
        Code::Ra08,
        Code::Ra09,
        Code::Ra10,
        Code::Ra11,
    ];
    CODES.get(usize::from(n).wrapping_sub(1)).copied().unwrap_or(Code::Ra10)
}

impl Abort {
    /// The run-time abort identifier.
    pub fn code(self) -> Code {
        match self {
            Abort::Overflow => Code::Ra01,
            Abort::DivideByZero => Code::Ra02,
            Abort::Conversion => Code::Ra03,
            Abort::FloatConversion => Code::Ra04,
            Abort::ShiftCount => Code::Ra05,
            Abort::Index => Code::Ra06,
            Abort::Panic => Code::Ra10,
            Abort::Called(n) => runtime_code(n),
        }
    }
}

/// Checks an exact result against the type.
fn fit(t: IntTy, exact: Option<i128>) -> Result<i128, Abort> {
    match exact {
        Some(v) if t.fits(v) => Ok(v),
        _ => Err(Abort::Overflow),
    }
}

/// `a + b`.
pub fn add(t: IntTy, a: i128, b: i128) -> Result<i128, Abort> {
    fit(t, a.checked_add(b))
}

/// `a - b`.
pub fn sub(t: IntTy, a: i128, b: i128) -> Result<i128, Abort> {
    fit(t, a.checked_sub(b))
}

/// `a * b`. The product of two 64-bit values can exceed even `i128`, which is
/// also an overflow.
pub fn mul(t: IntTy, a: i128, b: i128) -> Result<i128, Abort> {
    fit(t, a.checked_mul(b))
}

/// `a / b`, rounded toward zero.
pub fn div(t: IntTy, a: i128, b: i128) -> Result<i128, Abort> {
    if b == 0 {
        return Err(Abort::DivideByZero);
    }
    fit(t, a.checked_div(b))
}

/// `a % b`, with the sign of `a`. `MIN % -1` is `0`.
pub fn rem(t: IntTy, a: i128, b: i128) -> Result<i128, Abort> {
    if b == 0 {
        return Err(Abort::DivideByZero);
    }
    // in `i128` no Topiq value is `i128::MIN`, so this cannot itself overflow
    fit(t, a.checked_rem(b))
}

/// `-a`.
pub fn neg(t: IntTy, a: i128) -> Result<i128, Abort> {
    fit(t, a.checked_neg())
}

/// `!a`: every bit flipped within the type's width.
pub fn not(t: IntTy, a: i128) -> i128 {
    if t.signed {
        // sign-extended, `!a` in `i128` is already the right value
        !a
    } else {
        t.max() - a
    }
}

/// Checks a shift count against the width of the value being shifted.
fn shift_count(t: IntTy, count: i128) -> Result<u32, Abort> {
    if (0..i128::from(t.bits)).contains(&count) {
        Ok(count as u32)
    } else {
        Err(Abort::ShiftCount)
    }
}

/// `a << count`: `a × 2^count`, which must fit.
pub fn shl(t: IntTy, a: i128, count: i128) -> Result<i128, Abort> {
    let n = shift_count(t, count)?;
    fit(t, a.checked_mul(1i128 << n))
}

/// `a >> count`: `a ÷ 2^count` rounded down, so arithmetic on signed types.
pub fn shr(t: IntTy, a: i128, count: i128) -> Result<i128, Abort> {
    let n = shift_count(t, count)?;
    Ok(a >> n)
}

/// `a & b`.
pub fn bit_and(a: i128, b: i128) -> i128 {
    a & b
}

/// `a | b`.
pub fn bit_or(a: i128, b: i128) -> i128 {
    a | b
}

/// `a ^ b`.
pub fn bit_xor(a: i128, b: i128) -> i128 {
    a ^ b
}

/// `a as to`, which must fit.
pub fn cast(to: IntTy, a: i128) -> Result<i128, Abort> {
    if to.fits(a) {
        Ok(a)
    } else {
        Err(Abort::Conversion)
    }
}

/// `x as T`, an integer becoming a character: every character is a Unicode
/// scalar value, and not every number is one.
pub fn int_to_char(x: i128) -> Result<char, Abort> {
    u32::try_from(x)
        .ok()
        .and_then(char::from_u32)
        .ok_or(Abort::Conversion)
}

/// `x as T`, a floating value becoming an integer: the value truncated toward
/// zero, when that is a value of `to`. A NaN, an infinity and anything outside
/// the range abort instead, which is why the total form `saturating_as` exists.
pub fn float_to_int(to: IntTy, x: f64) -> Result<i128, Abort> {
    let t = x.trunc();
    if !t.is_finite() {
        return Err(Abort::FloatConversion);
    }
    // every Topiq integer fits in an `i128`; the round trip catches a value
    // too large for even that, which Rust's conversion would clamp
    let v = t as i128;
    if v as f64 != t || !to.fits(v) {
        return Err(Abort::FloatConversion);
    }
    Ok(v)
}

/// `x as T`, an integer becoming a floating value: the nearest value of `to`,
/// which is exact for anything a `u32` can hold and rounds beyond that.
pub fn int_to_float(to: FloatTy, x: i128) -> f64 {
    to.round(x as f64)
}

/// `x as T` between floating types: the nearest value of `to`.
pub fn float_to_float(to: FloatTy, x: f64) -> f64 {
    to.round(x)
}

/// What a floating operator computes.
///
/// Floating arithmetic never aborts. Overflow gives an infinity, an invalid
/// operation gives a NaN, and division by zero gives an infinity or a NaN,
/// all as [IEEE 754] requires, and all identical on every implementation,
/// which is what lets a constant be folded here and produce what the compiled
/// program would have.
///
/// Each result is rounded to the operands' type, so an `f32` computation is
/// an `f32` computation and not an `f64` one narrowed at the end.
///
/// [IEEE 754]: https://en.wikipedia.org/wiki/IEEE_754
pub mod float {
    use super::FloatTy;

    /// `a + b`.
    pub fn add(t: FloatTy, a: f64, b: f64) -> f64 {
        binary(t, a, b, |x, y| x + y, |x, y| x + y)
    }

    /// `a - b`.
    pub fn sub(t: FloatTy, a: f64, b: f64) -> f64 {
        binary(t, a, b, |x, y| x - y, |x, y| x - y)
    }

    /// `a * b`.
    pub fn mul(t: FloatTy, a: f64, b: f64) -> f64 {
        binary(t, a, b, |x, y| x * y, |x, y| x * y)
    }

    /// `a / b`, which is an infinity or a NaN when `b` is zero.
    pub fn div(t: FloatTy, a: f64, b: f64) -> f64 {
        binary(t, a, b, |x, y| x / y, |x, y| x / y)
    }

    /// `a % b`, the remainder with the sign of the dividend.
    pub fn rem(t: FloatTy, a: f64, b: f64) -> f64 {
        binary(t, a, b, |x, y| x % y, |x, y| x % y)
    }

    /// `-a`, which flips the sign of every value, zeros and NaNs included.
    pub fn neg(t: FloatTy, a: f64) -> f64 {
        match t {
            FloatTy::F32 => f64::from(-(a as f32)),
            FloatTy::F64 => -a,
        }
    }

    /// Computes in the operands' own precision.
    fn binary(t: FloatTy, a: f64, b: f64, at32: fn(f32, f32) -> f32, at64: fn(f64, f64) -> f64) -> f64 {
        match t {
            FloatTy::F32 => f64::from(at32(a as f32, b as f32)),
            FloatTy::F64 => at64(a, b),
        }
    }
}

/// The value an integer literal's digits denote.
///
/// Separators are ignored. Returns `None` for digits that do not belong to
/// the base or a value too large to be any Topiq integer, both of which a
/// well-formed literal can still produce, since the lexer does not
/// range-check.
pub fn literal_value(digits: &str, radix: u32) -> Option<i128> {
    let clean: String = digits.chars().filter(|&c| c != '_').collect();
    if clean.is_empty() {
        return None;
    }
    let v = u128::from_str_radix(&clean, radix).ok()?;
    // nothing above `u64::MAX` is a value of any Topiq integer type
    (v <= u128::from(u64::MAX)).then_some(v as i128)
}

#[cfg(test)]
mod tests {
    use super::*;

    const I8: IntTy = IntTy::I8;
    const U8: IntTy = IntTy::U8;

    fn i8s() -> impl Iterator<Item = i128> {
        (-128..=127).map(i128::from)
    }

    fn u8s() -> impl Iterator<Item = i128> {
        (0..=255).map(i128::from)
    }

    /// Converts a Rust `checked_*` result on a primitive into this module's
    /// shape, for comparing against it as an oracle.
    fn oracle<T: Into<i128>>(r: Option<T>) -> Result<i128, Abort> {
        r.map(Into::into).ok_or(Abort::Overflow)
    }

    #[test]
    fn add_sub_mul_agree_with_rusts_checked_arithmetic_on_every_i8_pair() {
        for a in i8s() {
            for b in i8s() {
                let (x, y) = (a as i8, b as i8);
                assert_eq!(add(I8, a, b), oracle(x.checked_add(y)), "{a} + {b}");
                assert_eq!(sub(I8, a, b), oracle(x.checked_sub(y)), "{a} - {b}");
                assert_eq!(mul(I8, a, b), oracle(x.checked_mul(y)), "{a} * {b}");
            }
        }
    }

    #[test]
    fn add_sub_mul_agree_with_rusts_checked_arithmetic_on_every_u8_pair() {
        for a in u8s() {
            for b in u8s() {
                let (x, y) = (a as u8, b as u8);
                assert_eq!(add(U8, a, b), oracle(x.checked_add(y)), "{a} + {b}");
                assert_eq!(sub(U8, a, b), oracle(x.checked_sub(y)), "{a} - {b}");
                assert_eq!(mul(U8, a, b), oracle(x.checked_mul(y)), "{a} * {b}");
            }
        }
    }

    #[test]
    fn division_on_every_i8_pair() {
        for a in i8s() {
            for b in i8s() {
                let (x, y) = (a as i8, b as i8);
                let want_div = if b == 0 {
                    Err(Abort::DivideByZero)
                } else {
                    oracle(x.checked_div(y))
                };
                assert_eq!(div(I8, a, b), want_div, "{a} / {b}");
                // Rust calls `MIN % -1` an overflow; the answer is `0`
                let want_rem = if b == 0 {
                    Err(Abort::DivideByZero)
                } else {
                    Ok(i128::from(x.wrapping_rem(y)))
                };
                assert_eq!(rem(I8, a, b), want_rem, "{a} % {b}");
            }
        }
    }

    #[test]
    fn the_one_signed_division_that_overflows() {
        assert_eq!(div(I8, -128, -1), Err(Abort::Overflow));
        assert_eq!(rem(I8, -128, -1), Ok(0));
        assert_eq!(div(IntTy::I64, i128::from(i64::MIN), -1), Err(Abort::Overflow));
    }

    #[test]
    fn division_rounds_toward_zero_and_remainder_follows_the_dividend() {
        assert_eq!(div(I8, -7, 2), Ok(-3));
        assert_eq!(rem(I8, -7, 2), Ok(-1));
        assert_eq!(rem(I8, 7, -2), Ok(1));
    }

    #[test]
    fn negation() {
        for a in i8s() {
            assert_eq!(neg(I8, a), oracle((a as i8).checked_neg()), "-{a}");
        }
        assert_eq!(neg(U8, 0), Ok(0));
        assert_eq!(neg(U8, 1), Err(Abort::Overflow), "a non-zero unsigned value has no negation");
    }

    #[test]
    fn complement_matches_the_bit_pattern() {
        for a in i8s() {
            assert_eq!(not(I8, a), i128::from(!(a as i8)), "!{a}");
        }
        for a in u8s() {
            assert_eq!(not(U8, a), i128::from(!(a as u8)), "!{a}");
        }
    }

    #[test]
    fn shifts_on_every_i8_value_and_count() {
        for a in i8s() {
            for n in -2..=9i128 {
                let want_shl = if !(0..8).contains(&n) {
                    Err(Abort::ShiftCount)
                } else if I8.fits(a << n) {
                    Ok(a << n)
                } else {
                    Err(Abort::Overflow)
                };
                assert_eq!(shl(I8, a, n), want_shl, "{a} << {n}");
                let want_shr = if (0..8).contains(&n) {
                    Ok(i128::from((a as i8) >> n))
                } else {
                    Err(Abort::ShiftCount)
                };
                assert_eq!(shr(I8, a, n), want_shr, "{a} >> {n}");
            }
        }
    }

    #[test]
    fn a_left_shift_that_loses_bits_overflows() {
        assert_eq!(shl(U8, 1, 7), Ok(128));
        assert_eq!(shl(U8, 3, 7), Err(Abort::Overflow));
        assert_eq!(shl(I8, 1, 7), Err(Abort::Overflow), "128 is not an i8");
        assert_eq!(shl(I8, -1, 7), Ok(-128));
        assert_eq!(shl(IntTy::U64, 1, 63), Ok(1i128 << 63));
    }

    #[test]
    fn a_shift_count_equal_to_the_width_is_rejected() {
        assert_eq!(shl(IntTy::U32, 1, 32), Err(Abort::ShiftCount));
        assert_eq!(shr(IntTy::U32, 1, 32), Err(Abort::ShiftCount));
        assert_eq!(shr(IntTy::U32, 1, -1), Err(Abort::ShiftCount));
    }

    #[test]
    fn unsigned_right_shift_is_logical_and_signed_is_arithmetic() {
        assert_eq!(shr(U8, 255, 4), Ok(15));
        assert_eq!(shr(I8, -16, 2), Ok(-4));
        assert_eq!(shr(I8, -1, 7), Ok(-1));
    }

    #[test]
    fn bitwise_operators_act_on_the_twos_complement_pattern() {
        for a in i8s() {
            for b in [-128i128, -1, 0, 1, 85, 127] {
                let (x, y) = (a as i8, b as i8);
                assert_eq!(bit_and(a, b), i128::from(x & y));
                assert_eq!(bit_or(a, b), i128::from(x | y));
                assert_eq!(bit_xor(a, b), i128::from(x ^ y));
            }
        }
    }

    #[test]
    fn a_cast_keeps_the_value_or_aborts() {
        assert_eq!(cast(U8, 255), Ok(255));
        assert_eq!(cast(U8, 256), Err(Abort::Conversion));
        assert_eq!(cast(U8, -1), Err(Abort::Conversion));
        assert_eq!(cast(I8, -128), Ok(-128));
        assert_eq!(cast(IntTy::U64, i128::from(u64::MAX)), Ok(i128::from(u64::MAX)));
    }

    #[test]
    fn a_64_bit_product_that_exceeds_i128_is_still_just_an_overflow() {
        let big = i128::from(u64::MAX);
        assert_eq!(mul(IntTy::U64, big, big), Err(Abort::Overflow));
    }

    #[test]
    fn each_abort_names_its_identifier() {
        assert_eq!(Abort::Overflow.code(), Code::Ra01);
        assert_eq!(Abort::DivideByZero.code(), Code::Ra02);
        assert_eq!(Abort::Conversion.code(), Code::Ra03);
        assert_eq!(Abort::FloatConversion.code(), Code::Ra04);
        assert_eq!(Abort::ShiftCount.code(), Code::Ra05);
    }

    #[test]
    fn floating_arithmetic_answers_where_integer_arithmetic_aborts() {
        use FloatTy::{F32, F64};
        assert_eq!(float::div(F64, 1.0, 0.0), f64::INFINITY);
        assert!(float::div(F64, 0.0, 0.0).is_nan());
        assert_eq!(float::mul(F64, f64::MAX, 2.0), f64::INFINITY);
        // an `f32` computation stays an `f32` computation: this sum is exact
        // in `f64` and rounds away in `f32`
        let tiny = f64::from(f32::EPSILON) / 4.0;
        assert_eq!(float::add(F32, 1.0, tiny), 1.0);
        assert_ne!(float::add(F64, 1.0, tiny), 1.0);
        assert_eq!(float::neg(F64, 0.0).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn a_float_converts_to_an_integer_only_when_one_stands_for_it() {
        assert_eq!(float_to_int(IntTy::I32, 2.75), Ok(2));
        assert_eq!(float_to_int(IntTy::I32, -2.75), Ok(-2), "toward zero");
        // truncation happens first, so a fraction below zero is still zero
        assert_eq!(float_to_int(IntTy::U8, -0.5), Ok(0));
        assert_eq!(float_to_int(IntTy::U8, 255.9), Ok(255));
        assert_eq!(float_to_int(IntTy::U8, 256.0), Err(Abort::FloatConversion));
        assert_eq!(float_to_int(IntTy::U8, -1.0), Err(Abort::FloatConversion));
        assert_eq!(float_to_int(IntTy::I64, f64::NAN), Err(Abort::FloatConversion));
        assert_eq!(float_to_int(IntTy::I64, f64::INFINITY), Err(Abort::FloatConversion));
        // `2^63` is one past the largest `i64`, and is exactly representable
        assert_eq!(float_to_int(IntTy::I64, 9_223_372_036_854_775_808.0), Err(Abort::FloatConversion));
        assert_eq!(float_to_int(IntTy::U64, 1.0e40), Err(Abort::FloatConversion));
    }

    #[test]
    fn an_integer_becomes_a_character_only_if_it_is_one() {
        assert_eq!(int_to_char(70), Ok('F'));
        assert_eq!(int_to_char(0x1_F600), Ok('😀'));
        assert_eq!(int_to_char(0xD800), Err(Abort::Conversion), "a surrogate is not a character");
        assert_eq!(int_to_char(0x11_0000), Err(Abort::Conversion));
        assert_eq!(int_to_char(-1), Err(Abort::Conversion));
    }

    #[test]
    fn a_number_converts_to_a_float_by_rounding() {
        assert_eq!(int_to_float(FloatTy::F64, 3), 3.0);
        // 2^24 + 1 has no `f32`, so it rounds to the nearest one that does
        assert_eq!(int_to_float(FloatTy::F32, 16_777_217), 16_777_216.0);
        assert_eq!(float_to_float(FloatTy::F32, 0.1), f64::from(0.1f32));
    }

    #[test]
    fn literals_are_read_in_their_base_with_separators_ignored() {
        assert_eq!(literal_value("1_000", 10), Some(1000));
        assert_eq!(literal_value("FF", 16), Some(255));
        assert_eq!(literal_value("1010", 2), Some(10));
        assert_eq!(literal_value("755", 8), Some(493));
        assert_eq!(literal_value("18446744073709551615", 10), Some(i128::from(u64::MAX)));
        assert_eq!(literal_value("18446744073709551616", 10), None, "past u64::MAX");
        assert_eq!(literal_value("12", 2), None, "not a binary digit");
        assert_eq!(literal_value("_", 10), None);
    }
}
