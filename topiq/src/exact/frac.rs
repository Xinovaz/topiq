//! `frac`: arbitrary-precision rational arithmetic.
//!
//! Arithmetic on exact scalars is exact at arbitrary precision, both while a
//! program is translated and while it runs, so this is a bignum rational rather
//! than a fixed-width approximation. Conversion to `f32`, `f64` or `cplx<f64>`
//! is always written out with `as`, and is lossy.
//!
//! This type exists rather than a float for one reason: no judgement fact may
//! be decided by comparing floating-point values, and no floating
//! representation may stand in for an exact scalar where a judgement depends on
//! it. Two floats being close together proves nothing.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// An arbitrary-precision rational.
///
/// Always kept in lowest terms with a positive denominator, which is what makes
/// equality a structural comparison.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Frac(BigRational);

impl Frac {
    /// Zero.
    pub fn zero() -> Frac {
        Frac(BigRational::zero())
    }

    /// One.
    pub fn one() -> Frac {
        Frac(BigRational::one())
    }

    /// From an integer.
    pub fn from_int(n: i64) -> Frac {
        Frac(BigRational::from_integer(BigInt::from(n)))
    }

    /// From a big integer.
    pub fn from_bigint(n: BigInt) -> Frac {
        Frac(BigRational::from_integer(n))
    }

    /// From a numerator and denominator.
    ///
    /// # Panics
    ///
    /// Panics if `denominator` is zero. Callers reachable from source text
    /// should check first and report `RA02`/`EC04` instead.
    pub fn new(numerator: i64, denominator: i64) -> Frac {
        assert!(denominator != 0, "frac denominator is zero");
        Frac(BigRational::new(
            BigInt::from(numerator),
            BigInt::from(denominator),
        ))
    }

    /// From a numerator and denominator, or `None` if the denominator is zero.
    pub fn checked_new(numerator: BigInt, denominator: BigInt) -> Option<Frac> {
        if denominator.is_zero() {
            None
        } else {
            Some(Frac(BigRational::new(numerator, denominator)))
        }
    }

    /// Parses a decimal literal as written, including digit separators and an
    /// optional exponent, exactly.
    ///
    /// `2.5e-3` is exactly 1/400, not the nearest `f64`.
    pub fn parse_decimal(text: &str) -> Option<Frac> {
        let text: String = text.chars().filter(|c| *c != '_').collect();
        let (mantissa, exponent) = match text.find(['e', 'E']) {
            Some(i) => (&text[..i], text[i + 1..].parse::<i32>().ok()?),
            None => (&text[..], 0),
        };
        let (int_part, frac_part) = match mantissa.find('.') {
            Some(i) => (&mantissa[..i], &mantissa[i + 1..]),
            None => (mantissa, ""),
        };
        let digits = format!("{int_part}{frac_part}");
        let numerator: BigInt = digits.parse().ok()?;
        let scale = exponent - frac_part.len() as i32;
        let ten = BigInt::from(10);
        Some(if scale >= 0 {
            Frac(BigRational::from_integer(
                numerator * ten.pow(scale as u32),
            ))
        } else {
            Frac(BigRational::new(numerator, ten.pow((-scale) as u32)))
        })
    }

    /// The numerator.
    pub fn numer(&self) -> &BigInt {
        self.0.numer()
    }

    /// The denominator.
    pub fn denom(&self) -> &BigInt {
        self.0.denom()
    }

    /// Whether this is zero.
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Whether this is one.
    pub fn is_one(&self) -> bool {
        self.0.is_one()
    }

    /// Whether this is an integer.
    pub fn is_integer(&self) -> bool {
        self.0.is_integer()
    }

    /// Whether this is strictly positive.
    pub fn is_positive(&self) -> bool {
        self.0.is_positive()
    }

    /// Whether this is strictly negative.
    pub fn is_negative(&self) -> bool {
        self.0.is_negative()
    }

    /// The multiplicative inverse, or `None` for zero.
    pub fn inv(&self) -> Option<Frac> {
        if self.is_zero() {
            None
        } else {
            Some(Frac(self.0.recip()))
        }
    }

    /// The absolute value.
    pub fn abs(&self) -> Frac {
        Frac(self.0.abs())
    }

    /// Raises to a non-negative integer power.
    pub fn pow(&self, exp: u32) -> Frac {
        Frac(num_traits::Pow::pow(self.0.clone(), exp as i32))
    }

    /// An `f64` approximation.
    ///
    /// Lossy: for printing and for diagnostics only, never for a judgement.
    pub fn to_f64_lossy(&self) -> f64 {
        self.0.to_f64().unwrap_or(f64::NAN)
    }

    /// The underlying rational.
    pub fn as_ratio(&self) -> &BigRational {
        &self.0
    }
}

impl From<BigRational> for Frac {
    fn from(r: BigRational) -> Frac {
        Frac(r)
    }
}

impl From<i64> for Frac {
    fn from(n: i64) -> Frac {
        Frac::from_int(n)
    }
}

impl Add for Frac {
    type Output = Frac;
    fn add(self, rhs: Frac) -> Frac {
        Frac(self.0 + rhs.0)
    }
}

impl Sub for Frac {
    type Output = Frac;
    fn sub(self, rhs: Frac) -> Frac {
        Frac(self.0 - rhs.0)
    }
}

impl Mul for Frac {
    type Output = Frac;
    fn mul(self, rhs: Frac) -> Frac {
        Frac(self.0 * rhs.0)
    }
}

impl Div for Frac {
    type Output = Frac;
    /// # Panics
    ///
    /// Panics on division by zero. Use [`Frac::inv`] where the divisor may be
    /// zero; a source-reachable division reports `RA02` or `EC04` instead.
    fn div(self, rhs: Frac) -> Frac {
        assert!(!rhs.is_zero(), "frac division by zero");
        Frac(self.0 / rhs.0)
    }
}

impl Neg for Frac {
    type Output = Frac;
    fn neg(self) -> Frac {
        Frac(-self.0)
    }
}

impl fmt::Display for Frac {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_integer() {
            write!(f, "{}", self.numer())
        } else {
            write!(f, "{}/{}", self.numer(), self.denom())
        }
    }
}

impl fmt::Debug for Frac {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Frac({self})")
    }
}

impl Default for Frac {
    fn default() -> Frac {
        Frac::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_and_predicates() {
        assert!(Frac::zero().is_zero());
        assert!(Frac::one().is_one());
        assert!(Frac::from_int(7).is_integer());
        assert!(!Frac::new(1, 2).is_integer());
        assert!(Frac::from_int(3).is_positive());
        assert!(Frac::from_int(-3).is_negative());
    }

    #[test]
    fn fractions_are_kept_in_lowest_terms() {
        let f = Frac::new(6, 8);
        assert_eq!(f.numer().to_string(), "3");
        assert_eq!(f.denom().to_string(), "4");
        // equality is therefore structural
        assert_eq!(Frac::new(6, 8), Frac::new(3, 4));
        assert_eq!(Frac::new(-1, 2), Frac::new(1, -2));
    }

    #[test]
    fn arithmetic_is_exact() {
        // the classic float failure: 1/10 + 2/10 == 3/10 holds here
        let a = Frac::new(1, 10);
        let b = Frac::new(2, 10);
        assert_eq!(a + b, Frac::new(3, 10));

        // a third is exact, and thirds sum to one
        let third = Frac::new(1, 3);
        assert_eq!(
            third.clone() + third.clone() + third,
            Frac::one(),
            "1/3 must be exact: reflections about presented spans produce \
             denominators like this one"
        );
    }

    #[test]
    fn arbitrary_precision_does_not_overflow() {
        // precision is arbitrary, not fixed-width. a 24-qubit ket index
        // is around 1.1e23, past u64::MAX, and must be representable
        let big: BigInt = "111111111111111111111111".parse().unwrap();
        let f = Frac::from_bigint(big.clone());
        assert_eq!(f.numer(), &big);
        let squared = f.clone() * f;
        assert!(squared.numer().to_string().len() > 40);
    }

    #[test]
    fn division_and_inversion() {
        assert_eq!(Frac::from_int(1) / Frac::from_int(4), Frac::new(1, 4));
        assert_eq!(Frac::new(2, 3).inv(), Some(Frac::new(3, 2)));
        assert_eq!(Frac::zero().inv(), None);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn dividing_by_zero_panics_rather_than_returning_a_sentinel() {
        let _ = Frac::one() / Frac::zero();
    }

    #[test]
    #[should_panic(expected = "denominator is zero")]
    fn a_zero_denominator_panics() {
        let _ = Frac::new(1, 0);
    }

    #[test]
    fn checked_construction_reports_a_zero_denominator() {
        assert!(Frac::checked_new(BigInt::from(1), BigInt::from(0)).is_none());
        assert_eq!(
            Frac::checked_new(BigInt::from(1), BigInt::from(2)),
            Some(Frac::new(1, 2))
        );
    }

    #[test]
    fn decimal_literals_parse_exactly() {
        assert_eq!(Frac::parse_decimal("1.0"), Some(Frac::one()));
        assert_eq!(Frac::parse_decimal("0.5"), Some(Frac::new(1, 2)));
        assert_eq!(Frac::parse_decimal("42"), Some(Frac::from_int(42)));
        // 2.5e-3 is exactly 1/400, which no binary float represents
        assert_eq!(Frac::parse_decimal("2.5e-3"), Some(Frac::new(1, 400)));
        assert_eq!(Frac::parse_decimal("1e3"), Some(Frac::from_int(1000)));
        // 0.1 is exactly a tenth here, unlike as an f64
        assert_eq!(Frac::parse_decimal("0.1"), Some(Frac::new(1, 10)));
    }

    #[test]
    fn decimal_literals_admit_digit_separators() {
        assert_eq!(Frac::parse_decimal("1_000"), Some(Frac::from_int(1000)));
        assert_eq!(Frac::parse_decimal("1_0.2_5"), Some(Frac::new(41, 4)));
    }

    #[test]
    fn malformed_decimals_are_rejected() {
        assert_eq!(Frac::parse_decimal("abc"), None);
        assert_eq!(Frac::parse_decimal(""), None);
        assert_eq!(Frac::parse_decimal("1e"), None);
        assert_eq!(Frac::parse_decimal("1.2.3"), None);
    }

    #[test]
    fn powers_and_absolute_value() {
        assert_eq!(Frac::new(1, 2).pow(3), Frac::new(1, 8));
        assert_eq!(Frac::from_int(2).pow(0), Frac::one());
        assert_eq!(Frac::from_int(-5).abs(), Frac::from_int(5));
    }

    #[test]
    fn negation_and_subtraction() {
        assert_eq!(-Frac::new(1, 3), Frac::new(-1, 3));
        assert_eq!(Frac::one() - Frac::new(1, 4), Frac::new(3, 4));
    }

    #[test]
    fn ordering_is_by_value() {
        let mut v = vec![Frac::new(1, 2), Frac::from_int(-1), Frac::new(1, 3)];
        v.sort();
        assert_eq!(v, vec![Frac::from_int(-1), Frac::new(1, 3), Frac::new(1, 2)]);
    }

    #[test]
    fn display_reads_naturally() {
        assert_eq!(Frac::from_int(3).to_string(), "3");
        assert_eq!(Frac::new(3, 4).to_string(), "3/4");
        assert_eq!(Frac::new(-1, 2).to_string(), "-1/2");
        assert_eq!(format!("{:?}", Frac::new(1, 2)), "Frac(1/2)");
    }

    #[test]
    fn the_lossy_float_conversion_is_marked_as_such() {
        // only for printing; never for a judgement
        assert!((Frac::new(1, 4).to_f64_lossy() - 0.25).abs() < 1e-12);
    }
}
