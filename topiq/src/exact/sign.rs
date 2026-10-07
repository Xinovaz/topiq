//! Which side of zero a real element of K lies on.
//!
//! Equality in K is exact, but order is not algebraic: whether a real
//! element is positive depends on where `ζ_N` sits in the plane. The value is
//! a sum of rational multiples of cosines of rational multiples of π, and
//! each cosine is enclosed in a rational interval that is certain to hold it:
//! π by Machin's formula, whose alternating series bound themselves, and the
//! cosine by its Taylor series, with the next term bounding what is left. The
//! intervals are narrowed until their sum excludes zero, which it does once
//! they are narrow enough, the element not being zero. Nothing is rounded.

use std::cmp::Ordering;

use super::cyclo::Cyclo;
use super::frac::Frac;

/// A rational interval (i.e. `lo ≤ x ≤ hi`).
#[derive(Clone, Debug)]
struct Interval {
    lo: Frac,
    hi: Frac,
}

impl Interval {
    fn point(x: Frac) -> Interval {
        Interval { lo: x.clone(), hi: x }
    }

    fn around(x: Frac, r: Frac) -> Interval {
        Interval {
            lo: x.clone() - r.clone(),
            hi: x + r,
        }
    }

    fn add(&self, o: &Interval) -> Interval {
        Interval {
            lo: self.lo.clone() + o.lo.clone(),
            hi: self.hi.clone() + o.hi.clone(),
        }
    }

    /// Scaled by an exact rational.
    fn scale(&self, c: &Frac) -> Interval {
        let (a, b) = (self.lo.clone() * c.clone(), self.hi.clone() * c.clone());
        if c.is_negative() { Interval { lo: b, hi: a } } else { Interval { lo: a, hi: b } }
    }
}

/// `2^-bits`.
fn tolerance(bits: u32) -> Frac {
    Frac::from_bigint(num_bigint::BigInt::from(1) << bits).inv().expect("not zero")
}

/// arctan(1/m), as an interval of width at most `2·2^-bits`.
fn arctan_inverse(m: i64, bits: u32) -> Interval {
    let eps = tolerance(bits);
    let m2 = Frac::from_int(m * m);
    let mut power = Frac::new(1, m);
    let mut sum = Frac::zero();
    let mut j: i64 = 0;
    loop {
        let term = power.clone() / Frac::from_int(2 * j + 1);
        if j % 2 == 0 {
            sum = sum + term;
        } else {
            sum = sum - term;
        }
        power = power / m2.clone();
        j += 1;
        let next = power.clone() / Frac::from_int(2 * j + 1);
        if (next.clone() - eps.clone()).is_negative() {
            // an alternating series with shrinking terms is within its next
            // term of its sum
            return Interval::around(sum, next);
        }
    }
}

/// π, as an interval of width about `2^-bits`.
fn pi(bits: u32) -> Interval {
    let a = arctan_inverse(5, bits + 6);
    let b = arctan_inverse(239, bits + 6);
    Interval {
        lo: a.lo.clone() * Frac::from_int(16) - b.hi.clone() * Frac::from_int(4),
        hi: a.hi * Frac::from_int(16) - b.lo * Frac::from_int(4),
    }
}

/// cos(2π·k/n), as an interval.
fn cos_turn(k: i64, n: i64, pi: &Interval, bits: u32) -> Interval {
    // reduce the angle to [0, π/2] by the symmetries of the cosine, exactly
    let mut q = Frac::new(k.rem_euclid(n), n);
    let half = Frac::new(1, 2);
    if (q.clone() - half.clone()).is_positive() {
        q = Frac::one() - q;
    }
    let mut negate = false;
    if (q.clone() - Frac::new(1, 4)).is_positive() {
        q = half - q;
        negate = true;
    }
    // θ = 2πq, enclosed; the cosine is taken at the middle of the enclosure,
    // and moves by at most the enclosure's half-width across it
    let theta = pi.scale(&(q * Frac::from_int(2)));
    let mid = (theta.lo.clone() + theta.hi.clone()) * Frac::new(1, 2);
    let spread = (theta.hi - theta.lo) * Frac::new(1, 2);
    let eps = tolerance(bits);
    let x2 = mid.clone() * mid;
    let mut term = Frac::one();
    let mut sum = Frac::one();
    let mut j: i64 = 1;
    loop {
        term = -(term * x2.clone()) / Frac::from_int((2 * j - 1) * (2 * j));
        sum = sum + term.clone();
        j += 1;
        let next = (term.clone() * x2.clone() / Frac::from_int((2 * j - 1) * (2 * j))).abs();
        if (next.clone() - eps.clone()).is_negative() {
            let c = Interval::around(sum, next + spread);
            return if negate { c.scale(&Frac::from_int(-1)) } else { c };
        }
    }
}

/// The sign of `x`, which must be real: `None` when it is not.
pub fn real_sign(x: &Cyclo) -> Option<Ordering> {
    if *x != x.conj() {
        return None;
    }
    if x.is_zero() {
        return Some(Ordering::Equal);
    }
    let n = i64::from(x.conductor());
    let mut bits = 48;
    loop {
        let p = pi(bits);
        let mut total = Interval::point(Frac::zero());
        for (k, c) in x.coeffs().iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            total = total.add(&cos_turn(k as i64, n, &p, bits).scale(c));
        }
        if total.lo.is_positive() {
            return Some(Ordering::Greater);
        }
        if total.hi.is_negative() {
            return Some(Ordering::Less);
        }
        bits *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signs_of_real_elements_are_decided() {
        let s = Cyclo::sqrt2(8).unwrap();
        assert_eq!(real_sign(&s), Some(Ordering::Greater));
        assert_eq!(real_sign(&s.neg()), Some(Ordering::Less));
        // √2 − 1.4142 is small and positive; √2 − 1.4143 small and negative
        let a = s.sub(&Cyclo::from_frac(8, Frac::new(14142, 10000)));
        let b = s.sub(&Cyclo::from_frac(8, Frac::new(14143, 10000)));
        assert_eq!(real_sign(&a), Some(Ordering::Greater));
        assert_eq!(real_sign(&b), Some(Ordering::Less));
        // √3 at conductor 24, against 1.7320508075688772 and just above it
        let r3 = Cyclo::zeta(12).add(&Cyclo::zeta_pow(12, -1)).embed(24).unwrap();
        let close = r3.sub(&Cyclo::from_frac(24, Frac::new(17_320_508_075_688_772, 10_000_000_000_000_000)));
        assert_eq!(real_sign(&close), Some(Ordering::Greater));
        assert_eq!(real_sign(&Cyclo::zeta(8)), None);
        assert_eq!(real_sign(&Cyclo::zero(8)), Some(Ordering::Equal));
    }
}
