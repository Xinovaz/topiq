//! `cyclo<N>`: the cyclotomic field K = `Q(ζ_N)`.
//!
//! An element is held in the power basis of `ζ_N`, reduced by the N-th
//! cyclotomic polynomial `Φ_N`. Equality is then simply coefficient comparison,
//! with no normalisation step that could go wrong. The degree is φ(N): for the
//! conforming conductors 8, 16 and 24 that is 4, 8 and 8.
//!
//! # Why a field and not a ring
//!
//! K is a field, and the widening is deliberate. Reflections about presented
//! spans, and the reconstruction of unnormalised Schmidt branches, produce
//! denominators such as 1/3, which no ring built from roots of unity can
//! absorb. So coefficients here are [`Frac`] rather than integers, and
//! [`Cyclo::inv`] is total on non-zero elements.
//!
//! # Why `cyclo<N>` is not one fixed field
//!
//! No single field smaller than `Q(ζ_48)` holds every conductor's: the
//! conductors are 8, 16 and 24, and 16 does not divide 24, so `Q(ζ_16)` is
//! not a subfield of `Q(ζ_24)`. Fixing one ambient field would either
//! exclude a conforming conductor or quadruple the degree of every element.
//!
//! The construction stays generic in N instead, and [`Cyclo::embed`] performs
//! the conversion from `cyclo<M>` to `cyclo<N>` whenever M divides N.

use std::fmt;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Zero};

use super::frac::Frac;

/// The coefficients of `Φ_N`.
///
/// Computed from the defining identity x^N − 1 = Π_{d | N} `Φ_d(x)` by dividing
/// out the proper divisors. Every quotient here is exact over the integers.
fn cyclotomic_poly(n: u32) -> Vec<BigInt> {
    assert!(n >= 1, "the cyclotomic index must be positive");
    if n == 1 {
        // Phi_1 = x - 1
        return vec![BigInt::from(-1), BigInt::one()];
    }
    // start from x^n - 1
    let mut numer = vec![BigInt::zero(); n as usize + 1];
    numer[0] = BigInt::from(-1);
    numer[n as usize] = BigInt::one();

    for d in 1..n {
        if n.is_multiple_of(d) {
            numer = poly_div_exact(&numer, &cyclotomic_poly(d));
        }
    }
    numer
}

/// Exact division of integer polynomials, where `divisor` is monic and the
/// quotient is known to be integral.
fn poly_div_exact(numer: &[BigInt], divisor: &[BigInt]) -> Vec<BigInt> {
    let dn = poly_degree_int(numer);
    let dd = poly_degree_int(divisor);
    assert!(dd <= dn, "cyclotomic division by a larger polynomial");
    let mut rem = numer.to_vec();
    let mut quot = vec![BigInt::zero(); dn - dd + 1];
    let lead = divisor[dd].clone();
    for shift in (0..=dn - dd).rev() {
        let coeff = &rem[shift + dd] / &lead;
        if coeff.is_zero() {
            continue;
        }
        for (j, dv) in divisor.iter().enumerate().take(dd + 1) {
            rem[shift + j] -= &coeff * dv;
        }
        quot[shift] = coeff;
    }
    debug_assert!(
        rem.iter().all(BigInt::is_zero),
        "cyclotomic division left a remainder"
    );
    quot
}

fn poly_degree_int(p: &[BigInt]) -> usize {
    p.iter().rposition(|c| !c.is_zero()).unwrap_or(0)
}

/// Euler's totient.
fn totient(n: u32) -> u32 {
    let mut result = n;
    let mut m = n;
    let mut p = 2;
    while p * p <= m {
        if m.is_multiple_of(p) {
            while m.is_multiple_of(p) {
                m /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if m > 1 {
        result -= result / m;
    }
    result
}

/// An element of `Q(ζ_N)`, in the power basis of `ζ_N` reduced by `Φ_N`.
///
/// The arithmetic is inherent rather than through `std::ops`, because every
/// operation takes `&self` and may **widen the conductor**: adding an element
/// of `Q(ζ_8)` to one of `Q(ζ_12)` yields an element of `Q(ζ_24)`.
/// `std::ops::Add` cannot express that, and would invite by-value arithmetic
/// that clones a bignum vector per operation.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Cyclo {
    /// The order of the root of unity.
    n: u32,
    /// Coefficients from the constant term upwards; length is exactly φ(N).
    c: Vec<Frac>,
}

#[allow(
    clippy::should_implement_trait,
    reason = "these widen the conductor and take &self; see the type's documentation"
)]
impl Cyclo {
    /// The zero of `Q(ζ_N)`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero.
    pub fn zero(n: u32) -> Cyclo {
        assert!(n >= 1, "the cyclotomic index must be positive");
        Cyclo {
            n,
            c: vec![Frac::zero(); totient(n) as usize],
        }
    }

    /// The one of `Q(ζ_N)`.
    pub fn one(n: u32) -> Cyclo {
        Cyclo::from_frac(n, Frac::one())
    }

    /// A rational.
    pub fn from_frac(n: u32, f: Frac) -> Cyclo {
        let mut z = Cyclo::zero(n);
        z.c[0] = f;
        z
    }

    /// An integer.
    pub fn from_int(n: u32, v: i64) -> Cyclo {
        Cyclo::from_frac(n, Frac::from_int(v))
    }

    /// The element with these power-basis coefficients, reduced by `Φ_N`:
    /// there may be any number of them.
    pub fn from_coeffs(n: u32, coeffs: Vec<Frac>) -> Cyclo {
        let degree = totient(n) as usize;
        Cyclo {
            n,
            c: reduce(coeffs, n, degree),
        }
    }

    /// `ζ_N^k`, that is e^(2πik/N), what the literal `w(k, N)` denotes.
    ///
    /// `k` is reduced modulo N first, so a negative or large exponent is fine.
    pub fn zeta_pow(n: u32, k: i64) -> Cyclo {
        let k = k.rem_euclid(n as i64) as usize;
        let mut raw = vec![Frac::zero(); k + 1];
        raw[k] = Frac::one();
        Cyclo::from_coeffs(n, raw)
    }

    /// `ζ_N` itself.
    pub fn zeta(n: u32) -> Cyclo {
        Cyclo::zeta_pow(n, 1)
    }

    /// The imaginary unit.
    ///
    /// Returns `None` unless 4 divides N. Every conforming conductor is a
    /// multiple of eight, so this always succeeds in practice; the
    /// option exists because the type admits any N.
    pub fn imaginary_unit(n: u32) -> Option<Cyclo> {
        n.is_multiple_of(4).then(|| Cyclo::zeta_pow(n, (n / 4) as i64))
    }

    /// One over the square root of two, what the literal `isq2` denotes.
    ///
    /// Returns `None` unless 8 divides N, which is why a conductor must be a
    /// multiple of eight.
    ///
    /// The value is `(ζ_8 − ζ_8³)/2`: `ζ_8` is e^(iπ/4) and `ζ_8³` is
    /// e^(3iπ/4), so their difference is √2.
    pub fn isqrt2(n: u32) -> Option<Cyclo> {
        if !n.is_multiple_of(8) {
            return None;
        }
        let e = (n / 8) as i64;
        let z = Cyclo::zeta_pow(n, e);
        let z3 = Cyclo::zeta_pow(n, 3 * e);
        Some(z.sub(&z3).scale(&Frac::new(1, 2)))
    }

    /// The argument of this element as a whole number of N-th parts of a
    /// turn: the m for which `ζ_N^(−m)` times it is real and positive.
    /// `None` for zero, and for an element whose argument is no such angle.
    ///
    /// Each of the N candidates is tested exactly: realness by comparison
    /// with the conjugate, and positivity by [`super::sign::real_sign`].
    pub fn arg(&self) -> Option<u32> {
        if self.is_zero() {
            return None;
        }
        (0..self.n).find(|&m| {
            let w = self.mul_zeta_pow(-i64::from(m));
            w == w.conj() && super::sign::real_sign(&w) == Some(std::cmp::Ordering::Greater)
        })
    }

    /// The square root of two, `ζ_8` − `ζ_8³`; `None` unless 8 divides N.
    pub fn sqrt2(n: u32) -> Option<Cyclo> {
        Some(Cyclo::isqrt2(n)?.scale(&Frac::from_int(2)))
    }

    /// This times `ζ_N^k`, which is a shift of the coefficients followed by
    /// a reduction, and cheaper than a general product.
    pub fn mul_zeta_pow(&self, k: i64) -> Cyclo {
        let k = k.rem_euclid(i64::from(self.n)) as usize;
        let mut raw = vec![Frac::zero(); k];
        raw.extend_from_slice(&self.c);
        Cyclo::from_coeffs(self.n, raw)
    }

    /// The largest power of two in any coefficient's denominator.
    pub fn two_denominator(&self) -> u64 {
        self.c.iter().filter_map(|x| x.denom().trailing_zeros()).max().unwrap_or(0)
    }

    /// Whether this is an algebraic integer, an element of `Z[ζ_N]`.
    ///
    /// The power basis of `ζ_N` is an integral basis of the ring of integers
    /// of K, so that is exactly when every coefficient is an integer.
    pub fn is_integral(&self) -> bool {
        self.c.iter().all(Frac::is_integer)
    }

    /// The odd part of the least common denominator of the coefficients: 1
    /// exactly when this lies in `Z[1/2, ζ_N]`, the ring every amplitude a
    /// circuit of the standard gates prepares lies in.
    pub fn odd_denominator(&self) -> BigInt {
        let mut d = BigInt::one();
        for x in &self.c {
            d = d.lcm(x.denom());
        }
        while d.is_even() {
            d /= 2;
        }
        d
    }

    /// The order of the root of unity (i.e. the N of `cyclo<N>`).
    pub fn conductor(&self) -> u32 {
        self.n
    }

    /// The coefficients in the power basis.
    pub fn coeffs(&self) -> &[Frac] {
        &self.c
    }

    /// The degree of the field over Q (i.e. φ(N)).
    pub fn degree(&self) -> usize {
        self.c.len()
    }

    /// Whether this is zero.
    pub fn is_zero(&self) -> bool {
        self.c.iter().all(Frac::is_zero)
    }

    /// Whether this is one.
    pub fn is_one(&self) -> bool {
        self.c[0].is_one() && self.c[1..].iter().all(Frac::is_zero)
    }

    /// The rational value.
    pub fn as_rational(&self) -> Option<Frac> {
        self.c[1..]
            .iter()
            .all(Frac::is_zero)
            .then(|| self.c[0].clone())
    }

    /// Embeds this element into `Q(ζ_M)`.
    ///
    /// Returns `None` unless the current conductor divides `m`. This is the
    /// implicit conversion: `cyclo<M>` converts to `cyclo<N>` when M divides
    /// N.
    pub fn embed(&self, m: u32) -> Option<Cyclo> {
        if m == self.n {
            return Some(self.clone());
        }
        if m == 0 || !m.is_multiple_of(self.n) {
            return None;
        }
        let step = (m / self.n) as usize;
        let mut raw = vec![Frac::zero(); (self.c.len() - 1) * step + 1];
        for (k, coeff) in self.c.iter().enumerate() {
            raw[k * step] = coeff.clone();
        }
        Some(Cyclo::from_coeffs(m, raw))
    }

    /// This element as one of `Q(ζ_m)`, when it lies in that subfield:
    /// the inverse of [`Cyclo::embed`]. `None` when `m` does not divide the
    /// conductor or the element is not in the subfield.
    ///
    /// The embedding is Q-linear, so the element's coordinates in the
    /// subfield are the solution of a linear system over Q, the columns being
    /// the embedded powers `ζ_m^j`.
    pub fn restrict(&self, m: u32) -> Option<Cyclo> {
        if m == self.n {
            return Some(self.clone());
        }
        if m == 0 || !self.n.is_multiple_of(m) {
            return None;
        }
        let cols = totient(m) as usize;
        let rows = self.c.len();
        // the augmented matrix [ζ_m^0 … ζ_m^(k−1) | self], one row per
        // coordinate of the larger field
        let basis: Vec<Cyclo> = (0..cols)
            .map(|j| Cyclo::zeta_pow(m, j as i64).embed(self.n).expect("m divides the conductor"))
            .collect();
        let mut a: Vec<Vec<Frac>> = (0..rows)
            .map(|r| {
                let mut row: Vec<Frac> = basis.iter().map(|b| b.c[r].clone()).collect();
                row.push(self.c[r].clone());
                row
            })
            .collect();
        let mut pivots = Vec::with_capacity(cols);
        let mut top = 0;
        for col in 0..cols {
            let Some(p) = (top..rows).find(|&r| !a[r][col].is_zero()) else { continue };
            a.swap(top, p);
            let lead = a[top][col].inv().expect("the pivot is not zero");
            for x in &mut a[top] {
                *x = x.clone() * lead.clone();
            }
            let pivot = a[top].clone();
            for (r, row) in a.iter_mut().enumerate() {
                if r != top && !row[col].is_zero() {
                    let f = row[col].clone();
                    for (x, p) in row.iter_mut().zip(&pivot) {
                        *x = x.clone() - p.clone() * f.clone();
                    }
                }
            }
            pivots.push(col);
            top += 1;
        }
        // consistent exactly when no row left without a pivot asks for a
        // non-zero value
        if a[top..].iter().any(|row| !row[cols].is_zero()) {
            return None;
        }
        let mut c = vec![Frac::zero(); cols];
        for (r, &col) in pivots.iter().enumerate() {
            c[col] = a[r][cols].clone();
        }
        Some(Cyclo { n: m, c })
    }

    /// The smallest common field of two elements.
    ///
    /// The common conductor is the least common multiple of the two. This is
    /// the case that arises when a judgement crosses a boundary between units
    /// of conductors M and N.
    pub fn unify(a: &Cyclo, b: &Cyclo) -> (Cyclo, Cyclo) {
        if a.n == b.n {
            return (a.clone(), b.clone());
        }
        let m = a.n.lcm(&b.n);
        (
            a.embed(m).expect("lcm is a multiple"),
            b.embed(m).expect("lcm is a multiple"),
        )
    }

    /// Addition, widening to the common field if the conductors differ.
    pub fn add(&self, other: &Cyclo) -> Cyclo {
        Cyclo::pointwise(self, other, |x, y| x + y)
    }

    /// Subtraction, widening to the common field if the conductors differ.
    pub fn sub(&self, other: &Cyclo) -> Cyclo {
        Cyclo::pointwise(self, other, |x, y| x - y)
    }

    /// `f` of each pair of coefficients, in the common field.
    fn pointwise(a: &Cyclo, b: &Cyclo, f: impl Fn(Frac, Frac) -> Frac) -> Cyclo {
        let (a, b) = Cyclo::unify(a, b);
        Cyclo {
            n: a.n,
            c: a.c.into_iter().zip(b.c).map(|(x, y)| f(x, y)).collect(),
        }
    }

    /// Negation.
    pub fn neg(&self) -> Cyclo {
        Cyclo {
            n: self.n,
            c: self.c.iter().map(|x| -x.clone()).collect(),
        }
    }

    /// Multiplication, widening to the common field if the conductors differ.
    pub fn mul(&self, other: &Cyclo) -> Cyclo {
        let (a, b) = Cyclo::unify(self, other);
        let mut raw = vec![Frac::zero(); a.c.len() + b.c.len()];
        for (i, x) in a.c.iter().enumerate() {
            if x.is_zero() {
                continue;
            }
            for (j, y) in b.c.iter().enumerate() {
                if y.is_zero() {
                    continue;
                }
                raw[i + j] = raw[i + j].clone() + x.clone() * y.clone();
            }
        }
        Cyclo::from_coeffs(a.n, raw)
    }

    /// Multiplication by a rational.
    pub fn scale(&self, f: &Frac) -> Cyclo {
        Cyclo {
            n: self.n,
            c: self.c.iter().map(|x| x.clone() * f.clone()).collect(),
        }
    }

    /// Complex conjugation, which sends `ζ_N` to `ζ_N^(−1)`.
    ///
    /// This is an automorphism of K, so the result stays exact, which is what
    /// lets the state-normalisation check be decided in K rather
    /// than numerically.
    pub fn conj(&self) -> Cyclo {
        let n = self.n as usize;
        let mut raw = vec![Frac::zero(); n];
        for (k, coeff) in self.c.iter().enumerate() {
            raw[(n - k) % n] = coeff.clone();
        }
        Cyclo::from_coeffs(self.n, raw)
    }

    /// The squared modulus z·conj(z), which is exact and lies in K.
    ///
    /// Used by the normalisation test Σ|c|² = 1.
    pub fn norm_sq(&self) -> Cyclo {
        self.mul(&self.conj())
    }

    /// The multiplicative inverse, or `None` for zero.
    ///
    /// Computed by the extended Euclidean algorithm in `Q[x]` against `Φ_N`, which
    /// terminates because `Φ_N` is irreducible over Q and so every non-zero
    /// element is a unit.
    pub fn inv(&self) -> Option<Cyclo> {
        if self.is_zero() {
            return None;
        }
        Some(Cyclo::from_coeffs(self.n, poly_mod_inverse(&self.c, &modulus(self.n))?))
    }

    /// Division, or `None` if the divisor is zero.
    pub fn div(&self, other: &Cyclo) -> Option<Cyclo> {
        let (a, b) = Cyclo::unify(self, other);
        Some(a.mul(&b.inv()?))
    }

    /// Raises to a non-negative integer power.
    pub fn pow(&self, exp: u32) -> Cyclo {
        let mut acc = Cyclo::one(self.n);
        let mut base = self.clone();
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base);
            }
            base = base.mul(&base);
            e >>= 1;
        }
        acc
    }

    /// An `f64` approximation of the real and imaginary parts.
    ///
    /// For printing and diagnostics only. No judgement fact may be decided by
    /// a numerical comparison, and no floating representation may stand in
    /// for an exact one where a judgement depends on it.
    pub fn to_complex_lossy(&self) -> (f64, f64) {
        let (mut re, mut im) = (0.0, 0.0);
        for (k, coeff) in self.c.iter().enumerate() {
            let theta = 2.0 * std::f64::consts::PI * (k as f64) / (self.n as f64);
            let a = coeff.to_f64_lossy();
            re += a * theta.cos();
            im += a * theta.sin();
        }
        (re, im)
    }
}

/// `Φ_N` with rational coefficients, computed once for each N.
fn modulus(n: u32) -> std::sync::Arc<[Frac]> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u32, Arc<[Frac]>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    map.entry(n)
        .or_insert_with(|| cyclotomic_poly(n).into_iter().map(Frac::from_bigint).collect())
        .clone()
}

/// Reduces a raw power-basis vector modulo `Φ_N` to exactly `degree` terms.
fn reduce(mut raw: Vec<Frac>, n: u32, degree: usize) -> Vec<Frac> {
    if raw.len() <= degree {
        raw.resize(degree, Frac::zero());
        return raw;
    }
    let modulus = modulus(n);
    debug_assert_eq!(modulus.len() - 1, degree);
    // Phi_N is monic, so each step subtracts a shifted copy
    for i in (degree..raw.len()).rev() {
        let lead = raw[i].clone();
        if lead.is_zero() {
            continue;
        }
        raw[i] = Frac::zero();
        let shift = i - degree;
        for (j, m) in modulus.iter().enumerate().take(degree) {
            if !m.is_zero() {
                raw[shift + j] = raw[shift + j].clone() - lead.clone() * m.clone();
            }
        }
    }
    raw.truncate(degree);
    raw
}

fn poly_trim(mut p: Vec<Frac>) -> Vec<Frac> {
    while p.len() > 1 && p.last().is_some_and(Frac::is_zero) {
        p.pop();
    }
    p
}

fn poly_is_zero(p: &[Frac]) -> bool {
    p.iter().all(Frac::is_zero)
}

fn poly_mul(a: &[Frac], b: &[Frac]) -> Vec<Frac> {
    let mut out = vec![Frac::zero(); a.len() + b.len()];
    for (i, x) in a.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        for (j, y) in b.iter().enumerate() {
            out[i + j] = out[i + j].clone() + x.clone() * y.clone();
        }
    }
    poly_trim(out)
}

fn poly_sub(a: &[Frac], b: &[Frac]) -> Vec<Frac> {
    let n = a.len().max(b.len());
    let mut out = vec![Frac::zero(); n];
    for (i, slot) in out.iter_mut().enumerate() {
        let x = a.get(i).cloned().unwrap_or_else(Frac::zero);
        let y = b.get(i).cloned().unwrap_or_else(Frac::zero);
        *slot = x - y;
    }
    poly_trim(out)
}

/// Divides `a` by `b` over Q, returning (quotient, remainder).
fn poly_divmod(a: &[Frac], b: &[Frac]) -> (Vec<Frac>, Vec<Frac>) {
    let b = poly_trim(b.to_vec());
    let mut rem = poly_trim(a.to_vec());
    let db = rem_degree(&b);
    let mut quot = vec![Frac::zero(); rem.len().saturating_sub(db).max(1)];
    let lead_inv = b[db].inv().expect("a trimmed divisor has a non-zero lead");
    while !poly_is_zero(&rem) && rem_degree(&rem) >= db {
        let dr = rem_degree(&rem);
        let coeff = rem[dr].clone() * lead_inv.clone();
        let shift = dr - db;
        quot[shift] = coeff.clone();
        let mut sub = vec![Frac::zero(); shift];
        sub.extend(b.iter().map(|x| x.clone() * coeff.clone()));
        rem = poly_sub(&rem, &sub);
    }
    (poly_trim(quot), rem)
}

fn rem_degree(p: &[Frac]) -> usize {
    p.iter().rposition(|c| !c.is_zero()).unwrap_or(0)
}

/// The inverse of `a` modulo `m` in Q[x].
fn poly_mod_inverse(a: &[Frac], m: &[Frac]) -> Option<Vec<Frac>> {
    let mut old_r = poly_trim(m.to_vec());
    let mut r = poly_trim(a.to_vec());
    let mut old_s = vec![Frac::zero()];
    let mut s = vec![Frac::one()];

    while !poly_is_zero(&r) {
        let (q, rem) = poly_divmod(&old_r, &r);
        old_r = std::mem::replace(&mut r, rem);
        let new_s = poly_sub(&old_s, &poly_mul(&q, &s));
        old_s = std::mem::replace(&mut s, new_s);
    }
    // old_r is the gcd; it must be a non-zero constant for an inverse to exist
    let g = poly_trim(old_r);
    if rem_degree(&g) != 0 || g[0].is_zero() {
        return None;
    }
    let scale = g[0].inv()?;
    Some(old_s.into_iter().map(|c| c * scale.clone()).collect())
}

impl fmt::Display for Cyclo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut first = true;
        for (k, coeff) in self.c.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            match k {
                0 => write!(f, "{coeff}")?,
                1 => write!(f, "{coeff}*z{}", self.n)?,
                _ => write!(f, "{coeff}*z{}^{k}", self.n)?,
            }
        }
        Ok(())
    }
}

impl fmt::Debug for Cyclo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cyclo<{}>({})", self.n, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn cyclotomic_polynomials_match_the_classical_values() {
        let show = |n: u32| {
            cyclotomic_poly(n)
                .iter()
                .map(BigInt::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        assert_eq!(show(1), "-1,1"); // x - 1
        assert_eq!(show(2), "1,1"); // x + 1
        assert_eq!(show(3), "1,1,1"); // x^2 + x + 1
        assert_eq!(show(4), "1,0,1"); // x^2 + 1
        assert_eq!(show(6), "1,-1,1"); // x^2 - x + 1
        assert_eq!(show(8), "1,0,0,0,1"); // x^4 + 1
        assert_eq!(show(12), "1,0,-1,0,1"); // x^4 - x^2 + 1
    }

    #[test]
    fn the_degree_is_the_totient() {
        assert_eq!(totient(8), 4);
        assert_eq!(totient(16), 8);
        assert_eq!(totient(24), 8);
        for n in [8, 16, 24] {
            assert_eq!(Cyclo::zero(n).degree(), totient(n) as usize);
            assert_eq!(cyclotomic_poly(n).len() - 1, totient(n) as usize);
        }
    }

    #[test]
    fn the_conforming_conductors_do_not_share_one_ambient_field() {
        // 16 does not divide 24, so Q(zeta_16) is not inside Q(zeta_24). this
        // is why cyclo<N> stays generic in N rather than collapsing to a fixed
        // field; the test guards that reasoning
        assert_ne!(24 % 16, 0);
        assert!(Cyclo::zeta(16).embed(24).is_none());
        assert!(Cyclo::zeta(8).embed(24).is_some());
        assert!(Cyclo::zeta(8).embed(16).is_some());
    }

    #[test]
    fn field_axioms_hold() {
        let n = 8;
        let a = Cyclo::zeta_pow(n, 1).add(&Cyclo::from_int(n, 2));
        let b = Cyclo::zeta_pow(n, 3).sub(&Cyclo::from_frac(n, Frac::new(1, 3)));
        let c = Cyclo::isqrt2(n).unwrap();

        // commutativity and associativity
        assert_eq!(a.add(&b), b.add(&a));
        assert_eq!(a.mul(&b), b.mul(&a));
        assert_eq!(a.add(&b).add(&c), a.add(&b.add(&c)));
        assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));
        // distributivity
        assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
        // identities and inverses
        assert_eq!(a.add(&Cyclo::zero(n)), a);
        assert_eq!(a.mul(&Cyclo::one(n)), a);
        assert!(a.add(&a.neg()).is_zero());
        assert!(a.mul(&a.inv().unwrap()).is_one());
    }

    #[test]
    fn zeta_is_a_primitive_root_of_unity() {
        for n in [8, 12, 16, 24] {
            let z = Cyclo::zeta(n);
            assert!(z.pow(n).is_one(), "zeta_{n}^{n} should be 1");
            // primitive: no smaller positive power is 1
            for k in 1..n {
                assert!(!z.pow(k).is_one(), "zeta_{n}^{k} should not be 1");
            }
        }
    }

    #[test]
    fn w_k_n_is_zeta_to_the_k() {
        // `w(k, N)` denotes e^(2 pi i k / N)
        let n = 8;
        for k in 0..8i64 {
            let z = Cyclo::zeta_pow(n, k);
            let (re, im) = z.to_complex_lossy();
            let theta = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
            assert!(approx(re, theta.cos()), "re of w({k},{n})");
            assert!(approx(im, theta.sin()), "im of w({k},{n})");
        }
    }

    #[test]
    fn exponents_are_reduced_modulo_n() {
        let n = 8;
        assert_eq!(Cyclo::zeta_pow(n, 9), Cyclo::zeta_pow(n, 1));
        assert_eq!(Cyclo::zeta_pow(n, -1), Cyclo::zeta_pow(n, 7));
        assert_eq!(Cyclo::zeta_pow(n, 0), Cyclo::one(n));
    }

    #[test]
    fn the_imaginary_unit_squares_to_minus_one() {
        for n in [8, 16, 24] {
            let i = Cyclo::imaginary_unit(n).expect("4 divides every conductor");
            assert_eq!(i.mul(&i), Cyclo::from_int(n, -1), "i^2 at N={n}");
        }
        // not available where 4 does not divide N
        assert!(Cyclo::imaginary_unit(3).is_none());
    }

    #[test]
    fn isq2_squares_to_one_half() {
        // a conductor is a multiple of eight so that 1/sqrt(2) and i are both
        // available; this checks it
        for n in [8, 16, 24] {
            let s = Cyclo::isqrt2(n).expect("8 divides every conforming conductor");
            assert_eq!(
                s.mul(&s),
                Cyclo::from_frac(n, Frac::new(1, 2)),
                "isq2^2 at N={n}"
            );
            let (re, im) = s.to_complex_lossy();
            assert!(approx(re, std::f64::consts::FRAC_1_SQRT_2));
            assert!(approx(im, 0.0));
        }
        assert!(Cyclo::isqrt2(4).is_none(), "8 does not divide 4");
        assert!(Cyclo::isqrt2(12).is_none(), "8 does not divide 12");
    }

    #[test]
    fn sqrt_three_needs_conductor_twenty_four() {
        // a conductor of 24 makes a coefficient needing sqrt(3) a matter of
        // re-declaration rather than refusal
        let root3 = |n: u32| {
            let z = Cyclo::zeta_pow(n, (n / 12) as i64); // e^(i pi / 6)
            z.add(&z.conj()) // 2 cos(pi/6) = sqrt(3)
        };
        let s = root3(24);
        assert_eq!(s.mul(&s), Cyclo::from_int(24, 3), "should be exactly 3");

        // and 24 is the smallest that admits it: the value needs a twelfth
        // root of unity, and of the multiples of eight up to 24 only 24 is a
        // multiple of 12
        let conforming: Vec<u32> = (1..=24).filter(|n| n % 8 == 0).collect();
        assert_eq!(conforming, vec![8, 16, 24]);
        let admitting: Vec<u32> = conforming.into_iter().filter(|n| n % 12 == 0).collect();
        assert_eq!(admitting, vec![24]);
    }

    #[test]
    fn conjugation_is_an_involution_and_inverts_zeta() {
        let n = 8;
        let z = Cyclo::zeta(n);
        assert_eq!(z.conj(), Cyclo::zeta_pow(n, -1));
        assert_eq!(z.conj().conj(), z);
        let a = z.add(&Cyclo::from_int(n, 3)).mul(&Cyclo::isqrt2(n).unwrap());
        assert_eq!(a.conj().conj(), a);
        // a rational is its own conjugate
        let q = Cyclo::from_frac(n, Frac::new(2, 5));
        assert_eq!(q.conj(), q);
    }

    #[test]
    fn a_root_of_unity_has_unit_modulus() {
        // this is the shape of the state-normalisation test
        for n in [8, 16, 24] {
            for k in 0..n as i64 {
                assert!(
                    Cyclo::zeta_pow(n, k).norm_sq().is_one(),
                    "|zeta_{n}^{k}|^2 should be 1"
                );
            }
        }
    }

    #[test]
    fn a_bell_state_is_normalized_exactly() {
        // (|00> + |11>) * isq2 has coefficients isq2 and isq2; the sum of
        // squared moduli must be exactly one, decided in K and not numerically
        let n = 8;
        let c = Cyclo::isqrt2(n).unwrap();
        let total = c.norm_sq().add(&c.norm_sq());
        assert!(total.is_one());
        assert_eq!(total.as_rational(), Some(Frac::one()));
    }

    #[test]
    fn restriction_undoes_embedding_and_refuses_what_the_subfield_lacks() {
        for n in [8u32, 16, 24] {
            let a = Cyclo::zeta(n).add(&Cyclo::isqrt2(n).unwrap()).add(&Cyclo::from_int(n, 3));
            assert_eq!(a.embed(48).unwrap().restrict(n), Some(a));
        }
        // ζ16 is not in Q(ζ8), and ζ3 not in Q(ζ16)
        assert_eq!(Cyclo::zeta(16).restrict(8), None);
        assert_eq!(Cyclo::zeta(24).embed(48).unwrap().restrict(16), None);
        // i is in every one of them
        let i = Cyclo::imaginary_unit(48).unwrap();
        assert_eq!(i.restrict(8), Cyclo::imaginary_unit(8));
    }

    #[test]
    fn an_argument_in_the_phase_group_is_found_exactly() {
        assert_eq!(Cyclo::zeta_pow(8, 3).arg(), Some(3));
        assert_eq!(Cyclo::from_int(8, -2).arg(), Some(4));
        let one_plus_i = Cyclo::one(8).add(&Cyclo::imaginary_unit(8).unwrap());
        assert_eq!(one_plus_i.arg(), Some(1));
        assert_eq!(Cyclo::sqrt2(8).unwrap().arg(), Some(0));
        // 1 + 2i is at no multiple of π/4
        let off = Cyclo::one(8).add(&Cyclo::imaginary_unit(8).unwrap().scale(&Frac::from_int(2)));
        assert_eq!(off.arg(), None);
        assert_eq!(Cyclo::zero(8).arg(), None);
    }

    #[test]
    fn the_ring_of_the_standard_gates_is_told_apart() {
        let s = Cyclo::sqrt2(8).unwrap();
        assert_eq!(s.mul(&s).as_rational(), Some(Frac::from_int(2)));
        assert!(s.is_integral());
        assert!(!Cyclo::isqrt2(8).unwrap().is_integral());
        assert_eq!(Cyclo::isqrt2(8).unwrap().odd_denominator(), BigInt::one());
        assert_eq!(Cyclo::from_frac(8, Frac::new(1, 6)).odd_denominator(), BigInt::from(3));
    }

    #[test]
    fn embedding_preserves_value_and_arithmetic() {
        let a = Cyclo::zeta(8).add(&Cyclo::from_int(8, 2));
        let b = a.embed(24).unwrap();
        assert_eq!(b.conductor(), 24);
        let (ar, ai) = a.to_complex_lossy();
        let (br, bi) = b.to_complex_lossy();
        assert!(approx(ar, br) && approx(ai, bi));
        // arithmetic agrees across the embedding
        assert_eq!(a.mul(&a).embed(24).unwrap(), b.mul(&b));
    }

    #[test]
    fn mixed_conductors_widen_to_the_least_common_multiple() {
        // a judgement crossing a boundary between conductors M
        // and N is checked at their least common multiple
        let a = Cyclo::zeta(8);
        let b = Cyclo::zeta(12);
        let sum = a.add(&b);
        assert_eq!(sum.conductor(), 24);
        let (re, im) = sum.to_complex_lossy();
        let (ar, ai) = a.to_complex_lossy();
        let (br, bi) = b.to_complex_lossy();
        assert!(approx(re, ar + br) && approx(im, ai + bi));
    }

    #[test]
    fn thirds_survive_because_k_is_a_field() {
        // reflections about presented spans
        // generate denominators such as 1/3 that no ring of roots of unity
        // absorbs, which is why the widening to a field is deliberate
        let n = 8;
        let third = Cyclo::from_frac(n, Frac::new(1, 3));
        let sum = third.add(&third).add(&third);
        assert!(sum.is_one(), "three thirds must be exactly one");
        assert_eq!(
            Cyclo::one(n).div(&Cyclo::from_int(n, 3)),
            Some(third.clone())
        );
    }

    #[test]
    fn inversion_of_a_general_element_is_exact() {
        let n = 8;
        for e in [
            Cyclo::zeta(n).add(&Cyclo::one(n)),
            Cyclo::isqrt2(n).unwrap(),
            Cyclo::from_frac(n, Frac::new(-3, 7)),
            Cyclo::zeta_pow(n, 3).sub(&Cyclo::from_int(n, 5)),
        ] {
            let inv = e.inv().expect("non-zero elements are units");
            assert!(e.mul(&inv).is_one(), "{e:?} times its inverse");
        }
        assert!(Cyclo::zero(n).inv().is_none());
        assert!(Cyclo::zero(n).div(&Cyclo::zero(n)).is_none());
    }

    #[test]
    fn equality_is_coefficient_comparison_in_the_reduced_basis() {
        // at N=8, zeta^4 = -1, so a raw
        // fourth power must reduce rather than occupy a fifth coefficient
        let n = 8;
        let z4 = Cyclo::zeta_pow(n, 4);
        assert_eq!(z4, Cyclo::from_int(n, -1));
        assert_eq!(z4.degree(), 4);
        assert_eq!(Cyclo::zeta(n).pow(4), Cyclo::from_int(n, -1));
    }

    #[test]
    fn powers_use_repeated_squaring_correctly() {
        let n = 8;
        let base = Cyclo::zeta(n).add(&Cyclo::from_int(n, 1));
        let mut expected = Cyclo::one(n);
        for k in 0..12u32 {
            assert_eq!(base.pow(k), expected, "power {k}");
            expected = expected.mul(&base);
        }
    }

    #[test]
    fn rational_elements_are_recognized() {
        let n = 8;
        assert_eq!(Cyclo::from_int(n, 5).as_rational(), Some(Frac::from_int(5)));
        assert_eq!(Cyclo::zeta(n).as_rational(), None);
        assert!(Cyclo::zero(n).is_zero());
        assert!(Cyclo::one(n).is_one());
    }

    #[test]
    fn display_is_readable() {
        let n = 8;
        assert_eq!(Cyclo::zero(n).to_string(), "0");
        assert_eq!(Cyclo::one(n).to_string(), "1");
        assert_eq!(Cyclo::zeta(n).to_string(), "1*z8");
        assert_eq!(Cyclo::zeta_pow(n, 2).to_string(), "1*z8^2");
        assert!(format!("{:?}", Cyclo::one(n)).starts_with("cyclo<8>"));
    }

    #[test]
    #[should_panic(expected = "index must be positive")]
    fn a_zero_conductor_is_rejected() {
        let _ = Cyclo::zero(0);
    }
}
