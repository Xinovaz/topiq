//! `phase<N>`: the phase group `Θ_N` ≅ Z/NZ.
//!
//! An element is the angle 2πm/N with exact modular arithmetic. The type
//! supports addition, negation and comparison, and `phase::<8>::of(k)` is the
//! angle 2πk/8.
//!
//! # Why this is not an angle in radians
//!
//! The judgement checker is allowed no numerical tolerance at all, and a
//! floating representation may never stand in for an exact scalar where a
//! judgement depends on it. Schedules, holonomies and contract data all live in
//! `Θ_N`, so the group itself has to be exact: comparing two schedules is
//! comparing two integers modulo N, and two integers are never *nearly* equal.
//!
//! At N = 8 this is the Clifford + T phase group, with phases in (π/4)Z.

use std::fmt;

/// An element of `Θ_N`.
///
/// The representative `m` is always reduced into `0..N`, so equality is
/// structural.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Phase {
    n: u32,
    m: u32,
}

#[allow(
    clippy::should_implement_trait,
    reason = "`add` and `sub` are checked against the group's order rather than being total"
)]
impl Phase {
    /// The angle 2πk/N.
    ///
    /// `k` is reduced modulo N, so a negative or large index is fine at
    /// translation time. At run time an out-of-range index is `RA09`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero.
    pub fn of(n: u32, k: i64) -> Phase {
        assert!(n >= 1, "the phase group order must be positive");
        Phase {
            n,
            m: k.rem_euclid(n as i64) as u32,
        }
    }

    /// The zero angle.
    pub fn zero(n: u32) -> Phase {
        Phase::of(n, 0)
    }

    /// π, which exists only when N is even.
    pub fn pi(n: u32) -> Option<Phase> {
        n.is_multiple_of(2).then(|| Phase::of(n, (n / 2) as i64))
    }

    /// The order of the group.
    pub fn order(self) -> u32 {
        self.n
    }

    /// The representative in `0..N`.
    pub fn index(self) -> u32 {
        self.m
    }

    /// Whether this is the zero angle.
    pub fn is_zero(self) -> bool {
        self.m == 0
    }

    /// Addition in `Θ_N`.
    ///
    /// # Panics
    ///
    /// Panics if the two orders differ. A judgement crossing conductors is
    /// checked at their least common multiple, so callers embed with
    /// [`Phase::embed`] first rather than mixing silently.
    pub fn add(self, other: Phase) -> Phase {
        assert_eq!(
            self.n, other.n,
            "phase groups of different order do not add; embed first"
        );
        Phase::of(self.n, (self.m + other.m) as i64)
    }

    /// Subtraction in `Θ_N`.
    ///
    /// # Panics
    ///
    /// As [`Phase::add`].
    pub fn sub(self, other: Phase) -> Phase {
        assert_eq!(
            self.n, other.n,
            "phase groups of different order do not subtract; embed first"
        );
        Phase::of(self.n, self.m as i64 - other.m as i64)
    }

    /// Negation.
    pub fn neg(self) -> Phase {
        Phase::of(self.n, -(self.m as i64))
    }

    /// Multiplication by an integer.
    pub fn scale(self, k: i64) -> Phase {
        Phase::of(self.n, self.m as i64 * k)
    }

    /// Embeds into `Θ_M`, which exists when N divides M.
    ///
    /// The inclusion `Θ_N` into `Θ_M` sends 2πm/N to 2π(m·M/N)/M.
    pub fn embed(self, m: u32) -> Option<Phase> {
        if m == 0 || !m.is_multiple_of(self.n) {
            return None;
        }
        Some(Phase::of(m, (self.m * (m / self.n)) as i64))
    }

    /// Embeds two phases into their least common multiple group.
    pub fn unify(a: Phase, b: Phase) -> (Phase, Phase) {
        if a.n == b.n {
            return (a, b);
        }
        let m = num_integer::lcm(a.n, b.n);
        (a.embed(m).expect("lcm"), b.embed(m).expect("lcm"))
    }

    /// An `f64` approximation in radians.
    ///
    /// For printing and diagnostics only; never for deciding a judgement.
    pub fn to_radians_lossy(self) -> f64 {
        2.0 * std::f64::consts::PI * (self.m as f64) / (self.n as f64)
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "phase::<{}>::of({})", self.n, self.m)
    }
}

impl fmt::Debug for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representatives_are_reduced_into_range() {
        assert_eq!(Phase::of(8, 8).index(), 0);
        assert_eq!(Phase::of(8, 9).index(), 1);
        assert_eq!(Phase::of(8, -1).index(), 7);
        assert_eq!(Phase::of(8, -9).index(), 7);
        // equality is therefore structural
        assert_eq!(Phase::of(8, 9), Phase::of(8, 1));
    }

    #[test]
    fn the_group_is_z_mod_n() {
        let n = 8;
        let a = Phase::of(n, 3);
        let b = Phase::of(n, 6);
        assert_eq!(a.add(b), Phase::of(n, 1), "3 + 6 = 9 = 1 mod 8");
        assert_eq!(a.sub(b), Phase::of(n, 5), "3 - 6 = -3 = 5 mod 8");
        assert!(a.add(a.neg()).is_zero());
        assert_eq!(a.add(Phase::zero(n)), a);
    }

    #[test]
    fn addition_is_commutative_and_associative() {
        let n = 24;
        let (a, b, c) = (Phase::of(n, 5), Phase::of(n, 11), Phase::of(n, 19));
        assert_eq!(a.add(b), b.add(a));
        assert_eq!(a.add(b).add(c), a.add(b.add(c)));
    }

    #[test]
    fn pi_is_half_the_group() {
        // a closed loop with holonomy phase::<8>::of(4),
        // which is pi
        let p = Phase::pi(8).unwrap();
        assert_eq!(p, Phase::of(8, 4));
        assert!((p.to_radians_lossy() - std::f64::consts::PI).abs() < 1e-12);
        // twice pi is zero
        assert!(p.add(p).is_zero());
        assert!(Phase::pi(7).is_none(), "an odd group has no pi");
    }

    #[test]
    fn at_conductor_eight_the_phases_are_multiples_of_a_quarter_pi() {
        // at N = 8 this is the Clifford + T phase group, with
        // phases in (pi/4)Z
        for k in 0..8i64 {
            let want = (k as f64) * std::f64::consts::FRAC_PI_4;
            assert!((Phase::of(8, k).to_radians_lossy() - want).abs() < 1e-12);
        }
    }

    #[test]
    fn scaling_repeats_addition() {
        let n = 8;
        let a = Phase::of(n, 3);
        assert_eq!(a.scale(0), Phase::zero(n));
        assert_eq!(a.scale(1), a);
        assert_eq!(a.scale(2), a.add(a));
        assert_eq!(a.scale(-1), a.neg());
        assert_eq!(a.scale(3), a.add(a).add(a));
    }

    #[test]
    fn embedding_preserves_the_angle() {
        let a = Phase::of(8, 3);
        let b = a.embed(24).unwrap();
        assert_eq!(b.order(), 24);
        assert_eq!(b.index(), 9, "3/8 = 9/24");
        assert!((a.to_radians_lossy() - b.to_radians_lossy()).abs() < 1e-12);
    }

    #[test]
    fn embedding_needs_divisibility() {
        // the conforming conductors again: 16 does not divide 24
        assert!(Phase::of(16, 1).embed(24).is_none());
        assert!(Phase::of(8, 1).embed(24).is_some());
        assert!(Phase::of(8, 1).embed(16).is_some());
    }

    #[test]
    fn unify_widens_to_the_least_common_multiple() {
        let (a, b) = Phase::unify(Phase::of(8, 1), Phase::of(12, 1));
        assert_eq!(a.order(), 24);
        assert_eq!(b.order(), 24);
        assert_eq!(a.index(), 3);
        assert_eq!(b.index(), 2);
    }

    #[test]
    #[should_panic(expected = "different order")]
    fn mixing_orders_without_embedding_is_a_programming_error() {
        // coercing silently would decide a judgement in the wrong group
        let _ = Phase::of(8, 1).add(Phase::of(16, 1));
    }

    #[test]
    #[should_panic(expected = "order must be positive")]
    fn a_zero_order_is_rejected() {
        let _ = Phase::of(0, 1);
    }

    #[test]
    fn display_matches_the_source_spelling() {
        assert_eq!(Phase::of(8, 4).to_string(), "phase::<8>::of(4)");
    }
}
