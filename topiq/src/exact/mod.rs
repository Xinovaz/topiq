//! Exact scalar arithmetic.
//!
//! The exact scalars are first-class types, used for linear algebra and for
//! judgements. Arithmetic on them is exact at arbitrary precision, both while
//! the program is translated and while it runs. Conversion to `f32`, `f64` or
//! `cplx<f64>` is always written out with `as`, and is lossy.
//!
//! | type | module | what it is |
//! |---|---|---|
//! | `frac` | [`frac`] | an arbitrary-precision rational |
//! | `cyclo<N>` | [`cyclo`] | an element of K = `Q(ζ_N)`, in the reduced power basis |
//! | `phase<N>` | [`phase`] | an element of `Θ_N` ≅ Z/NZ, the angle 2πm/N |
//!
//! `cplx<T>` has no counterpart here. Over the exact scalars, complex numbers
//! already live in `cyclo<N>`: i is `ζ_N^(N/4)`, and every conforming
//! conductor is a multiple of eight. `core`'s `cplx<T>` is a library type,
//! computed while the program runs.
//!
//! These are the values of the translation-time computations; while the
//! program runs, the same scalars are `core`'s types, written in Topiq, and
//! `sema::exact` converts between the two, so a constant computed here and a
//! value computed by the program are the same bytes.
//!
//! # The rule this module exists to make enforceable
//!
//! No judgement fact may ever be decided by comparing floating-point values,
//! and a floating representation may never be substituted for an exact scalar
//! anywhere a judgement depends on it. A judgement is a proof, and two floats
//! being close is not a proof of anything.
//!
//! Every lossy conversion here is therefore named `*_lossy` and documented as
//! being for printing and diagnostics only, so that using one where it matters
//! is a visible act rather than an accident.
//!
//! This module depends on nothing else in the crate (no spans, no diagnostics),
//! so it can be tested purely as arithmetic.

pub mod cyclo;
pub mod frac;
pub mod phase;
pub mod sign;

pub use cyclo::Cyclo;
pub use frac::Frac;
pub use phase::Phase;

/// The smallest conforming conductor that admits `required`, or `None` if there
/// is none within the implementation limit.
///
/// This is the computation behind the `EJ04` diagnostic, which names the
/// required conductor when it is computable: a QUON coefficient written at
/// order `required` is expressible at a unit conductor N exactly when
/// `required` divides N, and N must be a positive multiple of eight, so the
/// answer is the least common multiple of the three.
pub fn conductor_admitting(required: u32, current: u32) -> Option<u32> {
    if required == 0 {
        return None;
    }
    let n = num_integer::lcm(num_integer::lcm(required, current.max(1)), 8);
    (n <= crate::diag::Limit::Conductor.value()).then_some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_coefficient_already_expressible_needs_no_change() {
        // w(1, 8) in a unit of conductor 8
        assert_eq!(conductor_admitting(8, 8), Some(8));
        // w(1, 4) in a unit of conductor 8: 4 divides 8 already
        assert_eq!(conductor_admitting(4, 8), Some(8));
    }

    #[test]
    fn a_coefficient_needing_more_names_the_conductor_that_would_suffice() {
        // w(1, 12) in a unit of conductor 8 needs lcm(12, 8) = 24
        assert_eq!(conductor_admitting(12, 8), Some(24));
        // w(1, 3) in a unit of conductor 8 needs lcm(3, 8) = 24
        assert_eq!(conductor_admitting(3, 8), Some(24));
        // w(1, 16) in a unit of conductor 8 needs 16
        assert_eq!(conductor_admitting(16, 8), Some(16));
    }

    #[test]
    fn the_suggested_conductor_is_always_a_conforming_one() {
        // a conductor is a positive multiple of eight, at most 24
        for required in 1..=24u32 {
            for current in [8u32, 16, 24] {
                if let Some(n) = conductor_admitting(required, current) {
                    assert_eq!(n % 8, 0, "N={n} is not a multiple of eight");
                    assert!(n <= 24, "N={n} is above the implementation limit");
                    assert_eq!(n % required, 0, "N={n} does not admit {required}");
                    assert_eq!(n % current, 0, "N={n} does not contain the current field");
                }
            }
        }
    }

    #[test]
    fn a_coefficient_beyond_the_limit_has_no_answer() {
        // 5 requires lcm(5, 8) = 40, past the cap of 24, so `EJ04`'s
        // "when computable" does not apply and the diagnostic omits the figure
        assert_eq!(conductor_admitting(5, 8), None);
        assert_eq!(conductor_admitting(7, 8), None);
        assert_eq!(conductor_admitting(0, 8), None);
    }

    #[test]
    fn the_three_exact_types_agree_on_a_worked_value() {
        // isq2 at N=8 has squared modulus 1/2; the phase of zeta_8 is 2 pi / 8
        let s = Cyclo::isqrt2(8).unwrap();
        assert_eq!(s.norm_sq().as_rational(), Some(Frac::new(1, 2)));
        assert_eq!(Phase::of(8, 1).index(), 1);
    }
}
