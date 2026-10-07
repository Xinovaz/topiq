//! The implementation limits, and the `EM05` diagnostic that guards them.
//!
//! Each value below is read in two directions at once. It is a **minimum on
//! the compiler**: any program staying within it must translate and run. It is
//! a **maximum on a program**: one that exceeds it is rejected, by name, with
//! `EM05`.
//!
//! A program over a limit is refused, naming the limit and the value it
//! reached. It is never truncated, approximated or read some weaker way,
//! which would leave a program that only appears to work.
//!
//! # Three limits that are not arbitrary
//!
//! Most of these are engineering headroom. Three are chosen for a reason:
//!
//! - **`@matrix_of` at 8 qubits** leaves room for three-qubit gate extraction
//!   several times over, while bounding the largest constant the compiler can
//!   be asked to fold at a 256 by 256 array of exact entries.
//! - **A conductor of 24** is the smallest multiple of eight admitting √3. That
//!   makes a coefficient needing √3 a matter of re-declaring the unit's
//!   conductor rather than a flat refusal.
//! - **Gauge patches at n + 1** is mathematics, not engineering. Every cover of
//!   `ℂℙⁿ` admits a gauge with at most n + 1 patches, so this bound can never
//!   obstruct a type that is well formed in the first place.

use super::{Code, Diagnostic};
use crate::span::Span;

/// One implementation limit.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[non_exhaustive]
pub enum Limit {
    /// Nesting of blocks.
    BlockNesting,
    /// Nesting of parenthesised expressions.
    ParenNesting,
    /// How deeply one macro may expand into another.
    MacroDepth,
    /// How deeply a constant function may recurse while being evaluated during
    /// translation.
    ConstRecursion,
    /// Generic instantiation depth.
    GenericDepth,
    /// Fields in a structure.
    StructFields,
    /// Variants in an enumeration.
    EnumVariants,
    /// Qubits in a fixed register `[qubit; N]`.
    RegisterQubits,
    /// Qubits whose operator `@matrix_of` will expand into an explicit matrix.
    MatrixOfQubits,
    /// The unit's conductor.
    Conductor,
    /// Literals in a `fin` or `span` cover.
    CoverLiterals,
    /// Stages in a chain.
    ChainStages,
    /// Entries in a `qmap`.
    QmapEntries,
}

impl Limit {
    /// Every limit.
    pub const ALL: &'static [Limit] = &[
        Limit::BlockNesting,
        Limit::ParenNesting,
        Limit::MacroDepth,
        Limit::ConstRecursion,
        Limit::GenericDepth,
        Limit::StructFields,
        Limit::EnumVariants,
        Limit::RegisterQubits,
        Limit::MatrixOfQubits,
        Limit::Conductor,
        Limit::CoverLiterals,
        Limit::ChainStages,
        Limit::QmapEntries,
    ];

    /// The limit's value.
    ///
    /// The limit on a gauge's patches depends on the cover, so
    /// [`gauge_patches`] computes it instead.
    pub fn value(self) -> u32 {
        match self {
            Limit::BlockNesting => 64,
            Limit::ParenNesting => 64,
            Limit::MacroDepth => 64,
            Limit::ConstRecursion => 64,
            Limit::GenericDepth => 32,
            Limit::StructFields => 1023,
            Limit::EnumVariants => 255,
            Limit::RegisterQubits => 4096,
            Limit::MatrixOfQubits => 8,
            Limit::Conductor => 24,
            Limit::CoverLiterals => 256,
            Limit::ChainStages => 64,
            Limit::QmapEntries => 1024,
        }
    }

    /// The limit's name.
    pub fn name(self) -> &'static str {
        match self {
            Limit::BlockNesting => "nesting of blocks",
            Limit::ParenNesting => "nesting of parenthesised expressions",
            Limit::MacroDepth => "macro expansion depth",
            Limit::ConstRecursion => "constant-function recursion depth",
            Limit::GenericDepth => "generic instantiation depth",
            Limit::StructFields => "fields in a structure",
            Limit::EnumVariants => "variants in an enumeration",
            Limit::RegisterQubits => "qubits in a fixed register",
            Limit::MatrixOfQubits => "qubits for @matrix_of",
            Limit::Conductor => "conductor N",
            Limit::CoverLiterals => "literals in a fin or span cover",
            Limit::ChainStages => "stages in a chain",
            Limit::QmapEntries => "entries in a qmap",
        }
    }

    /// Whether `count` is within the limit.
    pub fn permits(self, count: u32) -> bool {
        count <= self.value()
    }

    /// Builds the `EM05` diagnostic for exceeding this limit, naming both the
    /// limit and the value the program reached.
    pub fn exceeded(self, span: Span, count: u32) -> Diagnostic {
        Diagnostic::new(Code::Em05)
            .with_message(format!(
                "too many: {} is limited to {}, and this program reaches {}",
                self.name(),
                self.value(),
                count
            ))
            .at(span)
            .with_note(format!(
                "the limit on {} is {}; every Topiq compiler accepts at least \
                 this many, so a program staying within it is portable",
                self.name(),
                self.value()
            ))
            .with_help(
                "this limit is enforced rather than approximated: a program \
                 over it is rejected, never translated under a weaker reading",
            )
    }

    /// Checks a count, returning the diagnostic if it is over.
    ///
    /// ```
    /// use topiq::diag::Limit;
    /// use topiq::span::{SourceId, Span};
    ///
    /// let s = Span::new(SourceId(0), 0, 1);
    /// assert!(Limit::EnumVariants.check(255, s).is_none());
    /// assert!(Limit::EnumVariants.check(256, s).is_some());
    /// ```
    pub fn check(self, count: u32, span: Span) -> Option<Diagnostic> {
        if self.permits(count) {
            None
        } else {
            Some(self.exceeded(span, count))
        }
    }
}

impl std::fmt::Display for Limit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The maximum number of patches in a gauge on a cover of `ℂℙⁿ`.
///
/// Every cover of `ℂℙⁿ` admits a gauge with at most n + 1 patches, so this
/// bound never obstructs a well-formed type.
pub fn gauge_patches(projective_dimension: u32) -> u32 {
    projective_dimension.saturating_add(1)
}

/// A running depth counter that reports `EM05` the first time it goes over.
///
/// Used by the nesting limits, where the interesting event is the moment the
/// depth is exceeded rather than the final depth. Reporting once keeps a deeply
/// over-nested file from producing hundreds of identical diagnostics.
#[derive(Clone, Debug)]
pub struct DepthGuard {
    limit: Limit,
    depth: u32,
    max_seen: u32,
    reported: bool,
}

impl DepthGuard {
    /// A guard for `limit`.
    pub fn new(limit: Limit) -> DepthGuard {
        DepthGuard {
            limit,
            depth: 0,
            max_seen: 0,
            reported: false,
        }
    }

    /// Enters one level.
    ///
    /// Returns the `EM05` diagnostic the first time the limit is exceeded, and
    /// `None` on every later entry, however deep.
    pub fn enter(&mut self, span: Span) -> Option<Diagnostic> {
        self.depth += 1;
        self.max_seen = self.max_seen.max(self.depth);
        if self.limit.permits(self.depth) || self.reported {
            return None;
        }
        self.reported = true;
        Some(self.limit.exceeded(span, self.depth))
    }

    /// Leaves one level. Saturates at zero, so unbalanced input cannot
    /// underflow the counter.
    pub fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// The current depth.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// The deepest level reached.
    pub fn max_depth(&self) -> u32 {
        self.max_seen
    }

    /// Whether the limit has been exceeded at any point.
    pub fn exceeded(&self) -> bool {
        self.reported
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::SourceId;

    fn span() -> Span {
        Span::new(SourceId(0), 0, 1)
    }

    #[test]
    fn the_table_holds_its_documented_values() {
        assert_eq!(Limit::BlockNesting.value(), 64);
        assert_eq!(Limit::ParenNesting.value(), 64);
        assert_eq!(Limit::MacroDepth.value(), 64);
        assert_eq!(Limit::ConstRecursion.value(), 64);
        assert_eq!(Limit::GenericDepth.value(), 32);
        assert_eq!(Limit::StructFields.value(), 1023);
        assert_eq!(Limit::EnumVariants.value(), 255);
        assert_eq!(Limit::RegisterQubits.value(), 4096);
        assert_eq!(Limit::MatrixOfQubits.value(), 8);
        assert_eq!(Limit::Conductor.value(), 24);
        assert_eq!(Limit::CoverLiterals.value(), 256);
        assert_eq!(Limit::ChainStages.value(), 64);
        assert_eq!(Limit::QmapEntries.value(), 1024);
        assert_eq!(Limit::ALL.len(), 13);
    }

    #[test]
    fn limits_are_inclusive_maxima() {
        // each value is the largest a program may use, so the value
        // itself is permitted and one more is not
        assert!(Limit::EnumVariants.permits(255));
        assert!(!Limit::EnumVariants.permits(256));
        assert!(Limit::StructFields.permits(1023));
        assert!(!Limit::StructFields.permits(1024));
    }

    #[test]
    fn the_diagnostic_names_the_limit_and_the_value_reached() {
        let d = Limit::ChainStages.exceeded(span(), 65);
        assert_eq!(d.code, Code::Em05);
        assert!(d.message.contains("stages in a chain"), "{}", d.message);
        assert!(d.message.contains("64"), "{}", d.message);
        assert!(d.message.contains("65"), "{}", d.message);
        assert_eq!(d.primary_span(), Some(span()));
        assert_eq!(d.notes.len(), 1);
    }

    #[test]
    fn check_is_silent_inside_the_limit() {
        assert!(Limit::QmapEntries.check(1024, span()).is_none());
        let d = Limit::QmapEntries.check(1025, span()).unwrap();
        assert_eq!(d.code, Code::Em05);
    }

    #[test]
    fn every_limit_has_a_distinct_name() {
        let mut names: Vec<_> = Limit::ALL.iter().map(|l| l.name()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn the_conductor_limit_admits_the_multiples_of_eight_below_it() {
        // a conductor is a positive multiple of eight, and this table
        // caps it at 24, so exactly three conductors are conforming
        let conforming: Vec<u32> = (1..=64)
            .filter(|n| n % 8 == 0 && Limit::Conductor.permits(*n))
            .collect();
        assert_eq!(conforming, vec![8, 16, 24]);
    }

    #[test]
    fn gauge_patch_bound_is_one_more_than_the_projective_dimension() {
        assert_eq!(gauge_patches(0), 1);
        assert_eq!(gauge_patches(1), 2);
        assert_eq!(gauge_patches(3), 4);
        // saturating, so a nonsense dimension cannot wrap
        assert_eq!(gauge_patches(u32::MAX), u32::MAX);
    }

    #[test]
    fn a_depth_guard_reports_once_and_only_once() {
        let mut g = DepthGuard::new(Limit::BlockNesting);
        for _ in 0..64 {
            assert!(g.enter(span()).is_none());
        }
        assert_eq!(g.depth(), 64);
        // the 65th entry is the first over the limit
        let d = g.enter(span()).expect("EM05 at depth 65");
        assert_eq!(d.code, Code::Em05);
        assert!(d.message.contains("65"), "{}", d.message);
        // deeper still, but silent
        assert!(g.enter(span()).is_none());
        assert!(g.enter(span()).is_none());
        assert!(g.exceeded());
        assert_eq!(g.max_depth(), 67);
    }

    #[test]
    fn a_depth_guard_cannot_underflow_on_unbalanced_input() {
        // the marking pass runs on possibly-unbalanced token streams, so more
        // closes than opens must not panic
        let mut g = DepthGuard::new(Limit::ParenNesting);
        g.leave();
        g.leave();
        assert_eq!(g.depth(), 0);
        assert!(g.enter(span()).is_none());
        assert_eq!(g.depth(), 1);
    }

    #[test]
    fn depth_returns_to_zero_when_balanced() {
        let mut g = DepthGuard::new(Limit::BlockNesting);
        g.enter(span());
        g.enter(span());
        g.leave();
        g.leave();
        assert_eq!(g.depth(), 0);
        assert_eq!(g.max_depth(), 2);
        assert!(!g.exceeded());
    }

    #[test]
    fn limits_display_by_name() {
        assert_eq!(Limit::MacroDepth.to_string(), "macro expansion depth");
    }
}
