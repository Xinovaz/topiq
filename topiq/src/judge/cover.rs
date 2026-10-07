//! Covers, gauges on them, and what their topology is.
//!
//! A **cover** is the set of states a judgement is made over: one point, a
//! finite set of points, the projective span of some basis kets, or the code
//! a stabiliser group fixes, or the part two covers share, which a
//! restriction takes ([`intersect`]). A **gauge** chooses, for each point, which of
//! its unit vectors stands for it: a fiducial ω picks the one whose overlap
//! with ω is real and positive, on the points not orthogonal to ω. A gauge of
//! several fiducials is an atlas, whose patches are the cover's points each
//! fiducial reaches, and which must between them reach every point; a gauge
//! of none declares the cover stationary-only.
//!
//! # Rules
//!
//! - **A single fiducial reaches every point** (the fiducial condition), or
//!   the gauge is refused naming a point it misses. A subspace of dimension
//!   two or more always holds a point orthogonal to any one vector, so a
//!   `span` or `code` cover of projective dimension one or more takes an
//!   atlas or `none`, never one fiducial.
//! - **An atlas reaches every point,** or it is refused naming one it
//!   misses, and it has at most one patch more than the projective dimension
//!   of the register's states, which is enough for any cover.
//! - **The nerve** of a gauge has a vertex per patch and a simplex per set of
//!   patches whose common part of the cover is not empty. On a subspace, a
//!   finite set of patches that are each non-empty always share a point, a
//!   space over an infinite field not being a finite union of proper
//!   subspaces.
//! - **The transition class** is intrinsic to the cover: trivial on a point
//!   or a finite set, over which the phase bundle is trivial, and the Chern
//!   class of the tautological bundle, −1, on a subspace of projective
//!   dimension one or more. A gauge reduces to a single patch exactly when
//!   the class is trivial.
//! - **The orthogonality graph** joins the orthogonal pairs of a finite
//!   cover's points; a subspace has infinitely many points and no such
//!   graph to list.

use super::linalg;
use super::pauli::{self, Pauli};
use super::stabilizer::{self, Projector};
use crate::quon::eval::Amplitudes;

/// The widest code cover whose code space is written out, its dimension
/// being exponential in the width. A wider code is decided through its
/// group and its projector ([`super::stabilizer`]): restrictions and gauges
/// on it at any width, and judgements over it for Clifford circuits.
pub const CODE_WIDTH: usize = 10;

/// A cover.
#[derive(Clone, PartialEq, Debug)]
pub enum Cover {
    /// One point.
    Pt(Amplitudes),
    /// Finitely many distinct points.
    Fin(Vec<Amplitudes>),
    /// The projective span of distinct basis kets.
    Span(Vec<Amplitudes>),
    /// The projective span of independent vectors, which is what a
    /// restriction of one subspace to another leaves.
    Subspace(Vec<Amplitudes>),
    /// The code fixed by commuting, independent generators, on `width`
    /// qubits at conductor `n`.
    Code {
        /// The generators.
        gens: Vec<Pauli>,
        /// The qubits.
        width: usize,
        /// The conductor.
        n: u32,
    },
}

impl Cover {
    /// How many qubits its states are on.
    pub fn width(&self) -> usize {
        match self {
            Cover::Pt(p) => p.width,
            Cover::Fin(ps) | Cover::Span(ps) | Cover::Subspace(ps) => ps.first().map_or(0, |p| p.width),
            Cover::Code { width, .. } => *width,
        }
    }

    /// Its points, when it has finitely many that can be written out: the
    /// point of a code is not, beyond [`CODE_WIDTH`] qubits.
    pub fn points(&self) -> Option<Vec<Amplitudes>> {
        match self {
            Cover::Pt(p) => Some(vec![p.clone()]),
            Cover::Fin(ps) => Some(ps.clone()),
            Cover::Span(ps) | Cover::Subspace(ps) if ps.len() == 1 => Some(ps.clone()),
            Cover::Code { gens, width, n } if gens.len() == *width && *width <= CODE_WIDTH => {
                Some(pauli::code_basis(gens, *width, *n))
            }
            _ => None,
        }
    }

    /// This cover with its states in the field of conductor `n`, where their
    /// own conductor divides `n`, so that what is computed of them there,
    /// such as a phase in `n`-th parts of a turn, is exact.
    pub fn at(&self, n: u32) -> Cover {
        let all = |ps: &[Amplitudes]| ps.iter().map(|p| p.at(n)).collect();
        match self {
            Cover::Pt(p) => Cover::Pt(p.at(n)),
            Cover::Fin(ps) => Cover::Fin(all(ps)),
            Cover::Span(ps) => Cover::Span(all(ps)),
            Cover::Subspace(ps) => Cover::Subspace(all(ps)),
            Cover::Code { gens, width, n: m } => Cover::Code {
                gens: gens.clone(),
                width: *width,
                n: if n.is_multiple_of(*m) { n } else { *m },
            },
        }
    }

    /// The projector onto a code cover.
    fn projector(&self) -> Option<Projector> {
        match self {
            Cover::Code { gens, n, .. } => Some(Projector::new(gens, *n)),
            _ => None,
        }
    }

    /// The projective dimension of a subspace cover, one less than the
    /// dimension of the subspace; `None` for a point or a finite set.
    pub fn projective_dimension(&self) -> Option<u64> {
        match self {
            Cover::Pt(_) | Cover::Fin(_) => None,
            Cover::Span(ps) | Cover::Subspace(ps) => Some(ps.len() as u64 - 1),
            Cover::Code { gens, width, .. } => Some((1u64 << (width - gens.len()).min(63)) - 1),
        }
    }

    /// A basis of a subspace cover, or `None` for a point, a finite set, or a
    /// code too wide to write out.
    pub fn basis(&self) -> Option<Vec<Amplitudes>> {
        match self {
            Cover::Span(ps) | Cover::Subspace(ps) => Some(ps.clone()),
            Cover::Code { gens, width, n } if *width <= CODE_WIDTH => Some(pauli::code_basis(gens, *width, *n)),
            _ => None,
        }
    }

    /// Whether `v` is one of the cover's points.
    pub fn contains(&self, v: &Amplitudes) -> bool {
        match self {
            Cover::Pt(p) => linalg::same_point(p, v),
            Cover::Fin(ps) => ps.iter().any(|p| linalg::same_point(p, v)),
            Cover::Span(ps) => v.terms.keys().all(|k| ps.iter().any(|p| p.terms.contains_key(k))),
            Cover::Subspace(ps) => linalg::in_span(v, ps),
            Cover::Code { gens, .. } => pauli::fixes(gens, v),
        }
    }

    /// Whether the cover shares a point with `other`.
    pub fn meets(&self, other: &Cover) -> bool {
        intersect(self, other).is_some()
    }
}

/// The common part of two covers, `None` when it is empty. Two codes share
/// the code of both their groups, and a code and a subspace what of the
/// subspace the code's projector keeps, however wide the code.
pub fn intersect(a: &Cover, b: &Cover) -> Option<Cover> {
    if let (Cover::Code { gens: g, width, n }, Cover::Code { gens: h, .. }) = (a, b) {
        return stabilizer::joint(g, h).map(|gens| Cover::Code { gens, width: *width, n: *n });
    }
    if let Some(ps) = a.points() {
        let mut kept: Vec<Amplitudes> = ps.into_iter().filter(|p| b.contains(p)).collect();
        return match kept.len() {
            0 => None,
            1 => Some(Cover::Pt(kept.remove(0))),
            _ => Some(Cover::Fin(kept)),
        };
    }
    if b.points().is_some() {
        return intersect(b, a);
    }
    // neither is written out as points: each is a subspace, or a code
    let common = match (a.projector(), b.projector(), a.basis(), b.basis()) {
        (Some(p), None, _, Some(other)) | (None, Some(p), Some(other), _) => p.within(&other),
        (_, _, Some(x), Some(y)) => linalg::intersection(&x, &y),
        _ => unreachable!("of two covers not both codes, one is a subspace written out"),
    };
    match common.len() {
        0 => None,
        1 => Some(Cover::Pt(common[0].clone())),
        _ => Some(Cover::Subspace(common)),
    }
}

/// A gauge: its fiducials, one for each patch; none for a stationary-only
/// cover.
#[derive(Clone, PartialEq, Debug)]
pub struct Gauge {
    /// The fiducials.
    pub fiducials: Vec<Amplitudes>,
}

impl Gauge {
    /// This gauge with its fiducials in the field of conductor `n`, as
    /// [`Cover::at`] carries a cover's states.
    pub fn at(&self, n: u32) -> Gauge {
        Gauge {
            fiducials: self.fiducials.iter().map(|f| f.at(n)).collect(),
        }
    }
}

/// Why a gauge is not one on a cover.
#[derive(Clone, PartialEq, Debug)]
pub enum Refusal {
    /// A fiducial on a different number of qubits: its index.
    Width(usize),
    /// One fiducial on a subspace of projective dimension one or more.
    NoPregauge(u64),
    /// A point no fiducial reaches.
    Missed(Amplitudes),
    /// More patches than the most any cover needs.
    TooManyPatches(u32),
    /// A point of a code too wide to write out no fiducial reaches: the
    /// fiducials reach only `reached` of the code's `dimension` dimensions.
    MissedCode {
        /// The dimension of the part of the code the fiducials reach.
        reached: u64,
        /// The code's dimension.
        dimension: u64,
    },
}

/// The most patches a gauge on states of `width` qubits needs.
pub fn most_patches(width: usize) -> u32 {
    let dim = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
    crate::diag::limits::gauge_patches(dim)
}

/// Whether `gauge` is a gauge on `cover`.
///
/// # Errors
///
/// Why it is not.
pub fn check(cover: &Cover, gauge: &Gauge) -> Result<(), Refusal> {
    let width = cover.width();
    if let Some(i) = gauge.fiducials.iter().position(|f| f.width != width) {
        return Err(Refusal::Width(i));
    }
    let count = gauge.fiducials.len();
    if count == 0 {
        return Ok(());
    }
    if count as u64 > u64::from(most_patches(width)) {
        return Err(Refusal::TooManyPatches(count as u32));
    }
    if let Some(points) = cover.points() {
        for p in points {
            if gauge.fiducials.iter().all(|f| linalg::orthogonal(f, &p)) {
                return Err(Refusal::Missed(p));
            }
        }
        return Ok(());
    }
    let dim = cover.projective_dimension().unwrap_or(0);
    if count == 1 && dim >= 1 {
        return Err(Refusal::NoPregauge(dim));
    }
    if let Some(basis) = cover.basis() {
        return match linalg::orthogonal_member(&basis, &gauge.fiducials) {
            Some(p) => Err(Refusal::Missed(p)),
            None => Ok(()),
        };
    }
    // a code too wide to write out: the fiducials reach every point of it
    // exactly when their projections onto it span it
    let dimension = dim.saturating_add(1);
    let p = cover.projector().expect("a cover with no basis written out is a code");
    let reached = p.reach(&gauge.fiducials) as u64;
    if reached < dimension {
        return Err(Refusal::MissedCode { reached, dimension });
    }
    Ok(())
}

/// The patches of `gauge` on `cover` that reach some point.
fn live_patches(cover: &Cover, gauge: &Gauge) -> Vec<usize> {
    (0..gauge.fiducials.len())
        .filter(|&j| {
            let f = &gauge.fiducials[j];
            match (cover.points(), cover.basis(), cover.projector()) {
                (Some(ps), _, _) => ps.iter().any(|p| !linalg::orthogonal(f, p)),
                (None, Some(b), _) => b.iter().any(|p| !linalg::orthogonal(f, p)),
                (None, None, Some(p)) => !p.between(f, f).is_zero(),
                (None, None, None) => true,
            }
        })
        .collect()
}

/// The nerve of a gauge on its cover: every set of patches whose common
/// part is not empty, each as its patches' indices in order, the sets by
/// size and then in order.
pub fn nerve(cover: &Cover, gauge: &Gauge) -> Vec<Vec<usize>> {
    let live = live_patches(cover, gauge);
    let points = cover.points();
    let mut out = Vec::new();
    for mask in 1u64..(1u64 << live.len().min(20)) {
        let set: Vec<usize> = (0..live.len()).filter(|i| mask & (1 << i) != 0).map(|i| live[i]).collect();
        let meets = match &points {
            Some(ps) => ps
                .iter()
                .any(|p| set.iter().all(|&j| !linalg::orthogonal(&gauge.fiducials[j], p))),
            None => true,
        };
        if meets {
            out.push(set);
        }
    }
    out.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    out
}

/// The transition class of a cover: 0 when its phase bundle is trivial, and
/// otherwise the Chern class of the tautological bundle, −1.
pub fn transition_class(cover: &Cover) -> i64 {
    match cover.projective_dimension() {
        Some(d) if d >= 1 => -1,
        _ => 0,
    }
}

/// The orthogonality graph of a finite cover: each orthogonal pair of its
/// points, by index; `None` for a subspace.
pub fn orthogonality_graph(cover: &Cover) -> Option<Vec<(usize, usize)>> {
    if cover.projective_dimension().is_some_and(|d| d >= 1) {
        return None;
    }
    let ps = cover.points()?;
    let mut out = Vec::new();
    for i in 0..ps.len() {
        for j in i + 1..ps.len() {
            if linalg::orthogonal(&ps[i], &ps[j]) {
                out.push((i, j));
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quon::eval::evaluate;
    use crate::quon::parse::testing::state_in;

    fn st(src: &str) -> Amplitudes {
        let (s, i) = state_in(src);
        evaluate(&s, 8, &i, &mut |_, _| None).unwrap()
    }

    fn fid(srcs: &[&str]) -> Gauge {
        Gauge {
            fiducials: srcs.iter().map(|s| st(s)).collect(),
        }
    }

    #[test]
    fn a_fiducial_must_reach_every_point() {
        let b2 = Cover::Fin(vec![st("|00>"), st("|11>")]);
        assert_eq!(check(&b2, &fid(&["isq2 * |00> + isq2 * |11>"])), Ok(()));
        assert_eq!(check(&b2, &fid(&["|00>"])), Err(Refusal::Missed(st("|11>"))));
        assert_eq!(check(&b2, &Gauge { fiducials: vec![] }), Ok(()));
        assert_eq!(check(&b2, &fid(&["|0>"])), Err(Refusal::Width(0)));
    }

    #[test]
    fn a_subspace_takes_an_atlas_or_none() {
        let um = Cover::Span(vec![st("|00>"), st("|01>"), st("|10>")]);
        assert_eq!(check(&um, &fid(&["|00>"])), Err(Refusal::NoPregauge(2)));
        // three basis fiducials cover it; two leave |10> out
        assert_eq!(check(&um, &fid(&["|00>", "|01>", "|10>"])), Ok(()));
        match check(&um, &fid(&["|00>", "|01>"])) {
            Err(Refusal::Missed(p)) => assert!(linalg::same_point(&p, &st("|10>"))),
            other => panic!("{other:?}"),
        }
        // a one-dimensional span is a point
        assert_eq!(check(&Cover::Span(vec![st("|01>")]), &fid(&["|01>"])), Ok(()));
    }

    #[test]
    fn nerves_transition_classes_and_orthogonality_graphs() {
        let sphere = Cover::Span(vec![st("|0>"), st("|1>")]);
        let two = fid(&["|0>", "|1>"]);
        assert_eq!(nerve(&sphere, &two), vec![vec![0], vec![1], vec![0, 1]]);
        assert_eq!(transition_class(&sphere), -1);
        assert!(orthogonality_graph(&sphere).is_none());
        let pair = Cover::Fin(vec![st("|0>"), st("|1>")]);
        // on the two points the two patches do not meet
        assert_eq!(nerve(&pair, &two), vec![vec![0], vec![1]]);
        assert_eq!(transition_class(&pair), 0);
        assert_eq!(orthogonality_graph(&pair), Some(vec![(0, 1)]));
    }

    #[test]
    fn covers_meet_where_they_share_a_point() {
        let marked = Cover::Pt(st("|11>"));
        let all = Cover::Span(vec![st("|00>"), st("|01>"), st("|10>"), st("|11>")]);
        let um = Cover::Span(vec![st("|00>"), st("|01>"), st("|10>")]);
        assert!(marked.meets(&all));
        assert!(!marked.meets(&um));
        assert!(um.meets(&all));
        let bell = Cover::Code {
            gens: vec![Pauli::parse(false, "XX").unwrap(), Pauli::parse(false, "ZZ").unwrap()],
            width: 2,
            n: 8,
        };
        assert!(!bell.meets(&um));
        assert!(bell.meets(&all));
    }

    #[test]
    fn a_restriction_keeps_the_common_part() {
        let low = Cover::Span(vec![st("|00>"), st("|01>")]);
        let odd = Cover::Span(vec![st("|01>"), st("|10>")]);
        match intersect(&low, &odd) {
            Some(Cover::Pt(p)) => assert!(linalg::same_point(&p, &st("|01>"))),
            other => panic!("{other:?}"),
        }
        let um = Cover::Span(vec![st("|00>"), st("|01>"), st("|10>")]);
        let three = Cover::Span(vec![st("|00>"), st("|01>"), st("|11>")]);
        assert!(matches!(intersect(&um, &three), Some(Cover::Subspace(b)) if b.len() == 2));
        assert_eq!(intersect(&Cover::Pt(st("|11>")), &um), None);
    }
}
