//! Certificates, and the contract classes read off them.
//!
//! The **certificate** of an operator U from a base type (σ, ς) to another
//! (σ', ς') is the pair [f; θ]: the base map f carrying each point s of σ to
//! the point of σ' that U·ς(s) is a multiple of, and the schedule θ(s), the
//! phase of that multiple against ς'(f(s)). Both are derived exactly, by
//! applying the operator to each point and testing proportionality and
//! argument in the field; nothing is estimated.
//!
//! With the representatives chosen by fiducials ω and ω', θ(s) is the
//! argument of ⟨t|U s⟩·⟨s|ω⟩·⟨ω'|t⟩ for the literal points s and t = f(s),
//! which is what ⟨ς'(t)|U ς(s)⟩ is up to a positive factor. It must be a
//! whole number of N-th parts of a turn; when it is not, the derivation has
//! left the checked fragment, and says so rather than approximating.
//!
//! Over a subspace cover, which no single convention reaches, an operator
//! has a schedule only when it is stationary there: a scalar ϑ on the whole
//! subspace, which needs no gauge ([`Certificate::Scalar`]).
//!
//! # Classes
//!
//! A contract class is a predicate on a certificate ([`satisfies`]):
//! `rigid` is θ ≡ 0 and `flat(c)` θ ≡ c, both against a gauge; `locflat` is
//! θ constant on each connected component of the gauge's nerve; `cyc(θ̂)` is
//! θ = θ̂ + c; `stat(ϑ)` is f = id with θ ≡ ϑ, which needs no gauge; and
//! `free` constrains nothing. [`strongest`] reports the strongest class a
//! certificate satisfies, for `@judgment`.

use std::fmt;

use super::cover::{Cover, Gauge};
use super::linalg::{inner, same_point};
use crate::exact::{Cyclo, Phase};
use crate::quon::eval::Amplitudes;

/// A base type: a cover, and a gauge on it.
#[derive(Clone, PartialEq, Debug)]
pub struct Base {
    /// The cover.
    pub cover: Cover,
    /// The gauge.
    pub gauge: Gauge,
}

impl Base {
    /// This base type with its states in the field of conductor `n`, as
    /// [`Cover::at`] carries a cover's.
    pub fn at(&self, n: u32) -> Base {
        Base {
            cover: self.cover.at(n),
            gauge: self.gauge.at(n),
        }
    }
}

/// A certificate [f; θ].
#[derive(Clone, PartialEq, Debug)]
pub enum Certificate {
    /// Over a finite cover: for each point, in order, the index of its image
    /// among the target's points and its phase. `stationary` when every
    /// point is carried to itself.
    Points {
        /// f, by index.
        images: Vec<usize>,
        /// θ.
        schedule: Vec<Phase>,
        /// Whether f is the identity.
        stationary: bool,
    },
    /// Over a subspace: the operator is the scalar e^(iϑ) there.
    Scalar(Phase),
    /// Over a subspace: the operator carries it into the target, but not by
    /// a scalar, so no schedule can be written.
    Moving,
}

/// Why no certificate is derived.
#[derive(Clone, PartialEq, Debug)]
pub enum Fail {
    /// A point is carried out of the target cover: the point and its image.
    Leaves(Amplitudes, Amplitudes),
    /// A phase is no whole number of N-th parts of a turn: the point.
    Outside(Amplitudes),
    /// The cover is a code too wide to write out, which exact action
    /// cannot take.
    Undecided,
    /// What the operator makes of the source code, which is described, is
    /// not fixed by this generator of the target code.
    Unfixed(String, String),
    /// The operator carries the point of the source code, which is
    /// described, to none of the target's points.
    Unreached(String),
    /// The phase the operator gives the source code, which is described, is
    /// no whole number of N-th parts of a turn.
    OutsideCode(String),
}

/// The phase of `z`.
///
/// `z` may be in a smaller field than the conductor `n`'s, as an amplitude
/// of a unit of lower conductor is; its argument is read in `n`'s field, so
/// that a part of a turn keeps its size.
pub(crate) fn phase_of(z: &Cyclo, n: u32) -> Option<Phase> {
    let z = match z.embed(n) {
        Some(w) => w,
        None => z.restrict(n)?,
    };
    z.arg().map(|m| Phase::of(n, i64::from(m)))
}

/// The certificate of the operator `u` from `source` to `target`, at
/// conductor `n`.
///
/// # Errors
///
/// Why there is none.
pub fn derive(
    u: &mut dyn FnMut(&Amplitudes) -> Amplitudes,
    source: &Base,
    target: &Base,
    n: u32,
) -> Result<Certificate, Fail> {
    if let Some(points) = source.cover.points() {
        return points_certificate(u, &points, source, target, n);
    }
    let basis = source.cover.basis().ok_or(Fail::Undecided)?;
    let mut scalar: Option<Cyclo> = None;
    let mut stationary = true;
    for b in &basis {
        let v = u(b);
        if !target.cover.contains(&v) {
            return Err(Fail::Leaves(b.clone(), v));
        }
        if !same_point(&v, b) {
            stationary = false;
            continue;
        }
        // v = λ·b, λ = ⟨b|v⟩ / ⟨b|b⟩, whose phase is that of ⟨b|v⟩
        let lambda = inner(b, &v).div(&inner(b, b)).expect("a basis vector is not zero");
        match &scalar {
            None => scalar = Some(lambda),
            Some(s) if *s != lambda => stationary = false,
            Some(_) => {}
        }
    }
    if !stationary {
        return Ok(Certificate::Moving);
    }
    let lambda = scalar.expect("a subspace has a basis");
    phase_of(&lambda, n)
        .map(Certificate::Scalar)
        .ok_or_else(|| Fail::Outside(basis[0].clone()))
}

fn points_certificate(
    u: &mut dyn FnMut(&Amplitudes) -> Amplitudes,
    points: &[Amplitudes],
    source: &Base,
    target: &Base,
    n: u32,
) -> Result<Certificate, Fail> {
    let targets = target.cover.points();
    let mut images = Vec::with_capacity(points.len());
    let mut schedule = Vec::with_capacity(points.len());
    let mut stationary = true;
    for (i, s) in points.iter().enumerate() {
        let v = u(s);
        let (index, t) = match &targets {
            Some(ts) => match ts.iter().position(|t| same_point(&v, t)) {
                Some(j) => (j, ts[j].clone()),
                None => return Err(Fail::Leaves(s.clone(), v)),
            },
            None => {
                if !target.cover.contains(&v) || !same_point(&v, s) {
                    return Err(Fail::Leaves(s.clone(), v));
                }
                (i, s.clone())
            }
        };
        if !same_point(&t, s) {
            stationary = false;
        }
        let omega = source.gauge.fiducials.iter().find(|w| !inner(w, s).is_zero());
        let omega_t = target.gauge.fiducials.iter().find(|w| !inner(w, &t).is_zero());
        let z = match (omega, omega_t) {
            (Some(w), Some(w2)) => inner(&t, &v).mul(&inner(s, w)).mul(&inner(w2, &t)),
            // without a gauge at one end, a phase is defined only where the
            // point is carried to itself, and needs no gauge there
            _ if same_point(&v, s) => inner(s, &v),
            _ => return Err(Fail::Outside(s.clone())),
        };
        images.push(index);
        schedule.push(phase_of(&z, n).ok_or_else(|| Fail::Outside(s.clone()))?);
    }
    Ok(Certificate::Points {
        images,
        schedule,
        stationary,
    })
}

/// A contract class.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Class {
    /// θ ≡ 0.
    Rigid,
    /// θ ≡ c; with `None`, some constant.
    Flat(Option<Phase>),
    /// θ constant on each connected component of the gauge's nerve.
    Locflat,
    /// θ = θ̂ + c for the profile θ̂; with `None`, any schedule.
    Cyc(Option<Vec<Phase>>),
    /// f = id and θ ≡ ϑ; with `None`, some constant.
    Stat(Option<Phase>),
    /// Each sector stationary, with the phases given; with `None`, any.
    Sector(Option<Vec<Phase>>),
    /// No constraint.
    Free,
}

impl Class {
    /// Whether satisfying this class consults a phase convention.
    pub fn needs_gauge(&self) -> bool {
        matches!(self, Class::Rigid | Class::Flat(_) | Class::Locflat | Class::Cyc(_))
    }

    /// The class with its phases in `Θ_n`, which holds them when their
    /// order divides `n`; a phase it does not hold is kept as it is.
    pub fn at(&self, n: u32) -> Class {
        let one = |p: &Option<Phase>| p.map(|p| p.embed(n).unwrap_or(p));
        let all = |ps: &Option<Vec<Phase>>| ps.as_ref().map(|ps| ps.iter().map(|p| p.embed(n).unwrap_or(*p)).collect());
        match self {
            Class::Flat(p) => Class::Flat(one(p)),
            Class::Stat(p) => Class::Stat(one(p)),
            Class::Cyc(ps) => Class::Cyc(all(ps)),
            Class::Sector(ps) => Class::Sector(all(ps)),
            c => c.clone(),
        }
    }
}

/// A phase as a multiple of π.
pub fn show_phase(p: Phase) -> String {
    let (m, n) = (u64::from(p.index()) * 2, u64::from(p.order()));
    if m == 0 {
        return "0".to_owned();
    }
    let g = num_integer::gcd(m, n);
    let (a, b) = (m / g, n / g);
    match (a, b) {
        (1, 1) => "π".to_owned(),
        (a, 1) => format!("{a}π"),
        (1, b) => format!("π/{b}"),
        (a, b) => format!("{a}π/{b}"),
    }
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Class::Rigid => write!(f, "rigid"),
            Class::Flat(None) => write!(f, "flat"),
            Class::Flat(Some(c)) => write!(f, "flat({})", show_phase(*c)),
            Class::Locflat => write!(f, "locflat"),
            Class::Cyc(None) => write!(f, "cyc"),
            Class::Cyc(Some(p)) => write!(f, "cyc{}", show_schedule(p)),
            Class::Stat(None) => write!(f, "stat"),
            Class::Stat(Some(c)) => write!(f, "stat({})", show_phase(*c)),
            Class::Sector(None) => write!(f, "sector"),
            Class::Sector(Some(p)) => write!(f, "sector{}", show_schedule(p)),
            Class::Free => write!(f, "free"),
        }
    }
}

/// A schedule as it would be written (e.g. `(0, π/4)`).
pub fn show_schedule(s: &[Phase]) -> String {
    format!("({})", s.iter().map(|&p| show_phase(p)).collect::<Vec<_>>().join(", "))
}

/// The one value every element of `s` has.
pub(crate) fn constant(s: &[Phase]) -> Option<Phase> {
    let first = *s.first()?;
    s.iter().all(|&p| p == first).then_some(first)
}

/// The connected component of the gauge's nerve each point's first patch
/// lies in, for `locflat`.
fn components(source: &Base) -> Option<Vec<usize>> {
    let points = source.cover.points()?;
    let simplices = super::cover::nerve(&source.cover, &source.gauge);
    let patches = source.gauge.fiducials.len();
    let mut parent: Vec<usize> = (0..patches).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        if p[i] == i { i } else { let r = find(p, p[i]); p[i] = r; r }
    }
    for s in simplices.iter().filter(|s| s.len() >= 2) {
        let a = find(&mut parent, s[0]);
        for &j in &s[1..] {
            let b = find(&mut parent, j);
            parent[b] = a;
        }
    }
    points
        .iter()
        .map(|p| {
            let j = source.gauge.fiducials.iter().position(|w| !inner(w, p).is_zero())?;
            Some(find(&mut parent, j))
        })
        .collect()
}

/// Whether the certificate `c`, from `source`, satisfies `class`.
pub fn satisfies(c: &Certificate, source: &Base, class: &Class) -> bool {
    let gauged = !source.gauge.fiducials.is_empty();
    match (c, class) {
        (_, Class::Free) => true,
        (_, Class::Sector(_)) => false,
        (Certificate::Moving, _) => false,
        (Certificate::Scalar(v), Class::Stat(want)) => want.is_none_or(|w| w == *v),
        (Certificate::Scalar(v), Class::Rigid) => gauged && v.is_zero(),
        (Certificate::Scalar(v), Class::Flat(want)) => gauged && want.is_none_or(|w| w == *v),
        (Certificate::Scalar(_), Class::Locflat | Class::Cyc(None)) => gauged,
        (Certificate::Scalar(v), Class::Cyc(Some(p))) => gauged && p.len() == 1 && p[0] == *v,
        (Certificate::Points { schedule, stationary, .. }, class) => match class {
            Class::Stat(want) => *stationary && constant(schedule).is_some_and(|c| want.is_none_or(|w| w == c)),
            _ if !gauged => false,
            Class::Rigid => schedule.iter().all(|p| p.is_zero()),
            Class::Flat(want) => constant(schedule).is_some_and(|c| want.is_none_or(|w| w == c)),
            Class::Locflat => match components(source) {
                Some(comp) => {
                    let mut seen: std::collections::HashMap<usize, Phase> = std::collections::HashMap::new();
                    comp.iter().zip(schedule).all(|(k, &p)| *seen.entry(*k).or_insert(p) == p)
                }
                None => false,
            },
            Class::Cyc(None) => true,
            Class::Cyc(Some(profile)) => {
                profile.len() == schedule.len()
                    && constant(&schedule.iter().zip(profile).map(|(&a, &b)| a.sub(b)).collect::<Vec<_>>()).is_some()
            }
            Class::Free | Class::Sector(_) => unreachable!("handled above"),
        },
    }
}

/// The strongest class `c` satisfies.
pub fn strongest(c: &Certificate, source: &Base) -> Class {
    match c {
        Certificate::Scalar(v) => Class::Stat(Some(*v)),
        Certificate::Moving => Class::Free,
        Certificate::Points { schedule, stationary, .. } => {
            let gauged = !source.gauge.fiducials.is_empty();
            match constant(schedule) {
                Some(v) if *stationary => Class::Stat(Some(v)),
                _ if !gauged => Class::Free,
                Some(v) if v.is_zero() => Class::Rigid,
                Some(v) => Class::Flat(Some(v)),
                None if satisfies(c, source, &Class::Locflat) && source.gauge.fiducials.len() > 1 => Class::Locflat,
                None => Class::Cyc(Some(schedule.clone())),
            }
        }
    }
}

/// The certificate of `second` after `first`, by the cocycle law: the
/// composite base map, and θ(s) = θ₂(f₁(s)) + θ₁(s). `None` when either has
/// no finite schedule.
pub fn compose(first: &Certificate, second: &Certificate) -> Option<Certificate> {
    match (first, second) {
        (
            Certificate::Points { images: f1, schedule: t1, stationary: s1 },
            Certificate::Points { images: f2, schedule: t2, stationary: s2 },
        ) => Some(Certificate::Points {
            images: f1.iter().map(|&i| f2[i]).collect(),
            schedule: f1.iter().zip(t1).map(|(&i, &a)| t2[i].add(a)).collect(),
            stationary: *s1 && *s2,
        }),
        (Certificate::Scalar(a), Certificate::Scalar(b)) => Some(Certificate::Scalar(a.add(*b))),
        _ => None,
    }
}

/// Why `U ** id` is not branch-relatively `flat(k)`: two points whose
/// phases differ, or one whose phase is not `k`; and, when the two points
/// are orthogonal, that they are, which makes their entangled member a
/// witness too.
#[derive(Clone, PartialEq, Debug)]
pub struct FrameRefusal {
    /// The schedule.
    pub schedule: Vec<Phase>,
    /// The points, by index: two whose phases differ, or one alone.
    pub points: Vec<usize>,
    /// Whether the two points are orthogonal.
    pub orthogonal: bool,
}

/// The frame rule: `U ** id` holds branch-relatively at `flat(k)` exactly
/// when θ ≡ k on the whole source cover, whatever its orthogonality.
///
/// # Errors
///
/// The witness of failure.
pub fn frame(c: &Certificate, source: &Base, want: Option<Phase>) -> Result<Phase, Option<FrameRefusal>> {
    let schedule = match c {
        Certificate::Points { schedule, .. } => schedule.clone(),
        Certificate::Scalar(v) => return want.is_none_or(|w| w == *v).then_some(*v).ok_or(None),
        Certificate::Moving => return Err(None),
    };
    let points = source.cover.points().unwrap_or_default();
    if let Some(j) = (1..schedule.len()).find(|&j| schedule[j] != schedule[0]) {
        let i = (0..j)
            .find(|&i| schedule[i] != schedule[j] && super::linalg::orthogonal(&points[i], &points[j]))
            .unwrap_or(0);
        let orthogonal = super::linalg::orthogonal(&points[i], &points[j]);
        return Err(Some(FrameRefusal {
            schedule,
            points: vec![i, j],
            orthogonal,
        }));
    }
    let c0 = schedule[0];
    if want.is_some_and(|w| w != c0) {
        return Err(Some(FrameRefusal {
            schedule,
            points: vec![0],
            orthogonal: false,
        }));
    }
    Ok(c0)
}

/// The holonomy at the point `start` of a closed chain whose stages have
/// the certificates `stages`, over finite covers: the sum of every stage's
/// phase along the orbit, until it returns to `start`. `None` when a stage
/// has no finite schedule.
pub fn holonomy(stages: &[Certificate], start: usize, n: u32) -> Option<Phase> {
    Some(orbit_sums(stages, start, n)?.into_iter().fold(Phase::zero(n), Phase::add))
}

/// The stage a discrepancy between the holonomies at `a` and `b` indicts:
/// the first whose phases, summed along each point's orbit, differ between
/// them.
pub fn blame(stages: &[Certificate], a: usize, b: usize, n: u32) -> Option<usize> {
    let (x, y) = (orbit_sums(stages, a, n)?, orbit_sums(stages, b, n)?);
    (0..stages.len()).find(|&k| x[k] != y[k])
}

/// Each stage's phases summed along the orbit of the point `start`, until
/// it returns there.
fn orbit_sums(stages: &[Certificate], start: usize, n: u32) -> Option<Vec<Phase>> {
    let mut sums = vec![Phase::zero(n); stages.len()];
    let mut at = start;
    for _ in 0..=1024 {
        for (k, s) in stages.iter().enumerate() {
            let Certificate::Points { images, schedule, .. } = s else { return None };
            sums[k] = sums[k].add(schedule[at]);
            at = images[at];
        }
        if at == start {
            return Some(sums);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::action::gate_matrix;
    use crate::circuit::ir::GateOp;
    use crate::quon::eval::evaluate;
    use crate::quon::parse::testing::state_in;

    fn st(src: &str) -> Amplitudes {
        let (s, i) = state_in(src);
        evaluate(&s, 8, &i, &mut |_, _| None).unwrap()
    }

    fn base(points: &[&str], fid: &[&str]) -> Base {
        Base {
            cover: Cover::Fin(points.iter().map(|p| st(p)).collect()),
            gauge: Gauge {
                fiducials: fid.iter().map(|p| st(p)).collect(),
            },
        }
    }

    /// A one-qubit gate as an operator on states.
    fn gate(g: GateOp) -> impl FnMut(&Amplitudes) -> Amplitudes {
        let m = gate_matrix(&g, 8).unwrap();
        move |v: &Amplitudes| {
            let mut out = Amplitudes { width: 1, n: 8, terms: Default::default() };
            for r in 0..2 {
                let mut sum = Cyclo::zero(8);
                for c in 0..2 {
                    if let Some(x) = v.terms.get(&vec![c == 1]) {
                        sum = sum.add(&m.at(r, c).mul(x));
                    }
                }
                if !sum.is_zero() {
                    out.terms.insert(vec![r == 1], sum);
                }
            }
            out
        }
    }

    const PLUS: &str = "isq2 * |0> + isq2 * |1>";
    const MINUS: &str = "isq2 * |0> - isq2 * |1>";

    fn schedule(c: &Certificate) -> Vec<u32> {
        match c {
            Certificate::Points { schedule, .. } => schedule.iter().map(|p| p.index()).collect(),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_elementary_judgments_of_the_paper() {
        let t01 = base(&["|0>", "|1>"], &[PLUS]);
        // Z and T over the computational pair: the identity map, (0, π) and (0, π/4)
        let z = derive(&mut gate(GateOp::Z), &t01, &t01, 8).unwrap();
        assert_eq!(schedule(&z), [0, 4]);
        assert_eq!(strongest(&z, &t01), Class::Cyc(Some(vec![Phase::of(8, 0), Phase::of(8, 4)])));
        let t = derive(&mut gate(GateOp::T), &t01, &t01, 8).unwrap();
        assert_eq!(schedule(&t), [0, 1]);
        // X swaps the points and is rigid over the pair, with |+> as fiducial
        let x = derive(&mut gate(GateOp::X), &t01, &t01, 8).unwrap();
        assert!(matches!(&x, Certificate::Points { images, .. } if *images == [1, 0]));
        assert_eq!(strongest(&x, &t01), Class::Rigid);
        // over the Hadamard pair with fiducial |0>, X is (0, π) and stationary
        let tpm = base(&[PLUS, MINUS], &["|0>"]);
        let x2 = derive(&mut gate(GateOp::X), &tpm, &tpm, 8).unwrap();
        assert_eq!(schedule(&x2), [0, 4]);
        assert!(satisfies(&x2, &tpm, &Class::Cyc(None)));
        assert!(!satisfies(&x2, &tpm, &Class::Flat(None)));
        // H carries the computational pair to the Hadamard pair, rigidly
        let h = derive(&mut gate(GateOp::H), &t01, &tpm, 8).unwrap();
        assert_eq!(schedule(&h), [0, 0]);
    }

    #[test]
    fn a_phase_keeps_its_size_at_a_higher_conductor() {
        // -1 is half a turn: 4 eighths, and 8 sixteenths
        let minus_one = Cyclo::from_int(8, -1);
        assert_eq!(phase_of(&minus_one, 8), Some(Phase::of(8, 4)));
        assert_eq!(phase_of(&minus_one, 16), Some(Phase::of(16, 8)));
        // i, a quarter turn written at 16, read at 8
        assert_eq!(phase_of(&Cyclo::zeta_pow(16, 4), 8), Some(Phase::of(8, 2)));
        // the sixteenth of a turn is no eighth
        assert_eq!(phase_of(&Cyclo::zeta_pow(16, 1), 8), None);
    }

    #[test]
    fn the_t_refusal_has_both_witnesses() {
        let t01 = base(&["|0>", "|1>"], &[PLUS]);
        let t = derive(&mut gate(GateOp::T), &t01, &t01, 8).unwrap();
        let why = frame(&t, &t01, Some(Phase::zero(8))).unwrap_err().unwrap();
        assert_eq!(why.points, [0, 1]);
        assert!(why.orthogonal);
        assert_eq!(show_schedule(&why.schedule), "(0, π/4)");
        let z_twice = compose(&t, &t).unwrap();
        assert_eq!(schedule(&z_twice), [0, 2]);
    }

    #[test]
    fn holonomy_and_blame_in_a_three_stage_pipeline() {
        let t01 = base(&["|0>", "|1>"], &[PLUS]);
        let tpm = base(&[PLUS, MINUS], &["|0>"]);
        let stages = [
            derive(&mut gate(GateOp::H), &t01, &tpm, 8).unwrap(),
            derive(&mut gate(GateOp::X), &tpm, &tpm, 8).unwrap(),
            derive(&mut gate(GateOp::H), &tpm, &t01, 8).unwrap(),
        ];
        assert_eq!(holonomy(&stages, 0, 8), Some(Phase::zero(8)));
        assert_eq!(holonomy(&stages, 1, 8), Some(Phase::of(8, 4)));
        assert_eq!(blame(&stages, 0, 1, 8), Some(1));
        // re-gauging the endpoints with |-> leaves the holonomies fixed
        let t01m = base(&["|0>", "|1>"], &[MINUS]);
        let again = [
            derive(&mut gate(GateOp::H), &t01m, &tpm, 8).unwrap(),
            derive(&mut gate(GateOp::X), &tpm, &tpm, 8).unwrap(),
            derive(&mut gate(GateOp::H), &tpm, &t01m, 8).unwrap(),
        ];
        assert_eq!(holonomy(&again, 0, 8), Some(Phase::zero(8)));
        assert_eq!(holonomy(&again, 1, 8), Some(Phase::of(8, 4)));
    }

    #[test]
    fn a_cover_without_orthogonal_pairs_still_refuses_flat() {
        // the paper's three points |0>, |+> and (|0> + i|1>)/√2, fiducial |0>,
        // and the unitary taking |0> to |+> and |1> to (−i|0> + i|1>)/√2
        let c = base(&["|0>", PLUS, "isq2 * |0> + isq2 * i * |1>"], &["|0>"]);
        let u = |v: &Amplitudes| {
            let s = Cyclo::isqrt2(8).unwrap();
            let i = Cyclo::imaginary_unit(8).unwrap();
            let a = v.terms.get(&vec![false]).cloned().unwrap_or(Cyclo::zero(8));
            let b = v.terms.get(&vec![true]).cloned().unwrap_or(Cyclo::zero(8));
            let r0 = s.mul(&a).sub(&s.mul(&i).mul(&b));
            let r1 = s.mul(&a).add(&s.mul(&i).mul(&b));
            let mut out = Amplitudes { width: 1, n: 8, terms: Default::default() };
            if !r0.is_zero() {
                out.terms.insert(vec![false], r0);
            }
            if !r1.is_zero() {
                out.terms.insert(vec![true], r1);
            }
            out
        };
        let mut u = u;
        let cert = derive(&mut u, &c, &c, 8).unwrap();
        assert_eq!(schedule(&cert), [0, 7, 0]);
        let why = frame(&cert, &c, None).unwrap_err().unwrap();
        assert!(!why.orthogonal);
    }
}
