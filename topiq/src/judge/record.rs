//! The judgement record: what is derived of one operator, and what a unit
//! publishes of it for the units that import it.

use super::cert::{Base, Certificate, Class};
use crate::circuit::ir::{Angle, GateOp, Op};

/// Yes, no, or not decided.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tri {
    /// Decided true.
    Yes,
    /// Decided false.
    No,
    /// Not decided: the derivation left the checked fragment.
    Unknown,
}

/// What an instrument leaves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Kernel {
    /// How many qubits it measures.
    pub measured: u32,
    /// How many it forgets.
    pub forgotten: u32,
}

/// Which fragment a derivation stayed inside.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Fragment {
    /// The conductor it was derived at.
    pub conductor: u32,
    /// The step that left the checked fragment.
    pub departure: Option<String>,
    /// Whether every gate is a Clifford gate.
    pub stabilizer: bool,
}

/// A certificate with the endpoints it is against.
#[derive(Clone, PartialEq, Debug)]
pub struct Judged {
    /// The certificate.
    pub cert: Certificate,
    /// The source.
    pub source: Base,
    /// The target.
    pub target: Base,
}

impl Judged {
    /// The source's points, which a certificate over a finite cover gives a
    /// phase and an image each, in its order; none for any other, nor for
    /// an operator on no qubits, whose one point no state names.
    pub fn points(&self) -> Vec<crate::quon::eval::Amplitudes> {
        match self.cert {
            Certificate::Points { .. } if self.source.cover.width() > 0 => self.source.cover.points().unwrap_or_default(),
            _ => Vec::new(),
        }
    }
}

/// What is derived of an operator.
#[derive(Clone, PartialEq, Debug)]
pub struct Record {
    /// Whether it forgets nothing.
    pub monic: Tri,
    /// Whether it is monic and gives back as many qubits as it takes.
    pub unitary: Tri,
    /// The strongest class its certificate satisfies; `None` when not
    /// decided.
    pub contract: Option<Class>,
    /// Its certificate, when it has one.
    pub cert: Option<Judged>,
    /// When it acts on each basis state of its first register separately,
    /// each such sector's judgement of the rest.
    pub sectors: Option<Vec<Judged>>,
    /// Its kernel, when it measures or forgets.
    pub kernel: Option<Kernel>,
    /// The fragment it stayed inside.
    pub fragment: Fragment,
    /// Why its contract is not decided, when that is not a departure from
    /// the fragment: no cover written for too many qubits, or a supplied
    /// operator, which is judged with what is supplied.
    pub undecided: Option<String>,
    /// Whether it lifts a measured value.
    pub dynamic: bool,
}

/// One claim an operator's `[expect: …]` publishes.
#[derive(Clone, PartialEq, Debug)]
pub enum Claim {
    /// `monic`.
    Monic,
    /// `unitary`.
    Unitary,
    /// A contract class.
    Contract(Class),
    /// `frame(flat(k))`, with `None` for some constant.
    Frame(Option<crate::exact::Phase>),
    /// `kernel.outcomes = T`, by the type's name.
    Outcomes(String),
}

impl Claim {
    /// Whether satisfying the claim consults a phase convention, so that
    /// it means nothing without the cover and gauge it is stated against.
    pub fn needs_gauge(&self) -> bool {
        match self {
            Claim::Contract(c) => c.needs_gauge(),
            Claim::Frame(_) => true,
            _ => false,
        }
    }

    /// Whether `r` bears the claim out, where the record alone decides it:
    /// `None` for a frame claim, which is about the operator tensored with
    /// others rather than about its record.
    pub fn holds(&self, r: &Record, outcomes: &str) -> Option<bool> {
        Some(match self {
            Claim::Monic => r.monic == Tri::Yes,
            Claim::Unitary => r.unitary == Tri::Yes,
            Claim::Contract(Class::Free) => true,
            Claim::Contract(Class::Sector(want)) => match &r.contract {
                Some(Class::Sector(Some(got))) => want.as_ref().is_none_or(|w| w == got),
                _ => false,
            },
            Claim::Contract(c) => r.cert.as_ref().is_some_and(|j| super::cert::satisfies(&j.cert, &j.source, c)),
            Claim::Frame(_) => return None,
            Claim::Outcomes(t) => r.kernel.is_some() && t == outcomes,
        })
    }
}

/// A cover or gauge declaration an operator's endpoints name.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Declared {
    /// The declaration's name.
    pub name: String,
    /// Whether it has program linkage, so that an importer may name it.
    pub program: bool,
}

/// What a unit publishes of one of its operators with program linkage: its
/// record, the claims its `[expect: …]` makes, and the declarations its
/// endpoints name, if they name any.
#[derive(Clone, PartialEq, Debug)]
pub struct Published {
    /// The operator's name.
    pub name: String,
    /// Its record.
    pub record: Record,
    /// The name of its kernel's outcome type.
    pub outcomes: String,
    /// What its `[expect: …]` claims.
    pub claims: Vec<Claim>,
    /// The cover declaration its `[cover:]` names; `None` when it is written
    /// out, or not written.
    pub cover: Option<Declared>,
    /// The gauge declaration its `[gauge:]` names, or the base type's whose
    /// name its `[cover:]` gives.
    pub gauge: Option<Declared>,
}

/// How many qubits `ops` measure and forget, however deep, a register whose
/// length is known only as the circuit runs counting once.
pub fn kernel_of(ops: &[Op]) -> Kernel {
    fn count(ops: &[Op], k: &mut Kernel) {
        for op in ops {
            match op {
                Op::Measure { .. } | Op::MeasureAll { .. } => k.measured += 1,
                Op::Forget { .. } => k.forgotten += 1,
                Op::If { then, els, .. } => {
                    count(then, k);
                    count(els, k);
                }
                Op::For { body, .. } => count(body, k),
                _ => {}
            }
        }
    }
    let mut k = Kernel { measured: 0, forgotten: 0 };
    count(ops, &mut k);
    k
}

/// Whether every gate of `ops` is a Clifford gate: the Paulis, `h`, `s`,
/// `sdag`, `swap`, the controlled bit and phase flips with one control,
/// phases of quarter turns, and global phases.
pub fn clifford(ops: &[Op]) -> Result<(), String> {
    for op in ops {
        match op {
            Op::Gate { gate, controls, .. } => {
                let ok = match gate {
                    GateOp::X | GateOp::Z => controls.len() <= 1,
                    GateOp::Y | GateOp::H | GateOp::S | GateOp::Sdg | GateOp::Swap => controls.is_empty(),
                    GateOp::Phase(Angle::Fixed(p)) => controls.is_empty() && (u64::from(p.index()) * 4).is_multiple_of(u64::from(p.order())),
                    GateOp::GPhase(Angle::Fixed(_)) => controls.is_empty(),
                    _ => false,
                };
                if !ok {
                    let control = if controls.is_empty() { String::new() } else { format!(" with {} controls", controls.len()) };
                    return Err(format!("`{}`{control}", gate_name(gate)));
                }
            }
            Op::Unitary { .. } => return Err("an `apply` of a matrix".to_owned()),
            Op::If { then, els, .. } => {
                clifford(then)?;
                clifford(els)?;
            }
            Op::For { body, .. } => clifford(body)?,
            _ => {}
        }
    }
    Ok(())
}

fn gate_name(g: &GateOp) -> &'static str {
    match g {
        GateOp::X => "x",
        GateOp::Y => "y",
        GateOp::Z => "z",
        GateOp::H => "h",
        GateOp::S => "s",
        GateOp::Sdg => "sdag",
        GateOp::T => "t",
        GateOp::Tdg => "tdag",
        GateOp::Swap => "swap",
        GateOp::Phase(_) => "rz",
        GateOp::GPhase(_) => "gphase",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::ir::{Angle, Bit, CExpr, Circuit, Var, Wire};
    use crate::exact::Phase;
    use crate::judge::cover::{Cover, Gauge};
    use crate::quon::eval::Amplitudes;

    fn plus() -> Amplitudes {
        let mut a = Amplitudes::basis(vec![false], 8);
        a.terms.insert(vec![true], crate::exact::Cyclo::one(8));
        a
    }

    fn flat(k: i64) -> Record {
        let source = Base {
            cover: Cover::Fin(vec![Amplitudes::basis(vec![false], 8), Amplitudes::basis(vec![true], 8)]),
            gauge: Gauge { fiducials: vec![plus()] },
        };
        let p = Phase::of(8, k);
        Record {
            monic: Tri::Yes,
            unitary: Tri::Yes,
            contract: Some(Class::Flat(Some(p))),
            cert: Some(Judged {
                cert: Certificate::Points {
                    images: vec![0, 1],
                    schedule: vec![p, p],
                    stationary: true,
                },
                target: source.clone(),
                source,
            }),
            sectors: None,
            kernel: None,
            fragment: Fragment {
                conductor: 8,
                departure: None,
                stabilizer: true,
            },
            undecided: None,
            dynamic: false,
        }
    }

    #[test]
    fn a_published_claim_is_checked_against_the_record_alone() {
        let r = flat(4);
        assert_eq!(Claim::Unitary.holds(&r, ""), Some(true));
        assert_eq!(Claim::Contract(Class::Flat(Some(Phase::of(8, 4)))).holds(&r, ""), Some(true));
        assert_eq!(Claim::Contract(Class::Rigid).holds(&r, ""), Some(false));
        assert_eq!(Claim::Outcomes("bool".to_owned()).holds(&r, "bool"), Some(false), "no kernel");
        assert_eq!(Claim::Frame(None).holds(&r, ""), None, "a frame claim needs more than the record");
        assert!(Claim::Contract(Class::Rigid).needs_gauge());
        assert!(!Claim::Contract(Class::Stat(None)).needs_gauge());
    }

    #[test]
    fn allocating_in_a_loop_asks_for_qubits_as_the_circuit_runs() {
        let alloc = Op::Alloc { wire: Wire(0) };
        let looped = Circuit {
            body: vec![Op::For {
                var: Var(0),
                start: CExpr::Param(0),
                end: CExpr::Param(0),
                body: vec![alloc.clone(), Op::Measure { wire: Wire(0), bit: Bit(0) }],
            }],
            ..Circuit::default()
        };
        assert!(looped.allocates_while_running());
        let once = Circuit {
            body: vec![alloc, Op::Gate { gate: GateOp::Phase(Angle::Fixed(Phase::of(8, 1))), targets: vec![Wire(0)], controls: vec![] }],
            ..Circuit::default()
        };
        assert!(!once.allocates_while_running());
    }
}
