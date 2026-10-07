//! What a circuit does to a state.
//!
//! A [`State`] is a vector of amplitudes in the field `Q(ζ_N)` over a few
//! named wires, and applying an operation to it is exact matrix action:
//! nothing is rounded. What cannot be applied (a measurement, a forgetting,
//! an operator a caller supplies, an angle known only when the circuit runs)
//! is reported as [`Opaque`].
//!
//! [`returns_to_zero`] uses this to decide whether a sequence of operations
//! returns some qubits to |0> whatever the other qubits they meet hold, which
//! is how an ancilla's uncomputation and the reset of a register are
//! checked.

use std::collections::{BTreeMap, BTreeSet};

use super::ir::{Angle, Control, GateOp, Matrix, Op, Wire};
use crate::exact::Cyclo;

/// An operation whose action cannot be computed now.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Opaque;

/// The matrix of a gate, its first target the most significant bit of the
/// row and column index; `None` for an angle known only when the circuit
/// runs.
pub fn gate_matrix(g: &GateOp, n: u32) -> Option<Matrix> {
    let zero = || Cyclo::zero(n);
    let one = || Cyclo::one(n);
    let i = || Cyclo::imaginary_unit(n).expect("the conductor is a multiple of eight");
    let diag = |a: Cyclo| Matrix {
        size: 2,
        entries: vec![one(), zero(), zero(), a],
    };
    // the angle's own field when it is not in `n`'s, whose arithmetic then
    // widens: a part of a turn is never rounded to none
    let zeta = |m: u32, of: u32| {
        let z = Cyclo::zeta_pow(of, i64::from(m));
        z.embed(n).unwrap_or(z)
    };
    Some(match g {
        GateOp::X => Matrix { size: 2, entries: vec![zero(), one(), one(), zero()] },
        GateOp::Y => Matrix { size: 2, entries: vec![zero(), i().neg(), i(), zero()] },
        GateOp::Z => diag(one().neg()),
        GateOp::H => {
            let s = Cyclo::isqrt2(n).expect("the conductor is a multiple of eight");
            Matrix {
                size: 2,
                entries: vec![s.clone(), s.clone(), s.clone(), s.neg()],
            }
        }
        GateOp::S => diag(i()),
        GateOp::Sdg => diag(i().neg()),
        GateOp::T => diag(Cyclo::zeta_pow(n, i64::from(n / 8))),
        GateOp::Tdg => diag(Cyclo::zeta_pow(n, -i64::from(n / 8))),
        GateOp::Swap => {
            let mut e = vec![zero(); 16];
            for (r, c) in [(0, 0), (1, 2), (2, 1), (3, 3)] {
                e[r * 4 + c] = one();
            }
            Matrix { size: 4, entries: e }
        }
        GateOp::Phase(Angle::Fixed(p)) => diag(zeta(p.index(), p.order())),
        GateOp::GPhase(Angle::Fixed(p)) => Matrix {
            size: 1,
            entries: vec![zeta(p.index(), p.order())],
        },
        GateOp::Phase(Angle::Runtime(_)) | GateOp::GPhase(Angle::Runtime(_)) => return None,
    })
}

/// An exact state of some wires: the amplitude of each basis state, the
/// first wire the most significant bit of the index.
#[derive(Clone, PartialEq, Debug)]
pub struct State {
    /// The wires.
    pub wires: Vec<Wire>,
    /// The amplitudes.
    pub amps: Vec<Cyclo>,
    n: u32,
}

impl State {
    /// The basis state whose index is `bits`.
    pub fn basis(wires: Vec<Wire>, bits: usize, n: u32) -> State {
        let mut amps = vec![Cyclo::zero(n); 1 << wires.len()];
        amps[bits] = Cyclo::one(n);
        State { wires, amps, n }
    }

    /// The bit position of a wire in the index.
    fn bit(&self, w: Wire) -> Option<usize> {
        let k = self.wires.len();
        self.wires.iter().position(|&x| x == w).map(|p| k - 1 - p)
    }

    /// Applies `op`.
    ///
    /// # Errors
    ///
    /// When the operation's action cannot be computed now, or it names a wire
    /// the state does not have.
    pub fn apply(&mut self, op: &Op) -> Result<(), Opaque> {
        match op {
            Op::Gate { gate, targets, controls } => {
                let m = gate_matrix(gate, self.n).ok_or(Opaque)?;
                self.apply_matrix(&m, targets, controls)
            }
            Op::Unitary { matrix, targets, controls } => self.apply_matrix(matrix, targets, controls),
            // allocation and release name a qubit that is |0>, and change
            // nothing
            Op::Alloc { .. } | Op::Release { .. } => Ok(()),
            _ => Err(Opaque),
        }
    }

    /// Applies `m` to `targets` where every control holds.
    fn apply_matrix(&mut self, m: &Matrix, targets: &[Wire], controls: &[Control]) -> Result<(), Opaque> {
        let tbits: Vec<usize> = targets.iter().map(|&t| self.bit(t).ok_or(Opaque)).collect::<Result<_, _>>()?;
        let cbits: Vec<(usize, bool)> = controls
            .iter()
            .map(|c| self.bit(c.wire).map(|b| (b, c.on)).ok_or(Opaque))
            .collect::<Result<_, _>>()?;
        let tmask: usize = tbits.iter().map(|b| 1 << b).sum();
        let size = 1usize << tbits.len();
        for base in 0..self.amps.len() {
            if base & tmask != 0 || !cbits.iter().all(|&(b, on)| ((base >> b) & 1 == 1) == on) {
                continue;
            }
            // the indices of the target sub-space from this base, in the
            // matrix's order: the first target most significant
            let idx: Vec<usize> = (0..size)
                .map(|j| {
                    let mut i = base;
                    for (t, &b) in tbits.iter().enumerate() {
                        if (j >> (tbits.len() - 1 - t)) & 1 == 1 {
                            i |= 1 << b;
                        }
                    }
                    i
                })
                .collect();
            let v: Vec<Cyclo> = idx.iter().map(|&i| self.amps[i].clone()).collect();
            for (r, &i) in idx.iter().enumerate() {
                let mut sum = Cyclo::zero(self.n);
                for (c, x) in v.iter().enumerate() {
                    let e = m.at(r, c);
                    if !e.is_zero() && !x.is_zero() {
                        sum = sum.add(&e.mul(x));
                    }
                }
                self.amps[i] = sum;
            }
        }
        Ok(())
    }
}

/// A sparse state over some wires: the amplitude of each basis state it
/// holds, by the state's bits in wire order.
pub type Sparse = BTreeMap<Vec<bool>, Cyclo>;

/// Applies `m` to `targets` of a sparse state, the first the most
/// significant bit of its index, where every control holds.
pub fn apply_sparse(v: &mut Sparse, m: &Matrix, targets: &[Wire], controls: &[Control], n: u32) {
    let holds = |bits: &[bool]| controls.iter().all(|c| bits[c.wire.0 as usize] == c.on);
    let t: Vec<usize> = targets.iter().map(|w| w.0 as usize).collect();
    let size = 1usize << t.len();
    let bases: BTreeSet<Vec<bool>> = v
        .keys()
        .filter(|b| holds(b))
        .map(|b| {
            let mut b = b.clone();
            for &i in &t {
                b[i] = false;
            }
            b
        })
        .collect();
    for base in bases {
        let keys: Vec<Vec<bool>> = (0..size)
            .map(|j| {
                let mut k = base.clone();
                for (p, &i) in t.iter().enumerate() {
                    k[i] = (j >> (t.len() - 1 - p)) & 1 == 1;
                }
                k
            })
            .collect();
        let x: Vec<Option<Cyclo>> = keys.iter().map(|k| v.get(k).cloned()).collect();
        for (r, k) in keys.iter().enumerate() {
            let mut sum = Cyclo::zero(n);
            for (c, xc) in x.iter().enumerate() {
                if let Some(xc) = xc {
                    let e = m.at(r, c);
                    if !e.is_zero() {
                        sum = sum.add(&e.mul(xc));
                    }
                }
            }
            if sum.is_zero() {
                v.remove(k);
            } else {
                v.insert(k.clone(), sum);
            }
        }
    }
}

/// Whether the vector `v` is `λ·s` for some scalar λ.
pub fn proportional(v: &[Cyclo], s: &[Cyclo]) -> bool {
    let Some(k) = s.iter().position(|x| !x.is_zero()) else {
        return v.iter().all(Cyclo::is_zero);
    };
    let Some(lambda) = v[k].div(&s[k]) else { return false };
    v.iter().zip(s).all(|(a, b)| {
        let (x, y) = Cyclo::unify(a, &lambda.mul(b));
        x == y
    })
}

/// The state of `wires` after `ops` from `start`, when every operation acts
/// only on them.
///
/// # Errors
///
/// When an operation names another wire or cannot be computed now.
pub fn evolve(wires: &[Wire], start: Vec<Cyclo>, ops: &[Op], n: u32) -> Result<Vec<Cyclo>, Opaque> {
    let mut s = State {
        wires: wires.to_vec(),
        amps: start,
        n,
    };
    for op in ops {
        s.apply(op)?;
    }
    Ok(s.amps)
}

/// What [`returns_to_zero`] found.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// The qubits end in |0> whatever the others held.
    Yes,
    /// For some state of the others they do not; the index is the first
    /// operation after which they could no longer be returned.
    No,
    /// It could not be decided: an operation that bears on them cannot be
    /// computed now, or too many qubits bear on them.
    Unknown,
}

/// The most qubits [`returns_to_zero`] simulates together.
pub const WIDTH: usize = 8;

/// Whether `ops`, applied with the qubits `zeroed` in |0>, leave them in |0>
/// again whatever the other qubits hold.
///
/// Only the operations that can bear on those qubits are simulated: walking
/// back from the last one to act on them, an operation bears on them when it
/// names one of them or a qubit an operation after it that bears on them
/// names. What happens afterwards to the other qubits, measurements
/// included, cannot change them. For each basis state of the other qubits
/// that bear on them, the operations are applied exactly and the result
/// checked to have nothing outside |0> on `zeroed`; by linearity that decides
/// every state of the others.
pub fn returns_to_zero(ops: &[Op], zeroed: &[Wire], n: u32) -> Verdict {
    let zero: BTreeSet<Wire> = zeroed.iter().copied().collect();
    let Some(last) = ops.iter().rposition(|op| op.wires().iter().any(|w| zero.contains(w))) else {
        return Verdict::Yes;
    };
    let mut bearing: BTreeSet<Wire> = zero.clone();
    let mut kept = Vec::new();
    for op in ops[..=last].iter().rev() {
        let ws = op.wires();
        if ws.iter().any(|w| bearing.contains(w)) {
            bearing.extend(ws);
            kept.push(op.clone());
        }
    }
    kept.reverse();
    if bearing.len() > WIDTH {
        return Verdict::Unknown;
    }
    let others: Vec<Wire> = bearing.iter().copied().filter(|w| !zero.contains(w)).collect();
    let mut wires: Vec<Wire> = zeroed.to_vec();
    wires.extend(&others);
    let k0 = zeroed.len();
    for r in 0..(1usize << others.len()) {
        let mut s = State::basis(wires.clone(), r, n);
        for op in &kept {
            if s.apply(op).is_err() {
                return Verdict::Unknown;
            }
        }
        let high = ((1usize << k0) - 1) << others.len();
        if s.amps.iter().enumerate().any(|(i, a)| i & high != 0 && !a.is_zero()) {
            return Verdict::No;
        }
    }
    Verdict::Yes
}

/// Whether `ops` change nothing: every state of the qubits they name is
/// left exactly as it was.
pub fn is_identity(ops: &[Op], n: u32) -> Verdict {
    let wires: BTreeSet<Wire> = ops.iter().flat_map(Op::wires).collect();
    if wires.len() > WIDTH {
        return Verdict::Unknown;
    }
    let wires: Vec<Wire> = wires.into_iter().collect();
    for r in 0..(1usize << wires.len()) {
        let mut s = State::basis(wires.clone(), r, n);
        for op in ops {
            if s.apply(op).is_err() {
                return Verdict::Unknown;
            }
        }
        if s != State::basis(wires.clone(), r, n) {
            return Verdict::No;
        }
    }
    Verdict::Yes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Frac;

    fn half(n: u32) -> Cyclo {
        Cyclo::from_frac(n, Frac::new(1, 2))
    }

    fn gate(g: GateOp, t: u32, controls: &[(u32, bool)]) -> Op {
        Op::Gate {
            gate: g,
            targets: vec![Wire(t)],
            controls: controls.iter().map(|&(w, on)| Control { wire: Wire(w), on }).collect(),
        }
    }

    #[test]
    fn a_hadamard_twice_is_the_identity() {
        let mut s = State::basis(vec![Wire(0)], 0, 8);
        s.apply(&gate(GateOp::H, 0, &[])).unwrap();
        assert_eq!(s.amps[0].mul(&s.amps[0]), half(8));
        s.apply(&gate(GateOp::H, 0, &[])).unwrap();
        assert!(s.amps[0].is_one() && s.amps[1].is_zero());
    }

    #[test]
    fn a_controlled_gate_acts_only_where_its_control_holds() {
        // |10> goes to |11>, the first wire being the control
        let mut s = State::basis(vec![Wire(0), Wire(1)], 0b10, 8);
        s.apply(&gate(GateOp::X, 1, &[(0, true)])).unwrap();
        assert!(s.amps[0b11].is_one());
        let mut s = State::basis(vec![Wire(0), Wire(1)], 0b00, 8);
        s.apply(&gate(GateOp::X, 1, &[(0, true)])).unwrap();
        assert!(s.amps[0b00].is_one());
    }

    #[test]
    fn computing_and_uncomputing_a_conjunction_returns_the_ancilla() {
        let ops = vec![gate(GateOp::X, 2, &[(0, true), (1, true)]), gate(GateOp::X, 3, &[(2, true)]), gate(GateOp::X, 2, &[(0, true), (1, true)])];
        assert_eq!(returns_to_zero(&ops, &[Wire(2)], 8), Verdict::Yes);
        assert_eq!(returns_to_zero(&ops[..2], &[Wire(2)], 8), Verdict::No);
    }

    #[test]
    fn a_phase_kicked_back_leaves_the_ancilla_to_be_undone_alone() {
        // a = |->; cx(q, a) kicks back a phase; h, x on a return it to |0>
        let ops = vec![
            gate(GateOp::X, 1, &[]),
            gate(GateOp::H, 1, &[]),
            gate(GateOp::H, 0, &[]),
            gate(GateOp::X, 1, &[(0, true)]),
            gate(GateOp::H, 0, &[]),
            Op::Measure { wire: Wire(0), bit: super::super::ir::Bit(0) },
            gate(GateOp::H, 1, &[]),
            gate(GateOp::X, 1, &[]),
        ];
        assert_eq!(returns_to_zero(&ops, &[Wire(1)], 8), Verdict::Yes);
    }

    #[test]
    fn an_eigenvector_is_proportional_to_its_image() {
        let n = 8;
        let s = Cyclo::isqrt2(n).unwrap();
        let minus = vec![s.clone(), s.neg()];
        let x = gate_matrix(&GateOp::X, n).unwrap();
        let image: Vec<Cyclo> = (0..2).map(|r| (0..2).fold(Cyclo::zero(n), |acc, c| acc.add(&x.at(r, c).mul(&minus[c])))).collect();
        assert!(proportional(&image, &minus));
        let plus_zero = vec![Cyclo::one(n), Cyclo::zero(n)];
        let image: Vec<Cyclo> = (0..2).map(|r| (0..2).fold(Cyclo::zero(n), |acc, c| acc.add(&x.at(r, c).mul(&plus_zero[c])))).collect();
        assert!(!proportional(&image, &plus_zero));
    }
}
