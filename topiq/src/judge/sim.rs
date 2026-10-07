//! What a circuit does to a state on its inputs.
//!
//! The state is sparse, the amplitudes of the basis states it holds over all
//! of the circuit's wires, so a register of many qubits costs only what its
//! state holds. Each gate is applied as its exact matrix. A circuit that
//! measures, forgets, lifts, chooses by an outcome, loops over a bound known
//! only when it runs, or calls a supplied operator has no fixed action on
//! its inputs, and why is reported instead.

use std::collections::{BTreeMap, BTreeSet};

use crate::circuit::action::{Sparse, apply_sparse, gate_matrix};
use crate::circuit::ir::{Angle, Circuit, GateOp, Op};
use crate::quon::eval::Amplitudes;

/// Why a circuit has no fixed action on its inputs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Why {
    /// It measures, or forgets: it is an instrument.
    Instrument,
    /// Its structure or an angle depends on a value known only when it runs.
    Runtime,
    /// It applies an operator its caller supplies.
    Supplied,
}

/// What `c` does to `input`, a state of its input wires in order: the state
/// of its output wires in order.
///
/// # Errors
///
/// Why the circuit has no fixed action.
pub fn action(c: &Circuit, input: &Amplitudes) -> Result<Amplitudes, Why> {
    let wires = c.wires as usize;
    let n = input.n;
    let mut v: Sparse = BTreeMap::new();
    for (bits, amp) in &input.terms {
        let mut all = vec![false; wires];
        for (i, w) in c.inputs.iter().enumerate() {
            all[w.0 as usize] = bits[i];
        }
        v.insert(all, amp.clone());
    }
    run(&c.body, &mut v, n)?;
    let mut out = Amplitudes {
        width: c.outputs.len(),
        n,
        terms: BTreeMap::new(),
    };
    let kept: BTreeSet<usize> = c.outputs.iter().map(|w| w.0 as usize).collect();
    for (bits, amp) in v {
        // every other wire is back in |0>; generation has shown it
        debug_assert!(bits.iter().enumerate().all(|(i, &b)| !b || kept.contains(&i)));
        let o: Vec<bool> = c.outputs.iter().map(|w| bits[w.0 as usize]).collect();
        out.terms.insert(o, amp);
    }
    Ok(out)
}

/// Whether `c` has a fixed action on its inputs; why not, otherwise.
///
/// # Errors
///
/// Why it has none.
pub fn fixed(ops: &[Op]) -> Result<(), Why> {
    for op in ops {
        match op {
            Op::Measure { .. } | Op::Forget { .. } | Op::MeasureAll { .. } => return Err(Why::Instrument),
            Op::Lift { .. }
            | Op::If { .. }
            | Op::For { .. }
            | Op::Let { .. }
            | Op::Grow { .. }
            | Op::Element { .. } => return Err(Why::Runtime),
            Op::Call { .. } => return Err(Why::Supplied),
            Op::Gate {
                gate: GateOp::Phase(Angle::Runtime(_)) | GateOp::GPhase(Angle::Runtime(_)),
                ..
            } => return Err(Why::Runtime),
            _ => {}
        }
    }
    Ok(())
}

fn run(ops: &[Op], v: &mut Sparse, n: u32) -> Result<(), Why> {
    fixed(ops)?;
    for op in ops {
        match op {
            Op::Gate { gate, targets, controls } => {
                let m = gate_matrix(gate, n).ok_or(Why::Runtime)?;
                apply_sparse(v, &m, targets, controls, n);
            }
            Op::Unitary { matrix, targets, controls } => apply_sparse(v, matrix, targets, controls, n),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::circuit::ir::{Control, Wire};
    use crate::exact::{Cyclo, Phase};

    fn gate(g: GateOp, t: &[u32], c: &[(u32, bool)]) -> Op {
        Op::Gate {
            gate: g,
            targets: t.iter().map(|&w| Wire(w)).collect(),
            controls: c.iter().map(|&(w, on)| Control { wire: Wire(w), on }).collect(),
        }
    }

    #[test]
    fn a_bell_circuit_acts_on_its_inputs() {
        let c = Circuit {
            conductor: 8,
            inputs: vec![Wire(0), Wire(1)],
            outputs: vec![Wire(0), Wire(1)],
            wires: 2,
            body: vec![gate(GateOp::H, &[0], &[]), gate(GateOp::X, &[1], &[(0, true)])],
            ..Circuit::default()
        };
        let out = action(&c, &Amplitudes::basis(vec![false, false], 8)).unwrap();
        assert_eq!(out.terms.len(), 2);
        assert_eq!(out.terms[&vec![true, true]], Cyclo::isqrt2(8).unwrap());
    }

    #[test]
    fn an_instrument_has_no_fixed_action() {
        let body = vec![Op::Measure { wire: Wire(0), bit: crate::circuit::ir::Bit(0) }];
        assert_eq!(fixed(&body), Err(Why::Instrument));
        let t = gate(GateOp::Phase(Angle::Fixed(Phase::of(8, 1))), &[0], &[]);
        assert_eq!(fixed(&[t]), Ok(()));
    }
}
