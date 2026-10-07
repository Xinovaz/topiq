//! Circuits: what a quantum unit's operators become.
//!
//! [`ir`] defines a circuit: operations on numbered qubits, with classical
//! bits that measurements write and classical expressions for what is known
//! only when it runs. [`qasm`] writes one as OpenQASM 3. [`action`] computes
//! exactly what operations do to a state, to decide what can be undone.
//! [`synth`] makes the circuit that prepares a state, exactly.
//! [`document`] writes an entry circuit as the TCON document it travels as,
//! with its judgement. Circuit generation, which makes
//! circuits, is translation phase 9, in [`crate::lower`].

pub mod action;
pub mod document;
pub mod ir;
pub mod qasm;
pub mod synth;

pub use ir::{Angle, Bit, CExpr, Circuit, Control, GateOp, Matrix, Op, Param, Slot, Var, Wire};
