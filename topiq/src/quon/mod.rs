//! QUON: the notation for writing quantum states down.
//!
//! Like [TCON](crate::tcon), QUON is native syntax rather than a second
//! language embedded in strings: a document is parsed by this compiler and
//! checked while the program is translated.
//!
//! A QUON value at run time is a *classical description* of a quantum state
//! (a list of amplitudes over basis states), not the state itself. Nothing here
//! hands a program a quantum state it can read back, which the rest of the
//! language forbids.
//!
//! [`parse`] holds the grammar and [`ket`] the reassembly of kets from
//! ordinary tokens. [`eval`] computes a state's amplitudes exactly, at the
//! unit's conductor, which is what `prep` prepares.

pub mod ast;
pub mod eval;
pub mod ket;
pub mod parse;

pub use ket::{NotAKet, reconstruct};
pub use ast::{
    Exact, Ket, Pauli, QContext, QCover, QDoc, QExpr, QFactor, QForm, QGauge, QItem, QOp, QState,
    QTerm, QType, Sign,
};
