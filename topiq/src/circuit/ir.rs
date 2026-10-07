//! The circuit intermediate representation.
//!
//! A [`Circuit`] is what one operator becomes once every generation-time
//! value in it is known: a sequence of [`Op`]s on numbered wires, one wire per
//! qubit, and classical bits that measurements write. What is not known until
//! the circuit runs (a measured outcome, a classical parameter of the entry
//! operator, a value `lift` makes) is carried as a [`CExpr`], a classical
//! expression the executor evaluates.
//!
//! Control is part of every quantum operation rather than a gate of its own:
//! a gate applied under a quantum `if` carries the condition's qubits as its
//! [`Control`]s, so `cx` is `x` with one control and `ccx` is `x` with two.

use std::fmt;

use crate::exact::{Cyclo, Phase};
use crate::tir::{BinOp, IntTy, LogicalOp, UnOp, Value};

/// A qubit of the circuit.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Wire(pub u32);

/// A classical bit a measurement writes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Bit(pub u32);

/// A classical variable of the circuit: a loop index, a value `lift` makes,
/// or one computed from outcomes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Var(pub u32);

/// A qubit an operation is controlled on, and the value it is controlled on:
/// `on` is `true` when the operation applies where the qubit is |1>.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Control {
    /// The qubit.
    pub wire: Wire,
    /// Whether the operation applies on |1> (`true`) or on |0>.
    pub on: bool,
}

/// An angle: a whole number of N-th parts of a turn, known now or computed
/// when the circuit runs.
#[derive(Clone, PartialEq, Debug)]
pub enum Angle {
    /// Known now.
    Fixed(Phase),
    /// Computed when the circuit runs: the index `m` of the angle 2πm/N,
    /// for the circuit's conductor N.
    Runtime(CExpr),
}

/// A gate of the standard set.
#[derive(Clone, PartialEq, Debug)]
pub enum GateOp {
    /// The bit flip.
    X,
    /// The bit and phase flip.
    Y,
    /// The phase flip.
    Z,
    /// The Hadamard gate.
    H,
    /// `diag(1, i)`.
    S,
    /// `diag(1, -i)`.
    Sdg,
    /// `diag(1, e^(iπ/4))`.
    T,
    /// `diag(1, e^(-iπ/4))`.
    Tdg,
    /// The exchange of two qubits.
    Swap,
    /// `diag(1, e^(iθ))`.
    Phase(Angle),
    /// `e^(iθ)` on no qubit at all: a global phase, relative once controlled.
    GPhase(Angle),
}

impl GateOp {
    /// How many target qubits it acts on.
    pub fn arity(&self) -> usize {
        match self {
            GateOp::Swap => 2,
            GateOp::GPhase(_) => 0,
            _ => 1,
        }
    }

    /// Its inverse, which undoes it.
    pub fn inverse(&self) -> GateOp {
        match self {
            GateOp::S => GateOp::Sdg,
            GateOp::Sdg => GateOp::S,
            GateOp::T => GateOp::Tdg,
            GateOp::Tdg => GateOp::T,
            GateOp::Phase(a) => GateOp::Phase(a.negated()),
            GateOp::GPhase(a) => GateOp::GPhase(a.negated()),
            g => g.clone(),
        }
    }
}

impl Angle {
    /// The opposite angle.
    pub fn negated(&self) -> Angle {
        match self {
            Angle::Fixed(p) => Angle::Fixed(p.neg()),
            Angle::Runtime(e) => Angle::Runtime(CExpr::Unary(UnOp::Neg, Box::new(e.clone()))),
        }
    }
}

/// A square matrix with exact entries.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix {
    /// How many rows, and columns: `2^k` for a gate on `k` qubits.
    pub size: usize,
    /// The entries.
    pub entries: Vec<Cyclo>,
}

impl Matrix {
    /// The entry in row `r` and column `c`.
    pub fn at(&self, r: usize, c: usize) -> &Cyclo {
        &self.entries[r * self.size + c]
    }
}

/// One operation of a circuit.
#[derive(Clone, PartialEq, Debug)]
pub enum Op {
    /// A gate on `targets`.
    Gate {
        /// The gate.
        gate: GateOp,
        /// The qubits it acts on.
        targets: Vec<Wire>,
        /// The qubits it is controlled on.
        controls: Vec<Control>,
    },
    /// An exact unitary matrix applied as a gate. The first target is the
    /// most significant qubit of the matrix's row and column index.
    Unitary {
        /// The matrix.
        matrix: std::sync::Arc<Matrix>,
        /// The qubits it acts on.
        targets: Vec<Wire>,
        /// The qubits it is controlled on.
        controls: Vec<Control>,
    },
    /// Measures a qubit in the computational basis, writing what is
    /// observed to a bit. The qubit is used up.
    Measure {
        /// The qubit.
        wire: Wire,
        /// Where the outcome goes.
        bit: Bit,
    },
    /// A fresh qubit.
    Alloc {
        /// The qubit.
        wire: Wire,
    },
    /// A qubit given back.
    Release {
        /// The qubit.
        wire: Wire,
    },
    /// A qubit discarded without an outcome.
    Forget {
        /// The qubit.
        wire: Wire,
    },
    /// Operations chosen by a classical condition known when the circuit
    /// runs: feed-forward.
    If {
        /// The condition.
        cond: CExpr,
        /// What is done where it holds.
        then: Vec<Op>,
        /// What is done where it does not.
        els: Vec<Op>,
    },
    /// Operations repeated for each index from `start` up to `end`, not
    /// including it, bounds known when the circuit runs.
    For {
        /// The index.
        var: Var,
        /// The first index.
        start: CExpr,
        /// The bound.
        end: CExpr,
        /// What each iteration does.
        body: Vec<Op>,
    },
    /// Gives a variable a classical value computed from outcomes, parameters
    /// and other variables.
    Let {
        /// The variable.
        var: Var,
        /// Its value.
        value: CExpr,
    },
    /// Gives a variable an outcome's value as one the rest of the circuit's
    /// structure may depend on. A circuit containing one is dynamic.
    Lift {
        /// The variable.
        var: Var,
        /// The outcome.
        value: CExpr,
    },
    /// Adds the qubit on `wire` to the end of the register `reg`, whose
    /// length is known only as the circuit runs; `wire` no longer names it.
    Grow {
        /// The register.
        reg: u32,
        /// The qubit.
        wire: Wire,
    },
    /// Makes `wire` name the qubit at `index` of the register `reg`.
    Element {
        /// The wire.
        wire: Wire,
        /// The register.
        reg: u32,
        /// Which of its qubits, counted from 0.
        index: CExpr,
    },
    /// Measures every qubit of the register `reg`, in order, giving the
    /// array of what was observed to `var`. The qubits are used up.
    MeasureAll {
        /// The register.
        reg: u32,
        /// Where the outcomes go.
        var: Var,
    },
    /// Applies the operator a caller supplies for an operator-typed
    /// parameter of the entry operator.
    Call {
        /// Which of the circuit's slots.
        slot: u32,
        /// The qubits passed.
        args: Vec<Vec<Wire>>,
        /// The qubits the call is controlled on.
        controls: Vec<Control>,
        /// Whether the operator's adjoint is applied rather than the
        /// operator, undoing it.
        adjoint: bool,
    },
}

impl Op {
    /// Every qubit the operation names, targets and controls both.
    pub fn wires(&self) -> Vec<Wire> {
        match self {
            Op::Gate { targets, controls, .. } | Op::Unitary { targets, controls, .. } => {
                controls.iter().map(|c| c.wire).chain(targets.iter().copied()).collect()
            }
            Op::Measure { wire, .. } | Op::Alloc { wire } | Op::Release { wire } | Op::Forget { wire } => vec![*wire],
            Op::Call { args, controls, .. } => controls.iter().map(|c| c.wire).chain(args.iter().flatten().copied()).collect(),
            Op::If { then, els, .. } => then.iter().chain(els).flat_map(Op::wires).collect(),
            Op::For { body, .. } => body.iter().flat_map(Op::wires).collect(),
            Op::Grow { wire, .. } | Op::Element { wire, .. } => vec![*wire],
            Op::Let { .. } | Op::Lift { .. } | Op::MeasureAll { .. } => Vec::new(),
        }
    }

    /// The qubits the operation may change: its targets, or everything a
    /// supplied operator or a nested operation is given.
    pub fn writes(&self) -> Vec<Wire> {
        match self {
            Op::Gate { targets, .. } | Op::Unitary { targets, .. } => targets.clone(),
            Op::Call { args, .. } => args.iter().flatten().copied().collect(),
            Op::If { .. } | Op::For { .. } => self.wires(),
            Op::Measure { wire, .. } | Op::Alloc { wire } | Op::Release { wire } | Op::Forget { wire } => vec![*wire],
            Op::Grow { wire, .. } => vec![*wire],
            Op::Element { .. } | Op::Let { .. } | Op::Lift { .. } | Op::MeasureAll { .. } => Vec::new(),
        }
    }

    /// The operation that undoes this one, if one does: a gate's inverse, a
    /// matrix's conjugate transpose, a release for an allocation, and a loop
    /// whose body has one run backwards. A measurement, a forgetting, a
    /// lift, a choice by a value and a classical assignment have none.
    pub fn inverse(&self) -> Option<Op> {
        Some(match self {
            Op::Gate { gate, targets, controls } => Op::Gate {
                gate: gate.inverse(),
                targets: targets.clone(),
                controls: controls.clone(),
            },
            Op::Unitary { matrix, targets, controls } => Op::Unitary {
                matrix: std::sync::Arc::new(matrix.dagger()),
                targets: targets.clone(),
                controls: controls.clone(),
            },
            Op::Alloc { wire } => Op::Release { wire: *wire },
            Op::Release { wire } => Op::Alloc { wire: *wire },
            Op::Call { slot, args, controls, adjoint } => Op::Call {
                slot: *slot,
                args: args.clone(),
                controls: controls.clone(),
                adjoint: !adjoint,
            },
            // the loop over the same range, each round undoing the body of
            // the round it mirrors: round i sets the variable to
            // start + end − 1 − i first
            Op::For { var, start, end, body } => {
                let usize = |v: i128| Box::new(CExpr::Value(Value::Int(v, IntTy::USIZE)));
                let sum = CExpr::Binary(BinOp::Add, Box::new(start.clone()), Box::new(end.clone()));
                let last = CExpr::Binary(BinOp::Sub, Box::new(sum), usize(1));
                let mut undone = vec![Op::Let {
                    var: *var,
                    value: CExpr::Binary(BinOp::Sub, Box::new(last), Box::new(CExpr::Var(*var))),
                }];
                for op in body.iter().rev() {
                    undone.push(op.inverse()?);
                }
                Op::For {
                    var: *var,
                    start: start.clone(),
                    end: end.clone(),
                    body: undone,
                }
            }
            Op::Measure { .. }
            | Op::Forget { .. }
            | Op::Lift { .. }
            | Op::If { .. }
            | Op::Let { .. }
            | Op::Grow { .. }
            | Op::Element { .. }
            | Op::MeasureAll { .. } => {
                return None;
            }
        })
    }

    /// Whether the operation keeps what it acts on: it measures, forgets and
    /// lifts nothing, however deep.
    pub fn is_monic(&self) -> bool {
        match self {
            Op::Measure { .. } | Op::Forget { .. } | Op::Lift { .. } | Op::MeasureAll { .. } => false,
            Op::If { then, els, .. } => then.iter().chain(els).all(Op::is_monic),
            Op::For { body, .. } => body.iter().all(Op::is_monic),
            _ => true,
        }
    }
}

impl Matrix {
    /// The conjugate transpose.
    pub fn dagger(&self) -> Matrix {
        let n = self.size;
        let mut entries = Vec::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                entries.push(self.at(c, r).conj());
            }
        }
        Matrix { size: n, entries }
    }
}

/// A classical expression evaluated when the circuit runs. Its operations
/// are those of the language, with the same meaning.
#[derive(Clone, PartialEq, Debug)]
pub enum CExpr {
    /// A value known now.
    Value(Value),
    /// What a measurement wrote.
    Bit(Bit),
    /// A variable.
    Var(Var),
    /// A classical parameter of the entry operator.
    Param(u32),
    /// A unary operator.
    Unary(UnOp, Box<CExpr>),
    /// A binary operator.
    Binary(BinOp, Box<CExpr>, Box<CExpr>),
    /// `&&` or `||`.
    Logical(LogicalOp, Box<CExpr>, Box<CExpr>),
    /// An integer converted to another integer type.
    Cast(Box<CExpr>, IntTy),
    /// An element of an array.
    Index(Box<CExpr>, Box<CExpr>),
    /// A field of a structure or tuple.
    Field(Box<CExpr>, u32),
    /// An array of values.
    Array(Vec<CExpr>),
    /// A structure or tuple of values.
    Struct(Vec<CExpr>),
    /// A variant of an enumeration and its fields.
    Variant(u32, Vec<CExpr>),
    /// Whether an enumeration's value is the variant given.
    Is(Box<CExpr>, u32),
    /// A field of an enumeration's variant.
    Payload(Box<CExpr>, u32),
    /// `then` where `cond` holds and `els` where it does not.
    Select(Box<CExpr>, Box<CExpr>, Box<CExpr>),
    /// How many qubits the register of run-time length holds.
    RegLen(u32),
    /// How many elements an array computed as the circuit runs has.
    Len(Box<CExpr>),
    /// The array with one element added at its end.
    Append(Box<CExpr>, Box<CExpr>),
}

impl CExpr {
    /// The expression, or the value it is when known now.
    pub fn known(&self) -> Option<&Value> {
        match self {
            CExpr::Value(v) => Some(v),
            _ => None,
        }
    }
}

/// A classical parameter of an entry operator, which the caller supplies
/// each time the circuit runs.
#[derive(Clone, PartialEq, Debug)]
pub struct Param {
    /// Its name.
    pub name: String,
    /// Its type, as a program writes it.
    pub ty: String,
}

/// An operator-typed parameter of an entry operator, which the caller
/// supplies an operator for each time the circuit runs.
#[derive(Clone, PartialEq, Debug)]
pub struct Slot {
    /// Its name.
    pub name: String,
    /// The number of qubits each of the operator's parameters names.
    pub widths: Vec<u32>,
    /// Whether the circuit needs what is supplied to be monic.
    pub monic: bool,
    /// Arguments the supplied operator must leave in the state they are
    /// given in, up to a phase: ancillae it acts on by kickback, which the
    /// circuit returns to |0> on that understanding. Each is the argument's
    /// index and its state, over the argument's qubits.
    pub keeps: Vec<(u32, Vec<Cyclo>)>,
}

/// One circuit: what an operator becomes.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Circuit {
    /// The operator's name.
    pub name: String,
    /// The conductor its phases are counted at.
    pub conductor: u32,
    /// Its classical parameters.
    pub params: Vec<Param>,
    /// Its operator-typed parameters.
    pub slots: Vec<Slot>,
    /// The qubits holding its quantum parameters, in order, which the
    /// caller supplies: in |0>, unless it prepares them.
    pub inputs: Vec<Wire>,
    /// How many of `inputs` each quantum parameter holds, in order.
    pub registers: Vec<u32>,
    /// The qubits it gives back.
    pub outputs: Vec<Wire>,
    /// Processor nodes it names, each with the wire standing for it.
    pub nodes: Vec<(u64, Wire)>,
    /// How many wires it uses.
    pub wires: u32,
    /// How many bits it uses.
    pub bits: u32,
    /// How many variables it uses.
    pub vars: u32,
    /// How many registers of run-time length it grows ([`Op::Grow`]).
    pub grown: u32,
    /// What it does.
    pub body: Vec<Op>,
    /// The classical part of what it returns, if it returns anything
    /// classical: the outcome a caller receives.
    pub result: Option<CExpr>,
    /// Whether its structure depends on an outcome, through `lift`.
    pub dynamic: bool,
}

impl Circuit {
    /// Whether it asks for qubits as it runs, in a number known only then:
    /// it allocates inside a loop, whose bounds are known only as it runs,
    /// or grows a register of run-time length.
    pub fn allocates_while_running(&self) -> bool {
        fn inside(ops: &[Op], looped: bool) -> bool {
            ops.iter().any(|o| match o {
                Op::Alloc { .. } => looped,
                Op::Grow { .. } => true,
                Op::If { then, els, .. } => inside(then, looped) || inside(els, looped),
                Op::For { body, .. } => inside(body, true),
                _ => false,
            })
        }
        inside(&self.body, false)
    }
}

impl fmt::Display for Wire {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "q[{}]", self.0)
    }
}

impl fmt::Display for Bit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "c[{}]", self.0)
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}
