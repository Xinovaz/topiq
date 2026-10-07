//! Circuit generation: running an operator's body to find the circuit it
//! makes.
//!
//! [`Gen`] interprets an operator's typed body. Classical code runs, loops
//! over ranges known now are unrolled, and calls are followed, so that what
//! is left is the sequence of operations on qubits, each recorded as it is
//! met. Values not known until the circuit runs are carried as expressions
//! ([`super::value`]).
//!
//! # Control
//!
//! - **Quantum `if` and `match`** apply their branches controlled on the
//!   condition's qubits: every operation inside, however deeply called,
//!   gains those controls. A condition built with `!`, `&&`, `||` and `^` is
//!   computed into an ancilla where one control cannot say it, and the
//!   ancilla is returned to |0> after the branches; a branch may not change
//!   a qubit the condition read (`EQ04`), or the ancilla would not return. Inside such a branch
//!   nothing may measure, forget, prepare or lift, write to a qubit of the
//!   condition, leave allocated what it allocates, change a binding declared
//!   outside it, or jump out of it (`EQ04`, `forget` being `EQ05`).
//! - **An `if` or `match` on a measured value** becomes feed-forward: both
//!   branches are generated, each under the condition, and a binding either
//!   one changes afterwards holds whichever value the run chose.
//! - **A `while` or `loop`** runs while the circuit is generated, and may not
//!   act on qubits (`EQ03`). A `for` over a range known now is unrolled; one
//!   over a range fixed by a parameter or a lifted value becomes a loop of
//!   the circuit, whose body is generated once and repeated as it runs.
//! - **In a loop of the circuit,** a classical binding of the loop's own
//!   function that the body assigns whole, or grows, is carried from one
//!   time round to the next in a variable of the circuit; any other change
//!   to what is declared outside the body is refused (`EQ16`). A growable
//!   register of qubits the body names becomes a register whose length is
//!   known only as the circuit runs: it may grow, have its qubits named by
//!   index, and be measured or forgotten whole, but not shrink, and an
//!   operator cannot give one back (`EQ16`). So may a classical array of
//!   outcomes grow as the circuit runs.
//!
//! # Scope's end
//!
//! - **An `aux` ancilla** is returned to |0> by undoing its forward cone: the
//!   operations since its allocation that wrote it. One that found it in a
//!   state it leaves it in, up to a phase (an eigenvector of what it
//!   applies) is not part of the cone: the phase is kicked back to its
//!   controls, and undoing it would undo that too, as in Deutsch's
//!   algorithm. The ancilla's state is followed exactly while it is joined
//!   to no other qubit, which is when that can be known. The cone is undone
//!   in reverse order, and the result checked exactly: the ancilla must be
//!   |0> again for every state of the qubits it met. Where that cannot be
//!   decided, the cone must be free of measurement, and every qubit it read
//!   live and unchanged since (`EQ02` otherwise). A supplied operator given
//!   the ancilla is required to be monic and to leave it as it found it.
//! - **A register its declaration allocated** is given back when nothing
//!   changed it, or when what did is shown, exactly, to be undone; otherwise
//!   it may still hold qubits (`EQ01`).
//!
//! # Operators made of operators
//!
//! - **`adjoint(f)`** applies what `f` does, undone: its operations
//!   inverted in reverse order (a loop of the circuit run with its rounds
//!   reversed) or the operator `[adjoint: g]` declares,
//!   once checked, where that can be decided exactly, to undo `f` (`EQ17`
//!   otherwise). An operator that measures, forgets or prepares has no
//!   adjoint (`EQ17`).
//! - **`controlled(f)`** applies `f` controlled on its first argument, under
//!   the rules of a quantum branch (`EQ04`).
//! - **`f.then(g)`** applies `f` and then `g` to the same arguments.
//! - **`replay f(args)`** makes the call again, which prepares the same state
//!   only when `f` is monic and every argument is known now (`EJ02`).
//! - **`qcopy(&x)`** repeats on fresh qubits the operations that prepared
//!   `x`, which gives the same state only when they are known: `x`'s qubits
//!   were allocated in this operator, among the operations open now, and
//!   every operation that made them what they are is a gate or a matrix,
//!   acting on them and on qubits allocated since and given back in |0>
//!   (`EJ19` otherwise). A value the operator was given, half of an
//!   entangled pair, and a measured history are refused. Its classical
//!   parts are copied as values, each whose type defines `$copy` by it.
//! - **`t ^= c`** flips `t` controlled on the condition `c`: a parity is a
//!   flip for each side, a negation a flip and a bit flip, and `a | b` is
//!   `!(!a & !b)`, so none of these needs an ancilla. A flip controlled on
//!   its own target is `EQ20`.
//! - **`__exchange`**, which `core::swap` is, swaps the states of what two
//!   handles name with SWAP gates, as `gates::swap` does, and exchanges
//!   classical values as they are.
//! - **`query t[k]`** applies each entry of the map locale controlled on the
//!   key register holding its key, or, for a map locale of states, prepares
//!   each entry's state under that control. The entry **`measure t[k]`**
//!   gives is applied, or prepared, under feed-forward: each entry, where
//!   the measured key is its own.
//! - **A quantum enumeration's classical field** is put on its qubits as
//!   their basis state when a variant is made, by gates, or by gates chosen
//!   as the circuit runs for a value known only then; `match measure`
//!   measures those qubits with the tag, giving the field's value. A
//!   quantum `match` cannot read such a field (`EQ04`).
//! - **A judgement the body reads** has the value phase 9 derived for it
//!   before the body was generated ([`super::reads`]).
//!
//! `adjoint`, `controlled` and `replay` need a circuit fixed in advance, so
//! a dynamic operator, one that lifts a measured value, is refused there
//! (`EQ08`).

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::circuit::{Angle, CExpr, Circuit, Control, GateOp, Matrix, Op, Param, Slot, Var, Wire};
use crate::diag::{Code, Diagnostic};
use crate::exact::{Cyclo, Phase};
use crate::intern::Interner;
use crate::sema::arith::Abort;
use crate::span::Span;
use crate::tir::{
    Arg, BinOp, Block, Callee, Expr, ExprKind, FnId, Gate, GlobalId, GrowOp, Intrinsic, LocalId, LogicalOp, LoopId,
    Pat, PatKind, QuantumOp, Stmt, Ty, UnOp, Unit, Value,
};

use super::value::{Functor, Kind, Place, V};
use crate::circuit::action::{Verdict, returns_to_zero};

/// How many steps generation takes before it is taken not to finish.
const MAX_STEPS: u64 = 20_000_000;

/// The deepest calls may nest while a circuit is generated.
const MAX_CALLS: u32 = 1024;

/// Why evaluation stopped before giving a value.
enum Flow {
    /// A `return`.
    Return(V),
    /// A `break` out of a loop.
    Break(LoopId, Option<V>),
    /// A `continue` of a loop.
    Continue(LoopId),
    /// A problem was reported, and this circuit is abandoned.
    Stop,
}

type R<T> = Result<T, Flow>;

/// One function activation.
#[derive(Clone, Debug)]
struct Frame {
    func: FnId,
    locals: Vec<V>,
    /// Where its operations begin: the depth of the operation list open
    /// when it was called, and that list's length then.
    start: (usize, usize),
}

/// A branch of a quantum `if` or `match` being generated.
#[derive(Clone, Debug)]
struct Region {
    /// What makes it one.
    kind: RegionKind,
    /// Where the `if` or `match`, `controlled` or `adjoint` is.
    span: Span,
    /// How many controls were in force outside it.
    controls: usize,
    /// The qubits live when it began.
    live: BTreeSet<Wire>,
    /// The activation it began in.
    frame: usize,
    /// The bindings declared inside it.
    declared: HashSet<(usize, LocalId)>,
    /// The loops begun inside it.
    loops: HashSet<LoopId>,
    /// For a branch, the qubits its condition read, controls or not: one
    /// written inside would leave an ancilla computed from it unreturned.
    reads: Vec<Wire>,
}

/// What makes a region of generation one in which only what can be
/// controlled, or undone, may happen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RegionKind {
    /// A branch of a quantum `if` or `match`, applied controlled on its
    /// condition.
    Branch,
    /// An operator applied through `controlled`.
    Controlled,
    /// An operator whose adjoint is taken.
    Adjoint,
}

impl RegionKind {
    /// What the region is, in words.
    fn what(self) -> &'static str {
        match self {
            RegionKind::Branch => "a branch of a quantum `if` or `match`",
            RegionKind::Controlled => "an operator applied through `controlled`",
            RegionKind::Adjoint => "an operator whose adjoint is taken",
        }
    }

    /// What the place it begins is, in words.
    fn begins(self) -> &'static str {
        match self {
            RegionKind::Branch => "the branch is controlled on a qubit here",
            RegionKind::Controlled => "controlled here",
            RegionKind::Adjoint => "its adjoint is taken here",
        }
    }
}

impl Region {
    fn functor(span: Span, controls: usize, live: &BTreeSet<Wire>, frame: usize, kind: RegionKind) -> Region {
        Region {
            kind,
            span,
            controls,
            live: live.clone(),
            frame,
            declared: HashSet::new(),
            loops: HashSet::new(),
            reads: Vec::new(),
        }
    }
}

/// Every qubit a condition reads: the controls it comes to, and those the
/// operations computing its ancillae read.
fn condition_reads(lits: &Lits, undo: &[(Op, Option<Wire>)]) -> Vec<Wire> {
    let mut out: Vec<Wire> = match lits {
        Lits::Conj(cs) => cs.iter().map(|c| c.wire).collect(),
        _ => Vec::new(),
    };
    for (op, _) in undo {
        if let Op::Gate { controls, .. } = op {
            out.extend(controls.iter().map(|c| c.wire));
        }
    }
    out
}

/// A loop of the circuit, whose body is generated once and repeated as the
/// circuit runs, being generated.
#[derive(Clone, Debug)]
struct Repeated {
    /// The activation it began in.
    frame: usize,
    /// The bindings declared inside it.
    declared: HashSet<(usize, LocalId)>,
    /// The bindings of its activation it carries from one time round to the
    /// next in variables of the circuit.
    carried: HashSet<LocalId>,
}

impl Repeated {
    /// Whether a change to `p` inside the loop stays inside one time round,
    /// or is one the loop carries.
    fn admits(&self, p: &Place) -> bool {
        match p {
            Place::Local { frame, local, path } => {
                *frame > self.frame
                    || self.declared.contains(&(*frame, *local))
                    || (*frame == self.frame && path.is_empty() && self.carried.contains(local))
            }
            Place::Global { .. } => false,
            Place::Temp(..) => true,
        }
    }
}

/// What a condition of a quantum `if` comes to.
enum Lits {
    /// It always holds: a `bool` known now to be true.
    Always,
    /// It never holds.
    Never,
    /// It holds where every one of these controls does.
    Conj(Vec<Control>),
}

/// State saved before one branch of feed-forward, to be restored for the
/// other.
#[derive(Clone)]
struct Snapshot {
    frames: Vec<Frame>,
    globals: HashMap<GlobalId, V>,
    live: BTreeSet<Wire>,
}

/// Generates the circuits of one unit.
pub struct Gen<'a> {
    unit: &'a Unit,
    interner: &'a Interner,
    conductor: u32,
    reuse: bool,
    /// What generation reported.
    pub diags: Vec<Diagnostic>,
    reported: HashSet<(Code, u32, u32)>,
    /// The unit-scope objects' values.
    globals: HashMap<GlobalId, V>,
    /// The objects' values as the initialiser left them, which each circuit
    /// begins from.
    initialized: HashMap<GlobalId, V>,
    c: Circuit,
    /// The operations recorded, in the lists open (the circuit's own, and
    /// a feed-forward branch's or a loop's inside it), each with where in the
    /// unit it comes from.
    ops: Vec<Vec<(Op, Span)>>,
    free: Vec<Wire>,
    live: BTreeSet<Wire>,
    written: HashSet<Wire>,
    controls: Vec<Control>,
    regions: Vec<Region>,
    /// The loops of the circuit being generated.
    repeated: Vec<Repeated>,
    /// How many `while`s and `loop`s are running.
    loops: u32,
    /// How many feed-forward branches are being generated.
    ff: u32,
    frames: Vec<Frame>,
    steps: u64,
    in_init: bool,
    nodes: HashMap<u64, Wire>,
    /// For each binding allocated by its declaration, where its allocation
    /// begins: the depth of the operation list open then, and its length.
    allocated: HashMap<(usize, LocalId), (usize, usize)>,
    /// The expression being evaluated.
    here: Span,
    /// Each call being generated: where it is, and what it calls.
    sites: Vec<(Span, FnId)>,
    /// The value of each judgement a body reads, by its number among the
    /// unit's; `None` for one that has none, as was reported.
    derived: HashMap<u32, Option<Value>>,
}

impl<'a> Gen<'a> {
    /// A generator for `unit`.
    /// With `reuse`, a qubit given back in |0> is allocated again.
    pub fn new(unit: &'a Unit, interner: &'a Interner, conductor: u32, reuse: bool) -> Gen<'a> {
        Gen {
            unit,
            interner,
            conductor,
            reuse,
            diags: Vec::new(),
            reported: HashSet::new(),
            globals: HashMap::new(),
            initialized: HashMap::new(),
            c: Circuit::default(),
            ops: Vec::new(),
            free: Vec::new(),
            live: BTreeSet::new(),
            written: HashSet::new(),
            controls: Vec::new(),
            regions: Vec::new(),
            repeated: Vec::new(),
            loops: 0,
            ff: 0,
            frames: Vec::new(),
            steps: 0,
            in_init: false,
            nodes: HashMap::new(),
            sites: Vec::new(),
            here: Span::synthetic(),
            allocated: HashMap::new(),
            derived: HashMap::new(),
        }
    }

    fn name(&self, s: crate::intern::Symbol) -> &'a str {
        self.interner.resolve(s)
    }

    fn report(&mut self, mut d: Diagnostic) {
        // what goes wrong inside another unit's operator (a library gate)
        // is reported where this unit's own code called it
        let own = self.unit.source;
        let at_foreign = d.labels.first().is_some_and(|l| l.span.source != own);
        if at_foreign && let Some(&(site, callee)) = self.sites.iter().rev().find(|(s, _)| s.source == own) {
            let name = self.name(self.unit.func(callee).name);
            d.labels.retain(|l| l.span.source == own);
            d.labels.insert(
                0,
                crate::diag::Label {
                    span: site,
                    message: format!("inside `{name}`, called here"),
                    primary: true,
                },
            );
        }
        let span = d.labels.first().map_or(Span::synthetic(), |l| l.span);
        if self.reported.insert((d.code, span.start, span.end)) {
            self.diags.push(d);
        }
    }

    fn fail<T>(&mut self, d: Diagnostic) -> R<T> {
        self.report(d);
        Err(Flow::Stop)
    }

    fn reset(&mut self, name: String) {
        self.c = Circuit {
            name,
            conductor: self.conductor,
            ..Circuit::default()
        };
        self.ops = vec![Vec::new()];
        self.free.clear();
        self.live.clear();
        self.written.clear();
        self.controls.clear();
        self.regions.clear();
        self.repeated.clear();
        self.loops = 0;
        self.ff = 0;
        self.frames.clear();
        self.steps = 0;
        self.nodes.clear();
        self.sites.clear();
        self.allocated.clear();
    }

    //////////////////
    // ENTRY POINTS //
    //////////////////

    /// Runs the unit initialiser, which may set unit-scope objects but may
    /// not act on qubits (`EQ11`), since no circuit exists while it runs.
    pub fn initialize(&mut self, init: FnId) {
        self.reset("initialiser".to_owned());
        self.in_init = true;
        let _ = self.call(init, Vec::new(), self.unit.func(init).span);
        self.in_init = false;
        self.initialized = self.globals.clone();
    }

    /// Gives the judgement numbered `k` among the unit's the value its
    /// reading in a body has; `None` when it has none, as was reported.
    pub fn provide(&mut self, k: u32, v: Option<Value>) {
        self.derived.insert(k, v);
    }

    /// The conductor the circuits' phases are counted at.
    pub fn conductor(&self) -> u32 {
        self.conductor
    }

    /// The circuit the operator `f` makes, or `None` when a problem was
    /// reported. Each classical parameter becomes a parameter of the circuit,
    /// each quantum one a register it is given, and each operator-typed one a
    /// slot the caller fills.
    pub fn operator(&mut self, f: FnId) -> Option<Circuit> {
        let func = self.unit.func(f);
        self.reset(self.name(func.name).to_owned());
        // each circuit begins from the objects as the initialiser left them,
        // whatever generating another changed
        self.globals = self.initialized.clone();
        let mut args = Vec::with_capacity(func.params.len());
        let mut handles = Vec::new();
        for &p in &func.params {
            let l = func.local(p);
            let name = self.name(l.name).to_owned();
            let before = self.live.len();
            let v = self.parameter(l.ty, &name, l.span).ok()?;
            if self.live.len() > before {
                self.c.registers.push((self.live.len() - before) as u32);
            }
            if let V::Ref(Place::Temp(inner, _)) = &v {
                handles.extend(inner.wires());
            }
            args.push(v);
        }
        self.c.inputs = self.live.iter().copied().collect();
        let v = self.call(f, args, func.span).ok()?;
        if v.has_register() {
            self.report(runtime_structure(
                func.span,
                "the operator returns a register whose length is known only when the circuit runs",
                "a circuit's caller receives the qubits of its result on wires fixed when it is generated",
            ));
            return None;
        }
        let mut outputs = handles;
        outputs.extend(v.wires());
        self.c.outputs = outputs;
        self.c.result = v.classical_part();
        self.c.body = self.pop_ops();
        self.c.nodes = self.nodes.iter().map(|(&k, &w)| (k, w)).collect();
        self.c.nodes.sort();
        Some(std::mem::take(&mut self.c))
    }

    /// The value a parameter of type `ty` stands for in a circuit made for
    /// its operator.
    fn parameter(&mut self, ty: Ty, name: &str, span: Span) -> R<V> {
        // qubits, or a handle to them: input wires
        if self.unit.types.is_quantum(ty) {
            return self.fresh(ty, span);
        }
        if let Some((_, target)) = self.unit.types.as_ref(ty)
            && self.unit.types.is_quantum(target)
        {
            let v = self.fresh(target, span)?;
            return Ok(V::Ref(Place::Temp(Box::new(v), Vec::new())));
        }

        // an operator on qubits: a slot, filled when the circuit runs
        if let Some((params, ret)) = self.unit.types.as_sig(ty).filter(|_| matches!(ty, Ty::Fn(_))) {
            let mut widths = Vec::with_capacity(params.len());
            for &p in params {
                match self.handle_width(p) {
                    Some(w) => widths.push(w as u32),
                    None => {
                        return self.fail(runtime_structure(
                            span,
                            &format!("the operator-typed parameter `{name}` takes a classical value"),
                            "an operator supplied when the circuit runs is applied to qubits the circuit \
                             names; a classical argument would choose its circuit, which is fixed before",
                        ));
                    }
                }
            }
            if ret != Ty::Void {
                return self.fail(runtime_structure(
                    span,
                    &format!("the operator-typed parameter `{name}` returns a value"),
                    "an operator supplied when the circuit runs acts on qubits in place; what it would \
                     return has nowhere to go in a circuit fixed before it is chosen",
                ));
            }
            self.c.slots.push(Slot {
                name: name.to_owned(),
                widths,
                monic: false,
                keeps: Vec::new(),
            });
            return Ok(V::Slot(self.c.slots.len() as u32 - 1));
        }

        // anything else classical: an input, symbolic until the circuit runs
        if let Ty::Qmap(_) = ty {
            return self.fail(runtime_structure(
                span,
                &format!("the map locale `{name}` would be given when the circuit runs"),
                "a map locale's entries are parts of the circuit, chosen when it is generated",
            ));
        }
        let index = self.c.params.len() as u32;
        self.c.params.push(Param {
            name: name.to_owned(),
            ty: self.unit.types.display(ty, self.interner),
        });
        Ok(V::Sym(CExpr::Param(index), Kind::Input))
    }

    /// Fresh qubits for a value of quantum type `ty`, in |0>.
    fn fresh(&mut self, ty: Ty, span: Span) -> R<V> {
        if ty == Ty::Qubit {
            return Ok(V::Wire(self.alloc(span)?));
        }
        if let Some((elem, n)) = self.unit.types.as_array(ty) {
            let mut items = Vec::with_capacity(n as usize);
            for _ in 0..n {
                items.push(self.fresh(elem, span)?);
            }
            return Ok(V::Agg { array: true, items });
        }
        if let Ty::Adt(id) = ty {
            let def = self.unit.types.adt(id);
            if def.is_struct() {
                let fields: Vec<Ty> = def.fields().iter().map(|f| f.ty).collect();
                let mut items = Vec::with_capacity(fields.len());
                for t in fields {
                    items.push(if self.unit.types.is_quantum(t) { self.fresh(t, span)? } else { V::Moved });
                }
                return Ok(V::Agg { array: false, items });
            }
            let (tag, payload) = self.unit.types.enum_qubits(id);
            let (tag, payload) = (self.alloc_n(tag, span)?, self.alloc_n(payload, span)?);
            return Ok(V::QEnum { adt: id, tag, payload });
        }
        if self.unit.types.as_growable(ty).is_some() {
            return Ok(V::Agg {
                array: true,
                items: Vec::new(),
            });
        }
        Ok(V::Moved)
    }

    /// `n` fresh qubits in |0>.
    fn alloc_n(&mut self, n: u64, span: Span) -> R<Vec<Wire>> {
        (0..n).map(|_| self.alloc(span)).collect()
    }

    /// Where variant `v`'s fields lie in its enumeration's payload register:
    /// each field's type, its first qubit, and how many it takes: a
    /// quantum field its own, and a classical one a qubit for each bit, held
    /// as their basis state.
    fn variant_layout(&self, adt: crate::tir::AdtId, v: u32) -> Vec<(Ty, usize, usize)> {
        let mut at = 0usize;
        let mut out = Vec::new();
        for f in &self.unit.types.adt(adt).variants()[v as usize].fields {
            let w = self.unit.types.payload_qubits(f.ty) as usize;
            out.push((f.ty, at, w));
            at += w;
        }
        out
    }

    /// The number of qubits a handle's type names.
    fn handle_width(&self, t: Ty) -> Option<u64> {
        let (_, target) = self.unit.types.as_ref(t)?;
        self.unit.types.is_quantum(target).then(|| self.unit.types.qubits(target))
    }

    //////////////
    // EMITTING //
    //////////////

    fn push(&mut self, op: Op) {
        let at = self.site();
        self.ops.last_mut().expect("an operation list is open").push((op, at));
    }

    /// The operations of the innermost open list, without where each came
    /// from, closing it.
    fn pop_ops(&mut self) -> Vec<Op> {
        self.ops.pop().unwrap_or_default().into_iter().map(|(op, _)| op).collect()
    }

    /// Where in this unit's own code what is being generated comes from: the
    /// expression being evaluated, or, inside another unit's operator, where
    /// this unit called it.
    fn site(&self) -> Span {
        let own = self.unit.source;
        if self.here.source == own {
            return self.here;
        }
        self.sites.iter().rev().find(|(s, _)| s.source == own).map_or(self.here, |(s, _)| *s)
    }

    /// The checks every quantum operation is subject to, wherever it is.
    fn acting(&mut self, what: &str, span: Span) -> R<()> {
        self.steps += 1;
        if self.in_init {
            return self.fail(
                Diagnostic::new(Code::Eq11)
                    .with_message(format!("the unit initialiser {what}, but no circuit exists while it runs"))
                    .at(span)
                    .with_note(
                        "a quantum unit's initialiser runs before any of its circuits is generated, to \
                         compute what they are built from; it may set the unit's objects, not act on qubits",
                    )
                    .with_help("do this in an operator"),
            );
        }
        if self.loops > 0 {
            return self.fail(
                Diagnostic::new(Code::Eq03)
                    .with_message(format!("unbounded quantum control: a `while` or `loop` {what}"))
                    .at(span)
                    .with_note(
                        "a `while` or `loop` runs while the circuit is generated, and only over \
                         classical values; a circuit is a fixed sequence of operations, so what is \
                         repeated on qubits must repeat a number of times known in advance",
                    )
                    .with_help("repeat it with `for` over a range known when the circuit is generated"),
            );
        }
        Ok(())
    }

    /// A fresh qubit in |0>.
    fn alloc(&mut self, span: Span) -> R<Wire> {
        self.acting("allocates a qubit", span)?;
        let w = match self.free.pop() {
            Some(w) if self.reuse && self.ff == 0 => w,
            Some(w) => {
                self.free.push(w);
                self.new_wire()
            }
            None => self.new_wire(),
        };
        self.live.insert(w);
        self.written.remove(&w);
        self.push(Op::Alloc { wire: w });
        Ok(w)
    }

    fn new_wire(&mut self) -> Wire {
        let w = Wire(self.c.wires);
        self.c.wires += 1;
        w
    }

    /// Gives back a qubit in |0>.
    fn release(&mut self, w: Wire) {
        self.live.remove(&w);
        self.push(Op::Release { wire: w });
        if self.ff == 0 {
            self.free.push(w);
        }
    }

    /// Refuses an operation that cannot be controlled or undone, inside a
    /// region where that is needed: `code` in a quantum branch, `EQ05` for
    /// `forget` there, `EQ08` for a `lift` under `controlled` or `adjoint`,
    /// and `EQ17` for what has no inverse under `adjoint`.
    fn in_region(&mut self, what: &str, span: Span, code: Code) -> R<()> {
        let Some(r) = self.regions.last() else { return Ok(()) };
        let (kind, at) = (r.kind, r.span);
        let lift = what.starts_with("lifts");
        let code = match kind {
            RegionKind::Branch => code,
            _ if lift => Code::Eq08,
            RegionKind::Controlled => Code::Eq04,
            RegionKind::Adjoint => Code::Eq17,
        };
        let note = match (kind, lift) {
            (RegionKind::Branch, _) => {
                "a quantum branch is applied controlled on its condition's qubits, without measuring \
                 them, so what it does must be an operation that can be controlled: measuring, \
                 forgetting, preparing and lifting cannot"
            }
            (_, true) => {
                "a lift makes the rest of the circuit depend on a measured value, so the operator's \
                 circuit is dynamic; controlling or undoing it needs a circuit fixed in advance"
            }
            (RegionKind::Controlled, _) => {
                "a controlled operator is applied controlled on a qubit, without measuring it, so \
                 what it does must be an operation that can be controlled: measuring, forgetting \
                 and preparing cannot"
            }
            (RegionKind::Adjoint, _) => {
                "an adjoint undoes an operator, which is possible only for one that is unitary: \
                 measuring, forgetting and preparing destroy or make what undoing would need"
            }
        };
        let d = Diagnostic::new(code)
            .with_message(format!("{} {what}", kind.what()))
            .at_with(span, "here")
            .also(at, kind.begins())
            .with_note(note);
        self.fail(d)
    }

    /// Applies a gate to `targets`, controlled on `extra` and on whatever
    /// quantum control is in force.
    fn gate(&mut self, gate: GateOp, targets: Vec<Wire>, extra: Vec<Control>, span: Span) -> R<()> {
        self.acting("acts on a qubit", span)?;
        let mut controls = self.controls.clone();
        controls.extend(extra);
        self.check_operands(&targets, &controls, span)?;
        self.written.extend(&targets);
        self.push(Op::Gate { gate, targets, controls });
        Ok(())
    }

    /// Checks that no qubit is named twice by one operation, and that none of
    /// a quantum branch's condition qubits is written inside it.
    fn check_operands(&mut self, targets: &[Wire], controls: &[Control], span: Span) -> R<()> {
        for &t in targets {
            if let Some(i) = self.controls.iter().position(|c| c.wire == t) {
                let at = self.regions.iter().rev().find(|r| r.controls <= i).map_or(span, |r| r.span);
                return self.fail(
                    Diagnostic::new(Code::Eq04)
                        .with_message("a branch of a quantum `if` or `match` acts on a qubit its condition tests")
                        .at_with(span, "this acts on it")
                        .also(at, "the branch is controlled on it here")
                        .with_note(
                            "the branch is applied controlled on the condition's qubits; changing one \
                             of them inside would change whether the branch applies",
                        )
                        .with_help("act on the qubit outside the `if`"),
                );
            }
            if let Some(r) = self.regions.iter().rev().find(|r| r.reads.contains(&t)) {
                let at = r.span;
                return self.fail(
                    Diagnostic::new(Code::Eq04)
                        .with_message("a branch of a quantum `if` acts on a qubit its condition reads")
                        .at_with(span, "this acts on it")
                        .also(at, "the condition reads it here")
                        .with_note(
                            "the condition was computed into an ancilla from this qubit, which is \
                             returned to |0> after the branches by computing it again; changing the \
                             qubit inside would leave the ancilla holding something",
                        )
                        .with_help("act on the qubit outside the `if`"),
                );
            }
        }
        let mut seen = HashSet::new();
        for w in targets.iter().copied().chain(controls.iter().map(|c| c.wire)) {
            if !seen.insert(w) {
                return self.fail(
                    Diagnostic::new(Code::Eq15)
                        .with_message("one qubit is named twice among the operands of this operation")
                        .at(span)
                        .with_note(
                            "a gate acts on distinct qubits; two handles naming one qubit cannot be \
                             two operands of it",
                        )
                        .with_help("pass handles to different qubits"),
                );
            }
        }
        Ok(())
    }

    ////////////
    // PLACES //
    ////////////

    fn frame(&self) -> usize {
        self.frames.len() - 1
    }

    /// Where `e` is held, if it is a place.
    fn place_of(&mut self, e: &Expr) -> R<Option<Place>> {
        Ok(match &e.kind {
            ExprKind::Local(l) => Some(Place::Local {
                frame: self.frame(),
                local: *l,
                path: Vec::new(),
            }),
            ExprKind::Global(g) => Some(Place::Global { id: *g, path: Vec::new() }),
            ExprKind::Field { base, field } => self.place_of(base)?.map(|p| p.child(*field)),
            ExprKind::Index { base, index } => {
                let Some(p) = self.place_of(base)? else { return Ok(None) };
                if let V::Reg(reg) = self.read(&p) {
                    return self.element(reg, index).map(Some);
                }
                match self.expr(index)? {
                    V::Val(Value::Int(i, _)) => {
                        let i = u32::try_from(i).unwrap_or(u32::MAX);
                        let len = self.read(&p).part_count();
                        if len.is_some_and(|n| i as usize >= n) {
                            return self.abort(Abort::Index, e.span);
                        }
                        Some(p.child(i))
                    }
                    _ => None,
                }
            }
            ExprKind::Deref(r) => match self.expr(r)? {
                V::Ref(p) => Some(p),
                _ => None,
            },
            _ => None,
        })
    }

    fn read(&self, p: &Place) -> V {
        match p {
            Place::Local { frame, local, path } => {
                self.frames[*frame].locals[local.index()].at(path).unwrap_or(V::Moved)
            }
            Place::Global { id, path } => self.global_value(*id).at(path).unwrap_or(V::Moved),
            Place::Temp(v, path) => v.at(path).unwrap_or(V::Moved),
        }
    }

    /// Moves the value out of a place, which then holds nothing.
    fn take(&mut self, p: &Place) -> V {
        match p {
            Place::Local { frame, local, path } => {
                self.frames[*frame].locals[local.index()].replace(path, V::Moved).unwrap_or(V::Moved)
            }
            _ => self.read(p),
        }
    }

    fn write(&mut self, p: &Place, v: V, span: Span) -> R<()> {
        if let Some(r) = self.regions.last() {
            let inside = match p {
                Place::Local { frame, local, .. } => {
                    *frame > r.frame || (*frame == r.frame && r.declared.contains(&(*frame, *local)))
                }
                Place::Global { .. } => false,
                Place::Temp(..) => true,
            };
            if !inside {
                let at = r.span;
                return self.fail(
                    Diagnostic::new(Code::Eq04)
                        .with_message("a branch of a quantum `if` or `match` changes a binding declared outside it")
                        .at_with(span, "changed here")
                        .also(at, "the branch is controlled on a qubit here")
                        .with_note(
                            "a quantum branch is applied to every part of a superposition at once; which \
                             value a classical binding held afterwards would depend on a qubit no one has \
                             measured",
                        )
                        .with_help("keep classical changes inside the branch, on bindings declared there"),
                );
            }
        }
        self.repeats(p, span)?;
        // what the place held is dropped: qubits never acted on are given
        // back, and any acted on would be lost
        let old: Vec<Wire> = self.read(p).wires().into_iter().filter(|w| self.live.contains(w)).collect();
        if old.iter().any(|w| self.written.contains(w)) {
            return self.fail(
                Diagnostic::new(Code::Eq01)
                    .with_message("assigning here would drop the qubits the place may still hold")
                    .at(span)
                    .with_note(
                        "a qubit's state may be entangled with qubits that live on, so it cannot be \
                         dropped silently, by an assignment or otherwise",
                    )
                    .with_help("measure what the place holds, `forget` it, or move it elsewhere first"),
            );
        }
        for w in old {
            self.release(w);
        }
        self.store(p, v);
        Ok(())
    }

    /// Refuses a change to `p` inside a loop of the circuit that the loop
    /// cannot repeat: one to a binding declared outside it that it does not
    /// carry from one time round to the next.
    fn repeats(&mut self, p: &Place, span: Span) -> R<()> {
        match self.repeated.last() {
            Some(r) if !r.admits(p) => self.fail(runtime_structure(
                span,
                "a loop whose count is known only when the circuit runs changes this, which is declared outside it",
                "such a loop's body is generated once and repeated as the circuit runs; it can carry a \
                 classical binding of its own function, assigned whole, and grow a register of qubits, \
                 but not change where qubits are held or reach other bindings",
            )),
            _ => Ok(()),
        }
    }

    /// Puts `v` in `p`, with no check.
    fn store(&mut self, p: &Place, v: V) {
        match p {
            Place::Local { frame, local, path } => {
                let slot = &mut self.frames[*frame].locals[local.index()];
                if path.is_empty() {
                    *slot = v;
                } else {
                    slot.replace(path, v);
                }
            }
            Place::Global { id, path } => {
                let mut g = self.global_value(*id);
                g.replace(path, v);
                self.globals.insert(*id, g);
            }
            Place::Temp(..) => {}
        }
    }

    fn global_value(&self, id: GlobalId) -> V {
        if let Some(v) = self.globals.get(&id) {
            return v.clone();
        }
        match &self.unit.global(id).value {
            Some(v) => V::Val(v.clone()),
            None => V::Moved,
        }
    }

    ////////////////
    // EVALUATION //
    ////////////////

    fn abort<T>(&mut self, abort: Abort, span: Span) -> R<T> {
        let code = abort.code();
        self.fail(
            Diagnostic::new(Code::Ec04)
                .with_message(format!(
                    "generating this circuit would abort with {code}: {}",
                    code.message()
                ))
                .at_with(span, format!("{code} here"))
                .with_note(
                    "a quantum unit's classical code runs while its circuits are generated, during \
                     translation, so an operation that would abort is reported now",
                ),
        )
    }

    fn step(&mut self, span: Span) -> R<()> {
        self.steps += 1;
        if self.steps > MAX_STEPS {
            return self.fail(
                Diagnostic::new(Code::Eq14)
                    .with_message("generating this circuit did not finish")
                    .at(span)
                    .with_note(format!(
                        "circuit generation runs the classical code around the quantum operations; \
                         after {MAX_STEPS} steps it is taken to be a loop that never ends"
                    ))
                    .with_help("check that every loop here ends"),
            );
        }
        Ok(())
    }

    /// The value of a block. The quantum bindings it declares that still
    /// hold qubits when it is left (ancillae, and bindings never acted on)
    /// are given back in |0>.
    fn block(&mut self, b: &Block) -> R<V> {
        let mut declared = Vec::new();
        let r = self.block_inner(b, &mut declared);
        if matches!(r, Err(Flow::Stop)) {
            return r;
        }
        self.leave(&declared, b.span)?;
        r
    }

    fn block_inner(&mut self, b: &Block, declared: &mut Vec<LocalId>) -> R<V> {
        for s in &b.stmts {
            match s {
                Stmt::Let { local, init } => {
                    let l = self.unit.func(self.frames[self.frame()].func).local(*local);
                    let (ty, span) = (l.ty, l.span);
                    let v = match init {
                        Some(e) => self.expr(e)?,
                        None if self.unit.types.is_quantum(ty) => {
                            // where the allocation begins, for undoing or
                            // checking what is done to it before its scope ends
                            let at = (self.ops.len() - 1, self.ops.last().map_or(0, Vec::len));
                            let f = self.frame();
                            self.allocated.insert((f, *local), at);
                            self.fresh(ty, span)?
                        }
                        None => V::Moved,
                    };
                    self.bind(*local, v);
                    declared.push(*local);
                }
                Stmt::LetPat { pat, init } => {
                    let v = self.expr(init)?;
                    let mut out = Vec::new();
                    self.bind_pattern(pat, v, None, &mut out)?;
                    for (l, v) in out {
                        self.bind(l, v);
                        declared.push(l);
                    }
                }
                Stmt::Expr(e) => {
                    self.expr(e)?;
                }
            }
        }
        match &b.value {
            Some(v) => self.expr(v),
            None => Ok(V::Val(Value::Void)),
        }
    }

    fn bind(&mut self, local: LocalId, v: V) {
        let f = self.frame();
        self.frames[f].locals[local.index()] = v;
        if let Some(r) = self.regions.last_mut() {
            r.declared.insert((f, local));
        }
        for r in &mut self.repeated {
            r.declared.insert((f, local));
        }
    }

    /// Gives back the qubits the bindings `declared` still hold as their scope
    /// ends.
    fn leave(&mut self, declared: &[LocalId], span: Span) -> R<()> {
        let f = self.frame();
        for &l in declared.iter().rev() {
            let v = std::mem::replace(&mut self.frames[f].locals[l.index()], V::Moved);
            let at = self.allocated.remove(&(f, l));
            let wires: Vec<Wire> = v.wires().into_iter().filter(|w| self.live.contains(w)).collect();
            if wires.is_empty() {
                continue;
            }
            let local = self.unit.func(self.frames[f].func).local(l);
            let (aux, name, decl) = (local.aux, self.name(local.name), local.span);
            let acted = wires.iter().any(|w| self.written.contains(w));
            match at {
                Some(at) if acted && aux => self.uncompute(&wires, at, name, decl, span)?,
                Some(at) if acted => self.prove_reset(&wires, at, name, decl, span)?,
                _ => {}
            }
            for w in wires {
                self.release(w);
            }
        }
        Ok(())
    }

    /// The operations recorded since `at`.
    fn since(&self, at: (usize, usize)) -> Option<&[(Op, Span)]> {
        self.ops.get(at.0).filter(|_| at.0 + 1 == self.ops.len()).and_then(|l| l.get(at.1..))
    }

    /// Checks that a binding allocated by its declaration and acted on is
    /// back in |0> as its scope ends (because what was done to it has been
    /// undone) or reports it dropped while it may be live (`EQ01`).
    fn prove_reset(&mut self, wires: &[Wire], at: (usize, usize), name: &str, decl: Span, end: Span) -> R<()> {
        let n = self.conductor;
        let verdict = self.since(at).map_or(Verdict::Unknown, |ops| {
            let ops: Vec<Op> = ops.iter().map(|(o, _)| o.clone()).collect();
            returns_to_zero(&ops, wires, n)
        });
        if verdict == Verdict::Yes {
            return Ok(());
        }
        let why = match verdict {
            Verdict::No => "what was done to it is not undone: for some state of the qubits it met, it is not |0> here",
            _ => "whether it is |0> again cannot be decided exactly here: an operation that bears on it is supplied when \
                  the circuit runs, is controlled by a measured value, or joins it to too many qubits",
        };
        let at = Span::new(end.source, end.end.saturating_sub(1).max(end.start), end.end);
        self.fail(
            Diagnostic::new(Code::Eq01)
                .with_message(format!("`{name}` may still hold qubits when its scope ends"))
                .at_with(at, "its scope ends here")
                .also(decl, "allocated here, and acted on after")
                .with_note(format!(
                    "a qubit allocated in |0> and acted on is given back only if it is shown to be |0> again; \
                     {why}"
                ))
                .with_help(format!("measure `{name}`, `forget` it, return it, or declare it `aux let` to have it uncomputed")),
        )
    }

    /// Returns an ancilla to |0> as its scope ends, by undoing its forward
    /// cone: the operations since its allocation that wrote it, other than
    /// those that found it in a state they leave it in.
    ///
    /// The ancilla's state is followed exactly while it is joined to no
    /// other qubit. An operation writing it from such a state is left out of
    /// the cone when that state is an eigenvector of what it applies: a
    /// phase kicked back to its controls, which leaves the ancilla as it
    /// was. The cone is undone in reverse order, and the result checked
    /// exactly: for every state of the qubits it met, the ancilla must be |0>
    /// again. Where that cannot be decided, the cone must at least be monic,
    /// and every other qubit it reads live and unchanged since.
    fn uncompute(&mut self, anc: &[Wire], at: (usize, usize), name: &str, decl: Span, end: Span) -> R<()> {
        let n = self.conductor;
        let Some(window) = self.since(at).map(<[(Op, Span)]>::to_vec) else {
            return self.fail(self.ancilla_error(name, decl, end, "its allocation and its scope's end are in different parts of the circuit", None));
        };
        let set: HashSet<Wire> = anc.iter().copied().collect();
        let mut state: Option<Vec<Cyclo>> = Some(crate::circuit::action::State::basis(anc.to_vec(), 0, n).amps);
        let mut cone: Vec<(usize, Op, Span)> = Vec::new();
        let mut joined: Option<Span> = None;
        let mut keeps: Vec<(u32, u32, Vec<Cyclo>)> = Vec::new();
        let mut monic_slots: Vec<u32> = Vec::new();
        for (i, (op, span)) in window.iter().enumerate() {
            let ws = op.wires();
            if !ws.iter().any(|w| set.contains(w)) {
                continue;
            }
            match op {
                Op::Alloc { .. } | Op::Release { .. } => {}
                Op::Measure { .. } | Op::Forget { .. } => {
                    let what = if matches!(op, Op::Measure { .. }) { "measures" } else { "forgets" };
                    return self.fail(self.ancilla_error(name, decl, end, &format!("its cone {what} it"), Some(*span)));
                }
                Op::If { .. }
                | Op::For { .. }
                | Op::Lift { .. }
                | Op::Let { .. }
                | Op::Grow { .. }
                | Op::Element { .. }
                | Op::MeasureAll { .. } => {
                    return self.fail(self.ancilla_error(
                        name,
                        decl,
                        end,
                        "what is done to it depends on a value known only when the circuit runs",
                        Some(*span),
                    ));
                }
                Op::Call { slot, args, .. } => {
                    monic_slots.push(*slot);
                    // a supplied operator given the ancilla, in a state
                    // followed exactly, as one argument: it is required to
                    // leave it in that state
                    let whole = args.iter().position(|a| a.as_slice() == anc);
                    match (&state, whole) {
                        (Some(s), Some(k)) if args.iter().enumerate().all(|(j, a)| j == k || !a.iter().any(|w| set.contains(w))) => {
                            keeps.push((*slot, k as u32, s.clone()));
                        }
                        _ => {
                            cone.push((i, op.clone(), *span));
                            state = None;
                            joined.get_or_insert(*span);
                        }
                    }
                }
                Op::Gate { targets, controls, .. } | Op::Unitary { targets, controls, .. } => {
                    let writes = targets.iter().any(|t| set.contains(t));
                    let inside = ws.iter().all(|w| set.contains(w));
                    if !writes {
                        // read as a control: harmless while the ancilla is a
                        // basis state on those qubits, joining otherwise
                        if let Some(s) = &state
                            && !definite(anc, s, controls)
                        {
                            state = None;
                            joined.get_or_insert(*span);
                        }
                        continue;
                    }
                    match &state {
                        Some(s) if inside => {
                            state = crate::circuit::action::evolve(anc, s.clone(), std::slice::from_ref(op), n).ok();
                            cone.push((i, op.clone(), *span));
                        }
                        Some(s) if targets.iter().all(|t| set.contains(t))
                            && controls.iter().all(|c| !set.contains(&c.wire))
                            && kicks_back(op, anc, s, n) => {}
                        _ => {
                            cone.push((i, op.clone(), *span));
                            state = None;
                            joined.get_or_insert(*span);
                        }
                    }
                }
            }
        }
        // undo the cone, in reverse order
        let mut undo = Vec::with_capacity(cone.len());
        for (_, op, span) in cone.iter().rev() {
            match op.inverse() {
                Some(inv) => undo.push(inv),
                None => return self.fail(self.ancilla_error(name, decl, end, "its cone holds an operation with no inverse", Some(*span))),
            }
        }
        let mut all: Vec<Op> = window.iter().map(|(o, _)| o.clone()).collect();
        all.extend(undo.iter().cloned());
        let verdict = returns_to_zero(&all, anc, n);
        if verdict == Verdict::No {
            let why = "undoing its cone does not return it to |0> for every state of the qubits it met: an operation \
                       joined it to them in a way undoing it alone cannot part";
            return self.fail(self.ancilla_error(name, decl, end, why, joined));
        }
        if verdict == Verdict::Unknown
            && let Some((what, span)) = self.stale_read(&window, &cone, &set)
        {
            return self.fail(self.ancilla_error(name, decl, end, &what, Some(span)));
        }
        for s in monic_slots {
            if let Some(slot) = self.c.slots.get_mut(s as usize) {
                slot.monic = true;
            }
        }
        for (s, k, state) in keeps {
            if let Some(slot) = self.c.slots.get_mut(s as usize)
                && !slot.keeps.iter().any(|(a, st)| *a == k && *st == state)
            {
                slot.keeps.push((k, state));
            }
        }
        for op in undo {
            self.written.extend(op.writes());
            self.push(op);
        }
        Ok(())
    }

    /// A qubit other than the ancilla that an operation of its cone names,
    /// and that is no longer live, or is changed after that operation by
    /// one outside the cone: undoing the cone would then undo it against
    /// something other than what it read.
    fn stale_read(&self, window: &[(Op, Span)], cone: &[(usize, Op, Span)], anc: &HashSet<Wire>) -> Option<(String, Span)> {
        let in_cone: HashSet<usize> = cone.iter().map(|(i, _, _)| *i).collect();
        for (i, op, _) in cone {
            for w in op.wires().into_iter().filter(|w| !anc.contains(w)) {
                if !self.live.contains(&w) {
                    let at = window[*i + 1..].iter().find(|(o, _)| o.writes().contains(&w)).map_or(window[*i].1, |(_, s)| *s);
                    return Some(("its cone reads a qubit that is measured or forgotten before its scope ends".to_owned(), at));
                }
                for (j, (later, span)) in window.iter().enumerate().skip(i + 1) {
                    if !in_cone.contains(&j) && later.writes().contains(&w) {
                        return Some(("its cone reads a qubit that is changed afterwards".to_owned(), *span));
                    }
                }
            }
        }
        None
    }

    fn ancilla_error(&self, name: &str, decl: Span, end: Span, why: &str, op: Option<Span>) -> Diagnostic {
        let at = Span::new(end.source, end.end.saturating_sub(1).max(end.start), end.end);
        let mut d = Diagnostic::new(Code::Eq02)
            .with_message(format!("the ancilla `{name}` cannot be uncomputed: {why}"))
            .at_with(at, "its scope ends here, where it would be returned to |0>");
        if let Some(op) = op {
            d = d.also(op, "this operation");
        }
        d.also(decl, "declared here")
            .with_note(
                "an ancilla is returned to |0> at the end of its scope by undoing what was done to it, which is \
                 possible only when that was a unitary transformation of it that nothing since has disturbed",
            )
            .with_help(format!("declare `{name}` with plain `let`, and measure or `forget` it"))
    }

    /// Calls the function `f` with `args`.
    fn call(&mut self, f: FnId, args: Vec<V>, span: Span) -> R<V> {
        if self.frames.len() as u32 >= MAX_CALLS {
            return self.fail(
                Diagnostic::new(Code::Eq14)
                    .with_message("generating this circuit calls functions too deeply")
                    .at(span)
                    .with_note(format!(
                        "calls may nest at most {MAX_CALLS} deep while a circuit is generated, since \
                         each is inlined"
                    )),
            );
        }
        let func = self.unit.func(f);
        let mut locals = vec![V::Moved; func.locals.len()];
        for (&p, v) in func.params.iter().zip(args) {
            locals[p.index()] = v;
        }
        let start = (self.ops.len().saturating_sub(1), self.ops.last().map_or(0, Vec::len));
        self.frames.push(Frame { func: f, locals, start });
        self.sites.push((span, f));
        let r = stacker::maybe_grow(256 * 1024, 4 * 1024 * 1024, || self.block(&func.body));
        self.sites.pop();
        self.frames.pop();
        match r {
            Ok(v) | Err(Flow::Return(v)) => Ok(v),
            Err(Flow::Stop) => Err(Flow::Stop),
            Err(Flow::Break(..) | Flow::Continue(_)) => Ok(V::Val(Value::Void)),
        }
    }

    /// The value of `e`. A quantum value is moved out of the place holding it.
    fn expr(&mut self, e: &Expr) -> R<V> {
        let outer = std::mem::replace(&mut self.here, e.span);
        let r = stacker::maybe_grow(256 * 1024, 4 * 1024 * 1024, || self.expr_inner(e));
        self.here = outer;
        r
    }

    fn expr_inner(&mut self, e: &Expr) -> R<V> {
        self.step(e.span)?;
        let span = e.span;
        match &e.kind {
            ExprKind::Const(v) => Ok(V::Val(v.clone())),
            ExprKind::Local(_) | ExprKind::Global(_) | ExprKind::Field { .. } | ExprKind::Index { .. } | ExprKind::Deref(_) => {
                if let Some(p) = self.place_of(e)? {
                    let v = if self.unit.types.is_quantum(e.ty) { self.take(&p) } else { self.read(&p) };
                    return Ok(v);
                }
                self.projection(e)
            }
            ExprKind::FnRef(Callee::Fn(f)) | ExprKind::ThunkRef(f) => Ok(V::Func(*f)),
            ExprKind::FnRef(Callee::Extern(_)) => self.foreign(span),
            ExprKind::FnAsClosure(x) | ExprKind::Coerce(x) | ExprKind::Grow(x) => self.expr(x),
            ExprKind::Closure { code, captures } => Ok(V::Closure {
                code: *code,
                env: self.exprs(captures)?,
            }),
            ExprKind::Call { callee, args } => self.call_expr(*callee, args, span),
            ExprKind::IndirectCall { callee, args } => {
                let target = self.expr(callee)?;
                let target = self.deref(target);
                let values = self.exprs(args)?;
                self.call_value(target, values, span)
            }
            ExprKind::Intrinsic { which, args } => self.intrinsic(*which, args, e),
            ExprKind::Growable { op, array, args } => self.growable(*op, array, args, e),
            ExprKind::Unary { op, operand } => {
                let v = self.expr(operand)?;
                match v {
                    V::Val(x) => {
                        let r = match (op, x) {
                            (UnOp::Neg, Value::Int(x, t)) => crate::sema::arith::neg(t, x).map(|r| Value::Int(r, t)),
                            (UnOp::Neg, Value::Float(x, t)) => Ok(Value::Float(crate::sema::arith::float::neg(t, x), t)),
                            (UnOp::Not, Value::Int(x, t)) => Ok(Value::Int(crate::sema::arith::not(t, x), t)),
                            (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
                            _ => unreachable!("the checker admits no other operand"),
                        };
                        r.map(V::Val).or_else(|a| self.abort(a, span))
                    }
                    V::Sym(x, k) => Ok(V::Sym(CExpr::Unary(*op, Box::new(x)), k)),
                    _ => self.not_data(span),
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.expr(lhs)?;
                let r = self.expr(rhs)?;
                match (l, r) {
                    (V::Val(a), V::Val(b)) => crate::sema::consteval::binary(*op, a, b).map(V::Val).or_else(|a| self.abort(a, span)),
                    (a, b) => {
                        let (Some((x, kx)), Some((y, ky))) = (a.classical(), b.classical()) else {
                            return self.not_data(span);
                        };
                        let k = super::value::join_kind(kx, ky).unwrap_or(Kind::Input);
                        Ok(V::Sym(CExpr::Binary(*op, Box::new(x), Box::new(y)), k))
                    }
                }
            }
            ExprKind::Logical { op, lhs, rhs } => {
                let l = self.expr(lhs)?;
                match l {
                    V::Val(Value::Bool(b)) => {
                        let decided = match op {
                            LogicalOp::And => !b,
                            LogicalOp::Or => b,
                        };
                        if decided { Ok(V::Val(Value::Bool(b))) } else { self.expr(rhs) }
                    }
                    V::Sym(x, k) => {
                        let before = self.ops.last().map_or(0, Vec::len);
                        let r = self.expr(rhs)?;
                        if self.ops.last().map_or(0, Vec::len) != before {
                            return self.fail(runtime_structure(
                                rhs.span,
                                "whether this runs depends on a measured value",
                                "the right operand of `&&` or `||` runs only when the left does not decide; \
                                 when the left is known only as the circuit runs, what the right operand \
                                 does to qubits cannot be part of it",
                            ));
                        }
                        let (y, ky) = r.classical().ok_or(Flow::Stop)?;
                        let k = Kind::join(k, ky.unwrap_or(k));
                        Ok(V::Sym(CExpr::Logical(*op, Box::new(x), Box::new(y)), k))
                    }
                    _ => self.not_data(span),
                }
            }
            ExprKind::Cast { expr, to } => match self.expr(expr)? {
                V::Val(v) => crate::sema::consteval::cast(&v, *to).map(V::Val).or_else(|a| self.abort(a, span)),
                V::Sym(x, k) => match to {
                    Ty::Int(t) => Ok(V::Sym(CExpr::Cast(Box::new(x), *t), k)),
                    _ => self.fail(runtime_structure(
                        span,
                        "this conversion of a value known only when the circuit runs",
                        "a circuit computes with integers and `bool`s while it runs",
                    )),
                },
                _ => self.not_data(span),
            },
            ExprKind::Ref(x) => match self.place_of(x)? {
                Some(p) => Ok(V::Ref(p)),
                None => {
                    let v = self.expr(x)?;
                    Ok(V::Ref(Place::Temp(Box::new(v), Vec::new())))
                }
            },
            ExprKind::StructLit { fields } => {
                let n = match e.ty {
                    Ty::Adt(id) => self.unit.types.adt(id).fields().len(),
                    _ => fields.len(),
                };
                let items = self.members(n, fields)?;
                Ok(V::Agg { array: false, items }.normal())
            }
            ExprKind::Variant { variant, fields } => {
                let Ty::Adt(id) = e.ty else { unreachable!("a variant is of an enumeration") };
                let n = self.unit.types.adt(id).variants()[*variant as usize].fields.len();
                let items = self.members(n, fields)?;
                if self.unit.types.is_quantum(e.ty) {
                    return self.inject(id, *variant, items, span);
                }
                Ok(V::Variant(*variant, items).normal())
            }
            ExprKind::ArrayLit(items) => Ok(V::Agg {
                array: true,
                items: self.exprs(items)?,
            }
            .normal()),
            ExprKind::ArrayRepeat { elem, len } => {
                let v = self.expr(elem)?;
                Ok(V::Agg {
                    array: true,
                    items: vec![v; *len as usize],
                }
                .normal())
            }
            ExprKind::Assign { place, value } => {
                let v = self.expr(value)?;
                let Some(p) = self.place_of(place)? else {
                    return self.fail(runtime_structure(
                        place.span,
                        "which element this assigns is known only when the circuit runs",
                        "an element of an array is chosen while the circuit is generated",
                    ));
                };
                self.write(&p, v, span)?;
                Ok(V::Val(Value::Void))
            }
            ExprKind::Block(b) => self.block(b),
            ExprKind::If { cond, then, els } => self.if_expr(cond, then, els.as_deref(), span),
            ExprKind::Match { scrutinee, arms } => self.match_expr(scrutinee, arms, e),
            ExprKind::Loop { id, body } => {
                self.enter_loop(*id);
                self.loops += 1;
                let r = loop {
                    self.step(span)?;
                    match self.block(body) {
                        Ok(_) | Err(Flow::Continue(_)) => {}
                        Err(Flow::Break(t, v)) if t == *id => break Ok(v.unwrap_or(V::Val(Value::Void))),
                        Err(f) => break Err(f),
                    }
                };
                self.loops -= 1;
                r
            }
            ExprKind::While { id, cond, body } => {
                self.enter_loop(*id);
                self.loops += 1;
                let r = loop {
                    self.step(span)?;
                    match self.expr(cond) {
                        Ok(V::Val(Value::Bool(true))) => {}
                        Ok(V::Val(Value::Bool(false))) => break Ok(V::Val(Value::Void)),
                        Ok(_) => {
                            break self.fail(runtime_structure(
                                cond.span,
                                "whether this loop goes on is known only when the circuit runs",
                                "how many times a loop runs is part of the circuit's structure",
                            ));
                        }
                        Err(f) => break Err(f),
                    }
                    match self.block(body) {
                        Ok(_) => {}
                        Err(Flow::Continue(t)) if t == *id => {}
                        Err(Flow::Break(t, _)) if t == *id => break Ok(V::Val(Value::Void)),
                        Err(f) => break Err(f),
                    }
                };
                self.loops -= 1;
                r
            }
            ExprKind::ForRange {
                id,
                var,
                start,
                end,
                inclusive,
                body,
            } => self.for_range(*id, *var, start, end, *inclusive, body, span),
            ExprKind::Break { target, value } => {
                let v = value.as_deref().map(|v| self.expr(v)).transpose()?;
                self.jump(*target, span)?;
                Err(Flow::Break(*target, v))
            }
            ExprKind::Continue { target } => {
                self.jump(*target, span)?;
                Err(Flow::Continue(*target))
            }
            ExprKind::Return(v) => {
                let v = match v {
                    Some(v) => self.expr(v)?,
                    None => V::Val(Value::Void),
                };
                if let Some(r) = self.regions.last()
                    && r.frame == self.frame()
                {
                    let at = r.span;
                    return self.fail(jump_out(span, at, "`return`"));
                }
                Err(Flow::Return(v))
            }
            ExprKind::Quantum { op, args } => self.quantum(op, args, e),
        }
    }

    fn enter_loop(&mut self, id: LoopId) {
        if let Some(r) = self.regions.last_mut() {
            r.loops.insert(id);
        }
    }

    /// A `break` or `continue` of the loop `target`, which may not leave a
    /// quantum branch.
    fn jump(&mut self, target: LoopId, span: Span) -> R<()> {
        if let Some(r) = self.regions.last()
            && r.frame == self.frame()
            && !r.loops.contains(&target)
        {
            let at = r.span;
            return self.fail(jump_out(span, at, "a `break` or `continue`"));
        }
        Ok(())
    }

    /// A field or element of a value that is not a place.
    fn projection(&mut self, e: &Expr) -> R<V> {
        match &e.kind {
            ExprKind::Field { base, field } => {
                let b = self.expr(base)?;
                b.part(*field).map_or_else(|| self.not_data(e.span), Ok)
            }
            ExprKind::Index { base, index } => {
                let b = self.expr(base)?;
                let i = self.expr(index)?;
                match (b, i) {
                    (b, V::Val(Value::Int(i, _))) => {
                        let i = u32::try_from(i).unwrap_or(u32::MAX);
                        if b.part_count().is_some_and(|n| i as usize >= n) {
                            return self.abort(Abort::Index, e.span);
                        }
                        b.part(i).map_or_else(|| self.not_data(e.span), Ok)
                    }
                    (b, V::Sym(i, k)) => match b.classical() {
                        Some((a, ka)) => {
                            let k = Kind::join(k, ka.unwrap_or(k));
                            Ok(V::Sym(CExpr::Index(Box::new(a), Box::new(i)), k))
                        }
                        None => self.fail(runtime_structure(
                            e.span,
                            "which qubit this names is known only when the circuit runs",
                            "each operation of a circuit acts on qubits chosen when it is generated",
                        )),
                    },
                    _ => self.not_data(e.span),
                }
            }
            ExprKind::Deref(r) => match self.expr(r)? {
                V::Ref(p) => Ok(self.read(&p)),
                _ => self.not_data(e.span),
            },
            _ => self.not_data(e.span),
        }
    }

    fn members(&mut self, n: usize, fields: &[(u32, Expr)]) -> R<Vec<V>> {
        let mut out = vec![V::Moved; n];
        for (i, f) in fields {
            out[*i as usize] = self.expr(f)?;
        }
        Ok(out)
    }

    /// The values of `es`.
    fn exprs(&mut self, es: &[Expr]) -> R<Vec<V>> {
        es.iter().map(|e| self.expr(e)).collect()
    }

    /// What `v` refers to, if it is a reference, and otherwise `v`.
    fn deref(&self, v: V) -> V {
        match v {
            V::Ref(p) => self.read(&p),
            v => v,
        }
    }

    /// The value of `e`, read where it is held if it is a place (leaving
    /// any qubits there) and seen through a reference.
    fn peek(&mut self, e: &Expr) -> R<V> {
        let v = match self.place_of(e)? {
            Some(p) => self.read(&p),
            None => self.expr(e)?,
        };
        Ok(self.deref(v))
    }

    /// Something circuit generation cannot compute with, which analysis has
    /// already refused.
    fn not_data<T>(&mut self, span: Span) -> R<T> {
        self.fail(runtime_structure(
            span,
            "this value is known only when the circuit runs, where its structure needs it now",
            "a circuit is fixed before it runs; what shapes it must be known while it is generated",
        ))
    }

    fn foreign<T>(&mut self, span: Span) -> R<T> {
        self.fail(crate::sema::report::unsupported(span, "calling a function of another unit whose body is not at hand while a circuit is generated"))
    }

    fn call_expr(&mut self, callee: Callee, args: &[Expr], span: Span) -> R<V> {
        // an exact scalar's operation is computed exactly, as constant
        // evaluation computes it
        if let Some(op) = crate::sema::exact::exact_op(self.unit, self.interner, callee) {
            let vs: Vec<V> = self.exprs(args)?.into_iter().map(|v| self.deref(v)).collect();
            // a copy is the value copied, whenever that is known
            if op.is_copy()
                && let Some(v) = vs.first()
            {
                return Ok(v.clone());
            }
            let known: Option<Vec<Value>> = vs
                .into_iter()
                .map(|v| if let V::Val(x) = v { Some(x) } else { None })
                .collect();
            let Some(values) = known else {
                return self.fail(runtime_structure(
                    span,
                    "exact arithmetic on a value known only when the circuit runs",
                    "a circuit computes with integers and `bool`s while it runs; `frac`, `cyclo` and \
                     `phase` values are computed while it is generated",
                ));
            };
            match crate::sema::exact::eval_exact(&op, &values) {
                Ok(v) => return Ok(V::Val(v)),
                Err(crate::sema::exact::ExactError::Abort(a)) => return self.abort(a, span),
                Err(crate::sema::exact::ExactError::Unsupported) => {
                    let Callee::Fn(f) = callee else { return self.foreign(span) };
                    return self.call(f, values.into_iter().map(V::Val).collect(), span);
                }
            }
        }
        let Callee::Fn(f) = callee else { return self.foreign(span) };
        let values = self.exprs(args)?;
        self.call(f, values, span)
    }

    /// A call of an operator value.
    fn call_value(&mut self, target: V, mut values: Vec<V>, span: Span) -> R<V> {
        match target {
            V::Func(f) => self.call(f, values, span),
            V::Closure { code, env } => {
                let env = V::Ref(Place::Temp(Box::new(V::Agg { array: false, items: env }), Vec::new()));
                values.insert(0, env);
                self.call(code, values, span)
            }
            V::Slot(i) => self.call_slot(i, values, span, false),
            V::Functor(f) => match *f {
                Functor::Adjoint(inner) => {
                    self.needs_static(&inner, "taking an adjoint", span)?;
                    self.call_adjoint(inner, values, span)
                }
                Functor::Controlled(inner) => {
                    self.needs_static(&inner, "`controlled`", span)?;
                    let mut it = values.into_iter();
                    let control = it.next().unwrap_or(V::Moved);
                    let ws = self.handle_wires(control, span)?;
                    let rest: Vec<V> = it.collect();
                    let mut out = V::Val(Value::Void);
                    self.region(span, RegionKind::Controlled, vec![on(ws[0])], &[], |g| {
                        out = g.call_value(inner, rest, span)?;
                        Ok(V::Val(Value::Void))
                    })?;
                    Ok(out)
                }
                Functor::Then(a, b) => {
                    self.call_value(a, values.clone(), span)?;
                    self.call_value(b, values, span)
                }
                Functor::Query(map, key) => {
                    let entries = self.unit.geometry.qmaps[map].def.entries.clone();
                    for (k, e) in &entries {
                        let f = self.expr(e)?;
                        let controls = key_controls(&key, *k);
                        let args = values.clone();
                        self.region(span, RegionKind::Controlled, controls, &[], |g| g.call_value(f, args, span))?;
                    }
                    Ok(V::Val(Value::Void))
                }
                Functor::Lookup(map, bits) => {
                    let entries = self.unit.geometry.qmaps[map].def.entries.clone();
                    self.call_entry(&entries, &bits, values, span)
                }
            },
            _ => self.fail(runtime_structure(
                span,
                "which operator this calls is known only when the circuit runs",
                "a circuit is fixed before it runs, and each call in it is part of it",
            )),
        }
    }

    /// A call of the entry of a map locale that the measured key `bits`
    /// holds: for each entry, whether it is that one is tested as the
    /// circuit runs, and the entry applied if it is.
    fn call_entry(&mut self, entries: &[(u64, Expr)], bits: &[CExpr], values: Vec<V>, span: Span) -> R<V> {
        let Some(((k, e), rest)) = entries.split_first() else {
            return Ok(V::Val(Value::Void));
        };
        let f = self.expr(e)?;
        let args = values.clone();
        self.feed_forward(
            key_condition(bits, *k),
            |g| {
                g.call_value(f, args, span)?;
                Ok(V::Val(Value::Void))
            },
            |g| g.call_entry(rest, bits, values, span),
        )
    }

    /// Refuses a dynamic operator where `what` needs its circuit fixed in
    /// advance (`EQ08`).
    fn needs_static(&mut self, v: &V, what: &str, span: Span) -> R<()> {
        let f = match v {
            V::Func(f) => *f,
            V::Functor(inner) => {
                return match inner.as_ref() {
                    Functor::Adjoint(x) | Functor::Controlled(x) => self.needs_static(x, what, span),
                    Functor::Then(a, b) => {
                        self.needs_static(a, what, span)?;
                        self.needs_static(b, what, span)
                    }
                    Functor::Query(map, _) | Functor::Lookup(map, _) => {
                        for (_, e) in self.unit.geometry.qmaps[*map].def.entries.clone() {
                            if let ExprKind::FnRef(crate::tir::Callee::Fn(f)) = e.kind {
                                self.needs_static(&V::Func(f), what, span)?;
                            }
                        }
                        Ok(())
                    }
                };
            }
            _ => return Ok(()),
        };
        if !dynamic(self.unit, f) {
            return Ok(());
        }
        let name = self.name(self.unit.func(f).name);
        self.fail(
            Diagnostic::new(Code::Eq08)
                .with_message(format!("{what} needs a circuit fixed in advance, but `{name}` is dynamic"))
                .at_with(span, "here")
                .also(self.unit.func(f).span, format!("`{name}` lifts a measured value, or calls what does"))
                .with_note(
                    "a lift makes the rest of an operator's circuit depend on what was measured; controlling \
                     it, undoing it, or preparing its state again needs the circuit known before it runs",
                ),
        )
    }

    /// `adjoint(f)(args)`: `f`'s declared adjoint where it has one, and
    /// otherwise what `f` does, undone: its operations inverted, in reverse
    /// order. An operator that measures, forgets or lifts has no adjoint.
    fn call_adjoint(&mut self, inner: V, values: Vec<V>, span: Span) -> R<V> {
        // a declared adjoint, or a slot's, is the caller's to give
        if let V::Func(f) = inner
            && let Some(adj) = self.unit.func(f).attrs.adjoint
        {
            return self.declared_adjoint(f, adj, values, span);
        }
        if let V::Slot(i) = inner {
            return self.call_slot(i, values, span, true);
        }

        // record the operator's steps apart
        self.regions.push(Region::functor(span, self.controls.len(), &self.live, self.frame(), RegionKind::Adjoint));
        self.ops.push(Vec::new());
        let r = self.call_value(inner, values, span);
        let recorded = self.ops.pop().unwrap_or_default();
        self.regions.pop();
        let v = r?;

        // then emit each step's inverse, last first
        let mut undone = Vec::with_capacity(recorded.len());
        for (op, at) in recorded.iter().rev() {
            match op.inverse() {
                Some(inv) => undone.push((inv, *at)),
                None => {
                    let what = match op {
                        Op::Measure { .. } | Op::MeasureAll { .. } => "measures",
                        Op::Forget { .. } => "forgets",
                        Op::Lift { .. } => "lifts a measured value",
                        _ => "is chosen by a value known only when the circuit runs",
                    };
                    return self.fail(
                        Diagnostic::new(Code::Eq17)
                            .with_message(format!("this operator has no adjoint: it {what}"))
                            .at_with(span, "its adjoint is taken here")
                            .also(*at, "here")
                            .with_note(
                                "an adjoint undoes an operator, which is possible only for one that is \
                                 unitary; a measurement or a forgetting destroys what undoing it would need",
                            ),
                    );
                }
            }
        }
        for (op, at) in undone {
            self.written.extend(op.writes());
            self.ops.last_mut().expect("an operation list is open").push((op, at));
        }
        Ok(v)
    }

    /// `adjoint(f)(args)` for an operator `f` declaring its adjoint `adj`:
    /// `adj` is applied, having been checked, where that can be decided
    /// exactly, to undo `f`.
    fn declared_adjoint(&mut self, f: FnId, adj: FnId, values: Vec<V>, span: Span) -> R<V> {
        self.ops.push(Vec::new());
        let forward = self.call(f, values.clone(), span);
        let backward = forward.and_then(|_| self.call(adj, values.clone(), span));
        let trial: Vec<Op> = self.pop_ops();
        backward?;
        if crate::circuit::action::is_identity(&trial, self.conductor) == Verdict::No {
            let (fname, aname) = (self.name(self.unit.func(f).name), self.name(self.unit.func(adj).name));
            return self.fail(
                Diagnostic::new(Code::Eq17)
                    .with_message(format!("`{aname}`, declared the adjoint of `{fname}`, does not undo it"))
                    .at_with(span, "the adjoint is taken here")
                    .also(self.unit.func(f).span, "declared here")
                    .with_note(format!(
                        "applying `{fname}` and then `{aname}` should change nothing, and computed exactly, it changes \
                         the state of what they act on"
                    ))
                    .with_help("correct the declared adjoint, or remove `[adjoint: …]` to have the compiler undo the operator itself"),
            );
        }
        // the trial ran forward and back; the qubits it allocated were given
        // back inside it. now only the adjoint is applied
        self.call(adj, values, span)
    }

    /// A call of the operator the caller supplies for a slot.
    fn call_slot(&mut self, slot: u32, args: Vec<V>, span: Span, adjoint: bool) -> R<V> {
        self.acting("calls an operator supplied when the circuit runs", span)?;
        let wires: Vec<Vec<Wire>> = args.into_iter().map(|a| self.deref(a).wires()).collect();
        let flat: Vec<Wire> = wires.iter().flatten().copied().collect();
        let controls = self.controls.clone();
        self.check_operands(&flat, &controls, span)?;
        self.written.extend(flat);
        self.push(Op::Call {
            slot,
            args: wires,
            controls,
            adjoint,
        });
        Ok(V::Val(Value::Void))
    }

    ////////////////
    // INTRINSICS //
    ////////////////

    fn intrinsic(&mut self, which: Intrinsic, args: &[Expr], e: &Expr) -> R<V> {
        let span = e.span;
        match which {
            Intrinsic::Gate(g) => self.gate_intrinsic(g, args, span),
            Intrinsic::Adjoint | Intrinsic::Controlled | Intrinsic::Then => {
                let mut it = self.exprs(args)?.into_iter();
                let f = it.next().unwrap_or(V::Moved);
                Ok(V::Functor(Box::new(match which {
                    Intrinsic::Adjoint => Functor::Adjoint(f),
                    Intrinsic::Controlled => Functor::Controlled(f),
                    _ => Functor::Then(f, it.next().unwrap_or(V::Moved)),
                })))
            }
            Intrinsic::Apply => self.apply(args, span),
            Intrinsic::Node => {
                let V::Val(Value::Int(k, _)) = self.expr(&args[0])? else {
                    return self.not_data(args[0].span);
                };
                let k = u64::try_from(k).unwrap_or(0);
                self.acting("names a processor node", span)?;
                let w = match self.nodes.get(&k) {
                    Some(&w) => w,
                    None => {
                        let w = self.new_wire();
                        self.live.insert(w);
                        self.nodes.insert(k, w);
                        w
                    }
                };
                Ok(V::Ref(Place::Temp(Box::new(V::Wire(w)), Vec::new())))
            }
            Intrinsic::Len => {
                let v = self.peek(&args[0])?;
                match (v.part_count(), v) {
                    (Some(n), _) => Ok(V::Val(Value::Int(n as i128, crate::tir::IntTy::USIZE))),
                    (None, V::Reg(r)) => {
                        // the length now, which the register's growing later
                        // does not change
                        let var = self.new_var();
                        self.push(Op::Let { var, value: CExpr::RegLen(r) });
                        Ok(V::Sym(CExpr::Var(var), Kind::Input))
                    }
                    // an array's length depends on a measured value only
                    // where one chose between arrays
                    (None, V::Sym(x, _)) => {
                        let k = if chooses(&x) { Kind::Outcome } else { Kind::Input };
                        Ok(V::Sym(CExpr::Len(Box::new(x)), k))
                    }
                    (None, _) => self.not_data(span),
                }
            }
            Intrinsic::Panic => self.abort(Abort::Panic, span),
            Intrinsic::Abort(n) => self.abort(Abort::Called(n), span),
            Intrinsic::Derived(k) => match self.derived.get(&k) {
                Some(Some(v)) => Ok(V::Val(v.clone())),
                // why it has no value was reported where it is read
                _ => Err(Flow::Stop),
            },
            Intrinsic::Clone => {
                let v = self.expr(&args[0])?;
                Ok(self.deref(v))
            }
            Intrinsic::Exchange => {
                let (a, b) = (self.expr(&args[0])?, self.expr(&args[1])?);
                let (V::Ref(pa), V::Ref(pb)) = (a, b) else {
                    return self.not_data(span);
                };
                let (va, vb) = (self.read(&pa), self.read(&pb));
                let (wa, wb) = (va.wires(), vb.wires());
                if wa.is_empty() && wb.is_empty() {
                    self.store(&pa, vb);
                    self.store(&pb, va);
                    return Ok(V::Val(Value::Void));
                }
                if va.has_register() || vb.has_register() || wa.len() != wb.len() {
                    return self.fail(crate::sema::report::unsupported(
                        span,
                        "swapping registers whose length is known only as the circuit runs",
                    ));
                }
                // the states change places by gates, and each place keeps
                // its qubits, as `gates::swap` does; classical parts are
                // exchanged as values
                for (a, b) in wa.iter().zip(&wb) {
                    self.gate(GateOp::Swap, vec![*a, *b], Vec::new(), span)?;
                }
                self.store(&pa, vb.with_wires(&mut wa.iter().copied()));
                self.store(&pb, va.with_wires(&mut wb.iter().copied()));
                Ok(V::Val(Value::Void))
            }
            _ => {
                // what is pure is computed as constant evaluation computes it
                let mut consts = Vec::with_capacity(args.len());
                for a in args {
                    let v = self.expr(a)?;
                    match v {
                        V::Val(x) => consts.push(Expr::constant(x, a.ty, a.span)),
                        _ => return self.not_data(a.span),
                    }
                }
                let call = Expr::intrinsic(which, consts, e.ty, span);
                let mut ev = crate::sema::consteval::Evaluator::new(self.unit, self.interner);
                let mut frame = crate::sema::consteval::Frame::empty();
                match ev.top(&call, &mut frame) {
                    Ok(v) => Ok(V::Val(v)),
                    Err(crate::sema::consteval::EvalError::Abort { abort, .. }) => self.abort(abort, span),
                    Err(_) => self.fail(
                        crate::sema::report::unsupported(span, format_args!("`{}` while a circuit is generated", which.name())),
                    ),
                }
            }
        }
    }

    /// The qubits a handle names.
    fn handle_wires(&mut self, v: V, span: Span) -> R<Vec<Wire>> {
        let ws = self.deref(v).wires();
        if ws.is_empty() {
            return self.not_data(span);
        }
        Ok(ws)
    }

    /// An angle given as a `phase<N>`.
    fn phase_angle(&mut self, v: V, ty: Ty, span: Span) -> R<Angle> {
        let n = phase_order(self.unit, ty).unwrap_or(u64::from(self.conductor));
        let n32 = u32::try_from(n).unwrap_or(u32::MAX);
        if n32 == 0 || !self.conductor.is_multiple_of(n32) {
            let suffices = crate::exact::conductor_admitting(n32, self.conductor);
            let help = match suffices {
                Some(m) => format!("add `#pragma conductor({m})` to the unit"),
                None => "use a phase whose order divides the unit's conductor".to_owned(),
            };
            return self.fail(
                Diagnostic::new(Code::Ej04)
                    .with_message(format!(
                        "this angle is a `phase<{n}>`, but the unit's conductor is {}",
                        self.conductor
                    ))
                    .at(span)
                    .with_note(
                        "a circuit's phases are whole numbers of N-th parts of a turn for the unit's \
                         conductor N, so an angle must be one of them",
                    )
                    .with_help(help),
            );
        }
        let scale = self.conductor / n32;
        match v {
            V::Val(p) => {
                let m = crate::sema::exact::phase_of(&p).unwrap_or(0);
                Ok(Angle::Fixed(Phase::of(self.conductor, (m * u64::from(scale)) as i64)))
            }
            V::Sym(x, _) => {
                let m = CExpr::Field(Box::new(x), 0);
                let m = if scale == 1 { m } else { CExpr::Binary(BinOp::Mul, Box::new(m), Box::new(usize_lit(i128::from(scale)))) };
                Ok(Angle::Runtime(m))
            }
            _ => self.not_data(span),
        }
    }

    fn gate_intrinsic(&mut self, g: Gate, args: &[Expr], span: Span) -> R<V> {
        let mut it = self.exprs(args)?.into_iter();
        let angle = if g.takes_phase() {
            let v = it.next().expect("a phase gate takes a phase");
            Some(self.phase_angle(v, args[0].ty, args[0].span)?)
        } else {
            None
        };
        // each qubit operand names one qubit; the controls come first
        let mut qubits = Vec::new();
        for (v, a) in it.zip(args.iter().skip(usize::from(g.takes_phase()))) {
            qubits.push(self.handle_wires(v, a.span)?[0]);
        }
        let (op, controls) = match g {
            Gate::X => (GateOp::X, 0),
            Gate::Y => (GateOp::Y, 0),
            Gate::Z => (GateOp::Z, 0),
            Gate::H => (GateOp::H, 0),
            Gate::S => (GateOp::S, 0),
            Gate::Sdg => (GateOp::Sdg, 0),
            Gate::T => (GateOp::T, 0),
            Gate::Tdg => (GateOp::Tdg, 0),
            Gate::Cx => (GateOp::X, 1),
            Gate::Cz => (GateOp::Z, 1),
            Gate::Ccx => (GateOp::X, 2),
            Gate::Swap => (GateOp::Swap, 0),
            Gate::Rz => (GateOp::Phase(angle.expect("a phase")), 0),
            Gate::GPhase => (GateOp::GPhase(angle.expect("a phase")), 0),
        };
        let (controls, targets) = qubits.split_at(controls);
        self.gate(op, targets.to_vec(), controls.iter().copied().map(on).collect(), span)?;
        Ok(V::Val(Value::Void))
    }

    /// `apply(u, r)`: an exact matrix, checked to be unitary unless a block
    /// around says to trust it, applied to the qubits `r` names.
    fn apply(&mut self, args: &[Expr], span: Span) -> R<V> {
        let m = self.expr(&args[0])?;
        let r = self.expr(&args[1])?;
        let trusted = matches!(self.expr(&args[2])?, V::Val(Value::Bool(true)));
        let V::Val(m) = m else {
            return self.not_data(args[0].span);
        };

        // entries in a field the unit's conductor includes
        let n = matrix_order(self.unit, args[0].ty).unwrap_or(u64::from(self.conductor));
        let n32 = u32::try_from(n).unwrap_or(u32::MAX);
        if !self.conductor.is_multiple_of(n32) {
            let help = crate::exact::conductor_admitting(n32, self.conductor)
                .map_or_else(|| "use entries of a field the unit's conductor includes".to_owned(), |m| format!("add `#pragma conductor({m})` to the unit"));
            return self.fail(
                Diagnostic::new(Code::Ej04)
                    .with_message(format!(
                        "this matrix's entries are `cyclo<{n}>`, but the unit's conductor is {}",
                        self.conductor
                    ))
                    .at(args[0].span)
                    .with_help(help),
            );
        }

        // the matrix, its entries lifted to the conductor, and unitary unless trusted
        let rows = match &m {
            Value::Struct(fs) => match fs.first() {
                Some(Value::Array(rows)) => rows.clone(),
                _ => return self.not_data(span),
            },
            _ => return self.not_data(span),
        };
        let size = rows.len();
        let mut entries = Vec::with_capacity(size * size);
        for row in &rows {
            let Value::Array(cells) = row else { return self.not_data(span) };
            for cell in cells {
                let Some(c) = crate::sema::exact::cyclo_of(cell, n32) else { return self.not_data(span) };
                entries.push(c.embed(self.conductor).expect("the conductor is a multiple"));
            }
        }
        let matrix = Matrix { size, entries };
        if !trusted && let Some((i, j)) = not_unitary(&matrix, self.conductor) {
            return self.fail(
                Diagnostic::new(Code::Eq13)
                    .with_message("the matrix given to `apply` is not unitary")
                    .at(args[0].span)
                    .with_note(format!(
                        "a gate is a unitary matrix U, one with U†U = I; for this one, entry ({i}, {j}) of \
                         U†U is not what the identity has there, which was computed exactly"
                    ))
                    .with_help(
                        "correct the matrix, or, where it is unitary by an argument the compiler cannot \
                         follow, write the `apply` in a block marked `[trusted_unitary]`",
                    ),
            );
        }

        // applied to the qubits, under any controls in force
        let targets = self.handle_wires(r, args[1].span)?;
        self.acting("acts on qubits", span)?;
        let controls = self.controls.clone();
        self.check_operands(&targets, &controls, span)?;
        self.written.extend(&targets);
        self.push(Op::Unitary {
            matrix: std::sync::Arc::new(matrix),
            targets,
            controls,
        });
        Ok(V::Val(Value::Void))
    }

    fn growable(&mut self, op: GrowOp, array: &Expr, args: &[Expr], e: &Expr) -> R<V> {
        let span = e.span;
        let Some(p) = self.place_of(array)? else {
            return self.fail(crate::sema::report::unsupported(span, "changing a temporary growable array while a circuit is generated"));
        };
        let values = self.exprs(args)?;
        match self.read(&p) {
            V::Reg(reg) => return self.grow_register(op, reg, values, span),
            V::Sym(a, k) => return self.grow_runtime(op, a, k, &p, values, span),
            _ => {}
        }

        // an array known now: change it, and write it back
        let mut items = match self.read(&p).opened() {
            V::Agg { items, .. } => items,
            _ => Vec::new(),
        };
        let out = match op {
            GrowOp::Push => {
                items.push(values.into_iter().next().unwrap_or(V::Moved));
                V::Val(Value::Void)
            }
            GrowOp::Pop => {
                let last = items.pop();
                let Ty::Adt(opt) = e.ty else { return self.not_data(span) };
                let def = self.unit.types.adt(opt);
                let index = |name: &str| def.variants().iter().position(|v| self.interner.resolve(v.name) == name).unwrap_or(0) as u32;
                match last {
                    Some(v) => V::Variant(index("Some"), vec![v]).normal(),
                    None => V::Val(Value::Enum {
                        variant: index("None"),
                        fields: Vec::new(),
                    }),
                }
            }
            GrowOp::Clear => {
                items.clear();
                V::Val(Value::Void)
            }
            GrowOp::Truncate => {
                if let Some(V::Val(Value::Int(n, _))) = values.first() {
                    items.truncate(usize::try_from(*n).unwrap_or(usize::MAX));
                }
                V::Val(Value::Void)
            }
            GrowOp::Reserve => V::Val(Value::Void),
            GrowOp::Take | GrowOp::ForgetFront => {
                return self.fail(crate::sema::report::unsupported(span, format_args!("`{}` while a circuit is generated", op.name())));
            }
        };
        self.write(&p, V::Agg { array: true, items }.normal(), span)?;
        Ok(out)
    }

    /// An operation on a register whose length is known only as the circuit
    /// runs: it may grow, and nothing else changes its length.
    fn grow_register(&mut self, op: GrowOp, reg: u32, values: Vec<V>, span: Span) -> R<V> {
        match op {
            GrowOp::Push => {
                self.in_region("grows a register whose length is known only as the circuit runs", span, Code::Eq04)?;
                let Some(V::Wire(w)) = values.into_iter().next() else {
                    return self.not_data(span);
                };
                self.live.remove(&w);
                self.push(Op::Grow { reg, wire: w });
                Ok(V::Val(Value::Void))
            }
            GrowOp::Reserve => Ok(V::Val(Value::Void)),
            _ => self.fail(runtime_structure(
                span,
                &format!("`{}` on a register whose length is known only when the circuit runs", op.name()),
                "such a register can grow, have its qubits named by index, be measured or be forgotten; \
                 which qubit would leave it is not fixed while the circuit is generated",
            )),
        }
    }

    /// An operation on a classical array computed as the circuit runs: it
    /// may grow, or be emptied.
    fn grow_runtime(&mut self, op: GrowOp, a: CExpr, k: Kind, p: &Place, values: Vec<V>, span: Span) -> R<V> {
        match op {
            GrowOp::Push => {
                let Some((x, kx)) = values.first().and_then(V::classical) else {
                    return self.not_data(span);
                };
                let k = Kind::join(k, kx.unwrap_or(k));
                self.write(p, V::Sym(CExpr::Append(Box::new(a), Box::new(x)), k), span)?;
                Ok(V::Val(Value::Void))
            }
            GrowOp::Clear => {
                self.write(p, V::Val(Value::Array(Vec::new())), span)?;
                Ok(V::Val(Value::Void))
            }
            GrowOp::Reserve => Ok(V::Val(Value::Void)),
            _ => self.fail(runtime_structure(
                span,
                &format!("`{}` on an array computed only when the circuit runs", op.name()),
                "while the circuit runs, such an array can grow or be emptied",
            )),
        }
    }

    /// A place naming the qubit at `index` of the register `reg`, whose
    /// length is known only as the circuit runs: a wire made to name it then.
    fn element(&mut self, reg: u32, index: &Expr) -> R<Place> {
        let i = self.expr(index)?;
        let Some((i, k)) = i.classical() else {
            return self.not_data(index.span);
        };
        if k == Some(Kind::Outcome) {
            return self.fail(runtime_structure(
                index.span,
                "which qubit of the register this names depends on a measured value",
                "a measured value may choose what happens as the circuit runs, but not which qubits it acts on",
            ));
        }
        let wire = self.new_wire();
        self.push(Op::Element { wire, reg, index: i });
        Ok(Place::Temp(Box::new(V::Wire(wire)), Vec::new()))
    }

    /// Before a loop whose count is known only as the circuit runs, makes
    /// each growable register of qubits the loop's `body` names, held outside
    /// it, a register whose length is known only then, so that the body,
    /// generated once, can grow it and name its qubits by index.
    fn grow_registers(&mut self, body: &Block) {
        for e in registers_named(&self.unit.types, body) {
            let Some(p) = self.static_place(&e) else { continue };
            let v = self.read(&p);
            if matches!(v, V::Reg(_) | V::Moved) {
                continue;
            }
            let reg = self.c.grown;
            self.c.grown += 1;
            for w in v.wires() {
                self.live.remove(&w);
                self.push(Op::Grow { reg, wire: w });
            }
            self.store(&p, V::Reg(reg));
        }
    }

    /// Where `e` is held, when no index need be computed to say.
    fn static_place(&self, e: &Expr) -> Option<Place> {
        let f = self.frame();
        match &e.kind {
            ExprKind::Local(l) => Some(Place::Local {
                frame: f,
                local: *l,
                path: Vec::new(),
            }),
            ExprKind::Field { base, field } => self.static_place(base).map(|p| p.child(*field)),
            ExprKind::Deref(r) => match &r.kind {
                ExprKind::Local(l) => match &self.frames[f].locals[l.index()] {
                    V::Ref(p) => Some(p.clone()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    ////////////////////////
    // QUANTUM OPERATIONS //
    ////////////////////////

    fn quantum(&mut self, op: &QuantumOp, args: &[Expr], e: &Expr) -> R<V> {
        let span = e.span;
        match op {
            QuantumOp::Measure => {
                self.in_region("measures", span, Code::Eq04)?;
                let v = self.expr(&args[0])?;
                let bits = self.measure(v, span)?;
                Ok(self.measured_numbers(args[0].ty, bits))
            }
            QuantumOp::Forget => {
                self.in_region("forgets", span, Code::Eq05)?;
                let v = self.expr(&args[0])?;
                self.acting("forgets qubits", span)?;
                if let V::Reg(reg) = v {
                    let (var, wire) = (self.new_var(), self.new_wire());
                    self.push(Op::For {
                        var,
                        start: usize_lit(0),
                        end: CExpr::RegLen(reg),
                        body: vec![
                            Op::Element {
                                wire,
                                reg,
                                index: CExpr::Var(var),
                            },
                            Op::Forget { wire },
                        ],
                    });
                    return Ok(V::Val(Value::Void));
                }
                for w in v.wires() {
                    self.live.remove(&w);
                    self.push(Op::Forget { wire: w });
                }
                Ok(V::Val(Value::Void))
            }
            QuantumOp::Lift => {
                self.in_region("lifts a measured value", span, Code::Eq04)?;
                let v = self.expr(&args[0])?;
                match v {
                    V::Sym(x, _) => {
                        let var = self.new_var();
                        self.push(Op::Lift { var, value: x });
                        self.c.dynamic = true;
                        Ok(V::Sym(CExpr::Var(var), Kind::Input))
                    }
                    v => Ok(v),
                }
            }
            QuantumOp::Tensor => {
                let a = self.expr(&args[0])?;
                let b = self.expr(&args[1])?;
                if a.has_register() || b.has_register() {
                    return self.fail(runtime_structure(
                        span,
                        "this joins a register whose length is known only when the circuit runs",
                        "the qubits of a register joined with `**` are fixed while the circuit is generated",
                    ));
                }
                let items = a.wires().into_iter().chain(b.wires()).map(V::Wire).collect();
                Ok(V::Agg { array: true, items })
            }
            QuantumOp::Replay => self.replay(&args[0], span),
            QuantumOp::Flip => self.flip(&args[0], &args[1], span),
            QuantumOp::Copy => self.qcopy(&args[0], span),
            QuantumOp::Query(map) => self.query(*map, &args[0], span),
            QuantumOp::Lookup(map) => self.lookup(*map, &args[0], span),
            QuantumOp::Table(map) => Ok(V::Table(*map)),
            QuantumOp::QueryAt | QuantumOp::LookupAt => {
                let t = self.expr(&args[0])?;
                let V::Table(map) = self.deref(t) else { return self.not_data(span) };
                if matches!(op, QuantumOp::QueryAt) {
                    self.query(map, &args[1], span)
                } else {
                    self.lookup(map, &args[1], span)
                }
            }
            QuantumOp::Prep(prepared) => {
                self.in_region("prepares a state", span, Code::Eq04)?;
                let wires = self.alloc_n(prepared.width as u64, span)?;
                self.acting("prepares a state", span)?;
                for op in prepared.on(&wires) {
                    self.written.extend(op.writes());
                    self.push(op);
                }
                Ok(V::Agg {
                    array: true,
                    items: wires.into_iter().map(V::Wire).collect(),
                })
            }
        }
    }

    /// `query t[k]` of the map locale `map`, `key` a handle to the key: an
    /// operator applying each entry controlled on its key, or, for a map
    /// locale of states, a register prepared in each entry's state
    /// controlled on its key, entangled with the key register.
    fn query(&mut self, map: usize, key: &Expr, span: Span) -> R<V> {
        let key = self.expr(key)?;
        let wires = self.handle_wires(key, span)?;
        let def = &self.unit.geometry.qmaps[map].def;
        if !self.unit.types.is_quantum(def.entry) {
            return Ok(V::Functor(Box::new(Functor::Query(map, wires))));
        }
        let (entry, entries) = (def.entry, def.entries.clone());
        self.in_region("prepares a state", span, Code::Eq04)?;
        let register = self.alloc_n(self.unit.types.qubits(entry), span)?;
        self.acting("prepares a state", span)?;
        for (k, e) in &entries {
            let controls = key_controls(&wires, *k);
            for mut op in prepared_ops(e, &register) {
                match &mut op {
                    Op::Gate { controls: c, .. } | Op::Unitary { controls: c, .. } => c.extend(controls.iter().copied()),
                    _ => {}
                }
                self.written.extend(op.writes());
                self.push(op);
            }
        }
        Ok(shape_register(entry, &register))
    }

    /// `measure t[k]` of the map locale `map`, `key` the key register: the
    /// index it held, measured, and the entry there (for a map locale of
    /// states, a register prepared in that entry's state).
    fn lookup(&mut self, map: usize, key: &Expr, span: Span) -> R<V> {
        self.in_region("measures", span, Code::Eq04)?;
        let key = self.expr(key)?;
        let measured = self.measure(key, span)?;
        let bits: Vec<CExpr> = match measured {
            V::Agg { items, .. } => items
                .into_iter()
                .filter_map(|v| match v {
                    V::Sym(x, _) => Some(x),
                    _ => None,
                })
                .collect(),
            V::Sym(x, _) => vec![x],
            _ => Vec::new(),
        };
        // the index: each bit worth its power of two
        let width = bits.len();
        let index = bits
            .iter()
            .enumerate()
            .map(|(i, b)| CExpr::Select(Box::new(b.clone()), Box::new(usize_lit(1 << (width - 1 - i))), Box::new(usize_lit(0))))
            .reduce(|a, b| CExpr::Binary(BinOp::Add, Box::new(a), Box::new(b)))
            .unwrap_or(usize_lit(0));
        let def = &self.unit.geometry.qmaps[map].def;
        let entry = if self.unit.types.is_quantum(def.entry) {
            let (ty, entries) = (def.entry, def.entries.clone());
            let register = self.alloc_n(self.unit.types.qubits(ty), span)?;
            for (k, e) in &entries {
                let ops = prepared_ops(e, &register);
                for op in &ops {
                    self.written.extend(op.writes());
                }
                self.push(Op::If {
                    cond: CExpr::Binary(BinOp::Eq, Box::new(index.clone()), Box::new(usize_lit(i128::from(*k)))),
                    then: ops,
                    els: Vec::new(),
                });
            }
            shape_register(ty, &register)
        } else {
            V::Functor(Box::new(Functor::Lookup(map, bits)))
        };
        Ok(V::Agg {
            array: false,
            items: vec![V::Sym(index, Kind::Outcome), entry],
        })
    }

    /// `replay f(args)`: the call made again, which is a second preparation
    /// of one state only when `f` is monic and every argument is known
    /// while the circuit is generated (`EJ02` otherwise).
    fn replay(&mut self, call: &Expr, span: Span) -> R<V> {
        // an operator known now, and arguments known now
        let (target, cargs) = match &call.kind {
            ExprKind::Call { callee: Callee::Fn(f), args } => (V::Func(*f), args),
            ExprKind::IndirectCall { callee, args } => {
                let t = self.expr(callee)?;
                (self.deref(t), args)
            }
            _ => return self.foreign(span),
        };
        self.needs_static(&target, "`replay`", span)?;
        let mut values = Vec::with_capacity(cargs.len());
        for a in cargs {
            let v = self.expr(a)?;
            if !matches!(v, V::Val(_)) {
                return self.fail(
                    Diagnostic::new(Code::Ej02)
                        .with_message("`replay` needs every argument known while the circuit is generated")
                        .at_with(a.span, "this is not")
                        .with_note(
                            "a replay prepares again the state a monic operator prepared from the same \
                             arguments; an argument holding qubits, or known only as the circuit runs, is \
                             not the same argument twice, and replaying it would amount to copying a state",
                        ),
                );
            }
            values.push(v);
        }

        // the call recorded apart, kept only if every step is monic
        self.ops.push(Vec::new());
        let r = self.call_value(target, values, span);
        let recorded = self.ops.pop().unwrap_or_default();
        let v = r?;
        if let Some((op, at)) = recorded.iter().find(|(op, _)| !op.is_monic()) {
            let what = match op {
                Op::Measure { .. } | Op::MeasureAll { .. } => "measures",
                Op::Forget { .. } => "forgets",
                Op::Lift { .. } => "lifts a measured value",
                _ => "measures under feed-forward",
            };
            return self.fail(
                Diagnostic::new(Code::Ej02)
                    .with_message(format!("`replay` needs a monic operator, and this one {what}"))
                    .at_with(span, "replayed here")
                    .also(*at, "here")
                    .with_note(
                        "a monic operator keeps everything it acts on, so the same arguments always give \
                         the same state; one that measures or forgets gives a state that depends on what \
                         it observed",
                    ),
            );
        }
        self.ops.last_mut().expect("an operation list is open").extend(recorded);
        Ok(v)
    }

    /// `qcopy(&x)`: new qubits prepared as those `x` holds were, by
    /// repeating on fresh qubits every operation that made them what they
    /// are. Each qubit of `x` must have been allocated in this operator,
    /// among the operations open now; what prepared them may read other
    /// qubits only if those were allocated since and given back in |0>, which
    /// are allocated afresh and given back again; and every operation must be
    /// a gate or a matrix, so that repeating it repeats its effect (`EJ19`
    /// otherwise). A value its operator was given, or one joined to qubits
    /// that outlive the copy, has a state the operator does not know. A
    /// classical part is copied as it is, or by its type's `$copy` where it
    /// has one.
    fn qcopy(&mut self, handle: &Expr, span: Span) -> R<V> {
        self.in_region("prepares a state", span, Code::Eq04)?;
        let h = self.expr(handle)?;
        let V::Ref(place) = h else { return self.not_data(span) };
        let v = self.read(&place);
        if v.has_register() {
            return self.fail(crate::sema::report::unsupported(
                span,
                "copying a register whose length is known only as the circuit runs",
            ));
        }
        let wires = v.wires();
        if wires.is_empty() {
            return Ok(v);
        }
        let depth = self.ops.len() - 1;
        let (from_depth, from) = self.frames.last().map_or((0, 0), |f| f.start);
        let first = if from_depth == depth { from } else { 0 };
        let list = self.ops[depth].clone();

        // the allocation of each qubit `x` holds, among the operations open
        let mut born = Vec::with_capacity(wires.len());
        for &w in &wires {
            match list.iter().rposition(|(op, _)| matches!(op, Op::Alloc { wire } if *wire == w)) {
                Some(i) if i >= first => born.push(i),
                _ => {
                    return self.uncopyable(
                        span,
                        None,
                        "it holds qubits this operator did not allocate here, whose state it does not know",
                    );
                }
            }
        }
        let start = *born.iter().min().expect("a qubit at least");

        // each qubit an operation touches, as the allocation it belongs to:
        // `None` for one allocated before the history began
        type Held = (Wire, Option<usize>);
        let mut current: HashMap<Wire, Option<usize>> = HashMap::new();
        let mut touched: Vec<Vec<Held>> = Vec::with_capacity(list.len() - start);
        let mut released: HashSet<Held> = HashSet::new();
        for (i, (op, _)) in list.iter().enumerate().skip(start) {
            if let Op::Alloc { wire } = op {
                current.insert(*wire, Some(i));
            }
            let held: Vec<Held> = op.wires().into_iter().map(|w| (w, current.get(&w).copied().flatten())).collect();
            if let Op::Release { wire } = op {
                released.insert((*wire, current.get(wire).copied().flatten()));
            }
            touched.push(held);
        }

        // the qubits whose history is the copy's: those of `x`, and those
        // allocated since, joined to them, and given back in |0>
        let mut closure: HashSet<Held> = wires.iter().zip(&born).map(|(&w, &i)| (w, Some(i))).collect();
        loop {
            let mut grew = false;
            for (k, held) in touched.iter().enumerate() {
                if !held.iter().any(|x| closure.contains(x)) {
                    continue;
                }
                let (op, at) = &list[start + k];
                let what = match op {
                    Op::Gate { .. } | Op::Unitary { .. } | Op::Alloc { .. } | Op::Release { .. } => None,
                    Op::Measure { .. } | Op::MeasureAll { .. } => Some("it was measured from"),
                    Op::Forget { .. } => Some("a qubit it was made from was forgotten"),
                    Op::If { .. } => Some("what was done to it depends on a measured value"),
                    Op::For { .. } | Op::Grow { .. } | Op::Element { .. } => {
                        Some("it was made in a loop the circuit repeats as it runs")
                    }
                    Op::Call { .. } => Some("an operator supplied when the circuit runs acted on it"),
                    Op::Let { .. } | Op::Lift { .. } => None,
                };
                if let Some(why) = what {
                    return self.uncopyable(span, Some(*at), why);
                }
                for x in held {
                    if closure.contains(x) {
                        continue;
                    }
                    if x.1.is_some() && released.contains(x) {
                        closure.insert(*x);
                        grew = true;
                    } else {
                        return self.uncopyable(
                            span,
                            Some(*at),
                            "it is joined here to a qubit outside it, which a copy could not reproduce \
                             without disturbing",
                        );
                    }
                }
            }
            if !grew {
                break;
            }
        }

        // the history repeated on fresh qubits
        let mut fresh: HashMap<Held, Wire> = HashMap::new();
        for (k, held) in touched.iter().enumerate() {
            if !held.iter().any(|x| closure.contains(x)) {
                continue;
            }
            let map = |fresh: &HashMap<Held, Wire>, w: Wire| -> Wire {
                let x = held.iter().find(|x| x.0 == w).copied().expect("touched");
                fresh[&x]
            };
            match &list[start + k].0 {
                Op::Alloc { wire } => {
                    let f = self.alloc(span)?;
                    fresh.insert((*wire, Some(start + k)), f);
                }
                Op::Release { wire } => {
                    let f = map(&fresh, *wire);
                    self.release(f);
                }
                Op::Gate { gate, targets, controls } => {
                    let targets = targets.iter().map(|&t| map(&fresh, t)).collect();
                    let controls = controls.iter().map(|c| Control { wire: map(&fresh, c.wire), on: c.on }).collect();
                    self.gate(gate.clone(), targets, controls, span)?;
                }
                Op::Unitary { matrix, targets, controls } => {
                    let targets: Vec<Wire> = targets.iter().map(|&t| map(&fresh, t)).collect();
                    let controls = controls.iter().map(|c| Control { wire: map(&fresh, c.wire), on: c.on }).collect();
                    self.written.extend(&targets);
                    self.push(Op::Unitary {
                        matrix: matrix.clone(),
                        targets,
                        controls,
                    });
                }
                _ => {}
            }
        }
        let copied: Vec<Wire> = wires.iter().zip(&born).map(|(&w, &i)| fresh[&(w, Some(i))]).collect();
        let mut out = v.with_wires(&mut copied.into_iter());

        // a classical part whose type has `$copy` is copied by it
        let ty = self.unit.types.as_ref(handle.ty).map_or(handle.ty, |(_, t)| t);
        let mut parts = Vec::new();
        self.copied_by_method(ty, &mut Vec::new(), &mut parts);
        for (path, callee) in parts {
            let at = path.iter().fold(place.clone(), |p, &i| p.child(i));
            let made = match callee {
                Callee::Fn(f) => self.call(f, vec![V::Ref(at)], span)?,
                Callee::Extern(_) => return self.foreign(span),
            };
            out.replace(&path, made);
        }
        Ok(out)
    }

    /// The parts of a value of `ty`, at `path` within it, whose type defines
    /// `$copy`, with each one's `$copy`: those of structures, arrays and
    /// tuples, followed in until a type with one is met. An enumeration's
    /// variant is not known while the circuit is generated, so its parts are
    /// not followed.
    fn copied_by_method(&self, ty: Ty, path: &mut Vec<u32>, out: &mut Vec<(Vec<u32>, Callee)>) {
        let types = &self.unit.types;
        let mut part = |g: &Self, i: u64, t: Ty, out: &mut Vec<(Vec<u32>, Callee)>| {
            path.push(i as u32);
            g.copied_by_method(t, path, out);
            path.pop();
        };
        if let Ty::Adt(id) = ty {
            if let Some(&c) = self.unit.copies.get(&id) {
                out.push((path.clone(), c));
                return;
            }
            let def = types.adt(id);
            if def.is_struct() {
                for (i, f) in def.fields().iter().enumerate() {
                    part(self, i as u64, f.ty, out);
                }
            }
        } else if let Some((elem, n)) = types.as_array(ty) {
            for i in 0..n {
                part(self, i, elem, out);
            }
        } else if let Some(elems) = types.as_tuple(ty) {
            for (i, &t) in elems.iter().enumerate() {
                part(self, i as u64, t, out);
            }
        }
    }

    /// Refuses a `qcopy`, saying why, and where in the history if that is
    /// one operation.
    fn uncopyable<T>(&mut self, span: Span, at: Option<Span>, why: &str) -> R<T> {
        let mut d = Diagnostic::new(Code::Ej19)
            .with_message(format!("this value cannot be copied: {why}"))
            .at_with(span, "copied here")
            .with_note(
                "a copy is a second preparation, made by repeating what prepared the value; that is \
                 possible only for a value this operator prepared from |0> by gates alone, joined to \
                 no qubit that outlives it, so that its state is known",
            )
            .with_help("copy the value where it is prepared, or `replay` the operator that prepares it");
        if let Some(at) = at {
            d = d.also(at, "here");
        }
        self.fail(d)
    }

    fn new_var(&mut self) -> Var {
        let v = Var(self.c.vars);
        self.c.vars += 1;
        v
    }

    /// Measures the qubits a value holds, giving what is observed.
    fn measure(&mut self, v: V, span: Span) -> R<V> {
        self.acting("measures", span)?;
        match v {
            V::Wire(w) => Ok(self.measure_wire(w)),
            V::Agg { array, items } => {
                let mut out = Vec::with_capacity(items.len());
                for i in items {
                    // a structure's classical field not known here stays so
                    out.push(if i == V::Moved { i } else { self.measure(i, span)? });
                }
                Ok(V::Agg { array, items: out }.normal())
            }
            // a quantum structure's classical field is kept as it is
            v @ (V::Val(_) | V::Sym(..)) => Ok(v),
            V::Reg(reg) => {
                let var = self.new_var();
                self.push(Op::MeasureAll { reg, var });
                Ok(V::Sym(CExpr::Var(var), Kind::Outcome))
            }
            _ => self.not_data(span),
        }
    }

    /// The width of `t` when it is the `gates` library's `quint<N>`.
    fn quint_width(&self, t: Ty) -> Option<u64> {
        let Ty::Adt(id) = t else { return None };
        let def = self.unit.types.adt(id);
        let in_gates = def.origin.as_deref().unwrap_or(&self.unit.name) == "gates";
        if !in_gates || self.name(def.name) != "quint" {
            return None;
        }
        match def.args.first() {
            Some(crate::tir::Arg::Const(n)) => u64::try_from(*n).ok(),
            _ => None,
        }
    }

    /// Whether a value of `t` holds a `quint` anywhere.
    fn holds_quint(&self, t: Ty) -> bool {
        if self.quint_width(t).is_some() {
            return true;
        }
        if let Ty::Adt(id) = t {
            let def = self.unit.types.adt(id);
            return def.is_struct() && def.fields().iter().any(|f| self.holds_quint(f.ty));
        }
        self.unit.types.as_array(t).is_some_and(|(elem, _)| self.holds_quint(elem))
    }

    /// What measuring a value of the quantum type `t` gives, from the bits
    /// `v` it was measured into: each `quint` among them the number its bits
    /// spell, the first the most significant.
    fn measured_numbers(&self, t: Ty, v: V) -> V {
        if !self.holds_quint(t) {
            return v;
        }
        if let Some(n) = self.quint_width(t) {
            let int = crate::sema::quantum::quint_measured(n);
            let bits = v.part(0).unwrap_or(V::Moved);
            let mut acc = CExpr::Value(Value::Int(0, int));
            for i in 0..n {
                let bit = match bits.part(i as u32) {
                    Some(V::Sym(c, _)) => c,
                    Some(V::Val(b)) => CExpr::Value(b),
                    _ => CExpr::Value(Value::Bool(false)),
                };
                let one = CExpr::Select(
                    Box::new(bit),
                    Box::new(CExpr::Value(Value::Int(1, int))),
                    Box::new(CExpr::Value(Value::Int(0, int))),
                );
                let doubled = CExpr::Binary(BinOp::Mul, Box::new(acc), Box::new(CExpr::Value(Value::Int(2, int))));
                acc = CExpr::Binary(BinOp::Add, Box::new(doubled), Box::new(one));
            }
            return V::Sym(acc, Kind::Outcome);
        }
        let parts: Vec<Ty> = match t {
            Ty::Adt(id) => self.unit.types.adt(id).fields().iter().map(|f| f.ty).collect(),
            _ => match self.unit.types.as_array(t) {
                Some((elem, n)) => vec![elem; n as usize],
                None => return v,
            },
        };
        let array = !matches!(t, Ty::Adt(_));
        let items = parts
            .into_iter()
            .enumerate()
            .map(|(i, pt)| self.measured_numbers(pt, v.part(i as u32).unwrap_or(V::Moved)))
            .collect();
        V::Agg { array, items }.normal()
    }

    fn measure_wire(&mut self, w: Wire) -> V {
        let bit = crate::circuit::Bit(self.c.bits);
        self.c.bits += 1;
        self.live.remove(&w);
        self.push(Op::Measure { wire: w, bit });
        V::Sym(CExpr::Bit(bit), Kind::Outcome)
    }

    /// A variant of a quantum enumeration: its tag set to the variant, and its
    /// payload the variant's fields, with fresh qubits beyond them.
    fn inject(&mut self, adt: crate::tir::AdtId, variant: u32, fields: Vec<V>, span: Span) -> R<V> {
        let (tw, pw) = self.unit.types.enum_qubits(adt);
        let mut tag = Vec::with_capacity(tw as usize);
        for i in 0..tw {
            let w = self.alloc(span)?;
            if (u64::from(variant) >> (tw - 1 - i)) & 1 == 1 {
                self.gate(GateOp::X, vec![w], vec![], span)?;
            }
            tag.push(w);
        }
        let mut payload: Vec<Wire> = Vec::with_capacity(pw as usize);
        for ((ty, _, w), v) in self.variant_layout(adt, variant).into_iter().zip(&fields) {
            if self.unit.types.is_quantum(ty) {
                payload.extend(v.wires());
                continue;
            }
            let ws = self.alloc_n(w as u64, span)?;
            self.encode(ty, v, &ws, span)?;
            payload.extend(ws);
        }
        while (payload.len() as u64) < pw {
            payload.push(self.alloc(span)?);
        }
        Ok(V::QEnum { adt, tag, payload })
    }

    /// Puts the classical value `v`, of type `ty`, on the fresh qubits `ws`
    /// as their basis state, its first bit the most significant.
    fn encode(&mut self, ty: Ty, v: &V, ws: &[Wire], span: Span) -> R<()> {
        let mut bits = Vec::with_capacity(ws.len());
        self.bits_of(ty, v, &mut bits, span)?;
        for (&w, bit) in ws.iter().zip(bits) {
            match bit {
                CExpr::Value(Value::Bool(false)) => {}
                CExpr::Value(Value::Bool(true)) => self.gate(GateOp::X, vec![w], vec![], span)?,
                // a bit known only as the circuit runs sets its qubit then
                cond => {
                    self.acting("sets a qubit", span)?;
                    self.written.insert(w);
                    let flip = Op::Gate {
                        gate: GateOp::X,
                        targets: vec![w],
                        controls: self.controls.clone(),
                    };
                    self.push(Op::If {
                        cond,
                        then: vec![flip],
                        els: Vec::new(),
                    });
                }
            }
        }
        Ok(())
    }

    /// The bits of the classical value `v` of type `ty`, the most
    /// significant first, each known now or as the circuit runs.
    fn bits_of(&mut self, ty: Ty, v: &V, out: &mut Vec<CExpr>, span: Span) -> R<()> {
        let types = &self.unit.types;
        match ty {
            Ty::Bool => match v.classical() {
                Some((e, _)) => out.push(e),
                None => return self.not_data(span),
            },
            Ty::Int(it) => {
                let n = u32::from(it.bits);
                match v {
                    V::Val(Value::Int(x, _)) => {
                        out.extend((0..n).map(|k| CExpr::Value(Value::Bool((x >> (n - 1 - k)) & 1 == 1))));
                    }
                    V::Sym(e, _) => {
                        let int = |i: i128| Box::new(CExpr::Value(Value::Int(i, it)));
                        for k in 0..n {
                            let shifted = CExpr::Binary(BinOp::Shr, Box::new(e.clone()), int(i128::from(n - 1 - k)));
                            let bit = CExpr::Binary(BinOp::BitAnd, Box::new(shifted), int(1));
                            out.push(CExpr::Binary(BinOp::Ne, Box::new(bit), int(0)));
                        }
                    }
                    _ => return self.not_data(span),
                }
            }
            _ => {
                let Some((parts, _)) = parts(types, ty) else { return self.not_data(span) };
                for (i, t) in parts.into_iter().enumerate() {
                    let Some(part) = v.part(i as u32) else { return self.not_data(span) };
                    self.bits_of(t, &part, out, span)?;
                }
            }
        }
        Ok(())
    }

    /// The classical value of type `ty` held on `ws` as their basis state,
    /// measured, the first qubit its most significant bit.
    fn decode(&mut self, ty: Ty, ws: &[Wire]) -> CExpr {
        let types = &self.unit.types;
        match ty {
            Ty::Bool => match self.measure_wire(ws[0]) {
                V::Sym(b, _) => b,
                _ => unreachable!("a measurement gives a bit"),
            },
            Ty::Int(it) => {
                let n = ws.len();
                let int = |i: i128| Box::new(CExpr::Value(Value::Int(i, it)));
                let mut sum: Option<CExpr> = None;
                for (k, &w) in ws.iter().enumerate() {
                    let V::Sym(b, _) = self.measure_wire(w) else { unreachable!("a measurement gives a bit") };
                    // two's complement: a signed type's first bit weighs
                    // −2^(n−1)
                    let place = 1i128 << (n - 1 - k);
                    let weight = if it.signed && k == 0 { -place } else { place };
                    let term = CExpr::Select(Box::new(b), int(weight), int(0));
                    sum = Some(match sum {
                        None => term,
                        Some(s) => CExpr::Binary(BinOp::Add, Box::new(s), Box::new(term)),
                    });
                }
                sum.unwrap_or(CExpr::Value(Value::Int(0, it)))
            }
            _ => {
                let (parts, array) = parts(types, ty).unwrap_or_default();
                let mut at = 0usize;
                let mut items = Vec::with_capacity(parts.len());
                for t in parts {
                    let w = self.unit.types.basis_bits(t).unwrap_or(0) as usize;
                    items.push(self.decode(t, &ws[at..at + w]));
                    at += w;
                }
                if array { CExpr::Array(items) } else { CExpr::Struct(items) }
            }
        }
    }

    /////////////
    // CONTROL //
    /////////////

    fn is_quantum_condition(&self, t: Ty) -> bool {
        t == Ty::Qubit || self.unit.types.as_ref(t).is_some_and(|(_, x)| x == Ty::Qubit)
    }

    fn if_expr(&mut self, cond: &Expr, then: &Block, els: Option<&Expr>, span: Span) -> R<V> {
        if self.is_quantum_condition(cond.ty) {
            return self.quantum_if(cond, then, els, span);
        }
        match self.expr(cond)? {
            V::Val(Value::Bool(true)) => self.block(then),
            V::Val(Value::Bool(false)) => match els {
                Some(e) => self.expr(e),
                None => Ok(V::Val(Value::Void)),
            },
            V::Sym(c, _) => self.feed_forward(c, |g| g.block(then), |g| match els {
                Some(e) => g.expr(e),
                None => Ok(V::Val(Value::Void)),
            }),
            _ => self.not_data(cond.span),
        }
    }

    /// Both branches of feed-forward, generated under `cond`, and the
    /// bindings either changes made to hold whichever value the run chose.
    fn feed_forward(
        &mut self,
        cond: CExpr,
        then: impl FnOnce(&mut Self) -> R<V>,
        els: impl FnOnce(&mut Self) -> R<V>,
    ) -> R<V> {
        let before = Snapshot {
            frames: self.frames.clone(),
            globals: self.globals.clone(),
            live: self.live.clone(),
        };
        self.ff += 1;
        self.ops.push(Vec::new());
        let t = then(self);
        let tops = self.pop_ops();
        let after_then = Snapshot {
            frames: std::mem::replace(&mut self.frames, before.frames.clone()),
            globals: std::mem::replace(&mut self.globals, before.globals.clone()),
            live: std::mem::replace(&mut self.live, before.live.clone()),
        };
        self.ops.push(Vec::new());
        let e = els(self);
        let eops = self.pop_ops();
        self.ff -= 1;
        let (t, e, returning) = match (t, e) {
            (Err(Flow::Stop), _) | (_, Err(Flow::Stop)) => return Err(Flow::Stop),
            (Ok(t), Ok(e)) => (t, e, false),
            (Err(Flow::Return(t)), Err(Flow::Return(e))) => (t, e, true),
            _ => {
                return self.fail(runtime_structure(
                    self.frames.last().map_or(Span::synthetic(), |f| self.unit.func(f.func).span),
                    "whether control leaves here depends on a measured value",
                    "a `return`, `break` or `continue` on one side of an `if` on a measured value, and \
                     not on the other, would leave the rest of the circuit to be chosen as it runs",
                ));
            }
        };
        // merge the two ends
        if after_then.live != self.live {
            return self.fail(runtime_structure(
                self.frames.last().map_or(Span::synthetic(), |f| self.unit.func(f.func).span),
                "which qubits are held after this depends on a measured value",
                "the two sides of an `if` on a measured value must end holding the same qubits",
            ));
        }
        for (fi, (tf, ef)) in after_then.frames.iter().zip(self.frames.clone().iter()).enumerate() {
            for (li, (tv, ev)) in tf.locals.iter().zip(&ef.locals).enumerate() {
                if tv != ev {
                    let Some(m) = merge(&cond, tv, ev) else {
                        return self.fail(runtime_structure(
                            self.unit.func(tf.func).locals[li].span,
                            "which qubits this binding holds depends on a measured value",
                            "a binding holding qubits must hold the same ones whichever side of an \
                             `if` on a measured value runs",
                        ));
                    };
                    self.frames[fi].locals[li] = m;
                }
            }
        }
        for (id, tv) in after_then.globals {
            let ev = self.global_value(id);
            if tv != ev {
                let Some(m) = merge(&cond, &tv, &ev) else { return self.not_data(Span::synthetic()) };
                self.globals.insert(id, m);
            }
        }
        if !tops.is_empty() || !eops.is_empty() {
            self.push(Op::If {
                cond: cond.clone(),
                then: tops,
                els: eops,
            });
        }
        let Some(v) = merge(&cond, &t, &e) else {
            return self.not_data(Span::synthetic());
        };
        if returning { Err(Flow::Return(v)) } else { Ok(v) }
    }

    /// A quantum `if`: `then` controlled on the condition, `els` on its
    /// negation.
    fn quantum_if(&mut self, cond: &Expr, then: &Block, els: Option<&Expr>, span: Span) -> R<V> {
        let mut undo = Vec::new();
        let mut lits = self.condition(cond, &mut undo)?;
        if els.is_some()
            && let Lits::Conj(cs) = &lits
            && cs.len() > 1
        {
            let anc = self.conjunction(cs.clone(), &mut undo, cond.span)?;
            lits = Lits::Conj(vec![on(anc)]);
        }
        let reads = condition_reads(&lits, &undo);
        let result = match lits {
            Lits::Always => self.block(then).map(|_| ()),
            Lits::Never => match els {
                Some(e) => self.expr(e).map(|_| ()),
                None => Ok(()),
            },
            Lits::Conj(cs) => {
                let negated: Vec<Control> = cs.iter().map(|c| Control { wire: c.wire, on: !c.on }).collect();
                self.region(span, RegionKind::Branch, cs, &reads, |g| g.block(then))?;
                if let Some(e) = els {
                    self.region(span, RegionKind::Branch, negated, &reads, |g| g.expr(e))?;
                }
                Ok(())
            }
        };
        result?;
        self.undo(undo);
        Ok(V::Val(Value::Void))
    }

    /// Runs `body` as a quantum branch controlled on `controls`, checking
    /// that it leaves the qubits live as it found them.
    fn region(
        &mut self,
        span: Span,
        kind: RegionKind,
        controls: Vec<Control>,
        reads: &[Wire],
        body: impl FnOnce(&mut Self) -> R<V>,
    ) -> R<()> {
        let outer = self.controls.len();
        let mut region = Region::functor(span, outer, &self.live, self.frame(), kind);
        region.reads = reads.to_vec();
        self.regions.push(region);
        self.controls.extend(controls);
        let r = body(self);
        self.controls.truncate(outer);
        let region = self.regions.pop().expect("the region pushed above");
        let v = r?;
        if v.classical().is_some_and(|(_, k)| k.is_some()) || !matches!(v, V::Val(Value::Void)) && !v.wires().is_empty() {
            return self.fail(
                Diagnostic::new(Code::Eq04)
                    .with_message(format!("{} gives a value", kind.what()))
                    .at(span)
                    .with_note(
                        "it is applied controlled on qubits in superposition, so there is no one value \
                         it gives",
                    ),
            );
        }
        if region.live != self.live {
            return self.fail(
                Diagnostic::new(Code::Eq04)
                    .with_message(format!("{} ends holding other qubits than it began with", kind.what()))
                    .at(span)
                    .with_note(
                        "it is applied controlled on qubits in superposition; what it allocates, it must \
                         give back before it ends",
                    )
                    .with_help("allocate what it needs as an `aux` ancilla"),
            );
        }
        Ok(())
    }

    /// The controls a quantum condition comes to, with the operations that
    /// computed any ancilla it needed recorded in `undo`.
    fn condition(&mut self, e: &Expr, undo: &mut Vec<(Op, Option<Wire>)>) -> R<Lits> {
        match &e.kind {
            ExprKind::Unary { op: UnOp::Not, operand } => {
                let l = self.condition(operand, undo)?;
                self.negate(l, undo, e.span)
            }
            // a ^ b, computed into an ancilla by flipping it on each
            ExprKind::Binary {
                op: BinOp::BitXor,
                lhs,
                rhs,
            } if e.ty == Ty::Qubit => {
                let l = self.condition(lhs, undo)?;
                let r = self.condition(rhs, undo)?;
                match (l, r) {
                    (Lits::Never, x) | (x, Lits::Never) => Ok(x),
                    (Lits::Always, x) | (x, Lits::Always) => self.negate(x, undo, e.span),
                    (Lits::Conj(a), Lits::Conj(b)) => {
                        let x = self.literal(a, undo, lhs.span)?;
                        let y = self.literal(b, undo, rhs.span)?;
                        let anc = self.alloc(e.span)?;
                        let flip = |c: Control| Op::Gate {
                            gate: GateOp::X,
                            targets: vec![anc],
                            controls: vec![c],
                        };
                        self.raw(flip(x), Some(anc), undo);
                        self.raw(flip(y), None, undo);
                        Ok(Lits::Conj(vec![on(anc)]))
                    }
                }
            }
            ExprKind::Logical { op, lhs, rhs } => {
                let l = self.condition(lhs, undo)?;
                let r = self.condition(rhs, undo)?;
                Ok(match (op, l, r) {
                    (LogicalOp::And, Lits::Never, _) | (LogicalOp::And, _, Lits::Never) => Lits::Never,
                    (LogicalOp::And, Lits::Always, x) | (LogicalOp::And, x, Lits::Always) => x,
                    (LogicalOp::And, Lits::Conj(mut a), Lits::Conj(b)) => {
                        for c in b {
                            if a.iter().any(|x| x.wire == c.wire && x.on != c.on) {
                                return Ok(Lits::Never);
                            }
                            if !a.contains(&c) {
                                a.push(c);
                            }
                        }
                        Lits::Conj(a)
                    }
                    (LogicalOp::Or, Lits::Always, _) | (LogicalOp::Or, _, Lits::Always) => Lits::Always,
                    (LogicalOp::Or, Lits::Never, x) | (LogicalOp::Or, x, Lits::Never) => x,
                    (LogicalOp::Or, Lits::Conj(a), Lits::Conj(b)) => {
                        let x = self.literal(a, undo, lhs.span)?;
                        let y = self.literal(b, undo, rhs.span)?;
                        // a || b is !(!a && !b)
                        let anc = self.alloc(e.span)?;
                        self.raw(Op::Gate { gate: GateOp::X, targets: vec![anc], controls: vec![] }, Some(anc), undo);
                        self.raw(
                            Op::Gate {
                                gate: GateOp::X,
                                targets: vec![anc],
                                controls: vec![Control { wire: x.wire, on: !x.on }, Control { wire: y.wire, on: !y.on }],
                            },
                            None,
                            undo,
                        );
                        Lits::Conj(vec![on(anc)])
                    }
                })
            }
            _ if e.ty == Ty::Bool => match self.expr(e)? {
                V::Val(Value::Bool(true)) => Ok(Lits::Always),
                V::Val(Value::Bool(false)) => Ok(Lits::Never),
                _ => self.fail(runtime_structure(
                    e.span,
                    "a measured value and a qubit are tested in one condition",
                    "a qubit's test is applied as control, while a measured value chooses what the \
                     circuit does as it runs; one condition cannot be both",
                )),
            },
            _ => match self.peek(e)? {
                V::Wire(w) => Ok(Lits::Conj(vec![on(w)])),
                _ => self.not_data(e.span),
            },
        }
    }

    /// What holds where `l` does not.
    fn negate(&mut self, l: Lits, undo: &mut Vec<(Op, Option<Wire>)>, span: Span) -> R<Lits> {
        Ok(match l {
            Lits::Always => Lits::Never,
            Lits::Never => Lits::Always,
            Lits::Conj(cs) if cs.len() == 1 => Lits::Conj(vec![Control { wire: cs[0].wire, on: !cs[0].on }]),
            Lits::Conj(cs) => {
                let anc = self.conjunction(cs, undo, span)?;
                Lits::Conj(vec![Control { wire: anc, on: false }])
            }
        })
    }

    /// `t ^= c`: the qubit `target` names flipped where the quantum
    /// condition `c` holds, under whatever control is in force.
    fn flip(&mut self, target: &Expr, cond: &Expr, span: Span) -> R<V> {
        let handle = self.expr(target)?;
        let wires = self.handle_wires(handle, span)?;
        let [t] = wires[..] else { return self.not_data(span) };
        self.flip_wire(t, cond, span)?;
        Ok(V::Val(Value::Void))
    }

    /// `t ^= c` on the qubit `t`. A parity is a flip for each side and a
    /// negation a flip and a bit flip, and `a | b` is `!(!a & !b)`, so none
    /// of these needs an ancilla.
    fn flip_wire(&mut self, t: Wire, cond: &Expr, span: Span) -> R<()> {
        let quantum = cond.ty == Ty::Qubit;
        match &cond.kind {
            ExprKind::Binary {
                op: BinOp::BitXor,
                lhs,
                rhs,
            } if quantum => {
                self.flip_wire(t, lhs, span)?;
                self.flip_wire(t, rhs, span)
            }
            ExprKind::Unary { op: UnOp::Not, operand } if quantum => {
                self.flip_wire(t, operand, span)?;
                self.gate(GateOp::X, vec![t], Vec::new(), span)
            }
            ExprKind::Logical {
                op: LogicalOp::Or,
                lhs,
                rhs,
            } if quantum => {
                let mut undo = Vec::new();
                let l = self.condition(lhs, &mut undo)?;
                let l = self.negate(l, &mut undo, lhs.span)?;
                let r = self.condition(rhs, &mut undo)?;
                let r = self.negate(r, &mut undo, rhs.span)?;
                let neither = match (l, r) {
                    (Lits::Never, _) | (_, Lits::Never) => Lits::Never,
                    (Lits::Always, x) | (x, Lits::Always) => x,
                    (Lits::Conj(mut a), Lits::Conj(b)) => {
                        let mut contradicts = false;
                        for c in b {
                            contradicts |= a.iter().any(|x| x.wire == c.wire && x.on != c.on);
                            if !a.contains(&c) {
                                a.push(c);
                            }
                        }
                        if contradicts { Lits::Never } else { Lits::Conj(a) }
                    }
                };
                self.flip_lits(t, neither, undo, span)?;
                self.gate(GateOp::X, vec![t], Vec::new(), span)
            }
            _ => {
                let mut undo = Vec::new();
                let lits = self.condition(cond, &mut undo)?;
                self.flip_lits(t, lits, undo, span)
            }
        }
    }

    /// Flips `t` where `lits` holds, then undoes what computed the ancillae
    /// the condition needed.
    fn flip_lits(&mut self, t: Wire, lits: Lits, undo: Vec<(Op, Option<Wire>)>, span: Span) -> R<()> {
        if condition_reads(&lits, &undo).contains(&t) {
            return self.fail(
                Diagnostic::new(Code::Eq20)
                    .with_message("`^=` flips a qubit that its right operand reads")
                    .at(span)
                    .with_note(
                        "the flip is controlled on what the right operand reads; a qubit cannot be \
                         flipped under its own control",
                    )
                    .with_help("compute the right operand into a fresh qubit first, and flip from that"),
            );
        }
        match lits {
            Lits::Always => self.gate(GateOp::X, vec![t], Vec::new(), span)?,
            Lits::Never => {}
            Lits::Conj(cs) => self.gate(GateOp::X, vec![t], cs, span)?,
        }
        self.undo(undo);
        Ok(())
    }

    /// One control standing for a conjunction.
    fn literal(&mut self, cs: Vec<Control>, undo: &mut Vec<(Op, Option<Wire>)>, span: Span) -> R<Control> {
        if cs.len() == 1 {
            return Ok(cs[0]);
        }
        Ok(on(self.conjunction(cs, undo, span)?))
    }

    /// An ancilla holding the conjunction of `cs`.
    fn conjunction(&mut self, cs: Vec<Control>, undo: &mut Vec<(Op, Option<Wire>)>, span: Span) -> R<Wire> {
        let anc = self.alloc(span)?;
        self.raw(Op::Gate { gate: GateOp::X, targets: vec![anc], controls: cs }, Some(anc), undo);
        Ok(anc)
    }

    /// An operation computing a condition's ancilla, outside any control, to
    /// be undone after the branches.
    fn raw(&mut self, op: Op, anc: Option<Wire>, undo: &mut Vec<(Op, Option<Wire>)>) {
        self.push(op.clone());
        undo.push((op, anc));
    }

    /// Undoes what computed a condition's ancillae, and gives them back.
    fn undo(&mut self, undo: Vec<(Op, Option<Wire>)>) {
        for (op, anc) in undo.into_iter().rev() {
            self.push(op);
            if let Some(a) = anc {
                self.release(a);
            }
        }
    }

    ///////////
    // LOOPS //
    ///////////

    #[allow(clippy::too_many_arguments)]
    fn for_range(&mut self, id: LoopId, var: LocalId, start: &Expr, end: &Expr, inclusive: bool, body: &Block, span: Span) -> R<V> {
        let s = self.expr(start)?;
        let e = self.expr(end)?;
        self.enter_loop(id);
        match (s, e) {
            (V::Val(Value::Int(a, t)), V::Val(Value::Int(b, _))) => {
                let last = if inclusive { b } else { b - 1 };
                let mut i = a;
                while i <= last {
                    self.step(span)?;
                    self.bind(var, V::Val(Value::Int(i, t)));
                    match self.block(body) {
                        Ok(_) => {}
                        Err(Flow::Continue(x)) if x == id => {}
                        Err(Flow::Break(x, _)) if x == id => break,
                        Err(f) => return Err(f),
                    }
                    i += 1;
                }
                Ok(V::Val(Value::Void))
            }
            (s, e) => {
                let (Some((s, ks)), Some((e, ke))) = (s.classical(), e.classical()) else {
                    return self.not_data(span);
                };
                if ks == Some(Kind::Outcome) || ke == Some(Kind::Outcome) {
                    return self.not_data(span);
                }
                let end = if inclusive { CExpr::Binary(BinOp::Add, Box::new(e), Box::new(usize_lit(1))) } else { e };
                let v = self.new_var();
                self.bind(var, V::Sym(CExpr::Var(v), Kind::Input));
                // the body is made once and repeated as the circuit runs, so a
                // register it names has a length known only then, and a local
                // it assigns is carried between rounds in a circuit variable
                self.grow_registers(body);
                let f = self.frame();
                let func = self.unit.func(self.frames[f].func);
                let mut carried = Vec::new();
                for l in assigned_locals(body) {
                    if self.unit.types.is_quantum(func.local(l).ty) {
                        continue;
                    }
                    let Some((now, kind)) = self.frames[f].locals[l.index()].classical() else { continue };
                    let cv = self.new_var();
                    self.push(Op::Let { var: cv, value: now });
                    self.frames[f].locals[l.index()] = V::Sym(CExpr::Var(cv), kind.unwrap_or(Kind::Input));
                    carried.push((l, cv, kind));
                }
                self.repeated.push(Repeated {
                    frame: f,
                    declared: HashSet::new(),
                    carried: carried.iter().map(|&(l, _, _)| l).collect(),
                });
                self.ops.push(Vec::new());
                let r = self.block(body);
                self.repeated.pop();
                let mut kinds = Vec::with_capacity(carried.len());
                for &(l, cv, kind) in &carried {
                    let (after, k) = self.frames[f].locals[l.index()]
                        .classical()
                        .unwrap_or((CExpr::Var(cv), kind));
                    self.push(Op::Let { var: cv, value: after });
                    kinds.push(super::value::join_kind(kind, k));
                }
                let ops = self.pop_ops();
                for (&(l, cv, _), k) in carried.iter().zip(kinds) {
                    self.frames[f].locals[l.index()] = V::Sym(CExpr::Var(cv), k.unwrap_or(Kind::Input));
                }
                match r {
                    Ok(_) => {}
                    Err(Flow::Break(..) | Flow::Continue(_)) => {
                        return self.fail(runtime_structure(
                            span,
                            "a loop whose count is known only when the circuit runs is left early",
                            "such a loop is a loop of the circuit, repeated whole each time round",
                        ));
                    }
                    Err(f) => return Err(f),
                }
                self.push(Op::For {
                    var: v,
                    start: s,
                    end,
                    body: ops,
                });
                Ok(V::Val(Value::Void))
            }
        }
    }

    ///////////
    // MATCH //
    ///////////

    fn match_expr(&mut self, scrutinee: &Expr, arms: &[crate::tir::Arm], e: &Expr) -> R<V> {
        // `match measure` on a quantum enumeration
        if let ExprKind::Quantum { op: QuantumOp::Measure, args } = &scrutinee.kind
            && let Ty::Adt(id) = args[0].ty
            && self.unit.types.is_quantum(args[0].ty)
        {
            self.in_region("measures", scrutinee.span, Code::Eq04)?;
            let v = self.expr(&args[0])?;
            return self.measured_match(id, v, arms, e.span);
        }
        let place = self.place_of(scrutinee)?;
        let owned = arms.iter().any(|a| binds_owned(self.unit, &a.pat));
        let v = match &place {
            Some(p) if owned => self.take(p),
            Some(p) => self.read(p),
            None => self.expr(scrutinee)?,
        };
        if let V::QEnum { adt, tag, payload } = &v {
            return self.quantum_match(*adt, tag.clone(), payload.clone(), arms, e.span);
        }
        self.dispatch(&v, place.as_ref(), arms, 0, e.span)
    }

    /// Matches `v` against the arms from `from` on, choosing now what can be
    /// chosen now and feeding forward what depends on a measured value.
    fn dispatch(&mut self, v: &V, place: Option<&Place>, arms: &[crate::tir::Arm], from: usize, span: Span) -> R<V> {
        for (i, arm) in arms.iter().enumerate().skip(from) {
            match test(&arm.pat, v) {
                Test::No => {}
                Test::Yes => {
                    let mut out = Vec::new();
                    self.bind_pattern(&arm.pat, v.clone(), place.cloned(), &mut out)?;
                    return self.arm(out, &arm.body);
                }
                Test::When(cond) => {
                    let v1 = v.clone();
                    let place1 = place.cloned();
                    let v2 = v.clone();
                    let place2 = place.cloned();
                    return self.feed_forward(
                        cond,
                        |g| {
                            let mut out = Vec::new();
                            g.bind_pattern(&arm.pat, v1, place1, &mut out)?;
                            g.arm(out, &arm.body)
                        },
                        |g| g.dispatch(&v2, place2.as_ref(), arms, i + 1, span),
                    );
                }
            }
        }
        self.fail(
            Diagnostic::new(Code::Tq011)
                .with_message("no arm of this `match` accepts the value, though analysis found every value covered")
                .at(span),
        )
    }

    /// An arm's body with its bindings made, the qubits of those left
    /// holding any given back as its scope ends.
    fn arm(&mut self, bound: Vec<(LocalId, V)>, body: &Expr) -> R<V> {
        let locals: Vec<LocalId> = bound.iter().map(|(l, _)| *l).collect();
        for (l, v) in bound {
            self.bind(l, v);
        }
        let r = self.expr(body);
        if matches!(r, Err(Flow::Stop)) {
            return r;
        }
        self.leave(&locals, body.span)?;
        r
    }

    /// Binds what a pattern names in `v`, which it matches.
    fn bind_pattern(&mut self, p: &Pat, v: V, place: Option<Place>, out: &mut Vec<(LocalId, V)>) -> R<()> {
        match &p.kind {
            PatKind::Wild | PatKind::Const(_) => Ok(()),
            PatKind::Bind(l) => {
                out.push((*l, v));
                Ok(())
            }
            PatKind::BindRef(l) => {
                let r = match place {
                    Some(p) => V::Ref(p),
                    None => V::Ref(Place::Temp(Box::new(v), Vec::new())),
                };
                out.push((*l, r));
                Ok(())
            }
            PatKind::Struct { fields } | PatKind::Variant { fields, .. } => {
                for (i, f) in fields {
                    let part = v.part(*i).unwrap_or(V::Moved);
                    let sub = place.as_ref().map(|p| p.child(*i));
                    self.bind_pattern(f, part, sub, out)?;
                }
                Ok(())
            }
        }
    }

    /// A `match` on a quantum enumeration, without measuring: each arm is
    /// applied controlled on the tag's being its variant, its bindings
    /// references to the variant's part of the payload.
    fn quantum_match(&mut self, adt: crate::tir::AdtId, tag: Vec<Wire>, payload: Vec<Wire>, arms: &[crate::tir::Arm], span: Span) -> R<V> {
        let count = self.unit.types.adt(adt).variants().len() as u32;
        let mut covered = vec![false; count as usize];
        for arm in arms {
            let variants: Vec<u32> = match &arm.pat.kind {
                PatKind::Variant { variant, .. } => vec![*variant],
                PatKind::Wild => (0..count).filter(|v| !covered[*v as usize]).collect(),
                _ => {
                    return self.fail(
                        Diagnostic::new(Code::Eq04)
                            .with_message("an arm of a `match` on a quantum enumeration binds the whole value")
                            .at(arm.pat.span)
                            .with_help("match each variant, binding its payload by reference"),
                    );
                }
            };
            for v in variants {
                if covered[v as usize] {
                    continue;
                }
                covered[v as usize] = true;
                let controls = key_controls(&tag, u64::from(v));
                let fields = self.variant_fields(adt, v, &payload);
                let mut out = Vec::new();
                if let PatKind::Variant { fields: pats, .. } = &arm.pat.kind {
                    for (i, fp) in pats {
                        let Some(part) = fields.get(*i as usize).cloned().flatten() else {
                            if matches!(fp.kind, PatKind::Wild) {
                                continue;
                            }
                            return self.fail(
                                Diagnostic::new(Code::Eq04)
                                    .with_message("an arm of a `match` on a quantum enumeration reads a classical field")
                                    .at(fp.span)
                                    .with_note(
                                        "the arm is applied controlled on the tag, to every variant's part of a \
                                         superposition at once; the field is held there as the basis state of \
                                         qubits, which may be in superposition too, and has no one value to read",
                                    )
                                    .with_help("read it with `match measure`, which gives its value, or match it with `_`"),
                            );
                        };
                        match &fp.kind {
                            PatKind::BindRef(l) => out.push((*l, V::Ref(Place::Temp(Box::new(part), Vec::new())))),
                            PatKind::Wild => {}
                            _ => {
                                return self.fail(
                                    Diagnostic::new(Code::Eq04)
                                        .with_message("an arm of a `match` on a quantum enumeration takes its payload")
                                        .at(fp.span)
                                        .with_note(
                                            "the arm is applied controlled on the tag, to every variant's part of a \
                                             superposition at once; it can act on the payload where it is, not take it",
                                        )
                                        .with_help("match through a reference, `match &e`, to bind the payload by reference"),
                                );
                            }
                        }
                    }
                }
                let body = &arm.body;
                self.region(span, RegionKind::Branch, controls, &[], |g| g.arm(out, body))?;
            }
        }
        Ok(V::Val(Value::Void))
    }

    /// The values of variant `v`'s fields, laid out at the front of the
    /// payload register. A classical field's is `None`: its qubits hold it
    /// only as a basis state, in superposition with the other variants.
    fn variant_fields(&self, adt: crate::tir::AdtId, v: u32, payload: &[Wire]) -> Vec<Option<V>> {
        self.variant_layout(adt, v)
            .into_iter()
            .map(|(ty, at, w)| self.unit.types.is_quantum(ty).then(|| shape(self.unit, ty, &payload[at..at + w])))
            .collect()
    }

    /// `match measure e`: the tag measured, and the arm for what was observed
    /// chosen as the circuit runs, binding the payload that survives.
    fn measured_match(&mut self, adt: crate::tir::AdtId, v: V, arms: &[crate::tir::Arm], span: Span) -> R<V> {
        let V::QEnum { tag, payload, .. } = v else { return self.not_data(span) };
        self.acting("measures", span)?;
        let bits: Vec<CExpr> = tag.iter().map(|&w| match self.measure_wire(w) {
            V::Sym(b, _) => b,
            _ => unreachable!("a measurement gives a bit"),
        }).collect();
        let count = self.unit.types.adt(adt).variants().len() as u32;
        self.measured_arms(adt, &bits, &payload, arms, 0, count, span)
    }

    #[allow(clippy::too_many_arguments)]
    fn measured_arms(
        &mut self,
        adt: crate::tir::AdtId,
        bits: &[CExpr],
        payload: &[Wire],
        arms: &[crate::tir::Arm],
        variant: u32,
        count: u32,
        span: Span,
    ) -> R<V> {
        let arm = arms.iter().find(|a| match &a.pat.kind {
            PatKind::Variant { variant: v, .. } => *v == variant,
            _ => true,
        });
        let Some(arm) = arm else {
            return self.fail(
                Diagnostic::new(Code::Tq011)
                    .with_message("no arm of this `match` accepts a variant, though analysis found every one covered")
                    .at(span),
            );
        };
        let run = |g: &mut Self| -> R<V> {
            // a quantum field survives the measurement; a classical one is
            // measured with the tag, giving its value
            let layout = g.variant_layout(adt, variant);
            let used: usize = layout.iter().map(|&(_, _, w)| w).sum();
            let mut fields = Vec::with_capacity(layout.len());
            for (ty, at, w) in layout {
                let ws = &payload[at..at + w];
                fields.push(if g.unit.types.is_quantum(ty) {
                    shape(g.unit, ty, ws)
                } else {
                    V::Sym(g.decode(ty, ws), Kind::Outcome)
                });
            }
            let mut out = Vec::new();
            match &arm.pat.kind {
                PatKind::Variant { fields: pats, .. } => {
                    for (i, fp) in pats {
                        let part = fields.get(*i as usize).cloned().unwrap_or(V::Moved);
                        g.bind_pattern(fp, part.clone(), None, &mut out)?;
                        if matches!(fp.kind, PatKind::Wild) && !part.wires().is_empty() {
                            return g.dropped(fp.span);
                        }
                    }
                    let bound: HashSet<usize> = pats.iter().map(|(i, _)| *i as usize).collect();
                    if fields.iter().enumerate().any(|(i, f)| !bound.contains(&i) && !f.wires().is_empty()) {
                        return g.dropped(arm.pat.span);
                    }
                }
                _ if used > 0 => return g.dropped(arm.pat.span),
                _ => {}
            }
            // in this variant's sector the rest of the payload register is
            // |0>, so it is given back
            for &w in &payload[used..] {
                if g.live.contains(&w) {
                    g.release(w);
                }
            }
            g.arm(out, &arm.body)
        };
        if variant + 1 >= count {
            return run(self);
        }
        let cond = key_condition(bits, u64::from(variant));
        self.feed_forward(cond, run, |g| g.measured_arms(adt, bits, payload, arms, variant + 1, count, span))
    }

    fn dropped<T>(&mut self, span: Span) -> R<T> {
        self.fail(
            Diagnostic::new(Code::Eq01)
                .with_message("this arm leaves qubits of the measured payload unbound")
                .at(span)
                .with_note(
                    "measuring a quantum enumeration's tag leaves its payload's qubits, which may be \
                     entangled with others; they cannot be dropped silently",
                )
                .with_help("bind the payload, and measure it, `forget` it, or return it"),
        )
    }
}

/// What a pattern's test on a value comes to.
enum Test {
    /// It does not match.
    No,
    /// It matches.
    Yes,
    /// It matches where this condition, known when the circuit runs, holds.
    When(CExpr),
}

fn test(p: &Pat, v: &V) -> Test {
    match (&p.kind, v) {
        (PatKind::Wild | PatKind::Bind(_) | PatKind::BindRef(_), _) => Test::Yes,
        (PatKind::Const(c), V::Val(x)) if c == x => Test::Yes,
        (PatKind::Const(_), V::Val(_)) => Test::No,
        (PatKind::Const(c), V::Sym(x, _)) => {
            Test::When(CExpr::Binary(BinOp::Eq, Box::new(x.clone()), Box::new(CExpr::Value(c.clone()))))
        }
        (PatKind::Variant { variant, fields }, V::Val(Value::Enum { variant: w, .. }) | V::Variant(w, _)) => {
            if variant != w {
                return Test::No;
            }
            all(fields.iter().map(|(i, f)| test(f, &v.part(*i).unwrap_or(V::Moved))))
        }
        (PatKind::Variant { variant, fields }, V::Sym(x, k)) => {
            let is = CExpr::Is(Box::new(x.clone()), *variant);
            let rest = all(fields.iter().map(|(i, f)| test(f, &V::Sym(CExpr::Payload(Box::new(x.clone()), *i), *k))));
            and(Test::When(is), rest)
        }
        (PatKind::Struct { fields }, _) => all(fields.iter().map(|(i, f)| test(f, &v.part(*i).unwrap_or(V::Moved)))),
        _ => Test::Yes,
    }
}

fn all(tests: impl Iterator<Item = Test>) -> Test {
    tests.fold(Test::Yes, and)
}

fn and(a: Test, b: Test) -> Test {
    match (a, b) {
        (Test::No, _) | (_, Test::No) => Test::No,
        (Test::Yes, x) | (x, Test::Yes) => x,
        (Test::When(x), Test::When(y)) => Test::When(CExpr::Logical(LogicalOp::And, Box::new(x), Box::new(y))),
    }
}

/// Whether the operator `f` is dynamic: it, or an operator it calls, is
/// marked `[dynamic]` or contains a `lift`.
pub fn dynamic(unit: &Unit, f: FnId) -> bool {
    struct Scan<'u> {
        unit: &'u Unit,
        seen: HashSet<FnId>,
        found: bool,
    }
    impl crate::tir::visit::Visit for Scan<'_> {
        fn expr(&mut self, e: &Expr) {
            if self.found {
                return;
            }
            match &e.kind {
                ExprKind::Quantum { op: QuantumOp::Lift, .. } => self.found = true,
                ExprKind::Call { callee: Callee::Fn(g), .. } | ExprKind::FnRef(Callee::Fn(g)) => self.visit_fn(*g),
                _ => {}
            }
            crate::tir::visit::walk_expr(self, e);
        }
    }
    impl Scan<'_> {
        fn visit_fn(&mut self, g: FnId) {
            if !self.seen.insert(g) {
                return;
            }
            let func = self.unit.func(g);
            if func.attrs.dynamic {
                self.found = true;
                return;
            }
            crate::tir::visit::Visit::block(self, &func.body);
        }
    }
    let mut s = Scan {
        unit,
        seen: HashSet::new(),
        found: false,
    };
    s.visit_fn(f);
    s.found
}

/// Whether a pattern binds part of the value by moving it.
fn binds_owned(unit: &Unit, p: &Pat) -> bool {
    match &p.kind {
        PatKind::Bind(_) => !unit.types.is_copyable(p.ty),
        PatKind::Struct { fields } | PatKind::Variant { fields, .. } => fields.iter().any(|(_, f)| binds_owned(unit, f)),
        _ => false,
    }
}

/// A value of quantum type `ty` made of the qubits `ws`, in order.
fn shape(unit: &Unit, ty: Ty, ws: &[Wire]) -> V {
    if ty == Ty::Qubit {
        return V::Wire(ws[0]);
    }
    if let Some((elem, n)) = unit.types.as_array(ty) {
        let w = ws.len() / n.max(1) as usize;
        return V::Agg {
            array: true,
            items: (0..n as usize).map(|i| shape(unit, elem, &ws[i * w..(i + 1) * w])).collect(),
        };
    }
    V::Agg {
        array: true,
        items: ws.iter().map(|&w| V::Wire(w)).collect(),
    }
}

/// The operations the entry `e` of a map locale of states, `prep` of its
/// state, prepares its state with on `register`.
fn prepared_ops(e: &Expr, register: &[Wire]) -> Vec<Op> {
    match &e.kind {
        ExprKind::Quantum {
            op: QuantumOp::Prep(p), ..
        } => p.on(register),
        _ => unreachable!("an entry of a map locale of states is a preparation"),
    }
}

/// The register `wires` as a value of the register type `ty`.
fn shape_register(ty: Ty, wires: &[Wire]) -> V {
    match ty {
        Ty::Qubit => V::Wire(wires[0]),
        _ => V::Agg {
            array: true,
            items: wires.iter().map(|&w| V::Wire(w)).collect(),
        },
    }
}

/// The value a binding holds after feed-forward under `cond`, when it holds
/// `a` where the condition holds and `b` where it does not; `None` when the
/// two differ in qubits.
fn merge(cond: &CExpr, a: &V, b: &V) -> Option<V> {
    if a == b {
        return Some(a.clone());
    }
    match (a, b) {
        (V::Moved, _) | (_, V::Moved) => Some(V::Moved),
        (V::Agg { array, items: x }, V::Agg { items: y, .. }) if x.len() == y.len() => Some(
            V::Agg {
                array: *array,
                items: x.iter().zip(y).map(|(p, q)| merge(cond, p, q)).collect::<Option<Vec<_>>>()?,
            }
            .normal(),
        ),
        _ => {
            let (x, _) = a.classical()?;
            let (y, _) = b.classical()?;
            Some(V::Sym(CExpr::Select(Box::new(cond.clone()), Box::new(x), Box::new(y)), Kind::Outcome))
        }
    }
}

/// Whether an array computed as the circuit runs is one of several chosen
/// between, whose length may then depend on what chose.
fn chooses(e: &CExpr) -> bool {
    match e {
        CExpr::Select(..) | CExpr::Payload(..) | CExpr::Field(..) | CExpr::Index(..) => true,
        CExpr::Append(a, _) => chooses(a),
        _ => false,
    }
}

fn on(w: Wire) -> Control {
    Control { wire: w, on: true }
}

/// Controls on `wires` that hold where they hold `k`, the first wire its
/// most significant bit.
fn key_controls(wires: &[Wire], k: u64) -> Vec<Control> {
    wires.iter().zip(key_bits(wires.len(), k)).map(|(&wire, on)| Control { wire, on }).collect()
}

/// The condition that the measured `bits`, the first the most significant,
/// hold `k`.
fn key_condition(bits: &[CExpr], k: u64) -> CExpr {
    bits.iter()
        .zip(key_bits(bits.len(), k))
        .map(|(b, set)| if set { b.clone() } else { CExpr::Unary(UnOp::Not, Box::new(b.clone())) })
        .reduce(|a, b| CExpr::Logical(LogicalOp::And, Box::new(a), Box::new(b)))
        .unwrap_or(CExpr::Value(Value::Bool(true)))
}

/// The bits of `k`.
fn key_bits(n: usize, k: u64) -> impl Iterator<Item = bool> {
    (0..n).map(move |i| (k >> (n - 1 - i)) & 1 == 1)
}

fn usize_lit(i: i128) -> CExpr {
    CExpr::Value(Value::Int(i, crate::tir::IntTy::USIZE))
}

/// The types of the parts of an array, tuple or structure type `ty`, and
/// whether it is an array.
fn parts(types: &crate::tir::TypeTable, ty: Ty) -> Option<(Vec<Ty>, bool)> {
    if let Some((elem, n)) = types.as_array(ty) {
        Some((vec![elem; n as usize], true))
    } else if let Some(elems) = types.as_tuple(ty) {
        Some((elems.to_vec(), false))
    } else if let Ty::Adt(id) = ty {
        Some((types.adt(id).fields().iter().map(|f| f.ty).collect(), false))
    } else {
        None
    }
}

/// Whether the state `s` of the wires `anc` has a definite value on each of
/// them that `controls` names: every basis state it holds agrees there.
fn definite(anc: &[Wire], s: &[Cyclo], controls: &[Control]) -> bool {
    let k = anc.len();
    let bits: Vec<usize> = controls
        .iter()
        .filter_map(|c| anc.iter().position(|w| *w == c.wire).map(|p| k - 1 - p))
        .collect();
    let mut seen: Option<usize> = None;
    for (i, a) in s.iter().enumerate() {
        if a.is_zero() {
            continue;
        }
        let v: usize = bits.iter().map(|&b| i & (1 << b)).sum();
        match seen {
            None => seen = Some(v),
            Some(x) if x != v => return false,
            Some(_) => {}
        }
    }
    true
}

/// Whether the gate `op`, all of whose targets are among `anc` and none of
/// whose controls are, leaves the state `s` of `anc` as it is up to a phase:
/// `s` is an eigenvector of what it applies, so applied under its controls it
/// kicks that phase back to them.
fn kicks_back(op: &Op, anc: &[Wire], s: &[Cyclo], n: u32) -> bool {
    let bare = match op {
        Op::Gate { gate, targets, .. } => Op::Gate {
            gate: gate.clone(),
            targets: targets.clone(),
            controls: Vec::new(),
        },
        Op::Unitary { matrix, targets, .. } => Op::Unitary {
            matrix: matrix.clone(),
            targets: targets.clone(),
            controls: Vec::new(),
        },
        _ => return false,
    };
    match crate::circuit::action::evolve(anc, s.to_vec(), &[bare], n) {
        Ok(image) => crate::circuit::action::proportional(&image, s),
        Err(_) => false,
    }
}

fn jump_out(span: Span, at: Span, what: &str) -> Diagnostic {
    Diagnostic::new(Code::Eq04)
        .with_message(format!("{what} leaves a branch of a quantum `if` or `match`"))
        .at_with(span, "leaves here")
        .also(at, "the branch is controlled on a qubit here")
        .with_note(
            "a quantum branch is applied controlled on qubits in superposition, so it cannot end \
             early for some parts of the superposition and not others",
        )
}

/// The locals a block assigns.
fn assigned_locals(body: &Block) -> Vec<LocalId> {
    struct Assigned(Vec<LocalId>);
    impl crate::tir::visit::Visit for Assigned {
        fn expr(&mut self, e: &Expr) {
            let changed = match &e.kind {
                ExprKind::Assign { place, .. } => Some(place.as_ref()),
                ExprKind::Growable { op, array, .. } if *op != GrowOp::Reserve => Some(array.as_ref()),
                _ => None,
            };
            if let Some(place) = changed {
                let mut p = place;
                while let ExprKind::Field { base, .. } | ExprKind::Index { base, .. } = &p.kind {
                    p = base;
                }
                if let ExprKind::Local(l) = p.kind
                    && !self.0.contains(&l)
                {
                    self.0.push(l);
                }
            }
            crate::tir::visit::walk_expr(self, e);
        }
    }
    let mut a = Assigned(Vec::new());
    crate::tir::visit::Visit::block(&mut a, body);
    a.0
}

/// The growable registers of qubits `body` names by binding, field or
/// dereferenced binding.
fn registers_named(types: &crate::tir::TypeTable, body: &Block) -> Vec<Expr> {
    struct Named<'t> {
        types: &'t crate::tir::TypeTable,
        found: Vec<Expr>,
    }
    impl crate::tir::visit::Visit for Named<'_> {
        fn expr(&mut self, e: &Expr) {
            let named = matches!(e.kind, ExprKind::Local(_) | ExprKind::Field { .. } | ExprKind::Deref(_));
            if named && self.types.as_growable(e.ty) == Some(Ty::Qubit) {
                self.found.push(e.clone());
                return;
            }
            crate::tir::visit::walk_expr(self, e);
        }
    }
    let mut n = Named { types, found: Vec::new() };
    crate::tir::visit::Visit::block(&mut n, body);
    n.found
}

/// A value known only when the circuit runs, where the circuit's structure
/// needs one now.
fn runtime_structure(span: Span, what: &str, why: &str) -> Diagnostic {
    Diagnostic::new(Code::Eq16)
        .with_message(format!("{what}, but this needs a value while the circuit is generated"))
        .at(span)
        .with_note(why.to_owned())
        .with_help("make the value `const`, or known from constants, so that it is fixed when the circuit is generated")
}

/// The order N of a `phase<N>` type.
fn phase_order(unit: &Unit, ty: Ty) -> Option<u64> {
    let Ty::Adt(id) = ty else { return None };
    match unit.types.adt(id).args.first() {
        Some(Arg::Const(n)) => u64::try_from(*n).ok(),
        _ => None,
    }
}

/// The order N of the entries of a `mat<cyclo<N>, R, C>` type.
fn matrix_order(unit: &Unit, ty: Ty) -> Option<u64> {
    let Ty::Adt(id) = ty else { return None };
    match unit.types.adt(id).args.first() {
        Some(Arg::Type(elem)) => phase_order(unit, *elem),
        _ => None,
    }
}

/// The entry of U†U that differs from the identity's, if one does.
fn not_unitary(m: &Matrix, conductor: u32) -> Option<(usize, usize)> {
    for i in 0..m.size {
        for j in 0..m.size {
            let mut sum = Cyclo::zero(conductor);
            for k in 0..m.size {
                sum = sum.add(&m.at(k, i).conj().mul(m.at(k, j)));
            }
            let want = if i == j { Cyclo::one(conductor) } else { Cyclo::zero(conductor) };
            let (a, b) = Cyclo::unify(&sum, &want);
            if a != b {
                return Some((i, j));
            }
        }
    }
    None
}

impl V {
    /// How many parts the value has, if it is an aggregate.
    pub fn part_count(&self) -> Option<usize> {
        match self {
            V::Val(Value::Array(xs) | Value::Struct(xs)) => Some(xs.len()),
            V::Agg { items, .. } => Some(items.len()),
            _ => None,
        }
    }
}
