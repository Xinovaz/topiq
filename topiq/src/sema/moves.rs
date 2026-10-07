//! Following every path through a function to check that no binding is used
//! after its value was moved away, or before it was given one.
//!
//! # Moves
//!
//! Every value has exactly one owner. Using a structure or enumeration as a
//! value (passing it to a function, assigning it, binding it with `let`,
//! returning it) hands it to a new owner, and the binding it came from holds
//! nothing afterwards. Using that binding again is an error. Integers, `bool`,
//! `char`, references, and arrays of those are copied instead, so using them
//! never empties anything.
//!
//! Moves are tracked per field: moving `s.a` out of a structure `s` leaves
//! `s.b` usable, but not `s` as a whole. Assigning a new value to a moved
//! binding makes it usable again.
//!
//! Three places cannot give up a value this way, because nothing could track
//! that they had: an element of an array (which one is only known at run
//! time), the object a reference refers to (someone else owns it), and a
//! unit-scope or persistent object (everything may read it). A value that
//! cannot be copied can only be used in place there (its fields read, a
//! reference taken), never moved out.
//!
//! A `match` does not move what it examines unless an arm binds part of it
//! that cannot be copied; then that arm's binding takes the value.
//!
//! # Values not yet given
//!
//! `let x: T;` declares a binding without a value. Every path from there to a
//! read of `x` must assign it first. A path that might not (an `if` with no
//! `else`, a loop that might not run) makes the read an error.
//!
//! # How paths are followed
//!
//! The body is walked in evaluation order, carrying the set of places that are
//! moved or not yet given a value. At an `if` or `match` each branch starts
//! from the same set and the results are merged: a place moved on *some* path
//! counts as moved. A loop's body is walked until the set at its start stops
//! growing, since a move late in the body affects the next iteration. After a
//! `return`, `break`, `continue`, or a call that never returns, nothing is
//! reachable, and the walk resumes wherever control can arrive again.
//!
//! # Qubits at the end of their scope
//!
//! A type holding a qubit is never copied, so the rules above already keep a
//! qubit from being used twice. The other half of the discipline is that a
//! qubit may not be dropped silently: its state may be entangled with qubits
//! that live on. So wherever a scope is left (the end of a block or of a
//! `match` arm, `return`, `break`, `continue`), each binding it declares that
//! may still hold a qubit is `EQ01`, unless
//!
//! - it was moved away on every path, being measured, forgotten, returned or
//!   passed on;
//! - it is an `aux` ancilla, which uncomputation returns to |0>; or
//! - it holds what its declaration allocated, and nothing else. Whether what
//!   was done to it through handles left it in |0> (nothing was, or what
//!   was is undone) depends on the operations applied, so circuit
//!   generation decides it exactly ([`crate::lower`]), and reports `EQ01`
//!   where it is not.
//!
//! The same holds of an assignment to a place that may still hold a qubit,
//! and of an expression statement whose value holds one.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::diag::{Code, Diagnostic};
use crate::intern::Interner;
use crate::span::Span;
use crate::tir::{
    Block, Expr, ExprKind, Fn, GlobalKind, Intrinsic, LocalId, LoopId, Pat, PatKind, QuantumOp, Stmt, Ty, Unit,
};

/// A binding, or a field within it, as a path of field indices.
type Path = (LocalId, Vec<u32>);

/// Why a place holds nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Empty {
    /// Its value was moved away, at this span.
    Moved(Span),
    /// It was declared without a value, at this span.
    Unset(Span),
}

/// What is known at one point of a function.
#[derive(Clone, Debug, Default)]
struct Places {
    /// The places that may hold nothing here, and why.
    empty: BTreeMap<Path, Empty>,
    /// The quantum bindings that may hold something other than what their
    /// declaration allocated: a value given, a parameter, a part bound.
    touched: BTreeSet<LocalId>,
}

impl Places {
    /// Records that `path`, and every part of it, holds a value.
    fn fill(&mut self, path: &Path) {
        self.empty.retain(|p, _| !(p.0 == path.0 && p.1.starts_with(&path.1)));
    }
}

/// What is known at one point, or `None` where control cannot reach.
type State = Option<Places>;

/// Records that `local` has just been given a whole value.
fn fresh_local(local: LocalId, st: &mut State) {
    if let Some(places) = st {
        places.empty.retain(|(l, _), _| *l != local);
    }
}

/// Merges the states of two paths that meet: anything true on either path
/// may be true here.
fn join(a: State, b: State) -> State {
    match (a, b) {
        (None, s) | (s, None) => s,
        (Some(mut a), Some(b)) => {
            for (k, v) in b.empty {
                a.empty.entry(k).or_insert(v);
            }
            a.touched.extend(b.touched);
            Some(a)
        }
    }
}

/// Whether two states say the same things, whatever the spans.
fn same(a: &State, b: &State) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.empty.keys().eq(b.empty.keys()) && a.touched == b.touched
        }
        _ => false,
    }
}

/// Checks every compiled function of the unit.
pub fn check(unit: &Unit, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    for f in &unit.fns {
        let mut m = Mover {
            unit,
            func: f,
            interner,
            diags,
            loops: HashMap::new(),
            reported: HashSet::new(),
            scopes: vec![f.params.clone()],
            loop_scopes: HashMap::new(),
        };
        // a parameter came from the caller, so what it holds is not known
        // to be |0>
        let touched = f.params.iter().copied().filter(|&p| m.quantum(p)).collect();
        let mut st: State = Some(Places {
            touched,
            ..Places::default()
        });
        m.block(&f.body, &mut st);
        let end = Span::new(f.body.span.source, f.body.span.end.saturating_sub(1), f.body.span.end);
        m.leave_scopes(0, end, Exit::Return, &st);
    }
}

/// How the scopes being left are left, for a message.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Exit {
    /// A block's end is reached.
    End,
    /// The function returns.
    Return,
    /// A loop is left by `break`.
    Break,
    /// A loop's next iteration begins.
    Continue,
}

#[derive(Default)]
struct LoopExits {
    breaks: State,
    continues: State,
}

struct Mover<'a> {
    unit: &'a Unit,
    func: &'a Fn,
    interner: &'a Interner,
    diags: &'a mut Vec<Diagnostic>,
    loops: HashMap<LoopId, LoopExits>,
    /// Uses already reported, so that walking a loop again does not repeat
    /// them.
    reported: HashSet<(u32, u32, Code)>,
    /// The bindings each open scope declares, outermost (the parameters)
    /// first.
    scopes: Vec<Vec<LocalId>>,
    /// For each open loop, how many scopes were open outside it.
    loop_scopes: HashMap<LoopId, usize>,
}

/// How a place is used.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    /// Read in place: a reference taken, a field read, a length asked for.
    Look,
    /// Stored into.
    Write,
}

impl Mover<'_> {
    fn copyable(&self, ty: Ty) -> bool {
        self.unit.types.is_copyable(ty)
    }

    /// The path of a place made only of a binding and fields.
    fn path_of(e: &Expr) -> Option<Path> {
        match &e.kind {
            ExprKind::Local(l) => Some((*l, Vec::new())),
            ExprKind::Field { base, field } => {
                let (l, mut fs) = Self::path_of(base)?;
                fs.push(*field);
                Some((l, fs))
            }
            _ => None,
        }
    }

    /// A path as a program writes it: `s.a.b`.
    fn show(&self, p: &Path) -> String {
        let local = self.func.local(p.0);
        let mut out = self.interner.resolve(local.name).to_owned();
        let mut ty = local.ty;
        for &f in &p.1 {
            let Ty::Adt(id) = ty else { break };
            let Some(field) = self.unit.types.adt(id).fields().get(f as usize) else {
                break;
            };
            out.push('.');
            out.push_str(self.interner.resolve(field.name));
            ty = field.ty;
        }
        out
    }

    fn once(&mut self, span: Span, code: Code) -> bool {
        self.reported.insert((span.start, span.end, code))
    }

    /// Reports a use of `path` if any part of it, or anything containing it,
    /// holds nothing.
    fn check_use(&mut self, path: &Path, span: Span, st: &State) {
        let Some(places) = st else { return };

        // an emptied place that is part of `path`, or contains it
        let hit = places.empty.iter().find(|(p, _)| {
            p.0 == path.0 && (p.1.starts_with(&path.1) || path.1.starts_with(&p.1))
        });
        let Some((emptied, why)) = hit.map(|(p, w)| (p.clone(), *w)) else {
            return;
        };

        // reported once for each place and kind
        let used = self.show(path);
        let whole = self.show(&emptied);
        match why {
            Empty::Unset(decl) => {
                if !self.once(span, Code::Es11) {
                    return;
                }
                self.diags.push(
                    Diagnostic::new(Code::Es11)
                        .with_message(format!("`{used}` is read here, but `{whole}` may not have been given a value yet"))
                        .at(span)
                        .also(decl, "declared here without a value")
                        .with_note("on at least one path from its declaration to here, nothing assigns it")
                        .with_help(format!("give `{whole}` a value on every path before this, or when it is declared")),
                );
            }
            Empty::Moved(at) => {
                if !self.once(span, Code::Es10) {
                    return;
                }
                let message = if emptied == *path {
                    format!("`{used}` is used here after its value was moved away")
                } else if emptied.1.len() > path.1.len() {
                    format!("`{used}` is used here, but its part `{whole}` was moved away")
                } else {
                    format!("`{used}` is used here, but `{whole}`, which contains it, was moved away")
                };
                let what = describe_owned(self.func.local(emptied.0).ty);
                self.diags.push(
                    Diagnostic::new(Code::Es10)
                        .with_message(message)
                        .at(span)
                        .also(at, "moved from here")
                        .with_note(format!(
                            "{what} is moved, not copied, when it is used as a value (passed to a \
                             function, assigned, bound with `let` or returned) and afterwards the \
                             place it came from holds nothing"
                        ))
                        .with_help(format!(
                            "if `{whole}` is still needed afterwards, lend it with `&{whole}` where \
                             it was moved, which does not move it"
                        )),
                );
            }
        }
    }

    /// Whether the place `e` is a constant or a field or element of one.
    fn part_of_constant(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Global(g) => self.unit.global(*g).constant,
            ExprKind::Field { base, .. } | ExprKind::Index { base, .. } => self.part_of_constant(base),
            _ => false,
        }
    }

    /// A value that cannot be copied, moved out of a place that must keep it.
    fn cannot_move(&mut self, e: &Expr, place: &str) {
        if !self.once(e.span, Code::Es12) {
            return;
        }
        let what = describe_owned(e.ty);
        self.diags.push(
            Diagnostic::new(Code::Es12)
                .with_message(format!("this would move {what} out of {place}, which must keep it"))
                .at(e.span)
                .with_note(
                    "a value that cannot be copied can be moved only out of a binding the \
                     function owns; elsewhere nothing could record that the place is empty",
                )
                .with_help("use it in place (read the fields you need, or take a reference with `&`) instead of moving it"),
        );
    }

    fn block(&mut self, b: &Block, st: &mut State) {
        self.scopes.push(Vec::new());
        self.statements(b, st);
        let depth = self.scopes.len() - 1;
        let end = Span::new(b.span.source, b.span.end.saturating_sub(1), b.span.end);
        self.leave_scopes(depth, end, Exit::End, st);
        self.scopes.pop();
    }

    /// A block's statements and value.
    fn statements(&mut self, b: &Block, st: &mut State) {
        for s in &b.stmts {
            if st.is_none() {
                return;
            }
            match s {
                Stmt::Let { local, init } => {
                    self.declare(*local);
                    let quantum = self.quantum(*local);
                    match init {
                        Some(e) => {
                            self.consume(e, st);
                            fresh_local(*local, st);
                            if quantum && let Some(places) = st {
                                places.touched.insert(*local);
                            }
                        }
                        // quantum storage declared without a value is
                        // allocated, in |0>
                        None if quantum => {
                            fresh_local(*local, st);
                            if let Some(places) = st {
                                places.touched.remove(local);
                            }
                        }
                        None => {
                            if let Some(places) = st {
                                let decl = self.func.local(*local).span;
                                places.empty.insert((*local, Vec::new()), Empty::Unset(decl));
                            }
                        }
                    }
                }
                // taking a value apart binds parts of it, exactly as a
                // `match` arm does, so the same rule applies: the value is
                // moved only where a part that cannot be copied is bound
                Stmt::LetPat { pat, init } => {
                    let moves = self.binds_owned(pat);
                    self.examine(init, moves, st);
                    self.declare_pattern(pat, st);
                }
                Stmt::Expr(e) => {
                    self.consume(e, st);
                    self.dropped_temporary(e, st);
                }
            }
        }
        if let Some(v) = &b.value {
            self.consume(v, st);
        }
    }

    /// Whether a binding holds qubits.
    fn quantum(&self, local: LocalId) -> bool {
        self.unit.types.is_quantum(self.func.local(local).ty)
    }

    /// Records that the innermost open scope declares `local`.
    fn declare(&mut self, local: LocalId) {
        if let Some(s) = self.scopes.last_mut() {
            s.push(local);
        }
    }

    /// Declares each binding a pattern makes, which takes part of a value
    /// that was already acted on.
    fn declare_pattern(&mut self, p: &Pat, st: &mut State) {
        let mut bound = Vec::new();
        pattern_locals(p, &mut bound);
        for l in bound {
            // in a loop, a binding moved away on the last time round holds
            // a new value each time the pattern binds it
            fresh_local(l, st);
            self.declare(l);
            if self.quantum(l) && let Some(places) = st {
                places.touched.insert(l);
            }
        }
    }

    /// Records that the quantum binding at the root of a place is given a
    /// value other than what its declaration allocated.
    fn touch(&self, e: &Expr, st: &mut State) {
        if let Some((l, _)) = Self::path_of(e)
            && self.quantum(l)
            && let Some(places) = st
        {
            places.touched.insert(l);
        }
    }

    /// Reports an assignment to a place that may still hold qubits, which
    /// would drop them.
    fn overwritten(&mut self, path: &Path, e: &Expr, st: &State) {
        let Some(places) = st else { return };
        if !self.unit.types.is_quantum(e.ty) || !places.touched.contains(&path.0) {
            return;
        }
        let Some(part) = self.live_part(path, e.ty, places) else {
            return;
        };
        if !self.once(e.span, Code::Eq01) {
            return;
        }
        let shown = self.show(&part);
        self.diags.push(
            Diagnostic::new(Code::Eq01)
                .with_message(format!("assigning to `{shown}` here would drop the qubits it may still hold"))
                .at(e.span)
                .with_note(
                    "a qubit's state may be entangled with qubits that live on, so it cannot be \
                     dropped silently, by an assignment or otherwise",
                )
                .with_help(format!("measure `{shown}`, `forget` it, or move it elsewhere first")),
        );
    }

    /// Reports an expression statement whose value holds qubits: nothing
    /// keeps it, so it would be dropped where it is made.
    fn dropped_temporary(&mut self, e: &Expr, st: &State) {
        if st.is_none() || !self.unit.types.is_quantum(e.ty) || !self.once(e.span, Code::Eq01) {
            return;
        }
        let what = self.unit.types.display(e.ty, self.interner);
        self.diags.push(
            Diagnostic::new(Code::Eq01)
                .with_message(format!("this gives `{what}`, which holds qubits, and nothing keeps it"))
                .at(e.span)
                .with_note(
                    "a qubit's state may be entangled with qubits that live on, so discarding it is \
                     an operation of its own, which the program writes",
                )
                .with_help("bind it and measure it, `forget` it, or return it"),
        );
    }

    /// Checks the scopes from `from` inward as they are left by `exit` at
    /// `at`: every binding they declare that still holds qubits is reported,
    /// unless it is an ancilla, which is uncomputed, or holds only what its
    /// declaration allocated, which circuit generation checks.
    fn leave_scopes(&mut self, from: usize, at: Span, exit: Exit, st: &State) {
        let Some(places) = st else { return };
        let leaving: Vec<LocalId> = self.scopes[from..].iter().flatten().copied().collect();
        for l in leaving {
            let local = self.func.local(l);
            if local.aux || !self.quantum(l) || !places.touched.contains(&l) {
                continue;
            }
            let Some(part) = self.live_part(&(l, Vec::new()), local.ty, places) else {
                continue;
            };
            if !self.reported.insert((local.span.start, at.start, Code::Eq01)) {
                continue;
            }
            let shown = self.show(&part);
            let (when, here) = match exit {
                Exit::End => ("its scope ends", "its scope ends here"),
                Exit::Return => ("the function returns", "the function returns here"),
                Exit::Break => ("`break` leaves its loop", "its loop is left here"),
                Exit::Continue => ("`continue` starts the loop again", "its loop is left here"),
            };
            self.diags.push(
                Diagnostic::new(Code::Eq01)
                    .with_message(format!("`{shown}` may still hold qubits when {when}"))
                    .at_with(at, here)
                    .also(local.span, "declared here")
                    .with_note(
                        "a qubit's state may be entangled with qubits that live on, so it cannot be \
                         dropped silently; only an `aux` ancilla, which is uncomputed, and a qubit \
                         allocated in the scope and returned to |0> there, are released at the end \
                         of their scope",
                    )
                    .with_help(format!("measure `{shown}`, `forget` it, or return it")),
            );
        }
    }

    /// The first part of the place `path`, of type `ty`, that may still hold
    /// a qubit, or `None` when every qubit in it was moved away.
    fn live_part(&self, path: &Path, ty: Ty, places: &Places) -> Option<Path> {
        if !self.unit.types.is_quantum(ty) || places.empty.contains_key(path) {
            return None;
        }
        let split = places
            .empty
            .keys()
            .any(|p| p.0 == path.0 && p.1.len() > path.1.len() && p.1.starts_with(&path.1));
        let Ty::Adt(id) = ty else {
            return Some(path.clone());
        };
        let def = self.unit.types.adt(id);
        if !split || !def.is_struct() {
            return Some(path.clone());
        }
        def.fields().iter().enumerate().find_map(|(i, f)| {
            let mut sub = path.clone();
            sub.1.push(i as u32);
            self.live_part(&sub, f.ty, places)
        })
    }

    /// Evaluates `e` for its value, which moves it if it is a place whose type
    /// cannot be copied.
    fn consume(&mut self, e: &Expr, st: &mut State) {
        if st.is_none() {
            return;
        }
        match &e.kind {
            ExprKind::Local(_) | ExprKind::Global(_) | ExprKind::Deref(_) | ExprKind::Index { .. } => self.read(e, st),
            ExprKind::Field { base, .. } => {
                if e.is_place() {
                    self.read(e, st);
                } else {
                    // a field of a temporary: the temporary is used up
                    self.consume(base, st);
                }
            }
            ExprKind::Call { args, .. } | ExprKind::Closure { captures: args, .. } => {
                for a in args {
                    self.consume(a, st);
                }
            }
            // calling a closure uses it without using it up
            ExprKind::IndirectCall { callee, args } => {
                self.examine(callee, false, st);
                for a in args {
                    self.consume(a, st);
                }
            }
            ExprKind::FnRef(_) | ExprKind::ThunkRef(_) => {}
            ExprKind::Intrinsic { which, args } => {
                for a in args {
                    if *which == Intrinsic::Len {
                        self.place(a, Use::Look, st);
                    } else {
                        self.consume(a, st);
                    }
                }
            }
            // `t ^= c` changes `t` where it is and only reads `c`
            ExprKind::Quantum {
                op: QuantumOp::Flip,
                args,
            } => {
                self.consume(&args[0], st);
                self.look_condition(&args[1], st);
            }
            ExprKind::Quantum { args, .. } => {
                for a in args {
                    self.consume(a, st);
                }
            }
            // the array is changed where it is, and stays owned there
            ExprKind::Growable { array, args, .. } => {
                self.place(array, Use::Look, st);
                for a in args {
                    self.consume(a, st);
                }
            }
            ExprKind::Grow(x) => self.consume(x, st),
            ExprKind::Unary { operand: x, .. }
            | ExprKind::Cast { expr: x, .. }
            | ExprKind::Coerce(x)
            | ExprKind::FnAsClosure(x)
            | ExprKind::ArrayRepeat { elem: x, .. } => self.consume(x, st),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.consume(lhs, st);
                self.consume(rhs, st);
            }
            ExprKind::Logical { lhs, rhs, .. } => {
                self.consume(lhs, st);
                let skipped = st.clone();
                self.consume(rhs, st);
                *st = join(skipped, st.take());
            }
            ExprKind::Ref(x) => self.place(x, Use::Look, st),
            ExprKind::StructLit { fields } | ExprKind::Variant { fields, .. } => {
                for (_, f) in fields {
                    self.consume(f, st);
                }
            }
            ExprKind::ArrayLit(items) => {
                for i in items {
                    self.consume(i, st);
                }
            }
            ExprKind::Assign { place, value } => {
                self.consume(value, st);
                self.place(place, Use::Write, st);
            }
            ExprKind::Block(b) => self.block(b, st),
            ExprKind::If { cond, then, els } => {
                if cond.ty == Ty::Qubit || self.unit.types.is_quantum(cond.ty) {
                    self.look_condition(cond, st);
                } else {
                    self.consume(cond, st);
                }
                let mut other = st.clone();
                self.block(then, st);
                if let Some(e) = els {
                    self.consume(e, &mut other);
                }
                *st = join(st.take(), other);
            }
            ExprKind::Match { scrutinee, arms } => {
                let moves = arms.iter().any(|a| self.binds_owned(&a.pat));
                self.examine(scrutinee, moves, st);
                let start = st.take();
                let mut end: State = None;
                for a in arms {
                    let mut s = start.clone();
                    self.scopes.push(Vec::new());
                    self.declare_pattern(&a.pat, &mut s);
                    self.consume(&a.body, &mut s);
                    let depth = self.scopes.len() - 1;
                    self.leave_scopes(depth, a.body.span, Exit::End, &s);
                    self.scopes.pop();
                    end = join(end, s);
                }
                *st = end;
            }
            // a `loop` is left only by `break`
            ExprKind::Loop { id, body } => {
                self.repeat(*id, st, |m, s| {
                    m.block(body, s);
                    None
                });
            }
            ExprKind::While { id, cond, body } => {
                self.repeat(*id, st, |m, s| {
                    m.consume(cond, s);
                    let at_test = s.clone();
                    m.block(body, s);
                    at_test
                });
            }
            ExprKind::ForRange {
                id, start, end, body, ..
            } => {
                self.consume(start, st);
                self.consume(end, st);
                self.repeat(*id, st, |m, s| {
                    let at_head = s.clone();
                    m.block(body, s);
                    at_head
                });
            }
            ExprKind::Break { target, value } => {
                if let Some(v) = value {
                    self.consume(v, st);
                }
                let from = self.loop_scopes.get(target).copied().unwrap_or(self.scopes.len());
                self.leave_scopes(from, e.span, Exit::Break, st);
                let exits = self.loops.entry(*target).or_default();
                exits.breaks = join(exits.breaks.take(), st.take());
            }
            ExprKind::Continue { target } => {
                let from = self.loop_scopes.get(target).copied().unwrap_or(self.scopes.len());
                self.leave_scopes(from, e.span, Exit::Continue, st);
                let exits = self.loops.entry(*target).or_default();
                exits.continues = join(exits.continues.take(), st.take());
            }
            ExprKind::Return(v) => {
                if let Some(v) = v {
                    self.consume(v, st);
                }
                self.leave_scopes(0, e.span, Exit::Return, st);
                *st = None;
            }
            ExprKind::Const(_) => {}
        }
        if e.ty == Ty::Never {
            *st = None;
        }
    }

    /// A quantum condition: the qubits it tests control what the `if` does,
    /// and stay where they are.
    fn look_condition(&mut self, e: &Expr, st: &mut State) {
        match &e.kind {
            ExprKind::Unary { operand, .. } => self.look_condition(operand, st),
            ExprKind::Logical { lhs, rhs, .. } | ExprKind::Binary { lhs, rhs, .. } if e.ty == Ty::Qubit => {
                self.look_condition(lhs, st);
                self.look_condition(rhs, st);
            }
            _ => self.examine(e, false, st),
        }
    }

    /// Evaluates `e` for its value when that `moves` it, or when it is not a
    /// place; otherwise reads it in place.
    fn examine(&mut self, e: &Expr, moves: bool, st: &mut State) {
        if moves || !e.is_place() {
            self.consume(e, st);
        } else {
            self.place(e, Use::Look, st);
        }
    }

    /// Walks the body of the loop `id` until the state at its start stops
    /// growing, leaving in `st` the state after the loop. `body` runs one
    /// iteration from the state it is given, leaving the state at the
    /// iteration's end, and returns the state at the point where the loop may
    /// be left without a `break` (for `while` and `for`, where the condition
    /// or the range is tested).
    fn repeat(&mut self, id: LoopId, st: &mut State, mut body: impl FnMut(&mut Self, &mut State) -> State) {
        self.loop_scopes.insert(id, self.scopes.len());
        let entry = st.clone();
        let mut head = entry.clone();
        let test = loop {
            self.loops.insert(id, LoopExits::default());
            let mut s = head.clone();
            let test = body(self, &mut s);
            let continues = self.loops.get_mut(&id).and_then(|x| x.continues.take());
            let next = join(join(entry.clone(), s), continues);
            if same(&next, &head) {
                break test;
            }
            head = next;
        };
        let exits = self.loops.remove(&id).unwrap_or_default();
        *st = join(test, exits.breaks);
    }

    /// Whether a pattern binds part of the value that cannot be copied, which
    /// moves that part into the binding.
    fn binds_owned(&self, p: &Pat) -> bool {
        match &p.kind {
            PatKind::Bind(_) => !self.copyable(p.ty),
            PatKind::Struct { fields } | PatKind::Variant { fields, .. } => {
                fields.iter().any(|(_, f)| self.binds_owned(f))
            }
            PatKind::Wild | PatKind::BindRef(_) | PatKind::Const(_) => false,
        }
    }

    /// Reads a place for its value: a copy, or a move when its type cannot be
    /// copied.
    fn read(&mut self, e: &Expr, st: &mut State) {
        let moves = !self.copyable(e.ty);
        // a value whose type has `$drop` is given to it whole as it is
        // destroyed, so no part of it may be moved out first
        if moves
            && let ExprKind::Field { base, .. } = &e.kind
            && let Ty::Adt(owner) = base.ty
            && self.unit.drops.contains_key(&owner)
        {
            let ty = self.unit.types.adt_name(owner, self.interner);
            if self.once(e.span, Code::Es12) {
                self.diags.push(
                    Diagnostic::new(Code::Es12)
                        .with_message(format!("this would move a field out of a `{ty}`, whose type has `$drop`"))
                        .at(e.span)
                        .with_note(format!(
                            "`$drop` runs on the whole `{ty}` when it is destroyed, so every field must still be there"
                        ))
                        .with_help("take a reference to the field instead, or give the type a method that hands the field over"),
                );
            }
            return;
        }
        if let Some(path) = Self::path_of(e) {
            self.check_use(&path, e.span, st);
            if moves && let Some(places) = st {
                places.fill(&path);
                places.empty.insert(path, Empty::Moved(e.span));
            }
            return;
        }
        match &e.kind {
            ExprKind::Global(g) => {
                let global = self.unit.global(*g);
                if moves && !global.constant {
                    let place = match global.kind {
                        GlobalKind::Persist(_) => format!("the persistent object `{}`", self.interner.resolve(global.name)),
                        _ => format!("the unit-scope object `{}`", self.interner.resolve(global.name)),
                    };
                    self.cannot_move(e, &place);
                }
            }
            // a part of a constant, like the whole, is a copy of its value
            ExprKind::Field { base, .. } => {
                self.place(base, Use::Look, st);
                if moves && !self.part_of_constant(base) {
                    self.cannot_move(e, where_of(base));
                }
            }
            ExprKind::Index { base, index } => {
                self.place(base, Use::Look, st);
                self.consume(index, st);
                if moves && !self.part_of_constant(base) {
                    self.cannot_move(e, "an element of an array");
                }
            }
            ExprKind::Deref(r) => {
                self.consume(r, st);
                if moves {
                    self.cannot_move(e, "the object a reference refers to");
                }
            }
            _ => self.consume(e, st),
        }
    }

    /// Evaluates a place without taking its value: its own sub-expressions
    /// run, and it must hold a value, unless it is being stored into.
    fn place(&mut self, e: &Expr, how: Use, st: &mut State) {
        if st.is_none() {
            return;
        }
        if let Some(path) = Self::path_of(e) {
            match how {
                Use::Look => self.check_use(&path, e.span, st),
                Use::Write => {
                    self.overwritten(&path, e, st);
                    self.touch(e, st);
                    // whatever contains the place must hold a value already:
                    // a field cannot be stored into an empty structure
                    if !path.1.is_empty() {
                        let parent = (path.0, path.1[..path.1.len() - 1].to_vec());
                        self.check_use(&parent, e.span, st);
                    }
                    if let Some(places) = st {
                        places.fill(&path);
                    }
                }
            }
            return;
        }
        match &e.kind {
            ExprKind::Global(_) => {}
            ExprKind::Field { base, .. } => self.place(base, Use::Look, st),
            ExprKind::Index { base, index } => {
                self.place(base, Use::Look, st);
                self.consume(index, st);
            }
            ExprKind::Deref(r) => self.consume(r, st),
            // a value that is not a place: evaluated, and used up
            _ => self.consume(e, st),
        }
    }
}

/// Where a field is read from, when that is not a binding the function owns.
fn where_of(base: &Expr) -> &'static str {
    match base.kind {
        ExprKind::Deref(_) => "the object a reference refers to",
        ExprKind::Index { .. } => "an element of an array",
        _ => "a unit-scope object",
    }
}

/// The bindings a pattern makes.
fn pattern_locals(p: &Pat, out: &mut Vec<LocalId>) {
    match &p.kind {
        PatKind::Bind(l) | PatKind::BindRef(l) => out.push(*l),
        PatKind::Struct { fields } | PatKind::Variant { fields, .. } => {
            for (_, f) in fields {
                pattern_locals(f, out);
            }
        }
        PatKind::Wild | PatKind::Const(_) => {}
    }
}

/// A value of a type that is moved.
fn describe_owned(ty: Ty) -> &'static str {
    match ty {
        Ty::Adt(_) => "a structure or enumeration value",
        Ty::Array(_) => "an array whose elements cannot be copied",
        Ty::Growable(_) => "a growable array, which owns its elements",
        _ => "this value",
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};

    const P: &str = "struct P { a: Q, b: Q, n: i32 }\nstruct Q { v: i32 }\nfn take(p: P) { }\nfn take_q(q: Q) { }\n";

    fn with(src: &str) -> String {
        format!("{P}{src}")
    }

    #[test]
    fn using_a_moved_structure_is_refused() {
        assert_eq!(codes(&with("fn f(p: P) { take(p); take(p); }")), [Code::Es10]);
        assert_eq!(codes(&with("fn f(p: P) { let q = p; let r = p; }")), [Code::Es10]);
    }

    #[test]
    fn copyable_values_are_never_moved() {
        check("fn g(x: i32) { }\nfn f(x: i32, s: *[u8]) { g(x); g(x); let t = s; let u = s; }");
        check("fn f(a: [i32; 3]) -> [i32; 3] { let b = a; a }");
    }

    #[test]
    fn a_reference_does_not_move() {
        check(&with("fn peek(p: *P) { }\nfn f(p: P) { peek(&p); peek(&p); take(p); }"));
    }

    #[test]
    fn fields_are_tracked_separately() {
        check(&with("fn f(p: P) { take_q(p.a); take_q(p.b); let n = p.n; }"));
        assert_eq!(codes(&with("fn f(p: P) { take_q(p.a); take(p); }")), [Code::Es10]);
        assert_eq!(codes(&with("fn f(p: P) { take(p); let n = p.n; }")), [Code::Es10]);
    }

    #[test]
    fn assigning_again_makes_a_binding_usable() {
        check(&with("fn f(q: Q) { take_q(q); q = Q { v: 1 }; take_q(q); }"));
    }

    #[test]
    fn a_move_on_one_branch_counts_after_the_branches_meet() {
        assert_eq!(codes(&with("fn f(q: Q, c: bool) { if c { take_q(q); } take_q(q); }")), [Code::Es10]);
        check(&with("fn f(q: Q, c: bool) { if c { take_q(q); } else { take_q(q); } }"));
    }

    #[test]
    fn a_move_inside_a_loop_affects_the_next_iteration() {
        assert_eq!(codes(&with("fn f(q: Q) { loop { take_q(q); } }")), [Code::Es10]);
        check(&with("fn f(q: Q) { loop { take_q(q); break; } }"));
        assert_eq!(codes(&with("fn f(q: Q) { for i in 0..3 { take_q(q); } }")), [Code::Es10]);
    }

    #[test]
    fn a_move_after_a_return_is_unreachable() {
        check(&with("fn f(q: Q) { take_q(q); return; }"));
        check(&with("fn f(q: Q, c: bool) { if c { take_q(q); return; } take_q(q); }"));
    }

    #[test]
    fn a_value_cannot_leave_an_element_a_referent_or_an_object() {
        assert_eq!(codes(&with("fn f(a: [Q; 2]) { take_q(a[0]); }")), [Code::Es12]);
        assert_eq!(codes(&with("fn f(p: *P) { take_q(p.a); }")), [Code::Es12]);
        assert_eq!(codes(&with("let O: Q = Q { v: 1 };\nfn f() { take_q(O); }")), [Code::Es12]);
        check(&with("fn f(p: *P, a: [Q; 2]) -> i32 { p.a.v + a[1].v }"));
    }

    #[test]
    fn a_match_moves_only_what_its_arms_bind() {
        let e = "enum E { A(Q), B }\n";
        check(&with(&format!("{e}fn f(x: E) {{ match x {{ E::A(_) => 1, E::B => 0 }}; match x {{ E::B => 0, _ => 1 }}; }}")));
        assert_eq!(
            codes(&with(&format!("{e}fn f(x: E) {{ match x {{ E::A(q) => take_q(q), E::B => {{ }} }}; let y = x; }}"))),
            [Code::Es10]
        );
        // through a reference, `q` refers to the part instead of taking it
        assert_eq!(
            codes(&with(&format!("{e}fn f(x: *E) {{ match x {{ E::A(q) => take_q(q), E::B => {{ }} }} }}"))),
            [Code::Es06]
        );
        check(&with(&format!(
            "{e}fn look(q: *Q) {{ }}\nfn f(x: *E) {{ match x {{ E::A(q) => look(q), E::B => {{ }} }}; match x {{ E::B => {{ }}, _ => {{ }} }} }}"
        )));
    }

    #[test]
    fn a_binding_must_be_given_a_value_on_every_path() {
        check("fn f(c: bool) -> i32 { let x: i32; if c { x = 1; } else { x = 2; } x }");
        assert_eq!(codes("fn f(c: bool) -> i32 { let x: i32; if c { x = 1; } x }"), [Code::Es11]);
        assert_eq!(codes("fn f() -> i32 { let x: i32; x }"), [Code::Es11]);
        check("fn f() -> i32 { let x: i32; loop { x = 3; break; } x }");
        assert_eq!(codes("fn f(n: u32) -> i32 { let x: i32; for i in 0..n { x = 3; } x }"), [Code::Es11]);
    }

    #[test]
    fn a_field_cannot_be_given_before_its_structure() {
        assert_eq!(codes(&with("fn f() { let q: Q; q.v = 1; }")), [Code::Es11]);
    }

    #[test]
    fn a_binding_declared_without_a_value_is_given_one_before_it_is_read() {
        check("fn f(c: bool) -> i32 { let x: i32; if c { x = 1; } else { x = 2; } x }");
        check("fn f() -> i32 { let x: i32; x = 1; x = 2; x }");
        check("fn f() { let x: i32; loop { x = 1; } }");
        assert_eq!(codes("fn f(c: bool) -> i32 { let x: i32; if c { x = 1; } x }"), [Code::Es11]);
    }
}
