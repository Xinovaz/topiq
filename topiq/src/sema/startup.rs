//! Unit initialisers.
//!
//! A unit may name, in its `#unit` directive, a function that runs once before
//! `main`: its initialiser. A unit object declared without a value (as a
//! growable array must be, since no constant can own an allocation) is
//! given one there. Every such object must be assigned by the initialiser
//! exactly once on every path through it, and not read before that; this
//! module checks that by following the initialiser's paths the way move
//! checking does, counting the assignments each object has had.
//!
//! The initialiser is not an ordinary function: nothing may call it, and it
//! is not part of the unit's interface. The program runs the initialisers of
//! all its units before `main`, those of imported units first.
//!
//! What the initialiser calls is not followed. A function it calls could read
//! an object not yet assigned, and would see the zero bytes its storage
//! starts with; keeping such reads out is the initialiser's own business.

use std::collections::HashMap;

use crate::diag::{Code, Diagnostic};
use crate::intern::Interner;
use crate::span::Span;
use crate::tir::visit::{self, Visit};
use crate::tir::{Block, Expr, ExprKind, GlobalId, LoopId, Stmt, Ty, Unit};

use super::body::Checker;

impl Checker<'_, '_> {
    /// The initialiser named where it may not be: called, or used as a value.
    pub(super) fn initializer_used(&self, name: &str, span: Span) -> Diagnostic {
        Diagnostic::new(Code::Eu06)
            .with_message(format!("`{name}` is this unit's initialiser, so the program may not call it"))
            .at(span)
            .with_note(
                "the initialiser runs exactly once, before `main`, and calling it again would \
                 give objects that already have values second ones",
            )
            .with_help(format!("move what `{name}` does that is needed here into another function, and call that"))
    }
}

/// How many times an object may have been assigned at a point: one bit each
/// for none, once, and more than once.
type Counts = u8;
const NONE: Counts = 1;
const ONCE: Counts = 2;
const MANY: Counts = 4;

/// The counts of every object the initialiser must assign, or `None` where
/// control cannot reach.
type State = Option<Vec<Counts>>;

fn join(a: State, b: State) -> State {
    match (a, b) {
        (None, s) | (s, None) => s,
        (Some(a), Some(b)) => Some(a.iter().zip(&b).map(|(x, y)| x | y).collect()),
    }
}

/// Checks that the unit's initialiser assigns each object declared without a
/// value exactly once on every path, and reads none of them first.
pub fn check(unit: &Unit, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    let Some(init) = unit.init else { return };
    let objects: Vec<GlobalId> = unit
        .globals
        .iter()
        .enumerate()
        .filter(|(_, g)| g.startup)
        .map(|(i, _)| GlobalId(i as u32))
        .collect();
    if objects.is_empty() {
        return;
    }

    // follow each object's assignments through the initialiser
    let f = unit.func(init);
    let names: Vec<&str> = objects.iter().map(|&g| interner.resolve(unit.global(g).name)).collect();
    let mut flow = Flow {
        objects: &objects,
        names: &names,
        st: Some(vec![NONE; objects.len()]),
        loops: HashMap::new(),
        exits: None,
        reported: vec![false; objects.len()],
        found: Vec::new(),
    };
    flow.block(&f.body);
    let end = join(flow.st.take(), flow.exits.take());
    diags.append(&mut flow.found);

    // then each one it can end without having assigned
    if let Some(counts) = end {
        for (i, &c) in counts.iter().enumerate() {
            if c & NONE != 0 && !flow.reported[i] {
                let g = unit.global(objects[i]);
                let name = names[i];
                diags.push(
                    Diagnostic::new(Code::Ec06)
                        .with_message(if c == NONE {
                            format!("the initialiser never gives `{name}` its value")
                        } else {
                            format!("the initialiser gives `{name}` its value on some paths but not all")
                        })
                        .at(f.span)
                        .also(g.span, "declared here without a value")
                        .with_note(
                            "an object declared without a value is given one by the unit's \
                             initialiser, on every path through it, before `main` runs",
                        )
                        .with_help(format!("assign `{name} = …` in the initialiser, on every path")),
                );
            }
        }
    }
}

/// Whether `e` reads the object `g` anywhere.
fn mentions(e: &Expr, g: GlobalId) -> bool {
    struct Finds(GlobalId, bool);
    impl Visit for Finds {
        fn expr(&mut self, e: &Expr) {
            if matches!(e.kind, ExprKind::Global(x) if x == self.0) {
                self.1 = true;
            }
            visit::walk_expr(self, e);
        }
    }
    let mut f = Finds(g, false);
    f.expr(e);
    f.1
}

struct Flow<'a> {
    objects: &'a [GlobalId],
    names: &'a [&'a str],
    st: State,
    /// Each loop's states where it continues and where it is left.
    loops: HashMap<LoopId, (State, State)>,
    /// The states where the initialiser returns early.
    exits: State,
    /// Which objects have been reported, so that each is reported once.
    reported: Vec<bool>,
    found: Vec<Diagnostic>,
}

impl Flow<'_> {
    fn index(&self, g: GlobalId) -> Option<usize> {
        self.objects.iter().position(|&o| o == g)
    }

    fn assign(&mut self, i: usize, span: Span) {
        let Some(st) = &mut self.st else { return };
        let c = st[i];
        let next = (if c & NONE != 0 { ONCE } else { 0 }) | (if c & (ONCE | MANY) != 0 { MANY } else { 0 });
        st[i] = next;
        if next & MANY != 0 && !self.reported[i] {
            self.reported[i] = true;
            self.found.push(
                Diagnostic::new(Code::Ec06)
                    .with_message(format!("the initialiser may give `{}` a value more than once", self.names[i]))
                    .at(span)
                    .with_note(
                        "an object declared without a value is given exactly one by the \
                         initialiser; a second assignment, including one repeated by a loop, \
                         would replace it",
                    )
                    .with_help("assign it once, after deciding what the value should be"),
            );
        }
    }

    fn read(&mut self, i: usize, span: Span) {
        let Some(st) = &self.st else { return };
        if st[i] & NONE != 0 && !self.reported[i] {
            self.reported[i] = true;
            self.found.push(
                Diagnostic::new(Code::Ec06)
                    .with_message(format!("`{}` is used here before the initialiser gives it its value", self.names[i]))
                    .at(span)
                    .with_note("until the initialiser assigns it, the object holds nothing meaningful")
                    .with_help("assign it first"),
            );
        }
    }

    /// Runs a loop body until the states at its head stop changing.
    fn repeat(&mut self, id: LoopId, mut body: impl FnMut(&mut Self)) -> State {
        let mut head = self.st.clone();
        loop {
            self.loops.insert(id, (None, None));
            self.st = head.clone();
            body(self);
            let (cont, _) = self.loops.get(&id).cloned().unwrap_or((None, None));
            let next = join(head.clone(), join(self.st.take(), cont));
            if next == head {
                break;
            }
            head = next;
        }
        head
    }
}

impl Visit for Flow<'_> {
    fn expr(&mut self, e: &Expr) {
        if self.st.is_none() {
            return;
        }
        match &e.kind {
            ExprKind::Global(g) => {
                if let Some(i) = self.index(*g) {
                    self.read(i, e.span);
                }
            }
            ExprKind::Assign { place, value } => {
                self.expr(value);
                match place.kind {
                    // `X += 1` changes the value `X` has; only an assignment
                    // that does not read `X` gives it one
                    ExprKind::Global(g) if self.index(g).is_some() && !mentions(value, g) => {
                        let i = self.index(g).expect("checked");
                        self.assign(i, place.span);
                    }
                    _ => self.expr(place),
                }
            }
            ExprKind::If { cond, then, els } => {
                self.expr(cond);
                let before = self.st.clone();
                self.block(then);
                let after_then = self.st.take();
                self.st = before;
                if let Some(x) = els {
                    self.expr(x);
                }
                self.st = join(after_then, self.st.take());
            }
            ExprKind::Logical { lhs, rhs, .. } => {
                self.expr(lhs);
                let before = self.st.clone();
                self.expr(rhs);
                self.st = join(before, self.st.take());
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                let before = self.st.take();
                let mut out = None;
                for a in arms {
                    self.st = before.clone();
                    self.expr(&a.body);
                    out = join(out, self.st.take());
                }
                self.st = out;
            }
            ExprKind::Loop { id, body } => {
                self.repeat(*id, |me| me.block(body));
                let (_, breaks) = self.loops.remove(id).unwrap_or((None, None));
                self.st = breaks;
            }
            ExprKind::While { id, cond, body } => {
                let head = self.repeat(*id, |me| {
                    me.expr(cond);
                    me.block(body);
                });
                // it is left when the condition fails, or by `break`
                self.st = head;
                self.expr(cond);
                let (_, breaks) = self.loops.remove(id).unwrap_or((None, None));
                self.st = join(self.st.take(), breaks);
            }
            ExprKind::ForRange { id, start, end, body, .. } => {
                self.expr(start);
                self.expr(end);
                let head = self.repeat(*id, |me| me.block(body));
                let (_, breaks) = self.loops.remove(id).unwrap_or((None, None));
                self.st = join(head, breaks);
            }
            ExprKind::Break { target, value } => {
                if let Some(v) = value {
                    self.expr(v);
                }
                let st = self.st.take();
                let entry = self.loops.entry(*target).or_insert((None, None));
                entry.1 = join(entry.1.take(), st);
            }
            ExprKind::Continue { target } => {
                let st = self.st.take();
                let entry = self.loops.entry(*target).or_insert((None, None));
                entry.0 = join(entry.0.take(), st);
            }
            ExprKind::Return(value) => {
                if let Some(v) = value {
                    self.expr(v);
                }
                let st = self.st.take();
                self.exits = join(self.exits.take(), st);
            }
            _ => {
                visit::walk_expr(self, e);
                // a call that never returns ends the path
                if e.ty == Ty::Never {
                    self.st = None;
                }
            }
        }
    }

    fn block(&mut self, b: &Block) {
        for s in &b.stmts {
            match s {
                Stmt::Let { init: Some(e), .. } | Stmt::LetPat { init: e, .. } | Stmt::Expr(e) => self.expr(e),
                Stmt::Let { init: None, .. } => {}
            }
        }
        if let Some(v) = &b.value {
            self.expr(v);
        }
    }
}
