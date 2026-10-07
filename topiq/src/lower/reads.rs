//! The judgements operators' bodies read.
//!
//! An operator's body may read a judgement (`@judgment(f)` or another of
//! the macros that ask of one) as a value known while its circuit is
//! generated. Each such value is derived before any body reading it is
//! generated, from the circuit of the operator it asks about, which is
//! generated first with the judgements its own body reads in place. A read
//! of a judgement that depends, through such reads, on the reading
//! operator's own circuit is refused (`EJ18`): that circuit would be
//! needed before it exists.

use std::collections::{HashMap, HashSet};

use crate::diag::{Code, Diagnostic};
use crate::intern::Interner;
use crate::judge::record::Record;
use crate::span::Span;
use crate::tir::claims::Derived;
use crate::tir::{Callee, Expr, ExprKind, FnId, Intrinsic, Unit};

use super::generate::Gen;
use super::judge::{self, Judgments};

/// What one function's body reads of judgements, and what it calls.
#[derive(Default)]
struct Body {
    reads: Vec<(u32, Span)>,
    calls: Vec<FnId>,
}

impl crate::tir::visit::Visit for Body {
    fn expr(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::Intrinsic {
                which: Intrinsic::Derived(k),
                ..
            } => self.reads.push((*k, e.span)),
            ExprKind::Call { callee: Callee::Fn(f), .. } | ExprKind::FnRef(Callee::Fn(f)) | ExprKind::ThunkRef(f) => {
                self.calls.push(*f);
            }
            ExprKind::Closure { code, .. } => self.calls.push(*code),
            _ => {}
        }
        crate::tir::visit::walk_expr(self, e);
    }
}

struct Reads<'a, 'g> {
    unit: &'a Unit,
    interner: &'a Interner,
    g: &'g mut Gen<'a>,
    bodies: Vec<Body>,
    /// What each function's body reads, with what the functions it calls
    /// read.
    needs: HashMap<FnId, Vec<(u32, Span)>>,
    /// Where each judgement is first read.
    at: HashMap<u32, Span>,
    visiting: HashSet<u32>,
    done: HashSet<u32>,
    failed: HashSet<u32>,
    pre: Judgments,
    holonomies: bool,
    diags: Vec<Diagnostic>,
}

/// Derives the value of every judgement a function body of `unit` reads,
/// and gives each to `g`, reporting the reads that have none.
pub(super) fn provide<'a>(
    unit: &'a Unit,
    interner: &'a Interner,
    g: &mut Gen<'a>,
    imported: &HashMap<FnId, Record>,
) -> Vec<Diagnostic> {
    let bodies: Vec<Body> = unit
        .fns
        .iter()
        .map(|f| {
            let mut b = Body::default();
            crate::tir::visit::Visit::block(&mut b, &f.body);
            b
        })
        .collect();
    let mut at = HashMap::new();
    for b in &bodies {
        for &(k, span) in &b.reads {
            at.entry(k).or_insert(span);
        }
    }
    if at.is_empty() {
        return Vec::new();
    }
    let mut r = Reads {
        unit,
        interner,
        g,
        bodies,
        needs: HashMap::new(),
        at,
        visiting: HashSet::new(),
        done: HashSet::new(),
        failed: HashSet::new(),
        pre: Judgments {
            records: imported.clone(),
            holonomies: HashMap::new(),
            diags: Vec::new(),
        },
        holonomies: false,
        diags: Vec::new(),
    };
    let mut wanted: Vec<u32> = r.at.keys().copied().collect();
    // a holonomy is derived with every chain's, so after the rest
    wanted.sort_by_key(|&k| (matches!(unit.derived[k as usize], Derived::Holonomy(..)), k));
    for k in wanted {
        r.visit(k);
    }
    r.diags
}

impl<'a> Reads<'a, '_> {
    /// What `f`'s body reads, with what the functions it calls read.
    fn needs(&mut self, f: FnId) -> Vec<(u32, Span)> {
        if let Some(n) = self.needs.get(&f) {
            return n.clone();
        }
        let mut seen = HashSet::from([f]);
        let mut stack = vec![f];
        let mut out = Vec::new();
        while let Some(h) = stack.pop() {
            let body = &self.bodies[h.0 as usize];
            out.extend(body.reads.iter().copied());
            for &c in &body.calls {
                if seen.insert(c) {
                    stack.push(c);
                }
            }
        }
        self.needs.insert(f, out.clone());
        out
    }

    /// The operators whose circuits the judgement numbered `k` is derived
    /// from.
    fn operators(&self, k: u32) -> Vec<FnId> {
        match self.unit.derived[k as usize] {
            Derived::Judgment(f)
            | Derived::Kernel(f)
            | Derived::Certificate(f)
            | Derived::MatrixOf(f)
            | Derived::FragmentOf(f)
            | Derived::ClassifyOp(f, _) => vec![f],
            Derived::Holonomy(chain, _) => self.unit.geometry.chains[chain]
                .def
                .stages
                .iter()
                .filter_map(|s| match s.op.kind {
                    ExprKind::FnRef(Callee::Fn(f)) => Some(f),
                    _ => None,
                })
                .collect(),
        }
    }

    fn visit(&mut self, k: u32) {
        if self.done.contains(&k) {
            return;
        }
        self.visiting.insert(k);
        let mut value = true;
        for op in self.operators(k) {
            for (k2, span) in self.needs(op) {
                if self.visiting.contains(&k2) {
                    self.cycle(k2, op, span);
                    value = false;
                    continue;
                }
                self.visit(k2);
                // one that depends on a judgement with no value has none, for
                // the reason reported there
                value &= !self.failed.contains(&k2);
            }
        }
        self.visiting.remove(&k);
        self.done.insert(k);
        let v = if value { self.value(k) } else { None };
        if v.is_none() {
            self.failed.insert(k);
        }
        self.g.provide(k, v);
    }

    /// The value of the judgement numbered `k`, once every judgement its
    /// operators read has one; `None`, as reported, when it has none.
    fn value(&mut self, k: u32) -> Option<crate::tir::Value> {
        let d = &self.unit.derived[k as usize];
        if matches!(d, Derived::Holonomy(..)) && !self.holonomies {
            self.pre.holonomies = judge::holonomies(self.unit, self.interner, self.g);
            self.holonomies = true;
        }
        let at = self.at[&k];
        match judge::derived_value(self.unit, self.interner, &mut self.pre, self.g, d, at) {
            Ok(v) => Some(v),
            Err(why) => {
                self.diags.push(why);
                None
            }
        }
    }

    /// Reports that `op`, whose judgement is read, reads at `span` the
    /// judgement numbered `k`, which is being derived and so depends on it.
    fn cycle(&mut self, k: u32, op: FnId, span: Span) {
        let name = self.interner.resolve(self.unit.func(op).name);
        let mut d = Diagnostic::new(Code::Ej18)
            .with_message(format!("this reads a judgment that depends on the circuit of `{name}`, which reads it"))
            .at(span);
        if self.at[&k] != span {
            d = d.also(self.at[&k], "the judgment is read here too");
        }
        self.diags.push(
            d.with_note(
                    "an operator's body reads a judgment while its circuit is generated, and a \
                     judgment is derived from circuits; one that depends on the reading operator's \
                     own circuit would be needed before that circuit exists",
                )
                .with_help("read the judgment of an operator the reading operator does not depend on"),
        );
    }
}
