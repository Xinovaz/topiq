//! Map locales: a family of operators or of states, one for each value of a
//! key register, and the two ways of reading one.
//!
//! `qmap Name: [qubit; k] -> V { key: entry, … }` declares one, at unit
//! scope or in a block. `query Name[key]` is the coherent reading, which
//! does not measure or consume the key: for operator entries, an operator
//! applying each entry controlled on the key register holding its key; for
//! state entries, a register prepared in each entry's state under the same
//! control, and so entangled with the key. `measure Name[key]` is the
//! classical reading: the key is measured, and it gives the index measured
//! and the entry there (for state entries, a register prepared in that
//! state). `Name[key]` alone is neither, and is `EQ09`.
//!
//! A map locale is a value too, of type `qmap<[qubit; k], V>`: its name
//! written alone gives it, and it may be held and passed like any value
//! fixed while circuits are generated, and read the same two ways.
//!
//! A structure or enumeration may be read the same two ways by defining the
//! operator methods `$query` and `$lookup`, which `query a[k]` and
//! `measure a[k]` call.
//!
//! # Rules
//!
//! - **The key type is a register,** `qubit` or `[qubit; k]`, and every key
//!   is a constant below 2^k, given once (`EC07`, `ES06`), with at most
//!   1024 entries (`EM05`). Distinct basis keys are orthogonal, so the
//!   entries lie on orthogonal supports.
//! - **The entry type is a state of a register, or an operator taking its
//!   qubits through handles and giving back nothing,** so that each entry
//!   can be applied under its sector's control (`ES06`). A state entry is
//!   `prep` of a state as wide as the entry type.
//! - **A map locale is chosen while circuits are generated:** an `[entry]`
//!   operator cannot be given one when its circuit runs (`EQ16`).

use std::collections::HashMap;

use crate::ast::{self, ItemKind};
use crate::diag::{Code, Diagnostic, Limit};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{Expr, ExprKind, Named, QmapDef, QuantumOp, Ty};

use super::body::Checker;
use super::items::{Progress, UnitCx};

/// A map locale's declaration, and how far resolving it has got.
pub struct QmapSrc<'a> {
    kind: &'a ItemKind,
    name: Spanned<Symbol>,
    state: Progress,
    index: Option<usize>,
    /// For one a block declares, the layers of block items it sees.
    layers: Option<Vec<super::local_items::Layer>>,
}

/// The map locales a unit declares.
pub type QmapDecls<'a> = HashMap<Symbol, QmapSrc<'a>>;

impl<'a> UnitCx<'a> {
    /// Declares a map locale at unit scope.
    pub(super) fn declare_qmap(&mut self, kind: &'a ItemKind, name: Spanned<Symbol>, span: Span) {
        if self.quantum {
            let earlier = self
                .qmap_decls
                .get(&name.node)
                .map(|d| d.name.span)
                .or_else(|| self.scope.get(name.node).and_then(|_| self.scope.span_of(name.node)));
            if let Some(earlier) = earlier {
                let text = self.interner.resolve(name.node);
                self.report(super::report::duplicate(name.span, earlier, text, "at unit scope"));
                return;
            }
        }
        self.declare_qmap_as(kind, name, name.node, span);
    }

    /// Declares a map locale written `name` under the name `key`: its own
    /// at unit scope, and one no other declaration shares for one a block
    /// declares.
    pub(super) fn declare_qmap_as(&mut self, kind: &'a ItemKind, name: Spanned<Symbol>, key: Symbol, span: Span) {
        if !self.quantum {
            self.report(
                Diagnostic::new(Code::Eu02)
                    .with_message("only a quantum unit can declare a map locale")
                    .at(span)
                    .with_note("a map locale's entries act on qubits")
                    .with_help("move this into a unit that begins with `#unit quantum`"),
            );
            return;
        }
        self.check_reserved(name.node, name.span);
        self.qmap_decls.insert(
            key,
            QmapSrc {
                kind,
                name,
                state: Progress::Pending,
                index: None,
                layers: None,
            },
        );
    }

    /// Gives the map locale `key`, which a block declares, the layers of
    /// block items it sees.
    pub(super) fn scope_qmap(&mut self, key: Symbol, layers: Vec<super::local_items::Layer>) {
        if let Some(d) = self.qmap_decls.get_mut(&key) {
            d.layers = Some(layers);
        }
    }

    /// Resolves every map locale, in the order written.
    pub(super) fn check_qmaps(&mut self) {
        let mut names: Vec<(Span, Symbol)> = self.qmap_decls.iter().map(|(&k, d)| (d.name.span, k)).collect();
        names.sort_by_key(|(s, _)| s.start);
        for (_, name) in names {
            self.qmap(name);
        }
    }

    /// Whether `name` is a map locale in scope.
    pub fn is_qmap(&self, name: Symbol) -> bool {
        self.qmap_decls.contains_key(&self.geometry_key(name))
    }

    /// The map locale `name`, by position, resolving it first if it has not
    /// been; `None` once why it has none is reported.
    pub fn qmap(&mut self, name: Symbol) -> Option<usize> {
        let name = self.geometry_key(name);
        let d = self.qmap_decls.get_mut(&name)?;
        match d.state {
            Progress::Done => return d.index,
            Progress::Active => {
                let (span, text) = (d.name.span, self.interner.resolve(d.name.node));
                self.report(
                    Diagnostic::new(Code::Es16)
                        .with_message(format!("the map locale `{text}` is written in terms of itself"))
                        .at(span),
                );
                return None;
            }
            Progress::Pending => d.state = Progress::Active,
        }
        let (kind, decl_name) = (d.kind, d.name);
        let make = |cx: &mut Self| match kind {
            ItemKind::Qmap { key, entry, entries, .. } => cx.qmap_def(key, entry, entries).map(|def| {
                cx.unit.geometry.qmaps.push(Named {
                    name: decl_name.node,
                    def,
                    span: decl_name.span,
                });
                cx.unit.geometry.qmaps.len() - 1
            }),
            _ => None,
        };
        let index = match d.layers.clone() {
            Some(layers) => self.with_layers(layers, make),
            None => make(self),
        };
        let d = self.qmap_decls.get_mut(&name).expect("declared");
        d.state = Progress::Done;
        d.index = index;
        index
    }

    fn qmap_def(
        &mut self,
        key: &Spanned<ast::Type>,
        entry: &Spanned<ast::Type>,
        entries: &'a [ast::QmapEntry],
    ) -> Option<QmapDef> {
        let width = self.register_type_width(key)?;
        let entry_ty = self.written_type(entry)?;
        self.entry_type(entry_ty, entry.span)?;
        let first = entries.first().map_or(key.span, |e| e.key.span);
        if let Some(d) = Limit::QmapEntries.check(u32::try_from(entries.len()).unwrap_or(u32::MAX), first) {
            self.report(d);
            return None;
        }
        let bound: u128 = 1u128 << width.min(127);
        let mut out: Vec<(u64, Expr)> = Vec::with_capacity(entries.len());
        let mut seen: HashMap<u64, Span> = HashMap::new();
        let mut ok = true;
        // an entry of a state type is the preparation of its state
        let states = self.unit.types.is_quantum(entry_ty);
        for e in entries {
            let k = self.const_value(&e.key, Ty::USIZE, "a map locale's key").and_then(|v| v.as_int());
            let value = if states {
                self.state_entry(&e.value, entry_ty)
            } else {
                super::body::check_const_as(self, &e.value, Some(entry_ty), "a map locale's entry")
            };
            let Some(k) = k else {
                ok = false;
                continue;
            };
            if value.ty == Ty::Never {
                ok = false;
                continue;
            }
            let Some(k) = u64::try_from(k).ok().filter(|&k| u128::from(k) < bound) else {
                self.report(
                    Diagnostic::new(Code::Ec07)
                        .with_message(format!("the key {k} is out of range for a key register of {width} qubits"))
                        .at(e.key.span)
                        .with_note(format!("a register of {width} qubits holds the keys 0 to {}", bound - 1)),
                );
                ok = false;
                continue;
            };
            if let Some(&earlier) = seen.get(&k) {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("the key {k} is given two entries"))
                        .at_with(e.key.span, "here")
                        .also(earlier, "and here")
                        .with_note("each key of a map locale has one entry, on its own basis state of the key register"),
                );
                ok = false;
                continue;
            }
            seen.insert(k, e.key.span);
            out.push((k, value));
        }
        ok.then_some(QmapDef {
            key: width,
            entry: entry_ty,
            entries: out,
        })
    }

    /// An entry of a map locale of states, of type `ty`: `prep` of a state
    /// as wide as the type.
    fn state_entry(&mut self, e: &Spanned<ast::Expr>, ty: Ty) -> Expr {
        let error = || super::body::Checker::error(e.span);
        let ast::Expr::Prep(arg) = &e.node else {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("an entry of a map locale of states is a state prepared, `prep` of one")
                    .at(e.span)
                    .with_help("write the entry as `prep |…>`, or `prep` of a constant `quon::State`"),
            );
            return error();
        };
        let Some(p) = self.prep_expr(arg, e.span) else { return error() };
        let want = self.unit.types.qubits(ty);
        let got = self.unit.types.qubits(p.ty);
        if want != got {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("this state is of {got} qubits, and the map locale's entries of {want}"))
                    .at(e.span),
            );
            return error();
        }
        Expr { ty, ..p }
    }

    /// Whether `t` can be a map locale's entry type.
    fn entry_type(&mut self, t: Ty, span: Span) -> Option<()> {
        match entry_problem(&self.unit.types, self.interner, t, span) {
            Some(d) => {
                self.report(d);
                None
            }
            None => Some(()),
        }
    }
}

/// Why `t` cannot be a map locale's entry type, if it cannot: an entry is a
/// state of a register, or an operator taking its qubits through handles
/// and giving back nothing.
pub(super) fn entry_problem(types: &crate::tir::TypeTable, interner: &crate::intern::Interner, t: Ty, span: Span) -> Option<Diagnostic> {
    if t == Ty::Qubit || types.as_array(t).is_some_and(|(e, _)| e == Ty::Qubit) {
        return None;
    }
    let Some((params, ret)) = types.as_sig(t) else {
        let what = super::report::describe(t, types, interner);
        return Some(
            Diagnostic::new(Code::Es06)
                .with_message(format!("a map locale's entries are states of a register or operators, and this is {what}"))
                .at(span)
                .with_help("write the entry type as `[qubit; N]`, for states, or `fn(*[qubit; N])`, for operators"),
        );
    };
    let by_value = params.iter().any(|&p| types.is_quantum(p));
    if by_value || types.is_quantum(ret) || ret != Ty::Void {
        return Some(
            Diagnostic::new(Code::Es06)
                .with_message("a map locale's entries take their qubits through handles and give back nothing")
                .at(span)
                .with_note(
                    "a coherent query applies every entry, each under the control of its key's basis \
                     state, so none can take its qubits away or give a result back",
                )
                .with_help("write the entry type as `fn(*[qubit; N])`"),
        );
    }
    None
}

impl Checker<'_, '_> {
    /// The map locale `e` names, when it is a single name that no binding
    /// in scope hides.
    fn named_qmap(&mut self, e: &Spanned<ast::Expr>) -> Option<Symbol> {
        let ast::Expr::Path { path, args } = &e.node else { return None };
        let [name] = path.segments.as_slice() else { return None };
        (args.is_empty() && self.scopes.in_blocks(name.node).is_none() && self.cx.is_qmap(name.node)).then_some(name.node)
    }

    /// `query t[k]`.
    pub fn query(&mut self, map: &Spanned<ast::Expr>, index: &Spanned<ast::Expr>, span: Span) -> Expr {
        let Some(name) = self.named_qmap(map) else {
            let receiver = self.expr(map);
            if self.shallow(receiver.ty) == Ty::Never {
                return Checker::error(span);
            }
            if matches!(self.shallow(receiver.ty), Ty::Qmap(_)) {
                return self.read_table(receiver, index, true, span).unwrap_or_else(|| Checker::error(span));
            }
            let key = self.expr(index);
            return self.operator_call("query", "query", receiver, Some(key), span);
        };
        let Some(id) = self.cx.qmap(name) else {
            return Checker::error(span);
        };
        let (width, entry) = {
            let def = &self.cx.unit.geometry.qmaps[id].def;
            (def.key, def.entry)
        };
        let key = self.expr(index);
        let Some(key) = self.key_handle(key, width) else {
            return Checker::error(span);
        };
        Expr {
            kind: ExprKind::Quantum {
                op: QuantumOp::Query(id),
                args: vec![key],
            },
            ty: entry,
            span,
        }
    }

    /// `measure t[k]`, when `operand` is `t[k]` of a map locale or of a type
    /// with `$lookup`; `None` when it is not an index.
    pub(super) fn lookup(&mut self, operand: &Spanned<ast::Expr>, span: Span) -> Option<Expr> {
        let ast::Expr::Index { receiver, index } = &operand.node else { return None };
        let Some(name) = self.named_qmap(receiver) else {
            let r = self.expr(receiver);
            let rt = self.shallow(r.ty);
            if let Ty::Qmap(_) = rt {
                return self.read_table(r, index, false, span);
            }
            if self.operator_method(rt, "lookup").is_none() {
                // an ordinary index, measured
                let e = self.index_checked(r, receiver.span, index, operand.span);
                return Some(self.measured(e, false, span));
            }
            let key = self.expr(index);
            return Some(self.operator_call("lookup", "measure", r, Some(key), span));
        };
        let Some(id) = self.cx.qmap(name) else {
            return Some(Checker::error(span));
        };
        let (width, entry) = {
            let def = &self.cx.unit.geometry.qmaps[id].def;
            (def.key, def.entry)
        };
        let key = self.expr(index);
        if !self.measured_key(&key, width as u64, index.span) {
            return Some(Checker::error(span));
        }
        let ty = self.types_mut().tuple(vec![Ty::USIZE, entry]);
        Some(Expr {
            kind: ExprKind::Quantum {
                op: QuantumOp::Lookup(id),
                args: vec![key],
            },
            ty,
            span,
        })
    }

    /// Whether `key`, written at `at`, is a key register of `width` qubits
    /// for `measure t[k]`, reporting it when it is not.
    fn measured_key(&mut self, key: &Expr, width: u64, at: Span) -> bool {
        let kt = self.shallow(key.ty);
        if self.register_width(kt) == Some(width) {
            return true;
        }
        let what = self.describe(kt);
        self.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!("the key register is of {width} qubits, and this is {what}"))
                .at(at)
                .with_note("`measure t[k]` measures the key register, which it takes"),
        );
        false
    }

    /// `t[k]` of a map locale written with neither `query` nor `measure`:
    /// `EQ09`. `None` when `receiver` is not a map locale.
    pub(super) fn bare_qmap_index(&mut self, receiver: &Spanned<ast::Expr>, span: Span) -> Option<Expr> {
        let name = self.named_qmap(receiver)?;
        let text = self.interner.resolve(name);
        self.report(
            Diagnostic::new(Code::Eq09)
                .with_message(format!("`{text}[…]` does not say how the map locale is read"))
                .at(span)
                .with_note(
                    "a map locale is read coherently, applying each entry under the control of its key \
                     without measuring, or classically, measuring the key and taking the entry it held",
                )
                .with_help(format!("write `query {text}[k]` or `measure {text}[k]`")),
        );
        Some(Checker::error(span))
    }

    /// A map locale's name written as a value, of type `qmap<K, V>`. `None`
    /// when `name` is not one.
    pub(super) fn qmap_as_value(&mut self, e: &Spanned<ast::Expr>) -> Option<Expr> {
        let name = self.named_qmap(e)?;
        let Some(id) = self.cx.qmap(name) else {
            return Some(Checker::error(e.span));
        };
        let (key, entry) = {
            let def = &self.cx.unit.geometry.qmaps[id].def;
            (def.key as u64, def.entry)
        };
        let ty = self.types_mut().qmap(key, entry);
        Some(Expr {
            kind: ExprKind::Quantum {
                op: QuantumOp::Table(id),
                args: Vec::new(),
            },
            ty,
            span: e.span,
        })
    }

    /// `query t[k]` or `measure t[k]` of `receiver`, a map locale held as a
    /// value; `None` when it is not one.
    fn read_table(&mut self, receiver: Expr, index: &Spanned<ast::Expr>, coherent: bool, span: Span) -> Option<Expr> {
        let (width, entry) = self.types().as_qmap(self.shallow(receiver.ty))?;
        let key = self.expr(index);
        let (op, key, ty) = if coherent {
            let Some(key) = self.key_handle(key, width as usize) else { return Some(Checker::error(span)) };
            (QuantumOp::QueryAt, key, entry)
        } else {
            if !self.measured_key(&key, width, index.span) {
                return Some(Checker::error(span));
            }
            (QuantumOp::LookupAt, key, self.types_mut().tuple(vec![Ty::USIZE, entry]))
        };
        Some(Expr {
            kind: ExprKind::Quantum {
                op,
                args: vec![receiver, key],
            },
            ty,
            span,
        })
    }

    /// The key operand of a coherent query: a handle to a register of
    /// `width` qubits, taken from the register itself when that is given.
    fn key_handle(&mut self, key: Expr, width: usize) -> Option<Expr> {
        let kt = self.shallow(key.ty);
        if kt == Ty::Never {
            return None;
        }
        let (direct, target) = match self.types().as_ref(kt) {
            Some((_, t)) => (false, t),
            None => (true, kt),
        };
        if self.register_width(target) != Some(width as u64) {
            let what = self.describe(kt);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("the key register is of {width} qubits, and this is {what}"))
                    .at(key.span)
                    .with_note("a query reads the key register without measuring or taking it"),
            );
            return None;
        }
        if !direct {
            return Some(key);
        }
        if !key.is_place() {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("a query's key is a register the program holds")
                    .at(key.span)
                    .with_note("a query does not take the key, so it must be somewhere the program keeps it"),
            );
            return None;
        }
        let at = key.span;
        Some(self.reference_to(key, at))
    }
}
