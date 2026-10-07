//! The geometry a quantum unit declares: covers, gauges, base types, locales
//! and chains.
//!
//! Each is declared at unit scope, in the name space of types, or in a
//! block, where it is the block's own ([`super::local_items`]); each is
//! resolved the first time something names it, so they may be written in
//! any order.
//! Their states are computed exactly at the unit's conductor, and what makes
//! each well formed is decided exactly ([`crate::judge::cover`]); the results
//! are the unit's [`crate::tir::Geometry`].
//!
//! # Rules
//!
//! - **A cover's points are unit vectors** (`EJ14`), of one width, and none
//!   is given twice; a `span`'s kets are distinct, and a `code`'s generators
//!   commute and are independent (`EJ15`). A `fin` or `span` has at most 256
//!   literals (`EM05`).
//! - **A gauge is checked against the cover of each base type it is used
//!   in:** one fiducial must reach every point (`EJ10`, naming a point it
//!   misses), and cannot gauge a subspace of projective dimension one or more
//!   (`EJ09`); an atlas must reach every point (`EJ10`), with at most one
//!   patch more than the projective dimension of the register's states
//!   (`EM05`). Every fiducial is a unit vector as wide as the cover.
//! - **A base type's register is as wide as its cover's states.**
//! - **A restriction `A of B`** takes the part of `B`'s cover that `A`'s
//!   covers, under `B`'s gauge. Both are base types over registers of one
//!   width, whose covers meet (`EJ11`).
//! - **A locale's members are interfaces of its container:** each takes the
//!   container's register, by value giving it back or through a handle
//!   giving no qubits back, with classical parameters besides (`EJ12`).
//! - **A chain's stages** each apply an operator that is an interface of
//!   their base types' register, from one base type to the next: each
//!   stage's target is the next one's source. A chain has at most 64 stages
//!   (`EM05`), and is closed when it ends where it began.
//! - **None of them is a type of values:** naming one where a value's type
//!   is written is `ES06`, and declaring one in a classical unit is `EU02`.

use crate::ast::{self, ItemKind, Type};
use crate::diag::{Code, Diagnostic, Limit};
use crate::intern::Symbol;
use crate::judge::cover::{self, Cover, Gauge, Refusal};
use crate::judge::pauli::{self, Pauli};
use crate::quon::ast::{DeclName, QCover, QGauge, Sign};
use crate::quon::eval::{Amplitudes, show};
use crate::span::{Span, Spanned};
use crate::tir::{BaseDef, ChainDef, ChainStage, Exported, LocaleDef, Named, Region, SigId, Ty};

use super::interface::GeometryDef;
use super::items::{Progress, UnitCx};
use super::scope::UnitRef;

/// A declaration of the geometry, and how far resolving it has got.
pub struct Decl<'a> {
    kind: &'a ItemKind,
    name: Spanned<Symbol>,
    linkage: crate::ast::Linkage,
    state: Progress,
    /// Its position in its list, once resolved; `None` when it could not be.
    index: Option<usize>,
    /// For one a block declares, the layers of block items it sees.
    layers: Option<Vec<super::local_items::Layer>>,
}

/// What kind of declaration a name is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Kind {
    Cover,
    Gauge,
    Base,
    Locale,
    Chain,
}

impl Kind {
    fn of(item: &ItemKind) -> Option<Kind> {
        Some(match item {
            ItemKind::Cover { .. } => Kind::Cover,
            ItemKind::Gauge { .. } => Kind::Gauge,
            ItemKind::Base { .. } => Kind::Base,
            ItemKind::Locale { .. } => Kind::Locale,
            ItemKind::Chain { .. } => Kind::Chain,
            _ => return None,
        })
    }

    fn noun(self) -> &'static str {
        match self {
            Kind::Cover => "cover",
            Kind::Gauge => "gauge",
            Kind::Base => "base type",
            Kind::Locale => "locale",
            Kind::Chain => "chain",
        }
    }
}

impl<'a> UnitCx<'a> {
    /// Declares a cover, gauge, base type, locale or chain at unit scope.
    pub(super) fn declare_geometry(&mut self, kind: &'a ItemKind, name: Spanned<Symbol>, linkage: crate::ast::Linkage, span: Span) {
        if self.quantum {
            let earlier = self.geometry_decls.get(&name.node).map(|d| d.name.span).or_else(|| {
                self.scope
                    .get_type(name.node)
                    .and_then(|_| self.scope.span_of(name.node))
            });
            if let Some(earlier) = earlier {
                let text = self.interner.resolve(name.node);
                self.report(super::report::duplicate(name.span, earlier, text, "at unit scope"));
                return;
            }
        }
        self.declare_geometry_as(kind, name, name.node, linkage, span);
    }

    /// Declares a cover, gauge, base type, locale or chain written `name`
    /// under the name `key`: its own at unit scope, and one no other
    /// declaration shares for one a block declares.
    pub(super) fn declare_geometry_as(
        &mut self,
        kind: &'a ItemKind,
        name: Spanned<Symbol>,
        key: Symbol,
        linkage: crate::ast::Linkage,
        span: Span,
    ) {
        let noun = Kind::of(kind).expect("a declaration of the geometry").noun();
        if !self.quantum {
            self.report(
                Diagnostic::new(Code::Eu02)
                    .with_message(format!("only a quantum unit can declare a {noun}"))
                    .at(span)
                    .with_note("covers, gauges, base types, locales and chains describe the states of qubits")
                    .with_help("move this into a unit that begins with `#unit quantum`"),
            );
            return;
        }
        self.check_reserved(name.node, name.span);
        self.geometry_decls.insert(
            key,
            Decl {
                kind,
                name,
                linkage,
                state: Progress::Pending,
                index: None,
                layers: None,
            },
        );
    }

    /// Gives the declaration `key`, which a block makes, the layers of block
    /// items it sees.
    pub(super) fn scope_geometry(&mut self, key: Symbol, layers: Vec<super::local_items::Layer>) {
        if let Some(d) = self.geometry_decls.get_mut(&key) {
            d.layers = Some(layers);
        } else {
            self.scope_qmap(key, layers);
        }
    }

    /// The linkage of the declaration of the geometry `name`, if it is one.
    pub(super) fn geometry_linkage(&self, name: Symbol) -> Option<crate::ast::Linkage> {
        self.geometry_decls.get(&self.geometry_key(name)).map(|d| d.linkage)
    }

    /// Resolves every declaration of the geometry, in the order written.
    pub(super) fn check_geometry(&mut self) {
        let mut names: Vec<(Span, Symbol)> = self.geometry_decls.iter().map(|(&k, d)| (d.name.span, k)).collect();
        names.sort_by_key(|(s, _)| s.start);
        for (_, name) in names {
            self.resolve_geometry(name);
        }
        // what other units may name: covers, gauges and base types at unit
        // scope, not `static`, in the order written
        let mut exports: Vec<(Span, Symbol, Exported)> = Vec::new();
        for (&key, d) in &self.geometry_decls {
            if d.layers.is_some() || key != d.name.node || d.linkage != crate::ast::Linkage::Program {
                continue;
            }
            let Some(i) = d.index else { continue };
            let e = match Kind::of(d.kind) {
                Some(Kind::Cover) => Exported::Cover(i),
                Some(Kind::Gauge) => Exported::Gauge(i),
                Some(Kind::Base) => Exported::Base(i),
                _ => continue,
            };
            exports.push((d.name.span, key, e));
        }
        exports.sort_by_key(|(s, _, _)| s.start);
        self.unit.geometry.exports = exports.into_iter().map(|(_, k, e)| (k, e)).collect();
    }

    /// What kind of declaration of the geometry `name` is, if it is one.
    pub(super) fn geometry_kind(&self, name: Symbol) -> Option<Kind> {
        Kind::of(self.geometry_decls.get(&self.geometry_key(name))?.kind)
    }

    /// What a name of the geometry is, as a noun with its article, if it is
    /// one: for a message where a type of values was wanted.
    pub fn geometry_noun(&self, name: Symbol) -> Option<String> {
        Some(format!("a {}", self.geometry_kind(name)?.noun()))
    }

    /// The position of the declaration `name` in its list, resolving it
    /// first if it has not been; `None` once why it has none is reported.
    pub(super) fn resolve_geometry(&mut self, name: Symbol) -> Option<usize> {
        let name = self.geometry_key(name);
        let d = self.geometry_decls.get_mut(&name)?;
        match d.state {
            Progress::Done => return d.index,
            Progress::Active => {
                let (span, text) = (d.name.span, self.interner.resolve(d.name.node));
                self.report(
                    Diagnostic::new(Code::Es16)
                        .with_message(format!("`{text}` is written in terms of itself"))
                        .at(span)
                        .with_note("each cover, gauge and base type names only ones that are complete without it"),
                );
                return None;
            }
            Progress::Pending => d.state = Progress::Active,
        }
        let (kind, decl_name) = (d.kind, d.name);
        let index = match d.layers.clone() {
            Some(layers) => self.with_layers(layers, |cx| cx.geometry_of(kind, decl_name)),
            None => self.geometry_of(kind, decl_name),
        };
        let d = self.geometry_decls.get_mut(&name).expect("declared");
        d.state = Progress::Done;
        d.index = index;
        index
    }

    /// Makes the declaration `kind`, written `decl_name`, and adds it to its
    /// list, giving its position there.
    fn geometry_of(&mut self, kind: &'a ItemKind, decl_name: Spanned<Symbol>) -> Option<usize> {
        fn add<T>(list: &mut Vec<Named<T>>, name: Spanned<Symbol>, def: T) -> usize {
            list.push(Named {
                name: name.node,
                def,
                span: name.span,
            });
            list.len() - 1
        }
        match kind {
            ItemKind::Cover { cover, .. } => {
                let c = self.eval_cover(cover)?;
                Some(add(&mut self.unit.geometry.covers, decl_name, c))
            }
            ItemKind::Gauge { gauge, .. } => {
                let g = self.eval_gauge(gauge)?;
                Some(add(&mut self.unit.geometry.gauges, decl_name, g))
            }
            ItemKind::Base { register, cover, gauge, .. } => {
                let b = self.base(register, *cover, *gauge)?;
                Some(add(&mut self.unit.geometry.bases, decl_name, b))
            }
            ItemKind::Locale { ty, members, .. } => {
                let l = self.locale(ty, members)?;
                Some(add(&mut self.unit.geometry.locales, decl_name, l))
            }
            ItemKind::Chain { stages, .. } => {
                let c = self.chain(decl_name, stages)?;
                Some(add(&mut self.unit.geometry.chains, decl_name, c))
            }
            _ => None,
        }
    }

    /// The declaration `name` of kind `want`, resolved; `None` once why not
    /// is reported.
    pub(super) fn named_geometry(&mut self, name: Spanned<Symbol>, want: Kind) -> Option<usize> {
        let text = self.interner.resolve(name.node);
        let key = self.geometry_key(name.node);
        let Some(found) = self.geometry_decls.get(&key).and_then(|d| Kind::of(d.kind)) else {
            let candidates: Vec<&str> = self
                .geometry_decls
                .values()
                .filter(|d| Kind::of(d.kind) == Some(want))
                .map(|d| self.interner.resolve(d.name.node))
                .collect();
            self.report(super::report::not_found(name.span, want.noun(), text, candidates));
            return None;
        };
        if found != want {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`{text}` is a {}, where a {} is wanted", found.noun(), want.noun()))
                    .at(name.span),
            );
            return None;
        }
        self.resolve_geometry(key)
    }

    /// The declaration `name` of kind `want`, this unit's or another's,
    /// resolved; `None` once why not is reported.
    pub(super) fn named_decl(&mut self, name: Spanned<DeclName>, want: Kind) -> Option<usize> {
        match name.node.unit {
            None => self.named_geometry(Spanned::new(name.node.name, name.span), want),
            Some(unit) => self.imported_decl(unit, name.node.name, name.span, want),
        }
    }

    /// What kind of declaration `name` is, if it is one this unit can name.
    pub(super) fn decl_kind(&self, name: DeclName) -> Option<Kind> {
        match name.unit {
            None => self.geometry_kind(name.name),
            Some(unit) => match self.offering_unit(unit)? {
                Offered::Own => self.geometry_kind(name.name),
                Offered::Other(iface) => iface.geometry_named(name.name).map(def_kind),
            },
        }
    }

    /// The linkage of the declaration `name`, if it is one: another unit's
    /// can be named only because it has program linkage.
    pub(super) fn decl_linkage(&self, name: DeclName) -> Option<crate::ast::Linkage> {
        match name.unit {
            None => self.geometry_linkage(name.name),
            Some(_) => self.decl_kind(name).map(|_| crate::ast::Linkage::Program),
        }
    }

    /// `name` as written: `T01`, or `gates::T01`.
    pub(super) fn decl_text(&self, name: DeclName) -> String {
        match name.unit {
            None => self.interner.resolve(name.name).to_owned(),
            Some(u) => format!("{}::{}", self.interner.resolve(u), self.interner.resolve(name.name)),
        }
    }

    /// The unit called `unit` here: this one, or an import.
    fn offering_unit(&self, unit: Symbol) -> Option<Offered<'a>> {
        let u = self.scope.get_unit(unit)?;
        Some(if u == UnitRef::SELF { Offered::Own } else { Offered::Other(self.imported[u.0 as usize]) })
    }

    /// The cover, gauge or base type `name` of the unit called `unit`, made
    /// this unit's the first time it is named: its states are carried to
    /// the conductor the two units' judgements meet at.
    fn imported_decl(&mut self, unit: Symbol, name: Symbol, span: Span, want: Kind) -> Option<usize> {
        let iface = match self.offering_unit(unit) {
            Some(Offered::Own) => return self.named_geometry(Spanned::new(name, span), want),
            Some(Offered::Other(iface)) => iface,
            None => {
                let d = super::imports::unknown_unit(self, unit, span);
                self.report(d);
                return None;
            }
        };
        let text = self.decl_text(DeclName { unit: Some(unit), name });
        let Some(def) = iface.geometry_named(name) else {
            let candidates: Vec<&str> = iface
                .geometry
                .iter()
                .filter(|g| def_kind(&g.def) == want)
                .map(|g| self.interner.resolve(g.name))
                .collect();
            let mut d = Diagnostic::new(Code::Es04)
                .with_message(format!(
                    "unit `{}` has no {} named `{}` that other units can use",
                    self.interner.resolve(unit),
                    want.noun(),
                    self.interner.resolve(name)
                ))
                .at(span)
                .with_note("only a cover, gauge or base type declared without `static` can be named from another unit");
            if let Some(s) = super::report::closest(self.interner.resolve(name), candidates) {
                d = d.with_help(format!("did you mean `{s}`?"));
            }
            self.report(d);
            return None;
        };
        let found = def_kind(def);
        if found != want {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`{text}` is a {}, where a {} is wanted", found.noun(), want.noun()))
                    .at(span),
            );
            return None;
        }
        let key = (iface.unit.clone(), name);
        if let Some(&i) = self.imported_geometry.get(&key) {
            return Some(i);
        }
        let n = num_integer::lcm(self.conductor, iface.conductor);
        let label = Spanned::new(self.interner.intern_late(&text), span);
        let push_cover = |cx: &mut Self, c: &Cover| {
            cx.unit.geometry.covers.push(Named { name: label.node, def: c.at(n), span });
            cx.unit.geometry.covers.len() - 1
        };
        let push_gauge = |cx: &mut Self, g: &Gauge| {
            cx.unit.geometry.gauges.push(Named { name: label.node, def: g.at(n), span });
            cx.unit.geometry.gauges.len() - 1
        };
        let index = match def {
            GeometryDef::Cover(c) => push_cover(self, c),
            GeometryDef::Gauge(g) => push_gauge(self, g),
            GeometryDef::Base(b) => {
                let width = b.cover.width();
                let (cover, gauge) = (push_cover(self, &b.cover), push_gauge(self, &b.gauge));
                self.unit.geometry.bases.push(Named { name: label.node, def: BaseDef { width, cover, gauge }, span });
                self.unit.geometry.bases.len() - 1
            }
        };
        self.imported_geometry.insert(key, index);
        Some(index)
    }

    ////////////
    // COVERS //
    ////////////

    pub(super) fn eval_cover(&mut self, c: &Spanned<QCover>) -> Option<Cover> {
        match &c.node {
            QCover::Name(name) => {
                let i = self.named_decl(Spanned::new(*name, c.span), Kind::Cover)?;
                Some(self.unit.geometry.covers[i].def.clone())
            }
            QCover::Pt(s) => Some(Cover::Pt(self.point(s)?)),
            QCover::Fin(states) => {
                self.literals(states.len(), c.span)?;
                // every point is checked, and each problem reported
                let points: Vec<Option<Amplitudes>> = states.iter().map(|s| self.point(s)).collect();
                let points: Vec<Amplitudes> = points.into_iter().collect::<Option<_>>()?;
                self.one_width(&points, states.iter().map(|s| s.span))?;
                for j in 0..points.len() {
                    if let Some(i) = (0..j).find(|&i| crate::judge::linalg::same_point(&points[i], &points[j])) {
                        self.report(
                            Diagnostic::new(Code::Ej15)
                                .with_message("this cover names one point twice")
                                .at_with(states[j].span, "this point")
                                .also(states[i].span, "is this one, up to a phase")
                                .with_note("a `fin` cover is a set of distinct points, and states differing by a phase are one point"),
                        );
                        return None;
                    }
                }
                Some(Cover::Fin(points))
            }
            QCover::Span(kets) => {
                self.literals(kets.len(), c.span)?;
                let points: Vec<Amplitudes> = kets
                    .iter()
                    .map(|k| Amplitudes::basis(k.node.bits.chars().map(|b| b == '1').collect(), self.conductor))
                    .collect();
                self.one_width(&points, kets.iter().map(|k| k.span))?;
                for j in 0..points.len() {
                    if let Some(i) = (0..j).find(|&i| points[i] == points[j]) {
                        self.report(
                            Diagnostic::new(Code::Ej15)
                                .with_message("this span names one ket twice")
                                .at_with(kets[j].span, "this ket")
                                .also(kets[i].span, "is named here already")
                                .with_note("a `span` is presented by distinct basis kets"),
                        );
                        return None;
                    }
                }
                Some(Cover::Span(points))
            }
            QCover::Code(strings) => {
                let gens: Vec<Pauli> = strings
                    .iter()
                    .map(|p| {
                        Pauli::parse(p.node.sign == Some(Sign::Minus), &p.node.letters).expect("the parser admits only I, X, Y and Z")
                    })
                    .collect();
                let width = gens.first().map_or(0, Pauli::width);
                if let Some(j) = gens.iter().position(|g| g.width() != width) {
                    self.report(
                        Diagnostic::new(Code::Ej15)
                            .with_message(format!(
                                "this code's generators act on different numbers of qubits: {width} and {}",
                                gens[j].width()
                            ))
                            .at(strings[j].span)
                            .with_note("a code is fixed by Pauli strings on one register, one letter for each qubit"),
                    );
                    return None;
                }
                for j in 0..gens.len() {
                    if let Some(i) = (0..j).find(|&i| !gens[i].commutes(&gens[j])) {
                        self.report(
                            Diagnostic::new(Code::Ej15)
                                .with_message("these two generators do not commute, so no state is fixed by both")
                                .at_with(strings[j].span, "this one")
                                .also(strings[i].span, "anticommutes with this one")
                                .with_note("a stabilizer code is the common +1 eigenspace of commuting Pauli strings"),
                        );
                        return None;
                    }
                }
                if !pauli::independent(&gens) {
                    self.report(
                        Diagnostic::new(Code::Ej15)
                            .with_message("this code's generators are not independent")
                            .at(c.span)
                            .with_note(
                                "a generator that is a product of the others, up to sign, either adds nothing \
                                 or makes −I a member, when no state is fixed at all",
                            )
                            .with_help("leave out the generators the others make"),
                    );
                    return None;
                }
                Some(Cover::Code {
                    gens,
                    width,
                    n: self.conductor,
                })
            }
        }
    }

    /// A point of a cover, or a fiducial: a state, which must be a unit
    /// vector.
    fn point(&mut self, s: &Spanned<crate::quon::ast::QState>) -> Option<Amplitudes> {
        let a = self.written_state(s)?;
        if !a.is_normalized() {
            self.report(super::quon::not_normalized(&a, s.span));
            return None;
        }
        Some(a)
    }

    /// `EM05` when a cover has too many literals.
    fn literals(&mut self, count: usize, span: Span) -> Option<()> {
        if let Some(d) = Limit::CoverLiterals.check(u32::try_from(count).unwrap_or(u32::MAX), span) {
            self.report(d);
            return None;
        }
        Some(())
    }

    /// `EJ15` unless every state is as wide as the first.
    fn one_width(&mut self, points: &[Amplitudes], spans: impl Iterator<Item = Span>) -> Option<()> {
        let spans: Vec<Span> = spans.collect();
        let w = points.first().map_or(0, |p| p.width);
        if let Some(j) = points.iter().position(|p| p.width != w) {
            self.report(
                Diagnostic::new(Code::Ej15)
                    .with_message(format!(
                        "this cover's points are of different widths: {w} qubits and {} qubits",
                        points[j].width
                    ))
                    .at_with(spans[j], format!("{} qubits", points[j].width))
                    .also(spans[0], format!("{w} qubits"))
                    .with_note("a cover is a set of states of one register"),
            );
            return None;
        }
        Some(())
    }

    ////////////
    // GAUGES //
    ////////////

    pub(super) fn eval_gauge(&mut self, g: &Spanned<QGauge>) -> Option<Gauge> {
        match &g.node {
            QGauge::None => Some(Gauge { fiducials: Vec::new() }),
            QGauge::Fid(s) => Some(Gauge {
                fiducials: vec![self.point(s)?],
            }),
            QGauge::Name(name) => {
                let i = self.named_decl(Spanned::new(*name, g.span), Kind::Gauge)?;
                Some(self.unit.geometry.gauges[i].def.clone())
            }
            QGauge::Atlas(parts) => {
                let mut fiducials = Vec::new();
                for p in parts {
                    let part = self.eval_gauge(p)?;
                    if part.fiducials.is_empty() {
                        self.report(
                            Diagnostic::new(Code::Ej15)
                                .with_message("an atlas is made of fiducials, and this part has none")
                                .at(p.span)
                                .with_note("`none` declares a cover stationary-only; it is a gauge alone, not a patch of one"),
                        );
                        return None;
                    }
                    fiducials.extend(part.fiducials);
                }
                Some(Gauge { fiducials })
            }
        }
    }

    ////////////////
    // BASE TYPES //
    ////////////////

    fn base(&mut self, register: &Spanned<Type>, cover: Spanned<DeclName>, gauge: Spanned<DeclName>) -> Option<BaseDef> {
        let width = self.register_type_width(register)?;
        let c = self.named_decl(cover, Kind::Cover);
        let g = self.named_decl(gauge, Kind::Gauge);
        let (c, g) = (c?, g?);
        let the_cover = self.unit.geometry.covers[c].def.clone();
        let the_gauge = self.unit.geometry.gauges[g].def.clone();
        if the_cover.width() != width {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!(
                        "this register has {width} qubits, but the cover's states have {}",
                        the_cover.width()
                    ))
                    .at_with(register.span, format!("{width} qubits"))
                    .also(cover.span, format!("{} qubits", the_cover.width()))
                    .with_note("a base type is a cover of the states of its register"),
            );
            return None;
        }
        if let Err(why) = cover::check(&the_cover, &the_gauge) {
            let cover_name = format!("`{}`", self.decl_text(cover.node));
            let d = gauge_refusal(why, &the_gauge, &cover_name, cover.span, gauge.span, width);
            self.report(d);
            return None;
        }
        Some(BaseDef { width, cover: c, gauge: g })
    }

    /// The number of qubits of a register type: `qubit` or `[qubit; N]`.
    pub(super) fn register_type_width(&mut self, t: &Spanned<Type>) -> Option<usize> {
        match self.written_type(t)? {
            Ty::Qubit => Some(1),
            other => match self.unit.types.as_array(other) {
                Some((Ty::Qubit, n)) => usize::try_from(n).ok(),
                _ => {
                    let what = super::report::describe(other, &self.unit.types, self.interner);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("a base type is over a register, `qubit` or `[qubit; N]`, and this is {what}"))
                            .at(t.span),
                    );
                    None
                }
            },
        }
    }

    //////////////////////////////
    // RESTRICTIONS AND LOCALES //
    //////////////////////////////

    /// The region a restriction type, or a base type alone, names.
    fn region(&mut self, t: &Spanned<Type>) -> Option<Region> {
        match &t.node {
            Type::Restriction { restricted, container } => {
                let a = self.region(restricted);
                let b = self.region(container);
                let (a, b) = (a?, b?);
                let (wa, wb) = (self.region_width(&a), self.region_width(&b));
                if wa != wb {
                    self.report(
                        Diagnostic::new(Code::Ej11)
                            .with_message(format!("a restriction is of states of one register, and these have {wa} and {wb} qubits"))
                            .at_with(restricted.span, format!("{wa} qubits"))
                            .also(container.span, format!("{wb} qubits"))
                            .with_note("`A of B` takes the part of `B`'s states that are also `A`'s"),
                    );
                    return None;
                }
                let Some(cover) = cover::intersect(&a.cover, &b.cover) else {
                    self.report(
                        Diagnostic::new(Code::Ej11)
                            .with_message("these two base types share no state, so the restriction is empty")
                            .at_with(restricted.span, "no state of this")
                            .also(container.span, "is one of this")
                            .with_note("a restriction names the states both covers hold, and a type holds at least one"),
                    );
                    return None;
                };
                let mut bases = a.bases;
                bases.extend(b.bases);
                Some(Region { bases, cover })
            }
            Type::Path { path, args } if args.is_empty() && path.segments.len() == 1 => {
                let name = path.segments[0];
                let b = self.named_geometry(name, Kind::Base)?;
                let cover = self.unit.geometry.covers[self.unit.geometry.bases[b].def.cover].def.clone();
                Some(Region { bases: vec![b], cover })
            }
            _ => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message("a locale is over a base type, or a restriction `A of B` of base types")
                        .at(t.span),
                );
                None
            }
        }
    }

    fn region_width(&self, r: &Region) -> usize {
        r.bases.first().map_or(0, |&b| self.unit.geometry.bases[b].def.width)
    }

    fn locale(&mut self, ty: &Spanned<Type>, members: &'a [ast::LocaleMember]) -> Option<LocaleDef> {
        let region = self.region(ty)?;
        let width = self.region_width(&region);
        let mut out = Vec::with_capacity(members.len());
        let mut ok = true;
        for m in members {
            super::items::check_annotations(
                &m.annotations,
                super::items::Target::Function,
                true,
                self.interner,
                &mut self.diags,
            );
            let Some(sig) = self.member_signature(m) else {
                ok = false;
                continue;
            };
            if let Err(why) = self.interface_on(Ty::Fn(sig), width) {
                let name = self.interner.resolve(m.name.node);
                self.report(
                    Diagnostic::new(Code::Ej12)
                        .with_message(format!("`{name}` is not an interface of the locale's container: {why}"))
                        .at(m.name.span)
                        .also(ty.span, format!("the container is a register of {width} qubits"))
                        .with_note(
                            "a locale declares some of the interfaces its container admits from itself to \
                             itself, never more: each takes the container's register, by value giving it back \
                             or through a handle",
                        ),
                );
                ok = false;
                continue;
            }
            out.push(Named {
                name: m.name.node,
                def: sig,
                span: m.name.span,
            });
        }
        ok.then_some(LocaleDef { region, members: out })
    }

    fn member_signature(&mut self, m: &ast::LocaleMember) -> Option<SigId> {
        let params = m.params.iter().map(|p| self.written_type(&p.ty)).collect::<Option<Vec<_>>>()?;
        let ret = match &m.ret {
            Some(r) => self.written_type(r)?,
            None => Ty::Void,
        };
        match self.unit.types.function(params, ret) {
            Ty::Fn(sig) => Some(sig),
            _ => None,
        }
    }

    /// Whether the function type `f` is an interface from a register of
    /// `width` qubits to itself; why not, otherwise.
    fn interface_on(&self, f: Ty, width: usize) -> Result<(), String> {
        let types = &self.unit.types;
        let Some((params, ret)) = types.as_sig(f) else {
            return Err("it is not an operator".to_owned());
        };
        let register = |t: Ty| match t {
            Ty::Qubit => Some(1usize),
            _ => types.as_array(t).and_then(|(e, n)| (e == Ty::Qubit).then_some(n as usize)),
        };
        let holds_qubits = |t: Ty| types.is_quantum(t) || types.as_ref(t).is_some_and(|(_, r)| types.is_quantum(r));
        let quantum: Vec<Ty> = params.iter().copied().filter(|&t| holds_qubits(t)).collect();
        let [q] = quantum.as_slice() else {
            return Err(format!("it takes {} registers, where an interface takes one", quantum.len()));
        };
        let (by_handle, target) = match types.as_ref(*q) {
            Some((_, t)) => (true, t),
            None => (false, *q),
        };
        match register(target) {
            Some(w) if w == width => {}
            Some(w) => return Err(format!("its register has {w} qubits")),
            None => return Err("what it takes is not a register".to_owned()),
        }
        if by_handle {
            if types.is_quantum(ret) {
                return Err("it takes the register through a handle but gives qubits back".to_owned());
            }
        } else if register(ret) != Some(width) || ret != target {
            return Err("it takes the register by value but does not give it back".to_owned());
        }
        Ok(())
    }

    ////////////
    // CHAINS //
    ////////////

    fn chain(&mut self, name: Spanned<Symbol>, stages: &'a [ast::ChainStage]) -> Option<ChainDef> {
        if let Some(d) = Limit::ChainStages.check(u32::try_from(stages.len()).unwrap_or(u32::MAX), name.span) {
            self.report(d);
            return None;
        }

        // each stage, checked in full so that every problem is reported
        let mut out: Vec<ChainStage> = Vec::with_capacity(stages.len());
        let mut ok = true;
        for (i, s) in stages.iter().enumerate() {
            // its two base types and operator
            let from = self.named_geometry(s.from, Kind::Base);
            let to = self.named_geometry(s.to, Kind::Base);
            let op = super::body::check_const_as(self, &s.operator, None, "a chain's stage");
            let (Some(from), Some(to)) = (from, to) else {
                ok = false;
                continue;
            };
            if op.ty == Ty::Never {
                ok = false;
                continue;
            }

            // one register throughout, which the operator acts on
            let (wf, wt) = (self.unit.geometry.bases[from].def.width, self.unit.geometry.bases[to].def.width);
            if wf != wt {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("this stage goes from a register of {wf} qubits to one of {wt}"))
                        .at_with(s.from.span, format!("{wf} qubits"))
                        .also(s.to.span, format!("{wt} qubits"))
                        .with_note("a stage applies one operator to one register"),
                );
                ok = false;
                continue;
            }
            let why = match op.ty {
                Ty::Fn(_) => self.interface_on(op.ty, wf).err(),
                _ => Some("it is not an operator".to_owned()),
            };
            if let Some(why) = why {
                let what = super::report::describe(op.ty, &self.unit.types, self.interner);
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("a stage applies an operator to its register, and this is {what}: {why}"))
                        .at(s.operator.span)
                        .with_note("an operator of a stage takes the register, by value giving it back or through a handle"),
                );
                ok = false;
                continue;
            }

            // starting where the stage before ends
            if let Some(prev) = out.last()
                && prev.to != from
                && i > 0
            {
                let (a, b) = (self.interner.resolve(stages[i - 1].to.node), self.interner.resolve(s.from.node));
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("this stage starts at `{b}`, but the stage before ends at `{a}`"))
                        .at_with(s.from.span, "starts here")
                        .also(stages[i - 1].to.span, "the stage before ends here")
                        .with_note("in a chain, each stage starts at the base type the one before ends at"),
                );
                ok = false;
            }
            out.push(ChainStage { op, from, to });
        }
        if !ok {
            return None;
        }

        // closed when it ends where it starts
        let closed = matches!((out.first(), out.last()), (Some(f), Some(l)) if f.from == l.to);
        Some(ChainDef {
            stages: out,
            closed,
            expects: Vec::new(),
        })
    }
}

impl super::body::Checker<'_, '_> {
    /// `@nerve(B)`, `@transition_class(B)`, `@single_patch(B)` and
    /// `@ortho_graph(B)`, of the base type `B`: its gauge's nerve, as the
    /// sets of patches that meet; the transition class of its cover; whether
    /// that is trivial, so that one patch would do; and the orthogonal pairs
    /// of its cover's points. `None` when `which` is none of them.
    pub(super) fn geometry_macro(&mut self, which: &str, args: &[Spanned<ast::MacroArg>], span: Span) -> Option<crate::tir::Expr> {
        use crate::tir::{Expr, IntTy, Value};
        if !matches!(which, "nerve" | "transition_class" | "single_patch" | "ortho_graph") {
            return None;
        }

        // the base type, a bare name that may have parsed as a type or an expression
        let name = match args {
            [a] => match &a.node {
                ast::MacroArg::Type(t) => match &t.node {
                    Type::Path { path, args } if args.is_empty() && path.segments.len() == 1 => Some(path.segments[0]),
                    _ => None,
                },
                ast::MacroArg::Expr(e) => match &e.node {
                    ast::Expr::Path { path, args } if args.is_empty() && path.segments.len() == 1 => Some(path.segments[0]),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        };
        let Some(name) = name else {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`@{which}` takes the name of a base type"))
                    .at(span)
                    .with_help(format!("write it as `@{which}(B)`, for `base B = …;`")),
            );
            return Some(Self::error(span));
        };
        let Some(b) = self.cx.named_geometry(name, Kind::Base) else {
            return Some(Self::error(span));
        };

        // what is asked of its cover and gauge
        let base = self.cx.unit.geometry.bases[b].def;
        let the_cover = self.cx.unit.geometry.covers[base.cover].def.clone();
        let the_gauge = self.cx.unit.geometry.gauges[base.gauge].def.clone();
        let usize_value = |i: usize| Value::Int(i as i128, IntTy::USIZE);
        let (value, ty) = match which {
            "nerve" => {
                let simplices = cover::nerve(&the_cover, &the_gauge);
                let inner = self.types_mut().growable(Ty::USIZE);
                let ty = self.types_mut().growable(inner);
                let value = Value::Array(
                    simplices
                        .into_iter()
                        .map(|s| Value::Array(s.into_iter().map(usize_value).collect()))
                        .collect(),
                );
                (value, ty)
            }
            "transition_class" => (
                Value::Int(i128::from(cover::transition_class(&the_cover)), IntTy::I64),
                Ty::Int(IntTy::I64),
            ),
            "single_patch" => (Value::Bool(cover::transition_class(&the_cover) == 0), Ty::Bool),
            _ => {
                let Some(edges) = cover::orthogonality_graph(&the_cover) else {
                    let text = self.interner.resolve(name.node);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!(
                                "`{text}`'s cover is a subspace, whose orthogonality graph has infinitely many points"
                            ))
                            .at(span)
                            .with_note("`@ortho_graph` lists the orthogonal pairs of a finite cover's points"),
                    );
                    return Some(Self::error(span));
                };
                let pair = self.types_mut().tuple(vec![Ty::USIZE, Ty::USIZE]);
                let ty = self.types_mut().growable(pair);
                let value = Value::Array(
                    edges
                        .into_iter()
                        .map(|(i, j)| Value::Struct(vec![usize_value(i), usize_value(j)]))
                        .collect(),
                );
                (value, ty)
            }
        };
        Some(Expr::constant(value, ty, span))
    }
}

/// The unit a qualified name of the geometry names.
enum Offered<'a> {
    /// The unit being analysed.
    Own,
    /// An import.
    Other(&'a super::Interface),
}

/// What kind of declaration another unit's is.
fn def_kind(def: &GeometryDef) -> Kind {
    match def {
        GeometryDef::Cover(_) => Kind::Cover,
        GeometryDef::Gauge(_) => Kind::Gauge,
        GeometryDef::Base(_) => Kind::Base,
    }
}

/// The diagnostic for a gauge that is not one on a cover: `cover_name` is
/// how the message names the cover, and the spans are where the two are
/// written.
pub(super) fn gauge_refusal(why: Refusal, gauge: &Gauge, cover_name: &str, cover_at: Span, gauge_at: Span, width: usize) -> Diagnostic {
    match why {
        Refusal::Width(i) => Diagnostic::new(Code::Es06)
            .with_message(format!(
                "this gauge's fiducial {} has {} qubits, but the cover's states have {width}",
                i + 1,
                gauge.fiducials[i].width
            ))
            .at(gauge_at)
            .with_note("a fiducial is a state of the register the cover is of"),
        Refusal::NoPregauge(dim) => Diagnostic::new(Code::Ej09)
            .with_message(format!(
                "one fiducial cannot gauge {cover_name}, a subspace of projective dimension {dim}"
            ))
            .at_with(gauge_at, "a single fiducial")
            .also(cover_at, "a subspace")
            .with_note(
                "a subspace of dimension two or more holds a point orthogonal to any one vector, \
                 and no single phase convention reaches all of it",
            )
            .with_help("declare the gauge `none`, making the cover stationary-only, or give it an `atlas{…}` of several fiducials"),
        Refusal::Missed(p) => {
            let what = if gauge.fiducials.len() == 1 {
                "the fiducial is orthogonal to"
            } else {
                "every fiducial of the atlas is orthogonal to"
            };
            Diagnostic::new(Code::Ej10)
                .with_message(format!("{what} the point {} of {cover_name}", show(&p)))
                .at_with(gauge_at, "this gauge")
                .also(cover_at, "on this cover")
                .with_note(
                    "a fiducial fixes the phase of each point it overlaps, and a point orthogonal to \
                     every fiducial is given none",
                )
        }
        Refusal::TooManyPatches(count) => Diagnostic::new(Code::Em05)
            .with_message(format!(
                "too many: patches in a gauge on {width} qubits are limited to {}, and this gauge has {count}",
                cover::most_patches(width)
            ))
            .at(gauge_at)
            .with_note("one patch more than the projective dimension of the register's states is enough for any cover"),
        Refusal::MissedCode { reached, dimension } => {
            let what = if gauge.fiducials.len() == 1 {
                "the fiducial is orthogonal to"
            } else {
                "every fiducial of the atlas is orthogonal to"
            };
            let point = if dimension == 1 { "the point" } else { "a point" };
            Diagnostic::new(Code::Ej10)
                .with_message(format!("{what} {point} of {cover_name}"))
                .at_with(gauge_at, "this gauge")
                .also(cover_at, "on this cover")
                .with_note(format!(
                    "what the fiducials reach of the code has dimension {reached}, and the code {dimension}; \
                     a point outside what they reach is orthogonal to every one"
                ))
                .with_note(
                    "a fiducial fixes the phase of each point it overlaps, and a point orthogonal to \
                     every fiducial is given none",
                )
        }
    }
}
