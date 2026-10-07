//! The claims a quantum unit makes of its operators, read from their
//! annotations into [`crate::tir::claims`].
//!
//! - **`[cover: c]` and `[gauge: g]`** give an operator's endpoints: the
//!   states of its register it is judged over, from and to. Each is written in
//!   QUON or names a declaration; a base type's name gives both. A cover
//!   without a gauge takes the fiducial |+…+> when it is finite, and is
//!   stationary-only when it is a subspace; a gauge without a cover is over
//!   the computational basis. The cover is of all the operator's qubits, and
//!   the gauge must be one on it. Written on the parameters instead, the
//!   finite covers of each are joined in order, as their tensor product.
//! - **`[expect: …]`** claims `monic`, `unitary`, a contract class:
//!   `contract = C`, or the class alone, `frame(flat(k))`, or
//!   `kernel.outcomes = T`. A class's phases are constant `phase<N>`s.
//!   `frame` takes only the flat family (`EA05` otherwise).
//! - **`[frame: a ** b]`** declares a pair grouping of the operator's
//!   quantum parameters, each once. Two groupings of the same parameters
//!   that associate differently are `EQ07`.
//! - **`[glue: piecewise]`** asks the operator to be checked piece by piece.
//! - **`[iso: f, …]`** on a quantum enumeration names its witnesses.
//! - **`[expect: holonomy(s) = k]`** on a chain claims its holonomy at a
//!   point of its first base type's cover, given by index or as a constant
//!   `quon::State`.
//!
//! The claims are checked once the circuits exist, in [`crate::lower`].

use crate::ast::{self, AnnArg, BinOp, ItemKind};
use crate::diag::{Code, Diagnostic};
use crate::exact::{Cyclo, Phase};
use crate::intern::Symbol;
use crate::judge::cert::{Base, Class};
use crate::judge::cover::{self, Cover, Gauge};
use crate::quon::ast::{DeclName, QForm};
use crate::quon::eval::{Amplitudes, ket_bits};
use crate::span::{Span, Spanned};
use crate::tir::claims::{Claims, Expect, Grouping, Iso};
use crate::tir::{Arg, FnId, Ty};

use super::geometry::Kind;
use super::items::UnitCx;
use super::scope::Def;

/// The computational-basis cover of `width` qubits.
pub fn basis_cover(width: usize, n: u32) -> Cover {
    Cover::Fin((0..1usize << width).map(|i| Amplitudes::basis(ket_bits(i, width), n)).collect())
}

/// |+…+> on `width` qubits.
pub fn plus(width: usize, n: u32) -> Amplitudes {
    let amp = (0..width).fold(Cyclo::one(n), |a, _| {
        a.mul(&Cyclo::isqrt2(n).expect("the conductor is a multiple of eight"))
    });
    Amplitudes {
        width,
        n,
        terms: (0..1usize << width).map(|i| (ket_bits(i, width), amp.clone())).collect(),
    }
}

/// The qubits a parameter of type `t` holds, directly or through a handle;
/// `None` for one that holds none, or is not a register.
pub fn register_qubits(types: &crate::tir::TypeTable, t: Ty) -> Option<usize> {
    let t = types.as_ref(t).map_or(t, |(_, x)| x);
    match t {
        Ty::Qubit => Some(1),
        _ => types.as_array(t).and_then(|(e, n)| (e == Ty::Qubit).then(|| usize::try_from(n).ok()).flatten()),
    }
}

/// The gauge a cover takes when none is written: |+…+> on a finite cover,
/// none on a subspace.
pub fn default_gauge(c: &Cover) -> Gauge {
    Gauge {
        fiducials: if c.points().is_some() && c.projective_dimension().is_none_or(|d| d == 0) {
            vec![plus(c.width(), c.points().and_then(|p| p.first().map(|a| a.n)).unwrap_or(8))]
        } else {
            Vec::new()
        },
    }
}

impl<'a> UnitCx<'a> {
    /// Reads every claim of the unit's annotations.
    pub(super) fn check_claims(&mut self, unit: &'a ast::Unit) {
        if !self.quantum {
            return;
        }
        for item in &unit.items {
            self.item_claims(&item.node);
        }
        // a block's items, seeing what the block sees
        for (item, layers) in std::mem::take(&mut self.annotated_items) {
            self.with_layers(layers, |cx| cx.item_claims(item));
        }
    }

    /// Reads the claims of one item's annotations.
    fn item_claims(&mut self, item: &'a ast::Item) {
        match &item.kind {
            ItemKind::Fn(f) => {
                let ast::FnName::Plain(name) = &f.name else { return };
                if item.annotations.is_empty() && f.params.iter().all(|p| p.annotations.is_empty()) {
                    return;
                }
                let Some(Def::Fn(id)) = self.local_value(name.node).or_else(|| self.scope.get(name.node)) else { return };
                self.fn_claims(id, &item.annotations, f, name.span);
            }
            ItemKind::Enum { name, .. } => self.iso_claims(name.node, &item.annotations),
            ItemKind::Chain { name, .. } => self.chain_claims(*name, &item.annotations),
            _ => {}
        }
    }

    /// Checks the `@static_assert`s written at unit scope, each as a
    /// constant expression: one whose condition reads a judgement is checked
    /// once the judgements exist.
    pub(super) fn check_asserts(&mut self) {
        for call in std::mem::take(&mut self.asserts) {
            super::body::check_const_as(self, call, Some(Ty::Void), "an assertion");
        }
    }

    /// Each annotation named `which` among `groups`.
    fn annotations_named(
        &self,
        groups: &'a [Spanned<ast::AnnotationGroup>],
        which: &str,
    ) -> Vec<&'a Spanned<ast::Annotation>> {
        groups
            .iter()
            .flat_map(|g| &g.node.annotations)
            .filter(|a| self.interner.resolve(a.node.name.node) == which)
            .collect()
    }

    fn fn_claims(&mut self, id: FnId, groups: &'a [Spanned<ast::AnnotationGroup>], f: &'a ast::FnItem, at: Span) {
        self.fn_sig(id);
        let params: Vec<Ty> = self.unit.func(id).param_types().collect();
        let widths: Vec<Option<usize>> = params.iter().map(|&t| register_qubits(&self.unit.types, t)).collect();
        let width: usize = widths.iter().flatten().sum();
        let mut claims = Claims::default();

        // endpoints
        let covers = self.annotations_named(groups, "cover");
        let gauges = self.annotations_named(groups, "gauge");
        if !covers.is_empty() || !gauges.is_empty() {
            let span = covers.first().or(gauges.first()).map_or(at, |a| a.span);
            if let Some(base) = self.endpoint(covers.first().copied(), gauges.first().copied(), width) {
                claims.endpoints = Some((base, span));
            }
            let named = |s: &Self, a: Option<&&Spanned<ast::Annotation>>| {
                let [arg] = a?.node.args.as_slice() else { return None };
                let name = s.geo_arg(arg)?;
                let linkage = s.decl_linkage(name)?;
                Some(crate::judge::record::Declared {
                    name: s.decl_text(name),
                    program: linkage == ast::Linkage::Program,
                })
            };
            claims.cover_decl = named(self, covers.first());
            claims.gauge_decl = named(self, gauges.first());
            // a base type's name gives the gauge as well as the cover
            if claims.gauge_decl.is_none()
                && let Some([arg]) = covers.first().map(|a| a.node.args.as_slice())
                && let Some(name) = self.geo_arg(arg)
                && self.decl_kind(name) == Some(Kind::Base)
            {
                claims.gauge_decl.clone_from(&claims.cover_decl);
            }
        } else if f.params.iter().any(|p| !p.annotations.is_empty()) {
            claims.endpoints = self.param_endpoints(f, &widths);
        }

        // expectations
        for a in self.annotations_named(groups, "expect") {
            for arg in &a.node.args {
                if let Some(e) = self.expect(arg) {
                    claims.expects.push((e, arg.span));
                }
            }
        }

        // pair groupings
        for a in self.annotations_named(groups, "frame") {
            let [arg] = a.node.args.as_slice() else {
                self.report(form(a.span, "frame", "one grouping of the operator's parameters, as `[frame: a ** b]`"));
                continue;
            };
            let Some(g) = self.grouping(arg, f, &widths) else { continue };
            let mut seen = g.params();
            seen.sort_unstable();
            if seen.windows(2).any(|w| w[0] == w[1]) {
                self.report(form(arg.span, "frame", "a grouping naming each parameter once"));
                continue;
            }
            if let Some((_, at)) = claims.frames.iter().find(|(e, _)| {
                let mut p = e.params();
                p.sort_unstable();
                p == seen && *e != g
            }) {
                self.report(
                    Diagnostic::new(Code::Eq07)
                        .with_message("these two pair groupings associate the same parameters differently")
                        .at_with(arg.span, "this grouping")
                        .also(*at, "reads the parameters across another cut")
                        .with_note(
                            "a grouping is read across its cuts, and a state may split acceptably across one \
                             and not another, as the W state does; one cannot stand for the other",
                        ),
                );
                continue;
            }
            claims.frames.push((g, arg.span));
        }

        // glue
        for a in self.annotations_named(groups, "glue") {
            let piecewise = matches!(a.node.args.as_slice(), [arg] if self.word(arg).as_deref() == Some("piecewise"));
            if piecewise {
                claims.glue = Some(a.span);
            } else {
                self.report(form(a.span, "glue", "`piecewise`, as `[glue: piecewise]`"));
            }
        }
        self.unit.claims.insert(id, claims);
    }

    /// The base type `[cover:]` and `[gauge:]` give, of a register of
    /// `width` qubits.
    fn endpoint(
        &mut self,
        cover: Option<&'a Spanned<ast::Annotation>>,
        gauge: Option<&'a Spanned<ast::Annotation>>,
        width: usize,
    ) -> Option<Base> {
        let n = self.conductor;
        let (mut c, mut g) = (None, None);
        if let Some(a) = cover {
            let [arg] = a.node.args.as_slice() else {
                self.report(form(a.span, "cover", "one cover"));
                return None;
            };
            let (cv, gv) = self.cover_arg(arg)?;
            (c, g) = (Some(cv), gv);
        }
        if let Some(a) = gauge {
            let [arg] = a.node.args.as_slice() else {
                self.report(form(a.span, "gauge", "one gauge"));
                return None;
            };
            g = Some(self.gauge_arg(arg)?);
        }
        let the_cover = c.unwrap_or_else(|| basis_cover(width, n));
        let the_gauge = g.unwrap_or_else(|| default_gauge(&the_cover));
        // an operator taking no qubits is judged from the trivial type, which
        // endpoints do not describe; they are not consulted
        if width == 0 {
            return None;
        }
        let at = cover.or(gauge).map_or(Span::synthetic(), |a| a.span);
        if the_cover.width() != width {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!(
                        "this cover's states have {} qubits, but the operator takes {width}",
                        the_cover.width()
                    ))
                    .at(at)
                    .with_note("an operator's endpoints are states of all the qubits it takes"),
            );
            return None;
        }
        if let Err(why) = cover::check(&the_cover, &the_gauge) {
            let gat = gauge.map_or(at, |a| a.span);
            let d = super::geometry::gauge_refusal(why, &the_gauge, "the operator's cover", at, gat, width);
            self.report(d);
            return None;
        }
        Some(Base {
            cover: the_cover,
            gauge: the_gauge,
        })
    }

    /// A cover written in `[cover:]`: in QUON, or the name of a cover or of a
    /// base type, which gives a gauge too.
    fn cover_arg(&mut self, arg: &Spanned<AnnArg>) -> Option<(Cover, Option<Gauge>)> {
        if let AnnArg::Quon(q) = &arg.node
            && let QForm::Cover(c) = q.as_ref()
            && !matches!(c.node, crate::quon::ast::QCover::Name(_))
        {
            return Some((self.eval_cover(c)?, None));
        }
        let Some(name) = self.geo_arg(arg) else {
            self.report(form(arg.span, "cover", "a cover in QUON, or the name of a cover or base type"));
            return None;
        };
        let at = Spanned::new(name, arg.span);
        match self.decl_kind(name) {
            Some(Kind::Base) => {
                let b = self.named_decl(at, Kind::Base)?;
                let def = self.unit.geometry.bases[b].def;
                Some((
                    self.unit.geometry.covers[def.cover].def.clone(),
                    Some(self.unit.geometry.gauges[def.gauge].def.clone()),
                ))
            }
            _ => {
                let c = self.named_decl(at, Kind::Cover)?;
                Some((self.unit.geometry.covers[c].def.clone(), None))
            }
        }
    }

    /// A gauge written in `[gauge:]`: in QUON, or the name of a gauge.
    fn gauge_arg(&mut self, arg: &Spanned<AnnArg>) -> Option<Gauge> {
        if let AnnArg::Quon(q) = &arg.node
            && let QForm::Gauge(g) = q.as_ref()
        {
            return self.eval_gauge(g);
        }
        let Some(name) = self.geo_arg(arg) else {
            self.report(form(arg.span, "gauge", "a gauge in QUON, or the name of a gauge"));
            return None;
        };
        let g = self.named_decl(Spanned::new(name, arg.span), Kind::Gauge)?;
        Some(self.unit.geometry.gauges[g].def.clone())
    }

    /// The endpoints the parameters' `[cover:]` and `[gauge:]` give: each
    /// parameter's finite cover, the computational basis for one with none
    /// written, joined in order.
    fn param_endpoints(&mut self, f: &'a ast::FnItem, widths: &[Option<usize>]) -> Option<(Base, Span)> {
        // start from the empty register's one point
        let n = self.conductor;
        let mut points: Vec<Amplitudes> = vec![Amplitudes {
            width: 0,
            n,
            terms: std::collections::BTreeMap::from([(Vec::new(), Cyclo::one(n))]),
        }];
        let mut fiducial = points[0].clone();
        let mut at = None;

        // each quantum parameter's cover and fiducial, tensored onto them
        for (p, w) in f.params.iter().zip(widths) {
            let Some(w) = *w else { continue };
            let covers = self.annotations_named(&p.annotations, "cover");
            let gauges = self.annotations_named(&p.annotations, "gauge");
            if at.is_none() {
                at = covers.first().or(gauges.first()).map(|a| a.span);
            }
            let base = if covers.is_empty() && gauges.is_empty() {
                let c = basis_cover(w, n);
                let g = default_gauge(&c);
                Base { cover: c, gauge: g }
            } else {
                self.endpoint(covers.first().copied(), gauges.first().copied(), w)?
            };
            let (Some(ps), [omega]) = (base.cover.points(), base.gauge.fiducials.as_slice()) else {
                let span = covers.first().or(gauges.first()).map_or(p.name.span, |a| a.span);
                self.report(
                    Diagnostic::new(Code::Ea05)
                        .with_message("the parameters' covers are joined only when each is finite with one fiducial")
                        .at(span)
                        .with_help("write the operator's cover and gauge on the operator, of all its qubits"),
                );
                return None;
            };
            points = points.iter().flat_map(|a| ps.iter().map(move |b| tensor(a, b))).collect();
            fiducial = tensor(&fiducial, omega);
        }

        // one point is a `pt` cover
        let cover = if points.len() == 1 { Cover::Pt(points.remove(0)) } else { Cover::Fin(points) };
        Some((
            Base {
                cover,
                gauge: Gauge {
                    fiducials: vec![fiducial],
                },
            },
            at.unwrap_or(Span::synthetic()),
        ))
    }

    /// One claim of `[expect: …]`.
    fn expect(&mut self, arg: &Spanned<AnnArg>) -> Option<Expect> {
        let bad = |s: &mut Self| {
            s.report(form(
                arg.span,
                "expect",
                "`monic`, `unitary`, a contract class, `contract = C`, `frame(flat(k))` or `kernel.outcomes = T`",
            ));
            None
        };
        match &arg.node {
            AnnArg::Named { name, value } if self.interner.resolve(name.node) == "contract" => {
                let e = match &value.node {
                    AnnArg::Expr(e) => e.as_ref().clone(),
                    _ => match self.word(value) {
                        Some(w) => Spanned::new(single_path(self.interner.intern_late(&w), value.span), value.span),
                        None => return bad(self),
                    },
                };
                self.class(&e).map(Expect::Contract)
            }
            AnnArg::Expr(e) => match &e.node {
                ast::Expr::Assign { op: None, place, value } => {
                    let ast::Expr::Field { receiver, name } = &place.node else { return bad(self) };
                    let is_kernel = matches!(&receiver.node, ast::Expr::Path { path, .. }
                        if path.segments.len() == 1 && self.interner.resolve(path.segments[0].node) == "kernel");
                    if !is_kernel || self.interner.resolve(name.node) != "outcomes" {
                        return bad(self);
                    }
                    let ast::Expr::Path { path, .. } = &value.node else { return bad(self) };
                    let t = Spanned::new(
                        ast::Type::Path {
                            path: path.clone(),
                            args: Vec::new(),
                        },
                        value.span,
                    );
                    self.written_type(&t).map(Expect::Outcomes)
                }
                ast::Expr::Call { callee, args } if self.word_of(callee).as_deref() == Some("frame") => {
                    let [inner] = args.as_slice() else { return bad(self) };
                    match self.class(inner)? {
                        Class::Rigid => Some(Expect::Frame(Some(Phase::zero(self.conductor)))),
                        Class::Flat(k) => Some(Expect::Frame(k)),
                        _ => {
                            self.report(
                                Diagnostic::new(Code::Ea05)
                                    .with_message("only the flat family frames")
                                    .at(inner.span)
                                    .with_note(
                                        "a branch-relative class other than flat would have to carry a schedule \
                                         through every Schmidt presentation of every member, which no rule licenses",
                                    )
                                    .with_help("write `frame(flat(k))` or `frame(rigid)`"),
                            );
                            None
                        }
                    }
                }
                ast::Expr::Path { .. } | ast::Expr::Call { .. } => {
                    match self.word_of(e).as_deref() {
                        Some("monic") => return Some(Expect::Monic),
                        Some("unitary") => return Some(Expect::Unitary),
                        _ => {}
                    }
                    self.class(e).map(Expect::Contract)
                }
                _ => bad(self),
            },
            AnnArg::Type(_) => match self.word(arg).as_deref() {
                Some("monic") => Some(Expect::Monic),
                Some("unitary") => Some(Expect::Unitary),
                Some(w) => {
                    let e = Spanned::new(single_path(self.interner.intern_late(w), arg.span), arg.span);
                    self.class(&e).map(Expect::Contract)
                }
                None => bad(self),
            },
            _ => bad(self),
        }
    }

    /// A contract class as written: its name, with its phases in
    /// parentheses where it takes them.
    fn class(&mut self, e: &Spanned<ast::Expr>) -> Option<Class> {
        let (name, args): (String, &[Spanned<ast::Expr>]) = match &e.node {
            ast::Expr::Call { callee, args } => (self.word_of(callee)?, args.as_slice()),
            _ => (self.word_of(e)?, &[]),
        };
        let phases = args.iter().map(|a| self.phase_value(a)).collect::<Option<Vec<_>>>()?;
        let one = |p: &[Phase]| p.first().copied();
        let class = match (name.as_str(), phases.len()) {
            ("rigid", 0) => Class::Rigid,
            ("flat", 0 | 1) => Class::Flat(one(&phases)),
            ("locflat", 0) => Class::Locflat,
            ("cyc", 0) => Class::Cyc(None),
            ("cyc", _) => Class::Cyc(Some(phases)),
            ("stat", 0 | 1) => Class::Stat(one(&phases)),
            ("sector", 0) => Class::Sector(None),
            ("sector", _) => Class::Sector(Some(phases)),
            ("free", 0) => Class::Free,
            _ => {
                self.report(
                    Diagnostic::new(Code::Ea05)
                        .with_message(format!("`{name}` with {} phases is not a contract class", phases.len()))
                        .at(e.span)
                        .with_note(
                            "the classes are `rigid`, `flat` or `flat(k)`, `locflat`, `cyc` or `cyc(θ…)`, \
                             `stat` or `stat(k)`, `sector` or `sector(θ…)`, and `free`",
                        ),
                );
                return None;
            }
        };
        Some(class)
    }

    /// A constant `phase<N>`.
    pub(super) fn phase_value(&mut self, e: &Spanned<ast::Expr>) -> Option<Phase> {
        if let Some(at) = estimate_in(e, self.interner) {
            self.report(
                Diagnostic::new(Code::Ej13)
                    .with_message("an estimated phase cannot satisfy a claim")
                    .at(at)
                    .with_note(
                        "an estimate is a measurement, drawn by running a circuit, not a derivation; a \
                         claim is checked against what is derived, and an estimate is never written back \
                         into a judgment",
                    )
                    .with_help("act on the estimate where the program runs, and claim here only what is derived"),
            );
            return None;
        }
        let (v, ty) = self.typed_const(e, "a phase")?;
        // a whole number k is the angle 2πk/N at the unit's conductor, as
        // in `flat(0)`
        if let Some(k) = v.as_int().filter(|_| ty.is_integer()) {
            return Some(Phase::of(self.conductor, (k % i128::from(self.conductor)) as i64));
        }
        let order = match ty {
            Ty::Adt(id) => {
                let def = self.unit.types.adt(id);
                let is_phase = def.origin.as_deref() == Some("core") && self.interner.resolve(def.name) == "phase";
                match (is_phase, def.args.first()) {
                    (true, Some(Arg::Const(n))) => u32::try_from(*n).ok(),
                    _ => None,
                }
            }
            _ => None,
        };
        let Some(order) = order.filter(|&n| n > 0) else {
            let what = super::report::describe(ty, &self.unit.types, self.interner);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("a contract's phase is a `phase<N>`, and this is {what}"))
                    .at(e.span)
                    .with_help("write it as `phase::<8>::of(k)`"),
            );
            return None;
        };
        let m = super::exact::phase_of(&v).unwrap_or(0);
        match Phase::of(order, i64::try_from(m).unwrap_or(0)).embed(self.conductor) {
            Some(p) => Some(p),
            None => {
                let d = crate::quon::eval::outside(e.span, u64::from(order), self.conductor);
                self.report(d);
                None
            }
        }
    }

    /// The grouping `[frame: …]` writes, of the operator's quantum
    /// parameters.
    fn grouping(&mut self, arg: &Spanned<AnnArg>, f: &ast::FnItem, widths: &[Option<usize>]) -> Option<Grouping> {
        match &arg.node {
            AnnArg::Expr(e) => return self.grouping_expr(e, f, widths),
            AnnArg::Quon(q) => match q.as_ref() {
                QForm::State(s) => return self.grouping_state(s, f, widths),
                QForm::Cover(c) => {
                    if let crate::quon::ast::QCover::Name(n) = &c.node {
                        return self.grouping_name(n.unit.is_none().then_some(n.name), c.span, f, widths);
                    }
                }
                QForm::Gauge(_) => {}
            },
            _ => {}
        }
        self.report(form(arg.span, "frame", "a grouping of the operator's parameters, as `a ** b`"));
        None
    }

    /// A grouping written as a tensor product of names, parenthesised as
    /// it associates.
    fn grouping_state(
        &mut self,
        s: &Spanned<crate::quon::ast::QState>,
        f: &ast::FnItem,
        widths: &[Option<usize>],
    ) -> Option<Grouping> {
        let term = &s.node.head.node;
        if !s.node.tail.is_empty() || term.coeff.is_some() {
            self.report(form(s.span, "frame", "a grouping of the operator's parameters, joined by `**`"));
            return None;
        }
        let mut parts = Vec::with_capacity(term.factors.len());
        for fac in &term.factors {
            parts.push(match &fac.node {
                crate::quon::ast::QFactor::Name(n) => self.grouping_name(Some(*n), fac.span, f, widths)?,
                crate::quon::ast::QFactor::Paren(inner) => self.grouping_state(inner, f, widths)?,
                crate::quon::ast::QFactor::Ket(_) => {
                    self.report(form(fac.span, "frame", "the names of the operator's quantum parameters"));
                    return None;
                }
            });
        }
        parts.into_iter().reduce(|a, b| Grouping::Pair(Box::new(a), Box::new(b)))
    }

    /// The quantum parameter `n` names.
    fn grouping_name(&mut self, n: Option<Symbol>, at: Span, f: &ast::FnItem, widths: &[Option<usize>]) -> Option<Grouping> {
        let found = n.and_then(|n| f.params.iter().position(|p| self.interner.resolve(p.name.node) == self.interner.resolve(n)));
        match found {
            Some(i) if widths.get(i).is_some_and(Option::is_some) => Some(Grouping::Param(i)),
            _ => {
                self.report(form(at, "frame", "the names of the operator's quantum parameters, joined by `**`"));
                None
            }
        }
    }

    fn grouping_expr(&mut self, e: &Spanned<ast::Expr>, f: &ast::FnItem, widths: &[Option<usize>]) -> Option<Grouping> {
        match &e.node {
            ast::Expr::Paren(inner) => self.grouping_expr(inner, f, widths),
            ast::Expr::Binary { op: BinOp::Tensor, lhs, rhs } => Some(Grouping::Pair(
                Box::new(self.grouping_expr(lhs, f, widths)?),
                Box::new(self.grouping_expr(rhs, f, widths)?),
            )),
            _ => self.grouping_name(single_name(e), e.span, f, widths),
        }
    }

    /// `[iso: f, …]` on a quantum enumeration.
    fn iso_claims(&mut self, name: Symbol, groups: &'a [Spanned<ast::AnnotationGroup>]) {
        let isos = self.annotations_named(groups, "iso");
        let Some(a) = isos.first() else { return };
        let Some(adt) = self.lookup_type(name) else { return };
        let mut witnesses = Vec::with_capacity(a.node.args.len());
        for arg in &a.node.args {
            let found = self.name_arg(arg).and_then(|s| match self.lookup_value(s) {
                Some(Def::Fn(id)) => Some(id),
                _ => None,
            });
            match found {
                Some(id) => witnesses.push((id, arg.span)),
                None => {
                    self.report(form(arg.span, "iso", "the names of the operators witnessing the variants' sameness"));
                    return;
                }
            }
        }
        self.unit.isos.push(Iso {
            adt,
            witnesses,
            span: a.span,
        });
    }

    /// `[expect: holonomy(s) = k]` on a chain.
    fn chain_claims(&mut self, name: Spanned<Symbol>, groups: &'a [Spanned<ast::AnnotationGroup>]) {
        let expects = self.annotations_named(groups, "expect");
        if expects.is_empty() {
            return;
        }
        let Some(c) = self.named_geometry(name, Kind::Chain) else { return };
        for a in expects {
            for arg in &a.node.args {
                let parsed = match &arg.node {
                    AnnArg::Expr(e) => match &e.node {
                        ast::Expr::Assign { op: None, place, value } => match &place.node {
                            ast::Expr::Call { callee, args } if self.word_of(callee).as_deref() == Some("holonomy") => {
                                match args.as_slice() {
                                    [p] => Some((p, value)),
                                    _ => None,
                                }
                            }
                            _ => None,
                        },
                        _ => None,
                    },
                    _ => None,
                };
                let Some((point, value)) = parsed else {
                    self.report(form(arg.span, "expect", "`holonomy(s) = k` on a chain"));
                    continue;
                };
                let Some(i) = self.chain_point(c, point) else { continue };
                let Some(k) = self.phase_value(value) else { continue };
                self.unit.geometry.chains[c].def.expects.push((i, k, arg.span));
            }
        }
    }

    /// The point of the chain `c`'s first base type's cover `e` names: by
    /// index, or as a constant `quon::State`.
    pub(super) fn chain_point(&mut self, c: usize, e: &Spanned<ast::Expr>) -> Option<usize> {
        let first = self.unit.geometry.chains[c].def.stages.first()?.from;
        let cover = self.unit.geometry.covers[self.unit.geometry.bases[first].def.cover].def.clone();
        let points = cover.points().unwrap_or_default();
        if let Some(v) = self.evaluate_const(e, Some(Ty::USIZE), "a point's index").ok().and_then(|v| v.as_int()) {
            return match usize::try_from(v).ok().filter(|&i| i < points.len()) {
                Some(i) => Some(i),
                None => {
                    self.report(
                        Diagnostic::new(Code::Ec07)
                            .with_message(format!("the chain's first cover has {} points, and this is point {v}", points.len()))
                            .at(e.span),
                    );
                    None
                }
            };
        }
        let state = self.constant_state(e)?;
        self.point_index(&points, &state, e.span)
    }

    /// The index among `points` of the point `state` is.
    pub(super) fn point_index(&mut self, points: &[Amplitudes], state: &Amplitudes, at: Span) -> Option<usize> {
        match points.iter().position(|p| crate::judge::linalg::same_point(p, state)) {
            Some(i) => Some(i),
            None => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!(
                            "{} is not a point of the chain's first cover",
                            crate::quon::eval::show(state)
                        ))
                        .at(at)
                        .with_note("a holonomy is taken at a point of a chain's first base type"),
                );
                None
            }
        }
    }

    /// A single name written as an annotation argument.
    fn name_arg(&self, arg: &Spanned<AnnArg>) -> Option<Symbol> {
        match &arg.node {
            AnnArg::Expr(e) => single_name(e),
            AnnArg::Type(t) => match &t.node {
                ast::Type::Path { path, args } if args.is_empty() && path.segments.len() == 1 => Some(path.segments[0].node),
                _ => None,
            },
            AnnArg::Quon(_) => self.geo_arg(arg).and_then(|d| d.unit.is_none().then_some(d.name)),
            AnnArg::Named { .. } => None,
        }
    }

    /// The name of a cover, gauge or base type written as an annotation
    /// argument: this unit's, or another's as `gates::T01`.
    fn geo_arg(&self, arg: &Spanned<AnnArg>) -> Option<DeclName> {
        let path = |segments: &[Spanned<Symbol>]| match segments {
            [n] => Some(DeclName::local(n.node)),
            [u, n] => Some(DeclName { unit: Some(u.node), name: n.node }),
            _ => None,
        };
        match &arg.node {
            AnnArg::Expr(e) => match &e.node {
                ast::Expr::Path { path: p, args } if args.is_empty() => path(&p.segments),
                _ => None,
            },
            AnnArg::Type(t) => match &t.node {
                ast::Type::Path { path: p, args } if args.is_empty() => path(&p.segments),
                _ => None,
            },
            AnnArg::Quon(q) => match q.as_ref() {
                QForm::Cover(c) => match &c.node {
                    crate::quon::ast::QCover::Name(d) => Some(*d),
                    _ => None,
                },
                QForm::Gauge(g) => match &g.node {
                    crate::quon::ast::QGauge::Name(d) => Some(*d),
                    _ => None,
                },
                QForm::State(_) => None,
            },
            AnnArg::Named { .. } => None,
        }
    }

    fn word(&self, arg: &Spanned<AnnArg>) -> Option<String> {
        self.name_arg(arg).map(|s| self.interner.resolve(s).to_owned())
    }

    fn word_of(&self, e: &Spanned<ast::Expr>) -> Option<String> {
        single_name(e).map(|s| self.interner.resolve(s).to_owned())
    }
}

/// The one name `e` is.
fn single_name(e: &Spanned<ast::Expr>) -> Option<Symbol> {
    match &e.node {
        ast::Expr::Path { path, args } if args.is_empty() && path.segments.len() == 1 => Some(path.segments[0].node),
        _ => None,
    }
}

/// `a ⊗ b`, `a`'s qubits first.
fn tensor(a: &Amplitudes, b: &Amplitudes) -> Amplitudes {
    let mut out = Amplitudes {
        width: a.width + b.width,
        n: a.n,
        terms: Default::default(),
    };
    for (x, p) in &a.terms {
        for (y, q) in &b.terms {
            let mut k = x.clone();
            k.extend_from_slice(y);
            out.terms.insert(k, p.mul(q));
        }
    }
    out
}

fn single_path(name: Symbol, span: Span) -> ast::Expr {
    ast::Expr::Path {
        path: ast::Path::single(Spanned::new(name, span)),
        args: Vec::new(),
    }
}

/// Where `e` asks a device for an estimate, `estimate_stat` or
/// `estimate_sector`, if it does.
fn estimate_in(e: &Spanned<ast::Expr>, interner: &crate::intern::Interner) -> Option<Span> {
    let is_estimate = |s: Symbol| matches!(interner.resolve(s), "estimate_stat" | "estimate_sector");
    let all = |xs: &[Spanned<ast::Expr>]| xs.iter().find_map(|x| estimate_in(x, interner));
    match &e.node {
        ast::Expr::MethodCall { receiver, name, args, .. } => {
            if is_estimate(name.node) {
                return Some(e.span);
            }
            estimate_in(receiver, interner).or_else(|| all(args))
        }
        ast::Expr::Call { callee, args } => {
            if let ast::Expr::Path { path, .. } = &callee.node
                && path.segments.last().is_some_and(|s| is_estimate(s.node))
            {
                return Some(e.span);
            }
            all(args)
        }
        ast::Expr::Paren(x) | ast::Expr::Try(x) => estimate_in(x, interner),
        ast::Expr::Unary { operand, .. } => estimate_in(operand, interner),
        ast::Expr::Binary { lhs, rhs, .. } => estimate_in(lhs, interner).or_else(|| estimate_in(rhs, interner)),
        ast::Expr::Field { receiver, .. } => estimate_in(receiver, interner),
        ast::Expr::Index { receiver, index } => estimate_in(receiver, interner).or_else(|| estimate_in(index, interner)),
        ast::Expr::Tuple(xs) | ast::Expr::Array(xs) => all(xs),
        _ => None,
    }
}

/// `EA05`: an annotation's argument not of the form it takes.
fn form(span: Span, name: &str, wants: &str) -> Diagnostic {
    Diagnostic::new(Code::Ea05)
        .with_message(format!("`[{name}: …]` takes {wants}"))
        .at(span)
}
