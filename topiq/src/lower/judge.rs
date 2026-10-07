//! Judgement derivation: what each operator does to quantum state, and
//! whether that is what its annotations claim.
//!
//! Each operator with a circuit gets a [`Record`]; an operator another unit
//! declares has the record that unit publishes, read from its interface and
//! never derived again here. A record's certificate is derived from the
//! circuit's exact action on its endpoints
//! ([`crate::judge::cert`]): the base type its `[cover:]` and `[gauge:]`
//! give, from and to itself; or, when none is written, the computational
//! basis with the fiducial |+…+>, carried to itself when the operator keeps
//! the basis and otherwise to the points it reaches, gauged by where it takes
//! the fiducial. An operator that takes no qubits is judged from the trivial
//! type to the state it prepares, rigidly. An operator taking two or more
//! registers that acts on each basis state of the first separately is also
//! judged sector by sector, each sector over the rest. A Clifford circuit
//! over a code cover is judged on the stabiliser path instead
//! ([`crate::judge::stabilizer`]), in time polynomial in the code's width,
//! which gives the same certificate.
//!
//! # Rules
//!
//! - **An instrument has no certificate:** phase lineage ends at its kernel,
//!   and its contract is `free`.
//! - **A field outside the checked fragment is `Unknown`,** with a warning
//!   naming the step (`EJ16`): an angle known only as the circuit runs, a
//!   supplied operator, or a phase that is no whole number of N-th parts of a
//!   turn. An `Unknown` field never satisfies a claim (`EJ01`).
//! - **Each claim is checked against the record:** a contract claim fails
//!   with the schedule that refutes it (`EJ03`); a frame claim with its
//!   product witness, and its entangled witness when the points are
//!   orthogonal (`EJ05`); a glue with the constants of its pieces
//!   (`EJ06`).
//! - **A chain's stages are judged between their base types,** and its
//!   holonomy claims checked, a failure naming the stage the discrepancy
//!   indicts (`EJ07`).
//! - **An iso-sum's witnesses compose by the cocycle law:** the witness
//!   from j to i after the one from k to j is the one from k to i, as
//!   certificates over the payload.
//! - **Under `#pragma fragment(stabilizer)`** every gate is a Clifford gate
//!   (`EJ17`).

use std::collections::HashMap;

use crate::circuit::Circuit;
use crate::circuit::ir::Op;
use crate::diag::{Code, Diagnostic};
use crate::exact::{Cyclo, Phase};
use crate::intern::Interner;
use crate::judge::cert::{self, Base, Certificate, Class, Fail, constant, show_phase, show_schedule};
use crate::judge::cover::{Cover, Gauge};
use crate::judge::linalg::same_point;
use crate::judge::record::{Fragment, Judged, Kernel, Record, Tri, clifford, kernel_of};
use crate::judge::sim;
use crate::quon::eval::{Amplitudes, ket, ket_bits, show};
use crate::sema::claims::{basis_cover, plus, register_qubits};
use crate::span::Span;
use crate::tir::claims::{Claims, Derived, Expect};
use crate::tir::{Callee, ExprKind, FnId, IntTy, Unit, Value};

use super::generate::Gen;

/// The judgements of a unit, and what checking its claims reported.
pub struct Judgments {
    /// Each operator's record, by operator.
    pub records: HashMap<FnId, Record>,
    /// The holonomies of each closed chain, by chain, one for each point of
    /// its first base type's cover.
    pub holonomies: HashMap<usize, Vec<Option<Phase>>>,
    /// What was reported.
    pub diags: Vec<Diagnostic>,
}

/// The most qubits the computational-basis cover is written out for, the
/// most literals a cover may have being 256.
const BASIS_WIDTH: usize = 8;

struct Judging<'a, 'g> {
    unit: &'a Unit,
    interner: &'a Interner,
    n: u32,
    g: &'g mut Gen<'a>,
    circuits: HashMap<FnId, Option<Circuit>>,
    records: HashMap<FnId, Record>,
    diags: Vec<Diagnostic>,
}

/// Derives the record of every operator with a circuit and checks the
/// unit's claims against them. `imported` holds the records the units it
/// imports publish of their operators it uses, which are read, never
/// derived again.
pub fn judge<'a>(
    unit: &'a Unit,
    interner: &'a Interner,
    g: &mut Gen<'a>,
    circuits: &[super::Lowered],
    stabilizer: bool,
    imported: HashMap<FnId, Record>,
) -> Judgments {
    let mut j = Judging::new(unit, interner, g);
    j.circuits = circuits.iter().map(|l| (l.func, Some(l.circuit.clone()))).collect();
    j.records = imported;

    // each operator's record, derived from its circuit
    for l in circuits {
        if stabilizer && let Err(step) = clifford(&l.circuit.body) {
            let name = interner.resolve(unit.func(l.func).name);
            j.diags.push(
                Diagnostic::new(Code::Ej17)
                    .with_message(format!("`{name}` applies {step}, which is not a Clifford gate"))
                    .at(unit.func(l.func).span)
                    .with_note(
                        "the unit declares `#pragma fragment(stabilizer)`, in which every judgment is \
                         derived by the polynomial-time stabilizer path, and this step leaves it",
                    )
                    .with_help("remove the pragma, or keep the operator to Clifford gates"),
            );
        }
        let endpoints = unit.claims.get(&l.func).and_then(|c| c.endpoints.clone());
        let r = j.derive(l.func, &l.circuit, endpoints);
        if let Some(step) = &r.fragment.departure {
            let name = interner.resolve(unit.func(l.func).name);
            j.diags.push(
                Diagnostic::new(Code::Ej16)
                    .with_message(format!("`{name}`'s judgment left the checked fragment, and its contract is `Unknown`"))
                    .at(unit.func(l.func).span)
                    .with_note(format!("the step: {step}"))
                    .as_warning(),
            );
        }
        j.records.insert(l.func, r);
    }

    // the claims, in declaration order, then the chains and isomorphisms
    let mut ids: Vec<FnId> = unit.claims.keys().copied().collect();
    ids.sort_by_key(|f| f.0);
    for f in ids {
        let claims = &unit.claims[&f];
        j.check(f, claims);
    }
    let holonomies = j.chains();
    j.isos();
    Judgments {
        records: j.records,
        holonomies,
        diags: j.diags,
    }
}

/// The holonomies of the unit's closed chains, as [`judge`] derives them;
/// what deriving them reports is left to [`judge`].
pub fn holonomies<'a>(unit: &'a Unit, interner: &'a Interner, g: &mut Gen<'a>) -> HashMap<usize, Vec<Option<Phase>>> {
    Judging::new(unit, interner, g).chains()
}

impl<'a, 'g> Judging<'a, 'g> {
    fn new(unit: &'a Unit, interner: &'a Interner, g: &'g mut Gen<'a>) -> Self {
        Judging {
            unit,
            interner,
            n: g.conductor(),
            g,
            circuits: HashMap::new(),
            records: HashMap::new(),
            diags: Vec::new(),
        }
    }

    fn name(&self, f: FnId) -> &'a str {
        self.interner.resolve(self.unit.func(f).name)
    }

    /// The circuit of `f`.
    fn circuit(&mut self, f: FnId) -> Option<Circuit> {
        if !self.circuits.contains_key(&f) {
            let c = self.g.operator(f);
            self.circuits.insert(f, c);
        }
        self.circuits[&f].clone()
    }

    /// The record of `f` from its own endpoints.
    fn record(&mut self, f: FnId) -> Option<Record> {
        if let Some(r) = self.records.get(&f) {
            return Some(r.clone());
        }
        let c = self.circuit(f)?;
        let endpoints = self.unit.claims.get(&f).and_then(|c| c.endpoints.clone());
        let r = self.derive(f, &c, endpoints);
        self.records.insert(f, r.clone());
        Some(r)
    }

    /// The record of `f`.
    fn derive(&mut self, f: FnId, c: &Circuit, endpoints: Option<(Base, Span)>) -> Record {
        let kernel = kernel_of(&c.body);
        let instrument = kernel.measured > 0 || kernel.forgotten > 0;
        let monic = if instrument || lifts(&c.body) {
            Tri::No
        } else if calls(&c.body).iter().any(|&s| !c.slots.get(s as usize).is_some_and(|slot| slot.monic)) {
            Tri::Unknown
        } else {
            Tri::Yes
        };
        let unitary = match monic {
            Tri::Yes if c.inputs.len() == c.outputs.len() => Tri::Yes,
            Tri::Unknown => Tri::Unknown,
            _ => Tri::No,
        };
        let mut record = Record {
            monic,
            unitary,
            contract: None,
            cert: None,
            sectors: None,
            kernel: instrument.then_some(Kernel {
                measured: kernel.measured,
                forgotten: kernel.forgotten,
            }),
            fragment: Fragment {
                conductor: self.n,
                departure: None,
                stabilizer: clifford(&c.body).is_ok(),
            },
            dynamic: c.dynamic,
            undecided: None,
        };
        if instrument {
            record.contract = Some(Class::Free);
            return record;
        }
        match sim::fixed(&c.body) {
            Ok(()) => {}
            Err(sim::Why::Supplied) => {
                record.undecided = Some("a call of an operator its caller supplies".to_owned());
                return record;
            }
            Err(_) => {
                record.fragment.departure = Some("an angle, a choice or a loop known only as the circuit runs".to_owned());
                return record;
            }
        }
        let n = self.n;
        let mut u = |v: &Amplitudes| sim::action(c, v).expect("the circuit's action is fixed");
        let width = c.inputs.len();
        let (source, target) = match &endpoints {
            Some((b, _)) => (b.clone(), b.clone()),
            None if width == 0 => {
                // from the trivial type to the state prepared, rigidly: the
                // target's fiducial is that state
                let empty = Amplitudes::basis(Vec::new(), n);
                let out = u(&empty);
                let source = Base {
                    cover: Cover::Pt(empty.clone()),
                    gauge: Gauge { fiducials: vec![empty] },
                };
                let target = Base {
                    cover: Cover::Pt(out.clone()),
                    gauge: Gauge { fiducials: vec![out] },
                };
                (source, target)
            }
            None if width > BASIS_WIDTH => {
                record.undecided = Some(format!(
                    "no cover is written for its {width} qubits, and the computational basis of more than {BASIS_WIDTH} is too many points"
                ));
                return record;
            }
            None => self.default_endpoints(&mut u, width, c.outputs.len()),
        };
        match certify(c, &source, &target, n).map_err(failure) {
            Ok(cert) => {
                record.contract = Some(cert::strongest(&cert, &source));
                record.cert = Some(Judged { cert, source, target });
            }
            Err(Failure::Leaves(how)) => {
                let at = endpoints.as_ref().map_or(self.unit.func(f).span, |(_, s)| *s);
                let name = self.name(f);
                self.diags.push(
                    Diagnostic::new(Code::Ej03)
                        .with_message(format!("`{name}` does not keep to its cover: {how}"))
                        .at(at)
                        .with_note("an operator judged over a cover carries every point of it to a point of it"),
                );
                return record;
            }
            Err(Failure::Outside(what)) => {
                record.fragment.departure = Some(format!("the phase it gives {what} is no whole number of {n}-th parts of a turn"));
                return record;
            }
            Err(Failure::Undecided) => {
                record.undecided = Some(format!(
                    "its cover is a code of more than {} qubits, which is judged without being written out only for a Clifford circuit",
                    crate::judge::cover::CODE_WIDTH
                ));
                return record;
            }
        }
        record.sectors = self.sectors(f, c);
        if let Some(phases) = record.sectors.as_ref().and_then(|s| s.iter().map(stationary).collect()) {
            record.contract = Some(Class::Sector(Some(phases)));
        }
        record
    }

    /// The default endpoints of an operator on `width` qubits giving back
    /// `out`: the computational basis with |+…+>, to itself when the
    /// operator keeps it, and otherwise to the points it reaches, gauged by
    /// where it takes the fiducial.
    fn default_endpoints(&self, u: &mut dyn FnMut(&Amplitudes) -> Amplitudes, width: usize, out: usize) -> (Base, Base) {
        let n = self.n;
        let source = basis_base(width, n);
        let points = source.cover.points().expect("a basis is finite");
        let images: Vec<Amplitudes> = points.iter().map(&mut *u).collect();
        let keeps = out == width && images.iter().all(|v| points.iter().any(|p| same_point(p, v)));
        if keeps {
            return (source.clone(), source);
        }
        let target = Base {
            cover: Cover::Fin(images),
            gauge: Gauge {
                fiducials: vec![u(&plus(width, n))],
            },
        };
        (source, target)
    }

    /// The judgement of `f`, sector by sector over the basis states of its
    /// first register, when it has two or more and acts on each separately.
    fn sectors(&mut self, f: FnId, c: &Circuit) -> Option<Vec<Judged>> {
        let widths: Vec<usize> = self.unit.func(f).param_types().filter_map(|t| register_qubits(&self.unit.types, t)).collect();
        if widths.len() < 2 || c.inputs.len() != c.outputs.len() {
            return None;
        }
        let (k, rest) = (widths[0], c.inputs.len() - widths[0]);
        if rest == 0 || rest > BASIS_WIDTH || k > BASIS_WIDTH {
            return None;
        }

        // one judgement for each basis state `key` of the first register
        let n = self.n;
        let rest_points = basis_cover(rest, n).points().expect("a basis is finite");
        let mut out = Vec::with_capacity(1 << k);
        for j in 0..1usize << k {
            // the rest of a state whose first register is `key`, and back
            let key = ket_bits(j, k);
            let project = |v: &Amplitudes| -> Option<Amplitudes> {
                let mut a = Amplitudes {
                    width: rest,
                    n,
                    terms: Default::default(),
                };
                for (bits, amp) in &v.terms {
                    if bits[..k] != key[..] {
                        return None;
                    }
                    a.terms.insert(bits[k..].to_vec(), amp.clone());
                }
                Some(a)
            };
            let lift = |p: &Amplitudes| -> Amplitudes {
                let mut a = Amplitudes {
                    width: k + rest,
                    n,
                    terms: Default::default(),
                };
                for (bits, amp) in &p.terms {
                    let mut b = key.clone();
                    b.extend_from_slice(bits);
                    a.terms.insert(b, amp.clone());
                }
                a
            };

            // the circuit must keep `key` fixed, so the sector is an operator of its own
            for p in &rest_points {
                project(&sim::action(c, &lift(p)).ok()?)?;
            }
            let mut u = |p: &Amplitudes| project(&sim::action(c, &lift(p)).expect("fixed")).expect("block diagonal");
            let (source, target) = self.default_endpoints(&mut u, rest, rest);
            let cert = cert::derive(&mut u, &source, &target, n).ok()?;
            out.push(Judged { cert, source, target });
        }
        Some(out)
    }

    ////////////
    // CLAIMS //
    ////////////

    fn check(&mut self, f: FnId, claims: &Claims) {
        if claims.expects.is_empty() && claims.glue.is_none() {
            return;
        }
        let name = self.name(f);
        let Some(r) = self.record(f) else {
            if let Some((_, at)) = claims.expects.first() {
                self.diags.push(
                    Diagnostic::new(Code::Ej01)
                        .with_message(format!("`{name}` has no judgment to check this against"))
                        .at(*at)
                        .with_note(
                            "an operator taking classical values is judged where it is called, with the \
                             values it is called with; alone it has no circuit",
                        ),
                );
            }
            return;
        };
        for (e, at) in &claims.expects {
            self.expect(f, &r, e, *at);
        }
        if let Some(at) = claims.glue {
            self.glue(f, &r, at);
        }
    }

    fn unknown(&mut self, f: FnId, r: &Record, what: &str, at: Span) {
        let name = self.name(f);
        let note = match (&r.fragment.departure, &r.undecided) {
            (Some(step), _) => format!("its derivation left the checked fragment: {step}"),
            (None, Some(why)) => format!("it is not decided: {why}"),
            (None, None) => "it was not derived".to_owned(),
        };
        self.diags.push(
            Diagnostic::new(Code::Ej01)
                .with_message(format!("`{name}`'s {what} is `Unknown`, which satisfies no claim"))
                .at(at)
                .with_note(note)
                .with_help("write the operator's cover, keep it inside the fragment, or remove the claim"),
        );
    }

    fn expect(&mut self, f: FnId, r: &Record, e: &Expect, at: Span) {
        let name = self.name(f);
        match e {
            Expect::Monic | Expect::Unitary => {
                let (what, v) = if matches!(e, Expect::Monic) { ("monic", r.monic) } else { ("unitary", r.unitary) };
                match v {
                    Tri::Yes => {}
                    Tri::Unknown => self.unknown(f, r, &format!("`{what}`"), at),
                    Tri::No => {
                        let why = if r.kernel.is_some() {
                            "it measures or forgets".to_owned()
                        } else if r.dynamic {
                            "it lifts a measured value".to_owned()
                        } else {
                            "it gives back a different number of qubits than it takes".to_owned()
                        };
                        self.diags.push(
                            Diagnostic::new(Code::Ej03)
                                .with_message(format!("`{name}` is not {what}: {why}"))
                                .at(at)
                                .with_note(if what == "monic" {
                                    "a monic operator forgets nothing: it neither measures, forgets nor lifts"
                                } else {
                                    "a unitary operator is monic, and gives back as many qubits as it takes"
                                }),
                        );
                    }
                }
            }
            Expect::Outcomes(t) => {
                let ret = self.unit.func(f).ret;
                if r.kernel.is_none() {
                    self.diags.push(
                        Diagnostic::new(Code::Ej03)
                            .with_message(format!("`{name}` has no kernel: it neither measures nor forgets"))
                            .at(at)
                            .with_note("a kernel's outcomes are what an instrument's measurement gives"),
                    );
                } else if ret != *t {
                    let (have, want) = (self.unit.types.display(ret, self.interner), self.unit.types.display(*t, self.interner));
                    self.diags.push(
                        Diagnostic::new(Code::Ej03)
                            .with_message(format!("`{name}`'s outcomes are `{have}`, not `{want}`"))
                            .at(at)
                            .with_note("an instrument's outcome type is the classical value it returns"),
                    );
                }
            }
            // a claim's phases are the unit's; the derivation's may be in a
            // larger group, when the unit meets an import of another
            // conductor
            Expect::Contract(class) => self.contract(f, r, &class.at(self.n), at),
            Expect::Frame(k) => self.frame(f, r, k.map(|p| p.embed(self.n).unwrap_or(p)), at),
        }
    }

    fn contract(&mut self, f: FnId, r: &Record, class: &Class, at: Span) {
        let name = self.name(f);

        // a sector class: each sector stationary, at the phases claimed
        if let Class::Sector(want) = class {
            let Some(sectors) = &r.sectors else {
                let why = if r.kernel.is_some() {
                    "it is an instrument".to_owned()
                } else if r.cert.is_none() {
                    return self.unknown(f, r, "contract", at);
                } else {
                    "it does not act on each basis state of its first register separately".to_owned()
                };
                self.diags.push(
                    Diagnostic::new(Code::Ej03)
                        .with_message(format!("`{name}` is not `{class}`: {why}"))
                        .at(at)
                        .with_note("a sector judgment is of an operator acting on each sector of a sum on its own"),
                );
                return;
            };
            let phases: Vec<Option<Phase>> = sectors.iter().map(stationary).collect();
            if let Some(i) = phases.iter().position(Option::is_none) {
                self.diags.push(
                    Diagnostic::new(Code::Ej03)
                        .with_message(format!("`{name}` is not `{class}`: its sector {i} is not stationary"))
                        .at(at)
                        .with_note(format!(
                            "sector {i}'s certificate is {}; a sector judgment needs each sector's operator a scalar",
                            describe(&sectors[i].cert)
                        )),
                );
                return;
            }
            let got: Vec<Phase> = phases.into_iter().flatten().collect();
            if let Some(w) = want
                && *w != got
            {
                self.diags.push(
                    Diagnostic::new(Code::Ej03)
                        .with_message(format!(
                            "`{name}` is not `{class}`: its sector phases are {}",
                            show_schedule(&got)
                        ))
                        .at(at),
                );
            }
            return;
        }

        // any other class: the certificate satisfies it, or why not
        let Some(j) = &r.cert else {
            if r.kernel.is_some() && *class != Class::Free {
                self.diags.push(
                    Diagnostic::new(Code::Ej03)
                        .with_message(format!("`{name}` is not `{class}`: it is an instrument, and has no certificate"))
                        .at(at)
                        .with_note("phase lineage ends at a measurement, so no contract is claimed across one"),
                );
            } else if r.kernel.is_none() {
                self.unknown(f, r, "contract", at);
            }
            return;
        };
        if cert::satisfies(&j.cert, &j.source, class) {
            return;
        }
        let strongest = cert::strongest(&j.cert, &j.source);
        let mut d = Diagnostic::new(Code::Ej03)
            .with_message(format!("`{name}` is not `{class}`: its certificate is {}", describe(&j.cert)))
            .at(at)
            .with_note(format!("the strongest class it satisfies is `{strongest}`"));
        if let Some(w) = witness(&j.cert, &j.source) {
            d = d.with_note(w);
        }
        if class.needs_gauge() && j.source.gauge.fiducials.is_empty() {
            d = d.with_note("its cover is stationary-only, and a class that consults a convention needs a gauge");
        }
        self.diags.push(d);
    }

    fn frame(&mut self, f: FnId, r: &Record, want: Option<Phase>, at: Span) {
        let name = self.name(f);
        let Some(j) = &r.cert else {
            if r.kernel.is_some() {
                self.diags.push(
                    Diagnostic::new(Code::Ej05)
                        .with_message(format!("`{name}` is an instrument, and frames at no contract"))
                        .at(at),
                );
            } else {
                self.unknown(f, r, "frame", at);
            }
            return;
        };
        let Err(why) = cert::frame(&j.cert, &j.source, want) else { return };
        let Some(why) = why else {
            self.diags.push(
                Diagnostic::new(Code::Ej05)
                    .with_message(format!("frame expectation failed: `{name}`'s certificate is {}", describe(&j.cert)))
                    .at(at),
            );
            return;
        };

        // the schedule it has, against the constant wanted
        let points = j.source.cover.points().unwrap_or_default();
        let target_points = j.target.cover.points().unwrap_or_default();
        let images = match &j.cert {
            Certificate::Points { images, .. } => images.clone(),
            _ => Vec::new(),
        };
        let message = if why.points.len() == 2 {
            format!("frame expectation failed: schedule is {}, not constant", show_schedule(&why.schedule))
        } else {
            format!(
                "frame expectation failed: schedule is constant at {}, not {}",
                show_phase(why.schedule[0]),
                want.map_or("that".to_owned(), show_phase)
            )
        };
        let mut d = Diagnostic::new(Code::Ej05).with_message(message).at(at);

        // the shift at each point, and for two orthogonal points, a superposition that shows it
        let product: Vec<String> = why
            .points
            .iter()
            .map(|&i| format!("{} ** β is shifted by {}", show(&points[i]), show_phase(why.schedule[i])))
            .collect();
        d = d.with_note(format!(
            "witness: for any state β of the rest, {}{}",
            product.join(", and "),
            if why.points.len() == 2 { "; no one constant matches both" } else { "" }
        ));
        if why.orthogonal {
            let (s, t) = (why.points[0], why.points[1]);
            let branch = |p: &Amplitudes, bit: bool, phase: Option<Phase>| {
                let prefix = match phase {
                    Some(ph) if !ph.is_zero() => format!("w({}, {}) * ", ph.index(), ph.order()),
                    _ => String::new(),
                };
                match p.terms.iter().next() {
                    Some((bits, c)) if p.terms.len() == 1 && c.is_one() => {
                        let mut b = bits.clone();
                        b.push(bit);
                        format!("{prefix}{}", ket(&b))
                    }
                    _ => format!("{prefix}{} ** |{}>", show(p), u8::from(bit)),
                }
            };
            let input = format!("({} + {}) * isq2", branch(&points[s], false, None), branch(&points[t], true, None));
            let (fs, ft) = (&target_points[images[s]], &target_points[images[t]]);
            let output = format!(
                "({} + {}) * isq2",
                branch(fs, false, Some(why.schedule[s])),
                branch(ft, true, Some(why.schedule[t]))
            );
            d = d.with_note(format!(
                "witness: (`{name}` ** id) applied to {input} gives {output}, whose branch-relative shift is {}",
                show_phase(why.schedule[t].sub(why.schedule[s]))
            ));
        }
        d = d.with_note(
            "an operator tensored with the identity is branch-relatively flat exactly when its schedule is \
             one constant on the whole of its cover, whether or not the cover has orthogonal points",
        );
        self.diags.push(d);
    }

    fn glue(&mut self, f: FnId, r: &Record, at: Span) {
        let name = self.name(f);
        // the pieces: the sectors, or the gauge's patches
        let constants: Option<Vec<Option<Phase>>> = if let Some(sectors) = &r.sectors {
            Some(
                sectors
                    .iter()
                    .map(|s| match &s.cert {
                        Certificate::Points { schedule, .. } => constant(schedule),
                        Certificate::Scalar(v) => Some(*v),
                        Certificate::Moving => None,
                    })
                    .collect(),
            )
        } else if let Some(j) = &r.cert
            && let Certificate::Points { schedule, .. } = &j.cert
        {
            let points = j.source.cover.points().unwrap_or_default();
            Some(
                j.source
                    .gauge
                    .fiducials
                    .iter()
                    .map(|w| {
                        let piece: Vec<Phase> = points
                            .iter()
                            .zip(schedule)
                            .filter(|(p, _)| !crate::judge::linalg::inner(w, p).is_zero())
                            .map(|(_, &s)| s)
                            .collect();
                        constant(&piece)
                    })
                    .collect(),
            )
        } else {
            None
        };
        let Some(constants) = constants else {
            if r.kernel.is_none() {
                self.unknown(f, r, "certificate", at);
            }
            return;
        };
        if let Some(i) = constants.iter().position(Option::is_none) {
            self.diags.push(
                Diagnostic::new(Code::Ej06)
                    .with_message(format!("glue failed: `{name}` is not flat on its piece {i}"))
                    .at(at)
                    .with_note("gluing assembles pieces that are each flat, with a constant each"),
            );
            return;
        }
        let cs: Vec<Phase> = constants.into_iter().flatten().collect();
        if cs.windows(2).any(|w| w[0] != w[1]) {
            self.diags.push(
                Diagnostic::new(Code::Ej06)
                    .with_message(format!(
                        "glue failed: `{name}`'s pieces are flat with the constants {}, which differ",
                        show_schedule(&cs)
                    ))
                    .at(at)
                    .with_note(format!(
                        "the operator is `cyc` with that profile, and may publish the sector schedule {}",
                        show_schedule(&cs)
                    )),
            );
        }
    }

    ////////////
    // CHAINS //
    ////////////

    fn chains(&mut self) -> HashMap<usize, Vec<Option<Phase>>> {
        let mut out = HashMap::new();
        let geometry = &self.unit.geometry;
        for (ci, chain) in geometry.chains.iter().enumerate() {
            let mut certs = Vec::with_capacity(chain.def.stages.len());
            let mut ok = true;
            for (k, stage) in chain.def.stages.iter().enumerate() {
                let ExprKind::FnRef(Callee::Fn(op)) = stage.op.kind else {
                    ok = false;
                    continue;
                };
                let Some(c) = self.circuit(op) else {
                    ok = false;
                    continue;
                };
                if sim::fixed(&c.body).is_err() {
                    self.diags.push(
                        Diagnostic::new(Code::Ej01)
                            .with_message(format!("stage {} of the chain applies `{}`, which has no fixed action", k + 1, self.name(op)))
                            .at(stage.op.span),
                    );
                    ok = false;
                    continue;
                }
                let (from, to) = (base_type(self.unit, stage.from), base_type(self.unit, stage.to));
                match certify(&c, &from, &to, self.n) {
                    Ok(cert) => certs.push(cert),
                    Err(why) => {
                        let what = match failure(why) {
                            Failure::Leaves(how) => format!("does not keep to its target: {how}"),
                            Failure::Outside(what) => format!("gives {what} a phase that is no whole number of {}-th parts of a turn", self.n),
                            Failure::Undecided => "is over a code too wide to write out, and is not a Clifford circuit".to_owned(),
                        };
                        let fname = self.name(op);
                        self.diags.push(
                            Diagnostic::new(Code::Ej03)
                                .with_message(format!("stage {} of the chain does not hold: `{fname}` {what}", k + 1))
                                .at(stage.op.span),
                        );
                        ok = false;
                    }
                }
            }
            if !ok || !chain.def.closed {
                if !chain.def.closed
                    && let Some((_, _, at)) = chain.def.expects.first()
                {
                    self.diags.push(
                        Diagnostic::new(Code::Ej07)
                            .with_message("a holonomy is taken around a closed chain, and this one ends elsewhere")
                            .at(*at),
                    );
                }
                continue;
            }
            let count = geometry.covers[geometry.bases[chain.def.stages[0].from].def.cover]
                .def
                .points()
                .map_or(0, |p| p.len());
            let hol: Vec<Option<Phase>> = (0..count).map(|s| cert::holonomy(&certs, s, self.n)).collect();
            for &(s, want, at) in &chain.def.expects {
                let want = want.embed(self.n).unwrap_or(want);
                let Some(got) = hol.get(s).copied().flatten() else { continue };
                if got == want {
                    continue;
                }
                let other = (0..count).find(|&t| hol[t] == Some(want)).or_else(|| (0..count).find(|&t| hol[t] != Some(got)));
                let stage = other.and_then(|t| cert::blame(&certs, s, t, self.n));
                let points = geometry.covers[geometry.bases[chain.def.stages[0].from].def.cover].def.points().unwrap_or_default();
                let mut d = Diagnostic::new(Code::Ej07)
                    .with_message(format!(
                        "holonomy expectation failed: at {} the holonomy is {}, not {}",
                        show(&points[s]),
                        show_phase(got),
                        show_phase(want)
                    ))
                    .at(at);
                if let (Some(k), Some(t)) = (stage, other) {
                    let op = match chain.def.stages[k].op.kind {
                        ExprKind::FnRef(Callee::Fn(op)) => self.name(op),
                        _ => "?",
                    };
                    d = d.also(chain.def.stages[k].op.span, "this stage is indicted").with_note(format!(
                        "stage {} (`{op}`) gives the orbits of {} and {} different phases, and no choice of gauges \
                         at any station removes the discrepancy, which is free of convention",
                        k + 1,
                        show(&points[s]),
                        show(&points[t])
                    ));
                }
                self.diags.push(d);
            }
            out.insert(ci, hol);
        }
        out
    }

    //////////////
    // ISO-SUMS //
    //////////////

    fn isos(&mut self) {
        for iso in &self.unit.isos {
            let def = self.unit.types.adt(iso.adt);
            let variants = def.variants();
            let k = variants.len();
            let needed = k * (k.saturating_sub(1)) / 2;
            let ename = self.interner.resolve(def.name);
            if iso.witnesses.len() != needed {
                self.diags.push(
                    Diagnostic::new(Code::Ea05)
                        .with_message(format!(
                            "`{ename}` has {k} variants, so `[iso: …]` names {needed} witnesses, and this names {}",
                            iso.witnesses.len()
                        ))
                        .at(iso.span)
                        .with_note("the witnesses are ι₁₀, ι₂₀, ι₂₁, …: to each variant from each variant before it"),
                );
                continue;
            }
            // ι_ij for i > j, in the order written
            let mut table: HashMap<(usize, usize), Certificate> = HashMap::new();
            let mut ok = true;
            let mut next = iso.witnesses.iter();
            for i in 1..k {
                for jj in 0..i {
                    let (f, at) = *next.next().expect("counted");
                    let Some(c) = self.circuit(f) else {
                        ok = false;
                        continue;
                    };
                    let width = c.inputs.len();
                    if sim::fixed(&c.body).is_err() || c.outputs.len() != width || width > BASIS_WIDTH {
                        self.diags.push(
                            Diagnostic::new(Code::Ej03)
                                .with_message(format!("`{}` witnesses no isomorphism: it has no fixed unitary action", self.name(f)))
                                .at(at),
                        );
                        ok = false;
                        continue;
                    }
                    let mut u = |v: &Amplitudes| sim::action(&c, v).expect("fixed");
                    let base = basis_base(width, self.n);
                    match cert::derive(&mut u, &base, &base, self.n) {
                        Ok(cert) => {
                            table.insert((i, jj), cert);
                        }
                        Err(_) => ok = false,
                    }
                }
            }
            if !ok {
                continue;
            }
            for i in 2..k {
                for jj in 1..i {
                    for kk in 0..jj {
                        let composite = cert::compose(&table[&(jj, kk)], &table[&(i, jj)]);
                        if composite.as_ref() != Some(&table[&(i, kk)]) {
                            self.diags.push(
                                Diagnostic::new(Code::Ej03)
                                    .with_message(format!(
                                        "`{ename}`'s witnesses do not compose: ι{i}{jj} after ι{jj}{kk} is not ι{i}{kk}"
                                    ))
                                    .at(iso.span)
                                    .with_note("an iso-sum's witnesses satisfy the cocycle law, as certificates"),
                            );
                        }
                    }
                }
            }
        }
    }
}

/// A certificate as it would be written: `[id; (0, π/4)]`.
pub fn describe(c: &Certificate) -> String {
    match c {
        Certificate::Points {
            images,
            schedule,
            stationary,
        } => {
            let map = if *stationary {
                "id".to_owned()
            } else {
                format!("({})", images.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
            };
            format!("[{map}; {}]", show_schedule(schedule))
        }
        Certificate::Scalar(v) => format!("[id; const {}]", show_phase(*v)),
        Certificate::Moving => "a map of its subspace that is not a scalar, which no schedule describes".to_owned(),
    }
}

/// A witness to a certificate's failing a class: two points whose phases
/// differ, or a point not carried to itself.
fn witness(c: &Certificate, source: &Base) -> Option<String> {
    let Certificate::Points {
        images,
        schedule,
        stationary,
    } = c
    else {
        return None;
    };
    let points = source.cover.points()?;
    if let Some(j) = (1..schedule.len()).find(|&j| schedule[j] != schedule[0]) {
        return Some(format!(
            "witness: its phase at {} is {}, and at {} is {}",
            show(&points[0]),
            show_phase(schedule[0]),
            show(&points[j]),
            show_phase(schedule[j])
        ));
    }
    if !stationary && let Some(i) = (0..images.len()).find(|&i| images[i] != i) {
        return Some(format!("witness: it does not keep {}", show(&points[i])));
    }
    None
}

/// A sector's phase.
fn stationary(s: &Judged) -> Option<Phase> {
    match cert::strongest(&s.cert, &s.source) {
        Class::Stat(Some(v)) => Some(v),
        _ => None,
    }
}

/// The computational basis of `width` qubits.
fn basis_base(width: usize, n: u32) -> Base {
    Base {
        cover: basis_cover(width, n),
        gauge: Gauge {
            fiducials: vec![plus(width, n)],
        },
    }
}

/// The base type numbered `b` among the unit's.
fn base_type(unit: &Unit, b: usize) -> Base {
    let geometry = &unit.geometry;
    Base {
        cover: geometry.covers[geometry.bases[b].def.cover].def.clone(),
        gauge: geometry.gauges[geometry.bases[b].def.gauge].def.clone(),
    }
}

/// The certificate of `c` from `source` to `target`: on the stabiliser path
/// where it applies, and by exact action otherwise.
fn certify(c: &Circuit, source: &Base, target: &Base, n: u32) -> Result<Certificate, Fail> {
    if let Some(r) = crate::judge::stabilizer::derive(c, source, target, n) {
        return r;
    }
    // the endpoints may be written at a lower conductor than the one the
    // circuit's angles are counted in, as a unit's are when it imports one
    // of higher conductor; the circuit acts on them in the field of `n`
    let (source, target) = (source.at(n), target.at(n));
    let mut u = |v: &Amplitudes| sim::action(c, v).expect("the circuit's action is fixed");
    cert::derive(&mut u, &source, &target, n)
}

/// What a failed derivation says.
enum Failure {
    /// The operator leaves its cover.
    Leaves(String),
    /// The phase it gives the point, or the code, named is no whole number
    /// of the conductor's parts of a turn.
    Outside(String),
    /// The cover is a code too wide to write out, under a circuit that is
    /// not Clifford.
    Undecided,
}

fn failure(why: Fail) -> Failure {
    match why {
        Fail::Leaves(p, v) => Failure::Leaves(format!("it carries {} to {}, which is not in it", show(&p), show(&v))),
        Fail::Unfixed(code, g) => Failure::Leaves(format!("`{g}`, which fixes every point of it, does not fix what it makes of {code}")),
        Fail::Unreached(code) => Failure::Leaves(format!("it carries the point of {code} to none of its points")),
        Fail::Outside(p) => Failure::Outside(show(&p)),
        Fail::OutsideCode(code) => Failure::Outside(code),
        Fail::Undecided => Failure::Undecided,
    }
}

fn lifts(ops: &[Op]) -> bool {
    ops.iter().any(|op| match op {
        Op::Lift { .. } => true,
        Op::If { then, els, .. } => lifts(then) || lifts(els),
        Op::For { body, .. } => lifts(body),
        _ => false,
    })
}

/// The slots `ops` call.
fn calls(ops: &[Op]) -> Vec<u32> {
    ops.iter()
        .flat_map(|op| match op {
            Op::Call { slot, .. } => vec![*slot],
            Op::If { then, els, .. } => [calls(then), calls(els)].concat(),
            Op::For { body, .. } => calls(body),
            _ => Vec::new(),
        })
        .collect()
}

/////////////////////////////////////////////
// THE VALUES `@judgment` AND ITS KIN GIVE //
/////////////////////////////////////////////

/// A phase as `core` holds a `phase<48>`.
fn phase48(p: Phase) -> Value {
    let p = p.embed(48).unwrap_or(p);
    Value::Struct(vec![Value::Int(i128::from(p.index()), IntTy::USIZE)])
}

fn tri(t: Tri) -> Value {
    Value::Enum {
        variant: match t {
            Tri::Yes => 0,
            Tri::No => 1,
            Tri::Unknown => 2,
        },
        fields: Vec::new(),
    }
}

fn class_value(c: Option<&Class>) -> Value {
    let list = |ps: &[Phase]| Value::Array(ps.iter().map(|&p| phase48(p)).collect());
    let (variant, fields) = match c {
        Some(Class::Rigid) => (0, vec![]),
        Some(Class::Flat(Some(p))) => (1, vec![phase48(*p)]),
        Some(Class::Flat(None)) => (1, vec![phase48(Phase::zero(48))]),
        Some(Class::Locflat) => (2, vec![]),
        Some(Class::Cyc(p)) => (3, vec![list(p.as_deref().unwrap_or(&[]))]),
        Some(Class::Stat(p)) => (4, vec![phase48(p.unwrap_or(Phase::zero(48)))]),
        Some(Class::Sector(p)) => (5, vec![list(p.as_deref().unwrap_or(&[]))]),
        Some(Class::Free) => (6, vec![]),
        None => (7, vec![]),
    };
    Value::Enum { variant, fields }
}

fn some(v: Value) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![v],
    }
}

fn none() -> Value {
    Value::Enum {
        variant: 1,
        fields: Vec::new(),
    }
}

fn cert_value(j: &Judged, sectors: &[Judged]) -> Value {
    let (map, schedule) = match &j.cert {
        Certificate::Points { images, schedule, .. } => (
            images.iter().map(|&i| Value::Int(i as i128, IntTy::USIZE)).collect(),
            schedule.iter().map(|&p| phase48(p)).collect(),
        ),
        Certificate::Scalar(v) => (Vec::new(), vec![phase48(*v)]),
        Certificate::Moving => (Vec::new(), Vec::new()),
    };
    Value::Struct(vec![
        Value::Array(map),
        Value::Array(schedule),
        Value::Array(sectors.iter().map(|s| cert_value(s, &[])).collect()),
        Value::Array(j.points().iter().map(crate::sema::quon::state_value).collect()),
    ])
}

fn text(s: &str) -> Value {
    Value::Array(s.chars().map(Value::Char).collect())
}

fn kernel_value(k: &Kernel, outcomes: &str) -> Value {
    Value::Struct(vec![
        Value::Int(i128::from(k.measured), IntTy::U32),
        Value::Int(i128::from(k.forgotten), IntTy::U32),
        text(outcomes),
    ])
}

fn fragment_value(f: &Fragment) -> Value {
    Value::Struct(vec![
        Value::Int(i128::from(f.conductor), IntTy::U32),
        Value::Bool(f.departure.is_none()),
        Value::Bool(f.stabilizer),
    ])
}

/// A record as `judge::Judgment` holds it; `outcomes` names the outcome
/// type of an instrument.
pub fn judgment_value(r: &Record, outcomes: &str) -> Value {
    let cert = r
        .cert
        .as_ref()
        .map_or_else(none, |j| some(cert_value(j, r.sectors.as_deref().unwrap_or(&[]))));
    Value::Struct(vec![
        tri(r.monic),
        tri(r.unitary),
        class_value(r.contract.as_ref()),
        cert,
        r.kernel.as_ref().map_or_else(none, |k| some(kernel_value(k, outcomes))),
        fragment_value(&r.fragment),
        Value::Bool(r.dynamic),
    ])
}

/// The value `d` asks for, or the diagnostic saying why there is none.
pub fn derived_value<'a>(
    unit: &'a Unit,
    interner: &'a Interner,
    j: &mut Judgments,
    g: &mut Gen<'a>,
    d: &Derived,
    at: Span,
) -> Result<Value, Diagnostic> {
    // an operator's name, outcome type, and record, derived once
    let name = |f: FnId| interner.resolve(unit.func(f).name).to_owned();
    let outcomes = |f: FnId| unit.types.display(unit.func(f).ret, interner);
    let no_record = |f: FnId| {
        Diagnostic::new(Code::Ej01)
            .with_message(format!("`{}` has no judgment: it has no circuit alone", name(f)))
            .at(at)
            .with_note("an operator taking classical values is judged where it is called, with the values it is called with")
    };
    let record = |j: &mut Judgments, g: &mut Gen<'a>, f: FnId| -> Result<Record, Diagnostic> {
        let mut jj = Judging::new(unit, interner, g);
        jj.records = std::mem::take(&mut j.records);
        let r = jj.record(f);
        j.records = jj.records;
        r.ok_or_else(|| no_record(f))
    };

    // each macro's value, as the `judge` library's types hold it
    match *d {
        Derived::Judgment(f) => Ok(judgment_value(&record(j, g, f)?, &outcomes(f))),
        Derived::Kernel(f) => {
            let r = record(j, g, f)?;
            match &r.kernel {
                Some(k) => Ok(kernel_value(k, &outcomes(f))),
                None => Err(Diagnostic::new(Code::Ej03)
                    .with_message(format!("`{}` has no kernel: it neither measures nor forgets", name(f)))
                    .at(at)),
            }
        }
        Derived::Certificate(f) => {
            let r = record(j, g, f)?;
            match &r.cert {
                Some(c) => Ok(cert_value(c, r.sectors.as_deref().unwrap_or(&[]))),
                None => Err(Diagnostic::new(Code::Ej01)
                    .with_message(format!("`{}` has no certificate", name(f)))
                    .at(at)
                    .with_note(r.fragment.departure.clone().or_else(|| r.undecided.clone()).unwrap_or_else(|| {
                        "an instrument's phase lineage ends at its kernel".to_owned()
                    }))),
            }
        }
        Derived::FragmentOf(f) => Ok(fragment_value(&record(j, g, f)?.fragment)),
        Derived::MatrixOf(f) => {
            let c = g.operator(f).ok_or_else(|| no_record(f))?;
            if sim::fixed(&c.body).is_err() || c.inputs.len() != c.outputs.len() {
                return Err(Diagnostic::new(Code::Ej01)
                    .with_message(format!("`{}` has no matrix: it is not a fixed unitary", name(f)))
                    .at(at));
            }
            // column i is what the circuit makes of basis state i
            let n = g.conductor();
            let w = c.inputs.len();
            let size = 1usize << w;
            let bits = |i: usize| ket_bits(i, w);
            let columns: Vec<Amplitudes> = (0..size)
                .map(|col| sim::action(&c, &Amplitudes::basis(bits(col), n)).expect("fixed"))
                .collect();
            let rows = (0..size)
                .map(|r| {
                    Value::Array(
                        columns
                            .iter()
                            .map(|col| {
                                let e = col.terms.get(&bits(r)).cloned().unwrap_or_else(|| Cyclo::zero(n));
                                crate::sema::exact::cyclo_value(&e)
                            })
                            .collect(),
                    )
                })
                .collect();
            Ok(Value::Struct(vec![Value::Array(rows)]))
        }
        Derived::ClassifyOp(f, b) => {
            let c = g.operator(f).ok_or_else(|| no_record(f))?;
            let r = Judging::new(unit, interner, g).derive_from(f, &c, base_type(unit, b));
            Ok(judgment_value(&r, &outcomes(f)))
        }
        Derived::Holonomy(chain, point) => match j.holonomies.get(&chain).and_then(|h| h.get(point).copied().flatten()) {
            Some(p) => {
                let p = p.embed(g.conductor()).unwrap_or(p);
                Ok(Value::Struct(vec![Value::Int(i128::from(p.index()), IntTy::USIZE)]))
            }
            None => Err(Diagnostic::new(Code::Ej01)
                .with_message("this chain has no holonomy there")
                .at(at)
                .with_note("a holonomy is taken at a point of a closed chain whose every stage holds")),
        },
    }
}

impl Judging<'_, '_> {
    /// The record of `f` from the base type `source`, to where it carries
    /// it, gauged by where it takes the source's fiducials.
    fn derive_from(&mut self, f: FnId, c: &Circuit, source: Base) -> Record {
        let mut r = self.derive(f, c, None);
        if r.kernel.is_some() || sim::fixed(&c.body).is_err() {
            return r;
        }
        let u = |v: &Amplitudes| sim::action(c, v).expect("fixed");
        // to the base type itself when the operator keeps its cover, and
        // otherwise to the points it reaches, gauged by where it takes the
        // fiducials
        let target = match source.cover.points() {
            Some(ps) => {
                let images: Vec<Amplitudes> = ps.iter().map(&u).collect();
                if images.iter().all(|v| ps.iter().any(|p| same_point(p, v))) {
                    source.clone()
                } else {
                    Base {
                        cover: Cover::Fin(images),
                        gauge: Gauge {
                            fiducials: source.gauge.fiducials.iter().map(&u).collect(),
                        },
                    }
                }
            }
            None => source.clone(),
        };
        match certify(c, &source, &target, self.n) {
            Ok(cert) => {
                r.contract = Some(cert::strongest(&cert, &source));
                r.cert = Some(Judged { cert, source, target });
                r.sectors = None;
            }
            Err(_) => {
                r.contract = None;
                r.cert = None;
            }
        }
        r
    }
}
