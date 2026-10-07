//! States written in QUON, and the `quon::State` values that hold them.
//!
//! `prep` of a state written out, of a named constant state, or of a constant
//! expression of type `quon::State` computes the state's amplitudes exactly at
//! the unit's conductor ([`crate::quon::eval`]), checks that they form a unit
//! vector, and makes the circuit that prepares it
//! ([`crate::circuit::synth`]), which the expression carries. The register it
//! allocates is as wide as the state.
//!
//! `@embed("states.quon")` reads a QUON document of named states as a
//! constant of the structure written for it, a `quon::State` field for each
//! item.
//!
//! # Rules
//!
//! - **A coefficient is exact at the unit's conductor.** One the field of
//!   the conductor does not hold is `EJ04`, naming the conductor that would.
//! - **A state is a unit vector:** its squared magnitudes sum to one, which
//!   is decided exactly (`EJ14`).
//! - **A state is prepared exactly or refused.** A state no circuit of the
//!   standard gates prepares exactly is `EQ10`, naming the amplitude that
//!   stops it and, when there is one, the conductor at which it is prepared.
//! - **`prep` of a `quon::State` needs a constant,** since the register's
//!   width is part of its type; a state computed as the program runs is
//!   prepared into a register the program has, by `qpu::prepare`.
//! - **A document's state is as wide as the type it is stated against:**
//!   `qubit`, `[qubit; N]`, or a quantum enumeration, whose register is its
//!   tag and then its payload (`ES06`).
//! - **A `quon::State` holds its amplitudes as `cyclo<48>`,** whose field
//!   holds those of every conforming conductor.

use std::sync::Arc;

use chumsky::Parser;

use crate::ast;
use crate::circuit::synth::{self, Obstruction};
use crate::diag::{Code, Diagnostic, Limit};
use crate::intern::Symbol;
use crate::parse::input::{Cx, TokenStream, stream};
use crate::quon::ast::{QState, QType};
use crate::quon::eval::{self, Amplitudes, ket};
use crate::span::{Span, Spanned};
use crate::tir::{AdtKind, Expr, ExprKind, IntTy, QuantumOp, Ty, Value};

use super::body::Checker;
use super::items::UnitCx;
use super::exact::{cyclo_of, cyclo_value};

/// The conductor a `quon::State` holds its amplitudes at.
pub const STATE_CONDUCTOR: u32 = 48;

impl Checker<'_, '_> {
    /// `prep` of a state written in QUON, or of a constant `quon::State`.
    pub fn prep(&mut self, arg: &ast::PrepArg, span: Span) -> Expr {
        if !self.cx.quantum {
            return self.quantum_in_classical("prepare a state", span);
        }
        self.cx.prep_expr(arg, span).unwrap_or_else(|| Checker::error(span))
    }

    /// How many qubits a document's type for a state holds: `qubit` one,
    /// and `[qubit; N]` N, which is written as a number or names a
    /// constant.
    fn qtype_width(&mut self, t: &Spanned<QType>) -> Option<usize> {
        match &t.node {
            QType::Qubit => Some(1),
            QType::Path(p) if p.len() == 1 && self.name(p[0]) == "qubit" => Some(1),
            // a quantum enumeration's register: its tag, then its payload
            QType::Path(p) => {
                let path = ast::Path {
                    segments: p.iter().map(|&s| Spanned::new(s, t.span)).collect(),
                };
                let written = Spanned::new(ast::Type::Path { path, args: Vec::new() }, t.span);
                let ty = super::types::resolve(self.cx, &written).map_err(|d| self.report(*d)).ok()?.ty;
                match ty {
                    Ty::Adt(id) if self.types().is_quantum(ty) && !self.adt(id).is_struct() => {
                        usize::try_from(self.types().qubits(ty)).ok()
                    }
                    _ => {
                        let what = self.describe(ty);
                        self.report(
                            Diagnostic::new(Code::Es06)
                                .with_message(format!("a state is stated against a register, and this is {what}"))
                                .at(t.span)
                                .with_help("state it against `qubit`, `[qubit; N]` or a quantum enumeration"),
                        );
                        None
                    }
                }
            }
            QType::Register(n) => {
                let text: String = self.name(n.node).chars().filter(|&c| c != '_').collect();
                if let Ok(w) = text.parse::<usize>() {
                    return Some(w);
                }
                let path = ast::Expr::Path {
                    path: ast::Path::single(*n),
                    args: Vec::new(),
                };
                let v = self.cx.const_value(&Spanned::new(path, n.span), Ty::USIZE, "the width of a register")?;
                v.as_int().and_then(|w| usize::try_from(w).ok())
            }
        }
    }

    /// Whether `ty` is the `quon` library's `State`.
    pub fn is_quon_state(&mut self, ty: Ty) -> bool {
        let ty = self.shallow(ty);
        self.cx.is_quon_state(ty)
    }
}

impl UnitCx<'_> {
    /// `prep` of `arg`: the register it allocates, prepared in the state by
    /// the circuit analysis made for it; `None` once why not is reported.
    pub(super) fn prep_expr(&mut self, arg: &ast::PrepArg, span: Span) -> Option<Expr> {
        let state = match arg {
            ast::PrepArg::State(s) => self.written_state(s),
            ast::PrepArg::Expr(e) => self.constant_state(e),
        }?;
        if let Some(d) = Limit::RegisterQubits.check(u32::try_from(state.width).unwrap_or(u32::MAX), span) {
            self.report(d);
            return None;
        }
        if !state.is_normalized() {
            self.report(not_normalized(&state, span));
            return None;
        }
        let prepared = match synth::prepare(&state) {
            Ok(p) => p,
            Err(why) => {
                self.report(unsynthesisable(&state, &why, span));
                return None;
            }
        };
        let ty = self.unit.types.array(Ty::Qubit, state.width as u64);
        Some(Expr {
            kind: ExprKind::Quantum {
                op: QuantumOp::Prep(Arc::new(prepared)),
                args: Vec::new(),
            },
            ty,
            span,
        })
    }

    /// The amplitudes of a state written in QUON, at the unit's conductor,
    /// its names being constant `quon::State`s; `None` once why there are
    /// none has been reported.
    pub fn written_state(&mut self, s: &Spanned<QState>) -> Option<Amplitudes> {
        let n = self.conductor;
        let interner = self.interner;
        let mut names = |name: Symbol, at: Span| {
            let path = ast::Expr::Path {
                path: ast::Path::single(Spanned::new(name, at)),
                args: Vec::new(),
            };
            self.constant_state(&Spanned::new(path, at))
        };
        match eval::evaluate(s, n, interner, &mut names) {
            Ok(a) => Some(a),
            Err(diags) => {
                for d in diags {
                    self.report(d);
                }
                None
            }
        }
    }

    /// The amplitudes of `e`, a constant `quon::State`, at the unit's
    /// conductor.
    pub fn constant_state(&mut self, e: &Spanned<ast::Expr>) -> Option<Amplitudes> {
        let (value, ty) = self.typed_const(e, "a state")?;
        if !self.is_quon_state(ty) {
            let what = super::report::describe(ty, &self.unit.types, self.interner);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("a state here is a `quon::State`, and this is {what}"))
                    .at(e.span)
                    .with_note("a state is written in QUON, or is a constant `quon::State`")
                    .with_help("write the state out, as in `isq2 * |0> + isq2 * |1>`"),
            );
            return None;
        }
        match amplitudes_of(&value, self.conductor) {
            Ok(a) => Some(a),
            Err(why) => {
                let d = why.diagnostic(e.span, self.conductor);
                self.report(d);
                None
            }
        }
    }

    /// Whether `ty`, settled, is the `quon` library's `State`.
    pub fn is_quon_state(&self, ty: Ty) -> bool {
        let Ty::Adt(id) = ty else { return false };
        let def = self.unit.types.adt(id);
        def.origin.as_deref() == Some("quon") && self.interner.resolve(def.name) == "State"
    }
}

impl Checker<'_, '_> {
    /// `@embed` of a QUON document, read as `want`: a structure with a
    /// `quon::State` field for each of the document's states.
    pub(super) fn embed_quon(&mut self, path: &str, source: crate::span::SourceId, spliced: &crate::source::Spliced, want: Option<Ty>, span: Span) -> Expr {
        // read the document
        let Some((tokens, eoi)) = self.lex_document(source, spliced) else {
            return Checker::error(span);
        };
        let parsed = {
            let cx = Cx::new(self.interner);
            crate::quon::parse::qdoc::<TokenStream>(cx)
                .parse(stream(&tokens, eoi))
                .into_result()
        };
        let doc = match parsed {
            Ok(doc) => doc,
            Err(errors) => {
                for e in errors {
                    self.report(
                        Diagnostic::new(Code::Es02)
                            .with_message(format!("`{path}` is not a QUON document here: {e}"))
                            .at(*e.span())
                            .with_note("a QUON document is a list of states, each written `name: [qubit; N] = state;`"),
                    );
                }
                return Checker::error(span);
            }
        };

        // the structure it is read as
        let names: Vec<String> = doc.items.iter().map(|i| self.name(i.name.node).to_owned()).collect();
        let fields = want.map(|t| self.settled(t)).and_then(|t| match t {
            Ty::Adt(id) => match &self.adt(id).kind {
                AdtKind::Struct { fields } => Some(fields.clone()),
                AdtKind::Enum { .. } => None,
            },
            _ => None,
        });
        let Some(fields) = fields else {
            let schema: Vec<String> = names.iter().map(|n| format!("{n}: quon::State")).collect();
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`{path}` is read as a structure with a `quon::State` for each state it names"))
                    .at(span)
                    .with_note("a QUON document is a list of named states, and a structure gives each a field")
                    .with_help(format!(
                        "declare `struct States {{ {} }}` and write `let S: const States = @embed(\"{path}\");`",
                        schema.join(", ")
                    )),
            );
            return Checker::error(span);
        };

        // each state, which may name the ones before it
        let n = self.cx.conductor;
        let mut done: Vec<(Symbol, Amplitudes)> = Vec::new();
        let mut ok = true;
        for item in &doc.items {
            let interner = self.interner;
            let mut lookup = |name: Symbol, at: Span| {
                let found = done.iter().find(|(n, _)| *n == name).map(|(_, a)| a.clone());
                if found.is_none() {
                    let text = interner.resolve(name);
                    self.report(
                        Diagnostic::new(Code::Es04)
                            .with_message(format!("no state `{text}` is named before this one in the document"))
                            .at(at)
                            .with_note("a state of a QUON document may name the states written before it"),
                    );
                }
                found
            };
            let Ok(state) = eval::evaluate(&item.state, n, interner, &mut lookup).map_err(|diags| {
                for d in diags {
                    self.report(d);
                }
            }) else {
                ok = false;
                continue;
            };
            let Some(declared) = self.qtype_width(&item.ty) else {
                ok = false;
                continue;
            };
            if declared != state.width {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!(
                            "this state has {} qubits, but its type says {declared}",
                            state.width
                        ))
                        .at_with(item.state.span, format!("{} qubits", state.width))
                        .also(item.ty.span, "the type written for it")
                        .with_note("a document's type for a state is the register it is prepared in"),
                );
                ok = false;
            } else if !state.is_normalized() {
                self.report(not_normalized(&state, item.state.span));
                ok = false;
            }
            done.push((item.name.node, state));
        }
        if !ok {
            return Checker::error(span);
        }

        // a field for each state, and a state for each field
        let mut values = Vec::with_capacity(fields.len());
        for f in &fields {
            let field = self.name(f.name).to_owned();
            if !self.is_quon_state(f.ty) {
                let what = self.show(f.ty);
                self.report(
                    Diagnostic::new(Code::Et01)
                        .with_message(format!("the field `{field}` is a `{what}`, but a QUON document holds states"))
                        .at(span)
                        .with_note("each field of the structure a QUON document is read as is a `quon::State`"),
                );
                return Checker::error(span);
            }
            let Some((_, state)) = done.iter().find(|(n, _)| *n == f.name) else {
                self.report(
                    Diagnostic::new(Code::Et01)
                        .with_message(format!("`{path}` names no state `{field}`"))
                        .at(span)
                        .with_note(format!("the document names {}", names.join(", ")))
                        .with_help("give the structure a field for each state the document names, and no others"),
                );
                return Checker::error(span);
            };
            values.push(state_value(state));
        }
        for (name, _) in &done {
            if !fields.iter().any(|f| f.name == *name) {
                let text = self.name(*name).to_owned();
                let what = want.map(|t| self.show(t)).unwrap_or_default();
                self.report(
                    Diagnostic::new(Code::Et01)
                        .with_message(format!("`{what}` has no field for the state `{text}`"))
                        .at(span)
                        .with_note("an embedded document is checked against the type it is read as"),
                );
                return Checker::error(span);
            }
        }
        Expr::constant(Value::Struct(values), want.expect("the type was found above"), span)
    }
}

/// A `quon::State` value holding `a`: `State { width, terms }`, each term
/// `Term { ket, amp }`.
pub fn state_value(a: &Amplitudes) -> Value {
    let terms = a
        .terms
        .iter()
        .map(|(bits, c)| {
            let amp = c.embed(STATE_CONDUCTOR).expect("every conforming conductor divides 48");
            Value::Struct(vec![
                Value::Array(bits.iter().map(|&b| Value::Bool(b)).collect()),
                cyclo_value(&amp),
            ])
        })
        .collect();
    Value::Struct(vec![Value::Int(a.width as i128, IntTy::U32), Value::Array(terms)])
}

/// Why a `quon::State` value is not a state at a conductor.
#[derive(Clone, Debug)]
pub enum NotAState {
    /// A term's ket is not as wide as the state: the ket, and the width.
    Width(Vec<bool>, usize),
    /// An amplitude the conductor's field does not hold: its ket, and the
    /// smallest conforming conductor whose field does, if there is one.
    Outside(Vec<bool>, Option<u32>),
    /// Not the shape of a `quon::State`.
    Shape,
}

impl NotAState {
    fn diagnostic(&self, span: Span, n: u32) -> Diagnostic {
        match self {
            NotAState::Width(bits, w) => Diagnostic::new(Code::Es06)
                .with_message(format!(
                    "this state is on {w} qubits, but its term {} names {}",
                    ket(bits),
                    bits.len()
                ))
                .at(span)
                .with_note("every term of a state names a basis state of the one register"),
            NotAState::Outside(bits, m) => {
                let help = match m.and_then(|m| crate::exact::conductor_admitting(m, n)) {
                    Some(m) => format!("add `#pragma conductor({m})` to the unit"),
                    None => "use amplitudes in the field of the unit's conductor".to_owned(),
                };
                let needs = match m {
                    Some(m) => format!("conductor {m}"),
                    None => "a conductor no unit may have".to_owned(),
                };
                Diagnostic::new(Code::Ej04)
                    .with_message(format!(
                        "the amplitude of {} in this state needs {needs}, but the unit's conductor is {n}",
                        ket(bits)
                    ))
                    .at(span)
                    .with_note("a state's amplitudes are elements of the field of the unit's conductor")
                    .with_help(help)
            }
            NotAState::Shape => Diagnostic::new(Code::Tq011)
                .with_message("this constant does not have the shape of a `quon::State`")
                .at(span),
        }
    }
}

/// The state a `quon::State` value holds, at conductor `n`.
///
/// # Errors
///
/// When a term is not as wide as the state, or an amplitude is not in the
/// field of `n`.
pub fn amplitudes_of(v: &Value, n: u32) -> Result<Amplitudes, NotAState> {
    // a width, and a list of terms
    let Value::Struct(parts) = v else { return Err(NotAState::Shape) };
    let [width, Value::Array(terms)] = parts.as_slice() else {
        return Err(NotAState::Shape);
    };
    let width = width.as_int().and_then(|w| usize::try_from(w).ok()).ok_or(NotAState::Shape)?;
    let mut a = Amplitudes {
        width,
        n,
        terms: Default::default(),
    };

    // each term's bits and amplitude, summed with any earlier term on the same bits
    for t in terms {
        let Value::Struct(fields) = t else { return Err(NotAState::Shape) };
        let [Value::Array(bits), amp] = fields.as_slice() else {
            return Err(NotAState::Shape);
        };
        let bits: Vec<bool> = bits
            .iter()
            .map(|b| match b {
                Value::Bool(b) => Ok(*b),
                _ => Err(NotAState::Shape),
            })
            .collect::<Result<_, _>>()?;
        if bits.len() != width {
            return Err(NotAState::Width(bits, width));
        }
        let amp = cyclo_of(amp, STATE_CONDUCTOR).ok_or(NotAState::Shape)?;
        let Some(c) = amp.restrict(n) else {
            // outside the unit's field: name the smallest conductor that would hold it
            let m = [8, 16, 24].into_iter().find(|&m| amp.restrict(m).is_some());
            return Err(NotAState::Outside(bits, m));
        };
        let sum = match a.terms.get(&bits) {
            Some(x) => x.add(&c),
            None => c,
        };
        if sum.is_zero() {
            a.terms.remove(&bits);
        } else {
            a.terms.insert(bits, sum);
        }
    }
    Ok(a)
}

/// `EJ14`: a state whose squared magnitudes do not sum to one.
pub(super) fn not_normalized(state: &Amplitudes, span: Span) -> Diagnostic {
    let sum = state.norm_sq();
    let said = match sum.as_rational() {
        Some(f) => f.to_string(),
        None => format!("about {:.4}", sum.to_complex_lossy().0),
    };
    Diagnostic::new(Code::Ej14)
        .with_message(format!(
            "this state is not a unit vector: the squared magnitudes of its amplitudes sum to {said}"
        ))
        .at(span)
        .with_note(
            "the squared magnitude of an amplitude is the probability of its basis state, and the \
             probabilities of a state sum to one, exactly",
        )
        .with_help("scale the amplitudes, as in `isq2 * |0> + isq2 * |1>`")
}

/// `EQ10`: a state no circuit of the standard gates prepares exactly, or
/// none was found for.
fn unsynthesisable(state: &Amplitudes, why: &Obstruction, span: Span) -> Diagnostic {
    let n = state.n;
    match why {
        Obstruction::Ring(bits, c, odd) => Diagnostic::new(Code::Eq10)
            .with_message(format!(
                "no circuit prepares this state exactly: the amplitude of {} is {c}, whose denominator has the factor {odd}",
                ket(bits)
            ))
            .at(span)
            .with_note(
                "every standard gate's matrix has entries whose denominators are powers of two, from \
                 the 1/√2 of `h`, so every amplitude a circuit of them prepares has such a denominator, \
                 at every conductor",
            )
            .with_help("prepare a state whose amplitudes are built from `isq2`, `i` and roots of unity; `prep` never approximates"),
        Obstruction::Stuck(bits, c) => {
            let elsewhere = [16u32, 24].into_iter().filter(|&m| m != n && m.is_multiple_of(n)).find(|&m| {
                let widened = Amplitudes {
                    width: state.width,
                    n: m,
                    terms: state
                        .terms
                        .iter()
                        .map(|(b, c)| (b.clone(), c.embed(m).expect("n divides m")))
                        .collect(),
                };
                synth::prepare(&widened).is_ok()
            });
            let help = match elsewhere {
                Some(m) => format!("add `#pragma conductor({m})` to the unit, at which it is prepared"),
                None => "no conforming conductor was found at which it is prepared".to_owned(),
            };
            Diagnostic::new(Code::Eq10)
                .with_message(format!(
                    "no exact preparation of this state was found at conductor {n}: the amplitude of {} ({c}) could not be reduced",
                    ket(bits)
                ))
                .at(span)
                .with_note(
                    "a state is prepared by reducing it to |0…0> two amplitudes at a time, each step \
                     lowering the powers of two in their denominators; for this state no step did",
                )
                .with_help(help)
        }
    }
}
