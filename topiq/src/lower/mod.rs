//! Translation phase 9: circuit lowering, uncomputation synthesis and
//! judgement derivation.
//!
//! [`lower`] generates a circuit ([`crate::circuit`]) for each operator
//! of a quantum unit that has one: each `[entry]` operator, and each other
//! operator whose parameters are all quantum, handles or operators, which is
//! generated to check it even though nothing may call it alone. An operator
//! with classical parameters is generated where it is called, for the
//! arguments it is called with. The unit's initialiser runs first, setting
//! what the operators read of the unit's objects.
//!
//! Generation is the interpretation of each body ([`generate`]), with the
//! values it computes with defined in [`value`]. Every problem generation
//! finds in the unit is reported, not only the first. What generation does
//! not do (an operation only a running program performs, such as reading a
//! file, or changing a growable array that is not a named binding) is
//! reported as `TQ003`.
//!
//! # Uncomputation
//!
//! An `aux` ancilla is returned to |0> as its scope ends by undoing its
//! forward cone, and a register its declaration allocated is given back
//! there when what was done to it is shown undone; [`generate`] says how.
//! Both are decided exactly, by computing what the operations do
//! ([`crate::circuit::action`]).
//!
//! # Preparation
//!
//! `prep` carries the circuit analysis made for its state
//! ([`crate::circuit::synth`]); generation allocates the register and
//! places that circuit on it.
//!
//! # Judgements
//!
//! Once the circuits exist, each operator's judgement record is derived
//! from its circuit's exact action, and the claims the unit's annotations
//! make are checked against the records ([`judge`]). An operator another
//! unit declares has the record that unit publishes, read from its
//! interface. The `@static_assert`s whose conditions read judgements are
//! checked then, with the values they ask for in place. A judgement an
//! operator's body reads is derived before that body is generated
//! ([`reads`]).
//!
//! What the unit then offers its importers (the record of each operator
//! with program linkage, with its claims, and each `[entry]` operator's
//! circuit as the document a program embeds) is made by [`publish`].
//!
//! # Rules
//!
//! - **Derivation is exact or refused.** There is no numerical tolerance.
//!   Where a program leaves the fragment the checker decides, each affected
//!   field of the judgement becomes `Unknown`, with a warning, because a
//!   judgement is a claim and an approximate claim is a false one.
//! - **Phase lineage ends at every kernel.** No certificate is claimed,
//!   propagated or optimised across a measurement, which destroys the phase a
//!   certificate is about.
//! - **The phase finishes before code generation,** so a refusal is reported
//!   before anything is emitted.

pub mod generate;
pub mod judge;
pub mod reads;
pub mod value;

use crate::circuit::Circuit;
use crate::diag::Diagnostic;
use crate::intern::Interner;
use crate::tir::{FnId, Ty, Unit};

/// How circuits are generated.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// The unit's conductor.
    pub conductor: u32,
    /// Whether a qubit given back in |0> is allocated again.
    pub reuse: bool,
    /// Whether the unit declares `#pragma fragment(stabilizer)`: every gate
    /// must be a Clifford gate.
    pub stabilizer: bool,
}

/// One operator's circuit.
#[derive(Clone, Debug)]
pub struct Lowered {
    /// The operator.
    pub func: FnId,
    /// Whether it is an `[entry]` operator.
    pub entry: bool,
    /// Its circuit.
    pub circuit: Circuit,
}

/// What phase 9 made of a unit.
#[derive(Clone, Debug, Default)]
pub struct Lowering {
    /// The circuits of the operators that have one.
    pub circuits: Vec<Lowered>,
    /// The judgement of each operator that has one.
    pub records: std::collections::HashMap<FnId, crate::judge::record::Record>,
    /// What it reported.
    pub diagnostics: Vec<Diagnostic>,
}

/// Generates the circuits of the analysed quantum unit `unit`, which was
/// analysed against the interfaces `imports`.
pub fn lower(unit: &Unit, interner: &Interner, options: Options, imports: &[crate::sema::Interface]) -> Lowering {
    let mut g = generate::Gen::new(unit, interner, options.conductor, options.reuse);
    if let Some(init) = unit.init {
        g.initialize(init);
    }
    let imported = imported_records(unit, interner, imports);
    let mut diagnostics = reads::provide(unit, interner, &mut g, &imported);
    let mut circuits = Vec::new();
    for (i, f) in unit.fns.iter().enumerate() {
        let id = FnId(i as u32);
        let entry = f.attrs.entry;
        if !own(unit, id) || !entry && !f.param_types().all(|t| stands_alone(unit, t)) {
            continue;
        }
        if let Some(circuit) = g.operator(id) {
            circuits.push(Lowered { func: id, entry, circuit });
        }
    }
    diagnostics.append(&mut g.diags);
    let mut judgments = judge::judge(unit, interner, &mut g, &circuits, options.stabilizer, imported);
    diagnostics.append(&mut judgments.diags);
    // the assertions that read judgements, now that they exist
    for d in &unit.deferred {
        let mut cond = d.cond.clone();
        let mut fill = Fill {
            unit,
            interner,
            judgments: &mut judgments,
            g: &mut g,
            failed: None,
        };
        crate::tir::visit::VisitMut::expr(&mut fill, &mut cond);
        if let Some(why) = fill.failed {
            diagnostics.push(why);
            continue;
        }
        let holds = crate::sema::consteval::Evaluator::new(unit, interner).top(&cond, &mut crate::sema::consteval::Frame::empty());
        match holds {
            Ok(crate::tir::Value::Bool(true)) => {}
            Ok(_) => diagnostics.push(
                Diagnostic::new(crate::diag::Code::Em01)
                    .with_message(d.message.clone())
                    .at(d.span)
                    .with_note("`@static_assert` makes a program ill-formed when its condition is false"),
            ),
            Err(e) => diagnostics.push(crate::sema::fold::describe(
                e,
                &crate::sema::fold::Need::Constant("the condition of `@static_assert`"),
                unit,
                interner,
            )),
        }
    }
    diagnostics.append(&mut g.diags);
    Lowering {
        circuits,
        records: judgments.records,
        diagnostics,
    }
}

/// The records the units `unit` imports publish of their operators it
/// uses, by the operator as `unit` holds it.
fn imported_records(
    unit: &Unit,
    interner: &Interner,
    imports: &[crate::sema::Interface],
) -> std::collections::HashMap<FnId, crate::judge::record::Record> {
    let mut out = std::collections::HashMap::new();
    for (i, f) in unit.fns.iter().enumerate() {
        let Some(from) = &f.origin else { continue };
        if !f.args.is_empty() {
            continue;
        }
        let published = imports
            .iter()
            .find(|iface| &iface.unit == from)
            .and_then(|iface| iface.record(interner.resolve(f.name)));
        if let Some(p) = published {
            out.insert(FnId(i as u32), p.record.clone());
        }
    }
    out
}

/// What the quantum unit `unit`, lowered as `lowering`, publishes: of each
/// operator with program linkage that has a record, the record, the claims
/// its `[expect: …]` makes and the declarations its endpoints name; and each
/// `[entry]` operator's circuit, as its document.
pub fn publish(
    unit: &Unit,
    interner: &Interner,
    lowering: &Lowering,
) -> (Vec<crate::judge::record::Published>, Vec<crate::sema::interface::EntryCircuit>) {
    use crate::judge::record::{Claim, Published};
    use crate::tir::claims::Expect;
    let mut ids: Vec<FnId> = lowering.records.keys().copied().filter(|&id| own(unit, id)).collect();
    ids.sort_by_key(|f| f.0);
    let outcomes = |id: FnId| unit.types.display(unit.func(id).ret, interner);
    let mut records = Vec::new();
    for id in ids {
        let f = unit.func(id);
        if f.linkage != crate::ast::Linkage::Program {
            continue;
        }
        let claims = unit.claims.get(&id);
        let record = lowering.records[&id].clone();
        // the claims' phases in the group the record's are in
        let n = record.fragment.conductor;
        records.push(Published {
            name: interner.resolve(f.name).to_owned(),
            record,
            outcomes: outcomes(id),
            claims: claims
                .map(|c| {
                    c.expects
                        .iter()
                        .map(|(e, _)| match e {
                            Expect::Monic => Claim::Monic,
                            Expect::Unitary => Claim::Unitary,
                            Expect::Contract(k) => Claim::Contract(k.at(n)),
                            Expect::Frame(k) => Claim::Frame(k.map(|p| p.embed(n).unwrap_or(p))),
                            Expect::Outcomes(t) => Claim::Outcomes(unit.types.display(*t, interner)),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            cover: claims.and_then(|c| c.cover_decl.clone()),
            gauge: claims.and_then(|c| c.gauge_decl.clone()),
        });
    }
    let mut circuits = Vec::new();
    for l in lowering.circuits.iter().filter(|l| l.entry) {
        let f = unit.func(l.func);
        let name = interner.resolve(f.name);
        let params: Vec<Ty> = f.param_types().collect();
        let sig = unit.types.qualified_signature(&params, f.ret, interner, &unit.name);
        let outcomes = outcomes(l.func);
        let record = lowering.records.get(&l.func);
        let (source, target) = record
            .and_then(|r| r.cert.as_ref())
            .map(|j| (crate::meta::encode::base_digest(&j.source), crate::meta::encode::base_digest(&j.target)))
            .unwrap_or_default();
        let heading = crate::circuit::document::Heading {
            name: &format!("{}::{name}", unit.name),
            sig: &sig,
            judgment: record.map(|r| (r, outcomes.as_str())),
            endpoints: (&source, &target),
        };
        circuits.push(crate::sema::interface::EntryCircuit {
            name: f.name,
            document: crate::circuit::document::document(&l.circuit, &heading),
            qasm: crate::circuit::qasm::write(&l.circuit),
            sig,
            dynamic: l.circuit.dynamic,
            allocates: l.circuit.allocates_while_running(),
            monic_slots: l.circuit.slots.iter().map(|s| s.monic).collect(),
        });
    }
    (records, circuits)
}

/// Puts the values asked of the judgements in place of the asking.
struct Fill<'a, 'j, 'g> {
    unit: &'a Unit,
    interner: &'a Interner,
    judgments: &'j mut judge::Judgments,
    g: &'g mut generate::Gen<'a>,
    failed: Option<Diagnostic>,
}

impl crate::tir::visit::VisitMut for Fill<'_, '_, '_> {
    fn expr(&mut self, e: &mut crate::tir::Expr) {
        if let crate::tir::ExprKind::Intrinsic {
            which: crate::tir::Intrinsic::Derived(k),
            ..
        } = e.kind
        {
            let d = &self.unit.derived[k as usize];
            match judge::derived_value(self.unit, self.interner, self.judgments, self.g, d, e.span) {
                Ok(v) => e.kind = crate::tir::ExprKind::Const(v),
                Err(why) => {
                    if self.failed.is_none() {
                        self.failed = Some(why);
                    }
                }
            }
            return;
        }
        crate::tir::visit::walk_expr_mut(self, e);
    }
}

/// Whether `id` is an operator the unit declares itself: not imported, not
/// an instance of a generic, not a closure's body, and not the initialiser.
fn own(unit: &Unit, id: FnId) -> bool {
    let f = unit.func(id);
    f.origin.is_none() && f.args.is_empty() && !f.closure && unit.init != Some(id)
}

/// Whether a parameter of type `t` lets its operator be generated without
/// its caller: it is quantum, a handle, or an operator.
fn stands_alone(unit: &Unit, t: Ty) -> bool {
    unit.types.is_quantum(t)
        || unit.types.as_ref(t).is_some_and(|(_, x)| unit.types.is_quantum(x))
        || matches!(t, Ty::Fn(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Op;
    use crate::sema::testing::quantum_with_names;

    fn lowered(src: &str, reuse: bool) -> Lowering {
        let (u, d, names) = quantum_with_names(src);
        assert!(d.is_empty(), "{d:?}");
        lower(&u, &names, Options { conductor: 8, reuse, stabilizer: false }, &[])
    }

    #[test]
    fn a_quantum_parameter_is_an_input_and_a_returned_register_an_output() {
        let l = lowered("fn f(r: [qubit; 2]) -> [qubit; 2] { r }", true);
        let c = &l.circuits[0].circuit;
        assert_eq!(c.inputs.len(), 2);
        assert_eq!(c.outputs, c.inputs);
        assert!(c.result.is_none(), "nothing classical is returned");
    }

    #[test]
    fn a_qubit_given_back_is_allocated_again_unless_allocation_is_linear() {
        let src = "fn f() { let a: qubit; } \n fn g() { f(); f(); }";
        let wires = |reuse| lowered(src, reuse).circuits.iter().find(|l| l.circuit.name == "g").unwrap().circuit.wires;
        assert_eq!(wires(true), 1);
        assert_eq!(wires(false), 2);
    }

    #[test]
    fn a_measurement_writes_a_bit_the_result_reads() {
        let l = lowered("fn f() -> bool { let q: [qubit; 1] = prep |1>; let m = measure q; m[0] }", true);
        let c = &l.circuits[0].circuit;
        assert!(matches!(c.body.last(), Some(Op::Measure { .. })));
        assert_eq!(c.result, Some(crate::circuit::CExpr::Bit(crate::circuit::Bit(0))));
    }
}
