//! A circuit's document: the TCON text one entry circuit travels as.
//!
//! A quantum unit's `.tqu` carries a document for each of its `[entry]`
//! operators, beside the circuit's OpenQASM 3, and a program holding a
//! handle to one embeds it: in the hybrid image, for the program's own
//! processor, and in a circuit archive. The document holds everything the
//! circuit is run and checked from without its source: the circuit and its
//! judgement.
//!
//! ```text
//! Circuit {
//!     name: "deutsch::deutsch",          // qualified by its unit
//!     sig: "fn(bool) -> bool",           // as a program writes the type
//!     conductor: 8,
//!     params: [{ name: "n", ty: "u32" }],
//!     slots: [{ name: "oracle", widths: [1, 1], monic: true, keeps: [] }],
//!     inputs: [0], registers: [1], outputs: [], nodes: [],
//!     wires: 2, bits: 1, vars: 0,
//!     grown: 0,                          // registers of run-time length
//!     body: [Gate { gate: H, targets: [0], controls: [] }, …],
//!     result: Bit(0),                    // or null
//!     dynamic: false,
//!     judgment: Judgment { … },          // or null
//!     source: "…", target: "…",          // digests of the certificate's endpoints
//! }
//! ```
//!
//! An expression made of others holds them in a list, as a Topiq
//! enumeration holding values of its own type must: `Binary("Add", [a, b])`,
//! `Select([c, t, f])`.
//!
//! Every name is one a Topiq program can declare, so the `tcon` library
//! reads a document into Topiq values: the judgement is written as
//! `judge::Judgment`, each phase as a `phase<48>` (`{ m: 12 }` is a
//! quarter turn), a certificate's points as states in QUON, as the `quon`
//! library writes them, and an exact number as the list of its coefficients
//! over the powers of the root of unity, each a fraction written as text,
//! `"1/2"`. An integer value is written as its digits and its type,
//! `Int("5", "u32")`, so that every value of every integer type reads back
//! exactly.

use crate::exact::{Cyclo, Phase};
use crate::judge::cert::{Certificate, Class};
use crate::judge::record::{Judged, Record, Tri};
use crate::quon::eval::{Amplitudes, ket};
use crate::tcon::write::{self, Doc};
use crate::tir::Value;

use super::ir::{Angle, CExpr, Circuit, Control, GateOp, Op, Wire};

/// What names a circuit's document and says what it is for.
pub struct Heading<'a> {
    /// The operator.
    pub name: &'a str,
    /// Its signature, as a program writes the type, every type qualified.
    pub sig: &'a str,
    /// Its judgement, and the name of an instrument's outcome type.
    pub judgment: Option<(&'a Record, &'a str)>,
    /// Digests of the base types its certificate is from and to, which
    /// tell whether one circuit's target is another's source; empty when it
    /// has no certificate.
    pub endpoints: (&'a str, &'a str),
}

/// The document of the circuit `c`.
pub fn document(c: &Circuit, h: &Heading<'_>) -> String {
    write::write(&doc(c, h))
}

fn object(fields: Vec<(&str, Doc)>) -> Doc {
    Doc::typed("", fields)
}

fn int(n: impl Into<u64>) -> Doc {
    Doc::Int(n.into())
}

fn wires(ws: &[Wire]) -> Doc {
    Doc::list(ws, |w| int(w.0))
}

fn doc(c: &Circuit, h: &Heading<'_>) -> Doc {
    let params = Doc::list(&c.params, |p| object(vec![("name", Doc::text(&p.name)), ("ty", Doc::text(&p.ty))]));
    let slots = Doc::list(&c.slots, |s| {
        let keeps = Doc::list(&s.keeps, |(arg, state)| {
            object(vec![("arg", int(*arg)), ("state", Doc::list(state, |x| cyclo(x, c.conductor)))])
        });
        object(vec![
            ("name", Doc::text(&s.name)),
            ("widths", Doc::list(&s.widths, |&w| int(w))),
            ("monic", Doc::Bool(s.monic)),
            ("keeps", keeps),
        ])
    });
    let nodes = Doc::list(&c.nodes, |(n, w)| object(vec![("node", int(*n)), ("wire", int(w.0))]));
    Doc::typed(
        "Circuit",
        vec![
            ("name", Doc::text(h.name)),
            ("sig", Doc::text(h.sig)),
            ("conductor", int(c.conductor)),
            ("params", params),
            ("slots", slots),
            ("inputs", wires(&c.inputs)),
            ("registers", Doc::list(&c.registers, |&w| int(w))),
            ("outputs", wires(&c.outputs)),
            ("nodes", nodes),
            ("wires", int(c.wires)),
            ("bits", int(c.bits)),
            ("vars", int(c.vars)),
            ("grown", int(c.grown)),
            ("body", ops(&c.body, c.conductor)),
            ("result", c.result.as_ref().map_or(Doc::tag("null"), expr)),
            ("dynamic", Doc::Bool(c.dynamic)),
            ("judgment", h.judgment.map_or(Doc::tag("null"), |(r, o)| judgment(r, o))),
            ("source", Doc::text(h.endpoints.0)),
            ("target", Doc::text(h.endpoints.1)),
        ],
    )
}

fn ops(ops: &[Op], n: u32) -> Doc {
    Doc::list(ops, |o| op(o, n))
}

fn controls(cs: &[Control]) -> Doc {
    Doc::list(cs, |c| object(vec![("wire", int(c.wire.0)), ("on", Doc::Bool(c.on))]))
}

fn angle(a: &Angle, n: u32) -> Doc {
    match a {
        Angle::Fixed(p) => Doc::call("Fixed", vec![int(p.embed(n).unwrap_or(*p).index())]),
        Angle::Runtime(e) => Doc::call("Runtime", vec![expr(e)]),
    }
}

fn gate(g: &GateOp, n: u32) -> Doc {
    match g {
        GateOp::X => Doc::tag("X"),
        GateOp::Y => Doc::tag("Y"),
        GateOp::Z => Doc::tag("Z"),
        GateOp::H => Doc::tag("H"),
        GateOp::S => Doc::tag("S"),
        GateOp::Sdg => Doc::tag("Sdg"),
        GateOp::T => Doc::tag("T"),
        GateOp::Tdg => Doc::tag("Tdg"),
        GateOp::Swap => Doc::tag("Swap"),
        GateOp::Phase(a) => Doc::call("Phase", vec![angle(a, n)]),
        GateOp::GPhase(a) => Doc::call("GPhase", vec![angle(a, n)]),
    }
}

fn op(o: &Op, n: u32) -> Doc {
    let (name, fields) = match o {
        Op::Gate { gate: g, targets, controls: cs } => {
            ("Gate", vec![("gate", gate(g, n)), ("targets", wires(targets)), ("controls", controls(cs))])
        }
        Op::Unitary { matrix, targets, controls: cs } => (
            "Unitary",
            vec![
                ("size", int(matrix.size as u64)),
                ("entries", Doc::list(&matrix.entries, |x| cyclo(x, n))),
                ("targets", wires(targets)),
                ("controls", controls(cs)),
            ],
        ),
        Op::Measure { wire, bit } => ("Measure", vec![("wire", int(wire.0)), ("bit", int(bit.0))]),
        Op::Alloc { wire } => ("Alloc", vec![("wire", int(wire.0))]),
        Op::Release { wire } => ("Release", vec![("wire", int(wire.0))]),
        Op::Forget { wire } => ("Forget", vec![("wire", int(wire.0))]),
        Op::If { cond, then, els } => ("If", vec![("cond", expr(cond)), ("then", ops(then, n)), ("els", ops(els, n))]),
        Op::For { var, start, end, body } => (
            "For",
            vec![("var", int(var.0)), ("start", expr(start)), ("end", expr(end)), ("body", ops(body, n))],
        ),
        Op::Let { var, value } => ("Let", vec![("var", int(var.0)), ("value", expr(value))]),
        Op::Lift { var, value } => ("Lift", vec![("var", int(var.0)), ("value", expr(value))]),
        Op::Call { slot, args, controls: cs, adjoint } => (
            "Call",
            vec![
                ("slot", int(*slot)),
                ("args", Doc::list(args, |a| wires(a))),
                ("controls", controls(cs)),
                ("adjoint", Doc::Bool(*adjoint)),
            ],
        ),
        Op::Grow { reg, wire } => ("Grow", vec![("reg", int(*reg)), ("wire", int(wire.0))]),
        Op::Element { wire, reg, index } => {
            ("Element", vec![("wire", int(wire.0)), ("reg", int(*reg)), ("index", expr(index))])
        }
        Op::MeasureAll { reg, var } => ("MeasureAll", vec![("reg", int(*reg)), ("var", int(var.0))]),
    };
    Doc::typed(name, fields)
}

/// A classical expression. The expressions an expression is made of are
/// written as a list, which is how a Topiq enumeration holds values of its
/// own type.
fn expr(e: &CExpr) -> Doc {
    let list = |xs: &[&CExpr]| Doc::list(xs, |x| expr(x));
    let op = |o: String| Doc::text(&o);
    match e {
        CExpr::Value(v) => Doc::call("Value", vec![value(v)]),
        CExpr::Bit(bit) => Doc::call("Bit", vec![int(bit.0)]),
        CExpr::Var(v) => Doc::call("Var", vec![int(v.0)]),
        CExpr::Param(i) => Doc::call("Param", vec![int(*i)]),
        CExpr::Unary(o, x) => Doc::call("Unary", vec![op(format!("{o:?}")), list(&[x])]),
        CExpr::Binary(o, x, y) => Doc::call("Binary", vec![op(format!("{o:?}")), list(&[x, y])]),
        CExpr::Logical(o, x, y) => Doc::call("Logical", vec![op(format!("{o:?}")), list(&[x, y])]),
        CExpr::Cast(x, t) => Doc::call("Cast", vec![list(&[x]), Doc::text(t.name())]),
        CExpr::Index(x, i) => Doc::call("Index", vec![list(&[x, i])]),
        CExpr::Field(x, i) => Doc::call("Field", vec![list(&[x]), int(*i)]),
        CExpr::Array(xs) => Doc::call("Array", vec![Doc::list(xs, expr)]),
        CExpr::Struct(xs) => Doc::call("Struct", vec![Doc::list(xs, expr)]),
        CExpr::Variant(k, xs) => Doc::call("Variant", vec![int(*k), Doc::list(xs, expr)]),
        CExpr::Is(x, k) => Doc::call("Is", vec![list(&[x]), int(*k)]),
        CExpr::Payload(x, k) => Doc::call("Payload", vec![list(&[x]), int(*k)]),
        CExpr::Select(c, t, f) => Doc::call("Select", vec![list(&[c, t, f])]),
        CExpr::RegLen(r) => Doc::call("RegLen", vec![int(*r)]),
        CExpr::Len(x) => Doc::call("Len", vec![list(&[x])]),
        CExpr::Append(a, x) => Doc::call("Append", vec![list(&[a, x])]),
    }
}

/// A value known when the circuit was made.
fn value(v: &Value) -> Doc {
    let all = |vs: &[Value]| Doc::list(vs, value);
    match v {
        Value::Int(x, t) => Doc::call("Int", vec![Doc::text(&x.to_string()), Doc::text(t.name())]),
        Value::Float(x, t) => Doc::call("Float", vec![Doc::text(&format!("{:#x}", x.to_bits())), Doc::text(t.name())]),
        Value::Bool(b) => Doc::call("Bool", vec![Doc::Bool(*b)]),
        Value::Char(c) => Doc::call("Char", vec![Doc::Char(*c)]),
        Value::Void => Doc::tag("Void"),
        // a string is only ever a constant's text, which the circuit holds
        // as its characters
        Value::Str(_) => Doc::call("Array", vec![Doc::Array(Vec::new())]),
        Value::Struct(fs) => Doc::call("Struct", vec![all(fs)]),
        Value::Enum { variant, fields } => Doc::call("Variant", vec![int(*variant), all(fields)]),
        Value::Array(items) => Doc::call("Array", vec![all(items)]),
    }
}

/// An exact number at the conductor `n`.
fn cyclo(x: &Cyclo, n: u32) -> Doc {
    let x = x.embed(n).unwrap_or_else(|| x.clone());
    Doc::list(x.coeffs(), |f| Doc::text(&f.to_string()))
}

fn phase48(p: Phase) -> Doc {
    let p = p.embed(48).unwrap_or(p);
    object(vec![("m", int(p.index()))])
}

fn tri(t: Tri) -> Doc {
    Doc::tag(match t {
        Tri::Yes => "Yes",
        Tri::No => "No",
        Tri::Unknown => "Unknown",
    })
}

fn class(c: Option<&Class>) -> Doc {
    let list = |ps: &[Phase]| Doc::list(ps, |&p| phase48(p));
    match c {
        Some(Class::Rigid) => Doc::tag("Rigid"),
        Some(Class::Flat(p)) => Doc::call("Flat", vec![phase48(p.unwrap_or(Phase::zero(48)))]),
        Some(Class::Locflat) => Doc::tag("Locflat"),
        Some(Class::Cyc(p)) => Doc::call("Cyc", vec![list(p.as_deref().unwrap_or(&[]))]),
        Some(Class::Stat(p)) => Doc::call("Stat", vec![phase48(p.unwrap_or(Phase::zero(48)))]),
        Some(Class::Sector(p)) => Doc::call("Sector", vec![list(p.as_deref().unwrap_or(&[]))]),
        Some(Class::Free) => Doc::tag("Free"),
        None => Doc::tag("Unknown"),
    }
}

/// A state in QUON, as the `quon` library reads it back: each amplitude a
/// sum of rational multiples of `w(k, 48)`.
fn quon_text(a: &Amplitudes) -> String {
    let terms: Vec<String> = a
        .terms
        .iter()
        .map(|(bits, c)| {
            let c = c.embed(48).unwrap_or_else(|| c.clone());
            let mut amp = String::new();
            for (k, f) in c.coeffs().iter().enumerate() {
                if f.is_zero() {
                    continue;
                }
                let sign = if f.is_negative() { "-" } else { "+" };
                if amp.is_empty() {
                    if f.is_negative() {
                        amp.push('-');
                    }
                } else {
                    amp.push_str(&format!(" {sign} "));
                }
                amp.push_str(&f.abs().to_string());
                if k > 0 {
                    amp.push_str(&format!(" * w({k}, {})", c.conductor()));
                }
            }
            format!("({amp}) * {}", ket(bits))
        })
        .collect();
    terms.join(" + ")
}

fn certificate(j: &Judged, sectors: &[Judged]) -> Doc {
    let (map, schedule) = match &j.cert {
        Certificate::Points { images, schedule, .. } => {
            (Doc::list(images, |&i| int(i as u64)), Doc::list(schedule, |&p| phase48(p)))
        }
        Certificate::Scalar(v) => (Doc::Array(Vec::new()), Doc::Array(vec![phase48(*v)])),
        Certificate::Moving => (Doc::Array(Vec::new()), Doc::Array(Vec::new())),
    };
    object(vec![
        ("map", map),
        ("schedule", schedule),
        ("sectors", Doc::list(sectors, |s| certificate(s, &[]))),
        ("points", Doc::list(&j.points(), |p| Doc::text(&quon_text(p)))),
    ])
}

/// A record as the `judge` library's `Judgment` holds it; `outcomes`
/// names an instrument's outcome type.
pub fn judgment(r: &Record, outcomes: &str) -> Doc {
    let null = || Doc::tag("null");
    let cert = r.cert.as_ref().map_or_else(null, |j| certificate(j, r.sectors.as_deref().unwrap_or(&[])));
    let kernel = r.kernel.as_ref().map_or_else(null, |k| {
        object(vec![
            ("measured", int(k.measured)),
            ("forgotten", int(k.forgotten)),
            ("outcomes", Doc::text(outcomes)),
        ])
    });
    let fragment = object(vec![
        ("conductor", int(r.fragment.conductor)),
        ("inside", Doc::Bool(r.fragment.departure.is_none())),
        ("stabilizer", Doc::Bool(r.fragment.stabilizer)),
    ]);
    Doc::typed(
        "Judgment",
        vec![
            ("monic", tri(r.monic)),
            ("unitary", tri(r.unitary)),
            ("contract", class(r.contract.as_ref())),
            ("cert", cert),
            ("kernel", kernel),
            ("fragment", fragment),
            ("dynamic", Doc::Bool(r.dynamic)),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::ir::{Bit, Circuit, Op};

    #[test]
    fn a_document_names_its_circuit_and_carries_its_judgment() {
        let c = Circuit {
            name: "f".to_owned(),
            conductor: 8,
            wires: 1,
            bits: 1,
            body: vec![
                Op::Alloc { wire: Wire(0) },
                Op::Gate {
                    gate: GateOp::Phase(Angle::Fixed(Phase::of(8, 2))),
                    targets: vec![Wire(0)],
                    controls: Vec::new(),
                },
                Op::Measure { wire: Wire(0), bit: Bit(0) },
            ],
            result: Some(CExpr::Bit(Bit(0))),
            ..Circuit::default()
        };
        let text = document(
            &c,
            &Heading {
                name: "u::f",
                sig: "fn() -> bool",
                judgment: None,
                endpoints: ("", ""),
            },
        );
        assert!(text.starts_with("Circuit {\n"), "{text}");
        assert!(text.contains("name: \"u::f\""), "{text}");
        assert!(text.contains("gate: Phase(Fixed(2))"), "{text}");
        assert!(text.contains("result: Bit(0)"), "{text}");
        assert!(text.contains("judgment: null"), "{text}");
    }

    #[test]
    fn a_judgment_is_written_as_the_judge_library_holds_it() {
        let r = Record {
            monic: Tri::Yes,
            unitary: Tri::Yes,
            contract: Some(Class::Flat(Some(Phase::of(8, 4)))),
            cert: None,
            sectors: None,
            kernel: None,
            fragment: crate::judge::record::Fragment {
                conductor: 8,
                departure: None,
                stabilizer: true,
            },
            undecided: None,
            dynamic: false,
        };
        let text = write::write(&judgment(&r, ""));
        assert!(text.starts_with("Judgment {"), "{text}");
        // π is 24 forty-eighths of a turn
        assert!(text.contains("contract: Flat(") && text.contains("m: 24"), "{text}");
        assert!(text.contains("cert: null"), "{text}");
        assert!(text.contains("inside: true"), "{text}");
    }
}
