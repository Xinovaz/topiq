//! Writing a circuit as OpenQASM 3.
//!
//! Each wire is an element of one register `q`, each bit of one register
//! `c`, and each variable an `int[64]`. A control is a `ctrl @` or `negctrl
//! @` modifier, in the order of the operation's controls, whose qubits come
//! first among the operands. The phase gate `diag(1, e^(iθ))` is `p(θ)`,
//! and an angle of `m` N-th parts of a turn is written `2*pi*m/N`.
//!
//! What OpenQASM 3 cannot say is written as a comment where it happens: an
//! exact matrix applied as a gate, a qubit forgotten, the classical result a
//! caller receives, a register whose length is known only as the circuit
//! runs (its growing, the qubit an index names, and its measurement) and
//! the operator a caller supplies for a slot, which is called as a gate of
//! the slot's name.

use std::fmt::Write as _;

use super::ir::{Angle, CExpr, Circuit, Control, GateOp, Op};
use crate::tir::Value;

/// The circuit as an OpenQASM 3 program.
pub fn write(c: &Circuit) -> String {
    // the header, and what the circuit takes
    let mut out = String::new();
    let _ = writeln!(out, "OPENQASM 3.0;");
    let _ = writeln!(out, "include \"stdgates.inc\";");
    let _ = writeln!(out, "// {}, at conductor {}", c.name, c.conductor);
    if c.dynamic {
        let _ = writeln!(out, "// dynamic: its structure depends on what it measures");
    }
    for (i, p) in c.params.iter().enumerate() {
        let _ = writeln!(out, "input int[64] p{i}; // {}: {}", p.name, p.ty);
    }
    for s in &c.slots {
        let widths: Vec<String> = s.widths.iter().map(u32::to_string).collect();
        let _ = writeln!(
            out,
            "// {}: an operator on qubits of widths ({}) supplied by the caller{}",
            s.name,
            widths.join(", "),
            if s.monic { ", which must be monic" } else { "" }
        );
        for (arg, state) in &s.keeps {
            let amps: Vec<String> = state.iter().map(ToString::to_string).collect();
            let _ = writeln!(
                out,
                "//   and must leave its argument {arg}, given in the state ({}), in that state",
                amps.join(", ")
            );
        }
    }

    // its registers and variables
    if c.wires > 0 {
        let _ = writeln!(out, "qubit[{}] q;", c.wires);
    }
    if c.bits > 0 {
        let _ = writeln!(out, "bit[{}] c;", c.bits);
    }
    for v in 0..c.vars {
        let _ = writeln!(out, "int[64] v{v};");
    }
    if !c.inputs.is_empty() {
        let _ = writeln!(out, "// inputs: {}", wires(&c.inputs));
    }
    for (k, w) in &c.nodes {
        let _ = writeln!(out, "// node {k}: {w}");
    }

    // the body, then what it gives back
    let mut w = Writer {
        out,
        conductor: c.conductor,
        slots: c.slots.iter().map(|s| s.name.clone()).collect(),
    };
    w.ops(&c.body, 0);
    let mut out = w.out;
    if !c.outputs.is_empty() {
        let _ = writeln!(out, "// outputs: {}", wires(&c.outputs));
    }
    if let Some(r) = &c.result {
        let _ = writeln!(out, "// result: {}", expr(r));
    }
    out
}

fn wires(ws: &[super::ir::Wire]) -> String {
    ws.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
}

struct Writer {
    out: String,
    conductor: u32,
    slots: Vec<String>,
}

impl Writer {
    fn line(&mut self, depth: usize, text: &str) {
        let _ = writeln!(self.out, "{}{text}", "    ".repeat(depth));
    }

    fn ops(&mut self, ops: &[Op], depth: usize) {
        for op in ops {
            self.op(op, depth);
        }
    }

    fn op(&mut self, op: &Op, depth: usize) {
        match op {
            Op::Gate { gate, targets, controls } => {
                let name = match gate {
                    GateOp::X => "x".to_owned(),
                    GateOp::Y => "y".to_owned(),
                    GateOp::Z => "z".to_owned(),
                    GateOp::H => "h".to_owned(),
                    GateOp::S => "s".to_owned(),
                    GateOp::Sdg => "sdg".to_owned(),
                    GateOp::T => "t".to_owned(),
                    GateOp::Tdg => "tdg".to_owned(),
                    GateOp::Swap => "swap".to_owned(),
                    GateOp::Phase(a) => format!("p({})", angle(a, self.conductor)),
                    GateOp::GPhase(a) => format!("gphase({})", angle(a, self.conductor)),
                };
                let text = format!("{}{name}{};", modifiers(controls), operands(controls, targets));
                self.line(depth, &text);
            }
            Op::Unitary { matrix, targets, controls } => {
                let rows: Vec<String> = (0..matrix.size)
                    .map(|r| {
                        let row: Vec<String> = (0..matrix.size).map(|c| matrix.at(r, c).to_string()).collect();
                        format!("[{}]", row.join(", "))
                    })
                    .collect();
                let text = format!(
                    "// unitary{} [{}]{};",
                    if controls.is_empty() { String::new() } else { format!(" {}", modifiers(controls).trim_end()) },
                    rows.join(", "),
                    operands(controls, targets)
                );
                self.line(depth, &text);
            }
            Op::Measure { wire, bit } => self.line(depth, &format!("{bit} = measure {wire};")),
            Op::Alloc { .. } | Op::Release { .. } => {}
            Op::Forget { wire } => self.line(depth, &format!("// forget {wire}")),
            Op::If { cond, then, els } => {
                self.line(depth, &format!("if ({}) {{", expr(cond)));
                self.ops(then, depth + 1);
                if els.is_empty() {
                    self.line(depth, "}");
                } else {
                    self.line(depth, "} else {");
                    self.ops(els, depth + 1);
                    self.line(depth, "}");
                }
            }
            Op::For { var, start, end, body } => {
                let text = format!(
                    "for int[64] {var} in [{}:({}) - 1] {{",
                    expr(start),
                    expr(end)
                );
                self.line(depth, &text);
                self.ops(body, depth + 1);
                self.line(depth, "}");
            }
            Op::Let { var, value } => self.line(depth, &format!("{var} = {};", expr(value))),
            Op::Lift { var, value } => {
                self.line(depth, &format!("{var} = {}; // lifted", expr(value)));
            }
            Op::Call { slot, args, controls, adjoint } => {
                let name = self.slots.get(*slot as usize).cloned().unwrap_or_else(|| format!("slot{slot}"));
                let flat: Vec<_> = args.iter().flatten().copied().collect();
                let inv = if *adjoint { "inv @ " } else { "" };
                let text = format!("{inv}{}{name}{};", modifiers(controls), operands(controls, &flat));
                self.line(depth, &text);
            }
            // a register whose length is known only as the circuit runs has
            // no OpenQASM 3 declaration
            Op::Grow { reg, wire } => self.line(depth, &format!("// append {wire} to register r{reg}")),
            Op::Element { wire, reg, index } => {
                self.line(depth, &format!("// {wire} is r{reg}[{}]", expr(index)));
            }
            Op::MeasureAll { reg, var } => self.line(depth, &format!("// {var} = measure r{reg};")),
        }
    }
}

fn modifiers(controls: &[Control]) -> String {
    controls.iter().map(|c| if c.on { "ctrl @ " } else { "negctrl @ " }).collect()
}

fn operands(controls: &[Control], targets: &[super::ir::Wire]) -> String {
    let all: Vec<String> = controls.iter().map(|c| c.wire.to_string()).chain(targets.iter().map(ToString::to_string)).collect();
    if all.is_empty() {
        String::new()
    } else {
        format!(" {}", all.join(", "))
    }
}

fn angle(a: &Angle, conductor: u32) -> String {
    match a {
        Angle::Fixed(p) => {
            if p.is_zero() {
                "0".to_owned()
            } else {
                format!("2*pi*{}/{}", p.index(), p.order())
            }
        }
        Angle::Runtime(e) => format!("2*pi*({})/{conductor}", expr(e)),
    }
}

/// A classical expression.
pub fn expr(e: &CExpr) -> String {
    let sub = expr;
    match e {
        CExpr::Value(v) => value(v),
        CExpr::Bit(b) => b.to_string(),
        CExpr::Var(v) => v.to_string(),
        CExpr::Param(i) => format!("p{i}"),
        CExpr::Unary(op, x) => format!("{}({})", op.text(), sub(x)),
        CExpr::Binary(op, l, r) => format!("({} {} {})", sub(l), op.text(), sub(r)),
        CExpr::Logical(op, l, r) => format!("({} {} {})", sub(l), op.text(), sub(r)),
        CExpr::Cast(x, t) => format!("{}({})", t.name(), sub(x)),
        CExpr::Index(a, i) => format!("{}[{}]", sub(a), sub(i)),
        CExpr::Field(x, f) => format!("{}.{f}", sub(x)),
        CExpr::Array(xs) => format!("{{{}}}", list(xs, sub)),
        CExpr::Struct(xs) => format!("({})", list(xs, sub)),
        CExpr::Variant(v, xs) => variant(*v, &list(xs, sub)),
        CExpr::Is(x, v) => format!("({} is variant{v})", sub(x)),
        CExpr::Payload(x, f) => format!("{}.payload{f}", sub(x)),
        CExpr::Select(c, t, f) => format!("({} ? {} : {})", sub(c), sub(t), sub(f)),
        CExpr::RegLen(r) => format!("sizeof(r{r})"),
        CExpr::Len(x) => format!("sizeof({})", sub(x)),
        CExpr::Append(a, x) => format!("append({}, {})", sub(a), sub(x)),
    }
}

fn value(v: &Value) -> String {
    match v {
        Value::Int(i, _) => i.to_string(),
        Value::Float(x, _) => x.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Char(c) => format!("{}", u32::from(*c)),
        Value::Void => "()".to_owned(),
        Value::Str(_) => "\"…\"".to_owned(),
        Value::Struct(fs) => format!("({})", list(fs, value)),
        Value::Enum { variant: v, fields } => variant(*v, &list(fields, value)),
        Value::Array(xs) => format!("{{{}}}", list(xs, value)),
    }
}

/// Each of `xs` written by `f`, separated by commas.
fn list<T>(xs: &[T], f: impl Fn(&T) -> String) -> String {
    xs.iter().map(f).collect::<Vec<_>>().join(", ")
}

/// A variant of an enumeration.
fn variant(v: u32, fields: &str) -> String {
    if fields.is_empty() { format!("variant{v}") } else { format!("variant{v}({fields})") }
}
