//! Snapshots of what each phase produces, so a grammar change shows up as a
//! reviewable diff rather than as a silently different tree.
//!
//! The fixture is deliberately dense: it exercises the preprocessor, both
//! disambiguation passes, the judgement-system declarations and the quantum
//! expression forms in one unit. Regenerate with `cargo insta review` after an
//! intended change.

use topiq::driver::{Options, Session, Stage, compile};
use topiq::lex::token::spell;

/// A unit that touches every part of the frontend at least once.
const FIXTURE: &str = "\
#unit quantum(build_tables)
#pragma conductor(24)
#define WIDTH 2
#define REG(N) [qubit; N]

import gates;

cover  B2   = fin{ |00>, |11> };
gauge  GB   = fid((|00> + |11>) * isq2);
base   Bell = REG(WIDTH), B2, GB;

enum Guess { Constant, Balanced }

[entry]
[cover: B2] [gauge: GB]
[expect: monic; contract = flat]
fn bell(r: *REG(WIDTH)) -> Guess {
    let q: REG(WIDTH) = prep |00>;
    aux let a: qubit;
    h(&q[0]);
    for i in 0..WIDTH { cx(&q[0], &q[1]); }
    let bits = measure q;
    if bits[0] && !bits[1] { Guess::Balanced } else { Guess::Constant }
}

chain Loop = bell : Bell -> Bell ;
";

/// Runs the pipeline to `stage` and returns the session and outcome.
fn run(stage: Stage) -> (Session, topiq::driver::Outcome) {
    let mut session = Session::new();
    let id = session.add("snapshot.tq", FIXTURE);
    let out = compile(
        &mut session,
        id,
        Options {
            stage,
            ..Options::default()
        },
    );
    assert!(
        !out.has_errors(),
        "the snapshot fixture should be clean: {:?}",
        out.diagnostics
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
    );
    (session, out)
}

#[test]
fn tokens() {
    let (session, out) = run(Stage::Tokens);
    let text = out
        .tokens
        .iter()
        .map(|(t, _)| spell(*t, session.interner()))
        .collect::<Vec<_>>()
        .join(" ");
    insta::assert_snapshot!(text);
}

#[test]
fn preprocessed() {
    // phase 4's output: directives executed, `REG(WIDTH)` expanded, and the
    // `#unit`, `#pragma` and `#define` lines gone
    let (session, out) = run(Stage::Preprocess);
    let pp = out.preprocessed.as_ref().expect("phase 4 ran");
    let text = pp
        .tokens
        .iter()
        .map(|(t, _)| spell(*t, session.interner()))
        .collect::<Vec<_>>()
        .join(" ");
    insta::assert_snapshot!(format!(
        "conductor {}\nfragment {:?}\nqalloc {:?}\n\n{text}",
        pp.conductor, pp.fragment, pp.qalloc
    ));
}

#[test]
fn marked_brackets() {
    // which `[` opens an annotation group and which an array type. a
    // regression here is otherwise invisible until a whole file misparses
    let (session, out) = run(Stage::Mark);
    let marked = out.marked.as_ref().expect("the marking pass ran");
    let text = marked
        .tokens
        .iter()
        .map(|(t, _)| {
            if t.is(topiq::lex::Punct::LBracketAnnot) {
                "@[".to_owned()
            } else {
                spell(*t, session.interner())
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    insta::assert_snapshot!(format!(
        "{} annotation group(s)\n\n{text}",
        marked.annotation_count()
    ));
}

#[test]
fn syntax_tree() {
    let (session, out) = run(Stage::Parse);
    let unit = out.unit.as_ref().expect("phase 5 ran");
    insta::assert_snapshot!(topiq::ast::print::unit(unit, session.interner()));
}

/// A classical unit exercising analysis and code generation: literal
/// inference, a constant function, a unit-scope object, a persistent one, and
/// each loop form.
const CLASSICAL: &str = "\
#unit classical

let LIMIT: const u32 = 10;
let TOTAL: u64 = 0;

fn square(n: constexpr u32) -> constexpr u32 { n * n }

fn step(i: u32) -> u64 {
    persist let calls: u32 = 0;
    calls += 1;
    if i % 2 == 0 { i as u64 } else { (i * 3) as u64 }
}

fn main() -> i32 {
    let k = 0;
    for i in 0..LIMIT { TOTAL += step(i); }
    while k < square(3) && TOTAL > 0 { k += 1; }
    let bonus = loop { if k > 5 { break k; } k += 1; };
    (TOTAL as i32) + (bonus as i32)
}
";

/// A classical unit with structures, enumerations, arrays, references,
/// strings and `match`.
const AGGREGATES: &str = "\
#unit classical

struct Point { x: i32, y: i32 }
enum Shape { Empty, Circle(i32), Rect { corner: Point, w: i32, h: i32 } }

let ORIGIN: const Point = Point { x: 0, y: 0 };
let NAMES: [*[char]; 2] = [\"circle\", \"rect\"];

fn area(s: *Shape) -> i32 {
    match s {
        Shape::Empty => 0,
        Shape::Circle(r) => 3 * r * r,
        Shape::Rect { w, h } => w * h,
    }
}

fn grow(p: *Point, by: i32) {
    p.x += by;
}

fn main() -> i32 {
    let shapes = [Shape::Circle(2), Shape::Rect { corner: ORIGIN, w: 3, h: 4 }];
    let total = 0;
    for i in 0..shapes.len() { total += area(&shapes[i]); }
    let p = Point { y: 1, x: 2 };
    grow(&p, 5);
    print(NAMES[1]);
    total + p.x
}
";

/// Runs the classical unit to `stage`.
fn run_classical(stage: Stage) -> (Session, topiq::driver::Outcome) {
    run_source("classical.tq", CLASSICAL, stage)
}

/// Runs `text`, named `file`, to `stage`, which it must reach cleanly.
fn run_source(file: &str, text: &str, stage: Stage) -> (Session, topiq::driver::Outcome) {
    let mut session = Session::new();
    let id = session.add(file, text);
    let out = compile(
        &mut session,
        id,
        Options {
            stage,
            ..Options::default()
        },
    );
    assert!(
        !out.has_errors(),
        "the fixture should be clean: {:?}",
        out.diagnostics
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
    );
    (session, out)
}

#[test]
fn typed_unit() {
    // what analysis concluded: every binding's type, every name's target,
    // every constant folded
    let (session, out) = run_classical(Stage::Check);
    insta::assert_snapshot!(topiq::tir::print::unit(
        out.tir().expect("analysis ran"),
        session.interner()
    ));
}

#[cfg(feature = "llvm")]
#[test]
fn llvm_ir() {
    // the unoptimised IR, which follows the source closely: a regression in
    // lowering shows up here as a readable diff
    let (_, out) = run_classical(Stage::Llvm);
    insta::assert_snapshot!(out.llvm_ir.expect("code generation ran"));
}

#[test]
fn typed_aggregates() {
    // declared types, implicit dereferences, coercions and patterns, as
    // analysis writes them out
    let (session, out) = run_source("aggregates.tq", AGGREGATES, Stage::Check);
    insta::assert_snapshot!(topiq::tir::print::unit(
        out.tir().expect("analysis ran"),
        session.interner()
    ));
}

#[cfg(feature = "llvm")]
#[test]
fn llvm_aggregates() {
    // aggregates built in place, byte-offset field access, bounds checks,
    // the discriminant tests of `match`, and string data
    let (_, out) = run_source("aggregates.tq", AGGREGATES, Stage::Llvm);
    insta::assert_snapshot!(out.llvm_ir.expect("code generation ran"));
}

#[test]
fn metadata() {
    // what the unit's object tells importers and the linker
    let (session, out) = run_source("aggregates.tq", AGGREGATES, Stage::Metadata);
    insta::assert_snapshot!(topiq::meta::encode::encode(
        out.metadata.as_ref().expect("analysis succeeded"),
        session.interner()
    ));
}

/// A quantum unit exercising circuit generation: gates, a parameter, a loop
/// unrolled and one left to the circuit, a quantum `if`, feed-forward on a
/// measurement, `lift`, a slot, and a quantum enumeration.
const CIRCUITS: &str = "\
#unit quantum
import gates;

enum QB { Zero(qubit), One(qubit) }

[entry]
fn bell() -> [bool; 2] {
    let r: [qubit; 2] = prep |00>;
    h(&r[0]);
    cx(&r[0], &r[1]);
    measure r
}

[entry]
fn steps(n: usize, k: phase<8>) -> bool {
    let r: [qubit; 3] = prep |000>;
    for i in 0..3 { h(&r[i]); }
    for j in 0..n { rz(k, &r[0]); }
    if &r[0] && !&r[1] { x(&r[2]); } else { z(&r[2]); }
    let m = measure r;
    let again: usize = lift (if m[0] { 2 } else { 1 });
    let s: [qubit; 1] = prep |1>;
    for t in 0..again { h(&s[0]); }
    let last = measure s;
    if m[1] { last[0] } else { false }
}

[entry]
fn query(oracle: fn(*qubit, *qubit)) -> bool {
    let q: [qubit; 2] = prep |01>;
    h(&q[0]);
    h(&q[1]);
    oracle(&q[0], &q[1]);
    h(&q[0]);
    let m = measure q;
    m[0]
}

enum Guess { Constant, Balanced }

[entry]
fn deutsch(oracle: fn(*qubit, *qubit)) -> Guess {
    let q: [qubit; 1] = prep |0>;
    aux let a: qubit;
    x(&a); h(&a);
    h(&q[0]);
    oracle(&q[0], &a);
    h(&q[0]);
    let bits = measure q;
    if bits[0] { Guess::Balanced } else { Guess::Constant }
}

[entry]
fn prepared() -> [bool; 3] {
    let r: [qubit; 3] = prep isq2 * |000> + w(1, 8) * isq2 * |101>;
    measure r
}

fn id1(t: *[qubit; 1]) {}
fn flip1(t: *[qubit; 1]) { z(&t[0]); }
qmap Phases: [qubit; 1] -> fn(*[qubit; 1]) { 0: id1, 1: flip1 }

[entry]
fn keyed() -> [bool; 2] {
    let key: [qubit; 1] = prep |0>;
    let t: [qubit; 1] = prep |1>;
    h(&key[0]);
    query Phases[key](&t);
    h(&key[0]);
    measure (key ** t)
}

[entry]
fn looked_up() -> usize {
    let key: [qubit; 1] = prep |1>;
    let t: [qubit; 1] = prep |0>;
    let (i, f) = measure Phases[key];
    f(&t);
    forget t;
    i
}

[entry]
fn sectors() -> bool {
    let q: qubit;
    let e = QB::One(q);
    match &e {
        QB::Zero(p) => { x(p); }
        QB::One(p) => { z(p); }
    }
    match measure e {
        QB::Zero(p) => { forget p; false }
        QB::One(p) => { forget p; true }
    }
}
";

#[test]
fn circuits() {
    // each entry operator's circuit, as OpenQASM 3
    let (_, out) = run_source("circuits.tq", CIRCUITS, Stage::Check);
    let text: Vec<String> = out
        .circuits
        .iter()
        .filter(|l| l.entry)
        .map(|l| topiq::circuit::qasm::write(&l.circuit))
        .collect();
    insta::assert_snapshot!(text.join("\n"));
}
