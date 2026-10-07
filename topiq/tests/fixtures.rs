//! Every worked example program must parse clean, and every one that is
//! complete as written must analyse clean too.
//!
//! These are not invented test inputs. Each is a program of the kind Topiq
//! exists to be written in, and between them they use nearly every construct
//! the language has. If one fails, the compiler is wrong, not the fixture.
//!
//! Some are not complete as written: the classical half of the hybrid
//! program names an oracle its quantum half does not declare, and the Grover
//! iterate, the queried table and the locale name operators and a base type
//! they do not define. These are only parsed here; completed, they are
//! analysed and judged in `tests/acceptance`, and the hybrid program is run
//! in `tests/programs/deutsch`.

use std::path::Path;

use topiq::driver::{Options, Session, Stage, compile};

/// Translates a fixture as far as the parser and returns its diagnostics
/// rendered as plain text.
fn check(name: &str) -> (bool, String) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name);
    let mut session = Session::new();
    let id = session
        .load(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let out = compile(&mut session, id, Options::default());

    let mut buf = Vec::new();
    topiq::diag::render::write_all(
        &out.diagnostics,
        session.sources(),
        topiq::diag::render::Style::Plain,
        &mut buf,
    )
    .expect("rendering to a buffer cannot fail");
    (
        !out.has_errors() && out.reached == Some(Stage::Parse),
        String::from_utf8_lossy(&buf).into_owned(),
    )
}

fn assert_parses(name: &str) {
    let (ok, report) = check(name);
    assert!(ok, "{name} should parse clean, but:\n{report}");
}

/// Asserts that a fixture is not only parsed but analysed clean, with the
/// units it imports and the documents it embeds; a quantum one is lowered to
/// circuits and judged too.
fn assert_analyzes(name: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name);
    let loaded = topiq::driver::program::load(
        &[path],
        &[],
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    )
    .unwrap_or_else(|e| panic!("cannot load {name}: {e}"));
    let mut diagnostics: Vec<topiq::diag::Diagnostic> = loaded.diagnostics.clone();
    for m in &loaded.members {
        if let topiq::driver::program::MemberKind::Source(out) = &m.kind {
            diagnostics.extend(out.diagnostics.iter().cloned());
        }
    }
    let mut buf = Vec::new();
    topiq::diag::render::write_all(&diagnostics, loaded.session.sources(), topiq::diag::render::Style::Plain, &mut buf)
        .expect("rendering to a buffer cannot fail");
    assert!(
        diagnostics.iter().all(|d| !d.is_error()),
        "{name} should analyse clean, but:\n{}",
        String::from_utf8_lossy(&buf)
    );
}

#[test]
fn h1_the_hybrid_deutsch_quantum_unit() {
    // exercises `prep` with a ket, an `aux` ancilla, `measure`,
    // a quantum `if`, a function-typed parameter, and three annotations of
    // which one opens a QUON context
    assert_parses("deutsch.tq");
    assert_analyzes("deutsch.tq");
}

#[test]
fn h1_the_hybrid_deutsch_classical_unit() {
    // the classical half: imports, method calls, `unwrap`, and a
    // reference argument. it names `deutsch::ORACLE_Z`, which `deutsch`
    // does not declare; the program test runs it with `gates`' oracles
    assert_parses("main.tq");
}

#[test]
fn h3_the_grover_iterate_typed_over_its_orbit() {
    // exercises cover, gauge and base declarations with states
    // scaled by `/ 2` and a leading `-`, plus a closed `chain`
    assert_parses("grover.tq");
}

#[test]
fn h4_a_coherently_queried_table() {
    // exercises a `qmap` declaration and `query t[k](args)`, where
    // the call applies to the query's result
    assert_parses("table.tq");
}

#[test]
fn h5_data_driven_configuration() {
    // exercises `@embed`, optional types, `match` with tuple-variant
    // patterns, turbofish generics, and `?` error propagation
    assert_parses("config.tq");
    assert_analyzes("config.tq");
}

#[test]
fn h6_a_prefix_hierarchy_without_inheritance() {
    // exercises a `for` loop over a range, indexing, field access,
    // and the checked downcast `s as *Circle` inside a `match`
    assert_parses("shapes.tq");
    assert_analyzes("shapes.tq");
}

#[test]
fn h7_generics_bounded_by_introspection() {
    // exercises a generic parameter, a `where` clause built from
    // introspection macros, and `&&` inside it
    assert_parses("sum.tq");
    assert_analyzes("sum.tq");
}

#[test]
fn the_locale_and_contract_forms() {
    // a locale and both contract-ascription forms: a keyword as a path
    // segment, a path continued after a turbofish, and a `;` continuing an
    // argument list, the three easiest things to get wrong in the grammar
    assert_parses("locale.tq");
}

#[test]
fn every_fixture_in_the_directory_is_covered_by_a_test() {
    // a fixture added without a test would silently never run
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    let mut found: Vec<String> = std::fs::read_dir(&dir)
        .expect("the fixture directory exists")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "tq").then(|| p.file_name()?.to_str().map(str::to_owned))?
        })
        .collect();
    found.sort();

    let mut expected = vec![
        "config.tq".to_owned(),
        "deutsch.tq".to_owned(),
        "grover.tq".to_owned(),
        "locale.tq".to_owned(),
        "main.tq".to_owned(),
        "shapes.tq".to_owned(),
        "sum.tq".to_owned(),
        "table.tq".to_owned(),
    ];
    expected.sort();
    assert_eq!(found, expected, "fixtures and tests have drifted apart");
}
