//! Every diagnostic the compiler raises must fire on a program that breaks
//! its rule, and must fire **by name**.
//!
//! Rejecting the program is not enough on its own. A diagnostic identifier is
//! a promise: it means one thing for the life of the language, so a test, a
//! lint or a suppression can be written against it. A check that fired with
//! the wrong identifier, or with none, would break that promise while still
//! looking like it worked.
//!
//! Each test below therefore asserts the identifier, not merely that
//! translation failed.

use topiq::diag::Code;
use topiq::driver::{Options, Outcome, Session, Stage, compile};

/// Translates a source string as far as the parser.
fn run(src: &str) -> Outcome {
    let mut session = Session::new();
    let id = session.add("demo.tq", src);
    compile(&mut session, id, Options::default())
}

/// Asserts that translating `src` reports `code`.
fn reports(src: &str, code: Code) {
    let out = run(src);
    assert!(
        out.reported(code),
        "expected {code}, got {:?}\nsource:\n{src}",
        out.diagnostics
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
    );
}

/// Translates a classical unit's body through analysis. `body` is everything
/// after the `#unit` directive.
fn analyze(body: &str) -> Outcome {
    let mut session = Session::new();
    let id = session.add("demo.tq", &format!("#unit classical\n{body}"));
    compile(
        &mut session,
        id,
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    )
}

/// Asserts that analysing `body` reports exactly `codes`, in order.
fn analyzes_to(body: &str, codes: &[Code]) {
    let out = analyze(body);
    let got: Vec<Code> = out.diagnostics.iter().map(|d| d.code).collect();
    assert_eq!(
        got,
        codes,
        "{:?}\nsource:\n{body}",
        out.diagnostics
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
    );
}

/// Asserts that translating `src` reports nothing at all.
fn clean(src: &str) {
    let out = run(src);
    assert!(
        out.diagnostics.is_empty(),
        "expected no diagnostics, got {:?}\nsource:\n{src}",
        out.diagnostics
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
    );
}

///////////////////
//
// EU01:
//   The unit directive.
//
///////////////////

#[test]
fn eu01_missing_unit_directive() {
    reports("fn main() -> i32 { 0 }\n", Code::Eu01);
}

#[test]
fn eu01_duplicate_unit_directive() {
    reports("#unit classical\n#unit quantum\n", Code::Eu01);
}

#[test]
fn eu01_misplaced_unit_directive() {
    // the directive must come before any item
    reports("fn f() { }\n#unit classical\n", Code::Eu01);
}

#[test]
fn eu01_unit_directive_naming_neither_kind() {
    reports("#unit hybrid\n", Code::Eu01);
}

#[test]
fn a_well_formed_unit_directive_is_clean() {
    clean("#unit classical\nfn main() -> i32 { 0 }\n");
    clean("#unit quantum(build_tables)\nfn build_tables() { }\n");
}

///////////////////
//
// EU04:
//   The conductor.
//
///////////////////

#[test]
fn eu04_a_conductor_that_is_not_a_multiple_of_eight() {
    for n in [1u32, 7, 12, 20] {
        reports(
            &format!("#unit quantum\n#pragma conductor({n})\n"),
            Code::Eu04,
        );
    }
}

#[test]
fn eu04_a_conductor_above_the_implementation_limit() {
    reports("#unit quantum\n#pragma conductor(32)\n", Code::Eu04);
    reports("#unit quantum\n#pragma conductor(64)\n", Code::Eu04);
}

#[test]
fn the_three_conforming_conductors_are_clean() {
    // positive multiples of eight up to the limit of 24: exactly these
    for n in [8u32, 16, 24] {
        clean(&format!("#unit quantum\n#pragma conductor({n})\n"));
    }
}

///////////////////
//
// EA02:
//   A `[` that could open either an annotation or an array.
//
///////////////////

#[test]
fn ea02_a_bracket_in_item_position_that_opens_no_annotation() {
    let out = run("#unit classical\n[1, 2, 3]\nfn f() { }\n");
    assert!(out.reported(Code::Ea02));
    let d = out
        .diagnostics
        .iter()
        .find(|d| d.code == Code::Ea02)
        .unwrap();
    assert_eq!(d.notes.len(), 3, "both readings, and why neither is chosen");
    assert!(!d.helps.is_empty(), "and a way out");
}

#[test]
fn a_real_annotation_does_not_trigger_ea02() {
    clean("#unit quantum\n[entry]\nfn f() { }\n");
    clean("#unit classical\n[packed; align: 8]\nstruct H { tag: u8 }\n");
}

#[test]
fn an_array_expression_in_a_statement_is_not_ea02() {
    // only an item position is refused outright. inside a block a `[` is
    // always an array, so an array-literal statement is read as one
    clean("#unit classical\nfn f() { [1, 2, 3]; }\n");
}

///////////////////
//
// EA03:
//   Unknown and malformed pragmas.
//
///////////////////

#[test]
fn ea03_an_unknown_pragma_warns_and_translation_continues() {
    let out = run("#unit classical\n#pragma nonesuch(1)\nfn main() -> i32 { 0 }\n");
    assert!(out.reported(Code::Ea03));
    assert!(!out.has_errors(), "an unknown pragma is a warning, not an error");
    assert_eq!(
        out.reached,
        Some(Stage::Parse),
        "translation continues past a diagnosed construct"
    );
}

#[test]
fn ea03_a_malformed_pragma_argument_warns_and_is_ignored() {
    let out = run("#unit quantum\n#pragma conductor(\"eight\")\n");
    assert!(out.reported(Code::Ea03));
    assert!(!out.has_errors());
    assert_eq!(
        out.preprocessed.unwrap().conductor,
        8,
        "the pragma is ignored, so the default stands"
    );
}

///////////////////
//
// EA04:
//   The predefined macros.
//
///////////////////

#[test]
fn ea04_redefining_a_predefined_macro() {
    for name in [
        "__TOPIQ__",
        "__FILE__",
        "__LINE__",
        "__UNIT__",
        "__UNIT_KIND__",
        "__CONDUCTOR__",
    ] {
        reports(
            &format!("#unit classical\n#define {name} 1\n"),
            Code::Ea04,
        );
        reports(&format!("#unit classical\n#undef {name}\n"), Code::Ea04);
    }
}

#[test]
fn defining_an_ordinary_macro_is_clean() {
    clean("#unit classical\n#define LIMIT 64\nlet N: const u32 = LIMIT;\n");
}

///////////////////
//
// EM05:
//   The implementation limits.
//
///////////////////

#[test]
fn em05_block_nesting() {
    let src = format!(
        "#unit classical\nfn f() {}{}\n",
        "{".repeat(70),
        "}".repeat(70)
    );
    reports(&src, Code::Em05);
}

#[test]
fn em05_parenthesis_nesting() {
    let src = format!(
        "#unit classical\nfn f() {{ let x = {}1{}; }}\n",
        "(".repeat(70),
        ")".repeat(70)
    );
    reports(&src, Code::Em05);
}

#[test]
fn em05_macro_expansion_depth() {
    let mut src = String::from("#unit classical\n");
    for k in 0..80 {
        src.push_str(&format!("#define M{k} M{}\n", k + 1));
    }
    src.push_str("#define M80 1\nlet N: const u32 = M0;\n");
    reports(&src, Code::Em05);
}

#[test]
fn em05_fields_in_a_structure_and_variants_in_an_enumeration() {
    let fields = |n: usize| (0..n).map(|i| format!("f{i}: u8")).collect::<Vec<_>>().join(", ");
    let variants = |n: usize| (0..n).map(|i| format!("V{i}")).collect::<Vec<_>>().join(", ");
    analyzes_to(&format!("struct S {{ {} }}\n", fields(1024)), &[Code::Em05]);
    analyzes_to(&format!("struct S {{ {} }}\n", fields(1023)), &[]);
    analyzes_to(&format!("enum E {{ {} }}\n", variants(256)), &[Code::Em05]);
    analyzes_to(&format!("enum E {{ {} }}\n", variants(255)), &[]);
}

#[test]
fn constant_recursion_within_the_limit_translates() {
    // each level of a body with a branch takes a good deal of stack while it
    // is evaluated; the whole depth the limit allows must still fit
    let deep = "fn down(n: constexpr i32) -> constexpr i32 { if n == 0 { 0 } else { down(n - 1) } }\n";
    analyzes_to(&format!("{deep}let X: const i32 = down(63);\n"), &[]);
    analyzes_to(&format!("{deep}let X: const i32 = down(64);\n"), &[Code::Em05]);
}

#[test]
fn em05_generic_instantiation_depth() {
    let out = analyze("fn f<T>(x: T) -> i32 { f([x]) }\nfn g() -> i32 { f(1i32) }\n");
    assert!(out.reported(Code::Em05), "{:?}", out.diagnostics);
}

#[test]
fn nesting_within_the_limits_is_clean() {
    let src = format!(
        "#unit classical\nfn f() {{ let x = {}1{}; }}\n",
        "(".repeat(60),
        ")".repeat(60)
    );
    clean(&src);
}

///////////////////
//
// ES02:
//   Text that does not parse.
//
///////////////////

#[test]
fn es02_a_syntax_error() {
    reports("#unit classical\nfn main( { }\n", Code::Es02);
    reports("#unit classical\nstruct S { x }\n", Code::Es02);
    reports("#unit classical\nlet x = ;\n", Code::Es02);
}

#[test]
fn es02_mut_is_not_part_of_the_language() {
    reports("#unit classical\nfn f() { let mut x = 1; }\n", Code::Es02);
    reports("#unit classical\nfn f(p: *mut i32) { }\n", Code::Es02);
}

#[test]
fn es02_an_unterminated_literal() {
    reports("#unit classical\nlet s = \"oops\n", Code::Es02);
}

#[test]
fn es02_a_comparison_chain_is_non_associative() {
    // comparisons are non-associative, so `a < b < c` is ill-formed
    let out = run("#unit classical\nfn f() { let b = a < b < c; }\n");
    assert!(out.has_errors());
    assert!(out.reported(Code::Es02));
}

#[test]
fn es02_include_is_rejected_with_an_explanation() {
    // `#include` is left out on purpose, so the diagnostic should
    // say so rather than merely calling it unknown
    let out = run("#unit classical\n#include \"other.tq\"\n");
    assert!(out.has_errors());
    let d = out
        .diagnostics
        .iter()
        .find(|d| d.code == Code::Es02)
        .unwrap();
    assert!(d.message.contains("no `#include`"), "{}", d.message);
    // and it should point at the two features that replace it
    let helps = d.helps.join(" ");
    assert!(helps.contains("import"), "{helps}");
    assert!(helps.contains("#embed"), "{helps}");
}

///////////////////
//
// TQ003:
//   A construct the compiler does not translate.
//
///////////////////

#[test]
fn tq003_an_unsupported_construct_names_itself() {
    let mut session = Session::new();
    let id = session.add(
        "demo.tq",
        "#unit quantum\nenum E { A(qubit), B(qubit) }\nfn f(e: E) -> bool { let m = measure e; true }\n",
    );
    let out = compile(
        &mut session,
        id,
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    );
    let d = out
        .diagnostics
        .iter()
        .find(|d| d.code == Code::Tq003)
        .expect("reported");
    assert!(d.message.contains("measuring a quantum enumeration outside"), "{}", d.message);
}

///////////////////
//
// ES03:
//   #error and #warning.
//
///////////////////

#[test]
fn es03_error_carries_the_programs_own_text() {
    let out = run("#unit classical\n#error \"this build is unsupported\"\n");
    assert!(out.reported(Code::Es03));
    assert!(out.has_errors());
    assert_eq!(out.diagnostics[0].message, "this build is unsupported");
}

#[test]
fn es03_warning_does_not_stop_translation() {
    let out = run("#unit classical\n#warning \"deprecated\"\nfn main() -> i32 { 0 }\n");
    assert!(out.reported(Code::Es03));
    assert!(!out.has_errors());
}

#[test]
fn a_message_in_an_inactive_branch_does_not_fire() {
    clean("#unit classical\n#if 0\n#error \"boom\"\n#endif\n");
}

////////////////////////////////
// THE IDENTIFIERS THEMSELVES //
////////////////////////////////

#[test]
fn every_identifier_reported_here_is_a_real_one() {
    // a diagnostic must name an identifier the language defines, or one of
    // this implementation's own, marked as such
    for code in [
        Code::Eu01,
        Code::Eu04,
        Code::Ea02,
        Code::Ea03,
        Code::Ea04,
        Code::Em05,
        Code::Es02,
        Code::Es03,
    ] {
        assert!(code.is_language_defined(), "{code} should be defined by the language");
    }
    assert!(!Code::Tq003.is_language_defined(), "TQ003 is this implementation's own and must say so");
}

#[test]
fn a_clean_program_reports_nothing_at_all() {
    clean(
        "#unit classical\n\
         import geometry;\n\
         let VERSION: const u32 = 2;\n\
         static let SALT: const u32 = 0xA5;\n\
         struct Vec3 { x: f64, y: f64, z: f64 }\n\
         enum Ord { Lt, Eq, Gt }\n\
         fn hypot(a: f64, b: f64) -> f64 { a }\n\
         fn main() -> i32 { 0 }\n",
    );
}

///////////////////
//
// Analysis:
//   Names, types and calls.
//
///////////////////

#[test]
fn es04_a_name_that_resolves_to_nothing() {
    analyzes_to("fn main() -> i32 { total }\n", &[Code::Es04]);
    analyzes_to("fn main() -> i32 { g() }\n", &[Code::Es04]);
    analyzes_to("fn main() -> i33 { 0 }\n", &[Code::Es04]);
}

#[test]
fn es04_suggests_the_name_that_was_probably_meant() {
    let out = analyze("fn main() -> i32 { let total = 1; totl }\n");
    let d = &out.diagnostics[0];
    assert!(d.helps.iter().any(|h| h.contains("`total`")), "{:?}", d.helps);
}

#[test]
fn es05_two_declarations_of_one_name() {
    analyzes_to("fn f() { }\nfn f() { }\n", &[Code::Es05]);
    analyzes_to("fn f(a: i32, a: i32) { }\n", &[Code::Es05]);
}

#[test]
fn es06_a_value_of_the_wrong_type() {
    analyzes_to("fn main() -> i32 { true }\n", &[Code::Es06]);
    analyzes_to("fn f(a: i32, b: i64) -> i32 { a + b }\n", &[Code::Es06]);
    analyzes_to("fn f() -> u8 { 256 }\n", &[Code::Es06]);
}

#[test]
fn es06_a_body_ending_without_its_value_explains_the_semicolon() {
    let out = analyze("fn f() -> i32 { 5; }\n");
    let d = &out.diagnostics[0];
    assert_eq!(d.code, Code::Es06);
    assert!(d.notes.iter().any(|n| n.contains("semicolon")), "{:?}", d.notes);
}

#[test]
fn es07_a_call_that_does_not_fit() {
    analyzes_to("fn g(a: i32) { }\nfn f() { g(); }\n", &[Code::Es07]);
    analyzes_to("fn f(x: i32) { x(1); }\n", &[Code::Es07]);
}

#[test]
fn es24_measure_of_classical_data_warns_and_gives_it_back() {
    // the shape is what measuring the quantum data would give
    let body = "struct In { a: bool, t: u8 }\nstruct Out { i: In, b: bool }\nenum E { A, B }\n\
                fn f(o: Out, e: E) -> ((bool, u8), bool, u8, [bool; 2], u32) {\n\
                let m: ((bool, u8), bool) = measure o;\n\
                let k = match measure e { E::A => 1u8, E::B => 2 };\n\
                (m.0, m.1, k, measure [true, false], measure 3u32)\n\
                }\n";
    analyzes_to(body, &[Code::Es24, Code::Es24, Code::Es24, Code::Es24]);
    // what nothing quantum measures to
    analyzes_to("fn f() -> f64 { measure 1.5 }\n", &[Code::Es06]);
    analyzes_to("fn f() -> i32 { measure -1i32 }\n", &[Code::Es06]);
    analyzes_to("enum E { A }\nfn f() -> E { measure E::A }\n", &[Code::Es06]);
    // and in a quantum unit, a measured value measured again
    assert_eq!(quantum_codes("fn f(q: qubit) -> bool { let m = measure q; measure m }\n"), [Code::Es24]);
}

#[test]
fn es08_assignment_to_a_value_that_is_not_a_place() {
    analyzes_to("fn g() -> i32 { 1 }\nfn f() { g() = 2; }\n", &[Code::Es08]);
}

#[test]
fn increments_step_integer_places() {
    analyzes_to("fn f(p: *u8) -> i32 { let x = 1; x++; --x; (*p)++; let a = [1, 2]; a[0]--; ++a[1] }\n", &[]);
    // `--` is one token: on a value it is not a double negation
    analyzes_to("fn f() -> i32 { --5 }\n", &[Code::Es08]);
    analyzes_to("fn f() -> i32 { let x = 5; -(-x) }\n", &[]);
    analyzes_to("fn f() { let x: const i32 = 1; x++; }\n", &[Code::Ec03]);
}

#[test]
fn es22_a_step_of_something_that_is_not_an_integer() {
    analyzes_to("fn f() { let x = 1.5; x++; }\n", &[Code::Es22]);
    analyzes_to("fn f() { let b = true; --b; }\n", &[Code::Es22]);
}

#[test]
fn a_reference_reads_and_writes_what_it_refers_to() {
    analyzes_to("fn f(p: *i32) { *p = 2; }\n", &[]);
    analyzes_to("fn f() { let x = 1; let p = &x; *p = 2; let q: *const i32 = p; }\n", &[]);
}

#[test]
fn text_is_a_string_as_core_spells_it() {
    analyzes_to("fn f(s: string) { s[0] = 'x'; }\nfn g() { f(\"hi\"); let t: core::string = \"t\"; f(t); }\n", &[]);
    analyzes_to("type string = i32;\nfn f(s: string) -> i32 { s + 1 }\n", &[]);
}

#[test]
fn a_units_own_alias_hides_a_core_type_of_its_name() {
    analyzes_to("type frac = i32;\nfn f(x: frac) -> i32 { x + 1 }\n", &[]);
    // a type declared in a block hides the alias in turn
    analyzes_to(
        "type frac = i32;\nfn f() -> bool { struct frac { n: u8 } let a = frac { n: 1 }; a.n == 1 }\n",
        &[],
    );
}

#[test]
fn text_becomes_a_string_where_one_is_wanted() {
    analyzes_to(
        "fn f(s: string) -> usize { s.len() }\n\
         fn g(a: *[char], b: [char]) -> usize {\n\
             let s: string = \"lit\";\n\
             f(\"x\") + f(a) + f(b) + s.len()\n\
         }\n",
        &[],
    );
}

#[test]
fn a_strings_operators_take_text_on_either_side() {
    analyzes_to(
        "fn f(s: string) -> bool {\n\
             let t = s + \"!\";\n\
             let u = \"<\" + t;\n\
             u += \">\";\n\
             s == \"x\" || \"x\" != s || s < \"y\" || u >= s\n\
         }\n",
        &[],
    );
}

#[test]
fn two_slices_of_text_still_compare_as_references() {
    analyzes_to("fn f(a: *[char], b: *[char]) -> bool { a == b }\n", &[]);
    // and slices have no `+`
    analyzes_to("fn f(a: *[char], b: *[char]) { let c = a + b; }\n", &[Code::Es06]);
}

#[test]
fn a_string_lends_its_characters_as_a_slice() {
    // a reference to one is a view of its characters, without a copy
    analyzes_to("fn f(s: string) -> bool { str::eq(&s, \"x\") && str::eq(&s.chars, \"x\") }\n", &[]);
    analyzes_to("fn f(s: *const string) { print(s); }\n", &[]);
    // but not one that writes through a `*const string`
    analyzes_to("fn g(t: *[char]) { }\nfn f(s: *const string) { g(s); }\n", &[Code::Es06]);
    // the string itself is not a slice
    analyzes_to("fn f(s: string) -> bool { str::eq(s, \"x\") }\n", &[Code::Es06]);
}

#[test]
fn text_becomes_a_string_a_const_reference_refers_to() {
    analyzes_to(
        "fn n(s: *const string) -> usize { s.len() }\n\
         fn f(a: *[char]) -> usize { n(\"lit\") + n(a) }\n",
        &[],
    );
    // a `*string` could change it, and would change only the copy
    analyzes_to("fn n(s: *string) { }\nfn f() { n(\"lit\"); }\n", &[Code::Es06]);
}

#[test]
fn a_tuple_written_out_takes_each_element_as_wanted() {
    analyzes_to("fn f() { let xs: [(string, i32)] = []; xs.push((\"a\", 1)); let t: (string, u8) = (\"b\", 2); }\n", &[]);
}

#[test]
fn enumerations_whose_variants_carry_nothing_compare_by_variant() {
    analyzes_to(
        "enum C { R, G, B }\n\
         fn f(a: C, b: *const C) -> bool { a == C::R || *b != a || C::G == C::B }\n",
        &[],
    );
    analyzes_to("fn f(k: TypeKind) -> bool { k == TypeKind::Enum }\n", &[]);
    // a variant that carries something leaves `==` to `$eq`
    analyzes_to("enum E { A(i32), B }\nfn f(a: E) -> bool { a == E::B }\n", &[Code::Es06]);
    // and so does an enumeration's own `$eq`
    analyzes_to(
        "enum C { R, G }\nimpl C { fn $eq(self: *const C, o: *const C) -> bool { true } }\nfn f() -> bool { C::R == C::G }\n",
        &[],
    );
    // the two sides are of one type
    analyzes_to("enum C { R }\nenum D { S }\nfn f() -> bool { C::R == D::S }\n", &[Code::Es06]);
}

#[test]
fn every_binding_but_a_constant_may_be_assigned() {
    analyzes_to("fn f(y: i32) { let x = 1; x = 2; y = x; }\n", &[]);
    analyzes_to("fn f() { let x: const i32 = 1; x = 2; }\n", &[Code::Ec03]);
}

#[test]
fn es09_break_or_continue_outside_a_loop_that_accepts_it() {
    analyzes_to("fn f() { break; }\n", &[Code::Es09]);
    analyzes_to("fn f() { while true { break 1; } }\n", &[Code::Es09]);
}

///////////////////
//
// Analysis:
//   Constants.
//
///////////////////

#[test]
fn ec01_a_constexpr_whose_value_is_only_known_at_run_time() {
    analyzes_to("fn f(x: i32) { let k: constexpr i32 = x; }\n", &[Code::Ec01]);
    analyzes_to("let A: i32 = 1;\nlet B: const i32 = A;\n", &[Code::Ec01]);
}

#[test]
fn a_const_binding_is_folded_when_it_can_be_and_kept_when_it_cannot() {
    // `k` is only known at run time, so it stays a binding that is never
    // assigned; `j` is computed now
    analyzes_to("fn f(x: i32) -> i32 { let k: const i32 = x; let j: const i32 = 2 * 3; k + j }\n", &[]);
    analyzes_to("fn f(x: i32) { let k: const i32 = x; k = 2; }\n", &[Code::Ec03]);
    analyzes_to("fn f(x: i32) { let (a, b): const (i32, i32) = (x, 1); a = 2; }\n", &[Code::Ec03]);
}

#[test]
fn a_const_binding_may_hold_a_measured_outcome() {
    // only known while the circuit runs: kept as a binding, never assigned
    let body = |ty: &str| {
        format!(
            "#unit quantum\nimport gates;\n[entry]\nfn f(n: u32) -> bool {{\n\
             let q: [qubit; 2] = prep |00>;\nh(&q[0]);\n\
             let bits: {ty} [bool; 2] = measure q;\nbits[0] && !bits[1]\n}}\n"
        )
    };
    assert_eq!(unit_codes(&body("const")), []);
    assert_eq!(unit_codes(&body("constexpr")), [Code::Ec01]);
}

#[test]
fn a_const_parameter_is_never_assigned_and_a_constexpr_one_is_known_now() {
    analyzes_to("fn f(x: const i32) -> i32 { x + 1 }\nfn g(y: i32) -> i32 { f(y) }\n", &[]);
    analyzes_to("fn f(x: const i32) { x = 2; }\n", &[Code::Ec03]);
    analyzes_to("fn f(n: constexpr i32, x: i32) -> i32 { n * x }\nfn g(y: i32) -> i32 { f(3, y) }\n", &[]);
}

#[test]
fn es06_misplaced_const_and_constexpr() {
    analyzes_to("fn f() -> const i32 { 1 }\n", &[Code::Es06]);
    analyzes_to("fn f(p: *constexpr i32) { }\n", &[Code::Es06]);
    analyzes_to("fn f() { let g = fn(x: constexpr i32) -> i32 with () { x }; }\n", &[Code::Es06]);
    analyzes_to("fn f() { let (a, b): constexpr (i32, i32) = (1, 2); }\n", &[Code::Es06]);
}

#[test]
fn a_constant_is_lent_only_to_what_promises_to_read_it() {
    let k = "let K: const i32 = 1;\n";
    analyzes_to(&format!("{k}fn read<T>(x: *const T) -> T {{ *x }}\nfn f() -> i32 {{ read(&K) }}\n"), &[]);
    analyzes_to(&format!("{k}fn write<T>(x: *T, v: T) {{ *x = v; }}\nfn f() {{ write(&K, 2); }}\n"), &[Code::Es06]);
}

#[test]
fn a_constant_is_indexed_by_index_rd() {
    let read = "fn read() -> i64 { let t = 0; for i in 0..3 { t += i; } t }\n";
    // a new constant's value, and any read of a constant, use `$index_rd`;
    // a change uses `$index`
    analyzes_to(
        &format!(
            "{read}fn f() -> i64 {{ let v: vec<i64, 2> = vec {{ e: [read(), 2] }}; let a: const i64 = v[0]; v[1] = 3; \
             let k: const mat<i64, 2, 2> = mat {{ e: [[read(), 0], [0, 1]] }}; let d: const i64 = k[1][1]; a + d + k[0][0] }}\n"
        ),
        &[],
    );
    analyzes_to(
        &format!("{read}fn f() {{ let k: const vec<i64, 2> = vec {{ e: [read(), 4] }}; k[0] = 1; }}\n"),
        &[Code::Ec03],
    );
    let w = "struct W { x: i64 }\nimpl W { fn $index(self: *W, i: usize) -> *i64 { &self.x } }\n";
    analyzes_to(&format!("{read}{w}fn f() -> i64 {{ let w: const W = W {{ x: read() }}; w[0] }}\n"), &[Code::Es06]);
    let bad = "struct B { x: i64 }\nimpl B { fn $index_rd(self: *B, i: usize) -> *i64 { &self.x } }\n";
    analyzes_to(&format!("{bad}fn f() -> i64 {{ let b = B {{ x: 1 }}; let z: const i64 = b[0]; z }}\n"), &[Code::Es06]);
}

#[test]
fn a_method_that_may_write_cannot_be_called_on_a_constant() {
    let p = "struct P { x: i32 }\nimpl P { fn set(self: *P) { self.x = 1; } fn get(self: *const P) -> i32 { self.x } }\n";
    analyzes_to(&format!("{p}fn f(x: i32) -> i32 {{ let p: const P = P {{ x: x }}; p.get() }}\n"), &[]);
    analyzes_to(&format!("{p}fn f(x: i32) {{ let p: const P = P {{ x: x }}; p.set(); }}\n"), &[Code::Es06]);
}

#[test]
fn ec02_a_run_time_argument_for_a_const_parameter() {
    analyzes_to(
        "fn sq(n: constexpr i32) -> constexpr i32 { n * n }\nfn f(x: i32) -> i32 { sq(x) }\n",
        &[Code::Ec02],
    );
}

#[test]
fn ec03_assignment_to_a_constant() {
    analyzes_to("let X: const i32 = 1;\nfn f() { X = 2; }\n", &[Code::Ec03]);
}

#[test]
fn ec04_a_constant_that_would_abort_names_the_abort() {
    let out = analyze("let X: const i32 = 1 / 0;\n");
    let d = out.diagnostics.iter().find(|d| d.code == Code::Ec04).expect("reported");
    assert!(d.message.contains("RA02"), "{}", d.message);
}

#[test]
fn ec05_a_unit_scope_object_never_given_a_value() {
    analyzes_to("let X: i32;\n", &[Code::Ec05]);
}

/// The codes analysing a whole unit reports.
fn unit_codes(src: &str) -> Vec<Code> {
    let mut session = Session::new();
    let id = session.add("demo.tq", src);
    let out = compile(
        &mut session,
        id,
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    );
    out.diagnostics.iter().map(|d| d.code).collect()
}

#[test]
fn an_initializer_gives_each_object_its_one_value() {
    let ok = "#unit classical(setup)\nlet X: i32;\nlet XS: [u8];\nfn setup() { X = 3; XS = [1, 2]; XS.push(X as u8); }\n";
    assert_eq!(unit_codes(ok), []);
}

#[test]
fn eu05_an_initializer_that_is_not_a_function_taking_nothing() {
    assert_eq!(unit_codes("#unit classical(setup)\nlet X: i32;\n"), [Code::Eu05]);
    assert_eq!(
        unit_codes("#unit classical(setup)\nlet X: i32 = 1;\nfn setup(n: i32) { }\n"),
        [Code::Eu05]
    );
}

#[test]
fn eu06_calling_the_initializer() {
    assert_eq!(
        unit_codes("#unit classical(setup)\nlet X: i32;\nfn setup() { X = 1; }\nfn again() { setup(); }\n"),
        [Code::Eu06]
    );
}

#[test]
fn ec06_an_object_the_initializer_does_not_assign_exactly_once() {
    let unit = |body: &str| format!("#unit classical(setup)\nlet X: i32;\nfn setup() {{ {body} }}\n");
    assert_eq!(unit_codes(&unit("")), [Code::Ec06], "never");
    assert_eq!(unit_codes(&unit("if true { X = 1; }")), [Code::Ec06], "on some paths");
    assert_eq!(unit_codes(&unit("X = 1; X = 2;")), [Code::Ec06], "twice");
    assert_eq!(unit_codes(&unit("while false { X = 1; }")), [Code::Ec06], "in a loop");
    assert_eq!(unit_codes(&unit("let y = X; X = y;")), [Code::Ec06], "read first");
    assert_eq!(unit_codes(&unit("if true { X = 1; } else { X = 2; }")), [], "once on each path");
    assert_eq!(unit_codes(&unit("X = 1; X += 1;")), [], "changed after its value");
}

#[test]
fn em05_constant_recursion_past_the_limit() {
    analyzes_to(
        "fn r(n: constexpr u32) -> constexpr u32 { r(n + 1) }\nlet X: const u32 = r(0);\n",
        &[Code::Em05],
    );
}

///////////////////
//
// Analysis:
//   Unit structure and annotations.
//
///////////////////

#[test]
fn eu02_quantum_storage_in_a_classical_unit() {
    analyzes_to("fn f(q: qubit) { }\n", &[Code::Eu02]);
    analyzes_to("fn f() { aux let a: i32 = 0; }\n", &[Code::Eu02]);
    // measuring classical data is a warning (ES24), and an `i64` stands for no quantum data
    analyzes_to("fn f() -> i64 { measure 1 }\n", &[Code::Es06]);
    analyzes_to("fn f() { forget 1; }\n", &[Code::Eu02]);
    // an operator's adjoint undoes it, which only a quantum unit can
    analyzes_to("fn g(x: i32) -> i32 { x }\nfn f() { let h = adjoint(g); }\n", &[Code::Eu02]);
}

///////////////////
//
// Analysis:
//   Quantum units.
//
///////////////////

/// The codes analysing a quantum unit's `body` reports.
fn quantum_codes(body: &str) -> Vec<Code> {
    unit_codes(&format!("#unit quantum\n{body}"))
}

#[test]
fn a_quantum_unit_holds_measures_and_joins_registers() {
    let ok = "fn f(q: qubit) -> bool { measure q }\n\
              fn g() -> [bool; 3] { let a: [qubit; 1] = prep |0>; let b: [qubit; 2] = prep |01>; measure (a ** b) }\n\
              fn h() { aux let t: qubit; let fresh: [qubit; 4]; }\n\
              struct Pair { a: qubit, n: u32 }\n\
              fn pair(p: Pair) { forget p; }\n";
    assert_eq!(quantum_codes(ok), []);
}

#[test]
fn eq01_a_quantum_value_dropped_while_possibly_live() {
    // a parameter, a prepared register, one left on an early return, one
    // assigned over, and one made and kept by nothing
    assert_eq!(quantum_codes("fn f(q: qubit) { }\n"), [Code::Eq01]);
    assert_eq!(quantum_codes("fn f() { let q: [qubit; 1] = prep |1>; }\n"), [Code::Eq01]);
    assert_eq!(
        quantum_codes("fn f(q: qubit, c: bool) -> bool { if c { return true; } measure q }\n"),
        [Code::Eq01]
    );
    assert_eq!(
        quantum_codes("fn f() { let q: [qubit; 1] = prep |0>; q = prep |1>; forget q; }\n"),
        [Code::Eq01]
    );
    assert_eq!(quantum_codes("fn f() { prep |0>; }\n"), [Code::Eq01]);
    // allocated and never changed is still |0>; an ancilla is uncomputed
    assert_eq!(quantum_codes("fn peek(q: *qubit) { }\nfn f() { let q: qubit; aux let a: qubit; peek(&a); peek(&q); }\n"), []);
    // allocated and changed is |0> again only where the change is undone
    assert_eq!(quantum_codes("import gates;\nfn f() { let q: qubit; h(&q); }\n"), [Code::Eq01]);
    assert_eq!(quantum_codes("import gates;\nfn f() { let q: qubit; t(&q); h(&q); h(&q); tdag(&q); }\n"), []);
}

#[test]
fn eq01_says_what_to_do_instead() {
    let mut session = Session::new();
    let id = session.add("demo.tq", "#unit quantum\nfn f(q: qubit) { }\n");
    let out = compile(&mut session, id, Options { stage: Stage::Check, ..Options::default() });
    let d = out.diagnostics.iter().find(|d| d.code == Code::Eq01).expect("reported");
    assert!(d.helps.join(" ").contains("forget"), "{:?}", d.helps);
}

#[test]
fn eq06_an_outcome_value_shaping_the_circuit() {
    let measured = "fn use_it(b: bool) { }\nlet SEEN: bool = false;\n";
    let body = |s: &str| format!("{measured}fn f() {{ let q: [qubit; 2] = prep |01>; let bits = measure q; {s} }}\n");
    assert_eq!(quantum_codes(&body("use_it(bits[0]);")), [Code::Eq06]);
    assert_eq!(quantum_codes(&body("SEEN = bits[1];")), [Code::Eq06]);
    assert_eq!(
        quantum_codes(&body("let n: usize = 0; if bits[0] { n = 1; } for i in 0..n { }")),
        [Code::Eq06]
    );
    assert_eq!(quantum_codes(&body("let n: usize = 0; if bits[0] { n = 1; } let t = [1, 2]; let x = t[n];")), [Code::Eq06]);
    // an array of outcomes is as long as what was measured, or as the
    // rounds that pushed onto it, unless it grew under a test of one
    assert_eq!(
        quantum_codes(&body("for i in 0..bits.len() { } let s: [bool] = []; s.push(bits[0]); for i in 0..s.len() { }")),
        []
    );
    assert_eq!(
        quantum_codes(&body("let s: [bool] = []; if bits[0] { s.push(true); } for i in 0..s.len() { }")),
        [Code::Eq06]
    );
    // feed-forward, combination and lifting are what an outcome is for
    assert_eq!(
        quantum_codes(&body("let both = bits[0] && bits[1]; if both { } let k: usize = if bits[0] { 1 } else { 2 }; let n = lift k; for i in 0..n { }")),
        []
    );
}

#[test]
fn eq11_a_quantum_unit_acting_on_the_programs_surroundings() {
    assert_eq!(quantum_codes("fn f() { print(\"hello\"); }\n"), [Code::Eq11]);
}

#[test]
fn ec01_a_constant_holding_qubits() {
    assert_eq!(quantum_codes("fn f(q: constexpr qubit) { }\n"), [Code::Ec01]);
}

#[test]
fn ec09_clone_of_a_quantum_value() {
    assert_eq!(quantum_codes("fn f(q: *qubit) { let r = clone(q); forget r; }\n"), [Code::Ec09]);
}

#[test]
fn em05_a_register_over_the_qubit_limit() {
    assert_eq!(quantum_codes("fn f(r: *[qubit; 4097]) { }\n"), [Code::Em05]);
    assert_eq!(quantum_codes("fn f(r: *[qubit; 4096]) { }\n"), []);
}

#[test]
fn quantum_operations_take_quantum_operands() {
    // classical data measured gives itself back, with a warning
    assert_eq!(quantum_codes("fn f() -> bool { measure true }\n"), [Code::Es24]);
    assert_eq!(quantum_codes("fn f() { forget 3; }\n"), [Code::Es06]);
    assert_eq!(quantum_codes("fn f() { aux let a: u32; }\n"), [Code::Es06]);
    assert_eq!(quantum_codes("fn f() { let n = lift 3; }\n"), [Code::Es06]);
    assert_eq!(quantum_codes("fn f() { let r: [qubit; 2] = prep |0> + |11>; forget r; }\n"), [Code::Es06]);
}

#[test]
fn a_closure_in_a_quantum_unit_captures_no_qubits() {
    assert_eq!(
        quantum_codes("fn f(q: qubit) { let c = fn() with (q) { }; }\n"),
        [Code::Es06]
    );
}

#[test]
fn quantum_annotations_describe_quantum_operators() {
    analyzes_to("[entry]\nfn f() { }\n", &[Code::Es18]);
    assert_eq!(quantum_codes("[entry]\nfn f() { }\n"), []);
    assert_eq!(quantum_codes("[entry]\nfn f<T>(x: T) { }\n"), [Code::Es18]);
}

#[test]
fn eu03_a_quantum_unit_using_a_classical_units_code() {
    let dir = std::env::temp_dir().join(format!("topiq-eu03-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("helpers.tq"),
        "#unit classical\nlet N: const u32 = 2;\nlet COUNT: u32 = 0;\nfn twice(n: constexpr u32) -> u32 { n * 2 }\nfn runtime(n: u32) -> u32 { n }\n",
    )
    .unwrap();
    let main = dir.join("q.tq");
    std::fs::write(
        &main,
        "#unit quantum\nimport helpers;\n\
         fn a() -> u32 { helpers::twice(helpers::N) }\n\
         fn b() -> u32 { helpers::runtime(1) }\n\
         fn c() -> u32 { helpers::COUNT }\n",
    )
    .unwrap();
    let loaded = topiq::driver::program::load(
        &[main],
        &[],
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    )
    .unwrap();
    let mut codes: Vec<Code> = loaded.diagnostics.iter().map(|d| d.code).collect();
    for m in &loaded.members {
        if let topiq::driver::program::MemberKind::Source(out) = &m.kind {
            codes.extend(out.diagnostics.iter().filter(|d| d.is_error()).map(|d| d.code));
        }
    }
    assert_eq!(codes, [Code::Eu03, Code::Eu03]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn el01_static_where_it_restricts_nothing() {
    // `core` is always available, so only the misplaced `static` is wrong
    analyzes_to("static import core;\n", &[Code::El01]);
}

#[test]
fn ea01_an_unknown_annotation_is_a_warning_and_analysis_continues() {
    let out = analyze("[my_tool]\nfn main() -> i32 { 0 }\n");
    assert!(out.reported(Code::Ea01));
    assert!(!out.has_errors());
    assert!(out.tir().is_some());
}

#[test]
fn an_analyzed_program_reports_nothing() {
    analyzes_to(
        "let LIMIT: const u32 = 10;\n\
         fn sq(n: constexpr u32) -> constexpr u32 { n * n }\n\
         fn main() -> i32 {\n\
             let total = 0;\n\
             for i in 0..LIMIT { total += i; }\n\
             (total + sq(2)) as i32\n\
         }\n",
        &[],
    );
}

/////////////////////////////////////////////////////
// STRUCTURES, ENUMERATIONS, ARRAYS AND REFERENCES //
/////////////////////////////////////////////////////

const SHAPES: &str = "struct Point { x: i32, y: i32 }\n\
                      enum Shape { Empty, Circle(i32), Rect { w: i32, h: i32 } }\n\
                      fn take(p: Point) { }\n";

#[test]
fn a_program_using_every_new_form_reports_nothing() {
    analyzes_to(
        &format!(
            "{SHAPES}\
             fn area(s: *Shape) -> i32 {{\n\
                 match s {{ Shape::Empty => 0, Shape::Circle(r) => 3 * r * r, Shape::Rect {{ w, h }} => w * h }}\n\
             }}\n\
             fn main() -> i32 {{\n\
                 let ps = [Point {{ x: 1, y: 2 }}, Point {{ x: 3, y: 4 }}];\n\
                 ps[1].y = 9;\n\
                 let r: *[Point] = &ps;\n\
                 let s = Shape::Rect {{ w: r[0].x, h: r.len() as i32 }};\n\
                 print(\"ok\\n\");\n\
                 area(&s) + ps[1].y\n\
             }}\n"
        ),
        &[],
    );
}

#[test]
fn es10_a_structure_used_after_it_was_moved() {
    analyzes_to(&format!("{SHAPES}fn f(p: Point) {{ take(p); take(p); }}\n"), &[Code::Es10]);
}

#[test]
fn es11_a_binding_read_before_it_has_a_value() {
    analyzes_to("fn f(c: bool) -> i32 { let x: i32; if c { x = 1; } x }\n", &[Code::Es11]);
}

#[test]
fn es12_a_structure_moved_out_of_an_array() {
    analyzes_to(&format!("{SHAPES}fn f(a: [Point; 2]) {{ take(a[0]); }}\n"), &[Code::Es12]);
}

#[test]
fn es13_a_match_that_misses_a_variant_names_it() {
    let out = analyze(&format!(
        "{SHAPES}fn f(s: Shape) -> i32 {{ match s {{ Shape::Empty => 0, Shape::Circle(r) => r }} }}\n"
    ));
    let d = out.diagnostics.iter().find(|d| d.code == Code::Es13).expect("reported");
    assert!(d.message.contains("Shape::Rect { w: _, h: _ }"), "{}", d.message);
}

#[test]
fn es14_an_unreachable_arm_is_a_warning() {
    let out = analyze("fn f(x: i32) -> i32 { match x { _ => 0, 1 => 1 } }\n");
    assert!(out.reported(Code::Es14));
    assert!(!out.has_errors());
}

#[test]
fn es15_a_structure_literal_missing_a_field() {
    analyzes_to(&format!("{SHAPES}fn f() -> Point {{ Point {{ x: 1 }} }}\n"), &[Code::Es15]);
}

#[test]
fn es16_a_structure_that_contains_itself() {
    analyzes_to("struct Node { value: i32, next: Node }\n", &[Code::Es16]);
}

#[test]
fn es17_an_import_of_a_unit_that_cannot_be_found() {
    analyzes_to("import nowhere;\n", &[Code::Es17]);
}

#[test]
fn es18_a_layout_annotation_on_a_function() {
    analyzes_to("[align: 8]\nfn f() { }\n", &[Code::Es18]);
}

#[test]
fn es20_a_name_the_library_reserves() {
    analyzes_to("fn __helper() { }\n", &[Code::Es20]);
    analyzes_to("fn f() { let __x = 1; }\n", &[Code::Es20]);
    let mut session = Session::new();
    let id = session.add("time.tq", "#unit classical\nfn f() { }\n");
    let out = compile(
        &mut session,
        id,
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    );
    assert!(out.reported(Code::Es20), "a unit named after a reserved library unit");
}

#[test]
fn a_library_unit_is_reached_without_an_import() {
    clean("#unit classical\nfn f(x: i64) -> i64? { if x > 0 { Some(x) } else { None } }\nfn g() -> i64 { f(3).unwrap_or(0) + core::wrapping_add(1, 2) }\n");
}

#[test]
fn ep01_a_reference_to_a_type_that_does_not_begin_with_the_other() {
    let types = "struct Shape { kind: u8, area: f64 }\nstruct Circle { kind: u8, area: f64, r: f64 }\n\
                 struct Wrong { kind: u8, size: f64 }\n";
    analyzes_to(&format!("{types}fn f(c: *Circle) -> *Shape {{ c }}\n"), &[]);
    analyzes_to(&format!("{types}fn f(c: *Circle) -> *Wrong {{ c }}\n"), &[Code::Ep01]);
    // going the other way is a downcast, which is written and checked
    let out = analyze(&format!("{types}fn f(s: *Shape) -> *Circle {{ s }}\n"));
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, Code::Es06);
    assert!(format!("{:?}", out.diagnostics[0]).contains("write `as *Circle`"), "the help suggests the downcast");
    analyzes_to(&format!("{types}fn f(s: *Shape) -> Opt<*Circle> {{ s as *Circle }}\n"), &[]);
}

#[test]
fn em03_run_time_type_information_for_a_type_that_refuses_it() {
    let hidden = "[no_typeinfo]\nstruct Secret { key: u64 }\n";
    analyzes_to(&format!("{hidden}fn f() -> usize {{ @sizeof(Secret) }}\n"), &[]);
    analyzes_to(&format!("{hidden}fn f() -> *const TypeInfo {{ @typeinfo(Secret) }}\n"), &[Code::Em03]);
    analyzes_to(&format!("{hidden}fn f(s: Secret) -> dyn {{ dyn::of(s) }}\n"), &[Code::Em03]);
}

#[test]
fn em04_a_capability_the_target_does_not_name() {
    analyzes_to("fn f() -> bool { @target_has(\"dynamic_lifting\") }\n", &[]);
    analyzes_to("fn f() -> bool { @target_has(\"teleport\") }\n", &[Code::Em04]);
}

#[test]
fn et02_a_type_with_no_written_form() {
    let types = "struct Holder { at: *i64 }\nstruct Outer { name: [char], inner: Holder }\n\
                 [tcon: opaque]\nstruct Secret { key: u64 }\nstruct Plain { x: i64 }\n";
    analyzes_to(&format!("{types}fn f(p: *Plain) -> string {{ tcon::emit(p) }}\n"), &[]);
    analyzes_to(&format!("{types}fn f(p: *Outer) -> string {{ tcon::emit(p) }}\n"), &[Code::Et02]);
    analyzes_to(&format!("{types}fn f(p: *Secret) -> string {{ tcon::emit(p) }}\n"), &[Code::Et02]);
    analyzes_to(
        &format!("{types}fn f(s: *[char]) -> bool {{ tcon::parse::<Outer>(s).is_ok() }}\n"),
        &[Code::Et02],
    );
    let out = analyze(&format!("{types}fn f(p: *Outer) -> string {{ tcon::emit(p) }}\n"));
    assert!(format!("{:?}", out.diagnostics[0]).contains("inner.at"), "the part is named");
    // a type that writes itself is taken as it is
    analyzes_to(
        &format!(
            "{types}impl Holder {{ fn to_tcon(self: *Holder, w: *tcon::Writer) {{ }} }}\n\
             fn f(p: *Outer) -> string {{ tcon::emit(p) }}\n"
        ),
        &[],
    );
}

#[test]
fn et01_an_embedded_document_that_does_not_fit_its_type_is_reported_in_it() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("bad.tcon"), "{\n  host: \"h\",\n  port: 70000,\n}\n").unwrap();
    std::fs::write(dir.path().join("missing.tcon"), "{ host: \"h\" }\n").unwrap();
    let main = dir.path().join("demo.tq");
    let check = |src: &str| {
        std::fs::write(&main, src).unwrap();
        let mut session = Session::new();
        let id = session.load(&main).unwrap();
        let out = compile(
            &mut session,
            id,
            Options {
                stage: Stage::Check,
                ..Options::default()
            },
        );
        let found: Vec<(Code, String, u32)> = out
            .diagnostics
            .iter()
            .map(|d| {
                let span = d.primary_span().expect("a place");
                let file = session.sources().file(span.source);
                let name = std::path::Path::new(file.name()).file_name().unwrap().to_string_lossy().into_owned();
                (d.code, name, file.line_col(span.start).0)
            })
            .collect();
        found
    };
    let types = "#unit classical\nstruct C { host: [char], port: u16 }\n";
    let found = check(&format!("{types}let K: const C = @embed(\"bad.tcon\");\n"));
    assert_eq!(found, [(Code::Et01, "bad.tcon".to_owned(), 3)], "reported at the value's line in the document");
    let found = check(&format!("{types}let K: const C = @embed(\"missing.tcon\");\n"));
    assert_eq!(found.iter().map(|f| f.0).collect::<Vec<_>>(), [Code::Et01]);
    let found = check(&format!("{types}fn f() {{ let k = @embed(\"bad.tcon\"); }}\n"));
    assert_eq!(found.iter().map(|f| f.0).collect::<Vec<_>>(), [Code::Es06], "nothing says the type");
    let found = check(&format!("{types}let K: const C = @embed(\"absent.tcon\");\n"));
    assert_eq!(found.iter().map(|f| f.0).collect::<Vec<_>>(), [Code::Es02]);
}

#[test]
fn ec09_clone_of_an_opaque_type() {
    let types = "[tcon: opaque]\nstruct Secret { key: u64 }\nstruct Vault { s: Secret }\n";
    analyzes_to(&format!("{types}fn f(v: *Vault) -> Vault {{ clone(v) }}\n"), &[Code::Ec09]);
    analyzes_to("[tcon: shut]\nstruct S { x: i32 }\n", &[Code::Es06]);
}

#[test]
fn fmt_takes_the_text_so_far_and_gives_it_back() {
    let p = "struct P { x: i64 }\n";
    analyzes_to(&format!("{p}impl P {{ fn $fmt(self: *const P, out: [char]) -> [char] {{ out }} }}\n"), &[]);
    analyzes_to(&format!("{p}impl P {{ fn $fmt(self: *const P) -> [char] {{ [] }} }}\n"), &[Code::Es07]);
}

#[test]
fn what_an_erased_reference_refers_to_cannot_be_read() {
    assert!(!analyze("fn f(p: *any) -> i32 { *p }\n").diagnostics.is_empty());
}

#[test]
fn em02_an_operator_method_the_language_does_not_define() {
    analyzes_to(
        &format!("{SHAPES}impl Point {{ fn $plus(self: Point, o: Point) -> Point {{ self }} }}\n"),
        &[Code::Em02],
    );
}

#[test]
fn ec03_assignment_through_a_reference_that_only_reads() {
    analyzes_to(&format!("{SHAPES}fn f(p: *const Point) {{ p.x = 1; }}\n"), &[Code::Ec03]);
}

#[test]
fn ec03_assignment_through_a_reference_to_a_constant() {
    analyzes_to("let K: const i32 = 1;\nfn f() { let r = &K; *r = 2; }\n", &[Code::Ec03]);
    analyzes_to("fn f(p: *const i32) { *p = 2; }\n", &[Code::Ec03]);
}

#[test]
fn ec07_a_constant_index_out_of_bounds() {
    analyzes_to("fn f(a: [u8; 4]) -> u8 { a[4] }\n", &[Code::Ec07]);
}

#[test]
fn ec08_growing_an_array_through_a_slice() {
    analyzes_to("fn f(xs: *[u8]) { xs.push(1); }\n", &[Code::Ec08]);
    analyzes_to("fn f(xs: *[u8]) { xs.reserve(8); }\n", &[Code::Ec08]);
    // looking at it through one is fine
    analyzes_to("fn f(xs: *[u8]) -> usize { xs.as_ref().len() }\n", &[]);
}

#[test]
fn es04_a_tuple_element_past_the_end() {
    analyzes_to("fn f(t: (i32, bool)) -> i32 { t.0 }\n", &[]);
    analyzes_to("fn f(t: (i32, bool)) -> bool { t.2 }\n", &[Code::Es04]);
}

#[test]
fn ec09_cloning_a_value_that_holds_a_closure() {
    let out = analyze(
        "struct Job { id: i64, run: closure<fn() -> i64> }\nfn f(j: *Job) -> Job { clone(j) }\n",
    );
    let d = out.diagnostics.iter().find(|d| d.code == Code::Ec09).expect("reported");
    assert!(d.message.contains("`run`"), "names the field: {}", d.message);
}

#[test]
fn es07_a_copy_method_of_the_wrong_shape() {
    analyzes_to(
        &format!("{SHAPES}impl Point {{ fn $copy(self: Point) -> Point {{ self }} }}\n"),
        &[Code::Es07],
    );
}

#[test]
fn any_binding_of_a_growable_array_may_grow() {
    analyzes_to("fn f() { let xs: [u8] = [1]; xs.push(2); }\n", &[]);
}

#[test]
fn es06_a_function_given_to_map_that_takes_the_wrong_thing() {
    analyzes_to(
        "fn g(x: *bool) -> bool { *x }\nfn f(xs: [u8]) { let ys = xs.map(&g); }\n",
        &[Code::Es06],
    );
}

#[test]
fn es12_moving_an_element_out_of_a_growable_array() {
    analyzes_to(
        &format!("{SHAPES}fn f(xs: [Point]) -> Point {{ xs[0] }}\n"),
        &[Code::Es12],
    );
}

#[test]
fn ec04_a_constant_that_would_panic_reports_ra10() {
    let out = analyze("fn f(n: constexpr i32) -> constexpr i32 { if n > 0 { n } else { panic(\"no\") } }\nlet X: const i32 = f(0);\n");
    let d = out.diagnostics.iter().find(|d| d.code == Code::Ec04).expect("reported");
    assert!(d.message.contains("RA10"), "{}", d.message);
}

#[test]
fn ec04_an_exact_constant_dividing_by_zero() {
    let out = analyze("let X: const frac = frac::of(1, 3) / frac::from(0);\n");
    let d = out.diagnostics.iter().find(|d| d.code == Code::Ec04).expect("reported");
    assert!(d.message.contains("RA02"), "{}", d.message);
}

#[test]
fn ec04_a_phase_out_of_range_in_a_constant() {
    let out = analyze("let P: const phase<8> = phase::<8>::of(9);\n");
    let d = out.diagnostics.iter().find(|d| d.code == Code::Ec04).expect("reported");
    assert!(d.message.contains("RA09"), "{}", d.message);
}

#[test]
fn es06_vectors_and_matrices_of_shapes_that_do_not_fit() {
    let a = "let a: mat<i64, 2, 3> = [[1, 2, 3], [4, 5, 6]];";
    let v = "let v: vec<i64, 2> = [1, 2];";
    let out = analyze(&format!("fn f() {{ {a} {v} let r = a * v; }}\n"));
    let d = out.diagnostics.iter().find(|d| d.code == Code::Es06).expect("reported");
    assert!(d.message.contains("2×3 matrix") && d.message.contains("2 elements"), "{}", d.message);
    analyzes_to(&format!("fn f() {{ {a} let r = a * a; }}\n"), &[Code::Es06]);
    analyzes_to(
        &format!("fn f() {{ {v} let w: vec<i64, 3> = [1, 2, 3]; let s = v == w; }}\n"),
        &[Code::Es06],
    );
    analyzes_to(
        &format!("fn f() {{ {v} let w: vec<f64, 2> = [1.0, 2.0]; let s = v + w; }}\n"),
        &[Code::Es06],
    );
    analyzes_to(&format!("fn f() {{ {a} let w: vec<i64, 3> = [1, 2, 3]; let r = a * w; }}\n"), &[]);
}

#[test]
fn es06_vectors_covectors_and_matrices_of_kinds_that_do_not_combine() {
    let lets = "let v: vec<i64, 2> = [1, 2]; let c: covec<i64, 2> = [3, 4]; let m: mat<i64, 2, 2> = [[1, 0], [0, 1]];";
    // two vectors have no product, and the diagnostic writes the two that
    // `adjoint` makes
    let out = analyze(&format!("fn f() {{ {lets} let r = v * v; }}\n"));
    let d = out.diagnostics.iter().find(|d| d.code == Code::Es06).expect("reported");
    assert!(d.helps.iter().any(|h| h.contains("adjoint(a) * b")), "{d}");
    for bad in ["c * c", "v * m", "m * c", "v + c", "v == c"] {
        analyzes_to(&format!("fn f() {{ {lets} let r = {bad}; }}\n"), &[Code::Es06]);
    }
    analyzes_to(
        &format!("fn f() {{ {lets} let w: covec<i64, 3> = [1, 2, 3]; let r = w * v; }}\n"),
        &[Code::Es06],
    );
    // a covector times a vector is a number, a vector times a covector a
    // matrix, and a covector times a matrix a covector
    analyzes_to(
        &format!(
            "fn f() {{ {lets} let n: i64 = c * v; let o: mat<i64, 2, 2> = v * c; let r: covec<i64, 2> = c * m; \
             let s: covec<i64, 2> = 2 * c; let t: covec<i64, 4> = c ** c; }}\n"
        ),
        &[],
    );
}

#[test]
fn es06_es07_the_adjoint_of_a_value() {
    // a vector's adjoint is a covector and a covector's a vector; a
    // matrix's is its conjugate transpose, and a number's its conjugate
    analyzes_to(
        "fn f() { let v: vec<cyclo<8>, 2> = [1, i]; let c: covec<cyclo<8>, 2> = adjoint(v); \
         let w: vec<cyclo<8>, 2> = adjoint(&c); let m: mat<cyclo<8>, 3, 2> = [[1, 0], [0, i], [1, 1]]; \
         let d: mat<cyclo<8>, 2, 3> = adjoint(m); let n: cyclo<8> = adjoint(v) * w; let x: i32 = adjoint(3); \
         let y: f64 = adjoint(2.5); }\n",
        &[],
    );
    analyzes_to(
        "@static_assert(adjoint(i) == -i, \"i\");\n\
         @static_assert(adjoint(frac::of(1, 2)) == frac::of(1, 2), \"a real number\");\n\
         @static_assert(adjoint(isq2 * i) * (isq2 * i) == cyclo::<8>::of(&frac::of(1, 2)), \"a modulus\");\n",
        &[],
    );
    analyzes_to("fn f() { let s = adjoint(\"text\"); }\n", &[Code::Es06]);
    analyzes_to("struct P { x: i32 }\nfn f() { let s = adjoint(P { x: 1 }); }\n", &[Code::Es06]);
    analyzes_to("fn f() { let s = adjoint(1, 2); }\n", &[Code::Es07]);
    // a unit's own `adjoint` hides the library's
    analyzes_to("fn adjoint(s: *const [char]) -> i32 { 0 }\nfn f() -> i32 { adjoint(\"text\") }\n", &[]);
}

#[test]
fn es06_cyclotomic_numbers_of_orders_neither_dividing_the_other() {
    let lets = "let a: cyclo<4> = w(1, 4); let e: cyclo<3> = w(1, 3);";
    let out = analyze(&format!("fn f() {{ {lets} let s = a + e; }}\n"));
    let d = out.diagnostics.iter().find(|d| d.code == Code::Es06).expect("reported");
    assert!(d.helps.iter().any(|h| h.contains("cyclo<12>")), "{d}");
    analyzes_to("fn f() { let a: cyclo<4> = w(1, 4); let b: cyclo<8> = isq2; let s = a + b; let t = b * a; }\n", &[]);
}

#[test]
fn no_instance_of_a_generic_type_is_a_prefix_of_another() {
    let m = "struct Mod<N: const usize> { v: u64 }\n";
    analyzes_to(&format!("{m}fn f(x: *Mod<3>) -> *Mod<4> {{ x }}\n"), &[Code::Es06]);
    analyzes_to(&format!("{m}fn g() {{ @static_assert(!@is_prefix_of(Mod<4>, Mod<3>), \"not\"); }}\n"), &[]);
}

#[test]
fn es21_a_use_of_a_deprecated_item_warns_with_its_message() {
    let decl = "[deprecated: \"use g\"]\nfn f() -> i32 { 1 }\n[deprecated]\nstruct Old { v: i32 }\n";
    let out = analyze(&format!("{decl}fn h() -> i32 {{ let o = Old {{ v: 1 }}; f() + o.v }}\n"));
    let got: Vec<Code> = out.diagnostics.iter().map(|d| d.code).collect();
    assert_eq!(got, [Code::Es21, Code::Es21], "{:?}", out.diagnostics);
    assert!(out.diagnostics.iter().all(|d| !d.is_error()), "a warning, not an error");
    assert!(out.diagnostics.iter().any(|d| d.helps.iter().any(|h| h == "use g")));
    // inside a deprecated item, its own kind are not warned of
    analyzes_to(&format!("{decl}[deprecated]\nfn k(o: *Old) -> i32 {{ f() + o.v }}\n"), &[]);
}

#[test]
fn ea05_an_annotation_argument_of_the_wrong_form() {
    analyzes_to("[export: 3]\nfn f() { }\n", &[Code::Ea05]);
    analyzes_to("[export: \"no-dashes\"]\nfn f() { }\n", &[Code::Ea05]);
    analyzes_to("[export: \"main\"]\nfn f() { }\n", &[Code::Ea05]);
    analyzes_to("[export: \"__hidden\"]\nfn f() { }\n", &[Code::Ea05]);
    analyzes_to("[inline: always]\nfn f() { }\n", &[Code::Ea05]);
    analyzes_to("[test: 1]\nfn f() { }\n", &[Code::Ea05]);
    analyzes_to("[deprecated: 1]\nfn f() { }\n", &[Code::Ea05]);
    analyzes_to("[export: \"f_sym\"]\nfn f() { }\n[inline]\nfn g() { }\n[inline: never]\nfn h() { }\n", &[]);
}

#[test]
fn es18_export_and_test_where_they_cannot_apply() {
    analyzes_to("[export: \"f\"]\nfn f<T>(x: T) { }\n", &[Code::Es18]);
    analyzes_to("[export: \"f\"]\nstatic fn f() { }\n", &[Code::Es18]);
    analyzes_to("[export: \"x\"]\nlet X: const i32 = 1;\n", &[Code::Es18]);
    analyzes_to("[export: \"s\"]\nstruct S { v: i32 }\n", &[Code::Es18]);
    analyzes_to("struct S { v: i32 }\n[export: \"s_get\"]\nfn S.get(self: *S) -> i32 { self.v }\n", &[Code::Es18]);
    analyzes_to("[test]\nfn t(x: i32) { }\n", &[Code::Es18]);
    analyzes_to("[test]\nfn t() -> i32 { 1 }\n", &[Code::Es18]);
    analyzes_to("[test]\nstruct S { v: i32 }\n", &[Code::Es18]);
}

#[test]
fn es05_one_symbol_exported_twice() {
    analyzes_to("[export: \"same\"]\nfn f() { }\n[export: \"same\"]\nfn g() { }\n", &[Code::Es05]);
}

#[test]
fn items_inside_a_block_are_scoped_to_it() {
    analyzes_to("fn f() -> i32 { fn g() -> i32 { 1 } g() }\n", &[]);
    analyzes_to("fn f() -> i32 { fn g() -> i32 { 1 } g() }\nfn h() -> i32 { g() }\n", &[Code::Es04]);
    let out = analyze("fn f() -> i32 { let x = 1; fn g() -> i32 { x } g() }\n");
    let d = out.diagnostics.iter().find(|d| d.code == Code::Es04).expect("reported");
    assert!(d.notes.iter().any(|n| n.contains("only a closure captures")), "{d}");
    analyzes_to("fn f() { fn g() { } fn g() { } }\n", &[Code::Es05]);
    analyzes_to("fn f() { fn g<T>(x: T) { } }\n", &[Code::Es18]);
    analyzes_to("fn f() { type T = i32; }\n", &[Code::Es18]);
}

#[test]
fn a_library_function_as_a_value() {
    analyzes_to("fn f() { let p = print; p(\"x\"); }\n", &[]);
    analyzes_to("fn f() { let c = clone; }\n", &[Code::Es06]);
}

#[test]
fn an_open_range_and_a_tensor_without_one_are_errors() {
    assert!(analyze("fn f() { let r = 1..; }\n").reported(Code::Es02));
    analyzes_to("fn f() -> i32 { 2 ** 3 }\n", &[Code::Es06]);
    analyzes_to("fn f(xs: *[i32]) -> usize { xs.len::<i32>() }\n", &[Code::Es06]);
}

#[test]
fn a_function_mixing_const_and_ordinary_parameters_is_specialized() {
    let f = "fn scale(n: constexpr i64, x: i64) -> i64 { n * x }\n";
    analyzes_to(&format!("{f}fn g(x: i64) -> i64 {{ scale(3, x) + scale(4, x) }}\n"), &[]);
    // the `const` argument must be known while translating
    let out = analyze(&format!("{f}fn g(x: i64) -> i64 {{ scale(x, x) }}\n"));
    assert!(out.has_errors(), "{:?}", out.diagnostics);
    analyzes_to("fn f(n: constexpr f64, x: i64) -> i64 { x }\n", &[Code::Es06]);
    analyzes_to("fn f<T>(n: constexpr i64, x: T) -> T { x }\n", &[Code::Es06]);
    analyzes_to("struct S { v: i64 }\nimpl S { fn m(self: *S, n: constexpr i64) -> i64 { self.v } }\n", &[Code::Es06]);
}

/////////////
// LINKING //
/////////////

#[cfg(feature = "llvm")]
#[test]
fn el05_a_program_without_a_main_that_can_start_it() {
    use topiq::driver::build::{BuildOptions, build};
    let dir = tempfile::tempdir().unwrap();
    for (name, src) in [
        ("none.tq", "#unit classical\nfn helper() { }\n"),
        ("wrong.tq", "#unit classical\nfn main() { }\n"),
    ] {
        let path = dir.path().join(name);
        std::fs::write(&path, src).unwrap();
        let program = build(
            &[path],
            &BuildOptions {
                out_dir: dir.path().to_owned(),
                link: true,
                ..BuildOptions::default()
            },
        )
        .unwrap();
        assert!(
            program.diagnostics.iter().any(|d| d.code == Code::El05),
            "{name}: {:?}",
            program.diagnostics
        );
        assert!(program.executable.is_none());
    }
}

////////////////////////
// CIRCUIT GENERATION //
////////////////////////

/// The codes a quantum unit importing `gates`, with `body` after it,
/// reports.
fn gated(body: &str) -> Vec<Code> {
    quantum_codes(&format!("import gates;\n{body}"))
}

#[test]
fn a_quantum_if_controls_its_branches() {
    assert_eq!(gated("fn f(c: *qubit, t: *qubit) { if c { x(t); } else { z(t); } }\n"), []);
    assert_eq!(gated("fn f(a: *qubit, b: *qubit, t: *qubit) { if a || !b { x(t); } }\n"), []);
}

#[test]
fn eq04_each_lifting_rule() {
    // measuring, lifting or preparing in a branch
    assert_eq!(gated("fn f(c: *qubit, t: qubit) { if c { let m = measure t; } }\n"), [Code::Eq04]);
    assert_eq!(gated("fn f(c: *qubit) { if c { let r: [qubit; 1] = prep |0>; forget r; } }\n"), [Code::Eq04]);
    // acting on the condition's own qubit
    assert_eq!(gated("fn f(c: *qubit) { if c { x(c); } }\n"), [Code::Eq04]);
    // changing a classical binding declared outside
    assert_eq!(gated("fn f(c: *qubit) { let n: u32 = 0; if c { n = 1; } }\n"), [Code::Eq04]);
    // and leaving the branch early
    assert_eq!(gated("fn f(c: *qubit) -> u32 { if c { return 1; } 0 }\n"), [Code::Eq04]);
}

#[test]
fn eq04_a_branch_writing_a_qubit_its_condition_computed_from() {
    // `a || b` is computed into an ancilla from `a`, which the branch changes
    assert_eq!(gated("fn f(a: *qubit, b: *qubit) { if a || b { x(a); } }\n"), [Code::Eq04]);
    assert_eq!(gated("fn f(a: *qubit, b: *qubit, t: *qubit) { if a ^ b { x(t); } }\n"), []);
}

#[test]
fn operators_on_qubits_are_the_gates_they_stand_for() {
    let ok = "fn f(a: *qubit, b: *qubit, t: *qubit) {\n\
              *t ^= true; *t ^= a; *t ^= a & b; *t ^= a | !b; *t ^= a ^ b; *t ^= *a == *b; *t ^= *a != *b;\n\
              let p = a & b; let q = !p; forget p; forget q;\n\
              aux let s = a ^ b; *t ^= s;\n\
              }\n\
              fn g(r: *[qubit; 3], s: *[qubit; 3], k: u8) {\n\
              *r ^= s; *r ^= 5; *r ^= [true, false, true]; let u = r ^ s; let v = !u; forget u; forget v;\n\
              }\n";
    assert_eq!(gated(ok), []);
    // two handles compared are still two references
    assert_eq!(gated("fn f(a: *qubit, b: *qubit) -> bool { let s = a; s == b }\n").len(), 1);
}

#[test]
fn eq18_an_operator_with_no_reversible_meaning_on_qubits() {
    assert_eq!(gated("fn f(a: *qubit, t: *qubit) { *t &= a; }\n"), [Code::Eq18]);
    assert_eq!(gated("fn f(a: *qubit, t: *qubit) { *t |= a; }\n"), [Code::Eq18]);
    assert_eq!(gated("fn f(r: *[qubit; 2]) { *r <<= 1; }\n"), [Code::Eq18]);
    assert_eq!(gated("fn f(a: *qubit) { let q = a + a; forget q; }\n"), [Code::Eq18]);
    assert_eq!(gated("fn f(t: qubit) -> qubit { t++; t }\n"), [Code::Eq18]);
}

#[test]
fn eq20_a_flip_controlled_on_its_own_target() {
    assert_eq!(gated("fn f(t: *qubit) { *t ^= t; }\n"), [Code::Eq20]);
    assert_eq!(gated("fn f(a: *qubit, t: *qubit) { *t ^= *t ^ *a; }\n"), [Code::Eq20]);
    assert_eq!(gated("fn f(a: *qubit, t: *qubit) { *t ^= a | t; }\n"), [Code::Eq20]);
}

#[test]
fn ec10_a_constant_wider_than_its_register() {
    assert_eq!(gated("fn f(r: *[qubit; 2]) { *r ^= 4; }\n"), [Code::Ec10]);
    assert_eq!(gated("fn f() -> u8 { let x: quint<2> = 4; measure x }\n"), [Code::Ec10]);
    assert_eq!(gated("fn f(r: *[qubit; 8], v: u8) { *r ^= v; }\n"), []);
    // a value known only as the circuit is generated is checked then
    assert_eq!(gated("fn f() -> u8 { let k: u8 = 4; let x: quint<2> = k; measure x }\n"), [Code::Ec04]);
}

#[test]
fn a_quint_measures_to_the_smallest_unsigned_type() {
    let ok = "fn f() -> (u8, u16, u64) {\n\
              let a: quint<3> = 5; let b: quint<9> = 300; let c: quint<64> = 1; let d: quint<9> = 2;\n\
              let a = wrapping_add(a, 1); let b = wrapping_sub(b, d); let c = rotate_left(c, 63);\n\
              forget d;\n\
              (measure a, measure b, measure c)\n\
              }\n";
    assert_eq!(gated(ok), []);
}

#[test]
fn es23_a_quint_width_outside_1_to_64() {
    assert_eq!(gated("fn f(x: *quint<0>) { }\n"), [Code::Es23]);
    assert_eq!(gated("fn f(x: *quint<65>) { }\n"), [Code::Es23]);
}

#[test]
fn eq19_quint_arithmetic_written_without_wrapping() {
    assert_eq!(gated("fn f(x: *quint<3>, y: *quint<3>) { *x += y; }\n"), [Code::Eq19]);
    assert_eq!(gated("fn f(x: quint<3>) -> quint<3> { let y = x * 3; forget x; y }\n"), [Code::Eq19]);
    assert_eq!(gated("fn f(x: quint<3>) -> quint<3> { x++; x }\n"), [Code::Eq19]);
    assert_eq!(gated("fn f(x: quint<3>) -> quint<3> { let y = -x; forget x; y }\n"), [Code::Eq19]);
    // an even factor is not reversible
    assert_eq!(gated("fn f(x: quint<3>) -> quint<3> { wrapping_mul(x, 2) }\n"), [Code::Eq18]);
}

#[test]
fn qcopy_prepares_again_what_this_operator_prepared() {
    let ok = "fn f() -> ([bool; 2], [bool; 2], u8, u8) {\n\
              let r: [qubit; 2]; h(&r[0]); r[1] ^= r[0]; let s = qcopy(&r);\n\
              let x: quint<3> = 2; let x = wrapping_add(x, 5); let y = qcopy(&x);\n\
              (measure r, measure s, measure x, measure y)\n\
              }\n";
    assert_eq!(gated(ok), []);
    // what qcopy takes
    assert_eq!(gated("fn f() -> u8 { let a: u8 = 1; qcopy(&a) }\n"), [Code::Es06]);
    assert_eq!(gated("fn f() -> qubit { let q: qubit; let p = qcopy(q); forget q; p }\n"), [Code::Es06]);
    // a type holding qubits does not define `$copy`
    assert_eq!(
        gated("struct B { q: qubit }\nimpl B { fn $copy(self: *const B) -> B { B { q: qcopy(&self.q) } } }\n")[0],
        Code::Es06
    );
}

#[test]
fn ej19_a_copy_of_a_state_not_prepared_here() {
    // a value the operator was given
    assert_eq!(gated("fn f(q: *qubit) -> qubit { qcopy(q) }\n"), [Code::Ej19]);
    // half of an entangled pair
    assert_eq!(
        gated("fn f() -> ([bool; 2], bool) { let r: [qubit; 2]; h(&r[0]); r[1] ^= r[0]; let s = qcopy(&r[0]); (measure r, measure s) }\n"),
        [Code::Ej19]
    );
    // a history that depends on a measurement
    assert_eq!(
        gated("fn f() -> (bool, bool) { let a: qubit; h(&a); let m = measure a; let b: qubit; if m { b ^= true; } let c = qcopy(&b); (measure b, measure c) }\n"),
        [Code::Ej19]
    );
    // and a copy inside a quantum branch is a preparation there
    assert_eq!(
        gated("fn f(c: *qubit) { if c { let q: qubit; let p = qcopy(&q); forget p; forget q; } }\n")[0],
        Code::Eq04
    );
}

#[test]
fn an_operator_written_with_operators_judges_as_its_gates_do() {
    let unit = "cover C = fin{ |0> ** (|0> - |1>) * isq2, |1> ** (|0> - |1>) * isq2 };\n\
                gauge G = fid((|0> + |1>) ** (|0> - |1>) / 2);\n\
                [cover: C] [gauge: G] [expect: monic; contract = rigid]\n\
                fn sugared(x: *qubit, y: *qubit) { *y ^= x; }\n\
                [cover: C] [gauge: G] [expect: monic; contract = rigid]\n\
                fn by_hand(x: *qubit, y: *qubit) { cx(x, y); }\n";
    let mut session = Session::new();
    let id = session.add("demo.tq", &format!("#unit quantum\nimport gates;\n{unit}"));
    let out = compile(
        &mut session,
        id,
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    );
    let refusals: Vec<String> = out
        .diagnostics
        .iter()
        .filter(|d| d.code == Code::Ej03)
        .map(|d| d.message.replace("sugared", "f").replace("by_hand", "f"))
        .collect();
    // both refused alike, naming one certificate
    assert_eq!(refusals.len(), 2, "{refusals:?}");
    assert_eq!(refusals[0], refusals[1]);
}

#[test]
fn eq05_forget_in_a_controlled_region() {
    assert_eq!(gated("fn f(c: *qubit, t: qubit) { if c { forget t; } }\n"), [Code::Eq05]);
}

#[test]
fn eq03_a_loop_of_unknown_count_acting_on_qubits() {
    assert_eq!(gated("fn f(t: *qubit) { let i: u32 = 0; while i < 3 { h(t); i = i + 1; } }\n"), [Code::Eq03]);
    assert_eq!(gated("fn f(t: *qubit) { while t { } }\n"), [Code::Eq03]);
    // a loop over a range known when the circuit is generated is unrolled
    assert_eq!(gated("fn f(r: *[qubit; 3]) { for i in 0..3 { h(&r[i]); } }\n"), []);
}

#[test]
fn eq11_an_initializer_acting_on_qubits() {
    assert_eq!(
        unit_codes("#unit quantum(init)\nlet N: u32 = 0;\nfn init() { N = 1; let q: [qubit; 1] = prep |0>; forget q; }\n"),
        [Code::Eq11]
    );
    assert_eq!(unit_codes("#unit quantum(init)\nlet N: u32 = 0;\nfn init() { N = 1; }\n"), []);
}

#[test]
fn eq12_and_eq13_the_matrix_given_to_apply() {
    assert_eq!(
        quantum_codes("fn f(r: *[qubit; 1]) { let u: mat<f64, 2, 2> = [[0.0, 1.0], [1.0, 0.0]]; apply(u, r); }\n"),
        [Code::Eq12]
    );
    assert_eq!(
        quantum_codes("fn f(r: *[qubit; 1]) { let u: mat<cyclo<8>, 2, 2> = [[1, 1], [1, 0]]; apply(u, r); }\n"),
        [Code::Eq13]
    );
    assert_eq!(
        quantum_codes("fn f(r: *[qubit; 1]) { let u: mat<cyclo<8>, 2, 2> = [[isq2, isq2], [isq2, -isq2]]; apply(u, r); }\n"),
        []
    );
    // a program may vouch for its matrices, at its own risk
    assert_eq!(
        quantum_codes("[trusted_unitary]\nfn f(r: *[qubit; 1]) { let u: mat<cyclo<8>, 2, 2> = [[1, 1], [1, 0]]; apply(u, r); }\n"),
        []
    );
    // a gate written as a sum of outer products, and a lone one, which is
    // never unitary
    let kets = "let k0: vec<cyclo<8>, 2> = [1, 0]; let k1: vec<cyclo<8>, 2> = [0, 1];";
    assert_eq!(
        quantum_codes(&format!("fn f(r: *[qubit; 1]) {{ {kets} apply(k0 * adjoint(k1) + k1 * adjoint(k0), r); }}\n")),
        []
    );
    assert_eq!(
        quantum_codes(&format!("fn f(r: *[qubit; 1]) {{ {kets} apply(k0 * adjoint(k0), r); }}\n")),
        [Code::Eq13]
    );
    // the adjoint of a matrix and of the operator applying it
    assert_eq!(
        quantum_codes(
            "fn f(r: *[qubit; 1]) { let u: mat<cyclo<8>, 2, 2> = [[isq2, isq2], [i * isq2, -i * isq2]]; apply(adjoint(u), r); }\n\
             fn g(r: *[qubit; 1]) { f(r); adjoint(f)(r); }\n"
        ),
        []
    );
}

#[test]
fn eq15_one_qubit_twice_in_one_operation() {
    assert_eq!(gated("fn f(t: *qubit) { cx(t, t); }\n"), [Code::Eq15]);
}

#[test]
fn eq16_a_runtime_value_shaping_the_circuit() {
    assert_eq!(
        gated("[entry]\nfn f(n: usize) -> bool { let r: [qubit; 4] = prep |0000>; h(&r[n]); let m = measure r; m[0] }\n"),
        [Code::Eq16]
    );
    // a parameter may bound a loop and turn a gate, as a circuit's input
    assert_eq!(
        gated("[entry]\nfn f(n: usize, k: phase<8>) -> bool { let r: [qubit; 1] = prep |0>; for i in 0..n { rz(k, &r[0]); } let m = measure r; m[0] }\n"),
        []
    );
}

#[test]
fn a_register_grown_as_the_circuit_runs() {
    let grown = |rest: &str, ret: &str, tail: &str| {
        format!(
            "[entry]\nfn f(n: usize) -> {ret} {{ let r: [qubit] = []; \
             for i in 0..n {{ let q: qubit; r.push(q); }} {rest} {tail} }}\n"
        )
    };
    let measured = |rest: &str| grown(rest, "bool", "let m = measure r; m.len() > 0");
    // its qubits are named by index, and it may grow again
    assert_eq!(gated(&measured("for i in 0..r.len() { h(&r[i]); } let q: qubit; r.push(q); h(&r[0]);")), []);
    assert_eq!(gated(&grown("let k = r.len() as u32;", "u32", "forget r; k")), []);
    // its caller cannot receive it on wires fixed in advance
    assert_eq!(gated(&grown("", "[qubit]", "r")), [Code::Eq16]);
    // a loop of the circuit changes no binding outside it that it cannot
    // carry from one time round to the next
    assert_eq!(gated(&measured("let s = (0u32, 1u32); for i in 0..n { s.0 = 2; }")), [Code::Eq16]);
    assert_eq!(gated(&measured("let c: u32 = 0; for i in 0..n { c = c + 1; }")), []);
}

#[test]
fn eq14_generation_that_does_not_finish() {
    assert_eq!(quantum_codes("[entry]\nfn f() { loop { } }\n"), [Code::Eq14]);
}

#[test]
fn ej04_a_phase_outside_the_conductor() {
    assert_eq!(gated("fn f(q: *qubit) { rz(phase::<16>::of(1), q); }\n"), [Code::Ej04]);
    assert_eq!(unit_codes("#unit quantum\n#pragma conductor(16)\nimport gates;\nfn f(q: *qubit) { rz(phase::<16>::of(1), q); }\n"), []);
}

#[test]
fn a_quantum_enumeration_is_controlled_on_and_measured() {
    let e = "enum QB { Zero(qubit), One(qubit) }\n";
    assert_eq!(
        gated(&format!("{e}fn f() -> bool {{ let q: qubit; let v = QB::One(q); match &v {{ QB::Zero(p) => {{ x(p); }} QB::One(p) => {{ z(p); }} }} match measure v {{ QB::Zero(p) => {{ forget p; false }} QB::One(p) => {{ forget p; true }} }} }}\n")),
        []
    );
    // an arm that drops the measured payload
    assert_eq!(
        gated(&format!("{e}fn f(v: QB) -> bool {{ match measure v {{ QB::Zero(_) => false, QB::One(p) => {{ forget p; true }} }} }}\n")),
        [Code::Eq01]
    );
    // a classical field is held as the basis state of qubits: read by
    // measuring, never under the tag's control
    let tagged = "enum E { A(qubit), B(qubit, u32) }\n";
    assert_eq!(gated(tagged), []);
    assert_eq!(
        gated(&format!("{tagged}fn f(v: E) -> u32 {{ match measure v {{ E::A(p) => {{ forget p; 0 }} E::B(p, n) => {{ forget p; n }} }} }}\n")),
        []
    );
    assert_eq!(
        gated(&format!("{tagged}fn f(v: *E) {{ match v {{ E::A(p) => {{ x(p); }} E::B(p, _) => {{ z(p); }} }} }}\n")),
        []
    );
    assert_eq!(
        gated(&format!("{tagged}fn f(v: *E) {{ match v {{ E::A(p) => {{ x(p); }} E::B(p, n) => {{ z(p); }} }} }}\n")),
        [Code::Eq04]
    );
    assert_eq!(quantum_codes("enum E { A(qubit), B(f64) }\n"), [Code::Es06]);
    // a quantum structure measures field by field, keeping what is
    // classical
    let pair = "struct P { a: qubit, r: [qubit; 2], n: u32 }\n";
    assert_eq!(
        gated(&format!("{pair}fn f(p: P) -> u32 {{ let (a, r, n) = measure p; if a && r[1] {{ n }} else {{ 0 }} }}\n")),
        []
    );
    assert_eq!(quantum_codes(&format!("{e}struct S {{ v: QB }}\nfn f(s: S) {{ let m = measure s; }}\n")), [Code::Es06]);
}

///////////////////////////////////////////////////////
// UNCOMPUTATION, MONICITY AND THE OPERATOR FUNCTORS //
///////////////////////////////////////////////////////

#[test]
fn an_ancilla_is_uncomputed_by_undoing_its_cone() {
    // the example of the ancillae: t := a AND b, used, and undone
    assert_eq!(
        gated("fn f(a: qubit, b: qubit, out: qubit) -> (qubit, qubit, qubit) { aux let t: qubit; ccx(&a, &b, &t); cx(&t, &out); (a, b, out) }\n"),
        []
    );
    // a phase kicked back through it leaves it to be undone alone, as in
    // Deutsch's algorithm, even after what it met is measured
    assert_eq!(
        gated("fn f() -> bool { let q: [qubit; 1] = prep |0>; aux let a: qubit; x(&a); h(&a); h(&q[0]); ORACLE_Z(&q[0], &a); h(&q[0]); let m = measure q; m[0] }\n"),
        []
    );
}

#[test]
fn eq02_an_ancilla_that_cannot_be_uncomputed() {
    // joined to a qubit that lives on, undoing it alone cannot part them
    assert_eq!(gated("fn f(q: *qubit) { aux let t: qubit; h(&t); cx(&t, q); }\n"), [Code::Eq02]);
    // what its cone read has changed since
    assert_eq!(gated("fn f(q: *qubit) { aux let t: qubit; cx(q, &t); h(q); }\n"), [Code::Eq02]);
}

#[test]
fn the_operator_functors() {
    let body = "fn p(q: *qubit) { h(q); t(q); }\n\
                fn f(q: *qubit, c: *qubit) { adjoint(p)(q); controlled(p)(c, q); p.then(p)(q); }\n";
    assert_eq!(gated(body), []);
    // an instrument has no adjoint, and cannot be controlled
    let m = "fn m(q: *qubit) { let r: qubit; let b = measure r; }\n";
    assert_eq!(gated(&format!("{m}fn f(q: *qubit) {{ adjoint(m)(q); }}\n")), [Code::Eq17]);
    assert_eq!(gated(&format!("{m}fn f(q: *qubit, c: *qubit) {{ controlled(m)(c, q); }}\n")), [Code::Eq04]);
}

#[test]
fn eq08_a_dynamic_operator_where_a_static_circuit_is_needed() {
    let d = "fn d(q: *qubit) { let r: qubit; let b = measure r; let k: usize = lift (if b { 1 } else { 2 }); }\n";
    assert_eq!(gated(&format!("{d}fn f(q: *qubit, c: *qubit) {{ controlled(d)(c, q); }}\n")), [Code::Eq08]);
}

#[test]
fn a_declared_adjoint_is_checked() {
    assert_eq!(
        gated("[adjoint: undo]\nfn p(q: *qubit) { h(q); s(q); }\nfn undo(q: *qubit) { sdag(q); h(q); }\nfn f(q: *qubit) { adjoint(p)(q); }\n"),
        []
    );
    assert_eq!(
        gated("[adjoint: undo]\nfn p(q: *qubit) { h(q); s(q); }\nfn undo(q: *qubit) { h(q); }\nfn f(q: *qubit) { adjoint(p)(q); }\n"),
        [Code::Eq17]
    );
}

#[test]
fn ej02_replay_of_what_is_not_monic_or_not_known() {
    let bell = "fn bell() -> [qubit; 2] { let r: [qubit; 2] = prep |00>; h(&r[0]); cx(&r[0], &r[1]); r }\n";
    assert_eq!(gated(&format!("{bell}fn f() -> ([qubit; 2], [qubit; 2]) {{ let a = bell(); let b = replay bell(); (a, b) }}\n")), []);
    let coin = "fn coin() -> bool { let r: qubit; h(&r); measure r }\n";
    assert_eq!(gated(&format!("{coin}fn f() -> bool {{ replay coin() }}\n")), [Code::Ej02]);
    assert_eq!(gated("fn p(q: qubit) -> qubit { h(&q); q }\nfn f(q: qubit) -> qubit { replay p(q) }\n"), [Code::Ej02]);
}

//////////////////////////
// QUON AND PREPARATION //
//////////////////////////

/// A register of `n` qubits prepared in `state` and measured.
fn prepared(n: usize, state: &str) -> String {
    format!("fn f() -> [bool; {n}] {{ let r: [qubit; {n}] = prep {state}; measure r }}\n")
}

#[test]
fn a_state_written_out_is_prepared_exactly() {
    assert_eq!(quantum_codes(&prepared(2, "isq2 * |00> + isq2 * |11>")), []);
    assert_eq!(quantum_codes(&prepared(1, "(1 + i) / 2 * |0> + (1 - i) / 2 * |1>")), []);
    assert_eq!(quantum_codes(&prepared(3, "1/2 * |000> + 1/2 * i * |011> + isq2 * w(1, 8) * |110>")), []);
    // the register is as wide as the state
    assert_eq!(quantum_codes(&prepared(1, "isq2 * |00> + isq2 * |11>")), [Code::Es06]);
}

#[test]
fn ej14_a_state_that_is_not_a_unit_vector() {
    assert_eq!(quantum_codes(&prepared(1, "|0> + |1>")), [Code::Ej14]);
    assert_eq!(quantum_codes(&prepared(1, "|0> - |0>")), [Code::Ej14]);
}

#[test]
fn eq10_a_state_no_circuit_prepares_exactly() {
    // a unit vector, but no gate divides by five
    assert_eq!(quantum_codes(&prepared(1, "3/5 * |0> + 4/5 * |1>")), [Code::Eq10]);
}

#[test]
fn ej04_a_coefficient_outside_the_unit_conductor() {
    let src = prepared(1, "isq2 * |0> + w(1, 16) * isq2 * |1>");
    assert_eq!(quantum_codes(&src), [Code::Ej04]);
    assert_eq!(unit_codes(&format!("#unit quantum\n#pragma conductor(16)\n{src}")), []);
    // reduced first: w(2, 16) is w(1, 8)
    assert_eq!(quantum_codes(&prepared(1, "isq2 * |0> + w(2, 16) * isq2 * |1>")), []);
}

#[test]
fn a_named_state_is_a_constant_quon_state() {
    let one = "import quon;\n\
               let ONE: const quon::State = quon::State { width: 1, terms: [quon::Term { ket: [true], amp: cyclo::<48>::from(1) }] };\n";
    assert_eq!(quantum_codes(&format!("{one}{}", prepared(2, "|0> ** ONE"))), []);
    assert_eq!(quantum_codes(&format!("{one}{}", prepared(1, "ONE"))), []);
    assert_eq!(quantum_codes(&format!("let N: const u32 = 1;\n{}", prepared(2, "|0> ** N"))), [Code::Es06]);
    // an amplitude the unit's field does not hold
    let fine = "import quon;\n\
                let Z: const quon::State = quon::State { width: 1, terms: [quon::Term { ket: [false], amp: cyclo::<48>::zeta(3) }] };\n";
    assert_eq!(quantum_codes(&format!("{fine}{}", prepared(1, "Z"))), [Code::Ej04]);
}

#[test]
fn a_quon_document_is_embedded_as_a_structure_of_states() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("states.quon"),
        "bell: [qubit; 2] = isq2 * |00> + isq2 * |11>;\nplus: qubit = isq2 * |0> + isq2 * |1>;\nboth: [qubit; 3] = bell ** plus;\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("wide.quon"), "one: [qubit; 2] = |0>;\n").unwrap();
    std::fs::write(dir.path().join("long.quon"), "one: qubit = |0> + |1>;\n").unwrap();
    let main = dir.path().join("demo.tq");
    let check = |src: &str| {
        std::fs::write(&main, src).unwrap();
        let mut session = Session::new();
        let id = session.load(&main).unwrap();
        let out = compile(
            &mut session,
            id,
            Options {
                stage: Stage::Check,
                ..Options::default()
            },
        );
        out.diagnostics.iter().map(|d| d.code).collect::<Vec<_>>()
    };
    let head = "#unit quantum\nimport quon;\nstruct States { bell: quon::State, plus: quon::State, both: quon::State }\n";
    let use_both = "let BOTH: const quon::State = S.both;\nfn f() -> [bool; 3] { let r: [qubit; 3] = prep BOTH; measure r }\n";
    assert_eq!(check(&format!("{head}let S: const States = @embed(\"states.quon\");\n{use_both}")), []);
    // nothing says what the document is read as
    assert_eq!(check(&format!("{head}fn g() {{ let s = @embed(\"states.quon\"); }}\n")), [Code::Es06]);
    // a field the document has no state for
    let more = "#unit quantum\nimport quon;\nstruct More { bell: quon::State, plus: quon::State, both: quon::State, other: quon::State }\n";
    assert_eq!(check(&format!("{more}let S: const More = @embed(\"states.quon\");\n")), [Code::Et01]);
    // a state as wide as its type says, and a unit vector
    let one = "#unit quantum\nimport quon;\nstruct One { one: quon::State }\n";
    assert_eq!(check(&format!("{one}let S: const One = @embed(\"wide.quon\");\n")), [Code::Es06]);
    assert_eq!(check(&format!("{one}let S: const One = @embed(\"long.quon\");\n")), [Code::Ej14]);
    // a state stated against a quantum enumeration is of its register: the
    // tag, then the widest variant's payload, here a qubit and a `bool`
    std::fs::write(dir.path().join("variants.quon"), "one: QB = isq2 * |000> + isq2 * |110>;\n").unwrap();
    std::fs::write(dir.path().join("narrow.quon"), "one: QB = |00>;\n").unwrap();
    std::fs::write(dir.path().join("plain.quon"), "one: One = |0>;\n").unwrap();
    let qb = "enum QB { Zero(qubit), One(qubit, bool) }\n";
    assert_eq!(check(&format!("{one}{qb}let S: const One = @embed(\"variants.quon\");\n")), []);
    assert_eq!(check(&format!("{one}{qb}let S: const One = @embed(\"narrow.quon\");\n")), [Code::Es06]);
    assert_eq!(check(&format!("{one}let S: const One = @embed(\"plain.quon\");\n")), [Code::Es06]);
}

/////////////////////////////////////////////////////////////////
// COVERS, GAUGES, BASE TYPES, LOCALES, CHAINS AND MAP LOCALES //
/////////////////////////////////////////////////////////////////

const GEOMETRY: &str = "cover B2 = fin{ |00>, |11> };\n\
                        gauge GB = fid((|00> + |11>) * isq2);\n\
                        base Bell = [qubit; 2], B2, GB;\n\
                        cover Unmarked = span{ |00>, |01>, |10> };\n\
                        gauge GNone = none;\n\
                        base UM = [qubit; 2], Unmarked, GNone;\n\
                        cover All = span{ |00>, |01>, |10>, |11> };\n\
                        base Search = [qubit; 2], All, GNone;\n\
                        cover Marked = pt(|11>);\n\
                        base M = [qubit; 2], Marked, GNone;\n";

#[test]
fn the_declared_geometry_of_the_examples_is_well_formed() {
    let locale = "locale MarkedIn = M of Search { fn reflect(r: *[qubit; 2]); fn swap(r: [qubit; 2], k: u32) -> [qubit; 2]; }\n";
    assert_eq!(gated(&format!("{GEOMETRY}{locale}")), []);
    let chain = "cover C01 = fin{ |0>, |1> };\ncover Cpm = fin{ isq2 * |0> + isq2 * |1>, isq2 * |0> - isq2 * |1> };\n\
                 gauge GPlus = fid(isq2 * |0> + isq2 * |1>);\ngauge GZero = fid(|0>);\n\
                 base T01 = [qubit; 1], C01, GPlus;\nbase Tpm = [qubit; 1], Cpm, GZero;\n\
                 chain HXH = h : T01 -> Tpm ; x : Tpm -> Tpm ; h : Tpm -> T01 ;\n";
    assert_eq!(gated(chain), []);
    assert_eq!(gated(&format!("{chain}chain Broken = h : T01 -> Tpm ; h : T01 -> Tpm ;\n")), [Code::Es06]);
}

#[test]
fn ej09_one_fiducial_on_a_subspace() {
    assert_eq!(gated(&format!("{GEOMETRY}gauge G1 = fid(|00>);\nbase B = [qubit; 2], Unmarked, G1;\n")), [Code::Ej09]);
    // an atlas reaching every point is a gauge on it
    assert_eq!(
        gated(&format!("{GEOMETRY}gauge A = atlas{{ fid(|00>), fid(|01>), fid(|10>) }};\nbase B = [qubit; 2], Unmarked, A;\n")),
        []
    );
}

#[test]
fn ej10_a_point_no_fiducial_reaches() {
    assert_eq!(gated(&format!("{GEOMETRY}gauge G0 = fid(|00>);\nbase B = [qubit; 2], B2, G0;\n")), [Code::Ej10]);
    assert_eq!(
        gated(&format!("{GEOMETRY}gauge A = atlas{{ fid(|00>), fid(|01>) }};\nbase B = [qubit; 2], Unmarked, A;\n")),
        [Code::Ej10]
    );
}

#[test]
fn ej11_a_restriction_that_is_empty_or_mixes_widths() {
    assert_eq!(gated(&format!("{GEOMETRY}locale L = M of UM {{ }}\n")), [Code::Ej11]);
    let one = "cover C = pt(|0>);\nbase One = qubit, C, GNone;\n";
    assert_eq!(gated(&format!("{GEOMETRY}{one}locale L = One of Search {{ }}\n")), [Code::Ej11]);
}

#[test]
fn ej12_a_locale_member_that_is_not_an_interface_of_its_container() {
    assert_eq!(gated(&format!("{GEOMETRY}locale L = M of Search {{ fn f(a: *[qubit; 2], b: *[qubit; 2]); }}\n")), [Code::Ej12]);
    assert_eq!(gated(&format!("{GEOMETRY}locale L = M of Search {{ fn f(a: *[qubit; 1]); }}\n")), [Code::Ej12]);
    assert_eq!(gated(&format!("{GEOMETRY}locale L = M of Search {{ fn f(a: [qubit; 2]); }}\n")), [Code::Ej12]);
}

#[test]
fn ej15_a_cover_presented_badly() {
    assert_eq!(gated("cover D = fin{ |0>, i * |0> };\n"), [Code::Ej15]);
    assert_eq!(gated("cover D = span{ |01>, |01> };\n"), [Code::Ej15]);
    assert_eq!(gated("cover D = code<XI, ZI>;\n"), [Code::Ej15]);
    assert_eq!(gated("cover D = code<XX, ZZ, YY>;\n"), [Code::Ej15]);
    assert_eq!(gated("cover D = fin{ |0>, |01> };\n"), [Code::Ej15]);
    assert_eq!(gated("cover D = fin{ |0> + |1> };\n"), [Code::Ej14]);
    assert_eq!(gated("cover D = code<XX, ZZ>;\ngauge G = fid(isq2 * |00> + isq2 * |11>);\nbase B = [qubit; 2], D, G;\n"), []);
}

#[test]
fn geometry_is_not_a_type_of_values_nor_classical() {
    assert_eq!(gated(&format!("{GEOMETRY}fn f(x: Bell) {{}}\n")), [Code::Es06]);
    assert_eq!(gated("cover A = B;\ncover B = A;\n"), [Code::Es16]);
    assert_eq!(unit_codes("#unit classical\ncover C = pt(|0>);\n"), [Code::Eu02]);
}

#[test]
fn the_topology_of_a_base_type_is_asked_of_it() {
    let sphere = "cover Sphere = span{ |0>, |1> };\ngauge Two = atlas{ fid(|0>), fid(|1>) };\nbase S = qubit, Sphere, Two;\n\
                  cover Pair = fin{ |0>, |1> };\ngauge Plus = fid(isq2 * |0> + isq2 * |1>);\nbase P = qubit, Pair, Plus;\n";
    let checks = "fn f() {\n\
                  @static_assert(@transition_class(S) == -1, \"a sphere\");\n\
                  @static_assert(!@single_patch(S) && @single_patch(P), \"one patch\");\n\
                  @static_assert(@nerve(S).len() == 3 && @nerve(P).len() == 1, \"nerves\");\n\
                  @static_assert(@ortho_graph(P).len() == 1, \"one orthogonal pair\");\n}\n";
    assert_eq!(gated(&format!("{sphere}{checks}")), []);
    assert_eq!(gated(&format!("{sphere}fn g() {{ let e = @ortho_graph(S); }}\n")), [Code::Es06]);
}

const TABLE: &str = "fn id1(t: *[qubit; 1]) {}\nfn neg1(t: *[qubit; 1]) { z(&t[0]); x(&t[0]); z(&t[0]); x(&t[0]); }\n\
                     qmap Phases: [qubit; 2] -> fn(*[qubit; 1]) { 0: id1, 1: id1, 2: id1, 3: neg1, }\n";

#[test]
fn a_map_locale_is_queried_and_measured() {
    let query = "fn mark(key: *[qubit; 2], t: *[qubit; 1]) { query Phases[key](t); }\n";
    assert_eq!(gated(&format!("{TABLE}{query}")), []);
    let lookup = "fn read(key: [qubit; 2], t: *[qubit; 1]) -> usize { let (i, f) = measure Phases[key]; f(t); i }\n";
    assert_eq!(gated(&format!("{TABLE}{lookup}")), []);
}

#[test]
fn a_block_declares_geometry_of_its_own() {
    let local = "fn outer(q: *qubit) {\n\
                     cover C01 = fin{ |0>, |1> };\n\
                     gauge GPlus = fid(isq2 * |0> + isq2 * |1>);\n\
                     base T01 = qubit, C01, GPlus;\n\
                     [cover: T01]\n[expect: contract = CLASS]\n\
                     fn flip(q: *qubit) { x(q); }\n\
                     flip(q);\n\
                 }\n";
    assert_eq!(judged(&local.replace("CLASS", "rigid")), []);
    assert_eq!(judged(&local.replace("CLASS", "stat")), [Code::Ej03]);
    // a block's geometry is its own: outside the block nothing names it,
    // and inside, it hides the unit's of the same name
    assert_eq!(judged(&format!("{}[cover: T01]\nfn g(q: *qubit) {{ }}\n", local.replace("CLASS", "rigid"))), [Code::Es04]);
    let hides = "cover C01 = fin{ |0> };\n";
    assert_eq!(judged(&format!("{hides}{}", local.replace("CLASS", "rigid"))), []);
    // a map locale, queried in the block that declares it
    let table = "fn id1(t: *[qubit; 1]) {}\nfn neg1(t: *[qubit; 1]) { z(&t[0]); x(&t[0]); z(&t[0]); x(&t[0]); }\n\
                 fn mark(key: *[qubit; 2], t: *[qubit; 1]) {\n\
                     qmap Phases: [qubit; 2] -> fn(*[qubit; 1]) { 0: id1, 1: id1, 2: id1, 3: neg1, };\n\
                     query Phases[key](t);\n\
                 }\n";
    assert_eq!(gated(table), []);
}

#[test]
fn eq09_a_map_locale_read_neither_way() {
    assert_eq!(gated(&format!("{TABLE}fn f(key: *[qubit; 2], t: *[qubit; 1]) {{ Phases[key](t); }}\n")), [Code::Eq09]);
    // a map locale held as a value is read the same two ways
    assert_eq!(gated(&format!("{TABLE}fn f() {{ let p = Phases; }}\n")), []);
    let held = "fn mark(m: qmap<[qubit; 2], fn(*[qubit; 1])>, key: *[qubit; 2], t: *[qubit; 1]) { query m[key](t); }\n\
                fn g(key: *[qubit; 2], t: *[qubit; 1]) { mark(Phases, key, t); }\n";
    assert_eq!(gated(&format!("{TABLE}{held}")), []);
    assert_eq!(
        gated(&format!("{TABLE}fn f(m: qmap<[qubit; 2], fn(*[qubit; 1])>, key: *[qubit; 2], t: *[qubit; 1]) {{ m[key](t); }}\n")),
        [Code::Eq09]
    );
    assert_eq!(gated("fn f(m: qmap<[qubit; 2], u32>) { }\n"), [Code::Es06]);
    assert_eq!(unit_codes("#unit classical\nfn f(m: qmap<[qubit; 2], fn(*[qubit; 1])>) { }\n"), [Code::Eu02]);
}

#[test]
fn a_map_locale_of_states_prepares_them() {
    let load = "qmap Load: [qubit; 1] -> [qubit; 2] { 0: prep |00>, 1: prep isq2 * |00> + isq2 * |11>, }\n";
    // coherently, each entry's state is prepared under its key's control
    assert_eq!(gated(&format!("{load}fn f(k: *[qubit; 1]) -> [bool; 2] {{ let r = query Load[k]; measure r }}\n")), []);
    // classically, the key measured chooses the state prepared
    assert_eq!(
        gated(&format!("{load}fn f(k: [qubit; 1]) -> usize {{ let (i, r) = measure Load[k]; let m = measure r; i }}\n")),
        []
    );
    // an entry is a state prepared, as wide as the entries are
    assert_eq!(gated("qmap L: [qubit; 1] -> [qubit; 2] { 0: prep |0>, }\n"), [Code::Es06]);
    assert_eq!(gated("qmap L: [qubit; 1] -> [qubit; 2] { 0: 3, }\n"), [Code::Es06]);
}

#[test]
fn a_map_locale_s_keys_and_entries_are_checked() {
    let id = "fn id1(t: *[qubit; 1]) {}\n";
    assert_eq!(gated(&format!("{id}qmap P: [qubit; 1] -> fn(*[qubit; 1]) {{ 2: id1 }}\n")), [Code::Ec07]);
    assert_eq!(gated(&format!("{id}qmap P: [qubit; 1] -> fn(*[qubit; 1]) {{ 0: id1, 0: id1 }}\n")), [Code::Es06]);
    let take = "fn take(t: [qubit; 1]) -> [qubit; 1] { t }\n";
    assert_eq!(gated(&format!("{take}qmap P: [qubit; 1] -> fn([qubit; 1]) -> [qubit; 1] {{ 0: take }}\n")), [Code::Es06]);
}

////////////////
// JUDGEMENTS //
////////////////

/// A quantum unit importing `gates` and `judge`.
fn judged(body: &str) -> Vec<Code> {
    quantum_codes(&format!("import gates;\nimport judge;\n{body}"))
}

/// The text of the first diagnostic of `code` a quantum unit importing
/// `gates` reports: its message and notes.
fn judged_text(body: &str, code: Code) -> String {
    let mut session = Session::new();
    let id = session.add("demo.tq", &format!("#unit quantum\nimport gates;\n{body}"));
    let out = compile(
        &mut session,
        id,
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    );
    let d = out.diagnostics.iter().find(|d| d.code == code).expect("reported");
    format!("{d:?}")
}

const PAIR: &str = "cover C01 = fin{ |0>, |1> };\ncover Cpm = fin{ isq2 * |0> + isq2 * |1>, isq2 * |0> - isq2 * |1> };\n\
                    gauge GPlus = fid(isq2 * |0> + isq2 * |1>);\ngauge GZero = fid(|0>);\n\
                    base T01 = qubit, C01, GPlus;\nbase Tpm = qubit, Cpm, GZero;\n";

/// The repetition code on twelve qubits, too wide to write out, with the
/// state fixed by it and X on every qubit, and base types of the two.
fn wide_codes() -> String {
    let width = 12;
    let zz: Vec<String> = (0..width - 1)
        .map(|i| (0..width).map(|q| if q == i || q == i + 1 { 'Z' } else { 'I' }).collect())
        .collect();
    let (zeros, ones) = ("0".repeat(width), "1".repeat(width));
    format!(
        "cover Rep = code<{zz}>;\ngauge Free = none;\nbase R = [qubit; 12], Rep, Free;\n\
         cover Ghz = code<{all}, {zz}>;\ngauge GG = fid(isq2 * |{zeros}> + isq2 * |{ones}>);\n\
         base G = [qubit; 12], Ghz, GG;\n",
        zz = zz.join(", "),
        all = "X".repeat(width),
    )
}

#[test]
fn a_code_too_wide_to_write_out_is_judged_on_the_stabilizer_path() {
    let codes = wide_codes();
    // Z on two qubits is the identity on the code, and X on every qubit
    // keeps the GHZ state rigidly
    assert_eq!(
        judged(&format!("{codes}[cover: R]\n[expect: contract = stat(phase::<8>::of(0))]\nfn f(r: *[qubit; 12]) {{ z(&r[0]); z(&r[5]); }}\n")),
        []
    );
    assert_eq!(
        judged(&format!(
            "{codes}[cover: G]\n[expect: contract = rigid]\nfn f(r: *[qubit; 12]) {{ for i in 0..12 {{ x(&r[i]); }} }}\n"
        )),
        []
    );
    // X on one qubit carries the code out of itself
    assert_eq!(judged(&format!("{codes}[cover: R]\nfn f(r: *[qubit; 12]) {{ x(&r[0]); }}\n")), [Code::Ej03]);
    // a global phase makes it stationary at that phase, not rigid
    assert_eq!(
        judged(&format!("{codes}[cover: R]\n[expect: contract = stat(phase::<8>::of(0))]\nfn f(r: *[qubit; 12]) {{ gphase(phase::<8>::of(2)); }}\n")),
        [Code::Ej03]
    );
    // restricting one code to another keeps what both fix, which may be
    // nothing; a fiducial orthogonal to the GHZ state reaches none of it
    assert_eq!(gated(&format!("{codes}locale L = G of R {{ }}\n")), []);
    let odd = "cover Odd = code<-ZZIIIIIIIIII>;\nbase O = [qubit; 12], Odd, Free;\n";
    assert_eq!(gated(&format!("{codes}{odd}locale L = O of R {{ }}\n")), [Code::Ej11]);
    let zeros = "0".repeat(12);
    assert_eq!(
        gated(&format!("{codes}gauge Off = fid(isq2 * |{zeros}> - isq2 * |{}>);\nbase H = [qubit; 12], Ghz, Off;\n", "1".repeat(12))),
        [Code::Ej10]
    );
}

#[test]
fn ej05_the_t_refusal_with_both_witnesses() {
    let bad = "[cover: fin{ |0>, |1> }] [gauge: fid((|0> + |1>) * isq2)]\n[expect: frame(flat(0))]\nfn bad(q: *qubit) { t(q); }\n";
    assert_eq!(judged(bad), [Code::Ej05]);
    let text = judged_text(bad, Code::Ej05);
    assert!(text.contains("schedule is (0, π/4), not constant"), "{text}");
    assert!(text.contains("(|00> + |11>) * isq2 gives (|00> + w(1, 8) * |11>) * isq2"), "{text}");
    let fine = "[cover: fin{ |0>, |1> }] [gauge: fid((|0> + |1>) * isq2)]\n[expect: frame(flat(0))]\nfn fine(q: *qubit) { x(q); }\n";
    assert_eq!(judged(fine), []);
}

#[test]
fn ej03_a_contract_the_certificate_refutes() {
    assert_eq!(judged(&format!("{PAIR}[cover: T01]\n[expect: contract = rigid]\nfn f(q: *qubit) {{ x(q); }}\n")), []);
    assert_eq!(judged(&format!("{PAIR}[cover: T01]\n[expect: contract = flat]\nfn f(q: *qubit) {{ z(q); }}\n")), [Code::Ej03]);
    assert_eq!(judged(&format!("{PAIR}[cover: T01]\n[expect: contract = cyc(0, 4)]\nfn f(q: *qubit) {{ z(q); }}\n")), []);
    // one gate, two behaviours: x is rigid over the computational pair, and
    // stationary with (0, π) over the Hadamard pair
    assert_eq!(judged(&format!("{PAIR}[cover: Tpm]\n[expect: contract = stat]\nfn f(q: *qubit) {{ x(q); }}\n")), [Code::Ej03]);
    assert_eq!(judged(&format!("{PAIR}[cover: Tpm]\n[expect: contract = cyc(0, 4)]\nfn f(q: *qubit) {{ x(q); }}\n")), []);
    // a global phase is stationary, and not rigid
    assert_eq!(
        judged(&format!("{PAIR}[cover: T01]\n[expect: stat(phase::<8>::of(4))]\nfn f(q: *qubit) {{ gphase(phase::<8>::of(4)); }}\n")),
        []
    );
    assert_eq!(
        judged(&format!("{PAIR}[cover: T01]\n[expect: contract = rigid]\nfn f(q: *qubit) {{ gphase(phase::<8>::of(4)); }}\n")),
        [Code::Ej03]
    );
    // an operator that leaves its cover does not hold over it
    assert_eq!(judged(&format!("{PAIR}[cover: T01]\nfn f(q: *qubit) {{ h(q); }}\n")), [Code::Ej03]);
    // monic, unitary, and an instrument's outcomes
    assert_eq!(judged("[expect: monic, unitary]\nfn f(q: *qubit) { h(q); }\n"), []);
    assert_eq!(judged("[expect: monic]\nfn f() -> bool { let q: qubit; h(&q); measure q }\n"), [Code::Ej03]);
    assert_eq!(judged("[expect: kernel.outcomes = bool]\nfn f() -> bool { let q: qubit; h(&q); measure q }\n"), []);
    assert_eq!(judged("[expect: kernel.outcomes = u8]\nfn f() -> bool { let q: qubit; h(&q); measure q }\n"), [Code::Ej03]);
}

#[test]
fn ej01_an_unknown_field_satisfies_no_claim() {
    assert_eq!(judged("[expect: monic]\nfn f(g: fn(*qubit), q: *qubit) { g(q); }\n"), [Code::Ej01]);
    assert_eq!(judged("[expect: contract = rigid]\nfn f(r: *[qubit; 9]) { }\n"), [Code::Ej01]);
}

#[test]
fn a_query_is_a_sector_judgment_and_a_failed_glue_keeps_its_profile() {
    let table = "fn id1(t: *[qubit; 1]) {}\nfn neg1(t: *[qubit; 1]) { gphase(phase::<8>::of(4)); }\n\
                 qmap Phases: [qubit; 2] -> fn(*[qubit; 1]) { 0: id1, 1: id1, 2: id1, 3: neg1, }\n";
    let mark = "fn mark(key: *[qubit; 2], t: *[qubit; 1]) { query Phases[key](t); }\n";
    assert_eq!(judged(&format!("{table}[expect: sector]\n{mark}")), []);
    assert_eq!(judged(&format!("{table}[expect: sector(0, 0, 0, 4)]\n{mark}")), []);
    assert_eq!(judged(&format!("{table}[expect: sector(0, 0, 0, 0)]\n{mark}")), [Code::Ej03]);
    assert_eq!(judged(&format!("{table}[glue: piecewise]\n{mark}")), [Code::Ej06]);
    let text = judged_text(&format!("{table}[glue: piecewise]\n{mark}"), Code::Ej06);
    assert!(text.contains("(0, 0, 0, π)"), "{text}");
}

#[test]
fn ej07_a_chain_blames_its_middle_stage_and_holonomies_are_constants() {
    let chain = "chain HXH = h : T01 -> Tpm ; x : Tpm -> Tpm ; h : Tpm -> T01 ;\n";
    let holds = "@static_assert(@holonomy(HXH, |0>) == phase::<8>::of(0), \"\");\n\
                 @static_assert(@holonomy(HXH, |1>) == phase::<8>::of(4), \"\");\n";
    assert_eq!(judged(&format!("{PAIR}{chain}{holds}")), []);
    assert_eq!(
        judged(&format!("{PAIR}{chain}@static_assert(@holonomy(HXH, |1>) == phase::<8>::of(0), \"no\");\n")),
        [Code::Em01]
    );
    let expect = format!("[expect: holonomy(1) = phase::<8>::of(0)]\n{chain}");
    assert_eq!(judged(&format!("{PAIR}{expect}")), [Code::Ej07]);
    let text = judged_text(&format!("{PAIR}{expect}"), Code::Ej07);
    assert!(text.contains("stage 2 (`x`)"), "{text}");
    // rebuilding the endpoint with the fiducial |-> leaves the holonomies
    // fixed
    let minus = PAIR.replace("gauge GPlus = fid(isq2 * |0> + isq2 * |1>);", "gauge GPlus = fid(isq2 * |0> - isq2 * |1>);");
    assert_eq!(judged(&format!("{minus}{chain}{holds}")), []);
}

#[test]
fn the_grover_iterate_is_cyclic_over_its_orbit_with_holonomy_pi() {
    let src = "cover Orb = fin{ (|00> + |01> + |10> + |11>) / 2, |11>, (-|00> - |01> - |10> + |11>) / 2 };\n\
               gauge GOrb = fid((|00> + |01> + |10> + |11>) / 2);\nbase Torb = [qubit; 2], Orb, GOrb;\n\
               fn oracle(r: *[qubit; 2]) { cz(&r[0], &r[1]); }\n\
               fn diffuse(r: *[qubit; 2]) { h(&r[0]); h(&r[1]); x(&r[0]); x(&r[1]); cz(&r[0], &r[1]); \
               x(&r[0]); x(&r[1]); h(&r[0]); h(&r[1]); gphase(phase::<8>::of(4)); }\n\
               [cover: Orb] [gauge: GOrb]\n[expect: contract = cyc(0, 4, 0)]\nfn iterate(r: *[qubit; 2]) { oracle(r); diffuse(r); }\n\
               chain Loop = iterate : Torb -> Torb ;\n\
               @static_assert(@holonomy(Loop, |11>) == phase::<8>::of(4), \"loop holonomy is pi\");\n";
    assert_eq!(judged(src), []);
    assert_eq!(judged(&src.replace("[expect: contract = cyc(0, 4, 0)]", "[expect: contract = flat]")), [Code::Ej03]);
}

#[test]
fn the_deutsch_oracles_compose_by_the_cocycle_law() {
    // Z followed by the global phase π has the schedules of NEGZ
    let cover = "[cover: fin{ |0> ** (|0> - |1>) * isq2, |1> ** (|0> - |1>) * isq2 }]\n[gauge: fid((|0> + |1>) ** (|0> - |1>) / 2)]\n";
    let composite = format!("{cover}[expect: contract = cyc(4, 0)]\nfn negz(x: *qubit, y: *qubit) {{ ORACLE_Z(x, y); gphase(phase::<8>::of(4)); }}\n");
    assert_eq!(judged(&composite), []);
    let checks = "@static_assert(match @judgment(ORACLE_Z).monic { judge::Tri::Yes => true, _ => false }, \"monic\");\n";
    assert_eq!(judged(checks), []);
}

#[test]
fn judgment_macros_answer_once_the_circuits_exist() {
    let src = "fn tt(q: *qubit) { t(q); }\nfn hh(q: *qubit) { h(q); }\nfn coin() -> bool { let r: qubit; h(&r); measure r }\n\
               @static_assert(@certificate(tt).schedule[1] == phase::<48>::of(6), \"t's schedule\");\n\
               @static_assert(match @judgment(coin).monic { judge::Tri::No => true, _ => false }, \"a coin is not monic\");\n\
               @static_assert(@kernel(coin).measured == 1, \"one measured\");\n\
               @static_assert(@matrix_of(hh).e[1][1] == -isq2, \"h's corner\");\n\
               @static_assert(@fragment_of(tt).inside && !@fragment_of(tt).stabilizer, \"t is in the fragment\");\n\
               @static_assert(@bargmann(|0>, isq2 * |0> + isq2 * |1>, isq2 * |0> + isq2 * i * |1>) == phase::<8>::of(1), \"one octant\");\n";
    assert_eq!(judged(src), []);
    // an operator's body reads a judgement as a value fixed while its
    // circuit is generated, and chooses by it
    assert_eq!(judged("fn hh(q: *qubit) { h(q); }\nfn g(q: *qubit) { let j = @judgment(hh); }\n"), []);
    let chooses = "fn hh(q: *qubit) { h(q); }\nfn tt(q: *qubit) { t(q); }\n\
                   [expect: contract = rigid]\n[cover: fin{ |0>, |1> }] [gauge: fid(isq2 * |0> + isq2 * |1>)]\n\
                   fn g(q: *qubit) { if @fragment_of(tt).stabilizer { tt(q); } else { x(q); } }\n";
    assert_eq!(judged(chooses), []);
    // a judgement that depends on the reading operator's own circuit
    assert_eq!(judged("fn g(q: *qubit) { if @fragment_of(g).stabilizer { h(q); } }\n"), [Code::Ej18]);
    assert_eq!(
        judged("fn a(q: *qubit) { let j = @judgment(b); }\nfn b(q: *qubit) { let j = @judgment(a); }\n"),
        [Code::Ej18]
    );
    assert_eq!(quantum_codes("fn hh(q: *qubit) { }\nfn g() { let j = @judgment(hh); }\n"), [Code::Es04]);
}

#[test]
fn eq07_and_the_frame_family() {
    let two = "[frame: a ** (b ** c)]\n[frame: (a ** b) ** c]\nfn g(a: *qubit, b: *qubit, c: *qubit) { }\n";
    assert_eq!(judged(two), [Code::Eq07]);
    assert_eq!(judged("[frame: a ** (b ** c)]\nfn g(a: *qubit, b: *qubit, c: *qubit) { }\n"), []);
    assert_eq!(judged("[expect: frame(cyc)]\nfn g(q: *qubit) { }\n"), [Code::Ea05]);
}

#[test]
fn an_iso_sum_s_witnesses_compose() {
    assert_eq!(judged("[iso: x]\nenum QBit { Zero(qubit), One(qubit) }\n"), []);
    assert_eq!(judged("[iso: x, z, x]\nenum Three { A(qubit), B(qubit), C(qubit) }\n"), [Code::Ej03]);
    assert_eq!(judged("[iso: x, x]\nenum QBit { Zero(qubit), One(qubit) }\n"), [Code::Ea05]);
}

#[test]
fn ej17_a_step_outside_the_declared_fragment() {
    let src = "#unit quantum\n#pragma fragment(stabilizer)\nimport gates;\nfn f(q: *qubit) { t(q); }\n";
    assert_eq!(unit_codes(src), [Code::Ej17]);
    assert_eq!(unit_codes(&src.replace("t(q)", "s(q)")), []);
}

#[test]
fn a_unit_scope_assertion_is_checked() {
    assert_eq!(unit_codes("#unit classical\n@static_assert(1 + 1 == 2, \"sum\");\n"), []);
    assert_eq!(unit_codes("#unit classical\n@static_assert(1 + 1 == 3, \"sum\");\n"), [Code::Em01]);
    // a binding the condition's own pattern makes
    let bound = "#unit classical\nlet N: const i32? = Some(3);\n@static_assert(match N { Some(v) => v == 3, None => false }, \"bound\");\n";
    assert_eq!(unit_codes(bound), []);
    assert_eq!(unit_codes(&bound.replace("v == 3", "v == 4")), [Code::Em01]);
}

///////////////////
//
// Running circuits:
//   `qpu` and the circuit algebra.
//
///////////////////

/// What checking `user`, a classical unit beside a quantum unit `qk`,
/// reports: its errors' identifiers.
fn running(user: &str) -> Vec<Code> {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("qk.tq"),
        "#unit quantum\nimport gates;\n\
         [entry]\nfn flip(q: *qubit) { x(q); }\n\
         [entry]\nfn run(n: u32, f: fn(*qubit), q: [qubit; 1]) -> ([qubit; 1], bool) {\n\
             f(&q[0]);\n\
             let r: [qubit; 1] = prep |0>;\n\
             let m = measure r;\n\
             (q, m[0])\n\
         }\n\
         [entry]\nfn peek(q: *qubit) { let t: qubit; cx(q, &t); let m = measure t; }\n\
         [entry]\nfn kick(f: fn(*qubit), q: *qubit) { aux let a: qubit; cx(q, &a); f(&a); }\n",
    )
    .unwrap();
    let main = dir.path().join("user.tq");
    std::fs::write(&main, user).unwrap();
    let loaded = topiq::driver::program::load(
        &[main],
        &[],
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    )
    .unwrap();
    let mut codes: Vec<Code> = loaded.diagnostics.iter().map(|d| d.code).collect();
    for m in &loaded.members {
        if let topiq::driver::program::MemberKind::Source(out) = &m.kind {
            codes.extend(out.diagnostics.iter().filter(|d| d.is_error()).map(|d| d.code));
        }
    }
    codes
}

#[test]
fn a_run_takes_the_circuits_arguments_in_its_order() {
    let body = |call: &str| {
        format!(
            "#unit classical\nimport qk;\n\
             fn main() -> i32 {{ let dev = qpu::connect(\"sim://exact\").unwrap(); let r = dev.register(1); {call}; 0 }}\n"
        )
    };
    // a value, a circuit for the operator, a register for the qubits; the
    // kernel is the classical part of what the operator returns
    assert_eq!(running(&body("let b: bool = dev.run(qk::run, 3, qk::flip, r).unwrap()")), []);
    assert_eq!(running(&body("let b: bool = qk::run(3, qk::flip, r)")), []);
    assert_eq!(running(&body("dev.run(qk::run, 3, qk::flip)")), [Code::Es07]);
    assert_eq!(running(&body("dev.run(qk::run, true, qk::flip, r)")), [Code::Es06]);
    assert_eq!(running(&body("dev.run(qk::run, 3, 4, r)")), [Code::Es06]);
    assert_eq!(running(&body("dev.run(5, 3)")), [Code::Es06]);
    // an operator the circuit uncomputes through is monic, which a handle
    // named where it is supplied is checked for then
    assert_eq!(running(&body("dev.run(qk::kick, qk::flip, r).unwrap()")), []);
    assert_eq!(running(&body("dev.run(qk::kick, qk::peek, r).unwrap()")), [Code::Eq02]);
    assert_eq!(running(&body("qk::kick(qk::peek, r)")), [Code::Eq02]);
}

#[test]
fn a_controlled_circuit_takes_its_control_first() {
    let src = "#unit classical\nimport qk;\n\
               fn main() -> i32 { let c: circuit<fn(*qubit, *qubit)> = circuit::controlled(qk::flip); 0 }\n";
    assert_eq!(running(src), []);
    assert_eq!(running(&src.replace("circuit<fn(*qubit, *qubit)>", "circuit<fn(*qubit)>")), [Code::Es06]);
    assert_eq!(running(&src.replace("circuit::controlled(qk::flip)", "circuit::controlled(3)")), [Code::Es06]);
}

#[test]
fn ej13_an_estimate_where_a_claim_is_made() {
    let src = "#unit quantum\nimport gates;\n\
               [expect: contract = stat(dev.estimate_stat(c, 0, 3).unwrap())]\nfn f(q: *qubit) { s(q); }\n";
    assert_eq!(unit_codes(src), [Code::Ej13]);
}

////////////////////////
// THE QUANTUM LIMITS //
////////////////////////

#[test]
fn em05_qubits_for_matrix_of() {
    let at = |n: u32| {
        format!(
            "fn op(r: *[qubit; {n}]) {{ }}\n@static_assert(@matrix_of(op).e[0][0] == @matrix_of(op).e[0][0], \"the matrix\");\n"
        )
    };
    assert_eq!(judged(&at(3)), []);
    assert!(judged(&at(9)).contains(&Code::Em05));
}

/// Every basis ket of `width` qubits, the first `count` of them.
fn kets(width: u32, count: u32) -> String {
    (0..count)
        .map(|i| format!("|{:0w$b}>", i, w = width as usize))
        .collect::<Vec<_>>()
        .join(", ")
}

#[test]
fn em05_literals_in_a_cover() {
    assert_eq!(quantum_codes(&format!("cover C = fin{{ {} }};\n", kets(8, 256))), []);
    assert_eq!(quantum_codes(&format!("cover C = fin{{ {} }};\n", kets(9, 257))), [Code::Em05]);
}

#[test]
fn em05_patches_in_a_gauge() {
    // a cover of CP¹ needs at most two patches
    let atlas = |fids: &str| format!("cover Line = span{{ |0>, |1> }};\ngauge A = atlas{{ {fids} }};\nbase B = qubit, Line, A;\n");
    assert_eq!(gated(&atlas("fid(|0>), fid(|1>)")), []);
    assert_eq!(gated(&atlas("fid(|0>), fid(|1>), fid((|0> + |1>) * isq2)")), [Code::Em05]);
}

#[test]
fn em05_stages_in_a_chain() {
    let chain = |n: usize| {
        let stages = vec!["x : T01 -> T01 ;"; n].join(" ");
        format!(
            "cover C01 = fin{{ |0>, |1> }};\ngauge GPlus = fid((|0> + |1>) * isq2);\nbase T01 = qubit, C01, GPlus;\nchain L = {stages}\n"
        )
    };
    assert_eq!(gated(&chain(64)), []);
    assert_eq!(gated(&chain(65)), [Code::Em05]);
}

#[test]
fn em05_entries_in_a_qmap() {
    let table = |n: u32| {
        let entries: String = (0..n).map(|k| format!("{k}: id1, ")).collect();
        format!("fn id1(t: *[qubit; 1]) {{ }}\nqmap P: [qubit; 11] -> fn(*[qubit; 1]) {{ {entries} }}\n")
    };
    assert_eq!(gated(&table(1024)), []);
    assert_eq!(gated(&table(1025)), [Code::Em05]);
}

#[test]
fn eu04_a_conductor_an_import_brings_in() {
    // `a` at 16 imports `b` at 8, which imports `c` at 24: `b`'s operators
    // may be made of `c`'s, whose judgements meet `a`'s at 48
    let dir = tempfile::tempdir().unwrap();
    let unit = |name: &str, n: u32, imports: &str| {
        std::fs::write(
            dir.path().join(format!("{name}.tq")),
            format!("#unit quantum\n#pragma conductor({n})\n{imports}import gates;\nfn {name}_op(q: *qubit) {{ h(q); }}\n"),
        )
        .unwrap();
    };
    unit("c", 24, "");
    unit("b", 8, "import c;\n");
    unit("a", 16, "import b;\n");
    let loaded = topiq::driver::program::load(
        &[dir.path().join("a.tq")],
        &[],
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    )
    .unwrap();
    let mut found = Vec::new();
    for m in &loaded.members {
        if let topiq::driver::program::MemberKind::Source(out) = &m.kind {
            found.extend(out.diagnostics.iter().filter(|d| d.is_error()).map(|d| (d.code, d.message.clone())));
        }
    }
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].0, Code::Eu04);
    assert!(found[0].1.contains("through `c`"), "{}", found[0].1);
}

#[test]
fn es01_a_file_that_is_not_utf8() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.tq");
    std::fs::write(&path, b"#unit classical\n\xff\xfe fn f() { }\n").unwrap();
    let loaded = topiq::driver::program::load(
        std::slice::from_ref(&path),
        &[],
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
    )
    .unwrap();
    let codes: Vec<Code> = loaded.all_diagnostics().map(|d| d.code).collect();
    assert_eq!(codes, [Code::Es01]);
    let d = loaded.all_diagnostics().next().unwrap();
    assert!(d.message.contains("offset 16"), "{}", d.message);
}

///////////////////
//
// Units of either kind:
//   `#unit any`, and the kinds asked of it.
//
///////////////////

/// The codes of loading the program whose files are `files`, by name and
/// text, the first given, with `kinds` for the `#unit any` units given.
fn program_codes(files: &[(&str, &str)], kinds: &topiq::driver::KindArgs) -> Vec<Code> {
    let dir = tempfile::tempdir().unwrap();
    for (name, text) in files {
        std::fs::write(dir.path().join(name), text).unwrap();
    }
    let loaded = topiq::driver::program::load_as(
        &[dir.path().join(files[0].0)],
        &[],
        Options {
            stage: Stage::Check,
            ..Options::default()
        },
        kinds,
    )
    .unwrap();
    loaded.all_diagnostics().filter(|d| d.is_error()).map(|d| d.code).collect()
}

#[test]
fn eu07_an_any_unit_whose_kind_nothing_decides() {
    assert_eq!(unit_codes("#unit any\nfn f() { }\n"), [Code::Eu07]);
    assert_eq!(unit_codes("#unit any(both(init))\nfn init() { }\n"), [Code::Eu07]);
    // a preference decides it, or `--kind`
    assert_eq!(unit_codes("#unit any(quantum)\nfn f(q: *qubit) { }\n"), []);
    let quantum = topiq::driver::KindArgs {
        all: Some(topiq::pp::UnitKind::Quantum),
        named: Vec::new(),
    };
    assert_eq!(program_codes(&[("dual.tq", "#unit any\nfn f(q: *qubit) { }\n")], &quantum), []);
}

#[test]
fn eu01_a_malformed_any_directive() {
    for src in [
        "#unit any(maybe)\n",
        "#unit any(classical(a), classical(b))\nfn a() { }\nfn b() { }\n",
        "#unit any(both(a), quantum(b))\nfn a() { }\nfn b() { }\n",
        "#unit any(quantum(a), classical)\nfn a() { }\n",
        "#unit any(classical\n",
        "#unit anything\n",
    ] {
        // the unit is also left without a directive, which is EU01 again
        let codes = unit_codes(src);
        assert!(!codes.is_empty() && codes.iter().all(|&c| c == Code::Eu01), "{src}: {codes:?}");
    }
}

#[test]
fn an_any_unit_runs_the_initialiser_of_its_kind() {
    let src = "#unit any(quantum, classical(c), quantum(q))\nlet K: u32 = 0;\nfn c() { K = 1; }\nfn q() { K = 2; }\n\
               [entry]\nfn read(n: u32) -> u32 { K }\n";
    assert_eq!(unit_codes(src), []);
}

#[test]
fn eu08_a_kind_asked_of_a_unit_of_fixed_kind() {
    let none = topiq::driver::KindArgs::default();
    let fixed = "#unit quantum\nimport gates;\n[entry]\nfn flip(n: u32) -> bool { let q: qubit; x(&q); measure q }\n";
    // even its own kind is refused
    assert_eq!(
        program_codes(&[("main.tq", "#unit classical\nimport(quantum) circ;\nfn main() -> i32 { 0 }\n"), ("circ.tq", fixed)], &none),
        [Code::Eu08]
    );
    assert_eq!(
        program_codes(&[("main.tq", "#unit classical\nimport circ;\nfn main() -> i32 { 0 }\n"), ("circ.tq", fixed)], &none),
        []
    );
    // and `--kind` naming such a unit
    let named = topiq::driver::KindArgs {
        all: None,
        named: vec![("main".to_owned(), topiq::pp::UnitKind::Classical)],
    };
    assert_eq!(program_codes(&[("main.tq", "#unit classical\nfn main() -> i32 { 0 }\n")], &named), [Code::Eu08]);
}

#[test]
fn both_kinds_of_one_unit_need_a_name_each() {
    let dual = "#unit any(classical)\n#alias dit = bool | qubit\nfn read(d: *dit) { }\n";
    let none = topiq::driver::KindArgs::default();
    assert_eq!(
        program_codes(
            &[
                ("main.tq", "#unit classical\nimport dual;\nimport(quantum) dual as qdual;\nfn main() -> i32 { 0 }\n"),
                ("dual.tq", dual),
            ],
            &none
        ),
        []
    );
    let codes = program_codes(
        &[
            ("main.tq", "#unit classical\nimport dual;\nimport(quantum) dual;\nfn main() -> i32 { 0 }\n"),
            ("dual.tq", dual),
        ],
        &none,
    );
    assert_eq!(codes.len(), 1, "{codes:?}");
    // a kind is `classical` or `quantum`
    assert_eq!(
        program_codes(&[("main.tq", "#unit classical\nimport(both) dual;\nfn main() -> i32 { 0 }\n"), ("dual.tq", dual)], &none),
        [Code::Es02]
    );
}
