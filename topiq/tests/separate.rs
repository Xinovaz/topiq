//! Programs built from several pieces.
//!
//! - **Separate compilation:** a unit compiled earlier is used through its
//!   object alone, and an object built against an older version of a unit it
//!   imports is refused when the program is linked.
//! - **Whole-program checks:** initialiser order, methods added to `[open]`
//!   types, exported symbols, and units that import one another.
//! - **Modules:** a module built on its own and loaded by a program while it
//!   runs, checked against it, and called into.

#![cfg(feature = "llvm")]

use std::path::{Path, PathBuf};

use topiq::diag::Code;
use topiq::driver::build::{BuildOptions, build};

fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p
}

fn opts(dir: &Path, link: bool) -> BuildOptions {
    BuildOptions {
        out_dir: dir.to_owned(),
        link,
        ..BuildOptions::default()
    }
}

const GEO: &str = "#unit classical\n\
                   struct P { x: i32 }\n\
                   fn area(p: *P) -> i32 { p.x * 2 }\n\
                   let LIMIT: const i32 = 7;\n";

const APP: &str = "#unit classical\n\
                   import geo;\n\
                   fn main() -> i32 { let p = geo::P { x: 3 }; geo::area(&p) + geo::LIMIT }\n";

/// Builds `app` and `geo` together, then replaces `geo` with `changed` and
/// rebuilds only it, then links the two objects. Returns what the link
/// reported.
fn relink(changed: &str) -> Vec<Code> {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "geo.tq", GEO);
    let app = write(dir.path(), "app.tq", APP);
    let first = build(&[app], &opts(dir.path(), true)).unwrap();
    assert!(!first.has_errors(), "{:?}", first.all_diagnostics().collect::<Vec<_>>());

    let geo = write(dir.path(), "geo.tq", changed);
    let second = build(&[geo], &opts(dir.path(), false)).unwrap();
    assert!(!second.has_errors());

    let objects = [dir.path().join("app.tcu"), dir.path().join("geo.tcu")];
    let linked = build(&objects, &opts(dir.path(), true)).unwrap();
    if linked.has_errors() {
        assert!(linked.executable.is_none(), "a refused program is not linked");
    }
    linked.all_diagnostics().map(|d| d.code).collect()
}

#[test]
fn objects_that_still_agree_link_and_run() {
    assert_eq!(relink(GEO), Vec::<Code>::new());
}

#[test]
fn a_changed_structure_is_refused_as_el03() {
    let changed = GEO.replace("struct P { x: i32 }", "struct P { tag: u8, x: i32 }");
    assert_eq!(relink(&changed), [Code::El03]);
}

#[test]
fn a_changed_signature_is_refused_as_el02() {
    let changed = GEO.replace("fn area(p: *P) -> i32", "fn area(p: *const P) -> i32");
    assert_eq!(relink(&changed), [Code::El02]);
}

#[test]
fn a_removed_function_is_refused_as_el06() {
    let changed = GEO.replace("fn area", "static fn area");
    assert_eq!(relink(&changed), [Code::El06]);
}

#[test]
fn a_changed_constant_is_refused_as_el12() {
    let changed = GEO.replace("= 7", "= 8");
    assert_eq!(relink(&changed), [Code::El12]);
}

#[test]
fn the_metadata_stays_in_the_object_and_out_of_the_executable() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "geo.tq", GEO);
    let app = write(dir.path(), "app.tq", APP);
    let program = build(&[app], &opts(dir.path(), true)).unwrap();
    let exe = program.executable.expect("linked");
    let bytes = std::fs::read(&exe).unwrap();
    assert!(
        topiq::meta::object::read_section(&std::fs::read(dir.path().join("geo.tcu")).unwrap()).is_ok(),
        "the object carries it"
    );
    let needle = b"Metadata {";
    assert!(
        !bytes.windows(needle.len()).any(|w| w == needle),
        "the executable does not"
    );
    let status = std::process::Command::new(&exe).status().unwrap();
    assert_eq!(status.code(), Some(13));
}

const GEN: &str = "#unit classical\n\
                   static fn twice(x: i64) -> i64 { x * 2 }\n\
                   fn doubled<T>(x: T) -> T { x + x }\n\
                   fn plus_twice<T>(x: T) -> i64 { twice(x as i64) }\n\
                   struct Pair<A, B> { a: A, b: B }\n\
                   enum Opt<T> { None, Some(T) }\n\
                   fn first<A, B>(p: Pair<A, B>) -> A { p.a }\n\
                   fn cube(n: constexpr i64) -> constexpr i64 { n * n * n }\n";

const OTHER: &str = "#unit classical\n\
                     import gen;\n\
                     fn six() -> i32 { gen::doubled(3i32) }\n";

const USER: &str = "#unit classical\n\
                    import gen;\n\
                    import other;\n\
                    let N: const i64 = gen::cube(2);\n\
                    fn main() -> i32 {\n\
                        let p = gen::Pair { a: 5i32, b: true };\n\
                        let o: gen::Opt<i32> = gen::Opt::Some(other::six());\n\
                        let v = match o { gen::Opt::Some(x) => x, gen::Opt::None => 0 };\n\
                        gen::doubled(10i32) + gen::first(p) + v + (gen::plus_twice(3i32) as i32) + (N as i32)\n\
                    }\n";

fn run(exe: &Path) -> Option<i32> {
    std::process::Command::new(exe).status().unwrap().code()
}

#[test]
fn another_units_generics_are_instantiated_where_they_are_used() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "gen.tq", GEN);
    write(dir.path(), "other.tq", OTHER);
    let user = write(dir.path(), "user.tq", USER);
    let program = build(&[user], &opts(dir.path(), true)).unwrap();
    assert!(!program.has_errors(), "{:?}", program.all_diagnostics().collect::<Vec<_>>());
    // `doubled<i32>` is made in both `user` and `other`; the linker keeps one
    assert_eq!(run(&program.executable.expect("linked")), Some(45));
}

#[test]
fn a_compiled_unit_carries_the_bodies_of_its_generics() {
    let dir = tempfile::tempdir().unwrap();
    let gen_src = write(dir.path(), "gen.tq", GEN);
    let first = build(std::slice::from_ref(&gen_src), &opts(dir.path(), false)).unwrap();
    assert!(!first.has_errors(), "{:?}", first.all_diagnostics().collect::<Vec<_>>());
    std::fs::remove_file(&gen_src).unwrap();
    write(dir.path(), "other.tq", OTHER);
    let user = write(dir.path(), "user.tq", USER);
    let program = build(&[user], &opts(dir.path(), true)).unwrap();
    assert!(!program.has_errors(), "{:?}", program.all_diagnostics().collect::<Vec<_>>());
    assert_eq!(run(&program.executable.expect("linked")), Some(45));
}

#[test]
fn a_changed_generic_body_is_refused_as_el02() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "gen.tq", GEN);
    write(dir.path(), "other.tq", OTHER);
    let user = write(dir.path(), "user.tq", USER);
    let first = build(&[user], &opts(dir.path(), true)).unwrap();
    assert!(!first.has_errors(), "{:?}", first.all_diagnostics().collect::<Vec<_>>());

    let gen_src = write(dir.path(), "gen.tq", &GEN.replace("{ x + x }", "{ x + x + x }"));
    let second = build(&[gen_src], &opts(dir.path(), false)).unwrap();
    assert!(!second.has_errors());

    let objects = ["user.tcu", "other.tcu", "gen.tcu"].map(|n| dir.path().join(n));
    let linked = build(&objects, &opts(dir.path(), true)).unwrap();
    let codes: Vec<Code> = linked.all_diagnostics().map(|d| d.code).collect();
    assert_eq!(codes, [Code::El02, Code::El02], "both importers borrowed the old body");
}

#[test]
fn two_units_adding_one_method_to_an_open_type_is_el07() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "shape.tq", "#unit classical\n[open]\nstruct S { n: i64 }\n");
    let add = |name: &str, body: &str| {
        format!("#unit classical\nimport shape;\nfn shape::S.twice(self: *shape::S) -> i64 {{ {body} }}\nfn {name}() {{ }}\n")
    };
    write(dir.path(), "left.tq", &add("left", "self.n * 2"));
    write(dir.path(), "right.tq", &add("right", "self.n + self.n"));
    let main = write(
        dir.path(),
        "main.tq",
        "#unit classical\nimport left;\nimport right;\nfn main() -> i32 { 0 }\n",
    );
    let program = build(&[main], &opts(dir.path(), true)).unwrap();
    let codes: Vec<Code> = program.all_diagnostics().map(|d| d.code).collect();
    assert_eq!(codes, [Code::El07]);
    assert!(program.executable.is_none());
}

const COUNTED: &str = "#unit classical\n\
                       struct Counter { n: i32 }\n\
                       impl Counter {\n\
                           fn $copy(self: *const Counter) -> Counter { Counter { n: self.n + 1 } }\n\
                       }\n\
                       struct Box<T> { v: T }\n\
                       impl Box<T> {\n\
                           fn $copy(self: *const Box<T>) -> Box<T> { Box { v: self.v } }\n\
                       }\n";

const COPIER: &str = "#unit classical\n\
                      import counted;\n\
                      fn main() -> i32 {\n\
                          let a = counted::Counter { n: 1 };\n\
                          let b = a;\n\
                          let c = b;\n\
                          let x = counted::Box { v: c };\n\
                          let y = x;\n\
                          a.n + y.v.n * 10\n\
                      }\n";

#[test]
fn another_units_type_is_copied_by_its_own_copy_method() {
    let dir = tempfile::tempdir().unwrap();
    let counted = write(dir.path(), "counted.tq", COUNTED);
    let first = build(std::slice::from_ref(&counted), &opts(dir.path(), false)).unwrap();
    assert!(!first.has_errors(), "{:?}", first.all_diagnostics().collect::<Vec<_>>());
    let user = write(dir.path(), "copier.tq", COPIER);
    let program = build(&[user], &opts(dir.path(), true)).unwrap();
    assert!(!program.has_errors(), "{:?}", program.all_diagnostics().collect::<Vec<_>>());
    // `b` and `c` are copies of copies, `x.v` copies `c` once more, and `y`
    // copies `x`, whose `$copy` copies its `v` again: 1 + 5 * 10
    assert_eq!(run(&program.executable.expect("linked")), Some(51));
}

/// Records the order the initialisers run in, one digit each.
const LOG: &str = "#unit classical(start)\n\
                   let L: i32;\n\
                   fn start() { L = 0; }\n\
                   fn note(d: i32) { L = L * 10 + d; }\n";

fn noting(name: &str, digit: u32, imports: &[&str]) -> String {
    let imports: String = imports.iter().map(|i| format!("import {i};\n")).collect();
    format!("#unit classical(init_{name})\nimport log;\n{imports}fn init_{name}() {{ log::note({digit}); }}\n")
}

#[test]
fn initializers_run_after_those_of_the_units_imported_then_by_name() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "log.tq", LOG);
    write(dir.path(), "a.tq", &noting("a", 1, &[]));
    write(dir.path(), "b.tq", &noting("b", 2, &["a"]));
    write(dir.path(), "z.tq", &noting("z", 3, &[]));
    write(dir.path(), "y.tq", &noting("y", 4, &[]));
    let main = write(
        dir.path(),
        "app.tq",
        "#unit classical\nimport log;\nimport z;\nimport y;\nimport b;\nfn main() -> i32 { log::L }\n",
    );
    let program = build(&[main], &opts(dir.path(), true)).unwrap();
    assert!(!program.has_errors(), "{:?}", program.all_diagnostics().collect::<Vec<_>>());
    // `log` first, since every unit imports it; then `a`, which `b` needs;
    // then `b`, `y` and `z` by name
    assert_eq!(run(&program.executable.expect("linked")), Some(1243));
}

#[test]
fn units_with_initializers_importing_one_another_are_el04() {
    let dir = tempfile::tempdir().unwrap();
    let q_first = "#unit classical(init_q)\nlet Q: i32;\nfn init_q() { Q = 1; }\n";
    write(dir.path(), "q.tq", q_first);
    let p = write(
        dir.path(),
        "p.tq",
        "#unit classical(init_p)\nimport q;\nlet P: i32;\nfn init_p() { P = q::Q; }\nfn main() -> i32 { P }\n",
    );
    let first = build(&[p], &opts(dir.path(), false)).unwrap();
    assert!(!first.has_errors(), "{:?}", first.all_diagnostics().collect::<Vec<_>>());
    // `q` now imports `p` too, and is compiled against `p`'s object
    std::fs::remove_file(dir.path().join("p.tq")).unwrap();
    let q = write(
        dir.path(),
        "q.tq",
        "#unit classical(init_q)\nimport p;\nlet Q: i32;\nfn init_q() { Q = 1; }\n",
    );
    let second = build(&[q], &opts(dir.path(), false)).unwrap();
    assert!(!second.has_errors(), "{:?}", second.all_diagnostics().collect::<Vec<_>>());
    let objects = [dir.path().join("p.tcu"), dir.path().join("q.tcu")];
    let linked = build(&objects, &opts(dir.path(), true)).unwrap();
    let codes: Vec<Code> = linked.all_diagnostics().map(|d| d.code).collect();
    assert_eq!(codes, [Code::El04]);
    assert!(linked.executable.is_none());
}

#[test]
fn a_classical_unit_using_a_quantum_units_operators_or_objects_is_eu03() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "qk.tq",
        "#unit quantum\nstruct Reg { q: [qubit; 2] }\nlet N: u32 = 0;\nfn helper(q: *qubit) { }\n\
         [entry]\nfn run(n: u32) -> bool { true }\n",
    );
    let app = write(
        dir.path(),
        "user.tq",
        "#unit classical\nimport qk;\n\
         fn a(r: qk::Reg) { }\n\
         fn b() -> u32 { qk::N }\n\
         fn c() { qk::helper(); }\n\
         fn d() -> bool { qk::run(1) }\n\
         fn main() -> i32 { 0 }\n",
    );
    let program = build(&[app], &opts(dir.path(), true)).unwrap();
    let codes: Vec<Code> = program.all_diagnostics().filter(|d| d.is_error()).map(|d| d.code).collect();
    // the type, the object and the operator are the quantum side's; the
    // entry operator is a circuit, which a call runs
    assert_eq!(codes, [Code::Eu03, Code::Eu03, Code::Eu03]);
}

///////////////////
//
// Modules:
//   Libraries loaded while a program runs.
//
///////////////////

/// Builds `module` (its files in `module_dir`) as a module, then `host` as a
/// program beside it, and runs the program there. Returns its status and
/// what it printed.
fn load_and_run(dir: &Path, module: &[(&str, &str)], host: &[(&str, &str)]) -> (Option<i32>, String) {
    let module_dir = dir.join("module");
    std::fs::create_dir_all(&module_dir).unwrap();
    let files: Vec<PathBuf> = module.iter().map(|(n, t)| write(&module_dir, n, t)).collect();
    let built = build(
        &files[..1],
        &BuildOptions {
            module: true,
            executable: Some(dir.join("plugin.dll")),
            ..opts(&module_dir, true)
        },
    )
    .unwrap();
    assert!(!built.has_errors(), "{:?}", built.all_diagnostics().collect::<Vec<_>>());
    let host_dir = dir.join("host");
    std::fs::create_dir_all(&host_dir).unwrap();
    let files: Vec<PathBuf> = host.iter().map(|(n, t)| write(&host_dir, n, t)).collect();
    let program = build(
        &files[..1],
        &BuildOptions {
            executable: Some(dir.join("host.exe")),
            ..opts(&host_dir, true)
        },
    )
    .unwrap();
    assert!(!program.has_errors(), "{:?}", program.all_diagnostics().collect::<Vec<_>>());
    let out = std::process::Command::new(dir.join("host.exe")).current_dir(dir).output().unwrap();
    (out.status.code(), String::from_utf8_lossy(&out.stdout).into_owned())
}

/// A program that loads `plugin.dll` and prints why it could not, if it
/// could not.
const LOADER: &str = "#unit classical\n\
    fn main() -> i32 {\n\
        match module::load(\"plugin.dll\") {\n\
            Ok(_) => print(\"loaded\"),\n\
            Err(module::Error::LayoutMismatch(t)) => { print(\"layout \"); print(&t); }\n\
            Err(module::Error::DuplicateMethod(m)) => { print(\"method \"); print(&m); }\n\
            Err(module::Error::Malformed) => print(\"malformed\"),\n\
            Err(module::Error::NotFound) => print(\"not found\"),\n\
            Err(_) => print(\"other\"),\n\
        }\n\
        0\n\
    }\n";

#[test]
fn a_module_runs_its_initializers_and_offers_its_functions() {
    let dir = tempfile::tempdir().unwrap();
    let plugin = "#unit classical(setup)\n\
        let SCALE: i64;\n\
        fn setup() { SCALE = 10; }\n\
        [typeinfo]\n\
        struct Shape { side: i64 }\n\
        fn scaled(x: i64) -> i64 { x * SCALE }\n\
        fn greet(name: *[char]) -> [char] {\n\
            let out: [char] = [];\n\
            for c in \"hello, \" { out.push(*c); }\n\
            for c in name { out.push(*c); }\n\
            out\n\
        }\n";
    let host = "#unit classical\n\
        fn main() -> i32 {\n\
            let m = match module::load(\"plugin.dll\") {\n\
                Ok(m) => m,\n\
                Err(_) => { return 1; }\n\
            };\n\
            match m.symbol::<fn(i64) -> i64>(\"plugin::scaled\") {\n\
                Some(f) => print(&tcon::emit(&invoke::<fn(i64) -> i64>(f, 4))),\n\
                None => { return 2; }\n\
            }\n\
            match m.symbol::<fn(*[char]) -> [char]>(\"plugin::greet\") {\n\
                Some(g) => { print(\" \"); print(&invoke::<fn(*[char]) -> [char]>(g, \"host\")); }\n\
                None => { return 3; }\n\
            }\n\
            match m.symbol::<fn(i32) -> i64>(\"plugin::scaled\") {\n\
                Some(_) => { return 4; }\n\
                None => print(\" / refused\"),\n\
            }\n\
            match m.symbol::<fn(i64) -> i64>(\"plugin::nothing\") {\n\
                Some(_) => { return 5; }\n\
                None => print(\" / absent\"),\n\
            }\n\
            match m.typeinfo(\"plugin::Shape\") {\n\
                Some(ti) => { print(\" / \"); print(ti.name); }\n\
                None => { return 7; }\n\
            }\n\
            match m.typeinfo(\"plugin::Nothing\") {\n\
                Some(_) => { return 8; }\n\
                None => print(\" / no table\"),\n\
            }\n\
            match m.unload() {\n\
                Ok(()) => print(\" / unloaded\"),\n\
                Err(_) => { return 6; }\n\
            }\n\
            0\n\
        }\n";
    let (status, out) = load_and_run(dir.path(), &[("plugin.tq", plugin)], &[("host.tq", host)]);
    assert_eq!((status, out.as_str()), (Some(0), "40 hello, host / refused / absent / plugin::Shape / no table / unloaded"));
}

#[test]
fn a_module_laying_a_shared_type_out_differently_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let plugin = "#unit classical\nimport shapes;\nfn area(p: *shapes::P) -> i32 { p.x }\n";
    let shapes = |fields: &str| format!("#unit classical\nstruct P {{ {fields} }}\n");
    let before = shapes("x: i32");
    let after = shapes("tag: u8, x: i32");
    let host = LOADER.replace("#unit classical\n", "#unit classical\nimport shapes;\nfn use_it(p: *shapes::P) -> i32 { p.x }\n");
    let (status, out) = load_and_run(
        dir.path(),
        &[("plugin.tq", plugin), ("shapes.tq", &before)],
        &[("host.tq", &host), ("shapes.tq", &after)],
    );
    assert_eq!((status, out.as_str()), (Some(0), "layout shapes::P"));
}

#[test]
fn a_module_adding_a_method_the_program_adds_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let counting = "#unit classical\n[open]\nstruct Counter { n: i64 }\n";
    let adds = "fn counting::Counter.twice(self: *counting::Counter) -> i64 { self.n * 2 }\n";
    let plugin = format!("#unit classical\nimport counting;\n{adds}");
    let host = LOADER.replace("#unit classical\n", &format!("#unit classical\nimport counting;\n{adds}"));
    let (status, out) = load_and_run(
        dir.path(),
        &[("plugin.tq", &plugin), ("counting.tq", counting)],
        &[("host.tq", &host), ("counting.tq", counting)],
    );
    assert_eq!((status, out.as_str()), (Some(0), "method counting::Counter::twice"));
}

#[test]
fn a_file_that_is_not_a_module_is_malformed_and_nothing_is_not_found() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "plugin.dll", "not a library");
    let host_dir = dir.path().join("host");
    std::fs::create_dir_all(&host_dir).unwrap();
    let host = write(&host_dir, "host.tq", LOADER);
    let program = build(
        &[host],
        &BuildOptions {
            executable: Some(dir.path().join("host.exe")),
            ..opts(&host_dir, true)
        },
    )
    .unwrap();
    assert!(!program.has_errors());
    let run = || {
        let out = std::process::Command::new(dir.path().join("host.exe")).current_dir(dir.path()).output().unwrap();
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    assert_eq!(run(), "malformed");
    std::fs::remove_file(dir.path().join("plugin.dll")).unwrap();
    assert_eq!(run(), "not found");
}

#[test]
fn a_value_made_by_the_program_is_recognized_inside_a_module() {
    let dir = tempfile::tempdir().unwrap();
    let shapes = "#unit classical\nstruct P { x: i32 }\n";
    let plugin = "#unit classical\nimport shapes;\n\
        fn x_of(p: *any) -> i32 {\n\
            match p as *shapes::P { Some(q) => q.x, None => 0 - 1 }\n\
        }\n";
    let host = "#unit classical\nimport shapes;\n\
        fn main() -> i32 {\n\
            let m = match module::load(\"plugin.dll\") { Ok(m) => m, Err(_) => { return 1; } };\n\
            let f = match m.symbol::<fn(*any) -> i32>(\"plugin::x_of\") { Some(f) => f, None => { return 2; } };\n\
            let p = shapes::P { x: 5 };\n\
            let n = 7;\n\
            print(&tcon::emit(&invoke::<fn(*any) -> i32>(f, &p)));\n\
            print(\" \");\n\
            print(&tcon::emit(&invoke::<fn(*any) -> i32>(f, &n)));\n\
            0\n\
        }\n";
    let (status, out) = load_and_run(
        dir.path(),
        &[("plugin.tq", plugin), ("shapes.tq", shapes)],
        &[("host.tq", host), ("shapes.tq", shapes)],
    );
    assert_eq!((status, out.as_str()), (Some(0), "5 -1"));
}

#[test]
fn methods_added_to_another_units_generic_type_need_it_open() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "boxes.tq", "#unit classical\nstruct Shut<T> { v: T }\n");
    let main = write(
        dir.path(),
        "main.tq",
        "#unit classical\nimport boxes;\nimpl boxes::Shut<T> {\n    fn get(self: *boxes::Shut<T>) -> *T { &self.v }\n}\n\
         fn main() -> i32 { 0 }\n",
    );
    let program = build(&[main], &opts(dir.path(), false)).unwrap();
    let codes: Vec<Code> = program.all_diagnostics().map(|d| d.code).collect();
    assert_eq!(codes, [Code::Es19]);
}

#[test]
fn the_data_driven_configuration_example_runs_as_written() {
    // the program binds a document to a type the module offers, with a `main`
    // to run it
    let dir = tempfile::tempdir().unwrap();
    let example = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("config.tq"),
    )
    .unwrap();
    let defaults = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("defaults.tcon"),
    )
    .unwrap();
    let host = format!(
        "{example}\n\
         fn main() -> i32 {{\n\
             print(&ACTIVE.host);\n\
             let m = match module::load(\"plugin.dll\") {{ Ok(m) => m, Err(_) => {{ return 1; }} }};\n\
             match fs::write(\"server.tcon\", \"ServerConfig {{ host: \\\"example.org\\\", port: 443, tls: null }}\") {{\n\
                 Ok(()) => {{}}\n\
                 Err(_) => {{ return 2; }}\n\
             }}\n\
             let c = load(\"server.tcon\");\n\
             print(\" \");\n\
             print(&c.host);\n\
             print(\" \");\n\
             print(&tcon::emit(&load(\"missing.tcon\").port));\n\
             match load_dyn(\"server.tcon\", \"ServerConfig\", &m) {{\n\
                 Ok(d) => {{ print(\" \"); print(&dyn::emit(&d)); }}\n\
                 Err(_) => {{ return 3; }}\n\
             }}\n\
             0\n\
         }}\n"
    );
    let plugin = "#unit classical\n\
        [typeinfo]\n\
        struct ServerConfig { host: [char], port: u16, tls: TlsConfig? }\n\
        struct TlsConfig    { cert: [char], key: [char] }\n";
    let (status, out) = load_and_run(
        dir.path(),
        &[("plugin.tq", plugin)],
        &[("config.tq", &host), ("defaults.tcon", &defaults)],
    );
    assert_eq!(
        (status, out.as_str()),
        (Some(0), "localhost example.org 8080 { host: \"example.org\", port: 443, tls: null }")
    );
}

#[test]
fn a_value_of_a_modules_type_is_made_and_called_by_name() {
    // the example of loading a module by path, finding a type it holds by
    // name, binding a document to it, and calling its method dynamically
    let dir = tempfile::tempdir().unwrap();
    let plugin = "#unit classical\n\
        [typeinfo]\n\
        struct Circle { r: f64 }\n\
        impl Circle { fn area(self: *Circle) -> f64 { 3.0 * self.r * self.r } }\n";
    let host = "#unit classical\n\
        fn main() -> i32 {\n\
            let m = match module::load(\"plugin.dll\") { Ok(m) => m, Err(_) => { return 1; } };\n\
            let ti = match m.typeinfo(\"Circle\") { Some(t) => t, None => { return 2; } };\n\
            let src = \"Circle { r: 2.0 }\";\n\
            let v = match tcon::parse_dyn(src) { Ok(v) => v, Err(_) => { return 3; } };\n\
            let d = match dyn::bind(&v, ti) { Ok(d) => d, Err(_) => { return 4; } };\n\
            let a = match dyn::call(&d, \"area\", &[]) { Ok(a) => a, Err(_) => { return 5; } };\n\
            print(&dyn::emit(&a));\n\
            0\n\
        }\n";
    let (status, out) = load_and_run(dir.path(), &[("plugin.tq", plugin)], &[("host.tq", host)]);
    assert_eq!((status, out.as_str()), (Some(0), "12.0"));
}

#[test]
fn units_importing_one_another_link_unless_they_have_initializers() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "a.tq", "#unit classical\nimport b;\nfn fa() -> i64 { b::fb() + 1 }\n");
    write(dir.path(), "b.tq", "#unit classical\nimport a;\nfn fb() -> i64 { 41 }\nfn fc() -> i64 { a::fa() }\n");
    let main = write(dir.path(), "main.tq", "#unit classical\nimport a;\nfn main() -> i32 { a::fa() as i32 }\n");
    let program = build(std::slice::from_ref(&main), &opts(dir.path(), true)).unwrap();
    assert!(!program.has_errors(), "{:?}", program.all_diagnostics().collect::<Vec<_>>());
    let status = std::process::Command::new(program.executable.unwrap()).status().unwrap();
    assert_eq!(status.code(), Some(42));

    write(dir.path(), "a.tq", "#unit classical(ia)\nimport b;\nlet X: i64;\nfn ia() { X = 1; }\nfn fa() -> i64 { b::fb() + X }\n");
    write(dir.path(), "b.tq", "#unit classical(ib)\nimport a;\nlet Y: i64;\nfn ib() { Y = 2; }\nfn fb() -> i64 { Y }\nfn fc() -> i64 { a::fa() }\n");
    let program = build(&[main], &opts(dir.path(), true)).unwrap();
    let codes: Vec<Code> = program.all_diagnostics().map(|d| d.code).collect();
    assert_eq!(codes, [Code::El04]);
}

///////////////////
//
// Quantum units:
//   `.tqu`, published records, archives.
//
///////////////////

/// A quantum unit publishing a record: an operator judged over a cover and
/// gauge it declares, with claims, and an entry operator.
const QK: &str = "#unit quantum\n\
    import gates;\n\
    cover BITS = fin{ |0>, |1> };\n\
    gauge PLUS = fid((|0> + |1>) * isq2);\n\
    [cover: BITS] [gauge: PLUS]\n\
    [expect: unitary, contract = rigid]\n\
    fn ident(q: *qubit) { x(q); x(q); }\n\
    [entry]\n\
    fn run(flip: bool) -> bool { let q: [qubit; 1] = prep |0>; h(&q[0]); let m = measure q; m[0] }\n";

/// A quantum unit using `qk`'s operator, and asking about its judgement.
const USES: &str = "#unit quantum\n\
    import qk;\n\
    import judge;\n\
    @static_assert(@fragment_of(qk::ident).stabilizer, \"`ident` is a Clifford circuit\");\n\
    [entry]\n\
    fn twice(n: u32) -> bool { let q: [qubit; 1] = prep |1>; qk::ident(&q[0]); let m = measure q; m[0] }\n";

/// A classical program holding handles to both units' entry operators.
const HOLDER: &str = "#unit classical\n\
    import qk;\n\
    import uses;\n\
    fn main() -> i32 { let a = qk::run; let b = uses::twice; 7 }\n";

fn codes_of(program: &topiq::driver::build::Program) -> Vec<Code> {
    program.all_diagnostics().filter(|d| d.is_error()).map(|d| d.code).collect()
}

#[test]
fn a_quantum_unit_compiled_alone_is_imported_from_its_tqu() {
    let dir = tempfile::tempdir().unwrap();
    let qk = write(dir.path(), "qk.tq", QK);
    let alone = build(&[qk], &opts(dir.path(), false)).unwrap();
    assert!(!alone.has_errors(), "{:?}", alone.all_diagnostics().collect::<Vec<_>>());
    let tqu = dir.path().join("qk.tqu");
    assert!(alone.quantum.contains(&tqu), "a quantum unit's output is its `.tqu`");
    let text = std::fs::read_to_string(&tqu).unwrap();
    assert!(text.contains("records: ["), "{text}");
    assert!(text.contains("OPENQASM 3.0;"), "the circuit's OpenQASM travels in the `.tqu`");
    std::fs::remove_file(dir.path().join("qk.tq")).unwrap();

    // what an importer knows of `ident`'s judgement is what `qk.tqu` says,
    // never derived again from its body: a record saying it leaves the
    // Clifford gates is believed, and the assertion fails
    let untrue = text.replacen("stabilizer: true", "stabilizer: false", 1);
    std::fs::write(&tqu, &untrue).unwrap();
    let uses = write(dir.path(), "uses.tq", USES);
    let program = build(std::slice::from_ref(&uses), &opts(dir.path(), false)).unwrap();
    assert_eq!(codes_of(&program), [Code::Em01], "the record is read, not derived");
    std::fs::write(&tqu, &text).unwrap();

    let holder = write(dir.path(), "holder.tq", HOLDER);
    let program = build(&[holder], &opts(dir.path(), true)).unwrap();
    assert!(!program.has_errors(), "{:?}", program.all_diagnostics().collect::<Vec<_>>());
    assert_eq!(program.compiled.len(), 1, "`qk` came from its `.tqu`");
    let status = std::process::Command::new(program.executable.unwrap()).status().unwrap();
    assert_eq!(status.code(), Some(7), "the handles refer to circuits the image embeds");
}

#[test]
fn a_changed_judgment_record_is_refused_as_el03() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "qk.tq", QK);
    write(dir.path(), "uses.tq", USES);
    let holder = write(dir.path(), "holder.tq", HOLDER);
    let first = build(&[holder], &opts(dir.path(), true)).unwrap();
    assert!(!first.has_errors(), "{:?}", first.all_diagnostics().collect::<Vec<_>>());

    // `ident` becomes the bit flip: still unitary, now not rigid, so its
    // record and its claims change
    let changed = QK
        .replace("fn ident(q: *qubit) { x(q); x(q); }", "fn ident(q: *qubit) { x(q); }")
        .replace("contract = rigid", "contract = free");
    let qk = write(dir.path(), "qk.tq", &changed);
    let second = build(&[qk], &opts(dir.path(), false)).unwrap();
    assert!(!second.has_errors(), "{:?}", second.all_diagnostics().collect::<Vec<_>>());
    for f in ["qk.tq", "uses.tq", "holder.tq"] {
        std::fs::remove_file(dir.path().join(f)).unwrap();
    }
    let objects = [dir.path().join("holder.tcu"), dir.path().join("uses.tqu"), dir.path().join("qk.tqu")];
    let linked = build(&objects, &opts(dir.path(), true)).unwrap();
    let codes = codes_of(&linked);
    assert!(codes.contains(&Code::El03), "{codes:?}");
    assert!(linked.executable.is_none());
}

/// Builds `qk`, written as `source`, with `uses` importing it and a program
/// holding both, and returns what was reported.
fn published(source: &str) -> Vec<Code> {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "qk.tq", source);
    write(dir.path(), "uses.tq", USES);
    let holder = write(dir.path(), "holder.tq", HOLDER);
    let program = build(&[holder], &opts(dir.path(), true)).unwrap();
    codes_of(&program)
}

#[test]
fn a_convention_dependent_contract_needs_its_cover_and_gauge_published() {
    assert_eq!(published(QK), Vec::<Code>::new());
    // written out rather than declared, the cover and gauge cannot be named
    let inline = QK.replace(
        "[cover: BITS] [gauge: PLUS]",
        "[cover: fin{ |0>, |1> }] [gauge: fid((|0> + |1>) * isq2)]",
    );
    assert_eq!(published(&inline), [Code::Ej08]);
    // declared `static`, they cannot be named either
    let hidden = QK.replace("cover BITS", "static cover BITS");
    assert_eq!(published(&hidden), [Code::Ej08]);
    // a stationary claim needs no convention
    let stat = inline.replace("contract = rigid", "contract = stat(0)");
    assert_eq!(published(&stat), Vec::<Code>::new());
}

#[test]
fn conductors_that_meet_above_the_limit_are_eu04_then_el10() {
    let dir = tempfile::tempdir().unwrap();
    let qk16 = QK.replace("#unit quantum\n", "#unit quantum\n#pragma conductor(16)\n");
    let qk24 = qk16.replace("(16)", "(24)");
    let uses16 = USES.replace("#unit quantum\n", "#unit quantum\n#pragma conductor(16)\n");
    write(dir.path(), "qk.tq", &qk24);
    let uses = write(dir.path(), "uses.tq", &uses16);
    let program = build(std::slice::from_ref(&uses), &opts(dir.path(), false)).unwrap();
    assert_eq!(codes_of(&program), [Code::Eu04], "16 and 24 meet at 48");

    // compiled when the two agreed, and `qk` changed since
    write(dir.path(), "qk.tq", &qk16);
    let holder = write(dir.path(), "holder.tq", HOLDER);
    let first = build(&[holder], &opts(dir.path(), false)).unwrap();
    assert!(!first.has_errors(), "{:?}", first.all_diagnostics().collect::<Vec<_>>());
    let qk = write(dir.path(), "qk.tq", &qk24);
    let second = build(&[qk], &opts(dir.path(), false)).unwrap();
    assert!(!second.has_errors(), "{:?}", second.all_diagnostics().collect::<Vec<_>>());
    for f in ["qk.tq", "uses.tq", "holder.tq"] {
        std::fs::remove_file(dir.path().join(f)).unwrap();
    }
    let objects = [dir.path().join("holder.tcu"), dir.path().join("uses.tqu"), dir.path().join("qk.tqu")];
    let linked = build(&objects, &opts(dir.path(), true)).unwrap();
    let codes = codes_of(&linked);
    assert!(codes.contains(&Code::El10), "{codes:?}");
}

fn archive(dir: &Path, source: &str) -> topiq::driver::build::Program {
    let qk = write(dir, "qk.tq", source);
    build(
        &[qk],
        &BuildOptions {
            archive: true,
            ..opts(dir, true)
        },
    )
    .unwrap()
}

#[test]
fn an_archive_holds_static_circuits_in_openqasm_with_their_judgments() {
    let dir = tempfile::tempdir().unwrap();
    let program = archive(dir.path(), QK);
    assert!(!program.has_errors(), "{:?}", program.all_diagnostics().collect::<Vec<_>>());
    let path = program.archive.expect("an archive was written");
    assert_eq!(path, dir.path().join("qk.tqar"));
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("Archive {"), "{text}");
    assert!(text.contains("name: \"qk::run\""), "{text}");
    assert!(text.contains("OPENQASM 3.0;"), "{text}");
    assert!(text.contains("judgment: Judgment {"), "each member carries its judgment");
    assert!(program.executable.is_none());

    let lifted = QK.replace(
        "let m = measure q; m[0] }",
        "let m = measure q; let k: usize = lift (if m[0] { 1 } else { 2 }); for i in 0..k { } m[0] }",
    );
    assert_eq!(codes_of(&archive(dir.path(), &lifted)), [Code::El08]);
    let allocating = QK.replace(
        "fn run(flip: bool) -> bool {",
        "fn run(flip: bool, n: u32) -> bool { \
         for i in 0..n { let r: [qubit; 1] = prep |1>; let b = measure r; }",
    );
    assert_eq!(codes_of(&archive(dir.path(), &allocating)), [Code::El09]);
    // a register a loop of the circuit names is one whose qubits are
    // chosen as it runs
    let growing = QK.replace(
        "fn run(flip: bool) -> bool {",
        "fn run(flip: bool, n: u32) -> bool { \
         let r: [qubit] = []; let a: qubit; r.push(a); \
         for i in 0..n { h(&r[0]); } forget r;",
    );
    assert_eq!(codes_of(&archive(dir.path(), &growing)), [Code::El09]);
}

#[test]
fn a_target_s_capabilities_are_those_of_what_is_built() {
    // a program's own processor runs dynamic circuits; an archive's target
    // does not, and a unit can choose by asking
    let dir = tempfile::tempdir().unwrap();
    let asks = QK.replace(
        "import gates;\n",
        "import gates;\n@static_assert(!@target_has(\"dynamic_lifting\"), \"a static target\");\n",
    );
    assert_eq!(codes_of(&archive(dir.path(), &asks)), []);
    let program = dir.path().join("program");
    std::fs::create_dir_all(&program).unwrap();
    write(&program, "qk.tq", &asks);
    let holder = write(&program, "main.tq", "#unit classical\nimport qk;\nfn main() -> i32 { 0 }\n");
    assert_eq!(codes_of(&build(&[holder], &opts(&program, true)).unwrap()), [Code::Em01]);
}

#[test]
fn a_module_loaded_from_an_archive_or_a_tqu_gives_circuits_and_judgments() {
    let dir = tempfile::tempdir().unwrap();
    let built = archive(dir.path(), QK);
    assert!(!built.has_errors(), "{:?}", built.all_diagnostics().collect::<Vec<_>>());
    let host_dir = dir.path().join("host");
    std::fs::create_dir_all(&host_dir).unwrap();
    let host = write(
        &host_dir,
        "host.tq",
        "#unit classical\n\
         fn check(path: *[char]) -> i32 {\n\
             let m = match module::load(path) { Ok(m) => m, Err(_) => { return 1; } };\n\
             if m.circuit::<fn(bool) -> bool>(\"qk::run\").is_none() { return 2; }\n\
             if m.circuit::<fn(u32) -> bool>(\"qk::run\").is_some() { return 3; }\n\
             let j = match m.judgment(\"qk::run\") { Some(j) => j, None => { return 4; } };\n\
             match j.monic { judge::Tri::No => {} _ => { return 5; } }\n\
             match &j.kernel { Some(k) => { if k.measured != 1 { return 6; } } None => { return 7; } }\n\
             if m.judgment(\"qk::none\").is_some() { return 8; }\n\
             0\n\
         }\n\
         fn main() -> i32 {\n\
             let a = check(\"qk.tqar\");\n\
             if a != 0 { return a; }\n\
             let b = check(\"qk.tqu\");\n\
             if b != 0 { return 10 + b; }\n\
             42\n\
         }\n",
    );
    let program = build(
        &[host],
        &BuildOptions {
            executable: Some(dir.path().join("host.exe")),
            ..opts(&host_dir, true)
        },
    )
    .unwrap();
    assert!(!program.has_errors(), "{:?}", program.all_diagnostics().collect::<Vec<_>>());
    let status = std::process::Command::new(dir.path().join("host.exe")).current_dir(dir.path()).status().unwrap();
    assert_eq!(status.code(), Some(42));
}

/// A `#unit any` unit.
const DUAL: &str = "#unit any(classical)\n\
                    #alias dit = bool | qubit\n\
                    #if __UNIT_KIND__ == __QUANTUM__\n\
                    import gates;\n\
                    #define ENTRY [entry]\n\
                    #define FRESH(x) let x: dit\n\
                    #else\n\
                    #define ENTRY\n\
                    #define FRESH(x) let x: dit = false\n\
                    #endif\n\
                    ENTRY\n\
                    fn odd(n: u32) -> bool { FRESH(b); b ^= true; b ^= false; measure b }\n";

const BOTH: &str = "#unit classical\n\
                    import dual;\n\
                    import(quantum) dual as qdual;\n\
                    fn main() -> i32 { if dual::odd(0) && qdual::odd(0) { 7 } else { 1 } }\n";

#[test]
fn a_unit_of_either_kind_is_compiled_as_each_and_linked_from_its_objects() {
    use topiq::driver::KindArgs;
    use topiq::pp::UnitKind;
    let dir = tempfile::tempdir().unwrap();
    let (src, objs, app) = (dir.path().join("src"), dir.path().join("objs"), dir.path().join("app"));
    for d in [&src, &objs, &app] {
        std::fs::create_dir_all(d).unwrap();
    }
    let dual = write(&src, "dual.tq", DUAL);
    for kind in [UnitKind::Classical, UnitKind::Quantum] {
        let built = build(
            std::slice::from_ref(&dual),
            &BuildOptions {
                kinds: KindArgs {
                    all: Some(kind),
                    named: Vec::new(),
                },
                ..opts(&objs, false)
            },
        )
        .unwrap();
        assert!(!built.has_errors(), "{:?}", built.all_diagnostics().collect::<Vec<_>>());
    }
    // one object of each kind, side by side
    assert!(objs.join("dual.tcu").is_file() && objs.join("dual.tqu").is_file());

    // a program finds each by the kind it imports
    let main = write(&app, "main.tq", BOTH);
    let program = build(
        &[main],
        &BuildOptions {
            unit_path: vec![objs.clone()],
            ..opts(&app, true)
        },
    )
    .unwrap();
    assert!(!program.has_errors(), "{:?}", program.all_diagnostics().collect::<Vec<_>>());
    assert_eq!(run(&program.executable.expect("linked")), Some(7));

    // without the quantum object, the quantum import finds nothing, and the
    // classical object does not stand in for it
    std::fs::remove_file(objs.join("dual.tqu")).unwrap();
    let main = app.join("main.tq");
    let program = build(
        &[main],
        &BuildOptions {
            unit_path: vec![objs],
            ..opts(&app, true)
        },
    )
    .unwrap();
    // and what it is used for is then unknown too
    let codes: Vec<Code> = program.all_diagnostics().filter(|d| d.is_error()).map(|d| d.code).collect();
    assert_eq!(codes.first(), Some(&Code::Es17), "{codes:?}");
}
