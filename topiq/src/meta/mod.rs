//! A compiled unit's metadata: what another unit needs to be compiled
//! against it, and what the linker needs to check a program is consistent.
//!
//! Every `.tcu` carries its unit's metadata in a section of its own,
//! `.topiq`, written as a TCON document; a quantum unit's `.tqu` is its
//! metadata alone. It holds:
//!
//! - the **language version** the unit was translated against, and its
//!   **conductor**;
//! - the unit's **interface** (its exported functions with their
//!   signatures, its types with their full definitions, its constants with
//!   their values), enough to compile an importer without the source; and
//!   for a quantum unit, the **judgement record** of each operator with
//!   program linkage, with the claims its `[expect: …]` publishes, and each
//!   `[entry]` operator's **circuit**, in OpenQASM 3 and as the document a
//!   program embeds;
//! - whether the unit declares a **`main`**, and whether that `main` can start
//!   a program;
//! - what it **uses** from each unit it imports: every function, object and
//!   type, recorded as the signature, type or layout it was compiled against,
//!   and every quantum operator, recorded as the record it read.
//!
//! The last is how a stale object is caught. Changing a structure in one unit
//! and recompiling only that unit leaves its importers built against the old
//! layout; when the program is linked, each importer's record is compared
//! with what the imported unit now offers, and any difference is reported
//! rather than producing a program that reads the wrong bytes.
//!
//! [`encode`] and [`decode`] turn metadata into text and back; [`check`]
//! computes the canonical forms that are compared; [`object`] finds the
//! section in an object file.

pub mod check;
pub mod decode;
pub mod encode;
pub mod object;

use crate::ast::Linkage;
use crate::intern::{Interner, Symbol};
use crate::sema::Interface;
use crate::tir::visit::Visit;
use crate::tir::{self, FnKind, GlobalKind, IntTy, Ty};

/// The name of the section metadata is stored in.
pub const SECTION: &str = ".topiq";

/// Whether a unit declares `main`, and whether it can start a program.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MainDecl {
    /// The unit declares no function named `main`.
    Absent,
    /// It declares a `main` that can start a program.
    Valid,
    /// It declares a `main` that cannot, for this reason.
    Invalid(String),
}

/// What kind of item a [`Use`] is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UseKind {
    /// A function.
    Function,
    /// An object or constant.
    Object,
    /// A structure or enumeration.
    Type,
    /// The unit's source, recorded by its digest: generic or constant
    /// function bodies were checked from it.
    Source,
    /// A quantum operator, recorded by the digest of what its unit
    /// publishes of it: its judgement was read, and its gates placed in the
    /// importer's circuits.
    Record,
}

impl UseKind {
    /// The word a message uses.
    pub fn noun(self) -> &'static str {
        match self {
            UseKind::Function => "function",
            UseKind::Object => "object",
            UseKind::Type => "type",
            UseKind::Source => "source of",
            UseKind::Record => "judgment record of",
        }
    }
}

/// One item a unit uses from a unit it imports, as it was when the importer
/// was compiled.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Use {
    /// For a method, the type it belongs to.
    pub owner: Option<MethodUse>,
    /// The unit that declares it.
    pub unit: String,
    /// Its name there.
    pub name: Symbol,
    /// What it is.
    pub kind: UseKind,
    /// Its signature, type or layout digest, in the canonical form of
    /// [`check`].
    pub expect: String,
    /// For a constant, a digest of the value that was folded in.
    pub value: Option<String>,
}

/// The type a method an importer uses belongs to.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MethodUse {
    /// The unit that declared the type.
    pub unit: String,
    /// The type's name.
    pub ty: Symbol,
    /// Whether the method is an operator method.
    pub operator: bool,
}

/// A unit's metadata.
#[derive(Clone, Debug)]
pub struct Metadata {
    /// The version of the language the unit was translated against.
    pub version: String,
    /// The edition of the language.
    pub edition: u32,
    /// The unit's name.
    pub unit: String,
    /// The unit's conductor.
    pub conductor: u32,
    /// Its `main`, if any.
    pub main: MainDecl,
    /// Whether it has an initialiser, which the program runs before `main`,
    /// after those of the units it imports.
    pub init: bool,
    /// What it offers importers.
    pub interface: Interface,
    /// What it uses from its imports.
    pub uses: Vec<Use>,
}

impl Metadata {
    /// The metadata of an analysed unit.
    pub fn of(unit: &tir::Unit, conductor: u32, interner: &Interner) -> Metadata {
        let main = match unit.main {
            None => MainDecl::Absent,
            Some(id) => match main_problem(unit.func(id), &unit.types) {
                None => MainDecl::Valid,
                Some(why) => MainDecl::Invalid(why.to_owned()),
            },
        };
        Metadata {
            version: crate::LANGUAGE_VERSION.to_owned(),
            edition: crate::LANGUAGE_EDITION,
            unit: unit.name.clone(),
            conductor,
            main,
            init: unit.init.is_some(),
            interface: Interface {
                conductor,
                ..if unit.quantum {
                    Interface::of_quantum(unit, &unit.entries, unit.quantum_only.clone())
                } else {
                    Interface::of(unit)
                }
            },
            uses: uses(unit, interner),
        }
    }
}

/// What a unit uses from the units it imports.
fn uses(unit: &tir::Unit, interner: &Interner) -> Vec<Use> {
    let this = unit.name.as_str();
    let types = &unit.types;
    let mut out = Vec::new();
    // another unit's generic bodies bring its items into view whether used or
    // not, so only what is used is recorded; a constant always is, since its
    // uses are folded away
    let mut used = Referenced::default();
    for f in &unit.fns {
        used.block(&f.body);
    }
    for g in &unit.globals {
        if let Some(e) = &g.init {
            used.expr(e);
        }
    }
    for (i, x) in unit.externs.iter().enumerate() {
        if !used.externs.contains(&tir::ExternId(i as u32)) {
            continue;
        }
        out.push(Use {
            owner: x.method.map(|m| {
                let a = types.adt(m.owner);
                MethodUse {
                    unit: a.origin.clone().unwrap_or_else(|| this.to_owned()),
                    ty: a.name,
                    operator: m.operator,
                }
            }),
            unit: x.unit.clone(),
            name: x.name,
            kind: UseKind::Function,
            expect: check::signature_text(types, &x.params, x.ret, this, interner),
            value: None,
        });
    }
    for (i, g) in unit.globals.iter().enumerate() {
        // a circuit handle stands for a quantum unit's entry operator, whose
        // signature is what must still agree
        if let GlobalKind::Imported(from) = &g.kind
            && let Some((params, ret)) = types.as_sig(g.ty).filter(|_| matches!(g.ty, Ty::Circuit(_)))
        {
            out.push(Use {
                owner: None,
                unit: from.clone(),
                name: g.name,
                kind: UseKind::Function,
                expect: check::signature_text(types, params, ret, this, interner),
                value: None,
            });
            continue;
        }
        if let GlobalKind::Imported(from) = &g.kind
            && (g.constant || used.globals.contains(&tir::GlobalId(i as u32)))
        {
            out.push(Use {
                owner: None,
                unit: from.clone(),
                name: g.name,
                kind: UseKind::Object,
                expect: check::object_text(types, g.ty, g.constant, this, interner),
                value: g.value.as_ref().map(|v| check::value_digest(v, interner)),
            });
        }
    }
    for (i, a) in types.adts().iter().enumerate() {
        // an instance was made here from the declaring unit's source, which
        // the source's own digest covers
        if let Some(from) = a.origin.as_ref().filter(|_| a.args.is_empty()) {
            out.push(Use {
                owner: None,
                unit: from.clone(),
                name: a.name,
                kind: UseKind::Type,
                expect: check::adt_digest(types, tir::AdtId(i as u32), this, interner),
                value: None,
            });
        }
    }
    for (from, digest) in &unit.borrowed {
        // every unit whose bodies were borrowed was named by an import, so
        // its name is interned
        if let Some(name) = interner.get(from) {
            out.push(Use {
                owner: None,
                unit: from.clone(),
                name,
                kind: UseKind::Source,
                expect: digest.clone(),
                value: None,
            });
        }
    }
    out
}

/// The quantum operators of other units the quantum unit `unit`, analysed
/// against `imports`, uses (those its own code reaches, directly or through
/// the operators it uses), each one whose unit publishes a record of it,
/// recorded by the record's digest.
pub fn record_uses(unit: &tir::Unit, interner: &Interner, imports: &[Interface]) -> Vec<Use> {
    /// The functions an expression names.
    #[derive(Default)]
    struct Named(Vec<tir::FnId>);
    impl tir::visit::Visit for Named {
        fn expr(&mut self, e: &tir::Expr) {
            if let tir::ExprKind::FnRef(tir::Callee::Fn(f))
            | tir::ExprKind::Call {
                callee: tir::Callee::Fn(f),
                ..
            } = &e.kind
            {
                self.0.push(*f);
            }
            tir::visit::walk_expr(self, e);
        }
    }
    let mut reached: Vec<bool> = unit.fns.iter().map(|f| f.origin.is_none()).collect();
    // operators named outside any body: a chain's stages, a map locale's
    // entries, an iso-sum's witnesses, and those whose judgements the unit
    // asks about
    let mut named = Named::default();
    for c in &unit.geometry.chains {
        for s in &c.def.stages {
            named.expr(&s.op);
        }
    }
    for q in &unit.geometry.qmaps {
        for (_, e) in &q.def.entries {
            named.expr(e);
        }
    }
    named.0.extend(unit.isos.iter().flat_map(|i| i.witnesses.iter().map(|(f, _)| *f)));
    named.0.extend(unit.derived.iter().filter_map(|d| match *d {
        tir::claims::Derived::Judgment(f)
        | tir::claims::Derived::Kernel(f)
        | tir::claims::Derived::Certificate(f)
        | tir::claims::Derived::MatrixOf(f)
        | tir::claims::Derived::FragmentOf(f)
        | tir::claims::Derived::ClassifyOp(f, _) => Some(f),
        tir::claims::Derived::Holonomy(..) => None,
    }));
    for f in named.0 {
        reached[f.index()] = true;
    }
    let mut work: Vec<tir::FnId> = (0..unit.fns.len()).filter(|&i| reached[i]).map(|i| tir::FnId(i as u32)).collect();
    while let Some(id) = work.pop() {
        let mut named = Named::default();
        named.block(&unit.func(id).body);
        for f in named.0 {
            if !reached[f.index()] {
                reached[f.index()] = true;
                work.push(f);
            }
        }
    }
    let mut out: Vec<Use> = Vec::new();
    for (i, f) in unit.fns.iter().enumerate() {
        let Some(from) = &f.origin else { continue };
        if !reached[i] || !f.args.is_empty() || out.iter().any(|u| &u.unit == from && u.name == f.name) {
            continue;
        }
        let published = imports
            .iter()
            .find(|i| &i.unit == from)
            .and_then(|i| i.record(interner.resolve(f.name)));
        if let Some(p) = published {
            out.push(Use {
                owner: None,
                unit: from.clone(),
                name: f.name,
                kind: UseKind::Record,
                expect: encode::record_digest(p),
                value: None,
            });
        }
    }
    out
}

/// The functions of other units a unit calls, and the objects it reads or
/// writes.
#[derive(Default)]
struct Referenced {
    externs: std::collections::HashSet<tir::ExternId>,
    globals: std::collections::HashSet<tir::GlobalId>,
}

impl tir::visit::Visit for Referenced {
    fn expr(&mut self, e: &tir::Expr) {
        match &e.kind {
            tir::ExprKind::Call {
                callee: tir::Callee::Extern(x),
                ..
            } => {
                self.externs.insert(*x);
            }
            tir::ExprKind::Global(g) => {
                self.globals.insert(*g);
            }
            _ => {}
        }
        tir::visit::walk_expr(self, e);
    }
}

/// Why a function named `main` cannot start a program, if it cannot.
///
/// A program starts by calling `main` in exactly one of its units. It must be
/// visible outside its unit (not `static`), compiled (not a constant
/// function), take nothing or the program's arguments as a `*[*[char]]`,
/// and return an `i32`, which becomes the program's exit status.
pub fn main_problem(f: &tir::Fn, types: &tir::TypeTable) -> Option<&'static str> {
    let params: Vec<Ty> = f.param_types().collect();
    if f.linkage == Linkage::Unit {
        Some("it is declared `static`, so it is private to its unit and cannot start the program")
    } else if f.kind == FnKind::Constant {
        Some("its parameters are `const`, so it only exists during translation")
    } else if !(params.is_empty() || (params.len() == 1 && is_argv(params[0], types))) {
        Some("it takes parameters other than the program's arguments, `argv: *[*[char]]`")
    } else if f.ret != Ty::Int(IntTy::I32) {
        Some("it does not return an `i32`, which becomes the program's exit status")
    } else {
        None
    }
}

/// Whether `t` is `*[*[char]]`: the program's arguments, as `main` may take
/// them, each one a slice of characters.
pub fn is_argv(t: Ty, types: &tir::TypeTable) -> bool {
    types
        .as_slice(t)
        .and_then(|(_, arg)| types.as_slice(arg))
        .is_some_and(|(_, c)| c == Ty::Char)
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::sema::testing::analyzed_in;

    /// The metadata of `src`, analysed as a clean unit named `name` that may
    /// import any of `imports`, with names in `interner`.
    pub fn metadata_of(name: &str, src: &str, imports: &[Interface], interner: &mut Interner) -> Metadata {
        let (u, d) = analyzed_in(name, src, imports, interner);
        assert!(d.iter().all(|d| !d.is_error()), "{d:?}");
        Metadata::of(&u, 8, interner)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::metadata_of;
    use super::*;
    use crate::sema::testing::check;

    #[test]
    fn a_valid_main_is_recorded() {
        let mut i = Interner::new();
        let m = metadata_of("app", "fn main() -> i32 { 0 }", &[], &mut i);
        assert_eq!(m.main, MainDecl::Valid);
        assert_eq!(m.version, crate::LANGUAGE_VERSION);
        assert_eq!(m.unit, "app");
    }

    #[test]
    fn what_an_importer_uses_is_recorded_as_compiled() {
        let mut i = Interner::new();
        let geo = metadata_of(
            "geo",
            "struct P { x: i32 }\nfn area(p: *P) -> i32 { p.x }\nlet LIMIT: const i32 = 7;",
            &[],
            &mut i,
        );
        let app = metadata_of(
            "app",
            "import geo;\nfn main() -> i32 { let p = geo::P { x: 2 }; geo::area(&p) + geo::LIMIT }",
            std::slice::from_ref(&geo.interface),
            &mut i,
        );
        let kinds: Vec<(UseKind, &str)> = app.uses.iter().map(|u| (u.kind, i.resolve(u.name))).collect();
        assert!(kinds.contains(&(UseKind::Function, "area")), "{kinds:?}");
        assert!(kinds.contains(&(UseKind::Object, "LIMIT")), "{kinds:?}");
        assert!(kinds.contains(&(UseKind::Type, "P")), "{kinds:?}");
        let area = app.uses.iter().find(|u| u.kind == UseKind::Function).unwrap();
        assert_eq!(area.expect, "fn(*geo::P) -> i32");
        assert!(app.uses.iter().all(|u| u.unit == "geo"));
        let limit = app.uses.iter().find(|u| u.kind == UseKind::Object).unwrap();
        assert!(limit.value.is_some(), "a constant's value is recorded");
    }

    #[test]
    fn a_main_that_cannot_start_a_program_says_why() {
        let reason = |src: &str| {
            let u = check(src);
            main_problem(u.func(u.main.expect("declares main")), &u.types).map(str::to_owned)
        };
        assert_eq!(reason("fn main() -> i32 { 0 }"), None);
        assert!(reason("static fn main() -> i32 { 0 }").unwrap().contains("static"));
        assert!(reason("fn main(x: i32) -> i32 { x }").unwrap().contains("parameters"));
        assert_eq!(reason("fn main(argv: *[*[char]]) -> i32 { argv.len() as i32 }"), None);
        assert!(reason("fn main() { }").unwrap().contains("i32"));
        assert!(reason("fn main() -> u8 { 0 }").unwrap().contains("i32"));
    }

    #[test]
    fn a_unit_with_no_imports_uses_nothing() {
        let mut i = Interner::new();
        let m = metadata_of("geo", "struct P { x: i32 }\nfn f(p: *P) -> i32 { p.x }", &[], &mut i);
        assert!(m.uses.is_empty());
        assert_eq!(m.main, MainDecl::Absent);
    }
}
