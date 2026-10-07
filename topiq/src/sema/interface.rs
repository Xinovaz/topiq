//! What a unit offers the units that import it.
//!
//! An importing unit is analysed against its imports' *interfaces*, never
//! their bodies: the functions it may call with their signatures, the types it
//! may name with their full definitions (so it can build and lay out values of
//! them), and the constants it may use with their values (so they fold into
//! its own constant expressions). An interface is built from a finished
//! analysis when both units are compiled together, and read back from the
//! imported unit's object file when it was compiled earlier; either way the
//! importer sees the same thing.
//!
//! Only what has program linkage may be *named* by an importer. An item
//! declared `static` is not merely hidden: an importer naming it is told there
//! is no such item. It still travels, flagged as `static`, because an importer
//! may need it on another unit's behalf (see below).
//!
//! # Bodies that travel
//!
//! A generic function is checked once for each set of arguments, and a
//! constant function is evaluated wherever it is called, so an importer using
//! either needs its *body*, not just its signature. A unit that declares any
//! therefore carries its source text in its interface, with the text of every
//! file it embeds, and the importer parses it again and checks the bodies it
//! needs in the declaring unit's own scope: a name in them means what it means
//! there, including its `static` items and its own imports.
//!
//! # Records that travel
//!
//! A quantum unit's operators are judged once, in the unit that declares
//! them. Its interface carries what it publishes of each operator with
//! program linkage (the record, the claims its `[expect: …]` makes, and the
//! declarations its cover and gauge name), and an importer asking about
//! such an operator reads the record rather than deriving it again. The
//! interface also carries each `[entry]` operator's circuit, which a program
//! holding a handle to it embeds.

use std::sync::Arc;

use crate::ast::{self, Linkage};
use crate::intern::Symbol;
use crate::tir::{self, FnKind, GlobalKind, Ty, TypeTable, Value};

/// A function another unit may call.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExportFn {
    /// Its name.
    pub name: Symbol,
    /// Its parameter types, in the interface's type table.
    pub params: Vec<Ty>,
    /// Its return type.
    pub ret: Ty,
    /// Whether it is a constant function, evaluated during translation
    /// rather than compiled.
    pub constant: bool,
    /// Whether importers may name it. A `static` one is only reached through
    /// the declaring unit's generic bodies.
    pub linkage: Linkage,
    /// The type it is a method or associated function of, in the interface's
    /// type table. A method is found through its type, never by name alone.
    pub method: Option<tir::MethodOf>,
    /// Whether it is a quantum unit's `[entry]` operator, which a classical
    /// importer holds as a circuit handle rather than calls.
    pub entry: bool,
    /// The symbol `[export: "sym"]` gave it, which importers call it by.
    pub symbol: Option<String>,
    /// The message of `[deprecated: "msg"]`, repeated at each use.
    pub deprecated: Option<String>,
}

/// A type alias another unit may use: another name for a type, in the
/// interface's type table.
#[derive(Clone, PartialEq, Debug)]
pub struct ExportAlias {
    /// Its name.
    pub name: Symbol,
    /// The type it stands for.
    pub ty: Ty,
    /// Whether other units may name it.
    pub linkage: Linkage,
}

/// An object or constant another unit may use.
#[derive(Clone, PartialEq, Debug)]
pub struct ExportGlobal {
    /// Its name.
    pub name: Symbol,
    /// Its type, in the interface's type table.
    pub ty: Ty,
    /// Whether it is a constant, whose value travels with the interface.
    pub constant: bool,
    /// A constant's value.
    pub value: Option<Value>,
    /// Whether importers may name it.
    pub linkage: Linkage,
    /// The symbol `[export: "sym"]` gave it, which importers reach it by.
    pub symbol: Option<String>,
    /// The message of `[deprecated: "msg"]`, repeated at each use.
    pub deprecated: Option<String>,
}

/// A quantum unit's `[entry]` operator's circuit, as it travels.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EntryCircuit {
    /// The operator's name.
    pub name: Symbol,
    /// Its signature, as a program writes the type, every type qualified.
    pub sig: String,
    /// The circuit in OpenQASM 3.
    pub qasm: String,
    /// The circuit's document ([`crate::circuit::document`]).
    pub document: String,
    /// Whether it lifts a measured value, which only a target able to run
    /// dynamic circuits can do.
    pub dynamic: bool,
    /// Whether it asks for qubits as it runs, a number known only then,
    /// which only a target able to allocate registers while running can
    /// give.
    pub allocates: bool,
    /// For each operator-typed parameter, in order, whether the operator
    /// supplied for it must be monic.
    pub monic_slots: Vec<bool>,
}

/// A unit's source, carried so that importers can check its generic and
/// constant functions' bodies.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Source {
    /// The file it was read from.
    pub path: String,
    /// Its text.
    pub text: String,
    /// Every file it embeds, by the path it names it with, and that file's
    /// text.
    pub embeds: Vec<(String, String)>,
}

impl Source {
    /// A digest of the text, embedded files included, that tells whether two
    /// copies of a unit's source are the same.
    pub fn digest(&self) -> String {
        let mut all = self.text.clone();
        for (path, text) in &self.embeds {
            all.push('\0');
            all.push_str(path);
            all.push('\0');
            all.push_str(text);
        }
        format!("{:016x}", crate::meta::check::fnv1a(all.as_bytes()))
    }
}

/// A unit's interface.
#[derive(Clone, Debug)]
pub struct Interface {
    /// The unit's name.
    pub unit: String,
    /// Every structure and enumeration an export mentions (the unit's own,
    /// including `static` ones a signature needs, and any it imported), each
    /// with its origin recorded, plus the compound types built from them.
    pub types: TypeTable,
    /// Its functions, `static` ones included.
    pub fns: Vec<ExportFn>,
    /// Its objects and constants, `static` ones included.
    pub globals: Vec<ExportGlobal>,
    /// The units it imports.
    pub imports: Vec<String>,
    /// Its source, when it declares generic or constant functions whose
    /// bodies importers need.
    pub source: Option<Source>,
    /// That source parsed, ready for an importer to check. Not written to
    /// metadata: whoever reads the metadata parses [`Interface::source`]
    /// again.
    pub ast: Option<Arc<ast::Unit>>,
    /// Whether the unit is a quantum unit. A classical unit importing one
    /// may use only its types, its constants and its `[entry]` operators.
    pub quantum: bool,
    /// For a quantum unit, the names it declares that a classical unit may
    /// not use: operators other than its entries, objects that are not
    /// constant, and types that hold quantum values.
    pub quantum_only: Vec<Symbol>,
    /// The structures and enumerations it declares that are marked
    /// `[deprecated: "msg"]`, by name, with their messages.
    pub deprecated_types: Vec<(Symbol, String)>,
    /// The unit's conductor, which judgements crossing into it are checked
    /// at together with the importer's.
    pub conductor: u32,
    /// For a quantum unit, what it publishes of each operator with program
    /// linkage that has a judgement: the record importers read rather than
    /// derive again.
    pub records: Vec<crate::judge::record::Published>,
    /// For a quantum unit, each `[entry]` operator's circuit.
    pub circuits: Vec<EntryCircuit>,
    /// Its type aliases that take no generic parameters, `static` ones
    /// included, each with the type it stands for.
    pub aliases: Vec<ExportAlias>,
    /// For a `#unit any` unit, the kind it prefers, if it prefers one; `None`
    /// for a unit of fixed kind. Such a unit's name, [`Interface::unit`],
    /// carries the kind it was translated as, as `dsp.quantum`.
    pub any: Option<Option<crate::pp::UnitKind>>,
    /// For a quantum unit, the covers, gauges and base types other units may
    /// name, as `gates::T01`.
    pub geometry: Vec<ExportGeometry>,
}

/// A cover, gauge or base type another unit may name.
#[derive(Clone, PartialEq, Debug)]
pub struct ExportGeometry {
    /// Its name.
    pub name: Symbol,
    /// What it declares, with its states at the declaring unit's conductor.
    pub def: GeometryDef,
}

/// What a cover, gauge or base type declares.
#[derive(Clone, PartialEq, Debug)]
pub enum GeometryDef {
    /// A cover.
    Cover(crate::judge::cover::Cover),
    /// A gauge.
    Gauge(crate::judge::cover::Gauge),
    /// A base type: its register's states are its cover's.
    Base(crate::judge::cert::Base),
}

impl Interface {
    /// The interface of an analysed unit. Its imports, source and syntax
    /// tree are the driver's to fill in; analysis does not know them.
    pub fn of(unit: &tir::Unit) -> Interface {
        let mut types = unit.types.clone();
        for i in 0..types.adts().len() {
            let a = types.adt_mut(tir::AdtId(i as u32));
            if a.origin.is_none() {
                a.origin = Some(unit.name.clone());
            }
        }
        // instances and other units' functions are not this unit's to offer:
        // an importer makes its own. nor is the unit's initialiser, which
        // only the program's startup runs
        let fns = unit
            .fns
            .iter()
            .enumerate()
            .filter(|(i, f)| {
                f.origin.is_none() && f.args.is_empty() && !f.attrs.hidden && unit.init != Some(tir::FnId(*i as u32))
            })
            .map(|(_, f)| f)
            .map(|f| ExportFn {
                name: f.name,
                params: f.param_types().collect(),
                ret: f.ret,
                constant: f.kind == FnKind::Constant,
                linkage: f.linkage,
                method: f.method,
                entry: false,
                symbol: f.attrs.symbol.clone(),
                deprecated: f.attrs.deprecated.clone(),
            })
            .collect();
        let globals = unit
            .globals
            .iter()
            .filter(|g| g.kind == GlobalKind::Unit)
            .map(|g| ExportGlobal {
                name: g.name,
                ty: g.ty,
                constant: g.constant,
                value: if g.constant { g.value.clone() } else { None },
                linkage: g.linkage,
                symbol: g.symbol.clone(),
                deprecated: g.deprecated.clone(),
            })
            .collect();
        let aliases = unit
            .aliases
            .iter()
            .map(|a| ExportAlias {
                name: a.name,
                ty: a.ty,
                linkage: a.linkage,
            })
            .collect();
        Interface {
            unit: unit.name.clone(),
            types,
            fns,
            globals,
            aliases,
            imports: Vec::new(),
            source: None,
            ast: None,
            quantum: false,
            quantum_only: Vec::new(),
            deprecated_types: unit.deprecated_types.clone(),
            conductor: 8,
            records: Vec::new(),
            circuits: Vec::new(),
            any: None,
            geometry: Vec::new(),
        }
    }

    /// The cover, gauge or base type `name` that other units may name.
    pub fn geometry_named(&self, name: Symbol) -> Option<&GeometryDef> {
        self.geometry.iter().find(|g| g.name == name).map(|g| &g.def)
    }

    /// What the unit publishes of its operator `name`, if it publishes a
    /// record of it.
    pub fn record(&self, name: &str) -> Option<&crate::judge::record::Published> {
        self.records.iter().find(|p| p.name == name)
    }

    /// The circuit of its `[entry]` operator `name`.
    pub fn circuit(&self, name: Symbol) -> Option<&EntryCircuit> {
        self.circuits.iter().find(|c| c.name == name)
    }

    /// The interface of a quantum unit. A quantum importer may use all of
    /// it; a classical importer only its types that hold no qubits, its
    /// constants and its `entry` operators, and is told of the rest, whose
    /// names `quantum_only` lists.
    pub fn of_quantum(unit: &tir::Unit, entries: &[tir::FnId], quantum_only: Vec<Symbol>) -> Interface {
        let mut iface = Interface::of(unit);
        for f in &mut iface.fns {
            f.entry = entries.iter().any(|&id| unit.func(id).name == f.name && unit.func(id).method.is_none());
            if f.entry {
                f.symbol = None;
            }
        }
        iface.quantum = true;
        iface.quantum_only = quantum_only;
        let g = &unit.geometry;
        iface.geometry = g
            .exports
            .iter()
            .map(|&(name, e)| ExportGeometry {
                name,
                def: match e {
                    tir::Exported::Cover(i) => GeometryDef::Cover(g.covers[i].def.clone()),
                    tir::Exported::Gauge(i) => GeometryDef::Gauge(g.gauges[i].def.clone()),
                    tir::Exported::Base(i) => {
                        let b = g.bases[i].def;
                        GeometryDef::Base(crate::judge::cert::Base {
                            cover: g.covers[b.cover].def.clone(),
                            gauge: g.gauges[b.gauge].def.clone(),
                        })
                    }
                },
            })
            .collect();
        iface
    }

    /// The function named `name` that importers may call.
    pub fn function(&self, name: Symbol) -> Option<&ExportFn> {
        self.any_function(name).filter(|f| f.linkage == Linkage::Program)
    }

    /// The object or constant named `name` that importers may use.
    pub fn global(&self, name: Symbol) -> Option<&ExportGlobal> {
        self.any_global(name).filter(|g| g.linkage == Linkage::Program)
    }

    /// The function named `name`.
    pub fn any_function(&self, name: Symbol) -> Option<&ExportFn> {
        self.fns.iter().find(|f| f.name == name && f.method.is_none())
    }

    /// The method or associated function `name` this unit gives the type
    /// declared by `origin` as `ty`: its own type, or another unit's
    /// `[open]` one.
    pub fn method(&self, origin: &str, ty: Symbol, name: Symbol, operator: bool) -> Option<&ExportFn> {
        self.fns.iter().find(|f| {
            f.name == name
                && f.method.is_some_and(|m| {
                    let a = self.types.adt(m.owner);
                    m.operator == operator &&
                    a.name == ty && a.origin.as_deref() == Some(origin) && a.args.is_empty()
                })
        })
    }

    /// The object or constant named `name`.
    pub fn any_global(&self, name: Symbol) -> Option<&ExportGlobal> {
        self.globals.iter().find(|g| g.name == name)
    }

    /// The structure or enumeration this unit declares under `name`, if an
    /// importer may name it.
    pub fn named_type(&self, name: Symbol) -> Option<tir::AdtId> {
        let id = self.any_type(name)?;
        (self.types.adt(id).linkage == Linkage::Program).then_some(id)
    }

    /// The structure or enumeration this unit declares under `name`, `static`
    /// or not.
    pub fn any_type(&self, name: Symbol) -> Option<tir::AdtId> {
        let id = self.types.find_adt(Some(&self.unit), name)?;
        Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sema::testing::check;

    #[test]
    fn only_program_linkage_can_be_named() {
        let u = check(
            "struct Point { x: i32 }\n\
             static struct Hidden { y: i32 }\n\
             fn area(p: *Point) -> i32 { p.x }\n\
             static fn helper() -> i32 { 1 }\n\
             let LIMIT: const i32 = 7;\n\
             static let SALT: const i32 = 5;\n\
             let COUNT: u32 = 0;",
        );
        let i = Interface::of(&u);
        assert_eq!(i.unit, "test");
        assert_eq!(i.fns.iter().filter(|f| f.linkage == Linkage::Program).count(), 1);
        assert_eq!(i.fns.len(), 2, "a `static` function travels, flagged");
        assert_eq!(i.globals.iter().filter(|g| g.linkage == Linkage::Program).count(), 2);
        assert_eq!(i.globals.len(), 3);
        let limit = i.globals.iter().find(|g| g.constant && g.linkage == Linkage::Program).unwrap();
        assert_eq!(limit.value, Some(Value::Int(7, crate::tir::IntTy::I32)));
        let count = i.globals.iter().find(|g| !g.constant).unwrap();
        assert_eq!(count.value, None, "an object's value is not part of the interface");
    }

    #[test]
    fn a_static_type_travels_but_cannot_be_named() {
        let u = check("static struct Hidden { y: i32 }\nstruct Shown { h: *Hidden }");
        let i = Interface::of(&u);
        assert_eq!(i.types.adts().len(), 2);
        assert!(i.types.adts().iter().all(|a| a.origin.as_deref() == Some("test")));
        let names: Vec<bool> = i
            .types
            .adts()
            .iter()
            .map(|a| i.named_type(a.name).is_some())
            .collect();
        assert_eq!(names, [false, true]);
    }
}
