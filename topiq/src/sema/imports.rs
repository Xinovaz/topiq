//! `import`, and names reached through an imported unit.
//!
//! `import geometry;` makes the unit `geometry` available under its own name,
//! and `import geometry as g;` under `g`. Its items are then reached by
//! qualified paths (`geometry::area(…)`, `geometry::Point { … }`,
//! `geometry::Shape::Circle(…)`): never unqualified, so reading a name always
//! tells you where it comes from.
//!
//! What an import provides is the other unit's [`Interface`]. Using one of its
//! items copies what this unit needs into its own tables: a function becomes
//! an [`tir::ExternFn`] called by its symbol, an object an imported
//! [`tir::Global`], a constant its value, and a type its full definition,
//! including any types that definition mentions, whichever unit they came
//! from. Types stay identified by the unit that declared them and their name,
//! so `geometry::Point` reached through two different imports is still one
//! type.
//!
//! Another unit's generic items and constant functions are different: using
//! them needs their bodies. For those, the unit's own declarations are opened
//! as a second *frame* of this analysis (see [`UnitCx`]), from the
//! source its interface carries, and `geometry::sum::<i32>` is checked there
//! and made here. The instance is named after `geometry`, so every unit that
//! makes it makes the same one, and the linker keeps a single copy.
//!
//! `import shapes::solid::cube;` is the unit `cube`, found as
//! `shapes/solid/cube.tq` on the unit path, and known as `cube`.
//!
//! `import(quantum) dsp;` asks a `#unit any` unit for a kind. Without one,
//! it is the kind the unit prefers, or else this unit's. Each kind of such a
//! unit is a unit of its own, known as `dsp.classical` or `dsp.quantum`, and
//! a unit importing both gives each its own name with `as`. A unit of fixed
//! kind is not asked for one, even its own (`EU08`).
//!
//! The `core` library needs no import: its items are visible everywhere,
//! both unqualified and as `core::name`. The other library units are named
//! `unit::item` without an import too.
//!
//! Units that import one another are analysed from each other's source
//! ([`crate::driver::program`]). A peer's declarations, opened here, import
//! the unit being analysed, and that import names its own items
//! ([`UnitRef::SELF`]).
//!
//! # Quantum units
//!
//! A quantum unit's operators are placed in the circuits of a quantum unit
//! using them, so their bodies are checked in the declaring unit's frame, as
//! generic bodies are; a quantum library unit such as `gates`, which a
//! quantum unit names unqualified, is named so in each unit's own code. A
//! classical unit holds an `[entry]` operator as a circuit handle, an object
//! the program defines when it is linked.
//!
//! A judgement crossing between two units is checked at the least common
//! multiple of their conductors; importing a quantum unit whose conductor
//! meets this unit's above the limit is `EU04`.

use crate::ast::Path;
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{self, AdtDef, AdtId, AdtKind, Arg, Compound, FieldDef, GlobalKind, Ty, VariantDef};

use super::interface::Interface;
use super::items::UnitCx;
use super::report;
use super::scope::{Def, UnitRef};

/// Declares an `import` item: `import shapes::solid::cube;`, or with the
/// kind asked of a `#unit any` unit, `import(quantum) dsp;`.
///
/// A `#unit any` unit is the unit of the kind asked for, or else of the kind
/// it prefers, or else of this unit's kind, as the program's loader chose it;
/// its name among the units available carries that kind, as `dsp.quantum`.
/// Asking a unit of fixed kind for a kind is `EU08`, even for its own.
pub fn declare(cx: &mut UnitCx<'_>, path: &Path, alias: Option<&Spanned<Symbol>>, kind: Option<&Spanned<Symbol>>, span: Span) {
    // `import shapes::solid::cube;` is the unit `cube`, found in the
    // directory `shapes/solid`
    let name = *path.segments.last().expect("a path has a part");
    let text = cx.interner.resolve(name.node);
    let asked = match kind.map(|k| (k, cx.interner.resolve(k.node))) {
        None => None,
        Some((_, "classical")) => Some(crate::pp::UnitKind::Classical),
        Some((_, "quantum")) => Some(crate::pp::UnitKind::Quantum),
        Some((k, other)) => {
            cx.report(
                Diagnostic::new(Code::Es02)
                    .with_message(format!("`import({other})` names no kind of unit"))
                    .at(k.span)
                    .with_help("write `import(classical)` or `import(quantum)`"),
            );
            return;
        }
    };
    if text == "core" {
        // always available; importing it changes nothing
        return;
    }
    let local = alias.copied().unwrap_or(name);
    let own = crate::driver::written(&cx.unit.name);
    if text == own && cx.frame != 0 {
        // another unit's declarations, open here, import this one: the two
        // import one another. its name means this unit's own items
        let _ = cx.scope.declare_unit(local.node, UnitRef::SELF, local.span);
        return;
    }
    if text == own {
        cx.report(
            Diagnostic::new(Code::Es05)
                .with_message(format!("unit `{text}` imports itself"))
                .at(span)
                .with_note("a unit's own items are already visible throughout it")
                .with_help("remove this `import`"),
        );
        return;
    }
    let available = cx.available;
    let fixed = available.iter().find(|i| i.unit == text);
    if let (Some(_), Some(k)) = (fixed, kind) {
        cx.report(
            Diagnostic::new(Code::Eu08)
                .with_message(format!("`{text}` is a unit of fixed kind, which is not imported with a kind"))
                .at(k.span)
                .with_note(
                    "only a `#unit any` unit takes its kind from what imports it; a unit that declares \
                     its kind keeps it, and asking for one, even its own, would hide a change to it",
                )
                .with_help(format!("write `import {text};`, or declare `{text}` with `#unit any`")),
        );
        return;
    }
    let chosen = fixed.or_else(|| {
        // a `#unit any` unit, of the kind asked, preferred, or this unit's
        let prefers = available
            .iter()
            .find(|i| i.any.is_some() && crate::driver::written(&i.unit) == text)
            .and_then(|i| i.any.flatten());
        let own_kind = if cx.quantum { crate::pp::UnitKind::Quantum } else { crate::pp::UnitKind::Classical };
        let k = asked.or(prefers).unwrap_or(own_kind);
        let identity = format!("{text}.{}", k.name());
        available.iter().find(|i| i.unit == identity)
    });
    let Some(iface) = chosen else {
        let d = if crate::library::is_library(text) {
            report::unsupported(span, format_args!("the library unit `{text}`"))
                .with_note(
                    "this compiler provides every library unit the language defines, but this one was \
                     not made available to this translation",
                )
        } else {
            Diagnostic::new(Code::Es17)
                .with_message(format!("there is no unit named `{text}` to import"))
                .at(name.span)
                .with_note(format!(
                    "a unit is found by its file name: `import {text};` looks for `{text}.tq`, \
                     or `{text}.tcu` or `{text}.tqu` if it was compiled already, beside the files \
                     being compiled and in each directory given with `--unit-path`"
                ))
                .with_help(format!(
                    "check the spelling, or pass the directory holding `{text}.tq` with \
                     `--unit-path`"
                ))
        };
        cx.report(d);
        return;
    };
    // a quantum unit's judgements meet the importer's at the least common
    // multiple of their conductors, and of those of the quantum units it
    // brings in, which the limit must admit
    let limit = crate::diag::Limit::Conductor.value();
    let (source, conductor) = highest_meeting(cx.available, iface, cx.conductor);
    let both = num_integer::lcm(cx.conductor.max(1), conductor.max(1));
    if cx.frame == 0 && iface.quantum && both > limit {
        let through = if source == iface.unit {
            String::new()
        } else {
            format!(", through `{source}`, which it imports,")
        };
        cx.report(
            Diagnostic::new(Code::Eu04)
                .with_message(format!(
                    "`{text}`{through} brings conductor {conductor}, and this unit has {}: their judgments meet \
                     at conductor {both}, above the limit of {limit}",
                    cx.conductor
                ))
                .at(span)
                .with_note(
                    "a judgment crossing between units is checked at the least common multiple of their \
                     conductors, whose phase group holds both units' phases",
                )
                .with_help(format!(
                    "declare the two units with conductors whose least common multiple is at most {limit}, \
                     such as the same one"
                )),
        );
    }
    let index = match cx.imported.iter().position(|i| std::ptr::eq(*i, iface)) {
        Some(i) => i,
        None => {
            cx.imported.push(iface);
            cx.imported.len() - 1
        }
    };
    if let Err(t) = cx.scope.declare_unit(local.node, UnitRef(index as u32), local.span) {
        let d = report::duplicate(local.span, t.earlier, cx.interner.resolve(local.node), "at unit scope");
        let mut d = d.with_note(
            "an imported unit's name starts every path into it, so nothing else at unit scope \
             may share it",
        );
        if iface.any.is_some() {
            d = d.with_help(format!(
                "a `#unit any` unit imported as both kinds needs a name for each: \
                 `import(classical) {text} as c{text};`"
            ));
        }
        cx.report(d);
    }
    // a quantum library unit's operators are the vocabulary of a quantum
    // unit importing it, and are named unqualified there: `h(&q)`, in the
    // unit's own code, and in another quantum unit's, checked here
    let by = cx.frame;
    if cx.quantum
        && iface.quantum
        && crate::library::source(&iface.unit).is_some()
        && let Some(frame) = cx.frame_for(iface)
    {
        cx.opened.push((by, frame));
    }
}

/// Of the quantum unit `iface` and the quantum units it imports, directly or
/// not, among `available`, the one whose conductor meets `own` highest, and
/// that conductor.
fn highest_meeting(available: &[Interface], iface: &Interface, own: u32) -> (String, u32) {
    let mut best = (iface.unit.clone(), iface.conductor);
    let mut seen: Vec<&str> = Vec::new();
    let mut work: Vec<&Interface> = vec![iface];
    while let Some(i) = work.pop() {
        if seen.contains(&i.unit.as_str()) {
            continue;
        }
        seen.push(&i.unit);
        if i.quantum && num_integer::lcm(own.max(1), i.conductor.max(1)) > num_integer::lcm(own.max(1), best.1.max(1)) {
            best = (i.unit.clone(), i.conductor);
        }
        for name in &i.imports {
            let unit = crate::driver::unit_of(name);
            if let Some(next) = available.iter().find(|a| a.unit == unit) {
                work.push(next);
            }
        }
    }
    best
}

/// The type `name` declared by an imported unit.
pub fn import_type(cx: &mut UnitCx<'_>, u: UnitRef, name: Symbol) -> Option<AdtId> {
    if u == UnitRef::SELF {
        return cx.own_type(name);
    }
    let iface = cx.imported[u.0 as usize];
    if !cx.quantum && iface.quantum_only.contains(&name) {
        return None;
    }
    let id = iface.named_type(name)?;
    Some(import_adt(cx, iface, id))
}

/// A type of another unit whose frame is being declared, `static` or not.
pub fn foreign_type(cx: &mut UnitCx<'_>, iface: &Interface, name: Symbol) -> Option<AdtId> {
    let id = iface.any_type(name)?;
    Some(import_adt(cx, iface, id))
}

/// A value (function, object or constant) declared by an imported unit.
///
/// # Errors
///
/// A diagnostic when the item exists but cannot be used from
/// another unit.
pub fn import_value(cx: &mut UnitCx<'_>, u: UnitRef, name: Symbol, span: Span) -> Option<Result<Def, Box<Diagnostic>>> {
    if u == UnitRef::SELF {
        return cx.own_value(name).map(Ok);
    }
    let iface = cx.imported[u.0 as usize];
    if !cx.quantum && iface.quantum_only.contains(&name) {
        return None;
    }
    // a quantum unit may use what of a classical unit can be evaluated while
    // a circuit is generated: types, constants, constant functions, macros
    // its other functions and objects exist only while a program runs
    if cx.quantum && !iface.quantum {
        let what = match (iface.function(name), iface.global(name)) {
            (Some(f), _) if !f.constant => Some("function that is not constant"),
            (None, Some(g)) if !g.constant => Some("object"),
            _ => None,
        };
        if let Some(what) = what {
            return Some(Err(Box::new(
                Diagnostic::new(Code::Eu03)
                    .with_message(format!(
                        "`{}::{}` is a classical {what}, which a quantum unit cannot use",
                        iface.unit,
                        cx.interner.resolve(name)
                    ))
                    .at(span)
                    .with_note(
                        "a quantum unit's code runs while its circuits are generated, before any \
                         classical program does; from a classical unit it may use types, constants, \
                         constant functions and macros",
                    )
                    .with_help("make it a constant or a constant function, or compute it in the quantum unit"),
            )));
        }
    }
    // a quantum unit's operators are inlined into the circuits of a quantum
    // unit using them, which needs their bodies
    if cx.quantum
        && iface.quantum
        && iface.function(name).is_some_and(|f| f.method.is_none())
        && let Some(frame) = cx.frame_for(iface)
    {
        return cx.frame_value(frame, name).map(Ok);
    }
    if let Some(f) = iface.function(name) {
        if f.entry && !cx.quantum {
            return Some(Ok(circuit_handle(cx, iface, f, span)));
        }
        if f.constant {
            // evaluating it needs its body, which the unit's source carries
            let Some(frame) = cx.frame_for(iface) else {
                return Some(Err(Box::new(
                    Diagnostic::new(Code::Es17)
                        .with_message(format!(
                            "`{}::{}` is a constant function, but unit `{}` was compiled without its \
                             source",
                            iface.unit,
                            cx.interner.resolve(name),
                            iface.unit
                        ))
                        .at(span)
                        .with_note(
                            "a constant function is evaluated wherever it is called, so its body \
                             must travel with the unit's interface",
                        )
                        .with_help(format!("compile `{}` again with this `tqc`", iface.unit)),
                )));
            };
            return cx.frame_value(frame, name).map(Ok);
        }
        return Some(Ok(extern_fn(cx, iface, name)));
    }
    if iface.global(name).is_some() {
        return Some(Ok(import_global(cx, iface, name, span)));
    }
    None
}

/// A quantum unit's `[entry]` operator, as a classical unit holds it: a
/// circuit handle, `circuit<Sig>`: an object the program's startup defines,
/// referring to the circuit's document, which the program embeds when it is
/// linked.
fn circuit_handle(cx: &mut UnitCx<'_>, iface: &Interface, f: &super::interface::ExportFn, span: Span) -> Def {
    let key = (std::ptr::from_ref(iface) as usize, f.name);
    if let Some(&id) = cx.imported_globals.get(&key) {
        return Def::Global(id);
    }
    let params = f.params.iter().map(|&t| translate(cx, iface, t)).collect();
    let ret = translate(cx, iface, f.ret);
    let ty = cx.unit.types.circuit(params, ret);
    let global = tir::Global {
        deprecated: f.deprecated.clone(),
        ..imported(iface, f.name, ty, span)
    };
    add_imported(cx, key, global)
}

/// An object `name` of type `ty` that the unit `iface` defines, as this
/// unit refers to it.
fn imported(iface: &Interface, name: Symbol, ty: Ty, span: Span) -> tir::Global {
    tir::Global {
        name,
        ty,
        constant: false,
        linkage: crate::ast::Linkage::Program,
        kind: GlobalKind::Imported(iface.unit.clone()),
        init: None,
        value: None,
        startup: false,
        span,
        symbol: None,
        deprecated: None,
    }
}

/// Adds `global` to this unit, as what `key` names from now on.
fn add_imported(cx: &mut UnitCx<'_>, key: (usize, Symbol), global: tir::Global) -> Def {
    let id = tir::GlobalId(cx.unit.globals.len() as u32);
    cx.unit.globals.push(global);
    cx.imported_globals.insert(key, id);
    Def::Global(id)
}

/// A function or object of another unit whose frame is being declared,
/// `static` or not.
pub fn foreign_value(cx: &mut UnitCx<'_>, iface: &Interface, name: Symbol, span: Span) -> Option<Def> {
    if iface.any_function(name).is_some() {
        return Some(extern_fn(cx, iface, name));
    }
    iface.any_global(name)?;
    Some(import_global(cx, iface, name, span))
}

/// An imported function.
fn extern_fn(cx: &mut UnitCx<'_>, iface: &Interface, name: Symbol) -> Def {
    let f = iface.any_function(name).expect("the caller found it");
    extern_of(cx, iface, f)
}

/// A function of an imported unit (a method included) as one this unit
/// calls by its symbol.
pub fn extern_of(cx: &mut UnitCx<'_>, iface: &Interface, f: &super::interface::ExportFn) -> Def {
    let name = f.name;
    let key = (std::ptr::from_ref(f) as usize, name);
    if let Some(&id) = cx.extern_ids.get(&key) {
        return Def::Extern(id);
    }
    let params = f.params.iter().map(|&t| translate(cx, iface, t)).collect();
    let ret = translate(cx, iface, f.ret);
    let id = tir::ExternId(cx.unit.externs.len() as u32);
    let method = f.method.map(|m| tir::MethodOf {
        owner: import_adt(cx, iface, m.owner),
        ..m
    });
    cx.unit.externs.push(tir::ExternFn {
        unit: iface.unit.clone(),
        name,
        method,
        params,
        ret,
        symbol: f.symbol.clone(),
        deprecated: f.deprecated.clone(),
    });
    cx.extern_ids.insert(key, id);
    Def::Extern(id)
}

/// An imported object or constant.
fn import_global(cx: &mut UnitCx<'_>, iface: &Interface, name: Symbol, span: Span) -> Def {
    let key = (std::ptr::from_ref(iface) as usize, name);
    if let Some(&id) = cx.imported_globals.get(&key) {
        return Def::Global(id);
    }
    let g = iface.any_global(name).expect("the caller found it");
    let ty = translate(cx, iface, g.ty);
    let global = tir::Global {
        constant: g.constant,
        value: g.value.clone(),
        symbol: g.symbol.clone(),
        deprecated: g.deprecated.clone(),
        ..imported(iface, name, ty, span)
    };
    add_imported(cx, key, global)
}

/// The name an imported unit was given.
fn unit_name<'x>(cx: &UnitCx<'x>, u: UnitRef) -> &'x str {
    &cx.imported[u.0 as usize].unit
}

/// An imported unit has no such item.
pub fn no_such_item(cx: &UnitCx<'_>, u: UnitRef, name: Symbol, span: Span, kind: &str) -> Diagnostic {
    let interner = cx.interner;
    let text = interner.resolve(name);
    if u == UnitRef::SELF {
        return Diagnostic::new(Code::Es04)
            .with_message(format!("unit `{}` has no {kind} named `{text}`", cx.unit.name))
            .at(span)
            .with_note("the units that import one another name each other's items as they declare them");
    }
    let iface = cx.imported[u.0 as usize];
    if !cx.quantum && iface.quantum_only.contains(&name) {
        return Diagnostic::new(Code::Eu03)
            .with_message(format!(
                "`{}::{text}` belongs to the quantum side of `{}`, which a classical unit cannot use",
                iface.unit, iface.unit
            ))
            .at(span)
            .with_note(
                "from a quantum unit, a classical unit may use its types that hold no quantum \
                 values, its constants, and its `[entry]` operators, as circuit handles; its \
                 other operators and its objects exist only inside circuits",
            )
            .with_help("mark the operator `[entry]` to run it as a circuit, or move what is needed into a classical unit");
    }

    // not there, or `static`: suggest the closest name it does offer
    let candidates: Vec<&str> = iface
        .fns
        .iter()
        .filter(|f| f.linkage == crate::ast::Linkage::Program)
        .map(|f| interner.resolve(f.name))
        .chain(
            iface
                .globals
                .iter()
                .filter(|g| g.linkage == crate::ast::Linkage::Program)
                .map(|g| interner.resolve(g.name)),
        )
        .chain(
            iface
                .types
                .adts()
                .iter()
                .filter(|a| iface.named_type(a.name).is_some())
                .map(|a| interner.resolve(a.name)),
        )
        .collect();
    let mut d = Diagnostic::new(Code::Es04)
        .with_message(format!(
            "unit `{}` has no {kind} named `{text}` that other units can use",
            unit_name(cx, u)
        ))
        .at(span)
        .with_note(
            "only what a unit declares without `static` can be used from another unit; a \
             `static` item is private to its own unit",
        );
    if let Some(s) = report::closest(text, candidates) {
        d = d.with_help(format!("did you mean `{s}`?"));
    }
    d
}

/// A path starting with a name that is neither an imported unit nor a type.
pub fn unknown_unit(cx: &UnitCx<'_>, name: Symbol, span: Span) -> Diagnostic {
    let text = cx.interner.resolve(name);
    let mut d = Diagnostic::new(Code::Es04)
        .with_message(format!("there is no imported unit or type named `{text}` here"))
        .at(span);
    if cx.available.iter().any(|i| crate::driver::written(&i.unit) == text) {
        d = d.with_help(format!("add `import {text};` to use the unit `{text}`"));
    } else {
        d = d.with_note(
            "a path `a::b` starts with a unit this one imports, or with an enumeration \
             whose variant `b` is",
        );
    }
    d
}

/// A type of an imported unit's interface.
pub(super) fn translate(cx: &mut UnitCx<'_>, iface: &Interface, ty: Ty) -> Ty {
    match ty {
        Ty::Adt(id) => Ty::Adt(import_adt(cx, iface, id)),
        Ty::Tuple(_) => {
            let elems: Vec<Ty> = iface.types.as_tuple(ty).expect("a tuple type").to_vec();
            let elems = elems.into_iter().map(|e| translate(cx, iface, e)).collect();
            cx.unit.types.tuple(elems)
        }
        Ty::Fn(_) | Ty::Closure(_) | Ty::Circuit(_) => {
            let (params, ret) = iface.types.as_sig(ty).expect("a signature");
            let params: Vec<Ty> = params.to_vec();
            let params = params.into_iter().map(|p| translate(cx, iface, p)).collect();
            let ret = translate(cx, iface, ret);
            match ty {
                Ty::Fn(_) => cx.unit.types.function(params, ret),
                Ty::Closure(_) => cx.unit.types.closure(params, ret),
                _ => cx.unit.types.circuit(params, ret),
            }
        }
        Ty::Ref(id) | Ty::Slice(id) | Ty::Array(id) | Ty::Growable(id) => match iface.types.compound(id) {
            Compound::Ref { access, target } => {
                let t = translate(cx, iface, target);
                cx.unit.types.reference(access, t)
            }
            Compound::Slice { access, elem } => {
                let e = translate(cx, iface, elem);
                cx.unit.types.slice(access, e)
            }
            Compound::Array { elem, len } => {
                let e = translate(cx, iface, elem);
                cx.unit.types.array(e, len)
            }
            Compound::Growable { elem } => {
                let e = translate(cx, iface, elem);
                cx.unit.types.growable(e)
            }
            Compound::Qmap { .. } => unreachable!("only a map locale type is a map locale's compound"),
        },
        Ty::Qmap(_) => {
            let (key, entry) = iface.types.as_qmap(ty).expect("a map locale type");
            let e = translate(cx, iface, entry);
            cx.unit.types.qmap(key, e)
        }
        other => other,
    }
}

/// A structure or enumeration of an interface, copied into this unit unless it
/// is already there.
fn import_adt(cx: &mut UnitCx<'_>, iface: &Interface, id: AdtId) -> AdtId {
    let def = iface.types.adt(id);
    // one of this unit's own types, come back through another unit, is the
    // unit's own again (an instance of its generic included, even one only
    // the other unit made)
    let origin = def.origin.clone().filter(|o| *o != cx.unit.name);
    let args: Vec<Arg> = def
        .args
        .iter()
        .map(|&a| match a {
            Arg::Type(t) => Arg::Type(translate(cx, iface, t)),
            c => c,
        })
        .collect();
    if let Some(existing) = cx.unit.types.find_instance(origin.as_deref(), def.name, &args) {
        return existing;
    }
    // add it before its fields, so that a field referring back to it finds it
    let new = cx.unit.types.add_adt(AdtDef {
        args,
        name: def.name,
        origin,
        linkage: def.linkage,
        open: def.open,
        opaque: def.opaque,
        kind: match def.kind {
            AdtKind::Struct { .. } => AdtKind::Struct { fields: Vec::new() },
            AdtKind::Enum { .. } => AdtKind::Enum { variants: Vec::new() },
        },
        repr: def.repr,
        span: def.span,
    });
    let field = |cx: &mut UnitCx<'_>, f: &FieldDef| FieldDef {
        name: f.name,
        ty: translate(cx, iface, f.ty),
        span: f.span,
    };
    let kind = match &def.kind {
        AdtKind::Struct { fields } => AdtKind::Struct {
            fields: fields.iter().map(|f| field(cx, f)).collect(),
        },
        AdtKind::Enum { variants } => AdtKind::Enum {
            variants: variants
                .iter()
                .map(|v| VariantDef {
                    name: v.name,
                    shape: v.shape,
                    fields: v.fields.iter().map(|f| field(cx, f)).collect(),
                    span: v.span,
                })
                .collect(),
        },
    };
    cx.unit.types.adt_mut(new).kind = kind;
    mark_if_copyable(cx, new);
    new
}

/// Records that another unit's type is copyable if some unit gives it
/// `$copy`: the unit that declares it, or, for an `[open]` type, another.
fn mark_if_copyable(cx: &mut UnitCx<'_>, id: AdtId) {
    let def = cx.unit.types.adt(id);
    let (name, origin) = (def.name, def.origin.clone());
    let Some(copy) = cx.interner.get("copy") else { return };
    let owner = origin.clone().unwrap_or_else(|| cx.unit.name.clone());
    let available = cx.available;
    // a generic type's methods are in its unit's source; declaring them
    // marks it
    if let Some(iface) = available.iter().find(|i| i.unit == owner)
        && iface.ast.is_some()
    {
        cx.frame_for(iface);
    }
    if available.iter().any(|i| i.method(&owner, name, copy, true).is_some()) {
        cx.unit.types.mark_copyable(origin, name);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::diag::Code;
    use crate::intern::Interner;
    use crate::sema::interface::Source;
    use crate::sema::testing::analyzed_in;
    use crate::tir::{FnKind, Value};

    use super::*;

    /// The interface of the clean unit `name`, carrying its source and tree
    /// as the driver would give them.
    fn exporting(name: &str, src: &str, imports: &[Interface], interner: &mut Interner) -> Interface {
        let (u, d) = analyzed_in(name, src, imports, interner);
        assert!(d.iter().all(|d| !d.is_error()), "{d:?}");
        let mut i = Interface::of(&u);
        i.ast = Some(Arc::new(crate::parse::testing::unit_in(src, interner)));
        i.source = Some(Source {
            path: format!("{name}.tq"),
            text: src.to_owned(),
            embeds: Vec::new(),
        });
        i
    }

    fn clean(name: &str, src: &str, imports: &[Interface], interner: &mut Interner) -> tir::Unit {
        let (u, d) = analyzed_in(name, src, imports, interner);
        let errors: Vec<String> = d.iter().filter(|d| d.is_error()).map(|d| format!("{}: {}", d.code, d.message)).collect();
        assert!(errors.is_empty(), "{errors:?}");
        u
    }

    const GEO: &str = "static fn twice(x: i64) -> i64 { x * 2 }\n\
                       fn doubled<T>(x: T) -> T { x + x }\n\
                       fn scaled<T>(x: T) -> i64 { twice(3) }\n\
                       struct Pair<A, B> { a: A, b: B }\n\
                       enum Opt<T> { None, Some(T) }\n\
                       fn cube(n: constexpr i64) -> constexpr i64 { n * n * n }\n\
                       static fn hidden<T>(x: T) -> T { x }";

    #[test]
    fn another_units_generic_function_is_instantiated_here() {
        let mut i = Interner::new();
        let geo = exporting("geo", GEO, &[], &mut i);
        let u = clean(
            "app",
            "import geo;\nfn f() -> i32 { geo::doubled(3i32) + geo::doubled::<i32>(4) }",
            &[geo],
            &mut i,
        );
        let instances: Vec<&tir::Fn> = u.fns.iter().filter(|f| !f.args.is_empty()).collect();
        assert_eq!(instances.len(), 1, "one set of arguments makes one instance");
        assert_eq!(instances[0].origin.as_deref(), Some("geo"));
        assert_eq!(instances[0].kind, FnKind::Runtime);
        assert!(u.borrowed.iter().any(|(unit, _)| unit == "geo"));
    }

    #[test]
    fn another_units_alias_of_an_enumeration_names_its_variants() {
        let mut i = Interner::new();
        let low = exporting("low", "enum Error { NotFound, Io(u32) }", &[], &mut i);
        let mid = exporting("mid", "import low;\ntype Error = low::Error;", std::slice::from_ref(&low), &mut i);
        clean(
            "app",
            "import low;\nimport mid;\n\
             fn f() -> mid::Error { mid::Error::Io(5) }\n\
             fn g(e: low::Error) -> u32 { match e { mid::Error::Io(c) => c, mid::Error::NotFound => 0 } }",
            &[low, mid],
            &mut i,
        );
    }

    #[test]
    fn a_generic_body_reaches_its_units_static_items() {
        let mut i = Interner::new();
        let geo = exporting("geo", GEO, &[], &mut i);
        let u = clean("app", "import geo;\nfn f() -> i64 { geo::scaled(true) }", &[geo], &mut i);
        assert!(u.externs.iter().any(|x| x.unit == "geo" && i.resolve(x.name) == "twice"));
    }

    #[test]
    fn another_units_generic_types_are_one_type_however_named() {
        let mut i = Interner::new();
        let geo = exporting("geo", GEO, &[], &mut i);
        let u = clean(
            "app",
            "import geo;\n\
             fn f() -> bool { let p: geo::Pair<i32, bool> = geo::Pair { a: 1i32, b: true }; p.b }\n\
             fn g() -> i32 { let o: geo::Opt<i32> = geo::Opt::None; match o { geo::Opt::Some(x) => x, geo::Opt::None => 0 } }",
            &[geo],
            &mut i,
        );
        let pairs = u.types.adts().iter().filter(|a| i.resolve(a.name) == "Pair").count();
        assert_eq!(pairs, 1);
        let pair = u.types.adts().iter().find(|a| i.resolve(a.name) == "Pair").unwrap();
        assert_eq!(pair.origin.as_deref(), Some("geo"));
    }

    #[test]
    fn another_units_constant_function_is_evaluated_here() {
        let mut i = Interner::new();
        let geo = exporting("geo", GEO, &[], &mut i);
        let u = clean("app", "import geo;\nlet N: const i64 = geo::cube(3);", &[geo], &mut i);
        let n = u.globals.iter().find(|g| i.resolve(g.name) == "N").unwrap();
        assert_eq!(n.value, Some(Value::Int(27, tir::IntTy::I64)));
    }

    #[test]
    fn a_static_generic_cannot_be_named_from_outside() {
        let mut i = Interner::new();
        let geo = exporting("geo", GEO, &[], &mut i);
        let (_, d) = analyzed_in("app", "import geo;\nfn f() -> i32 { geo::hidden(1i32) }", &[geo], &mut i);
        assert_eq!(d.iter().map(|d| d.code).collect::<Vec<_>>(), [Code::Es04]);
    }

    #[test]
    fn a_generic_body_uses_its_own_units_imports() {
        let mut i = Interner::new();
        let low = exporting("low", "fn one() -> i64 { 1 }\nstruct Tag { n: i64 }", &[], &mut i);
        let mid = exporting(
            "mid",
            "import low as b;\nfn plus_one<T>(x: T) -> b::Tag { b::Tag { n: b::one() } }",
            std::slice::from_ref(&low),
            &mut i,
        );
        let u = clean(
            "app",
            "import mid;\nfn f() -> i64 { mid::plus_one(true).n }",
            &[mid, low],
            &mut i,
        );
        assert!(u.externs.iter().any(|x| x.unit == "low"));
        let tags: Vec<_> = u.types.adts().iter().filter(|a| i.resolve(a.name) == "Tag").collect();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].origin.as_deref(), Some("low"));
    }

    #[test]
    fn a_mistake_in_an_instance_says_where_it_was_asked_for() {
        let mut i = Interner::new();
        let geo = exporting("geo", GEO, &[], &mut i);
        let (_, d) = analyzed_in("app", "import geo;\nfn f() -> bool { geo::doubled(true) }", &[geo], &mut i);
        let e = d.iter().find(|d| d.is_error()).expect("`true + true` is refused");
        assert!(
            e.labels.iter().any(|l| l.message.contains("geo::doubled<bool>")),
            "{:?}",
            e.labels
        );
    }

    const SHAPES: &str = "[open]\nstruct Circle { r: i64 }\n\
                          struct Square { s: i64 }\n\
                          impl Circle { fn area(self: *Circle) -> i64 { 3 * self.r * self.r } }\n\
                          impl Square {\n\
                              fn area(self: *Square) -> i64 { self.s * self.s }\n\
                              fn $add(self: Square, o: Square) -> Square { Square { s: self.s + o.s } }\n\
                          }\n\
                          struct Boxed<T> { v: T }\n\
                          impl Boxed<T> { fn get(self: *Boxed<T>) -> *T { &self.v } }";

    #[test]
    fn another_units_methods_are_called_through_its_types() {
        let mut i = Interner::new();
        let shapes = exporting("shapes", SHAPES, &[], &mut i);
        let u = clean(
            "app",
            "import shapes;\n\
             fn f(c: shapes::Circle, s: shapes::Square) -> i64 {\n\
                 let t = s + shapes::Square { s: 1 };\n\
                 let b = shapes::Boxed { v: 5i64 };\n\
                 c.area() + t.area() + *b.get()\n\
             }",
            &[shapes],
            &mut i,
        );
        let methods: Vec<&tir::ExternFn> = u.externs.iter().filter(|x| x.method.is_some()).collect();
        assert!(methods.len() >= 3, "{methods:?}");
        assert!(u.fns.iter().any(|f| f.method.is_some() && f.origin.as_deref() == Some("shapes")));
    }

    #[test]
    fn an_open_type_takes_methods_from_other_units() {
        let mut i = Interner::new();
        let shapes = exporting("shapes", SHAPES, &[], &mut i);
        let extra = exporting(
            "extra",
            "import shapes;\nfn shapes::Circle.diameter(self: *shapes::Circle) -> i64 { 2 * self.r }",
            std::slice::from_ref(&shapes),
            &mut i,
        );
        assert!(extra.fns.iter().any(|f| f.method.is_some()));
        clean(
            "app",
            "import shapes;\nimport extra;\nfn f(c: shapes::Circle) -> i64 { c.diameter() + c.area() }",
            &[shapes, extra],
            &mut i,
        );
    }

    #[test]
    fn a_closed_type_takes_no_methods_from_other_units() {
        let mut i = Interner::new();
        let shapes = exporting("shapes", SHAPES, &[], &mut i);
        let (_, d) = analyzed_in(
            "extra",
            "import shapes;\nfn shapes::Square.side(self: *shapes::Square) -> i64 { self.s }",
            &[shapes],
            &mut i,
        );
        assert_eq!(d.iter().map(|d| d.code).collect::<Vec<_>>(), [Code::Es19]);
    }

    #[test]
    fn a_method_the_type_already_has_cannot_be_added() {
        let mut i = Interner::new();
        let shapes = exporting("shapes", SHAPES, &[], &mut i);
        let (_, d) = analyzed_in(
            "extra",
            "import shapes;\nfn shapes::Circle.area(self: *shapes::Circle) -> i64 { 0 }",
            &[shapes],
            &mut i,
        );
        assert_eq!(d.iter().map(|d| d.code).collect::<Vec<_>>(), [Code::Es05]);
    }
}
