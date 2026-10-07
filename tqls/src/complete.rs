//! What may be written at the cursor.
//!
//! The text before the cursor on its line says what kind of thing is being
//! written: a directive after `#`, a macro after `@`, an annotation after the
//! `[` opening a group, a unit after `import`, an item after `unit::`, a
//! field or method after `.`, and otherwise a keyword or a name in scope. The
//! names come from the file's last check that reached a typed unit, since the
//! text being typed seldom checks.

use std::collections::HashSet;
use std::path::Path;

use lsp_types::{CompletionItem, CompletionItemKind};
use topiq::ast::{ItemKind, Linkage};
use topiq::driver::program::MemberKind;
use topiq::lex::Keyword;
use topiq::span::SourceId;
use topiq::tir::{self, Ty};

use crate::analysis::Checked;

/// The preprocessing directives.
const DIRECTIVES: [&str; 14] = [
    "unit", "define", "undef", "if", "ifdef", "ifndef", "elif", "else", "endif", "error", "warning", "pragma",
    "embed", "include",
];

/// Functions the compiler provides rather than a library unit declares.
const PROVIDED: [&str; 7] = ["print", "eprint", "exit", "panic", "abort", "clone", "qcopy"];

/// What may be written at `offset` of `text`, the file at `path`, whose last
/// typed check, if any, is `checked`.
pub fn complete(checked: Option<&Checked>, text: &str, offset: usize, path: &Path) -> Vec<CompletionItem> {
    let offset = offset.min(text.len());
    let line = &text[text[..offset].rfind('\n').map_or(0, |i| i + 1)..offset];
    let word_len = line.bytes().rev().take_while(|&b| b.is_ascii_alphanumeric() || b == b'_').count();
    let before = &line[..line.len() - word_len];
    let lead = before.trim_start();
    let file = checked.and_then(|c| c.file(path));

    // a directive, and a pragma's name and argument
    if let Some(directive) = lead.strip_prefix('#') {
        let words: Vec<&str> = directive.split_whitespace().collect();
        let open = directive.is_empty() || directive.ends_with(char::is_whitespace);
        return match (words.as_slice(), open) {
            ([], _) => items(&DIRECTIVES, CompletionItemKind::KEYWORD),
            (["unit"], true) => items(&["classical", "quantum"], CompletionItemKind::KEYWORD),
            (["pragma"], true) => items(&["conductor", "fragment", "qalloc"], CompletionItemKind::KEYWORD),
            (["pragma", "fragment"], true) => items(&["finite", "stabilizer"], CompletionItemKind::VALUE),
            (["pragma", "qalloc"], true) => items(&["linear", "reuse"], CompletionItemKind::VALUE),
            _ => Vec::new(),
        };
    }

    // a macro
    if before.ends_with('@') {
        let mut names: Vec<&str> = topiq::sema::macros::MACROS.to_vec();
        names.extend(topiq::sema::macros::QUANTUM);
        names.push("embed");
        return items(&names, CompletionItemKind::FUNCTION);
    }

    // an annotation, at the start of its group
    if lead.starts_with('[') && before.trim_end().ends_with('[') {
        let names: Vec<&str> = topiq::ast::annot::StdAnnotation::ALL.iter().map(|a| a.name()).collect();
        return items(&names, CompletionItemKind::PROPERTY);
    }

    // a unit to import: the library's, and the files beside this one
    if lead == "import " || (lead.starts_with("import") && lead.trim_end() == "import") {
        let mut names: Vec<String> = topiq::library::NAMES.iter().map(|&n| n.to_owned()).collect();
        names.extend(sibling_units(path));
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        return items(&names, CompletionItemKind::MODULE);
    }

    let Some((checked, file)) = checked.zip(file) else {
        return general(None, None, offset);
    };

    // `unit::name` or `Enum::Variant`
    if let Some(qualified) = before.strip_suffix("::") {
        let q_len = qualified.bytes().rev().take_while(|&b| b.is_ascii_alphanumeric() || b == b'_').count();
        let q = &qualified[qualified.len() - q_len..];
        return qualified_items(checked, file, q);
    }

    // a field or method of what comes before the `.`
    if before.ends_with('.') && !before.ends_with("..") {
        let dot = offset - word_len - 1;
        return members(checked, file, dot);
    }

    general(Some(checked), Some(file), offset)
}

fn items(names: &[&str], kind: CompletionItemKind) -> Vec<CompletionItem> {
    names.iter().map(|&n| item(n.to_owned(), kind, None)).collect()
}

fn item(label: String, kind: CompletionItemKind, detail: Option<String>) -> CompletionItem {
    CompletionItem {
        label,
        kind: Some(kind),
        detail,
        ..CompletionItem::default()
    }
}

/// The units beside `path`.
fn sibling_units(path: &Path) -> Vec<String> {
    let own = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let Some(dir) = path.parent().and_then(|d| std::fs::read_dir(d).ok()) else {
        return Vec::new();
    };
    dir.filter_map(|e| {
        let p = e.ok()?.path();
        let ext = p.extension()?.to_str()?;
        let stem = p.file_stem()?.to_str()?.to_owned();
        (matches!(ext, "tq" | "tcu" | "tqu") && stem != own).then_some(stem)
    })
    .collect::<HashSet<_>>()
    .into_iter()
    .collect()
}

/// The items of the unit named `unit` that another unit may use, or all of
/// them for the unit's own file.
fn unit_items(checked: &Checked, unit: &str, own: bool) -> Vec<CompletionItem> {
    let Some(m) = checked.member(unit) else { return Vec::new() };
    let MemberKind::Source(out) = &m.kind else { return Vec::new() };
    let Some(ast) = out.unit.as_ref() else { return Vec::new() };
    let interner = checked.loaded.session.interner();
    ast.items
        .iter()
        .filter(|decl| own || decl.node.linkage == Linkage::Program)
        .filter_map(|decl| {
            let name = crate::navigate::item_name(&decl.node.kind)?;
            let kind = match &decl.node.kind {
                ItemKind::Fn(_) => CompletionItemKind::FUNCTION,
                ItemKind::Struct { .. } | ItemKind::TypeAlias { .. } | ItemKind::Base { .. } => CompletionItemKind::STRUCT,
                ItemKind::Enum { .. } => CompletionItemKind::ENUM,
                ItemKind::Let { .. } => CompletionItemKind::CONSTANT,
                ItemKind::Locale { .. } => CompletionItemKind::MODULE,
                _ => CompletionItemKind::VALUE,
            };
            Some(item(interner.resolve(name.node).to_owned(), kind, Some(unit.to_owned())))
        })
        .collect()
}

/// After `q::`: a unit's items, or an enumeration's variants.
fn qualified_items(checked: &Checked, file: SourceId, q: &str) -> Vec<CompletionItem> {
    let own = checked.loaded.session.sources().file(file).unit_name() == q;
    if checked.member(q).is_some() {
        return unit_items(checked, q, own);
    }
    let Some(unit) = checked.outcome(file).and_then(|o| o.tir()) else { return Vec::new() };
    let interner = checked.loaded.session.interner();
    let Some(def) = unit.types.adts().iter().find(|d| interner.resolve(d.name) == q) else {
        return Vec::new();
    };
    def.variants()
        .iter()
        .map(|v| item(interner.resolve(v.name).to_owned(), CompletionItemKind::ENUM_MEMBER, Some(q.to_owned())))
        .collect()
}

/// After `.` at `dot`: the fields and methods of the type of the expression
/// ending there.
fn members(checked: &Checked, file: SourceId, dot: usize) -> Vec<CompletionItem> {
    let Some(unit) = checked.outcome(file).and_then(|o| o.tir()) else { return Vec::new() };
    let Some(mut ty) = crate::navigate::type_ending_at(unit, file, dot) else { return Vec::new() };
    while let Some((_, inner)) = unit.types.as_ref(ty) {
        ty = inner;
    }
    let interner = checked.loaded.session.interner();
    let show = |t: Ty| unit.types.display(t, interner);
    let mut out = Vec::new();
    match ty {
        Ty::Adt(id) => {
            for f in unit.types.adt(id).fields() {
                out.push(item(interner.resolve(f.name).to_owned(), CompletionItemKind::FIELD, Some(show(f.ty))));
            }
            let mut seen = HashSet::new();
            let methods = unit
                .fns
                .iter()
                .filter(|f| !f.attrs.hidden)
                .filter_map(|f| f.method.filter(|m| m.owner == id && m.receiver && !m.operator).map(|_| (f.name, f.ret)))
                .chain(
                    unit.externs
                        .iter()
                        .filter_map(|x| x.method.filter(|m| m.owner == id && m.receiver && !m.operator).map(|_| (x.name, x.ret))),
                );
            for (name, ret) in methods {
                if seen.insert(name) {
                    out.push(item(interner.resolve(name).to_owned(), CompletionItemKind::METHOD, Some(show(ret))));
                }
            }
        }
        Ty::Growable(_) | Ty::Slice(_) => {
            out.extend(items(&topiq::sema::growable::METHODS, CompletionItemKind::METHOD));
        }
        _ => {}
    }
    out
}

/// Anywhere else: keywords, types, the bindings in scope, and the items of
/// this unit, of `core` and of the units it imports.
fn general(checked: Option<&Checked>, file: Option<SourceId>, offset: usize) -> Vec<CompletionItem> {
    let mut out: Vec<CompletionItem> = Keyword::ALL
        .iter()
        .map(|k| item(k.text().to_owned(), CompletionItemKind::KEYWORD, None))
        .collect();
    out.extend(items(&["true", "false", "self"], CompletionItemKind::KEYWORD));
    out.extend(items(&topiq::sema::types::PRIMITIVES, CompletionItemKind::STRUCT));
    out.extend(items(&["qubit"], CompletionItemKind::STRUCT));
    out.extend(items(&PROVIDED, CompletionItemKind::FUNCTION));

    let (Some(checked), Some(file)) = (checked, file) else { return out };
    let interner = checked.loaded.session.interner();

    // the bindings of the function around the cursor, declared before it
    if let Some(unit) = checked.outcome(file).and_then(|o| o.tir()) {
        let mut seen = HashSet::new();
        for f in unit.fns.iter().filter(|f| around(f, file, offset)) {
            for l in &f.locals {
                let name = interner.resolve(l.name);
                if !name.is_empty() && (l.span.end as usize) <= offset && seen.insert(name) {
                    out.push(item(name.to_owned(), CompletionItemKind::VARIABLE, Some(unit.types.display(l.ty, interner))));
                }
            }
        }
    }

    // the items of this unit and of `core`, and the units it imports
    let own = checked.loaded.session.sources().file(file).unit_name().to_owned();
    out.extend(unit_items(checked, &own, true));
    out.extend(unit_items(checked, "core", false));
    for m in &checked.loaded.members {
        if m.name != own && m.name != "core" {
            out.push(item(m.name.clone(), CompletionItemKind::MODULE, None));
        }
    }
    out
}

/// Whether `offset` of `file` is inside the body of `f`.
fn around(f: &tir::Fn, file: SourceId, offset: usize) -> bool {
    let body = f.body.span;
    body.source == file && body.start as usize <= offset && offset <= body.end as usize
}
