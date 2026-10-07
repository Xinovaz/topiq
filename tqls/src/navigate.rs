//! What a name in a file is: its hover, where it is declared, and the file's
//! outline.
//!
//! The typed unit says what each expression and pattern refers to, and where
//! that was declared. Walking it once gives a list of facts, each a span of
//! the file and what it is; a name under the cursor is the smallest fact
//! around it that carries that name. What the typed unit does not hold (a
//! type written in an annotation, a cover, a unit named in a path) is found
//! among the syntax tree's items by name.

use lsp_types::{
    DocumentSymbol, Hover, HoverContents, Location, MarkupContent, MarkupKind, SymbolKind,
};
use topiq::ast::{self, FnName, ItemKind};
use topiq::driver::Outcome;
use topiq::driver::program::MemberKind;
use topiq::intern::{Interner, Symbol};
use topiq::lex::{Punct, Token};
use topiq::span::{SourceId, Span, Spanned};
use topiq::tir::visit::{Visit, walk_expr, walk_pat};
use topiq::tir::{self, Callee, ExprKind, PatKind, Ty};

use crate::analysis::{Checked, range};
use crate::lines::Encoding;

/// What a span of the file is.
struct Fact {
    span: Span,
    /// The name written there.
    name: Option<String>,
    /// What hovering over it shows, as Topiq.
    shown: String,
    /// Where what it names is declared.
    target: Option<Target>,
    /// Whether the comment above its declaration describes it, as it does
    /// an item's but not a local binding's.
    documented: bool,
}

/// Where a name is declared.
#[derive(Clone)]
enum Target {
    /// At a span of the program.
    At(Span),
    /// As an item of another unit, found by name.
    Item { unit: String, name: String },
}

/// Walks one file's part of a typed unit, recording a fact for each name.
struct Collector<'a> {
    unit: &'a tir::Unit,
    interner: &'a Interner,
    file: SourceId,
    func: Option<&'a tir::Fn>,
    out: Vec<Fact>,
}

impl Collector<'_> {
    fn name(&self, sym: Symbol) -> String {
        self.interner.resolve(sym).to_owned()
    }

    fn ty(&self, ty: Ty) -> String {
        self.unit.types.display(ty, self.interner)
    }

    /// Records that `span` names an item, if it is in this file.
    fn named(&mut self, span: Span, name: String, shown: String, target: Option<Target>) {
        if span.source == self.file && !name.is_empty() {
            self.out.push(Fact {
                span,
                name: Some(name),
                shown,
                target,
                documented: true,
            });
        }
    }

    fn local(&mut self, id: tir::LocalId, span: Span) {
        let Some(f) = self.func else { return };
        let l = f.local(id);
        let name = self.name(l.name);
        if span.source == self.file && !name.is_empty() {
            self.out.push(Fact {
                span,
                name: Some(name),
                shown: local_shown(self, f, id),
                target: Some(Target::At(l.span)),
                documented: false,
            });
        }
    }

    fn callee(&mut self, callee: Callee, span: Span) {
        match callee {
            Callee::Fn(id) => {
                let f = self.unit.func(id);
                if f.closure || f.attrs.hidden {
                    return;
                }
                let target = (!f.span.source.is_synthetic()).then_some(Target::At(f.span));
                self.named(span, self.name(f.name), signature(self.unit, self.interner, f), target);
            }
            Callee::Extern(id) => {
                let x = self.unit.extern_fn(id);
                let shown = extern_signature(self.unit, self.interner, x);
                let target = Target::Item {
                    unit: x.unit.clone(),
                    name: self.name(x.name),
                };
                self.named(span, self.name(x.name), shown, Some(target));
            }
        }
    }

    fn adt(&mut self, ty: Ty, span: Span) {
        if let Ty::Adt(id) = ty {
            let def = self.unit.types.adt(id);
            self.named(span, self.name(def.name), adt_shown(self.unit, self.interner, id), Some(Target::At(def.span)));
        }
    }

    fn variant(&mut self, ty: Ty, variant: u32, span: Span) {
        let Ty::Adt(id) = ty else { return };
        let def = self.unit.types.adt(id);
        let Some(v) = def.variants().get(variant as usize) else { return };
        let shown = format!(
            "{}::{}",
            self.unit.types.adt_name(id, self.interner),
            variant_shown(self.unit, self.interner, v)
        );
        self.named(span, self.name(v.name), shown, Some(Target::At(v.span)));
    }
}

impl Visit for Collector<'_> {
    fn expr(&mut self, e: &tir::Expr) {
        match &e.kind {
            ExprKind::Local(id) => self.local(*id, e.span),
            ExprKind::Global(id) => {
                let g = self.unit.global(*id);
                let target = match &g.kind {
                    tir::GlobalKind::Imported(unit) => Some(Target::Item {
                        unit: unit.clone(),
                        name: self.name(g.name),
                    }),
                    _ => Some(Target::At(g.span)),
                };
                self.named(e.span, self.name(g.name), global_shown(self.unit, self.interner, g), target);
            }
            ExprKind::Call { callee, .. } | ExprKind::FnRef(callee) => self.callee(*callee, e.span),
            ExprKind::Field { base, field } => {
                if let Ty::Adt(id) = base.ty
                    && let Some(f) = self.unit.types.adt(id).fields().get(*field as usize)
                {
                    let owner = self.unit.types.adt_name(id, self.interner);
                    let shown = format!("{owner}.{}: {}", self.name(f.name), self.ty(f.ty));
                    self.named(e.span, self.name(f.name), shown, Some(Target::At(f.span)));
                }
            }
            ExprKind::StructLit { .. } => self.adt(e.ty, e.span),
            ExprKind::Variant { variant, .. } => self.variant(e.ty, *variant, e.span),
            _ => {}
        }
        // any expression shows its type, for what is not a name
        if e.span.source == self.file && e.ty != Ty::Never {
            self.out.push(Fact {
                span: e.span,
                name: None,
                shown: self.ty(e.ty),
                target: None,
                documented: false,
            });
        }
        walk_expr(self, e);
    }

    fn pat(&mut self, p: &tir::Pat) {
        match &p.kind {
            PatKind::Bind(id) | PatKind::BindRef(id) => self.local(*id, p.span),
            PatKind::Variant { variant, .. } => self.variant(p.ty, *variant, p.span),
            PatKind::Struct { .. } => self.adt(p.ty, p.span),
            _ => {}
        }
        walk_pat(self, p);
    }
}

/// Every fact of the file `file` in `unit`.
fn facts(unit: &tir::Unit, interner: &Interner, file: SourceId) -> Vec<Fact> {
    let mut c = Collector {
        unit,
        interner,
        file,
        func: None,
        out: Vec::new(),
    };
    for f in &unit.fns {
        if f.span.source != file || f.attrs.hidden {
            continue;
        }
        c.func = Some(f);
        // each binding where it is declared, parameters included
        for (i, l) in f.locals.iter().enumerate() {
            c.local(tir::LocalId(i as u32), l.span);
        }
        if !f.closure {
            c.named(f.span, c.name(f.name), signature(unit, interner, f), Some(Target::At(f.span)));
        }
        c.block(&f.body);
    }
    c.func = None;
    for g in &unit.globals {
        c.named(g.span, c.name(g.name), global_shown(unit, interner, g), Some(Target::At(g.span)));
        if let Some(init) = &g.init {
            c.expr(init);
        }
    }
    for (i, def) in unit.types.adts().iter().enumerate() {
        if def.span.source != file {
            continue;
        }
        let id = tir::AdtId(i as u32);
        c.adt(Ty::Adt(id), def.span);
        for f in def.fields() {
            let shown = format!("{}.{}: {}", unit.types.adt_name(id, interner), c.name(f.name), c.ty(f.ty));
            c.named(f.span, c.name(f.name), shown, Some(Target::At(f.span)));
        }
        for v in 0..def.variants().len() {
            c.variant(Ty::Adt(id), v as u32, def.variants()[v].span);
        }
    }
    c.out
}

fn local_shown(c: &Collector<'_>, f: &tir::Fn, id: tir::LocalId) -> String {
    let l = f.local(id);
    let written = format!(
        "{}: {}{}",
        c.name(l.name),
        if l.constexpr { "constexpr " } else if l.constant { "const " } else { "" },
        c.ty(l.ty)
    );
    match (f.params.contains(&id), l.aux) {
        (true, _) => written,
        (false, true) => format!("aux let {written}"),
        (false, false) => format!("let {written}"),
    }
}

fn global_shown(unit: &tir::Unit, interner: &Interner, g: &tir::Global) -> String {
    let unit_of = match &g.kind {
        tir::GlobalKind::Imported(u) => format!("{u}::"),
        _ => String::new(),
    };
    format!(
        "let {unit_of}{}: {}{}",
        interner.resolve(g.name),
        if g.constant { "const " } else { "" },
        unit.types.display(g.ty, interner)
    )
}

/// `fn name(a: A, b: B) -> R`, with `Owner.` before a method's name.
fn signature(unit: &tir::Unit, interner: &Interner, f: &tir::Fn) -> String {
    let params: Vec<String> = f
        .params
        .iter()
        .map(|&p| {
            let l = f.local(p);
            format!(
                "{}: {}",
                interner.resolve(l.name),
                unit.types.display(l.ty, interner)
            )
        })
        .collect();
    let owner = f.method.map(|m| method_owner(unit, interner, m)).unwrap_or_default();
    format!(
        "fn {owner}{}({}){}",
        interner.resolve(f.name),
        params.join(", "),
        returns(unit, interner, f.ret)
    )
}

fn extern_signature(unit: &tir::Unit, interner: &Interner, x: &tir::ExternFn) -> String {
    let params: Vec<String> = x.params.iter().map(|&t| unit.types.display(t, interner)).collect();
    let owner = match x.method {
        Some(m) => method_owner(unit, interner, m),
        None => format!("{}::", x.unit),
    };
    format!(
        "fn {owner}{}({}){}",
        interner.resolve(x.name),
        params.join(", "),
        returns(unit, interner, x.ret)
    )
}

fn method_owner(unit: &tir::Unit, interner: &Interner, m: tir::MethodOf) -> String {
    let sigil = if m.operator { "$" } else { "" };
    format!("{}.{sigil}", unit.types.adt_name(m.owner, interner))
}

fn returns(unit: &tir::Unit, interner: &Interner, ret: Ty) -> String {
    if ret == Ty::Void {
        String::new()
    } else {
        format!(" -> {}", unit.types.display(ret, interner))
    }
}

/// `struct Name { a: A, b: B }` or `enum Name { A, B(T) }`.
fn adt_shown(unit: &tir::Unit, interner: &Interner, id: tir::AdtId) -> String {
    let def = unit.types.adt(id);
    let name = unit.types.adt_name(id, interner);
    let (kind, parts): (&str, Vec<String>) = match &def.kind {
        tir::AdtKind::Struct { fields } => (
            "struct",
            fields
                .iter()
                .map(|f| format!("{}: {}", interner.resolve(f.name), unit.types.display(f.ty, interner)))
                .collect(),
        ),
        tir::AdtKind::Enum { variants } => ("enum", variants.iter().map(|v| variant_shown(unit, interner, v)).collect()),
    };
    if parts.is_empty() {
        format!("{kind} {name} {{}}")
    } else {
        format!("{kind} {name} {{ {} }}", parts.join(", "))
    }
}

/// `A`, `B(T)` or `C { x: T }`.
fn variant_shown(unit: &tir::Unit, interner: &Interner, v: &tir::VariantDef) -> String {
    let name = interner.resolve(v.name);
    let ty = |t: Ty| unit.types.display(t, interner);
    match v.shape {
        tir::VariantShape::Unit => name.to_owned(),
        tir::VariantShape::Tuple => {
            let types: Vec<String> = v.fields.iter().map(|f| ty(f.ty)).collect();
            format!("{name}({})", types.join(", "))
        }
        tir::VariantShape::Struct => {
            let fields: Vec<String> = v
                .fields
                .iter()
                .map(|f| format!("{}: {}", interner.resolve(f.name), ty(f.ty)))
                .collect();
            format!("{name} {{ {} }}", fields.join(", "))
        }
    }
}

/// What the cursor at `offset` of the file `id` rests on.
struct Found {
    /// The token's span.
    at: Span,
    /// What hovering shows, as Topiq.
    shown: String,
    /// Where it is declared.
    target: Option<Target>,
    /// Whether the comment above its declaration describes it.
    documented: bool,
    /// What describes it otherwise.
    note: Option<String>,
}

/// The token at `offset`, or the word just before it.
fn token_at(out: &Outcome, id: SourceId, offset: usize) -> Option<(Token, Span)> {
    let tokens = out.tokens.iter().filter(|(_, s)| s.source == id);
    let word = |t: &Token| matches!(t, Token::Ident(_) | Token::OpName(_) | Token::MacroName(_) | Token::Kw(_));
    tokens
        .clone()
        .find(|(_, s)| s.start as usize <= offset && offset < s.end as usize)
        .or_else(|| tokens.clone().find(|(t, s)| s.end as usize == offset && word(t)))
        .copied()
}

fn find(checked: &Checked, id: SourceId, offset: usize) -> Option<Found> {
    let out = checked.outcome(id)?;
    let interner = checked.loaded.session.interner();
    let (token, at) = token_at(out, id, offset)?;
    let facts = out.tir().map(|u| facts(u, interner, id)).unwrap_or_default();
    let around = |f: &&Fact| f.span.start <= at.start && at.end <= f.span.end;
    let found = |f: &Fact| Found {
        at,
        shown: f.shown.clone(),
        target: f.target.clone(),
        documented: f.documented,
        note: None,
    };

    // a name: the smallest fact around it that names it
    if let Token::Ident(s) | Token::OpName(s) = token {
        let word = interner.resolve(s);
        let named = facts.iter().filter(around).filter(|f| f.name.as_deref() == Some(word));
        return match named.min_by_key(|f| f.span.len()) {
            Some(f) => Some(found(f)),
            None => by_name(checked, out, id, at, word),
        };
    }

    // anything else: the type of the smallest expression around it
    let f = facts.iter().filter(around).filter(|f| f.name.is_none()).min_by_key(|f| f.span.len())?;
    Some(found(f))
}

/// A name the typed unit does not place: a unit, one of a unit's items
/// written `unit::name`, or an item of this unit, by name.
fn by_name(checked: &Checked, out: &Outcome, id: SourceId, at: Span, word: &str) -> Option<Found> {
    let tokens: Vec<&(Token, Span)> = out.tokens.iter().filter(|(_, s)| s.source == id).collect();
    let i = tokens.iter().position(|(_, s)| *s == at)?;
    let interner = checked.loaded.session.interner();
    let item = |(span, shown): (Option<Span>, String)| Found {
        at,
        shown,
        target: span.map(Target::At),
        documented: true,
        note: None,
    };

    // `unit::name`
    if i >= 2
        && tokens[i - 1].0.is(Punct::ColonColon)
        && let Token::Ident(u) = tokens[i - 2].0
        && let Some(found) = item_of_unit(checked, interner.resolve(u), word)
    {
        return Some(item(found));
    }

    // a unit, described by its opening comment
    if let Some(m) = checked.member(word)
        && let MemberKind::Source(o) = &m.kind
        && let Some(file) = o.source
    {
        let head = leading_comment(checked.text(file));
        return Some(Found {
            at,
            shown: format!("import {word};"),
            target: Some(Target::At(Span::new(file, 0, 0))),
            documented: false,
            note: (!head.is_empty()).then_some(head),
        });
    }

    // an item of this unit
    let unit = checked.loaded.session.sources().file(id).unit_name();
    item_of_unit(checked, unit, word).map(item)
}

/// The item `name` of the unit `unit`: where its name is written, if in a
/// file on disk, and its declaration as written.
fn item_of_unit(checked: &Checked, unit: &str, name: &str) -> Option<(Option<Span>, String)> {
    let m = checked.member(unit)?;
    let MemberKind::Source(out) = &m.kind else { return None };
    let file = out.source?;
    let interner = checked.loaded.session.interner();
    let ast = out.unit.as_ref()?;
    let item = ast
        .items
        .iter()
        .find(|item| item_name(&item.node.kind).is_some_and(|n| interner.resolve(n.node) == name))?;
    let span = item_name(&item.node.kind)?.span;
    let shown = declaration(checked.text(file), item.span);
    Some((checked.path(file).is_some().then_some(span), shown))
}

/// An item's declaration as written: whole if short, else its first line,
/// annotations left out.
fn declaration(text: &str, span: Span) -> String {
    let written = text.get(span.range()).unwrap_or("");
    let lines: Vec<&str> = written.lines().skip_while(|l| l.trim_start().starts_with('[')).collect();
    if lines.len() <= 6 {
        return lines.join("\n");
    }
    let first = lines[0].trim_end();
    match first.find('{') {
        Some(i) => format!("{} {{ … }}", first[..i].trim_end()),
        None => format!("{first} …"),
    }
}

/// The type of the smallest expression of `file` ending at `end`: what is
/// written before a `.` there.
pub(crate) fn type_ending_at(unit: &tir::Unit, file: SourceId, end: usize) -> Option<Ty> {
    struct Ending {
        file: SourceId,
        end: usize,
        best: Option<(u32, Ty)>,
    }
    impl Visit for Ending {
        fn expr(&mut self, e: &tir::Expr) {
            if e.span.source == self.file
                && e.span.end as usize == self.end
                && self.best.is_none_or(|(len, _)| e.span.len() < len)
            {
                self.best = Some((e.span.len(), e.ty));
            }
            walk_expr(self, e);
        }
    }
    let mut v = Ending { file, end, best: None };
    for f in unit.fns.iter().filter(|f| f.body.span.source == file) {
        v.block(&f.body);
    }
    for init in unit.globals.iter().filter_map(|g| g.init.as_ref()) {
        v.expr(init);
    }
    v.best.map(|(_, ty)| ty).filter(|&t| t != Ty::Never)
}

/// The name an item declares.
pub(crate) fn item_name(kind: &ItemKind) -> Option<Spanned<Symbol>> {
    match kind {
        ItemKind::Fn(f) => Some(fn_name(&f.name)),
        ItemKind::Struct { name, .. }
        | ItemKind::Enum { name, .. }
        | ItemKind::Let { name, .. }
        | ItemKind::TypeAlias { name, .. }
        | ItemKind::Cover { name, .. }
        | ItemKind::Gauge { name, .. }
        | ItemKind::Base { name, .. }
        | ItemKind::Locale { name, .. }
        | ItemKind::Chain { name, .. }
        | ItemKind::Qmap { name, .. } => Some(*name),
        ItemKind::Impl { .. } | ItemKind::Import { .. } | ItemKind::Assert { .. } => None,
    }
}

fn fn_name(name: &FnName) -> Spanned<Symbol> {
    match name {
        FnName::Plain(s) | FnName::Operator(s) | FnName::External { name: s, .. } => *s,
    }
}

/// The comment lines directly above the line holding `offset`, over any
/// annotations between them, without their `//`.
fn comments_above(text: &str, offset: usize) -> String {
    let start = text[..offset.min(text.len())].rfind('\n').map_or(0, |i| i + 1);
    let mut docs = Vec::new();
    for line in text[..start].lines().rev() {
        let line = line.trim();
        if let Some(comment) = line.strip_prefix("//") {
            docs.push(comment.strip_prefix(' ').unwrap_or(comment));
        } else if line.starts_with('[') && docs.is_empty() {
            continue;
        } else {
            break;
        }
    }
    docs.reverse();
    docs.join("\n")
}

/// A file's opening comment.
fn leading_comment(text: &str) -> String {
    let docs: Vec<&str> = text
        .lines()
        .map(str::trim)
        .take_while(|l| l.starts_with("//"))
        .map(|l| {
            let c = l.trim_start_matches('/');
            c.strip_prefix(' ').unwrap_or(c)
        })
        .collect();
    docs.join("\n")
}

/// Resolves a target to a span of the program.
fn resolve(checked: &Checked, target: &Target) -> Option<Span> {
    match target {
        Target::At(span) => Some(*span),
        Target::Item { unit, name } => item_of_unit(checked, unit, name)?.0,
    }
}

/// The hover for the cursor at `offset` of the file `id`.
pub fn hover(checked: &Checked, id: SourceId, offset: usize, enc: Encoding) -> Option<Hover> {
    let found = find(checked, id, offset)?;
    let mut value = format!("```topiq\n{}\n```", found.shown);
    // what the comment above its declaration says
    let docs = match (&found.note, &found.target) {
        (Some(note), _) => note.clone(),
        (None, Some(target)) if found.documented => resolve(checked, target)
            .filter(|s| !s.source.is_synthetic())
            .map(|s| comments_above(checked.text(s.source), s.start as usize))
            .unwrap_or_default(),
        _ => String::new(),
    };
    if !docs.is_empty() {
        value.push_str("\n\n");
        value.push_str(&docs);
    }
    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value,
        }),
        range: Some(range(checked.text(id), found.at, enc)),
    })
}

/// Where what the cursor at `offset` of the file `id` names is declared.
pub fn definition(checked: &Checked, id: SourceId, offset: usize, enc: Encoding) -> Option<Location> {
    let found = find(checked, id, offset)?;
    let span = resolve(checked, found.target.as_ref()?)?;
    let (path, range) = checked.location(span, enc)?;
    Some(Location::new(crate::url(&path)?, range))
}

/// The outline of the file `id`: its items, with a structure's fields, an
/// enumeration's variants and an `impl` block's methods beneath them.
pub fn symbols(checked: &Checked, id: SourceId, enc: Encoding) -> Vec<DocumentSymbol> {
    let Some(unit) = checked.outcome(id).and_then(|o| o.unit.as_ref()) else {
        return Vec::new();
    };
    let text = checked.text(id);
    let interner = checked.loaded.session.interner();
    let name = |s: Symbol| interner.resolve(s).to_owned();
    let range_of = |s: Span| range(text, s, enc);
    let symbol = |name: String, detail: Option<String>, kind: SymbolKind, whole: Span, at: Span, children: Vec<DocumentSymbol>| {
        // the selection must lie within the whole
        let whole = if whole.start <= at.start && at.end <= whole.end { whole } else { whole.join(at) };
        #[allow(deprecated)]
        DocumentSymbol {
            name,
            detail,
            kind,
            tags: None,
            deprecated: None,
            range: range_of(whole),
            selection_range: range_of(at),
            children: (!children.is_empty()).then_some(children),
        }
    };
    let method = |f: &ast::FnItem| {
        let at = fn_name(&f.name);
        let shown = match &f.name {
            FnName::Operator(_) => format!("${}", name(at.node)),
            _ => name(at.node),
        };
        symbol(shown, None, SymbolKind::METHOD, at.span, at.span, Vec::new())
    };

    let mut out = Vec::new();
    for item in &unit.items {
        let whole = item.span;
        let s = match &item.node.kind {
            ItemKind::Fn(f) => {
                let at = fn_name(&f.name);
                let (shown, kind) = match &f.name {
                    FnName::Plain(_) => (name(at.node), SymbolKind::FUNCTION),
                    FnName::Operator(_) => (format!("${}", name(at.node)), SymbolKind::METHOD),
                    FnName::External { ty, .. } => (format!("{}.{}", path_text(interner, ty), name(at.node)), SymbolKind::METHOD),
                };
                symbol(shown, None, kind, whole, at.span, Vec::new())
            }
            ItemKind::Struct { name: n, fields, .. } => {
                let children = fields
                    .iter()
                    .map(|f| symbol(name(f.name.node), None, SymbolKind::FIELD, f.name.span.join(f.ty.span), f.name.span, Vec::new()))
                    .collect();
                symbol(name(n.node), None, SymbolKind::STRUCT, whole, n.span, children)
            }
            ItemKind::Enum { name: n, variants, .. } => {
                let children = variants
                    .iter()
                    .map(|v| symbol(name(v.name.node), None, SymbolKind::ENUM_MEMBER, v.name.span, v.name.span, Vec::new()))
                    .collect();
                symbol(name(n.node), None, SymbolKind::ENUM, whole, n.span, children)
            }
            ItemKind::Impl { path, items, .. } => {
                let at = path.segments.first().map_or(whole, |s| s.span);
                let children = items.iter().map(method).collect();
                symbol(format!("impl {}", path_text(interner, path)), None, SymbolKind::CLASS, whole, at, children)
            }
            ItemKind::Let { name: n, ty, .. } => {
                let kind = if matches!(ty.node, ast::Type::Const(_)) { SymbolKind::CONSTANT } else { SymbolKind::VARIABLE };
                let detail = text.get(ty.span.range()).map(str::to_owned);
                symbol(name(n.node), detail, kind, whole, n.span, Vec::new())
            }
            ItemKind::TypeAlias { name: n, .. } => symbol(name(n.node), Some("type".to_owned()), SymbolKind::TYPE_PARAMETER, whole, n.span, Vec::new()),
            ItemKind::Import { path, .. } => {
                let at = path.segments.first().map_or(whole, |s| s.span);
                symbol(path_text(interner, path), Some("import".to_owned()), SymbolKind::MODULE, whole, at, Vec::new())
            }
            ItemKind::Cover { name: n, .. } => symbol(name(n.node), Some("cover".to_owned()), SymbolKind::OBJECT, whole, n.span, Vec::new()),
            ItemKind::Gauge { name: n, .. } => symbol(name(n.node), Some("gauge".to_owned()), SymbolKind::OBJECT, whole, n.span, Vec::new()),
            ItemKind::Base { name: n, .. } => symbol(name(n.node), Some("base".to_owned()), SymbolKind::CLASS, whole, n.span, Vec::new()),
            ItemKind::Locale { name: n, members, .. } => {
                let children = members
                    .iter()
                    .map(|m| symbol(name(m.name.node), None, SymbolKind::FUNCTION, m.name.span, m.name.span, Vec::new()))
                    .collect();
                symbol(name(n.node), Some("locale".to_owned()), SymbolKind::NAMESPACE, whole, n.span, children)
            }
            ItemKind::Chain { name: n, .. } => symbol(name(n.node), Some("chain".to_owned()), SymbolKind::OBJECT, whole, n.span, Vec::new()),
            ItemKind::Qmap { name: n, .. } => symbol(name(n.node), Some("qmap".to_owned()), SymbolKind::OBJECT, whole, n.span, Vec::new()),
            ItemKind::Assert { .. } => continue,
        };
        out.push(s);
    }
    out
}

fn path_text(interner: &Interner, path: &ast::Path) -> String {
    let parts: Vec<&str> = path.segments.iter().map(|s| interner.resolve(s.node)).collect();
    parts.join("::")
}
