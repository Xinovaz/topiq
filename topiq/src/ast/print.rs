//! Rendering a tree back to readable text.
//!
//! `tqc emit --stage=ast` writes a *dump*, not a formatter: it shows the tree's shape, with symbols
//! resolved through the [`crate::intern::Interner`] so that names are
//! names rather than numbers. Reading it should make clear which production
//! matched, which is what a grammar dump is for.

use std::fmt::Write;

use crate::intern::{Interner, Symbol};
use crate::span::Spanned;

use super::annot::AnnotationGroup;
use super::expr::{MacroArg, Param, PrepArg};
use super::item::{FnName, ItemKind, VariantPayload};
use super::ty::{Path, TArg, Type};
use super::{Block, Expr, Item, Pattern, Stmt, Unit};

/// Renders a unit as an indented tree.
pub fn unit(u: &Unit, interner: &Interner) -> String {
    let mut p = Printer { out: String::new(), depth: 0, interner };
    match u.kind {
        Some(k) => p.line(&format!("unit {}", k.name())),
        None => p.line("unit <no directive>"),
    }
    p.indented(|p| {
        p.doc(u.doc.as_ref());
        for item in &u.items {
            p.item(&item.node);
        }
    });
    p.out
}

/// Renders one expression, for a focused test assertion.
pub fn expr(e: &Expr, interner: &Interner) -> String {
    let mut p = Printer { out: String::new(), depth: 0, interner };
    p.expr(e);
    p.out
}

struct Printer<'a> {
    out: String,
    depth: usize,
    interner: &'a Interner,
}

impl<'a> Printer<'a> {
    fn line(&mut self, text: &str) {
        for _ in 0..self.depth {
            self.out.push_str("  ");
        }
        let _ = writeln!(self.out, "{text}");
    }

    fn doc(&mut self, doc: Option<&super::Doc>) {
        if let Some(d) = doc {
            self.line(&format!("doc {:?}", d.text));
        }
    }

    fn indented(&mut self, f: impl FnOnce(&mut Self)) {
        self.depth += 1;
        f(self);
        self.depth -= 1;
    }

    fn sym(&self, s: Symbol) -> &'a str {
        self.interner.try_resolve(s).unwrap_or("<unknown>")
    }

    fn decl(&self, d: crate::quon::ast::DeclName) -> String {
        match d.unit {
            Some(u) => format!("{}::{}", self.sym(u), self.sym(d.name)),
            None => self.sym(d.name).to_owned(),
        }
    }

    fn path(&self, p: &Path) -> String {
        p.segments
            .iter()
            .map(|s| self.sym(s.node))
            .collect::<Vec<_>>()
            .join("::")
    }

    ///////////
    // ITEMS //
    ///////////

    fn item(&mut self, it: &Item) {
        self.doc(it.doc.as_ref());
        for g in &it.annotations {
            self.annotation_group(&g.node);
        }
        let linkage = if it.is_exported() { "" } else { "static " };
        match &it.kind {
            ItemKind::Fn(f) => {
                let name = match &f.name {
                    FnName::Plain(s) => self.sym(s.node).to_owned(),
                    FnName::Operator(s) => format!("${}", self.sym(s.node)),
                    FnName::External { ty, name } => {
                        format!("{}.{}", self.path(ty), self.sym(name.node))
                    }
                };
                self.line(&format!("{linkage}fn {name}"));
                self.indented(|p| {
                    for g in &f.generics {
                        p.line(&format!("generic {}", p.sym(g.name.node)));
                    }
                    for param in &f.params {
                        p.param(param);
                    }
                    if let Some(r) = &f.ret {
                        p.line(&format!("-> {}", p.ty(&r.node)));
                    }
                    if f.where_clause.is_some() {
                        p.line("where <constant expression>");
                    }
                    p.block(&f.body);
                });
            }
            ItemKind::Struct { name, fields, .. } => {
                self.line(&format!("{linkage}struct {}", self.sym(name.node)));
                self.indented(|p| {
                    for f in fields {
                        p.doc(f.doc.as_ref());
                        let t = p.ty(&f.ty.node);
                        p.line(&format!("field {}: {t}", p.sym(f.name.node)));
                    }
                });
            }
            ItemKind::Enum { name, variants, .. } => {
                self.line(&format!("{linkage}enum {}", self.sym(name.node)));
                self.indented(|p| {
                    for v in variants {
                        p.doc(v.doc.as_ref());
                        let shape = match &v.payload {
                            None => "",
                            Some(VariantPayload::Tuple(_)) => " (tuple)",
                            Some(VariantPayload::Struct(_)) => " (struct)",
                        };
                        p.line(&format!("variant {}{shape}", p.sym(v.name.node)));
                    }
                });
            }
            ItemKind::Impl { path, items, .. } => {
                self.line(&format!("{linkage}impl {}", self.path(path)));
                self.indented(|p| {
                    for f in items {
                        p.doc(f.doc.as_ref());
                        let name = match &f.name {
                            FnName::Plain(s) | FnName::Operator(s) => p.sym(s.node).to_owned(),
                            FnName::External { name, .. } => p.sym(name.node).to_owned(),
                        };
                        p.line(&format!("fn {name}"));
                    }
                });
            }
            ItemKind::Let { name, ty, init } => {
                let t = self.ty(&ty.node);
                self.node(&format!("{linkage}let {}: {t}", self.sym(name.node)), init);
            }
            ItemKind::TypeAlias { name, ty, .. } => {
                let t = self.ty(&ty.node);
                self.line(&format!("{linkage}type {} = {t}", self.sym(name.node)));
            }
            ItemKind::Import { path, alias, kind } => {
                let a = match alias {
                    Some(a) => format!(" as {}", self.sym(a.node)),
                    None => String::new(),
                };
                let k = match kind {
                    Some(k) => format!("({})", self.sym(k.node)),
                    None => String::new(),
                };
                self.line(&format!("{linkage}import{k} {}{a}", self.path(path)));
            }
            ItemKind::Cover { name, .. } => {
                self.line(&format!("{linkage}cover {}", self.sym(name.node)));
            }
            ItemKind::Gauge { name, .. } => {
                self.line(&format!("{linkage}gauge {}", self.sym(name.node)));
            }
            ItemKind::Base {
                name, cover, gauge, ..
            } => {
                self.line(&format!(
                    "{linkage}base {} over {} with {}",
                    self.sym(name.node),
                    self.decl(cover.node),
                    self.decl(gauge.node)
                ));
            }
            ItemKind::Locale { name, members, .. } => {
                self.line(&format!("{linkage}locale {}", self.sym(name.node)));
                self.indented(|p| {
                    for m in members {
                        p.doc(m.doc.as_ref());
                        p.line(&format!("member {}", p.sym(m.name.node)));
                    }
                });
            }
            ItemKind::Chain { name, stages } => {
                self.line(&format!(
                    "{linkage}chain {} ({} stages)",
                    self.sym(name.node),
                    stages.len()
                ));
                self.indented(|p| {
                    for s in stages {
                        p.line(&format!(
                            "stage {} -> {}",
                            p.sym(s.from.node),
                            p.sym(s.to.node)
                        ));
                    }
                });
            }
            ItemKind::Qmap { name, entries, .. } => {
                self.line(&format!(
                    "{linkage}qmap {} ({} entries)",
                    self.sym(name.node),
                    entries.len()
                ));
            }
            ItemKind::Assert { call } => self.node("assertion", [call]),
        }
    }

    fn param(&mut self, p: &Param) {
        let t = self.ty(&p.ty.node);
        self.line(&format!("param {}: {t}", self.sym(p.name.node)));
    }

    fn annotation_group(&mut self, g: &AnnotationGroup) {
        let names: Vec<&str> = g.annotations.iter().map(|a| self.sym(a.node.name.node)).collect();
        self.line(&format!("[{}]", names.join("; ")));
    }

    ///////////
    // TYPES //
    ///////////

    fn ty(&self, t: &Type) -> String {
        match t {
            Type::Const(i) => format!("const {}", self.ty(&i.node)),
            Type::Constexpr(i) => format!("constexpr {}", self.ty(&i.node)),
            Type::Ref { target } => format!("*{}", self.ty(&target.node)),
            Type::Array { elem, len } => match len {
                Some(_) => format!("[{}; _]", self.ty(&elem.node)),
                None => format!("[{}]", self.ty(&elem.node)),
            },
            Type::Tuple(ts) => format!(
                "({})",
                ts.iter().map(|t| self.ty(&t.node)).collect::<Vec<_>>().join(", ")
            ),
            Type::Fn { params, ret } => {
                let ps = params
                    .iter()
                    .map(|t| self.ty(&t.node))
                    .collect::<Vec<_>>()
                    .join(", ");
                match ret {
                    Some(r) => format!("fn({ps}) -> {}", self.ty(&r.node)),
                    None => format!("fn({ps})"),
                }
            }
            Type::Closure(i) => format!("closure<{}>", self.ty(&i.node)),
            Type::Circuit(i) => format!("circuit<{}>", self.ty(&i.node)),
            Type::Qmap { key, value } => {
                format!("qmap<{}, {}>", self.ty(&key.node), self.ty(&value.node))
            }
            Type::Dyn => "dyn".to_owned(),
            Type::Macro { name, args } => {
                use crate::ast::expr::MacroArg;
                let args: Vec<String> = args
                    .iter()
                    .map(|a| match &a.node {
                        MacroArg::Type(t) => self.ty(&t.node),
                        MacroArg::Str(s) => format!("{:?}", self.sym(s.node)),
                        MacroArg::Expr(_) | MacroArg::Quon(_) => "…".to_owned(),
                    })
                    .collect();
                format!("@{}({})", self.sym(name.node), args.join(", "))
            }
            Type::Void => "void".to_owned(),
            Type::Restriction {
                restricted,
                container,
            } => format!(
                "{} of {}",
                self.ty(&restricted.node),
                self.ty(&container.node)
            ),
            Type::Optional(i) => format!("{}?", self.ty(&i.node)),
            Type::Path { path, args } => {
                if args.is_empty() {
                    self.path(path)
                } else {
                    let a = args
                        .iter()
                        .map(|x| match &x.node {
                            TArg::Type(t) => self.ty(&t.node),
                            TArg::Const(_) => "_".to_owned(),
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{}<{a}>", self.path(path))
                }
            }
        }
    }

    ////////////////////////////////
    // STATEMENTS AND EXPRESSIONS //
    ////////////////////////////////

    fn block(&mut self, b: &Block) {
        self.line("block");
        self.indented(|p| {
            for s in &b.stmts {
                p.stmt(&s.node);
            }
            if let Some(v) = &b.value {
                p.line("value");
                p.indented(|p| p.expr(&v.node));
            }
        });
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let(l) => {
                let storage = match l.storage {
                    Some(st) => format!("{} ", st.node.text()),
                    None => String::new(),
                };
                let bound = match &l.binding.node {
                    Pattern::Binding(s) => self.sym(s.node).to_owned(),
                    other => self.pattern_text(other),
                };
                self.line(&format!("{storage}let {bound}"));
                self.indented(|p| {
                    if let Some(t) = &l.ty {
                        let t = p.ty(&t.node);
                        p.line(&format!(": {t}"));
                    }
                    if let Some(e) = &l.init {
                        p.expr(&e.node);
                    }
                });
            }
            Stmt::Item(i) => self.item(i),
            Stmt::Flow(f) => self.line(&format!("{f:?}").split('(').next().unwrap_or("flow").to_lowercase()),
            Stmt::Forget(e) => self.node("forget", [e]),
            Stmt::BlockExpr(e) => self.node("block-statement", [e]),
            Stmt::Expr(e) => self.expr(&e.node),
            Stmt::Empty => self.line("empty"),
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Int { raw, .. } => self.line(&format!("int {}", self.sym(*raw))),
            Expr::Float { raw, .. } => self.line(&format!("float {}", self.sym(*raw))),
            Expr::Char(c) => self.line(&format!("char {c:?}")),
            Expr::Bool(b) => self.line(&format!("bool {b}")),
            Expr::Str { value, .. } => self.line(&format!("string {:?}", self.sym(*value))),
            Expr::Path { path, args } => {
                let t = if args.is_empty() { "" } else { "::<…>" };
                self.line(&format!("path {}{t}", self.path(path)));
            }
            Expr::Paren(i) => self.expr(&i.node),
            Expr::Tuple(es) => self.node("tuple", es),
            Expr::Array(es) => self.node("array", es),
            Expr::ArrayRepeat { elem, len } => self.node("array-repeat", [&**elem, &**len]),
            Expr::StructLit { path, fields } => {
                self.line(&format!("struct-literal {}", self.path(path)));
                self.indented(|p| {
                    for f in fields {
                        p.line(&format!("field {}", p.sym(f.name.node)));
                        p.indented(|p| p.expr(&f.value.node));
                    }
                });
            }
            Expr::Block(b) => self.block(b),
            Expr::If { cond, then, els } => {
                self.line("if");
                self.indented(|p| {
                    p.expr(&cond.node);
                    p.block(then);
                    if let Some(e) = els {
                        p.line("else");
                        p.indented(|p| p.expr(&e.node));
                    }
                });
            }
            Expr::Match {
                measuring,
                scrutinee,
                arms,
            } => {
                self.line(if *measuring { "match measure" } else { "match" });
                self.indented(|p| {
                    p.expr(&scrutinee.node);
                    for a in arms {
                        p.pattern(&a.pattern.node);
                        p.indented(|p| p.expr(&a.body.node));
                    }
                });
            }
            Expr::While { cond, body } => {
                self.line("while");
                self.indented(|p| {
                    p.expr(&cond.node);
                    p.block(body);
                });
            }
            Expr::Loop { body } => {
                self.line("loop");
                self.indented(|p| p.block(body));
            }
            Expr::For {
                pattern,
                iter,
                body,
            } => {
                self.line("for");
                self.indented(|p| {
                    p.pattern(&pattern.node);
                    p.expr(&iter.node);
                    p.block(body);
                });
            }
            Expr::Closure {
                params,
                captures,
                body,
                ..
            } => {
                self.line(&format!(
                    "closure ({} params, {} captures)",
                    params.len(),
                    captures.len()
                ));
                self.indented(|p| p.block(body));
            }
            Expr::Measure(i) => self.node("measure", [&**i]),
            Expr::Prep(a) => {
                self.line("prep");
                self.indented(|p| match a {
                    PrepArg::State(_) => p.line("<QUON state>"),
                    PrepArg::Expr(e) => p.expr(&e.node),
                });
            }
            Expr::Query { map, index } => self.node("query", [&**map, &**index]),
            Expr::Lift(i) => self.node("lift", [&**i]),
            Expr::Replay(i) => self.node("replay", [&**i]),
            Expr::Macro { name, args } => {
                self.line(&format!("@{}", self.sym(name.node)));
                self.indented(|p| {
                    for a in args {
                        match &a.node {
                            MacroArg::Type(t) => {
                                let t = p.ty(&t.node);
                                p.line(&format!("type {t}"));
                            }
                            MacroArg::Str(s) => {
                                p.line(&format!("string {:?}", p.sym(s.node)));
                            }
                            MacroArg::Expr(e) => p.expr(&e.node),
                            MacroArg::Quon(_) => p.line("<QUON form>"),
                        }
                    }
                });
            }
            Expr::Unary { op, operand } => self.node(&format!("unary {op:?}"), [&**operand]),
            Expr::Step { op, post, place } => {
                let label = if *post { format!("step x{}", op.text()) } else { format!("step {}x", op.text()) };
                self.node(&label, [&**place]);
            }
            Expr::Binary { op, lhs, rhs } => self.node(&format!("binary {}", op.text()), [&**lhs, &**rhs]),
            Expr::Assign { op, place, value } => {
                let o = match op {
                    Some(o) => format!("{}=", o.text()),
                    None => "=".to_owned(),
                };
                // written place-first, but the value is evaluated
                // first; the dump shows the syntax, not the order
                self.node(&format!("assign {o}"), [&**place, &**value]);
            }
            Expr::Range {
                start,
                end,
                inclusive,
            } => {
                let label = if *inclusive { "range ..=" } else { "range .." };
                self.node(label, start.iter().chain(end).map(|b| &**b));
            }
            Expr::Call { callee, args } => self.node("call", std::iter::once(&**callee).chain(args)),
            Expr::Index { receiver, index } => self.node("index", [&**receiver, &**index]),
            Expr::Field { receiver, name } => self.node(&format!("field .{}", self.sym(name.node)), [&**receiver]),
            Expr::MethodCall {
                receiver,
                name,
                args,
                ..
            } => self.node(&format!("method .{}", self.sym(name.node)), std::iter::once(&**receiver).chain(args)),
            Expr::Try(i) => self.node("try ?", [&**i]),
            Expr::Cast { expr, ty } => self.node(&format!("as {}", self.ty(&ty.node)), [&**expr]),
        }
    }

    /// A line `label`, and beneath it each of `children`.
    fn node<'e>(&mut self, label: &str, children: impl IntoIterator<Item = &'e Spanned<Expr>>) {
        self.line(label);
        self.indented(|p| {
            for e in children {
                p.expr(&e.node);
            }
        });
    }

    fn pattern(&mut self, p: &Pattern) {
        let text = match p {
            Pattern::Binding(s) => format!("bind {}", self.sym(s.node)),
            other => self.pattern_text(other),
        };
        self.line(&format!("pattern {text}"));
    }

    /// A pattern in one line.
    fn pattern_text(&self, p: &Pattern) -> String {
        match p {
            Pattern::Wildcard => "_".to_owned(),
            Pattern::Literal(_) => "<literal>".to_owned(),
            Pattern::Binding(s) => self.sym(s.node).to_owned(),
            Pattern::Path(path) => self.path(path),
            Pattern::TupleStruct { path, elements } => {
                format!("{}({})", self.path(path), elements.len())
            }
            Pattern::Struct { path, fields } => {
                format!("{} {{{}}}", self.path(path), fields.len())
            }
            Pattern::Tuple(ps) => {
                let parts: Vec<String> = ps.iter().map(|x| self.pattern_text(&x.node)).collect();
                format!("({})", parts.join(", "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{Options, Session, compile};

    fn dump(src: &str) -> String {
        let mut session = Session::new();
        let id = session.add("demo.tq", src);
        let out = compile(&mut session, id, Options::default());
        assert!(
            !out.has_errors(),
            "{src:?} failed: {:?}",
            out.diagnostics.iter().map(|d| d.to_string()).collect::<Vec<_>>()
        );
        unit(out.unit.as_ref().unwrap(), session.interner())
    }

    #[test]
    fn a_unit_names_its_kind() {
        assert!(dump("#unit classical\n").starts_with("unit classical"));
        assert!(dump("#unit quantum\n").starts_with("unit quantum"));
    }

    #[test]
    fn a_function_shows_its_parameters_and_return_type() {
        let d = dump("#unit classical\nfn hypot(a: f64, b: f64) -> f64 { a }\n");
        assert!(d.contains("fn hypot"), "{d}");
        assert!(d.contains("param a: f64"), "{d}");
        assert!(d.contains("param b: f64"), "{d}");
        assert!(d.contains("-> f64"), "{d}");
    }

    #[test]
    fn static_linkage_is_shown() {
        let d = dump("#unit classical\nstatic fn scratch(n: usize) { }\n");
        assert!(d.contains("static fn scratch"), "{d}");
    }

    #[test]
    fn a_structure_shows_its_fields_with_types() {
        let d = dump("#unit classical\nstruct Packet { tag: u8, len: u32, payload: [u8] }\n");
        assert!(d.contains("struct Packet"), "{d}");
        assert!(d.contains("field tag: u8"), "{d}");
        assert!(d.contains("field payload: [u8]"), "{d}");
    }

    #[test]
    fn an_import_shows_its_alias() {
        let d = dump("#unit classical\nimport qpu::gates as g;\n");
        assert!(d.contains("import qpu::gates as g"), "{d}");
    }

    #[test]
    fn operators_are_shown_by_their_spelling() {
        let d = dump("#unit classical\nfn f() { let x = 1 + 2 * 3; }\n");
        assert!(d.contains("binary +"), "{d}");
        assert!(d.contains("binary *"), "{d}");
    }

    #[test]
    fn precedence_is_visible_in_the_shape() {
        // `1 + 2 * 3` nests the multiplication under the addition
        let d = dump("#unit classical\nfn f() { let x = 1 + 2 * 3; }\n");
        let plus = d.find("binary +").unwrap();
        let star = d.find("binary *").unwrap();
        assert!(plus < star, "the addition should be the outer node:\n{d}");
    }

    #[test]
    fn a_block_shows_its_value_separately_from_its_statements() {
        let d = dump("#unit classical\nfn f() -> i32 { let x = 1; x }\n");
        assert!(d.contains("value"), "{d}");
        assert!(d.contains("let x"), "{d}");
    }

    #[test]
    fn annotations_are_shown_above_their_item() {
        let d = dump("#unit quantum\n[entry]\nfn f() { }\n");
        let annot = d.find("[entry]").expect("the annotation should appear");
        let item = d.find("fn f").expect("the item should appear");
        assert!(annot < item, "{d}");
    }

    #[test]
    fn the_dump_resolves_names_rather_than_printing_numbers() {
        let d = dump("#unit classical\nlet LIMIT: const usize = 64;\n");
        assert!(d.contains("LIMIT"), "{d}");
        assert!(!d.contains("Symbol("), "{d}");
    }

    #[test]
    fn an_expression_can_be_dumped_on_its_own() {
        let mut i = Interner::new();
        let s = i.intern("x");
        let e = Expr::Path {
            path: Path::single(Spanned::new(
                s,
                crate::span::Span::new(crate::span::SourceId(0), 0, 1),
            )),
            args: Vec::new(),
        };
        assert_eq!(expr(&e, &i).trim(), "path x");
    }
}
