//! What each page of the site shows, built from the program's units.
//!
//! A unit compiled from source, or compiled earlier with its source carried
//! in its metadata, is documented from its tree: each declaration it offers
//! has a page, with its signature as written and its documentation, and its
//! methods are gathered onto the page of the type they belong to. A unit
//! compiled earlier without its source is documented from its interface
//! alone, which has types but no names for parameters and no documentation.
//!
//! What a unit keeps to itself (a `static` declaration, or a name beginning
//! with `__`, which only the library's internals use) is left out unless
//! private declarations are asked for.

use std::borrow::Cow;
use std::collections::HashMap;

use serde::Serialize;

use crate::ast::{self, Doc, FnItem, FnName, Item, ItemKind, Type};
use crate::driver::graph::Graph;
use crate::driver::program::{Loaded, Member, MemberKind};
use crate::intern::Interner;
use crate::lex::{Keyword, Punct, Token};
use crate::source::SourceFile;
use crate::span::{Span, Spanned};

use super::highlight::{self, Gaps};
use super::markdown::{self, Context, Rendered};
use super::resolve::{Index, ItemEntry, UnitEntry};

/// The whole site.
#[derive(Debug, Default)]
pub struct Site {
    /// The units given.
    pub project: String,
    /// Every unit, those given first, then the program's own, then the
    /// library's.
    pub units: Vec<UnitView>,
    /// The dependency tree.
    pub tree: String,
    /// Every page but the root one.
    pub pages: Vec<Page>,
    /// What search finds: for each entry, its name, its unit, what it is,
    /// its address and its summary.
    pub search: Vec<[String; 5]>,
    /// Links that name nothing documented.
    pub warnings: Vec<String>,
}

/// One unit, as lists of units show it.
#[derive(Clone, Debug, Serialize)]
pub struct UnitView {
    /// Its name.
    pub name: String,
    /// Its page.
    pub href: String,
    /// `classical` or `quantum`.
    pub kind: String,
    /// What else is true of it: `library`, `compiled`, `given`.
    pub notes: Vec<String>,
    /// The first paragraph of its documentation.
    pub summary: String,
}

/// A page and where it is written.
#[derive(Debug)]
pub struct Page {
    /// Its path from the root.
    pub path: String,
    /// What it shows.
    pub content: Content,
    /// The navigation beside it.
    pub sidebar: Sidebar,
}

/// What a page shows.
#[derive(Debug)]
pub enum Content {
    /// A unit.
    Unit(UnitPage),
    /// One declaration.
    Item(ItemPage),
    /// A unit's source.
    Source(SourcePage),
}

/// The navigation beside a unit's pages.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Sidebar {
    /// The unit.
    pub unit: String,
    /// Its page.
    pub href: String,
    /// Its declarations, by kind.
    pub sections: Vec<Section>,
}

/// A unit's page.
#[derive(Debug, Serialize)]
pub struct UnitPage {
    /// The unit.
    pub unit: UnitView,
    /// Its documentation.
    pub doc: String,
    /// The units it imports.
    pub imports: Vec<Link>,
    /// The units that import it.
    pub imported_by: Vec<Link>,
    /// Its declarations, by kind.
    pub sections: Vec<Section>,
    /// Its methods on types of other units.
    pub foreign: Vec<MemberView>,
    /// For a unit known only from its interface, that interface, one
    /// declaration per entry.
    pub interface: Vec<String>,
    /// Its source page, if it has one.
    pub source: Option<String>,
}

/// A name and, when it has a page, the page.
#[derive(Clone, Debug, Serialize)]
pub struct Link {
    /// The name.
    pub name: String,
    /// The page.
    pub href: Option<String>,
}

/// Declarations of one kind.
#[derive(Clone, Debug, Serialize)]
pub struct Section {
    /// The heading.
    pub title: &'static str,
    /// The heading's anchor.
    pub id: &'static str,
    /// The declarations.
    pub entries: Vec<Entry>,
}

/// One declaration in a list.
#[derive(Clone, Debug, Serialize)]
pub struct Entry {
    /// Its name.
    pub name: String,
    /// Its page.
    pub href: String,
    /// The first paragraph of its documentation.
    pub summary: String,
    /// Whether it is deprecated.
    pub deprecated: bool,
}

/// A declaration's page.
#[derive(Debug, Serialize)]
pub struct ItemPage {
    /// Its unit.
    pub unit: String,
    /// What it is, such as `Structure`.
    pub kind: &'static str,
    /// Its name.
    pub name: String,
    /// Its signature, as HTML.
    pub signature: String,
    /// Its documentation.
    pub doc: String,
    /// The message of `[deprecated]`.
    pub deprecated: Option<String>,
    /// Its line on the unit's source page.
    pub source: Option<String>,
    /// Its fields, variants, members, methods and operators.
    pub groups: Vec<Group>,
}

/// Parts of a declaration of one kind.
#[derive(Debug, Serialize)]
pub struct Group {
    /// The heading.
    pub title: &'static str,
    /// The heading's anchor.
    pub id: &'static str,
    /// The parts.
    pub members: Vec<MemberView>,
}

/// A field, variant, locale member or method.
#[derive(Clone, Debug, Serialize)]
pub struct MemberView {
    /// Its anchor.
    pub id: String,
    /// Its name.
    pub name: String,
    /// Its signature, as HTML.
    pub signature: String,
    /// Its documentation.
    pub doc: String,
    /// Its line on the unit's source page.
    pub source: Option<String>,
}

/// A unit's source.
#[derive(Debug, Serialize)]
pub struct SourcePage {
    /// The unit.
    pub unit: String,
    /// The file it was read from.
    pub path: String,
    /// The line numbers.
    pub gutter: String,
    /// The text.
    pub code: String,
}

//////////////////
// DECLARATIONS //
//////////////////

/// What a declaration with a page of its own is.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Category {
    Struct,
    Enum,
    TypeAlias,
    Base,
    Cover,
    Gauge,
    Locale,
    Chain,
    Qmap,
    Function,
    Constant,
    Object,
}

impl Category {
    /// Every category, in the order a unit's page lists them.
    const ALL: [Category; 12] = [
        Category::Struct,
        Category::Enum,
        Category::TypeAlias,
        Category::Base,
        Category::Cover,
        Category::Gauge,
        Category::Locale,
        Category::Chain,
        Category::Qmap,
        Category::Function,
        Category::Constant,
        Category::Object,
    ];

    /// What begins the name of a page of this kind.
    fn prefix(self) -> &'static str {
        match self {
            Category::Struct => "struct",
            Category::Enum => "enum",
            Category::TypeAlias => "type",
            Category::Base => "base",
            Category::Cover => "cover",
            Category::Gauge => "gauge",
            Category::Locale => "locale",
            Category::Chain => "chain",
            Category::Qmap => "qmap",
            Category::Function => "fn",
            Category::Constant => "const",
            Category::Object => "let",
        }
    }

    /// One declaration of this kind.
    fn label(self) -> &'static str {
        match self {
            Category::Struct => "Structure",
            Category::Enum => "Enumeration",
            Category::TypeAlias => "Type alias",
            Category::Base => "Base type",
            Category::Cover => "Cover",
            Category::Gauge => "Gauge",
            Category::Locale => "Locale",
            Category::Chain => "Chain",
            Category::Qmap => "Map locale",
            Category::Function => "Function",
            Category::Constant => "Constant",
            Category::Object => "Object",
        }
    }

    /// Several declarations of this kind, and their section's anchor.
    fn heading(self) -> (&'static str, &'static str) {
        match self {
            Category::Struct => ("Structures", "structs"),
            Category::Enum => ("Enumerations", "enums"),
            Category::TypeAlias => ("Type aliases", "types"),
            Category::Base => ("Base types", "bases"),
            Category::Cover => ("Covers", "covers"),
            Category::Gauge => ("Gauges", "gauges"),
            Category::Locale => ("Locales", "locales"),
            Category::Chain => ("Chains", "chains"),
            Category::Qmap => ("Map locales", "qmaps"),
            Category::Function => ("Functions", "functions"),
            Category::Constant => ("Constants", "constants"),
            Category::Object => ("Objects", "objects"),
        }
    }

    /// Whether a declaration of this kind is a type, which methods belong
    /// to.
    fn is_type(self) -> bool {
        matches!(self, Category::Struct | Category::Enum | Category::TypeAlias | Category::Base)
    }
}

/// A declaration with a page of its own.
struct Decl<'a> {
    item: &'a Spanned<Item>,
    category: Category,
    name: String,
    generics: Vec<String>,
    methods: Vec<Method<'a>>,
}

impl Decl<'_> {
    /// Its page, relative to its unit's directory.
    fn page(&self) -> String {
        format!("{}.{}.html", self.category.prefix(), self.name)
    }
}

/// A method: a function in an `impl` block, one defined as `T.name`, or an
/// operator method.
struct Method<'a> {
    /// The path of the type it belongs to.
    owner: Vec<String>,
    f: &'a FnItem,
    /// Where its signature starts: its first annotation, or `fn`.
    start: u32,
    doc: Option<&'a Doc>,
    generics: Vec<String>,
}

impl Method<'_> {
    /// Its name as written, `$` and all, and its anchor.
    fn name_and_anchor(&self, interner: &Interner) -> (String, String) {
        let name = interner.resolve(self.f.name.name());
        if self.f.name.is_operator() {
            (format!("${name}"), format!("op.{name}"))
        } else {
            (name.to_owned(), format!("method.{name}"))
        }
    }
}

/// A unit's declarations and the methods that belong to no type of its own.
#[derive(Default)]
struct Declared<'a> {
    decls: Vec<Decl<'a>>,
    foreign: Vec<Method<'a>>,
}

/// What a unit offers, gathered from its tree.
fn declarations<'a>(unit: &'a ast::Unit, interner: &Interner, private: bool) -> Declared<'a> {
    let shown = |item: &Item, name: &str| private || (item.is_exported() && !name.starts_with("__"));
    let names = |gs: &[ast::GenericParam]| gs.iter().map(|g| interner.resolve(g.name.node).to_owned()).collect::<Vec<_>>();
    let mut decls: Vec<Decl<'a>> = Vec::new();
    let mut methods: Vec<Method<'a>> = Vec::new();
    for item in &unit.items {
        let decl = |category, name: &Spanned<crate::intern::Symbol>, generics: Vec<String>| Decl {
            item,
            category,
            name: interner.resolve(name.node).to_owned(),
            generics,
            methods: Vec::new(),
        };
        let found = match &item.node.kind {
            ItemKind::Fn(f) => {
                let name = interner.resolve(f.name.name());
                let owner = match &f.name {
                    FnName::Plain(n) => {
                        if shown(&item.node, name) {
                            decls.push(decl(Category::Function, n, names(&f.generics)));
                        }
                        continue;
                    }
                    FnName::External { ty, .. } => Some(segments(ty, interner)),
                    // an operator at unit scope belongs to the type of `self`
                    FnName::Operator(_) => f.params.first().and_then(|p| type_path(&p.ty.node)).map(|p| segments(p, interner)),
                };
                if let Some(owner) = owner
                    && shown(&item.node, name)
                {
                    methods.push(Method {
                        owner,
                        f,
                        start: item.span.start,
                        doc: item.node.doc.as_ref(),
                        generics: names(&f.generics),
                    });
                }
                continue;
            }
            ItemKind::Impl { path, generics, items } => {
                for f in items {
                    let name = interner.resolve(f.name.name());
                    if !shown(&item.node, name) {
                        continue;
                    }
                    let mut all = names(generics);
                    all.extend(names(&f.generics));
                    methods.push(Method {
                        owner: segments(path, interner),
                        f,
                        start: f.signature.start,
                        doc: f.doc.as_ref(),
                        generics: all,
                    });
                }
                continue;
            }
            ItemKind::Struct { name, generics, .. } => decl(Category::Struct, name, names(generics)),
            ItemKind::Enum { name, generics, .. } => decl(Category::Enum, name, names(generics)),
            ItemKind::TypeAlias { name, generics, .. } => decl(Category::TypeAlias, name, names(generics)),
            ItemKind::Let { name, ty, .. } => {
                decl(if ty.node.is_const() { Category::Constant } else { Category::Object }, name, Vec::new())
            }
            ItemKind::Cover { name, .. } => decl(Category::Cover, name, Vec::new()),
            ItemKind::Gauge { name, .. } => decl(Category::Gauge, name, Vec::new()),
            ItemKind::Base { name, .. } => decl(Category::Base, name, Vec::new()),
            ItemKind::Locale { name, .. } => decl(Category::Locale, name, Vec::new()),
            ItemKind::Chain { name, .. } => decl(Category::Chain, name, Vec::new()),
            ItemKind::Qmap { name, .. } => decl(Category::Qmap, name, Vec::new()),
            ItemKind::Import { .. } | ItemKind::Assert { .. } => continue,
        };
        if shown(&item.node, &found.name) {
            decls.push(found);
        }
    }

    // a method goes on its type's page when the type is this unit's
    let mut foreign = Vec::new();
    for m in methods {
        let home = match m.owner.as_slice() {
            [name] => decls.iter_mut().find(|d| d.category.is_type() && d.name == *name),
            _ => None,
        };
        match home {
            Some(d) => d.methods.push(m),
            None => foreign.push(m),
        }
    }
    Declared { decls, foreign }
}

/// The segments of a path.
fn segments(path: &ast::Path, interner: &Interner) -> Vec<String> {
    path.segments.iter().map(|s| interner.resolve(s.node).to_owned()).collect()
}

/// The path naming a type.
fn type_path(ty: &Type) -> Option<&ast::Path> {
    match ty {
        Type::Path { path, .. } => Some(path),
        Type::Const(t) | Type::Constexpr(t) => type_path(&t.node),
        Type::Ref { target } => type_path(&target.node),
        _ => None,
    }
}

/// Where a declaration's text ends: before the first of `stops`, or the
/// bracket closing around it, outside any bracket of its own.
fn end_of(tokens: &[(Token, Span)], start: u32, stops: &[Punct]) -> u32 {
    let mut depth = 0usize;
    let mut end = start;
    for &(tok, span) in &tokens[tokens.partition_point(|(_, s)| s.start < start)..] {
        match tok {
            Token::Punct(Punct::LParen | Punct::LBracket | Punct::LBracketAnnot | Punct::LBrace) => depth += 1,
            Token::Punct(Punct::RParen | Punct::RBracket | Punct::RBrace) if depth == 0 => return end,
            Token::Punct(Punct::RParen | Punct::RBracket | Punct::RBrace) => depth -= 1,
            Token::Punct(p) if depth == 0 && stops.contains(&p) => return end,
            _ => {}
        }
        end = span.end;
    }
    end
}

/// The message of a `[deprecated]` annotation.
fn deprecation(item: &Item, file: &SourceFile, interner: &Interner) -> Option<String> {
    let a = item
        .annotations
        .iter()
        .flat_map(|g| &g.node.annotations)
        .find(|a| interner.resolve(a.node.name.node) == "deprecated")?;
    let text = a.node.args.first().and_then(|arg| file.slice(arg.span)).unwrap_or("");
    Some(text.trim().trim_matches('"').to_owned())
}

/////////////////
// THE BUILDER //
/////////////////

/// A unit's source: its file and tokens.
struct Text<'a> {
    file: &'a SourceFile,
    tokens: Cow<'a, [(Token, Span)]>,
}

/// One unit, while the site is built.
struct UnitInfo<'a> {
    member: &'a Member,
    tree: Option<&'a ast::Unit>,
    text: Option<Text<'a>>,
    declared: Declared<'a>,
}

/// Builds the site for a loaded program.
pub fn build(loaded: &Loaded, private: bool) -> Site {
    let interner = loaded.session.interner();
    let sources = loaded.session.sources();
    let members = &loaded.members;
    let graph = crate::driver::program::graph(loaded);

    // each unit's tree and text, and what it declares
    let units: Vec<UnitInfo<'_>> = members
        .iter()
        .map(|member| {
            let (tree, text) = match &member.kind {
                MemberKind::Source(out) => {
                    let text = out.source.map(|id| Text {
                        file: sources.file(id),
                        tokens: Cow::Borrowed(out.tokens.as_slice()),
                    });
                    (out.unit.as_ref(), text)
                }
                MemberKind::Compiled { metadata, .. } => {
                    let tree = metadata.interface.ast.as_deref();
                    // the source a compiled unit carries is in the session,
                    // but its tokens are not kept
                    let text = tree.and_then(|t| t.items.first()).map(|i| {
                        let file = sources.file(i.span.source);
                        let tokens = Cow::Owned(crate::lex::lex(file.id(), file.spliced(), interner).tokens);
                        Text { file, tokens }
                    });
                    (tree, text)
                }
            };
            let declared = tree.map(|t| declarations(t, interner, private)).unwrap_or_default();
            UnitInfo { member, tree, text, declared }
        })
        .collect();

    let index = index(&units, &graph, interner);
    let order = display_order(members);
    let mut site = Site {
        project: members.iter().filter(|m| m.given).map(|m| m.name.clone()).collect::<Vec<_>>().join(", "),
        ..Site::default()
    };

    // every unit's view, for lists of units
    let mut views: Vec<Option<UnitView>> = vec![None; members.len()];
    for &u in &order {
        let mut ignored = Vec::new();
        let summary = units[u].tree.and_then(|t| t.doc.as_ref()).map_or_else(String::new, |d| {
            let resolve = |p: &[&str]| index.resolve(u, p, &[]);
            let mut cx = Context { root: "", resolve: &resolve, interner, warnings: &mut ignored, owner: "" };
            markdown::render(&d.text, &mut cx).summary
        });
        let m = units[u].member;
        let mut notes = Vec::new();
        if m.given {
            notes.push("given".to_owned());
        }
        if crate::library::source(&m.name).is_some() {
            notes.push("library".to_owned());
        }
        if matches!(m.kind, MemberKind::Compiled { .. }) {
            notes.push("compiled".to_owned());
        }
        views[u] = Some(UnitView {
            name: m.name.clone(),
            href: index.unit_page(u),
            kind: m.unit_kind().map_or("", |k| k.name()).to_owned(),
            notes,
            summary,
        });
    }
    site.units = order.iter().filter_map(|&u| views[u].clone()).collect();

    // the dependency tree, each unit linked to its page
    let given: Vec<usize> = (0..members.len()).filter(|&i| members[i].given).collect();
    site.tree = graph.render_with(&given, &|i| match index.units.get(i) {
        Some(_) => format!("<a href=\"{}\">{}</a>", index.unit_page(i), highlight::escape(graph.name(i))),
        None => highlight::escape(graph.name(i)),
    });

    for &u in &order {
        let view = views[u].clone().expect("every unit has a view");
        let mut builder = Builder {
            u,
            info: &units[u],
            index: &index,
            interner,
            warnings: &mut site.warnings,
            search: &mut site.search,
        };
        let pages = builder.pages(view, &graph, members.len());
        site.pages.extend(pages);
    }
    site
}

/// The units in the order lists show them: those given, then the
/// program's own by name, then the library's by name.
fn display_order(members: &[Member]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..members.len()).collect();
    order.sort_by_key(|&i| {
        let m = &members[i];
        (!m.given, crate::library::source(&m.name).is_some(), m.name.clone())
    });
    order
}

/// Which page documents each name, for every unit.
fn index(units: &[UnitInfo<'_>], graph: &Graph, interner: &Interner) -> Index {
    let dir = |name: &str| name.chars().map(|c| if c.is_ascii_alphanumeric() || "_.-".contains(c) { c } else { '_' }).collect::<String>();
    let mut index = Index {
        units: units
            .iter()
            .map(|info| UnitEntry {
                name: info.member.name.clone(),
                dir: dir(&info.member.name),
                ..UnitEntry::default()
            })
            .collect(),
    };
    for (u, info) in units.iter().enumerate() {
        // types first, so that a type wins a name a function shares
        let mut decls: Vec<&Decl<'_>> = info.declared.decls.iter().collect();
        decls.sort_by_key(|d| !d.category.is_type());
        for d in decls {
            let mut anchors: HashMap<String, String> = HashMap::new();
            match &d.item.node.kind {
                ItemKind::Struct { fields, .. } => {
                    for f in fields {
                        let name = interner.resolve(f.name.node);
                        anchors.insert(name.to_owned(), format!("field.{name}"));
                    }
                }
                ItemKind::Enum { variants, .. } => {
                    for v in variants {
                        let name = interner.resolve(v.name.node);
                        anchors.insert(name.to_owned(), format!("variant.{name}"));
                    }
                }
                ItemKind::Locale { members, .. } => {
                    for m in members {
                        let name = interner.resolve(m.name.node);
                        anchors.insert(name.to_owned(), format!("member.{name}"));
                    }
                }
                _ => {}
            }
            for m in &d.methods {
                let (name, anchor) = m.name_and_anchor(interner);
                anchors.insert(name, anchor);
            }
            index.units[u].items.entry(d.name.clone()).or_insert(ItemEntry { page: d.page(), anchors });
        }

        // what each import names, among the units this one depends on
        let Some(tree) = info.tree else { continue };
        let deps: Vec<usize> = graph.deps(u).iter().copied().filter(|&j| j < units.len()).collect();
        for item in &tree.items {
            let ItemKind::Import { path, alias, kind } = &item.node.kind else { continue };
            let Some(last) = path.segments.last().map(|s| interner.resolve(s.node)) else { continue };
            let asked = kind.as_ref().map(|k| interner.resolve(k.node));
            let own = info.member.unit_kind().map(|k| k.name());
            let candidates: Vec<usize> =
                deps.iter().copied().filter(|&j| crate::driver::written(&units[j].member.name) == last).collect();
            let of_kind = |k: Option<&str>| {
                k.and_then(|k| candidates.iter().copied().find(|&j| units[j].member.name.ends_with(&format!(".{k}"))))
            };
            let Some(target) = of_kind(asked).or_else(|| of_kind(own)).or_else(|| candidates.first().copied()) else {
                continue;
            };
            let name = alias.as_ref().map_or(last, |a| interner.resolve(a.node));
            index.units[u].imports.insert(name.to_owned(), target);
        }
    }
    index
}

/// Builds one unit's pages.
struct Builder<'a, 'b> {
    u: usize,
    info: &'a UnitInfo<'a>,
    index: &'a Index,
    interner: &'a Interner,
    warnings: &'b mut Vec<String>,
    search: &'b mut Vec<[String; 5]>,
}

impl Builder<'_, '_> {
    /// The unit's directory.
    fn dir(&self) -> &str {
        &self.index.units[self.u].dir
    }

    /// Documentation as HTML, for a page one level below the root.
    fn markdown(&mut self, doc: Option<&Doc>, owner: &str) -> Rendered {
        let Some(doc) = doc else { return Rendered::default() };
        let (index, u) = (self.index, self.u);
        let resolve = |p: &[&str]| index.resolve(u, p, &[]);
        let mut cx = Context {
            root: "../",
            resolve: &resolve,
            interner: self.interner,
            warnings: &mut *self.warnings,
            owner,
        };
        markdown::render(&doc.text, &mut cx)
    }

    /// The source from `from` to `to` as a signature, on the page `here`,
    /// with `generics` in scope.
    fn signature(&self, from: u32, to: u32, generics: &[String], here: &str) -> String {
        let Some(text) = &self.info.text else { return String::new() };
        let generics: Vec<&str> = generics.iter().map(String::as_str).collect();
        let (index, u) = (self.index, self.u);
        highlight::render(text.file.original(), &text.tokens, from, to, Gaps::Signature, &mut |p| {
            index.resolve(u, p, &generics).filter(|h| h != here).map(|h| format!("../{h}"))
        })
    }

    /// The address of the source line holding `offset`.
    fn line(&self, offset: u32) -> Option<String> {
        let text = self.info.text.as_ref()?;
        Some(format!("{}/source.html#L{}", self.dir(), text.file.line_index(offset) + 1))
    }

    /// Every page of the unit.
    fn pages(&mut self, view: UnitView, graph: &Graph, members: usize) -> Vec<Page> {
        let info = self.info;
        let dir = self.dir().to_owned();
        let unit_name = view.name.clone();
        self.search.push([unit_name.clone(), String::new(), "Unit".to_owned(), view.href.clone(), String::new()]);

        // the lists of declarations, which the sidebar shows too
        let decls = &info.declared.decls;
        let mut rendered: Vec<Rendered> = Vec::with_capacity(decls.len());
        for d in decls {
            let owner = format!("{unit_name}::{}", d.name);
            rendered.push(self.markdown(d.item.node.doc.as_ref(), &owner));
        }
        let mut sections = Vec::new();
        for category in Category::ALL {
            let mut entries: Vec<Entry> = decls
                .iter()
                .zip(&rendered)
                .filter(|(d, _)| d.category == category)
                .map(|(d, r)| Entry {
                    name: d.name.clone(),
                    href: format!("{dir}/{}", d.page()),
                    summary: r.summary.clone(),
                    deprecated: d.item.node.annotations.iter().flat_map(|g| &g.node.annotations).any(|a| {
                        self.interner.resolve(a.node.name.node) == "deprecated"
                    }),
                })
                .collect();
            if entries.is_empty() {
                continue;
            }
            entries.sort_by_cached_key(|e| (e.name.to_lowercase(), e.name.clone()));
            let (title, id) = category.heading();
            sections.push(Section { title, id, entries });
        }
        let sidebar = Sidebar {
            unit: unit_name.clone(),
            href: view.href.clone(),
            sections: sections.clone(),
        };

        let mut pages = Vec::new();
        for (d, r) in decls.iter().zip(rendered) {
            let page = self.item_page(d, r, &unit_name);
            pages.push(Page {
                path: format!("{dir}/{}", d.page()),
                content: Content::Item(page),
                sidebar: sidebar.clone(),
            });
        }

        // the unit's own page
        let doc = self.markdown(info.tree.and_then(|t| t.doc.as_ref()), &unit_name).html;
        let link = |j: usize| Link {
            name: graph.name(j).to_owned(),
            href: (j < members).then(|| self.index.unit_page(j)),
        };
        let imports: Vec<Link> = graph.deps(self.u).iter().map(|&j| link(j)).collect();
        let imported_by: Vec<Link> =
            (0..members).filter(|&j| j != self.u && graph.deps(j).contains(&self.u)).map(link).collect();
        let page = format!("{dir}/index.html");
        let foreign: Vec<MemberView> =
            info.declared.foreign.iter().map(|m| self.method(m, &unit_name, &page, true)).collect();
        let source = info.text.as_ref().map(|_| format!("{dir}/source.html"));
        let unit_page = UnitPage {
            unit: view,
            doc,
            imports,
            imported_by,
            sections,
            foreign,
            interface: self.interface(),
            source: source.clone(),
        };
        pages.push(Page {
            path: page,
            content: Content::Unit(unit_page),
            sidebar: sidebar.clone(),
        });

        // the source, every line numbered
        if let (Some(text), Some(path)) = (&info.text, source) {
            let original = text.file.original();
            let code = highlight::render(original, &text.tokens, 0, original.len() as u32, Gaps::Verbatim, &mut |_| None);
            let gutter: String = (1..=text.file.line_count()).map(|n| format!("<a id=\"L{n}\" href=\"#L{n}\">{n}</a>\n")).collect();
            pages.push(Page {
                path,
                content: Content::Source(SourcePage {
                    unit: unit_name,
                    path: text.file.path().display().to_string(),
                    gutter,
                    code,
                }),
                sidebar,
            });
        }
        pages
    }

    /// A declaration's page.
    fn item_page(&mut self, d: &Decl<'_>, doc: Rendered, unit: &str) -> ItemPage {
        let info = self.info;
        let here = format!("{}/{}", self.dir(), d.page());
        let item = &d.item.node;
        let end = match &item.kind {
            ItemKind::Fn(f) => f.signature.end,
            _ => d.item.span.end,
        };
        let signature = self.signature(d.item.span.start, end, &d.generics, &here);
        let deprecated = info.text.as_ref().and_then(|t| deprecation(item, t.file, self.interner));
        self.search.push([d.name.clone(), unit.to_owned(), d.category.label().to_owned(), here.clone(), doc.plain.clone()]);

        let mut groups = Vec::new();
        let tokens: &[(Token, Span)] = info.text.as_ref().map_or(&[], |t| &t.tokens);
        let mut parts: Vec<MemberView> = Vec::new();
        let (title, id) = match &item.kind {
            ItemKind::Struct { fields, .. } => {
                for f in fields {
                    let name = self.interner.resolve(f.name.node).to_owned();
                    let start = f.annotations.first().map_or(f.name.span.start, |a| a.span.start);
                    let owner = format!("{unit}::{}::{name}", d.name);
                    let doc = self.markdown(f.doc.as_ref(), &owner).html;
                    parts.push(MemberView {
                        id: format!("field.{name}"),
                        signature: self.signature(start, f.ty.span.end, &d.generics, &here),
                        name,
                        doc,
                        source: self.line(start),
                    });
                }
                ("Fields", "fields")
            }
            ItemKind::Enum { variants, .. } => {
                for v in variants {
                    let name = self.interner.resolve(v.name.node).to_owned();
                    let start = v.name.span.start;
                    let owner = format!("{unit}::{}::{name}", d.name);
                    let doc = self.markdown(v.doc.as_ref(), &owner).html;
                    parts.push(MemberView {
                        id: format!("variant.{name}"),
                        signature: self.signature(start, end_of(tokens, start, &[Punct::Comma]), &d.generics, &here),
                        name,
                        doc,
                        source: self.line(start),
                    });
                }
                ("Variants", "variants")
            }
            ItemKind::Locale { members, .. } => {
                for m in members {
                    let name = self.interner.resolve(m.name.node).to_owned();
                    // the member's `fn` is the token before its name
                    let at = tokens.partition_point(|(_, s)| s.start < m.name.span.start);
                    let keyword = at.checked_sub(1).map(|k| tokens[k]).filter(|(t, _)| t.is_kw(Keyword::Fn));
                    let start = m
                        .annotations
                        .first()
                        .map(|a| a.span.start)
                        .or(keyword.map(|(_, s)| s.start))
                        .unwrap_or(m.name.span.start);
                    let owner = format!("{unit}::{}::{name}", d.name);
                    let doc = self.markdown(m.doc.as_ref(), &owner).html;
                    parts.push(MemberView {
                        id: format!("member.{name}"),
                        signature: self.signature(start, end_of(tokens, start, &[Punct::Semi]), &d.generics, &here),
                        name,
                        doc,
                        source: self.line(start),
                    });
                }
                ("Members", "members")
            }
            _ => ("", ""),
        };
        if !parts.is_empty() {
            groups.push(Group { title, id, members: parts });
        }

        let (operators, methods): (Vec<&Method<'_>>, Vec<&Method<'_>>) = d.methods.iter().partition(|m| m.f.name.is_operator());
        for (list, title, id) in [(methods, "Methods", "methods"), (operators, "Operators", "operators")] {
            let members: Vec<MemberView> = list.into_iter().map(|m| self.method(m, unit, &here, false)).collect();
            if !members.is_empty() {
                groups.push(Group { title, id, members });
            }
        }

        ItemPage {
            unit: unit.to_owned(),
            kind: d.category.label(),
            name: d.name.clone(),
            signature,
            doc: doc.html,
            deprecated,
            source: self.line(d.item.span.start),
            groups,
        }
    }

    /// A method's entry on the page `here`.
    fn method(&mut self, m: &Method<'_>, unit: &str, here: &str, foreign: bool) -> MemberView {
        let (name, anchor) = m.name_and_anchor(self.interner);
        let owner = m.owner.join("::");
        let shown = if foreign { format!("{owner}.{name}") } else { name.clone() };
        let doc = self.markdown(m.doc, &format!("{unit}::{owner}::{name}"));
        let href = format!("{here}#{anchor}");
        self.search.push([format!("{owner}::{name}"), unit.to_owned(), "Method".to_owned(), href, doc.plain.clone()]);
        MemberView {
            id: anchor,
            name: shown,
            signature: self.signature(m.start, m.f.signature.end, &m.generics, here),
            doc: doc.html,
            source: self.line(m.start),
        }
    }

    /// The interface of a unit known only from it: one entry per function
    /// and global it offers, as HTML.
    fn interface(&self) -> Vec<String> {
        if self.info.tree.is_some() {
            return Vec::new();
        }
        let MemberKind::Compiled { metadata, .. } = &self.info.member.kind else { return Vec::new() };
        let i = &metadata.interface;
        let shown = |linkage: ast::Linkage, name: &str| linkage == ast::Linkage::Program && !name.starts_with("__");
        let mut out = Vec::new();
        for f in &i.fns {
            let name = self.interner.resolve(f.name);
            if !shown(f.linkage, name) {
                continue;
            }
            let params: Vec<String> = f.params.iter().map(|&t| i.types.display(t, self.interner)).collect();
            let ret = i.types.display(f.ret, self.interner);
            let text = format!("fn {name}({}) -> {ret}", params.join(", "));
            out.push(format!("<span class=\"kw\">fn</span> {}", highlight::escape(&text[3..])));
        }
        for g in &i.globals {
            let name = self.interner.resolve(g.name);
            if !shown(g.linkage, name) {
                continue;
            }
            let text = format!("{name}: {}", i.types.display(g.ty, self.interner));
            out.push(format!("<span class=\"kw\">let</span> {}", highlight::escape(&text)));
        }
        out
    }
}
