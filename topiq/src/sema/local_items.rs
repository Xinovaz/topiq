//! Items declared inside a block.
//!
//! A block may declare functions, structures and enumerations of its own, and
//! add methods to them in an `impl` block. In a quantum unit it may also
//! declare covers, gauges, base types, locales, chains and map locales, each
//! under a unique name the block's own names map to. They are in scope
//! throughout the block and the blocks inside it, in any order, as a unit's
//! items are; outside it nothing can name them. A function declared inside
//! another cannot use the outer function's bindings: only a closure captures.
//!
//! Every such item is declared with the unit's own, before any body is
//! checked, by walking each function's body for blocks that declare items.
//! Each block's items make a *layer*, looked up before unit scope. A function
//! or type declared in a block keeps the layers around it, so that its
//! signature, fields and body see what the block sees, wherever they are
//! checked from.
//!
//! Such an item is the unit's own, kept from other units: its symbol is its
//! own, and an importer cannot name it.

use crate::ast::{self, ItemKind, Linkage};
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{AdtDef, AdtId, FnId};

use super::items::UnitCx;
use super::report;
use super::scope::Def;

/// The items one block declares.
#[derive(Clone, Debug, Default)]
pub struct Layer {
    /// Its functions.
    pub values: Vec<(Symbol, Def)>,
    /// Its structures and enumerations.
    pub types: Vec<(Symbol, AdtId)>,
    /// Its covers, gauges, base types, locales, chains and map locales: each
    /// name as written, and the name it is declared under, which no other
    /// declaration shares.
    pub geometry: Vec<(Symbol, Symbol)>,
}

/// The address a block is known by: it does not move while the unit is
/// analysed.
pub fn key(b: &ast::Block) -> usize {
    std::ptr::from_ref(b) as usize
}

impl<'a> UnitCx<'a> {
    /// Declares the items of every block in `body`, however deeply nested.
    pub(super) fn declare_block_items(&mut self, body: &'a ast::Block) {
        self.walk_block(body);
    }

    /// Declares the items of every block in the expression `e`.
    pub(super) fn declare_expr_items(&mut self, e: &'a Spanned<ast::Expr>) {
        self.walk_expr(e);
    }

    /// The function or type `name` a block around the code being checked
    /// declares, innermost first.
    pub(super) fn local_value(&self, name: Symbol) -> Option<Def> {
        self.local_items
            .iter()
            .rev()
            .find_map(|l| l.values.iter().find(|(n, _)| *n == name).map(|&(_, d)| d))
    }

    /// The structure or enumeration `name` a block around the code being
    /// checked declares.
    pub(super) fn local_type(&self, name: Symbol) -> Option<AdtId> {
        self.local_items
            .iter()
            .rev()
            .find_map(|l| l.types.iter().find(|(n, _)| *n == name).map(|&(_, t)| t))
    }

    /// The name the cover, gauge, base type, locale, chain or map locale
    /// `name` is declared under: a block's own, innermost first, or the
    /// unit's, whose names are their own.
    pub(super) fn geometry_key(&self, name: Symbol) -> Symbol {
        self.local_items
            .iter()
            .rev()
            .find_map(|l| l.geometry.iter().find(|(n, _)| *n == name).map(|&(_, k)| k))
            .unwrap_or(name)
    }

    /// Runs `f` with the layers `layers` in scope in place of the current
    /// ones.
    pub(super) fn with_layers<T>(&mut self, layers: Vec<Layer>, f: impl FnOnce(&mut Self) -> T) -> T {
        let saved = std::mem::replace(&mut self.local_items, layers);
        let out = f(self);
        self.local_items = saved;
        out
    }

    fn walk_block(&mut self, b: &'a ast::Block) {
        let items: Vec<(&'a ast::Item, Span)> = b
            .stmts
            .iter()
            .filter_map(|s| match &s.node {
                ast::Stmt::Item(item) => Some((&**item, s.span)),
                _ => None,
            })
            .collect();
        let pushed = !items.is_empty();
        if pushed {
            let (layer, fns, adts, impls) = self.declare_layer(&items);
            self.block_layers.insert(key(b), layer.clone());
            let geometry: Vec<Symbol> = layer.geometry.iter().map(|&(_, k)| k).collect();
            self.local_items.push(layer);
            // the block's geometry is resolved, and its claims read, seeing
            // what the block sees
            for k in geometry {
                self.scope_geometry(k, self.local_items.clone());
            }
            for &(item, _) in &items {
                let annotated = !item.annotations.is_empty()
                    || matches!(&item.kind, ItemKind::Fn(f) if f.params.iter().any(|p| !p.annotations.is_empty()));
                if annotated {
                    self.annotated_items.push((item, self.local_items.clone()));
                }
            }
            let before = self.unit.fns.len();
            for (item, span) in impls {
                self.declare_methods_in(item, span, None);
            }
            let methods: Vec<FnId> = (before..self.unit.fns.len()).map(|i| FnId(i as u32)).collect();
            for &id in fns.iter().chain(&methods) {
                self.unit.fns[id.index()].attrs.hidden = true;
                self.fn_layers.insert(id, self.local_items.clone());
            }
            for &id in &adts {
                self.adt_layers.insert(id, self.local_items.clone());
            }
        }
        for s in &b.stmts {
            match &s.node {
                ast::Stmt::Let(l) => {
                    if let Some(e) = &l.init {
                        self.walk_expr(e);
                    }
                }
                ast::Stmt::Item(item) => match &item.kind {
                    ItemKind::Fn(f) => self.walk_block(&f.body),
                    ItemKind::Impl { items, .. } => {
                        for f in items {
                            self.walk_block(&f.body);
                        }
                    }
                    _ => {}
                },
                ast::Stmt::Flow(ast::Flow::Return(Some(e)) | ast::Flow::Break(Some(e))) => self.walk_expr(e),
                ast::Stmt::Forget(e) | ast::Stmt::BlockExpr(e) | ast::Stmt::Expr(e) => self.walk_expr(e),
                ast::Stmt::Flow(_) | ast::Stmt::Empty => {}
            }
        }
        if let Some(v) = &b.value {
            self.walk_expr(v);
        }
        if pushed {
            self.local_items.pop();
        }
    }

    fn walk_expr(&mut self, e: &'a Spanned<ast::Expr>) {
        use ast::Expr as E;
        match &e.node {
            E::Block(b) | E::Loop { body: b } => self.walk_block(b),
            E::If { cond, then, els } => {
                self.walk_expr(cond);
                self.walk_block(then);
                if let Some(e) = els {
                    self.walk_expr(e);
                }
            }
            E::Match { scrutinee, arms, .. } => {
                self.walk_expr(scrutinee);
                for a in arms {
                    self.walk_expr(&a.body);
                }
            }
            E::While { cond, body } => {
                self.walk_expr(cond);
                self.walk_block(body);
            }
            E::For { iter, body, .. } => {
                self.walk_expr(iter);
                self.walk_block(body);
            }
            E::Closure { body, .. } => self.walk_block(body),
            E::Paren(x)
            | E::Measure(x)
            | E::Lift(x)
            | E::Replay(x)
            | E::Try(x)
            | E::Unary { operand: x, .. }
            | E::Step { place: x, .. }
            | E::Field { receiver: x, .. }
            | E::Cast { expr: x, .. } => self.walk_expr(x),
            E::Tuple(xs) | E::Array(xs) => {
                for x in xs {
                    self.walk_expr(x);
                }
            }
            E::ArrayRepeat { elem: a, len: b }
            | E::Query { map: a, index: b }
            | E::Binary { lhs: a, rhs: b, .. }
            | E::Assign { place: a, value: b, .. }
            | E::Index { receiver: a, index: b } => {
                self.walk_expr(a);
                self.walk_expr(b);
            }
            E::StructLit { fields, .. } => {
                for f in fields {
                    self.walk_expr(&f.value);
                }
            }
            E::Range { start, end, .. } => {
                for x in [start, end].into_iter().flatten() {
                    self.walk_expr(x);
                }
            }
            E::Call { callee, args } => {
                self.walk_expr(callee);
                for a in args {
                    self.walk_expr(a);
                }
            }
            E::MethodCall { receiver, args, .. } => {
                self.walk_expr(receiver);
                for a in args {
                    self.walk_expr(a);
                }
            }
            E::Int { .. }
            | E::Float { .. }
            | E::Char(_)
            | E::Bool(_)
            | E::Str { .. }
            | E::Path { .. }
            | E::Prep(_)
            | E::Macro { .. } => {}
        }
    }

    /// Declares one block's items: its functions and types, whose names make
    /// the layer, and the `impl` blocks to declare once the layer is in
    /// scope. Anything else a block cannot declare is reported.
    #[allow(clippy::type_complexity)]
    fn declare_layer(
        &mut self,
        items: &[(&'a ast::Item, Span)],
    ) -> (Layer, Vec<FnId>, Vec<AdtId>, Vec<(&'a ast::Item, Span)>) {
        let mut layer = Layer::default();
        let (mut fns, mut adts, mut impls) = (Vec::new(), Vec::new(), Vec::new());
        let mut seen: Vec<(Symbol, Span)> = Vec::new();
        for &(item, span) in items {
            if item.linkage == Linkage::Unit {
                // reported where the block is checked
                continue;
            }
            let name = match &item.kind {
                ItemKind::Fn(f) => match f.name {
                    ast::FnName::Plain(n) if f.generics.is_empty() => Some(n),
                    ast::FnName::Plain(_) => {
                        self.report(not_in_a_block(span, "a generic function"));
                        None
                    }
                    _ => {
                        impls.push((item, span));
                        None
                    }
                },
                ItemKind::Struct { name, generics, .. } | ItemKind::Enum { name, generics, .. } => {
                    if generics.is_empty() {
                        Some(*name)
                    } else {
                        self.report(not_in_a_block(span, "a generic type"));
                        None
                    }
                }
                ItemKind::Impl { generics, .. } => {
                    if generics.is_empty() {
                        impls.push((item, span));
                    } else {
                        self.report(not_in_a_block(span, "a generic `impl` block"));
                    }
                    None
                }
                ItemKind::Import { .. } => {
                    self.report(not_in_a_block(span, "an `import`"));
                    None
                }
                ItemKind::TypeAlias { .. } => {
                    self.report(not_in_a_block(span, "a type alias"));
                    None
                }
                ItemKind::Let { .. } => {
                    self.report(not_in_a_block(span, "a unit-scope object"));
                    None
                }
                ItemKind::Cover { name, .. }
                | ItemKind::Gauge { name, .. }
                | ItemKind::Base { name, .. }
                | ItemKind::Locale { name, .. }
                | ItemKind::Chain { name, .. }
                | ItemKind::Qmap { name, .. } => Some(*name),
                // one without annotations is a statement
                ItemKind::Assert { .. } => {
                    self.report(not_in_a_block(span, "an annotated `@static_assert`"));
                    None
                }
            };
            let Some(name) = name else { continue };
            if let Some(&(_, earlier)) = seen.iter().find(|(n, _)| *n == name.node) {
                let d = report::duplicate(name.span, earlier, self.interner.resolve(name.node), "in one block");
                self.report(d);
                continue;
            }
            seen.push((name.node, name.span));
            self.check_reserved(name.node, name.span);
            match &item.kind {
                ItemKind::Fn(f) => {
                    let mut attrs = self.item_attrs(&item.annotations, super::attrs::Subject::function(Linkage::Unit, false));
                    attrs.hidden = true;
                    let id = self.add_fn(f, Linkage::Unit, name, span);
                    self.unit.fns[id.index()].attrs = attrs;
                    layer.values.push((name.node, Def::Fn(id)));
                    fns.push(id);
                }
                ItemKind::Struct { .. } | ItemKind::Enum { .. } => {
                    let is_struct = matches!(item.kind, ItemKind::Struct { .. });
                    // a type's table and layout digest go by its qualified
                    // name, so one that shares its name with another of the
                    // unit's is told apart by where it is declared
                    let shared = self.unit.types.adts().iter().any(|a| a.origin.is_none() && a.name == name.node);
                    let written = self.interner.resolve(name.node);
                    let unique = if shared {
                        self.interner.intern_late(&format!("{written}#{}", self.unit.types.adts().len()))
                    } else {
                        name.node
                    };
                    let def = AdtDef::unresolved(unique, None, Linkage::Unit, Vec::new(), is_struct, name.span);
                    let id = self.add_adt(def, &item.kind, &item.annotations, self.frame, Vec::new());
                    layer.types.push((name.node, id));
                    adts.push(id);
                }
                ItemKind::Cover { .. }
                | ItemKind::Gauge { .. }
                | ItemKind::Base { .. }
                | ItemKind::Locale { .. }
                | ItemKind::Chain { .. }
                | ItemKind::Qmap { .. } => {
                    // declared under a name of its own, which the layer maps
                    // the written one to
                    let written = self.interner.resolve(name.node);
                    let key = self.interner.intern_late(&format!("{written}#{}", name.span.start));
                    if matches!(item.kind, ItemKind::Qmap { .. }) {
                        self.declare_qmap_as(&item.kind, name, key, span);
                    } else {
                        self.declare_geometry_as(&item.kind, name, key, Linkage::Unit, span);
                    }
                    layer.geometry.push((name.node, key));
                }
                _ => {}
            }
        }
        (layer, fns, adts, impls)
    }
}

/// An item a block cannot declare.
fn not_in_a_block(span: Span, what: &str) -> Diagnostic {
    Diagnostic::new(Code::Es18)
        .with_message(format!("{what} cannot be declared inside a block"))
        .at(span)
        .with_note(
            "a block may declare functions, structures, enumerations, and `impl` blocks for its own \
             types; generic declarations, imports, type aliases and unit-scope objects belong at unit scope",
        )
        .with_help("move it to unit scope")
}
