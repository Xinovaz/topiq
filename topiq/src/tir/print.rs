//! Rendering an analysed unit as readable text.
//!
//! The `tqc emit --stage=tir` dump is meant to answer the questions analysis settles: what each name
//! resolved to, what type each binding ended up with, and which values were
//! folded. So bindings print with an id (`n#0`), constants print with their
//! type (`1i32`), and control flow prints as an indented tree while plain
//! arithmetic stays on one line.
//!
//! Two things the program left implicit are shown explicitly: reading through
//! a reference prints as `(*p)`, and a reference converted to a weaker type
//! prints as `coerce(e, T)`. Neither is Topiq syntax (Topiq has no way to
//! write them), which is exactly why the dump spells them out.

use std::fmt::Write;

use crate::ast::Linkage;
use crate::intern::Interner;

use super::adt::{AdtKind, VariantShape};
use super::{
    Arm, Block, Callee, Expr, ExprKind, Fn, FnKind, GlobalKind, Pat, PatKind, Stmt, Ty, TypeTable,
    Unit, Value,
};

/// Renders a unit.
pub fn unit(u: &Unit, interner: &Interner) -> String {
    let mut p = Printer {
        out: String::new(),
        depth: 0,
        interner,
        unit: u,
        func: None,
    };
    p.line(&format!("unit {}", u.name));
    p.depth += 1;
    for (i, a) in u.types.adts().iter().enumerate() {
        p.adt(super::AdtId(i as u32), a.origin.is_some());
    }
    for x in &u.externs {
        let params: Vec<String> = x.params.iter().map(|&t| p.ty(t)).collect();
        let line = format!(
            "extern fn {}::{}({}) -> {}",
            x.unit,
            interner.resolve(x.name),
            params.join(", "),
            p.ty(x.ret)
        );
        p.line(&line);
    }
    for g in &u.globals {
        let mut head = String::from("global ");
        // a persistent object is private by nature, not because the program
        // wrote `static`, so only a unit-scope one shows the word
        if g.linkage == Linkage::Unit && g.kind == GlobalKind::Unit {
            head.push_str("static ");
        }
        match &g.kind {
            GlobalKind::Persist(f) => {
                let _ = write!(head, "persist in {} ", interner.resolve(u.func(*f).name));
            }
            GlobalKind::Imported(from) => {
                let _ = write!(head, "from {from} ");
            }
            GlobalKind::Unit => {}
        }
        let _ = write!(head, "{}: ", interner.resolve(g.name));
        if g.constant {
            head.push_str("const ");
        }
        head.push_str(&p.ty(g.ty));
        match &g.value {
            Some(v) => {
                let _ = write!(head, " = {}", value(v, g.ty, &u.types, interner));
            }
            None if matches!(g.kind, GlobalKind::Imported(_)) => {}
            None => head.push_str(" = <not yet evaluated>"),
        }
        p.line(&head);
    }
    for f in &u.fns {
        p.func(f);
    }
    p.out
}

/// Renders a value of type `ty` the way the dump shows constants.
pub fn value(v: &Value, ty: Ty, types: &TypeTable, interner: &Interner) -> String {
    match v {
        Value::Int(x, t) => format!("{x}{t}"),
        Value::Float(x, t) => format!("{}{t}", t.text(*x)),
        Value::Bool(b) => b.to_string(),
        Value::Char(c) => format!("'{}'", c.escape_debug()),
        Value::Void => "void".to_owned(),
        Value::Str(s) => format!("\"{}\"", interner.resolve(*s).escape_debug()),
        Value::Array(items) => {
            let elem = types.as_array(ty).map_or(Ty::Void, |(e, _)| e);
            let parts: Vec<String> = items.iter().map(|x| value(x, elem, types, interner)).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Struct(fields) => {
            let Ty::Adt(id) = ty else {
                return format!("{v:?}");
            };
            let def = types.adt(id);
            let parts: Vec<String> = fields
                .iter()
                .zip(def.fields())
                .map(|(x, f)| format!("{}: {}", interner.resolve(f.name), value(x, f.ty, types, interner)))
                .collect();
            format!("{} {{ {} }}", types.adt_name(id, interner), parts.join(", "))
        }
        Value::Enum { variant, fields } => {
            let Ty::Adt(id) = ty else {
                return format!("{v:?}");
            };
            let def = types.adt(id);
            let Some(var) = def.variants().get(*variant as usize) else {
                return format!("{v:?}");
            };
            let head = format!("{}::{}", types.adt_name(id, interner), interner.resolve(var.name));
            let parts = fields
                .iter()
                .zip(&var.fields)
                .map(|(x, f)| (interner.resolve(f.name).to_owned(), value(x, f.ty, types, interner)));
            shaped(head, var.shape, parts)
        }
    }
}

/// `head`, `head(a, b)` or `head { x: a, y: b }`, as a variant of `shape`
/// writes its parts, each given with its name.
fn shaped(head: String, shape: VariantShape, parts: impl Iterator<Item = (String, String)>) -> String {
    match shape {
        VariantShape::Unit => head,
        VariantShape::Tuple => format!("{head}({})", parts.map(|(_, x)| x).collect::<Vec<_>>().join(", ")),
        VariantShape::Struct => {
            let parts: Vec<String> = parts.map(|(n, x)| format!("{n}: {x}")).collect();
            format!("{head} {{ {} }}", parts.join(", "))
        }
    }
}

struct Printer<'a> {
    out: String,
    depth: usize,
    interner: &'a Interner,
    unit: &'a Unit,
    func: Option<&'a Fn>,
}

impl<'a> Printer<'a> {
    fn line(&mut self, text: &str) {
        for _ in 0..self.depth {
            self.out.push_str("  ");
        }
        let _ = writeln!(self.out, "{text}");
    }

    fn nested(&mut self, f: impl FnOnce(&mut Self)) {
        self.depth += 1;
        f(self);
        self.depth -= 1;
    }

    fn ty(&self, t: Ty) -> String {
        self.unit.types.display(t, self.interner)
    }

    fn name(&self, s: crate::intern::Symbol) -> &'a str {
        self.interner.resolve(s)
    }

    fn adt(&mut self, id: super::AdtId, imported: bool) {
        let a = self.unit.types.adt(id);
        let mut head = String::new();
        if a.repr.packed {
            head.push_str("[packed] ");
        }
        if let Some(n) = a.repr.align {
            let _ = write!(head, "[align: {n}] ");
        }
        if let Some(t) = a.repr.tag {
            let _ = write!(head, "[repr: {t}] ");
        }
        if a.linkage == Linkage::Unit {
            head.push_str("static ");
        }
        let name = self.unit.types.adt_name(id, self.interner);
        let fields = |fs: &[super::FieldDef]| fs.iter().map(|f| (self.name(f.name).to_owned(), self.ty(f.ty))).collect::<Vec<_>>();
        match &a.kind {
            AdtKind::Struct { fields: fs } => {
                head.push_str(&shaped(format!("struct {name}"), VariantShape::Struct, fields(fs).into_iter()));
            }
            AdtKind::Enum { variants } => {
                let vs: Vec<String> = variants
                    .iter()
                    .map(|v| shaped(self.name(v.name).to_owned(), v.shape, fields(&v.fields).into_iter()))
                    .collect();
                let _ = write!(head, "enum {name} {{ {} }}", vs.join(", "));
            }
        }
        if imported {
            head.push_str("  [imported]");
        }
        self.line(&head);
    }

    fn func(&mut self, f: &'a Fn) {
        // the head: parameters, instance arguments, return type and notes
        self.func = Some(f);
        let params: Vec<String> = f
            .params
            .iter()
            .map(|&id| {
                let l = f.local(id);
                format!(
                    "{}: {}{}",
                    self.local_name(id),
                    if l.constexpr { "constexpr " } else if l.constant { "const " } else { "" },
                    self.ty(l.ty)
                )
            })
            .collect();
        let args = if f.args.is_empty() {
            String::new()
        } else {
            let parts: Vec<String> = f
                .args
                .iter()
                .map(|&a| match a {
                    super::Arg::Type(t) => self.ty(t),
                    super::Arg::Const(v) => v.to_string(),
                })
                .collect();
            format!("<{}>", parts.join(", "))
        };
        let mut head = format!(
            "fn {}{args}({}) -> {}",
            self.name(f.name),
            params.join(", "),
            self.ty(f.ret)
        );
        if f.linkage == Linkage::Unit {
            head.push_str("  [unit linkage]");
        }
        if f.kind == FnKind::Constant {
            head.push_str("  [constant: evaluated, never compiled]");
        }
        self.line(&head);

        // the body
        self.nested(|p| p.block_body(&f.body));
        self.func = None;
    }

    fn local_name(&self, id: super::LocalId) -> String {
        match self.func {
            Some(f) => format!("{}#{}", self.name(f.local(id).name), id.0),
            None => format!("#{}", id.0),
        }
    }

    fn global_name(&self, id: super::GlobalId) -> String {
        let g = self.unit.global(id);
        match &g.kind {
            GlobalKind::Imported(from) => format!("{from}::{}", self.name(g.name)),
            _ => self.name(g.name).to_owned(),
        }
    }

    fn callee(&self, c: Callee) -> String {
        match c {
            Callee::Fn(id) => self.name(self.unit.func(id).name).to_owned(),
            Callee::Extern(id) => {
                let x = self.unit.extern_fn(id);
                format!("{}::{}", x.unit, self.name(x.name))
            }
        }
    }

    fn block_body(&mut self, b: &Block) {
        for s in &b.stmts {
            match s {
                Stmt::Let { local, init } => {
                    let (ty, constant) = match self.func {
                        Some(f) => (self.ty(f.local(*local).ty), f.local(*local).constant),
                        None => ("?".to_owned(), false),
                    };
                    let head = format!(
                        "let {}: {}{ty}",
                        self.local_name(*local),
                        if constant { "const " } else { "" }
                    );
                    match init {
                        Some(e) => self.labeled(&format!("{head} ="), e),
                        None => self.line(&head),
                    }
                }
                Stmt::LetPat { pat, init } => {
                    let head = format!("let {} =", self.pat(pat));
                    self.labeled(&head, init);
                }
                Stmt::Expr(e) => self.stmt_expr(e),
            }
        }
        if let Some(v) = &b.value {
            self.labeled("=>", v);
        }
    }

    /// Prints `head` followed by the expression: on the same line when it is
    /// short, below it when it has structure.
    fn labeled(&mut self, head: &str, e: &Expr) {
        match self.inline(e) {
            Some(text) => self.line(&format!("{head} {text}")),
            None => {
                self.line(head);
                self.nested(|p| p.stmt_expr(e));
            }
        }
    }

    /// Prints each of `items` as a child line.
    fn children<'e>(&mut self, items: impl IntoIterator<Item = &'e Expr>) {
        self.nested(|p| {
            for x in items {
                p.stmt_expr(x);
            }
        });
    }

    /// Prints an expression as a statement.
    fn stmt_expr(&mut self, e: &Expr) {
        if let Some(text) = self.inline(e) {
            self.line(&text);
            return;
        }
        match &e.kind {
            ExprKind::Block(b) => {
                self.line(&format!("block: {}", self.ty(b.ty)));
                self.nested(|p| p.block_body(b));
            }
            ExprKind::If { cond, then, els } => {
                self.labeled(&format!("if: {}", self.ty(e.ty)), cond);
                self.line("then");
                self.nested(|p| p.block_body(then));
                if let Some(els) = els {
                    self.line("else");
                    self.nested(|p| p.stmt_expr(els));
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.labeled(&format!("match: {}", self.ty(e.ty)), scrutinee);
                self.nested(|p| {
                    for Arm { pat, body } in arms {
                        let head = format!("{} =>", p.pat(pat));
                        p.labeled(&head, body);
                    }
                });
            }
            ExprKind::Loop { id, body } => {
                self.line(&format!("loop L{}: {}", id.0, self.ty(e.ty)));
                self.nested(|p| p.block_body(body));
            }
            ExprKind::While { id, cond, body } => {
                self.labeled(&format!("while L{}", id.0), cond);
                self.nested(|p| p.block_body(body));
            }
            ExprKind::ForRange {
                id,
                var,
                start,
                end,
                inclusive,
                body,
            } => {
                let range = match (self.inline(start), self.inline(end)) {
                    (Some(a), Some(b)) => {
                        format!("{a}{}{b}", if *inclusive { "..=" } else { ".." })
                    }
                    _ => "<range>".to_owned(),
                };
                self.line(&format!("for L{} {} in {range}", id.0, self.local_name(*var)));
                self.nested(|p| p.block_body(body));
            }
            ExprKind::Break { target, value } => match value {
                Some(v) => self.labeled(&format!("break L{}", target.0), v),
                None => self.line(&format!("break L{}", target.0)),
            },
            ExprKind::Return(Some(v)) => self.labeled("return", v),
            ExprKind::Assign { place, value } => match self.inline(place) {
                Some(target) => self.labeled(&format!("{target} ="), value),
                None => {
                    self.line("assign");
                    self.children([value.as_ref(), place.as_ref()]);
                }
            },
            // anything else only fails to inline because a child has
            // structure; show the node and its children beneath it
            ExprKind::Binary { op, lhs, rhs } => {
                self.line(&format!("binary {}: {}", op.text(), self.ty(e.ty)));
                self.children([lhs.as_ref(), rhs.as_ref()]);
            }
            ExprKind::Logical { op, lhs, rhs } => {
                self.line(&format!("logical {}", op.text()));
                self.children([lhs.as_ref(), rhs.as_ref()]);
            }
            ExprKind::Unary { op, operand } => {
                self.line(&format!("unary {}: {}", op.text(), self.ty(e.ty)));
                self.children([operand.as_ref()]);
            }
            ExprKind::Cast { expr, to } => {
                self.line(&format!("as {}", self.ty(*to)));
                self.children([expr.as_ref()]);
            }
            ExprKind::Call { callee, args } => {
                self.line(&format!("call {}", self.callee(*callee)));
                self.children(args);
            }
            ExprKind::IndirectCall { callee, args } => {
                self.line(&format!("call through: {}", self.ty(e.ty)));
                self.children(std::iter::once(callee.as_ref()).chain(args));
            }
            ExprKind::Closure { code, captures } => {
                let name = self.interner.resolve(self.unit.func(*code).name);
                self.line(&format!("closure {name}#{}: {}", code.0, self.ty(e.ty)));
                self.children(captures);
            }
            ExprKind::FnRef(_) | ExprKind::ThunkRef(_) | ExprKind::FnAsClosure(_) => {
                unreachable!("a function value always prints inline")
            }
            ExprKind::Intrinsic { which, args } => {
                self.line(&format!("intrinsic {}", which.name()));
                self.children(args);
            }
            ExprKind::Quantum { op, args } => {
                self.line(&format!("quantum {}: {}", op.name(), self.ty(e.ty)));
                self.children(args);
            }
            ExprKind::Growable { op, array, args } => {
                self.line(&format!("growable {}: {}", op.name(), self.ty(e.ty)));
                self.children(std::iter::once(array.as_ref()).chain(args));
            }
            ExprKind::Grow(x) => {
                self.line(&format!("grow: {}", self.ty(e.ty)));
                self.children([x.as_ref()]);
            }
            ExprKind::Field { base, field } => {
                self.line(&format!("field {}", self.field_name(base.ty, *field)));
                self.children([base.as_ref()]);
            }
            ExprKind::Index { base, index } => {
                self.line(&format!("index: {}", self.ty(e.ty)));
                self.children([base.as_ref(), index.as_ref()]);
            }
            ExprKind::Deref(r) => {
                self.line("deref");
                self.children([r.as_ref()]);
            }
            ExprKind::Ref(x) => {
                self.line(&format!("ref: {}", self.ty(e.ty)));
                self.children([x.as_ref()]);
            }
            ExprKind::Coerce(x) => {
                self.line(&format!("coerce to {}", self.ty(e.ty)));
                self.children([x.as_ref()]);
            }
            ExprKind::StructLit { fields } | ExprKind::Variant { fields, .. } => {
                let head = match e.ty {
                    Ty::Tuple(_) => format!("tuple: {}", self.ty(e.ty)),
                    _ => format!("{}:", self.constructor(e)),
                };
                self.line(&head);
                self.nested(|p| {
                    for (i, x) in fields {
                        let head = format!("{}:", p.member_name(e, *i));
                        p.labeled(&head, x);
                    }
                });
            }
            ExprKind::ArrayLit(items) => {
                self.line(&format!("array: {}", self.ty(e.ty)));
                self.children(items);
            }
            ExprKind::ArrayRepeat { elem, len } => {
                self.line(&format!("array of {len}: {}", self.ty(e.ty)));
                self.children([elem.as_ref()]);
            }
            ExprKind::Const(_)
            | ExprKind::Local(_)
            | ExprKind::Global(_)
            | ExprKind::Continue { .. }
            | ExprKind::Return(None) => unreachable!("always inline"),
        }
    }

    /// The name of field `i` of the structure `ty`.
    fn field_name(&self, ty: Ty, i: u32) -> String {
        match ty {
            Ty::Adt(id) => self
                .unit
                .types
                .adt(id)
                .fields()
                .get(i as usize)
                .map_or_else(|| format!("#{i}"), |f| self.name(f.name).to_owned()),
            _ => format!("#{i}"),
        }
    }

    /// `Point` for a structure literal, `Shape::Circle` for a variant.
    fn constructor(&self, e: &Expr) -> String {
        let Ty::Adt(id) = e.ty else {
            // a tuple's constructor is written by the parentheses alone
            return String::new();
        };
        let name = self.unit.types.adt_name(id, self.interner);
        match &e.kind {
            ExprKind::Variant { variant, .. } => {
                let v = &self.unit.types.adt(id).variants()[*variant as usize];
                format!("{name}::{}", self.name(v.name))
            }
            _ => name,
        }
    }

    /// Whether `e` builds a variant whose values are positional.
    fn is_tuple_variant(&self, e: &Expr) -> bool {
        match (&e.kind, e.ty) {
            (ExprKind::Variant { variant, .. }, Ty::Adt(id)) => self
                .unit
                .types
                .adt(id)
                .variants()
                .get(*variant as usize)
                .is_some_and(|v| v.shape == VariantShape::Tuple),
            _ => false,
        }
    }

    /// The name of member `i` of a structure literal or variant: a field name,
    /// or a position for a tuple variant.
    fn member_name(&self, e: &Expr, i: u32) -> String {
        let Ty::Adt(id) = e.ty else {
            return format!("{i}");
        };

        let def = self.unit.types.adt(id);
        let field = match &e.kind {
            ExprKind::Variant { variant, .. } => def.variants()[*variant as usize].fields.get(i as usize),
            _ => def.fields().get(i as usize),
        };
        match field {
            Some(f) if f.name != crate::intern::Symbol::EMPTY => self.name(f.name).to_owned(),
            _ => format!("{i}"),
        }
    }

    fn pat(&self, p: &Pat) -> String {
        match &p.kind {
            PatKind::Wild => "_".to_owned(),
            PatKind::Bind(l) => self.local_name(*l),
            PatKind::BindRef(l) => format!("&{}", self.local_name(*l)),
            PatKind::Const(v) => value(v, p.ty, &self.unit.types, self.interner),
            // a tuple pattern names its elements by position, and is written
            // the way it was: `(a, _)`
            PatKind::Struct { fields } if matches!(p.ty, Ty::Tuple(_)) => {
                let fs: Vec<String> = fields.iter().map(|(_, f)| self.pat(f)).collect();
                format!("({})", fs.join(", "))
            }
            PatKind::Struct { fields } => {
                let name = match p.ty {
                    Ty::Adt(id) => self.unit.types.adt_name(id, self.interner),
                    _ => "?".to_owned(),
                };
                let parts = fields.iter().map(|(i, f)| (self.field_name(p.ty, *i), self.pat(f)));
                shaped(name, VariantShape::Struct, parts)
            }
            PatKind::Variant { variant, fields } => {
                let Ty::Adt(id) = p.ty else {
                    return "?".to_owned();
                };
                let v = &self.unit.types.adt(id).variants()[*variant as usize];
                let head = format!("{}::{}", self.unit.types.adt_name(id, self.interner), self.name(v.name));
                let parts = fields.iter().map(|(i, f)| (self.name(v.fields[*i as usize].name).to_owned(), self.pat(f)));
                shaped(head, v.shape, parts)
            }
        }
    }

    /// The one-line form of an expression.
    fn inline(&self, e: &Expr) -> Option<String> {
        let list = |items: &mut dyn Iterator<Item = &Expr>| -> Option<String> {
            let parts: Option<Vec<String>> = items.map(|a| self.inline(a)).collect();
            Some(parts?.join(", "))
        };
        Some(match &e.kind {
            ExprKind::Const(v) => value(v, e.ty, &self.unit.types, self.interner),
            ExprKind::Local(l) => self.local_name(*l),
            ExprKind::Global(g) => self.global_name(*g),
            ExprKind::Call { callee, args } => {
                format!("{}({})", self.callee(*callee), list(&mut args.iter())?)
            }
            ExprKind::FnRef(callee) => self.callee(*callee),
            ExprKind::FnAsClosure(f) => format!("as_closure({})", self.inline(f)?),
            ExprKind::ThunkRef(code) => {
                format!("{}#{}", self.interner.resolve(self.unit.func(*code).name), code.0)
            }
            ExprKind::IndirectCall { callee, args } => {
                format!("({})({})", self.inline(callee)?, list(&mut args.iter())?)
            }
            ExprKind::Intrinsic { which, args } => {
                format!("{}({})", which.name(), list(&mut args.iter())?)
            }
            ExprKind::Quantum { op, args } => {
                format!("{}({})", op.name(), list(&mut args.iter())?)
            }
            ExprKind::Growable { op, array, args } => {
                format!("{}.{}({})", self.inline(array)?, op.name(), list(&mut args.iter())?)
            }
            ExprKind::Grow(x) => format!("grow({})", self.inline(x)?),
            ExprKind::Unary { op, operand } => format!("({}{})", op.text(), self.inline(operand)?),
            ExprKind::Binary { op, lhs, rhs } => format!(
                "({} {} {})",
                self.inline(lhs)?,
                op.text(),
                self.inline(rhs)?
            ),
            ExprKind::Logical { op, lhs, rhs } => format!(
                "({} {} {})",
                self.inline(lhs)?,
                op.text(),
                self.inline(rhs)?
            ),
            ExprKind::Cast { expr, to } => format!("({} as {})", self.inline(expr)?, self.ty(*to)),
            ExprKind::Field { base, field } => {
                format!("{}.{}", self.inline(base)?, self.field_name(base.ty, *field))
            }
            ExprKind::Index { base, index } => {
                format!("{}[{}]", self.inline(base)?, self.inline(index)?)
            }
            ExprKind::Deref(r) => format!("(*{})", self.inline(r)?),
            ExprKind::Ref(x) => format!("&{}", self.inline(x)?),
            ExprKind::Coerce(x) => format!("coerce({}, {})", self.inline(x)?, self.ty(e.ty)),
            ExprKind::StructLit { fields } | ExprKind::Variant { fields, .. } => {
                if matches!(e.ty, Ty::Tuple(_)) || self.is_tuple_variant(e) {
                    let parts: Option<Vec<String>> = fields.iter().map(|(_, x)| self.inline(x)).collect();
                    return Some(format!("{}({})", self.constructor(e), parts?.join(", ")));
                }
                let parts: Option<Vec<String>> = fields
                    .iter()
                    .map(|(i, x)| Some(format!("{}: {}", self.member_name(e, *i), self.inline(x)?)))
                    .collect();
                let parts = parts?;
                if parts.is_empty() {
                    self.constructor(e)
                } else {
                    format!("{} {{ {} }}", self.constructor(e), parts.join(", "))
                }
            }
            ExprKind::ArrayLit(items) => format!("[{}]", list(&mut items.iter())?),
            ExprKind::ArrayRepeat { elem, len } => format!("[{}; {len}]", self.inline(elem)?),
            ExprKind::Assign { place, value } => {
                format!("{} = {}", self.inline(place)?, self.inline(value)?)
            }
            ExprKind::Break { target, value: None } => format!("break L{}", target.0),
            ExprKind::Break {
                target,
                value: Some(v),
            } => format!("break L{} {}", target.0, self.inline(v)?),
            ExprKind::Continue { target } => format!("continue L{}", target.0),
            ExprKind::Return(None) => "return".to_owned(),
            ExprKind::Return(Some(v)) => format!("return {}", self.inline(v)?),
            ExprKind::Block(_)
            | ExprKind::If { .. }
            | ExprKind::Match { .. }
            | ExprKind::Loop { .. }
            | ExprKind::While { .. }
            | ExprKind::Closure { .. }
            | ExprKind::ForRange { .. } => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::{SourceId, Span};
    use crate::tir::{
        AdtDef, BinOp, FieldDef, Global, IntTy, Local, LocalId, LoopId, Repr,
    };

    fn sp() -> Span {
        Span::new(SourceId(0), 0, 1)
    }

    fn e(kind: ExprKind, ty: Ty) -> Expr {
        Expr { kind, ty, span: sp() }
    }

    fn int(v: i128) -> Expr {
        Expr::constant(Value::Int(v, IntTy::I32), Ty::Int(IntTy::I32), sp())
    }

    fn sample(interner: &mut Interner) -> Unit {
        let i32_ = Ty::Int(IntTy::I32);
        let n = LocalId(0);
        let body = Block {
            stmts: vec![],
            value: Some(Box::new(e(
                ExprKind::If {
                    cond: Box::new(e(
                        ExprKind::Binary {
                            op: BinOp::Lt,
                            lhs: Box::new(e(ExprKind::Local(n), i32_)),
                            rhs: Box::new(int(2)),
                        },
                        Ty::Bool,
                    )),
                    then: Box::new(Block {
                        stmts: vec![],
                        value: Some(Box::new(e(ExprKind::Local(n), i32_))),
                        ty: i32_,
                        span: sp(),
                    }),
                    els: Some(Box::new(e(
                        ExprKind::Loop {
                            id: LoopId(0),
                            body: Box::new(Block {
                                stmts: vec![Stmt::Expr(e(
                                    ExprKind::Break {
                                        target: LoopId(0),
                                        value: Some(Box::new(int(0))),
                                    },
                                    Ty::Never,
                                ))],
                                value: None,
                                ty: Ty::Never,
                                span: sp(),
                            }),
                        },
                        i32_,
                    ))),
                },
                i32_,
            ))),
            ty: i32_,
            span: sp(),
        };
        let mut u = Unit::new("demo", SourceId(0));
        u.fns.push(Fn {
            args: Vec::new(),
            name: interner.intern("f"),
            linkage: Linkage::Program,
            origin: None,
            method: None,
            closure: false,
            kind: FnKind::Runtime,
            params: vec![n],
            ret: i32_,
            locals: vec![Local {
                name: interner.intern("n"),
                ty: i32_,
                constant: false,
                constexpr: false,
                aux: false,
                span: sp(),
            }],
            body,
            span: sp(),
            attrs: crate::tir::FnAttrs::default(),
        });
        u.globals.push(Global {
            name: interner.intern("LIMIT"),
            ty: i32_,
            constant: true,
            linkage: Linkage::Unit,
            kind: GlobalKind::Unit,
            init: None,
            value: Some(Value::Int(64, IntTy::I32)),
            startup: false,
            span: sp(),
            symbol: None,
            deprecated: None,
        });
        u
    }

    #[test]
    fn a_unit_prints_its_globals_and_functions() {
        let mut i = Interner::new();
        let u = sample(&mut i);
        let text = unit(&u, &i);
        assert!(text.starts_with("unit demo\n"), "{text}");
        assert!(text.contains("global static LIMIT: const i32 = 64i32"), "{text}");
        assert!(text.contains("fn f(n#0: i32) -> i32"), "{text}");
    }

    #[test]
    fn plain_expressions_stay_on_one_line_and_control_flow_nests() {
        let mut i = Interner::new();
        let u = sample(&mut i);
        let text = unit(&u, &i);
        assert!(text.contains("if: i32 (n#0 < 2i32)"), "{text}");
        assert!(text.contains("loop L0: i32"), "{text}");
        assert!(text.contains("break L0 0i32"), "{text}");
    }

    #[test]
    fn an_unevaluated_global_says_so() {
        let mut i = Interner::new();
        let mut u = sample(&mut i);
        u.globals[0].value = None;
        assert!(unit(&u, &i).contains("<not yet evaluated>"));
    }

    #[test]
    fn declared_types_and_aggregate_values_print_as_written() {
        let mut i = Interner::new();
        let mut u = sample(&mut i);
        let (x, y) = (i.intern("x"), i.intern("y"));
        let field = |name| FieldDef {
            name,
            ty: Ty::Int(IntTy::I32),
            span: sp(),
        };
        let p = u.types.add_adt(AdtDef {
            args: Vec::new(),
            name: i.intern("Point"),
            origin: None,
            linkage: Linkage::Program,
            open: false,
            opaque: false,
            kind: AdtKind::Struct {
                fields: vec![field(x), field(y)],
            },
            repr: Repr {
                packed: true,
                ..Repr::default()
            },
            span: sp(),
        });
        let text = unit(&u, &i);
        assert!(text.contains("[packed] struct Point { x: i32, y: i32 }"), "{text}");
        let v = Value::Struct(vec![Value::Int(1, IntTy::I32), Value::Int(2, IntTy::I32)]);
        assert_eq!(value(&v, Ty::Adt(p), &u.types, &i), "Point { x: 1i32, y: 2i32 }");
        let s = Value::Str(i.intern("hi\n"));
        let st = u.types.string();
        assert_eq!(value(&s, st, &u.types, &i), "\"hi\\n\"");
        assert_eq!(value(&Value::Char('a'), Ty::Char, &u.types, &i), "'a'");
    }
}
