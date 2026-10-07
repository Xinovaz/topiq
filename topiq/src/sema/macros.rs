//! The builtin macros, `@name(args)`.
//!
//! Most of them ask a question about a type and answer it during translation,
//! as a constant: its size, its name, its fields and methods, whether one type
//! is a prefix of another. None of these needs the type's run-time type
//! information, so asking never makes a table be emitted. A few do something
//! else: `@static_assert` refuses a program whose condition is false,
//! `@aborts` says whether evaluating an expression during translation would
//! abort, and `@method` names a method as a function value.
//!
//! # What a type's operators are
//!
//! `@has_method(T, "$add")` asks whether `a + b` works on a `T`. For a
//! structure or enumeration that is whether it defines `$add`. The primitive
//! types have the operators the language gives them: numbers add, integers
//! also shift and mask, everything that can be copied has `$copy`, and so on,
//! so that a generic function bounded by `@has_method` accepts them too.

use crate::ast::{self, MacroArg};
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{Access, AdtKind, Expr, ExprKind, IntTy, Intrinsic, Ty, Value, layout};

use super::body::Checker;
use super::consteval::EvalError;
use super::items::UnitCx;

/// The capabilities a target may have; `@target_has` names one of these.
pub const CAPABILITIES: [&str; 2] = ["dynamic_registers", "dynamic_lifting"];

/// The macros of quantum units.
pub const QUANTUM: [&str; 14] = [
    "node",
    "matrix_of",
    "judgment",
    "kernel",
    "certificate",
    "classify",
    "classify_op",
    "holonomy",
    "nerve",
    "transition_class",
    "single_patch",
    "ortho_graph",
    "bargmann",
    "fragment_of",
];

/// Every macro an expression may use in a classical unit.
pub const MACROS: [&str; 25] = [
    "sizeof",
    "alignof",
    "name_of",
    "kind_of",
    "is_quantum",
    "is_const",
    "field_count",
    "has_field",
    "field_type",
    "field_offset",
    "has_method",
    "method",
    "method_count",
    "variant_count",
    "static_assert",
    "target_has",
    "is_prefix_of",
    "aborts",
    "typeinfo",
    "typeinfo_of",
    "raw_parts",
    "from_raw_parts",
    "foreach_field",
    "foreach_method",
    "foreach_variant",
];

/// Why `U` is not a prefix of `T`.
pub enum NotPrefix {
    /// They are not both structures, nor both arrays.
    Shape,
    /// They have different layout annotations.
    Representation,
    /// `T` has fewer fields than `U`.
    Short,
    /// The field at this position differs.
    Field(Symbol),
    /// They are two instances of one generic type, with different
    /// arguments.
    Instance,
}

impl<'a> UnitCx<'a> {
    /// Whether `u` is a prefix of `t`: both structures, with the fields of
    /// `u` the first fields of `t`, the same in name and type, and the same
    /// layout annotations; or both fixed arrays of one length whose elements
    /// are. Every type is a prefix of itself, and no instance of a generic
    /// type is a prefix of another instance of it.
    pub fn prefix_of(&mut self, u: Ty, t: Ty) -> Result<(), NotPrefix> {
        if u == t {
            return Ok(());
        }
        match (u, t) {
            (Ty::Adt(a), Ty::Adt(b)) => {
                self.adt(a);
                self.adt(b);
                let (da, db) = (self.unit.types.adt(a).clone(), self.unit.types.adt(b).clone());
                let (AdtKind::Struct { fields: fa }, AdtKind::Struct { fields: fb }) = (&da.kind, &db.kind) else {
                    return Err(NotPrefix::Shape);
                };
                if da.repr != db.repr {
                    return Err(NotPrefix::Representation);
                }
                // a generic type's arguments are part of what its values
                // mean, even where no field shows them: the coefficients of
                // a `cyclo<3>` are a different number as a `cyclo<4>`
                if !da.args.is_empty() && da.name == db.name && da.origin == db.origin {
                    return Err(NotPrefix::Instance);
                }
                for (i, f) in fa.iter().enumerate() {
                    let Some(g) = fb.get(i) else {
                        return Err(NotPrefix::Short);
                    };
                    if f.name != g.name || f.ty != g.ty {
                        return Err(NotPrefix::Field(f.name));
                    }
                }
                Ok(())
            }
            (Ty::Array(_), Ty::Array(_)) => {
                let (Some((eu, nu)), Some((et, nt))) = (self.unit.types.as_array(u), self.unit.types.as_array(t)) else {
                    return Err(NotPrefix::Shape);
                };
                // elements of different sizes would sit at different places
                if nu != nt || layout::of(&self.unit.types, eu).size != layout::of(&self.unit.types, et).size {
                    return Err(NotPrefix::Shape);
                }
                self.prefix_of(eu, et)
            }
            _ => Err(NotPrefix::Shape),
        }
    }

    /// Whether a value of `ty` has the operator or method `name` (with its
    /// `$` for an operator).
    pub fn has_method(&mut self, ty: Ty, name: &str) -> bool {
        let (operator, bare) = match name.strip_prefix('$') {
            Some(b) => (true, b),
            None => (false, name),
        };
        match ty {
            Ty::Adt(id) => {
                if operator && bare == "copy" && self.unit.types.is_copyable(ty) {
                    return true;
                }
                let Some(sym) = self.interner.get(bare) else { return false };
                self.method_of_type(id, sym, operator).is_some()
            }
            _ if !operator => matches!(ty, Ty::Array(_) | Ty::Slice(_) | Ty::Growable(_)) && bare == "len",
            _ => self.builtin_operator(ty, bare),
        }
    }

    /// Whether the language gives values of the primitive or compound type
    /// `ty` the operator `$name`.
    fn builtin_operator(&self, ty: Ty, name: &str) -> bool {
        const ARITH: [&str; 5] = ["add", "sub", "mul", "div", "rem"];
        const BITS: [&str; 5] = ["and", "or", "xor", "shl", "shr"];
        let assign = name.strip_suffix("_assign");
        let base = assign.unwrap_or(name);
        match ty {
            Ty::Int(_) => {
                ARITH.contains(&base)
                    || BITS.contains(&base)
                    || (assign.is_none() && ["neg", "not", "eq", "ord", "copy", "adj"].contains(&name))
            }
            Ty::Float(_) => ARITH.contains(&base) || (assign.is_none() && ["neg", "eq", "ord", "copy", "adj"].contains(&name)),
            Ty::Bool => ["and", "or", "xor"].contains(&base) || (assign.is_none() && ["not", "eq", "copy"].contains(&name)),
            Ty::Char => ["eq", "ord", "copy"].contains(&name),
            Ty::Ref(_) | Ty::Slice(_) => ["eq", "copy"].contains(&name) || (ty.is_slice() && ["index", "index_rd"].contains(&name)),
            Ty::Array(_) | Ty::Growable(_) => {
                ["index", "index_rd"].contains(&name) || (name == "copy" && self.unit.types.is_copyable(ty))
            }
            Ty::Fn(_) | Ty::Closure(_) => name == "call" || (name == "copy" && self.unit.types.is_copyable(ty)),
            _ => name == "copy" && self.unit.types.is_copyable(ty),
        }
    }

    /// `@field_type(T, "name")`, written where a type is.
    pub fn macro_type(&mut self, name: Spanned<Symbol>, args: &[Spanned<MacroArg>], span: Span) -> Result<Ty, Box<Diagnostic>> {
        let text = self.interner.resolve(name.node).to_owned();
        if text != "field_type" {
            return Err(Box::new(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("`@{text}` does not give a type"))
                    .at(span)
                    .with_help("the one macro that gives a type is `@field_type(T, \"field\")`"),
            ));
        }
        let (Some(t), Some(field)) = (args.first(), args.get(1)) else {
            return Err(Box::new(arity(&text, "a type and a field name", span)));
        };
        let MacroArg::Type(t) = &t.node else {
            return Err(Box::new(wants(&text, "a type", t.span)));
        };
        let MacroArg::Str(field) = &field.node else {
            return Err(Box::new(wants(&text, "a field name, in quotes", field.span)));
        };
        let ty = super::types::resolve(self, t)?.ty;
        let fields = self.fields_of_type(ty);
        let interner = self.interner;
        let wanted = interner.resolve(field.node);
        match fields.iter().find(|(n, _)| interner.resolve(*n) == wanted) {
            Some(&(_, t)) => Ok(t),
            None => Err(Box::new(no_field(&self.unit.types.display(ty, interner), wanted, field.span))),
        }
    }

    /// The fields of a structure, or the elements of a tuple named by
    /// position; nothing for any other type.
    pub fn fields_of_type(&mut self, ty: Ty) -> Vec<(Symbol, Ty)> {
        match ty {
            Ty::Adt(id) => {
                self.adt(id);
                match &self.unit.types.adt(id).kind {
                    AdtKind::Struct { fields } => fields.iter().map(|f| (f.name, f.ty)).collect(),
                    AdtKind::Enum { .. } => Vec::new(),
                }
            }
            Ty::Tuple(_) => {
                let elems = self.unit.types.as_tuple(ty).map(<[Ty]>::to_vec).unwrap_or_default();
                elems
                    .into_iter()
                    .enumerate()
                    .map(|(i, t)| (self.interner.intern_late(&i.to_string()), t))
                    .collect()
            }
            _ => Vec::new(),
        }
    }
}

impl Checker<'_, '_> {
    /// `@name(args)` in an expression.
    pub(super) fn macro_call(&mut self, name: Spanned<Symbol>, args: &[Spanned<MacroArg>], span: Span) -> Expr {
        let text = self.name(name.node);
        match self.expand(text, args, span) {
            Some(e) => e,
            None => Checker::error(span),
        }
    }

    fn expand(&mut self, text: &str, args: &[Spanned<MacroArg>], span: Span) -> Option<Expr> {
        let usize_ = |v: u64| Expr::constant(Value::Int(i128::from(v), IntTy::USIZE), Ty::USIZE, span);
        let boolean = |b: bool| Expr::constant(Value::Bool(b), Ty::Bool, span);
        Some(match text {
            "sizeof" | "alignof" => {
                let ty = self.one_type(text, args, span)?;
                if !self.cx.layout_ready(ty, span) {
                    return None;
                }
                let l = layout::of(self.types(), ty);
                usize_(if text == "sizeof" { l.size } else { l.align })
            }
            "name_of" => {
                let ty = self.one_type(text, args, span)?;
                let name = self.types().qualified(ty, self.interner, &self.cx.unit.name);
                let sym = self.interner.intern_late(&name);
                let string = self.types_mut().string();
                Expr::constant(Value::Str(sym), string, span)
            }
            "kind_of" => {
                let ty = self.one_type(text, args, span)?;
                self.kind_of(ty, span)?
            }
            "is_quantum" => {
                let ty = self.one_type(text, args, span)?;
                boolean(self.types().is_quantum(ty))
            }
            "is_const" => {
                let [MacroArg::Type(t)] = args.iter().map(|a| &a.node).collect::<Vec<_>>()[..] else {
                    self.report(arity(text, "one type", span));
                    return None;
                };
                boolean(self.resolve_type(t).constant)
            }
            "field_count" => {
                let ty = self.one_type(text, args, span)?;
                let n = match ty {
                    Ty::Adt(id) if self.adt(id).is_struct() => self.adt(id).fields().len(),
                    Ty::Tuple(_) => self.types().as_tuple(ty).map_or(0, <[Ty]>::len),
                    _ => {
                        let what = self.describe(ty);
                        self.report(no_fields(text, &what, span));
                        return None;
                    }
                };
                usize_(n as u64)
            }
            "variant_count" => {
                let ty = self.one_type(text, args, span)?;
                match ty {
                    Ty::Adt(id) if !self.adt(id).is_struct() => usize_(self.adt(id).variants().len() as u64),
                    _ => {
                        let what = self.describe(ty);
                        self.report(
                            Diagnostic::new(Code::Es06)
                                .with_message(format!("`@variant_count` asks about an enumeration, but this is {what}"))
                                .at(span),
                        );
                        return None;
                    }
                }
            }
            "method_count" => {
                let ty = self.one_type(text, args, span)?;
                let Ty::Adt(id) = ty else {
                    return Some(usize_(0));
                };
                usize_(self.cx.method_names(id).len() as u64)
            }
            "has_field" | "field_offset" => {
                let (ty, field, at) = self.type_and_name(text, args, span)?;
                let fields = self.cx.fields_of_type(ty);
                let index = fields.iter().position(|(n, _)| self.name(*n) == field);
                if text == "has_field" {
                    return Some(boolean(index.is_some()));
                }
                let Some(i) = index else {
                    let shown = self.show(ty);
                    self.report(no_field(&shown, &field, at));
                    return None;
                };
                if !self.cx.layout_ready(ty, span) {
                    return None;
                }
                usize_(layout::fields_of(self.types(), ty).offsets[i])
            }
            "field_type" => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message("`@field_type` gives a type, and this is where a value is wanted")
                        .at(span)
                        .with_help("write it where a type goes, as in `let x: @field_type(T, \"f\") = …`"),
                );
                return None;
            }
            "has_method" => {
                let (ty, method, _) = self.type_and_name(text, args, span)?;
                boolean(self.cx.has_method(ty, &method))
            }
            "method" => {
                let (ty, method, at) = self.type_and_name(text, args, span)?;
                return self.method_value(ty, &method, at, span);
            }
            "static_assert" => {
                let (Some(cond), msg) = (args.first(), args.get(1)) else {
                    self.report(arity(text, "a condition and a message", span));
                    return None;
                };
                let Some(cond) = as_expr(cond) else {
                    self.report(wants(text, "a condition", cond.span));
                    return None;
                };
                let cond = &cond;
                let message = match msg.map(|m| &m.node) {
                    Some(MacroArg::Str(s)) => self.name(s.node).to_owned(),
                    None => "the assertion is false".to_owned(),
                    Some(_) => {
                        self.report(wants(text, "a message in quotes, second", msg.map_or(span, |m| m.span)));
                        return None;
                    }
                };
                let role = "the condition of `@static_assert`";
                let before = self.cx.diags.len();
                let checked = super::body::check_const_as(self.cx, cond, Some(Ty::Bool), role);
                if checked.ty == Ty::Never || self.cx.diags[before..].iter().any(Diagnostic::is_error) {
                    return None;
                }
                // a condition reading a judgement is checked once the
                // judgements exist, after the unit's circuits are generated
                if reads_judgment(&checked) {
                    self.cx.unit.deferred.push(crate::tir::claims::Deferred {
                        cond: checked,
                        message,
                        span: cond.span,
                    });
                    return Some(Expr::constant(Value::Void, Ty::Void, span));
                }
                match self.cx.evaluate_checked(&checked, role)? {
                    Value::Bool(true) => {}
                    _ => self.report(
                        Diagnostic::new(Code::Em01)
                            .with_message(message)
                            .at(cond.span)
                            .with_note("`@static_assert` makes a program ill-formed when its condition is false"),
                    ),
                }
                Expr::constant(Value::Void, Ty::Void, span)
            }
            "target_has" => {
                let [a] = args else {
                    self.report(arity(text, "one capability name", span));
                    return None;
                };
                let MacroArg::Str(s) = &a.node else {
                    self.report(wants(text, "a capability name in quotes", a.span));
                    return None;
                };
                let name = self.name(s.node);
                if !CAPABILITIES.contains(&name) {
                    let mut d = Diagnostic::new(Code::Em04)
                        .with_message(format!("`{name}` is not a capability a target can have"))
                        .at(a.span)
                        .with_note(format!("the capabilities are `{}` and `{}`", CAPABILITIES[0], CAPABILITIES[1]));
                    if let Some(c) = super::report::closest(name, CAPABILITIES) {
                        d = d.with_help(format!("did you mean `{c}`?"));
                    }
                    self.report(d);
                    return None;
                }
                // a program's own processor, the exact simulator, has both;
                // a circuit archive is for a target running static circuits,
                // which has neither
                boolean(!self.cx.static_circuits)
            }
            "is_prefix_of" => {
                let (Some(u), Some(t)) = (args.first(), args.get(1)) else {
                    self.report(arity(text, "two types", span));
                    return None;
                };
                let (MacroArg::Type(u), MacroArg::Type(t)) = (&u.node, &t.node) else {
                    self.report(wants(text, "two types", span));
                    return None;
                };
                let (u, t) = (self.resolve_type(u).ty, self.resolve_type(t).ty);
                boolean(self.cx.prefix_of(u, t).is_ok())
            }
            "aborts" => {
                let (e, _) = self.one_expr(text, "expression", args, span)?;
                match self.cx.evaluate_const(&e, None, "the expression of `@aborts`") {
                    Ok(_) => boolean(false),
                    Err(Some(EvalError::Abort { .. })) => boolean(true),
                    Err(None) => return None,
                    Err(Some(err)) => {
                        let d = super::fold::describe(
                            err,
                            &super::fold::Need::Constant("the expression of `@aborts`"),
                            &self.cx.unit,
                            self.interner,
                        );
                        self.report(d);
                        return None;
                    }
                }
            }
            "foreach_field" | "foreach_method" | "foreach_variant" => return self.foreach(text, args, span),
            "typeinfo" => {
                let ty = self.one_type(text, args, span)?;
                if !self.cx.need_table(ty, span) {
                    return None;
                }
                let info = self.cx.unit.typeinfo?.info;
                let result = self.types_mut().reference(Access::Const, Ty::Adt(info));
                Expr::intrinsic(Intrinsic::TypeInfo(ty), Vec::new(), result, span)
            }
            "typeinfo_of" => {
                let (e, at) = self.one_expr(text, "reference", args, span)?;
                let r = self.expr(&e);
                let t = self.settled(r.ty);
                if t == Ty::Never {
                    return None;
                }
                let Some((_, referent)) = self.types().as_ref(t) else {
                    let what = self.describe(t);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`@typeinfo_of` reads the type a reference carries, but this is {what}"))
                            .at(at)
                            .with_help("for a type written in the program, `@typeinfo(T)` gives its table"),
                    );
                    return None;
                };
                let result = self.table_ref(referent, span)?;
                Expr::intrinsic(Intrinsic::TypeInfoOf, vec![r], result, span)
            }
            "raw_parts" => {
                let (e, at) = self.one_expr(text, "reference", args, span)?;
                let r = self.expr(&e);
                let t = self.settled(r.ty);
                if t == Ty::Never {
                    return None;
                }
                let second = if self.types().as_slice(t).is_some() {
                    Ty::USIZE
                } else if let Some((_, referent)) = self.types().as_ref(t) {
                    self.table_ref(referent, span)?
                } else {
                    let what = self.describe(t);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`@raw_parts` takes a reference apart, but this is {what}"))
                            .at(at)
                            .with_note("a reference to a slice is its address and its length; any other reference is its address and the table of what it refers to")
                            .with_help("take a reference to the value with `&`"),
                    );
                    return None;
                };
                let ty = self.types_mut().tuple(vec![Ty::USIZE, second]);
                Expr::intrinsic(Intrinsic::RawParts, vec![r], ty, span)
            }
            "from_raw_parts" => {
                let [a, n] = args else {
                    self.report(arity(text, "an address and a length or table", span));
                    return None;
                };
                let (Some(a_e), Some(n_e)) = (as_expr(a), as_expr(n)) else {
                    self.report(wants(text, "an address and a length or table", span));
                    return None;
                };
                let addr = self.expr(&a_e);
                let addr = self.coerce(addr, Ty::USIZE, "the address");
                let second = self.expr(&n_e);
                let st = self.settled(second.ty);
                // what is made is left to the context: a slice from a
                // length, a reference from a table
                let referent = self.infer.fresh_any();
                let (second, ty) = if self.types().as_ref(st).is_some() {
                    let table = self.table_type(span)?;
                    let second = self.coerce(second, table, "the table");
                    (second, self.types_mut().reference(Access::Write, referent))
                } else {
                    let second = self.coerce(second, Ty::USIZE, "the length");
                    (second, self.types_mut().slice(Access::Write, referent))
                };
                Expr::intrinsic(Intrinsic::FromRawParts, vec![addr, second], ty, span)
            }
            "node" if self.cx.quantum => {
                let [MacroArg::Expr(k)] = args.iter().map(|a| &a.node).collect::<Vec<_>>()[..] else {
                    self.report(arity(text, "one node number", span));
                    return None;
                };
                let k = self.expr_for(k, Ty::USIZE);
                let k = self.coerce(k, Ty::USIZE, "the node's number");
                let ty = self.types_mut().reference(Access::Write, Ty::Qubit);
                Expr::intrinsic(Intrinsic::Node, vec![k], ty, span)
            }
            q if QUANTUM.contains(&q) && self.cx.quantum => {
                if let Some(e) = self.geometry_macro(q, args, span) {
                    return Some(e);
                }
                if let Some(e) = self.judgment_macro(q, args, span) {
                    return Some(e);
                }
                return Some(self.unsupported(span, &format!("the `@{q}` macro")));
            }
            q if QUANTUM.contains(&q) => {
                self.report(
                    Diagnostic::new(Code::Eu02)
                        .with_message(format!("`@{q}` asks about quantum operators or states, which a classical unit has none of"))
                        .at(span)
                        .with_help("use it in a quantum unit"),
                );
                return None;
            }
            other => return self.later_macro(other, args, span),
        })
    }

    /// `@embed` where nothing says its type, and the macros that do not
    /// exist.
    fn later_macro(&mut self, text: &str, args: &[Spanned<MacroArg>], span: Span) -> Option<Expr> {
        if text == "embed" {
            return Some(self.embed(args, None, span));
        }
        if MACROS.contains(&text) {
            return Some(self.unsupported(span, &format!("the `@{text}` macro")));
        }
        let mut d = Diagnostic::new(Code::Es04)
            .with_message(format!("there is no builtin macro named `@{text}`"))
            .at(span)
            .with_note("names beginning with `@` are the language's own macros; a program cannot declare one");
        if let Some(s) = super::report::closest(text, MACROS) {
            d = d.with_help(format!("did you mean `@{s}`?"));
        }
        self.report(d);
        None
    }

    /// `@foreach_field(T, M)`, `@foreach_method(T, M)` and
    /// `@foreach_variant(T, M)`: the preprocessor macro `M` expanded once for
    /// each member, as `M(name, "name", Type, index)`, and the expansions
    /// read as the statements of a block.
    ///
    /// A field's type is its own; a method's is its function type; a
    /// variant's is the tuple of what it carries.
    fn foreach(&mut self, text: &str, args: &[Spanned<MacroArg>], span: Span) -> Option<Expr> {
        use crate::lex::{Punct, Token};
        use crate::lex::token::{IntBase, StrKind};
        let [t, m] = args else {
            self.report(arity(text, "a type and a macro name", span));
            return None;
        };
        let MacroArg::Type(t) = &t.node else {
            self.report(wants(text, "a type first", t.span));
            return None;
        };
        let macro_name = match &m.node {
            MacroArg::Type(Spanned {
                node: ast::Type::Path { path, args },
                ..
            }) if args.is_empty() && path.is_simple() => path.last(),
            _ => None,
        };
        let Some(macro_name) = macro_name else {
            self.report(wants(text, "the name of a macro `#define`d with four parameters, second", m.span));
            return None;
        };
        let takes_four = self
            .cx
            .macros
            .and_then(|table| table.get(macro_name))
            .is_some_and(|d| d.accepts(4));
        if !takes_four {
            let name = self.name(macro_name);
            self.report(
                Diagnostic::new(Code::Es04)
                    .with_message(format!("`{name}` is not a macro taking four arguments"))
                    .at(m.span)
                    .with_note(format!(
                        "`@{text}` expands the macro once per member as `{name}(name, \"name\", Type, index)`"
                    ))
                    .with_help(format!("define it: `#define {name}(NAME, TEXT, TYPE, INDEX) …`")),
            );
            return None;
        }
        let ty = self.resolve_type(t).ty;
        if ty == Ty::Never {
            return None;
        }
        let members = self.members(text, ty, span)?;

        // `M(name, "name", Type, index)` for each member, as tokens
        let at = |tok: Token| crate::pp::PpToken::new(tok, span);
        let mut call = Vec::new();
        for (i, (name, member_ty)) in members.iter().enumerate() {
            call.push(at(Token::Ident(macro_name)));
            call.push(at(Token::Punct(Punct::LParen)));
            call.push(at(Token::Ident(*name)));
            call.push(at(Token::Punct(Punct::Comma)));
            call.push(at(Token::Str {
                value: *name,
                kind: StrKind::Normal,
            }));
            call.push(at(Token::Punct(Punct::Comma)));
            let shown = self.show(*member_ty);
            let lexed = crate::lex::lex(
                crate::span::SourceId::SYNTHETIC,
                &crate::source::Spliced::from_text(&shown),
                self.interner,
            );
            call.extend(lexed.tokens.into_iter().map(|(tok, _)| at(tok)));
            call.push(at(Token::Punct(Punct::Comma)));
            call.push(at(Token::Int {
                raw: self.interner.intern_late(&i.to_string()),
                base: IntBase::Decimal,
                suffix: None,
            }));
            call.push(at(Token::Punct(Punct::RParen)));
        }
        let mut table = self.cx.macros.cloned().unwrap_or_default();
        let sources = crate::source::SourceMap::new();
        let mut diags = Vec::new();
        let expanded = crate::pp::Expander::new(&mut table, self.interner, &sources, &mut diags, 8).expand(call);

        // read as the body of a function, whose block is then checked here
        let mut tokens = vec![
            (Token::Kw(crate::lex::Keyword::Fn), span),
            (Token::Ident(self.interner.intern_late("__foreach")), span),
            (Token::Punct(Punct::LParen), span),
            (Token::Punct(Punct::RParen), span),
            (Token::Punct(Punct::LBrace), span),
        ];
        tokens.extend(expanded.into_iter().map(|p| (p.tok, span)));
        tokens.push((Token::Punct(Punct::RBrace), span));
        let marked = crate::mark::mark(&tokens);
        diags.extend(marked.diagnostics);
        let parsed = crate::parse::parse_unit(
            crate::parse::Cx::new(self.interner),
            Some(crate::pp::UnitKind::Classical),
            &marked.tokens,
            span,
        );
        diags.extend(parsed.diagnostics);
        if diags.iter().any(Diagnostic::is_error) {
            for d in diags {
                self.report(d.with_note(format!("in the expansion of `@{text}`, which is written here")));
            }
            return None;
        }
        let body = parsed.unit.items.into_iter().find_map(|item| match item.node.kind {
            ast::ItemKind::Fn(f) => Some(f.body),
            _ => None,
        })?;
        let b = self.block(&body, span);
        Some(Expr {
            ty: b.ty,
            kind: ExprKind::Block(Box::new(b)),
            span,
        })
    }

    /// The members `@foreach_…` goes through: each one's name and type.
    fn members(&mut self, text: &str, ty: Ty, span: Span) -> Option<Vec<(Symbol, Ty)>> {
        match text {
            "foreach_field" => {
                if !matches!(ty, Ty::Tuple(_)) && !matches!(ty, Ty::Adt(id) if self.adt(id).is_struct()) {
                    let what = self.describe(ty);
                    self.report(no_fields(text, &what, span));
                    return None;
                }
                Some(self.cx.fields_of_type(ty))
            }
            "foreach_variant" => {
                let Ty::Adt(id) = ty else {
                    let what = self.describe(ty);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`@{text}` goes through an enumeration's variants, but this is {what}"))
                            .at(span),
                    );
                    return None;
                };
                let variants = self.adt(id).variants().to_vec();
                Some(
                    variants
                        .into_iter()
                        .map(|v| {
                            let carried = v.fields.iter().map(|f| f.ty).collect();
                            (v.name, self.types_mut().tuple(carried))
                        })
                        .collect(),
                )
            }
            _ => {
                let Ty::Adt(id) = ty else {
                    return Some(Vec::new());
                };
                let mut out = Vec::new();
                for name in self.cx.method_names(id) {
                    // a generic method has no one type to name
                    let Some(callee) = self.cx.method_of_type(id, name, false).and_then(|e| e.target.callee()) else {
                        continue;
                    };
                    let (params, ret) = self.cx.signature(callee);
                    out.push((name, self.types_mut().function(params, ret)));
                }
                // declaration order is not kept for methods; name order is at
                // least the same every time
                let interner = self.interner;
                out.sort_by(|a, b| interner.resolve(a.0).cmp(interner.resolve(b.0)));
                Some(out)
            }
        }
    }

    /// `@kind_of(T)`: the `TypeKind` of `ty`.
    fn kind_of(&mut self, ty: Ty, span: Span) -> Option<Expr> {
        let kind = self.types().kind_name(ty);
        let Some(id) = self.interner.get("TypeKind").and_then(|s| self.cx.lookup_type(s)) else {
            return Some(self.unsupported(span, "`@kind_of` without the `core` library"));
        };
        let interner = self.interner;
        let variant = self
            .adt(id)
            .variants()
            .iter()
            .position(|v| interner.resolve(v.name) == kind)
            .expect("`core` names every kind") as u32;
        Some(Expr::constant(
            Value::Enum {
                variant,
                fields: Vec::new(),
            },
            Ty::Adt(id),
            span,
        ))
    }

    /// `@method(T, "name")`: the method as a function value, called as
    /// `@method(T, "name")(args)`.
    fn method_value(&mut self, ty: Ty, method: &str, at: Span, span: Span) -> Option<Expr> {
        let Ty::Adt(id) = ty else {
            let what = self.describe(ty);
            self.report(
                Diagnostic::new(Code::Es04)
                    .with_message(format!("{what} has no method `{method}` that can be named as a value"))
                    .at(at),
            );
            return None;
        };

        // the method, `$` naming an operator method
        let (operator, bare) = match method.strip_prefix('$') {
            Some(b) => (true, b),
            None => (false, method),
        };
        let entry = self
            .interner
            .get(bare)
            .and_then(|sym| self.cx.method_of_type(id, sym, operator));
        let Some(entry) = entry else {
            let ty_name = self.show(ty);
            self.report(
                Diagnostic::new(Code::Es04)
                    .with_message(format!("`{ty_name}` has no method named `{method}`"))
                    .at(at)
                    .with_note("`@method` names a method that exists; ask first with `@has_method`"),
            );
            return None;
        };

        // a function with one type, which a generic method is not until called
        let Some(callee) = entry.target.callee() else {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!("`{method}` is generic, so it is only a value once its arguments are known"))
                    .at(at)
                    .with_help("call it on a value, as in `v.method(…)`, and its arguments are worked out"),
            );
            return None;
        };
        let path = ast::Path {
            segments: vec![Spanned::new(self.interner.intern_late(method), at)],
        };
        Some(self.function_value(callee, &path, span))
    }

    /// The one expression a macro takes, a `noun` such as "reference", and
    /// where it is written.
    fn one_expr(&mut self, text: &str, noun: &str, args: &[Spanned<MacroArg>], span: Span) -> Option<(Spanned<ast::Expr>, Span)> {
        let [a] = args else {
            self.report(arity(text, &format!("one {noun}"), span));
            return None;
        };
        let Some(e) = as_expr(a) else {
            let article = if noun.starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" };
            self.report(wants(text, &format!("{article} {noun}"), a.span));
            return None;
        };
        Some((e, a.span))
    }

    /// The type of a reference to a type's table.
    fn table_type(&mut self, span: Span) -> Option<Ty> {
        let info = self.cx.typeinfo_types(span)?.info;
        Some(self.types_mut().reference(Access::Const, Ty::Adt(info)))
    }

    /// The type of the table a reference to a `referent` carries. Where
    /// the referent's type is written, not `void`, that type's table must
    /// then exist.
    fn table_ref(&mut self, referent: Ty, span: Span) -> Option<Ty> {
        if referent != Ty::Void && !self.cx.need_table(referent, span) {
            return None;
        }
        self.table_type(span)
    }

    /// The one type a macro takes.
    fn one_type(&mut self, text: &str, args: &[Spanned<MacroArg>], span: Span) -> Option<Ty> {
        let [a] = args else {
            self.report(arity(text, "one type", span));
            return None;
        };
        let MacroArg::Type(t) = &a.node else {
            self.report(wants(text, "a type", a.span));
            return None;
        };
        let ty = self.resolve_type(t).ty;
        (ty != Ty::Never).then_some(ty)
    }

    /// A type and a name in quotes.
    fn type_and_name(&mut self, text: &str, args: &[Spanned<MacroArg>], span: Span) -> Option<(Ty, String, Span)> {
        let [t, n] = args else {
            self.report(arity(text, "a type and a name in quotes", span));
            return None;
        };
        let MacroArg::Type(t) = &t.node else {
            self.report(wants(text, "a type first", t.span));
            return None;
        };
        let MacroArg::Str(s) = &n.node else {
            self.report(wants(text, "a name in quotes second", n.span));
            return None;
        };
        let ty = self.resolve_type(t).ty;
        (ty != Ty::Never).then(|| (ty, self.name(s.node).to_owned(), n.span))
    }
}

/// An argument read as an expression. A lone name, or a macro, is parsed as a
/// type when it stands alone, since it could be one; where an expression is
/// wanted it is read as one.
fn as_expr(arg: &Spanned<MacroArg>) -> Option<Spanned<ast::Expr>> {
    match &arg.node {
        MacroArg::Expr(e) => Some(e.clone()),
        MacroArg::Type(t) => match &t.node {
            ast::Type::Macro { name, args } => Some(Spanned::new(
                ast::Expr::Macro {
                    name: *name,
                    args: args.clone(),
                },
                t.span,
            )),
            ast::Type::Path { path, args } if args.is_empty() => Some(Spanned::new(
                ast::Expr::Path {
                    path: path.clone(),
                    args: Vec::new(),
                },
                t.span,
            )),
            _ => None,
        },
        MacroArg::Str(_) | MacroArg::Quon(_) => None,
    }
}

fn arity(text: &str, wanted: &str, span: Span) -> Diagnostic {
    Diagnostic::new(Code::Es07)
        .with_message(format!("`@{text}` takes {wanted}"))
        .at(span)
}

fn wants(text: &str, wanted: &str, span: Span) -> Diagnostic {
    Diagnostic::new(Code::Es06)
        .with_message(format!("`@{text}` takes {wanted} here"))
        .at(span)
}

fn no_field(ty: &str, field: &str, span: Span) -> Diagnostic {
    Diagnostic::new(Code::Es04)
        .with_message(format!("`{ty}` has no field named `{field}`"))
        .at(span)
        .with_note("ask first with `@has_field`, which answers either way")
}

fn no_fields(text: &str, what: &str, span: Span) -> Diagnostic {
    Diagnostic::new(Code::Es06)
        .with_message(format!("`@{text}` asks about a structure or tuple, but this is {what}"))
        .at(span)
}

/// Whether `e` reads a value derived from a judgement.
fn reads_judgment(e: &Expr) -> bool {
    struct Find(bool);
    impl crate::tir::visit::Visit for Find {
        fn expr(&mut self, e: &Expr) {
            if matches!(
                e.kind,
                ExprKind::Intrinsic {
                    which: crate::tir::Intrinsic::Derived(_),
                    ..
                }
            ) {
                self.0 = true;
            }
            crate::tir::visit::walk_expr(self, e);
        }
    }
    let mut f = Find(false);
    crate::tir::visit::Visit::expr(&mut f, e);
    f.0
}
