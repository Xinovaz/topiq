//! Which run-time type information a unit makes.
//!
//! Every type has a table, `TypeInfo`, describing it: its name, size and
//! kind, its fields, methods and variants. A table is made only where
//! something needs it: `@typeinfo(T)`, a reference erased to `*any`, a
//! downcast to `*T`, or `[typeinfo]` written on the type. A table mentions the
//! tables of the types it describes (a field's type, a method's signature)
//! so those come with it.
//!
//! Each unit that needs a table makes its own copy, and the linker keeps one,
//! so that a table's address is its type's identity within a program. A
//! module loaded while it runs has its own tables, and a type of one name
//! in both is the same type. A reference carries its
//! referent's table as its second word, whichever unit made it; where no unit
//! of the program made one, that word is a blank table.
//!
//! `[no_typeinfo]` forbids a table for its type: asking for one is `EM03`.

use std::collections::HashSet;

use crate::ast::{AnnotationGroup, StdAnnotation};
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{self, AdtId, AdtKind, Callee, TableMethod, Ty, TypeInfoTypes};

use super::items::UnitCx;
use super::method::MethodTarget;

/// What `[tcon: …]` says of a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TconForm {
    /// A document may give fields the type does not have, which are skipped.
    Open,
    /// The type is kept out of documents, and out of `clone`.
    Opaque,
}

/// An annotation argument that is a single name, as `open` is.
fn single_name(arg: &crate::ast::AnnArg) -> Option<Symbol> {
    match arg {
        crate::ast::AnnArg::Expr(e) => match &e.node {
            crate::ast::Expr::Path { path, args } if args.is_empty() && path.is_simple() => path.last(),
            _ => None,
        },
        crate::ast::AnnArg::Type(t) => match &t.node {
            crate::ast::Type::Path { path, args } if args.is_empty() && path.is_simple() => path.last(),
            _ => None,
        },
        _ => None,
    }
}

impl UnitCx<'_> {
    /// `ET02`: `tcon::emit`, `tcon::parse`, `tcon::bind` or `str::parse`
    /// instantiated with a type that has no written form. `false` when so,
    /// after reporting it where it was asked for.
    pub(super) fn serializable_instance(&mut self, key: super::items::GenericKey, args: &[tir::Arg], span: Span) -> bool {
        let origin = self.generic_origin(key);
        let name = self.interner.resolve(key.name);
        let reads = match (origin.as_deref(), name) {
            (Some("tcon"), "emit") => false,
            (Some("tcon"), "parse" | "bind") | (Some("str"), "parse") => true,
            _ => return true,
        };
        let Some(tir::Arg::Type(t)) = args.first() else { return true };
        let Some((part, why)) = self.unwritable(*t, String::new(), reads, &mut Vec::new()) else {
            return true;
        };
        let ty = self.unit.types.display(*t, self.interner);
        let label = self.generic_label(key);
        let doing = if reads { "read" } else { "write" };
        let message = if part.is_empty() {
            format!("`{label}` cannot {doing} a `{ty}`: it is {why}")
        } else {
            format!("`{label}` cannot {doing} a `{ty}`: its part `{part}` is {why}")
        };
        let help = if reads {
            format!("give `{ty}` a method `from_tcon(self: *{ty}, v: *const tcon::Value) -> Result<(), tcon::Error>`, which is used instead")
        } else {
            format!("give `{ty}` a method `to_tcon(self: *const {ty}, w: *tcon::Writer)`, which is used instead")
        };
        self.report(
            Diagnostic::new(Code::Et02)
                .with_message(message)
                .at(span)
                .with_note(
                    "a document holds values: numbers, characters, text, and structures, arrays and \
                     variants of them. A reference, a function or a closure is part of the running \
                     program, and a type marked `[tcon: opaque]` has asked to be left out",
                )
                .with_help(help),
        );
        false
    }

    /// The first part of a `ty` that has no written form, as a path from
    /// `at`, and what it is. A type with its own way of being written (or,
    /// when `reads`, of being read) is taken as it is.
    fn unwritable(&mut self, ty: Ty, at: String, reads: bool, seen: &mut Vec<AdtId>) -> Option<(String, &'static str)> {
        let join = |at: &str, part: &str| if at.is_empty() { part.to_owned() } else { format!("{at}.{part}") };
        match ty {
            Ty::Ref(_) | Ty::Slice(_) => Some((at, "a reference, an address in the running program")),
            Ty::Fn(_) | Ty::Closure(_) | Ty::Circuit(_) => Some((at, "code, which has no written form")),
            Ty::Dyn if reads => Some((at, "a `dyn`, which does not say what type to read it as")),
            Ty::Adt(id) => {
                self.adt(id);
                let own = if reads {
                    self.has_method(ty, "from_tcon")
                } else {
                    self.has_method(ty, "to_tcon") || self.has_method(ty, "$fmt")
                };
                if own {
                    return None;
                }
                if self.unit.types.adt(id).opaque {
                    return Some((at, "marked `[tcon: opaque]`"));
                }
                if seen.contains(&id) {
                    return None;
                }
                seen.push(id);
                let def = self.unit.types.adt(id).clone();
                let mut found = None;
                match &def.kind {
                    AdtKind::Struct { fields } => {
                        for f in fields {
                            let part = join(&at, self.interner.resolve(f.name));
                            if let Some(hit) = self.unwritable(f.ty, part, reads, seen) {
                                found = Some(hit);
                                break;
                            }
                        }
                    }
                    AdtKind::Enum { variants } => {
                        'variants: for v in variants {
                            let vname = self.interner.resolve(v.name);
                            for (i, f) in v.fields.iter().enumerate() {
                                let field = if f.name == Symbol::EMPTY {
                                    i.to_string()
                                } else {
                                    self.interner.resolve(f.name).to_owned()
                                };
                                let part = join(&at, &format!("{vname}.{field}"));
                                if let Some(hit) = self.unwritable(f.ty, part, reads, seen) {
                                    found = Some(hit);
                                    break 'variants;
                                }
                            }
                        }
                    }
                }
                seen.pop();
                found
            }
            Ty::Array(_) | Ty::Growable(_) => {
                let types = &self.unit.types;
                let elem = types.as_growable(ty).or_else(|| types.as_array(ty).map(|(e, _)| e))?;
                self.unwritable(elem, format!("{at}[…]"), reads, seen)
            }
            Ty::Tuple(_) => {
                let elems = self.unit.types.as_tuple(ty).map(<[Ty]>::to_vec).unwrap_or_default();
                for (i, e) in elems.into_iter().enumerate() {
                    if let Some(hit) = self.unwritable(e, join(&at, &i.to_string()), reads, seen) {
                        return Some(hit);
                    }
                }
                None
            }
            _ => None,
        }
    }
}

impl<'a> UnitCx<'a> {
    /// Records what is written on a type this unit declares: its annotations,
    /// which its table lists, and whether it asks for a table or forbids one.
    pub(super) fn note_type_annotations(&mut self, id: AdtId, groups: &[Spanned<AnnotationGroup>]) {
        let mut names = Vec::new();
        for g in groups {
            for a in &g.node.annotations {
                let name = a.node.name.node;
                let value = match a.node.args.as_slice() {
                    [arg] => single_name(&arg.node),
                    _ => None,
                };
                names.push((name, value));
                match StdAnnotation::from_name(self.interner.resolve(name)) {
                    Some(StdAnnotation::Typeinfo) => {
                        self.typeinfo_policy.insert(id, true);
                    }
                    Some(StdAnnotation::NoTypeinfo) => {
                        self.typeinfo_policy.insert(id, false);
                    }
                    Some(StdAnnotation::Tcon) => match value.map(|v| self.interner.resolve(v)) {
                        Some("open") => {
                            self.tcon_forms.insert(id, TconForm::Open);
                        }
                        Some("opaque") => {
                            self.tcon_forms.insert(id, TconForm::Opaque);
                            self.unit.types.adt_mut(id).opaque = true;
                        }
                        _ => self.report(
                            Diagnostic::new(Code::Es06)
                                .with_message("`[tcon: …]` takes `open` or `opaque`")
                                .at(a.span)
                                .with_note(
                                    "`open` lets a document give fields the type does not have, which \
                                     are skipped; `opaque` keeps the type out of documents altogether",
                                )
                                .with_help("write `[tcon: open]` or `[tcon: opaque]`"),
                        ),
                    },
                    _ => {}
                }
            }
        }
        self.annotation_names.insert(id, names);
    }

    /// Makes this unit make `ty`'s table. `false`, and `EM03` reported, when
    /// the type forbids one.
    pub fn need_table(&mut self, ty: Ty, span: Span) -> bool {
        if let Ty::Adt(id) = ty
            && self.typeinfo_policy.get(&id) == Some(&false)
        {
            let name = self.unit.types.adt_name(id, self.interner);
            self.report(
                Diagnostic::new(Code::Em03)
                    .with_message(format!("`{name}` is marked `[no_typeinfo]`, so it has no run-time type information"))
                    .at(span)
                    .with_note(
                        "its table is what `@typeinfo`, `*any` and downcasts use; the macros that \
                         ask about the type during translation, such as `@has_field`, still work",
                    )
                    .with_help("remove `[no_typeinfo]`, or ask about the type with those macros"),
            );
            return false;
        }
        if self.typeinfo_types(span).is_none() {
            return false;
        }
        if !self.unit.tables.contains(&ty) {
            self.unit.tables.push(ty);
        }
        true
    }

    /// The `core` library's types a table is made of.
    pub fn typeinfo_types(&mut self, span: Span) -> Option<TypeInfoTypes> {
        if let Some(t) = self.unit.typeinfo {
            return Some(t);
        }
        let mut find = |name: &str| self.interner.get(name).and_then(|s| self.lookup_type(s));
        let found = (|| {
            Some(TypeInfoTypes {
                info: find("TypeInfo")?,
                field: find("FieldInfo")?,
                method: find("MethodInfo")?,
                variant: find("VariantInfo")?,
                annot: find("AnnotInfo")?,
                kind: find("TypeKind")?,
            })
        })();
        match found {
            Some(t) => {
                for id in [t.info, t.field, t.method, t.variant, t.annot, t.kind] {
                    self.adt(id);
                }
                self.unit.typeinfo = Some(t);
                Some(t)
            }
            None => {
                self.report(super::report::unsupported(
                    span,
                    "run-time type information without the `core` library",
                ));
                None
            }
        }
    }

    /// Settles every table the unit makes: the types asked for, those marked
    /// `[typeinfo]`, and every type their tables mention, with the methods
    /// and annotations each structure's and enumeration's table lists.
    pub fn finish_tables(&mut self) {
        let forced: Vec<AdtId> = self
            .typeinfo_policy
            .iter()
            .filter(|&(_, &on)| on)
            .map(|(&id, _)| id)
            .collect();
        for id in forced {
            let span = self.unit.types.adt(id).span;
            self.need_table(Ty::Adt(id), span);
        }
        let Some(ti) = self.unit.typeinfo else { return };
        // every table points at `TypeInfo`'s own, as the type of what its
        // references refer to
        let mut work: Vec<Ty> = self.unit.tables.clone();
        work.push(Ty::Adt(ti.info));
        let mut seen: HashSet<Ty> = HashSet::new();
        let mut order = Vec::new();
        while let Some(t) = work.pop() {
            if !seen.insert(t) {
                continue;
            }
            order.push(t);
            work.extend(self.mentioned(t));
            if let Ty::Adt(id) = t {
                let methods = self.table_methods_of(id);
                work.extend(methods.iter().map(|m| m.sig));
                self.unit.table_methods.insert(id, methods);
                let annots = self.annotation_names.get(&id).cloned().unwrap_or_default();
                self.unit.table_annots.insert(id, annots);
            }
        }
        self.unit.tables = order;
    }

    /// The types whose tables `t`'s mentions.
    fn mentioned(&mut self, t: Ty) -> Vec<Ty> {
        let types = &self.unit.types;
        match t {
            Ty::Adt(id) => {
                self.adt(id);
                self.unit.types.adt(id).field_types().collect()
            }
            Ty::Tuple(_) => types.as_tuple(t).map(<[Ty]>::to_vec).unwrap_or_default(),
            Ty::Ref(_) => types.as_ref(t).map(|(_, x)| vec![x]).unwrap_or_default(),
            Ty::Slice(_) => types.as_slice(t).map(|(_, x)| vec![x]).unwrap_or_default(),
            Ty::Array(_) => types.as_array(t).map(|(x, _)| vec![x]).unwrap_or_default(),
            Ty::Growable(_) => types.as_growable(t).map(|x| vec![x]).unwrap_or_default(),
            Ty::Fn(_) | Ty::Closure(_) | Ty::Circuit(_) => types
                .as_sig(t)
                .map(|(p, r)| p.iter().copied().chain([r]).collect())
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// The methods a structure's or enumeration's table lists: every method
    /// that can be called through a reference (those taking `*self` or
    /// `*const self`, and associated functions), with the receiver erased.
    fn table_methods_of(&mut self, id: AdtId) -> Vec<TableMethod> {
        let def = self.unit.types.adt(id);
        let (name, origin) = (def.name, def.origin.clone());
        // another unit's type lists every method its unit gives it, not only
        // the ones this unit has called: a library reading the type through
        // its table, as `tcon` does with `from_tcon`, needs them all
        if let Some(o) = origin.as_deref()
            && o != self.unit.name
            && let Some(iface) = self.available.iter().find(|i| i.unit == o)
        {
            let given: Vec<(Symbol, bool)> = iface
                .fns
                .iter()
                .filter_map(|f| {
                    let m = f.method?;
                    let a = iface.types.adt(m.owner);
                    let own = a.name == name && a.origin.as_deref() == Some(o) && a.args.is_empty();
                    (own && f.linkage == crate::ast::Linkage::Program).then_some((f.name, m.operator))
                })
                .collect();
            for (method, operator) in given {
                self.find_method(Some(o), name, method, operator);
            }
        }
        let mut keys: Vec<(Symbol, bool)> = self
            .methods
            .keys()
            .filter(|k| k.ty == name && k.origin == origin)
            .map(|k| (k.name, k.operator))
            .collect();
        keys.sort_by(|a, b| {
            self.interner
                .resolve(a.0)
                .cmp(self.interner.resolve(b.0))
                .then(a.1.cmp(&b.1))
        });
        let mut out = Vec::new();
        for (method, operator) in keys {
            let Some(entry) = self.method_of_type(id, method, operator) else { continue };
            let callee = match entry.target {
                MethodTarget::Fn(f) => Callee::Fn(f),
                MethodTarget::Extern(x) => Callee::Extern(x),
                MethodTarget::Generic(key) => {
                    let args = self.unit.types.adt(id).args.clone();
                    // a method with parameters of its own has no one entry,
                    // nor has one whose `where` clause refuses this instance,
                    // such as the determinant of a matrix that is not square
                    if self.generic_params(key).is_some_and(|p| p.len() > args.len()) {
                        continue;
                    }
                    let span = self.unit.types.adt(id).span;
                    if !self.where_allows(key, &args, span) {
                        continue;
                    }
                    let span = self.unit.types.adt(id).span;
                    match self.instantiate_fn(key, args, span) {
                        Some(f) => Callee::Fn(f),
                        None => continue,
                    }
                }
            };
            let (mut params, ret, constant) = match callee {
                Callee::Fn(f) => {
                    self.fn_sig(f);
                    let f = self.unit.func(f);
                    (f.param_types().collect::<Vec<_>>(), f.ret, f.kind == tir::FnKind::Constant)
                }
                Callee::Extern(x) => {
                    let x = self.unit.extern_fn(x);
                    (x.params.clone(), x.ret, false)
                }
            };
            if constant {
                continue;
            }
            match params.first().copied() {
                // by value: the table has no value to give it
                Some(p) if p == Ty::Adt(id) => continue,
                Some(p) => {
                    if let Some((access, Ty::Adt(owner))) = self.unit.types.as_ref(p)
                        && owner == id
                    {
                        params[0] = self.unit.types.reference(access, Ty::Void);
                    }
                }
                None => {}
            }
            let sig = self.unit.types.function(params, ret);
            let shown = if operator {
                self.interner.intern_late(&format!("${}", self.interner.resolve(method)))
            } else {
                method
            };
            out.push(TableMethod {
                name: shown,
                sig,
                callee,
            });
        }
        out
    }
}
