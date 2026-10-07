//! Methods, associated functions and operator methods.
//!
//! A method is a function whose first parameter is `self`, of type `T`, `*T`
//! or `*const T`, called as `value.name(…)`. A function of a type without
//! `self` is an *associated function*, called as `T::name(…)`. Both are
//! written in an `impl` block, or outside one with the type in front of the
//! name, `fn Point.norm(self: *Point) -> f64 { … }`; the two spellings declare
//! the same thing, so writing a method both ways, or twice, is refused.
//!
//! A method belongs to its type, not to the unit scope: two types may each
//! have a `len`, and neither hides a function called `len`. So methods live
//! in a table of their own, keyed by the type (the unit that declared it and
//! its name) and the method's name.
//!
//! # The receiver
//!
//! `value.name(…)` adjusts `value` by one step to fit the method's `self`: a
//! method taking `*T` or `*const T` is given a reference to `value` when `value`
//! is a `T`, and a method taking `T` is given what `value` refers to when
//! `value` is a reference to one. Nothing else is adjusted, so a mismatch is
//! reported rather than guessed at.
//!
//! # Operators
//!
//! An operator whose operand is a structure or enumeration is a call of the
//! operand type's operator method: `a + b` is `a.$add(b)`, `-a` is
//! `a.$neg()`, `a == b` is `a.$eq(b)` and `a != b` its negation, `a[i]` is
//! `*a.$index(i)`, and calling a value that is not a function calls its
//! `$call`. A type may also have `$index_rd`, which returns a `*const`
//! reference: `a[i]` calls it where the element is only read (as the whole
//! value of a new `const` binding, `let x: const T = a[i];`), and wherever
//! `$index` cannot serve, because `a` is a constant or the type has no
//! `$index`. `a += b` calls `$add_assign` when the type has one, and is
//! otherwise `a = a + b`. The second operand is adjusted the way a receiver
//! is, so `$add(self: *V, other: *V)` serves `a + b` with `a` and `b` values.
//! There is no interface to implement: a method with the right name and a
//! signature that fits is all an operator needs.
//!
//! # Other units' types
//!
//! A unit may add methods to another unit's type only if that unit marked it
//! `[open]`: a generic one in an `impl` block naming its parameters. Methods
//! so added count as the type's own wherever the adding unit is imported.
//! Two units adding the same method to one type is found when the program is
//! linked.
//!
//! # Generic methods
//!
//! `impl Pair<A, B> { … }` declares methods of every instance of `Pair`: each
//! is generic over the block's parameters, then its own. A call takes the
//! block's arguments from the receiver's type and deduces the method's own
//! from its arguments, as for any generic function.

use crate::ast::{self, FnItem, ItemKind, Path};
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{self, AdtId, Arg, Callee, Expr, ExternId, FnId, Ty};

use super::body::Checker;
use super::generic;
use super::imports;
use super::interface::Interface;
use super::items::{GenericKey, GenericSrc, UnitCx};
use super::report;

/// Every operator method name a type may define, without its `$`. Anything
/// else written with `$` is reserved.
pub const OPERATORS: [&str; 37] = [
    "add", "sub", "mul", "div", "rem", "neg", "not", "and", "or", "xor", "shl", "shr", "tensor", "eq",
    "ord", "index", "index_rd", "call", "iter", "next", "drop", "copy", "fmt", "adj", "hash", "query", "lookup",
    "add_assign", "sub_assign", "mul_assign", "div_assign", "rem_assign", "and_assign", "or_assign",
    "xor_assign", "shl_assign", "shr_assign",
];

/// Whether `name` is an operator method's name.
pub fn is_operator(name: &str) -> bool {
    OPERATORS.contains(&name)
}

/// What a generic method belongs to, as part of its [`GenericKey`], so that a
/// method and a function of one name do not collide.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OwnerKey {
    /// The type's name.
    pub ty: Symbol,
    /// The unit that declared the type.
    pub unit: Option<Symbol>,
    /// Whether the method is an operator method.
    pub operator: bool,
}

/// A method's place in the table: its type, by the unit that declared it and
/// its name, and its own name.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MethodKey {
    /// The unit that declared the type.
    pub origin: Option<String>,
    /// The type's name.
    pub ty: Symbol,
    /// The method's name.
    pub name: Symbol,
    /// Whether it is an operator method.
    pub operator: bool,
}

/// What a method is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MethodTarget {
    /// A function of this unit.
    Fn(FnId),
    /// A function of another unit.
    Extern(ExternId),
    /// A generic method.
    Generic(GenericKey),
}

impl MethodTarget {
    /// The function it is.
    pub fn callee(self) -> Option<Callee> {
        match self {
            MethodTarget::Fn(f) => Some(Callee::Fn(f)),
            MethodTarget::Extern(x) => Some(Callee::Extern(x)),
            MethodTarget::Generic(_) => None,
        }
    }
}

/// A method in the table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MethodEntry {
    /// What it is.
    pub target: MethodTarget,
    /// Where it is declared, or a synthetic span for another unit's.
    pub span: Span,
}

/// How a generic method's type is found for an instance.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OwnerSpec {
    /// A type that is not generic.
    Plain(AdtId),
    /// A generic type.
    Generic(GenericKey),
}

/// The method part of a generic declaration.
#[derive(Clone, Copy, Debug)]
pub struct GenericMethod<'a> {
    /// The method as written.
    pub item: &'a FnItem,
    /// Its type.
    pub owner: OwnerSpec,
    /// How many of the generic parameters are the `impl` block's.
    pub impl_arity: usize,
    /// Whether it takes `self`.
    pub receiver: bool,
    /// Whether it is an operator method.
    pub operator: bool,
}

/// The type a method is being declared for.
#[derive(Clone, Debug)]
struct Owner {
    spec: OwnerSpec,
    origin: Option<String>,
    ty: Symbol,
    unit: Option<Symbol>,
    /// The type as written.
    text: String,
    /// Whether it is another unit's type, which then must be `[open]`.
    foreign: bool,
}

impl<'a> UnitCx<'a> {
    /// Declares the methods an item declares, if it declares any: an `impl`
    /// block, a function written `T.name`, or an operator method.
    pub(super) fn declare_methods(&mut self, item: &'a Spanned<ast::Item>) {
        self.declare_methods_in(&item.node, item.span, None);
    }

    /// Declares the methods of an item of another unit, in its frame: those
    /// that are not generic are that unit's compiled functions.
    pub(super) fn declare_foreign_methods(&mut self, iface: &'a Interface, item: &'a Spanned<ast::Item>) {
        self.declare_methods_in(&item.node, item.span, Some(iface));
    }

    /// Declares the methods of the item `it`, written at `span`: one of the
    /// unit's, of another unit's, or of a block's.
    pub(super) fn declare_methods_in(&mut self, it: &'a ast::Item, span: Span, foreign: Option<&'a Interface>) {
        match &it.kind {
            ItemKind::Impl { path, generics, items } => {
                // another unit's non-generic methods are its compiled functions,
                // found through its interface when called; naming their type now
                // would bring it into a unit that may never use it
                if foreign.is_some() && generics.is_empty() && items.iter().all(|f| f.generics.is_empty()) {
                    return;
                }
                let Some(owner) = self.owner_of(path, span, foreign.is_some()) else {
                    return;
                };
                if !self.impl_arity_fits(&owner, generics, path, span) {
                    return;
                }
                for f in items {
                    self.declare_method(&owner, generics, f, it, foreign);
                }
            }
            ItemKind::Fn(f) => {
                let (path, at) = match &f.name {
                    ast::FnName::Plain(_) => return,
                    ast::FnName::External { ty, .. } => (ty, ty.segments.last().map_or(span, |s| s.span)),
                    ast::FnName::Operator(n) => {
                        // outside an `impl`, an operator method's type is the
                        // type of its `self`
                        let written = f.params.first().filter(|p| p.is_self).and_then(|p| receiver_path(&p.ty.node));
                        let Some(path) = written else {
                            if foreign.is_none() {
                                self.report(
                                    Diagnostic::new(Code::Es07)
                                        .with_message(format!(
                                            "`${}` is an operator method, so it needs a `self` naming its type",
                                            self.interner.resolve(n.node)
                                        ))
                                        .at(n.span)
                                        .with_help("write `self: T` first, or declare it in `impl T { … }`"),
                                );
                            }
                            return;
                        };
                        (path, n.span)
                    }
                };
                let Some(owner) = self.owner_of(path, span, foreign.is_some()) else {
                    return;
                };
                if let OwnerSpec::Generic(_) = owner.spec {
                    if foreign.is_none() {
                        self.report(generic_owner_needs_impl(&owner.text, at));
                    }
                    return;
                }
                self.declare_method(&owner, &[], f, it, foreign);
            }
            _ => {}
        }
    }

    /// The type a method is declared for, written as `path`. `quiet` is set
    /// for another unit's declarations, checked when it was compiled.
    fn owner_of(&mut self, path: &Path, span: Span, quiet: bool) -> Option<Owner> {
        let frame_origin = self.frame_origin();
        let text = path
            .segments
            .iter()
            .map(|s| self.interner.resolve(s.node))
            .collect::<Vec<_>>()
            .join("::");
        let found = match path.segments.as_slice() {
            [name] => {
                if let Some(id) = self.lookup_type(name.node) {
                    let def = self.unit.types.adt(id);
                    let origin = def.origin.clone();
                    Some(Owner {
                        spec: OwnerSpec::Plain(id),
                        foreign: origin != frame_origin,
                        unit: origin.as_deref().and_then(|o| self.interner.get(o)),
                        origin,
                        ty: def.name,
                        text,
                    })
                } else if let Some(key) = self.generic(name.node).filter(|&k| !self.is_generic_fn(k)) {
                    Some(Owner {
                        spec: OwnerSpec::Generic(key),
                        unit: frame_origin.as_deref().and_then(|o| self.interner.get(o)),
                        origin: frame_origin.clone(),
                        ty: name.node,
                        text,
                        foreign: false,
                    })
                } else {
                    if !quiet {
                        let names: Vec<&str> = self.scope.type_names().map(|s| self.interner.resolve(s)).collect();
                        let d = report::not_found(name.span, "type", self.interner.resolve(name.node), names)
                            .with_note("methods are declared for a structure or enumeration");
                        self.report(d);
                    }
                    None
                }
            }
            [unit, name] => {
                let Some(u) = self.scope.get_unit(unit.node) else {
                    if !quiet {
                        let d = imports::unknown_unit(self, unit.node, unit.span);
                        self.report(d);
                    }
                    return None;
                };
                match imports::import_type(self, u, name.node) {
                    Some(id) => {
                        let def = self.unit.types.adt(id);
                        Some(Owner {
                            spec: OwnerSpec::Plain(id),
                            origin: def.origin.clone(),
                            ty: def.name,
                            unit: Some(unit.node),
                            text,
                            foreign: true,
                        })
                    }
                    None => match self.imported_generic(u, name.node).filter(|&k| !self.is_generic_fn(k)) {
                        // another unit's generic type: the methods are
                        // generic over its parameters, made with each
                        // instance, and are this unit's to check
                        Some(key) => {
                            let open = self.generic_src(key).is_some_and(|s| {
                                s.annotations.iter().any(|g| {
                                    g.node.annotations.iter().any(|a| self.interner.resolve(a.node.name.node) == "open")
                                })
                            });
                            let own = u == super::scope::UnitRef::SELF;
                            let origin = if own { self.unit.name.clone() } else { self.imported[u.0 as usize].unit.clone() };
                            if !open && !quiet {
                                self.report(
                                    Diagnostic::new(Code::Es19)
                                        .with_message(format!(
                                            "`{text}` belongs to unit `{origin}`, which did not mark it `[open]`, so \
                                             no other unit may add methods to it"
                                        ))
                                        .at(span)
                                        .with_help(format!(
                                            "mark `{}` `[open]` in `{origin}`, or declare a function taking it instead",
                                            self.interner.resolve(name.node)
                                        )),
                                );
                                return None;
                            }
                            Some(Owner {
                                spec: OwnerSpec::Generic(key),
                                // this unit's own type, named by a unit that
                                // it imports and that imports it back
                                origin: (!own).then_some(origin),
                                ty: name.node,
                                unit: Some(unit.node),
                                text,
                                foreign: !own,
                            })
                        }
                        None => {
                            if !quiet {
                                let d = imports::no_such_item(self, u, name.node, name.span, "type");
                                self.report(d);
                            }
                            None
                        }
                    },
                }
            }
            _ => {
                if !quiet {
                    self.report(report::path_too_long(span, &text, 2));
                }
                None
            }
        }?;
        if found.foreign && !quiet {
            let OwnerSpec::Plain(id) = found.spec else { return Some(found) };
            if !self.unit.types.adt(id).open {
                let unit = found.origin.clone().unwrap_or_default();
                self.report(
                    Diagnostic::new(Code::Es19)
                        .with_message(format!(
                            "`{}` belongs to unit `{unit}`, which did not mark it `[open]`, so no other \
                             unit may add methods to it",
                            found.text
                        ))
                        .at(span)
                        .with_note(
                            "a type's methods are part of what its unit promises about it; `[open]` \
                             is how a unit lets others add to them",
                        )
                        .with_help(format!(
                            "mark the type `[open]` in `{unit}`, or write a plain function that takes \
                             the value instead"
                        )),
                );
                return None;
            }
        }
        Some(found)
    }

    /// Whether an `impl` block names as many parameters as its type has.
    fn impl_arity_fits(&mut self, owner: &Owner, generics: &[ast::GenericParam], path: &Path, span: Span) -> bool {
        let params = match owner.spec {
            OwnerSpec::Plain(_) => &[][..],
            OwnerSpec::Generic(k) => self.generic_params(k).unwrap_or_default(),
        };
        let needed = params.len();
        if needed == generics.len() {
            return true;
        }
        let names: Vec<&str> = params.iter().map(|p| self.interner.resolve(p.name.node)).collect();
        let at = path.segments.last().map_or(span, |s| s.span);
        let d = if needed == 0 {
            Diagnostic::new(Code::Es06)
                .with_message(format!("`{}` is not generic, so its `impl` has no parameters to declare", owner.text))
                .at(at)
                .with_help(format!("write `impl {} {{ … }}`", owner.text))
        } else {
            Diagnostic::new(Code::Es06)
                .with_message(format!(
                    "`{}` takes {needed} generic parameter{}, so its `impl` names {}",
                    owner.text,
                    if needed == 1 { "" } else { "s" },
                    if needed == 1 { "one" } else { "that many" }
                ))
                .at(at)
                .with_note("the methods of a generic type are generic over the same parameters")
                .with_help(format!("write `impl {}<{}> {{ … }}`", owner.text, names.join(", ")))
        };
        self.report(d);
        false
    }

    /// Declares one method of `owner`, `f`, written in the item `it`: an
    /// `impl` block, whose methods carry no annotations of their own, or a
    /// method declared on its own.
    fn declare_method(
        &mut self,
        owner: &Owner,
        impl_generics: &'a [ast::GenericParam],
        f: &'a FnItem,
        it: &'a ast::Item,
        foreign: Option<&'a Interface>,
    ) {
        let (linkage, item) = (it.linkage, &it.kind);
        let annotations: &'a [Spanned<ast::AnnotationGroup>] = match item {
            ItemKind::Impl { .. } => &[],
            _ => &it.annotations,
        };
        let (name, operator) = match &f.name {
            ast::FnName::Plain(n) | ast::FnName::External { name: n, .. } => (*n, false),
            ast::FnName::Operator(n) => (*n, true),
        };
        let text = self.interner.resolve(name.node);
        if foreign.is_none() {
            self.check_reserved(name.node, name.span);
        }
        if operator && !is_operator(text) {
            if foreign.is_none() {
                let d = unknown_operator(text, name.span);
                self.report(d);
            }
            return;
        }
        // `$copy` makes the type copyable, which analysis needs to know
        // before any body that copies one is checked
        if operator && text == "copy" {
            self.unit.types.mark_copyable(owner.origin.clone(), owner.ty);
        }
        let receiver = f.params.first().is_some_and(|p| p.is_self);
        let key = MethodKey {
            origin: owner.origin.clone(),
            ty: owner.ty,
            name: name.node,
            operator,
        };
        let shown = if operator { format!("${text}") } else { text.to_owned() };
        if let Some(earlier) = self.methods.get(&key) {
            if foreign.is_none() {
                let d = report::duplicate(name.span, earlier.span, &shown, &format!("as a method of `{}`", owner.text));
                self.report(d.with_note(
                    "a method written in an `impl` block and one written `fn T.name` are the same \
                     declaration, so a type has at most one method of each name",
                ));
            }
            return;
        }
        if owner.foreign
            && foreign.is_none()
            && let Some(origin) = &owner.origin
            && let Some(iface) = self.available.iter().find(|i| i.unit == *origin)
            && iface.method(origin, owner.ty, name.node, operator).is_some()
        {
            self.report(
                Diagnostic::new(Code::Es05)
                    .with_message(format!("`{}` already has a method `{shown}`", owner.text))
                    .at(name.span)
                    .with_note(format!("unit `{origin}` declares it"))
                    .with_help("give this one another name"),
            );
            return;
        }
        let generic = !impl_generics.is_empty() || !f.generics.is_empty();
        // its body is checked here when it is this unit's, or generic
        if foreign.is_none() || generic {
            self.declare_block_items(&f.body);
        }
        let target = match (generic, owner.spec, foreign) {
            (false, OwnerSpec::Plain(adt), None) => {
                let id = self.add_fn(f, linkage, name, name.span);
                self.unit.fns[id.index()].method = Some(tir::MethodOf {
                    owner: adt,
                    operator,
                    receiver,
                });
                let subject = super::attrs::Subject {
                    method: true,
                    ..super::attrs::Subject::function(linkage, false)
                };
                self.unit.fns[id.index()].attrs = self.item_attrs(annotations, subject);
                MethodTarget::Fn(id)
            }
            // another unit's compiled method is found through its interface
            // when something calls it
            (false, OwnerSpec::Plain(_), Some(_)) => return,
            (_, spec, _) => {
                let mut generics = impl_generics.to_vec();
                // `impl Ring<N>` names the type's parameters without their
                // kinds; a constant one, `N: const usize`, is one here too
                if let OwnerSpec::Generic(k) = spec
                    && let Some(declared) = self.generic_params(k).map(<[ast::GenericParam]>::to_vec)
                {
                    for (g, d) in generics.iter_mut().zip(declared) {
                        if g.ty.is_none() {
                            g.ty = d.ty;
                        }
                    }
                }
                generics.extend(f.generics.iter().cloned());
                let src = GenericSrc {
                    generics,
                    item,
                    method: Some(GenericMethod {
                        item: f,
                        owner: spec,
                        impl_arity: impl_generics.len(),
                        receiver,
                        operator,
                    }),
                    annotations: &[],
                    name,
                    linkage,
                };
                let of = OwnerKey {
                    ty: owner.ty,
                    unit: owner.unit,
                    operator,
                };
                let Some(key) = self.add_generic(Some(of), src) else {
                    return;
                };
                MethodTarget::Generic(key)
            }
        };
        self.methods.insert(
            key,
            MethodEntry {
                target,
                span: name.span,
            },
        );
    }

    /// Whether a `self` parameter may be written in function `id`: only as
    /// the first parameter of a method.
    pub(super) fn self_allowed(&mut self, id: FnId, first: bool, span: Span) -> bool {
        let f = self.unit.func(id);
        if f.method.is_none() {
            let name = self.interner.resolve(f.name);
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!("`self` is the value a method is called on, and `{name}` is not a method"))
                    .at(span)
                    .with_help(format!(
                        "declare it in `impl T {{ … }}`, or as `fn T.{name}(self: …)`, or rename the parameter"
                    )),
            );
            return false;
        }
        if !first {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message("`self` must be a method's first parameter")
                    .at(span)
                    .with_note("`value.m(a, b)` passes `value` first"),
            );
            return false;
        }
        true
    }

    /// Checks that a method's `self` is its type, or a reference to it.
    pub(super) fn check_receiver(&mut self, id: FnId) {
        let f = self.unit.func(id);
        let Some(m) = f.method.filter(|m| m.receiver) else {
            return;
        };
        let Some(&first) = f.params.first() else { return };
        let local = f.local(first);
        let (ty, span) = (local.ty, local.span);
        let inner = self.unit.types.as_ref(ty).map_or(ty, |(_, t)| t);
        if inner == Ty::Adt(m.owner) || ty == Ty::Never {
            return;
        }
        let owner = self.unit.types.adt_name(m.owner, self.interner);
        let found = self.unit.types.display(ty, self.interner);
        self.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!("a method of `{owner}` receives a `{owner}`, but this `self` is a `{found}`"))
                .at(span)
                .with_help(format!("write `self: {owner}`, `self: *{owner}` or `self: *const {owner}`")),
        );
    }

    /// The type an instance of a generic method belongs to, made with the
    /// `impl` block's share of `args`.
    pub(super) fn method_of_instance(&mut self, m: GenericMethod<'a>, args: &[Arg], span: Span) -> Option<tir::MethodOf> {
        let owner = match m.owner {
            OwnerSpec::Plain(id) => id,
            OwnerSpec::Generic(k) => self.instantiate_adt(k, args[..m.impl_arity].to_vec(), span)?,
        };
        Some(tir::MethodOf {
            owner,
            operator: m.operator,
            receiver: m.receiver,
        })
    }

    /// The method `name` of the type declared by `origin` as `ty`, looking in
    /// this unit, in the unit that declared the type, and among the methods
    /// the units this one imports add to it.
    pub fn find_method(&mut self, origin: Option<&str>, ty: Symbol, name: Symbol, operator: bool) -> Option<MethodEntry> {
        let key = MethodKey {
            origin: origin.map(str::to_owned),
            ty,
            name,
            operator,
        };
        if let Some(e) = self.methods.get(&key) {
            return Some(*e);
        }
        let available = self.available;
        let home = self.unit.name.clone();
        let owner_unit = origin.unwrap_or(&home);
        // the declaring unit's own: its generic methods need its source,
        // and its others are its compiled functions
        if let Some(o) = origin
            && let Some(iface) = available.iter().find(|i| i.unit == o)
        {
            if iface.ast.is_some() {
                self.frame_for(iface);
                if let Some(e) = self.methods.get(&key) {
                    return Some(*e);
                }
            }
            if let Some(x) = iface.method(o, ty, name, operator)
                && let super::scope::Def::Extern(id) = imports::extern_of(self, iface, x)
            {
                return Some(self.remember(key, MethodTarget::Extern(id)));
            }
        }
        // methods other units add to an `[open]` type
        for iface in available {
            if iface.unit == owner_unit {
                continue;
            }
            if let Some(x) = iface.method(owner_unit, ty, name, operator)
                && let super::scope::Def::Extern(id) = imports::extern_of(self, iface, x)
            {
                return Some(self.remember(key, MethodTarget::Extern(id)));
            }
        }
        None
    }

    fn remember(&mut self, key: MethodKey, target: MethodTarget) -> MethodEntry {
        let e = MethodEntry {
            target,
            span: Span::synthetic(),
        };
        self.methods.insert(key, e);
        e
    }

    /// The method `name` of a structure or enumeration.
    pub fn method_of_type(&mut self, id: AdtId, name: Symbol, operator: bool) -> Option<MethodEntry> {
        let def = self.unit.types.adt(id);
        let (origin, ty) = (def.origin.clone(), def.name);
        self.find_method(origin.as_deref(), ty, name, operator)
    }

    /// Records every type's `$drop` and `$copy`, making the instances a
    /// generic type's need, until no new type appears. A `$drop` takes
    /// `*self` alone and returns nothing; a `$copy` takes `*const self` alone
    /// and returns the copy. One that does not is reported, as is a `$fmt`
    /// that does not take `*const self` and the text so far and give the
    /// text back.
    pub fn collect_drops(&mut self) {
        let drop = self.interner.get("drop");
        let copy = self.interner.get("copy");
        let fmt = self.interner.get("fmt");
        let mut done = 0;
        while done < self.unit.types.adts().len() {
            let id = AdtId(done as u32);
            done += 1;
            if let Some(sym) = drop
                && let Some(callee) = self.special_method(id, sym, "drop")
            {
                self.unit.drops.insert(id, callee);
            }
            if let Some(sym) = copy
                && let Some(callee) = self.special_method(id, sym, "copy")
            {
                // reading a place of a type holding qubits moves them; a copy
                // of quantum state is a second preparation, made by `qcopy`
                if self.unit.types.is_quantum(Ty::Adt(id)) {
                    let name = self.unit.types.adt_name(id, self.interner);
                    let at = self.method_of_type(id, sym, true).map_or(Span::synthetic(), |m| m.span);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`{name}` holds qubits, so it cannot define `$copy`"))
                            .at(at)
                            .with_note(
                                "a type with `$copy` is copied whenever a place of it is read; a value \
                                 holding qubits is moved instead, since copying an unknown state is \
                                 impossible",
                            )
                            .with_help(
                                "remove `$copy`; a value whose preparation is known is copied with \
                                 `qcopy(&value)`, which prepares it again",
                            ),
                    );
                    continue;
                }
                self.unit.copies.insert(id, callee);
            }
            if let Some(sym) = fmt {
                // checked for its shape; `tcon` finds it through the table
                let _ = self.special_method(id, sym, "fmt");
            }
        }
    }

    /// The function a type's `$drop`, `$copy` or `$fmt` is, if it has one of
    /// the right shape.
    fn special_method(&mut self, id: AdtId, sym: Symbol, which: &str) -> Option<Callee> {
        // the method, with a generic one's instance for this type made
        let entry = self.method_of_type(id, sym, true)?;
        let span = self.unit.types.adt(id).span;
        let callee = match entry.target {
            MethodTarget::Generic(key) => {
                let args = self.unit.types.adt(id).args.clone();
                Callee::Fn(self.instantiate_fn(key, args, span)?)
            }
            target => target.callee()?,
        };

        // the signature it must have
        let (params, ret) = self.signature(callee);
        let ty = self.unit.types.adt_name(id, self.interner);
        let (access, want_ret, shape, note) = match which {
            "drop" => (
                tir::Access::Write,
                Ty::Void,
                format!("fn $drop(self: *{ty})"),
                "`$drop` runs as a value is destroyed, with the value lent to it to tidy up; it takes nothing else and gives nothing back",
            ),
            "copy" => (
                tir::Access::Const,
                Ty::Adt(id),
                format!("fn $copy(self: *const {ty}) -> {ty}"),
                "`$copy` runs wherever a value is copied, with the original lent to it; it gives back the copy",
            ),
            _ => (
                tir::Access::Const,
                self.unit.types.growable(Ty::Char),
                format!("fn $fmt(self: *const {ty}, out: [char]) -> [char]"),
                "`$fmt` writes the value as text: it is given the text written so far and gives it back with the \
                 value's own added, which is how `tcon::emit` writes the value",
            ),
        };

        // and the one it has
        let receiver = self.unit.types.reference(access, Ty::Adt(id));
        let mut wanted = vec![receiver];
        if which == "fmt" {
            wanted.push(want_ret);
        }
        if params != wanted || ret != want_ret {
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!("`${which}` of `{ty}` must be `{shape}`"))
                    .at(entry.span)
                    .with_note(note),
            );
            return None;
        }
        Some(callee)
    }

    /// The names of the methods this unit knows `id` to have, for a
    /// suggestion.
    pub fn method_names(&self, id: AdtId) -> Vec<Symbol> {
        let def = self.unit.types.adt(id);
        self.methods
            .keys()
            .filter(|k| k.ty == def.name && k.origin == def.origin && !k.operator)
            .map(|k| k.name)
            .collect()
    }
}

/// The path of the type a `self` parameter is written with: `T`, `*T` or
/// `*const T`.
fn receiver_path(t: &ast::Type) -> Option<&Path> {
    match t {
        ast::Type::Path { path, .. } => Some(path),
        ast::Type::Ref { target, .. } => match &target.node {
            ast::Type::Path { path, .. } => Some(path),
            _ => None,
        },
        _ => None,
    }
}

/// A method written outside an `impl` for a generic type.
fn generic_owner_needs_impl(ty: &str, span: Span) -> Diagnostic {
    Diagnostic::new(Code::Es06)
        .with_message(format!("`{ty}` is generic, so its methods are written in an `impl` block"))
        .at(span)
        .with_note("the block names the type's parameters, which its methods are generic over")
        .with_help(format!("write `impl {ty}<…> {{ fn … }}`"))
}

/// A `$` name that is not an operator method's.
pub fn unknown_operator(name: &str, span: Span) -> Diagnostic {
    let mut d = Diagnostic::new(Code::Em02)
        .with_message(format!("`${name}` is not an operator method"))
        .at(span)
        .with_note("names beginning with `$` are reserved for the operator methods the language defines");
    if let Some(s) = report::closest(name, OPERATORS) {
        d = d.with_help(format!("did you mean `${s}`?"));
    } else {
        d = d.with_help("drop the `$` to declare an ordinary method");
    }
    d
}

impl Checker<'_, '_> {
    /// `receiver.name(args)`.
    pub(super) fn method_call(
        &mut self,
        receiver: &Spanned<ast::Expr>,
        name: Spanned<Symbol>,
        targs: &[Spanned<ast::TArg>],
        args: &[Spanned<ast::Expr>],
        span: Span,
    ) -> Expr {
        let recv = self.expr(receiver);
        // which methods a value has depends on its type, so a variant still
        // waiting on its context, as `Some(5)` is, is settled now
        if self.infer.unsettled_instance(recv.ty).is_some() {
            self.settled(recv.ty);
        }
        let t = self.shallow(recv.ty);
        let base = self.types().as_ref(t).map_or(t, |(_, x)| self.shallow(x));
        match base {
            Ty::Never => {
                self.check_only(args);
                Checker::error(span)
            }
            Ty::Array(_) if self.name(name.node) == "len" => self.len_call(recv, targs, args, span),
            Ty::Fn(_) if self.cx.quantum && self.name(name.node) == "then" && targs.is_empty() => {
                self.then_call(recv, args, span)
            }
            Ty::Slice(_) | Ty::Growable(_) => self.array_method(recv, name, targs, args, span),
            Ty::Adt(id)
                if targs.is_empty()
                    && self.is_qpu_device(id)
                    && let Some(run) = super::qpu::Run::of_method(self.name(name.node)) =>
            {
                self.run_call(recv, run, args, span)
            }
            Ty::Adt(id) => {
                let Some(entry) = self.cx.method_of_type(id, name.node, false) else {
                    if targs.is_empty()
                        && let Some(e) = self.unwrap_call(&recv, self.interner.resolve(name.node), args, span)
                    {
                        return e;
                    }
                    self.check_only(args);
                    let d = self.no_method(id, name);
                    self.report(d);
                    return Checker::error(span);
                };
                let label = format!("{}.{}", self.types().adt_name(id, self.interner), self.name(name.node));
                let checked: Vec<Expr> = args.iter().map(|a| self.expr(a)).collect();
                self.call_method(entry, Some(recv), targs, checked, &label, span)
            }
            other => {
                self.check_only(args);
                let what = self.describe(other);
                let mut d = Diagnostic::new(Code::Es04)
                    .with_message(format!("{what} has no methods, so `.{}(…)` names nothing", self.name(name.node)))
                    .at(name.span)
                    .with_note("methods belong to structures and enumerations");
                if matches!(other, Ty::Array(_) | Ty::Slice(_)) {
                    d = d.with_help("an array or slice has one method, `len()`");
                }
                self.report(d);
                Checker::error(span)
            }
        }
    }

    /// A type has no method of this name.
    fn no_method(&mut self, id: AdtId, name: Spanned<Symbol>) -> Diagnostic {
        let ty = self.types().adt_name(id, self.interner);
        let text = self.name(name.node);
        let interner = self.interner;
        let names: Vec<&str> = self.cx.method_names(id).into_iter().map(|s| interner.resolve(s)).collect();
        let mut d = Diagnostic::new(Code::Es04)
            .with_message(format!("`{ty}` has no method named `{text}`"))
            .at(name.span);
        if let Some(s) = report::closest(text, names) {
            d = d.with_help(format!("did you mean `{s}`?"));
        } else if self.adt(id).is_struct() && self.adt(id).field_index(name.node).is_some() {
            d = d.with_help(format!("`{text}` is a field; read it with `.{text}`, without parentheses"));
        }
        d
    }

    /// Calls a method or associated function with arguments already checked.
    /// `recv` is the value it is called on, if it is called with `.`.
    pub(super) fn call_method(
        &mut self,
        entry: MethodEntry,
        recv: Option<Expr>,
        targs: &[Spanned<ast::TArg>],
        checked: Vec<Expr>,
        label: &str,
        span: Span,
    ) -> Expr {
        // the function called, with a generic one's instance made
        let (callee, decl) = match entry.target {
            MethodTarget::Fn(id) => {
                if !targs.is_empty() {
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`{label}` takes no generic arguments"))
                            .at(span),
                    );
                    return Checker::error(span);
                }
                self.cx.fn_sig(id);
                (Callee::Fn(id), Some(self.cx.unit.func(id).span))
            }
            MethodTarget::Extern(id) => (Callee::Extern(id), None),
            MethodTarget::Generic(key) => {
                let Some(id) = self.generic_method(key, recv.as_ref(), targs, &checked, label, span) else {
                    return Checker::error(span);
                };
                (Callee::Fn(id), Some(self.cx.unit.func(id).span))
            }
        };
        self.cx.use_callee(callee, span);

        // its parameters, the first taken by the receiver if there is one
        let (params, ret, method) = match callee {
            Callee::Fn(id) => {
                let f = self.cx.unit.func(id);
                (f.param_types().collect::<Vec<_>>(), f.ret, f.method)
            }
            Callee::Extern(id) => {
                let x = self.cx.unit.extern_fn(id);
                (x.params.clone(), x.ret, x.method)
            }
        };
        let receiver = method.is_some_and(|m| m.receiver);
        let mut args = Vec::with_capacity(params.len());
        let rest = match recv {
            Some(r) => {
                if !receiver {
                    self.report(
                        Diagnostic::new(Code::Es07)
                            .with_message(format!("`{label}` takes no `self`, so it is not called on a value"))
                            .at(span)
                            .with_help(format!("call it as `{}(…)`", label.replacen('.', "::", 1))),
                    );
                    return Checker::error(span);
                }
                args.push(self.adjust_receiver(r, params[0], label));
                &params[1..]
            }
            None => &params[..],
        };

        // the rest of the arguments
        if checked.len() != rest.len() {
            let mut d = Checker::wrong_arity(label, rest.len(), checked.len(), span);
            if let Some(decl) = decl {
                d = d.also(decl, format!("`{label}` is declared here"));
            }
            self.report(d);
            return Checker::error(span);
        }
        for (i, (a, &p)) in checked.into_iter().zip(rest).enumerate() {
            args.push(self.coerce(a, p, &format!("argument {} of `{label}`", i + 1)));
        }
        Expr::call(callee, args, ret, span)
    }

    /// The instance of a generic method a call needs: the `impl` block's
    /// arguments from the receiver's type, the method's own from `targs` or
    /// else deduced from the arguments.
    fn generic_method(
        &mut self,
        key: GenericKey,
        recv: Option<&Expr>,
        targs: &[Spanned<ast::TArg>],
        checked: &[Expr],
        label: &str,
        span: Span,
    ) -> Option<FnId> {
        let decl = self.cx.generic_fn_item(key)?;
        let params = self.cx.generic_params(key)?.to_vec();
        let m = self.cx.generic_src(key)?.method?;
        let names: Vec<Symbol> = params.iter().map(|p| p.name.node).collect();
        let mut subst = generic::Subst::new();
        // the receiver's type is an instance of the method's type, whose
        // arguments are the `impl` block's
        if let Some(r) = recv {
            let t = self.settled(r.ty);
            let t = self.types().as_ref(t).map_or(t, |(_, x)| x);
            if let Ty::Adt(id) = t {
                let def_args = self.adt(id).args.clone();
                for (p, a) in params.iter().zip(def_args).take(m.impl_arity) {
                    subst.push((p.name.node, a));
                }
            }
        }
        if !targs.is_empty() {
            let own = &params[m.impl_arity..];
            // `Type::<A>::f(…)` writes the type's arguments, and
            // `Type::<A>::f::<B>(…)` both; the path carries them together
            let (for_type, for_own) = if recv.is_none()
                && m.impl_arity > 0
                && (targs.len() == m.impl_arity || targs.len() == m.impl_arity + own.len())
                && targs.len() != own.len()
            {
                targs.split_at(m.impl_arity)
            } else {
                (&[][..], targs)
            };
            if !for_type.is_empty() {
                let written = self.cx.generic_args(label, &params[..m.impl_arity], for_type, span)?;
                for (p, a) in params.iter().zip(written) {
                    subst.push((p.name.node, a));
                }
            }
            if !for_own.is_empty() {
                let written = self.cx.generic_args(label, own, for_own, span)?;
                for (p, a) in own.iter().zip(written) {
                    subst.push((p.name.node, a));
                }
            }
        }
        let skip = usize::from(recv.is_some() && m.receiver);
        for (p, e) in decl.params.iter().skip(skip).zip(checked) {
            let found = self.settled(e.ty);
            generic::match_written(self.cx, &p.ty.node, found, &names, &mut subst);
        }
        let mut args = Vec::with_capacity(params.len());
        for p in &params {
            match generic::lookup(&subst, p.name.node) {
                Some(a) => args.push(a),
                None => {
                    let d = generic::undeduced(label, self.name(p.name.node), span);
                    self.report(d);
                    return None;
                }
            }
        }
        self.cx.instantiate_fn(key, args, span)
    }

    /// Fits the value a method is called on to its `self`, by one step: a
    /// reference taken of a value, or a value read through a reference.
    pub(super) fn adjust_receiver(&mut self, r: Expr, want: Ty, label: &str) -> Expr {
        let span = r.span;
        let have = self.shallow(r.ty);
        if have == Ty::Never || want == Ty::Never {
            return r;
        }
        if let Some((_, target)) = self.types().as_ref(want)
            && self.types().as_ref(have).is_none()
            && self.unify(have, target).is_ok()
        {
            // a `*const T` of a constant does not become the `*T` a method
            // that may write takes
            let r = self.reference_to(r, span);
            return self.coerce(r, want, &format!("the value `{label}` is called on"));
        }
        if let Some((_, target)) = self.types().as_ref(have)
            && self.types().as_ref(want).is_none()
            && self.unify(target, want).is_ok()
        {
            return Expr::deref(r, target, span);
        }
        self.coerce(r, want, &format!("the value `{label}` is called on"))
    }

    /// The operator method `$name` of an operand's type, if the operand is a
    /// structure or enumeration that has one.
    pub(super) fn operator_method(&mut self, operand: Ty, name: &str) -> Option<(MethodEntry, String)> {
        let t = self.shallow(operand);
        let Ty::Adt(id) = t else { return None };
        let sym = self.interner.get(name)?;
        let entry = self.cx.method_of_type(id, sym, true)?;
        let label = format!("{}.${name}", self.types().adt_name(id, self.interner));
        Some((entry, label))
    }

    /// Whether a type is one whose operators are its operator methods.
    pub(super) fn is_structured(&self, t: Ty) -> bool {
        matches!(self.infer.shallow(t), Ty::Adt(_))
    }

    /// `l op r` for a structure or enumeration `l`: a call of its operator
    /// method `$name`, with `r` adjusted as a receiver would be.
    pub(super) fn operator_call(&mut self, name: &str, op: &str, l: Expr, r: Option<Expr>, span: Span) -> Expr {
        let lt = self.shallow(l.ty);
        let Some((entry, label)) = self.operator_method(lt, name) else {
            let what = self.describe(lt);
            let ty = self.show(lt);
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("{what} has no `{op}`: `{ty}` defines no `${name}`"))
                    .at(l.span)
                    .with_note(format!("`{op}` on a structure or enumeration calls its operator method `${name}`"))
                    .with_help(format!("declare `fn ${name}(self: {ty}, …)` in `impl {ty} {{ … }}`")),
            );
            return Checker::error(span);
        };
        let args = match r {
            Some(a) => vec![self.adjust_operand(entry.target, a, &label)],
            None => Vec::new(),
        };
        self.call_method(entry, Some(l), &[], args, &label, span)
    }

    /// The second operand of an operator method `target`, adjusted as the
    /// receiver is.
    fn adjust_operand(&mut self, target: MethodTarget, a: Expr, label: &str) -> Expr {
        let want = match target {
            MethodTarget::Fn(id) => {
                self.cx.fn_sig(id);
                self.cx.unit.func(id).param_types().nth(1)
            }
            MethodTarget::Extern(id) => self.cx.unit.extern_fn(id).params.get(1).copied(),
            // a generic method's instance is not made yet, but its parameter
            // is written a reference or not
            MethodTarget::Generic(key) => {
                let written = self.cx.generic_fn_item(key).and_then(|d| d.params.get(1)).map(|p| &p.ty.node);
                let t = self.shallow(a.ty);
                let is_ref = self.types().as_ref(t).is_some() || self.types().as_slice(t).is_some();
                return match written {
                    Some(ast::Type::Ref { .. }) if !is_ref => {
                        let at = a.span;
                        self.reference_to(a, at)
                    }
                    _ => a,
                };
            }
        };
        match want {
            Some(w) => {
                // text given to a `string`'s operator is made a `string`
                let target = self.types().as_ref(w).map_or(w, |(_, t)| t);
                let a = self.text_for(a, target);
                self.adjust_receiver(a, w, label)
            }
            None => a,
        }
    }

    /// The associated function or method a path such as `Point::new` or
    /// `geo::Point::new` names, if it names one. A variant of an enumeration
    /// takes precedence, as `Shape::Circle` is always the variant.
    pub(super) fn associated(&mut self, path: &Path) -> Option<(MethodEntry, String)> {
        let (owner, name) = match path.segments.as_slice() {
            [owner @ .., name] if (1..=2).contains(&owner.len()) => (owner, *name),
            _ => return None,
        };
        let ty = *owner.last().expect("an owner");
        // the type, or else the generic type, the owner names
        let found = match owner {
            [t] => self.cx.lookup_type(t.node).ok_or_else(|| self.cx.generic(t.node)),
            [u, t] => {
                let unit = self.cx.scope.get_unit(u.node)?;
                imports::import_type(self.cx, unit, t.node).ok_or_else(|| self.cx.imported_generic(unit, t.node))
            }
            _ => return None,
        };
        let (origin, key_ty) = match found {
            Ok(id) => {
                let def = self.adt(id);
                if def.variant_index(name.node).is_some() {
                    return None;
                }
                (def.origin.clone(), def.name)
            }
            Err(key) => {
                let key = key?;
                if self.cx.generic_enum_variant(key, name.node).is_some() || self.cx.is_generic_fn(key) {
                    return None;
                }
                (self.cx.generic_origin(key), ty.node)
            }
        };
        let entry = self.cx.find_method(origin.as_deref(), key_ty, name.node, false)?;
        let label = format!("{}::{}", self.name(ty.node), self.name(name.node));
        Some((entry, label))
    }

    /// `callee(args)` where `callee` is a value, not a function: a call of
    /// its `$call`.
    pub(super) fn value_call(&mut self, callee: Expr, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let t = self.shallow(callee.ty);
        let base = self.types().as_ref(t).map_or(t, |(_, x)| x);
        if matches!(base, Ty::Fn(_) | Ty::Closure(_)) {
            return self.call_value(callee, args, span).expect("a function or closure");
        }
        let checked: Vec<Expr> = args.iter().map(|a| self.expr(a)).collect();
        match self.operator_method(base, "call") {
            Some((entry, label)) => self.call_method(entry, Some(callee), &[], checked, &label, span),
            None => {
                if t != Ty::Never {
                    let what = self.describe(t);
                    self.report(
                        Diagnostic::new(Code::Es07)
                            .with_message(format!("{what} cannot be called"))
                            .at(callee.span)
                            .with_note("a call names a function, or a value whose type defines `$call`"),
                    );
                }
                Checker::error(span)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};
    use crate::tir::visit::{self, Visit};
    use crate::tir::{Callee, ExprKind};

    const V: &str = "struct V { x: i64, y: i64 }\n";

    #[test]
    fn a_method_is_called_with_its_receiver_adjusted() {
        check(&format!(
            "{V}impl V {{\n\
                 fn new(x: i64, y: i64) -> V {{ V {{ x: x, y: y }} }}\n\
                 fn dot(self: *V, o: *V) -> i64 {{ self.x * o.x + self.y * o.y }}\n\
                 fn scale(self: *V, k: i64) {{ self.x = self.x * k; self.y = self.y * k; }}\n\
                 fn sum(self: V) -> i64 {{ self.x + self.y }}\n\
             }}\n\
             fn f(p: *V) -> i64 {{ let a = V::new(1, 2); a.scale(3); a.dot(&a) + p.dot(p) + a.sum() }}"
        ));
    }

    #[test]
    fn the_two_spellings_are_one_declaration() {
        check(&format!("{V}fn V.norm1(self: *V) -> i64 {{ self.x + self.y }}\nfn f(v: V) -> i64 {{ v.norm1() }}"));
        assert_eq!(
            codes(&format!("{V}impl V {{ fn m(self: *V) {{ }} }}\nfn V.m(self: *V) {{ }}")),
            [Code::Es05]
        );
    }

    #[test]
    fn an_associated_function_has_no_self() {
        assert_eq!(codes(&format!("{V}impl V {{ fn zero() -> i64 {{ 0 }} }}\nfn f(v: V) -> i64 {{ v.zero() }}")), [Code::Es07]);
        check(&format!("{V}impl V {{ fn zero() -> i64 {{ 0 }} }}\nfn f() -> i64 {{ V::zero() }}"));
    }

    #[test]
    fn a_missing_method_names_the_type() {
        assert_eq!(codes(&format!("{V}fn f(v: V) -> i64 {{ v.len() }}")), [Code::Es04]);
        assert_eq!(codes("fn f(x: i32) -> i32 { x.abs() }"), [Code::Es04]);
    }

    #[test]
    fn self_is_only_a_methods_first_parameter() {
        assert_eq!(codes("fn f(self: i32) { }"), [Code::Es07]);
        assert_eq!(codes(&format!("{V}impl V {{ fn m(a: i32, self: V) {{ }} }}")), [Code::Es07]);
        assert_eq!(codes(&format!("{V}impl V {{ fn m(self: i32) {{ }} }}")), [Code::Es06]);
    }

    #[test]
    fn operators_on_a_structure_call_its_operator_methods() {
        let u = check(&format!(
            "{V}impl V {{\n\
                 fn $add(self: V, o: V) -> V {{ V {{ x: self.x + o.x, y: self.y + o.y }} }}\n\
                 fn $neg(self: V) -> V {{ V {{ x: -self.x, y: -self.y }} }}\n\
                 fn $eq(self: *V, o: *V) -> bool {{ self.x == o.x && self.y == o.y }}\n\
                 fn $index(self: *V, i: usize) -> *i64 {{ if i == 0 {{ &self.x }} else {{ &self.y }} }}\n\
                 fn $add_assign(self: *V, o: V) {{ self.x = self.x + o.x; }}\n\
             }}\n\
             fn f(a: V, b: V, c: V) -> bool {{ let s = -(a + b); let t = s; t += c; t[0] == 3 && t != t }}"
        ));
        struct Calls(usize);
        impl Visit for Calls {
            fn expr(&mut self, e: &crate::tir::Expr) {
                if let ExprKind::Call { callee: Callee::Fn(_), .. } = e.kind {
                    self.0 += 1;
                }
                visit::walk_expr(self, e);
            }
        }
        let mut calls = Calls(0);
        calls.block(&u.fns.iter().find(|f| f.method.is_none()).unwrap().body);
        assert_eq!(calls.0, 5, "`-`, `+`, `+=`, `[…]` and `!=` each became a call");
    }

    #[test]
    fn an_operator_the_type_lacks_is_explained() {
        assert_eq!(codes(&format!("{V}fn f(a: V, b: V) -> V {{ a * b }}")), [Code::Es06]);
    }

    #[test]
    fn an_unknown_dollar_name_is_em02() {
        assert_eq!(codes(&format!("{V}impl V {{ fn $plus(self: V, o: V) -> V {{ self }} }}")), [Code::Em02]);
    }

    #[test]
    fn a_top_level_operator_method_belongs_to_its_self_type() {
        check(&format!("{V}fn $mul(self: V, k: i64) -> V {{ V {{ x: self.x * k, y: self.y * k }} }}\nfn f(v: V) -> V {{ v * 2 }}"));
    }

    #[test]
    fn a_value_with_call_can_be_called() {
        check(
            "struct Adder { n: i64 }\n\
             impl Adder { fn $call(self: *Adder, x: i64) -> i64 { self.n + x } }\n\
             fn f() -> i64 { let a = Adder { n: 2 }; a(40) }",
        );
    }

    #[test]
    fn a_generic_types_methods_take_its_arguments_from_the_receiver() {
        let u = check(
            "struct Pair<A, B> { a: A, b: B }\n\
             impl Pair<A, B> {\n\
                 fn swap(self: Pair<A, B>) -> Pair<B, A> { Pair { a: self.b, b: self.a } }\n\
                 fn first(self: *Pair<A, B>) -> *A { &self.a }\n\
                 fn make(a: A, b: B) -> Pair<A, B> { Pair { a: a, b: b } }\n\
                 fn also<C>(self: Pair<A, B>, c: C) -> Pair<A, C> { Pair { a: self.a, b: c } }\n\
             }\n\
             fn f() -> bool { let p = Pair::make(1i32, true); let q = p.swap(); let r = q.also('c'); *r.first() }",
        );
        let instances = u.fns.iter().filter(|f| !f.args.is_empty()).count();
        assert_eq!(instances, 4, "make, swap, also and first, one instance each");
        assert!(u.fns.iter().filter(|f| !f.args.is_empty()).all(|f| f.method.is_some()));
    }

    #[test]
    fn a_drop_takes_self_alone() {
        check("struct R { n: i64 }\nimpl R { fn $drop(self: *R) { } }");
        assert_eq!(codes("struct R { n: i64 }\nimpl R { fn $drop(self: *const R) { } }"), [Code::Es07]);
        assert_eq!(codes("struct R { n: i64 }\nimpl R { fn $drop(self: *R) -> i64 { 0 } }"), [Code::Es07]);
    }

    #[test]
    fn nothing_is_moved_out_of_a_value_that_drops() {
        let src = "struct Inner { v: i64 }\n\
                   struct R { inner: Inner }\n\
                   impl R { fn $drop(self: *R) { } }\n\
                   fn take(i: Inner) { }\n\
                   fn f(r: R) { take(r.inner); }";
        assert_eq!(codes(src), [Code::Es12]);
    }

    #[test]
    fn an_impl_names_its_types_parameters() {
        assert_eq!(codes("struct Pair<A, B> { a: A, b: B }\nimpl Pair { fn m() { } }"), [Code::Es06]);
        assert_eq!(codes(&format!("{V}impl V<T> {{ fn m() {{ }} }}")), [Code::Es06]);
    }
}
