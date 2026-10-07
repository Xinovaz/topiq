//! Item syntax: everything that can be declared at unit scope.
//!
//! # There is no export keyword
//!
//! A unit's interface is what it declares. Privacy is the marked case: `static`
//! on an item states that it stays inside the unit's boundary, so the boundary
//! is visible in the source.
//!
//! [`Linkage::Program`] is the default, and `static` is the only linkage
//! specifier. It says nothing about storage duration; the word has one
//! meaning in Topiq.
//!
//! # Unit scope is unordered
//!
//! An item at unit scope may be used above the line that declares it. Block
//! scope is ordered; unit scope is not. So as far as name resolution is
//! concerned the item list of a [`Unit`](super::Unit) is a set, even though it
//! is stored in source order so that diagnostics can point at the right place.

use crate::intern::Symbol;
use crate::quon::ast::{DeclName, QCover, QGauge};
use crate::span::{Span, Spanned};

use super::Doc;
use super::annot::AnnotationGroup;
use super::expr::{Expr, Param};
use super::stmt::Block;
use super::ty::{Path, Type};

/// Whether an identifier at unit scope is visible outside its unit.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Linkage {
    /// Program linkage: the default. The identifier denotes the same entity in
    /// every unit of the program that imports its unit and names it.
    #[default]
    Program,
    /// Unit linkage, written `static`. The name is not exported in the unit's
    /// metadata section, and an importer naming it is ill-formed.
    Unit,
}

/// A generic parameter.
///
/// A parameter may be declared `N: const usize` and instantiated with a
/// constant expression. There is no separate species of compile-time
/// parameter: a const generic is simply a parameter whose type happens to be a
/// const type.
#[derive(Clone, PartialEq, Debug)]
pub struct GenericParam {
    /// The parameter's name.
    pub name: Spanned<Symbol>,
    /// Its type, present for a const generic such as `N: const usize`.
    pub ty: Option<Spanned<Type>>,
}

/// A function's name.
#[derive(Clone, PartialEq, Debug)]
pub enum FnName {
    /// A plain name.
    Plain(Spanned<Symbol>),
    /// `Type.method`: defines a method on a type from outside its `impl`
    /// block.
    ///
    /// This and a method written inside an `impl` are the same declaration in
    /// different clothing. Writing one both ways, or twice, is ill-formed.
    External {
        /// The type the method is added to.
        ty: Path,
        /// The method name.
        name: Spanned<Symbol>,
    },
    /// `$name`: the method behind an operator. The symbol excludes the `$`.
    Operator(Spanned<Symbol>),
}

impl FnName {
    /// The bare name.
    pub fn name(&self) -> Symbol {
        match self {
            FnName::Plain(s) | FnName::Operator(s) => s.node,
            FnName::External { name, .. } => name.node,
        }
    }

    /// Whether this is an operator method.
    pub fn is_operator(&self) -> bool {
        matches!(self, FnName::Operator(_))
    }
}

/// A function or operator definition.
#[derive(Clone, PartialEq, Debug)]
pub struct FnItem {
    /// The name.
    pub name: FnName,
    /// Generic parameters.
    pub generics: Vec<GenericParam>,
    /// Parameters. A parameter of qubit-bearing type is consumed by the
    /// call, not borrowed.
    pub params: Vec<Param>,
    /// The return type.
    pub ret: Option<Spanned<Type>>,
    /// A `where` clause: a constant boolean expression, usually built from
    /// the introspection macros.
    ///
    /// An instantiation for which it evaluates false is ill-formed at the use
    /// site, and the diagnostic names the expression that failed (`EM01`).
    ///
    /// **There are no trait bounds.** A bound here is an ordinary condition
    /// over what the compiler can already see about a type, not a declaration
    /// that the type implements an interface.
    pub where_clause: Option<Spanned<Expr>>,
    /// The body.
    pub body: Block,
    /// The signature: from `fn` to the end of the return type or `where`
    /// clause, which is everything but the body.
    pub signature: Span,
    /// The documentation of a method in an `impl` block. A function at unit
    /// scope has its documentation on its [`Item`] instead.
    pub doc: Option<Doc>,
}

/// A structure field.
#[derive(Clone, PartialEq, Debug)]
pub struct Field {
    /// Annotations on the field.
    pub annotations: Vec<Spanned<AnnotationGroup>>,
    /// The field name.
    pub name: Spanned<Symbol>,
    /// Its type.
    pub ty: Spanned<Type>,
    /// Its documentation.
    pub doc: Option<Doc>,
}

/// An enumeration variant's payload.
#[derive(Clone, PartialEq, Debug)]
pub enum VariantPayload {
    /// `(T1, T2, …)`.
    Tuple(Vec<Spanned<Type>>),
    /// `{ f: T, … }`.
    Struct(Vec<Field>),
}

/// An enumeration variant.
#[derive(Clone, PartialEq, Debug)]
pub struct Variant {
    /// The variant's name.
    pub name: Spanned<Symbol>,
    /// Its payload, absent for a unit variant.
    pub payload: Option<VariantPayload>,
    /// Its documentation.
    pub doc: Option<Doc>,
}

/// One stage of a chain.
///
/// A chain names a closed or open sequence of staged judgements, in which the
/// target base type of each stage is the source of the next. It is a
/// translation-time object: it generates no code.
#[derive(Clone, PartialEq, Debug)]
pub struct ChainStage {
    /// The operator this stage applies.
    pub operator: Spanned<Expr>,
    /// The source base type.
    pub from: Spanned<Symbol>,
    /// The target base type.
    pub to: Spanned<Symbol>,
}

/// One entry of a map locale.
#[derive(Clone, PartialEq, Debug)]
pub struct QmapEntry {
    /// The key.
    pub key: Spanned<Expr>,
    /// The entry.
    pub value: Spanned<Expr>,
}

/// A locale member.
///
/// A member declares an *interface*, never a definition. Its contract is
/// carried by the `[expect: …]` annotation written on it.
#[derive(Clone, PartialEq, Debug)]
pub struct LocaleMember {
    /// Annotations, which is where the contract lives.
    pub annotations: Vec<Spanned<AnnotationGroup>>,
    /// The member's name.
    pub name: Spanned<Symbol>,
    /// Its parameters.
    pub params: Vec<Param>,
    /// Its return type, if written.
    pub ret: Option<Spanned<Type>>,
    /// Its documentation.
    pub doc: Option<Doc>,
}

/// What an item declares.
#[derive(Clone, PartialEq, Debug)]
pub enum ItemKind {
    /// A function or operator definition.
    Fn(Box<FnItem>),
    /// `struct Name<…> { … }`.
    Struct {
        /// The name.
        name: Spanned<Symbol>,
        /// Generic parameters.
        generics: Vec<GenericParam>,
        /// The fields.
        fields: Vec<Field>,
    },
    /// `enum Name<…> { … }`.
    ///
    /// An enumeration with qubit-bearing payloads is a *quantum enumeration*,
    /// which is a sum over sectors rather than an ordinary tagged union.
    Enum {
        /// The name.
        name: Spanned<Symbol>,
        /// Generic parameters.
        generics: Vec<GenericParam>,
        /// The variants.
        variants: Vec<Variant>,
    },
    /// `impl Path<…> { … }`.
    Impl {
        /// The type being implemented on.
        path: Path,
        /// Generic parameters.
        generics: Vec<GenericParam>,
        /// The methods.
        items: Vec<FnItem>,
    },
    /// A unit-scope binding.
    ///
    /// Such an object is initialised in exactly one of two ways: by a
    /// constant expression written here, or by assignment in the unit's
    /// initialiser. There is no lazy initialisation, so no read can find it
    /// unset.
    Let {
        /// The name.
        name: Spanned<Symbol>,
        /// The declared type.
        ty: Spanned<Type>,
        /// A constant initialiser.
        init: Option<Spanned<Expr>>,
    },
    /// `type Name<…> = T;`.
    TypeAlias {
        /// The alias name.
        name: Spanned<Symbol>,
        /// Generic parameters.
        generics: Vec<GenericParam>,
        /// The aliased type.
        ty: Spanned<Type>,
    },
    /// `import path (as name)?;`.
    ///
    /// Code reuse is by `import`, which is semantic and typed. There is no
    /// textual inclusion.
    Import {
        /// The unit path.
        path: Path,
        /// An alias (e.g. `import qpu::gates as g;`).
        alias: Option<Spanned<Symbol>>,
        /// The kind asked of a `#unit any` unit, as in `import(quantum) dsp;`,
        /// as written: `classical` or `quantum`, which analysis checks.
        kind: Option<Spanned<Symbol>>,
    },
    /// `cover Name = …;`
    Cover {
        /// The name.
        name: Spanned<Symbol>,
        /// The cover.
        cover: Spanned<QCover>,
    },
    /// `gauge Name = …;`
    Gauge {
        /// The name.
        name: Spanned<Symbol>,
        /// The gauge.
        gauge: Spanned<QGauge>,
    },
    /// `base Name = RegisterType, Cover, Gauge;`
    ///
    /// **A base type carries no contract.** A contract appears in exactly two
    /// places (an ascription and an interface) and always travels
    /// explicitly, so nothing is ever claimed by association.
    Base {
        /// The name.
        name: Spanned<Symbol>,
        /// The ambient register type.
        register: Spanned<Type>,
        /// The cover's name.
        cover: Spanned<DeclName>,
        /// The gauge's name.
        gauge: Spanned<DeclName>,
    },
    /// `locale Name = Restriction { … }`
    Locale {
        /// The name.
        name: Spanned<Symbol>,
        /// The restriction type the locale is over.
        ty: Spanned<Type>,
        /// The members.
        members: Vec<LocaleMember>,
    },
    /// `chain Name = stage; stage; …;`
    Chain {
        /// The name.
        name: Spanned<Symbol>,
        /// The stages.
        stages: Vec<ChainStage>,
    },
    /// `qmap Name: Key -> Entry { … }`
    Qmap {
        /// The name.
        name: Spanned<Symbol>,
        /// The key register type.
        key: Spanned<Type>,
        /// The common entry type.
        entry: Spanned<Type>,
        /// The entries.
        entries: Vec<QmapEntry>,
    },
    /// `@static_assert(cond, "msg");` at unit scope.
    Assert {
        /// The macro call.
        call: Spanned<Expr>,
    },
}

impl ItemKind {
    /// The name this item declares.
    ///
    /// An `impl` block declares no name of its own, so it has none.
    pub fn declared_name(&self) -> Option<Symbol> {
        Some(match self {
            ItemKind::Fn(f) => f.name.name(),
            ItemKind::Struct { name, .. }
            | ItemKind::Enum { name, .. }
            | ItemKind::Let { name, .. }
            | ItemKind::TypeAlias { name, .. }
            | ItemKind::Cover { name, .. }
            | ItemKind::Gauge { name, .. }
            | ItemKind::Base { name, .. }
            | ItemKind::Locale { name, .. }
            | ItemKind::Chain { name, .. }
            | ItemKind::Qmap { name, .. } => name.node,
            ItemKind::Import { alias, path, .. } => match alias {
                Some(a) => a.node,
                None => path.last()?,
            },
            ItemKind::Impl { .. } | ItemKind::Assert { .. } => return None,
        })
    }

    /// A short noun for diagnostics.
    pub fn describe(&self) -> &'static str {
        match self {
            ItemKind::Fn(_) => "function",
            ItemKind::Struct { .. } => "structure",
            ItemKind::Enum { .. } => "enumeration",
            ItemKind::Impl { .. } => "impl block",
            ItemKind::Let { .. } => "binding",
            ItemKind::TypeAlias { .. } => "type alias",
            ItemKind::Import { .. } => "import",
            ItemKind::Cover { .. } => "cover",
            ItemKind::Gauge { .. } => "gauge",
            ItemKind::Base { .. } => "base type",
            ItemKind::Locale { .. } => "locale",
            ItemKind::Chain { .. } => "chain",
            ItemKind::Qmap { .. } => "map locale",
            ItemKind::Assert { .. } => "assertion",
        }
    }

    /// Which name space the declared name occupies.
    ///
    /// Cover, gauge, base-type, locale and chain declarations share **one**
    /// name space with `struct`, `enum` and `type`. That is why a cover and a
    /// structure cannot both be called `Bell`.
    pub fn name_space(&self) -> NameSpace {
        match self {
            ItemKind::Struct { .. }
            | ItemKind::Enum { .. }
            | ItemKind::TypeAlias { .. }
            | ItemKind::Cover { .. }
            | ItemKind::Gauge { .. }
            | ItemKind::Base { .. }
            | ItemKind::Locale { .. }
            | ItemKind::Chain { .. } => NameSpace::Type,
            ItemKind::Fn(_) | ItemKind::Let { .. } | ItemKind::Qmap { .. } => NameSpace::Value,
            ItemKind::Import { .. } => NameSpace::Value,
            ItemKind::Impl { .. } | ItemKind::Assert { .. } => NameSpace::Value,
        }
    }
}

/// The name space an item can occupy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NameSpace {
    /// Objects, functions and generic parameters.
    Value,
    /// Types, including cover, gauge, base-type, locale and chain
    /// declarations, which share this one space with `struct`, `enum` and
    /// `type`.
    Type,
}

/// One item, with its annotations and linkage.
#[derive(Clone, PartialEq, Debug)]
pub struct Item {
    /// Annotation groups written before the item, in order.
    pub annotations: Vec<Spanned<AnnotationGroup>>,
    /// Program or unit linkage.
    pub linkage: Linkage,
    /// Its documentation: the `///` comments before it, or between its
    /// annotation groups.
    pub doc: Option<Doc>,
    /// What the item declares.
    pub kind: ItemKind,
}

impl Item {
    /// Whether the item carries an annotation with this name.
    pub fn has_annotation(&self, name: Symbol) -> bool {
        self.annotations.iter().any(|g| g.node.has(name))
    }

    /// Whether the item is exported from its unit.
    pub fn is_exported(&self) -> bool {
        self.linkage == Linkage::Program
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::span::{SourceId, Span};

    fn sp<T>(node: T) -> Spanned<T> {
        Spanned::new(node, Span::new(SourceId(0), 0, 1))
    }

    fn ty(i: &mut Interner, name: &str) -> Spanned<Type> {
        sp(Type::Path {
            path: Path::single(sp(i.intern(name))),
            args: Vec::new(),
        })
    }

    #[test]
    fn program_linkage_is_the_default() {
        // an identifier at unit scope is exported by default; privacy is the
        // marked case, and `static` is the mark
        assert_eq!(Linkage::default(), Linkage::Program);
        let mut i = Interner::new();
        let item = Item {
            annotations: Vec::new(),
            linkage: Linkage::default(),
            doc: None,
            kind: ItemKind::Struct {
                name: sp(i.intern("S")),
                generics: Vec::new(),
                fields: Vec::new(),
            },
        };
        assert!(item.is_exported());
    }

    #[test]
    fn static_marks_unit_linkage() {
        let mut i = Interner::new();
        let item = Item {
            annotations: Vec::new(),
            linkage: Linkage::Unit,
            doc: None,
            kind: ItemKind::Let {
                name: sp(i.intern("SALT")),
                ty: ty(&mut i, "u32"),
                init: None,
            },
        };
        assert!(!item.is_exported());
    }

    #[test]
    fn items_report_the_name_they_declare() {
        let mut i = Interner::new();
        let s = i.intern("Vec3");
        let k = ItemKind::Struct {
            name: sp(s),
            generics: Vec::new(),
            fields: Vec::new(),
        };
        assert_eq!(k.declared_name(), Some(s));
        assert_eq!(k.describe(), "structure");
    }

    #[test]
    fn an_impl_block_declares_no_name() {
        let mut i = Interner::new();
        let k = ItemKind::Impl {
            path: Path::single(sp(i.intern("Cplx"))),
            generics: Vec::new(),
            items: Vec::new(),
        };
        assert_eq!(k.declared_name(), None);
    }

    #[test]
    fn an_import_declares_its_alias_or_its_last_segment() {
        let mut i = Interner::new();
        let plain = ItemKind::Import {
            path: Path {
                segments: vec![sp(i.intern("qpu")), sp(i.intern("gates"))],
            },
            alias: None,
            kind: None,
        };
        assert_eq!(plain.declared_name(), Some(i.intern("gates")));

        let aliased = ItemKind::Import {
            path: Path {
                segments: vec![sp(i.intern("qpu")), sp(i.intern("gates"))],
            },
            alias: Some(sp(i.intern("g"))),
            kind: None,
        };
        assert_eq!(aliased.declared_name(), Some(i.intern("g")));
    }

    #[test]
    fn cover_gauge_base_locale_and_chain_share_the_type_name_space() {
        // they share one name space with struct, enum and type,
        // so a cover named `B` and a structure named `B` would collide
        let mut i = Interner::new();
        let n = sp(i.intern("B"));
        let kinds = vec![
            ItemKind::Cover {
                name: n,
                cover: sp(QCover::Name(DeclName::local(n.node))),
            },
            ItemKind::Gauge {
                name: n,
                gauge: sp(QGauge::None),
            },
            ItemKind::Base {
                name: n,
                register: ty(&mut i, "qubit"),
                cover: sp(DeclName::local(n.node)),
                gauge: sp(DeclName::local(n.node)),
            },
            ItemKind::Locale {
                name: n,
                ty: ty(&mut i, "T"),
                members: Vec::new(),
            },
            ItemKind::Chain {
                name: n,
                stages: Vec::new(),
            },
            ItemKind::Struct {
                name: n,
                generics: Vec::new(),
                fields: Vec::new(),
            },
            ItemKind::TypeAlias {
                name: n,
                generics: Vec::new(),
                ty: ty(&mut i, "T"),
            },
        ];
        for k in kinds {
            assert_eq!(k.name_space(), NameSpace::Type, "{}", k.describe());
        }
    }

    #[test]
    fn functions_and_bindings_share_the_value_name_space() {
        let mut i = Interner::new();
        let n = sp(i.intern("x"));
        let f = ItemKind::Fn(Box::new(FnItem {
            name: FnName::Plain(n),
            generics: Vec::new(),
            params: Vec::new(),
            ret: None,
            where_clause: None,
            body: Block::empty(),
            signature: Span::synthetic(),
            doc: None,
        }));
        assert_eq!(f.name_space(), NameSpace::Value);
        let l = ItemKind::Let {
            name: n,
            ty: ty(&mut i, "u32"),
            init: None,
        };
        assert_eq!(l.name_space(), NameSpace::Value);
    }

    #[test]
    fn function_names_come_in_three_shapes() {
        // a function name is plain, an external method, or an operator method
        let mut i = Interner::new();
        let conj = i.intern("conj");
        let plain = FnName::Plain(sp(conj));
        let external = FnName::External {
            ty: Path::single(sp(i.intern("Cplx"))),
            name: sp(conj),
        };
        let op = FnName::Operator(sp(i.intern("add")));

        assert_eq!(plain.name(), conj);
        assert_eq!(external.name(), conj, "the bare name, unqualified");
        assert!(!plain.is_operator());
        assert!(op.is_operator());
        assert_ne!(plain, external, "the two forms are distinguishable");
    }

    #[test]
    fn a_where_clause_is_a_constant_expression_not_a_trait_bound() {
        // there are no trait bounds; a bound is an ordinary constant
        // condition, as in
        // `where @has_method(T, "$add") && @has_method(T, "$copy")`
        let mut i = Interner::new();
        let f = FnItem {
            name: FnName::Plain(sp(i.intern("sum"))),
            generics: vec![GenericParam {
                name: sp(i.intern("T")),
                ty: None,
            }],
            params: Vec::new(),
            ret: None,
            where_clause: Some(sp(Expr::Bool(true))),
            body: Block::empty(),
            signature: Span::synthetic(),
            doc: None,
        };
        assert!(f.where_clause.is_some());
        assert_eq!(f.generics.len(), 1);
        assert!(
            f.generics[0].ty.is_none(),
            "a plain generic parameter carries no bound"
        );
    }

    #[test]
    fn a_const_generic_is_a_parameter_of_const_type() {
        // there is no separate kind of compile-time parameter
        let mut i = Interner::new();
        let g = GenericParam {
            name: sp(i.intern("N")),
            ty: Some(sp(Type::Const(Box::new(ty(&mut i, "usize"))))),
        };
        assert!(g.ty.as_ref().unwrap().node.is_const());
    }

    #[test]
    fn annotations_are_visible_on_an_item() {
        let mut i = Interner::new();
        let entry = i.intern("entry");
        let item = Item {
            annotations: vec![sp(AnnotationGroup {
                annotations: vec![sp(super::super::annot::Annotation {
                    name: sp(entry),
                    args: Vec::new(),
                })],
            })],
            linkage: Linkage::Program,
            doc: None,
            kind: ItemKind::Struct {
                name: sp(i.intern("S")),
                generics: Vec::new(),
                fields: Vec::new(),
            },
        };
        assert!(item.has_annotation(entry));
        assert!(!item.has_annotation(i.intern("packed")));
    }

    #[test]
    fn a_chain_keeps_its_stages_in_order() {
        // the target base type of each stage is the source of
        // the next, so order is the content
        let mut i = Interner::new();
        let (t01, tpm) = (i.intern("T01"), i.intern("Tpm"));
        let stage = |from, to| ChainStage {
            operator: sp(Expr::Bool(true)),
            from: sp(from),
            to: sp(to),
        };
        let k = ItemKind::Chain {
            name: sp(i.intern("HXH")),
            stages: vec![stage(t01, tpm), stage(tpm, tpm), stage(tpm, t01)],
        };
        match k {
            ItemKind::Chain { stages, .. } => {
                assert_eq!(stages.len(), 3);
                assert_eq!(stages[0].to.node, stages[1].from.node);
                assert_eq!(stages[1].to.node, stages[2].from.node);
                assert_eq!(
                    stages[2].to.node, stages[0].from.node,
                    "this chain is closed, so it has a holonomy"
                );
            }
            _ => panic!("expected a chain"),
        }
    }
}
