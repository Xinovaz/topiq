//! Structures and enumerations as analysis records them.
//!
//! A declaration's fields and variants are kept in the order written, because
//! that order is the layout order: the first field of a structure is at its
//! lowest address, and an enumeration's variants are numbered from 0 in the
//! order they appear. Nothing reorders them, so a program that describes a
//! file format or a device register with a structure gets exactly the layout
//! it wrote.

use crate::ast::Linkage;
use crate::intern::Symbol;
use crate::span::Span;

use super::ty::{Arg, IntTy, Ty};

/// How a structure or enumeration is laid out, from its annotations.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Repr {
    /// `[packed]`: no padding between fields, so a field may sit at any byte.
    pub packed: bool,
    /// `[align: n]`: the whole type's alignment raised to `n` bytes.
    pub align: Option<u64>,
    /// `[repr: uN]`: the discriminant type of an enumeration, instead of the
    /// smallest unsigned type that can number its variants.
    pub tag: Option<IntTy>,
}

/// One field of a structure or of an enumeration variant.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FieldDef {
    /// The name, or [`Symbol::EMPTY`] for a positional field of a tuple
    /// variant such as `Circle(i64)`.
    pub name: Symbol,
    /// The field's type.
    pub ty: Ty,
    /// Where it was declared.
    pub span: Span,
}

/// How a variant's payload is written.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VariantShape {
    /// `Empty`: no payload.
    Unit,
    /// `Circle(i64)`: positional fields.
    Tuple,
    /// `Rect { w: i64, h: i64 }`: named fields.
    Struct,
}

/// One variant of an enumeration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VariantDef {
    /// The variant's name.
    pub name: Symbol,
    /// How its payload is written.
    pub shape: VariantShape,
    /// Its fields, in the order written; empty for a unit variant.
    pub fields: Vec<FieldDef>,
    /// Where it was declared.
    pub span: Span,
}

/// What kind of type a declaration introduces.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum AdtKind {
    /// `struct Name { … }`.
    Struct {
        /// The fields.
        fields: Vec<FieldDef>,
    },
    /// `enum Name { … }`: a discriminant, then the payload of whichever
    /// variant the value is.
    Enum {
        /// The variants.
        variants: Vec<VariantDef>,
    },
}

/// A structure or enumeration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AdtDef {
    /// The name as declared.
    pub name: Symbol,
    /// The unit that declared it, when that is not the unit this table
    /// belongs to. Two definitions are the same type exactly when they have
    /// the same origin and name.
    pub origin: Option<String>,
    /// Whether units that import the declaring unit may name it.
    pub linkage: Linkage,
    /// Whether other units may add methods to it, as `[open]` says.
    pub open: bool,
    /// Whether it is kept out of documents and out of `clone`, as
    /// `[tcon: opaque]` says.
    pub opaque: bool,
    /// The generic arguments this instance was made with, empty for a type
    /// declared without generic parameters. `Pair<i32, bool>` and
    /// `Pair<u8, u8>` share a name and differ here.
    pub args: Vec<Arg>,
    /// Fields or variants.
    pub kind: AdtKind,
    /// Layout annotations.
    pub repr: Repr,
    /// Where it was declared, or a synthetic span for a type read from
    /// another unit's metadata.
    pub span: Span,
}

impl AdtDef {
    /// A structure (`is_struct`) or enumeration declared at `span`, whose
    /// fields or variants are not yet resolved.
    pub fn unresolved(name: Symbol, origin: Option<String>, linkage: Linkage, args: Vec<Arg>, is_struct: bool, span: Span) -> AdtDef {
        AdtDef {
            name,
            origin,
            linkage,
            open: false,
            opaque: false,
            args,
            kind: if is_struct { AdtKind::Struct { fields: Vec::new() } } else { AdtKind::Enum { variants: Vec::new() } },
            repr: Repr::default(),
            span,
        }
    }

    /// Whether this is a structure.
    pub fn is_struct(&self) -> bool {
        matches!(self.kind, AdtKind::Struct { .. })
    }

    /// The fields of a structure; empty for an enumeration.
    pub fn fields(&self) -> &[FieldDef] {
        match &self.kind {
            AdtKind::Struct { fields } => fields,
            AdtKind::Enum { .. } => &[],
        }
    }

    /// The type of each field of a structure, or of each variant of an
    /// enumeration.
    pub fn field_types(&self) -> impl Iterator<Item = Ty> + '_ {
        let variants = self.variants().iter().flat_map(|v| &v.fields);
        self.fields().iter().chain(variants).map(|f| f.ty)
    }

    /// The variants of an enumeration; empty for a structure.
    pub fn variants(&self) -> &[VariantDef] {
        match &self.kind {
            AdtKind::Struct { .. } => &[],
            AdtKind::Enum { variants } => variants,
        }
    }

    /// The index of a structure's field by name.
    pub fn field_index(&self, name: Symbol) -> Option<u32> {
        self.fields()
            .iter()
            .position(|f| f.name == name)
            .map(|i| i as u32)
    }

    /// The index of an enumeration's variant by name.
    pub fn variant_index(&self, name: Symbol) -> Option<u32> {
        self.variants()
            .iter()
            .position(|v| v.name == name)
            .map(|i| i as u32)
    }

    /// The word a message uses for this kind of type.
    pub fn noun(&self) -> &'static str {
        if self.is_struct() {
            "structure"
        } else {
            "enumeration"
        }
    }
}

impl VariantDef {
    /// The index of a named field of this variant.
    pub fn field_index(&self, name: Symbol) -> Option<u32> {
        self.fields
            .iter()
            .position(|f| f.name == name)
            .map(|i| i as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::span::SourceId;

    fn sp() -> Span {
        Span::new(SourceId(0), 0, 1)
    }

    fn field(name: Symbol, ty: Ty) -> FieldDef {
        FieldDef { name, ty, span: sp() }
    }

    #[test]
    fn a_structure_finds_its_fields_by_name_in_declaration_order() {
        let mut i = Interner::new();
        let (x, y) = (i.intern("x"), i.intern("y"));
        let s = AdtDef {
            args: Vec::new(),
            name: i.intern("P"),
            origin: None,
            linkage: Linkage::Program,
            open: false,
            opaque: false,
            kind: AdtKind::Struct {
                fields: vec![field(x, Ty::Bool), field(y, Ty::Char)],
            },
            repr: Repr::default(),
            span: sp(),
        };
        assert!(s.is_struct());
        assert_eq!(s.field_index(y), Some(1));
        assert_eq!(s.field_index(i.intern("z")), None);
        assert!(s.variants().is_empty());
        assert_eq!(s.noun(), "structure");
    }

    #[test]
    fn an_enumeration_numbers_its_variants_in_order() {
        let mut i = Interner::new();
        let (a, b) = (i.intern("A"), i.intern("B"));
        let variant = |name| VariantDef {
            name,
            shape: VariantShape::Unit,
            fields: vec![],
            span: sp(),
        };
        let e = AdtDef {
            args: Vec::new(),
            name: i.intern("E"),
            origin: Some("other".to_owned()),
            linkage: Linkage::Program,
            open: false,
            opaque: false,
            kind: AdtKind::Enum {
                variants: vec![variant(a), variant(b)],
            },
            repr: Repr::default(),
            span: sp(),
        };
        assert_eq!(e.variant_index(b), Some(1));
        assert!(e.fields().is_empty());
        assert_eq!(e.noun(), "enumeration");
    }
}
