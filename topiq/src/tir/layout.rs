//! Where every byte of a value goes.
//!
//! Layout is part of the language, not a choice the compiler makes, so that a
//! structure describing a file header or a device register means the same
//! bytes everywhere. This module is its one definition: code generation
//! addresses fields at the offsets computed here, `@sizeof` and `@alignof`
//! report them, and a unit's metadata digests them so that two units compiled
//! against different versions of a type are caught when they are linked.
//!
//! # The rules
//!
//! - An integer of `N` bits occupies `N / 8` bytes and is aligned to its
//!   size; `bool` is one byte; `char` is four.
//! - A reference or slice is two words: an address, and either the
//!   referent's type description or a count.
//! - `[T; N]` is `N` copies of `T` with no gaps, since a type's size is always
//!   a multiple of its alignment.
//! - A structure's fields are placed **in the order written**, each at the
//!   next offset that suits its alignment. The structure is aligned to its
//!   most demanding field and padded at the end to a multiple of that.
//!   `[packed]` removes all padding and alignment; `[align: n]` raises the
//!   alignment to `n`.
//! - An enumeration is a discriminant (the smallest unsigned integer that
//!   can number its variants, unless `[repr: uN]` says otherwise), followed by
//!   room for its largest variant. Every variant's fields start at the same
//!   offset, laid out among themselves like a structure's.

use super::adt::{AdtKind, Repr};
use super::table::{Compound, TypeTable};
use super::ty::{AdtId, IntTy, Ty};

/// The size and alignment of a type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Layout {
    /// How many bytes a value occupies, always a multiple of `align`.
    pub size: u64,
    /// The boundary a value must start on: a power of two.
    pub align: u64,
}

impl Layout {
    /// A layout of `size` bytes aligned to `align`.
    pub const fn new(size: u64, align: u64) -> Layout {
        Layout { size, align }
    }

    /// The layout of something with no bytes at all.
    pub const EMPTY: Layout = Layout::new(0, 1);
}

/// The layout of a structure, or of one enumeration variant's fields.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FieldsLayout {
    /// The whole.
    pub layout: Layout,
    /// Each field's offset from the start, in field order.
    pub offsets: Vec<u64>,
}

/// The layout of an enumeration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EnumLayout {
    /// The whole.
    pub layout: Layout,
    /// The discriminant's type. The discriminant is always at offset 0.
    pub tag: IntTy,
    /// Where every variant's payload begins.
    pub payload: u64,
    /// For each variant, the offset of each of its fields from the start of
    /// the enumeration.
    pub variants: Vec<Vec<u64>>,
}

/// Rounds `n` up to a multiple of `align`.
pub fn align_up(n: u64, align: u64) -> u64 {
    n.div_ceil(align) * align
}

/// The discriminant type an enumeration of `count` variants gets without a
/// `[repr: uN]`: the smallest unsigned integer that can number them all.
pub fn default_tag(count: usize) -> IntTy {
    if count <= 1 << 8 {
        IntTy::U8
    } else if count <= 1 << 16 {
        IntTy::U16
    } else {
        IntTy::U32
    }
}

/// The layout of `ty`.
///
/// A structure that contains itself has no finite layout. Analysis reports
/// that before anything asks; should it be asked anyway, the inner occurrence
/// counts as empty so that this always terminates.
pub fn of(types: &TypeTable, ty: Ty) -> Layout {
    Walker {
        types,
        stack: Vec::new(),
    }
    .of(ty)
}

/// The layout of a structure's fields.
///
/// # Panics
///
/// If `id` is not a structure.
pub fn structure(types: &TypeTable, id: AdtId) -> FieldsLayout {
    Walker {
        types,
        stack: vec![id],
    }
    .structure(id)
}

/// Where the parts of a type that holds several values in a row begin: a
/// structure's fields, or a tuple's elements.
///
/// # Panics
///
/// If `ty` is neither a structure nor a tuple.
pub fn fields_of(types: &TypeTable, ty: Ty) -> FieldsLayout {
    match ty {
        Ty::Adt(id) => structure(types, id),
        Ty::Tuple(_) => {
            let elems: Vec<Ty> = types.as_tuple(ty).expect("a tuple type").to_vec();
            Walker {
                types,
                stack: Vec::new(),
            }
            .fields(&elems, Repr::default())
        }
        other => panic!("{other:?} has no fields"),
    }
}

/// The layout of an enumeration.
///
/// # Panics
///
/// If `id` is not an enumeration.
pub fn enumeration(types: &TypeTable, id: AdtId) -> EnumLayout {
    Walker {
        types,
        stack: vec![id],
    }
    .enumeration(id)
}

/// Whether `id` contains itself by value, directly or through other
/// structures, enumerations or arrays, which would make it infinitely large.
/// A reference to itself is fine: a reference is two words whatever it points
/// at.
pub fn contains_itself(types: &TypeTable, id: AdtId) -> bool {
    let mut seen = vec![id];
    types.adt(id).field_types().any(|f| holds(types, f, id, &mut seen))
}

/// Whether `from` contains `to` by value, or is it.
pub fn reaches(types: &TypeTable, from: AdtId, to: AdtId) -> bool {
    holds(types, Ty::Adt(from), to, &mut Vec::new())
}

/// Whether a value of `ty` holds `target` inline (is one, or holds one
/// through fields, variants or elements), looking inside each of `seen` no
/// more.
fn holds(types: &TypeTable, ty: Ty, target: AdtId, seen: &mut Vec<AdtId>) -> bool {
    match ty {
        Ty::Adt(a) if a == target => true,
        Ty::Adt(a) if seen.contains(&a) => false,
        Ty::Adt(a) => {
            seen.push(a);
            types.adt(a).field_types().any(|f| holds(types, f, target, seen))
        }
        Ty::Array(_) => holds(types, types.as_array(ty).expect("an array type").0, target, seen),
        _ => false,
    }
}

struct Walker<'a> {
    types: &'a TypeTable,
    /// The structures and enumerations being laid out, outermost first.
    stack: Vec<AdtId>,
}

impl Walker<'_> {
    fn of(&mut self, ty: Ty) -> Layout {
        match ty {
            Ty::Int(t) => {
                let bytes = u64::from(t.bits / 8);
                Layout::new(bytes, bytes)
            }
            Ty::Float(t) => {
                let bytes = u64::from(t.bits() / 8);
                Layout::new(bytes, bytes)
            }
            // an undecided literal never reaches layout; if one did, it would
            // become an `i64`
            Ty::Infer(_) => Layout::new(8, 8),
            Ty::Bool => Layout::new(1, 1),
            Ty::Char => Layout::new(4, 4),
            // a qubit is a wire of a circuit, and a map locale a table of a
            // quantum unit: neither is a classical object
            Ty::Void | Ty::Never | Ty::Qubit | Ty::Qmap(_) => Layout::EMPTY,
            Ty::Ref(_) | Ty::Slice(_) | Ty::Closure(_) | Ty::Growable(_) | Ty::Circuit(_) | Ty::Dyn => Layout::new(16, 8),
            Ty::Fn(_) => Layout::new(8, 8),
            Ty::Array(id) => match self.types.compound(id) {
                Compound::Array { elem, len } => {
                    let e = self.of(elem);
                    Layout::new(e.size * len, e.align)
                }
                _ => unreachable!("an array id names an array"),
            },
            Ty::Tuple(_) => {
                let elems: Vec<Ty> = self.types.as_tuple(ty).expect("a tuple type").to_vec();
                self.fields(&elems, Repr::default()).layout
            }
            Ty::Adt(id) => {
                if self.stack.contains(&id) {
                    return Layout::EMPTY;
                }
                self.stack.push(id);
                let l = if self.types.adt(id).is_struct() {
                    self.structure(id).layout
                } else {
                    self.enumeration(id).layout
                };
                self.stack.pop();
                l
            }
        }
    }

    /// Lays fields out one after another, as a structure does.
    fn fields(&mut self, tys: &[Ty], repr: Repr) -> FieldsLayout {
        let mut offset = 0;
        let mut align = 1;
        let mut offsets = Vec::with_capacity(tys.len());
        for &t in tys {
            let l = self.of(t);
            if !repr.packed {
                offset = align_up(offset, l.align);
                align = align.max(l.align);
            }
            offsets.push(offset);
            offset += l.size;
        }
        if let Some(a) = repr.align {
            align = align.max(a);
        }
        FieldsLayout {
            layout: Layout::new(align_up(offset, align), align),
            offsets,
        }
    }

    fn structure(&mut self, id: AdtId) -> FieldsLayout {
        let def = self.types.adt(id);
        let AdtKind::Struct { fields } = &def.kind else {
            panic!("{id:?} is not a structure");
        };
        let tys: Vec<Ty> = fields.iter().map(|f| f.ty).collect();
        let repr = def.repr;
        self.fields(&tys, repr)
    }

    fn enumeration(&mut self, id: AdtId) -> EnumLayout {
        let def = self.types.adt(id);
        let AdtKind::Enum { variants } = &def.kind else {
            panic!("{id:?} is not an enumeration");
        };
        let repr = def.repr;
        let tag = repr.tag.unwrap_or_else(|| default_tag(variants.len()));
        let tag_size = u64::from(tag.bits / 8);
        let inner = Repr {
            packed: repr.packed,
            align: None,
            tag: None,
        };
        let payloads: Vec<FieldsLayout> = variants
            .iter()
            .map(|v| {
                let tys: Vec<Ty> = v.fields.iter().map(|f| f.ty).collect();
                self.fields(&tys, inner)
            })
            .collect();
        let payload_align = if repr.packed {
            1
        } else {
            payloads.iter().map(|p| p.layout.align).max().unwrap_or(1)
        };
        let payload = align_up(tag_size, payload_align);
        let largest = payloads.iter().map(|p| p.layout.size).max().unwrap_or(0);
        let mut align = if repr.packed { 1 } else { tag_size.max(payload_align) };
        if let Some(a) = repr.align {
            align = align.max(a);
        }
        EnumLayout {
            layout: Layout::new(align_up(payload + largest, align), align),
            tag,
            payload,
            variants: payloads
                .into_iter()
                .map(|p| p.offsets.into_iter().map(|o| payload + o).collect())
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Linkage;
    use crate::intern::{Interner, Symbol};
    use crate::span::Span;
    use crate::tir::adt::{AdtDef, FieldDef, VariantDef, VariantShape};
    use crate::tir::ty::Access;

    fn field(ty: Ty) -> FieldDef {
        FieldDef {
            name: Symbol::EMPTY,
            ty,
            span: Span::synthetic(),
        }
    }

    fn strukt(t: &mut TypeTable, i: &mut Interner, fields: Vec<Ty>, repr: Repr) -> AdtId {
        t.add_adt(AdtDef {
            args: Vec::new(),
            name: i.intern("S"),
            origin: None,
            linkage: Linkage::Program,
            open: false,
            opaque: false,
            kind: AdtKind::Struct {
                fields: fields.into_iter().map(field).collect(),
            },
            repr,
            span: Span::synthetic(),
        })
    }

    fn enumeration_of(t: &mut TypeTable, i: &mut Interner, variants: Vec<Vec<Ty>>, repr: Repr) -> AdtId {
        t.add_adt(AdtDef {
            args: Vec::new(),
            name: i.intern("E"),
            origin: None,
            linkage: Linkage::Program,
            open: false,
            opaque: false,
            kind: AdtKind::Enum {
                variants: variants
                    .into_iter()
                    .map(|fs| VariantDef {
                        name: Symbol::EMPTY,
                        shape: if fs.is_empty() {
                            VariantShape::Unit
                        } else {
                            VariantShape::Tuple
                        },
                        fields: fs.into_iter().map(field).collect(),
                        span: Span::synthetic(),
                    })
                    .collect(),
            },
            repr,
            span: Span::synthetic(),
        })
    }

    const U8: Ty = Ty::Int(IntTy::U8);
    const U16: Ty = Ty::Int(IntTy::U16);
    const U32: Ty = Ty::Int(IntTy::U32);
    const U64: Ty = Ty::Int(IntTy::U64);

    #[test]
    fn primitives_are_aligned_to_their_size() {
        let t = TypeTable::new();
        assert_eq!(of(&t, U8), Layout::new(1, 1));
        assert_eq!(of(&t, Ty::Int(IntTy::I64)), Layout::new(8, 8));
        assert_eq!(of(&t, Ty::Bool), Layout::new(1, 1));
        assert_eq!(of(&t, Ty::Char), Layout::new(4, 4));
        assert_eq!(of(&t, Ty::Void), Layout::EMPTY);
    }

    #[test]
    fn references_and_slices_are_two_words() {
        let mut t = TypeTable::new();
        let r = t.reference(Access::Write, U8);
        let s = t.string();
        assert_eq!(of(&t, r), Layout::new(16, 8));
        assert_eq!(of(&t, s), Layout::new(16, 8));
    }

    #[test]
    fn fields_are_placed_in_order_with_padding() {
        // `{ tag: u8, len: u32 }`: three bytes of padding after `tag`
        let mut t = TypeTable::new();
        let mut i = Interner::new();
        let s = strukt(&mut t, &mut i, vec![U8, U32], Repr::default());
        let l = structure(&t, s);
        assert_eq!(l.offsets, [0, 4]);
        assert_eq!(l.layout, Layout::new(8, 4));
    }

    #[test]
    fn a_structure_is_padded_to_its_alignment() {
        // `{ a: u64, b: u8 }` is 16 bytes, so an array of them stays aligned
        let mut t = TypeTable::new();
        let mut i = Interner::new();
        let s = strukt(&mut t, &mut i, vec![U64, U8], Repr::default());
        assert_eq!(of(&t, Ty::Adt(s)), Layout::new(16, 8));
        let arr = t.array(Ty::Adt(s), 3);
        assert_eq!(of(&t, arr), Layout::new(48, 8));
    }

    #[test]
    fn packed_removes_padding_and_align_raises_alignment() {
        let mut t = TypeTable::new();
        let mut i = Interner::new();
        let packed = strukt(
            &mut t,
            &mut i,
            vec![U8, U32],
            Repr {
                packed: true,
                ..Repr::default()
            },
        );
        let l = structure(&t, packed);
        assert_eq!(l.offsets, [0, 1]);
        assert_eq!(l.layout, Layout::new(5, 1));

        let aligned = strukt(
            &mut t,
            &mut i,
            vec![U8],
            Repr {
                align: Some(16),
                ..Repr::default()
            },
        );
        assert_eq!(of(&t, Ty::Adt(aligned)), Layout::new(16, 16));

        let both = strukt(
            &mut t,
            &mut i,
            vec![U8, U32],
            Repr {
                packed: true,
                align: Some(8),
                tag: None,
            },
        );
        assert_eq!(of(&t, Ty::Adt(both)), Layout::new(8, 8));
    }

    #[test]
    fn an_enumeration_is_a_tag_then_its_largest_variant() {
        // `{ A, B(u32), C(u8, u16) }`: a u8 tag, payload at 4, largest 4 bytes
        let mut t = TypeTable::new();
        let mut i = Interner::new();
        let e = enumeration_of(&mut t, &mut i, vec![vec![], vec![U32], vec![U8, U16]], Repr::default());
        let l = enumeration(&t, e);
        assert_eq!(l.tag, IntTy::U8);
        assert_eq!(l.payload, 4);
        assert_eq!(l.variants, [vec![], vec![4], vec![4, 6]]);
        assert_eq!(l.layout, Layout::new(8, 4));
    }

    #[test]
    fn a_fieldless_enumeration_is_just_its_tag() {
        let mut t = TypeTable::new();
        let mut i = Interner::new();
        let e = enumeration_of(&mut t, &mut i, vec![vec![], vec![], vec![]], Repr::default());
        assert_eq!(of(&t, Ty::Adt(e)), Layout::new(1, 1));
        let wide = enumeration_of(
            &mut t,
            &mut i,
            vec![vec![], vec![]],
            Repr {
                tag: Some(IntTy::U32),
                ..Repr::default()
            },
        );
        assert_eq!(enumeration(&t, wide).tag, IntTy::U32);
        assert_eq!(of(&t, Ty::Adt(wide)), Layout::new(4, 4));
    }

    #[test]
    fn the_default_tag_is_the_smallest_that_numbers_every_variant() {
        assert_eq!(default_tag(0), IntTy::U8);
        assert_eq!(default_tag(256), IntTy::U8);
        assert_eq!(default_tag(257), IntTy::U16);
    }

    #[test]
    fn a_type_containing_itself_is_detected_but_a_reference_to_itself_is_not() {
        let mut t = TypeTable::new();
        let mut i = Interner::new();
        let s = strukt(&mut t, &mut i, vec![], Repr::default());
        // `struct S { next: S }`
        if let AdtKind::Struct { fields } = &mut t.adt_mut(s).kind {
            fields.push(field(Ty::Adt(s)));
        }
        assert!(contains_itself(&t, s));
        assert_eq!(of(&t, Ty::Adt(s)), Layout::EMPTY, "laying it out still terminates");

        let n = strukt(&mut t, &mut i, vec![], Repr::default());
        let r = t.reference(Access::Write, Ty::Adt(n));
        if let AdtKind::Struct { fields } = &mut t.adt_mut(n).kind {
            fields.push(field(r));
        }
        assert!(!contains_itself(&t, n));
        assert_eq!(of(&t, Ty::Adt(n)), Layout::new(16, 8));
    }
}
