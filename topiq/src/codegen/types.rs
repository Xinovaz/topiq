//! Topiq types and values as LLVM sees them.
//!
//! # Values in registers
//!
//! An `N`-bit integer is an LLVM `iN`, whatever its signedness: LLVM integers
//! carry no sign, and each operation says whether it treats its operands as
//! signed. `bool` is `i1` and `char` is `i32`. A reference or a slice is a
//! two-word LLVM structure, `{ ptr, ptr }` for a reference and `{ ptr, i64 }`
//! for a slice or a reference to an array: the address, then the referent's
//! type description or the element count. `void` and `never` have no values,
//! so they have no LLVM value type.
//!
//! # Values in memory
//!
//! Structures, enumerations and fixed arrays are never LLVM values. They live
//! in memory, as plain bytes: storage for one is `[size x i8]` aligned as
//! [`crate::tir::layout`] says, and a field is reached by adding its offset to
//! the address. LLVM's own struct layout is never relied on, so the layout
//! rules are applied in exactly one place.
//!
//! A constant aggregate (a unit-scope object's initial value, a constant
//! structure) is built as a packed LLVM structure whose pieces sit at exactly
//! the offsets the layout gives, with zero bytes in between. A growable array
//! in a constant has its buffer in the image too, header and all; nothing
//! ever frees or grows it, since a constant is only read.

use std::collections::BTreeMap;

use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::types::{BasicType, BasicTypeEnum, FloatType, IntType, StructType};
use inkwell::values::{BasicValueEnum, IntValue, PointerValue};

use crate::tir::layout::{self, Layout};
use crate::tir::{AdtKind, Compound, FloatTy, IntTy, Ty, TypeTable, Value};

/// The LLVM integer type of a Topiq integer type.
pub fn int<'ctx>(cx: &'ctx Context, t: IntTy) -> IntType<'ctx> {
    match t.bits {
        8 => cx.i8_type(),
        16 => cx.i16_type(),
        32 => cx.i32_type(),
        _ => cx.i64_type(),
    }
}

/// The LLVM floating type of a Topiq floating type.
pub fn float<'ctx>(cx: &'ctx Context, t: FloatTy) -> FloatType<'ctx> {
    match t {
        FloatTy::F32 => cx.f32_type(),
        FloatTy::F64 => cx.f64_type(),
    }
}

/// `{ ptr, ptr }`: a reference, carrying the referent's type description.
pub fn fat_ref(cx: &Context) -> StructType<'_> {
    let p = cx.ptr_type(AddressSpace::default());
    cx.struct_type(&[p.into(), p.into()], false)
}

/// `{ ptr, i64 }`: a slice, or a reference to an array, carrying a count.
pub fn fat_slice(cx: &Context) -> StructType<'_> {
    let p = cx.ptr_type(AddressSpace::default());
    cx.struct_type(&[p.into(), cx.i64_type().into()], false)
}

/// Whether a reference type's second word is a count: a reference to an
/// array.
pub fn counts(types: &TypeTable, ty: Ty) -> bool {
    match ty {
        Ty::Slice(_) => true,
        Ty::Ref(_) => types.as_ref(ty).is_some_and(|(_, t)| matches!(t, Ty::Array(_))),
        _ => false,
    }
}

/// The LLVM type of a value held in registers, or `None` for a type whose
/// values live in memory or that has no values.
pub fn scalar<'ctx>(cx: &'ctx Context, types: &TypeTable, ty: Ty) -> Option<BasicTypeEnum<'ctx>> {
    match ty {
        Ty::Int(t) => Some(int(cx, t).into()),
        Ty::Float(t) => Some(float(cx, t).into()),
        Ty::Bool => Some(cx.bool_type().into()),
        Ty::Char => Some(cx.i32_type().into()),
        Ty::Ref(_) | Ty::Slice(_) => Some(if counts(types, ty) {
            fat_slice(cx).into()
        } else {
            fat_ref(cx).into()
        }),
        // the address of element 0, then the count; a circuit handle is the
        // characters of its operator's name, held the same way
        Ty::Growable(_) | Ty::Circuit(_) => Some(fat_slice(cx).into()),
        Ty::Fn(_) => Some(cx.ptr_type(AddressSpace::default()).into()),
        // the environment's address, then the code's; a dynamic value's
        // address, then its table
        Ty::Closure(_) | Ty::Dyn => Some(fat_ref(cx).into()),
        Ty::Adt(_) | Ty::Array(_) | Ty::Tuple(_) | Ty::Void | Ty::Never => None,
        Ty::Qubit | Ty::Qmap(_) => {
            unreachable!("a classical unit holds no qubits or map locales, and only classical units generate code")
        }
        Ty::Infer(_) => unreachable!("analysis settles every type before code generation"),
    }
}

/// Storage for a value of `ty`: its scalar type, or bytes of its size.
pub fn storage<'ctx>(cx: &'ctx Context, types: &TypeTable, ty: Ty) -> BasicTypeEnum<'ctx> {
    scalar(cx, types, ty).unwrap_or_else(|| bytes(cx, layout::of(types, ty)).into())
}

/// `[size x i8]`.
pub fn bytes(cx: &Context, l: Layout) -> inkwell::types::ArrayType<'_> {
    cx.i8_type().array_type(l.size as u32)
}

/// An integer constant of type `t`.
///
/// The value is passed to LLVM as its two's-complement bit pattern at the
/// type's width, so a negative value never relies on LLVM truncating a wider
/// pattern.
pub fn const_int<'ctx>(cx: &'ctx Context, t: IntTy, v: i128) -> IntValue<'ctx> {
    let mask = if t.bits >= 64 {
        u128::from(u64::MAX)
    } else {
        (1u128 << t.bits) - 1
    };
    int(cx, t).const_int(((v as u128) & mask) as u64, false)
}

/// What of a constant lies outside it, placed in the image by whoever builds
/// the constant.
pub enum Part<'v> {
    /// A string's characters.
    Str(crate::intern::Symbol),
    /// A growable array's elements, placed after a header as a buffer of the
    /// array's own would be.
    Elements(&'v [Value]),
}

/// A scalar constant in register form, or `None` for `void` and for values
/// that live in memory. A string's characters and a growable array's
/// elements need placing somewhere, which `parts` does, returning their
/// address and how many there are.
pub fn const_scalar<'ctx>(
    cx: &'ctx Context,
    v: &Value,
    parts: &mut impl FnMut(Part<'_>) -> (PointerValue<'ctx>, u64),
) -> Option<BasicValueEnum<'ctx>> {
    match v {
        Value::Int(x, t) => Some(const_int(cx, *t, *x).into()),
        Value::Float(x, t) => Some(float(cx, *t).const_float(*x).into()),
        Value::Bool(b) => Some(cx.bool_type().const_int(u64::from(*b), false).into()),
        Value::Char(c) => Some(cx.i32_type().const_int(u64::from(u32::from(*c)), false).into()),
        Value::Str(s) => {
            let (p, n) = parts(Part::Str(*s));
            let len = cx.i64_type().const_int(n, false);
            Some(fat_slice(cx).const_named_struct(&[p.into(), len.into()]).into())
        }
        // an array in register form is a growable one: a fixed array lives
        // in memory
        Value::Array(items) => {
            let (p, n) = parts(Part::Elements(items));
            let len = cx.i64_type().const_int(n, false);
            Some(fat_slice(cx).const_named_struct(&[p.into(), len.into()]).into())
        }
        Value::Void | Value::Struct(_) | Value::Enum { .. } => None,
    }
}

/// The bytes of `items`, each of type `elem`, one after another at the
/// element size, as the elements of a buffer are laid out.
pub fn const_elements<'ctx>(
    cx: &'ctx Context,
    types: &TypeTable,
    items: &[Value],
    elem: Ty,
    parts: &mut impl FnMut(Part<'_>) -> (PointerValue<'ctx>, u64),
) -> BasicValueEnum<'ctx> {
    let stride = layout::of(types, elem).size;
    let mut pieces = BTreeMap::new();
    for (i, item) in items.iter().enumerate() {
        flatten(cx, types, item, elem, i as u64 * stride, &mut pieces, parts);
    }
    packed(cx, pieces, stride * items.len() as u64)
}

/// The bytes of a constant of type `ty`, as an LLVM constant laid out exactly
/// as [`crate::tir::layout`] says: a packed structure of the pieces at their
/// offsets, zero bytes between them, and padding to the full size.
pub fn const_blob<'ctx>(
    cx: &'ctx Context,
    types: &TypeTable,
    v: &Value,
    ty: Ty,
    parts: &mut impl FnMut(Part<'_>) -> (PointerValue<'ctx>, u64),
) -> BasicValueEnum<'ctx> {
    let mut pieces = BTreeMap::new();
    flatten(cx, types, v, ty, 0, &mut pieces, parts);
    packed(cx, pieces, layout::of(types, ty).size)
}

/// Scalar pieces at their offsets, with zero bytes between them and after
/// them up to `size`, as one packed structure.
fn packed<'ctx>(cx: &'ctx Context, pieces: BTreeMap<u64, (BasicValueEnum<'ctx>, u64)>, size: u64) -> BasicValueEnum<'ctx> {
    let mut out: Vec<BasicValueEnum<'ctx>> = Vec::new();
    let mut at = 0u64;
    let i8 = cx.i8_type();
    for (offset, (value, width)) in pieces {
        if offset > at {
            out.push(i8.array_type((offset - at) as u32).const_zero().into());
        }
        out.push(value);
        at = offset + width;
    }
    if size > at {
        out.push(i8.array_type((size - at) as u32).const_zero().into());
    }
    cx.const_struct(&out, true).into()
}

/// Collects the scalar pieces of a constant, keyed by their offset, with the
/// width each occupies.
fn flatten<'ctx>(
    cx: &'ctx Context,
    types: &TypeTable,
    v: &Value,
    ty: Ty,
    base: u64,
    out: &mut BTreeMap<u64, (BasicValueEnum<'ctx>, u64)>,
    parts: &mut impl FnMut(Part<'_>) -> (PointerValue<'ctx>, u64),
) {
    match (v, ty) {
        (Value::Struct(fields), Ty::Adt(id)) => {
            let l = layout::structure(types, id);
            let def = types.adt(id);
            for ((f, offset), decl) in fields.iter().zip(&l.offsets).zip(def.fields()) {
                flatten(cx, types, f, decl.ty, base + offset, out, parts);
            }
        }
        (Value::Enum { variant, fields }, Ty::Adt(id)) => {
            let l = layout::enumeration(types, id);
            let tag = const_int(cx, l.tag, i128::from(*variant));
            out.insert(base, (tag.into(), u64::from(l.tag.bits / 8)));
            let AdtKind::Enum { variants } = &types.adt(id).kind else {
                unreachable!("an enumeration value of an enumeration type");
            };
            let var = &variants[*variant as usize];
            for ((f, offset), decl) in fields.iter().zip(&l.variants[*variant as usize]).zip(&var.fields) {
                flatten(cx, types, f, decl.ty, base + offset, out, parts);
            }
        }
        (Value::Array(items), Ty::Array(id)) => {
            let Compound::Array { elem, .. } = types.compound(id) else {
                unreachable!("an array id names an array");
            };
            let stride = layout::of(types, elem).size;
            for (i, item) in items.iter().enumerate() {
                flatten(cx, types, item, elem, base + i as u64 * stride, out, parts);
            }
        }
        (Value::Bool(b), _) => {
            // in memory a `bool` is a whole byte, 0 or 1
            out.insert(base, (cx.i8_type().const_int(u64::from(*b), false).into(), 1));
        }
        (scalar_value, _) => {
            if let Some(c) = const_scalar(cx, scalar_value, parts) {
                let width = layout::of(types, ty).size;
                out.insert(base, (c, width));
            }
        }
    }
}

/// The LLVM value type a constant blob of `ty` has, for declaring storage
/// that holds one.
pub fn blob_type<'ctx>(value: BasicValueEnum<'ctx>) -> BasicTypeEnum<'ctx> {
    value.get_type().as_basic_type_enum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Linkage;
    use crate::intern::Interner;
    use crate::span::Span;
    use crate::tir::{AdtDef, FieldDef, Repr, VariantDef, VariantShape};
    use inkwell::targets::TargetData;

    fn no_strings<'ctx>(_: Part<'_>) -> (PointerValue<'ctx>, u64) {
        unreachable!("no strings or growable arrays in these constants")
    }

    fn target_data() -> TargetData {
        TargetData::create("e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128")
    }

    #[test]
    fn integers_map_to_their_width() {
        let cx = Context::create();
        for t in IntTy::ALL {
            assert_eq!(int(&cx, t).get_bit_width(), u32::from(t.bits), "{t}");
        }
    }

    #[test]
    fn values_in_memory_have_no_register_type() {
        let cx = Context::create();
        let mut types = TypeTable::new();
        assert!(scalar(&cx, &types, Ty::Void).is_none());
        assert!(scalar(&cx, &types, Ty::Never).is_none());
        let arr = types.array(Ty::Bool, 3);
        assert!(scalar(&cx, &types, arr).is_none());
        assert_eq!(scalar(&cx, &types, Ty::Char).unwrap().into_int_type().get_bit_width(), 32);
    }

    #[test]
    fn references_and_slices_are_two_words_of_the_right_kind() {
        let cx = Context::create();
        let mut types = TypeTable::new();
        let r = types.reference(crate::tir::Access::Write, Ty::Bool);
        let arr = types.array(Ty::Bool, 2);
        let ra = types.reference(crate::tir::Access::Write, arr);
        let s = types.string();
        assert!(!counts(&types, r));
        assert!(counts(&types, ra));
        assert!(counts(&types, s));
        let td = target_data();
        for t in [r, ra, s] {
            let llvm = scalar(&cx, &types, t).unwrap();
            assert_eq!(td.get_abi_size(&llvm), 16);
            assert_eq!(td.get_abi_alignment(&llvm), 8);
        }
    }

    #[test]
    fn constants_keep_their_bit_pattern() {
        let cx = Context::create();
        let minus_one = const_int(&cx, IntTy::I8, -1);
        assert_eq!(minus_one.get_zero_extended_constant(), Some(0xFF));
        assert_eq!(minus_one.get_sign_extended_constant(), Some(-1));
        let big = const_int(&cx, IntTy::U64, i128::from(u64::MAX));
        assert_eq!(big.get_zero_extended_constant(), Some(u64::MAX));
        let min = const_int(&cx, IntTy::I64, i128::from(i64::MIN));
        assert_eq!(min.get_sign_extended_constant(), Some(i64::MIN));
    }

    #[test]
    fn a_constant_blob_is_exactly_the_size_of_its_layout() {
        let cx = Context::create();
        let mut i = Interner::new();
        let mut types = TypeTable::new();
        let field = |ty| FieldDef {
            name: crate::intern::Symbol::EMPTY,
            ty,
            span: Span::synthetic(),
        };
        let s = types.add_adt(AdtDef {
            args: Vec::new(),
            name: i.intern("H"),
            origin: None,
            linkage: Linkage::Program,
            open: false,
            opaque: false,
            kind: AdtKind::Struct {
                fields: vec![field(Ty::Int(IntTy::U8)), field(Ty::Int(IntTy::U32)), field(Ty::Bool)],
            },
            repr: Repr::default(),
            span: Span::synthetic(),
        });
        let e = types.add_adt(AdtDef {
            args: Vec::new(),
            name: i.intern("E"),
            origin: None,
            linkage: Linkage::Program,
            open: false,
            opaque: false,
            kind: AdtKind::Enum {
                variants: vec![
                    VariantDef {
                        name: i.intern("A"),
                        shape: VariantShape::Unit,
                        fields: vec![],
                        span: Span::synthetic(),
                    },
                    VariantDef {
                        name: i.intern("B"),
                        shape: VariantShape::Tuple,
                        fields: vec![field(Ty::Int(IntTy::U64))],
                        span: Span::synthetic(),
                    },
                ],
            },
            repr: Repr::default(),
            span: Span::synthetic(),
        });
        let arr = types.array(Ty::Adt(s), 3);
        let td = target_data();
        let h = Value::Struct(vec![Value::Int(1, IntTy::U8), Value::Int(2, IntTy::U32), Value::Bool(true)]);
        let cases = [
            (h.clone(), Ty::Adt(s)),
            (
                Value::Enum {
                    variant: 1,
                    fields: vec![Value::Int(9, IntTy::U64)],
                },
                Ty::Adt(e),
            ),
            (
                Value::Enum {
                    variant: 0,
                    fields: vec![],
                },
                Ty::Adt(e),
            ),
            (Value::Array(vec![h.clone(), h.clone(), h]), arr),
        ];
        for (v, ty) in cases {
            let blob = const_blob(&cx, &types, &v, ty, &mut no_strings);
            let size = td.get_abi_size(&blob_type(blob));
            assert_eq!(size, layout::of(&types, ty).size, "{v:?}");
        }
    }
}
