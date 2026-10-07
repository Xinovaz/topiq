//! Resolving structure and enumeration declarations.
//!
//! A declaration's fields keep the order they were written in, because that
//! is their layout order, and an enumeration's variants are numbered from 0
//! in the order written. This module turns each field's written type into a
//! [`Ty`] and reads the three layout annotations:
//!
//! - `[packed]`: no padding between fields;
//! - `[align: n]`: the whole type aligned to `n` bytes, a power of two;
//! - `[repr: uN]`: an enumeration's discriminant stored as `uN` instead of
//!   the smallest unsigned type that numbers its variants.
//!
//! A quantum enumeration (one with a qubit-bearing field) is one register:
//! a tag numbering its variants and a payload they share. A classical field
//! of a variant is held there as the basis state of qubits of its own, one
//! for each bit, so that it is in superposition with its variant; it may be
//! an integer or a `bool`, or an array, tuple or structure of them
//! (`ES06` otherwise).

use crate::ast::{self, AnnotationGroup, AnnArg, ItemKind, StdAnnotation, VariantPayload};
use crate::diag::{Code, Diagnostic, Limit};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{AdtId, AdtKind, FieldDef, IntTy, Repr, Ty, VariantDef, VariantShape};

use super::items::{self, Target, UnitCx};
use super::report;
use super::types;

/// Resolves a declaration's fields or variants, and its layout annotations.
pub fn resolve(
    cx: &mut UnitCx<'_>,
    id: AdtId,
    kind: &ItemKind,
    annotations: &[Spanned<AnnotationGroup>],
) -> (AdtKind, Repr) {
    let is_enum = matches!(kind, ItemKind::Enum { .. });
    let target = if is_enum {
        Target::Enumeration
    } else {
        Target::Structure
    };
    items::check_annotations(annotations, target, cx.quantum, cx.interner, &mut cx.diags);
    cx.note_type_annotations(id, annotations);
    let mut repr = representation(cx, annotations, is_enum);
    let span = cx.unit.types.adt(id).span;
    let kind = match kind {
        ItemKind::Struct { fields, .. } => {
            if let Some(d) = items::check_count(Limit::StructFields, fields.len(), span) {
                cx.report(d);
            }
            AdtKind::Struct {
                fields: named_fields(cx, fields),
            }
        }
        ItemKind::Enum { variants, .. } => {
            if let Some(d) = items::check_count(Limit::EnumVariants, variants.len(), span) {
                cx.report(d);
            }
            let mut out: Vec<VariantDef> = Vec::with_capacity(variants.len());
            for v in variants {
                if let Some(earlier) = out.iter().find(|o| o.name == v.name.node) {
                    let d = report::duplicate(
                        v.name.span,
                        earlier.span,
                        cx.interner.resolve(v.name.node),
                        "in one enumeration",
                    );
                    cx.report(d);
                }
                let (shape, fields) = match &v.payload {
                    None => (VariantShape::Unit, Vec::new()),
                    Some(VariantPayload::Tuple(tys)) => {
                        let fields = tys
                            .iter()
                            .map(|t| FieldDef {
                                name: Symbol::EMPTY,
                                ty: field_type(cx, t),
                                span: t.span,
                            })
                            .collect();
                        (VariantShape::Tuple, fields)
                    }
                    Some(VariantPayload::Struct(fields)) => (VariantShape::Struct, named_fields(cx, fields)),
                };
                out.push(VariantDef {
                    name: v.name.node,
                    shape,
                    fields,
                    span: v.name.span,
                });
            }
            if let Some(tag) = repr.tag {
                let capacity = 1u128 << tag.bits;
                if out.len() as u128 > capacity {
                    let d = Diagnostic::new(Code::Es06)
                        .with_message(format!(
                            "a `{tag}` discriminant can number {capacity} variants, but this \
                             enumeration has {}",
                            out.len()
                        ))
                        .at(span)
                        .with_help("choose a wider type in `[repr: …]`, or remove it");
                    cx.report(d);
                    repr.tag = None;
                }
            }
            // a quantum enumeration's variants are sectors of one register:
            // what a variant holds is a state, so a classical field is held
            // as a basis state of qubits of its own, one per bit
            let quantum = out.iter().flat_map(|v| &v.fields).any(|f| cx.unit.types.is_quantum(f.ty));
            if quantum {
                for f in out.iter().flat_map(|v| &v.fields) {
                    let t = f.ty;
                    if t != Ty::Never && !cx.unit.types.is_quantum(t) && cx.unit.types.basis_bits(t).is_none() {
                        let what = cx.unit.types.display(t, cx.interner);
                        let d = Diagnostic::new(Code::Es06)
                            .with_message(format!("a quantum enumeration's variant cannot hold a `{what}`"))
                            .at(f.span)
                            .with_note(
                                "a quantum enumeration is one register whose variants are orthogonal \
                                 sectors, in superposition; a classical field is held there as the \
                                 basis state of qubits, one for each of its bits",
                            )
                            .with_help("hold integers or `bool`s, or arrays, tuples and structures of them");
                        cx.report(d);
                    }
                }
            }
            AdtKind::Enum { variants: out }
        }
        _ => unreachable!("only structures and enumerations are resolved here"),
    };
    (kind, repr)
}

/// Resolves named fields, reporting a name given twice.
fn named_fields(cx: &mut UnitCx<'_>, fields: &[ast::Field]) -> Vec<FieldDef> {
    let mut out: Vec<FieldDef> = Vec::with_capacity(fields.len());
    for f in fields {
        items::check_annotations(&f.annotations, Target::Field, cx.quantum, cx.interner, &mut cx.diags);
        if let Some(earlier) = out.iter().find(|o| o.name == f.name.node) {
            let d = report::duplicate(
                f.name.span,
                earlier.span,
                cx.interner.resolve(f.name.node),
                "in one structure",
            );
            cx.report(d);
        }
        let ty = field_type(cx, &f.ty);
        out.push(FieldDef {
            name: f.name.node,
            ty,
            span: f.name.span,
        });
    }
    out
}

/// A field's type.
fn field_type(cx: &mut UnitCx<'_>, t: &Spanned<ast::Type>) -> Ty {
    match types::resolve(cx, t) {
        Ok(r) => {
            if r.constant {
                cx.report(
                    Diagnostic::new(Code::Es06)
                        .with_message("a field cannot be `const`")
                        .at(t.span)
                        .with_note(
                            "a field is part of a value, and `const` describes a whole value known \
                             during translation, which has no storage when the program runs; part \
                             of a value cannot lack the storage the rest has",
                        )
                        .with_help("declare the whole structure value `const` instead"),
                );
            }
            if r.ty == Ty::Void {
                cx.report(items::void_binding(t.span));
                return Ty::Never;
            }
            r.ty
        }
        Err(d) => {
            cx.report(*d);
            Ty::Never
        }
    }
}

/// Reads `[packed]`, `[align: n]` and `[repr: uN]`.
fn representation(cx: &mut UnitCx<'_>, annotations: &[Spanned<AnnotationGroup>], is_enum: bool) -> Repr {
    let mut repr = Repr::default();
    for g in annotations {
        for a in &g.node.annotations {
            let name = cx.interner.resolve(a.node.name.node);
            match StdAnnotation::from_name(name) {
                Some(StdAnnotation::Packed) => repr.packed = true,
                Some(StdAnnotation::Align) => {
                    if let Some(n) = single_arg(cx, &a.node.args, a.span, "[align: n]")
                        .and_then(|arg| alignment(cx, arg, a.span))
                    {
                        repr.align = Some(n);
                    }
                }
                Some(StdAnnotation::Repr) if is_enum => {
                    if let Some(arg) = single_arg(cx, &a.node.args, a.span, "[repr: uN]") {
                        repr.tag = discriminant(cx, arg, a.span);
                    }
                }
                _ => {}
            }
        }
    }
    repr
}

/// The one argument an annotation takes.
fn single_arg<'x>(
    cx: &mut UnitCx<'_>,
    args: &'x [Spanned<AnnArg>],
    span: Span,
    form: &str,
) -> Option<&'x Spanned<AnnArg>> {
    match args {
        [one] => Some(one),
        _ => {
            cx.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("this annotation takes exactly one argument, as in `{form}`"))
                    .at(span),
            );
            None
        }
    }
}

/// The byte count of `[align: n]`.
fn alignment(cx: &mut UnitCx<'_>, arg: &Spanned<AnnArg>, span: Span) -> Option<u64> {
    let AnnArg::Expr(e) = &arg.node else {
        cx.report(
            Diagnostic::new(Code::Es06)
                .with_message("`[align: n]` takes a number of bytes")
                .at(arg.span),
        );
        return None;
    };
    let n = cx.const_usize(e, "the alignment in `[align: n]`")?;
    if n == 0 || !n.is_power_of_two() || n > 1 << 29 {
        cx.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!("an alignment must be a power of two, and {n} is not"))
                .at(span)
                .with_help("use 1, 2, 4, 8, 16 or another power of two"),
        );
        return None;
    }
    Some(n)
}

/// The discriminant type of `[repr: uN]`.
fn discriminant(cx: &mut UnitCx<'_>, arg: &Spanned<AnnArg>, span: Span) -> Option<IntTy> {
    let name = match &arg.node {
        AnnArg::Type(t) => match &t.node {
            ast::Type::Path { path, .. } if path.is_simple() => path.last(),
            _ => None,
        },
        AnnArg::Expr(e) => match &e.node {
            ast::Expr::Path { path, .. } if path.is_simple() => path.last(),
            _ => None,
        },
        _ => None,
    };
    let t = name.and_then(|n| IntTy::from_name(cx.interner.resolve(n)));
    match t {
        Some(t) if !t.signed && !t.pointer_sized => Some(t),
        _ => {
            cx.report(
                Diagnostic::new(Code::Es06)
                    .with_message("a discriminant is stored as `u8`, `u16`, `u32` or `u64`")
                    .at(span)
                    .with_note("variants are numbered from 0, so the type must be unsigned"),
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};
    use crate::tir::{AdtId, IntTy, Ty, VariantShape, layout};

    #[test]
    fn fields_keep_their_order_and_types() {
        let u = check("struct Header { tag: u8, len: u32, name: *[char] }");
        let d = u.types.adt(AdtId(0));
        let names: Vec<Ty> = d.fields().iter().map(|f| f.ty).collect();
        assert_eq!(names[0], Ty::Int(IntTy::U8));
        assert_eq!(names[1], Ty::Int(IntTy::U32));
        assert_eq!(layout::structure(&u.types, AdtId(0)).offsets, [0, 4, 8]);
    }

    #[test]
    fn variants_record_their_shape() {
        let u = check("enum Shape { Empty, Circle(i64), Rect { w: i64, h: i64 } }");
        let v = u.types.adt(AdtId(0)).variants();
        assert_eq!(v[0].shape, VariantShape::Unit);
        assert_eq!(v[1].shape, VariantShape::Tuple);
        assert_eq!(v[1].fields.len(), 1);
        assert_eq!(v[2].shape, VariantShape::Struct);
    }

    #[test]
    fn layout_annotations_are_read() {
        let u = check("[packed; align: 8]\nstruct S { a: u8, b: u32 }\n[repr: u32]\nenum E { A, B }");
        let s = u.types.adt(AdtId(0));
        assert!(s.repr.packed);
        assert_eq!(s.repr.align, Some(8));
        assert_eq!(u.types.adt(AdtId(1)).repr.tag, Some(IntTy::U32));
    }

    #[test]
    fn a_bad_alignment_or_discriminant_is_refused() {
        assert_eq!(codes("[align: 3]\nstruct S { }"), [Code::Es06]);
        assert_eq!(codes("[repr: i8]\nenum E { A }"), [Code::Es06]);
    }

    #[test]
    fn names_given_twice_are_duplicates() {
        assert_eq!(codes("struct S { a: u8, a: u8 }"), [Code::Es05]);
        assert_eq!(codes("enum E { A, A }"), [Code::Es05]);
        assert_eq!(codes("enum E { A { x: u8, x: u8 } }"), [Code::Es05]);
    }

    #[test]
    fn a_const_field_is_refused() {
        assert_eq!(codes("struct S { a: const u8 }"), [Code::Es06]);
    }
}
