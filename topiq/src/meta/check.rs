//! The canonical forms an importer records and the linker compares.
//!
//! Each is independent of how either unit happens to number its types, so the
//! same declaration gives the same text in every unit:
//!
//! - a **type** is written with every structure and enumeration qualified by
//!   the unit that declares it: `*geometry::Point`, `[u8; 4]`;
//! - a **signature** is `fn(params) -> result` in those terms;
//! - a structure's or enumeration's **layout digest** covers everything that
//!   decides where its bytes go (its fields or variants, their names and
//!   types, and its layout annotations) and, for a field holding another
//!   structure by value, that structure's own digest. A field that is a
//!   reference covers only the name of what it refers to, since a reference is
//!   two words whatever it refers to;
//! - a constant's **value digest** covers its value.
//!
//! A digest is 64 bits of FNV-1a written in hexadecimal. It detects stale
//! objects; it is not meant to resist a deliberate collision.

use std::fmt::Write;

use crate::intern::Interner;
use crate::tir::{AdtId, AdtKind, Compound, Ty, TypeTable, Value, VariantShape};

/// FNV-1a over `bytes`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A digest in the form metadata stores it.
fn hex(h: u64) -> String {
    format!("{h:016x}")
}

/// A structure's or enumeration's name, qualified by its unit. `this` is the
/// name of the unit whose table it is, for a type that table declares itself.
pub fn qualified(types: &TypeTable, id: AdtId, this: &str, interner: &Interner) -> String {
    let a = types.adt(id);
    format!(
        "{}::{}",
        a.origin.as_deref().unwrap_or(this),
        interner.resolve(a.name)
    )
}

/// A type.
pub fn type_text(types: &TypeTable, ty: Ty, this: &str, interner: &Interner) -> String {
    match ty {
        Ty::Adt(id) => qualified(types, id, this, interner),
        Ty::Ref(id) | Ty::Slice(id) | Ty::Array(id) | Ty::Growable(id) | Ty::Qmap(id) => match types.compound(id) {
            Compound::Ref { access, target } => {
                format!("{}{}", access.prefix(), type_text(types, target, this, interner))
            }
            Compound::Slice { access, elem } => {
                format!("{}[{}]", access.prefix(), type_text(types, elem, this, interner))
            }
            Compound::Array { elem, len } => format!("[{}; {len}]", type_text(types, elem, this, interner)),
            Compound::Growable { elem } => format!("[{}]", type_text(types, elem, this, interner)),
            Compound::Qmap { key, entry } => format!("qmap<[qubit; {key}], {}>", type_text(types, entry, this, interner)),
        },
        other => types.display(other, interner),
    }
}

/// A function's signature.
pub fn signature_text(types: &TypeTable, params: &[Ty], ret: Ty, this: &str, interner: &Interner) -> String {
    let ps: Vec<String> = params.iter().map(|&p| type_text(types, p, this, interner)).collect();
    format!("fn({}) -> {}", ps.join(", "), type_text(types, ret, this, interner))
}

/// An object's or constant's type, and whether it is a constant.
pub fn object_text(types: &TypeTable, ty: Ty, constant: bool, this: &str, interner: &Interner) -> String {
    let prefix = if constant { "const " } else { "" };
    format!("{prefix}{}", type_text(types, ty, this, interner))
}

/// The layout digest of a structure or enumeration.
pub fn adt_digest(types: &TypeTable, id: AdtId, this: &str, interner: &Interner) -> String {
    hex(fnv1a(adt_text(types, id, this, interner).as_bytes()))
}

/// Everything that decides where a structure's or enumeration's bytes go.
fn adt_text(types: &TypeTable, id: AdtId, this: &str, interner: &Interner) -> String {
    let a = types.adt(id);
    let mut out = String::new();
    let _ = write!(
        out,
        "{} {} packed:{} align:{} tag:{} {{",
        if a.is_struct() { "struct" } else { "enum" },
        qualified(types, id, this, interner),
        a.repr.packed,
        a.repr.align.unwrap_or(0),
        a.repr.tag.map_or("", |t| t.name())
    );
    let field = |out: &mut String, name: crate::intern::Symbol, ty: Ty| {
        let _ = write!(out, " {}: {};", interner.resolve(name), field_text(types, ty, this, interner));
    };
    match &a.kind {
        AdtKind::Struct { fields } => {
            for f in fields {
                field(&mut out, f.name, f.ty);
            }
        }
        AdtKind::Enum { variants } => {
            for v in variants {
                let shape = match v.shape {
                    VariantShape::Unit => "unit",
                    VariantShape::Tuple => "tuple",
                    VariantShape::Struct => "struct",
                };
                let _ = write!(out, " {} {shape} {{", interner.resolve(v.name));
                for f in &v.fields {
                    field(&mut out, f.name, f.ty);
                }
                out.push_str(" }");
            }
        }
    }
    out.push_str(" }");
    out
}

/// A field's type for a digest: a type held by value brings its own digest.
fn field_text(types: &TypeTable, ty: Ty, this: &str, interner: &Interner) -> String {
    match ty {
        Ty::Adt(id) => format!(
            "{}#{}",
            qualified(types, id, this, interner),
            adt_digest(types, id, this, interner)
        ),
        Ty::Array(_) => {
            let (elem, len) = types.as_array(ty).expect("an array");
            format!("[{}; {len}]", field_text(types, elem, this, interner))
        }
        other => type_text(types, other, this, interner),
    }
}

/// The digest of a constant's value.
pub fn value_digest(v: &Value, interner: &Interner) -> String {
    let mut text = String::new();
    value_text(&mut text, v, interner);
    hex(fnv1a(text.as_bytes()))
}

fn value_text(out: &mut String, v: &Value, interner: &Interner) {
    match v {
        Value::Int(x, t) => {
            let _ = write!(out, "{x}{t}");
        }
        // the bits, so that two values differ in the digest exactly when
        // they differ in memory: `0.0` and `-0.0` are not the same constant
        Value::Float(x, t) => {
            let _ = write!(out, "{:#x}{t}", x.to_bits());
        }
        Value::Bool(b) => {
            let _ = write!(out, "{b}");
        }
        Value::Char(c) => {
            let _ = write!(out, "{c:?}");
        }
        Value::Void => out.push_str("void"),
        Value::Str(s) => {
            let _ = write!(out, "{:?}", interner.resolve(*s));
        }
        Value::Struct(fs) | Value::Array(fs) => {
            out.push('(');
            for f in fs {
                value_text(out, f, interner);
                out.push(',');
            }
            out.push(')');
        }
        Value::Enum { variant, fields } => {
            let _ = write!(out, "#{variant}(");
            for f in fields {
                value_text(out, f, interner);
                out.push(',');
            }
            out.push(')');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sema::testing::analyzed_in;
    use crate::tir::{IntTy, Unit};

    /// Analyses a clean unit named `geo`, with the interner its names are in.
    fn unit(src: &str) -> (Unit, Interner) {
        let mut i = Interner::new();
        let (u, d) = analyzed_in("geo", src, &[], &mut i);
        assert!(d.iter().all(|d| !d.is_error()), "{d:?}");
        (u, i)
    }

    #[test]
    fn fnv_matches_its_reference_values() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn a_signature_names_types_by_their_unit() {
        let (u, i) = unit("struct P { x: i32 }\nfn f(p: *P, n: u8) -> [P; 2] { [P { x: 1 }, P { x: 2 }] }");
        let f = &u.fns[0];
        let params: Vec<Ty> = f.param_types().collect();
        let text = signature_text(&u.types, &params, f.ret, "geo", &i);
        assert_eq!(text, "fn(*geo::P, u8) -> [geo::P; 2]");
    }

    #[test]
    fn a_layout_digest_changes_with_the_layout_and_only_with_it() {
        let digest = |src: &str| {
            let (u, i) = unit(src);
            adt_digest(&u.types, AdtId(0), "geo", &i)
        };
        let base = digest("struct P { x: i32, y: i32 }");
        assert_eq!(base, digest("struct P { x: i32, y: i32 }\nfn unrelated() { }"));
        assert_ne!(base, digest("struct P { x: i64, y: i32 }"), "a field's type");
        assert_ne!(base, digest("[packed]\nstruct P { x: i32, y: i32 }"), "an annotation");
        assert_ne!(base, digest("struct P { y: i32, x: i32 }"), "the field order");
    }

    #[test]
    fn a_nested_structure_brings_its_digest_but_a_reference_does_not() {
        let outer = |inner: &str, field: &str| {
            let (u, i) = unit(&format!("struct Outer {{ i: {field} }}\n{inner}"));
            adt_digest(&u.types, AdtId(0), "geo", &i)
        };
        assert_ne!(
            outer("struct In { a: u8 }", "In"),
            outer("struct In { a: u16 }", "In"),
            "held by value, the inner layout matters"
        );
        assert_eq!(
            outer("struct In { a: u8 }", "*In"),
            outer("struct In { a: u16 }", "*In"),
            "through a reference, it does not"
        );
    }

    #[test]
    fn value_digests_tell_values_apart() {
        let i = Interner::new();
        let a = value_digest(&Value::Int(1, IntTy::I32), &i);
        let b = value_digest(&Value::Int(2, IntTy::I32), &i);
        assert_ne!(a, b);
        assert_eq!(a.len(), 16);
    }
}
