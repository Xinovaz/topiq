//! Writing metadata as a TCON document.
//!
//! The document is meant to be read by people as well as by the compiler
//! (`tqc emit --stage=metadata` prints it), so it spells things out: a type is
//! written as the form it takes, `Ref("write", Array(Int("u8"), 4))`, and a
//! structure or enumeration by its position in the document's `types` list,
//! `Adt(0)`. An integer value is written as a string of its decimal digits, so
//! every value of every integer type, 128-bit ones included, reads back
//! exactly.
//!
//! ```text
//! Metadata {
//!     version: "0.2",
//!     edition: 2,
//!     unit: "geometry",
//!     conductor: 8,
//!     main: Absent,
//!     types: [ Adt { origin: "geometry", name: "Point", … } ],
//!     fns: [ Fn { name: "area", params: [Ref("shared", Adt(0))], ret: Int("i64"), constant: false } ],
//!     globals: [ … ],
//!     uses: [ … ],
//! }
//! ```

use crate::ast::Linkage;
use crate::exact::Phase;
use crate::intern::Interner;
use crate::judge::cert::{Base, Certificate, Class};
use crate::judge::cover::Cover;
use crate::judge::pauli::Pauli;
use crate::judge::record::{Claim, Declared, Judged, Published, Record, Tri};
use crate::quon::eval::Amplitudes;
use crate::sema::interface::GeometryDef;
use crate::tcon::write::{self, Doc};
use crate::tir::{Access, AdtKind, Arg, Compound, Ty, TypeTable, Value, VariantShape};

use super::{MainDecl, Metadata, UseKind};

/// The metadata as a TCON document.
pub fn encode(m: &Metadata, interner: &Interner) -> String {
    write::write(&doc(m, interner))
}

/// A text field written only when there is text, so that a unit using none
/// of what it records reads as it did before the field existed.
fn optional<'f>(fields: &mut Vec<(&'f str, Doc)>, name: &'f str, value: Option<&str>) {
    if let Some(v) = value {
        fields.push((name, Doc::text(v)));
    }
}

fn doc(m: &Metadata, interner: &Interner) -> Doc {
    let name = |s| Doc::text(interner.resolve(s));
    let types = &m.interface.types;
    let main = match &m.main {
        MainDecl::Absent => Doc::tag("Absent"),
        MainDecl::Valid => Doc::tag("Valid"),
        MainDecl::Invalid(why) => Doc::call("Invalid", vec![Doc::text(why)]),
    };
    let adts = Doc::list(types.adts(), |a| {
        let kind = match &a.kind {
            AdtKind::Struct { fields } => Doc::typed("Struct", vec![("fields", fields_doc(fields, types, interner))]),
            AdtKind::Enum { variants } => {
                let variants = Doc::list(variants, |v| {
                    let shape = match v.shape {
                        VariantShape::Unit => "Unit",
                        VariantShape::Tuple => "Tuple",
                        VariantShape::Struct => "Struct",
                    };
                    Doc::typed(
                        "Variant",
                        vec![
                            ("name", name(v.name)),
                            ("shape", Doc::tag(shape)),
                            ("fields", fields_doc(&v.fields, types, interner)),
                        ],
                    )
                });
                Doc::typed("Enum", vec![("variants", variants)])
            }
        };
        let args = Doc::list(&a.args, |&x| match x {
            Arg::Type(t) => Doc::call("T", vec![ty(t, types)]),
            Arg::Const(v) => Doc::call("N", vec![Doc::text(&v.to_string())]),
        });
        Doc::typed(
            "Adt",
            vec![
                ("origin", Doc::text(a.origin.as_deref().unwrap_or(&m.unit))),
                ("name", name(a.name)),
                ("linkage", linkage(a.linkage)),
                ("open", Doc::Bool(a.open)),
                ("opaque", Doc::Bool(a.opaque)),
                ("packed", Doc::Bool(a.repr.packed)),
                ("align", Doc::Int(a.repr.align.unwrap_or(0))),
                ("tag", Doc::text(a.repr.tag.map_or("", |t| t.name()))),
                ("args", args),
                ("kind", kind),
            ],
        )
    });
    let fns = Doc::list(&m.interface.fns, |f| {
        let method = f.method.map_or_else(
            || Doc::tag("None"),
            |m| {
                Doc::typed(
                    "Method",
                    vec![
                        ("owner", ty(Ty::Adt(m.owner), types)),
                        ("operator", Doc::Bool(m.operator)),
                        ("receiver", Doc::Bool(m.receiver)),
                    ],
                )
            },
        );
        let mut fs = vec![
            ("name", name(f.name)),
            ("params", Doc::list(&f.params, |&t| ty(t, types))),
            ("ret", ty(f.ret, types)),
            ("constant", Doc::Bool(f.constant)),
            ("linkage", linkage(f.linkage)),
            ("method", method),
            ("entry", Doc::Bool(f.entry)),
        ];
        optional(&mut fs, "symbol", f.symbol.as_deref());
        optional(&mut fs, "deprecated", f.deprecated.as_deref());
        Doc::typed("Fn", fs)
    });
    let globals = Doc::list(&m.interface.globals, |g| {
        let mut fs = vec![
            ("name", name(g.name)),
            ("ty", ty(g.ty, types)),
            ("constant", Doc::Bool(g.constant)),
            ("linkage", linkage(g.linkage)),
        ];
        if let Some(v) = &g.value {
            fs.push(("value", value(v, interner)));
        }
        optional(&mut fs, "symbol", g.symbol.as_deref());
        optional(&mut fs, "deprecated", g.deprecated.as_deref());
        Doc::typed("Global", fs)
    });
    let aliases = Doc::list(&m.interface.aliases, |a| {
        Doc::typed(
            "Alias",
            vec![("name", name(a.name)), ("ty", ty(a.ty, types)), ("linkage", linkage(a.linkage))],
        )
    });
    let uses = Doc::list(&m.uses, |u| {
        let kind = match u.kind {
            UseKind::Function => "Function",
            UseKind::Object => "Object",
            UseKind::Type => "Type",
            UseKind::Source => "Source",
            UseKind::Record => "Record",
        };
        let mut fs = vec![
            ("unit", Doc::text(&u.unit)),
            ("name", name(u.name)),
            ("kind", Doc::tag(kind)),
            ("expect", Doc::text(&u.expect)),
        ];
        if let Some(o) = &u.owner {
            fs.push(("owner", Doc::call("Owner", vec![Doc::text(&o.unit), name(o.ty), Doc::Bool(o.operator)])));
        }
        optional(&mut fs, "value", u.value.as_deref());
        Doc::typed("Use", fs)
    });
    let source = match &m.interface.source {
        None => Doc::tag("Absent"),
        Some(src) => {
            let embeds = Doc::list(&src.embeds, |(path, text)| {
                Doc::typed("Embed", vec![("path", Doc::text(path)), ("text", Doc::text(text))])
            });
            Doc::typed(
                "Source",
                vec![("path", Doc::text(&src.path)), ("text", Doc::text(&src.text)), ("embeds", embeds)],
            )
        }
    };
    let mut fields = vec![
        ("version", Doc::text(&m.version)),
        ("edition", Doc::Int(u64::from(m.edition))),
        ("unit", Doc::text(&m.unit)),
        ("conductor", Doc::Int(u64::from(m.conductor))),
        ("main", main),
        ("init", Doc::Bool(m.init)),
        ("types", adts),
        ("fns", fns),
        ("globals", globals),
        ("uses", uses),
        ("imports", Doc::list(&m.interface.imports, |u| Doc::text(u))),
        ("quantum", Doc::Bool(m.interface.quantum)),
        ("quantum_only", Doc::list(&m.interface.quantum_only, |&s| name(s))),
        ("source", source),
    ];
    // written only when there are any, as for `optional`
    if !m.interface.aliases.is_empty() {
        fields.push(("aliases", aliases));
    }
    // a `#unit any` unit's preference, empty for none
    if let Some(prefers) = m.interface.any {
        fields.push(("any", Doc::text(prefers.map_or("", crate::pp::UnitKind::name))));
    }
    if !m.interface.records.is_empty() {
        fields.push(("records", Doc::list(&m.interface.records, published)));
    }
    if !m.interface.circuits.is_empty() {
        let circuits = Doc::list(&m.interface.circuits, |c| {
            Doc::typed(
                "Entry",
                vec![
                    ("name", name(c.name)),
                    ("dynamic", Doc::Bool(c.dynamic)),
                    ("allocates", Doc::Bool(c.allocates)),
                    ("monic", Doc::list(&c.monic_slots, |&b| Doc::Bool(b))),
                    ("sig", Doc::text(&c.sig)),
                    ("qasm", Doc::Text(c.qasm.clone())),
                    ("document", Doc::Text(c.document.clone())),
                ],
            )
        });
        fields.push(("circuits", circuits));
    }
    if !m.interface.geometry.is_empty() {
        let geometry = Doc::list(&m.interface.geometry, |g| match &g.def {
            GeometryDef::Cover(c) => Doc::call("Cover", vec![name(g.name), cover(c)]),
            GeometryDef::Gauge(k) => Doc::call("Gauge", vec![name(g.name), Doc::list(&k.fiducials, state)]),
            GeometryDef::Base(b) => Doc::call("Base", vec![name(g.name), base(b)]),
        });
        fields.push(("geometry", geometry));
    }
    if !m.interface.deprecated_types.is_empty() {
        let deprecated = Doc::list(&m.interface.deprecated_types, |(n, msg)| {
            Doc::typed("Deprecated", vec![("name", name(*n)), ("message", Doc::text(msg))])
        });
        fields.push(("deprecated_types", deprecated));
    }
    Doc::typed("Metadata", fields)
}

/// What a unit publishes of an operator.
pub fn published(p: &Published) -> Doc {
    let declared = |d: &Option<Declared>| match d {
        None => Doc::tag("None"),
        Some(d) => Doc::call("Declared", vec![Doc::text(&d.name), Doc::Bool(d.program)]),
    };
    let claims = Doc::list(&p.claims, |c| match c {
        Claim::Monic => Doc::tag("Monic"),
        Claim::Unitary => Doc::tag("Unitary"),
        Claim::Contract(k) => Doc::call("Contract", vec![class(k)]),
        Claim::Frame(k) => Doc::call("Frame", vec![option(k.map(phase))]),
        Claim::Outcomes(t) => Doc::call("Outcomes", vec![Doc::text(t)]),
    });
    Doc::typed(
        "Record",
        vec![
            ("name", Doc::text(&p.name)),
            ("outcomes", Doc::text(&p.outcomes)),
            ("claims", claims),
            ("cover", declared(&p.cover)),
            ("gauge", declared(&p.gauge)),
            ("record", record(&p.record)),
        ],
    )
}

/// A digest of what a unit publishes of an operator, which tells whether
/// two copies of it are the same.
pub fn record_digest(p: &Published) -> String {
    let text = write::write(&published(p));
    format!("{:016x}", super::check::fnv1a(text.as_bytes()))
}

fn option(d: Option<Doc>) -> Doc {
    d.map_or_else(|| Doc::tag("None"), |d| Doc::call("Some", vec![d]))
}

fn phase(p: Phase) -> Doc {
    Doc::call("P", vec![Doc::Int(u64::from(p.order())), Doc::Int(u64::from(p.index()))])
}

fn phases(ps: &[Phase]) -> Doc {
    Doc::list(ps, |&p| phase(p))
}

fn class(c: &Class) -> Doc {
    match c {
        Class::Rigid => Doc::tag("Rigid"),
        Class::Flat(p) => Doc::call("Flat", vec![option(p.map(phase))]),
        Class::Locflat => Doc::tag("Locflat"),
        Class::Cyc(p) => Doc::call("Cyc", vec![option(p.as_deref().map(phases))]),
        Class::Stat(p) => Doc::call("Stat", vec![option(p.map(phase))]),
        Class::Sector(p) => Doc::call("Sector", vec![option(p.as_deref().map(phases))]),
        Class::Free => Doc::tag("Free"),
    }
}

fn tri(t: Tri) -> Doc {
    Doc::tag(match t {
        Tri::Yes => "Yes",
        Tri::No => "No",
        Tri::Unknown => "Unknown",
    })
}

fn record(r: &Record) -> Doc {
    let kernel = r
        .kernel
        .map(|k| Doc::call("Kernel", vec![Doc::Int(u64::from(k.measured)), Doc::Int(u64::from(k.forgotten))]));
    Doc::typed(
        "Judgment",
        vec![
            ("monic", tri(r.monic)),
            ("unitary", tri(r.unitary)),
            ("contract", option(r.contract.as_ref().map(class))),
            ("cert", option(r.cert.as_ref().map(judged))),
            ("sectors", option(r.sectors.as_ref().map(|s| Doc::list(s, judged)))),
            ("kernel", option(kernel)),
            ("conductor", Doc::Int(u64::from(r.fragment.conductor))),
            ("departure", option(r.fragment.departure.as_deref().map(Doc::text))),
            ("stabilizer", Doc::Bool(r.fragment.stabilizer)),
            ("undecided", option(r.undecided.as_deref().map(Doc::text))),
            ("dynamic", Doc::Bool(r.dynamic)),
        ],
    )
}

fn judged(j: &Judged) -> Doc {
    let cert = match &j.cert {
        Certificate::Points {
            images,
            schedule,
            stationary,
        } => Doc::call(
            "Points",
            vec![Doc::list(images, |&i| Doc::Int(i as u64)), phases(schedule), Doc::Bool(*stationary)],
        ),
        Certificate::Scalar(p) => Doc::call("Scalar", vec![phase(*p)]),
        Certificate::Moving => Doc::tag("Moving"),
    };
    Doc::typed("Judged", vec![("cert", cert), ("source", base(&j.source)), ("target", base(&j.target))])
}

/// A digest of a base type.
pub fn base_digest(b: &Base) -> String {
    let text = write::write(&base(b));
    format!("{:016x}", super::check::fnv1a(text.as_bytes()))
}

fn base(b: &Base) -> Doc {
    Doc::typed("Base", vec![("cover", cover(&b.cover)), ("gauge", Doc::list(&b.gauge.fiducials, state))])
}

fn cover(c: &Cover) -> Doc {
    let states = |ps: &[Amplitudes]| Doc::list(ps, state);
    match c {
        Cover::Pt(p) => Doc::call("Pt", vec![state(p)]),
        Cover::Fin(ps) => Doc::call("Fin", vec![states(ps)]),
        Cover::Span(ps) => Doc::call("Span", vec![states(ps)]),
        Cover::Subspace(ps) => Doc::call("Subspace", vec![states(ps)]),
        Cover::Code { gens, width, n } => Doc::call(
            "Code",
            vec![Doc::list(gens, |g| Doc::text(&pauli(g))), Doc::Int(*width as u64), Doc::Int(u64::from(*n))],
        ),
    }
}

/// A Pauli string as its sign and letters (e.g. `-XZI`).
fn pauli(p: &Pauli) -> String {
    let mut s = String::new();
    if p.negative {
        s.push('-');
    }
    for (&x, &z) in p.x.iter().zip(&p.z) {
        s.push(match (x, z) {
            (false, false) => 'I',
            (true, false) => 'X',
            (true, true) => 'Y',
            (false, true) => 'Z',
        });
    }
    s
}

/// A state: each basis ket with a non-zero amplitude, and the amplitude as
/// its coefficients over the powers of the root of unity, each a fraction
/// as text.
fn state(a: &Amplitudes) -> Doc {
    let terms = Doc::list(&a.terms, |(ket, c)| {
        let bits: String = ket.iter().map(|&b| if b { '1' } else { '0' }).collect();
        let c = c.embed(a.n).unwrap_or_else(|| c.clone());
        Doc::call("Term", vec![Doc::text(&bits), Doc::list(c.coeffs(), |f| Doc::text(&f.to_string()))])
    });
    Doc::typed(
        "State",
        vec![("width", Doc::Int(a.width as u64)), ("n", Doc::Int(u64::from(a.n))), ("terms", terms)],
    )
}

fn linkage(l: Linkage) -> Doc {
    Doc::text(match l {
        Linkage::Program => "program",
        Linkage::Unit => "unit",
    })
}

fn fields_doc(fields: &[crate::tir::FieldDef], types: &TypeTable, interner: &Interner) -> Doc {
    Doc::list(fields, |f| {
        Doc::typed("Field", vec![("name", Doc::text(interner.resolve(f.name))), ("ty", ty(f.ty, types))])
    })
}

/// The name of an access.
pub fn access_name(a: Access) -> &'static str {
    match a {
        Access::Write => "write",
        Access::Const => "const",
    }
}

/// A type.
fn ty(t: Ty, types: &TypeTable) -> Doc {
    match t {
        Ty::Int(i) => Doc::call("Int", vec![Doc::text(i.name())]),
        Ty::Float(f) => Doc::call("Float", vec![Doc::text(f.name())]),
        Ty::Bool => Doc::tag("Bool"),
        Ty::Dyn => Doc::tag("Dyn"),
        Ty::Qubit => Doc::tag("Qubit"),
        Ty::Char => Doc::tag("Char"),
        Ty::Void => Doc::tag("Void"),
        Ty::Never | Ty::Infer(_) => Doc::tag("Never"),
        Ty::Adt(id) => Doc::call("Adt", vec![Doc::Int(u64::from(id.0))]),
        Ty::Tuple(_) => Doc::call("Tuple", vec![Doc::list(types.as_tuple(t).expect("a tuple type"), |&e| ty(e, types))]),
        Ty::Fn(_) | Ty::Closure(_) | Ty::Circuit(_) => {
            let (params, ret) = types.as_sig(t).expect("a signature");
            let tag = match t {
                Ty::Fn(_) => "Fn",
                Ty::Closure(_) => "Closure",
                _ => "Circuit",
            };
            Doc::call(tag, vec![Doc::list(params, |&p| ty(p, types)), ty(ret, types)])
        }
        Ty::Ref(id) | Ty::Slice(id) | Ty::Array(id) | Ty::Growable(id) => match types.compound(id) {
            Compound::Ref { access, target } => {
                Doc::call("Ref", vec![Doc::text(access_name(access)), ty(target, types)])
            }
            Compound::Slice { access, elem } => {
                Doc::call("Slice", vec![Doc::text(access_name(access)), ty(elem, types)])
            }
            Compound::Array { elem, len } => Doc::call("Array", vec![ty(elem, types), Doc::Int(len)]),
            Compound::Growable { elem } => Doc::call("Growable", vec![ty(elem, types)]),
            Compound::Qmap { .. } => unreachable!("only a map locale type is a map locale's compound"),
        },
        Ty::Qmap(_) => {
            let (key, entry) = types.as_qmap(t).expect("a map locale type");
            Doc::call("Qmap", vec![Doc::Int(key), ty(entry, types)])
        }
    }
}

/// A constant's value.
fn value(v: &Value, interner: &Interner) -> Doc {
    let all = |vs: &[Value]| Doc::list(vs, |x| value(x, interner));
    match v {
        Value::Int(x, t) => Doc::call("I", vec![Doc::text(t.name()), Doc::text(&x.to_string())]),
        // a float travels as its bits, which is exact and covers the
        // infinities and NaN, for which no literal exists
        Value::Float(x, t) => Doc::call(
            "F",
            vec![Doc::text(t.name()), Doc::text(&format!("{:#x}", x.to_bits()))],
        ),
        Value::Bool(b) => Doc::call("B", vec![Doc::Bool(*b)]),
        Value::Char(c) => Doc::call("C", vec![Doc::Char(*c)]),
        Value::Void => Doc::tag("Void"),
        Value::Str(s) => Doc::call("S", vec![Doc::text(interner.resolve(*s))]),
        Value::Struct(fs) => Doc::call("Struct", vec![all(fs)]),
        Value::Enum { variant, fields } => Doc::call("Enum", vec![Doc::Int(u64::from(*variant)), all(fields)]),
        Value::Array(items) => Doc::call("Array", vec![all(items)]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::testing::metadata_of;

    #[test]
    fn metadata_reads_as_a_document() {
        let mut i = Interner::new();
        let m = metadata_of(
            "geo",
            "struct P { x: i32 }\nenum S { A, B(i64) }\nfn area(p: *P) -> i32 { p.x }\nlet LIMIT: const i32 = -7;",
            &[],
            &mut i,
        );
        let text = encode(&m, &i);
        assert!(text.starts_with("Metadata {\n"), "{text}");
        assert!(text.contains("version: \"0.2\""), "{text}");
        assert!(text.contains("params: [Ref(\"write\", Adt(0))]"), "{text}");
        assert!(text.contains("value: I(\"i32\", \"-7\")"), "{text}");
        assert!(text.contains("shape: Tuple"), "{text}");
    }
}
