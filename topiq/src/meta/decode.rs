//! Reading metadata back from its TCON text.
//!
//! The text is parsed with the ordinary TCON parser and then read field by
//! field. Metadata comes from a file on disk that may have been damaged or
//! written by another version of the compiler, so every step checks what it
//! finds and a problem is an error naming what was wrong, never a panic.

use chumsky::Parser;

use crate::ast::Linkage;
use crate::exact::{Cyclo, Frac, Phase};
use crate::intern::Interner;
use crate::judge::cert::{Base, Certificate, Class};
use crate::judge::cover::{Cover, Gauge};
use crate::judge::pauli::Pauli;
use crate::judge::record::{Claim, Declared, Fragment, Judged, Kernel, Published, Record, Tri};
use crate::quon::eval::Amplitudes;
use crate::sema::interface::EntryCircuit;
use crate::parse::input::{Cx, TokenStream, eoi_span, stream};
use crate::sema::Interface;
use crate::sema::interface::{ExportAlias, ExportFn, ExportGeometry, ExportGlobal, GeometryDef, Source};
use crate::source::Spliced;
use crate::span::{SourceId, Span, Spanned};
use crate::tcon::ast::{Scalar, Value as Tcon};
use crate::tir::{
    Access, AdtDef, AdtId, AdtKind, Arg, FieldDef, MethodOf, FloatTy, IntTy, Repr, Ty, TypeTable, Value, VariantDef,
    VariantShape,
};

use super::{MainDecl, Metadata, MethodUse, Use, UseKind};

/// Metadata that could not be read.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DecodeError(pub String);

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DecodeError {}

type Read<T> = Result<T, DecodeError>;

fn bad<T>(what: impl Into<String>) -> Read<T> {
    Err(DecodeError(what.into()))
}

/// The file id spans in metadata text are given. Nothing reports against
/// them, so any id serves.
const TEXT: SourceId = SourceId(u32::MAX);

/// Reads metadata from its text, interning names into `interner`.
///
/// # Errors
///
/// A [`DecodeError`] saying what is malformed.
pub fn decode(text: &str, interner: &mut Interner) -> Read<Metadata> {
    let spliced = Spliced::from_text(text);
    let lexed = crate::lex::lex(TEXT, &spliced, interner);
    if let Some(d) = lexed.diagnostics.first() {
        return bad(format!("the metadata is not well-formed text: {}", d.message));
    }
    let eoi = eoi_span(TEXT, &lexed.tokens);
    let null = interner.intern("null");
    let parsed = {
        let cx = Cx::new(interner);
        crate::tcon::parse::document::<TokenStream>(cx, null)
            .parse(stream(&lexed.tokens, eoi))
            .into_result()
    };
    let doc = match parsed {
        Ok(d) => d,
        Err(e) => return bad(format!("the metadata does not parse: {}", e[0])),
    };
    Reader { interner }.metadata(&doc.value.node)
}

struct Reader<'i> {
    interner: &'i mut Interner,
}

impl Reader<'_> {
    fn field<'v>(&self, v: &'v Tcon, name: &str) -> Read<&'v Tcon> {
        let Some(sym) = self.interner.get(name) else {
            return bad(format!("the field `{name}` is missing"));
        };
        match v.field(sym) {
            Some(x) => Ok(&x.node),
            None => bad(format!("the field `{name}` is missing")),
        }
    }

    fn has(&self, v: &Tcon, name: &str) -> bool {
        self.interner.get(name).is_some_and(|s| v.field(s).is_some())
    }

    /// The field `name` read by `read`.
    fn optional<T>(&mut self, v: &Tcon, name: &str, read: impl FnOnce(&mut Self, &Tcon) -> Read<T>) -> Read<Option<T>> {
        if !self.has(v, name) {
            return Ok(None);
        }
        let f = self.field(v, name)?;
        read(self, f).map(Some)
    }

    /// The text field `name`.
    fn optional_string(&mut self, v: &Tcon, name: &str) -> Read<Option<String>> {
        self.optional(v, name, |s, x| s.string(x))
    }

    /// Each item of the list `v`, read by `read`.
    fn each<T>(&mut self, v: &Tcon, mut read: impl FnMut(&mut Self, &Tcon) -> Read<T>) -> Read<Vec<T>> {
        self.list(v)?.iter().map(|x| read(self, &x.node)).collect()
    }

    /// Each item of the list field `name`, read by `read`; none if `v` lacks
    /// it.
    fn each_optional<T>(&mut self, v: &Tcon, name: &str, read: impl FnMut(&mut Self, &Tcon) -> Read<T>) -> Read<Vec<T>> {
        Ok(self.optional(v, name, |s, x| s.each(x, read))?.unwrap_or_default())
    }

    fn string(&self, v: &Tcon) -> Read<String> {
        match v {
            Tcon::Scalar(Scalar::Str { value, .. }) => Ok(self.interner.resolve(*value).to_owned()),
            other => bad(format!("expected a string, found {}", other.describe())),
        }
    }

    fn int(&self, v: &Tcon) -> Read<u64> {
        match v {
            Tcon::Scalar(Scalar::Int { raw, base, .. }) => {
                let digits: String = self.interner.resolve(*raw).chars().filter(|c| *c != '_').collect();
                u64::from_str_radix(&digits, base.radix()).or_else(|_| bad(format!("`{digits}` is not a number")))
            }
            other => bad(format!("expected a number, found {}", other.describe())),
        }
    }

    fn boolean(&self, v: &Tcon) -> Read<bool> {
        match v {
            Tcon::Scalar(Scalar::Bool(b)) => Ok(*b),
            other => bad(format!("expected true or false, found {}", other.describe())),
        }
    }

    fn list<'v>(&self, v: &'v Tcon) -> Read<&'v [Spanned<Tcon>]> {
        match v {
            Tcon::Array(xs) => Ok(xs),
            other => bad(format!("expected a list, found {}", other.describe())),
        }
    }

    /// A tagged form `Tag(payload…)`: its tag and payload.
    fn tagged<'v>(&self, v: &'v Tcon) -> Read<(&str, &'v [Spanned<Tcon>])> {
        match v {
            Tcon::Enum { path, payload } if path.is_simple() => {
                Ok((self.interner.resolve(path.segments[0].node), payload))
            }
            other => bad(format!("expected a tagged form, found {}", other.describe())),
        }
    }

    fn sym(&mut self, v: &Tcon) -> Read<crate::intern::Symbol> {
        let s = self.string(v)?;
        Ok(self.interner.intern(&s))
    }

    fn metadata(&mut self, v: &Tcon) -> Read<Metadata> {
        let unit = self.string(self.field(v, "unit")?)?;
        let main = match self.tagged(self.field(v, "main")?)? {
            ("Absent", []) => MainDecl::Absent,
            ("Valid", []) => MainDecl::Valid,
            ("Invalid", [why]) => MainDecl::Invalid(self.string(&why.node)?),
            (other, _) => return bad(format!("`{other}` is not a kind of `main`")),
        };
        let mut types = TypeTable::new();
        let adts = self.list(self.field(v, "types")?)?;
        // every type is declared before any is filled in, so that fields may
        // refer to any of them by position
        for a in adts {
            let a = &a.node;
            let origin = self.string(self.field(a, "origin")?)?;
            let name = self.sym(self.field(a, "name")?)?;
            let linkage = self.linkage(self.field(a, "linkage")?)?;
            let tag = self.string(self.field(a, "tag")?)?;
            let align = self.int(self.field(a, "align")?)?;
            let repr = Repr {
                packed: self.boolean(self.field(a, "packed")?)?,
                align: (align != 0).then_some(align),
                tag: if tag.is_empty() {
                    None
                } else {
                    Some(IntTy::from_name(&tag).map_or_else(|| bad(format!("`{tag}` is not an integer type")), Ok)?)
                },
            };
            let is_struct = matches!(self.field(a, "kind")?, Tcon::Typed { path, .. } if self.interner.resolve(path.segments[0].node) == "Struct");
            let mut def = AdtDef::unresolved(name, Some(origin), linkage, Vec::new(), is_struct, Span::synthetic());
            def.open = self.boolean(self.field(a, "open")?)?;
            def.opaque = self.has(a, "opaque") && self.boolean(self.field(a, "opaque")?)?;
            def.repr = repr;
            types.add_adt(def);
        }
        let count = adts.len();
        let table = &mut types;
        for (i, a) in adts.iter().enumerate() {
            let kind_doc = self.field(&a.node, "kind")?;
            let kind = if table.adt(AdtId(i as u32)).is_struct() {
                AdtKind::Struct {
                    fields: self.fields(self.field(kind_doc, "fields")?, table, count)?,
                }
            } else {
                let variants = self.each(self.field(kind_doc, "variants")?, |s, vd| {
                    let shape = match s.tagged(s.field(vd, "shape")?)? {
                        ("Unit", []) => VariantShape::Unit,
                        ("Tuple", []) => VariantShape::Tuple,
                        ("Struct", []) => VariantShape::Struct,
                        (other, _) => return bad(format!("`{other}` is not a variant shape")),
                    };
                    Ok(VariantDef {
                        name: s.sym(s.field(vd, "name")?)?,
                        shape,
                        fields: s.fields(s.field(vd, "fields")?, table, count)?,
                        span: Span::synthetic(),
                    })
                })?;
                AdtKind::Enum { variants }
            };
            table.adt_mut(AdtId(i as u32)).kind = kind;
            let args = self.each(self.field(&a.node, "args")?, |s, x| match s.tagged(x)? {
                ("T", [t]) => Ok(Arg::Type(s.ty(&t.node, table, count)?)),
                ("N", [n]) => {
                    let n = s.string(&n.node)?;
                    Ok(Arg::Const(n.parse().or_else(|_| bad(format!("`{n}` is not an integer")))?))
                }
                (other, _) => bad(format!("`{other}` is not a generic argument")),
            })?;
            table.adt_mut(AdtId(i as u32)).args = args;
        }

        let fns = self.each(self.field(v, "fns")?, |s, f| {
            Ok(ExportFn {
                params: s.each(s.field(f, "params")?, |s, p| s.ty(p, table, count))?,
                name: s.sym(s.field(f, "name")?)?,
                ret: s.ty(s.field(f, "ret")?, table, count)?,
                constant: s.boolean(s.field(f, "constant")?)?,
                linkage: s.linkage(s.field(f, "linkage")?)?,
                method: s.method_of(s.field(f, "method")?, table, count)?,
                entry: s.boolean(s.field(f, "entry")?)?,
                symbol: s.optional_string(f, "symbol")?,
                deprecated: s.optional_string(f, "deprecated")?,
            })
        })?;
        let globals = self.each(self.field(v, "globals")?, |s, g| {
            Ok(ExportGlobal {
                value: s.optional(g, "value", Self::value)?,
                name: s.sym(s.field(g, "name")?)?,
                ty: s.ty(s.field(g, "ty")?, table, count)?,
                constant: s.boolean(s.field(g, "constant")?)?,
                linkage: s.linkage(s.field(g, "linkage")?)?,
                symbol: s.optional_string(g, "symbol")?,
                deprecated: s.optional_string(g, "deprecated")?,
            })
        })?;
        // written only when there are any, so older documents read as having none
        let aliases = self
            .optional(v, "aliases", |s, list| {
                s.each(list, |s, a| {
                    Ok(ExportAlias {
                        name: s.sym(s.field(a, "name")?)?,
                        ty: s.ty(s.field(a, "ty")?, table, count)?,
                        linkage: s.linkage(s.field(a, "linkage")?)?,
                    })
                })
            })?
            .unwrap_or_default();
        let uses = self.each(self.field(v, "uses")?, |s, u| {
            let kind = match s.tagged(s.field(u, "kind")?)? {
                ("Function", []) => UseKind::Function,
                ("Object", []) => UseKind::Object,
                ("Type", []) => UseKind::Type,
                ("Source", []) => UseKind::Source,
                ("Record", []) => UseKind::Record,
                (other, _) => return bad(format!("`{other}` is not a kind of item")),
            };
            let value = s.optional_string(u, "value")?;
            let owner = s.optional(u, "owner", |s, o| match s.tagged(o)? {
                ("Owner", [unit, ty, operator]) => {
                    let unit = s.string(&unit.node)?;
                    let operator = s.boolean(&operator.node)?;
                    Ok(MethodUse {
                        unit,
                        ty: s.sym(&ty.node)?,
                        operator,
                    })
                }
                (other, _) => bad(format!("`{other}` is not a method's owner")),
            })?;
            Ok(Use {
                owner,
                unit: s.string(s.field(u, "unit")?)?,
                name: s.sym(s.field(u, "name")?)?,
                kind,
                expect: s.string(s.field(u, "expect")?)?,
                value,
            })
        })?;
        let imports = self.each(self.field(v, "imports")?, |s, u| s.string(u))?;
        let quantum_only = self.each(self.field(v, "quantum_only")?, Self::sym)?;
        let source_doc = self.field(v, "source")?;
        let source = match source_doc {
            Tcon::Typed { .. } => {
                let embeds = self.each(self.field(source_doc, "embeds")?, |s, e| {
                    Ok((s.string(s.field(e, "path")?)?, s.string(s.field(e, "text")?)?))
                })?;
                Some(Source {
                    path: self.string(self.field(source_doc, "path")?)?,
                    text: self.string(self.field(source_doc, "text")?)?,
                    embeds,
                })
            }
            other => match self.tagged(other)? {
                ("Absent", []) => None,
                (tag, _) => return bad(format!("`{tag}` is not a kind of source")),
            },
        };
        let deprecated_types = self.each_optional(v, "deprecated_types", |s, d| {
            Ok((s.sym(s.field(d, "name")?)?, s.string(s.field(d, "message")?)?))
        })?;
        let records = self.each_optional(v, "records", Self::published)?;
        let geometry = self.each_optional(v, "geometry", Self::geometry)?;
        let circuits = self.each_optional(v, "circuits", |s, c| {
            Ok(EntryCircuit {
                name: s.sym(s.field(c, "name")?)?,
                dynamic: s.boolean(s.field(c, "dynamic")?)?,
                allocates: s.boolean(s.field(c, "allocates")?)?,
                sig: s.string(s.field(c, "sig")?)?,
                qasm: s.string(s.field(c, "qasm")?)?,
                document: s.string(s.field(c, "document")?)?,
                monic_slots: s.each_optional(c, "monic", |s, b| s.boolean(b))?,
            })
        })?;
        // written only for a `#unit any` unit: the kind it prefers, or nothing
        let any = match self.optional_string(v, "any")?.as_deref() {
            None => None,
            Some("") => Some(None),
            Some("classical") => Some(Some(crate::pp::UnitKind::Classical)),
            Some("quantum") => Some(Some(crate::pp::UnitKind::Quantum)),
            Some(_) => return bad("the kind a `#unit any` unit prefers is not `classical` or `quantum`"),
        };
        let edition = self.int(self.field(v, "edition")?)?;
        let conductor = self.int(self.field(v, "conductor")?)?;
        let conductor = u32::try_from(conductor).or_else(|_| bad("the conductor is out of range"))?;
        Ok(Metadata {
            version: self.string(self.field(v, "version")?)?,
            edition: u32::try_from(edition).or_else(|_| bad("the edition is out of range"))?,
            unit: unit.clone(),
            conductor,
            main,
            init: self.boolean(self.field(v, "init")?)?,
            interface: Interface {
                unit,
                types,
                fns,
                globals,
                aliases,
                imports,
                source,
                ast: None,
                quantum: self.boolean(self.field(v, "quantum")?)?,
                quantum_only,
                deprecated_types,
                conductor,
                records,
                circuits,
                any,
                geometry,
            },
            uses,
        })
    }

    fn published(&mut self, v: &Tcon) -> Read<Published> {
        let mut claims = Vec::new();
        for c in self.list(self.field(v, "claims")?)? {
            claims.push(match self.tagged(&c.node)? {
                ("Monic", []) => Claim::Monic,
                ("Unitary", []) => Claim::Unitary,
                ("Contract", [k]) => Claim::Contract(self.class(&k.node)?),
                ("Frame", [k]) => Claim::Frame(self.option(&k.node, |s, x| s.phase(x))?),
                ("Outcomes", [t]) => Claim::Outcomes(self.string(&t.node)?),
                (other, _) => return bad(format!("`{other}` is not a claim")),
            });
        }
        let declared = |s: &Self, d: &Tcon| -> Read<Option<Declared>> {
            match s.tagged(d)? {
                ("None", []) => Ok(None),
                ("Declared", [name, program]) => Ok(Some(Declared {
                    name: s.string(&name.node)?,
                    program: s.boolean(&program.node)?,
                })),
                (other, _) => bad(format!("`{other}` is not a declaration")),
            }
        };
        Ok(Published {
            name: self.string(self.field(v, "name")?)?,
            outcomes: self.string(self.field(v, "outcomes")?)?,
            claims,
            cover: declared(self, self.field(v, "cover")?)?,
            gauge: declared(self, self.field(v, "gauge")?)?,
            record: self.record(self.field(v, "record")?)?,
        })
    }

    /// `None`, or `Some(x)` read by `read`.
    fn option<T>(&mut self, v: &Tcon, read: impl FnOnce(&mut Self, &Tcon) -> Read<T>) -> Read<Option<T>> {
        match self.tagged(v)? {
            ("None", []) => Ok(None),
            ("Some", [x]) => Ok(Some(read(self, &x.node)?)),
            (other, _) => bad(format!("expected `None` or `Some`, found `{other}`")),
        }
    }

    fn small(&self, v: &Tcon, what: &str) -> Read<u32> {
        u32::try_from(self.int(v)?).or_else(|_| bad(format!("{what} is out of range")))
    }

    fn phase(&mut self, v: &Tcon) -> Read<Phase> {
        match self.tagged(v)? {
            ("P", [n, m]) => {
                let n = self.small(&n.node, "a phase group's order")?;
                if n == 0 || n > 48 {
                    return bad(format!("{n} is not the order of a phase group"));
                }
                Ok(Phase::of(n, i64::from(self.small(&m.node, "a phase")?)))
            }
            (other, _) => bad(format!("`{other}` is not a phase")),
        }
    }

    fn phases(&mut self, v: &Tcon) -> Read<Vec<Phase>> {
        self.each(v, Self::phase)
    }

    fn class(&mut self, v: &Tcon) -> Read<Class> {
        Ok(match self.tagged(v)? {
            ("Rigid", []) => Class::Rigid,
            ("Flat", [p]) => Class::Flat(self.option(&p.node, |s, x| s.phase(x))?),
            ("Locflat", []) => Class::Locflat,
            ("Cyc", [p]) => Class::Cyc(self.option(&p.node, |s, x| s.phases(x))?),
            ("Stat", [p]) => Class::Stat(self.option(&p.node, |s, x| s.phase(x))?),
            ("Sector", [p]) => Class::Sector(self.option(&p.node, |s, x| s.phases(x))?),
            ("Free", []) => Class::Free,
            (other, _) => return bad(format!("`{other}` is not a contract class")),
        })
    }

    fn tri(&self, v: &Tcon) -> Read<Tri> {
        Ok(match self.tagged(v)? {
            ("Yes", []) => Tri::Yes,
            ("No", []) => Tri::No,
            ("Unknown", []) => Tri::Unknown,
            (other, _) => return bad(format!("`{other}` is not yes, no or unknown")),
        })
    }

    fn record(&mut self, v: &Tcon) -> Read<Record> {
        let kernel = self.option(self.field(v, "kernel")?, |s, k| match s.tagged(k)? {
            ("Kernel", [m, f]) => Ok(Kernel {
                measured: s.small(&m.node, "a count of qubits")?,
                forgotten: s.small(&f.node, "a count of qubits")?,
            }),
            (other, _) => bad(format!("`{other}` is not a kernel")),
        })?;
        let text = |s: &mut Self, x: &Tcon| s.string(x);
        Ok(Record {
            monic: self.tri(self.field(v, "monic")?)?,
            unitary: self.tri(self.field(v, "unitary")?)?,
            contract: self.option(self.field(v, "contract")?, |s, x| s.class(x))?,
            cert: self.option(self.field(v, "cert")?, |s, x| s.judged(x))?,
            sectors: self.option(self.field(v, "sectors")?, |s, x| s.each(x, Self::judged))?,
            kernel,
            fragment: Fragment {
                conductor: self.small(self.field(v, "conductor")?, "the conductor")?,
                departure: self.option(self.field(v, "departure")?, text)?,
                stabilizer: self.boolean(self.field(v, "stabilizer")?)?,
            },
            undecided: self.option(self.field(v, "undecided")?, text)?,
            dynamic: self.boolean(self.field(v, "dynamic")?)?,
        })
    }

    fn judged(&mut self, v: &Tcon) -> Read<Judged> {
        let cert = match self.tagged(self.field(v, "cert")?)? {
            ("Points", [images, schedule, stationary]) => Certificate::Points {
                images: self.each(&images.node, |s, i| Ok(s.int(i)? as usize))?,
                schedule: self.phases(&schedule.node)?,
                stationary: self.boolean(&stationary.node)?,
            },
            ("Scalar", [p]) => Certificate::Scalar(self.phase(&p.node)?),
            ("Moving", []) => Certificate::Moving,
            (other, _) => return bad(format!("`{other}` is not a certificate")),
        };
        Ok(Judged {
            cert,
            source: self.base(self.field(v, "source")?)?,
            target: self.base(self.field(v, "target")?)?,
        })
    }

    fn base(&mut self, v: &Tcon) -> Read<Base> {
        Ok(Base {
            cover: self.cover(self.field(v, "cover")?)?,
            gauge: Gauge {
                fiducials: self.states(self.field(v, "gauge")?)?,
            },
        })
    }

    fn geometry(&mut self, v: &Tcon) -> Read<ExportGeometry> {
        let (name, def) = match self.tagged(v)? {
            ("Cover", [n, c]) => (n, GeometryDef::Cover(self.cover(&c.node)?)),
            ("Gauge", [n, g]) => (n, GeometryDef::Gauge(Gauge { fiducials: self.states(&g.node)? })),
            ("Base", [n, b]) => (n, GeometryDef::Base(self.base(&b.node)?)),
            (other, _) => return bad(format!("`{other}` is not a cover, gauge or base type")),
        };
        Ok(ExportGeometry {
            name: self.sym(&name.node)?,
            def,
        })
    }

    fn cover(&mut self, v: &Tcon) -> Read<Cover> {
        Ok(match self.tagged(v)? {
            ("Pt", [p]) => Cover::Pt(self.state(&p.node)?),
            ("Fin", [ps]) => Cover::Fin(self.states(&ps.node)?),
            ("Span", [ps]) => Cover::Span(self.states(&ps.node)?),
            ("Subspace", [ps]) => Cover::Subspace(self.states(&ps.node)?),
            ("Code", [gens, width, n]) => Cover::Code {
                gens: self.each(&gens.node, |s, g| {
                    let text = s.string(g)?;
                    let (negative, letters) = match text.strip_prefix('-') {
                        Some(rest) => (true, rest),
                        None => (false, text.as_str()),
                    };
                    Pauli::parse(negative, letters).map_or_else(|| bad(format!("`{text}` is not a Pauli string")), Ok)
                })?,
                width: self.int(&width.node)? as usize,
                n: self.conductor(&n.node)?,
            },
            (other, _) => return bad(format!("`{other}` is not a cover")),
        })
    }

    fn conductor(&self, v: &Tcon) -> Read<u32> {
        let n = self.small(v, "a conductor")?;
        if n == 0 || n > 48 {
            return bad(format!("{n} is not a conductor"));
        }
        Ok(n)
    }

    fn states(&mut self, v: &Tcon) -> Read<Vec<Amplitudes>> {
        self.each(v, Self::state)
    }

    fn state(&mut self, v: &Tcon) -> Read<Amplitudes> {
        let width = self.int(self.field(v, "width")?)? as usize;
        let n = self.conductor(self.field(v, "n")?)?;
        let mut terms = std::collections::BTreeMap::new();
        for t in self.list(self.field(v, "terms")?)? {
            let ("Term", [ket, coeffs]) = self.tagged(&t.node)? else {
                return bad("a state's term is written `Term(\"01\", [coefficients])`");
            };
            let ket = self.string(&ket.node)?;
            if ket.len() != width || !ket.chars().all(|c| c == '0' || c == '1') {
                return bad(format!("`{ket}` is not a ket of {width} qubits"));
            }
            let cs = self.each(&coeffs.node, |s, c| {
                let text = s.string(c)?;
                fraction(&text).map_or_else(|| bad(format!("`{text}` is not a fraction")), Ok)
            })?;
            terms.insert(ket.chars().map(|c| c == '1').collect(), Cyclo::from_coeffs(n, cs));
        }
        Ok(Amplitudes { width, n, terms })
    }

    fn fields(&mut self, v: &Tcon, types: &mut TypeTable, count: usize) -> Read<Vec<FieldDef>> {
        self.each(v, |s, f| {
            Ok(FieldDef {
                name: s.sym(s.field(f, "name")?)?,
                ty: s.ty(s.field(f, "ty")?, types, count)?,
                span: Span::synthetic(),
            })
        })
    }

    fn method_of(&mut self, v: &Tcon, types: &mut TypeTable, count: usize) -> Read<Option<MethodOf>> {
        if matches!(self.tagged(v), Ok(("None", []))) {
            return Ok(None);
        }
        let Ty::Adt(owner) = self.ty(self.field(v, "owner")?, types, count)? else {
            return bad("a method's owner is not a structure or enumeration");
        };
        Ok(Some(MethodOf {
            owner,
            operator: self.boolean(self.field(v, "operator")?)?,
            receiver: self.boolean(self.field(v, "receiver")?)?,
        }))
    }

    fn linkage(&self, v: &Tcon) -> Read<Linkage> {
        match self.string(v)?.as_str() {
            "program" => Ok(Linkage::Program),
            "unit" => Ok(Linkage::Unit),
            other => bad(format!("`{other}` is not a linkage")),
        }
    }

    fn access(&self, v: &Tcon) -> Read<Access> {
        match self.string(v)?.as_str() {
            "write" => Ok(Access::Write),
            "const" => Ok(Access::Const),
            other => bad(format!("`{other}` is not a kind of reference")),
        }
    }

    fn ty(&mut self, v: &Tcon, types: &mut TypeTable, count: usize) -> Read<Ty> {
        Ok(match self.tagged(v)? {
            ("Int", [name]) => {
                let n = self.string(&name.node)?;
                Ty::Int(IntTy::from_name(&n).map_or_else(|| bad(format!("`{n}` is not an integer type")), Ok)?)
            }
            ("Float", [name]) => {
                let n = self.string(&name.node)?;
                Ty::Float(FloatTy::from_name(&n).map_or_else(|| bad(format!("`{n}` is not a floating type")), Ok)?)
            }
            ("Tuple", [items]) => {
                let elems = self.each(&items.node, |s, i| s.ty(i, types, count))?;
                types.tuple(elems)
            }
            (tag @ ("Fn" | "Closure" | "Circuit"), [params, ret]) => {
                let make = match tag {
                    "Closure" => TypeTable::closure,
                    "Circuit" => TypeTable::circuit,
                    _ => TypeTable::function,
                };
                let ps = self.each(&params.node, |s, i| s.ty(i, types, count))?;
                let r = self.ty(&ret.node, types, count)?;
                make(types, ps, r)
            }
            ("Bool", []) => Ty::Bool,
            ("Dyn", []) => Ty::Dyn,
            ("Qubit", []) => Ty::Qubit,
            ("Char", []) => Ty::Char,
            ("Void", []) => Ty::Void,
            ("Never", []) => Ty::Never,
            ("Adt", [i]) => {
                let i = self.int(&i.node)? as usize;
                if i >= count {
                    return bad(format!("type {i} is not in the list of types"));
                }
                Ty::Adt(AdtId(i as u32))
            }
            ("Ref", [a, t]) => {
                let a = self.access(&a.node)?;
                let t = self.ty(&t.node, types, count)?;
                types.reference(a, t)
            }
            ("Slice", [a, t]) => {
                let a = self.access(&a.node)?;
                let t = self.ty(&t.node, types, count)?;
                types.slice(a, t)
            }
            ("Growable", [t]) => {
                let t = self.ty(&t.node, types, count)?;
                types.growable(t)
            }
            ("Array", [t, n]) => {
                let t = self.ty(&t.node, types, count)?;
                let n = self.int(&n.node)?;
                types.array(t, n)
            }
            ("Qmap", [k, t]) => {
                let k = self.int(&k.node)?;
                let t = self.ty(&t.node, types, count)?;
                types.qmap(k, t)
            }
            (other, _) => return bad(format!("`{other}` is not a type")),
        })
    }

    fn value(&mut self, v: &Tcon) -> Read<Value> {
        Ok(match self.tagged(v)? {
            ("I", [t, x]) => {
                let t = self.string(&t.node)?;
                let t = IntTy::from_name(&t).map_or_else(|| bad(format!("`{t}` is not an integer type")), Ok)?;
                let x = self.string(&x.node)?;
                let n: i128 = x.parse().or_else(|_| bad(format!("`{x}` is not an integer")))?;
                Value::Int(n, t)
            }
            ("F", [t, bits]) => {
                let t = self.string(&t.node)?;
                let t = FloatTy::from_name(&t).map_or_else(|| bad(format!("`{t}` is not a floating type")), Ok)?;
                let bits = self.string(&bits.node)?;
                let n = bits
                    .strip_prefix("0x")
                    .and_then(|h| u64::from_str_radix(h, 16).ok())
                    .map_or_else(|| bad(format!("`{bits}` is not a floating value's bits")), Ok)?;
                Value::Float(f64::from_bits(n), t)
            }
            ("B", [b]) => Value::Bool(self.boolean(&b.node)?),
            ("C", [c]) => match &c.node {
                Tcon::Scalar(Scalar::Char(c)) => Value::Char(*c),
                other => return bad(format!("expected a character, found {}", other.describe())),
            },
            ("Void", []) => Value::Void,
            ("S", [s]) => Value::Str(self.sym(&s.node)?),
            ("Struct", [fs]) => Value::Struct(self.values(&fs.node)?),
            ("Array", [xs]) => Value::Array(self.values(&xs.node)?),
            ("Enum", [v, fs]) => Value::Enum {
                variant: u32::try_from(self.int(&v.node)?).or_else(|_| bad("a variant number is out of range"))?,
                fields: self.values(&fs.node)?,
            },
            (other, _) => return bad(format!("`{other}` is not a kind of value")),
        })
    }

    fn values(&mut self, v: &Tcon) -> Read<Vec<Value>> {
        self.each(v, Self::value)
    }
}

/// A fraction written `p` or `p/q`.
fn fraction(text: &str) -> Option<Frac> {
    let (p, q) = text.split_once('/').unwrap_or((text, "1"));
    Frac::checked_new(p.parse().ok()?, q.parse().ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::encode::encode;
    use crate::meta::testing::metadata_of;

    #[test]
    fn metadata_round_trips() {
        let mut i = Interner::new();
        let m = metadata_of(
            "geo",
            "struct P { x: i32, name: *[char] }\n\
             [repr: u16]\nenum S { A, B(i64), C { w: [u8; 3] } }\n\
             fn area(p: *P, s: *const [S; 2]) -> [P; 1] { [P { x: p.x, name: \"n\" }] }\n\
             let LIMIT: const i32 = -7;\n\
             let ORIGIN: const S = S::C { w: [1, 2, 3] };\n\
             let COUNT: u64 = 0;\n\
             fn main() -> i32 { 0 }",
            &[],
            &mut i,
        );
        let text = encode(&m, &i);
        let back = decode(&text, &mut i).unwrap_or_else(|e| panic!("{e}\n{text}"));
        assert_eq!(encode(&back, &i), text, "reading and writing again changes nothing");
        assert_eq!(back.main, MainDecl::Valid);
        assert_eq!(back.interface.fns.len(), 2);
        let limit = back.interface.globals.iter().find(|g| i.resolve(g.name) == "LIMIT").unwrap();
        assert_eq!(limit.value, Some(Value::Int(-7, IntTy::I32)));
    }

    #[test]
    fn type_aliases_round_trip_and_an_importer_names_them() {
        let mut i = Interner::new();
        let m = metadata_of(
            "geo",
            "struct P { x: f64 }\ntype Meters = f64;\ntype Pts = [P; 2];\nstatic type Hidden = i32;",
            &[],
            &mut i,
        );
        let text = encode(&m, &i);
        let back = decode(&text, &mut i).unwrap_or_else(|e| panic!("{e}\n{text}"));
        assert_eq!(encode(&back, &i), text, "reading and writing again changes nothing");
        let names: Vec<&str> = back.interface.aliases.iter().map(|a| i.resolve(a.name)).collect();
        assert_eq!(names, ["Meters", "Pts", "Hidden"]);
        // the unit has no generic or constant function, so no source travels:
        // the interface alone is what an importer reads
        assert!(back.interface.source.is_none());
        let (_, d) = crate::sema::testing::analyzed_in(
            "app",
            "import geo;\nfn f(m: geo::Meters, p: *geo::Pts) -> f64 { m + p[1].x }",
            std::slice::from_ref(&back.interface),
            &mut i,
        );
        assert!(d.iter().all(|d| !d.is_error()), "{d:?}");
        let (_, d) = crate::sema::testing::analyzed_in(
            "app",
            "import geo;\nfn f(h: geo::Hidden) { }",
            std::slice::from_ref(&back.interface),
            &mut i,
        );
        assert_eq!(d.iter().filter(|d| d.is_error()).map(|d| d.code).collect::<Vec<_>>(), [crate::diag::Code::Es04]);
    }

    #[test]
    fn a_quantum_units_records_and_circuits_round_trip() {
        use crate::driver::{Options, Session, Stage, compile};
        let mut session = Session::new();
        let id = session.add(
            "qk.tq",
            "#unit quantum\nimport gates;\n\
             cover BITS = fin{ |0>, (|0> + |1>) * isq2 };\ngauge G = fid(|0>);\n\
             base B = [qubit; 1], BITS, G;\nstatic cover OWN = pt(|1>);\n\
             [cover: BITS] [gauge: G]\n[expect: unitary, contract = rigid]\n\
             fn keep(q: *qubit) { }\n\
             [expect: contract = stat(0)]\nfn nothing(q: *qubit) { }\n\
             [entry]\nfn run(n: u32) -> bool { let q: [qubit; 1] = prep |0>; h(&q[0]); let m = measure q; m[0] }\n",
        );
        let out = compile(
            &mut session,
            id,
            Options {
                stage: Stage::Check,
                ..Options::default()
            },
        );
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let m = out.metadata.expect("analysed");
        let names: Vec<&str> = m.interface.records.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["keep", "nothing", "run"]);
        let keep = &m.interface.records[0];
        assert_eq!(keep.claims, [Claim::Unitary, Claim::Contract(Class::Rigid)]);
        assert_eq!(keep.cover, Some(Declared { name: "BITS".to_owned(), program: true }));
        assert_eq!(m.interface.circuits.len(), 1);
        assert!(m.interface.circuits[0].qasm.contains("measure"));

        let mut i = session.interner().clone();
        let text = encode(&m, &i);
        let back = decode(&text, &mut i).unwrap_or_else(|e| panic!("{e}\n{text}"));
        assert_eq!(encode(&back, &i), text, "reading and writing again changes nothing");
        assert_eq!(back.interface.records, m.interface.records);
        assert_eq!(back.interface.circuits[0].document, m.interface.circuits[0].document);
        assert_eq!(back.interface.conductor, 8);

        // the covers, gauges and base types not `static` travel, and an
        // importer names them
        let geometry: Vec<&str> = back.interface.geometry.iter().map(|g| i.resolve(g.name)).collect();
        assert_eq!(geometry, ["BITS", "G", "B"]);
        assert_eq!(back.interface.geometry, m.interface.geometry);
        let (_, d) = crate::sema::testing::quantum_in(
            "app",
            "import qk;\nbase C = qubit, qk::BITS, qk::G;\n\
             [cover: qk::B]\n[expect: contract = rigid]\nfn f(q: *qubit) { }\n",
            std::slice::from_ref(&back.interface),
            &mut i,
        );
        assert!(d.iter().all(|d| !d.is_error()), "{d:?}");
        let (_, d) = crate::sema::testing::quantum_in(
            "app",
            "import qk;\n[cover: qk::OWN]\nfn f(q: *qubit) { }\n",
            std::slice::from_ref(&back.interface),
            &mut i,
        );
        assert_eq!(d.iter().filter(|d| d.is_error()).map(|d| d.code).collect::<Vec<_>>(), [crate::diag::Code::Es04]);
    }

    #[test]
    fn damaged_metadata_is_an_error_not_a_panic() {
        let mut i = Interner::new();
        for text in ["", "Metadata {", "Metadata { unit: 5 }", "[1, 2]", "Metadata { unit: \"u\", main: Nope }"] {
            assert!(decode(text, &mut i).is_err(), "{text:?}");
        }
    }
}
