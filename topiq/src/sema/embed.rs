//! `@embed("file.tcon")`: a TCON document read during translation, as a
//! constant of the type the place it is written in wants.
//!
//! The type comes from where the macro is written: the type of the `let` it
//! initialises, as in `let CONFIG: const Config = @embed("config.tcon");`, or
//! the type after `as`. The document is read with the unit, from a path
//! relative to the unit's file, and checked against that type field by field.
//! A mismatch is `ET01`, reported at the document's own line; a type that
//! has no written form is `ET02`. The value is then a constant like any
//! other, placed in the program and folded wherever it is used.
//!
//! The document is checked as `tcon::parse` checks one when the program
//! runs: an object for a structure, whose fields may come in any order and
//! whose optional fields may be left out; `null` for an absent optional; a
//! string for a `[char]`; an array for an array, of the right length when it
//! is fixed; a variant by its name, with its values in `(…)` or its fields in
//! `{…}`; and a number for a number, which must fit the type exactly.
//!
//! A path ending in `.quon` names a QUON document of named states instead,
//! read as a structure with a `quon::State` field for each
//! ([`super::quon`]).

use chumsky::Parser;

use crate::ast::{self, MacroArg};
use crate::diag::{Code, Diagnostic};
use crate::parse::input::{Cx, TokenStream, eoi_span, stream};
use crate::span::{Span, Spanned};
use crate::tcon::ast::{Pair, Scalar, Value as Doc};
use crate::tir::{AdtId, AdtKind, Expr, Ty, Value, VariantShape};

use super::body::Checker;
use super::typeinfo::TconForm;

impl Checker<'_, '_> {
    /// `@embed("path")`, read as a `want`; `None` when nothing says what
    /// type that is.
    pub(super) fn embed(&mut self, args: &[Spanned<MacroArg>], want: Option<Ty>, span: Span) -> Expr {
        // the path, written as a string
        let path = match args {
            [a] => match &a.node {
                MacroArg::Str(s) => Some(s.node),
                MacroArg::Expr(e) => match &e.node {
                    ast::Expr::Str { value, .. } => Some(*value),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        };
        let Some(path) = path else {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("`@embed` takes the path of a document, written as a string")
                    .at(span)
                    .with_help("write it as `@embed(\"config.tcon\")`, the path relative to this unit's file"),
            );
            return Checker::error(span);
        };

        // a QUON document says its own type; a TCON one needs `want`
        let path = self.name(path).to_owned();
        let want = want.map(|t| self.settled(t));
        if path.ends_with(".quon") {
            return match self.document(&path, span) {
                Some((source, spliced)) => self.embed_quon(&path, source, spliced, want, span),
                None => Checker::error(span),
            };
        }
        let Some(want) = want else {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!("nothing here says what type `{path}` is read as"))
                    .at(span)
                    .with_note("a document is checked against a type, which is what its values are read as")
                    .with_help(format!(
                        "write it where a type is given, as in `let CONFIG: const Config = @embed(\"{path}\");`, or as `@embed(\"{path}\") as Config`"
                    )),
            );
            return Checker::error(span);
        };

        // read and parse the document
        let Some((source, spliced)) = self.document(&path, span) else {
            return Checker::error(span);
        };
        let Some((tokens, eoi)) = self.lex_document(source, spliced) else {
            return Checker::error(span);
        };
        let null = self.interner.intern_late("null");
        let parsed = {
            let cx = Cx::new(self.interner);
            crate::tcon::parse::document::<TokenStream>(cx, null)
                .parse(stream(&tokens, eoi))
                .into_result()
        };
        let doc = match parsed {
            Ok(doc) => doc,
            Err(errors) => {
                for e in errors {
                    self.report(
                        Diagnostic::new(Code::Es02)
                            .with_message(format!("`{path}` is not a TCON document here: {e}"))
                            .at(*e.span())
                            .with_note("a document is one value, written as Topiq writes literals"),
                    );
                }
                return Checker::error(span);
            }
        };

        // its value, checked against `want`
        match self.document_value(&doc.value, want) {
            Some(value) => Expr::constant(value, want, span),
            None => Checker::error(span),
        }
    }

    /// The constant `v` denotes as a `ty`, or `None` once what is wrong
    /// with it has been reported.
    fn document_value(&mut self, v: &Spanned<Doc>, ty: Ty) -> Option<Value> {
        let ty = self.settled(ty);
        let types = self.types();
        if let Some(elem) = types.as_growable(ty) {
            return match &v.node {
                Doc::Scalar(Scalar::Str { value, .. }) if elem == Ty::Char => {
                    Some(Value::Array(self.name(*value).chars().map(Value::Char).collect()))
                }
                Doc::Array(items) => self.document_items(items, elem),
                _ => self.doc_mismatch(v, ty, "an array"),
            };
        }
        if let Some((elem, n)) = types.as_array(ty) {
            return match &v.node {
                Doc::Array(items) if items.len() as u64 == n => self.document_items(items, elem),
                Doc::Array(items) => {
                    let what = self.show(ty);
                    self.et01(v.span, format!("a `{what}` holds {n} elements, but this array has {}", items.len()))
                }
                _ => self.doc_mismatch(v, ty, "an array"),
            };
        }
        if let Some((_, Ty::Char)) = types.as_slice(ty) {
            return match &v.node {
                Doc::Scalar(Scalar::Str { value, .. }) => Some(Value::Str(*value)),
                _ => self.doc_mismatch(v, ty, "a string"),
            };
        }
        match ty {
            Ty::Int(t) => {
                let (negative, n) = match &v.node {
                    Doc::Negative(n) => (true, &**n),
                    _ => (false, v),
                };
                let Doc::Scalar(Scalar::Int { raw, base, .. }) = &n.node else {
                    return self.doc_mismatch(v, ty, "an integer");
                };
                let Some(x) = super::arith::literal_value(self.name(*raw), base.radix()) else {
                    return self.et01(v.span, "this integer is larger than any integer type can hold".to_owned());
                };
                let x = if negative { -x } else { x };
                if !t.fits(x) {
                    return self.et01(v.span, format!("{x} does not fit in a `{t}`, which holds {} to {}", t.min(), t.max()));
                }
                Some(Value::Int(x, t))
            }
            Ty::Float(t) => {
                let (negative, n) = match &v.node {
                    Doc::Negative(n) => (true, &**n),
                    _ => (false, v),
                };
                let text = match &n.node {
                    Doc::Scalar(Scalar::Float { raw, .. } | Scalar::Int { raw, .. }) => self.name(*raw).replace('_', ""),
                    _ => return self.doc_mismatch(v, ty, "a number"),
                };
                let x: f64 = text.parse().ok()?;
                Some(Value::Float(t.round(if negative { -x } else { x }), t))
            }
            Ty::Bool => match &v.node {
                Doc::Scalar(Scalar::Bool(b)) => Some(Value::Bool(*b)),
                _ => self.doc_mismatch(v, ty, "`true` or `false`"),
            },
            Ty::Char => match &v.node {
                Doc::Scalar(Scalar::Char(c)) => Some(Value::Char(*c)),
                _ => self.doc_mismatch(v, ty, "a character"),
            },
            Ty::Tuple(_) => {
                let elems = self.types().as_tuple(ty).map(<[Ty]>::to_vec).unwrap_or_default();
                match &v.node {
                    Doc::Array(items) if items.len() == elems.len() => {
                        let mut out = Vec::with_capacity(items.len());
                        for (item, &t) in items.iter().zip(&elems) {
                            out.push(self.document_value(item, t)?);
                        }
                        Some(Value::Struct(out))
                    }
                    _ => self.doc_mismatch(v, ty, "an array of one value for each element of the tuple"),
                }
            }
            Ty::Adt(id) => self.document_adt(v, id),
            _ => {
                let what = self.show(ty);
                self.report(
                    Diagnostic::new(Code::Et02)
                        .with_message(format!("a `{what}` cannot be read from a document"))
                        .at(v.span)
                        .with_note("a document holds values, and this is part of the running program"),
                );
                None
            }
        }
    }

    fn document_items(&mut self, items: &[Spanned<Doc>], elem: Ty) -> Option<Value> {
        let mut out = Vec::with_capacity(items.len());
        let mut ok = true;
        for item in items {
            match self.document_value(item, elem) {
                Some(x) => out.push(x),
                None => ok = false,
            }
        }
        ok.then_some(Value::Array(out))
    }

    /// A structure or enumeration: `null` or the value for an `Opt`, an
    /// object for a structure, a variant for an enumeration.
    fn document_adt(&mut self, v: &Spanned<Doc>, id: AdtId) -> Option<Value> {
        let def = self.adt(id).clone();
        let name = self.show(Ty::Adt(id));
        if def.opaque {
            self.report(
                Diagnostic::new(Code::Et02)
                    .with_message(format!("`{name}` is marked `[tcon: opaque]`, so it cannot be read from a document"))
                    .at(v.span),
            );
            return None;
        }
        if self.core_enum(Ty::Adt(id), "Opt").is_some() {
            let variant = |n: &str| def.variants().iter().position(|x| self.name(x.name) == n).expect("`Opt` has it") as u32;
            if v.node.is_null() {
                return Some(Value::Enum {
                    variant: variant("None"),
                    fields: Vec::new(),
                });
            }
            let some = variant("Some");
            let inner = def.variants()[some as usize].fields[0].ty;
            let x = self.document_value(v, inner)?;
            return Some(Value::Enum {
                variant: some,
                fields: vec![x],
            });
        }
        match &def.kind {
            AdtKind::Struct { fields } => {
                let pairs: &[Pair] = match &v.node {
                    Doc::Object(p) => p,
                    Doc::Typed { path, fields: p } => {
                        let written = self.path_text(path);
                        let last = written.rsplit("::").next().unwrap_or(&written).to_owned();
                        if last != self.name(def.name) {
                            return self.et01(v.span, format!("the document names `{written}` where a `{name}` is wanted"));
                        }
                        p
                    }
                    _ => return self.doc_mismatch(v, Ty::Adt(id), "an object"),
                };
                let open = self.cx.tcon_forms.get(&id) == Some(&TconForm::Open);
                let fields = fields.clone();
                self.object_fields(pairs, &fields, open, &name, v.span).map(Value::Struct)
            }
            AdtKind::Enum { variants } => {
                let (path, positional, named): (&ast::Path, &[Spanned<Doc>], &[Pair]) = match &v.node {
                    Doc::Enum { path, payload } => (path, payload, &[]),
                    Doc::Typed { path, fields } => (path, &[], fields),
                    _ => return self.doc_mismatch(v, Ty::Adt(id), "one of its variants"),
                };
                let written = self.path_text(path);
                let last = written.rsplit("::").next().unwrap_or(&written).to_owned();
                let Some(index) = variants.iter().position(|x| self.name(x.name) == last) else {
                    let names: Vec<String> = variants.iter().map(|x| format!("`{}`", self.name(x.name))).collect();
                    return self.et01_with(
                        v.span,
                        format!("`{name}` has no variant `{last}`"),
                        format!("its variants are {}", names.join(", ")),
                    );
                };
                let var = variants[index].clone();
                let fields = match var.shape {
                    VariantShape::Struct => {
                        let open = false;
                        self.object_fields(named, &var.fields, open, &format!("{name}::{last}"), v.span)?
                    }
                    _ => {
                        if positional.len() != var.fields.len() || !named.is_empty() {
                            return self.et01(
                                v.span,
                                format!("`{last}` carries {} values, but the document gives {}", var.fields.len(), positional.len()),
                            );
                        }
                        let mut out = Vec::with_capacity(positional.len());
                        for (item, f) in positional.iter().zip(&var.fields) {
                            out.push(self.document_value(item, f.ty)?);
                        }
                        out
                    }
                };
                Some(Value::Enum {
                    variant: index as u32,
                    fields,
                })
            }
        }
    }

    /// The values of a structure's fields from an object's pairs, in the
    /// structure's order: an optional field left out is `None`, any other
    /// left out is reported, and so is a pair the structure has no field
    /// for unless it is `[tcon: open]`.
    fn object_fields(
        &mut self,
        pairs: &[Pair],
        fields: &[crate::tir::FieldDef],
        open: bool,
        name: &str,
        at: Span,
    ) -> Option<Vec<Value>> {
        // each pair names a field, unless the type is open
        let mut ok = true;
        for p in pairs {
            if !fields.iter().any(|f| f.name == p.name.node) && !open {
                let field = self.name(p.name.node).to_owned();
                self.et01_with::<()>(
                    p.name.span,
                    format!("`{name}` has no field `{field}`"),
                    "a field the type does not have is allowed only when the type is `[tcon: open]`".to_owned(),
                );
                ok = false;
            }
        }

        // each field given, or else optional
        let mut out = Vec::with_capacity(fields.len());
        for f in fields {
            match pairs.iter().find(|p| p.name.node == f.name) {
                Some(p) => match self.document_value(&p.value, f.ty) {
                    Some(x) => out.push(x),
                    None => ok = false,
                },
                None => {
                    let ft = self.settled(f.ty);
                    if let Some(def) = self.core_enum(ft, "Opt") {
                        let none = def.variants().iter().position(|x| self.name(x.name) == "None").expect("`Opt` has it");
                        out.push(Value::Enum {
                            variant: none as u32,
                            fields: Vec::new(),
                        });
                    } else {
                        let field = self.name(f.name).to_owned();
                        self.et01_with::<()>(
                            at,
                            format!("the field `{field}` of `{name}` is missing"),
                            "every field must be given, except one whose type is optional, which is `null` when left out".to_owned(),
                        );
                        ok = false;
                    }
                }
            }
        }
        ok.then_some(out)
    }

    fn path_text(&self, path: &ast::Path) -> String {
        path.segments.iter().map(|s| self.name(s.node)).collect::<Vec<_>>().join("::")
    }

    fn doc_mismatch<T>(&mut self, v: &Spanned<Doc>, ty: Ty, wanted: &str) -> Option<T> {
        let what = self.show(ty);
        let found = v.node.describe();
        self.et01(v.span, format!("a `{what}` is read from {wanted}, but the document gives {found}"))
    }

    fn et01<T>(&mut self, span: Span, message: String) -> Option<T> {
        self.report(
            Diagnostic::new(Code::Et01)
                .with_message(message)
                .at(span)
                .with_note("an embedded document is checked against the type it is read as"),
        );
        None
    }

    fn et01_with<T>(&mut self, span: Span, message: String, note: String) -> Option<T> {
        self.report(Diagnostic::new(Code::Et01).with_message(message).at(span).with_note(note));
        None
    }
}

impl<'a> Checker<'_, 'a> {
    /// The tokens of a document read for `@embed`, and where they end;
    /// `None` once what could not be read as tokens is reported.
    pub(super) fn lex_document(
        &mut self,
        source: crate::span::SourceId,
        spliced: &crate::source::Spliced,
    ) -> Option<(Vec<(crate::lex::Token, Span)>, Span)> {
        let lexed = crate::lex::lex(source, spliced, self.interner);
        if !lexed.diagnostics.is_empty() {
            for d in lexed.diagnostics {
                self.report(d);
            }
            return None;
        }
        let eoi = eoi_span(source, &lexed.tokens);
        Some((lexed.tokens, eoi))
    }

    /// The document `@embed` names at `path`, read with the unit; `None`
    /// once why it cannot be read has been reported.
    pub(super) fn document(&mut self, path: &str, span: Span) -> Option<(crate::span::SourceId, &'a crate::source::Spliced)> {
        let found = self.cx.documents.iter().find(|(p, _)| p == path).map(|(_, r)| r.clone());
        match found {
            Some(Ok(read)) => Some(read),
            Some(Err(why)) => {
                self.report(
                    Diagnostic::new(Code::Es02)
                        .with_message(format!("cannot read the document `{path}`: {why}"))
                        .at(span)
                        .with_note("the path is relative to the directory of this unit's file"),
                );
                None
            }
            None => {
                self.report(
                    Diagnostic::new(Code::Es02)
                        .with_message(format!("the document `{path}` was not read with this unit"))
                        .at(span)
                        .with_note("`@embed` reads a document named by a string written directly in it"),
                );
                None
            }
        }
    }
}
