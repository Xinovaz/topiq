//! Whether a `match` covers every value, and whether each arm can be reached.
//!
//! A classical `match` must accept every value of the type it matches: there
//! is no implicit fall-through and no run-time failure for a value nobody
//! expected. When one is missed, the diagnostic names a value no arm accepts,
//! written as a pattern (`Shape::Rect { w: _, h: _ }`, or `3`), so the fix
//! is to add an arm for exactly that.
//!
//! An arm that no value can reach, because the arms above it already accept
//! everything it would, is almost always a mistake in the order of the arms.
//! It is a warning.
//!
//! # How it is decided
//!
//! The arms form a matrix of patterns, one row per arm. A pattern vector is
//! *useful* against the matrix if some value matches it and matches no row.
//! The match is exhaustive exactly when a lone `_` is not useful against all
//! the arms, and arm `i` is reachable exactly when it is useful against the
//! arms before it. Usefulness is decided by splitting on the constructor in
//! the first column (a variant, `true` or `false`, a particular integer or
//! character), which is exact for every type patterns can take apart.
//!
//! The integer and character types are finite, so listing all 256 values of a
//! `u8` covers it. In practice, a `match` on a wider integer needs a `_` or a
//! binding arm.

use std::collections::BTreeSet;

use crate::diag::{Code, Diagnostic};
use crate::intern::Interner;
use crate::span::Span;
use crate::tir::visit::{self, Visit};
use crate::tir::{Block, Expr, ExprKind, IntTy, Pat, PatKind, Ty, TypeTable, Value, VariantShape};

/// Checks every `match` in a function body.
pub fn check_block(b: &Block, types: &TypeTable, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    let mut c = Checker { types, interner, diags };
    c.block(b);
}

/// Checks every `match` in an expression.
pub fn check_expr(e: &Expr, types: &TypeTable, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    let mut c = Checker { types, interner, diags };
    c.expr(e);
}

struct Checker<'a> {
    types: &'a TypeTable,
    interner: &'a Interner,
    diags: &'a mut Vec<Diagnostic>,
}

impl Visit for Checker<'_> {
    fn expr(&mut self, e: &Expr) {
        if let ExprKind::Match { scrutinee, arms } = &e.kind {
            self.check(scrutinee.ty, arms.iter().map(|a| &a.pat).collect(), e.span);
        }
        visit::walk_expr(self, e);
    }
}

/// A constructor: what the outermost part of a value is.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Ctor {
    /// One variant of an enumeration.
    Variant(u32),
    /// A structure.
    Struct,
    /// A boolean.
    Bool(bool),
    /// One integer.
    Int(i128),
    /// One character.
    Char(char),
    /// One floating value, by its bits, which is how two literals are told
    /// apart exactly.
    Float(u64),
}

/// A pattern reduced to what usefulness needs.
#[derive(Clone, Debug)]
enum P {
    /// Matches anything.
    Wild,
    /// A constructor and patterns for all of its fields, in field order.
    Ctor(Ctor, Vec<P>),
}

impl Checker<'_> {
    fn check(&mut self, ty: Ty, pats: Vec<&Pat>, span: Span) {
        if ty == Ty::Never || pats.iter().any(|p| mentions_never(p)) {
            // something here was already reported; nothing sensible to add
            return;
        }
        let rows: Vec<Vec<P>> = pats.iter().map(|p| vec![self.reduce(p)]).collect();
        for (i, row) in rows.iter().enumerate() {
            if self.useful(&rows[..i], row, &[ty]).is_none() {
                self.diags.push(
                    Diagnostic::new(Code::Es14)
                        .with_message("this arm can never be chosen: the arms above it already match every value it would")
                        .at(pats[i].span)
                        .with_help("remove the arm, or move it above the arm that covers it")
                        .as_warning(),
                );
            }
        }
        if let Some(w) = self.useful(&rows, &[P::Wild], &[ty]) {
            let example = self.show(&w[0], ty);
            let what = self.types.display(ty, self.interner);
            self.diags.push(
                Diagnostic::new(Code::Es13)
                    .with_message(format!(
                        "this `match` does not cover every `{what}`: for example, `{example}` matches no arm"
                    ))
                    .at(span)
                    .with_note("a `match` must accept every value; nothing happens by default for a value no arm names")
                    .with_help(format!(
                        "add an arm for `{example}`, or a final `_ => …` arm for everything not listed"
                    )),
            );
        }
    }

    /// Reduces a checked pattern: bindings become wildcards, and a structure
    /// or variant pattern lists every field, with `_` for those not written.
    fn reduce(&self, p: &Pat) -> P {
        match &p.kind {
            PatKind::Wild | PatKind::Bind(_) | PatKind::BindRef(_) => P::Wild,
            PatKind::Const(Value::Bool(b)) => P::Ctor(Ctor::Bool(*b), vec![]),
            PatKind::Const(Value::Int(v, _)) => P::Ctor(Ctor::Int(*v), vec![]),
            PatKind::Const(Value::Char(c)) => P::Ctor(Ctor::Char(*c), vec![]),
            PatKind::Const(Value::Float(x, _)) => P::Ctor(Ctor::Float(x.to_bits()), vec![]),
            PatKind::Const(_) => P::Wild,
            PatKind::Struct { fields } => P::Ctor(Ctor::Struct, self.full(p.ty, Ctor::Struct, fields)),
            PatKind::Variant { variant, fields } => {
                P::Ctor(Ctor::Variant(*variant), self.full(p.ty, Ctor::Variant(*variant), fields))
            }
        }
    }

    fn full(&self, ty: Ty, c: Ctor, given: &[(u32, Pat)]) -> Vec<P> {
        let n = self.field_types(ty, c).len();
        (0..n)
            .map(|i| {
                given
                    .iter()
                    .find(|(j, _)| *j as usize == i)
                    .map_or(P::Wild, |(_, p)| self.reduce(p))
            })
            .collect()
    }

    /// The types of a constructor's fields.
    fn field_types(&self, ty: Ty, c: Ctor) -> Vec<Ty> {
        match (ty, c) {
            (Ty::Adt(id), Ctor::Struct) => self.types.adt(id).fields().iter().map(|f| f.ty).collect(),
            // a tuple has one constructor, like a structure, and its
            // elements are its fields
            (Ty::Tuple(_), Ctor::Struct) => self.types.as_tuple(ty).unwrap_or(&[]).to_vec(),
            (Ty::Adt(id), Ctor::Variant(v)) => self.types.adt(id).variants()[v as usize]
                .fields
                .iter()
                .map(|f| f.ty)
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Whether the constructors in `used` are every constructor of `ty`.
    fn complete(&self, ty: Ty, used: &BTreeSet<Ctor>) -> bool {
        match ty {
            Ty::Adt(id) => {
                let def = self.types.adt(id);
                if def.is_struct() {
                    used.contains(&Ctor::Struct)
                } else {
                    (0..def.variants().len() as u32).all(|v| used.contains(&Ctor::Variant(v)))
                }
            }
            Ty::Tuple(_) => used.contains(&Ctor::Struct),
            Ty::Bool => used.contains(&Ctor::Bool(false)) && used.contains(&Ctor::Bool(true)),
            Ty::Int(t) => used.len() as u128 == (t.max() - t.min() + 1) as u128,
            Ty::Char => used.len() == CHAR_COUNT,
            _ => false,
        }
    }

    /// A constructor of `ty` not in `used`.
    fn missing(&self, ty: Ty, used: &BTreeSet<Ctor>) -> Ctor {
        match ty {
            Ty::Adt(id) => {
                let def = self.types.adt(id);
                if def.is_struct() {
                    Ctor::Struct
                } else {
                    (0..def.variants().len() as u32)
                        .map(Ctor::Variant)
                        .find(|c| !used.contains(c))
                        .unwrap_or(Ctor::Variant(0))
                }
            }
            Ty::Tuple(_) => Ctor::Struct,
            Ty::Bool => {
                if used.contains(&Ctor::Bool(false)) {
                    Ctor::Bool(true)
                } else {
                    Ctor::Bool(false)
                }
            }
            Ty::Int(t) => Ctor::Int(missing_int(t, used)),
            Ty::Char => Ctor::Char(
                (0..=0x10FFFF)
                    .filter_map(char::from_u32)
                    .find(|c| !used.contains(&Ctor::Char(*c)))
                    .unwrap_or('\0'),
            ),
            // no list of floating literals covers a floating type: there is
            // always another value, so such a `match` needs a `_` arm
            Ty::Float(_) => {
                let mut x = 0.0f64;
                while used.contains(&Ctor::Float(x.to_bits())) {
                    x += 1.0;
                }
                Ctor::Float(x.to_bits())
            }
            _ => Ctor::Struct,
        }
    }

    /// If `q` is useful against `rows`, a witness: patterns for a value that
    /// matches `q` and no row.
    fn useful(&self, rows: &[Vec<P>], q: &[P], tys: &[Ty]) -> Option<Vec<P>> {
        let Some((head, rest)) = q.split_first() else {
            return rows.is_empty().then(Vec::new);
        };
        let ty = tys[0];
        match head {
            P::Ctor(c, args) => {
                let fts = self.field_types(ty, *c);
                let sub = specialize(rows, *c, fts.len());
                let q2: Vec<P> = args.iter().cloned().chain(rest.iter().cloned()).collect();
                let tys2: Vec<Ty> = fts.iter().copied().chain(tys[1..].iter().copied()).collect();
                self.useful(&sub, &q2, &tys2).map(|w| rebuild(*c, fts.len(), w))
            }
            P::Wild => {
                let used: BTreeSet<Ctor> = rows
                    .iter()
                    .filter_map(|r| match &r[0] {
                        P::Ctor(c, _) => Some(*c),
                        P::Wild => None,
                    })
                    .collect();
                if !used.is_empty() && self.complete(ty, &used) {
                    let ctors: Vec<Ctor> = match ty {
                        Ty::Adt(id) if !self.types.adt(id).is_struct() => {
                            (0..self.types.adt(id).variants().len() as u32).map(Ctor::Variant).collect()
                        }
                        _ => used.iter().copied().collect(),
                    };
                    for c in ctors {
                        let fts = self.field_types(ty, c);
                        let sub = specialize(rows, c, fts.len());
                        let q2: Vec<P> = std::iter::repeat_n(P::Wild, fts.len()).chain(rest.iter().cloned()).collect();
                        let tys2: Vec<Ty> = fts.iter().copied().chain(tys[1..].iter().copied()).collect();
                        if let Some(w) = self.useful(&sub, &q2, &tys2) {
                            return Some(rebuild(c, fts.len(), w));
                        }
                    }
                    None
                } else {
                    let d: Vec<Vec<P>> = rows
                        .iter()
                        .filter(|r| matches!(r[0], P::Wild))
                        .map(|r| r[1..].to_vec())
                        .collect();
                    let w = self.useful(&d, rest, &tys[1..])?;
                    let head = if used.is_empty() {
                        P::Wild
                    } else {
                        let c = self.missing(ty, &used);
                        P::Ctor(c, vec![P::Wild; self.field_types(ty, c).len()])
                    };
                    Some(std::iter::once(head).chain(w).collect())
                }
            }
        }
    }

    /// A witness written as a pattern.
    fn show(&self, p: &P, ty: Ty) -> String {
        let P::Ctor(c, args) = p else {
            return "_".to_owned();
        };
        let fts = self.field_types(ty, *c);
        let parts: Vec<String> = args.iter().zip(&fts).map(|(a, &t)| self.show(a, t)).collect();
        match (*c, ty) {
            (Ctor::Bool(b), _) => b.to_string(),
            (Ctor::Int(v), _) => v.to_string(),
            (Ctor::Char(ch), _) => format!("'{}'", ch.escape_debug()),
            (Ctor::Float(bits), Ty::Float(t)) => t.text(f64::from_bits(bits)),
            (Ctor::Float(bits), _) => f64::from_bits(bits).to_string(),
            (Ctor::Struct, Ty::Tuple(_)) => format!("({})", parts.join(", ")),
            (Ctor::Struct, Ty::Adt(id)) => {
                let def = self.types.adt(id);
                let fs: Vec<String> = def
                    .fields()
                    .iter()
                    .zip(&parts)
                    .map(|(f, s)| format!("{}: {s}", self.interner.resolve(f.name)))
                    .collect();
                format!("{} {{ {} }}", self.types.adt_name(id, self.interner), fs.join(", "))
            }
            (Ctor::Variant(v), Ty::Adt(id)) => {
                let var = &self.types.adt(id).variants()[v as usize];
                let head = format!("{}::{}", self.types.adt_name(id, self.interner), self.interner.resolve(var.name));
                match var.shape {
                    VariantShape::Unit => head,
                    VariantShape::Tuple => format!("{head}({})", parts.join(", ")),
                    VariantShape::Struct => {
                        let fs: Vec<String> = var
                            .fields
                            .iter()
                            .zip(&parts)
                            .map(|(f, s)| format!("{}: {s}", self.interner.resolve(f.name)))
                            .collect();
                        format!("{head} {{ {} }}", fs.join(", "))
                    }
                }
            }
            _ => "_".to_owned(),
        }
    }
}

/// How many Unicode scalar values there are: every code point except the
/// surrogates.
const CHAR_COUNT: usize = 0x11_0000 - 0x800;

/// Whether a pattern mentions a type that stands for an earlier error.
fn mentions_never(p: &Pat) -> bool {
    p.ty == Ty::Never
        || match &p.kind {
            PatKind::Struct { fields } | PatKind::Variant { fields, .. } => fields.iter().any(|(_, f)| mentions_never(f)),
            _ => false,
        }
}

/// The rows that can match constructor `c`, with its fields in place of the
/// first column.
fn specialize(rows: &[Vec<P>], c: Ctor, arity: usize) -> Vec<Vec<P>> {
    rows.iter()
        .filter_map(|r| {
            let rest = r[1..].iter().cloned();
            match &r[0] {
                P::Ctor(d, args) if *d == c => Some(args.iter().cloned().chain(rest).collect()),
                P::Ctor(..) => None,
                P::Wild => Some(std::iter::repeat_n(P::Wild, arity).chain(rest).collect()),
            }
        })
        .collect()
}

/// Puts a constructor back around the first `arity` witness patterns.
fn rebuild(c: Ctor, arity: usize, mut w: Vec<P>) -> Vec<P> {
    let rest = w.split_off(arity);
    std::iter::once(P::Ctor(c, w)).chain(rest).collect()
}

/// A value of `t` that no constructor in `used` names: the one nearest zero.
fn missing_int(t: IntTy, used: &BTreeSet<Ctor>) -> i128 {
    let mut k: i128 = 0;
    loop {
        for v in [k, -k] {
            if t.fits(v) && !used.contains(&Ctor::Int(v)) {
                return v;
            }
        }
        k += 1;
        if k > used.len() as i128 + 1 {
            return t.min();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{analyzed, check, codes};

    const SHAPE: &str = "enum Shape { Empty, Circle(i64), Rect { w: i64, h: i64 } }\n";

    fn message(src: &str) -> String {
        let (_, d) = analyzed(src);
        let d = d.iter().find(|d| d.code == Code::Es13).expect("ES13 reported");
        d.message.clone()
    }

    #[test]
    fn covering_every_variant_is_exhaustive() {
        check(&format!(
            "{SHAPE}fn f(s: Shape) -> i64 {{ match s {{ Shape::Empty => 0, Shape::Circle(r) => r, Shape::Rect {{ w, h }} => w * h }} }}"
        ));
    }

    #[test]
    fn a_missing_variant_is_named() {
        let m = message(&format!(
            "{SHAPE}fn f(s: Shape) -> i64 {{ match s {{ Shape::Empty => 0, Shape::Circle(r) => r }} }}"
        ));
        assert!(m.contains("`Shape::Rect { w: _, h: _ }`"), "{m}");
    }

    #[test]
    fn a_missing_nested_value_is_named() {
        let m = message(
            "enum K { A, B }\nstruct P { k: K, n: i32 }\n\
             fn f(p: P) -> i32 { match p { P { k: K::A, n } => n } }",
        );
        assert!(m.contains("`P { k: K::B, n: _ }`"), "{m}");
    }

    #[test]
    fn integers_need_a_catch_all_unless_every_value_is_listed() {
        let m = message("fn f(x: i32) -> i32 { match x { 0 => 1, 1 => 2 } }");
        assert!(m.contains("`-1`"), "the missing value nearest zero: {m}");
        let every: Vec<String> = (0..256).map(|v| format!("{v} => 0,")).collect();
        check(&format!("fn f(x: u8) -> i32 {{ match x {{ {} }} }}", every.join(" ")));
    }

    #[test]
    fn booleans_are_covered_by_both_values() {
        check("fn f(b: bool) -> i32 { match b { true => 1, false => 0 } }");
        let m = message("fn f(b: bool) -> i32 { match b { true => 1 } }");
        assert!(m.contains("`false`"), "{m}");
    }

    #[test]
    fn an_arm_after_a_catch_all_is_unreachable() {
        let (_, d) = analyzed("fn f(x: i32) -> i32 { match x { _ => 0, 3 => 1 } }");
        let w = d.iter().find(|d| d.code == Code::Es14).expect("warned");
        assert!(!w.is_error());
        assert_eq!(
            codes("fn f(b: bool) -> i32 { match b { true => 1, false => 0, true => 2 } }"),
            [Code::Es14]
        );
    }
}
