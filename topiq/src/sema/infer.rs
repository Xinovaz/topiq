//! Settling the type of every unsuffixed number literal.
//!
//! `5` on its own could be any integer type. It takes the type its context
//! demands (`let x: u8 = 5` makes it a `u8`, and `a + 5` makes it whatever
//! `a` is), and when nothing demands one, it is an `i64`. Context can arrive
//! from anywhere in the function, including after the literal:
//!
//! ```text
//! let n = 5;          // not yet known
//! let m: u32 = n;     // now `n`, and so the `5`, is a u32
//! ```
//!
//! So each such literal starts as a variable, [`Ty::Infer`], and the checker
//! *unifies* variables with each other and with concrete types as it meets
//! them. Unification looks inside compound types, so `[1, 2]` meeting a
//! `[u8; 2]` makes both elements `u8`. When the body is finished,
//! [`settle_fn`] replaces every variable with what it was unified to, or
//! `i64`, and only then checks that each literal's value fits its type: a
//! `300` is only an error once it is known to be a `u8`.
//!
//! A floating literal works the same way, with `f64` as its default, and the
//! two families never mix: a variable that began as `1` never becomes an
//! `f32`, so `1 + 1.5` is a type error rather than a promotion. Unifying a
//! variable with `bool` or a structure is a type error too, reported where it
//! happened.

use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::tir::visit::{self, VisitMut};
use crate::tir::{
    Compound, Expr, ExprKind, FloatTy, Fn, InferId, IntTy, Pat, PatKind, Ty, TypeTable, Value,
};

use super::report;

/// What family a variable's literal belongs to. The two never mix: `1 + 1.5`
/// is a type error, not a promotion, and a variable of one kind never settles
/// as a type of the other.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// An integer literal: settles as an integer type, and `i64` by default.
    Integer,
    /// A floating literal: settles as a floating type, and `f64` by default.
    Floating,
    /// A variant of a generic enumeration that carries nothing, such as
    /// `Maybe::Nothing`: some instance of the enumeration declared with this
    /// name, which only its context can say. It has no default.
    Instance(Symbol),
    /// Any type at all, which only its uses can say: a generic argument that
    /// nothing written mentions, such as the `E` of `Ok(5)`. It has no
    /// default.
    Any,
}

/// The literal variables of one function body.
#[derive(Clone, Debug, Default)]
pub struct InferTable {
    parent: Vec<u32>,
    bound: Vec<Option<Ty>>,
    kind: Vec<Kind>,
    /// For an instance variable made by a variant that carries values, the
    /// arguments those values gave (some of them literals not yet settled).
    /// When the variable meets an instance, these meet its arguments, which
    /// is how `Just(3)` passed where a `Maybe<i32>` is wanted makes the `3` an
    /// `i32`.
    args: std::collections::HashMap<u32, Vec<Ty>>,
    /// For an instance variable, the unit that declared the generic: `None`
    /// for this unit's own. Two units may each declare a `Maybe`.
    origins: std::collections::HashMap<u32, Option<String>>,
}

impl InferTable {
    /// An empty table.
    pub fn new() -> InferTable {
        InferTable::default()
    }

    /// A new integer variable.
    pub fn fresh(&mut self) -> Ty {
        self.variable(Kind::Integer)
    }

    /// A new variable that may become any type.
    pub fn fresh_any(&mut self) -> Ty {
        self.variable(Kind::Any)
    }

    /// Whether `t` mentions a variable of any type that nothing has said.
    pub fn unsettled_any(&self, types: &TypeTable, t: Ty) -> bool {
        match self.shallow(t) {
            Ty::Infer(v) => self.kind[self.root(v.0) as usize] == Kind::Any,
            t @ (Ty::Ref(_) | Ty::Slice(_)) => types
                .as_ref(t)
                .or_else(|| types.as_slice(t))
                .is_some_and(|(_, x)| self.unsettled_any(types, x)),
            t @ Ty::Array(_) => types.as_array(t).is_some_and(|(x, _)| self.unsettled_any(types, x)),
            t @ Ty::Tuple(_) => types
                .as_tuple(t)
                .is_some_and(|xs| xs.iter().any(|&x| self.unsettled_any(types, x))),
            t @ (Ty::Fn(_) | Ty::Closure(_) | Ty::Circuit(_)) => types
                .as_sig(t)
                .is_some_and(|(p, r)| p.iter().any(|&x| self.unsettled_any(types, x)) || self.unsettled_any(types, r)),
            _ => false,
        }
    }

    /// A new floating variable.
    pub fn fresh_float(&mut self) -> Ty {
        self.variable(Kind::Floating)
    }

    /// A new variable standing for some instance of the generic enumeration
    /// `name` declared by the unit `origin` (`None` for this one).
    pub fn fresh_instance(&mut self, name: Symbol, origin: Option<String>) -> Ty {
        let t = self.variable(Kind::Instance(name));
        if let Ty::Infer(InferId(v)) = t {
            self.origins.insert(v, origin);
        }
        t
    }

    /// A new variable standing for the instance of `name` whose arguments
    /// are `args`, some of which are literals not yet settled.
    pub fn fresh_instance_with(&mut self, name: Symbol, origin: Option<String>, args: Vec<Ty>) -> Ty {
        let t = self.fresh_instance(name, origin);
        if let Ty::Infer(InferId(v)) = t {
            self.args.insert(v, args);
        }
        t
    }

    /// The arguments recorded for an unsettled instance variable, if it has
    /// any.
    pub fn pending_args(&self, t: Ty) -> Option<Vec<Ty>> {
        match self.shallow(t) {
            Ty::Infer(InferId(v)) => self.args.get(&v).cloned(),
            _ => None,
        }
    }

    /// If `t` is a variable standing for an instance that nothing has said,
    /// the name of the generic it is an instance of.
    pub fn unsettled_instance(&self, t: Ty) -> Option<Symbol> {
        match self.shallow(t) {
            Ty::Infer(InferId(v)) => match self.kind[v as usize] {
                Kind::Instance(n) => Some(n),
                _ => None,
            },
            _ => None,
        }
    }

    fn variable(&mut self, kind: Kind) -> Ty {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        self.bound.push(None);
        self.kind.push(kind);
        Ty::Infer(InferId(id))
    }

    fn root(&self, mut v: u32) -> u32 {
        while self.parent[v as usize] != v {
            v = self.parent[v as usize];
        }
        v
    }

    /// What a type is as far as is known now: a variable already unified with
    /// a concrete type is that type. Only the outermost level is looked
    /// through; a compound keeps its variables until it is settled.
    pub fn shallow(&self, t: Ty) -> Ty {
        match t {
            Ty::Infer(InferId(v)) => {
                let r = self.root(v);
                match self.bound[r as usize] {
                    Some(t) => t,
                    None => Ty::Infer(InferId(r)),
                }
            }
            other => other,
        }
    }

    /// Whether a variable stands for an integer literal. A type that is not a
    /// variable is not one.
    pub fn is_integer_variable(&self, t: Ty) -> bool {
        match t {
            Ty::Infer(InferId(v)) => self.kind[self.root(v) as usize] == Kind::Integer,
            _ => false,
        }
    }

    /// Whether a variable stands for a floating literal. A type that is not a
    /// variable is not one.
    pub fn is_floating_variable(&self, t: Ty) -> bool {
        match t {
            Ty::Infer(InferId(v)) => self.kind[self.root(v) as usize] == Kind::Floating,
            _ => false,
        }
    }

    /// Makes two types the same, returning the type they now share.
    ///
    /// `never` unifies with anything and yields the other type, because a
    /// value that never arrives can stand wherever any type is expected.
    ///
    /// # Errors
    ///
    /// When the two types cannot be the same, with both as they now stand, for
    /// the caller to report.
    pub fn unify(&mut self, types: &TypeTable, a: Ty, b: Ty) -> Result<Ty, (Ty, Ty)> {
        let (a, b) = (self.shallow(a), self.shallow(b));
        match (a, b) {
            (Ty::Never, t) | (t, Ty::Never) => Ok(t),
            // a variable of any type becomes the other, or joins a variable
            // that knows more
            (Ty::Infer(InferId(x)), t) | (t, Ty::Infer(InferId(x))) if self.kind[x as usize] == Kind::Any => {
                if t == Ty::Infer(InferId(x)) {
                    return Ok(t);
                }
                match t {
                    Ty::Infer(InferId(y)) => self.parent[x as usize] = y,
                    other => self.bound[x as usize] = Some(other),
                }
                Ok(t)
            }
            (Ty::Infer(InferId(x)), Ty::Infer(InferId(y))) => {
                if self.kind[x as usize] != self.kind[y as usize] || self.origins.get(&x) != self.origins.get(&y) {
                    return Err((a, b));
                }
                if let Kind::Instance(_) = self.kind[x as usize] {
                    // two instances of one generic: their arguments agree
                    if let (Some(xa), Some(ya)) = (self.args.get(&x).cloned(), self.args.get(&y).cloned()) {
                        for (p, q) in xa.into_iter().zip(ya) {
                            self.unify(types, p, q).map_err(|_| (a, b))?;
                        }
                    } else if let Some(xa) = self.args.remove(&x) {
                        self.args.insert(y, xa);
                    }
                }
                if x != y {
                    self.parent[x as usize] = y;
                }
                Ok(Ty::Infer(InferId(y)))
            }
            (Ty::Infer(InferId(x)), t @ Ty::Adt(id)) | (t @ Ty::Adt(id), Ty::Infer(InferId(x))) => {
                let def = types.adt(id);
                match self.kind[x as usize] {
                    Kind::Instance(n)
                        if def.name == n
                            && !def.args.is_empty()
                            && self.origins.get(&x).is_some_and(|o| *o == def.origin) =>
                    {
                        let theirs = def.args.clone();
                        if let Some(mine) = self.args.get(&x).cloned() {
                            for (m, o) in mine.into_iter().zip(theirs) {
                                if let crate::tir::Arg::Type(o) = o {
                                    self.unify(types, m, o).map_err(|_| (a, b))?;
                                }
                            }
                        }
                        self.bound[x as usize] = Some(t);
                        Ok(t)
                    }
                    _ => Err((a, b)),
                }
            }
            (Ty::Infer(InferId(x)), t @ (Ty::Int(_) | Ty::Float(_)))
            | (t @ (Ty::Int(_) | Ty::Float(_)), Ty::Infer(InferId(x))) => {
                let wants = if t.is_float() { Kind::Floating } else { Kind::Integer };
                if self.kind[x as usize] != wants {
                    return Err((a, b));
                }
                self.bound[x as usize] = Some(t);
                Ok(t)
            }
            (x, y) if x == y => Ok(x),
            // two instances of one generic, one made with `never` where the
            // other has a type because of an error already reported
            (Ty::Adt(i), Ty::Adt(j)) => {
                let (p, q) = (types.adt(i), types.adt(j));
                let mentions_never = |args: &[crate::tir::Arg]| {
                    args.iter()
                        .any(|a| matches!(a, crate::tir::Arg::Type(t) if types_mention_never(types, *t)))
                };
                if p.name != q.name
                    || p.origin != q.origin
                    || p.args.len() != q.args.len()
                    || !(mentions_never(&p.args) || mentions_never(&q.args))
                {
                    return Err((a, b));
                }
                for (x, y) in p.args.clone().into_iter().zip(q.args.clone()) {
                    match (x, y) {
                        (crate::tir::Arg::Type(x), crate::tir::Arg::Type(y)) => {
                            self.unify(types, x, y).map_err(|_| (a, b))?;
                        }
                        (x, y) if x == y => {}
                        _ => return Err((a, b)),
                    }
                }
                Ok(if mentions_never(&q.args) { a } else { b })
            }
            (Ty::Fn(_), Ty::Fn(_)) | (Ty::Closure(_), Ty::Closure(_)) | (Ty::Circuit(_), Ty::Circuit(_)) => {
                let (xp, xr) = types.as_sig(a).map(|(p, r)| (p.to_vec(), r)).expect("a signature");
                let (yp, yr) = types.as_sig(b).map(|(p, r)| (p.to_vec(), r)).expect("a signature");
                if xp.len() != yp.len() {
                    return Err((a, b));
                }
                for (&x, &y) in xp.iter().zip(&yp) {
                    self.unify(types, x, y).map_err(|_| (a, b))?;
                }
                self.unify(types, xr, yr).map_err(|_| (a, b))?;
                Ok(b)
            }
            (Ty::Tuple(_), Ty::Tuple(_)) => {
                let xs = types.as_tuple(a).unwrap_or(&[]).to_vec();
                let ys = types.as_tuple(b).unwrap_or(&[]).to_vec();
                if xs.len() != ys.len() {
                    return Err((a, b));
                }
                for (&x, &y) in xs.iter().zip(&ys) {
                    self.unify(types, x, y).map_err(|_| (a, b))?;
                }
                Ok(if types_mention_never(types, b) { a } else { b })
            }
            (Ty::Ref(i), Ty::Ref(j))
            | (Ty::Slice(i), Ty::Slice(j))
            | (Ty::Array(i), Ty::Array(j))
            | (Ty::Growable(i), Ty::Growable(j)) => {
                let inner = match (types.compound(i), types.compound(j)) {
                    (
                        Compound::Ref {
                            access: p,
                            target: s,
                        },
                        Compound::Ref {
                            access: q,
                            target: t,
                        },
                    )
                    | (Compound::Slice { access: p, elem: s }, Compound::Slice { access: q, elem: t })
                        if p == q =>
                    {
                        Some((s, t))
                    }
                    (Compound::Array { elem: s, len: m }, Compound::Array { elem: t, len: n }) if m == n => {
                        Some((s, t))
                    }
                    (Compound::Growable { elem: s }, Compound::Growable { elem: t }) => Some((s, t)),
                    _ => None,
                };
                match inner {
                    Some((s, t)) => match self.unify(types, s, t) {
                        // whichever side had no `never` inside is the more
                        // precise; prefer the one that was expected
                        Ok(_) => Ok(if types_mention_never(types, b) { a } else { b }),
                        Err(_) => Err((a, b)),
                    },
                    None => Err((a, b)),
                }
            }
            (x, y) => Err((x, y)),
        }
    }

    /// The final type: every variable replaced by what it was unified with,
    /// and a variable nothing constrained by `i64`, or `f64` if it stands for
    /// a floating literal.
    pub fn settle(&self, types: &mut TypeTable, t: Ty) -> Ty {
        types.map_leaves(t, &mut |leaf| match self.shallow(leaf) {
            Ty::Infer(v) => match self.kind[self.root(v.0) as usize] {
                Kind::Integer => Ty::Int(IntTy::I64),
                Kind::Floating => Ty::Float(FloatTy::F64),
                // nothing said which instance; that is reported where the
                // variant was written
                Kind::Instance(_) | Kind::Any => Ty::Never,
            },
            other => other,
        })
    }

    /// `t` with every variable resolved as far as is known now, for a message.
    pub fn resolved(&self, types: &mut TypeTable, t: Ty) -> Ty {
        types.map_leaves(t, &mut |leaf| self.shallow(leaf))
    }
}

/// Whether `ty` is or contains `never`, as the element type of an empty array
/// literal does.
fn types_mention_never(types: &TypeTable, ty: Ty) -> bool {
    match ty {
        Ty::Never => true,
        Ty::Ref(id) | Ty::Slice(id) | Ty::Array(id) | Ty::Growable(id) => match types.compound(id) {
            Compound::Ref { target: t, .. }
            | Compound::Slice { elem: t, .. }
            | Compound::Array { elem: t, .. }
            | Compound::Growable { elem: t }
            | Compound::Qmap { entry: t, .. } => types_mention_never(types, t),
        },
        Ty::Tuple(_) => types
            .as_tuple(ty)
            .is_some_and(|es| es.iter().any(|&e| types_mention_never(types, e))),
        _ => false,
    }
}

/// Replaces every variable in a function with its final type, and checks every
/// integer literal against the type it ended up with.
pub fn settle_fn(f: &mut Fn, table: &InferTable, types: &mut TypeTable, diags: &mut Vec<Diagnostic>) {
    for l in &mut f.locals {
        l.ty = table.settle(types, l.ty);
    }
    f.ret = table.settle(types, f.ret);
    let mut s = Settler { table, types, diags };
    s.block(&mut f.body);
}

/// Settles a lone expression, such as a unit-scope initialiser.
pub fn settle_expr(e: &mut Expr, table: &InferTable, types: &mut TypeTable, diags: &mut Vec<Diagnostic>) {
    let mut s = Settler { table, types, diags };
    s.expr(e);
}

struct Settler<'a> {
    table: &'a InferTable,
    types: &'a mut TypeTable,
    diags: &'a mut Vec<Diagnostic>,
}

impl VisitMut for Settler<'_> {
    fn expr(&mut self, e: &mut Expr) {
        visit::walk_expr_mut(self, e);
        e.ty = self.table.settle(self.types, e.ty);
        if let ExprKind::Const(v) = &mut e.kind {
            fix_literal(v, e.ty, e.span, self.diags);
        }
    }

    fn block(&mut self, b: &mut crate::tir::Block) {
        visit::walk_block_mut(self, b);
        b.ty = self.table.settle(self.types, b.ty);
    }

    fn pat(&mut self, p: &mut Pat) {
        visit::walk_pat_mut(self, p);
        p.ty = self.table.settle(self.types, p.ty);
        if let PatKind::Const(v) = &mut p.kind {
            fix_literal(v, p.ty, p.span, self.diags);
        }
    }
}

/// Gives a literal its settled type, reporting an integer that does not fit.
///
/// A floating literal always fits: a value too large for the type becomes an
/// infinity and one too small becomes zero, which is what [IEEE 754] rounding
/// says and never an error.
///
/// [IEEE 754]: https://en.wikipedia.org/wiki/IEEE_754
fn fix_literal(v: &mut Value, ty: Ty, span: crate::span::Span, diags: &mut Vec<Diagnostic>) {
    match (v, ty) {
        (Value::Int(x, t), Ty::Int(it)) => {
            *t = it;
            if !it.fits(*x) {
                diags.push(out_of_range(span, *x, it));
            }
        }
        (Value::Float(x, t), Ty::Float(ft)) => {
            *t = ft;
            *x = ft.round(*x);
        }
        _ => {}
    }
}

/// A literal whose value does not fit the type it was given.
fn out_of_range(span: crate::span::Span, v: i128, t: IntTy) -> Diagnostic {
    Diagnostic::new(Code::Es06)
        .with_message(format!(
            "{v} does not fit in {}, which holds {} to {}",
            report::describe_int(t),
            t.min(),
            t.max()
        ))
        .at(span)
        .with_note(
            "an integer literal takes its type from where it is used, and this \
             position requires the type above",
        )
        .with_help(
            "use a wider type there, or give the literal a suffix naming the type you \
             mean, such as `300u16`",
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::{SourceId, Span};
    use crate::tir::Access;

    fn sp() -> Span {
        Span::new(SourceId(0), 0, 1)
    }

    #[test]
    fn a_variable_takes_the_concrete_type_it_meets() {
        let mut types = TypeTable::new();
        let mut t = InferTable::new();
        let v = t.fresh();
        assert_eq!(t.unify(&types, v, Ty::Int(IntTy::U8)), Ok(Ty::Int(IntTy::U8)));
        assert_eq!(t.shallow(v), Ty::Int(IntTy::U8));
        assert_eq!(t.settle(&mut types, v), Ty::Int(IntTy::U8));
    }

    #[test]
    fn a_variable_nothing_constrains_is_an_i64() {
        let mut types = TypeTable::new();
        let mut t = InferTable::new();
        let v = t.fresh();
        assert_eq!(t.settle(&mut types, v), Ty::Int(IntTy::I64));
    }

    #[test]
    fn two_variables_joined_then_bound_are_both_bound() {
        let mut types = TypeTable::new();
        let mut t = InferTable::new();
        let (a, b) = (t.fresh(), t.fresh());
        t.unify(&types, a, b).unwrap();
        t.unify(&types, b, Ty::Int(IntTy::I16)).unwrap();
        assert_eq!(t.settle(&mut types, a), Ty::Int(IntTy::I16));
        assert_eq!(t.settle(&mut types, b), Ty::Int(IntTy::I16));
    }

    #[test]
    fn a_bound_variable_rejects_a_different_type() {
        let types = TypeTable::new();
        let mut t = InferTable::new();
        let v = t.fresh();
        t.unify(&types, v, Ty::Int(IntTy::U8)).unwrap();
        assert_eq!(
            t.unify(&types, v, Ty::Int(IntTy::I32)),
            Err((Ty::Int(IntTy::U8), Ty::Int(IntTy::I32)))
        );
    }

    #[test]
    fn a_variable_is_never_a_bool() {
        let types = TypeTable::new();
        let mut t = InferTable::new();
        let v = t.fresh();
        assert!(t.unify(&types, v, Ty::Bool).is_err());
        assert!(t.unify(&types, Ty::Void, v).is_err());
    }

    #[test]
    fn never_unifies_with_anything() {
        let types = TypeTable::new();
        let mut t = InferTable::new();
        assert_eq!(t.unify(&types, Ty::Never, Ty::Bool), Ok(Ty::Bool));
        assert_eq!(t.unify(&types, Ty::Int(IntTy::U8), Ty::Never), Ok(Ty::Int(IntTy::U8)));
        let v = t.fresh();
        assert!(matches!(t.unify(&types, Ty::Never, v), Ok(Ty::Infer(_))));
    }

    #[test]
    fn concrete_types_unify_only_with_themselves() {
        let types = TypeTable::new();
        let mut t = InferTable::new();
        assert_eq!(t.unify(&types, Ty::Bool, Ty::Bool), Ok(Ty::Bool));
        assert!(t.unify(&types, Ty::Int(IntTy::U64), Ty::Int(IntTy::USIZE)).is_err());
    }

    #[test]
    fn unification_looks_inside_arrays_and_references() {
        let mut types = TypeTable::new();
        let mut t = InferTable::new();
        let v = t.fresh();
        let loose = types.array(v, 3);
        let exact = types.array(Ty::Int(IntTy::U8), 3);
        assert!(t.unify(&types, loose, exact).is_ok());
        assert_eq!(t.shallow(v), Ty::Int(IntTy::U8));
        assert_eq!(t.settle(&mut types, loose), exact);

        let longer = types.array(Ty::Int(IntTy::U8), 4);
        assert!(t.unify(&types, exact, longer).is_err(), "lengths must agree");

        let writes = types.reference(Access::Write, exact);
        let reads = types.reference(Access::Const, exact);
        assert!(t.unify(&types, writes, reads).is_err(), "access is part of the type");
    }

    #[test]
    fn an_empty_array_takes_the_element_type_it_meets() {
        let mut types = TypeTable::new();
        let mut t = InferTable::new();
        let empty = types.array(Ty::Never, 0);
        let bools = types.array(Ty::Bool, 0);
        assert_eq!(t.unify(&types, empty, bools), Ok(bools));
        assert_eq!(t.unify(&types, bools, empty), Ok(bools));
    }

    #[test]
    fn settling_fixes_a_literal_and_checks_its_range() {
        let mut types = TypeTable::new();
        let mut t = InferTable::new();
        let v = t.fresh();
        t.unify(&types, v, Ty::Int(IntTy::U8)).unwrap();
        let mut ok = Expr {
            kind: ExprKind::Const(Value::Int(255, IntTy::I64)),
            ty: v,
            span: sp(),
        };
        let mut diags = Vec::new();
        settle_expr(&mut ok, &t, &mut types, &mut diags);
        assert!(diags.is_empty());
        assert!(matches!(ok.kind, ExprKind::Const(Value::Int(255, IntTy::U8))));

        let mut bad = Expr {
            kind: ExprKind::Const(Value::Int(300, IntTy::I64)),
            ty: v,
            span: sp(),
        };
        settle_expr(&mut bad, &t, &mut types, &mut diags);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, Code::Es06);
        assert!(diags[0].message.contains("300 does not fit in a u8"), "{}", diags[0].message);
    }
}
