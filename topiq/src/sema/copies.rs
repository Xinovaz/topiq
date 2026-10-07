//! Implicit copies of types that define `$copy`.
//!
//! A structure or enumeration with `$copy` is copyable: reading it from a
//! place leaves the place as it was and gives a copy, which `$copy` makes.
//! Move checking treats such a read like any other copy. This pass then
//! writes out each one as the call it is: every place of such a type read for
//! its value becomes `$copy(&place)`. A fixed array or tuple of such values is
//! copied part by part, each part with its own `$copy`.
//!
//! A place that is only looked at (assigned to, lent with `&`, indexed, or
//! matched without binding anything that needs copying) is not copied. One
//! that a pattern takes apart and binds a part of is copied whole first, and
//! the copy taken apart, when its type has a `$copy`. When only the part's
//! type has one, as for an `Opt<frac>`, the pattern binds a reference to
//! the part instead, and the binding is declared as a copy made through it;
//! the place keeps its own value, and each is destroyed once.
//!
//! Within a type's own `$copy`, a value of that type is copied byte for byte,
//! so that it can read the value it is given without calling itself; values
//! of other types are copied as anywhere else.

use std::collections::HashMap;

use crate::tir::visit::{self, VisitMut};
use crate::tir::{
    AdtId, Arm, Block, Callee, Expr, ExprKind, FnId, IntTy, Intrinsic, Local, LocalId, Pat, PatKind, Stmt, Ty, Unit,
    Value,
};

/// Rewrites every implicit copy of a value whose type defines `$copy`.
pub fn expand(unit: &mut Unit) {
    if unit.copies.is_empty() {
        return;
    }
    // everything the rewritten expressions need of the type table is made
    // first, so that the walk only reads it
    let refs: HashMap<AdtId, (Callee, Ty)> = unit
        .copies
        .clone()
        .into_iter()
        .map(|(id, callee)| {
            let r = unit.types.reference(crate::tir::Access::Const, Ty::Adt(id));
            (id, (callee, r))
        })
        .collect();
    // each `$copy` and the type it copies
    let own: HashMap<FnId, AdtId> = unit
        .copies
        .iter()
        .filter_map(|(&id, c)| match c {
            Callee::Fn(f) => Some((*f, id)),
            Callee::Extern(_) => None,
        })
        .collect();
    // a reference to each type a pattern binds that is copied with some
    // `$copy`, for the parts of a place that is not copied whole
    let bound: Vec<Ty> = {
        let mut found = Bound(Vec::new());
        for f in &unit.fns {
            crate::tir::visit::Visit::block(&mut found, &f.body);
        }
        let types = unit.types.clone();
        let probe = Copies {
            types: &types,
            refs: &refs,
            within: None,
            by_ref: &HashMap::new(),
            locals: &mut Vec::new(),
        };
        found.0.into_iter().filter(|&t| probe.needs(t)).collect()
    };
    let by_ref: HashMap<Ty, Ty> = bound
        .into_iter()
        .map(|t| (t, unit.types.reference(crate::tir::Access::Write, t)))
        .collect();
    let types = unit.types.clone();
    for (i, f) in unit.fns.iter_mut().enumerate() {
        let mut pass = Copies {
            types: &types,
            refs: &refs,
            within: own.get(&FnId(i as u32)).copied(),
            by_ref: &by_ref,
            locals: &mut f.locals,
        };
        pass.block(&mut f.body);
    }
}

/// The types of the values patterns bind.
struct Bound(Vec<Ty>);

impl crate::tir::visit::Visit for Bound {
    fn pat(&mut self, p: &Pat) {
        if matches!(p.kind, PatKind::Bind(_)) && !self.0.contains(&p.ty) {
            self.0.push(p.ty);
        }
        crate::tir::visit::walk_pat(self, p);
    }
}

struct Copies<'a> {
    types: &'a crate::tir::TypeTable,
    refs: &'a HashMap<AdtId, (Callee, Ty)>,
    /// In a type's own `$copy`, that type: copied there byte for byte.
    within: Option<AdtId>,
    /// A reference to each type a pattern binds that needs copying.
    by_ref: &'a HashMap<Ty, Ty>,
    /// The locals of the function being rewritten.
    locals: &'a mut Vec<Local>,
}

impl Copies<'_> {
    /// Whether copying a value of `ty` runs some `$copy`.
    fn needs(&self, ty: Ty) -> bool {
        match ty {
            Ty::Adt(id) => self.refs.contains_key(&id) && self.within != Some(id),
            Ty::Array(_) => self.types.as_array(ty).is_some_and(|(e, _)| self.needs(e)),
            Ty::Tuple(_) => {
                self.types.is_copyable(ty) && self.types.as_tuple(ty).is_some_and(|es| es.iter().any(|&e| self.needs(e)))
            }
            _ => false,
        }
    }

    /// Whether a pattern binds a part whose copy runs some `$copy`.
    fn binds_copied(&self, p: &Pat) -> bool {
        match &p.kind {
            PatKind::Bind(_) => self.needs(p.ty),
            PatKind::Struct { fields } | PatKind::Variant { fields, .. } => {
                fields.iter().any(|(_, f)| self.binds_copied(f))
            }
            PatKind::Wild | PatKind::BindRef(_) | PatKind::Const(_) => false,
        }
    }

    /// A place that is looked at, not read: its own parts that are values
    /// (an index, the reference a dereference follows) are still visited.
    fn place(&mut self, e: &mut Expr) {
        match &mut e.kind {
            ExprKind::Field { base, .. } => self.place(base),
            ExprKind::Index { base, index } => {
                self.place(base);
                self.expr(index);
            }
            ExprKind::Deref(r) => self.expr(r),
            ExprKind::Local(_) | ExprKind::Global(_) => {}
            // not a place: a temporary, whose value is used up
            _ => self.expr(e),
        }
    }

    /// A scrutinee: looked at, unless the patterns bind a part that must be
    /// copied, in which case the whole is copied first if its type copies it.
    /// Whether the patterns must then bind references to the parts they
    /// copy, which is when the whole is a place that is not copied.
    fn scrutinee<'p>(&mut self, e: &mut Expr, pats: impl IntoIterator<Item = &'p Pat>) -> bool {
        let binds = pats.into_iter().any(|p| self.binds_copied(p));
        if e.is_place() && !(self.needs(e.ty) && binds) {
            self.place(e);
            binds
        } else {
            self.expr(e);
            false
        }
    }

    /// Makes each binding of `p` whose value is copied with `$copy` bind a
    /// reference to the part instead, and gives the statements that then
    /// declare the binding as a copy of the part.
    fn bind_by_reference(&mut self, p: &mut Pat) -> Vec<Stmt> {
        let mut out = Vec::new();
        self.rebind(p, &mut out);
        out
    }

    fn rebind(&mut self, p: &mut Pat, out: &mut Vec<Stmt>) {
        match &mut p.kind {
            PatKind::Bind(l) if self.needs(p.ty) => {
                let l = *l;
                let r = LocalId(self.locals.len() as u32);
                let ref_ty = self.by_ref[&p.ty];
                let span = p.span;
                self.locals.push(Local {
                    ty: ref_ty,
                    constant: false,
                    constexpr: false,
                    aux: false,
                    ..self.locals[l.index()].clone()
                });
                p.kind = PatKind::BindRef(r);
                let part = Expr::deref(Expr::local(r, ref_ty, span), p.ty, span);
                out.push(Stmt::Let {
                    local: l,
                    init: Some(self.copy(part)),
                });
            }
            PatKind::Struct { fields } | PatKind::Variant { fields, .. } => {
                for (_, f) in fields {
                    self.rebind(f, out);
                }
            }
            _ => {}
        }
    }

    /// A copy of the value at the place `e`.
    fn copy(&self, e: Expr) -> Expr {
        let (ty, span) = (e.ty, e.span);
        match ty {
            Ty::Adt(id) => {
                let (callee, r) = self.refs[&id];
                Expr {
                    kind: ExprKind::Call {
                        callee,
                        args: vec![Expr {
                            kind: ExprKind::Ref(Box::new(e)),
                            ty: r,
                            span,
                        }],
                    },
                    ty,
                    span,
                }
            }
            Ty::Array(_) => {
                let (elem, n) = self.types.as_array(ty).expect("an array");
                let items = (0..n)
                    .map(|i| {
                        let at = Expr {
                            kind: ExprKind::Index {
                                base: Box::new(e.clone()),
                                index: Box::new(Expr::constant(Value::Int(i128::from(i), IntTy::USIZE), Ty::USIZE, span)),
                            },
                            ty: elem,
                            span,
                        };
                        self.part(at)
                    })
                    .collect();
                Expr {
                    kind: ExprKind::ArrayLit(items),
                    ty,
                    span,
                }
            }
            Ty::Tuple(_) => {
                let elems = self.types.as_tuple(ty).expect("a tuple").to_vec();
                let fields = elems
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| {
                        let at = Expr {
                            kind: ExprKind::Field {
                                base: Box::new(e.clone()),
                                field: i as u32,
                            },
                            ty: t,
                            span,
                        };
                        (i as u32, self.part(at))
                    })
                    .collect();
                Expr {
                    kind: ExprKind::StructLit { fields },
                    ty,
                    span,
                }
            }
            _ => e,
        }
    }

    /// A part of a value being copied: copied with `$copy` if it needs it,
    /// and otherwise read as it is.
    fn part(&self, e: Expr) -> Expr {
        if self.needs(e.ty) { self.copy(e) } else { e }
    }
}

impl VisitMut for Copies<'_> {
    fn expr(&mut self, e: &mut Expr) {
        // a place read for its value
        if e.is_place() && !matches!(e.kind, ExprKind::FnRef(_) | ExprKind::ThunkRef(_)) {
            self.place(e);
            if self.needs(e.ty) {
                let span = e.span;
                let read = std::mem::replace(e, Expr::constant(Value::Void, Ty::Void, span));
                *e = self.copy(read);
            }
            return;
        }
        match &mut e.kind {
            ExprKind::Ref(x) => self.place(x),
            ExprKind::Assign { place, value } => {
                self.expr(value);
                self.place(place);
            }
            ExprKind::Growable { array, args, .. } => {
                self.place(array);
                for a in args {
                    self.expr(a);
                }
            }
            ExprKind::Intrinsic {
                which: Intrinsic::Len,
                args,
            } => {
                for a in args {
                    self.place(a);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                let by_ref = self.scrutinee(scrutinee, arms.iter().map(|a: &Arm| &a.pat));
                for a in arms {
                    self.expr(&mut a.body);
                    if by_ref {
                        let stmts = self.bind_by_reference(&mut a.pat);
                        if !stmts.is_empty() {
                            let (ty, span) = (a.body.ty, a.body.span);
                            let body = std::mem::replace(&mut a.body, Expr::constant(Value::Void, Ty::Void, span));
                            a.body = Expr {
                                kind: ExprKind::Block(Box::new(Block {
                                    stmts,
                                    value: Some(Box::new(body)),
                                    ty,
                                    span,
                                })),
                                ty,
                                span,
                            };
                        }
                    }
                }
            }
            _ => visit::walk_expr_mut(self, e),
        }
    }

    fn block(&mut self, b: &mut Block) {
        let mut i = 0;
        while i < b.stmts.len() {
            let mut after = Vec::new();
            match &mut b.stmts[i] {
                Stmt::Let { init, .. } => {
                    if let Some(e) = init {
                        self.expr(e);
                    }
                }
                Stmt::LetPat { pat, init } => {
                    let p = pat.clone();
                    if self.scrutinee(init, [&p]) {
                        after = self.bind_by_reference(pat);
                    }
                }
                Stmt::Expr(e) => self.expr(e),
            }
            let n = after.len();
            b.stmts.splice(i + 1..i + 1, after);
            i += 1 + n;
        }
        if let Some(v) = &mut b.value {
            self.expr(v);
        }
    }
}
