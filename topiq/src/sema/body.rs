//! Checking one function body, one unit-scope initialiser, or one constant
//! such as an array length.
//!
//! A body is resolved and type-checked in a single walk. Block scope is
//! ordered, so a name's meaning depends on where in the walk it is met, which
//! makes resolving it at the moment it is met the natural thing to do, rather
//! than a separate pass that would have to rebuild the same scopes.
//!
//! The walk itself lives in [`super::expr`], [`super::stmt`] and the modules
//! beside them; this module holds the state they share and the entry points.
//!
//! # Recovering from an error
//!
//! After reporting a problem the checker carries on, so that one mistake does
//! not hide the next. The expression it gives back for something ill-formed
//! has type `never`, which unifies with every type, so the mistake is reported
//! once, where it is, and not again by everything that uses the result.
//!
//! # Conversions at a boundary
//!
//! Where a value flows somewhere with a declared type (an argument, a `let`
//! with a type, an assignment, a returned value, a field of a structure
//! literal), a reference may convert to one that promises less: `*T` to
//! `*const T`, and a reference to an array `*[T; N]` to a slice `*[T]`; and an array written out, `[a, b]` or `[x; n]`, may become a
//! growable array `[T]`. [`Checker::coerce`] applies those. Nowhere else does
//! any value change type.

use crate::ast::{self, FnItem};
use crate::diag::{Code, Diagnostic};
use crate::intern::{Interner, Symbol};
use crate::span::{Span, Spanned};
use crate::tir::{
    Access, AdtDef, AdtId, Arg, Expr, ExprKind, FnId, GlobalId, Local, LocalId, LoopId, Ty, TypeTable, Value,
};

use super::core;
use super::exhaust;
use super::infer::{self, InferTable};
use super::items::UnitCx;
use super::report;
use super::scope::{Def, Scopes};
use super::types::{self, Resolved};

/// What kind of loop a `break` or `continue` is inside.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoopKind {
    /// `loop`, which a `break` may leave with a value.
    Loop,
    /// `while`.
    While,
    /// `for`.
    For,
}

/// One enclosing loop.
#[derive(Clone, Copy, Debug)]
pub struct LoopFrame {
    /// The loop's id in the function.
    pub id: LoopId,
    /// What kind of loop it is.
    pub kind: LoopKind,
    /// For a `loop`, the type its `break`s give it, once one has been seen.
    pub value: Option<Ty>,
}

/// Two types that cannot be made the same.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Mismatch;

/// The state of a walk over one body.
pub struct Checker<'c, 'a> {
    /// The unit being analysed.
    pub cx: &'c mut UnitCx<'a>,
    /// For resolving and printing names.
    pub interner: &'a Interner,
    /// The function being checked, or `None` for a unit-scope initialiser or
    /// a constant.
    pub func: Option<FnId>,
    /// The bindings of the function so far.
    pub locals: Vec<Local>,
    /// The block scopes currently open.
    pub scopes: Scopes,
    /// The loops currently open.
    pub loops: Vec<LoopFrame>,
    /// The next loop id to hand out.
    pub next_loop: u32,
    /// Integer inference variables.
    pub infer: InferTable,
    /// The function's return type.
    pub ret: Ty,
    /// Persistent objects declared in this body, whose initialisers are
    /// settled together with it.
    pub persist: Vec<GlobalId>,
    /// Whether the indexing being checked only reads its element: it is the
    /// whole value of a new `const` binding, or the base of one that is, so
    /// a type's `$index_rd` serves it.
    pub read_index: bool,
    /// Variants of generic enumerations whose instance only their context
    /// can say, as `Maybe::Nothing` written where a `Maybe<char>` is wanted,
    /// with where each was written and what it is called. One that nothing
    /// settles by the end of the body is reported.
    pub pending: Vec<Pending>,
    /// In a closure's body: the parameter holding a reference to its
    /// environment, and whether each capture is held by reference.
    pub env: Option<(LocalId, Vec<bool>)>,
    /// While the arms of a `match` on a reference are checked, what that
    /// reference permits: a name there binds a part that cannot be copied by
    /// reference.
    pub match_through: Option<Access>,
    /// Whether the operator being checked is marked `[trusted_unitary]`,
    /// which makes the program answerable for the matrices `apply` is given
    /// in it.
    pub trusted: bool,
}

/// A variant of a generic enumeration whose instance is not yet known.
#[derive(Clone, Debug)]
pub struct Pending {
    /// The instance variable standing for its type.
    pub ty: Ty,
    /// Where it was written.
    pub span: Span,
    /// How it was written, for a message: `Maybe::Nothing`.
    pub what: String,
    /// The generic it is an instance of, and the arguments the values it
    /// carries gave (`None` for a variant that carries nothing, which has
    /// no arguments to fall back on).
    pub args: Option<(super::items::GenericKey, Vec<crate::tir::Arg>)>,
}

impl<'c, 'a> Checker<'c, 'a> {
    /// A checker at unit scope.
    pub fn new(cx: &'c mut UnitCx<'a>) -> Checker<'c, 'a> {
        let interner = cx.interner;
        Checker {
            cx,
            interner,
            func: None,
            locals: Vec::new(),
            scopes: Scopes::new(),
            loops: Vec::new(),
            next_loop: 0,
            pending: Vec::new(),
            infer: InferTable::new(),
            ret: Ty::Void,
            persist: Vec::new(),
            read_index: false,
            env: None,
            match_through: None,
            trusted: false,
        }
    }

    /// Reports a problem.
    pub fn report(&mut self, d: Diagnostic) {
        self.cx.diags.push(d);
    }

    /// Checks each of `args` for what it reports, the values unused: the
    /// arguments of a call that is not made.
    pub(super) fn check_only(&mut self, args: &[Spanned<ast::Expr>]) {
        for a in args {
            self.expr(a);
        }
    }

    /// `name` takes `want` arguments and a call gives `got`.
    pub(super) fn wrong_arity(name: &str, want: usize, got: usize, span: Span) -> Diagnostic {
        let noun = if want == 1 { "argument" } else { "arguments" };
        Diagnostic::new(Code::Es07)
            .with_message(format!("`{name}` takes {want} {noun}, but this call gives {got}"))
            .at(span)
    }

    /// The stand-in for an expression that could not be checked: it has type
    /// `never`, so nothing that uses it reports the same mistake again.
    pub fn error(span: Span) -> Expr {
        Expr::constant(Value::Void, Ty::Never, span)
    }

    /// Reports `what` as a construct the compiler does not translate (`TQ003`),
    /// and stands in for it.
    pub fn unsupported(&mut self, span: Span, what: &str) -> Expr {
        self.report(report::unsupported(span, what));
        Checker::error(span)
    }

    /// A name as written.
    pub fn name(&self, sym: Symbol) -> &'a str {
        self.interner.resolve(sym)
    }

    /// The unit's types.
    pub fn types(&self) -> &TypeTable {
        &self.cx.unit.types
    }

    /// The unit's types.
    pub fn types_mut(&mut self) -> &mut TypeTable {
        &mut self.cx.unit.types
    }

    /// A structure or enumeration.
    pub fn adt(&mut self, id: AdtId) -> &AdtDef {
        self.cx.adt(id);
        self.cx.unit.types.adt(id)
    }

    /// Adds a binding to the function.
    pub fn new_local(&mut self, name: Symbol, ty: Ty, constant: bool, span: Span) -> LocalId {
        self.cx.check_reserved(name, span);
        let id = LocalId(self.locals.len() as u32);
        self.locals.push(Local {
            name,
            ty,
            constant,
            constexpr: false,
            aux: false,
            span,
        });
        id
    }

    /// A fresh loop id.
    pub fn fresh_loop(&mut self) -> LoopId {
        let id = LoopId(self.next_loop);
        self.next_loop += 1;
        id
    }

    /// A type as far as is known now.
    pub fn shallow(&self, t: Ty) -> Ty {
        self.infer.shallow(t)
    }

    /// Reports each variant of a generic enumeration whose instance nothing
    /// said.
    fn report_unsettled(&mut self) {
        for p in std::mem::take(&mut self.pending) {
            if p.args.is_some() {
                // its literals had their chance to settle; now they take their
                // defaults, and the instance is made from them
                self.force(&p);
                continue;
            }
            let (t, span, what) = (p.ty, p.span, p.what);
            if self.infer.unsettled_instance(t).is_some() {
                self.report(
                    Diagnostic::new(Code::Es07)
                        .with_message(format!("nothing here says what the arguments of `{what}` are"))
                        .at(span)
                        .with_note(
                            "a variant that carries nothing takes its enumeration's arguments from \
                             where it is used, and nothing here uses it as any particular one",
                        )
                        .with_help(format!("write them, as in `{what}::<…>`")),
                );
            }
        }
    }

    /// A type as it would finally settle: what is known now, with any literal
    /// nothing has constrained taking its default. Deducing a generic
    /// argument needs a real type rather than a variable, and a literal's
    /// default is the type it would have had anyway.
    pub fn settled(&mut self, t: Ty) -> Ty {
        // an instance still waiting on its literals is made now, with their
        // defaults: something wants a real type
        if self.infer.unsettled_instance(t).is_some() {
            let waiting = self
                .pending
                .iter()
                .find(|p| p.args.is_some() && self.infer.shallow(p.ty) == self.infer.shallow(t))
                .cloned();
            if let Some(p) = waiting {
                self.force(&p);
            }
        }
        self.infer.settle(&mut self.cx.unit.types, t)
    }

    /// Makes the instance a pending variant stands for, from its arguments as
    /// they have settled, unless its context already said which it is.
    fn force(&mut self, p: &Pending) {
        let Some((name, args)) = &p.args else { return };
        if self.infer.unsettled_instance(p.ty).is_none() {
            return;
        }
        // an argument no value mentioned, and no context said
        let loose = args.iter().any(|&a| matches!(a, crate::tir::Arg::Type(t) if self.infer.unsettled_any(&self.cx.unit.types, t)));
        if loose {
            let what = p.what.clone();
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!("nothing here says what all the arguments of `{what}` are"))
                    .at(p.span)
                    .with_note(
                        "the values a variant carries say some of its enumeration's arguments, and \
                         where it is used says the rest; here nothing says them",
                    )
                    .with_help(format!("write them, as in `{what}::<…>(…)`, or give the value a type")),
            );
            let _ = self.unify(p.ty, Ty::Never);
            return;
        }
        let args: Vec<crate::tir::Arg> = args
            .iter()
            .map(|&a| match a {
                crate::tir::Arg::Type(t) => crate::tir::Arg::Type(self.infer.settle(&mut self.cx.unit.types, t)),
                c => c,
            })
            .collect();
        if let Some(id) = self.cx.instantiate_adt(*name, args, p.span) {
            let _ = self.unify(p.ty, Ty::Adt(id));
        }
    }

    /// A type with every variable replaced by what is known of it now, and
    /// the ones nothing has said left as they are.
    pub fn resolved(&mut self, t: Ty) -> Ty {
        self.infer.resolved(&mut self.cx.unit.types, t)
    }

    /// A type in words for a message (e.g. "an i32", "a structure `Point`").
    pub fn describe(&mut self, t: Ty) -> String {
        let t = self.infer.resolved(&mut self.cx.unit.types, t);
        report::describe(t, &self.cx.unit.types, self.interner)
    }

    /// A type as written for a message (e.g. `*[u8; 4]`).
    pub fn show(&mut self, t: Ty) -> String {
        let t = self.infer.resolved(&mut self.cx.unit.types, t);
        self.cx.unit.types.display(t, self.interner)
    }

    /// Makes two types the same, if they can be.
    ///
    /// # Errors
    ///
    /// [`Mismatch`] when they cannot, for the caller to report in its own
    /// terms.
    pub fn unify(&mut self, a: Ty, b: Ty) -> Result<Ty, Mismatch> {
        self.infer.unify(&self.cx.unit.types, a, b).map_err(|_| Mismatch)
    }

    /// A mismatch diagnostic.
    pub fn mismatch(&mut self, span: Span, role: &str, expected: Ty, found: Ty) -> Diagnostic {
        let (e, f) = (self.describe(expected), self.describe(found));
        report::mismatch(span, role, &e, &f)
    }

    /// Requires `found` to be `expected`, reporting a mismatch at `span`.
    /// `role` says what the value is for, such as "the value of `x`".
    /// Returns the type the two now share, or `expected` after a mismatch.
    pub fn expect(&mut self, found: Ty, expected: Ty, span: Span, role: &str) -> Ty {
        match self.unify(found, expected) {
            Ok(t) => t,
            Err(_) => {
                let d = self.mismatch(span, role, expected, found);
                self.report(d);
                expected
            }
        }
    }

    /// Delivers `e` where a value of `expected` is wanted, converting a
    /// reference to a weaker one if that is what makes it fit, and reporting
    /// a mismatch otherwise.
    pub fn coerce(&mut self, mut e: Expr, expected: Ty, role: &str) -> Expr {
        if self.unify(e.ty, expected).is_ok() {
            if e.ty != Ty::Never {
                e.ty = expected;
            }
            return e;
        }
        // a cyclotomic field lies in each field whose order its order divides
        e = match self.widen_cyclo(e, expected) {
            Ok(widened) => return widened,
            Err(e) => e,
        };
        // an integer placed on the qubits of a `quint`, which it must fit
        if self.cx.quantum
            && let Some(n) = self.quint_width(expected)
            && self.infer.shallow(e.ty).is_integer()
            && !self.is_floating(e.ty)
        {
            let span = e.span;
            return match self.bits_for(e, n, span) {
                Some(k) => self.library_call("gates", "__quint_of", vec![Arg::Const(i128::from(n))], vec![k], span),
                None => Checker::error(span),
            };
        }
        // text where a `string`, or a reference to one, is wanted, and a
        // string where a view of its characters is
        if self.is_string(expected) {
            e = match self.text_to_string(e) {
                Ok(s) => return s,
                Err(e) => e,
            };
        }
        e = match self.text_to_string_ref(e, expected) {
            Ok(r) => return r,
            Err(e) => e,
        };
        e = match self.string_view(e, expected) {
            Ok(v) => return v,
            Err(e) => e,
        };
        if self.converts(e.ty, expected) {
            let span = e.span;
            return Expr {
                kind: ExprKind::Coerce(Box::new(e)),
                ty: expected,
                span,
            };
        }
        // `*T` to `*any`, and `*T` to `*U` where `U` is a prefix of `T`: the
        // same address, carrying the same table
        let from = self.infer.shallow(e.ty);
        // a slice erased to `*any` refers to its elements as a growable array
        // of them, whose table it carries
        if let (Some((a, elem)), Some((b, Ty::Void))) = (self.cx.unit.types.as_slice(from), self.cx.unit.types.as_ref(expected))
            && a.converts_to(b)
        {
            let whole = self.types_mut().growable(elem);
            self.cx.need_table(whole, e.span);
            let span = e.span;
            return Expr {
                kind: ExprKind::Coerce(Box::new(e)),
                ty: expected,
                span,
            };
        }
        if let (Some((a, t)), Some((b, u))) = (self.cx.unit.types.as_ref(from), self.cx.unit.types.as_ref(expected))
            && a.converts_to(b)
        {
            let t = self.infer.shallow(t);
            let coerced = |e: Expr| {
                let span = e.span;
                Expr {
                    kind: ExprKind::Coerce(Box::new(e)),
                    ty: expected,
                    span,
                }
            };
            if u == Ty::Void && t != Ty::Void {
                // recovering the type later needs its table
                self.cx.need_table(t, e.span);
                return coerced(e);
            }
            if let (Ty::Adt(_), Ty::Adt(_)) = (t, u) {
                match self.cx.prefix_of(u, t) {
                    Ok(()) => return coerced(e),
                    Err(super::macros::NotPrefix::Field(f)) => {
                        let d = self.not_prefix(u, t, f, e.span);
                        self.report(d);
                        return coerced(e);
                    }
                    Err(_) => {}
                }
            }
        }
        // an array written out serves where a growable one is wanted, its
        // elements moved into a new buffer. only one written out, so a held
        // fixed array never reaches the heap unasked
        if let Some(want) = self.cx.unit.types.as_growable(expected)
            && matches!(
                e.kind,
                ExprKind::ArrayLit(_) | ExprKind::ArrayRepeat { .. } | ExprKind::Const(Value::Array(_))
            )
            && let Some((elem, _)) = self.cx.unit.types.as_array(self.infer.shallow(e.ty))
            && self.unify(elem, want).is_ok()
        {
            let span = e.span;
            return Expr {
                kind: ExprKind::Grow(Box::new(e)),
                ty: expected,
                span,
            };
        }
        // a reference to a function serves where a reference to a closure
        // of its signature is wanted
        if let (Some((_, from)), Some((Access::Write, to))) =
            (self.cx.unit.types.as_ref(self.infer.shallow(e.ty)), self.cx.unit.types.as_ref(expected))
            && matches!(from, Ty::Fn(_))
            && matches!(to, Ty::Closure(_))
            && self.cx.unit.types.as_sig(from) == self.cx.unit.types.as_sig(to)
        {
            let span = e.span;
            return Expr {
                kind: ExprKind::Ref(Box::new(Expr {
                    kind: ExprKind::FnAsClosure(Box::new(e)),
                    ty: to,
                    span,
                })),
                ty: expected,
                span,
            };
        }
        // a closure that captures nothing is a plain function: written where
        // a `*fn` of its signature is wanted, it is one
        if let ExprKind::Closure { code, captures } = &e.kind
            && captures.is_empty()
            && let Some((_, target)) = self.cx.unit.types.as_ref(expected)
            && let Some((tp, tr)) = self.cx.unit.types.as_sig(target).map(|(p, r)| (p.to_vec(), r))
            && matches!(target, Ty::Fn(_))
            && self.cx.unit.types.as_sig(e.ty).is_some_and(|(p, r)| p == tp.as_slice() && r == tr)
        {
            let (code, span) = (*code, e.span);
            return Expr {
                kind: ExprKind::Ref(Box::new(Expr {
                    kind: ExprKind::ThunkRef(code),
                    ty: target,
                    span,
                })),
                ty: expected,
                span,
            };
        }
        let d = self.mismatch(e.span, role, expected, e.ty);
        let d = self.explain_reference_mismatch(d, e.ty, expected);
        self.report(d);
        e.ty = expected;
        e
    }

    /// Whether a reference `from` converts to `to` without a check when the
    /// program runs: to one that promises less, to `*any`, or to a reference
    /// to a prefix of its referent.
    pub(super) fn converts_silently(&mut self, from: Ty, to: Ty) -> bool {
        if self.converts(from, to) {
            return true;
        }
        let types = &self.cx.unit.types;
        let (Some((a, t)), Some((b, u))) = (types.as_ref(from), types.as_ref(to)) else {
            return false;
        };
        let t = self.infer.shallow(t);
        a.converts_to(b) && (u == Ty::Void || matches!((t, u), (Ty::Adt(_), Ty::Adt(_)) if self.cx.prefix_of(u, t).is_ok()))
    }

    /// `EP01`: `*T` converted to `*U`, where `U` starts as `T` does but then
    /// departs from it at the field `f`.
    fn not_prefix(&mut self, u: Ty, t: Ty, f: Symbol, span: Span) -> Diagnostic {
        let (us, ts) = (self.show(u), self.show(t));
        let field = self.name(f);
        Diagnostic::new(Code::Ep01)
            .with_message(format!(
                "a `*{ts}` is converted to a `*{us}`, but `{us}` is not a prefix of `{ts}`: they part at the field `{field}`"
            ))
            .at(span)
            .with_note(format!(
                "a reference to one structure converts to a reference to another only when the \
                 other's fields are its first fields, with the same names and types in the same \
                 order, so that each is at the same place in both; `{field}` is where `{us}` \
                 stops matching `{ts}`"
            ))
            .with_help(format!("give `{us}` exactly the leading fields of `{ts}`, or convert with `as *{us}`, which is checked"))
    }

    /// Whether a value of `from` converts implicitly to `to` (only ever a
    /// reference to one that promises less).
    fn converts(&mut self, from: Ty, to: Ty) -> bool {
        let types = &self.cx.unit.types;
        let (from_r, from_s, to_r, to_s) = (
            types.as_ref(from),
            types.as_slice(from),
            types.as_ref(to),
            types.as_slice(to),
        );
        match (from_r, from_s, to_r, to_s) {
            // `*T` to `*const T`
            (Some((a, t)), _, Some((b, u)), _) => a.converts_to(b) && self.unify(t, u).is_ok(),
            // `*[T; N]` to `*[T]`
            (Some((a, t)), _, _, Some((b, u))) => match self.cx.unit.types.as_array(self.infer.shallow(t)) {
                Some((elem, _)) => a.converts_to(b) && self.unify(elem, u).is_ok(),
                None => false,
            },
            // `*[T]` to `*const [T]`
            (_, Some((a, t)), _, Some((b, u))) => a.converts_to(b) && self.unify(t, u).is_ok(),
            _ => false,
        }
    }

    /// Adds a hint to a mismatch between two references that differ only in
    /// what they permit.
    fn explain_reference_mismatch(&mut self, d: Diagnostic, found: Ty, expected: Ty) -> Diagnostic {
        let types = &self.cx.unit.types;
        let access = |t: Ty| types.as_ref(t).map(|r| r.0).or(types.as_slice(t).map(|s| s.0));
        match (access(found), access(expected)) {
            (Some(a), Some(b)) if !a.converts_to(b) => d.with_note(format!(
                "a reference converts only to one that promises less, `*` to `*const`, never \
                 from `{}` to `{}`",
                a.prefix().trim(),
                b.prefix().trim()
            )),
            _ => {
                // the other way round from a prefix conversion: a downcast,
                // which is written with `as` and checked
                let (Some((_, t)), Some((_, u))) = (types.as_ref(found), types.as_ref(expected)) else {
                    return d;
                };
                let t = self.infer.shallow(t);
                if !matches!((t, u), (Ty::Adt(_), Ty::Adt(_))) || self.cx.prefix_of(t, u).is_err() {
                    return d;
                }
                let (ts, us) = (self.show(t), self.show(u));
                d.with_note(format!(
                    "`{ts}` is a prefix of `{us}`, so a `*{us}` converts to a `*{ts}` implicitly; \
                     this way round the `{ts}` may not be part of a `{us}` at all"
                ))
                .with_help(format!(
                    "write `as *{us}`, which checks what the reference refers to and gives `Opt<*{us}>`"
                ))
            }
        }
    }

    /// Resolves a name in the value name space: block scopes, then unit
    /// scope, then the core library, which every unit sees without importing
    /// it.
    pub fn resolve(&mut self, name: Symbol) -> Option<Def> {
        self.scopes
            .in_blocks(name)
            .or_else(|| self.cx.lookup_value(name))
            .or_else(|| core::prelude(self.name(name)).map(Def::Intrinsic))
            .or_else(|| {
                self.cx
                    .quantum
                    .then(|| core::quantum_prelude(self.name(name)).map(Def::Intrinsic))
                    .flatten()
            })
            .or_else(|| {
                self.in_library()
                    .then(|| core::internal(self.name(name)).map(Def::Intrinsic))
                    .flatten()
            })
    }

    /// The names visible here.
    pub fn visible(&self) -> Vec<&'a str> {
        let interner = self.interner;
        let mut v: Vec<&'a str> = self
            .scopes
            .visible(&self.cx.scope)
            .map(|s| interner.resolve(s))
            .collect();
        v.extend(core::NAMES);
        v
    }

    /// Resolves a written type, reporting a problem and standing in `never`.
    pub fn resolve_type(&mut self, t: &Spanned<ast::Type>) -> Resolved {
        match types::resolve(self.cx, t) {
            Ok(r) => r,
            Err(d) => {
                self.report(*d);
                Resolved {
                    ty: Ty::Never,
                    constant: false,
                    constexpr: false,
                }
            }
        }
    }
}

/// Checks a function's body and stores it, with the function's bindings, in
/// the unit.
pub fn check_fn(cx: &mut UnitCx<'_>, id: FnId, item: &FnItem) {
    check_body(cx, id, &item.body, None);
}

/// Checks the body of function `id`. For a closure's body, `captures` are
/// the names it captured, each with whether it holds a reference, and the
/// function's first parameter is the environment holding them.
pub fn check_body(cx: &mut UnitCx<'_>, id: FnId, block: &ast::Block, captures: Option<&[(Symbol, bool)]>) {
    let (locals, params, ret, span) = {
        let f = cx.unit.func(id);
        (f.locals.clone(), f.params.clone(), f.ret, f.body.span)
    };
    let trusted = cx.unit.func(id).attrs.trusted_unitary;
    let mut c = Checker::new(cx);
    c.func = Some(id);
    c.trusted = trusted;
    c.ret = ret;
    c.locals = locals;
    c.scopes.enter();
    let mut params = params.into_iter();
    if let Some(caps) = captures {
        let env = params.next().expect("a closure's body takes its environment first");
        for (i, &(name, _)) in caps.iter().enumerate() {
            c.scopes.bind(name, Def::Capture(i as u32));
        }
        c.env = Some((env, caps.iter().map(|&(_, r)| r).collect()));
    }
    for p in params {
        let name = c.locals[p.index()].name;
        c.scopes.bind(name, Def::Local(p));
    }
    let mut body = c.block_for(block, span, Some(ret));
    // the value falls out of the closing brace, so that is where a missing
    // one is reported
    let end = Span::new(span.source, span.end.saturating_sub(1).max(span.start), span.end);
    match body.value.take() {
        Some(v) => {
            let v = c.coerce(*v, ret, "the value this function returns");
            body.ty = if v.ty == Ty::Never { Ty::Never } else { ret };
            body.value = Some(Box::new(v));
        }
        None => match c.unify(body.ty, ret) {
            Ok(_) => {}
            Err(_) if c.shallow(body.ty) == Ty::Void => {
                let what = c.describe(ret);
                let d = missing_value(end, c.name(c.cx.unit.func(id).name), &what);
                c.report(d);
            }
            Err(_) => {
                let found = body.ty;
                let d = c.mismatch(end, "the value this function returns", ret, found);
                c.report(d);
            }
        },
    }
    c.scopes.leave();
    c.report_unsettled();

    let Checker {
        locals,
        infer: table,
        persist,
        ..
    } = c;
    let unit = &mut cx.unit;
    let f = &mut unit.fns[id.index()];
    f.body = body;
    f.locals = locals;
    infer::settle_fn(f, &table, &mut unit.types, &mut cx.diags);
    for g in persist {
        if let Some(e) = &mut unit.globals[g.index()].init {
            infer::settle_expr(e, &table, &mut unit.types, &mut cx.diags);
        }
        let ty = unit.globals[g.index()].ty;
        unit.globals[g.index()].ty = table.settle(&mut unit.types, ty);
    }
    let f = &cx.unit.fns[id.index()];
    exhaust::check_block(&f.body, &cx.unit.types, cx.interner, &mut cx.diags);
}

/// Checks a unit-scope initialiser against its binding's declared type.
pub fn check_global_init(cx: &mut UnitCx<'_>, id: GlobalId, init: &Spanned<ast::Expr>) -> Expr {
    let declared = cx.unit.global(id).ty;
    let name = cx.interner.resolve(cx.unit.global(id).name);
    check_const(cx, init, declared, &format!("the value of `{name}`"))
}

/// Checks an expression at unit scope (an initialiser, or a constant such as
/// an array length) as a value of `expected`.
pub fn check_const(cx: &mut UnitCx<'_>, e: &Spanned<ast::Expr>, expected: Ty, role: &str) -> Expr {
    check_const_as(cx, e, Some(expected), role)
}

/// Checks an expression at unit scope as a value of `expected`, or of
/// whatever type it has when `expected` is `None`.
pub fn check_const_as(cx: &mut UnitCx<'_>, e: &Spanned<ast::Expr>, expected: Option<Ty>, role: &str) -> Expr {
    let mut c = Checker::new(cx);
    c.scopes.enter();
    let mut checked = match expected {
        Some(t) => {
            let checked = c.expr_for(e, t);
            c.coerce(checked, t, role)
        }
        None => c.expr(e),
    };
    c.scopes.leave();
    c.report_unsettled();
    let table = c.infer;
    let unit = &mut cx.unit;
    infer::settle_expr(&mut checked, &table, &mut unit.types, &mut cx.diags);
    exhaust::check_expr(&checked, &cx.unit.types, cx.interner, &mut cx.diags);
    checked
}

/// A function that should return a value whose body ends without one.
fn missing_value(span: Span, name: &str, ret: &str) -> Diagnostic {
    Diagnostic::new(Code::Es06)
        .with_message(format!(
            "`{name}` should return {ret}, but its body ends without a value"
        ))
        .at(span)
        .with_note(
            "a block's value is its last expression, written without a semicolon; a \
             semicolon after it turns it into a statement and discards the value",
        )
        .with_help(
            "end the body with the value to return, without a `;`, or use `return value;`",
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sema::testing::{check, codes};
    use crate::span::SourceId;

    fn cx(i: &Interner) -> UnitCx<'_> {
        UnitCx::new("t", SourceId(0), i, &[])
    }

    #[test]
    fn a_stand_in_for_an_error_unifies_with_anything() {
        let e = Checker::error(Span::synthetic());
        assert_eq!(e.ty, Ty::Never);
        let i = Interner::new();
        let mut cx = cx(&i);
        let mut c = Checker::new(&mut cx);
        assert_eq!(c.unify(e.ty, Ty::Bool), Ok(Ty::Bool));
    }

    #[test]
    fn locals_and_loops_are_numbered_in_order() {
        let mut i = Interner::new();
        let x = i.intern("x");
        let mut cx = cx(&i);
        let mut c = Checker::new(&mut cx);
        let a = c.new_local(x, Ty::Bool, false, Span::synthetic());
        let b = c.new_local(x, Ty::Bool, false, Span::synthetic());
        assert_eq!((a, b), (LocalId(0), LocalId(1)));
        assert_eq!(c.fresh_loop(), LoopId(0));
        assert_eq!(c.fresh_loop(), LoopId(1));
    }

    #[test]
    fn a_failed_expectation_is_reported_once_and_recovers_to_the_expected_type() {
        let i = Interner::new();
        let mut cx = cx(&i);
        let mut c = Checker::new(&mut cx);
        let t = c.expect(Ty::Bool, Ty::Int(crate::tir::IntTy::I32), Span::synthetic(), "the value");
        assert_eq!(t, Ty::Int(crate::tir::IntTy::I32));
        assert_eq!(cx.diags.len(), 1);
        assert_eq!(cx.diags[0].code, Code::Es06);
    }

    #[test]
    fn a_reference_converts_to_one_that_promises_less() {
        check("fn f(p: *i32) -> *i32 { p }");
        check("fn f(p: *i32) -> *const i32 { p }");
        check("fn f(a: *[u8; 4]) -> *[u8] { a }");
        check("fn f(a: *[u8]) -> *const [u8] { a }");
    }

    #[test]
    fn a_reference_never_converts_to_one_that_promises_more() {
        assert_eq!(codes("fn f(p: *const i32) -> *i32 { p }"), [Code::Es06]);
        assert_eq!(codes("fn f(a: *const [u8]) -> *[u8] { a }"), [Code::Es06]);
        assert_eq!(codes("fn f(a: *[u8]) -> *[u8; 4] { a }"), [Code::Es06]);
    }
}
