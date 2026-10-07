//! Checking statements and blocks.
//!
//! # A block's type
//!
//! A block's value is its last expression, if it ends in one. Otherwise its
//! type depends on whether control can reach its end: a block whose statements
//! include a `return`, `break` or `continue`, or a call of `exit` or `panic`,
//! never finishes normally, so its type is `never`, which is what lets
//!
//! ```text
//! let x = if ready { 5 } else { return 0; };
//! ```
//!
//! type-check: the `else` block has no value, but it has no need of one.
//! A block that does reach its end without a value has type `void`.
//!
//! # A `let` sees the scope before it
//!
//! A new binding is visible only after its whole statement, so in
//! `let n = n + 1;` the `n` on the right is the earlier one. Reading a binding
//! inside its own initialiser could never be meaningful (it has no value yet),
//! so this is the only useful reading.
//!
//! # A `let` without a value
//!
//! `let x: i32;` declares a binding and gives it a value later, by
//! assignment. Every path to a read of `x` must assign it first; that is
//! checked once the whole function is known, by following each path through
//! it. The type must be written, since there is no value to take it from.

use crate::ast::{self, Flow, Storage};
use crate::diag::{Code, Diagnostic};
use crate::span::{Span, Spanned};
use crate::tir::{Block, Expr, ExprKind, GlobalId, GlobalKind, Pat, PatKind, Stmt, Ty, Value};

use super::body::{Checker, LoopKind};
use super::refutable;
use super::items::{self, Target};
use super::scope::Def;

impl Checker<'_, '_> {
    /// Checks a block in a new scope. `span` is where the block was written,
    /// or the construct it belongs to when the block has no span of its own.
    pub fn block(&mut self, b: &ast::Block, span: Span) -> Block {
        self.block_for(b, span, None)
    }

    /// Checks a block whose value is wanted as a `want`, which its last
    /// expression is checked for, as a `let` with a type checks its value.
    pub fn block_for(&mut self, b: &ast::Block, span: Span, want: Option<Ty>) -> Block {
        items::check_annotations(&b.annotations, Target::Block, self.cx.quantum, self.interner, &mut self.cx.diags);
        // the items the block declares are in scope throughout it
        let layer = self.cx.block_layers.get(&super::local_items::key(b)).cloned();
        let has_layer = layer.is_some();
        if let Some(layer) = layer {
            self.cx.local_items.push(layer);
        }
        self.scopes.enter();
        let mut stmts = Vec::new();
        let mut diverges = false;
        for s in &b.stmts {
            diverges |= self.stmt(s, &mut stmts);
        }
        let value = b.value.as_ref().map(|v| {
            Box::new(match want {
                Some(t) => self.expr_for(v, t),
                None => self.expr(v),
            })
        });
        self.scopes.leave();
        if has_layer {
            self.cx.local_items.pop();
        }
        let ty = match &value {
            Some(v) => v.ty,
            None if diverges => Ty::Never,
            None => Ty::Void,
        };
        Block {
            stmts,
            value,
            ty,
            span,
        }
    }

    /// Checks one statement, appending what it becomes. Returns whether it
    /// diverges (whether control can never pass beyond it).
    fn stmt(&mut self, s: &Spanned<ast::Stmt>, out: &mut Vec<Stmt>) -> bool {
        match &s.node {
            ast::Stmt::Let(l) => self.let_stmt(l, s.span, out),
            // declared with the unit's items, and in scope through the block
            ast::Stmt::Item(item) => {
                if item.linkage == ast::Linkage::Unit {
                    self.report(items::static_without_linkage(
                        s.span,
                        "a declaration inside a block",
                    ));
                }
                false
            }
            ast::Stmt::Flow(flow) => {
                let e = self.flow(flow, s.span);
                out.push(Stmt::Expr(e));
                true
            }
            ast::Stmt::Forget(operand) => {
                if let Some(e) = self.forget(operand, s.span) {
                    out.push(Stmt::Expr(e));
                }
                false
            }
            ast::Stmt::BlockExpr(e) | ast::Stmt::Expr(e) => {
                let e = self.expr(e);
                let diverges = self.shallow(e.ty) == Ty::Never;
                out.push(Stmt::Expr(e));
                diverges
            }
            ast::Stmt::Empty => false,
        }
    }

    fn let_stmt(&mut self, l: &ast::LetStmt, span: Span, out: &mut Vec<Stmt>) -> bool {
        // only a plain name takes the ordinary path; anything else is a
        // pattern that takes the value apart
        if !matches!(l.binding.node, ast::Pattern::Binding(_)) {
            return self.destructuring_let(l, span, out);
        }
        match l.storage.map(|s| s.node) {
            Some(Storage::Aux) if self.cx.quantum => return self.aux_let(l, span, out),
            Some(Storage::Aux) => {
                self.report(
                    Diagnostic::new(Code::Eu02)
                        .with_message("an ancilla (`aux let`) can only exist in a quantum unit")
                        .at(span)
                        .with_note(
                            "an ancilla is a borrowed qubit, and a classical unit may not \
                             hold quantum state",
                        ),
                );
                return false;
            }
            Some(Storage::Persist) => return self.persist_let(l, span),
            None => {}
        }
        let declared = l.ty.as_ref().map(|t| {
            let r = self.resolve_type(t);
            if r.ty == Ty::Void {
                self.report(items::void_binding(t.span));
            }
            r
        });
        let ast::Pattern::Binding(bound) = l.binding.node else {
            unreachable!("a pattern that is not a name took the other path");
        };
        let name = self.name(bound.node);
        let constant = declared.is_some_and(|d| d.constant);
        let Some(init) = &l.init else {
            let ty = match declared {
                Some(d) => {
                    if d.constant {
                        self.report(
                            Diagnostic::new(Code::Ec01)
                                .with_message(format!("`{name}` is declared `const`, so it needs its value here"))
                                .at(span)
                                .with_note("a `const` binding is never assigned, so its value is the one it is declared with"),
                        );
                    }
                    d.ty
                }
                None => {
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("`{name}` has neither a type nor a value, so its type is unknown"))
                            .at(span)
                            .with_help(format!("write its type, as in `let {name}: i32;`, or give it a value")),
                    );
                    Ty::Never
                }
            };
            let local = self.new_local(bound.node, ty, constant, bound.span);
            self.scopes.bind(bound.node, Def::Local(local));
            out.push(Stmt::Let { local, init: None });
            return false;
        };
        // the whole value of a new constant, when it is indexing, only reads
        // the element, so a type's `$index_rd` serves it
        self.read_index = constant && matches!(init.node, ast::Expr::Index { .. });
        let init = match declared {
            Some(d) => self.expr_for(init, d.ty),
            None => self.expr_inferred(init),
        };
        self.read_index = false;
        let diverges = self.shallow(init.ty) == Ty::Never;
        let role = format!("the value of `{name}`");
        let (init, ty) = match declared {
            Some(d) => {
                let e = self.coerce(init, d.ty, &role);
                (e, d.ty)
            }
            None => {
                if self.shallow(init.ty) == Ty::Void {
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!(
                                "this produces no value, so there is nothing to bind to `{name}`"
                            ))
                            .at(init.span)
                            .with_note("an expression of type void, such as a call to a function with no return type, has no value"),
                    );
                    let t = Ty::Never;
                    (init, t)
                } else {
                    let t = init.ty;
                    (init, t)
                }
            }
        };
        let local = self.new_local(bound.node, ty, constant, bound.span);
        self.locals[local.index()].constexpr = declared.is_some_and(|d| d.constexpr);
        self.scopes.bind(bound.node, Def::Local(local));
        out.push(Stmt::Let {
            local,
            init: Some(init),
        });
        diverges
    }

    /// `aux let t: T;`: an ancilla, allocated in |0>. Its scope's end
    /// returns it to |0> by undoing what was done to it, so it is written
    /// with a quantum type, and with a value only when that is a condition
    /// computed from other qubits or a constant, flipped onto it from |0>:
    /// `aux let t = a & b;` is `aux let t: qubit; t ^= a & b;`.
    fn aux_let(&mut self, l: &ast::LetStmt, span: Span, out: &mut Vec<Stmt>) -> bool {
        let ast::Pattern::Binding(bound) = l.binding.node else {
            unreachable!("a pattern that is not a name took the other path");
        };
        let name = self.name(bound.node);

        // the value first, which the ancilla's name does not reach
        let value = l.init.as_ref().map(|init| (self.expr(init), init.span));
        let ty = match (&l.ty, &value) {
            (Some(t), _) => {
                let ty = self.resolve_type(t).ty;
                if ty != Ty::Never && !self.types().is_quantum(ty) {
                    let shown = self.types().display(ty, self.interner);
                    self.report(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!("an ancilla holds qubits, but `{name}` is declared `{shown}`"))
                            .at(t.span)
                            .with_note("`aux` marks quantum storage that is returned to |0> when its scope ends")
                            .with_help(format!("declare `{name}` with plain `let`")),
                    );
                }
                ty
            }
            (None, Some((v, _))) if super::qbits::is_condition_node(v) => Ty::Qubit,
            (None, Some((v, at))) => {
                if v.ty != Ty::Never {
                    let d = Self::ancilla_value(name, *at);
                    self.report(d);
                }
                return false;
            }
            (None, None) => {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message(format!("the ancilla `{name}` needs its type written"))
                        .at(span)
                        .with_note("an ancilla is allocated by its declaration, so its type says how many qubits to allocate")
                        .with_help(format!("write its type, as in `aux let {name}: qubit;`")),
                );
                return false;
            }
        };
        // a value computed from other qubits, flipped onto the ancilla
        // from |0>, so that undoing it is undoing that computation
        let computed = value.map(|(v, at)| {
            let fits = (self.shallow(ty) == Ty::Qubit && super::qbits::is_condition_node(&v))
                || (self.types().is_quantum(ty) && matches!(v.kind, ExprKind::Const(Value::Int(..))));
            (v, fits, at)
        });
        if let Some((v, false, at)) = &computed
            && v.ty != Ty::Never
        {
            let d = Self::ancilla_value(name, *at);
            self.report(d);
        }

        // a local marked as an ancilla, declared without a value
        let local = self.new_local(bound.node, ty, false, bound.span);
        self.locals[local.index()].aux = true;
        self.scopes.bind(bound.node, Def::Local(local));
        out.push(Stmt::Let { local, init: None });
        if let Some((v, true, _)) = computed {
            let place = Expr::local(local, ty, bound.span);
            let flip = self.quantum_assign(crate::tir::BinOp::BitXor, place, v, span);
            out.push(Stmt::Expr(flip));
        }
        false
    }

    /// An ancilla given a value it cannot take.
    fn ancilla_value(name: &str, at: Span) -> Diagnostic {
        Diagnostic::new(Code::Es06)
            .with_message(format!("the ancilla `{name}` begins in |0>, so it takes no value"))
            .at(at)
            .with_note(
                "its end of scope undoes everything done to it, which is only possible from a state \
                 the compiler knows; it can be given a condition computed from other qubits, such as \
                 `a & b`, or a constant, which are flipped onto it",
            )
            .with_help(format!("write `aux let {name}: …;` and act on it afterwards"))
    }

    /// `let (a, b) = e;` and the other forms that take a value apart.
    ///
    /// The pattern must match every value of the type, since there is no
    /// second arm to fall to: a name, a tuple, a structure, or the only
    /// variant of an enumeration that has one. A pattern that could fail is
    /// refused, naming `match` as the way to handle both cases.
    fn destructuring_let(&mut self, l: &ast::LetStmt, span: Span, out: &mut Vec<Stmt>) -> bool {
        if let Some(st) = l.storage {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message(format!(
                        "`{} let` names one object, so it cannot take a value apart",
                        st.node.text()
                    ))
                    .at(span),
            );
            return false;
        }
        let declared = l.ty.as_ref().map(|t| self.resolve_type(t));
        let Some(init) = &l.init else {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("this `let` takes a value apart, so it needs a value")
                    .at(span)
                    .with_help("write `= …` with the value to take apart"),
            );
            return false;
        };
        let init = match declared {
            Some(d) => self.expr_for(init, d.ty),
            None => self.expr(init),
        };
        let diverges = self.shallow(init.ty) == Ty::Never;
        let init = match declared {
            Some(d) => self.coerce(init, d.ty, "the value taken apart"),
            None => init,
        };
        let mut bound = Vec::new();
        let pat = self.pattern(&l.binding, init.ty, &mut bound);
        // everything the pattern binds is `const` when the value is
        if let Some(d) = declared
            && d.constant
        {
            if d.constexpr {
                self.report(
                    Diagnostic::new(Code::Es06)
                        .with_message("a value taken apart cannot be `constexpr`")
                        .at(l.ty.as_ref().map_or(span, |t| t.span))
                        .with_note("only a single name is folded into its uses during translation")
                        .with_help("write `const`, or bind each part with its own `constexpr` binding"),
                );
            }
            self.make_bindings_constant(&pat);
        }
        if !refutable::is_irrefutable(&pat, self.types()) {
            self.report(
                Diagnostic::new(Code::Es13)
                    .with_message("this pattern does not match every value, so `let` cannot use it")
                    .at(l.binding.span)
                    .with_note("a `let` has no second arm to fall to when the pattern does not match")
                    .with_help("use `match`, which names what to do in each case"),
            );
            return diverges;
        }
        out.push(Stmt::LetPat { pat, init });
        diverges
    }

    /// Marks every binding a pattern introduces as never assigned, which is
    /// what `let (a, b): const (A, B) = …` asks for.
    fn make_bindings_constant(&mut self, p: &Pat) {
        match &p.kind {
            PatKind::Bind(l) | PatKind::BindRef(l) => self.locals[l.index()].constant = true,
            PatKind::Struct { fields } | PatKind::Variant { fields, .. } => {
                for (_, f) in fields {
                    self.make_bindings_constant(f);
                }
            }
            PatKind::Wild | PatKind::Const(_) => {}
        }
    }


    /// `persist let`: one object for the whole program, initialised from a
    /// constant before the program starts, visible only in this block.
    fn persist_let(&mut self, l: &ast::LetStmt, span: Span) -> bool {
        let Some(func) = self.func else {
            return false;
        };
        // a persistent object is one object with one name, so there is
        // nothing for a pattern to take apart
        let ast::Pattern::Binding(bound) = l.binding.node else {
            self.report(
                Diagnostic::new(Code::Es06)
                    .with_message("a `persist let` names one object, so it cannot take a value apart")
                    .at(l.binding.span)
                    .with_help("bind the whole value to a name, and take it apart afterwards"),
            );
            return false;
        };
        let Some(init) = &l.init else {
            self.report(
                Diagnostic::new(Code::Ec01)
                    .with_message(format!(
                        "`persist let {}` needs an initial value",
                        self.name(bound.node)
                    ))
                    .at(span)
                    .with_note(
                        "a persistent object is created once, before the program starts, \
                         from a value known during translation",
                    ),
            );
            return false;
        };
        let declared = l.ty.as_ref().map(|t| self.resolve_type(t));
        let init = match declared {
            Some(d) => self.expr_for(init, d.ty),
            None => self.expr(init),
        };
        let name = self.name(bound.node);
        let (init, ty) = match declared {
            Some(d) => (self.coerce(init, d.ty, &format!("the value of `{name}`")), d.ty),
            None => {
                let t = init.ty;
                (init, t)
            }
        };
        let id = GlobalId(self.cx.unit.globals.len() as u32);
        self.cx.unit.globals.push(crate::tir::Global {
            name: bound.node,
            ty,
            constant: declared.is_some_and(|d| d.constant),
            linkage: ast::Linkage::Unit,
            kind: GlobalKind::Persist(func),
            init: Some(init),
            value: None,
            startup: false,
            span: bound.span,
            symbol: None,
            deprecated: None,
        });
        self.persist.push(id);
        self.scopes.bind(bound.node, Def::Global(id));
        false
    }

    /// `return`, `break` or `continue`.
    fn flow(&mut self, flow: &Flow, span: Span) -> Expr {
        let kind = match flow {
            Flow::Return(value) => {
                let ret = self.ret;
                let value = value.as_ref().map(|v| self.expr_for(v, ret));
                let value = match value {
                    Some(v) => {
                        let ret = self.ret;
                        Some(self.coerce(v, ret, "the value returned"))
                    }
                    None => {
                        if !matches!(self.shallow(self.ret), Ty::Void | Ty::Never) {
                            let what = self.describe(self.ret);
                            let d = Diagnostic::new(Code::Es06)
                                .with_message(format!("this function returns {what}, so `return` needs a value"))
                                .at(span)
                                .with_help("write `return <value>;`");
                            self.report(d);
                        }
                        None
                    }
                };
                ExprKind::Return(value.map(Box::new))
            }
            Flow::Break(value) => {
                let value = value.as_ref().map(|v| self.expr(v));
                let Some(frame) = self.loops.last().copied() else {
                    self.report(outside_loop(span, "break"));
                    return Checker::error(span);
                };
                match (&value, frame.kind) {
                    (Some(v), LoopKind::Loop) => {
                        let t = match frame.value {
                            Some(prev) => self.expect(v.ty, prev, v.span, "the value of this `break`"),
                            None => v.ty,
                        };
                        self.loops.last_mut().expect("checked above").value = Some(t);
                    }
                    (Some(v), _) => {
                        self.report(
                            Diagnostic::new(Code::Es09)
                                .with_message("`break` can only carry a value out of a `loop`")
                                .at(v.span)
                                .with_note(
                                    "a `while` or `for` loop can end without any `break`, so \
                                     there would be no value to give; only `loop` ends solely \
                                     through `break`",
                                )
                                .with_help("remove the value, or rewrite the loop as `loop`"),
                        );
                    }
                    (None, LoopKind::Loop) => {
                        let t = match frame.value {
                            Some(prev) => self.expect(Ty::Void, prev, span, "the value of this `break`"),
                            None => Ty::Void,
                        };
                        self.loops.last_mut().expect("checked above").value = Some(t);
                    }
                    (None, _) => {}
                }
                ExprKind::Break {
                    target: frame.id,
                    value: value.map(Box::new),
                }
            }
            Flow::Continue => {
                let Some(frame) = self.loops.last().copied() else {
                    self.report(outside_loop(span, "continue"));
                    return Checker::error(span);
                };
                ExprKind::Continue { target: frame.id }
            }
        };
        Expr {
            kind,
            ty: Ty::Never,
            span,
        }
    }
}

/// `break` or `continue` with no loop around it.
fn outside_loop(span: Span, word: &str) -> Diagnostic {
    Diagnostic::new(Code::Es09)
        .with_message(format!("`{word}` is only meaningful inside a loop, and there is none here"))
        .at(span)
        .with_note("a function body is not a loop: leave a function with `return`")
}

#[cfg(test)]
mod tests {
    use crate::diag::Code;
    use crate::sema::testing::{check, codes};
    use crate::tir::Stmt;

    #[test]
    fn a_block_ending_in_a_value_has_that_value() {
        check("fn f() -> i32 { let a = 1; a }");
    }

    #[test]
    fn a_block_that_always_returns_needs_no_value() {
        check("fn f(c: bool) -> i32 { let x = if c { 5 } else { return 0; }; x }");
    }

    #[test]
    fn a_trailing_semicolon_loses_the_value() {
        assert_eq!(codes("fn f() -> i32 { 5; }"), [Code::Es06]);
    }

    #[test]
    fn a_let_initializer_sees_the_earlier_binding() {
        check("fn f() -> i32 { let n = 1; let n = n + 1; n }");
    }

    #[test]
    fn a_let_is_not_visible_before_its_statement() {
        assert_eq!(codes("fn f() -> i32 { let a = b; let b = 1; a }"), [Code::Es04]);
    }

    #[test]
    fn a_binding_leaves_scope_with_its_block() {
        assert_eq!(codes("fn f() -> i32 { { let a = 1; } a }"), [Code::Es04]);
    }

    #[test]
    fn break_and_continue_need_a_loop() {
        assert_eq!(codes("fn f() { break; }"), [Code::Es09]);
        assert_eq!(codes("fn f() { continue; }"), [Code::Es09]);
    }

    #[test]
    fn only_loop_can_break_with_a_value() {
        check("fn f() -> i32 { loop { break 3; } }");
        assert_eq!(codes("fn f() { while true { break 3; } }"), [Code::Es09]);
    }

    #[test]
    fn return_needs_a_value_in_a_function_that_returns_one() {
        assert_eq!(codes("fn f() -> i32 { return; }"), [Code::Es06]);
        check("fn f() { return; }");
    }

    #[test]
    fn an_ancilla_needs_a_quantum_unit() {
        assert_eq!(codes("fn f() { aux let a: i32 = 0; }"), [Code::Eu02]);
    }

    #[test]
    fn a_persistent_binding_becomes_an_object() {
        let u = check("fn next() -> u64 { persist let n: u64 = 0; n += 1; n }");
        assert_eq!(u.globals.len(), 1);
        assert!(!u.globals[0].constant);
    }

    #[test]
    fn a_local_may_be_declared_without_a_value() {
        let u = check("fn f(c: bool) -> i32 { let x: i32; if c { x = 1; } else { x = 2; } x }");
        assert!(matches!(u.fns[0].body.stmts[0], Stmt::Let { init: None, .. }));
    }

    #[test]
    fn a_local_without_a_value_needs_a_type() {
        assert_eq!(codes("fn f() { let x; }"), [Code::Es06]);
    }

    #[test]
    fn a_static_declaration_inside_a_block_restricts_nothing() {
        assert_eq!(codes("fn f() { static fn g() { } }"), [Code::El01]);
    }

    #[test]
    fn binding_the_result_of_a_void_call_is_refused() {
        assert_eq!(codes("fn g() { }\nfn f() { let x = g(); }"), [Code::Es06]);
    }
}
