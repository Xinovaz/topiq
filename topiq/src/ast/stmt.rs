//! Statements and blocks.
//!
//! # A block's value is its trailing expression
//!
//! A block is a sequence of statements that may end in an expression, and that
//! expression is the block's value. It is held in [`Block::value`] rather than
//! as a final statement, which makes the difference between `{ x }` and
//! `{ x; }` structural rather than a matter of inspection.
//!
//! # Storage specifiers
//!
//! A `let` may carry one of two specifiers, each giving the binding a storage
//! duration other than the ordinary automatic one:
//!
//! - `persist`: the object is initialised once at startup from a constant
//!   expression, keeps its value between calls of the enclosing function, and
//!   is **not destroyed when the program ends**.
//! - `aux`: an ancilla, allowed only at block scope in a quantum unit. It is
//!   allocated in |0⟩ and *automatically uncomputed* when the scope exits, by
//!   synthesising the adjoint of everything that touched it. This is why
//!   borrowing an ancilla needs no matching cleanup code.
//!
//! The two are mutually exclusive, and `persist` may not be applied to a
//! qubit-bearing type: quantum state cannot outlive the run that created it.

use crate::span::Spanned;

use super::annot::AnnotationGroup;
use super::expr::Expr;
use super::item::Item;
use super::pat::Pattern;
use super::ty::Type;

/// A storage specifier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Storage {
    /// `persist`: one object for the whole program, initialised at startup.
    Persist,
    /// `aux`: an ancilla, uncomputed automatically at scope exit.
    Aux,
}

impl Storage {
    /// The spelling.
    pub fn text(self) -> &'static str {
        match self {
            Storage::Persist => "persist",
            Storage::Aux => "aux",
        }
    }
}

/// A `let` binding.
#[derive(Clone, PartialEq, Debug)]
pub struct LetStmt {
    /// The storage specifier.
    pub storage: Option<Spanned<Storage>>,
    /// What the value is bound to: a name, or a pattern that takes it
    /// apart, as in `let (q, r) = divide(a, b);`.
    pub binding: Spanned<Pattern>,
    /// The declared type.
    pub ty: Option<Spanned<Type>>,
    /// The initialiser.
    pub init: Option<Spanned<Expr>>,
}

/// A jump statement.
///
/// Each of these runs the destructions of every scope it exits, including
/// uncomputing any ancilla borrowed in those scopes. A jump is never a way to
/// skip cleanup.
#[derive(Clone, PartialEq, Debug)]
pub enum Flow {
    /// `return e?;`
    Return(Option<Spanned<Expr>>),
    /// `break e?;`: a `loop` may be broken out of with a value.
    Break(Option<Spanned<Expr>>),
    /// `continue;`
    Continue,
}

/// A statement.
#[derive(Clone, PartialEq, Debug)]
pub enum Stmt {
    /// A `let` binding.
    Let(Box<LetStmt>),
    /// An item declared inside a block.
    Item(Box<Item>),
    /// `return`, `break` or `continue`.
    Flow(Flow),
    /// `forget e;`: deliberately discards a qubit-bearing value, in a way the
    /// judgement system records.
    ///
    /// There is no implicit forgetting: dropping a quantum value could corrupt
    /// whatever it was entangled with, so a program writes it down.
    Forget(Spanned<Expr>),
    /// A block-expression statement: `if`, `match`, `loop`, `while`, `for` or a
    /// bare block written where a statement is expected.
    ///
    /// Its closing `}` ends the statement, so `{ … } * x;`
    /// is two statements rather than one multiplication.
    BlockExpr(Spanned<Expr>),
    /// An expression followed by a semicolon.
    Expr(Spanned<Expr>),
    /// A stray `;`, which is a statement that does nothing.
    Empty,
}

/// A block.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Block {
    /// Annotations written on the block, if any.
    pub annotations: Vec<Spanned<AnnotationGroup>>,
    /// The statements.
    pub stmts: Vec<Spanned<Stmt>>,
    /// The trailing expression.
    pub value: Option<Spanned<Expr>>,
}

impl Block {
    /// An empty block.
    pub fn empty() -> Block {
        Block::default()
    }

    /// Whether the block has neither statements nor a value.
    pub fn is_empty(&self) -> bool {
        self.stmts.is_empty() && self.value.is_none()
    }

    /// Whether the block evaluates to a value.
    pub fn has_value(&self) -> bool {
        self.value.is_some()
    }
}

/// A `for` binding's pattern together with what it iterates.
///
/// Kept as a named type because `for` has two quite different
/// meanings in a quantum unit: over a const-evaluable range it is **unrolled at
/// circuit generation** and the index is a generation-time constant; over a
/// range not known at generation time it is permitted only over purely
/// classical computation, unless a bound obtained by `lift` makes it
/// generation-time on a target with `dynamic_lifting`.
#[derive(Clone, PartialEq, Debug)]
pub struct ForHeader {
    /// The binding.
    pub pattern: Spanned<Pattern>,
    /// The iterable.
    pub iter: Spanned<Expr>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::span::{SourceId, Span};

    fn sp<T>(node: T) -> Spanned<T> {
        Spanned::new(node, Span::new(SourceId(0), 0, 1))
    }

    #[test]
    fn an_empty_block_has_no_statements_and_no_value() {
        let b = Block::empty();
        assert!(b.is_empty());
        assert!(!b.has_value());
    }

    #[test]
    fn a_trailing_expression_is_the_blocks_value() {
        // `{ x }` and `{ x; }` differ structurally, not by a flag on the
        // last statement
        let with_value = Block {
            annotations: Vec::new(),
            stmts: Vec::new(),
            value: Some(sp(Expr::Bool(true))),
        };
        let without = Block {
            annotations: Vec::new(),
            stmts: vec![sp(Stmt::Expr(sp(Expr::Bool(true))))],
            value: None,
        };
        assert!(with_value.has_value());
        assert!(!without.has_value());
        assert!(!with_value.is_empty());
        assert_ne!(with_value, without);
    }

    #[test]
    fn there_are_exactly_two_storage_specifiers() {
        assert_eq!(Storage::Persist.text(), "persist");
        assert_eq!(Storage::Aux.text(), "aux");
        assert_ne!(Storage::Persist, Storage::Aux);
    }

    #[test]
    fn an_ancilla_binding_records_its_specifier() {
        // as in `aux let a: qubit;`
        let mut i = Interner::new();
        let l = LetStmt {
            storage: Some(sp(Storage::Aux)),
            binding: sp(Pattern::Binding(sp(i.intern("a")))),
            ty: None,
            init: None,
        };
        assert_eq!(l.storage.map(|s| s.node), Some(Storage::Aux));
        assert!(
            l.init.is_none(),
            "an ancilla is allocated in |0>, not initialised by an expression"
        );
    }

    #[test]
    fn a_persist_binding_records_its_specifier() {
        // as in `persist let counter: u64 = 0;`
        let mut i = Interner::new();
        let l = LetStmt {
            storage: Some(sp(Storage::Persist)),
            binding: sp(Pattern::Binding(sp(i.intern("counter")))),
            ty: None,
            init: Some(sp(Expr::Bool(false))),
        };
        assert_eq!(l.storage.map(|s| s.node), Some(Storage::Persist));
    }

    #[test]
    fn break_may_carry_a_value_but_continue_may_not() {
        // a `loop` may be broken out of with a value
        assert_eq!(Flow::Break(Some(sp(Expr::Bool(true)))).clone(), Flow::Break(Some(sp(Expr::Bool(true)))));
        assert_eq!(Flow::Continue, Flow::Continue);
        assert_ne!(Flow::Break(None), Flow::Continue);
    }

    #[test]
    fn forget_is_a_statement_of_its_own() {
        // `forget` is a statement rather than an expression, and it is the
        // only way a quantum value is ever discarded
        let s = Stmt::Forget(sp(Expr::Bool(true)));
        assert!(matches!(s, Stmt::Forget(_)));
    }

    #[test]
    fn a_block_expression_statement_is_distinct_from_an_expression_statement() {
        // the closing brace ends the statement, so these are not the
        // same node and the parser must not conflate them
        let b = Stmt::BlockExpr(sp(Expr::Block(Box::new(Block::empty()))));
        let e = Stmt::Expr(sp(Expr::Block(Box::new(Block::empty()))));
        assert_ne!(b, e);
    }
}
