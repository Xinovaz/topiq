//! Expression syntax.
//!
//! # Operators are method calls
//!
//! An operator applied to non-primitive operands becomes a call to a method
//! with a `$`-prefixed name: `a + b` is `a.$add(b)`. The method is found by
//! introspection at the point of use, so `a + b` is ill-formed unless a
//! `$add` with a compatible signature actually exists on `a`'s type.
//!
//! There is no trait to implement: the name and the signature are the
//! contract, checked structurally, so adding an operator to a type is
//! defining one method.
//!
//! [`BinOp::method`] and [`UnOp::method`] give the name each operator
//! desugars to. `&&` and `||` are defined on `bool` alone and cannot be
//! overloaded: a method call cannot leave its argument unevaluated.
//!
//! # Evaluation order is fixed
//!
//! Nothing about evaluation order is left open, so no program has to be written
//! defensively against a compiler's choices:
//!
//! - the operands of a binary operator evaluate **left to right**;
//! - `&&` and `||` evaluate their right operand only when the left does not
//!   already decide the result;
//! - the operands of an assignment evaluate **right to left** (the value
//!   first, then the place).
//!
//! The last is recorded on [`Expr::Assign`] as well. Evaluating the value first
//! means a place expression that aborts cannot mask an abort in the value it
//! was going to be assigned.

use crate::intern::Symbol;
use crate::lex::{FloatSuffix, IntBase, IntSuffix, StrKind};
use crate::span::Spanned;

use super::pat::Pattern;
use super::stmt::Block;
use super::ty::{Path, TArg, Type};

/// A unary operator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnOp {
    /// `-e`
    Neg,
    /// `!e`
    Not,
    /// `&e`, which yields `*T` from a place, or `*const T` from a constant.
    Ref,
    /// `*e`, the place a reference refers to.
    Deref,
}

impl UnOp {
    /// The method this operator desugars to.
    ///
    /// `&` and `*` move between a place and a reference to it rather
    /// than calling a method, so they have none.
    pub fn method(self) -> Option<&'static str> {
        match self {
            UnOp::Neg => Some("$neg"),
            UnOp::Not => Some("$not"),
            UnOp::Ref | UnOp::Deref => None,
        }
    }
}

/// The direction of `++` or `--`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepOp {
    /// `++`
    Inc,
    /// `--`
    Dec,
}

impl StepOp {
    /// The operator as written.
    pub fn text(self) -> &'static str {
        match self {
            StepOp::Inc => "++",
            StepOp::Dec => "--",
        }
    }
}

/// A binary operator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Rem,
    /// `**`, the tensor product. **Not** exponentiation: Topiq has no
    /// exponentiation operator at all.
    Tensor,
    /// `&`
    And,
    /// `|`
    Or,
    /// `^`
    Xor,
    /// `<<`
    Shl,
    /// `>>`, formed by the parser from two adjacent `>` tokens.
    Shr,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`, formed by the parser from adjacent `>` and `=`.
    Ge,
    /// `&&`: bool only, and not overloadable.
    AndAnd,
    /// `||`: bool only, and not overloadable.
    OrOr,
}

impl BinOp {
    /// The method this operator desugars to.
    ///
    /// The comparisons collapse onto two methods rather than six: `==` and `!=`
    /// both go through `$eq`, which returns `bool`, and the four orderings
    /// through `$ord`, which returns `Ord`.
    pub fn method(self) -> Option<&'static str> {
        Some(match self {
            BinOp::Add => "$add",
            BinOp::Sub => "$sub",
            BinOp::Mul => "$mul",
            BinOp::Div => "$div",
            BinOp::Rem => "$rem",
            BinOp::Tensor => "$tensor",
            BinOp::And => "$and",
            BinOp::Or => "$or",
            BinOp::Xor => "$xor",
            BinOp::Shl => "$shl",
            BinOp::Shr => "$shr",
            BinOp::Eq | BinOp::Ne => "$eq",
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => "$ord",
            // `&&` and `||` are bool-only and cannot be overloaded: a method
            // call cannot skip evaluating its argument
            BinOp::AndAnd | BinOp::OrOr => return None,
        })
    }

    /// Whether this is one of the comparison operators.
    ///
    /// Comparisons are non-associative, so `a < b < c` is ill-formed rather
    /// than meaning `(a < b) < c` (which would compare a bool with `c`).
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }

    /// Whether this operator may leave its right operand unevaluated.
    pub fn short_circuits(self) -> bool {
        matches!(self, BinOp::AndAnd | BinOp::OrOr)
    }

    /// The spelling.
    pub fn text(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Tensor => "**",
            BinOp::And => "&",
            BinOp::Or => "|",
            BinOp::Xor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::AndAnd => "&&",
            BinOp::OrOr => "||",
        }
    }
}

/// One field of a structure literal.
#[derive(Clone, PartialEq, Debug)]
pub struct FieldInit {
    /// The field name.
    pub name: Spanned<Symbol>,
    /// Its value.
    pub value: Spanned<Expr>,
}

/// One arm of a `match`.
#[derive(Clone, PartialEq, Debug)]
pub struct MatchArm {
    /// The pattern.
    pub pattern: Spanned<Pattern>,
    /// The body.
    pub body: Spanned<Expr>,
}

/// A closure capture.
///
/// Topiq has no implicit capture. A name from an enclosing scope that is not a
/// parameter, not a declared capture, and not an item is simply not in scope
/// inside the body, so what a closure holds is always written down.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Capture {
    /// The name.
    pub name: Spanned<Symbol>,
    /// Whether it is captured by reference (`&x`) rather than moved.
    pub by_ref: bool,
}

/// A parameter of a function or closure.
#[derive(Clone, PartialEq, Debug)]
pub struct Param {
    /// Annotations written on the parameter, such as `[cover: B]`.
    pub annotations: Vec<Spanned<super::annot::AnnotationGroup>>,
    /// The name. `self` is spelled as an ordinary identifier here.
    pub name: Spanned<Symbol>,
    /// The declared type.
    pub ty: Spanned<Type>,
    /// Whether this is the `self` parameter of a method.
    pub is_self: bool,
}

/// The operand of a `prep` expression.
#[derive(Clone, PartialEq, Debug)]
pub enum PrepArg {
    /// A state written out in quantum notation, synthesised while the program
    /// is translated. If the target's gate set cannot prepare it exactly, the
    /// program is refused rather than given an approximation of what it asked
    /// for.
    State(Box<Spanned<crate::quon::ast::QState>>),
    /// A state computed at run time, synthesised by the run-time library
    /// under the same exactness requirement.
    Expr(Box<Spanned<Expr>>),
}

/// An argument to a compiler-supplied `@macro`.
#[derive(Clone, PartialEq, Debug)]
pub enum MacroArg {
    /// A type argument (e.g. `@sizeof(T)`).
    Type(Spanned<Type>),
    /// An expression argument.
    Expr(Spanned<Expr>),
    /// A string literal argument (e.g. `@has_method(T, "$add")`).
    Str(Spanned<Symbol>),
    /// A QUON form.
    Quon(Box<crate::quon::ast::QForm>),
}

/// An expression.
#[derive(Clone, PartialEq, Debug)]
pub enum Expr {
    /// An integer literal.
    Int {
        /// The digits as written.
        raw: Symbol,
        /// The base.
        base: IntBase,
        /// The type suffix.
        suffix: Option<IntSuffix>,
    },
    /// A floating literal.
    Float {
        /// The literal as written.
        raw: Symbol,
        /// The type suffix.
        suffix: Option<FloatSuffix>,
    },
    /// A character literal.
    Char(char),
    /// `true` or `false`.
    Bool(bool),
    /// A string literal.
    Str {
        /// The contents.
        value: Symbol,
        /// How it was written.
        kind: StrKind,
    },
    /// A path.
    ///
    /// In expression position generic arguments are always written
    /// `path::<args>`, never `path<args>`. That is what lets `a < b` and
    /// `c > d` each have exactly one reading, with no lookahead.
    Path {
        /// The name.
        path: Path,
        /// Generic arguments from a `::<…>` suffix.
        args: Vec<Spanned<TArg>>,
    },
    /// A parenthesised expression.
    Paren(Box<Spanned<Expr>>),
    /// `(a, b, …)`: a tuple.
    Tuple(Vec<Spanned<Expr>>),
    /// `[a, b, …]`: an array literal.
    Array(Vec<Spanned<Expr>>),
    /// `[e; N]`: a repeated array literal.
    ArrayRepeat {
        /// The element.
        elem: Box<Spanned<Expr>>,
        /// The count.
        len: Box<Spanned<Expr>>,
    },
    /// `Path { field: value, … }`: a structure literal, which is also the
    /// form a data document uses for an object.
    StructLit {
        /// The structure's name.
        path: Path,
        /// The field initialisers, in the order written, which is the order
        /// they evaluate in.
        fields: Vec<FieldInit>,
    },
    /// A block used as an expression.
    Block(Box<Block>),
    /// `if cond { … } else { … }`.
    ///
    /// Inside a quantum unit, with a quantum condition, this is lifted to
    /// controlled circuit structure rather than executed.
    If {
        /// The condition.
        cond: Box<Spanned<Expr>>,
        /// The consequent.
        then: Box<Block>,
        /// The alternative.
        els: Option<Box<Spanned<Expr>>>,
    },
    /// `match e { … }`, or `match measure e { … }`.
    Match {
        /// Whether `match measure e` was written, which measures and
        /// dispatches in one step, binding the payload classically.
        measuring: bool,
        /// The scrutinee.
        scrutinee: Box<Spanned<Expr>>,
        /// The arms.
        arms: Vec<MatchArm>,
    },
    /// `while cond { … }`.
    While {
        /// The condition.
        cond: Box<Spanned<Expr>>,
        /// The body.
        body: Box<Block>,
    },
    /// `loop { … }`: runs forever unless broken out of, and `break` may
    /// carry a value.
    Loop {
        /// The body.
        body: Box<Block>,
    },
    /// `for pat in iter { … }`, desugaring through `$iter` and `$next`.
    For {
        /// The binding pattern.
        pattern: Box<Spanned<Pattern>>,
        /// The iterable.
        iter: Box<Spanned<Expr>>,
        /// The body.
        body: Box<Block>,
    },
    /// `fn (params) -> T with (captures) { … }`: a closure. The capture
    /// list is mandatory: nothing is captured implicitly.
    Closure {
        /// The parameters.
        params: Vec<Param>,
        /// The return type.
        ret: Option<Box<Spanned<Type>>>,
        /// The capture list.
        captures: Vec<Capture>,
        /// The body.
        body: Box<Block>,
    },
    /// `measure e`: collapses the operand to a classical outcome, consuming
    /// it.
    Measure(Box<Spanned<Expr>>),
    /// `prep ψ`: prepares a register in the named state, refusing rather
    /// than approximating if it cannot be prepared exactly.
    Prep(PrepArg),
    /// `query t[k]`: coherent map access, which does not measure and does not
    /// consume the key register.
    Query {
        /// The map.
        map: Box<Spanned<Expr>>,
        /// The key register.
        index: Box<Spanned<Expr>>,
    },
    /// `lift e`: the sole controlled crossing from outcome to generation time,
    /// which makes the enclosing circuit dynamic.
    Lift(Box<Spanned<Expr>>),
    /// `replay f(args)`: a second, independent preparation of the same state,
    /// valid only on a monic operator. This is not cloning: the state is
    /// rebuilt from its construction, not copied.
    Replay(Box<Spanned<Expr>>),
    /// `@name(args)`: a macro the compiler supplies, expanded in phase 7.
    Macro {
        /// The name.
        name: Spanned<Symbol>,
        /// The arguments.
        args: Vec<Spanned<MacroArg>>,
    },
    /// A unary operation.
    Unary {
        /// The operator.
        op: UnOp,
        /// The operand.
        operand: Box<Spanned<Expr>>,
    },
    /// A binary operation.
    Binary {
        /// The operator.
        op: BinOp,
        /// The left operand.
        lhs: Box<Spanned<Expr>>,
        /// The right operand.
        rhs: Box<Spanned<Expr>>,
    },
    /// `++x`, `x++`, `--x` or `x--`: the place changed by one, giving its
    /// value after the change when the operator comes first and before it
    /// when the operator comes last. The place is evaluated once.
    Step {
        /// Which way the place moves.
        op: StepOp,
        /// Whether the operator follows the place, giving the old value.
        post: bool,
        /// The place.
        place: Box<Spanned<Expr>>,
    },
    /// `a = b`, or a compound assignment such as `a += b`.
    ///
    /// `a op= b` becomes `a = a.$op(b)`, or a call of `$op_assign` where the
    /// type defines one, which can be cheaper than read-modify-write.
    ///
    /// The operands evaluate **right to left**, the value first, so an
    /// aborting place cannot mask an abort in the value.
    Assign {
        /// The compound operator.
        op: Option<BinOp>,
        /// The place being assigned.
        place: Box<Spanned<Expr>>,
        /// The value.
        value: Box<Spanned<Expr>>,
    },
    /// `a..b` or `a..=b`, yielding a `Range<T>`.
    Range {
        /// The start.
        start: Option<Box<Spanned<Expr>>>,
        /// The end.
        end: Option<Box<Spanned<Expr>>>,
        /// Whether the end is included.
        inclusive: bool,
    },
    /// `f(args)`, which desugars to `$call` on a non-function value.
    Call {
        /// The callee.
        callee: Box<Spanned<Expr>>,
        /// The arguments.
        args: Vec<Spanned<Expr>>,
    },
    /// `e[i]`: indexing, always bounds-checked.
    Index {
        /// The receiver.
        receiver: Box<Spanned<Expr>>,
        /// The index.
        index: Box<Spanned<Expr>>,
    },
    /// `e.f`: field access.
    Field {
        /// The receiver.
        receiver: Box<Spanned<Expr>>,
        /// The field name.
        name: Spanned<Symbol>,
    },
    /// `e.m(args)`: a method call, with one level of automatic reference or
    /// dereference adjustment on the receiver.
    MethodCall {
        /// The receiver.
        receiver: Box<Spanned<Expr>>,
        /// The method name.
        name: Spanned<Symbol>,
        /// Turbofish generic arguments.
        targs: Vec<Spanned<TArg>>,
        /// The arguments.
        args: Vec<Spanned<Expr>>,
    },
    /// `e?`: error propagation by early return, running the destructions of
    /// the scopes exited.
    Try(Box<Spanned<Expr>>),
    /// `e as T`: an explicit conversion.
    Cast {
        /// The operand.
        expr: Box<Spanned<Expr>>,
        /// The target type.
        ty: Box<Spanned<Type>>,
    },
}

impl Expr {
    /// Whether this expression form begins with a block-introducing keyword or
    /// a brace.
    ///
    /// A statement beginning with one of these ends at its closing `}`, so
    /// `{ … } * x;` is **two statements** (a block, then `* x;`) rather than
    /// one multiplication. Parenthesise the block to get the other reading.
    pub fn is_block_form(&self) -> bool {
        matches!(
            self,
            Expr::Block(_)
                | Expr::If { .. }
                | Expr::Match { .. }
                | Expr::While { .. }
                | Expr::Loop { .. }
                | Expr::For { .. }
        )
    }

    /// Whether this expression denotes a place, and so may be assigned to or
    /// have a reference taken of it.
    pub fn is_place(&self) -> bool {
        match self {
            Expr::Path { .. } | Expr::Index { .. } | Expr::Field { .. } => true,
            Expr::Paren(inner) => inner.node.is_place(),
            Expr::Unary { op: UnOp::Deref, .. } => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_overloadable_operator_names_its_method() {
        // the names are the contract, checked
        // structurally, so getting one wrong would silently change which method
        // an operator finds
        let pairs = [
            (BinOp::Add, "$add"),
            (BinOp::Sub, "$sub"),
            (BinOp::Mul, "$mul"),
            (BinOp::Div, "$div"),
            (BinOp::Rem, "$rem"),
            (BinOp::Tensor, "$tensor"),
            (BinOp::And, "$and"),
            (BinOp::Or, "$or"),
            (BinOp::Xor, "$xor"),
            (BinOp::Shl, "$shl"),
            (BinOp::Shr, "$shr"),
        ];
        for (op, name) in pairs {
            assert_eq!(op.method(), Some(name), "{op:?}");
        }
        assert_eq!(UnOp::Neg.method(), Some("$neg"));
        assert_eq!(UnOp::Not.method(), Some("$not"));
    }

    #[test]
    fn the_logical_operators_are_not_overloadable() {
        // both are bool-only, so neither desugars to a method
        assert_eq!(BinOp::AndAnd.method(), None);
        assert_eq!(BinOp::OrOr.method(), None);
    }

    #[test]
    fn taking_or_following_a_reference_is_not_a_method_call() {
        assert_eq!(UnOp::Ref.method(), None);
        assert_eq!(UnOp::Deref.method(), None);
    }

    #[test]
    fn equality_and_ordering_collapse_onto_two_methods() {
        // `$eq` returns bool and covers `==` and `!=`; `$ord`
        // returns Ord and covers the four orderings
        assert_eq!(BinOp::Eq.method(), Some("$eq"));
        assert_eq!(BinOp::Ne.method(), Some("$eq"));
        for op in [BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge] {
            assert_eq!(op.method(), Some("$ord"), "{op:?}");
        }
    }

    #[test]
    fn the_comparisons_are_exactly_the_six_non_associative_operators() {
        let cmp: Vec<&str> = [
            BinOp::Add,
            BinOp::Eq,
            BinOp::Ne,
            BinOp::Lt,
            BinOp::Le,
            BinOp::Gt,
            BinOp::Ge,
            BinOp::AndAnd,
            BinOp::Tensor,
        ]
        .into_iter()
        .filter(|op| op.is_comparison())
        .map(BinOp::text)
        .collect();
        assert_eq!(cmp, vec!["==", "!=", "<", "<=", ">", ">="]);
    }

    #[test]
    fn only_the_logical_operators_short_circuit() {
        assert!(BinOp::AndAnd.short_circuits());
        assert!(BinOp::OrOr.short_circuits());
        assert!(!BinOp::And.short_circuits());
        assert!(!BinOp::Or.short_circuits());
        assert!(!BinOp::Add.short_circuits());
    }

    #[test]
    fn the_tensor_operator_is_not_exponentiation() {
        // `**` is the tensor product, the Kronecker product on matrices, not a
        // power
        assert_eq!(BinOp::Tensor.text(), "**");
        assert_eq!(BinOp::Tensor.method(), Some("$tensor"));
    }

    #[test]
    fn block_forms_are_the_ones_a_5_4_names() {
        // a statement beginning with one of these ends at its closing brace
        let block = || Box::new(Block::default());
        let forms: Vec<bool> = vec![
            Expr::Block(block()).is_block_form(),
            Expr::Loop { body: block() }.is_block_form(),
            Expr::Bool(true).is_block_form(),
            Expr::Char('a').is_block_form(),
        ];
        assert_eq!(forms, vec![true, true, false, false]);
    }

    #[test]
    fn places_are_the_assignable_forms() {
        let mut i = crate::intern::Interner::new();
        let at = crate::span::Span::new(crate::span::SourceId(0), 0, 1);
        let sp = |e: Expr| Spanned::new(e, at);
        let path = Expr::Path {
            path: Path::single(Spanned::new(i.intern("x"), at)),
            args: Vec::new(),
        };
        assert!(path.is_place());
        assert!(Expr::Paren(Box::new(sp(path.clone()))).is_place());
        assert!(!Expr::Bool(true).is_place());
        assert!(
            !Expr::Unary {
                op: UnOp::Ref,
                operand: Box::new(sp(path.clone()))
            }
            .is_place(),
            "a reference is a value, not a place"
        );
        assert!(
            Expr::Unary {
                op: UnOp::Deref,
                operand: Box::new(sp(path))
            }
            .is_place(),
            "what a reference refers to is a place"
        );
    }

    #[test]
    fn assignment_records_that_its_value_evaluates_first() {
        // an assignment evaluates its value before its place
        let sp = |e: Expr| {
            Spanned::new(
                e,
                crate::span::Span::new(crate::span::SourceId(0), 0, 1),
            )
        };
        let a = Expr::Assign {
            op: Some(BinOp::Add),
            place: Box::new(sp(Expr::Bool(false))),
            value: Box::new(sp(Expr::Bool(true))),
        };
        match a {
            Expr::Assign { op, .. } => assert_eq!(op, Some(BinOp::Add)),
            _ => panic!("expected an assignment"),
        }
    }

    #[test]
    fn operator_spellings_are_unique() {
        let ops = [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::Div,
            BinOp::Rem,
            BinOp::Tensor,
            BinOp::And,
            BinOp::Or,
            BinOp::Xor,
            BinOp::Shl,
            BinOp::Shr,
            BinOp::Eq,
            BinOp::Ne,
            BinOp::Lt,
            BinOp::Le,
            BinOp::Gt,
            BinOp::Ge,
            BinOp::AndAnd,
            BinOp::OrOr,
        ];
        let mut seen = std::collections::HashSet::new();
        for op in ops {
            assert!(seen.insert(op.text()), "duplicate spelling {}", op.text());
        }
        assert_eq!(seen.len(), 19);
    }
}
