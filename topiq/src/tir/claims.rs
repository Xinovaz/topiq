//! What a quantum unit claims of its operators, and asks of their
//! judgements.
//!
//! Analysis reads the annotations that make claims (`[cover:]`,
//! `[gauge:]`, `[expect:]`, `[frame:]`, `[glue:]`, `[iso:]`): into
//! [`Claims`], and the macros that ask about judgements into [`Derived`]
//! values. Both are settled once the unit's circuits are generated, when the
//! judgements exist.

use crate::exact::Phase;
use crate::judge::cert::{Base, Class};
use crate::judge::record::Declared;
use crate::span::Span;

use super::{Expr, FnId, Ty};

/// The claims an operator's annotations make.
#[derive(Clone, Debug, Default)]
pub struct Claims {
    /// The endpoints `[cover:]` and `[gauge:]` on the operator give: its
    /// register's states, from and to.
    pub endpoints: Option<(Base, Span)>,
    /// The cover declaration `[cover:]` names, or the base type's whose
    /// name it gives; `None` when the cover is written out or not written.
    pub cover_decl: Option<Declared>,
    /// The gauge declaration `[gauge:]` names, or the base type's.
    pub gauge_decl: Option<Declared>,
    /// The `[expect: …]` claims, each where it is written.
    pub expects: Vec<(Expect, Span)>,
    /// The pair groupings `[frame: …]` declares.
    pub frames: Vec<(Grouping, Span)>,
    /// Where `[glue: piecewise]` is written, if it is.
    pub glue: Option<Span>,
}

/// One claim of `[expect: …]`.
#[derive(Clone, Debug)]
pub enum Expect {
    /// `monic`.
    Monic,
    /// `unitary`.
    Unitary,
    /// `contract = C`, or a class written alone.
    Contract(Class),
    /// `frame(flat(k))`: the operator, tensored with the identity, is
    /// branch-relatively flat; with `None`, at some constant.
    Frame(Option<Phase>),
    /// `kernel.outcomes = T`: the operator is an instrument whose outcome
    /// type is `T`.
    Outcomes(Ty),
}

/// A tensor grouping of an operator's quantum parameters.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Grouping {
    /// A parameter.
    Param(usize),
    /// `a ** b`.
    Pair(Box<Grouping>, Box<Grouping>),
}

impl Grouping {
    /// The parameters it groups.
    pub fn params(&self) -> Vec<usize> {
        match self {
            Grouping::Param(i) => vec![*i],
            Grouping::Pair(a, b) => {
                let mut v = a.params();
                v.extend(b.params());
                v
            }
        }
    }
}

/// A value asked of the unit's judgements.
#[derive(Clone, Debug)]
pub enum Derived {
    /// `@judgment(f)`.
    Judgment(FnId),
    /// `@kernel(f)`.
    Kernel(FnId),
    /// `@certificate(f)`.
    Certificate(FnId),
    /// `@matrix_of(f)`.
    MatrixOf(FnId),
    /// `@fragment_of(f)`.
    FragmentOf(FnId),
    /// `@classify_op(f, B)`: the judgement of `f` from the base type `B`,
    /// by position.
    ClassifyOp(FnId, usize),
    /// `@holonomy(C, s)`: the chain, by position, and the point of its first
    /// base type's cover, by index.
    Holonomy(usize, usize),
}

/// An `@static_assert` whose condition reads a judgement, which is checked
/// once the judgements exist.
#[derive(Clone, Debug)]
pub struct Deferred {
    /// The condition.
    pub cond: Expr,
    /// The message.
    pub message: String,
    /// Where the condition is written.
    pub span: Span,
}

/// An enumeration's `[iso: …]` witnesses, checked once their judgements
/// exist.
#[derive(Clone, Debug)]
pub struct Iso {
    /// The enumeration.
    pub adt: super::AdtId,
    /// The witnesses: ι₁₀, ι₂₀, ι₂₁, …, for each variant, one to it from
    /// each variant before it.
    pub witnesses: Vec<(FnId, Span)>,
    /// Where the annotation is written.
    pub span: Span,
}
