//! Syntax nodes for quantum states, covers and gauges.
//!
//! These describe quantum states and quantum type annotations. Unlike
//! classical data notation, **at run time one of these values is a classical
//! description of a quantum state**, not the state.
//!
//! Live quantum state cannot be serialised: that would be tomography, or
//! cloning. The other way, turning a description into state, is
//! preparation, which is a circuit.
//!
//! # Where these appear
//!
//! Only after `prep`, in cover and gauge declarations and annotations, and in
//! `.quon` documents. Those positions define a quantum context, which is what
//! makes the ket reassembly in [`super::ket`] sound.

use crate::intern::Symbol;
use crate::lex::IntBase;
use crate::span::{Span, Spanned};

/// An exact-scalar constant.
///
/// Each spelling is also an ordinary identifier, so these are recognised from
/// identifiers where the QUON grammar leaves no other reading. See
/// [`crate::lex::token`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Exact {
    /// `i`, the imaginary unit, which is `ζ_N^(N/4)`.
    I,
    /// `isq2`, one over the square root of two, which needs 8 to divide N.
    Isq2,
    /// `w(k, N)`, denoting e^(2πik/N) as a `cyclo<N>`.
    W {
        /// The numerator of the angle.
        k: u64,
        /// The order of the root of unity.
        n: u64,
    },
}

/// A coefficient expression.
///
/// A closed arithmetic over exact-scalar constants: no general expression
/// appears inside a quantum context, so no `|` there can be a bitwise or and
/// no `>` can close a comparison, which makes ket reconstruction sound.
#[derive(Clone, PartialEq, Debug)]
pub enum QExpr {
    /// An integer literal: its digits as written, and the base they are in.
    Int(Symbol, IntBase),
    /// A floating literal.
    Float(Symbol),
    /// An exact-scalar constant.
    Exact(Exact),
    /// Negation.
    Neg(Box<Spanned<QExpr>>),
    /// A binary operation over the closed coefficient arithmetic.
    Binary {
        /// The operator.
        op: QOp,
        /// Left operand.
        lhs: Box<Spanned<QExpr>>,
        /// Right operand.
        rhs: Box<Spanned<QExpr>>,
    },
}

/// The coefficient operators: addition, subtraction, multiplication and
/// division only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
}

/// A computational-basis ket.
///
/// `bits` holds the digits exactly as written, so `|01>` and `|1>` stay
/// distinct: they name different basis states of registers of different width.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ket {
    /// The bit string as written.
    pub bits: String,
}

impl Ket {
    /// The number of qubits the ket names.
    pub fn width(&self) -> usize {
        self.bits.len()
    }

    /// The basis index, or `None` if the ket is wider than 64 qubits.
    ///
    /// Wide kets are legal (a register may hold 4096 qubits), so this
    /// deliberately reports absence rather than truncating.
    pub fn index(&self) -> Option<u64> {
        if self.bits.len() > 64 {
            return None;
        }
        u64::from_str_radix(&self.bits, 2).ok()
    }
}

/// A factor of a tensor product.
#[derive(Clone, PartialEq, Debug)]
pub enum QFactor {
    /// A computational-basis ket.
    Ket(Ket),
    /// A named state.
    Name(Symbol),
    /// A parenthesised state.
    Paren(Box<Spanned<QState>>),
}

/// One term of a state: a coefficient times a tensor product of factors.
///
/// A declared tensor grouping must not be re-associated, so the
/// factors are kept as a sequence in the order written rather than folded into
/// a binary tree.
#[derive(Clone, PartialEq, Debug)]
pub struct QTerm {
    /// The coefficient.
    pub coeff: Option<Spanned<QExpr>>,
    /// The tensor factors.
    pub factors: Vec<Spanned<QFactor>>,
}

/// A quantum state: a signed sum of terms.
#[derive(Clone, PartialEq, Debug)]
pub struct QState {
    /// The first term.
    pub head: Spanned<QTerm>,
    /// Subsequent terms with their signs.
    pub tail: Vec<(Sign, Spanned<QTerm>)>,
}

impl QState {
    /// A state of a single term.
    pub fn single(term: Spanned<QTerm>) -> QState {
        QState {
            head: term,
            tail: Vec::new(),
        }
    }

    /// How many terms the state has.
    pub fn term_count(&self) -> usize {
        1 + self.tail.len()
    }

    /// Each term with the sign before it, the first's `+`.
    pub fn terms(&self) -> impl Iterator<Item = (Sign, &Spanned<QTerm>)> {
        std::iter::once((Sign::Plus, &self.head)).chain(self.tail.iter().map(|(sign, t)| (*sign, t)))
    }
}

/// Whether a state writes a ket anywhere, rather than only names and
/// coefficients.
pub fn writes_ket(s: &QState) -> bool {
    s.terms()
        .any(|(_, t)| {
            t.node.factors.iter().any(|f| match &f.node {
                QFactor::Ket(_) => true,
                QFactor::Paren(inner) => writes_ket(&inner.node),
                QFactor::Name(_) => false,
            })
        })
}

/// The sign joining two terms.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sign {
    /// `+`
    Plus,
    /// `-`
    Minus,
}

/// A Pauli string, as written in a `code<...>` cover.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Pauli {
    /// An optional overall sign.
    pub sign: Option<Sign>,
    /// The letters.
    pub letters: String,
}

/// A cover: the nonempty set of states a judgement is made over.
#[derive(Clone, PartialEq, Debug)]
pub enum QCover {
    /// `pt(ψ)`: a single point.
    Pt(Box<Spanned<QState>>),
    /// `fin{ ψ, … }`: a finite set of points.
    Fin(Vec<Spanned<QState>>),
    /// `span{ |b⟩, … }`: the projective span of basis kets.
    ///
    /// Not redundant with `code`. The three-dimensional
    /// unmarked subspace of a four-point Grover search is a span of basis
    /// states, and three not being a power of two, no stabiliser code presents
    /// it.
    Span(Vec<Spanned<Ket>>),
    /// `code<P, …>`: a stabiliser code given by its generators.
    Code(Vec<Spanned<Pauli>>),
    /// A named cover declared elsewhere.
    Name(DeclName),
}

/// The name of a cover, gauge or base type declared elsewhere: in this
/// unit, `T01`, or in another, with that unit's name in front, `gates::T01`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeclName {
    /// The unit that declares it.
    pub unit: Option<Symbol>,
    /// Its name there.
    pub name: Symbol,
}

impl DeclName {
    /// A name declared in this unit.
    pub fn local(name: Symbol) -> DeclName {
        DeclName { unit: None, name }
    }
}

/// A gauge: a phase convention on a cover.
#[derive(Clone, PartialEq, Debug)]
pub enum QGauge {
    /// `fid(ω)`: a single-patch gauge presented by one vector, a *fiducial
    /// pregauge*.
    ///
    /// The fiducial must have nonzero overlap with every point of the cover,
    /// since the gauge is built by projecting onto it. A point orthogonal to
    /// it is `EJ10`.
    Fid(Box<Spanned<QState>>),
    /// `atlas{ g, … }`: a multi-patch gauge.
    ///
    /// The patches must be relatively open in the cover, each avoiding the
    /// hyperplane of its own fiducial, and between them must cover it.
    Atlas(Vec<Spanned<QGauge>>),
    /// `none`: stationary-only, declaring that no phase convention exists.
    ///
    /// A `span` or `code` cover of projective dimension at least one admits
    /// no single-patch gauge at all, so it must be declared `none` or given a
    /// multi-patch `atlas`. A lone `fid` on such a cover is `EJ09`.
    None,
    /// A named gauge declared elsewhere.
    Name(DeclName),
}

/// The register type a quantum item is stated against.
#[derive(Clone, PartialEq, Debug)]
pub enum QType {
    /// `[qubit; N]`: a fixed register.
    Register(Spanned<Symbol>),
    /// `qubit`: a single qubit.
    Qubit,
    /// A path naming a quantum enumeration.
    Path(Vec<Symbol>),
}

/// One item of a `.quon` document.
#[derive(Clone, PartialEq, Debug)]
pub struct QItem {
    /// The item's name.
    pub name: Spanned<Symbol>,
    /// The register type it is stated against.
    pub ty: Spanned<QType>,
    /// The state.
    pub state: Spanned<QState>,
}

/// A `.quon` document.
///
/// `@embed("states.quon")` yields a constant structure of
/// `quon::State` values named per item, with a schema type generated
/// implicitly.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct QDoc {
    /// The items.
    pub items: Vec<QItem>,
}

/// Where a quantum form was written, which decides which production applies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QContext {
    /// After `prep`.
    Prep,
    /// Inside an argument of the `cover`, `gauge` or `frame` annotations.
    Annotation,
    /// The right-hand side of a `cover`, `gauge` or `base` declaration.
    Declaration,
    /// Inside a `macro-arg` of a macro documented as taking a QUON form.
    MacroArgument,
    /// A whole `.quon` file.
    Document,
}

/// A QUON form as it appears in an annotation argument or a macro argument.
#[derive(Clone, PartialEq, Debug)]
pub enum QForm {
    /// A state.
    State(Spanned<QState>),
    /// A cover.
    Cover(Spanned<QCover>),
    /// A gauge.
    Gauge(Spanned<QGauge>),
}

impl QForm {
    /// Where the form was written.
    pub fn span(&self) -> Span {
        match self {
            QForm::State(s) => s.span,
            QForm::Cover(c) => c.span,
            QForm::Gauge(g) => g.span,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::SourceId;

    fn sp<T>(node: T) -> Spanned<T> {
        Spanned::new(node, Span::new(SourceId(0), 0, 1))
    }

    fn ket(bits: &str) -> Ket {
        Ket {
            bits: bits.to_owned(),
        }
    }

    #[test]
    fn a_kets_width_is_its_digit_count() {
        assert_eq!(ket("0").width(), 1);
        assert_eq!(ket("00").width(), 2);
        assert_eq!(ket("0110").width(), 4);
    }

    #[test]
    fn kets_of_different_width_are_different_states() {
        // `|01>` names a two-qubit register, `|1>` a one-qubit one. a parsed
        // integer value could not tell them apart, which is why the digits are
        // kept as written
        assert_ne!(ket("01"), ket("1"));
        assert_eq!(ket("01").index(), ket("1").index());
        assert_ne!(ket("01").width(), ket("1").width());
    }

    #[test]
    fn a_kets_index_is_its_binary_value() {
        assert_eq!(ket("0").index(), Some(0));
        assert_eq!(ket("1").index(), Some(1));
        assert_eq!(ket("11").index(), Some(3));
        assert_eq!(ket("0110").index(), Some(6));
    }

    #[test]
    fn a_very_wide_ket_reports_no_index_rather_than_truncating() {
        // a register may hold 4096 qubits, so a ket can be far wider
        // than any integer type
        let wide = ket(&"1".repeat(100));
        assert_eq!(wide.width(), 100);
        assert_eq!(wide.index(), None);
        // at the boundary it still works
        assert!(ket(&"1".repeat(64)).index().is_some());
        assert!(ket(&"1".repeat(65)).index().is_none());
    }

    #[test]
    fn a_state_counts_its_terms() {
        let term = sp(QTerm {
            coeff: None,
            factors: vec![sp(QFactor::Ket(ket("0")))],
        });
        let single = QState::single(term.clone());
        assert_eq!(single.term_count(), 1);

        let bell = QState {
            head: term.clone(),
            tail: vec![(Sign::Plus, term)],
        };
        assert_eq!(bell.term_count(), 2);
    }

    #[test]
    fn tensor_factors_keep_the_order_written() {
        // a tensor grouping is judgement-significant and must
        // not be re-associated, so the factors are a sequence, not a tree
        let t = QTerm {
            coeff: None,
            factors: vec![
                sp(QFactor::Ket(ket("0"))),
                sp(QFactor::Ket(ket("1"))),
                sp(QFactor::Ket(ket("0"))),
            ],
        };
        let widths: Vec<usize> = t
            .factors
            .iter()
            .map(|f| match &f.node {
                QFactor::Ket(k) => k.width(),
                _ => 0,
            })
            .collect();
        assert_eq!(widths, vec![1, 1, 1]);
        assert_eq!(t.factors.len(), 3, "three factors, not a nested pair");
    }

    #[test]
    fn the_exact_constants_are_distinct() {
        assert_ne!(Exact::I, Exact::Isq2);
        assert_eq!(Exact::W { k: 1, n: 8 }, Exact::W { k: 1, n: 8 });
        assert_ne!(Exact::W { k: 1, n: 8 }, Exact::W { k: 2, n: 8 });
    }

    #[test]
    fn a_gauge_can_be_stationary_only() {
        // the required form for a subspace cover of
        // projective dimension at least one
        assert_eq!(QGauge::None, QGauge::None);
        assert_ne!(QGauge::None, QGauge::Name(DeclName::local(Symbol::EMPTY)));
    }

    #[test]
    fn a_form_reports_where_it_was_written() {
        let s = sp(QState::single(sp(QTerm {
            coeff: None,
            factors: vec![],
        })));
        let span = s.span;
        assert_eq!(QForm::State(s).span(), span);
        assert_eq!(QForm::Gauge(sp(QGauge::None)).span(), span);
    }

    #[test]
    fn covers_are_distinguished_by_their_presentation() {
        // `span` is needed in its own right and is not
        // redundant with `code`
        let s = QCover::Span(vec![sp(ket("00")), sp(ket("01")), sp(ket("10"))]);
        let c = QCover::Code(vec![sp(Pauli {
            sign: None,
            letters: "XZ".to_owned(),
        })]);
        assert_ne!(s, c);
        match s {
            QCover::Span(kets) => assert_eq!(kets.len(), 3, "three of four basis states"),
            _ => panic!("expected a span"),
        }
    }
}
