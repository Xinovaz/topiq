//! What a state written in QUON is: its amplitudes.
//!
//! [`evaluate`] turns a [`QState`] into [`Amplitudes`], the non-zero
//! amplitude of each basis state, at the unit's conductor N. Every
//! coefficient is an element of `cyclo<N>`, computed exactly; one that is not
//! in that field is refused (`EJ04`), naming the conductor that would hold it.
//! A factor's name is resolved by the caller, which knows what the name
//! refers to.
//!
//! Whether the result is a unit vector is a separate question,
//! [`Amplitudes::is_normalized`], answered exactly in the field.

use std::collections::BTreeMap;

use num_bigint::BigInt;

use super::ast::{Exact, QExpr, QFactor, QOp, QState, Sign};
use crate::diag::{Code, Diagnostic};
use crate::exact::{Cyclo, Frac};
use crate::intern::{Interner, Symbol};
use crate::span::{Span, Spanned};

/// A state's amplitudes over the computational basis.
///
/// A basis state is its bits, the register's first qubit first; `terms`
/// holds only the non-zero amplitudes.
#[derive(Clone, PartialEq, Debug)]
pub struct Amplitudes {
    /// How many qubits.
    pub width: usize,
    /// The conductor the amplitudes are in.
    pub n: u32,
    /// The non-zero amplitudes.
    pub terms: BTreeMap<Vec<bool>, Cyclo>,
}

impl Amplitudes {
    /// The basis state `bits`.
    pub fn basis(bits: Vec<bool>, n: u32) -> Amplitudes {
        Amplitudes {
            width: bits.len(),
            n,
            terms: BTreeMap::from([(bits, Cyclo::one(n))]),
        }
    }

    /// This state with its amplitudes in the field of conductor `n`, when
    /// its own conductor divides `n`; as it is otherwise.
    pub fn at(&self, n: u32) -> Amplitudes {
        let terms: Option<BTreeMap<Vec<bool>, Cyclo>> =
            self.terms.iter().map(|(k, c)| Some((k.clone(), c.embed(n)?))).collect();
        match terms {
            Some(terms) => Amplitudes {
                width: self.width,
                n,
                terms,
            },
            None => self.clone(),
        }
    }

    /// Σ|c|², the squared length of the vector.
    pub fn norm_sq(&self) -> Cyclo {
        self.terms
            .values()
            .fold(Cyclo::zero(self.n), |acc, c| acc.add(&c.norm_sq()))
    }

    /// Whether this is a unit vector, as a state must be.
    pub fn is_normalized(&self) -> bool {
        self.norm_sq().is_one()
    }

    /// The tensor product `self ⊗ other`.
    fn tensor(&self, other: &Amplitudes) -> Amplitudes {
        let mut terms = BTreeMap::new();
        for (a, x) in &self.terms {
            for (b, y) in &other.terms {
                let mut bits = a.clone();
                bits.extend_from_slice(b);
                terms.insert(bits, x.mul(y));
            }
        }
        Amplitudes {
            width: self.width + other.width,
            n: self.n,
            terms,
        }
    }

    /// Adds `c · other`, dropping amplitudes that cancel.
    fn add_scaled(&mut self, other: &Amplitudes, c: &Cyclo) {
        for (bits, y) in &other.terms {
            let sum = match self.terms.get(bits) {
                Some(x) => x.add(&c.mul(y)),
                None => c.mul(y),
            };
            if sum.is_zero() {
                self.terms.remove(bits);
            } else {
                self.terms.insert(bits.clone(), sum);
            }
        }
    }

    /// Multiplies every amplitude by `c`.
    fn scaled(mut self, c: &Cyclo) -> Amplitudes {
        if c.is_zero() {
            self.terms.clear();
        } else {
            for x in self.terms.values_mut() {
                *x = x.mul(c);
            }
        }
        self
    }
}

/// A state as it would be written, for a message: `|11>`, or its terms
/// with their amplitudes, `ζ` written `z8` for conductor 8.
pub fn show(a: &Amplitudes) -> String {
    if a.terms.is_empty() {
        return "0".to_owned();
    }
    a.terms
        .iter()
        .map(|(bits, c)| if c.is_one() { ket(bits) } else { format!("({c}) * {}", ket(bits)) })
        .collect::<Vec<_>>()
        .join(" + ")
}

/// The basis ket numbered `i` among those of `width` qubits: its bits, the
/// first qubit's the most significant.
pub fn ket_bits(i: usize, width: usize) -> Vec<bool> {
    (0..width).map(|q| (i >> (width - 1 - q)) & 1 == 1).collect()
}

/// A basis state as a ket (e.g. `|0110>`).
pub fn ket(bits: &[bool]) -> String {
    let digits: String = bits.iter().map(|&b| if b { '1' } else { '0' }).collect();
    format!("|{digits}>")
}

/// How [`evaluate`] finds the state a name in a tensor product refers to:
/// `None` when it reported why there is none.
pub type Names<'a> = dyn FnMut(Symbol, Span) -> Option<Amplitudes> + 'a;

/// The amplitudes of `s` at conductor `n`.
///
/// # Errors
///
/// What makes the state not one: a coefficient outside the field, terms of
/// different widths, a division by zero. A name `names` could not resolve
/// gives an empty list, `names` having reported why.
pub fn evaluate(
    s: &Spanned<QState>,
    n: u32,
    interner: &Interner,
    names: &mut Names<'_>,
) -> Result<Amplitudes, Vec<Diagnostic>> {
    let mut ev = Eval {
        n,
        interner,
        names,
        diags: Vec::new(),
    };
    match ev.state(s) {
        Some(a) if ev.diags.is_empty() => Ok(a),
        _ => Err(ev.diags),
    }
}

struct Eval<'a, 'b> {
    n: u32,
    interner: &'a Interner,
    names: &'a mut Names<'b>,
    diags: Vec<Diagnostic>,
}

impl Eval<'_, '_> {
    fn state(&mut self, s: &Spanned<QState>) -> Option<Amplitudes> {
        let terms = s.node.terms();
        let mut sum: Option<(Amplitudes, Span)> = None;
        let mut failed = false;
        for (sign, t) in terms {
            let Some(mut value) = self.term(t) else {
                failed = true;
                continue;
            };
            if sign == Sign::Minus {
                value = value.scaled(&Cyclo::one(self.n).neg());
            }
            match &mut sum {
                None => sum = Some((value, t.span)),
                Some((acc, first)) if acc.width != value.width => {
                    self.diags.push(
                        Diagnostic::new(Code::Es06)
                            .with_message(format!(
                                "this state's terms are of different widths: {} qubits and {} qubits",
                                acc.width, value.width
                            ))
                            .at_with(t.span, format!("this term has {} qubits", value.width))
                            .also(*first, format!("this term has {} qubits", acc.width))
                            .with_note("a state is a sum of states of one register, so every term names as many qubits")
                            .with_help("write each term with the same number of digits in its kets"),
                    );
                    failed = true;
                }
                Some((acc, _)) => acc.add_scaled(&value, &Cyclo::one(self.n)),
            }
        }
        if failed { None } else { sum.map(|(a, _)| a) }
    }

    fn term(&mut self, t: &Spanned<super::ast::QTerm>) -> Option<Amplitudes> {
        let coeff = match &t.node.coeff {
            Some(c) => Some(self.coeff(c)?),
            None => None,
        };
        let mut product: Option<Amplitudes> = None;
        for f in &t.node.factors {
            let value = match &f.node {
                QFactor::Ket(k) => Amplitudes::basis(k.bits.chars().map(|c| c == '1').collect(), self.n),
                QFactor::Name(name) => (self.names)(*name, f.span)?,
                QFactor::Paren(inner) => self.state(inner)?,
            };
            product = Some(match product {
                None => value,
                Some(p) => p.tensor(&value),
            });
        }
        let Some(product) = product else {
            self.diags.push(
                Diagnostic::new(Code::Es06)
                    .with_message("this term is a coefficient with no state to scale")
                    .at(t.span)
                    .with_note("each term of a state is a coefficient times a tensor product of kets or states")
                    .with_help("multiply the coefficient by the ket it is the amplitude of, as in `isq2 * |0>`"),
            );
            return None;
        };
        Some(match coeff {
            Some(c) => product.scaled(&c),
            None => product,
        })
    }

    fn coeff(&mut self, e: &Spanned<QExpr>) -> Option<Cyclo> {
        let n = self.n;
        match &e.node {
            QExpr::Int(raw, base) => {
                let digits: String = self.interner.resolve(*raw).chars().filter(|&c| c != '_').collect();
                let v = BigInt::parse_bytes(digits.as_bytes(), base.radix())?;
                Some(Cyclo::from_frac(n, Frac::from_bigint(v)))
            }
            QExpr::Float(raw) => match Frac::parse_decimal(&self.interner.resolve(*raw).replace('_', "")) {
                Some(f) => Some(Cyclo::from_frac(n, f)),
                None => {
                    self.diags.push(
                        Diagnostic::new(Code::Es06)
                            .with_message("this number is not one a coefficient can be")
                            .at(e.span)
                            .with_note("a coefficient is exact, so a number in it must be a whole number or a decimal fraction")
                            .with_help("write the number as a fraction, as in `1/8`"),
                    );
                    None
                }
            },
            QExpr::Exact(Exact::I) => Cyclo::imaginary_unit(n),
            QExpr::Exact(Exact::Isq2) => Cyclo::isqrt2(n),
            QExpr::Exact(Exact::W { k, n: order }) => self.root(*k, *order, e.span),
            QExpr::Neg(x) => Some(self.coeff(x)?.neg()),
            QExpr::Binary { op, lhs, rhs } => {
                let a = self.coeff(lhs);
                let b = self.coeff(rhs);
                let (a, b) = (a?, b?);
                match op {
                    QOp::Add => Some(a.add(&b)),
                    QOp::Sub => Some(a.sub(&b)),
                    QOp::Mul => Some(a.mul(&b)),
                    QOp::Div => match a.div(&b) {
                        Some(q) => Some(q),
                        None => {
                            self.diags.push(
                                Diagnostic::new(Code::Ec04)
                                    .with_message("this coefficient divides by zero")
                                    .at_with(rhs.span, "this is zero")
                                    .with_note("a coefficient is a constant, computed exactly while the program is translated")
                                    .with_help("write the amplitude the term is meant to have"),
                            );
                            None
                        }
                    },
                }
            }
        }
    }

    /// `w(k, m)`, e^(2πik/m), at the conductor, reduced to its order first:
    /// `w(2, 16)` is `w(1, 8)`, which a conductor of 8 holds.
    fn root(&mut self, k: u64, m: u64, span: Span) -> Option<Cyclo> {
        if m == 0 {
            self.diags.push(
                Diagnostic::new(Code::Es06)
                    .with_message("`w(k, N)` is e^(2πik/N), for which `N` must be at least 1")
                    .at(span)
                    .with_note("`N` is the order of the root of unity; there is no root of order 0")
                    .with_help("write the order as a positive number, as in `w(1, 8)` for e^(2πi/8)"),
            );
            return None;
        }
        let k = k % m;
        let g = num_integer::gcd(k, m);
        let order = m / g;
        match u32::try_from(order).ok().filter(|&o| self.n.is_multiple_of(o)) {
            Some(o) => Some(Cyclo::zeta_pow(self.n, i64::try_from(k / g).ok()? * i64::from(self.n / o))),
            None => {
                self.diags.push(outside(span, order, self.n));
                None
            }
        }
    }
}

/// `EJ04`: a coefficient whose root of unity is of order `order`, which the
/// conductor `n` does not hold.
pub fn outside(span: Span, order: u64, n: u32) -> Diagnostic {
    let suffices = u32::try_from(order).ok().and_then(|o| crate::exact::conductor_admitting(o, n));
    let help = match suffices {
        Some(m) => format!("add `#pragma conductor({m})` to the unit"),
        None => "use roots of unity whose order divides the unit's conductor".to_owned(),
    };
    Diagnostic::new(Code::Ej04)
        .with_message(format!(
            "this coefficient needs conductor {order}, but the unit's conductor is {n}"
        ))
        .at_with(span, format!("a root of unity of order {order}"))
        .with_note(
            "a coefficient is an element of the field of the unit's conductor N, which holds the roots \
             of unity whose order divides N",
        )
        .with_help(help)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::Code;
    use crate::quon::parse::testing::state_in;

    fn eval(src: &str, n: u32) -> Result<Amplitudes, Vec<Code>> {
        let (s, interner) = state_in(src);
        evaluate(&s, n, &interner, &mut |_, _| None).map_err(|d| d.iter().map(|d| d.code).collect())
    }

    fn bits(s: &str) -> Vec<bool> {
        s.chars().map(|c| c == '1').collect()
    }

    #[test]
    fn a_bell_state_has_two_amplitudes_of_one_over_root_two() {
        let a = eval("isq2 * |00> + isq2 * |11>", 8).unwrap();
        assert_eq!(a.width, 2);
        assert_eq!(a.terms.len(), 2);
        assert_eq!(a.terms[&bits("11")], Cyclo::isqrt2(8).unwrap());
        assert!(a.is_normalized());
    }

    #[test]
    fn tensor_factors_are_placed_first_qubit_first() {
        let a = eval("|1> ** (isq2 * |0> - isq2 * |1>)", 8).unwrap();
        assert_eq!(a.width, 2);
        assert_eq!(a.terms[&bits("11")], Cyclo::isqrt2(8).unwrap().neg());
        assert!(!a.terms.contains_key(&bits("00")));
    }

    #[test]
    fn terms_that_cancel_leave_no_amplitude() {
        let a = eval("|0> + |1> - |1>", 8).unwrap();
        assert_eq!(a.terms, BTreeMap::from([(bits("0"), Cyclo::one(8))]));
    }

    #[test]
    fn a_state_that_is_not_a_unit_vector_is_seen_to_be_one_exactly() {
        assert!(!eval("|0> + |1>", 8).unwrap().is_normalized());
        assert!(eval("1/2 * |00> + 1/2 * |01> + 1/2 * |10> + 1/2 * |11>", 8).unwrap().is_normalized());
        assert!(eval("0.6 * |0> + 0.8 * i * |1>", 8).unwrap().is_normalized());
    }

    #[test]
    fn a_root_of_unity_is_reduced_before_the_conductor_is_asked() {
        assert!(eval("isq2 * |0> + w(2, 16) * isq2 * |1>", 8).is_ok());
        assert_eq!(eval("isq2 * |0> + w(1, 16) * isq2 * |1>", 8), Err(vec![Code::Ej04]));
        assert!(eval("isq2 * |0> + w(1, 16) * isq2 * |1>", 16).is_ok());
        assert_eq!(eval("w(1, 3) * |0>", 16), Err(vec![Code::Ej04]));
    }

    #[test]
    fn a_coefficient_that_divides_by_zero_is_refused() {
        assert_eq!(eval("1 / (1 - 1) * |0>", 8), Err(vec![Code::Ec04]));
    }

    #[test]
    fn a_parenthesized_coefficient_groups_like_one() {
        let a = eval("(1 + i) / 2 * |0> + (1 - i) / 2 * |1>", 8).unwrap();
        assert!(a.is_normalized());
        assert_eq!(eval("1/2 + |0>", 8), Err(vec![Code::Es06]));
    }

    #[test]
    fn terms_of_different_widths_are_refused() {
        assert_eq!(eval("|0> + |11>", 8), Err(vec![Code::Es06]));
    }
}
