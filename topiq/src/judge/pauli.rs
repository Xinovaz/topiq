//! Pauli strings, and the stabiliser codes their groups fix.
//!
//! A string on n qubits is a sign and one of I, X, Y, Z per qubit, the first
//! letter for the register's first qubit. As a vector over F₂ it is its X
//! part and its Z part (Y having both); two strings commute exactly when
//! the symplectic product of those vectors is zero.
//!
//! Generators that commute and are independent over F₂ fix a code of
//! dimension 2^(n − r): no product of them is −I, since the only product
//! with an empty X and Z part is the empty one.

use crate::exact::Cyclo;
use crate::quon::eval::Amplitudes;

/// A Hermitian Pauli string.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Pauli {
    /// Whether it carries the sign −1.
    pub negative: bool,
    /// Where an X or Y stands.
    pub x: Vec<bool>,
    /// Where a Z or Y stands.
    pub z: Vec<bool>,
}

impl Pauli {
    /// The string written as `letters`.
    pub fn parse(negative: bool, letters: &str) -> Option<Pauli> {
        let mut x = Vec::new();
        let mut z = Vec::new();
        for c in letters.chars() {
            let (a, b) = match c {
                'I' => (false, false),
                'X' => (true, false),
                'Y' => (true, true),
                'Z' => (false, true),
                _ => return None,
            };
            x.push(a);
            z.push(b);
        }
        Some(Pauli { negative, x, z })
    }

    /// How many qubits it acts on.
    pub fn width(&self) -> usize {
        self.x.len()
    }

    /// Whether it commutes with `other`.
    pub fn commutes(&self, other: &Pauli) -> bool {
        let mut odd = false;
        for q in 0..self.width() {
            odd ^= (self.x[q] && other.z[q]) ^ (self.z[q] && other.x[q]);
        }
        !odd
    }

    /// The string applied to `v`.
    pub fn apply(&self, v: &Amplitudes) -> Amplitudes {
        let n = v.n;
        let mut out = Amplitudes {
            width: v.width,
            n,
            terms: Default::default(),
        };
        for (bits, c) in &v.terms {
            // Z then X on each qubit: Y = i·X·Z
            let mut quarter_turns = if self.negative { 2 } else { 0 };
            let mut image = bits.clone();
            for q in 0..self.width() {
                if self.z[q] && bits[q] {
                    quarter_turns += 2;
                }
                if self.x[q] && self.z[q] {
                    quarter_turns += 1;
                }
                if self.x[q] {
                    image[q] = !image[q];
                }
            }
            let phase = Cyclo::zeta_pow(n, i64::from(quarter_turns % 4) * i64::from(n / 4));
            out.terms.insert(image, c.mul(&phase));
        }
        out
    }
}

impl std::fmt::Display for Pauli {
    /// Its sign, when negative, and its letters, as a code's generators are
    /// written: `-XZY`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.negative {
            write!(f, "-")?;
        }
        for (x, z) in self.x.iter().zip(&self.z) {
            let c = match (x, z) {
                (false, false) => 'I',
                (true, false) => 'X',
                (true, true) => 'Y',
                (false, true) => 'Z',
            };
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

/// Whether `gens` are independent over F₂: none is a product of others,
/// signs aside.
pub fn independent(gens: &[Pauli]) -> bool {
    let mut rows: Vec<Vec<bool>> = gens.iter().map(|g| g.x.iter().chain(&g.z).copied().collect()).collect();
    let cols = rows.first().map_or(0, Vec::len);
    super::stabilizer::reduce(&mut rows, cols).len() == gens.len()
}

/// Whether every generator fixes `v`.
pub fn fixes(gens: &[Pauli], v: &Amplitudes) -> bool {
    gens.iter().all(|g| g.apply(v) == *v)
}

/// A basis of the code `gens` fix, on `width` qubits at conductor `n`; the
/// generators commute and are independent.
pub fn code_basis(gens: &[Pauli], width: usize, n: u32) -> Vec<Amplitudes> {
    let dim = 1usize << (width - gens.len());
    let half = Cyclo::from_frac(n, crate::exact::Frac::new(1, 2));
    let mut out: Vec<Amplitudes> = Vec::new();
    for i in 0..(1usize << width) {
        if out.len() == dim {
            break;
        }
        // the projector Π (I + g)/2 applied to the basis state
        let mut v = Amplitudes::basis(crate::quon::eval::ket_bits(i, width), n);
        for g in gens {
            let gv = g.apply(&v);
            v = super::linalg::combine(&[v.clone(), gv], &[half.clone(), half.clone()], width, n);
        }
        if !v.terms.is_empty() && !super::linalg::in_span(&v, &out) {
            out.push(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Pauli {
        match s.strip_prefix('-') {
            Some(rest) => Pauli::parse(true, rest).unwrap(),
            None => Pauli::parse(false, s).unwrap(),
        }
    }

    #[test]
    fn strings_commute_where_their_symplectic_product_is_zero() {
        assert!(p("XX").commutes(&p("ZZ")));
        assert!(!p("XI").commutes(&p("ZI")));
        assert!(p("XZ").commutes(&p("ZX")));
        assert!(independent(&[p("XX"), p("ZZ")]));
        assert!(!independent(&[p("XX"), p("ZZ"), p("-YY")]));
        assert!(!independent(&[p("II")]));
    }

    #[test]
    fn the_bell_code_is_the_bell_state() {
        let b = code_basis(&[p("XX"), p("ZZ")], 2, 8);
        assert_eq!(b.len(), 1);
        let bits = |s: &str| s.chars().map(|c| c == '1').collect::<Vec<_>>();
        assert_eq!(b[0].terms.len(), 2);
        assert!(b[0].terms.contains_key(&bits("00")) && b[0].terms.contains_key(&bits("11")));
        assert!(fixes(&[p("XX"), p("ZZ")], &b[0]));
        assert!(!fixes(&[p("-ZZ")], &b[0]));
    }

    #[test]
    fn a_code_of_fewer_generators_is_a_subspace() {
        assert_eq!(code_basis(&[p("ZZ")], 2, 8).len(), 2);
        assert_eq!(code_basis(&[p("ZZI"), p("IZZ")], 3, 8).len(), 2);
        // Y acts with its phase: Y|0> = i|1>
        let y = p("Y").apply(&Amplitudes::basis(vec![false], 8));
        assert_eq!(y.terms[&vec![true]], Cyclo::imaginary_unit(8).unwrap());
    }
}
