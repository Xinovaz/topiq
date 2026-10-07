//! Exact linear algebra over K = `Q(ζ_N)` on the vectors covers are made of.
//!
//! A vector is an [`Amplitudes`]: the non-zero entries of a state over the
//! computational basis. Everything here is decided exactly (a zero test,
//! a comparison of coefficients) with no tolerance anywhere.

use crate::exact::Cyclo;
use crate::quon::eval::Amplitudes;

/// The inner product ⟨a|b⟩: over the basis, the sum of each entry of `a`
/// conjugated times the entry of `b`.
pub fn inner(a: &Amplitudes, b: &Amplitudes) -> Cyclo {
    let mut acc = Cyclo::zero(a.n);
    for (k, x) in &a.terms {
        if let Some(y) = b.terms.get(k) {
            acc = acc.add(&x.conj().mul(y));
        }
    }
    acc
}

/// Whether `a` and `b` are orthogonal.
pub fn orthogonal(a: &Amplitudes, b: &Amplitudes) -> bool {
    inner(a, b).is_zero()
}

/// Whether `a` and `b` name one projective point: `a = λ·b` for some
/// non-zero λ.
pub fn same_point(a: &Amplitudes, b: &Amplitudes) -> bool {
    if a.width != b.width || a.terms.len() != b.terms.len() {
        return false;
    }
    let Some((k, x)) = a.terms.iter().next() else {
        return b.terms.is_empty();
    };
    let Some(y) = b.terms.get(k) else { return false };
    // a_i·y = b_i·x for every i, which is a = (x/y)·b without dividing
    a.terms.iter().all(|(i, ai)| b.terms.get(i).is_some_and(|bi| ai.mul(y) == bi.mul(x)))
}

/// The linear combination of the vectors `vs`, of one width, with the
/// coefficients `cs`.
pub fn combine(vs: &[Amplitudes], cs: &[Cyclo], width: usize, n: u32) -> Amplitudes {
    let mut out = Amplitudes {
        width,
        n,
        terms: Default::default(),
    };
    for (v, c) in vs.iter().zip(cs) {
        if c.is_zero() {
            continue;
        }
        for (k, x) in &v.terms {
            let sum = match out.terms.get(k) {
                Some(y) => y.add(&c.mul(x)),
                None => c.mul(x),
            };
            if sum.is_zero() {
                out.terms.remove(k);
            } else {
                out.terms.insert(k.clone(), sum);
            }
        }
    }
    out
}

/// A matrix over K in reduced row echelon form, and where its pivots are.
struct Echelon {
    rows: Vec<Vec<Cyclo>>,
    pivots: Vec<usize>,
}

/// Row-reduces `m`, of `cols` columns.
fn echelon(mut rows: Vec<Vec<Cyclo>>, cols: usize) -> Echelon {
    let mut pivots = Vec::new();
    let mut top = 0;
    for col in 0..cols {
        let Some(p) = (top..rows.len()).find(|&r| !rows[r][col].is_zero()) else { continue };
        rows.swap(top, p);
        let lead = rows[top][col].inv().expect("the pivot is not zero");
        for x in &mut rows[top] {
            *x = x.mul(&lead);
        }
        let pivot = rows[top].clone();
        for (r, row) in rows.iter_mut().enumerate() {
            if r != top && !row[col].is_zero() {
                let f = row[col].clone();
                for (x, p) in row.iter_mut().zip(&pivot) {
                    *x = x.sub(&p.mul(&f));
                }
            }
        }
        pivots.push(col);
        top += 1;
    }
    rows.truncate(top);
    Echelon { rows, pivots }
}

/// The rank of the matrix whose rows are `rows`.
pub fn rank(rows: Vec<Vec<Cyclo>>, cols: usize) -> usize {
    echelon(rows, cols).pivots.len()
}

/// A non-zero x with M·x = 0, for M with rows `rows` of `cols` entries, if
/// there is one; `n` is the conductor the entries are at.
pub fn null_vector(rows: Vec<Vec<Cyclo>>, cols: usize, n: u32) -> Option<Vec<Cyclo>> {
    null_space(rows, cols, n).into_iter().next()
}

/// A basis of the x with M·x = 0.
pub fn null_space(rows: Vec<Vec<Cyclo>>, cols: usize, n: u32) -> Vec<Vec<Cyclo>> {
    let e = echelon(rows, cols);
    (0..cols)
        .filter(|c| !e.pivots.contains(c))
        .map(|free| {
            let mut x = vec![Cyclo::zero(n); cols];
            x[free] = Cyclo::one(n);
            for (row, &p) in e.rows.iter().zip(&e.pivots) {
                x[p] = row[free].neg();
            }
            x
        })
        .collect()
}

/// A basis of the common part of the spans of `a` and `b`, each a set of
/// independent vectors.
pub fn intersection(a: &[Amplitudes], b: &[Amplitudes]) -> Vec<Amplitudes> {
    let Some(first) = a.first() else { return Vec::new() };
    let (width, n) = (first.width, first.n);
    let found: Vec<Amplitudes> = null_space(meeting(a, b), a.len() + b.len(), n)
        .into_iter()
        .map(|c| combine(a, &c[..a.len()], width, n))
        .filter(|v| !v.terms.is_empty())
        .collect();
    basis(&found)
}

/// The Gram matrix of `vs`, the inner product of each pair, whose rank is
/// the dimension of their span.
pub fn gram(vs: &[Amplitudes]) -> Vec<Vec<Cyclo>> {
    vs.iter().map(|a| vs.iter().map(|b| inner(a, b)).collect()).collect()
}

/// The dimension of the span of `vs`.
pub fn span_dimension(vs: &[Amplitudes]) -> usize {
    rank(gram(vs), vs.len())
}

/// A basis of the span of `vs`: those of them independent of the ones
/// before.
pub fn basis(vs: &[Amplitudes]) -> Vec<Amplitudes> {
    let mut out: Vec<Amplitudes> = Vec::new();
    for v in vs {
        if !in_span(v, &out) {
            out.push(v.clone());
        }
    }
    out
}

/// Whether `v` lies in the span of `basis`, a set of independent vectors.
pub fn in_span(v: &Amplitudes, basis: &[Amplitudes]) -> bool {
    let mut all = basis.to_vec();
    all.push(v.clone());
    span_dimension(&all) == basis.len()
}

/// A non-zero vector of the span of `basis` orthogonal to every one of
/// `against`, if there is one.
pub fn orthogonal_member(basis: &[Amplitudes], against: &[Amplitudes]) -> Option<Amplitudes> {
    let first = basis.first()?;
    let (width, n) = (first.width, first.n);
    // Σ c_i·⟨w|b_i⟩ = 0 for every w
    let rows: Vec<Vec<Cyclo>> = against.iter().map(|w| basis.iter().map(|b| inner(w, b)).collect()).collect();
    let rows = if rows.is_empty() { vec![vec![Cyclo::zero(n); basis.len()]] } else { rows };
    let c = null_vector(rows, basis.len(), n)?;
    Some(combine(basis, &c, width, n))
}

/// A non-zero vector in the span of both `a` and `b`, each a set of
/// independent vectors, if the spans meet.
pub fn common_member(a: &[Amplitudes], b: &[Amplitudes]) -> Option<Amplitudes> {
    let first = a.first()?;
    let (width, n) = (first.width, first.n);
    // the a_i are independent, so a non-zero solution has a non-zero x unless
    // the b_j were dependent, which they are not
    let c = null_vector(meeting(a, b), a.len() + b.len(), n)?;
    let v = combine(a, &c[..a.len()], width, n);
    (!v.terms.is_empty()).then_some(v)
}

/// The system `Σ x_i·a_i − Σ y_j·b_j = 0`, whose solutions say where the
/// spans of `a` and `b` meet, read through the inner products with every
/// vector of both, which have the same kernel.
fn meeting(a: &[Amplitudes], b: &[Amplitudes]) -> Vec<Vec<Cyclo>> {
    a.iter()
        .chain(b)
        .map(|w| a.iter().map(|v| inner(w, v)).chain(b.iter().map(|v| inner(w, v).neg())).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn ket(bits: &str, n: u32) -> Amplitudes {
        Amplitudes::basis(bits.chars().map(|c| c == '1').collect(), n)
    }

    fn plus(n: u32) -> Amplitudes {
        let s = Cyclo::isqrt2(n).unwrap();
        Amplitudes {
            width: 1,
            n,
            terms: BTreeMap::from([(vec![false], s.clone()), (vec![true], s)]),
        }
    }

    #[test]
    fn inner_products_and_points() {
        assert!(orthogonal(&ket("0", 8), &ket("1", 8)));
        assert_eq!(inner(&plus(8), &ket("1", 8)), Cyclo::isqrt2(8).unwrap());
        let mut i_plus = plus(8);
        for c in i_plus.terms.values_mut() {
            *c = c.mul(&Cyclo::imaginary_unit(8).unwrap());
        }
        assert!(same_point(&plus(8), &i_plus));
        assert!(!same_point(&plus(8), &ket("0", 8)));
    }

    #[test]
    fn a_span_of_basis_kets_and_what_is_orthogonal_to_it() {
        let kets = [ket("00", 8), ket("01", 8), ket("10", 8)];
        assert_eq!(span_dimension(&kets), 3);
        // nothing in the span is orthogonal to all three
        assert!(orthogonal_member(&kets, &kets).is_none());
        // something is orthogonal to two of them
        let v = orthogonal_member(&kets, &kets[..2]).unwrap();
        assert!(same_point(&v, &ket("10", 8)));
    }

    #[test]
    fn spans_meet_where_they_share_a_direction() {
        let a = [ket("00", 8), ket("01", 8)];
        let b = [ket("01", 8), ket("11", 8)];
        assert!(same_point(&common_member(&a, &b).unwrap(), &ket("01", 8)));
        assert!(common_member(&a, &[ket("11", 8)]).is_none());
    }
}
