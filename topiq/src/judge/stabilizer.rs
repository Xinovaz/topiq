//! The stabiliser path: judging a Clifford circuit over a code cover in time
//! polynomial in the code's width.
//!
//! A state a Clifford circuit makes from a code state is a stabiliser state.
//! It is held here as the group that fixes it (a generator per qubit, each
//! a Pauli operator with its phase), together with one basis state it has
//! weight on and its exact amplitude there (`State`). A gate conjugates
//! each generator, and carries the amplitude by acting on the basis states
//! that differ from the one held only on the gate's qubits. Every other
//! amplitude follows from the group: the element whose bit flips carry the
//! held basis state to another gives the ratio of their amplitudes, a power
//! of i.
//!
//! What an operator does to a code of dimension 2^m is read off one state:
//! the code entangled with m reference qubits, fixed by the code's
//! generators and by each logical operator of the code paired with one on
//! the reference (`logicals`). The operator is a scalar on the code
//! exactly when it multiplies that state by one, and carries the code into
//! a target code exactly when each of the target's generators fixes what it
//! makes of that state. A code of one point is its own such state.
//!
//! This gives the certificate the exact action of [`super::sim`] gives, at
//! a cost polynomial in the width rather than exponential.

use super::cert::{Base, Certificate, Fail};
use super::cover::Cover;
use super::pauli::Pauli;
use crate::circuit::ir::{Angle, Circuit, GateOp, Op};
use crate::exact::Cyclo;
use crate::quon::eval::Amplitudes;

/// A Pauli operator with its phase: `i^e` times, on each qubit, X to the
/// power `x` and then Z to the power `z`.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Phased {
    x: Vec<bool>,
    z: Vec<bool>,
    e: u8,
}

impl Phased {
    fn identity(width: usize) -> Phased {
        Phased {
            x: vec![false; width],
            z: vec![false; width],
            e: 0,
        }
    }

    /// The Hermitian operator with parts `x` and `z`, and the sign −1 when
    /// `negative`: a Y is i·X·Z.
    fn hermitian(x: Vec<bool>, z: Vec<bool>, negative: bool) -> Phased {
        let ys = dot(&x, &z);
        let e = ((ys + if negative { 2 } else { 0 }) % 4) as u8;
        Phased { x, z, e }
    }

    /// `p`, its qubits placed at `at` among `width`.
    fn placed(p: &Pauli, at: &[usize], width: usize) -> Phased {
        Phased::hermitian(spread(&p.x, at, width), spread(&p.z, at, width), p.negative)
    }

    /// Z on qubit `w` of `width`.
    fn z_on(w: usize, width: usize) -> Phased {
        let mut z = vec![false; width];
        z[w] = true;
        Phased::hermitian(vec![false; width], z, false)
    }

    /// The power of i by which it carries the basis state `x` to the one its
    /// X part flips `x` to: `P|x⟩ = i^e (−1)^(z·x) |x ⊕ X⟩`.
    fn quarter_on(&self, x: &[bool]) -> u8 {
        ((usize::from(self.e) + 2 * dot(&self.z, x)) % 4) as u8
    }

    /// The product `self · o`.
    fn mul(&self, o: &Phased) -> Phased {
        // Z^b X^a' = (−1)^(b·a') X^a' Z^b, qubit by qubit
        let swaps = dot(&self.z, &o.x);
        Phased {
            x: self.x.iter().zip(&o.x).map(|(a, b)| a ^ b).collect(),
            z: self.z.iter().zip(&o.z).map(|(a, b)| a ^ b).collect(),
            e: ((usize::from(self.e) + usize::from(o.e) + 2 * swaps) % 4) as u8,
        }
    }

    fn commutes(&self, o: &Phased) -> bool {
        let odd = (0..self.x.len()).filter(|&q| (self.x[q] && o.z[q]) ^ (self.z[q] && o.x[q])).count();
        odd % 2 == 0
    }

    /// The operator `g · self · g†`.
    fn conjugate(&mut self, g: Step) {
        let flip = |e: &mut u8, by: u8| *e = (*e + by) % 4;
        match g {
            Step::X(q) => {
                if self.z[q] {
                    flip(&mut self.e, 2);
                }
            }
            Step::Z(q) => {
                if self.x[q] {
                    flip(&mut self.e, 2);
                }
            }
            Step::Y(q) => {
                if self.x[q] ^ self.z[q] {
                    flip(&mut self.e, 2);
                }
            }
            Step::H(q) => {
                if self.x[q] && self.z[q] {
                    flip(&mut self.e, 2);
                }
                std::mem::swap(&mut self.x[q], &mut self.z[q]);
            }
            Step::S(q) => {
                // X ↦ iXZ, Z ↦ Z
                if self.x[q] {
                    flip(&mut self.e, 1);
                    self.z[q] = !self.z[q];
                }
            }
            Step::Sdg(q) => {
                if self.x[q] {
                    flip(&mut self.e, 3);
                    self.z[q] = !self.z[q];
                }
            }
            Step::Cx(c, t) => {
                // X_c ↦ X_c X_t, Z_t ↦ Z_c Z_t
                self.x[t] ^= self.x[c];
                self.z[c] ^= self.z[t];
            }
            Step::Cz(c, t) => {
                // X_c ↦ X_c Z_t, X_t ↦ Z_c X_t; putting Z_t after X_t again
                // costs a sign when both X parts are there
                if self.x[c] && self.x[t] {
                    flip(&mut self.e, 2);
                }
                self.z[c] ^= self.x[t];
                self.z[t] ^= self.x[c];
            }
            Step::Swap(a, b) => {
                self.x.swap(a, b);
                self.z.swap(a, b);
            }
        }
    }
}

/// A Clifford gate.
#[derive(Clone, Copy, Debug)]
enum Step {
    X(usize),
    Y(usize),
    Z(usize),
    H(usize),
    S(usize),
    Sdg(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
}

/// A stabiliser state: the group fixing it, a basis state `x0` it has
/// weight on, and its amplitude there.
#[derive(Clone, Debug)]
struct State {
    gens: Vec<Phased>,
    x0: Vec<bool>,
    amp: Cyclo,
    n: u32,
}

/// The rows of `rows` reduced so that each of the first `pivots.len()` has
/// an X part with a one where no other has: each pivot's column and row.
fn x_echelon(rows: &mut [Phased]) -> Vec<(usize, usize)> {
    let width = rows.first().map_or(0, |r| r.x.len());
    let mut pivots = Vec::new();
    let mut used = vec![false; rows.len()];
    for col in 0..width {
        let Some(r) = (0..rows.len()).find(|&r| !used[r] && rows[r].x[col]) else { continue };
        used[r] = true;
        let pivot = rows[r].clone();
        for (o, row) in rows.iter_mut().enumerate() {
            if o != r && row.x[col] {
                *row = row.mul(&pivot);
            }
        }
        pivots.push((col, r));
    }
    pivots
}

/// The rows of `m` reduced over F₂ on their first `width` columns, in
/// reduced echelon form: the columns of the pivots, the pivot of the r-th
/// being row r's.
pub(super) fn reduce(m: &mut [Vec<bool>], width: usize) -> Vec<usize> {
    let mut pivots = Vec::new();
    for col in 0..width {
        let top = pivots.len();
        let Some(p) = (top..m.len()).find(|&r| m[r][col]) else { continue };
        m.swap(top, p);
        let pivot = m[top].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r != top && row[col] {
                xor_into(row, &pivot);
            }
        }
        pivots.push(col);
    }
    pivots
}

/// A solution of the equations `a·x = b` over F₂, if they have one.
fn solve(equations: &[(Vec<bool>, bool)], width: usize) -> Option<Vec<bool>> {
    // each equation as one row, `b` after `a`
    let mut rows: Vec<Vec<bool>> = equations.iter().map(|(a, b)| a.iter().copied().chain([*b]).collect()).collect();
    let pivots = reduce(&mut rows, width);
    if rows[pivots.len()..].iter().any(|r| r[width]) {
        return None;
    }
    let mut x = vec![false; width];
    for (r, &col) in pivots.iter().enumerate() {
        x[col] = rows[r][width];
    }
    Some(x)
}

/// `a ^= b`, bit by bit.
fn xor_into(a: &mut [bool], b: &[bool]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= *y;
    }
}

/// How many positions `a` and `b` both have set.
fn dot(a: &[bool], b: &[bool]) -> usize {
    a.iter().zip(b).filter(|&(a, b)| *a && *b).count()
}

/// `bits`, bit q placed at `at[q]`, among `width` otherwise unset.
fn spread(bits: &[bool], at: &[usize], width: usize) -> Vec<bool> {
    let mut out = vec![false; width];
    for (&b, &w) in bits.iter().zip(at) {
        out[w] = b;
    }
    out
}

/// The product of the pivot rows whose X parts together make `flips`, if
/// they do: `rows` in X echelon form, with its `pivots`.
fn flipping(rows: &[Phased], pivots: &[(usize, usize)], mut flips: Vec<bool>) -> Option<Phased> {
    let mut p = Phased::identity(flips.len());
    for &(col, r) in pivots {
        if flips[col] {
            p = p.mul(&rows[r]);
            xor_into(&mut flips, &rows[r].x);
        }
    }
    (!flips.contains(&true)).then_some(p)
}

impl State {
    /// The state `gens` fix, which must be as many as its qubits, commuting
    /// and independent, given the amplitude one where it first has weight.
    fn new(gens: Vec<Phased>, n: u32) -> Option<State> {
        let width = gens.len();
        let mut rows = gens.clone();
        let pivots = x_echelon(&mut rows);
        // each element with no X part is ±Z^z, and fixes the basis states
        // on which z has the parity of its sign
        let pivot_rows: Vec<usize> = pivots.iter().map(|&(_, r)| r).collect();
        let equations: Vec<(Vec<bool>, bool)> = rows
            .iter()
            .enumerate()
            .filter(|(r, _)| !pivot_rows.contains(r))
            .map(|(_, row)| (row.z.clone(), row.e == 2))
            .collect();
        let x0 = solve(&equations, width)?;
        Some(State {
            gens,
            x0,
            amp: Cyclo::one(n),
            n,
        })
    }

    fn width(&self) -> usize {
        self.x0.len()
    }

    /// The power of i the amplitude at `y` is of the one at `x0`; `None`
    /// where the state has no weight.
    fn quarter(&self, y: &[bool]) -> Option<u8> {
        let mut rows = self.gens.clone();
        let pivots = x_echelon(&mut rows);
        let p = flipping(&rows, &pivots, y.iter().zip(&self.x0).map(|(a, b)| a ^ b).collect())?;
        // ψ(y) = ⟨y|P|ψ⟩ = ⟨y|P|x0⟩ ψ(x0)
        Some(p.quarter_on(&self.x0))
    }

    /// The amplitude at `y`.
    fn amplitude(&self, y: &[bool]) -> Cyclo {
        match self.quarter(y) {
            Some(k) => self.amp.mul_zeta_pow(i64::from(k) * i64::from(self.n / 4)),
            None => Cyclo::zero(self.n),
        }
    }

    /// Whether `p` fixes the state.
    fn fixes(&self, p: &Phased) -> bool {
        if !self.gens.iter().all(|g| g.commutes(p)) {
            return false;
        }
        // then ±p is in the group, and its sign is read off the amplitude
        // p carries x0's to
        let y: Vec<bool> = self.x0.iter().zip(&p.x).map(|(a, b)| a ^ b).collect();
        self.quarter(&y) == Some(p.quarter_on(&self.x0))
    }

    /// How many basis states it has weight on, as a power of two.
    fn support(&self) -> usize {
        let mut rows = self.gens.clone();
        x_echelon(&mut rows).len()
    }

    /// Applies the gate `g`.
    fn step(&mut self, g: Step) -> Option<()> {
        let i = |k: i64| k * i64::from(self.n / 4);
        match g {
            Step::X(q) => self.x0[q] = !self.x0[q],
            Step::Cx(c, t) => {
                if self.x0[c] {
                    self.x0[t] = !self.x0[t];
                }
            }
            Step::Y(q) => {
                // Y|0> = i|1>, Y|1> = −i|0>
                self.amp = self.amp.mul_zeta_pow(i(if self.x0[q] { 3 } else { 1 }));
                self.x0[q] = !self.x0[q];
            }
            Step::Z(q) => {
                if self.x0[q] {
                    self.amp = self.amp.neg();
                }
            }
            Step::S(q) | Step::Sdg(q) => {
                if self.x0[q] {
                    self.amp = self.amp.mul_zeta_pow(i(if matches!(g, Step::S(_)) { 1 } else { 3 }));
                }
            }
            Step::Cz(c, t) => {
                if self.x0[c] && self.x0[t] {
                    self.amp = self.amp.neg();
                }
            }
            Step::Swap(a, b) => self.x0.swap(a, b),
            Step::H(q) => {
                // (Hψ)(y) = (ψ(y with q = 0) ± ψ(y with q = 1)) / √2, the
                // sign that of y's bit q; y is x0 or x0 with q flipped
                let mut other = self.x0.clone();
                other[q] = !other[q];
                let here = self.amp.clone();
                let there = self.amplitude(&other);
                let (zero, one) = if self.x0[q] { (there, here) } else { (here, there) };
                let r = Cyclo::isqrt2(self.n)?;
                let same = if self.x0[q] { zero.sub(&one) } else { zero.add(&one) };
                if same.is_zero() {
                    let flipped = if self.x0[q] { zero.add(&one) } else { zero.sub(&one) };
                    self.amp = flipped.mul(&r);
                    self.x0 = other;
                } else {
                    self.amp = same.mul(&r);
                }
            }
        }
        for generator in &mut self.gens {
            generator.conjugate(g);
        }
        Some(())
    }

    /// Runs the gates of `ops` on the qubits `at` names for each wire.
    fn run(&mut self, ops: &[Op]) -> Option<()> {
        for op in ops {
            match op {
                Op::Gate { gate, targets, controls } => {
                    let t: Vec<usize> = targets.iter().map(|w| w.0 as usize).collect();
                    let negated: Vec<usize> = controls.iter().filter(|c| !c.on).map(|c| c.wire.0 as usize).collect();
                    for &c in &negated {
                        self.step(Step::X(c))?;
                    }
                    let control = controls.first().map(|c| c.wire.0 as usize);
                    match (gate, control) {
                        (GateOp::X, None) => self.step(Step::X(t[0]))?,
                        (GateOp::X, Some(c)) => self.step(Step::Cx(c, t[0]))?,
                        (GateOp::Z, None) => self.step(Step::Z(t[0]))?,
                        (GateOp::Z, Some(c)) => self.step(Step::Cz(c, t[0]))?,
                        (GateOp::Y, None) => self.step(Step::Y(t[0]))?,
                        (GateOp::H, None) => self.step(Step::H(t[0]))?,
                        (GateOp::S, None) => self.step(Step::S(t[0]))?,
                        (GateOp::Sdg, None) => self.step(Step::Sdg(t[0]))?,
                        (GateOp::Swap, None) => self.step(Step::Swap(t[0], t[1]))?,
                        (GateOp::Phase(Angle::Fixed(p)), None) => {
                            let quarters = u64::from(p.index()) * 4 / u64::from(p.order());
                            for _ in 0..quarters {
                                self.step(Step::S(t[0]))?;
                            }
                        }
                        (GateOp::GPhase(Angle::Fixed(p)), None) => {
                            let k = p.embed(self.n)?.index();
                            self.amp = self.amp.mul_zeta_pow(i64::from(k));
                        }
                        _ => return None,
                    }
                    for &c in &negated {
                        self.step(Step::X(c))?;
                    }
                }
                Op::Alloc { .. } | Op::Release { .. } => {}
                _ => return None,
            }
        }
        Some(())
    }

    /// The state on `total` qubits: qubit q of this one at `layout[q]`, and
    /// every other in |0>.
    fn embed(&self, layout: &[usize], total: usize) -> State {
        let place = |bits: &[bool]| spread(bits, layout, total);
        let mut gens: Vec<Phased> = self
            .gens
            .iter()
            .map(|g| Phased {
                x: place(&g.x),
                z: place(&g.z),
                e: g.e,
            })
            .collect();
        gens.extend((0..total).filter(|w| !layout.contains(w)).map(|w| Phased::z_on(w, total)));
        State {
            gens,
            x0: place(&self.x0),
            amp: self.amp.clone(),
            n: self.n,
        }
    }
}

/// The logical operators of the code `gens` fix on `width` qubits, in
/// pairs, one of X's part and one of Z's: each pair anticommutes, every
/// other two operators commute, and each commutes with the code's
/// generators and is no product of them.
fn logicals(gens: &[Pauli], width: usize) -> Vec<(Vec<bool>, Vec<bool>)> {
    // a vector (x | z) of 2·width bits; the symplectic form
    let form = |a: &[bool], b: &[bool]| {
        let mut odd = false;
        for q in 0..width {
            odd ^= (a[q] && b[width + q]) ^ (a[width + q] && b[q]);
        }
        odd
    };
    let vec_of = |p: &Pauli| -> Vec<bool> { p.x.iter().chain(&p.z).copied().collect() };
    let stab: Vec<Vec<bool>> = gens.iter().map(vec_of).collect();
    // the operators commuting with every generator: the kernel of the form
    // against them, found by reducing the system whose rows are (z | x)
    let equations: Vec<Vec<bool>> = stab.iter().map(|s| s[width..].iter().chain(&s[..width]).copied().collect()).collect();
    let kernel = nullspace(&equations, 2 * width);
    // those that are not products of the generators, completed from them
    let mut span = Span::default();
    for s in &stab {
        span.insert(s.clone());
    }
    let mut rest: Vec<Vec<bool>> = kernel.into_iter().filter(|v| span.insert(v.clone())).collect();
    // symplectic Gram–Schmidt
    let mut pairs = Vec::new();
    while let Some(a) = rest.pop() {
        let Some(j) = rest.iter().position(|b| form(&a, b)) else { continue };
        let b = rest.remove(j);
        for c in &mut rest {
            let (ca, cb) = (form(c, &a), form(c, &b));
            if cb {
                xor_into(c, &a);
            }
            if ca {
                xor_into(c, &b);
            }
        }
        pairs.push((a, b));
    }
    pairs
}

/// The vectors a set of vectors spans, kept reduced so that a new one can
/// be tested and added.
#[derive(Default)]
struct Span {
    rows: Vec<(usize, Vec<bool>)>,
}

impl Span {
    /// Adds `v`, giving whether it was not in the span already.
    fn insert(&mut self, mut v: Vec<bool>) -> bool {
        for (col, row) in &self.rows {
            if v[*col] {
                xor_into(&mut v, row);
            }
        }
        let Some(col) = v.iter().position(|&b| b) else { return false };
        for (_, row) in &mut self.rows {
            if row[col] {
                xor_into(row, &v);
            }
        }
        self.rows.push((col, v));
        true
    }
}

/// A basis of the vectors `v` of `width` bits with `r·v = 0` for each row.
fn nullspace(rows: &[Vec<bool>], width: usize) -> Vec<Vec<bool>> {
    let mut m: Vec<Vec<bool>> = rows.to_vec();
    let pivots = reduce(&mut m, width);
    let free: Vec<usize> = (0..width).filter(|c| !pivots.contains(c)).collect();
    free.iter()
        .map(|&f| {
            let mut v = vec![false; width];
            v[f] = true;
            for (r, &col) in pivots.iter().enumerate() {
                v[col] = m[r][f];
            }
            v
        })
        .collect()
}

/// The generators of the code `gens` fixes on `width` qubits, entangled
/// with one reference qubit per logical pair, which follow the code's.
fn entangled(gens: &[Pauli], width: usize) -> Vec<Phased> {
    let pairs = logicals(gens, width);
    let total = width + pairs.len();
    let mut out: Vec<Phased> = gens.iter().map(|g| Phased::placed(g, &(0..width).collect::<Vec<_>>(), total)).collect();
    for (j, (a, b)) in pairs.iter().enumerate() {
        for (v, reference_x) in [(a, true), (b, false)] {
            let mut x: Vec<bool> = v[..width].to_vec();
            let mut z: Vec<bool> = v[width..].to_vec();
            x.resize(total, false);
            z.resize(total, false);
            if reference_x {
                x[width + j] = true;
            } else {
                z[width + j] = true;
            }
            out.push(Phased::hermitian(x, z, false));
        }
    }
    out
}

/// The Hermitian operator `p` is.
fn string_of(p: &Phased) -> Pauli {
    let ys = dot(&p.x, &p.z);
    Pauli {
        negative: (usize::from(p.e) + 4 - ys % 4) % 4 == 2,
        x: p.x.clone(),
        z: p.z.clone(),
    }
}

/// Independent generators of the code both `a` and `b` fix, on the same
/// qubits; `None` when they fix no state in common: when two of their
/// elements anticommute, or their product is −I.
pub fn joint(a: &[Pauli], b: &[Pauli]) -> Option<Vec<Pauli>> {
    let width = a.first().or(b.first()).map_or(0, Pauli::width);
    let at: Vec<usize> = (0..width).collect();
    let all: Vec<Phased> = a.iter().chain(b).map(|p| Phased::placed(p, &at, width)).collect();
    for (i, p) in all.iter().enumerate() {
        if all[i + 1..].iter().any(|q| !p.commutes(q)) {
            return None;
        }
    }
    // reduce over the X and Z parts together; a row that vanishes is a
    // product of the others, or −I
    let mut rows = all;
    let mut kept = Vec::new();
    let mut used = vec![false; rows.len()];
    for col in 0..2 * width {
        let bit = |r: &Phased| if col < width { r.x[col] } else { r.z[col - width] };
        let Some(r) = (0..rows.len()).find(|&r| !used[r] && bit(&rows[r])) else { continue };
        used[r] = true;
        let pivot = rows[r].clone();
        for (o, row) in rows.iter_mut().enumerate() {
            if o != r && !used[o] && bit(row) {
                *row = row.mul(&pivot);
            }
        }
        kept.push(r);
    }
    if rows.iter().enumerate().any(|(r, row)| !used[r] && row.e == 2) {
        return None;
    }
    Some(kept.into_iter().map(|r| string_of(&rows[r])).collect())
}

/// The projector onto the code `gens` fix, as its entries between basis
/// states: the average of the group's elements, of which those carrying
/// one basis state to another make a coset of the ones that flip no bit.
pub struct Projector {
    rows: Vec<Phased>,
    pivots: Vec<(usize, usize)>,
    n: u32,
}

impl Projector {
    /// The projector onto the code `gens` fix.
    pub fn new(gens: &[Pauli], n: u32) -> Projector {
        let width = gens.first().map_or(0, Pauli::width);
        let at: Vec<usize> = (0..width).collect();
        let mut rows: Vec<Phased> = gens.iter().map(|p| Phased::placed(p, &at, width)).collect();
        let pivots = x_echelon(&mut rows);
        Projector { rows, pivots, n }
    }

    /// ⟨x|P|y⟩.
    pub fn entry(&self, x: &[bool], y: &[bool]) -> Cyclo {
        let zero = Cyclo::zero(self.n);
        // every element that flips no bit must fix y, or the coset sums to
        // nothing
        let pivot_rows: Vec<usize> = self.pivots.iter().map(|&(_, r)| r).collect();
        let fixes_y = |(r, row): (usize, &Phased)| pivot_rows.contains(&r) || row.quarter_on(y) == 0;
        if !self.rows.iter().enumerate().all(fixes_y) {
            return zero;
        }
        let Some(g) = flipping(&self.rows, &self.pivots, x.iter().zip(y).map(|(a, b)| a ^ b).collect()) else {
            return zero;
        };
        // the coset's elements each give ⟨x|g|y⟩, and there are 2^(r − k)
        // of them among the 2^r the average is over, k the flips' rank
        let quarter = g.quarter_on(y);
        let scale = Cyclo::from_frac(self.n, crate::exact::Frac::new(1, 1i64 << self.pivots.len().min(62)));
        scale.mul_zeta_pow(quarter as i64 * i64::from(self.n / 4))
    }

    /// ⟨a|P|b⟩.
    pub fn between(&self, a: &Amplitudes, b: &Amplitudes) -> Cyclo {
        let mut sum = Cyclo::zero(self.n);
        for (x, ax) in &a.terms {
            for (y, by) in &b.terms {
                let e = self.entry(x, y);
                if !e.is_zero() {
                    sum = sum.add(&ax.conj().mul(by).mul(&e));
                }
            }
        }
        sum
    }

    /// A basis of the part of the span of `basis`, independent vectors,
    /// that lies in the code: the combinations whose length the projector
    /// keeps.
    pub fn within(&self, basis: &[Amplitudes]) -> Vec<Amplitudes> {
        let Some(first) = basis.first() else { return Vec::new() };
        let (width, n) = (first.width, first.n);
        // v = Σ c_i b_i is in the code exactly when (G − M)c = 0, G the Gram
        // matrix and M the projector's; G − M is positive semidefinite
        let rows: Vec<Vec<Cyclo>> = basis
            .iter()
            .map(|a| basis.iter().map(|b| super::linalg::inner(a, b).sub(&self.between(a, b))).collect())
            .collect();
        let found: Vec<Amplitudes> = super::linalg::null_space(rows, basis.len(), n)
            .into_iter()
            .map(|c| super::linalg::combine(basis, &c, width, n))
            .filter(|v| !v.terms.is_empty())
            .collect();
        super::linalg::basis(&found)
    }

    /// The dimension of the part of the code the projections of `vs`
    /// span.
    pub fn reach(&self, vs: &[Amplitudes]) -> usize {
        let rows: Vec<Vec<Cyclo>> = vs.iter().map(|a| vs.iter().map(|b| self.between(a, b)).collect()).collect();
        super::linalg::rank(rows, vs.len())
    }
}

/// The code a set of generators fixes.
pub fn describe(gens: &[Pauli]) -> String {
    format!("the code of <{}>", gens.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
}

/// The certificate of the Clifford circuit `c` from `source`, a code cover,
/// to `target`, derived on the stabiliser path; `None` where the path does
/// not apply: another cover, a gate outside the Clifford group, or a target
/// it cannot compare with.
pub fn derive(c: &Circuit, source: &Base, target: &Base, n: u32) -> Option<Result<Certificate, Fail>> {
    let Cover::Code { gens, width, .. } = &source.cover else { return None };
    let width = *width;
    if width != c.inputs.len() || c.outputs.len() != target.cover.width() || !n.is_multiple_of(8) {
        return None;
    }
    let point = gens.len() == width;
    match &target.cover {
        Cover::Code { .. } => {}
        Cover::Pt(_) | Cover::Fin(_) if point => {}
        _ => return None,
    }
    let phi = State::new(entangled(gens, width), n)?;
    let m = phi.width() - width;
    let wires = c.wires as usize;
    let total = wires + m;
    let layout = |ws: &[crate::circuit::ir::Wire]| -> Vec<usize> {
        ws.iter().map(|w| w.0 as usize).chain(wires..total).collect()
    };
    let (inputs, outputs) = (layout(&c.inputs), layout(&c.outputs));
    let mut psi = phi.embed(&inputs, total);
    let reference = phi.embed(&outputs, total);
    psi.run(&c.body)?;
    // every wire the operator does not give back is in |0> again
    if (0..wires).filter(|w| !outputs.contains(w)).any(|w| !psi.fixes(&Phased::z_on(w, total))) {
        return None;
    }
    let out_at = &outputs[..c.outputs.len()];
    if let Cover::Code { gens: tgens, .. } = &target.cover
        && let Some(t) = tgens.iter().find(|t| !psi.fixes(&Phased::placed(t, out_at, total)))
    {
        return Some(Err(Fail::Unfixed(describe(gens), t.to_string())));
    }
    let stationary = reference.gens.iter().all(|g| psi.fixes(g));
    // the multiple of the code's state that the operator gives, when it is
    // one
    let lambda = || psi.amplitude(&reference.x0).div(&reference.amp);
    let outside = || Fail::OutsideCode(describe(gens));
    if !point {
        if !stationary {
            return Some(Ok(Certificate::Moving));
        }
        return Some(
            lambda()
                .and_then(|l| super::cert::phase_of(&l, n))
                .map(Certificate::Scalar)
                .ok_or_else(outside),
        );
    }
    // which of the target's points the operator carries the source's to; a
    // target of more than one point that is not written out takes it only
    // to itself
    let index = match &target.cover {
        Cover::Code { gens: tgens, width, .. } if tgens.len() == *width || stationary => Some(0),
        Cover::Code { .. } => None,
        Cover::Pt(t) => proportional(&psi, t, out_at, total).then_some(0),
        Cover::Fin(ts) => ts.iter().position(|t| proportional(&psi, t, out_at, total)),
        _ => return None,
    };
    let Some(index) = index else {
        return Some(Err(Fail::Unreached(describe(gens))));
    };
    // ⟨s|ω⟩ and ⟨ω'|v⟩ for the fiducials that reach the point and what it
    // becomes; θ is the argument of their product
    let s_omega = |w: &Amplitudes| {
        w.terms
            .iter()
            .fold(Cyclo::zero(n), |acc, (x, a)| acc.add(&a.mul(&phi.amplitude(x).conj())))
    };
    let omega_v = |w: &Amplitudes| {
        w.terms
            .iter()
            .fold(Cyclo::zero(n), |acc, (x, a)| acc.add(&a.conj().mul(&psi.amplitude(&spread(x, out_at, total)))))
    };
    let from = source.gauge.fiducials.iter().map(s_omega).find(|z| !z.is_zero());
    let to = target.gauge.fiducials.iter().map(omega_v).find(|z| !z.is_zero());
    let z = match (from, to) {
        (Some(a), Some(b)) => Some(a.mul(&b)),
        _ if stationary => lambda(),
        _ => None,
    };
    let Some(theta) = z.and_then(|z| super::cert::phase_of(&z, n)) else {
        return Some(Err(outside()));
    };
    Some(Ok(Certificate::Points {
        images: vec![index],
        schedule: vec![theta],
        stationary,
    }))
}

/// Whether `psi`, read on the qubits `at`, is a multiple of `t`.
fn proportional(psi: &State, t: &Amplitudes, at: &[usize], total: usize) -> bool {
    let k = psi.support();
    if k >= 64 || 1usize << k != t.terms.len() {
        return false;
    }
    let spread = |x: &[bool]| spread(x, at, total);
    let Some((x1, t1)) = t.terms.iter().next() else { return false };
    let p1 = psi.amplitude(&spread(x1));
    if p1.is_zero() {
        return false;
    }
    t.terms.iter().all(|(x, tx)| psi.amplitude(&spread(x)).mul(t1) == p1.mul(tx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::ir::{Control, Wire};
    use crate::exact::Frac;
    use crate::judge::cover::Gauge;
    use crate::judge::sim;

    fn p(s: &str) -> Pauli {
        match s.strip_prefix('-') {
            Some(rest) => Pauli::parse(true, rest).unwrap(),
            None => Pauli::parse(false, s).unwrap(),
        }
    }

    fn gate(g: GateOp, t: u32, c: Option<(u32, bool)>) -> Op {
        Op::Gate {
            gate: g,
            targets: vec![Wire(t)],
            controls: c.map(|(w, on)| Control { wire: Wire(w), on }).into_iter().collect(),
        }
    }

    /// A global phase of `k` eighths of a turn.
    fn global(k: i64) -> Op {
        Op::Gate {
            gate: GateOp::GPhase(Angle::Fixed(crate::exact::Phase::of(8, k))),
            targets: Vec::new(),
            controls: Vec::new(),
        }
    }

    fn circuit(width: u32, body: Vec<Op>) -> Circuit {
        Circuit {
            wires: width,
            inputs: (0..width).map(Wire).collect(),
            outputs: (0..width).map(Wire).collect(),
            body,
            conductor: 8,
            ..Circuit::default()
        }
    }

    fn code(gens: &[&str], fiducials: Vec<Amplitudes>) -> Base {
        let gens: Vec<Pauli> = gens.iter().map(|g| p(g)).collect();
        let width = gens[0].width();
        Base {
            cover: Cover::Code { gens, width, n: 8 },
            gauge: Gauge { fiducials },
        }
    }

    /// The state with equal amplitudes on each of `kets`.
    fn even(kets: &[&str]) -> Amplitudes {
        let mut a = Amplitudes::basis(kets[0].chars().map(|c| c == '1').collect(), 8);
        a.terms.clear();
        for k in kets {
            a.terms.insert(k.chars().map(|c| c == '1').collect(), Cyclo::from_frac(8, Frac::new(1, 2)));
        }
        a
    }

    /// The stabiliser path agrees with exact action, certificate for
    /// certificate, and refuses where it refuses.
    fn agrees(b: &Base, c: &Circuit) {
        let fast = derive(c, b, b, 8).expect("the path applies");
        let mut u = |v: &Amplitudes| sim::action(c, v).unwrap();
        let exact = crate::judge::cert::derive(&mut u, b, b, 8);
        match (&fast, &exact) {
            (Ok(a), Ok(e)) => assert_eq!(a, e, "{:?}", c.body),
            (Err(_), Err(_)) => {}
            _ => panic!("{fast:?} against {exact:?} for {:?}", c.body),
        }
    }

    #[test]
    fn conjugation_and_products_keep_phases() {
        // S X S† = Y, H Y H = −Y
        let mut x = Phased::placed(&p("X"), &[0], 1);
        x.conjugate(Step::S(0));
        assert_eq!(x, Phased::placed(&p("Y"), &[0], 1));
        x.conjugate(Step::H(0));
        assert_eq!(x, Phased::placed(&p("-Y"), &[0], 1));
        // X·Z = −i Y
        let xz = Phased::placed(&p("X"), &[0], 1).mul(&Phased::placed(&p("Z"), &[0], 1));
        assert_eq!(xz, Phased { x: vec![true], z: vec![true], e: 0 });
        // CZ carries XX to YY
        let mut xx = Phased::placed(&p("XX"), &[0, 1], 2);
        xx.conjugate(Step::Cz(0, 1));
        assert_eq!(xx, Phased::placed(&p("YY"), &[0, 1], 2));
    }

    #[test]
    fn amplitudes_follow_from_the_group() {
        // |Φ+> fixed by XX and ZZ; then (S ⊗ I) gives |00> + i|11>
        let mut s = State::new(vec![Phased::placed(&p("XX"), &[0, 1], 2), Phased::placed(&p("ZZ"), &[0, 1], 2)], 8).unwrap();
        assert_eq!(s.quarter(&[true, true]), Some(0));
        assert_eq!(s.quarter(&[true, false]), None);
        s.step(Step::S(0)).unwrap();
        let ratio = s.amplitude(&[true, true]).div(&s.amplitude(&[false, false])).unwrap();
        assert_eq!(ratio, Cyclo::imaginary_unit(8).unwrap());
        // H on |0>: |+>, weight on both
        let mut z = State::new(vec![Phased::placed(&p("Z"), &[0], 1)], 8).unwrap();
        z.step(Step::H(0)).unwrap();
        assert!(z.fixes(&Phased::placed(&p("X"), &[0], 1)));
        assert!(!z.fixes(&Phased::placed(&p("-X"), &[0], 1)));
    }

    #[test]
    fn a_point_code_agrees_with_exact_action() {
        let bell = code(&["XX", "ZZ"], vec![even(&["00", "11"])]);
        for body in [
            vec![gate(GateOp::X, 0, None), gate(GateOp::X, 1, None)],
            vec![gate(GateOp::Z, 0, None)],
            vec![gate(GateOp::H, 0, None), gate(GateOp::H, 1, None)],
            vec![gate(GateOp::S, 0, None), gate(GateOp::Sdg, 1, None)],
            vec![gate(GateOp::S, 0, None), gate(GateOp::S, 1, None)],
            vec![gate(GateOp::X, 1, Some((0, true)))],
            vec![gate(GateOp::Y, 0, None), gate(GateOp::Y, 1, None)],
            vec![gate(GateOp::Z, 1, Some((0, false)))],
            vec![global(1), gate(GateOp::S, 0, None), gate(GateOp::Sdg, 1, None)],
            vec![global(5), gate(GateOp::H, 0, None), gate(GateOp::H, 1, None)],
        ] {
            agrees(&bell, &circuit(2, body));
        }
    }

    #[test]
    fn a_subspace_code_agrees_with_exact_action() {
        let even_parity = code(&["ZZ"], vec![]);
        let three = code(&["ZZI", "IZZ"], vec![]);
        for body in [
            vec![gate(GateOp::X, 0, None), gate(GateOp::X, 1, None)],
            vec![gate(GateOp::Z, 0, None), gate(GateOp::Z, 1, None)],
            vec![gate(GateOp::Z, 0, None)],
            vec![gate(GateOp::S, 0, None), gate(GateOp::S, 1, None)],
            vec![gate(GateOp::S, 0, None), gate(GateOp::Sdg, 1, None)],
            vec![gate(GateOp::H, 0, None), gate(GateOp::H, 1, None)],
            vec![gate(GateOp::X, 1, Some((0, true))), gate(GateOp::X, 1, Some((0, true)))],
            vec![global(1), gate(GateOp::Z, 0, None), gate(GateOp::Z, 1, None)],
            vec![global(3), gate(GateOp::S, 0, None), gate(GateOp::Sdg, 1, None)],
        ] {
            agrees(&even_parity, &circuit(2, body));
        }
        for body in [
            vec![gate(GateOp::X, 0, None), gate(GateOp::X, 1, None), gate(GateOp::X, 2, None)],
            vec![gate(GateOp::Z, 0, None), gate(GateOp::Z, 2, None)],
            vec![gate(GateOp::X, 1, Some((0, true))), gate(GateOp::X, 1, Some((2, true)))],
        ] {
            agrees(&three, &circuit(3, body));
        }
    }

    #[test]
    fn a_wide_code_is_judged_in_polynomial_time() {
        // the repetition code on 40 qubits: ZZ on two neighbours is the
        // identity on it, X on every qubit a logical flip, and a global
        // phase a scalar
        let width = 40;
        let gens: Vec<String> = (0..width - 1)
            .map(|i| (0..width).map(|q| if q == i || q == i + 1 { 'Z' } else { 'I' }).collect())
            .collect();
        let refs: Vec<&str> = gens.iter().map(String::as_str).collect();
        let rep = code(&refs, vec![]);
        let zz = circuit(width as u32, vec![gate(GateOp::Z, 3, None), gate(GateOp::Z, 4, None)]);
        assert_eq!(derive(&zz, &rep, &rep, 8), Some(Ok(Certificate::Scalar(crate::exact::Phase::zero(8)))));
        let flip = circuit(width as u32, (0..width as u32).map(|q| gate(GateOp::X, q, None)).collect());
        assert_eq!(derive(&flip, &rep, &rep, 8), Some(Ok(Certificate::Moving)));
        let h = circuit(width as u32, vec![gate(GateOp::H, 0, None)]);
        assert!(matches!(derive(&h, &rep, &rep, 8), Some(Err(Fail::Unfixed(..)))));
        // the GHZ state on 40 qubits, gauged by itself: X on every qubit
        // keeps it, rigidly; Z on one carries it out
        let mut ghz_gens = vec!["X".repeat(width)];
        ghz_gens.extend(gens.iter().cloned());
        let refs: Vec<&str> = ghz_gens.iter().map(String::as_str).collect();
        let ghz = code(&refs, vec![even(&[&"0".repeat(width), &"1".repeat(width)])]);
        let kept = derive(&flip, &ghz, &ghz, 8).unwrap().unwrap();
        assert_eq!(
            kept,
            Certificate::Points {
                images: vec![0],
                schedule: vec![crate::exact::Phase::zero(8)],
                stationary: true
            }
        );
        let z = circuit(width as u32, vec![gate(GateOp::Z, 0, None)]);
        assert!(matches!(derive(&z, &ghz, &ghz, 8), Some(Err(Fail::Unfixed(..)))));
    }

    #[test]
    fn projectors_and_the_codes_two_groups_share() {
        let ket = |s: &str| Amplitudes::basis(s.chars().map(|c| c == '1').collect(), 8);
        let half = Cyclo::from_frac(8, Frac::new(1, 2));
        // onto the even states: |00> is kept whole, |01> not at all
        let even_parity = Projector::new(&[p("ZZ")], 8);
        assert_eq!(even_parity.between(&ket("00"), &ket("00")), Cyclo::one(8));
        assert!(even_parity.between(&ket("01"), &ket("01")).is_zero());
        // onto the XX-fixed states: |00> and |11> each half in, and joined
        let xx = Projector::new(&[p("XX")], 8);
        assert_eq!(xx.between(&ket("00"), &ket("11")), half);
        assert_eq!(Projector::new(&[p("-XX")], 8).between(&ket("00"), &ket("11")), half.neg());
        // the Bell state, as what both fix; nothing, against its negation
        assert_eq!(joint(&[p("XX")], &[p("ZZ")]), Some(vec![p("XX"), p("ZZ")]));
        assert_eq!(joint(&[p("ZZ")], &[p("-ZZ")]), None);
        assert_eq!(joint(&[p("XI")], &[p("ZI")]), None);
        assert_eq!(joint(&[p("ZZ"), p("XX")], &[p("-YY")]), Some(vec![p("XX"), p("ZZ")]));
        // what of a span the code keeps
        let kept = even_parity.within(&[ket("00"), ket("01"), ket("11")]);
        assert_eq!(kept.len(), 2);
        assert_eq!(xx.reach(&[ket("00"), ket("11")]), 1);
    }
}
