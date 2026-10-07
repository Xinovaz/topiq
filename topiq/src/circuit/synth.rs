//! Exact state preparation: a circuit of the standard gates that takes
//! |0…0> to a given state, with nothing approximated.
//!
//! # The method
//!
//! Every gate of the standard set has entries in the ring `Z[1/2, ζ_N]`, so
//! every amplitude a circuit of them prepares from |0…0> lies there too. A
//! state with an amplitude outside it (one whose denominator has an odd
//! factor, like 1/3) cannot be prepared at any conductor, and is refused
//! for that reason ([`Obstruction::Ring`]).
//!
//! A state inside the ring is reduced to |0…0> by column reduction:
//!
//! - Two has one prime factor δ in `Z[ζ_N]`, and each amplitude's
//!   *denominator exponent* is the least k for which `δ^k` times it is an
//!   algebraic integer. This is finer than counting powers of √2, and the
//!   reduction needs the finer count: two amplitudes can have the same power
//!   of √2 and yet need a step that lowers only their powers of δ.
//! - While some exponent is positive, two amplitudes x and y are combined by
//!   a two-level operation, a Hadamard after the phase `ζ_N^m` on y, which
//!   gives (x + ζ^m y)/√2 and (x − ζ^m y)/√2. The pair and the m chosen are
//!   those that lower the sum of the exponents most, looked for first among
//!   the amplitudes of the largest exponent.
//! - Once no exponent is positive, the state, a unit vector of algebraic
//!   integers, is a root of unity times one basis state, which bit flips and a
//!   global phase take to |0…0>.
//!
//! The reduction is greedy, and nothing here proves that a lowering step
//! always exists; a state for which none does is refused as
//! [`Obstruction::Stuck`], naming the amplitude the reduction could not
//! lower. On the states circuits of the standard gates reach, it has found
//! one at every conforming conductor.
//!
//! A two-level operation on basis states that differ in more than one qubit
//! is made one that differs in a single qubit by controlled bit flips, which
//! are undone after it. The single-qubit operation is controlled on every
//! other qubit.
//!
//! The preparation is the reduction reversed and inverted, and it is checked
//! before it is returned: applied to |0…0>, exactly, it must give the state.

use std::collections::BTreeMap;

use super::action::{Sparse, apply_sparse, gate_matrix};
use super::ir::{Angle, Control, GateOp, Op, Wire};
use crate::exact::{Cyclo, Frac, Phase};
use crate::quon::eval::Amplitudes;

/// A preparation: operations on the wires `0..width`, the register's first
/// qubit on wire 0, that take |0…0> to the state.
#[derive(Clone, PartialEq, Debug)]
pub struct Prepared {
    /// How many qubits.
    pub width: usize,
    /// The operations.
    pub ops: Vec<Op>,
}

impl Prepared {
    /// The operations.
    pub fn on(&self, wires: &[Wire]) -> Vec<Op> {
        let at = |w: &Wire| wires[w.0 as usize];
        self.ops
            .iter()
            .map(|op| match op {
                Op::Gate { gate, targets, controls } => Op::Gate {
                    gate: gate.clone(),
                    targets: targets.iter().map(at).collect(),
                    controls: controls.iter().map(|c| Control { wire: at(&c.wire), on: c.on }).collect(),
                },
                op => op.clone(),
            })
            .collect()
    }
}

/// Why a state is not prepared.
#[derive(Clone, PartialEq, Debug)]
pub enum Obstruction {
    /// An amplitude outside `Z[1/2, ζ_N]`, which no circuit of the standard
    /// gates prepares at any conductor: its basis state, its value, and the
    /// odd factor of its denominator.
    Ring(Vec<bool>, Cyclo, num_bigint::BigInt),
    /// An amplitude the reduction could not lower: its basis state and its
    /// value.
    Stuck(Vec<bool>, Cyclo),
}

/// A preparation of `state`.
///
/// # Errors
///
/// The amplitude that stops it, and why.
pub fn prepare(state: &Amplitudes) -> Result<Prepared, Obstruction> {
    let n = state.n;
    for (bits, c) in &state.terms {
        let odd = c.odd_denominator();
        if !num_traits::One::is_one(&odd) {
            return Err(Obstruction::Ring(bits.clone(), c.clone(), odd));
        }
    }
    let two = Two::new(n);
    let mut v = state.terms.clone();
    let mut forward: Vec<Op> = Vec::new();
    loop {
        if v.values().all(Cyclo::is_integral) {
            break;
        }
        // the step that lowers the sum of the exponents most, looked for
        // first among the amplitudes of the largest exponent
        let keys: Vec<&Vec<bool>> = v.keys().collect();
        let exps: Vec<i64> = keys.iter().map(|k| two.exponent(&v[*k])).collect();
        let top = exps.iter().copied().max().unwrap_or(0);
        let mut best: Option<(i64, usize, usize, u32)> = None;
        for all in [false, true] {
            for i in 0..keys.len() {
                for j in i + 1..keys.len() {
                    if !all && (exps[i] != top || exps[j] != top) {
                        continue;
                    }
                    let (a, b) = (keys[i], keys[j]);
                    let t = (0..a.len()).find(|&q| a[q] != b[q]).expect("two basis states differ");
                    let (x, y) = if a[t] { (&v[b], &v[a]) } else { (&v[a], &v[b]) };
                    let before = exps[i] + exps[j];
                    let mut zy = y.clone();
                    for m in 0..n {
                        if m > 0 {
                            zy = zy.mul_zeta_pow(1);
                        }
                        let after = two.halved(&x.add(&zy)) + two.halved(&x.sub(&zy));
                        let change = after - before;
                        if best.is_none_or(|b| change < b.0) {
                            best = Some((change, i, j, m));
                        }
                    }
                }
            }
            if best.is_some_and(|b| b.0 < 0) {
                break;
            }
        }
        match best {
            Some((change, i, j, m)) if change < 0 => {
                let ops = two_level(keys[i], keys[j], m, n);
                for op in &ops {
                    apply(&mut v, op, n);
                }
                forward.extend(ops);
            }
            _ => {
                let (bits, c) = v
                    .iter()
                    .max_by_key(|(_, c)| two.exponent(c))
                    .map(|(b, c)| (b.clone(), c.clone()))
                    .expect("a unit vector is not zero");
                return Err(Obstruction::Stuck(bits, c));
            }
        }
    }
    // one basis state is left, with a root of unity for its amplitude
    let (bits, c) = v.iter().next().map(|(b, c)| (b.clone(), c.clone())).expect("a unit vector is not zero");
    let Some(j) = (0..n).find(|&j| c == Cyclo::zeta_pow(n, i64::from(j))) else {
        return Err(Obstruction::Stuck(bits, c));
    };
    for (q, &b) in bits.iter().enumerate() {
        if b {
            forward.push(gate(GateOp::X, q, Vec::new()));
        }
    }
    if j != 0 {
        forward.push(Op::Gate {
            gate: GateOp::GPhase(Angle::Fixed(Phase::of(n, -i64::from(j)))),
            targets: Vec::new(),
            controls: Vec::new(),
        });
    }
    let ops: Vec<Op> = forward
        .iter()
        .rev()
        .map(|op| op.inverse().expect("a gate has an inverse"))
        .collect();
    let prepared = Prepared { width: state.width, ops };
    assert_eq!(
        run(&prepared, n),
        state.terms,
        "a preparation must prepare the state it was made for"
    );
    Ok(prepared)
}

/// What `p` makes of |0…0>, computed exactly.
pub fn run(p: &Prepared, n: u32) -> Sparse {
    let mut v = BTreeMap::from([(vec![false; p.width], Cyclo::one(n))]);
    for op in &p.ops {
        apply(&mut v, op, n);
    }
    v
}

/// The prime of `Z[ζ_N]` above two, δ = `1 − ζ` for ζ a root of unity of
/// the largest power `2^a` dividing N, which is the only one: two is a unit
/// times `δ^e`, e = `2^(a−1)`, and √2 a unit times `δ^(e/2)`.
struct Two {
    /// δ⁻¹.
    inv: Cyclo,
    /// e.
    e: i64,
}

impl Two {
    fn new(n: u32) -> Two {
        let two_part = 1u32 << n.trailing_zeros();
        let delta = Cyclo::one(n).sub(&Cyclo::zeta_pow(n, i64::from(n / two_part)));
        Two {
            inv: delta.inv().expect("δ is not zero"),
            e: i64::from(two_part / 2),
        }
    }

    /// How many times δ divides `c`, negative when it is in the
    /// denominator; `None` for zero. `c` is in `Z[1/2, ζ_N]`.
    fn valuation(&self, c: &Cyclo) -> Option<i64> {
        if c.is_zero() {
            return None;
        }
        // 2^d · c is an algebraic integer, of valuation e·d more
        let d = c.two_denominator();
        let mut y = c.scale(&Frac::from_bigint(num_bigint::BigInt::from(1) << d));
        let mut v = -self.e * d as i64;
        loop {
            let q = y.mul(&self.inv);
            if !q.is_integral() {
                return Some(v);
            }
            y = q;
            v += 1;
        }
    }

    /// The least k for which `δ^k · c` is an algebraic integer.
    fn exponent(&self, c: &Cyclo) -> i64 {
        self.valuation(c).map_or(0, |v| (-v).max(0))
    }

    /// The exponent of `c/√2`.
    fn halved(&self, c: &Cyclo) -> i64 {
        self.valuation(c).map_or(0, |v| (self.e / 2 - v).max(0))
    }
}

fn gate(g: GateOp, target: usize, controls: Vec<Control>) -> Op {
    Op::Gate {
        gate: g,
        targets: vec![Wire(target as u32)],
        controls,
    }
}

/// The operations that act on the basis states `a` and `b` alone, as the
/// Hadamard after `ζ^m` on the one whose first differing qubit is 1.
fn two_level(a: &[bool], b: &[bool], m: u32, n: u32) -> Vec<Op> {
    let diff: Vec<usize> = (0..a.len()).filter(|&q| a[q] != b[q]).collect();
    let t = diff[0];
    let lo = if a[t] { b } else { a };
    // bit flips controlled on `t` make the two differ only there
    let flips: Vec<Op> = diff[1..]
        .iter()
        .map(|&q| gate(GateOp::X, q, vec![Control { wire: Wire(t as u32), on: true }]))
        .collect();
    let controls: Vec<Control> = (0..a.len())
        .filter(|&q| q != t)
        .map(|q| Control { wire: Wire(q as u32), on: lo[q] })
        .collect();
    let mut ops = flips.clone();
    if m != 0 {
        ops.push(gate(GateOp::Phase(Angle::Fixed(Phase::of(n, i64::from(m)))), t, controls.clone()));
    }
    ops.push(gate(GateOp::H, t, controls));
    ops.extend(flips.into_iter().rev());
    ops
}

/// Applies a gate of a preparation to a sparse state.
fn apply(v: &mut Sparse, op: &Op, n: u32) {
    let Op::Gate { gate, targets, controls } = op else {
        unreachable!("a preparation is made of gates");
    };
    let m = gate_matrix(gate, n).expect("a preparation's angles are known");
    apply_sparse(v, &m, targets, controls, n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::action::State;
    use crate::quon::eval::evaluate;
    use crate::quon::parse::testing::state_in;

    fn amplitudes(src: &str, n: u32) -> Amplitudes {
        let (s, interner) = state_in(src);
        let a = evaluate(&s, n, &interner, &mut |_, _| None).unwrap();
        assert!(a.is_normalized(), "{src} is not a unit vector");
        a
    }

    /// Prepares `src` and checks the preparation by the dense simulation,
    /// independently of the sparse one the synthesis uses.
    fn round_trip(src: &str, n: u32) -> Prepared {
        let a = amplitudes(src, n);
        let p = prepare(&a).unwrap_or_else(|e| panic!("{src}: {e:?}"));
        let wires: Vec<Wire> = (0..a.width as u32).map(Wire).collect();
        let mut s = State::basis(wires, 0, n);
        for op in &p.ops {
            s.apply(op).unwrap();
        }
        for (i, amp) in s.amps.iter().enumerate() {
            let bits: Vec<bool> = (0..a.width).map(|q| (i >> (a.width - 1 - q)) & 1 == 1).collect();
            let want = a.terms.get(&bits).cloned().unwrap_or_else(|| Cyclo::zero(n));
            assert_eq!(*amp, want, "{src}: amplitude of {bits:?}");
        }
        p
    }

    #[test]
    fn basis_states_are_bit_flips() {
        let p = round_trip("|101>", 8);
        assert_eq!(p.ops.len(), 2);
        assert!(round_trip("|000>", 8).ops.is_empty());
    }

    #[test]
    fn a_root_of_unity_on_a_basis_state_is_a_global_phase() {
        let p = round_trip("i * |1>", 8);
        assert!(p.ops.iter().any(|op| matches!(op, Op::Gate { gate: GateOp::GPhase(_), .. })));
    }

    #[test]
    fn the_usual_states_are_prepared_exactly() {
        round_trip("isq2 * |0> + isq2 * |1>", 8);
        round_trip("isq2 * |0> - isq2 * |1>", 8);
        round_trip("isq2 * |00> + isq2 * |11>", 8);
        round_trip("isq2 * |01> - isq2 * |10>", 8);
        round_trip("isq2 * |000> + isq2 * |111>", 8);
        round_trip("1/2 * |00> + 1/2 * |01> + 1/2 * |10> + 1/2 * |11>", 8);
        round_trip("isq2 * |0> + w(1, 8) * isq2 * |1>", 8);
    }

    #[test]
    fn states_needing_two_levels_of_reduction_are_prepared() {
        // (1 + i)/2 = ω/√2, so these amplitudes have denominator exponents
        // of two and one
        round_trip("1/2 * |00> + 1/2 * i * |01> + isq2 * w(1, 8) * |11>", 8);
        round_trip("(1/2 + i/2) * 1/2 * |000> + (1/2 - i/2) * 1/2 * |011> + 1/2 * |101> + isq2 * |110>", 8);
    }

    #[test]
    fn finer_phases_are_prepared_at_the_conductors_that_hold_them() {
        round_trip("isq2 * |0> + w(1, 16) * isq2 * |1>", 16);
        round_trip("isq2 * |00> + w(3, 16) * isq2 * |11>", 16);
        round_trip("isq2 * |0> + w(1, 24) * isq2 * |1>", 24);
        round_trip("isq2 * |01> + w(1, 3) * isq2 * |10>", 24);
    }

    /// The state a pseudo-random circuit of `h`, `cx` and phases of order
    /// `order` makes from |000>, which the gates can therefore prepare.
    fn reachable(seed: u64, n: u32, order: u32, gates: usize) -> Amplitudes {
        let wires: Vec<Wire> = (0..3).map(Wire).collect();
        let mut s = State::basis(wires, 0, n);
        let mut x = seed;
        for _ in 0..gates {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let r = (x >> 33) as u32;
            let q = r % 3;
            let op = match (r >> 2) % 3 {
                0 => gate(GateOp::H, q as usize, Vec::new()),
                1 => gate(
                    GateOp::Phase(Angle::Fixed(Phase::of(n, i64::from(n / order)))),
                    q as usize,
                    Vec::new(),
                ),
                _ => gate(GateOp::X, q as usize, vec![Control { wire: Wire((q + 1) % 3), on: true }]),
            };
            s.apply(&op).unwrap();
        }
        let terms = s
            .amps
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(i, c)| ((0..3).map(|q| (i >> (2 - q)) & 1 == 1).collect(), c.clone()))
            .collect();
        Amplitudes { width: 3, n, terms }
    }

    #[test]
    fn states_the_gates_reach_are_prepared_at_each_conductor() {
        for (n, seeds) in [(8, 12), (16, 4), (24, 4)] {
            for seed in 0..seeds {
                let a = reachable(seed, n, n, 16);
                let p = prepare(&a).unwrap_or_else(|e| panic!("N={n}, seed {seed}: {e:?}"));
                assert_eq!(run(&p, n), a.terms);
            }
        }
    }

    #[test]
    fn an_amplitude_with_an_odd_denominator_is_refused_whatever_the_conductor() {
        // 3/5 and 4/5: a unit vector, but no gate divides by five
        let a = amplitudes("3/5 * |0> + 4/5 * |1>", 8);
        match prepare(&a) {
            Err(Obstruction::Ring(bits, _, odd)) => {
                assert_eq!(bits, vec![false]);
                assert_eq!(odd, num_bigint::BigInt::from(5));
            }
            other => panic!("{other:?}"),
        }
    }
}
