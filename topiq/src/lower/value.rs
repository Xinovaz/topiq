//! What circuit generation computes with.
//!
//! A [`V`] is a value while an operator is being generated. A classical value
//! known now is a [`Value`], as constant evaluation has it. One known only
//! when the circuit runs is a [`CExpr`], of one of two [`Kind`]s. A qubit is
//! the [`Wire`] that carries it, and a value mixing the two (a register, a
//! quantum structure) is an aggregate of both.

use crate::circuit::{CExpr, Wire};
use crate::tir::{AdtId, FnId, GlobalId, LocalId, Value};

/// Which of the two kinds of value known only when the circuit runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Derived from a measurement: it may control what follows, and become
    /// the operator's result, but not shape the circuit.
    Outcome,
    /// A classical parameter of the entry operator, or a lifted outcome,
    /// fixed for each run before its structure is: it may bound a loop and
    /// set an angle.
    Input,
}

impl Kind {
    /// The kind of a value computed from values of kinds `a` and `b`.
    pub fn join(a: Kind, b: Kind) -> Kind {
        if a == Kind::Outcome || b == Kind::Outcome { Kind::Outcome } else { Kind::Input }
    }
}

/// A value during circuit generation.
#[derive(Clone, PartialEq, Debug)]
pub enum V {
    /// A classical value known now.
    Val(Value),
    /// A classical value known when the circuit runs.
    Sym(CExpr, Kind),
    /// A qubit.
    Wire(Wire),
    /// A structure, tuple or array with a part that is not known now.
    Agg {
        /// Whether it is an array, rather than a structure or tuple.
        array: bool,
        /// Its fields or elements.
        items: Vec<V>,
    },
    /// A classical enumeration's variant with a part that is not known now.
    Variant(u32, Vec<V>),
    /// A quantum enumeration: a tag register saying which variant, in
    /// superposition, and the payload register the variants share.
    QEnum {
        /// The enumeration.
        adt: AdtId,
        /// The tag's qubits.
        tag: Vec<Wire>,
        /// The payload's qubits.
        payload: Vec<Wire>,
    },
    /// A register of qubits whose length is known only as the circuit runs,
    /// by its number in the circuit.
    Reg(u32),
    /// A reference.
    Ref(Place),
    /// A function of the unit.
    Func(FnId),
    /// An operator-typed parameter of the entry operator.
    Slot(u32),
    /// A map locale.
    Table(usize),
    /// A closure: its code, and what it captured.
    Closure {
        /// The function that is its body.
        code: FnId,
        /// What it captured, in order.
        env: Vec<V>,
    },
    /// An operator made of others by `adjoint`, `controlled` or `then`.
    Functor(Box<Functor>),
    /// Nothing: a binding moved from, or not yet given a value.
    Moved,
}

/// An operator made of others.
#[derive(Clone, PartialEq, Debug)]
pub enum Functor {
    /// `adjoint(f)`.
    Adjoint(V),
    /// `controlled(f)`.
    Controlled(V),
    /// `f.then(g)`.
    Then(V, V),
    /// `query t[k]`: the map locale `t`, by position, and the key register's
    /// qubits.
    Query(usize, Vec<Wire>),
    /// The entry `measure t[k]` gives: the map locale `t`, by position, and
    /// the bits the key was measured into, the first qubit's first.
    Lookup(usize, Vec<CExpr>),
}

/// Where a value is held.
#[derive(Clone, PartialEq, Debug)]
pub enum Place {
    /// A binding of an activation, and a path of field or element indices
    /// into it.
    Local {
        /// The activation.
        frame: usize,
        /// The binding.
        local: LocalId,
        /// The path.
        path: Vec<u32>,
    },
    /// A unit-scope object, and a path into it.
    Global {
        /// The object.
        id: GlobalId,
        /// The path.
        path: Vec<u32>,
    },
    /// A value with no binding of its own, such as a node handle's qubit,
    /// and a path into it.
    Temp(Box<V>, Vec<u32>),
}

impl Place {
    /// The place `index` further in.
    pub fn child(&self, index: u32) -> Place {
        let mut p = self.clone();
        match &mut p {
            Place::Local { path, .. } | Place::Global { path, .. } | Place::Temp(_, path) => path.push(index),
        }
        p
    }
}

impl V {
    /// The value, with an aggregate or variant every part of which is known
    /// now made a [`Value`].
    pub fn normal(self) -> V {
        match self {
            V::Agg { array, items } => match known(items) {
                Ok(values) => V::Val(if array { Value::Array(values) } else { Value::Struct(values) }),
                Err(items) => V::Agg { array, items },
            },
            V::Variant(variant, fields) => match known(fields) {
                Ok(fields) => V::Val(Value::Enum { variant, fields }),
                Err(fields) => V::Variant(variant, fields),
            },
            v => v,
        }
    }

    /// The value with its parts opened out, so that one may change: a known
    /// structure, array or variant becomes an aggregate of known parts.
    pub fn opened(self) -> V {
        match self {
            V::Val(Value::Struct(fs)) => V::Agg {
                array: false,
                items: fs.into_iter().map(V::Val).collect(),
            },
            V::Val(Value::Array(xs)) => V::Agg {
                array: true,
                items: xs.into_iter().map(V::Val).collect(),
            },
            V::Val(Value::Enum { variant, fields }) => V::Variant(variant, fields.into_iter().map(V::Val).collect()),
            v => v,
        }
    }

    /// The part at `index`.
    pub fn part(&self, index: u32) -> Option<V> {
        let i = index as usize;
        match self {
            V::Val(Value::Struct(fs) | Value::Array(fs)) => fs.get(i).cloned().map(V::Val),
            V::Val(Value::Enum { fields, .. }) => fields.get(i).cloned().map(V::Val),
            V::Agg { items, .. } | V::Variant(_, items) => items.get(i).cloned(),
            V::Sym(e, k) => Some(V::Sym(CExpr::Field(Box::new(e.clone()), index), *k)),
            _ => None,
        }
    }

    /// The part at the end of `path`.
    pub fn at(&self, path: &[u32]) -> Option<V> {
        let mut v = self.clone();
        for &i in path {
            v = v.part(i)?;
        }
        Some(v)
    }

    /// Puts `new` at the end of `path` in this value, giving back what was
    /// there.
    pub fn replace(&mut self, path: &[u32], new: V) -> Option<V> {
        let Some((&first, rest)) = path.split_first() else {
            return Some(std::mem::replace(self, new));
        };
        *self = std::mem::replace(self, V::Moved).opened();
        let old = match self {
            V::Agg { items, .. } | V::Variant(_, items) => items.get_mut(first as usize)?.replace(rest, new),
            _ => None,
        };
        *self = std::mem::replace(self, V::Moved).normal();
        old
    }

    /// Whether the value holds a register whose length is known only as the
    /// circuit runs.
    pub fn has_register(&self) -> bool {
        match self {
            V::Reg(_) => true,
            V::Agg { items, .. } | V::Variant(_, items) => items.iter().any(V::has_register),
            _ => false,
        }
    }

    /// Every qubit the value holds, in order.
    pub fn wires(&self) -> Vec<Wire> {
        let mut out = Vec::new();
        self.collect_wires(&mut out);
        out
    }

    /// The value with its qubits replaced, in the order [`V::wires`] gives
    /// them, by those `with` yields.
    pub fn with_wires(&self, with: &mut impl Iterator<Item = Wire>) -> V {
        match self {
            V::Wire(_) => V::Wire(with.next().expect("as many qubits as the value holds")),
            V::Agg { array, items } => V::Agg {
                array: *array,
                items: items.iter().map(|i| i.with_wires(with)).collect(),
            },
            V::Variant(v, items) => V::Variant(*v, items.iter().map(|i| i.with_wires(with)).collect()),
            V::QEnum { adt, tag, payload } => V::QEnum {
                adt: *adt,
                tag: tag.iter().map(|_| with.next().expect("as many qubits as the value holds")).collect(),
                payload: payload.iter().map(|_| with.next().expect("as many qubits as the value holds")).collect(),
            },
            other => other.clone(),
        }
    }

    fn collect_wires(&self, out: &mut Vec<Wire>) {
        match self {
            V::Wire(w) => out.push(*w),
            V::Agg { items, .. } | V::Variant(_, items) => {
                for i in items {
                    i.collect_wires(out);
                }
            }
            V::QEnum { tag, payload, .. } => {
                out.extend(tag);
                out.extend(payload);
            }
            _ => {}
        }
    }

    /// The value as a classical expression, if it holds no qubit and nothing
    /// that is not data; the kind is `None` when it is all known now.
    pub fn classical(&self) -> Option<(CExpr, Option<Kind>)> {
        match self {
            V::Val(v) => Some((CExpr::Value(v.clone()), None)),
            V::Sym(e, k) => Some((e.clone(), Some(*k))),
            V::Agg { array, items } => {
                let (parts, kind) = classical_all(items)?;
                Some((if *array { CExpr::Array(parts) } else { CExpr::Struct(parts) }, kind))
            }
            V::Variant(v, fields) => {
                let (parts, kind) = classical_all(fields)?;
                Some((CExpr::Variant(*v, parts), kind))
            }
            _ => None,
        }
    }

    /// The classical parts of the value, qubits left out: what a caller of
    /// a circuit returning it receives. `None` when there are none.
    pub fn classical_part(&self) -> Option<CExpr> {
        match self {
            V::Wire(_) | V::QEnum { .. } | V::Moved | V::Val(Value::Void) => None,
            V::Agg { array: false, items } => {
                let parts: Vec<CExpr> = items.iter().filter_map(V::classical_part).collect();
                match parts.len() {
                    0 => None,
                    1 => parts.into_iter().next(),
                    _ => Some(CExpr::Struct(parts)),
                }
            }
            v => v.classical().map(|(e, _)| e),
        }
    }
}

/// The values of `items`, each made normal, when every one is known now;
/// otherwise the normal items.
fn known(items: Vec<V>) -> Result<Vec<Value>, Vec<V>> {
    let items: Vec<V> = items.into_iter().map(V::normal).collect();
    if !items.iter().all(|i| matches!(i, V::Val(_))) {
        return Err(items);
    }
    Ok(items.into_iter().filter_map(|i| if let V::Val(v) = i { Some(v) } else { None }).collect())
}

/// The classical expressions of `items`, and the kind they join to.
fn classical_all(items: &[V]) -> Option<(Vec<CExpr>, Option<Kind>)> {
    let mut kind = None;
    let parts = items
        .iter()
        .map(|i| {
            let (e, k) = i.classical()?;
            kind = join_kind(kind, k);
            Some(e)
        })
        .collect::<Option<_>>()?;
    Some((parts, kind))
}

/// The kind of a value built from parts of kinds `a` and `b`, `None` being
/// known now.
pub fn join_kind(a: Option<Kind>, b: Option<Kind>) -> Option<Kind> {
    match (a, b) {
        (None, k) | (k, None) => k,
        (Some(a), Some(b)) => Some(Kind::join(a, b)),
    }
}
