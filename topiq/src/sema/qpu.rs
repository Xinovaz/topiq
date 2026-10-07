//! Running circuits: what `dev.run(c, args…)`, `dev.run_opts(c, o, args…)`,
//! `dev.sample(c, o, args…)`, `dev.distribution(c, args…)` and a circuit
//! handle's call `c(args…)` are.
//!
//! A circuit handle's type, `circuit<Sig>`, is its entry operator's
//! signature, and a run takes the operator's arguments after the circuit, in
//! its order, each checked against its parameter here: a classical
//! parameter takes a value of its type; an operator-typed one, `fn(…)`, a
//! handle to a circuit of that signature, applied where the circuit calls
//! it; and a quantum one a `qpu::RegHandle`, the register it acts on. What
//! the run gives is `Result<KernelOf<Sig>, qpu::Error>`, `KernelOf<Sig>`
//! being the classical part of what the operator returns ([`kernel_of`]);
//! a sample gives `Result<[qpu::Count<KernelOf<Sig>>], qpu::Error>`, and a
//! distribution `Result<[qpu::Chance<KernelOf<Sig>>], qpu::Error>`.
//!
//! The call becomes one of the `qpu` library's, generic in the kernel's
//! type: `__run::<K>(dev, doc, values, circuits, registers)`: the handle as
//! its document, each classical argument as a `dyn`, each supplied circuit
//! as its document. A handle called as a function, `c(args…)`, is run on
//! the device `qpu::calls()` gives, the exact simulator unless the program
//! chose another, and unwrapped where it is called, so a failed run aborts
//! there (`RA10`).

use crate::ast;
use crate::diag::{Code, Diagnostic};
use crate::intern::Symbol;
use crate::span::{Span, Spanned};
use crate::tir::{Arg, Expr, ExprKind, Intrinsic, Ty, TypeTable};

use super::body::Checker;
use super::items::GenericKey;
use super::interface::Interface;

/// The classical part of what an operator returning `t` gives a caller who
/// runs its circuit: `t` itself when it holds no qubits; otherwise the parts
/// of a structure or tuple that are classical, as a tuple, or the one part;
/// and the empty tuple `()` when there is none.
pub fn kernel_of(types: &mut TypeTable, t: Ty) -> Ty {
    classical_part(types, t).unwrap_or_else(|| types.tuple(Vec::new()))
}

/// The classical part of a value of type `t`.
fn classical_part(types: &mut TypeTable, t: Ty) -> Option<Ty> {
    if t == Ty::Void {
        return None;
    }
    if !types.is_quantum(t) {
        return Some(t);
    }
    let parts: Vec<Ty> = if let Some(elems) = types.as_tuple(t) {
        elems.to_vec()
    } else if let Ty::Adt(id) = t
        && types.adt(id).is_struct()
    {
        types.adt(id).fields().iter().map(|f| f.ty).collect()
    } else {
        return None;
    };
    let kept: Vec<Ty> = parts.into_iter().filter_map(|p| classical_part(types, p)).collect();
    match kept.len() {
        0 => None,
        1 => Some(kept[0]),
        _ => Some(types.tuple(kept)),
    }
}

/// Which of a device's calls that run a circuit a method is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Run {
    /// `dev.run(c, args…)`: one run, its outcome.
    Once,
    /// `dev.run_opts(c, o, args…)`: the outcome `o.shots` runs see most.
    WithOptions,
    /// `dev.sample(c, o, args…)`: each outcome of `o.shots` runs, counted.
    Sample,
    /// `dev.distribution(c, args…)`: each outcome, with its probability.
    Distribution,
}

impl Run {
    /// The device's method `name` is.
    pub fn of_method(name: &str) -> Option<Run> {
        Some(match name {
            "run" => Run::Once,
            "run_opts" => Run::WithOptions,
            "sample" => Run::Sample,
            "distribution" => Run::Distribution,
            _ => return None,
        })
    }

    /// The method's name.
    fn method(self) -> &'static str {
        match self {
            Run::Once => "run",
            Run::WithOptions => "run_opts",
            Run::Sample => "sample",
            Run::Distribution => "distribution",
        }
    }

    /// Whether the options to run with come after the circuit.
    fn takes_options(self) -> bool {
        matches!(self, Run::WithOptions | Run::Sample)
    }

    /// The `qpu` function the call becomes.
    fn library_function(self) -> &'static str {
        match self {
            Run::Once => "__run",
            Run::WithOptions => "__run_opts",
            Run::Sample => "__sample",
            Run::Distribution => "__distribution",
        }
    }
}

/// What a run is given for an entry operator's parameters, each list in
/// the operator's order.
struct Supplied {
    values: Expr,
    circuits: Expr,
    registers: Expr,
}

impl<'a> Checker<'_, 'a> {
    /// Whether the structure `id` is the `qpu` library's `Device`.
    pub(super) fn is_qpu_device(&self, id: crate::tir::AdtId) -> bool {
        let def = self.types().adt(id);
        def.origin.as_deref() == Some("qpu") && self.name(def.name) == "Device"
    }

    /// `dev.run(c, args…)`, `dev.run_opts(c, o, args…)`,
    /// `dev.sample(c, o, args…)` or `dev.distribution(c, args…)`, as `run`
    /// says.
    pub(super) fn run_call(&mut self, recv: Expr, run: Run, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let method = run.method();
        let opts = run.takes_options();
        let lead = if opts { 2 } else { 1 };
        if args.len() < lead {
            self.check_only(args);
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!(
                        "`{method}` takes the circuit to run{}, then the circuit's arguments",
                        if opts { " and the options to run it with" } else { "" }
                    ))
                    .at(span),
            );
            return Checker::error(span);
        }

        // the circuit, the options, and the circuit's own arguments
        let circuit = self.expr(&args[0]);
        let Some((params, ret)) = self.circuit_signature(&circuit, method) else {
            for a in &args[1..] {
                self.expr(a);
            }
            return Checker::error(span);
        };
        let device = self.library_type_named("qpu", "Device", span);
        let run_opts = if opts {
            let o = self.expr(&args[1]);
            let t = self.library_type_named("qpu", "RunOpts", span);
            let want = self.types_mut().reference(crate::tir::Access::Const, t);
            Some(self.coerce(o, want, "the options"))
        } else {
            None
        };
        let running = self.named_entry(&circuit);
        let Some(supplied) = self.supplied(running, &params, &args[lead..], method, span) else {
            return Checker::error(span);
        };

        // a call of `qpu`'s own run, for the circuit's kernel type; running
        // changes the device's registry entry, never the handle
        let want = self.types_mut().reference(crate::tir::Access::Const, device);
        let recv = self.adjust_receiver(recv, want, method);
        let doc = self.circuit_doc(circuit);
        let kernel = kernel_of(self.types_mut(), ret);
        let mut call_args = vec![recv];
        call_args.extend(run_opts);
        call_args.extend([doc, supplied.values, supplied.circuits, supplied.registers]);
        self.library_call("qpu", run.library_function(), vec![Arg::Type(kernel)], call_args, span)
    }

    /// `c(args…)` for the circuit handle `c`: the circuit run on the device
    /// `qpu::calls()` gives, and what it gives unwrapped where it is called.
    pub(super) fn circuit_call(&mut self, circuit: Expr, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let Some((params, ret)) = self.circuit_signature(&circuit, "a call of a circuit") else {
            self.check_only(args);
            return Checker::error(span);
        };
        let running = self.named_entry(&circuit);
        let Some(supplied) = self.supplied(running, &params, args, "the circuit", span) else {
            return Checker::error(span);
        };
        let doc = self.circuit_doc(circuit);
        let kernel = kernel_of(self.types_mut(), ret);
        let run = self.library_call(
            "qpu",
            "__call",
            vec![Arg::Type(kernel)],
            vec![doc, supplied.values, supplied.circuits, supplied.registers],
            span,
        );
        self.unwrap_call(&run, "unwrap", &[], span).unwrap_or_else(|| Checker::error(span))
    }

    /// `circuit::controlled(f)`: for `f` a `circuit<fn(P…) -> R>`, a
    /// `circuit<fn(*qubit, P…) -> R>`, the qubit taken first being the one
    /// every operation is controlled on.
    pub(super) fn controlled_call(&mut self, args: &[Spanned<ast::Expr>], span: Span) -> Expr {
        let [f] = args else {
            self.check_only(args);
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message("`circuit::controlled` takes the circuit to control")
                    .at(span),
            );
            return Checker::error(span);
        };
        let circuit = self.expr(f);
        let Some((params, ret)) = self.circuit_signature(&circuit, "circuit::controlled") else {
            return Checker::error(span);
        };
        let sig = self.types_mut().function(params.clone(), ret);
        let mut with = vec![self.types_mut().reference(crate::tir::Access::Write, Ty::Qubit)];
        with.extend(params);
        let ty = self.types_mut().circuit(with, ret);
        let doc = self.library_call("circuit", "__controlled_of", vec![Arg::Type(sig)], vec![circuit], span);
        Expr::intrinsic(Intrinsic::CircuitOf, vec![doc], ty, span)
    }

    /// The parameters and result of the circuit handle `c`, or `None` once
    /// it is reported not to be one.
    fn circuit_signature(&mut self, c: &Expr, what: &str) -> Option<(Vec<Ty>, Ty)> {
        let t = self.settled(c.ty);
        if t == Ty::Never {
            return None;
        }
        if matches!(t, Ty::Circuit(_))
            && let Some((params, ret)) = self.types().as_sig(t)
        {
            return Some((params.to_vec(), ret));
        }
        let have = self.describe(t);
        self.report(
            Diagnostic::new(Code::Es06)
                .with_message(format!("`{what}` runs a circuit handle, but this is {have}"))
                .at(c.span)
                .with_note("a classical unit holds a quantum unit's `[entry]` operator as a circuit handle, `circuit<Sig>`"),
        );
        None
    }

    /// The interface of the imported unit, and the name of its `[entry]`
    /// operator, that the circuit handle `e` names directly.
    fn named_entry(&self, e: &Expr) -> Option<(&'a Interface, Symbol)> {
        let ExprKind::Global(g) = e.kind else { return None };
        let global = &self.cx.unit.globals[g.index()];
        let crate::tir::GlobalKind::Imported(unit) = &global.kind else { return None };
        let available: &'a [Interface] = self.cx.available;
        let iface = available.iter().find(|i| &i.unit == unit)?;
        iface.circuit(global.name)?;
        Some((iface, global.name))
    }

    /// Refuses the operator `e` supplied for the operator-typed parameter
    /// numbered `slot` of the entry operator `running`, when both are known
    /// here and the entry needs a monic operator there that the one supplied
    /// is not (`EQ02`); otherwise the run checks it.
    fn check_supplied(&mut self, running: Option<(&'a Interface, Symbol)>, slot: usize, e: &Expr) {
        let Some((iface, name)) = running else { return };
        if iface.circuit(name).is_none_or(|c| c.monic_slots.get(slot) != Some(&true)) {
            return;
        }
        let Some((given_iface, given)) = self.named_entry(e) else { return };
        let text = self.interner.resolve(given);
        let Some(p) = given_iface.record(text) else { return };
        if p.record.monic != crate::judge::record::Tri::No {
            return;
        }
        let entry = self.interner.resolve(name);
        self.report(
            Diagnostic::new(Code::Eq02)
                .with_message(format!(
                    "`{text}` measures, forgets or lifts, and `{entry}` needs the operator supplied here monic"
                ))
                .at(e.span)
                .with_note(format!(
                    "`{entry}` undoes what the operator does to an ancilla, to give the ancilla back in |0>; \
                     an operator that measures, forgets or lifts cannot be undone"
                ))
                .with_help("supply an operator that measures, forgets and lifts nothing"),
        );
    }

    /// The arguments `args` for the parameters `params` of the circuit run,
    /// sorted into the values, circuits and registers a run takes;
    /// `running` is the entry operator it is, when the handle names one.
    fn supplied(
        &mut self,
        running: Option<(&'a Interface, Symbol)>,
        params: &[Ty],
        args: &[Spanned<ast::Expr>],
        what: &str,
        span: Span,
    ) -> Option<Supplied> {
        if args.len() != params.len() {
            self.check_only(args);
            self.report(
                Diagnostic::new(Code::Es07)
                    .with_message(format!(
                        "the circuit takes {} arguments, but {what} gives it {}",
                        params.len(),
                        args.len()
                    ))
                    .at(span)
                    .with_note(
                        "a run takes the circuit's operator's arguments in its order: a value for each \
                         classical parameter, a circuit handle for each operator-typed one, and a \
                         `qpu::RegHandle` for each quantum one",
                    ),
            );
            return None;
        }

        // each argument, sorted by the kind of parameter it is for
        let register = self.library_type_named("qpu", "RegHandle", span);
        let text = self.types_mut().string();
        let (mut values, mut circuits, mut registers) = (Vec::new(), Vec::new(), Vec::new());
        for (a, &p) in args.iter().zip(params) {
            let quantum = self.types().is_quantum(p)
                || self.types().as_ref(p).is_some_and(|(_, x)| self.types().is_quantum(x));
            if quantum {
                // a register, by its handle
                let e = self.expr(a);
                registers.push(self.coerce(e, register, "a register for a quantum parameter"));
            } else if matches!(p, Ty::Fn(_))
                && let Some((ps, r)) = self.types().as_sig(p)
            {
                // an operator, by the document of a circuit with its signature
                let ps = ps.to_vec();
                let want = self.types_mut().circuit(ps, r);
                let e = self.expr_for(a, want);
                self.check_supplied(running, circuits.len(), &e);
                let e = self.coerce(e, want, "the circuit supplied for an operator-typed parameter");
                circuits.push(self.circuit_doc(e));
            } else {
                // a classical value, dynamically typed
                let e = self.expr_for(a, p);
                let e = self.coerce(e, p, "an argument of the circuit");
                if !self.cx.need_table(p, a.span) {
                    return None;
                }
                values.push(Expr::intrinsic(Intrinsic::DynOf(p), vec![e], Ty::Dyn, a.span));
            }
        }

        // each kind as a growable array
        let list = |me: &mut Self, items: Vec<Expr>, elem: Ty| {
            let n = items.len() as u64;
            let fixed = me.types_mut().array(elem, n);
            let lit = Expr {
                kind: ExprKind::ArrayLit(items),
                ty: fixed,
                span,
            };
            let growable = me.types_mut().growable(elem);
            me.coerce(lit, growable, "the run's arguments")
        };
        Some(Supplied {
            values: list(self, values, Ty::Dyn),
            circuits: list(self, circuits, text),
            registers: list(self, registers, register),
        })
    }

    /// The document of the circuit the handle `c` refers to.
    fn circuit_doc(&mut self, c: Expr) -> Expr {
        let span = c.span;
        let ty = self.types_mut().string();
        Expr::intrinsic(Intrinsic::CircuitDoc, vec![c], ty, span)
    }

    /// The type `unit::name` of a library unit.
    fn library_type_named(&mut self, unit: &str, name: &str, span: Span) -> Ty {
        self.library_type(unit, name, span).unwrap_or(Ty::Never)
    }

    /// A call of the generic function `name` of the library unit `unit`,
    /// made for `targs`.
    pub(super) fn library_call(&mut self, unit: &str, name: &str, targs: Vec<Arg>, args: Vec<Expr>, span: Span) -> Expr {
        let available = self.cx.available;
        let key = available
            .iter()
            .find(|i| i.unit == unit)
            .and_then(|iface| self.cx.frame_for(iface))
            .and_then(|frame| {
                let key = GenericKey {
                    frame,
                    name: self.interner.get(name)?,
                    method: None,
                };
                self.cx.generic_src(key).is_some().then_some(key)
            });
        let Some(key) = key else {
            return self.unsupported(span, &format!("`{unit}::{name}` without the `{unit}` library"));
        };
        self.instance_call(key, targs, args, &format!("{unit}::{name}"), span)
    }
}
