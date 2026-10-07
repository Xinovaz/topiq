//! Lowering functions, statements and expressions to LLVM IR.
//!
//! # Where values live
//!
//! Every binding gets a stack slot, allocated at the top of its function's
//! entry block, and is read and written through it. The same is true of the
//! values of `if`, `match`, `loop` and the short-circuiting operators: each
//! branch stores its result into a slot, and the join point loads it. Nothing
//! here builds a phi node by hand. At `-O2` LLVM promotes the slots to
//! registers itself, and at `-O0` the IR stays a direct transcription of the
//! source, which is what someone reading `tqc emit --stage=llvm` wants.
//!
//! A structure, enumeration or array is never an LLVM value: an expression of
//! such a type is lowered *into* an address (see [`Dest`]). A structure
//! literal writes each field straight into the storage it is destined for,
//! and reading a binding of such a type copies its bytes.
//!
//! # Code after a jump
//!
//! A `return`, `break` or `continue` ends the block it is in, as does a call
//! that never returns. Code written after one is still lowered (it is well
//! formed, merely unreachable), so after every such jump lowering continues in
//! a fresh block that nothing branches to. LLVM drops those blocks. Keeping
//! the rule uniform means no part of the lowering has to ask whether it is
//! reachable.
//!
//! An expression of type `never` produces no value, and lowering it returns
//! `None`; everything that consumes an operand stops at the first `None`,
//! since whatever it was about to build could never execute.

use std::collections::HashMap;

use inkwell::IntPredicate;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::module::{Linkage as LlvmLinkage, Module};
use inkwell::values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, GlobalValue, IntValue, PointerValue};

use crate::ast::Linkage;
use crate::diag::Code;
use crate::intern::{Interner, Symbol};
use crate::source::SourceMap;
use crate::span::Span;
use crate::tir::{
    self, Block, Callee, Expr, ExprKind, FnId, FnKind, GlobalKind, Intrinsic, LogicalOp, LoopId, Stmt, Ty, UnOp,
    layout,
};

use super::abort::{self, Aborts, Runtime};
use super::text::{self, Text};
use super::{abi, mangle, types};

/// Unwraps a builder result. The builder only fails when it has not been
/// positioned in a block, which lowering always does first.
pub(super) fn ok<T>(r: Result<T, BuilderError>) -> T {
    r.expect("the builder is positioned inside a function")
}

/// A read-only global of this object alone.
pub(super) fn private_constant<'ctx>(
    module: &Module<'ctx>,
    init: impl BasicValue<'ctx>,
    name: &str,
    align: u32,
) -> GlobalValue<'ctx> {
    let init = init.as_basic_value_enum();
    let g = module.add_global(init.get_type(), None, name);
    g.set_initializer(&init);
    g.set_constant(true);
    g.set_linkage(LlvmLinkage::Private);
    g.set_alignment(align);
    g
}

/// Memory holding a value, and the alignment its address is known to have.
#[derive(Clone, Copy, Debug)]
pub struct Addr<'ctx> {
    /// The address.
    pub ptr: PointerValue<'ctx>,
    /// A power of two the address is a multiple of.
    pub align: u64,
}

/// Where an expression's value goes.
#[derive(Clone, Copy, Debug)]
pub enum Dest<'ctx> {
    /// Returned as an LLVM value: for everything held in registers.
    Value,
    /// Written at this address: for structures, enumerations and arrays.
    Mem(Addr<'ctx>),
}

/// Where the branches of an `if`, `match` or `loop` deliver its value.
#[derive(Clone, Copy, Debug)]
pub(super) enum Out<'ctx> {
    /// It has none.
    Nothing,
    /// Into a slot the join point loads from.
    Slot(Addr<'ctx>, Ty),
    /// Straight into the destination of a structure, enumeration or array.
    Mem(Addr<'ctx>),
}

/// Where `break` and `continue` go for one loop.
#[derive(Clone, Copy)]
struct LoopTarget<'ctx> {
    /// The block after the loop.
    exit: BasicBlock<'ctx>,
    /// Where the next iteration starts.
    next: BasicBlock<'ctx>,
    /// Where a `break` delivers the loop's value.
    out: Out<'ctx>,
}

/// Lowers one unit into one module.
pub struct Lowering<'ctx, 'a> {
    pub(super) cx: &'ctx Context,
    pub(super) module: &'a Module<'ctx>,
    pub(super) b: Builder<'ctx>,
    pub(super) unit: &'a tir::Unit,
    sources: &'a SourceMap,
    pub(super) interner: &'a Interner,
    pub(super) fns: Vec<Option<(FunctionValue<'ctx>, bool)>>,
    externs: Vec<(FunctionValue<'ctx>, bool)>,
    /// Closure bodies made callable without an environment.
    pub(super) thunks: HashMap<FnId, FunctionValue<'ctx>>,
    /// The code of closures made from references to functions, one for each
    /// signature.
    pub(super) trampolines: HashMap<crate::tir::SigId, FunctionValue<'ctx>>,
    /// The function destroying each type's values.
    pub(super) drop_fns: HashMap<Ty, FunctionValue<'ctx>>,
    /// The function cloning each type's values.
    pub(super) clone_fns: HashMap<Ty, FunctionValue<'ctx>>,
    /// Tables this unit refers to without making them, whose references
    /// fall back on the blank table when no unit makes them.
    pub(super) weak_tables: Vec<String>,
    /// What destruction needs while the current function is lowered.
    pub(super) drops: super::drop::Drops<'ctx>,
    /// The types of the current function's bindings.
    pub(super) local_types: Vec<Ty>,
    /// The constant slots references to functions point at.
    pub(super) fn_slots: HashMap<super::closure::FnKey, GlobalValue<'ctx>>,
    globals: Vec<Option<GlobalValue<'ctx>>>,
    pub(super) aborts: Aborts<'ctx>,
    text: Text<'ctx>,
    rt: Runtime,
    strings: HashMap<Symbol, (GlobalValue<'ctx>, u64)>,
    constants: u32,
    pub(super) func: Option<FunctionValue<'ctx>>,
    /// The Topiq function being lowered.
    current_fn: Option<FnId>,
    ret_slot: Option<Addr<'ctx>>,
    pub(super) locals: Vec<Option<Addr<'ctx>>>,
    loops: HashMap<LoopId, LoopTarget<'ctx>>,
}

impl<'ctx, 'a> Lowering<'ctx, 'a> {
    /// Prepares to lower `unit` into `module`, defining the helpers every
    /// object carries.
    pub fn new(
        cx: &'ctx Context,
        module: &'a Module<'ctx>,
        unit: &'a tir::Unit,
        sources: &'a SourceMap,
        interner: &'a Interner,
        rt: Runtime,
    ) -> Lowering<'ctx, 'a> {
        Lowering {
            cx,
            module,
            b: cx.create_builder(),
            unit,
            sources,
            interner,
            fns: vec![None; unit.fns.len()],
            externs: Vec::new(),
            thunks: HashMap::new(),
            trampolines: HashMap::new(),
            drop_fns: HashMap::new(),
            clone_fns: HashMap::new(),
            weak_tables: Vec::new(),
            drops: super::drop::Drops::default(),
            local_types: Vec::new(),
            fn_slots: HashMap::new(),
            globals: vec![None; unit.globals.len()],
            aborts: Aborts::define(cx, module, rt),
            text: Text::new(rt),
            rt,
            strings: HashMap::new(),
            constants: 0,
            func: None,
            current_fn: None,
            ret_slot: None,
            locals: Vec::new(),
            loops: HashMap::new(),
        }
    }

    /// The LLVM function of a Topiq function, once declared. A constant
    /// function has none.
    pub fn function(&self, id: FnId) -> Option<FunctionValue<'ctx>> {
        self.fns[id.index()].map(|(f, _)| f)
    }

    /// The layout of a type.
    pub(super) fn layout(&self, ty: Ty) -> layout::Layout {
        layout::of(&self.unit.types, ty)
    }

    /// Declares every compiled function, every function called from another
    /// unit, and every object with storage, so that bodies may refer to any of
    /// them regardless of declaration order.
    pub fn declare(&mut self) {
        let unit_name = self.unit.name.as_str();
        for (i, g) in self.unit.globals.iter().enumerate() {
            // a constant has no storage; every use was folded
            if g.constant || !g.ty.is_storable() {
                continue;
            }
            let name = self.interner.resolve(g.name);
            let align = self.layout(g.ty).align as u32;
            let global = match &g.kind {
                GlobalKind::Imported(from) => {
                    let ty = types::storage(self.cx, &self.unit.types, g.ty);
                    let symbol = g.symbol.clone().unwrap_or_else(|| mangle::global(from, name));
                    let global = self.module.add_global(ty, None, &symbol);
                    global.set_linkage(LlvmLinkage::External);
                    global
                }
                kind => {
                    let (symbol, linkage) = match kind {
                        GlobalKind::Persist(f) => (
                            mangle::persist(unit_name, self.interner.resolve(self.unit.func(*f).name), name, i as u32),
                            Linkage::Unit,
                        ),
                        _ => (
                            g.symbol.clone().unwrap_or_else(|| mangle::global(unit_name, name)),
                            g.linkage,
                        ),
                    };
                    // an object the unit's initialiser gives its value starts
                    // as zero bytes, and is written once before `main`
                    let init = match &g.value {
                        Some(value) => self.constant(value, g.ty),
                        None if g.startup => types::storage(self.cx, &self.unit.types, g.ty).const_zero(),
                        None => unreachable!("analysis folds every initial value"),
                    };
                    let global = self.module.add_global(init.get_type(), None, &symbol);
                    global.set_initializer(&init);
                    global.set_linkage(if matches!(kind, GlobalKind::Persist(_)) {
                        llvm_linkage(linkage)
                    } else {
                        self.linkage(linkage)
                    });
                    global.set_constant(g.constant && !g.startup);
                    global
                }
            };
            global.set_alignment(align);
            self.globals[i] = Some(global);
        }
        for (i, f) in self.unit.fns.iter().enumerate() {
            if f.kind != FnKind::Runtime {
                continue;
            }
            let name = self.interner.resolve(f.name);
            // an instance is one definition however many units make it: its
            // symbol names the generic's own unit and qualifies every argument
            // type, so it is the same everywhere and the linker keeps one copy
            let args: Vec<String> = f
                .args
                .iter()
                .map(|&a| match a {
                    tir::Arg::Type(t) => self.unit.types.qualified(t, self.interner, unit_name),
                    tir::Arg::Const(v) => v.to_string(),
                })
                .collect();
            let symbol = if let Some(sym) = &f.attrs.symbol {
                sym.clone()
            } else if f.attrs.hidden {
                // its name may be another function's too, as two blocks may
                // each declare a `helper`; `$$` never occurs in a symbol a
                // program's name makes
                mangle::function(unit_name, &format!("{name}$${i}"))
            } else if f.closure {
                // nothing names a closure's body, so its symbol need only be
                // unique within the unit
                mangle::function(unit_name, &format!("closure${i}"))
            } else if let Some(m) = f.method {
                let owner = self.unit.types.qualified(Ty::Adt(m.owner), self.interner, unit_name);
                mangle::method(&owner, &method_name(name, m), &args.join(","))
            } else if f.args.is_empty() {
                mangle::function(unit_name, name)
            } else {
                mangle::instance(f.origin.as_deref().unwrap_or(unit_name), name, &args.join(","))
            };
            let params: Vec<Ty> = f.param_types().collect();
            let sig = abi::signature(self.cx, &self.unit.types, &params, f.ret);
            let linkage = if f.args.is_empty() {
                self.linkage(f.linkage)
            } else {
                LlvmLinkage::LinkOnceODR
            };
            let func = self.module.add_function(&symbol, sig.llvm, Some(linkage));
            if let Some(hint) = f.attrs.inline {
                let name = match hint {
                    tir::Inline::Hint => "inlinehint",
                    tir::Inline::Never => "noinline",
                };
                let attr = super::abort::enum_attribute(self.cx, name);
                func.add_attribute(inkwell::attributes::AttributeLoc::Function, attr);
            }
            if !f.args.is_empty() {
                let comdat = self.module.get_or_insert_comdat(&symbol);
                comdat.set_selection_kind(inkwell::comdat::ComdatSelectionKind::Any);
                func.as_global_value().set_comdat(comdat);
            }
            self.fns[i] = Some((func, sig.sret));
        }
        for x in &self.unit.externs {
            let name = self.interner.resolve(x.name);
            let symbol = match x.method {
                _ if x.symbol.is_some() => x.symbol.clone().expect("just tested"),
                Some(m) => {
                    let owner = self.unit.types.qualified(Ty::Adt(m.owner), self.interner, unit_name);
                    mangle::method(&owner, &method_name(name, m), "")
                }
                None => mangle::function(&x.unit, name),
            };
            let sig = abi::signature(self.cx, &self.unit.types, &x.params, x.ret);
            let func = self
                .module
                .get_function(&symbol)
                .unwrap_or_else(|| self.module.add_function(&symbol, sig.llvm, Some(LlvmLinkage::External)));
            self.externs.push((func, sig.sret));
        }
    }

    /// The LLVM linkage of one of the unit's own functions or objects. A
    /// `static` one stays private to the object unless other units may make
    /// instances of the unit's generic functions, whose bodies may use it.
    fn linkage(&self, l: Linkage) -> LlvmLinkage {
        if self.unit.shares_bodies {
            LlvmLinkage::External
        } else {
            llvm_linkage(l)
        }
    }

    /// A constant's initial value in the form its storage takes: a scalar
    /// constant, or the exact bytes of an aggregate.
    fn constant(&mut self, v: &tir::Value, ty: Ty) -> BasicValueEnum<'ctx> {
        if ty.is_aggregate() {
            let mut buffers = HashMap::new();
            self.prepare_parts(v, ty, &mut buffers);
            let mut lookup = self.parts_lookup(&buffers);
            types::const_blob(self.cx, &self.unit.types, v, ty, &mut lookup)
        } else {
            self.scalar_constant(v, ty).expect("a stored scalar has a value")
        }
    }

    /// A constant in register form, with whatever it points to placed in the
    /// image; `None` for `void`.
    fn scalar_constant(&mut self, v: &tir::Value, ty: Ty) -> Option<BasicValueEnum<'ctx>> {
        let mut buffers = HashMap::new();
        self.prepare_parts(v, ty, &mut buffers);
        let mut lookup = self.parts_lookup(&buffers);
        types::const_scalar(self.cx, v, &mut lookup)
    }

    /// Finds a constant's parts once [`Self::prepare_parts`] has placed them.
    fn parts_lookup<'b>(
        &'b self,
        buffers: &'b HashMap<usize, (PointerValue<'ctx>, u64)>,
    ) -> impl FnMut(types::Part<'_>) -> (PointerValue<'ctx>, u64) + 'b {
        move |p| match p {
            types::Part::Str(s) => {
                let (g, n) = self.strings[&s];
                (g.as_pointer_value(), n)
            }
            types::Part::Elements(items) => buffers[&(items.as_ptr() as usize)],
        }
    }

    /// Places in the image the characters of every string in `v`, and the
    /// buffer of every growable array, keyed by where its elements are held.
    /// An array's own elements are placed before it.
    fn prepare_parts(&mut self, v: &tir::Value, ty: Ty, buffers: &mut HashMap<usize, (PointerValue<'ctx>, u64)>) {
        let types = &self.unit.types;
        let (parts, tys): (&[tir::Value], Vec<Ty>) = match (v, ty) {
            (tir::Value::Str(s), _) => {
                self.string(*s);
                return;
            }
            (tir::Value::Struct(fs), Ty::Adt(id)) => (fs, types.adt(id).fields().iter().map(|f| f.ty).collect()),
            (tir::Value::Struct(fs), Ty::Tuple(_)) => (fs, types.as_tuple(ty).map(<[Ty]>::to_vec).unwrap_or_default()),
            (tir::Value::Enum { variant, fields }, Ty::Adt(id)) => {
                (fields, types.adt(id).variants()[*variant as usize].fields.iter().map(|f| f.ty).collect())
            }
            (tir::Value::Array(items), _) => {
                let growable = types.as_growable(ty);
                let Some(elem) = growable.or_else(|| types.as_array(ty).map(|(e, _)| e)) else {
                    return;
                };
                for item in items {
                    self.prepare_parts(item, elem, buffers);
                }
                if growable.is_some() {
                    let placed = self.constant_buffer(items, elem, buffers);
                    buffers.insert(items.as_ptr() as usize, placed);
                }
                return;
            }
            _ => return,
        };
        for (f, t) in parts.iter().zip(tys) {
            self.prepare_parts(f, t, buffers);
        }
    }

    /// Whether `e` is a constant, or a field or element of one, whose
    /// buffers are in the image.
    fn part_of_constant(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Const(_) => true,
            ExprKind::Global(g) => self.unit.global(*g).constant,
            ExprKind::Field { base, .. } | ExprKind::Index { base, .. } => self.part_of_constant(base),
            _ => false,
        }
    }

    /// A growable array's buffer placed in the image, header and all: the
    /// address of its first element, and how many there are.
    fn constant_buffer(
        &mut self,
        items: &[tir::Value],
        elem: Ty,
        buffers: &HashMap<usize, (PointerValue<'ctx>, u64)>,
    ) -> (PointerValue<'ctx>, u64) {
        let n = items.len() as u64;
        if n == 0 {
            return (self.null(), 0);
        }
        let elements = {
            let mut lookup = self.parts_lookup(buffers);
            types::const_elements(self.cx, &self.unit.types, items, elem, &mut lookup)
        };
        let i64 = self.cx.i64_type();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let init = self.cx.const_struct(&[i64.const_int(n, false).into(), ptr.const_null().into(), elements], true);
        let g = private_constant(self.module, init, &format!("buffer.{}", self.constants), 16);
        self.constants += 1;
        // SAFETY: the elements follow the two-word header
        let data = unsafe {
            g.as_pointer_value()
                .const_gep(self.cx.i8_type(), &[i64.const_int(super::growable::HEADER, false)])
        };
        (data, n)
    }

    /// The characters of a string literal, placed once in the image as
    /// four-byte scalar values, and how many there are.
    pub(super) fn string(&mut self, s: Symbol) -> (PointerValue<'ctx>, u64) {
        if let Some(&(g, n)) = self.strings.get(&s) {
            return (g.as_pointer_value(), n);
        }
        let text = self.interner.resolve(s);
        let i32 = self.cx.i32_type();
        let chars: Vec<IntValue<'ctx>> = text.chars().map(|c| i32.const_int(u64::from(u32::from(c)), false)).collect();
        let n = chars.len() as u64;
        let g = private_constant(self.module, i32.const_array(&chars), &format!("string.{}", self.strings.len()), 4);
        g.set_unnamed_addr(true);
        self.strings.insert(s, (g, n));
        (g.as_pointer_value(), n)
    }

    /// A read-only copy of a constant aggregate in the image, to read or copy
    /// from.
    pub(super) fn constant_in_memory(&mut self, v: &tir::Value, ty: Ty) -> Addr<'ctx> {
        let init = self.constant(v, ty);
        let align = self.layout(ty).align;
        let g = private_constant(self.module, init, &format!("constant.{}", self.constants), align as u32);
        self.constants += 1;
        g.set_unnamed_addr(true);
        Addr {
            ptr: g.as_pointer_value(),
            align,
        }
    }

    /// The address of an object's storage.
    pub(super) fn global_addr(&self, id: tir::GlobalId) -> Addr<'ctx> {
        let g = self.globals[id.index()].expect("an object with storage");
        Addr {
            ptr: g.as_pointer_value(),
            align: self.layout(self.unit.global(id).ty).align,
        }
    }

    /// Lowers every compiled function's body.
    pub fn define(&mut self) {
        for i in 0..self.unit.fns.len() {
            if self.fns[i].is_some() {
                self.lower_fn(FnId(i as u32));
            }
        }
    }

    fn lower_fn(&mut self, id: FnId) {
        let f = self.unit.func(id);
        let (func, sret) = self.fns[id.index()].expect("declared");
        self.func = Some(func);
        self.current_fn = Some(id);
        self.loops.clear();
        let entry = self.cx.append_basic_block(func, "entry");
        self.b.position_at_end(entry);

        let first = usize::from(sret);
        self.ret_slot = sret.then(|| Addr {
            ptr: func.get_first_param().expect("the result's address").into_pointer_value(),
            align: self.layout(f.ret).align,
        });
        self.locals = vec![None; f.locals.len()];
        let param_index: HashMap<tir::LocalId, usize> = f.params.iter().enumerate().map(|(i, &p)| (p, i + first)).collect();
        for (i, l) in f.locals.iter().enumerate() {
            let id = tir::LocalId(i as u32);
            if !l.ty.is_storable() {
                continue;
            }
            match param_index.get(&id) {
                // the caller made a copy for this call; it is the parameter's
                // storage
                Some(&k) if l.ty.is_aggregate() => {
                    let p = func.get_nth_param(k as u32).expect("one LLVM parameter per parameter");
                    self.locals[i] = Some(Addr {
                        ptr: p.into_pointer_value(),
                        align: self.layout(l.ty).align,
                    });
                }
                Some(&k) => {
                    let slot = self.slot_here(l.ty, self.interner.resolve(l.name));
                    let p = func.get_nth_param(k as u32).expect("one LLVM parameter per parameter");
                    self.store_scalar(p, slot);
                    self.locals[i] = Some(slot);
                }
                None => {
                    let slot = self.slot_here(l.ty, self.interner.resolve(l.name));
                    self.locals[i] = Some(slot);
                }
            }
        }

        let dest = match self.ret_slot {
            Some(a) => Dest::Mem(a),
            None => Dest::Value,
        };
        // the parameters are the function's own, destroyed as it returns
        self.local_types = f.locals.iter().map(|l| l.ty).collect();
        self.prepare_drops(f);
        for &p in &f.params {
            self.own(p);
        }
        let value = self.block(&f.body, dest);
        if !self.terminated() {
            self.destroy_from(0);
        }
        if !self.terminated() {
            match value {
                Some(v) if self.ret_slot.is_none() && types::scalar(self.cx, &self.unit.types, f.ret).is_some() => {
                    ok(self.b.build_return(Some(&v)));
                }
                _ if f.ret == Ty::Void || self.ret_slot.is_some() => {
                    ok(self.b.build_return(None));
                }
                // control cannot reach here: the body diverged, and this is
                // the fresh block lowering continued in afterwards
                _ => {
                    ok(self.b.build_unreachable());
                }
            }
        }
        self.func = None;
        self.ret_slot = None;
    }

    /// Starts lowering the body of the helper `f` in its entry block, giving
    /// back where lowering was, for [`Self::leave_fn`].
    pub(super) fn enter_fn(&mut self, f: FunctionValue<'ctx>) -> (Option<BasicBlock<'ctx>>, Option<FunctionValue<'ctx>>) {
        let saved = (self.b.get_insert_block(), self.func.replace(f));
        self.b.position_at_end(self.cx.append_basic_block(f, "entry"));
        saved
    }

    /// Goes back to where lowering was before [`Self::enter_fn`].
    pub(super) fn leave_fn(&mut self, (block, func): (Option<BasicBlock<'ctx>>, Option<FunctionValue<'ctx>>)) {
        self.func = func;
        if let Some(bb) = block {
            self.b.position_at_end(bb);
        }
    }

    /// The line a span starts on.
    pub(super) fn line(&self, span: Span) -> u32 {
        self.sources
            .get(span.source)
            .map_or(0, |f| f.line_col(span.start).0)
    }

    /// Branches to an abort with `code` when `bad` holds, and continues lowering
    /// on the path where it does not.
    pub(super) fn abort_if(&mut self, bad: IntValue<'ctx>, code: Code, span: Span) {
        let func = self.func.expect("inside a function");
        let fail = self.cx.append_basic_block(func, &format!("abort.{code}"));
        let pass = self.cx.append_basic_block(func, "checked");
        ok(self.b.build_conditional_branch(bad, fail, pass));

        self.b.position_at_end(fail);
        let text = abort::line(code, &self.unit.name, self.line(span));
        let message = self.aborts.message(&self.b, &text);
        let len = self.cx.i32_type().const_int(text.len() as u64, false);
        ok(self.b.build_call(
            self.aborts.helper(),
            &[message.as_pointer_value().into(), len.into()],
            "",
        ));
        ok(self.b.build_unreachable());

        self.b.position_at_end(pass);
    }

    /// Whether the current block already ends in a jump.
    pub(super) fn terminated(&self) -> bool {
        self.b
            .get_insert_block()
            .is_some_and(|bb| bb.get_terminator().is_some())
    }

    /// Continues in a fresh block that nothing jumps to.
    pub(super) fn dead(&mut self) {
        let func = self.func.expect("inside a function");
        let bb = self.cx.append_basic_block(func, "unreachable");
        self.b.position_at_end(bb);
    }

    /// Storage for a value of `ty` at the top of the entry block, where LLVM
    /// expects stack slots.
    pub(super) fn temp(&self, ty: Ty) -> Addr<'ctx> {
        self.named_temp(ty, "tmp")
    }

    /// A binding's slot, allocated where the builder is (which, while a
    /// function's bindings are being given slots, is the end of its entry
    /// block), so the slots appear in the order the bindings were declared.
    fn slot_here(&self, ty: Ty, name: &str) -> Addr<'ctx> {
        let ptr = ok(self.b.build_alloca(types::storage(self.cx, &self.unit.types, ty), name));
        self.aligned_slot(ptr, ty)
    }

    /// The stack slot `ptr` for a value of `ty`.
    fn aligned_slot(&self, ptr: PointerValue<'ctx>, ty: Ty) -> Addr<'ctx> {
        let align = self.layout(ty).align;
        if let Some(i) = ptr.as_instruction() {
            let _ = i.set_alignment(align as u32);
        }
        Addr { ptr, align }
    }

    /// Stack storage of an LLVM type, at the top of the entry block.
    pub(super) fn entry_alloca(&self, llvm: inkwell::types::BasicTypeEnum<'ctx>, name: &str) -> PointerValue<'ctx> {
        let func = self.func.expect("inside a function");
        let entry = func.get_first_basic_block().expect("the entry block exists");
        let at = self.cx.create_builder();
        match entry.get_first_instruction() {
            Some(first) => at.position_before(&first),
            None => at.position_at_end(entry),
        }
        ok(at.build_alloca(llvm, name))
    }

    /// Storage for a value of `ty`, named for the IR.
    pub(super) fn named_temp(&self, ty: Ty, name: &str) -> Addr<'ctx> {
        let ptr = self.entry_alloca(types::storage(self.cx, &self.unit.types, ty), name);
        self.aligned_slot(ptr, ty)
    }

    /// Lowers an expression of a type held in registers, returning its value,
    /// or `None` for one of type `void` or `never`.
    pub(super) fn value(&mut self, e: &Expr) -> Option<BasicValueEnum<'ctx>> {
        self.eval(e, Dest::Value)
    }

    /// Lowers an expression of any type into memory at `a`.
    pub(super) fn store_expr(&mut self, e: &Expr, a: Addr<'ctx>) {
        if e.ty.is_aggregate() {
            self.eval(e, Dest::Mem(a));
        } else if let Some(v) = self.value(e) {
            self.store_scalar(v, a);
        }
    }

    /// Moves the value of type `ty` at `src` to `dst`.
    pub(super) fn move_bytes(&mut self, dst: Addr<'ctx>, src: Addr<'ctx>, ty: Ty) {
        if ty.is_aggregate() {
            self.copy(dst, src, ty);
        } else if let Some(v) = self.load_scalar(ty, src) {
            self.store_scalar(v, dst);
        }
    }

    /// Lowers an expression for its effect alone. A value that owns
    /// something, which nothing keeps, is destroyed at once.
    fn discard(&mut self, e: &Expr) {
        if self.needs_drop(e.ty) {
            let t = self.temp(e.ty);
            self.store_expr(e, t);
            self.drop_at(t, e.ty);
            return;
        }
        if e.ty.is_aggregate() {
            let t = self.temp(e.ty);
            self.eval(e, Dest::Mem(t));
        } else {
            self.value(e);
        }
    }

    /// Lowers an expression, delivering its value to `dest`.
    pub(super) fn eval(&mut self, e: &Expr, dest: Dest<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        let span = e.span;
        if let (Dest::Value, true) = (dest, e.ty.is_aggregate()) {
            self.discard(e);
            return None;
        }
        match &e.kind {
            // a constant that owns memory has its buffers in the image, which
            // nothing may free: each use of it by value is a deep copy
            ExprKind::Const(v) if self.needs_drop(e.ty) => {
                let src = self.constant_in_memory(v, e.ty);
                self.clone_to(dest, src, e.ty)
            }
            ExprKind::Const(v) => {
                if e.ty.is_aggregate() {
                    let src = self.constant_in_memory(v, e.ty);
                    self.copy_to(dest, src, e.ty);
                    None
                } else {
                    self.scalar_constant(v, e.ty)
                }
            }
            // likewise for a constant's part, or a constant kept in storage
            ExprKind::Global(_) | ExprKind::Field { .. } | ExprKind::Index { .. }
                if self.needs_drop(e.ty) && self.part_of_constant(e) =>
            {
                let src = self.addr(e)?;
                self.clone_to(dest, src, e.ty)
            }
            ExprKind::Local(_) | ExprKind::Global(_) | ExprKind::Field { .. } | ExprKind::Index { .. } | ExprKind::Deref(_) => {
                let src = self.addr(e)?;
                // reading a value that owns something moves it
                let r = if e.ty.is_aggregate() {
                    self.copy_to(dest, src, e.ty);
                    None
                } else {
                    self.load_scalar(e.ty, src)
                };
                self.moved(e);
                r
            }
            ExprKind::Call { callee, args } => self.call(*callee, args, e.ty, dest),
            ExprKind::FnRef(c) => Some(self.fn_pointer(super::closure::FnKey::Callee(*c)).into()),
            ExprKind::ThunkRef(id) => Some(self.fn_pointer(super::closure::FnKey::Thunk(*id)).into()),
            ExprKind::IndirectCall { callee, args } => self.indirect_call(callee, args, e.ty, dest),
            ExprKind::Closure { code, captures } => self.make_closure(*code, captures, span),
            ExprKind::FnAsClosure(f) => self.fn_as_closure(f, e.ty),
            ExprKind::Intrinsic { which, args } => self.intrinsic(*which, args, e.ty, dest, span),
            ExprKind::Growable { op, array, args } => self.growable_op(*op, array, args, e.ty, dest, span),
            ExprKind::Grow(x) => self.grow(x, span),
            ExprKind::Unary { op, operand } if operand.ty.is_float() => {
                let v = self.value(operand)?.into_float_value();
                match op {
                    UnOp::Neg => Some(ok(self.b.build_float_neg(v, "")).into()),
                    other => unreachable!("`{}` on a floating value", other.text()),
                }
            }
            ExprKind::Unary { op, operand } => {
                let v = self.value(operand)?.into_int_value();
                Some(match (op, operand.ty) {
                    (UnOp::Neg, Ty::Int(t)) => self.neg(t, v, span).into(),
                    (UnOp::Not, _) => ok(self.b.build_not(v, "")).into(),
                    (UnOp::Neg, other) => unreachable!("negation of {other:?}"),
                })
            }
            ExprKind::Binary { op, lhs, rhs } if lhs.ty.is_reference() => {
                Some(self.reference_eq(*op == tir::BinOp::Eq, lhs, rhs)?.into())
            }
            ExprKind::Binary { op, lhs, rhs } if lhs.ty.is_float() => {
                let l = self.value(lhs)?.into_float_value();
                let r = self.value(rhs)?.into_float_value();
                Some(self.float_binary(*op, l, r))
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.value(lhs)?.into_int_value();
                let r = self.value(rhs)?.into_int_value();
                Some(self.binary(*op, lhs.ty, rhs.ty, l, r, span).into())
            }
            ExprKind::Logical { op, lhs, rhs } => self.logical(*op, lhs, rhs),
            ExprKind::Cast { expr, to } => {
                let v = self.value(expr)?;
                Some(self.convert(expr.ty, *to, v, span))
            }
            ExprKind::Ref(x) => self.reference(x, e.ty),
            ExprKind::StructLit { .. } | ExprKind::Variant { .. } | ExprKind::ArrayLit(_) | ExprKind::ArrayRepeat { .. } => {
                let Dest::Mem(a) = dest else {
                    unreachable!("an aggregate is always built in memory");
                };
                self.construct(e, a);
                None
            }
            ExprKind::Coerce(x) => match self.erase_counted(x, e.ty) {
                Some(done) => done,
                None => self.eval(x, dest),
            },
            ExprKind::Assign { place, value } => {
                self.assign(place, value);
                None
            }
            ExprKind::Block(b) => self.block(b, dest),
            ExprKind::If { cond, then, els } => self.if_expr(cond, then, els.as_deref(), e.ty, dest),
            ExprKind::Match { scrutinee, arms } => self.match_expr(scrutinee, arms, e.ty, dest),
            ExprKind::Loop { id, body } => self.loop_expr(*id, body, e.ty, dest),
            ExprKind::While { id, cond, body } => {
                self.while_expr(*id, cond, body);
                None
            }
            ExprKind::ForRange {
                id,
                var,
                start,
                end,
                inclusive,
                body,
            } => {
                self.for_range(*id, *var, start, end, *inclusive, body);
                None
            }
            ExprKind::Break { target, value } => {
                let t = self.loops[target];
                if let Some(v) = value {
                    self.deliver(t.out, v);
                    if self.terminated() {
                        return None;
                    }
                }
                self.leave_loop_early(*target);
                ok(self.b.build_unconditional_branch(t.exit));
                self.dead();
                None
            }
            ExprKind::Continue { target } => {
                let t = self.loops[target];
                self.leave_loop_early(*target);
                ok(self.b.build_unconditional_branch(t.next));
                self.dead();
                None
            }
            ExprKind::Quantum { .. } => {
                unreachable!("only quantum units act on quantum state, and they generate circuits, not code")
            }
            // the value leaves first; then everything the function owns is
            // destroyed
            ExprKind::Return(value) => {
                let r = match (value, self.ret_slot) {
                    (Some(v), Some(slot)) => {
                        self.store_expr(v, slot);
                        None
                    }
                    (Some(v), None) => self.value(v),
                    (None, _) => None,
                };
                if !self.terminated() {
                    self.destroy_from(0);
                    ok(self.b.build_return(r.as_ref().map(|v| v as &dyn BasicValue)));
                }
                self.dead();
                None
            }
        }
    }

    /// Copies an aggregate from `src` to where `dest` says.
    fn copy_to(&mut self, dest: Dest<'ctx>, src: Addr<'ctx>, ty: Ty) {
        if let Dest::Mem(d) = dest {
            self.copy(d, src, ty);
        }
    }

    /// `place = value`: the value first, then the place.
    fn assign(&mut self, place: &Expr, value: &Expr) {
        // the initialiser's one assignment to an object declared without a
        // value finds zero bytes, which are nothing to destroy
        let first = matches!(place.kind, ExprKind::Global(g) if self.unit.global(g).startup)
            && self.current_fn.is_some()
            && self.current_fn == self.unit.init;
        if self.needs_drop(place.ty) && !first {
            // the value the place held is destroyed before the new one goes
            // in
            let t = self.temp(place.ty);
            self.store_expr(value, t);
            if self.terminated() {
                return;
            }
            let Some(dst) = self.addr(place) else { return };
            match super::drop::local_path(place) {
                Some((l, p)) => {
                    self.drop_path(l, p.clone(), dst, place.ty);
                    self.move_bytes(dst, t, place.ty);
                    self.set_flags(l, &p, true);
                }
                None => {
                    self.drop_at(dst, place.ty);
                    self.move_bytes(dst, t, place.ty);
                }
            }
            return;
        }
        if place.ty.is_aggregate() {
            // built aside first, so that a value reading the place it is
            // assigned to sees the old contents throughout
            let t = self.temp(place.ty);
            self.store_expr(value, t);
            if let Some(dst) = self.addr(place) {
                self.copy(dst, t, place.ty);
            }
        } else if let Some(v) = self.value(value)
            && let Some(dst) = self.addr(place)
        {
            self.store_scalar(v, dst);
        }
    }

    /// The LLVM function a callee is, and whether it returns through a
    /// hidden address.
    pub(super) fn callee_function(&self, callee: Callee) -> (FunctionValue<'ctx>, bool) {
        match callee {
            Callee::Fn(id) => self.fns[id.index()].expect("constant functions were folded away"),
            Callee::Extern(id) => self.externs[id.index()],
        }
    }

    /// The values a call passes: `prefix`, then the arguments, each an
    /// aggregate by the address of a copy the callee owns for the call.
    /// `None` if an argument does not finish.
    pub(super) fn arguments(
        &mut self,
        mut values: Vec<BasicMetadataValueEnum<'ctx>>,
        args: &[Expr],
    ) -> Option<Vec<BasicMetadataValueEnum<'ctx>>> {
        for a in args {
            if a.ty.is_aggregate() {
                let t = self.temp(a.ty);
                self.store_expr(a, t);
                values.push(t.ptr.into());
            } else {
                values.push(self.value(a)?.into());
            }
        }
        if self.terminated() {
            return None;
        }
        Some(values)
    }

    /// A call. A result that lives in memory is built at `dest`, or in a
    /// temporary when the value is not wanted.
    fn call(&mut self, callee: Callee, args: &[Expr], ty: Ty, dest: Dest<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        let (f, sret) = self.callee_function(callee);
        let mut prefix: Vec<BasicMetadataValueEnum<'ctx>> = Vec::with_capacity(args.len() + 1);
        if sret {
            let out = match dest {
                Dest::Mem(a) => a,
                Dest::Value => self.temp(ty),
            };
            prefix.push(out.ptr.into());
        }
        let values = self.arguments(prefix, args)?;
        let r = ok(self.b.build_call(f, &values, ""));
        if ty == Ty::Never {
            return self.diverged();
        }
        if sret { None } else { r.try_as_basic_value().basic() }
    }

    /// Ends the current block, which control never leaves, and continues in
    /// a fresh one; there is no value.
    pub(super) fn diverged(&mut self) -> Option<BasicValueEnum<'ctx>> {
        ok(self.b.build_unreachable());
        self.dead();
        None
    }

    /// A library function or `len`, its value delivered to `dest` where it
    /// is built in memory.
    fn intrinsic(
        &mut self,
        which: Intrinsic,
        args: &[Expr],
        ty: Ty,
        dest: Dest<'ctx>,
        span: Span,
    ) -> Option<BasicValueEnum<'ctx>> {
        let i32 = self.cx.i32_type();
        match which {
            Intrinsic::Overflowing(op) => self.overflowing_op(Some(op), args, ty, dest),
            Intrinsic::OverflowingNeg => self.overflowing_op(None, args, ty, dest),
            Intrinsic::Clone => self.clone_of(&args[0], ty, dest),
            Intrinsic::Exchange => {
                let (_, target) = self.unit.types.as_ref(args[0].ty).expect("checked: a reference");
                let align = self.layout(target).align;
                let mut at = |e: &Expr| -> Option<Addr<'ctx>> {
                    let v = self.value(e)?.into_struct_value();
                    let ptr = ok(self.b.build_extract_value(v, 0, "")).into_pointer_value();
                    Some(Addr { ptr, align })
                };
                let a = at(&args[0])?;
                let b = at(&args[1])?;
                let held = self.temp(target);
                self.copy(held, a, target);
                self.copy(a, b, target);
                self.copy(b, held, target);
                None
            }
            Intrinsic::Downcast(target) => self.downcast(&args[0], target, ty, dest),
            Intrinsic::DynAs(target) => self.dyn_as(&args[0], target, ty, dest),
            Intrinsic::DynTake => self.dyn_take(&args[0], ty, dest),
            Intrinsic::RawParts => self.raw_parts(&args[0], ty, dest),
            Intrinsic::Print | Intrinsic::Eprint => {
                let s = self.value(&args[0])?.into_struct_value();
                let (p, n) = self.slice_parts(s);
                let fd = if which == Intrinsic::Print { text::STDOUT } else { text::STDERR };
                let print = self.text.print(self.cx, self.module);
                ok(self.b.build_call(
                    print,
                    &[i32.const_int(fd, false).into(), p.into(), n.into()],
                    "",
                ));
                None
            }
            Intrinsic::Exit => {
                let code = self.value(&args[0])?;
                let (_, exit) = abort::runtime_functions(self.cx, self.module, self.rt);
                ok(self.b.build_call(exit, &[code.into()], ""));
                self.diverged()
            }
            Intrinsic::Saturating(op) => self.saturating_op(op, args),
            Intrinsic::Truncate => self.truncate_to(&args[0], ty),
            Intrinsic::SaturatingAs => self.saturate_to(&args[0], ty),
            Intrinsic::TypeInfo(t) => Some(self.typeinfo_ref(t).into()),
            Intrinsic::DynOf(t) => self.dyn_of(&args[0], t, span),
            Intrinsic::DynView => self.dyn_view(&args[0]),
            Intrinsic::FieldAt => self.field_at(args),
            Intrinsic::DynCall => self.dyn_call(args),
            Intrinsic::DynMake => self.dyn_make(&args[0], span),
            Intrinsic::DynStore => self.dyn_store(args),
            Intrinsic::IsNull => self.is_null(&args[0]),
            // a handle is its document's characters, laid out as they are
            Intrinsic::CircuitOf | Intrinsic::CircuitDoc => self.value(&args[0]),
            Intrinsic::FromRawParts => self.assemble_raw_parts(args),
            Intrinsic::GrowZeroed => self.grow_zeroed(args, span),
            Intrinsic::FloatDigits => self.float_digits(args),
            Intrinsic::ParseFloat => self.parse_float(args),
            Intrinsic::Sin => self.libm("sin", &args[0]),
            Intrinsic::Cos => self.libm("cos", &args[0]),
            Intrinsic::FileOpen => self.file_open(args),
            Intrinsic::FileSize => self.file_size(args),
            Intrinsic::FileRead => self.file_transfer(args, false),
            Intrinsic::FileWrite => self.file_transfer(args, true),
            Intrinsic::FileClose => self.file_close(args),
            Intrinsic::StdHandle => self.std_handle(args),
            Intrinsic::IsConsole => self.is_console(args),
            Intrinsic::ConsoleRead => self.console_read(args),
            Intrinsic::ModuleOpen => self.module_open(args),
            Intrinsic::ModuleFind => self.module_find(args),
            Intrinsic::ModuleClose => self.module_close(args),
            Intrinsic::CallVoid => self.call_void(args),
            Intrinsic::HostDescriptor => self.host_descriptor(),
            Intrinsic::Clock => self.clock(),
            Intrinsic::Gate(_)
            | Intrinsic::Apply
            | Intrinsic::Node
            | Intrinsic::Adjoint
            | Intrinsic::Controlled
            | Intrinsic::Then
            | Intrinsic::Derived(_) => {
                unreachable!("only quantum units act on qubits, and they generate circuits, not code")
            }
            Intrinsic::Invoke(sig) => {
                let addr = self.value(&args[0])?.into_struct_value();
                let code = ok(self.b.build_extract_value(addr, 0, "code")).into_pointer_value();
                let carried = ok(self.b.build_extract_value(addr, 1, "sig")).into_pointer_value();
                let wanted = self.table(sig);
                let wrong = ok(self.b.build_int_compare(IntPredicate::NE, carried, wanted, "wrong.sig"));
                self.abort_if(wrong, Code::Ra11, span);
                Some(code.into())
            }
            Intrinsic::TypeInfoOf => {
                let r = self.value(&args[0])?.into_struct_value();
                let table = ok(self.b.build_extract_value(r, 1, "table")).into_pointer_value();
                Some(self.typeinfo_value(table).into())
            }
            // without a message, the identifier's own words, as for the
            // aborts the compiler inserts
            Intrinsic::Abort(n) if args.is_empty() => {
                let bad = self.cx.bool_type().const_all_ones();
                self.abort_if(bad, crate::sema::arith::runtime_code(n), span);
                self.diverged()
            }
            Intrinsic::Panic | Intrinsic::Abort(_) => {
                let s = self.value(&args[0])?.into_struct_value();
                let (p, n) = self.slice_parts(s);
                let code = match which {
                    Intrinsic::Abort(n) => crate::sema::arith::runtime_code(n),
                    _ => Code::Ra10,
                };
                let head = text::abort_head(code, &self.unit.name, self.line(span));
                let h = self.aborts.message(&self.b, &head);
                let panic = self.text.panic(self.cx, self.module);
                ok(self.b.build_call(
                    panic,
                    &[
                        h.as_pointer_value().into(),
                        i32.const_int(head.len() as u64, false).into(),
                        p.into(),
                        n.into(),
                    ],
                    "",
                ));
                self.diverged()
            }
            Intrinsic::Len => {
                let x = &args[0];
                match x.ty {
                    Ty::Slice(_) => {
                        let s = self.value(x)?.into_struct_value();
                        Some(self.slice_parts(s).1.into())
                    }
                    // read where it is: asking the length moves nothing
                    Ty::Growable(_) => {
                        let at = self.held(x)?;
                        let s = self.load_scalar(x.ty, at)?.into_struct_value();
                        Some(self.slice_parts(s).1.into())
                    }
                    _ => {
                        // a fixed array's length is its type's; the array is
                        // still evaluated, for whatever else it does
                        self.held(x)?;
                        let (_, n) = self.unit.types.as_array(x.ty).expect("len of an array or slice");
                        Some(self.cx.i64_type().const_int(n, false).into())
                    }
                }
            }
        }
    }

    /// Delivers a branch's value.
    pub(super) fn deliver(&mut self, out: Out<'ctx>, e: &Expr) {
        match out {
            Out::Nothing => self.discard(e),
            Out::Slot(a, _) => {
                if let Some(v) = self.value(e)
                    && !self.terminated()
                {
                    self.store_scalar(v, a);
                }
            }
            Out::Mem(a) => {
                self.eval(e, Dest::Mem(a));
            }
        }
    }

    /// Where the branches of a construct of type `ty` deliver its value.
    pub(super) fn out_for(&mut self, ty: Ty, dest: Dest<'ctx>, name: &str) -> Out<'ctx> {
        match dest {
            Dest::Mem(a) => Out::Mem(a),
            Dest::Value if types::scalar(self.cx, &self.unit.types, ty).is_some() => {
                Out::Slot(self.named_temp(ty, name), ty)
            }
            Dest::Value => Out::Nothing,
        }
    }

    /// The value a construct's branches delivered.
    pub(super) fn result(&mut self, out: Out<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        match out {
            Out::Slot(a, ty) => self.load_scalar(ty, a),
            _ => None,
        }
    }

    /// Lowers a block, delivering its value to `dest`.
    fn block(&mut self, b: &Block, dest: Dest<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        self.enter_scope();
        self.statements(b);
        let v = match &b.value {
            Some(v) => self.eval(v, dest),
            None => None,
        };
        // the value is out of the block before its bindings are destroyed
        self.leave_scope();
        v
    }

    /// Lowers a block whose value goes to `out`.
    pub(super) fn block_out(&mut self, b: &Block, out: Out<'ctx>) {
        self.enter_scope();
        self.statements(b);
        if let Some(v) = &b.value {
            self.deliver(out, v);
        }
        self.leave_scope();
    }

    fn statements(&mut self, b: &Block) {
        for s in &b.stmts {
            match s {
                Stmt::Let { local, init } => {
                    if let (Some(e), Some(slot)) = (init, self.locals[local.index()]) {
                        self.store_expr(e, slot);
                        self.own(*local);
                    } else if let Some(e) = init {
                        self.discard(e);
                    } else {
                        self.own_unset(*local);
                    }
                }
                // the pattern matches every value, so its names are filled in
                // from the value stored once, with no test that could fail;
                // what it does not keep of a value given up to it is destroyed
                Stmt::LetPat { pat, init } => {
                    let place = init.is_place();
                    let at = self.addr(init);
                    if let Some(at) = at {
                        self.bind_all(pat, at);
                        self.own_bindings(pat);
                        if !place {
                            self.drop_unbound(pat, at);
                        } else if self.binds_owned(pat) {
                            self.moved(init);
                            self.drop_unbound(pat, at);
                        }
                    }
                }
                Stmt::Expr(e) => self.discard(e),
            }
        }
    }

    /// Finishes one branch of a construct that joins afterwards, jumping to
    /// the join point if control can still reach the end of the branch.
    pub(super) fn close_branch(&mut self, join: BasicBlock<'ctx>) {
        if !self.terminated() {
            ok(self.b.build_unconditional_branch(join));
        }
    }

    fn if_expr(
        &mut self,
        cond: &Expr,
        then: &Block,
        els: Option<&Expr>,
        ty: Ty,
        dest: Dest<'ctx>,
    ) -> Option<BasicValueEnum<'ctx>> {
        let c = self.value(cond)?.into_int_value();
        let func = self.func.expect("inside a function");
        let out = self.out_for(ty, dest, "if.value");
        let then_bb = self.cx.append_basic_block(func, "then");
        let else_bb = els.map(|_| self.cx.append_basic_block(func, "else"));
        let join = self.cx.append_basic_block(func, "endif");
        ok(self.b.build_conditional_branch(c, then_bb, else_bb.unwrap_or(join)));

        self.b.position_at_end(then_bb);
        self.block_out(then, out);
        self.close_branch(join);

        if let (Some(e), Some(bb)) = (els, else_bb) {
            self.b.position_at_end(bb);
            self.deliver(out, e);
            self.close_branch(join);
        }

        self.b.position_at_end(join);
        self.result(out)
    }

    fn loop_expr(&mut self, id: LoopId, body: &Block, ty: Ty, dest: Dest<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        let func = self.func.expect("inside a function");
        let out = self.out_for(ty, dest, "loop.value");
        let top = self.cx.append_basic_block(func, "loop");
        let exit = self.cx.append_basic_block(func, "loop.end");
        self.enter_loop(id);
        self.loops.insert(id, LoopTarget { exit, next: top, out });
        ok(self.b.build_unconditional_branch(top));
        self.b.position_at_end(top);
        self.block(body, Dest::Value);
        if !self.terminated() {
            ok(self.b.build_unconditional_branch(top));
        }
        self.b.position_at_end(exit);
        self.result(out)
    }

    fn while_expr(&mut self, id: LoopId, cond: &Expr, body: &Block) {
        let func = self.func.expect("inside a function");
        let test = self.cx.append_basic_block(func, "while");
        ok(self.b.build_unconditional_branch(test));
        self.b.position_at_end(test);
        let Some(c) = self.value(cond) else { return };
        let top = self.cx.append_basic_block(func, "while.body");
        let exit = self.cx.append_basic_block(func, "while.end");
        ok(self.b.build_conditional_branch(c.into_int_value(), top, exit));
        self.enter_loop(id);
        self.loops.insert(
            id,
            LoopTarget {
                exit,
                next: test,
                out: Out::Nothing,
            },
        );
        self.b.position_at_end(top);
        self.block(body, Dest::Value);
        if !self.terminated() {
            ok(self.b.build_unconditional_branch(test));
        }
        self.b.position_at_end(exit);
    }

    /// `for var in start..end` or `start..=end`.
    ///
    /// The bound is evaluated once, before the first iteration. The loop never
    /// computes a value past the bound, so a range ending at its type's
    /// maximum finishes rather than overflowing: an exclusive range stops when
    /// the successor *equals* the bound, and an inclusive one stops when the
    /// variable does, before any successor is used.
    fn for_range(
        &mut self,
        id: LoopId,
        var: tir::LocalId,
        start: &Expr,
        end: &Expr,
        inclusive: bool,
        body: &Block,
    ) {
        let Some(s) = self.value(start) else { return };
        let Some(e) = self.value(end) else { return };
        let (s, e) = (s.into_int_value(), e.into_int_value());
        let Ty::Int(t) = start.ty else {
            unreachable!("analysis only ranges over integers")
        };
        let slot = self.locals[var.index()].expect("the loop variable has a slot");
        let func = self.func.expect("inside a function");

        let enter = ok(self.b.build_int_compare(
            match (inclusive, t.signed) {
                (false, true) => IntPredicate::SLT,
                (false, false) => IntPredicate::ULT,
                (true, true) => IntPredicate::SLE,
                (true, false) => IntPredicate::ULE,
            },
            s,
            e,
            "",
        ));
        self.store_scalar(s.into(), slot);
        let top = self.cx.append_basic_block(func, "for.body");
        let step = self.cx.append_basic_block(func, "for.next");
        let exit = self.cx.append_basic_block(func, "for.end");
        ok(self.b.build_conditional_branch(enter, top, exit));
        self.enter_loop(id);
        self.loops.insert(
            id,
            LoopTarget {
                exit,
                next: step,
                out: Out::Nothing,
            },
        );

        self.b.position_at_end(top);
        self.block(body, Dest::Value);
        if !self.terminated() {
            ok(self.b.build_unconditional_branch(step));
        }

        self.b.position_at_end(step);
        let int = types::int(self.cx, t);
        let i = ok(self.b.build_load(int, slot.ptr, "")).into_int_value();
        // wrapping is harmless here: at the only point the successor could
        // wrap (the variable equal to the type's maximum) the loop exits
        // and the successor is never read
        let next = ok(self.b.build_int_add(i, int.const_int(1, false), ""));
        ok(self.b.build_store(slot.ptr, next));
        let done = ok(self.b.build_int_compare(
            IntPredicate::EQ,
            if inclusive { i } else { next },
            e,
            "",
        ));
        ok(self.b.build_conditional_branch(done, exit, top));

        self.b.position_at_end(exit);
    }

    /// `&&` and `||`, evaluating the right operand only when the left does not
    /// decide the result.
    fn logical(&mut self, op: LogicalOp, lhs: &Expr, rhs: &Expr) -> Option<BasicValueEnum<'ctx>> {
        let l = self.value(lhs)?.into_int_value();
        let func = self.func.expect("inside a function");
        let slot = self.named_temp(Ty::Bool, "logic");
        // if the left operand decides, it is the result
        self.store_scalar(l.into(), slot);
        let right = self.cx.append_basic_block(func, "logic.rhs");
        let join = self.cx.append_basic_block(func, "logic.end");
        match op {
            LogicalOp::And => ok(self.b.build_conditional_branch(l, right, join)),
            LogicalOp::Or => ok(self.b.build_conditional_branch(l, join, right)),
        };
        self.b.position_at_end(right);
        self.deliver(Out::Slot(slot, Ty::Bool), rhs);
        self.close_branch(join);
        self.b.position_at_end(join);
        self.load_scalar(Ty::Bool, slot)
    }
}

/// A method's name as its symbol carries it: an operator method keeps its
/// `$`, so that `$add` and a method called `add` stay apart.
fn method_name(name: &str, m: tir::MethodOf) -> String {
    if m.operator { format!("${name}") } else { name.to_owned() }
}

/// The LLVM linkage of a Topiq linkage.
fn llvm_linkage(l: Linkage) -> LlvmLinkage {
    match l {
        Linkage::Program => LlvmLinkage::External,
        Linkage::Unit => LlvmLinkage::Internal,
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{function, ir};

    #[test]
    fn every_binding_gets_a_stack_slot_and_parameters_are_stored_into_theirs() {
        let ir = ir("fn f(a: i32, b: i32) -> i32 { let c = a; c + b }");
        let f = function(&ir, "f");
        for slot in ["%a = alloca i32", "%b = alloca i32", "%c = alloca i32"] {
            assert!(f.contains(slot), "{slot} in:\n{f}");
        }
        assert!(f.contains("store i32 %0, ptr %a"), "{f}");
    }

    #[test]
    fn an_if_without_else_branches_straight_to_its_join() {
        let ir = ir("fn g() { }\nfn f(c: bool) { if c { g(); } }");
        let f = function(&ir, "f");
        assert!(f.contains("label %then, label %endif"), "{f}");
        assert!(!f.contains("else:"), "{f}");
    }

    #[test]
    fn a_value_producing_if_joins_through_a_slot() {
        let ir = ir("fn f(c: bool) -> i32 { if c { 1 } else { 2 } }");
        let f = function(&ir, "f");
        assert!(f.contains("%if.value = alloca i32"), "{f}");
        assert!(f.contains("store i32 1, ptr %if.value"), "{f}");
        assert!(f.contains("store i32 2, ptr %if.value"), "{f}");
    }

    #[test]
    fn a_loop_value_arrives_through_its_break() {
        let ir = ir("fn f() -> i32 { loop { break 3; } }");
        let f = function(&ir, "f");
        assert!(f.contains("%loop.value = alloca i32"), "{f}");
        assert!(f.contains("store i32 3, ptr %loop.value"), "{f}");
        assert!(f.contains("br label %loop.end"), "{f}");
    }

    #[test]
    fn the_right_operand_of_and_is_evaluated_only_on_its_own_branch() {
        let ir = ir("fn g() -> bool { true }\nfn f(a: bool) -> bool { a && g() }");
        let f = function(&ir, "f");
        let rhs = f.find("logic.rhs:").expect("a block for the right operand");
        let call = f.find("call i1 @\"topiq$demo$g\"").expect("the call");
        assert!(call > rhs, "the call is inside the right-operand block:\n{f}");
    }

    #[test]
    fn a_range_loop_steps_without_a_check_and_stops_on_equality() {
        let ir = ir("fn f() { for i in 0u8..=255 { } }");
        let f = function(&ir, "f");
        let step = &f[f.find("for.next:").expect("the step block")..];
        assert!(step.contains("add i8"), "{step}");
        assert!(!step.contains("with.overflow"), "the step can never overflow:\n{step}");
        assert!(step.contains("icmp eq i8"), "{step}");
    }

    #[test]
    fn code_after_a_return_is_lowered_into_an_unreachable_block() {
        // well formed, never run; the module still verifies
        let ir = ir("fn f() -> i32 { return 1; }");
        let f = function(&ir, "f");
        assert!(f.contains("ret i32 1"), "{f}");
        assert!(f.contains("unreachable:"), "{f}");
    }

    #[test]
    fn continue_in_a_while_jumps_back_to_the_condition() {
        let ir = ir("fn f(n: u32) { let i = 0u32; while i < n { i += 1; continue; } }");
        let f = function(&ir, "f");
        assert!(f.matches("br label %while").count() >= 2, "{f}");
    }

    #[test]
    fn assigning_a_unit_scope_object_stores_to_its_symbol() {
        let ir = ir("let N: u16 = 0;\nfn f() { N = 5; }");
        let f = function(&ir, "f");
        assert!(f.contains("store i16 5, ptr @\"topiq$demo$N\""), "{f}");
    }

    #[test]
    fn a_structure_is_returned_through_a_hidden_parameter() {
        let ir = ir("struct P { x: i32, y: i32 }\nfn f() -> P { P { x: 1, y: 2 } }");
        let f = function(&ir, "f");
        assert!(f.starts_with("define void @\"topiq$demo$f\"(ptr"), "{f}");
        assert!(f.contains("store i32 2"), "{f}");
    }

    #[test]
    fn a_structure_argument_is_passed_as_the_address_of_a_copy() {
        let ir = ir("struct P { x: i32 }\nfn g(p: P) -> i32 { p.x }\nfn f(p: P) -> i32 { g(p) }");
        let f = function(&ir, "f");
        assert!(f.contains("llvm.memcpy"), "the copy is made:\n{f}");
        assert!(f.contains("call i32 @\"topiq$demo$g\"(ptr"), "{f}");
    }

    #[test]
    fn indexing_checks_the_bound() {
        let ir = ir("fn f(a: [i32; 4], i: usize) -> i32 { a[i] }");
        let f = function(&ir, "f");
        assert!(f.contains("icmp uge i64"), "{f}");
        assert!(f.contains("abort.RA06"), "{f}");
    }

    #[test]
    fn a_string_is_stored_once_as_four_byte_characters() {
        let ir = ir("fn f() { print(\"hi\"); print(\"hi\"); }");
        assert_eq!(ir.matches("[2 x i32] [i32 104, i32 105]").count(), 1, "{ir}");
        assert!(ir.contains("call void @\"topiq$print\""), "{ir}");
    }
}
