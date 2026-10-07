//! Destroying values: RAII.
//!
//! A value that owns something (a closure's environment, a growable array's
//! buffer, a structure or enumeration with a `$drop`, or anything holding one
//! of those) is destroyed when the binding that holds it goes out of scope,
//! unless it was moved away first. Bindings are destroyed in the reverse of
//! the order they were declared, at the end of their block and on every way
//! out of it:
//! `break`, `continue` and `return`. An assignment destroys the value it
//! replaces. A value that nothing keeps (the result of a call written as a
//! statement, the parts of a value a pattern takes apart but does not bind)
//! is destroyed at once. An abort destroys nothing.
//!
//! # Knowing what is still there
//!
//! Whether a binding still holds its value can depend on the path taken, so
//! each binding whose type needs destroying has a flag, set when it is given
//! a value and cleared when the value is moved away; a field that is moved
//! out on its own has a flag of its own too. The flags are ordinary stack
//! slots, so where they are constant the optimiser removes them.
//!
//! # Destroying a value
//!
//! Each type that needs it gets a destroying function: its `$drop` first, if
//! it has one, then its fields, elements or payload, last first. A closure's
//! environment carries, just before it, the function that destroys and frees
//! it, or nothing, for one it does not own.

use std::collections::HashMap;

use inkwell::AddressSpace;
use inkwell::module::Linkage as LlvmLinkage;
use inkwell::values::{FunctionValue, PointerValue};

use crate::tir::visit::{self, Visit};
use crate::tir::{AdtId, AdtKind, Expr, ExprKind, Fn, FnId, LocalId, LoopId, Pat, PatKind, Ty, layout};

use super::func::{Addr, Lowering, ok};
use super::types;

/// Something a scope destroys when it ends.
#[derive(Clone, Copy, Debug)]
pub(super) enum Owned<'ctx> {
    /// A binding.
    Local(LocalId),
    /// A temporary a reference was taken to, which lasts until the function
    /// returns, with the flag saying whether it holds a value.
    Temp(Addr<'ctx>, Ty, PointerValue<'ctx>),
}

/// The state destruction needs while one function is lowered.
#[derive(Default)]
pub(super) struct Drops<'ctx> {
    /// The scopes open, outermost first, each with what it owns in the order
    /// it came to own it.
    scopes: Vec<Vec<Owned<'ctx>>>,
    /// The flag of each binding, and of each field moved out on its own.
    flags: HashMap<(LocalId, Vec<u32>), PointerValue<'ctx>>,
    /// How many scopes were open where each loop began.
    loops: HashMap<LoopId, usize>,
}

impl<'ctx> Lowering<'ctx, '_> {
    /// Whether a value of `ty` has anything to destroy.
    pub(super) fn needs_drop(&self, ty: Ty) -> bool {
        self.unit.types.holds_any(ty, &|t| match t {
            Ty::Closure(_) | Ty::Growable(_) | Ty::Dyn => true,
            Ty::Adt(id) => self.unit.drops.contains_key(&id),
            _ => false,
        })
    }

    /// Destroys the value of type `ty` at `a`.
    pub(super) fn drop_at(&mut self, a: Addr<'ctx>, ty: Ty) {
        if !self.needs_drop(ty) || self.terminated() {
            return;
        }
        let f = self.drop_glue(ty);
        ok(self.b.build_call(f, &[a.ptr.into()], ""));
    }

    /// The function that destroys a value of `ty` at the address it is given,
    /// made the first time it is needed.
    pub(super) fn drop_glue(&mut self, ty: Ty) -> FunctionValue<'ctx> {
        if let Some(&f) = self.drop_fns.get(&ty) {
            return f;
        }
        let ptr = self.cx.ptr_type(AddressSpace::default());
        let name = format!("drop.{}", self.drop_fns.len());
        let f = self
            .module
            .add_function(&name, self.cx.void_type().fn_type(&[ptr.into()], false), Some(LlvmLinkage::Private));
        // recorded first, so that a type holding itself behind an owning
        // pointer finds it
        self.drop_fns.insert(ty, f);
        let saved = self.enter_fn(f);
        let at = Addr {
            ptr: f.get_first_param().expect("the value's address").into_pointer_value(),
            align: self.layout(ty).align,
        };
        self.destroy_parts(at, ty);
        if !self.terminated() {
            ok(self.b.build_return(None));
        }
        self.leave_fn(saved);
        f
    }

    /// Branches on the variant the enumeration `id` at `at` holds and joins
    /// after, in blocks named for `name`: for each variant with a field
    /// `needed` accepts, `each` lowers what is done to its fields, given
    /// each one's type and offset.
    pub(super) fn per_variant(
        &mut self,
        id: AdtId,
        at: Addr<'ctx>,
        name: &str,
        needed: impl std::ops::Fn(&Self, Ty) -> bool,
        mut each: impl FnMut(&mut Self, &[(Ty, u64)]),
    ) {
        let l = layout::enumeration(&self.unit.types, id);
        let variants = self.unit.types.adt(id).variants().to_vec();
        let func = self.func.expect("inside a function");
        let tag = ok(self.b.build_load(types::int(self.cx, l.tag), at.ptr, "")).into_int_value();
        let done = self.cx.append_basic_block(func, &format!("{name}.done"));
        let mut cases = Vec::new();
        for (i, v) in variants.iter().enumerate() {
            if !v.fields.iter().any(|f| needed(self, f.ty)) {
                continue;
            }
            let bb = self.cx.append_basic_block(func, &format!("{name}.variant"));
            cases.push((types::const_int(self.cx, l.tag, i as i128), bb));
            let here = self.b.get_insert_block();
            self.b.position_at_end(bb);
            let fields: Vec<(Ty, u64)> = v.fields.iter().map(|f| f.ty).zip(l.variants[i].iter().copied()).collect();
            each(self, &fields);
            ok(self.b.build_unconditional_branch(done));
            if let Some(h) = here {
                self.b.position_at_end(h);
            }
        }
        ok(self.b.build_switch(tag, done, &cases));
        self.b.position_at_end(done);
    }

    /// The body of a destroying function: `$drop`, then the parts.
    fn destroy_parts(&mut self, at: Addr<'ctx>, ty: Ty) {
        match ty {
            Ty::Adt(id) => {
                if let Some(&callee) = self.unit.drops.get(&id) {
                    let (f, _) = self.callee_function(callee);
                    let table = self.table(ty);
                    let r = self.pair(at.ptr, table);
                    ok(self.b.build_call(f, &[r.into()], ""));
                }
                match self.unit.types.adt(id).kind.clone() {
                    AdtKind::Struct { fields } => {
                        let offsets = layout::fields_of(&self.unit.types, ty).offsets;
                        for (f, &off) in fields.iter().zip(&offsets).rev() {
                            let a = self.offset(at, off);
                            self.drop_at(a, f.ty);
                        }
                    }
                    AdtKind::Enum { .. } => self.per_variant(id, at, "drop", Self::needs_drop, |l, fields| {
                        for &(t, off) in fields.iter().rev() {
                            let a = l.offset(at, off);
                            l.drop_at(a, t);
                        }
                    }),
                }
            }
            Ty::Tuple(_) => {
                let elems = self.unit.types.as_tuple(ty).expect("a tuple").to_vec();
                let offsets = layout::fields_of(&self.unit.types, ty).offsets;
                for (&e, &off) in elems.iter().zip(&offsets).rev() {
                    let a = self.offset(at, off);
                    self.drop_at(a, e);
                }
            }
            Ty::Array(_) => {
                let (elem, n) = self.unit.types.as_array(ty).expect("an array");
                let size = self.layout(elem).size;
                for i in (0..n).rev() {
                    let a = self.offset(at, i * size);
                    self.drop_at(a, elem);
                }
            }
            Ty::Closure(_) => {
                // the environment is freed by the function stored just before
                // it, which is absent for one the closure does not own
                let ptr = self.cx.ptr_type(AddressSpace::default());
                let env = ok(self.b.build_load(ptr, at.ptr, "")).into_pointer_value();
                let func = self.func.expect("inside a function");
                let has_env = self.cx.append_basic_block(func, "drop.env");
                let has_fn = self.cx.append_basic_block(func, "drop.env.fn");
                let done = self.cx.append_basic_block(func, "drop.env.done");
                let null_env = ok(self.b.build_is_null(env, ""));
                ok(self.b.build_conditional_branch(null_env, done, has_env));
                self.b.position_at_end(has_env);
                let i8 = self.cx.i8_type();
                let minus = self.cx.i64_type().const_int((-8i64) as u64, true);
                // SAFETY: an owned environment is always preceded by its header
                let header = unsafe { ok(self.b.build_gep(i8, env, &[minus], "")) };
                let glue = ok(self.b.build_load(ptr, header, "")).into_pointer_value();
                let null_fn = ok(self.b.build_is_null(glue, ""));
                ok(self.b.build_conditional_branch(null_fn, done, has_fn));
                self.b.position_at_end(has_fn);
                let fn_type = self.cx.void_type().fn_type(&[ptr.into()], false);
                ok(self.b.build_indirect_call(fn_type, glue, &[env.into()], ""));
                ok(self.b.build_unconditional_branch(done));
                self.b.position_at_end(done);
            }
            Ty::Growable(_) => self.destroy_growable(at, ty),
            Ty::Dyn => self.destroy_dyn(at),
            _ => {}
        }
    }

    /// The function that destroys the environment of a closure whose body is
    /// `code` and frees it: what an owned environment's header holds.
    pub(super) fn env_glue(&mut self, code: FnId, env_ty: Ty, header: u64) -> FunctionValue<'ctx> {
        let ptr = self.cx.ptr_type(AddressSpace::default());
        let name = format!("closure.env.drop.{}", code.0);
        if let Some(f) = self.module.get_function(&name) {
            return f;
        }
        let f = self
            .module
            .add_function(&name, self.cx.void_type().fn_type(&[ptr.into()], false), Some(LlvmLinkage::Private));
        let saved = self.enter_fn(f);
        let env = f.get_first_param().expect("the environment").into_pointer_value();
        let at = Addr {
            ptr: env,
            align: self.layout(env_ty).align,
        };
        self.drop_at(at, env_ty);
        let i8 = self.cx.i8_type();
        let back = self.cx.i64_type().const_int(header.wrapping_neg(), true);
        // SAFETY: the allocation begins `header` bytes before the environment
        let block = unsafe { ok(self.b.build_gep(i8, env, &[back], "")) };
        let free = self.free_fn();
        ok(self.b.build_call(free, &[block.into()], ""));
        ok(self.b.build_return(None));
        self.leave_fn(saved);
        f
    }

    /// The C runtime's `free`.
    pub(super) fn free_fn(&self) -> FunctionValue<'ctx> {
        let ptr = self.cx.ptr_type(AddressSpace::default());
        self.module.get_function("free").unwrap_or_else(|| {
            self.module
                .add_function("free", self.cx.void_type().fn_type(&[ptr.into()], false), Some(LlvmLinkage::External))
        })
    }

    //////////////////////
    // FLAGS AND SCOPES //
    //////////////////////

    /// Prepares the flags of a function about to be lowered: one for every
    /// binding that needs destroying, and one for every field of one that is
    /// read on its own and needs destroying.
    pub(super) fn prepare_drops(&mut self, f: &Fn) {
        self.drops = Drops::default();
        struct Paths<'l, 'c, 'a> {
            l: &'l Lowering<'c, 'a>,
            out: Vec<(LocalId, Vec<u32>)>,
        }
        impl Visit for Paths<'_, '_, '_> {
            fn expr(&mut self, e: &Expr) {
                if matches!(e.kind, ExprKind::Field { .. })
                    && self.l.needs_drop(e.ty)
                    && let Some(p) = local_path(e)
                    && !self.out.contains(&p)
                {
                    self.out.push(p);
                }
                visit::walk_expr(self, e);
            }
        }
        let mut paths = Paths { l: self, out: Vec::new() };
        paths.block(&f.body);
        let mut wanted = paths.out;
        for (i, local) in f.locals.iter().enumerate() {
            if self.needs_drop(local.ty) {
                wanted.push((LocalId(i as u32), Vec::new()));
            }
        }
        for key in wanted {
            let slot = self.cleared_flag("live");
            self.drops.flags.insert(key, slot);
        }
        self.drops.scopes.push(Vec::new());
    }

    /// A new flag, cleared as the function starts, before anything can set
    /// it.
    fn cleared_flag(&mut self, name: &str) -> PointerValue<'ctx> {
        let slot = self.named_temp(Ty::Bool, name).ptr;
        let at = self.cx.create_builder();
        let alloca = slot.as_instruction().expect("a stack slot is an instruction");
        match alloca.get_next_instruction() {
            Some(next) => at.position_before(&next),
            None => at.position_at_end(alloca.get_parent().expect("in a block")),
        }
        ok(at.build_store(slot, self.cx.bool_type().const_zero()));
        slot
    }

    /// Opens a scope.
    pub(super) fn enter_scope(&mut self) {
        self.drops.scopes.push(Vec::new());
    }

    /// Closes the innermost scope, destroying what it owns if control reaches
    /// its end.
    pub(super) fn leave_scope(&mut self) {
        let depth = self.drops.scopes.len() - 1;
        if !self.terminated() {
            self.destroy_from(depth);
        }
        self.drops.scopes.pop();
    }

    /// Destroys what every scope from `depth` inward owns, innermost first,
    /// without closing them: for a jump out of them.
    pub(super) fn destroy_from(&mut self, depth: usize) {
        let owned: Vec<Owned<'ctx>> = self.drops.scopes[depth..]
            .iter()
            .rev()
            .flat_map(|s| s.iter().rev().copied())
            .collect();
        for o in owned {
            match o {
                Owned::Local(l) => self.drop_local(l),
                Owned::Temp(a, ty, flag) => self.drop_if(flag, |me| me.drop_at(a, ty)),
            }
        }
    }

    /// How many scopes are open, for a loop that starts here.
    pub(super) fn enter_loop(&mut self, id: LoopId) {
        let depth = self.drops.scopes.len();
        self.drops.loops.insert(id, depth);
    }

    /// Destroys what the scopes a jump out of loop `id` leaves own.
    pub(super) fn leave_loop_early(&mut self, id: LoopId) {
        if let Some(&depth) = self.drops.loops.get(&id) {
            self.destroy_from(depth);
        }
    }

    /// Makes the innermost scope own `local`, which now holds a value.
    pub(super) fn own(&mut self, local: LocalId) {
        self.own_unset(local);
        self.set_flags(local, &[], true);
    }

    /// Makes the innermost scope own `local`, which holds nothing yet.
    pub(super) fn own_unset(&mut self, local: LocalId) {
        if self.drops.flags.contains_key(&(local, Vec::new()))
            && let Some(s) = self.drops.scopes.last_mut()
        {
            s.push(Owned::Local(local));
        }
    }

    /// Storage for a temporary that a reference is taken to, which keeps its
    /// value until the function returns. Where this runs more than once, as
    /// in a loop, the value it held from the time before is destroyed first.
    /// The caller stores the value, then calls [`Self::keep_temp`].
    pub(super) fn lasting_temp(&mut self, ty: Ty) -> (Addr<'ctx>, Option<PointerValue<'ctx>>) {
        let a = self.temp(ty);
        if !self.needs_drop(ty) {
            return (a, None);
        }
        let flag = self.cleared_flag("temp.live");
        self.drop_if(flag, |me| me.drop_at(a, ty));
        if let Some(first) = self.drops.scopes.first_mut() {
            first.push(Owned::Temp(a, ty, flag));
        }
        (a, Some(flag))
    }

    /// Marks a lasting temporary as holding its value.
    pub(super) fn keep_temp(&mut self, flag: Option<PointerValue<'ctx>>) {
        if let Some(f) = flag
            && !self.terminated()
        {
            ok(self.b.build_store(f, self.cx.bool_type().const_int(1, false)));
        }
    }

    /// Sets or clears the flags of `local`'s value at `path` and of
    /// everything inside it.
    pub(super) fn set_flags(&mut self, local: LocalId, path: &[u32], on: bool) {
        let bool_ty = self.cx.bool_type();
        let v = bool_ty.const_int(u64::from(on), false);
        let slots: Vec<PointerValue<'ctx>> = self
            .drops
            .flags
            .iter()
            .filter(|((l, p), _)| *l == local && p.starts_with(path))
            .map(|(_, &s)| s)
            .collect();
        for s in slots {
            ok(self.b.build_store(s, v));
        }
    }

    /// Records that the value `e` was moved away, if it is a binding or a
    /// field of one whose type needs destroying.
    pub(super) fn moved(&mut self, e: &Expr) {
        if !self.needs_drop(e.ty) {
            return;
        }
        if let Some((l, p)) = local_path(e) {
            self.set_flags(l, &p, false);
        }
    }

    /// Destroys a binding's value, as far as it still holds one.
    pub(super) fn drop_local(&mut self, local: LocalId) {
        let Some(at) = self.locals[local.index()] else { return };
        let ty = self.current_fn_local_ty(local);
        self.drop_path(local, Vec::new(), at, ty);
    }

    /// Destroys what is at `path` in `local`, looking at the flags: a part
    /// with a flag only if the flag is set, and a part inside which some
    /// field was moved out on its own field by field.
    pub(super) fn drop_path(&mut self, local: LocalId, path: Vec<u32>, at: Addr<'ctx>, ty: Ty) {
        if !self.needs_drop(ty) {
            return;
        }
        let flag = self.drops.flags.get(&(local, path.clone())).copied();
        let deeper = self
            .drops
            .flags
            .keys()
            .any(|(l, p)| *l == local && p.len() > path.len() && p.starts_with(&path));
        let body = |me: &mut Self| {
            if deeper && let Some(fields) = me.struct_parts(ty) {
                let offsets = layout::fields_of(&me.unit.types, ty).offsets;
                for (i, (&fty, &off)) in fields.iter().zip(&offsets).enumerate().rev() {
                    let mut sub = path.clone();
                    sub.push(i as u32);
                    let a = me.offset(at, off);
                    me.drop_path(local, sub, a, fty);
                }
            } else {
                me.drop_at(at, ty);
            }
        };
        match flag {
            Some(f) => self.drop_if(f, body),
            None => body(self),
        }
    }

    /// Runs `body` where the flag at `flag` is set.
    fn drop_if(&mut self, flag: PointerValue<'ctx>, body: impl FnOnce(&mut Self)) {
        if self.terminated() {
            return;
        }
        let func = self.func.expect("inside a function");
        let live = ok(self.b.build_load(self.cx.bool_type(), flag, "")).into_int_value();
        let yes = self.cx.append_basic_block(func, "drop");
        let after = self.cx.append_basic_block(func, "drop.after");
        ok(self.b.build_conditional_branch(live, yes, after));
        self.b.position_at_end(yes);
        body(self);
        if !self.terminated() {
            ok(self.b.build_unconditional_branch(after));
        }
        self.b.position_at_end(after);
    }

    /// The field types of a structure or tuple.
    fn struct_parts(&self, ty: Ty) -> Option<Vec<Ty>> {
        match ty {
            Ty::Adt(id) => match &self.unit.types.adt(id).kind {
                AdtKind::Struct { fields } if !self.unit.drops.contains_key(&id) => {
                    Some(fields.iter().map(|f| f.ty).collect())
                }
                _ => None,
            },
            Ty::Tuple(_) => self.unit.types.as_tuple(ty).map(<[Ty]>::to_vec),
            _ => None,
        }
    }

    /// Destroys the parts of the value at `a` that the pattern `p`, which it
    /// matched, did not bind: the value was given up to the pattern, and what
    /// the pattern does not keep, nothing keeps.
    pub(super) fn drop_unbound(&mut self, p: &Pat, a: Addr<'ctx>) {
        match &p.kind {
            PatKind::Wild => self.drop_at(a, p.ty),
            PatKind::Bind(_) | PatKind::BindRef(_) | PatKind::Const(_) => {}
            PatKind::Struct { fields } => {
                let offsets = layout::fields_of(&self.unit.types, p.ty).offsets;
                let tys = self.struct_parts(p.ty).unwrap_or_default();
                for (i, (&fty, &off)) in tys.iter().zip(&offsets).enumerate().rev() {
                    let at = self.offset(a, off);
                    match fields.iter().find(|(j, _)| *j as usize == i) {
                        Some((_, sub)) => self.drop_unbound(sub, at),
                        None => self.drop_at(at, fty),
                    }
                }
            }
            PatKind::Variant { variant, fields } => {
                let Ty::Adt(id) = p.ty else { return };
                let def = self.unit.types.adt(id).clone();
                let v = &def.variants()[*variant as usize];
                let offsets = layout::enumeration(&self.unit.types, id).variants[*variant as usize].clone();
                for (i, (f, &off)) in v.fields.iter().zip(&offsets).enumerate().rev() {
                    let at = self.offset(a, off);
                    match fields.iter().find(|(j, _)| *j as usize == i) {
                        Some((_, sub)) => self.drop_unbound(sub, at),
                        None => self.drop_at(at, f.ty),
                    }
                }
            }
        }
    }

    /// Makes the innermost scope own every binding `p` binds.
    pub(super) fn own_bindings(&mut self, p: &Pat) {
        match &p.kind {
            PatKind::Bind(l) => self.own(*l),
            PatKind::Wild | PatKind::BindRef(_) | PatKind::Const(_) => {}
            PatKind::Struct { fields } | PatKind::Variant { fields, .. } => {
                for (_, f) in fields {
                    self.own_bindings(f);
                }
            }
        }
    }

    /// Whether a pattern keeps any part that needs destroying.
    pub(super) fn binds_owned(&self, p: &Pat) -> bool {
        match &p.kind {
            PatKind::Bind(_) => self.needs_drop(p.ty),
            PatKind::Wild | PatKind::BindRef(_) | PatKind::Const(_) => false,
            PatKind::Struct { fields } | PatKind::Variant { fields, .. } => {
                fields.iter().any(|(_, f)| self.binds_owned(f))
            }
        }
    }

    fn current_fn_local_ty(&self, local: LocalId) -> Ty {
        self.local_types[local.index()]
    }
}

/// The binding and field path a place is, if it is made of a binding and
/// fields alone.
pub(super) fn local_path(e: &Expr) -> Option<(LocalId, Vec<u32>)> {
    match &e.kind {
        ExprKind::Local(l) => Some((*l, Vec::new())),
        ExprKind::Field { base, field } => {
            let (l, mut p) = local_path(base)?;
            p.push(*field);
            Some((l, p))
        }
        _ => None,
    }
}
