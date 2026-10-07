//! Function values and closures.
//!
//! A value of a function type, `fn(T…) -> U`, is its code's address. A
//! function declared with `fn` is such a value, and `&f` is a reference to
//! it: to a constant slot holding that address, one per function, so the
//! reference is valid for as long as the program runs.
//!
//! A closure is two words: the address of its environment, which holds what
//! it captured, and the address of its code. The code is the closure's body,
//! lifted into a function whose first parameter is a reference to the
//! environment; a call passes the environment in front of the arguments. The
//! environment is allocated when the closure is made. A closure that captures
//! nothing has no environment, and can become a plain `*fn`: that is a small
//! function of the closure's own signature that calls the body with no
//! environment.

use inkwell::AddressSpace;
use inkwell::module::Linkage as LlvmLinkage;
use inkwell::types::{BasicMetadataTypeEnum, BasicType};
use inkwell::values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, PointerValue};

use crate::diag::Code;
use crate::tir::{Callee, Expr, FnId, Ty, layout};

use super::abi;
use super::func::{Dest, Lowering, ok};
use super::types;

/// Which function a slot or a function value is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum FnKey {
    /// A function of the unit, or of another.
    Callee(Callee),
    /// A closure's body without its environment.
    Thunk(FnId),
}

impl<'ctx> Lowering<'ctx, '_> {
    /// The code address of a function.
    pub(super) fn fn_pointer(&mut self, key: FnKey) -> PointerValue<'ctx> {
        let f = match key {
            FnKey::Callee(c) => self.callee_function(c).0,
            FnKey::Thunk(id) => self.thunk(id),
        };
        f.as_global_value().as_pointer_value()
    }

    /// The constant slot holding a function's address, which a reference to
    /// the function points at.
    pub(super) fn fn_slot(&mut self, key: FnKey) -> PointerValue<'ctx> {
        if let Some(g) = self.fn_slots.get(&key) {
            return g.as_pointer_value();
        }
        let code = self.fn_pointer(key);
        let g = super::func::private_constant(self.module, code, &format!("fn.slot.{}", self.fn_slots.len()), 8);
        self.fn_slots.insert(key, g);
        g.as_pointer_value()
    }

    /// A closure's body as a function without the environment, made the
    /// first time it is needed.
    fn thunk(&mut self, id: FnId) -> FunctionValue<'ctx> {
        if let Some(&f) = self.thunks.get(&id) {
            return f;
        }
        let (body, sret) = self.fns[id.index()].expect("a closure's body is compiled");
        let f = self.unit.func(id);
        let params: Vec<Ty> = f.param_types().skip(1).collect();
        let sig = abi::signature(self.cx, &self.unit.types, &params, f.ret);
        let name = format!("{}.plain", body.get_name().to_string_lossy());
        let thunk = self.module.add_function(&name, sig.llvm, Some(LlvmLinkage::Private));
        let b = self.cx.create_builder();
        b.position_at_end(self.cx.append_basic_block(thunk, "entry"));
        let null = self.cx.ptr_type(AddressSpace::default()).const_null();
        let env = types::fat_ref(self.cx).const_named_struct(&[null.into(), null.into()]);
        let mut args: Vec<BasicMetadataValueEnum<'ctx>> = Vec::new();
        let first = usize::from(sret);
        if sret {
            args.push(thunk.get_nth_param(0).expect("the result's address").into());
        }
        args.push(env.into());
        for i in first..thunk.count_params() as usize {
            args.push(thunk.get_nth_param(i as u32).expect("a parameter").into());
        }
        let r = ok(b.build_call(body, &args, ""));
        match r.try_as_basic_value().basic() {
            Some(v) if !sret => ok(b.build_return(Some(&v))),
            _ => ok(b.build_return(None)),
        };
        self.thunks.insert(id, thunk);
        thunk
    }

    /// Calls a function value or a closure.
    pub(super) fn indirect_call(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        ty: Ty,
        dest: Dest<'ctx>,
    ) -> Option<BasicValueEnum<'ctx>> {
        let (params, ret) = self.unit.types.as_sig(callee.ty).expect("a function or closure");
        let params = params.to_vec();
        // calling a closure uses it in place: it is not given up
        let target = if callee.is_place() {
            let at = self.addr(callee)?;
            self.load_scalar(callee.ty, at)?
        } else {
            self.value(callee)?
        };
        let closure = matches!(callee.ty, Ty::Closure(_));
        let (code, env) = if closure {
            let v = target.into_struct_value();
            let env = ok(self.b.build_extract_value(v, 0, "")).into_pointer_value();
            let code = ok(self.b.build_extract_value(v, 1, "")).into_pointer_value();
            let null = self.cx.ptr_type(AddressSpace::default()).const_null();
            (code, Some(self.pair(env, null)))
        } else {
            (target.into_pointer_value(), None)
        };
        let sig = abi::signature(self.cx, &self.unit.types, &params, ret);
        let fn_type = match env {
            Some(_) => {
                // the environment goes after the result's address and before
                // the arguments
                let mut ps: Vec<BasicMetadataTypeEnum<'ctx>> =
                    sig.llvm.get_param_types();
                ps.insert(usize::from(sig.sret), types::fat_ref(self.cx).into());
                match sig.llvm.get_return_type() {
                    Some(r) => r.fn_type(&ps, false),
                    None => self.cx.void_type().fn_type(&ps, false),
                }
            }
            None => sig.llvm,
        };
        let mut prefix: Vec<BasicMetadataValueEnum<'ctx>> = Vec::new();
        let out = if sig.sret {
            let out = match dest {
                Dest::Mem(a) => a,
                Dest::Value => self.temp(ty),
            };
            prefix.push(out.ptr.into());
            Some(out)
        } else {
            None
        };
        if let Some(e) = env {
            prefix.push(e.into());
        }
        let values = self.arguments(prefix, args)?;
        let r = ok(self.b.build_indirect_call(fn_type, code, &values, ""));
        if ty == Ty::Never {
            return self.diverged();
        }
        if out.is_some() { None } else { r.try_as_basic_value().basic() }
    }

    /// A reference to a function as a closure: the environment is where the
    /// reference points, and the code a trampoline that calls through it.
    pub(super) fn fn_as_closure(&mut self, f: &Expr, ty: Ty) -> Option<BasicValueEnum<'ctx>> {
        let r = self.value(f)?.into_struct_value();
        let target = ok(self.b.build_extract_value(r, 0, "")).into_pointer_value();
        // the environment is a cell holding the function's address, with an
        // empty header before it: nothing to destroy, and nothing to free
        let ptr = self.cx.ptr_type(AddressSpace::default());
        let pair_ty = self.cx.struct_type(&[ptr.into(), ptr.into()], false);
        let env = self.entry_alloca(pair_ty.into(), "fn.env");
        ok(self.b.build_store(env, ptr.const_null()));
        let i8 = self.cx.i8_type();
        let eight = self.cx.i64_type().const_int(8, false);
        // SAFETY: the cell is two pointers long
        let slot = unsafe { ok(self.b.build_in_bounds_gep(i8, env, &[eight], "")) };
        let code_ptr = ok(self.b.build_load(ptr, target, "")).into_pointer_value();
        ok(self.b.build_store(slot, code_ptr));
        let Ty::Closure(sig) = ty else {
            unreachable!("a function becomes a closure")
        };
        let code = self.trampoline(sig, ty).as_global_value().as_pointer_value();
        Some(self.pair(slot, code).as_basic_value_enum())
    }

    /// The code of a closure made from a reference to a function of this
    /// signature: it reads the function's address from its environment and
    /// calls it with the arguments it was given.
    fn trampoline(&mut self, sig: crate::tir::SigId, ty: Ty) -> FunctionValue<'ctx> {
        if let Some(&f) = self.trampolines.get(&sig) {
            return f;
        }
        let (params, ret) = self.unit.types.as_sig(ty).expect("a signature");
        let params = params.to_vec();
        let plain = abi::signature(self.cx, &self.unit.types, &params, ret);
        let mut ps: Vec<BasicMetadataTypeEnum<'ctx>> = plain.llvm.get_param_types();
        let first = usize::from(plain.sret);
        ps.insert(first, types::fat_ref(self.cx).into());
        let fn_type = match plain.llvm.get_return_type() {
            Some(r) => r.fn_type(&ps, false),
            None => self.cx.void_type().fn_type(&ps, false),
        };
        let name = format!("fn.trampoline.{}", self.trampolines.len());
        let t = self.module.add_function(&name, fn_type, Some(LlvmLinkage::Private));
        let b = self.cx.create_builder();
        b.position_at_end(self.cx.append_basic_block(t, "entry"));
        let env = t.get_nth_param(first as u32).expect("the environment").into_struct_value();
        let slot = ok(b.build_extract_value(env, 0, "")).into_pointer_value();
        let ptr = self.cx.ptr_type(AddressSpace::default());
        let code = ok(b.build_load(ptr, slot, "")).into_pointer_value();
        let args: Vec<BasicMetadataValueEnum<'ctx>> = (0..t.count_params())
            .filter(|&i| i as usize != first)
            .map(|i| t.get_nth_param(i).expect("a parameter").into())
            .collect();
        let r = ok(b.build_indirect_call(plain.llvm, code, &args, ""));
        match r.try_as_basic_value().basic() {
            Some(v) if !plain.sret => ok(b.build_return(Some(&v))),
            _ => ok(b.build_return(None)),
        };
        self.trampolines.insert(sig, t);
        t
    }

    /// Makes a closure: allocates its environment, moves what it captures
    /// in, and pairs it with its code.
    pub(super) fn make_closure(&mut self, code: FnId, captures: &[Expr], span: crate::span::Span) -> Option<BasicValueEnum<'ctx>> {
        let ptr = self.cx.ptr_type(AddressSpace::default());
        let body = self.fns[code.index()].expect("a closure's body is compiled").0;
        let env = if captures.is_empty() {
            ptr.const_null()
        } else {
            let f = self.unit.func(code);
            let env_ty = self.unit.types.as_ref(f.local(f.params[0]).ty).expect("an environment reference").1;
            let l = self.layout(env_ty);
            let malloc = self.module.get_function("malloc").unwrap_or_else(|| {
                self.module.add_function(
                    "malloc",
                    ptr.fn_type(&[self.cx.i64_type().into()], false),
                    Some(LlvmLinkage::External),
                )
            });
            // just before the environment, the function that destroys and
            // frees it; the header is padded to keep the environment aligned
            let header = l.align.max(8);
            let size = self.cx.i64_type().const_int(header + l.size.max(1), false);
            let block = ok(self.b.build_call(malloc, &[size.into()], ""))
                .try_as_basic_value()
                .basic()
                .expect("malloc returns an address")
                .into_pointer_value();
            let failed = ok(self.b.build_is_null(block, ""));
            self.abort_if(failed, Code::Ra07, span);
            let whole = super::func::Addr { ptr: block, align: 8 };
            let base = super::func::Addr {
                ptr: self.offset(whole, header).ptr,
                align: l.align,
            };
            let glue = self.env_glue(code, env_ty, header);
            let slot = self.offset(whole, header - 8);
            ok(self.b.build_store(slot.ptr, glue.as_global_value().as_pointer_value()));
            let offsets = layout::fields_of(&self.unit.types, env_ty).offsets;
            for (c, &off) in captures.iter().zip(&offsets) {
                let at = self.offset(base, off);
                self.store_expr(c, at);
            }
            base.ptr
        };
        let code = body.as_global_value().as_pointer_value();
        Some(self.pair(env, code).as_basic_value_enum())
    }
}
