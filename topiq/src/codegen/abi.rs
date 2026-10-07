//! How Topiq functions pass values to each other.
//!
//! A value held in registers (an integer, `bool`, `char`, reference or slice)
//! is passed and returned directly.
//!
//! A structure, enumeration or array is passed as the address of a copy the
//! caller made for the purpose. The callee owns that copy for the length of
//! the call: it reads the parameter from there, and may change it, without the
//! caller's own value being touched.
//!
//! Such a value is returned through a hidden first parameter: the caller
//! passes the address the result should be built at, and the callee builds it
//! there and returns nothing. A structure literal returned from a function is
//! therefore built once, directly where the caller wants it.
//!
//! This convention is Topiq's own. For a function taking and returning only
//! integers, floating values, `bool` and `char` it coincides with the
//! platform's C convention, which is what lets code in another language call
//! a function exported with `[export]`; references, slices and aggregates
//! follow the rules above, not the platform's.

use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, FunctionType};

use crate::tir::{Ty, TypeTable};

use super::types;

/// How a function receives its values.
#[derive(Clone, Copy)]
pub struct Signature<'ctx> {
    /// The LLVM function type.
    pub llvm: FunctionType<'ctx>,
    /// Whether the result is built through a hidden first parameter.
    pub sret: bool,
}

/// The signature of a function with these parameter and result types.
pub fn signature<'ctx>(cx: &'ctx Context, types: &TypeTable, params: &[Ty], ret: Ty) -> Signature<'ctx> {
    let ptr = cx.ptr_type(AddressSpace::default());
    let sret = ret.is_aggregate();
    let mut ps: Vec<BasicMetadataTypeEnum<'ctx>> = Vec::with_capacity(params.len() + 1);
    if sret {
        ps.push(ptr.into());
    }
    for &p in params {
        match types::scalar(cx, types, p) {
            Some(t) => ps.push(t.into()),
            None if p.is_aggregate() => ps.push(ptr.into()),
            // a parameter has a value, so this is never reached
            None => {}
        }
    }
    let llvm = match types::scalar(cx, types, ret) {
        Some(r) if !sret => r.fn_type(&ps, false),
        _ => cx.void_type().fn_type(&ps, false),
    };
    Signature { llvm, sret }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tir::IntTy;

    #[test]
    fn scalars_travel_directly() {
        let cx = Context::create();
        let types = TypeTable::new();
        let s = signature(&cx, &types, &[Ty::Int(IntTy::I32), Ty::Bool], Ty::Char);
        assert!(!s.sret);
        assert_eq!(s.llvm.count_param_types(), 2);
        assert!(s.llvm.get_return_type().is_some());
    }

    #[test]
    fn aggregates_travel_by_address_and_return_through_a_hidden_parameter() {
        let cx = Context::create();
        let mut types = TypeTable::new();
        let arr = types.array(Ty::Int(IntTy::U8), 16);
        let s = signature(&cx, &types, &[arr], arr);
        assert!(s.sret);
        assert_eq!(s.llvm.count_param_types(), 2, "the result's address, then the array's");
        assert!(s.llvm.get_return_type().is_none());
    }

    #[test]
    fn a_function_returning_nothing_returns_llvm_void() {
        let cx = Context::create();
        let types = TypeTable::new();
        let s = signature(&cx, &types, &[], Ty::Void);
        assert!(s.llvm.get_return_type().is_none());
        let s = signature(&cx, &types, &[], Ty::Never);
        assert!(s.llvm.get_return_type().is_none());
    }
}
