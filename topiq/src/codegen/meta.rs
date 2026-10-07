//! Placing a unit's metadata in its object.
//!
//! The metadata text becomes a constant in a section of its own,
//! [`crate::meta::SECTION`]. Two things keep it where it belongs:
//!
//! - it is listed in `llvm.used`, so the optimiser keeps it although no code
//!   refers to it;
//! - it carries LLVM's `exclude` mark, which on Windows makes the section one
//!   the linker drops, so the metadata travels in the `.tcu` for other units
//!   and the linker to read, but not into the executable.

use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};

/// The symbol the metadata constant is given. Private to the object.
pub const SYMBOL: &str = "topiq.metadata";

/// Places `text` in `module`'s metadata section.
pub fn embed<'ctx>(cx: &'ctx Context, module: &Module<'ctx>, text: &str) {
    let g = super::func::private_constant(module, cx.const_string(text.as_bytes(), true), SYMBOL, 1);
    g.set_section(Some(crate::meta::SECTION));
    let exclude = cx.get_kind_id("exclude");
    g.set_metadata(cx.metadata_node(&[]), exclude);

    let ptr = cx.ptr_type(AddressSpace::default());
    let used = module.add_global(ptr.array_type(1), None, "llvm.used");
    used.set_linkage(Linkage::Appending);
    used.set_section(Some("llvm.metadata"));
    used.set_initializer(&ptr.const_array(&[g.as_pointer_value()]));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_metadata_is_kept_in_its_own_section() {
        let cx = Context::create();
        let m = cx.create_module("t");
        embed(&cx, &m, "Metadata {}");
        assert!(m.verify().is_ok(), "{}", m.verify().unwrap_err());
        let ir = m.print_to_string().to_string();
        assert!(ir.contains("section \".topiq\""), "{ir}");
        assert!(ir.contains("!exclude"), "{ir}");
        assert!(ir.contains("@llvm.used"), "{ir}");
    }
}
