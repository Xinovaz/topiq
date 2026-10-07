//! What an image (a program, or a module loaded into one) records about
//! itself.
//!
//! A module is compiled and linked on its own, into a library the operating
//! system loads, with its own copy of every unit it uses. Loading it is only
//! sound if it agrees with the program on everything they share, which is
//! what the linker checks when units are linked together: the edition of the
//! language, the layout of every type both hold, and which unit added which
//! method to an `[open]` type. Each image therefore carries a *descriptor*, a
//! constant the library reads:
//!
//! ```text
//! struct Descriptor {
//!     edition: usize,
//!     init: usize,               // a module's initialisers, run once loaded
//!     symbols: *[Symbol],        // a module's functions, by qualified name
//!     types: *[TypeRecord],      // every type, with its layout digest
//!     methods: *[MethodRecord],  // methods added to other units' types
//!     circuits: *[CircuitRecord], // entry operators' circuits
//! }
//! struct Symbol { name: *[char], sig: *[char], code: usize }
//! struct TypeRecord { name: *[char], digest: *[char], table: usize }
//! struct MethodRecord { owner: *[char], name: *[char] }
//! struct CircuitRecord { name: *[char], sig: *[char], doc: *[char] }
//! ```
//!
//! An image embeds the circuit of every `[entry]` operator of its quantum
//! units, as its document ([`crate::circuit::document`]), the hybrid
//! image. A classical unit holds such an operator by a circuit handle, an
//! object referring to the document, which the image defines under the
//! symbol an imported object of that name has.
//!
//! `lib/module.tq` declares the same structures. A program's descriptor is
//! the symbol [`HOST`]; a module's is [`MODULE`], exported so the loader can
//! find it by name. A signature is written as a program writes the type,
//! with every type qualified by its unit, so the two images compare them as
//! text: each has its own tables, at its own addresses.

use inkwell::context::Context;
use inkwell::module::{Linkage as LlvmLinkage, Module};
use inkwell::values::{BasicValueEnum, StructValue};
use inkwell::{AddressSpace, DLLStorageClass};

use super::func::private_constant;
use super::{mangle, types, typeinfo};

/// The symbol of a program's own descriptor.
pub const HOST: &str = "topiq$$host";

/// The symbol a module exports its descriptor under.
pub const MODULE: &str = "topiq_module";

/// A function a module offers.
#[derive(Clone, Debug)]
pub struct Symbol {
    /// Its qualified name, `unit::function`.
    pub name: String,
    /// Its signature, as a program writes the type, every type qualified.
    pub sig: String,
    /// Its symbol in the objects.
    pub symbol: String,
}

/// A structure or enumeration an image holds.
#[derive(Clone, Debug)]
pub struct Type {
    /// Its qualified name.
    pub name: String,
    /// Its layout digest.
    pub digest: String,
}

/// What an image records.
#[derive(Clone, Debug, Default)]
pub struct Image {
    /// Whether it is a module rather than a program.
    pub module: bool,
    /// The functions it offers.
    pub symbols: Vec<Symbol>,
    /// Every structure and enumeration it holds.
    pub types: Vec<Type>,
    /// Each method a unit of it adds to another unit's type: the type's
    /// qualified name, and the method's, with `$` for an operator.
    pub methods: Vec<(String, String)>,
    /// Each `[entry]` operator of its quantum units, with its circuit.
    pub circuits: Vec<Circuit>,
}

/// An `[entry]` operator's circuit an image embeds.
#[derive(Clone, Debug)]
pub struct Circuit {
    /// The operator's qualified name (i.e. `unit::operator`).
    pub name: String,
    /// Its signature, as a program writes the type, every type qualified.
    pub sig: String,
    /// The circuit's document.
    pub document: String,
    /// The symbol of the handle a classical unit holds it by.
    pub handle: String,
}

/// Defines the image's descriptor in `module`.
pub fn define<'ctx>(cx: &'ctx Context, module: &Module<'ctx>, image: &Image) {
    let i64 = cx.i64_type();
    let ptr = cx.ptr_type(AddressSpace::default());
    let slice = types::fat_slice(cx);
    let mut strings = 0usize;
    let mut text = |s: &str| -> StructValue<'ctx> {
        let chars: Vec<_> = s.chars().map(|c| cx.i32_type().const_int(u64::from(u32::from(c)), false)).collect();
        let g = private_constant(module, cx.i32_type().const_array(&chars), &format!("topiq$$image.text.{strings}"), 4);
        strings += 1;
        slice.const_named_struct(&[g.as_pointer_value().into(), i64.const_int(chars.len() as u64, false).into()])
    };
    let address = |name: &str, function: bool| -> BasicValueEnum<'ctx> {
        let p = if function {
            let f = module
                .get_function(name)
                .unwrap_or_else(|| module.add_function(name, cx.void_type().fn_type(&[], false), Some(LlvmLinkage::External)));
            f.as_global_value().as_pointer_value()
        } else {
            let g = module.get_global(name).unwrap_or_else(|| {
                let g = module.add_global(cx.i8_type(), None, name);
                g.set_linkage(LlvmLinkage::External);
                g
            });
            g.as_pointer_value()
        };
        p.const_to_int(i64).into()
    };
    let record = cx.struct_type(&[slice.into(), slice.into(), i64.into()], false);
    let pair = cx.struct_type(&[slice.into(), slice.into()], false);
    let symbols: Vec<StructValue<'ctx>> = image
        .symbols
        .iter()
        .map(|s| record.const_named_struct(&[text(&s.name).into(), text(&s.sig).into(), address(&s.symbol, true)]))
        .collect();
    let mut weak = Vec::new();
    let types: Vec<StructValue<'ctx>> = image
        .types
        .iter()
        .map(|t| {
            let table = typeinfo::symbol(&t.name);
            weak.push(table.clone());
            record.const_named_struct(&[text(&t.name).into(), text(&t.digest).into(), address(&table, false)])
        })
        .collect();
    let methods: Vec<StructValue<'ctx>> = image
        .methods
        .iter()
        .map(|(owner, name)| pair.const_named_struct(&[text(owner).into(), text(name).into()]))
        .collect();
    let array = |ty: inkwell::types::StructType<'ctx>, items: &[StructValue<'ctx>], what: &str| -> StructValue<'ctx> {
        let at = if items.is_empty() {
            ptr.const_null()
        } else {
            private_constant(module, ty.const_array(items), &format!("topiq$$image.{what}"), 8).as_pointer_value()
        };
        slice.const_named_struct(&[at.into(), i64.const_int(items.len() as u64, false).into()])
    };
    // each circuit's document, which its record and its handle both refer
    // to
    let triple = cx.struct_type(&[slice.into(), slice.into(), slice.into()], false);
    let mut circuits = Vec::with_capacity(image.circuits.len());
    for c in &image.circuits {
        let document = text(&c.document);
        let handle = module.add_global(slice, None, &c.handle);
        handle.set_initializer(&document);
        handle.set_constant(true);
        handle.set_alignment(8);
        circuits.push(triple.const_named_struct(&[text(&c.name).into(), text(&c.sig).into(), document.into()]));
    }
    let symbols = array(record, &symbols, "symbols");
    let types = array(record, &types, "types");
    let methods = array(pair, &methods, "methods");
    let circuits = array(triple, &circuits, "circuits");
    let init = if image.module {
        address(mangle::STARTUP, true)
    } else {
        i64.const_zero().into()
    };
    let descriptor = cx.const_struct(
        &[
            i64.const_int(u64::from(crate::LANGUAGE_EDITION), false).into(),
            init,
            symbols.into(),
            types.into(),
            methods.into(),
            circuits.into(),
        ],
        false,
    );
    let name = if image.module { MODULE } else { HOST };
    let g = module.add_global(descriptor.get_type(), None, name);
    g.set_initializer(&descriptor);
    g.set_constant(true);
    g.set_alignment(8);
    if image.module {
        g.set_dll_storage_class(DLLStorageClass::Export);
    }
    // a table no unit of the image made is the blank one
    for sym in weak {
        let option = format!("/ALTERNATENAME:{sym}={}", typeinfo::BLANK);
        let node = cx.metadata_node(&[cx.metadata_string(&option).into()]);
        let _ = module.add_global_metadata("llvm.linker.options", &node);
    }
}
