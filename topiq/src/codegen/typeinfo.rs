//! Run-time type information: each type's `TypeInfo` table.
//!
//! A table is a constant in the image, laid out as `core`'s `TypeInfo`
//! structure, with its fields, methods, variants and annotations in constant
//! arrays beside it. It is named after its type's qualified name, and a unit
//! that makes one makes it in a COMDAT, so the linker keeps one per program
//! and its address is the type's identity.
//!
//! # References carry their referent's table
//!
//! A reference's second word is its referent's table. Every unit refers to
//! the table by its name, but only units that need one make it; the others
//! refer to it weakly. On Windows each weak reference is a linker directive
//! naming a blank table to use when no unit made the real one; elsewhere it
//! is a weak symbol, which is null then. So the second word is the real table
//! whenever any unit of the program made it, and every question the program
//! can ask of a table (`@typeinfo`, a downcast, erasure to `*any`) makes
//! it.
//!
//! # Tables across a module boundary
//!
//! A module loaded while the program runs has its own tables, at its own
//! addresses. A downcast or a `dyn` check therefore compares tables by
//! address and, failing that, by qualified name; loading a module has
//! already checked that a type of one name is laid out the same on both
//! sides.

use inkwell::basic_block::BasicBlock;
use inkwell::types::BasicTypeEnum;
use inkwell::module::Linkage as LlvmLinkage;
use inkwell::IntPredicate;
use inkwell::values::{BasicValueEnum, IntValue, PointerValue, StructValue};

use crate::tir::{AdtId, AdtKind, Callee, Expr, Ty, TypeInfoTypes, layout};

use super::func::{Addr, Dest, Lowering, ok};
use super::types;

/// The blank table a weak reference falls back on.
pub const BLANK: &str = "topiq$$ti$blank";

/// The symbol of `ty`'s table, from its qualified name, with everything but
/// letters, digits and `_` written as `$` and two hexadecimal digits.
pub fn symbol(qualified: &str) -> String {
    let mut out = String::from("topiq$ti$");
    for b in qualified.bytes() {
        if b.is_ascii_alphanumeric() || b == b'_' {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("${b:02x}"));
        }
    }
    out
}

impl<'ctx> Lowering<'ctx, '_> {
    /// The address of `ty`'s table: this unit's own if it makes one, and
    /// otherwise a weak reference to whichever unit's does.
    pub(super) fn table(&mut self, ty: Ty) -> PointerValue<'ctx> {
        let name = self.unit.types.qualified(ty, self.interner, &self.unit.name);
        let sym = symbol(&name);
        if let Some(g) = self.module.get_global(&sym) {
            return g.as_pointer_value();
        }
        let g = self.module.add_global(self.cx.i8_type(), None, &sym);
        if self.windows() {
            g.set_linkage(LlvmLinkage::External);
            self.weak_tables.push(sym);
        } else {
            g.set_linkage(LlvmLinkage::ExternalWeak);
        }
        g.as_pointer_value()
    }

    /// `@typeinfo(T)`: a `*const TypeInfo` to `ty`'s table.
    pub(super) fn typeinfo_ref(&mut self, ty: Ty) -> StructValue<'ctx> {
        let table = self.table(ty);
        self.typeinfo_value(table)
    }

    /// A `*const TypeInfo` to the table at `table`: that address, and
    /// `TypeInfo`'s own table as the type of what it refers to.
    pub(super) fn typeinfo_value(&mut self, table: PointerValue<'ctx>) -> StructValue<'ctx> {
        let own = match self.unit.typeinfo {
            Some(ti) => self.table(Ty::Adt(ti.info)),
            None => self.null(),
        };
        self.pair(table, own)
    }

    /// Whether the tables `carried` and `wanted` describe one type: the same
    /// table, or (for a value made in a module loaded while the program
    /// runs, which has tables of its own) tables of the same qualified name.
    /// Loading a module checks that every type it shares with the program is
    /// laid out the same, so the name says it all.
    pub(super) fn same_type(&mut self, carried: PointerValue<'ctx>, wanted: PointerValue<'ctx>) -> IntValue<'ctx> {
        const NAME: &str = "topiq$$same.type";
        let i1 = self.cx.bool_type();
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let f = self.module.get_function(NAME).unwrap_or_else(|| {
            let f = self.module.add_function(NAME, i1.fn_type(&[ptr.into(), ptr.into()], false), Some(LlvmLinkage::Private));
            let b = self.cx.create_builder();
            let (a, w) = (
                f.get_nth_param(0).expect("a").into_pointer_value(),
                f.get_nth_param(1).expect("b").into_pointer_value(),
            );
            let entry = self.cx.append_basic_block(f, "entry");
            let names = self.cx.append_basic_block(f, "names");
            let compare = self.cx.append_basic_block(f, "compare");
            let yes = self.cx.append_basic_block(f, "yes");
            let no = self.cx.append_basic_block(f, "no");
            b.position_at_end(entry);
            let same = ok(b.build_int_compare(IntPredicate::EQ, a, w, "same"));
            let a_null = ok(b.build_is_null(a, "a.null"));
            let w_null = ok(b.build_is_null(w, "b.null"));
            let either = ok(b.build_or(a_null, w_null, "either.null"));
            let other = self.cx.append_basic_block(f, "other");
            ok(b.build_conditional_branch(same, yes, other));
            b.position_at_end(other);
            ok(b.build_conditional_branch(either, no, names));
            // a table starts with its name, a slice of characters
            b.position_at_end(names);
            let slice = types::fat_slice(self.cx);
            let an = ok(b.build_load(slice, a, "a.name")).into_struct_value();
            let wn = ok(b.build_load(slice, w, "b.name")).into_struct_value();
            let a_at = ok(b.build_extract_value(an, 0, "")).into_pointer_value();
            let w_at = ok(b.build_extract_value(wn, 0, "")).into_pointer_value();
            let a_len = ok(b.build_extract_value(an, 1, "")).into_int_value();
            let w_len = ok(b.build_extract_value(wn, 1, "")).into_int_value();
            let lens = ok(b.build_int_compare(IntPredicate::EQ, a_len, w_len, "same.length"));
            ok(b.build_conditional_branch(lens, compare, no));
            b.position_at_end(compare);
            let i64 = self.cx.i64_type();
            let bytes = ok(b.build_int_mul(a_len, i64.const_int(4, false), "bytes"));
            let memcmp = self.runtime_function("memcmp", self.cx.i32_type().fn_type(&[ptr.into(), ptr.into(), i64.into()], false));
            let r = ok(b.build_call(memcmp, &[a_at.into(), w_at.into(), bytes.into()], "cmp"))
                .try_as_basic_value()
                .basic()
                .expect("an int")
                .into_int_value();
            let equal = ok(b.build_int_compare(IntPredicate::EQ, r, self.cx.i32_type().const_zero(), "equal"));
            ok(b.build_conditional_branch(equal, yes, no));
            b.position_at_end(yes);
            ok(b.build_return(Some(&i1.const_int(1, false))));
            b.position_at_end(no);
            ok(b.build_return(Some(&i1.const_zero())));
            f
        });
        ok(self.b.build_call(f, &[carried.into(), wanted.into()], "same.type"))
            .try_as_basic_value()
            .basic()
            .expect("a bool")
            .into_int_value()
    }

    /// `p as *T`, checked: `Some(p)` when the table `p` carries is `T`'s, and
    /// `None` otherwise, written as the `Opt` it is.
    pub(super) fn downcast(&mut self, r: &Expr, target: Ty, ty: Ty, dest: Dest<'ctx>) -> Option<BasicValueEnum<'ctx>> {
        let p = self.value(r)?.into_struct_value();
        let carried = ok(self.b.build_extract_value(p, 1, "carried")).into_pointer_value();
        let wanted = self.table(target);
        let same = self.same_type(carried, wanted);
        let Ty::Adt(opt) = ty else {
            unreachable!("a downcast gives an `Opt`")
        };
        let out = match dest {
            Dest::Mem(a) => a,
            Dest::Value => self.temp(ty),
        };
        let def = self.unit.types.adt(opt).clone();
        let some = def.variants().iter().position(|v| !v.fields.is_empty()).expect("`Some`");
        let none = def.variants().iter().position(|v| v.fields.is_empty()).expect("`None`");
        let l = layout::enumeration(&self.unit.types, opt);
        let tag = ok(self.b.build_select(
            same,
            types::const_int(self.cx, l.tag, some as i128),
            types::const_int(self.cx, l.tag, none as i128),
            "",
        ));
        self.store_scalar(tag, out);
        // the payload is written either way; where the tag says `None`
        // nothing reads it
        let payload = self.offset(out, l.variants[some][0]);
        self.store_scalar(p.into(), payload);
        None
    }

    /// The body of a `dyn`'s destroying function: the value, by the `drop`
    /// its table records, then the buffer it is in.
    pub(super) fn destroy_dyn(&mut self, at: Addr<'ctx>) {
        let fat = types::fat_ref(self.cx);
        let d = ok(self.b.build_load(fat, at.ptr, "")).into_struct_value();
        let value = ok(self.b.build_extract_value(d, 0, "value")).into_pointer_value();
        let table = ok(self.b.build_extract_value(d, 1, "table")).into_pointer_value();
        let func = self.func.expect("inside a function");
        let has = self.cx.append_basic_block(func, "dyn.value");
        let glue_block = self.cx.append_basic_block(func, "dyn.glue");
        let free_block = self.cx.append_basic_block(func, "dyn.free");
        let done = self.cx.append_basic_block(func, "dyn.done");
        let empty = ok(self.b.build_is_null(value, ""));
        ok(self.b.build_conditional_branch(empty, done, has));

        self.b.position_at_end(has);
        self.destroy_through(table, value, glue_block, free_block);
        self.b.position_at_end(free_block);
        let free = self.free_fn();
        ok(self.b.build_call(free, &[value.into()], ""));
        ok(self.b.build_unconditional_branch(done));

        self.b.position_at_end(done);
    }

    /// Destroys the value at `value` with the destroying function `table`
    /// records, if it records one, then continues at `after`. `glue_block`
    /// is an empty block to hold the call.
    pub(super) fn destroy_through(
        &mut self,
        table: PointerValue<'ctx>,
        value: PointerValue<'ctx>,
        glue_block: BasicBlock<'ctx>,
        after: BasicBlock<'ctx>,
    ) {
        let ptr = self.cx.ptr_type(inkwell::AddressSpace::default());
        let glue = match self.table_entry(table, 9, ptr.into()) {
            Some(g) => g.into_pointer_value(),
            None => self.null(),
        };
        let none = ok(self.b.build_is_null(glue, ""));
        ok(self.b.build_conditional_branch(none, after, glue_block));

        self.b.position_at_end(glue_block);
        let fn_type = self.cx.void_type().fn_type(&[ptr.into()], false);
        ok(self.b.build_indirect_call(fn_type, glue, &[value.into()], ""));
        ok(self.b.build_unconditional_branch(after));
    }

    /// Entry `index` of the table at `table`, read as `ty`; `None` when no
    /// table type is at hand.
    pub(super) fn table_entry(
        &mut self,
        table: PointerValue<'ctx>,
        index: usize,
        ty: BasicTypeEnum<'ctx>,
    ) -> Option<BasicValueEnum<'ctx>> {
        let ti = self.unit.typeinfo?;
        let offset = layout::fields_of(&self.unit.types, Ty::Adt(ti.info)).offsets[index];
        let slot = self.offset(Addr { ptr: table, align: 8 }, offset);
        Some(ok(self.b.build_load(ty, slot.ptr, "")))
    }

    /// Whether the target is Windows, where a weak reference is a linker
    /// directive.
    pub(super) fn windows(&self) -> bool {
        self.module.get_triple().as_str().to_string_lossy().contains("windows")
    }

    /// Makes every table the unit needs. Run before any code is lowered, so
    /// that code referring to a table finds this unit's own.
    pub(super) fn emit_tables(&mut self) {
        let Some(ti) = self.unit.typeinfo else { return };
        // every table has one shape, so all are declared first, and may then
        // refer to one another in any order
        let shape = self.blank_record(ti).get_type();
        let tables = self.unit.tables.clone();
        let mut globals = Vec::new();
        for &t in &tables {
            let name = self.unit.types.qualified(t, self.interner, &self.unit.name);
            let sym = symbol(&name);
            let g = self.module.add_global(shape, None, &sym);
            g.set_linkage(LlvmLinkage::LinkOnceODR);
            let comdat = self.module.get_or_insert_comdat(&sym);
            comdat.set_selection_kind(inkwell::comdat::ComdatSelectionKind::Any);
            g.set_comdat(comdat);
            g.set_constant(true);
            g.set_alignment(8);
            globals.push(g);
        }
        for (&t, g) in tables.iter().zip(globals) {
            let value = self.table_value(t, ti);
            g.set_initializer(&value);
        }
    }

    /// The linker directives for the tables this unit only refers to. Run
    /// once all code is lowered.
    pub(super) fn emit_weak_directives(&mut self) {
        for sym in std::mem::take(&mut self.weak_tables) {
            let option = format!("/ALTERNATENAME:{sym}={BLANK}");
            let node = self.cx.metadata_node(&[self.cx.metadata_string(&option).into()]);
            let _ = self.module.add_global_metadata("llvm.linker.options", &node);
        }
    }

    /// A table with nothing in it.
    fn blank_record(&self, ti: TypeInfoTypes) -> StructValue<'ctx> {
        let i64 = self.cx.i64_type();
        let kind = layout::enumeration(&self.unit.types, ti.kind).tag;
        let slice = self.empty_slice();
        let fat = self.fat(self.null(), self.null()).into();
        self.record(
            ti.info,
            &[
                (0, slice),
                (1, i64.const_zero().into()),
                (2, i64.const_zero().into()),
                (3, types::const_int(self.cx, kind, 0).into()),
                (4, slice),
                (5, slice),
                (6, slice),
                (7, slice),
                (8, self.cx.i8_type().const_zero().into()),
                (9, fat),
                (10, fat),
                (11, i64.const_zero().into()),
                (12, i64.const_zero().into()),
                (13, i64.const_zero().into()),
            ],
        )
    }

    /// The contents of `ty`'s table.
    fn table_value(&mut self, ty: Ty, ti: TypeInfoTypes) -> StructValue<'ctx> {
        let qualified = self.unit.types.qualified(ty, self.interner, &self.unit.name);
        let sym = symbol(&qualified);
        let name = self.text(&qualified);
        let l = layout::of(&self.unit.types, ty);
        let i64 = self.cx.i64_type();
        let kind = self.kind(ty, ti);
        let (fields, variants, methods, annots) = match ty {
            Ty::Adt(id) => {
                let def = self.unit.types.adt(id).clone();
                let (fields, variants) = match &def.kind {
                    AdtKind::Struct { fields } => {
                        let offsets = layout::fields_of(&self.unit.types, ty).offsets;
                        let list: Vec<(String, Ty, u64)> = fields
                            .iter()
                            .zip(&offsets)
                            .map(|(f, &o)| (self.interner.resolve(f.name).to_owned(), f.ty, o))
                            .collect();
                        (self.field_list(&list, ti, &format!("{sym}$fields")), self.empty_slice())
                    }
                    AdtKind::Enum { variants } => {
                        let el = layout::enumeration(&self.unit.types, id);
                        let mut records = Vec::new();
                        for (i, v) in variants.iter().enumerate() {
                            let list: Vec<(String, Ty, u64)> = v
                                .fields
                                .iter()
                                .zip(&el.variants[i])
                                .map(|(f, &o)| (self.interner.resolve(f.name).to_owned(), f.ty, o))
                                .collect();
                            let fields = self.field_list(&list, ti, &format!("{sym}$variant{i}"));
                            let vname = self.text(self.interner.resolve(v.name));
                            records.push(self.record(ti.variant, &[(0, vname), (1, fields)]));
                        }
                        (self.empty_slice(), self.array_of(ti.variant, &records, &format!("{sym}$variants")))
                    }
                };
                let methods = self.method_list(id, ti, &format!("{sym}$methods"));
                let annots = self.annot_list(id, ti, &format!("{sym}$annots"));
                (fields, variants, methods, annots)
            }
            Ty::Tuple(_) => {
                let offsets = layout::fields_of(&self.unit.types, ty).offsets;
                let elems = self.unit.types.as_tuple(ty).map(<[Ty]>::to_vec).unwrap_or_default();
                let list: Vec<(String, Ty, u64)> = elems
                    .iter()
                    .zip(&offsets)
                    .enumerate()
                    .map(|(i, (&t, &o))| (i.to_string(), t, o))
                    .collect();
                (self.field_list(&list, ti, &format!("{sym}$fields")), self.empty_slice(), self.empty_slice(), self.empty_slice())
            }
            // a function type's parameters, then its result
            Ty::Fn(_) | Ty::Closure(_) | Ty::Circuit(_) => {
                let (params, ret) = self.unit.types.as_sig(ty).map_or((Vec::new(), Ty::Void), |(p, r)| (p.to_vec(), r));
                let mut list: Vec<(String, Ty, u64)> = params.iter().enumerate().map(|(i, &p)| (i.to_string(), p, 0)).collect();
                list.push(("return".to_owned(), ret, 0));
                (self.field_list(&list, ti, &format!("{sym}$fields")), self.empty_slice(), self.empty_slice(), self.empty_slice())
            }
            _ => (self.empty_slice(), self.empty_slice(), self.empty_slice(), self.empty_slice()),
        };
        let drop = if self.needs_drop(ty) {
            let glue = self.drop_glue(ty).as_global_value().as_pointer_value();
            self.fat(glue, self.null())
        } else {
            self.fat(self.null(), self.null())
        };
        let types = &self.unit.types;
        let (elem, len) = match ty {
            Ty::Array(_) => types.as_array(ty).map_or((None, 0), |(e, n)| (Some(e), n)),
            Ty::Growable(_) => (types.as_growable(ty), 0),
            Ty::Slice(_) => (types.as_slice(ty).map(|(_, e)| e), 0),
            _ => (None, 0),
        };
        let elem = match elem {
            Some(e) => self.table_ref(e, ti),
            None => self.fat(self.null(), self.null()).into(),
        };
        let (tag_offset, tag_size) = match ty {
            Ty::Adt(id) if !self.unit.types.adt(id).is_struct() => {
                // the tag is at the start
                let el = layout::enumeration(&self.unit.types, id);
                (0, u64::from(el.tag.bits / 8))
            }
            _ => (0, 0),
        };
        self.record(
            ti.info,
            &[
                (0, name),
                (1, i64.const_int(l.size, false).into()),
                (2, i64.const_int(l.align, false).into()),
                (3, kind),
                (4, fields),
                (5, methods),
                (6, variants),
                (7, annots),
                (8, self.cx.i8_type().const_zero().into()),
                (9, drop.into()),
                (10, elem),
                (11, i64.const_int(len, false).into()),
                (12, i64.const_int(tag_offset, false).into()),
                (13, i64.const_int(tag_size, false).into()),
            ],
        )
    }

    /// A slice of characters holding `text`.
    fn text(&mut self, text: &str) -> BasicValueEnum<'ctx> {
        let sym = self.interner.intern_late(text);
        let (ptr, n) = self.string(sym);
        self.slice_value(ptr, n)
    }

    fn slice_value(&self, ptr: PointerValue<'ctx>, n: u64) -> BasicValueEnum<'ctx> {
        types::fat_slice(self.cx)
            .const_named_struct(&[ptr.into(), self.cx.i64_type().const_int(n, false).into()])
            .into()
    }

    fn empty_slice(&self) -> BasicValueEnum<'ctx> {
        self.slice_value(self.null(), 0)
    }

    /// A reference as a constant: an address and a table.
    fn fat(&self, addr: PointerValue<'ctx>, table: PointerValue<'ctx>) -> StructValue<'ctx> {
        types::fat_ref(self.cx).const_named_struct(&[addr.into(), table.into()])
    }

    /// A `*const TypeInfo` to `ty`'s table.
    fn table_ref(&mut self, ty: Ty, ti: TypeInfoTypes) -> BasicValueEnum<'ctx> {
        let table = self.table(ty);
        let own = self.table(Ty::Adt(ti.info));
        self.fat(table, own).into()
    }

    /// The `TypeKind` of `ty`.
    fn kind(&self, ty: Ty, ti: TypeInfoTypes) -> BasicValueEnum<'ctx> {
        let kind = self.unit.types.kind_name(ty);
        let index = self
            .unit
            .types
            .adt(ti.kind)
            .variants()
            .iter()
            .position(|v| self.interner.resolve(v.name) == kind)
            .unwrap_or(0);
        let el = layout::enumeration(&self.unit.types, ti.kind);
        types::const_int(self.cx, el.tag, index as i128).into()
    }

    /// A slice of `FieldInfo` records.
    fn field_list(&mut self, list: &[(String, Ty, u64)], ti: TypeInfoTypes, name: &str) -> BasicValueEnum<'ctx> {
        let mut records = Vec::new();
        for (n, t, offset) in list {
            let n = self.text(n);
            let t = self.table_ref(*t, ti);
            let o = self.cx.i64_type().const_int(*offset, false).into();
            records.push(self.record(ti.field, &[(0, n), (1, t), (2, o)]));
        }
        self.array_of(ti.field, &records, name)
    }

    /// A slice of `MethodInfo` records: each method's name, erased
    /// signature, and code.
    fn method_list(&mut self, id: AdtId, ti: TypeInfoTypes, name: &str) -> BasicValueEnum<'ctx> {
        let methods = self.unit.table_methods.get(&id).cloned().unwrap_or_default();
        let mut records = Vec::new();
        for m in methods {
            let n = self.text(self.interner.resolve(m.name));
            let sig = self.table_ref(m.sig, ti);
            let code = match m.callee {
                Callee::Fn(_) | Callee::Extern(_) => self.callee_function(m.callee).0.as_global_value().as_pointer_value(),
            };
            let sig_table = self.table(m.sig);
            let addr = self.fat(code, sig_table).into();
            let call = match self.call_thunk(&m, name, records.len()) {
                Some(f) => self.fat(f.as_global_value().as_pointer_value(), sig_table),
                None => self.fat(self.null(), self.null()),
            };
            records.push(self.record(ti.method, &[(0, n), (1, sig), (2, addr), (3, call.into())]));
        }
        self.array_of(ti.method, &records, name)
    }

    /// A slice of `AnnotInfo` records.
    fn annot_list(&mut self, id: AdtId, ti: TypeInfoTypes, name: &str) -> BasicValueEnum<'ctx> {
        let annots = self.unit.table_annots.get(&id).cloned().unwrap_or_default();
        let mut records = Vec::new();
        for (a, value) in annots {
            let n = self.text(self.interner.resolve(a));
            let v = self.text(value.map_or("", |v| self.interner.resolve(v)));
            records.push(self.record(ti.annot, &[(0, n), (1, v)]));
        }
        self.array_of(ti.annot, &records, name)
    }

    /// Constant records placed in the image as one array, as a slice of them.
    fn array_of(&mut self, adt: AdtId, records: &[StructValue<'ctx>], name: &str) -> BasicValueEnum<'ctx> {
        if records.is_empty() {
            return self.empty_slice();
        }
        let array = records[0].get_type().const_array(records);
        let align = layout::of(&self.unit.types, Ty::Adt(adt)).align as u32;
        let g = super::func::private_constant(self.module, array, name, align);
        self.slice_value(g.as_pointer_value(), records.len() as u64)
    }

    /// A constant laid out as the structure `adt`, with each given field at
    /// its offset and zero bytes between. The layout is the structure's own,
    /// so the program reads it with ordinary field accesses.
    fn record(&self, adt: AdtId, values: &[(usize, BasicValueEnum<'ctx>)]) -> StructValue<'ctx> {
        let ty = Ty::Adt(adt);
        let offsets = layout::fields_of(&self.unit.types, ty).offsets;
        let size = layout::of(&self.unit.types, ty).size;
        let i8 = self.cx.i8_type();
        let mut parts: Vec<BasicValueEnum<'ctx>> = Vec::new();
        let mut at = 0u64;
        let mut sorted: Vec<(u64, BasicValueEnum<'ctx>)> = values.iter().map(|&(i, v)| (offsets[i], v)).collect();
        sorted.sort_by_key(|(o, _)| *o);
        for (offset, v) in sorted {
            if offset > at {
                parts.push(i8.array_type((offset - at) as u32).const_zero().into());
            }
            parts.push(v);
            at = offset + self.store_size(v);
        }
        if size > at {
            parts.push(i8.array_type((size - at) as u32).const_zero().into());
        }
        self.cx.const_struct(&parts, true)
    }

    /// The bytes a constant takes: the sizes this module's values have.
    fn store_size(&self, v: BasicValueEnum<'ctx>) -> u64 {
        match v {
            BasicValueEnum::IntValue(i) => u64::from(i.get_type().get_bit_width().div_ceil(8)),
            BasicValueEnum::PointerValue(_) => 8,
            BasicValueEnum::StructValue(s) => {
                let t = s.get_type();
                if t.is_packed() {
                    t.get_field_types().iter().map(|f| type_size(*f)).sum()
                } else {
                    // a reference or a slice: two eight-byte words
                    16
                }
            }
            BasicValueEnum::ArrayValue(a) => type_size(a.get_type().into()),
            _ => 8,
        }
    }

}

/// The size of a type made of bytes, integers, pointers and packed
/// structures of them.
fn type_size(t: inkwell::types::BasicTypeEnum<'_>) -> u64 {
    use inkwell::types::BasicTypeEnum as T;
    match t {
        T::IntType(i) => u64::from(i.get_bit_width().div_ceil(8)),
        T::PointerType(_) => 8,
        T::ArrayType(a) => u64::from(a.len()) * type_size(a.get_element_type()),
        T::StructType(s) => s.get_field_types().iter().map(|f| type_size(*f)).sum(),
        _ => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::symbol;

    #[test]
    fn a_table_is_named_after_its_type_in_letters_a_linker_accepts() {
        assert_eq!(symbol("geo::P"), "topiq$ti$geo$3a$3aP");
        assert_eq!(symbol("[u8; 4]"), "topiq$ti$$5bu8$3b$204$5d");
    }
}
