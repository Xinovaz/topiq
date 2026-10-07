//! Finding a unit's metadata inside its object file.
//!
//! A `.tcu` is an ordinary relocatable object for the host (COFF on Windows)
//! with one extra section, `.topiq`, holding the metadata text. Reading it
//! needs only the object format, not LLVM, so a unit can be checked against
//! a compiled import by a `tqc` built without code generation.

use object::{Object, ObjectSection};

use super::SECTION;

/// The metadata text in an object file's bytes.
///
/// # Errors
///
/// A sentence saying why: the bytes are not an object file, the object has
/// no metadata section (it was not produced by this compiler) or the
/// section is not text.
pub fn read_section(bytes: &[u8]) -> Result<String, String> {
    let file = object::File::parse(bytes).map_err(|e| format!("it is not an object file ({e})"))?;
    let section = file
        .section_by_name(SECTION)
        .ok_or_else(|| format!("it has no `{SECTION}` section, so it was not compiled by tqc from a Topiq unit"))?;
    let data = section
        .data()
        .map_err(|e| format!("its `{SECTION}` section cannot be read ({e})"))?;
    // the section is a C string: the text, then a terminating zero
    let text = data.split(|&b| b == 0).next().unwrap_or_default();
    String::from_utf8(text.to_vec()).map_err(|_| format!("its `{SECTION}` section is not UTF-8 text"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn something_that_is_not_an_object_is_refused() {
        let e = read_section(b"not an object").unwrap_err();
        assert!(e.contains("not an object file"), "{e}");
    }
}
