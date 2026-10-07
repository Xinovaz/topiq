//! LLVM target initialisation and host target-machine construction.
//!
//! Code is generated for the host. Only the X86 target is initialised, and the
//! crate enables only inkwell's `target-x86` feature, so an LLVM built for X86
//! alone suffices; the default `target-all` would need every target's
//! `LLVMInitialize*` symbols at link time.

use inkwell::OptimizationLevel;
use inkwell::targets::{
    CodeModel, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};

/// Why a target machine could not be built.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TargetError {
    /// The triple that was asked for.
    pub triple: String,
    /// What LLVM said.
    pub detail: String,
}

impl std::fmt::Display for TargetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cannot build a target machine for {}: {}",
            self.triple, self.detail
        )
    }
}

impl std::error::Error for TargetError {}

/// Initialises the X86 target.
///
/// Idempotent: LLVM tolerates repeated initialisation, and every entry point in
/// this module calls it, so no caller has to remember to.
pub fn initialize() {
    Target::initialize_x86(&InitializationConfig::default());
}

/// The host's target triple.
pub fn host_triple() -> TargetTriple {
    TargetMachine::get_default_triple()
}

/// The host CPU name LLVM would select (e.g. `znver3` or `skylake`).
pub fn host_cpu() -> String {
    TargetMachine::get_host_cpu_name().to_string()
}

/// The host CPU feature string LLVM would select.
pub fn host_features() -> String {
    TargetMachine::get_host_cpu_features().to_string()
}

/// Builds a target machine for `triple`.
///
/// # Errors
///
/// Returns [`TargetError`] if the triple names a target this LLVM was not built
/// with, or if LLVM declines to create the machine.
pub fn machine_for(
    triple: &TargetTriple,
    opt: OptimizationLevel,
) -> Result<TargetMachine, TargetError> {
    initialize();
    let describe = || triple.as_str().to_string_lossy().into_owned();
    let target = Target::from_triple(triple).map_err(|e| TargetError {
        triple: describe(),
        detail: e.to_string(),
    })?;
    target
        .create_target_machine(
            triple,
            &host_cpu(),
            &host_features(),
            opt,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| TargetError {
            triple: describe(),
            detail: "LLVM returned no target machine".to_owned(),
        })
}

/// Builds a target machine for the host at the default optimisation level.
///
/// # Errors
///
/// As [`machine_for`].
pub fn host_machine() -> Result<TargetMachine, TargetError> {
    machine_for(&host_triple(), OptimizationLevel::Default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization_is_idempotent() {
        initialize();
        initialize();
    }

    #[test]
    fn the_host_triple_is_non_empty() {
        let t = host_triple();
        assert!(!t.as_str().to_string_lossy().is_empty());
    }

    #[test]
    fn a_host_target_machine_can_be_built() {
        // proves the LLVM link end to end, resolving libLLVM's symbols and
        // building a real TargetMachine, so a missing or foreign LLVM fails here
        // rather than much later
        let machine = host_machine().expect("host target machine");
        let triple = machine.get_triple();
        assert!(!triple.as_str().to_string_lossy().is_empty());
    }

    #[test]
    fn the_host_target_is_x86() {
        // the bundled LLVM is an X86-only build; if this ever stops holding,
        // Cargo.toml's inkwell feature list needs the matching target feature
        let triple = host_triple();
        let t = triple.as_str().to_string_lossy().into_owned();
        assert!(
            t.starts_with("x86_64") || t.starts_with("i686") || t.starts_with("i386"),
            "unexpected host triple {t}"
        );
    }

    #[test]
    fn an_unknown_triple_is_an_error_not_a_panic() {
        let bogus = TargetTriple::create("nonesuch-unknown-unknown");
        let err = machine_for(&bogus, OptimizationLevel::None).unwrap_err();
        assert_eq!(err.triple, "nonesuch-unknown-unknown");
        assert!(!err.detail.is_empty());
        assert!(err.to_string().contains("nonesuch"));
    }

    #[test]
    fn host_cpu_and_features_are_readable() {
        initialize();
        // the CPU name is always something; features may legitimately be empty
        assert!(!host_cpu().is_empty());
        let _ = host_features();
    }
}
