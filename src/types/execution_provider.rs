//! Private definition of the execution-provider enum.
//! Re-exported from `pipeline_v2` and, when `infer` is enabled, from `onnx`.

/// Where a session would run. Product kernels execute on the CPU only:
/// [`ExecutionProvider::Cpu`] and [`ExecutionProvider::auto`] (which resolves
/// to CPU) are the only values `PipelineBuilder::validate` accepts. The
/// other variants exist for tract builds and are rejected, never silently
/// downgraded. `#[non_exhaustive]`: match with a wildcard arm.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ExecutionProvider {
    Cpu,
    CoreMl,
    Nnapi,
    Cuda,
    XnnPack,
}

impl ExecutionProvider {
    /// Resolves to [`Self::Cpu`]. Product kernels and tract both execute on the CPU.
    pub fn auto() -> Self {
        Self::Cpu
    }

    /// Only [`Self::Cpu`] is available. Pipeline validation rejects the rest.
    pub fn is_available(self) -> bool {
        matches!(self, Self::Cpu)
    }
}
