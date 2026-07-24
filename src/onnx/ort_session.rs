//! ONNX Runtime (`ort`) implementation of [`InferenceRuntime`].
//!
//! **This is the only module that may import `ort::`.** Neural stages must go
//! through [`InferenceRuntime`] / [`RuntimeSession`](super::RuntimeSession)
//! instead. The optional tract backend lives in `tract_session` behind
//! `backend-tract`.

use super::runtime::{
    InferenceError, InferenceRuntime, InferenceTensor, NamedTensor, TensorData,
};
use super::{ExecutionProvider, validate_onnx_header};
use std::path::Path;

/// Thin wrapper around `ort::session::Session` implementing [`InferenceRuntime`].
///
/// Construction goes through [`OrtSession::from_path`] / [`build_session_with_ep`](super::build_session_with_ep)
/// so the validate-before-build invariant and EP wiring stay centralized.
#[derive(Debug)]
pub struct OrtSession {
    session: ort::session::Session,
    input_names: Vec<String>,
}

impl OrtSession {
    /// Load a model from `path` with the given execution provider and optional
    /// intra-op thread pin. Validates the ONNX header before ort parses the file.
    pub fn from_path(
        model_path: &Path,
        ep: ExecutionProvider,
        intra_threads: Option<usize>,
    ) -> anyhow::Result<Self> {
        validate_onnx_header(model_path)?;
        // ort::Error is not Send+Sync, so it cannot ride `?` into anyhow — stringify.
        let mut builder =
            ort::session::Session::builder().map_err(|e| anyhow::anyhow!("session builder: {e}"))?;
        if let Some(n) = intra_threads {
            builder = builder
                .with_intra_threads(n)
                .map_err(|e| anyhow::anyhow!("intra threads: {e}"))?;
        }
        match ep {
            ExecutionProvider::Cpu => {}
            ExecutionProvider::CoreMl => {
                #[cfg(all(feature = "coreml", target_os = "macos", target_arch = "aarch64"))]
                {
                    let coreml = ort::execution_providers::CoreMLExecutionProvider::default();
                    builder = builder
                        .with_execution_providers([coreml.build()])
                        .map_err(|e| anyhow::anyhow!("coreml ep: {e}"))?;
                }
                #[cfg(not(all(feature = "coreml", target_os = "macos", target_arch = "aarch64")))]
                tracing::warn!(
                    "execution provider CoreMl is not compiled in (needs the `coreml` feature on \
                     macOS aarch64) — falling back to CPU"
                );
            }
            ExecutionProvider::XnnPack => {
                #[cfg(feature = "xnnpack")]
                {
                    let xnnpack = ort::execution_providers::XNNPACKExecutionProvider::default();
                    builder = builder
                        .with_execution_providers([xnnpack.build()])
                        .map_err(|e| anyhow::anyhow!("xnnpack ep: {e}"))?;
                }
                #[cfg(not(feature = "xnnpack"))]
                tracing::warn!(
                    "execution provider XnnPack is not compiled in (needs the `xnnpack` feature) —                  falling back to CPU"
                );
            }
            ExecutionProvider::Nnapi | ExecutionProvider::Cuda => {
                tracing::warn!("execution provider {ep:?} is not wired yet — falling back to CPU");
            }
        }
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("commit_from_file: {e}"))?;
        let input_names = session
            .inputs()
            .iter()
            .map(|i| i.name().to_owned())
            .collect();
        Ok(Self {
            session,
            input_names,
        })
    }

    /// Custom `metadata_props` key/value pairs from the ONNX model.
    ///
    /// Empty when the model carries no custom metadata. Used by the
    /// self-describing model config loader (`models::metadata`).
    pub fn custom_metadata_props(&self) -> Result<std::collections::HashMap<String, String>, String> {
        let meta = self
            .session
            .metadata()
            .map_err(|e| format!("session metadata: {e}"))?;
        let keys = meta
            .custom_keys()
            .map_err(|e| format!("custom metadata keys: {e}"))?;
        let mut out = std::collections::HashMap::with_capacity(keys.len());
        for key in keys {
            if let Some(value) = meta.custom(&key) {
                out.insert(key, value);
            }
        }
        Ok(out)
    }
}

impl InferenceRuntime for OrtSession {
    fn input_names(&self) -> &[String] {
        &self.input_names
    }

    fn run(
        &mut self,
        inputs: &[NamedTensor<'_>],
    ) -> Result<Vec<InferenceTensor>, InferenceError> {
        let mut owned: Vec<(String, ort::session::SessionInputValue<'_>)> =
            Vec::with_capacity(inputs.len());
        // Build owned ort tensors first so SessionInputValue can borrow them.
        let mut tensors = Vec::with_capacity(inputs.len());
        for nt in inputs {
            tensors.push((
                nt.name.to_owned(),
                to_ort_tensor(nt.tensor).map_err(InferenceError::Run)?,
            ));
        }
        for (name, tensor) in &tensors {
            owned.push((name.clone(), tensor.into()));
        }
        let outputs = self
            .session
            .run(owned)
            .map_err(|e| InferenceError::Run(e.to_string()))?;
        extract_outputs(&outputs)
    }

    fn run_ordered(
        &mut self,
        inputs: &[&InferenceTensor],
    ) -> Result<Vec<InferenceTensor>, InferenceError> {
        let tensors: Result<Vec<_>, _> = inputs.iter().map(|t| to_ort_tensor(t)).collect();
        let tensors = tensors.map_err(InferenceError::Run)?;
        let values: Vec<ort::session::SessionInputValue<'_>> =
            tensors.iter().map(|t| t.into()).collect();
        let outputs = self
            .session
            .run(values.as_slice())
            .map_err(|e| InferenceError::Run(e.to_string()))?;
        extract_outputs(&outputs)
    }
}

fn to_ort_tensor(t: &InferenceTensor) -> Result<ort::value::DynTensor, String> {
    // Clone data into an owned ort Tensor. Inference cost dominates; DER is
    // unaffected. Scalar i64 uses empty shape (ONNX 0-d).
    match &t.data {
        TensorData::F32(data) => {
            let tensor = ort::value::Tensor::from_array((t.shape.clone(), data.clone()))
                .map_err(|e| e.to_string())?;
            Ok(tensor.upcast())
        }
        TensorData::I64(data) => {
            // 0-d scalar: ort accepts `()` shape more reliably than empty vec
            // for some EP paths — match historical ndarray::arr0 usage.
            if t.shape.is_empty() {
                let value = data.first().copied().unwrap_or(0);
                let tensor = ort::value::Tensor::from_array(((), vec![value]))
                    .map_err(|e| e.to_string())?;
                Ok(tensor.upcast())
            } else {
                let tensor = ort::value::Tensor::from_array((t.shape.clone(), data.clone()))
                    .map_err(|e| e.to_string())?;
                Ok(tensor.upcast())
            }
        }
    }
}

fn extract_outputs(
    outputs: &ort::session::SessionOutputs<'_>,
) -> Result<Vec<InferenceTensor>, InferenceError> {
    let mut result = Vec::with_capacity(outputs.len());
    for i in 0..outputs.len() {
        let value = &outputs[i];
        if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
            let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            result.push(InferenceTensor::f32(shape_vec, data.to_vec()));
            continue;
        }
        if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
            let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            result.push(InferenceTensor::i64(shape_vec, data.to_vec()));
            continue;
        }
        return Err(InferenceError::Run(format!(
            "output {i}: unsupported tensor element type (need f32 or i64)"
        )));
    }
    Ok(result)
}
