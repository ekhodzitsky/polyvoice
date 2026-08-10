//! ONNX Runtime (`ort`) implementation of [`InferenceRuntime`].
//!
//! **This is the only module that may import `ort::`.** Neural stages must go
//! through [`InferenceRuntime`] / [`RuntimeSession`](super::RuntimeSession)
//! instead. The optional tract backend lives in `tract_session` behind
//! `backend-tract`.

use super::runtime::{InferenceError, InferenceRuntime, InferenceTensor, NamedTensor, TensorData};
use super::{ExecutionProvider, OnnxError, validate_onnx_header};
use std::path::Path;

/// Thin wrapper around `ort::session::Session` implementing [`InferenceRuntime`].
///
/// Construction goes through [`OrtSession::from_path`] / [`build_session_with_ep`](super::build_session_with_ep)
/// so the validate-before-build invariant and EP wiring stay centralized.
#[derive(Debug)]
pub struct OrtSession {
    session: ort::session::Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OrtSession {
    /// Load a model from `path` with the given execution provider and optional
    /// intra-op thread pin. Validates the ONNX header before ort parses the file.
    pub fn from_path(
        model_path: &Path,
        ep: ExecutionProvider,
        intra_threads: Option<usize>,
    ) -> Result<Self, OnnxError> {
        validate_onnx_header(model_path)?;
        // ort::Error is not Send+Sync, so it cannot ride a typed source — stringify.
        let build = |detail: String| OnnxError::SessionBuild {
            path: model_path.to_path_buf(),
            detail,
        };
        let mut builder =
            ort::session::Session::builder().map_err(|e| build(format!("session builder: {e}")))?;
        if let Some(n) = intra_threads {
            builder = builder
                .with_intra_threads(n)
                .map_err(|e| build(format!("intra threads: {e}")))?;
            // App-level session pools already fan out windows/embeds. On the
            // pure CPU EP, pin inter-op to 1 so pool workers do not
            // oversubscribe. Do NOT set this for CoreML/XNNPACK: those EPs
            // own their own parallelism and inter_threads=1 has been observed
            // to change CoreML outputs (DER + RTF) on Apple Silicon.
            if matches!(ep, ExecutionProvider::Cpu) {
                builder = builder
                    .with_inter_threads(1)
                    .map_err(|e| build(format!("inter threads: {e}")))?;
            }
        }
        match ep {
            ExecutionProvider::Cpu => {}
            ExecutionProvider::CoreMl => {
                #[cfg(all(feature = "coreml", target_os = "macos", target_arch = "aarch64"))]
                {
                    let coreml = ort::execution_providers::CoreMLExecutionProvider::default();
                    builder = builder
                        .with_execution_providers([coreml.build()])
                        .map_err(|e| build(format!("coreml ep: {e}")))?;
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
                        .map_err(|e| build(format!("xnnpack ep: {e}")))?;
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
            .map_err(|e| build(format!("commit_from_file: {e}")))?;
        let input_names = session
            .inputs()
            .iter()
            .map(|i| i.name().to_owned())
            .collect();
        let output_names = session
            .outputs()
            .iter()
            .map(|o| o.name().to_owned())
            .collect();
        Ok(Self {
            session,
            input_names,
            output_names,
        })
    }

    /// Custom `metadata_props` key/value pairs from the ONNX model.
    ///
    /// Empty when the model carries no custom metadata. Used by the
    /// self-describing model config loader (`models::metadata`).
    pub fn custom_metadata_props(
        &self,
    ) -> Result<std::collections::HashMap<String, String>, OnnxError> {
        let meta = self.session.metadata().map_err(|e| OnnxError::Metadata {
            detail: format!("session metadata: {e}"),
        })?;
        let keys = meta.custom_keys().map_err(|e| OnnxError::Metadata {
            detail: format!("custom metadata keys: {e}"),
        })?;
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

    fn output_names(&self) -> &[String] {
        &self.output_names
    }

    fn run(&mut self, inputs: &[NamedTensor<'_>]) -> Result<Vec<InferenceTensor>, InferenceError> {
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
                let tensor =
                    ort::value::Tensor::from_array(((), vec![value])).map_err(|e| e.to_string())?;
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn silero_path() -> Option<PathBuf> {
        let p = Path::new("models/silero_vad.onnx");
        if p.is_file() {
            Some(p.to_path_buf())
        } else {
            None
        }
    }

    /// Standard 16 kHz Silero step: 512-sample chunk + 64-sample context.
    fn silero_inputs() -> (InferenceTensor, InferenceTensor, InferenceTensor) {
        (
            InferenceTensor::f32(vec![1, 576], vec![0.01f32; 576]),
            InferenceTensor::f32(vec![2, 1, 128], vec![0.0f32; 2 * 128]),
            InferenceTensor::i64_scalar(16_000),
        )
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn from_path_reports_input_and_output_names() {
        let Some(path) = silero_path() else {
            return;
        };
        let session = OrtSession::from_path(&path, ExecutionProvider::Cpu, None).unwrap();
        assert!(format!("{session:?}").contains("OrtSession"));
        let inputs = session.input_names();
        assert!(inputs.iter().any(|n| n == "input"), "inputs: {inputs:?}");
        assert!(inputs.iter().any(|n| n == "state"), "inputs: {inputs:?}");
        assert!(inputs.iter().any(|n| n == "sr"), "inputs: {inputs:?}");
        assert!(!session.output_names().is_empty());
        assert_eq!(session.primary_input_name(), Some("input"));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn from_path_missing_file_is_validation_error() {
        let err = OrtSession::from_path(
            Path::new("models/definitely_not_a_model.onnx"),
            ExecutionProvider::Cpu,
            None,
        )
        .expect_err("missing file must fail validation");
        assert!(
            matches!(err, OnnxError::Validation(_)),
            "unexpected error: {err}"
        );
        assert!(err.to_string().contains("cannot read metadata"));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn from_path_valid_header_garbage_body_is_session_build_error() {
        // Passes header validation (ONNX magic) but is not a parseable model,
        // so the failure must surface as SessionBuild, not Validation.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let mut data = vec![0u8; 128];
        data[..4].copy_from_slice(b"ONNX");
        tmp.write_all(&data).unwrap();
        let err = OrtSession::from_path(tmp.path(), ExecutionProvider::Cpu, None)
            .expect_err("unparseable model must fail session build");
        match err {
            OnnxError::SessionBuild { path, detail } => {
                assert!(detail.contains("commit_from_file"), "detail: {detail}");
                assert_eq!(path, tmp.path());
            }
            other => panic!("expected SessionBuild, got: {other}"),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn from_path_intra_threads_pin() {
        let Some(path) = silero_path() else {
            return;
        };
        assert!(OrtSession::from_path(&path, ExecutionProvider::Cpu, Some(1)).is_ok());
        assert!(OrtSession::from_path(&path, ExecutionProvider::Cpu, Some(2)).is_ok());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn unwired_execution_providers_fall_back_to_cpu() {
        let Some(path) = silero_path() else {
            return;
        };
        // Nnapi / Cuda are not wired: they warn and run on CPU — never an error.
        assert!(OrtSession::from_path(&path, ExecutionProvider::Nnapi, None).is_ok());
        assert!(OrtSession::from_path(&path, ExecutionProvider::Cuda, None).is_ok());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn optional_execution_providers_build_or_fall_back() {
        let Some(path) = silero_path() else {
            return;
        };
        // CoreMl / XnnPack register when their cargo feature is compiled in on
        // a supported target, otherwise they warn and fall back to CPU. Either
        // way session construction must succeed.
        assert!(OrtSession::from_path(&path, ExecutionProvider::CoreMl, None).is_ok());
        assert!(OrtSession::from_path(&path, ExecutionProvider::XnnPack, None).is_ok());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn custom_metadata_props_real_model() {
        let Some(path) = silero_path() else {
            return;
        };
        let session = OrtSession::from_path(&path, ExecutionProvider::Cpu, Some(1)).unwrap();
        // Silero carries no custom props; the call itself must succeed.
        let props = session.custom_metadata_props().unwrap();
        assert!(props.keys().all(|k| !k.is_empty()));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn run_named_silero_chunk() {
        let Some(path) = silero_path() else {
            return;
        };
        let mut session = OrtSession::from_path(&path, ExecutionProvider::Cpu, Some(1)).unwrap();
        let (input, state, sr) = silero_inputs();
        let out = session
            .run(&[
                NamedTensor::new("input", &input),
                NamedTensor::new("state", &state),
                NamedTensor::new("sr", &sr),
            ])
            .unwrap();
        assert_eq!(out.len(), 2);
        // Speech probability: f32 [1, 1] in [0, 1].
        assert_eq!(out[0].shape, vec![1, 1]);
        let prob = out[0].as_f32_slice().unwrap()[0];
        assert!((0.0..=1.0).contains(&prob), "prob out of range: {prob}");
        // Next LSTM state: f32 [2, 1, 128].
        assert_eq!(out[1].shape, vec![2, 1, 128]);
        assert_eq!(out[1].as_f32_slice().unwrap().len(), 2 * 128);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn run_ordered_silero_chunk() {
        let Some(path) = silero_path() else {
            return;
        };
        let mut session = OrtSession::from_path(&path, ExecutionProvider::Cpu, Some(1)).unwrap();
        let (input, state, sr) = silero_inputs();
        let out = session.run_ordered(&[&input, &state, &sr]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].shape, vec![1, 1]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn run_rejects_unknown_input_name() {
        let Some(path) = silero_path() else {
            return;
        };
        let mut session = OrtSession::from_path(&path, ExecutionProvider::Cpu, Some(1)).unwrap();
        let (input, state, sr) = silero_inputs();
        let err = session
            .run(&[
                NamedTensor::new("nope", &input),
                NamedTensor::new("state", &state),
                NamedTensor::new("sr", &sr),
            ])
            .expect_err("unknown input name must fail the run");
        assert!(
            matches!(err, InferenceError::Run(_)),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn run_rejects_shape_data_mismatch() {
        let Some(path) = silero_path() else {
            return;
        };
        let mut session = OrtSession::from_path(&path, ExecutionProvider::Cpu, Some(1)).unwrap();
        let bad = InferenceTensor::f32(vec![3], vec![1.0, 2.0]);
        let (_, state, sr) = silero_inputs();
        let err = session
            .run(&[
                NamedTensor::new("input", &bad),
                NamedTensor::new("state", &state),
                NamedTensor::new("sr", &sr),
            ])
            .expect_err("shape/data mismatch must fail tensor conversion");
        assert!(
            matches!(err, InferenceError::Run(_)),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn to_ort_tensor_f32_and_i64_variants() {
        // f32 tensor.
        assert!(to_ort_tensor(&InferenceTensor::f32(vec![2], vec![1.0, 2.0])).is_ok());
        // i64 scalar (0-d): empty shape, one element.
        assert!(to_ort_tensor(&InferenceTensor::i64_scalar(16_000)).is_ok());
        // i64 scalar with empty data falls back to 0.
        assert!(to_ort_tensor(&InferenceTensor::i64(Vec::new(), Vec::new())).is_ok());
        // i64 with a real shape takes the shaped path.
        assert!(to_ort_tensor(&InferenceTensor::i64(vec![2], vec![1, 2])).is_ok());
        // Shape/data length mismatch is an error string, not a panic.
        assert!(to_ort_tensor(&InferenceTensor::f32(vec![3], vec![1.0])).is_err());
        assert!(to_ort_tensor(&InferenceTensor::i64(vec![3], vec![1])).is_err());
    }
}
