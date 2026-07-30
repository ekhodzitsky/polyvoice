//! Pure-Rust [`InferenceRuntime`] backed by [tract](https://github.com/sonos/tract).
//!
//! **This is the only module that may import `tract_onnx` / `tract_*`.** Stages
//! must go through [`InferenceRuntime`] / [`RuntimeSession`](super::RuntimeSession).
//!
//! Gated behind the `backend-tract` cargo feature. tract-onnx declares MSRV 1.91
//! (higher than this crate's declared 1.88) — enable only on a newer toolchain.

use super::runtime::{InferenceError, InferenceRuntime, InferenceTensor, NamedTensor, TensorData};
use super::{OnnxError, validate_onnx_header};
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;
use tract_onnx::tract_hir::internal::DimLike;

/// tract-backed inference session implementing [`InferenceRuntime`].
///
/// Loads ONNX via `tract_onnx`, optimizes to a typed runnable plan, and runs
/// named or ordered tensors. Stateful models (Silero LSTM) pass state as
/// ordinary named I/O tensors — same contract as [`super::OrtSession`].
///
/// EP / thread-pool knobs from ort are ignored: tract is pure-Rust CPU only
/// in this spike (no Metal/CUDA wiring).
pub struct TractSession {
    /// Arc because `SimplePlan::run` requires `&Arc<Self>` in tract 0.23.
    model: Arc<TypedRunnableModel>,
    input_names: Vec<String>,
}

impl std::fmt::Debug for TractSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TractSession")
            .field("input_names", &self.input_names)
            .finish_non_exhaustive()
    }
}

impl TractSession {
    /// Load a model from `path`. Validates the ONNX header before tract parses.
    ///
    /// `intra_threads` is accepted for API parity with the ort builder and is
    /// currently ignored (tract's default executor is used).
    pub fn from_path(model_path: &Path, _intra_threads: Option<usize>) -> Result<Self, OnnxError> {
        validate_onnx_header(model_path)?;
        let model = load_runnable(model_path).map_err(|e| OnnxError::SessionBuild {
            path: model_path.to_path_buf(),
            detail: format!("{e}"),
        })?;

        let input_names = (0..model.model().inputs.len())
            .map(|i| {
                let outlet = model.model().inputs[i];
                model.model().node(outlet.node).name.to_owned()
            })
            .collect();

        Ok(Self { model, input_names })
    }
}

/// Load + optimize strategies:
/// 1. Direct `into_optimized` (works for fixed / self-describing feed-forward).
/// 2. Bind free ONNX dims to symbols, then optimize (helps dynamic B/T graphs).
fn load_runnable(model_path: &Path) -> anyhow::Result<Arc<TypedRunnableModel>> {
    let base = tract_onnx::onnx()
        .model_for_path(model_path)
        .map_err(|e| anyhow::anyhow!("tract model_for_path: {e}"))?;

    match try_optimize_runnable(base.clone()) {
        Ok(m) => Ok(m),
        Err(direct_err) => match try_optimize_with_symbols(base) {
            Ok(m) => Ok(m),
            Err(sym_err) => Err(anyhow::anyhow!(
                "tract load failed (direct: {direct_err}; with-symbols: {sym_err})"
            )),
        },
    }
}

fn try_optimize_runnable(model: InferenceModel) -> anyhow::Result<Arc<TypedRunnableModel>> {
    // into_runnable() returns Arc in tract 0.23.
    model
        .into_optimized()
        .map_err(|e| anyhow::anyhow!("into_optimized: {e}"))?
        .into_runnable()
        .map_err(|e| anyhow::anyhow!("into_runnable: {e}"))
}

fn try_optimize_with_symbols(mut model: InferenceModel) -> anyhow::Result<Arc<TypedRunnableModel>> {
    for i in 0..model.inputs.len() {
        let fact = model
            .input_fact(i)
            .map_err(|e| anyhow::anyhow!("input_fact {i}: {e}"))?
            .clone();
        let Some(dt) = fact.datum_type.concretize() else {
            continue;
        };
        let dims_vec: Vec<_> = fact.shape.dims().cloned().collect();
        let mut dims: TVec<TDim> = tvec!();
        for (ax, d) in dims_vec.iter().enumerate() {
            match d.concretize() {
                Some(td) if td.to_usize().is_ok() => {
                    // SAFETY: checked is_ok above.
                    dims.push(td.to_usize().map_err(|e| anyhow::anyhow!("{e}"))?.to_dim());
                }
                _ => {
                    let s = model.sym(&format!("I{i}A{ax}"));
                    dims.push(s.to_dim());
                }
            }
        }
        model
            .set_input_fact(i, InferenceFact::dt_shape(dt, dims))
            .map_err(|e| anyhow::anyhow!("set_input_fact {i}: {e}"))?;
    }
    try_optimize_runnable(model)
}

impl InferenceRuntime for TractSession {
    fn input_names(&self) -> &[String] {
        &self.input_names
    }

    fn run(&mut self, inputs: &[NamedTensor<'_>]) -> Result<Vec<InferenceTensor>, InferenceError> {
        let n = self.model.model().inputs.len();
        let mut ordered: Vec<Option<&InferenceTensor>> = vec![None; n];
        for nt in inputs {
            let idx = self
                .input_names
                .iter()
                .position(|name| name == nt.name)
                .ok_or_else(|| {
                    InferenceError::Run(format!(
                        "unknown input name {:?} (model inputs: {:?})",
                        nt.name, self.input_names
                    ))
                })?;
            if ordered[idx].is_some() {
                return Err(InferenceError::Run(format!(
                    "duplicate input name {:?}",
                    nt.name
                )));
            }
            ordered[idx] = Some(nt.tensor);
        }
        let missing: Vec<_> = ordered
            .iter()
            .enumerate()
            .filter_map(|(i, t)| t.is_none().then_some(self.input_names[i].as_str()))
            .collect();
        if !missing.is_empty() {
            return Err(InferenceError::Run(format!(
                "missing inputs for run: {missing:?}"
            )));
        }
        let mut refs: Vec<&InferenceTensor> = Vec::with_capacity(n);
        for t in ordered {
            match t {
                Some(tensor) => refs.push(tensor),
                None => {
                    return Err(InferenceError::Run(
                        "internal: missing input after validation".into(),
                    ));
                }
            }
        }
        self.run_ordered(&refs)
    }

    fn run_ordered(
        &mut self,
        inputs: &[&InferenceTensor],
    ) -> Result<Vec<InferenceTensor>, InferenceError> {
        let expected = self.model.model().inputs.len();
        if inputs.len() != expected {
            return Err(InferenceError::Run(format!(
                "expected {expected} inputs, got {}",
                inputs.len()
            )));
        }
        let mut tvec = TVec::new();
        for t in inputs {
            tvec.push(to_tract_tvalue(t).map_err(InferenceError::Run)?);
        }
        let outputs = self
            .model
            .run(tvec)
            .map_err(|e| InferenceError::Run(format!("tract run: {e}")))?;
        outputs
            .into_iter()
            .enumerate()
            .map(|(i, tv)| from_tract_tvalue(tv, i))
            .collect()
    }
}

fn to_tract_tvalue(t: &InferenceTensor) -> Result<TValue, String> {
    match &t.data {
        TensorData::F32(data) => {
            let tensor = Tensor::from_shape(&t.shape, data.as_slice())
                .map_err(|e| format!("tract f32 tensor: {e}"))?;
            Ok(tensor.into_tvalue())
        }
        TensorData::I64(data) => {
            let tensor = Tensor::from_shape(&t.shape, data.as_slice())
                .map_err(|e| format!("tract i64 tensor: {e}"))?;
            Ok(tensor.into_tvalue())
        }
    }
}

fn from_tract_tvalue(tv: TValue, index: usize) -> Result<InferenceTensor, InferenceError> {
    let tensor = tv.into_tensor();
    let shape: Vec<usize> = tensor.shape().to_vec();
    match tensor.datum_type() {
        DatumType::F32 => {
            let view = tensor
                .to_plain_array_view::<f32>()
                .map_err(|e| InferenceError::Run(format!("output {index} f32 view: {e}")))?;
            Ok(InferenceTensor::f32(shape, view.iter().copied().collect()))
        }
        DatumType::I64 => {
            let view = tensor
                .to_plain_array_view::<i64>()
                .map_err(|e| InferenceError::Run(format!("output {index} i64 view: {e}")))?;
            Ok(InferenceTensor::i64(shape, view.iter().copied().collect()))
        }
        other => Err(InferenceError::Run(format!(
            "output {index}: unsupported tract datum type {other:?} (need f32 or i64)"
        ))),
    }
}
