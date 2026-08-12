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
/// 3. Bind free dims to concrete `[1,1,T]` facts for powerset-style waveform
///    inputs (rank-3). Fixed T unblocks Conv analyse that fails under symbols.
///    Tries product window lengths: 10 s, then 1 s @ 16 kHz.
fn load_runnable(model_path: &Path) -> anyhow::Result<Arc<TypedRunnableModel>> {
    let base = tract_onnx::onnx()
        .model_for_path(model_path)
        .map_err(|e| anyhow::anyhow!("tract model_for_path: {e}"))?;

    match try_optimize_runnable(base.clone()) {
        Ok(m) => Ok(m),
        Err(direct_err) => match try_optimize_with_symbols(base.clone()) {
            Ok(m) => Ok(m),
            Err(sym_err) => {
                let mut last = String::new();
                // 10 s window (product default), then 1 s smoke window.
                for t in [160_000usize, 16_000usize] {
                    match try_optimize_with_concrete_nct(base.clone(), t) {
                        Ok(m) => return Ok(m),
                        Err(e) => last = format!("concrete-N1C1T{t}: {e}"),
                    }
                }
                Err(anyhow::anyhow!(
                    "tract load failed (direct: {direct_err}; with-symbols: {sym_err}; {last})"
                ))
            }
        },
    }
}

/// Bind the single rank-3 input to concrete `[1, 1, t]` (powerset waveform).
fn try_optimize_with_concrete_nct(
    mut model: InferenceModel,
    t: usize,
) -> anyhow::Result<Arc<TypedRunnableModel>> {
    if model.inputs.len() != 1 {
        anyhow::bail!("concrete-N1C1T only for single-input models");
    }
    let fact = model
        .input_fact(0)
        .map_err(|e| anyhow::anyhow!("input_fact: {e}"))?
        .clone();
    let Some(dt) = fact.datum_type.concretize() else {
        anyhow::bail!("input datum type not concrete");
    };
    let rank = fact.shape.dims().count();
    if rank != 3 {
        anyhow::bail!("expected rank-3 input, got {rank}");
    }
    model
        .set_input_fact(0, InferenceFact::dt_shape(dt, tvec!(1, 1, t)))
        .map_err(|e| anyhow::anyhow!("set_input_fact concrete T={t}: {e}"))?;
    try_optimize_runnable(model)
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Path to a checked-in feed-forward model, or `None` (test skips) when the
    /// blob is not present in this checkout.
    fn cam_pp_path() -> Option<std::path::PathBuf> {
        let p = Path::new("models").join("cam_pp_fp32.onnx");
        if p.is_file() { Some(p) } else { None }
    }

    fn cam_pp_session() -> Option<TractSession> {
        let path = cam_pp_path()?;
        Some(TractSession::from_path(&path, Some(1)).expect("cam++ loads on tract"))
    }

    fn cam_pp_input() -> InferenceTensor {
        let time = 200usize;
        let n_mels = 80usize;
        InferenceTensor::f32(vec![1, time, n_mels], vec![0.05f32; time * n_mels])
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn from_path_loads_feed_forward_model() {
        let Some(session) = cam_pp_session() else {
            eprintln!("skip: models/cam_pp_fp32.onnx missing");
            return;
        };
        assert!(!session.input_names().is_empty());
        let dbg = format!("{session:?}");
        assert!(dbg.contains("TractSession"), "unexpected Debug: {dbg}");
        assert!(dbg.contains("input_names"), "unexpected Debug: {dbg}");
    }

    #[test]
    fn from_path_rejects_missing_file() {
        let err = TractSession::from_path(Path::new("models/definitely_not_here.onnx"), None)
            .expect_err("missing file must fail header validation");
        assert!(
            matches!(err, OnnxError::Validation(_)),
            "expected validation error, got: {err}"
        );
    }

    #[test]
    fn from_path_rejects_garbage_header() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&[0xAB; 64]).unwrap();
        let err = TractSession::from_path(tmp.path(), None)
            .expect_err("garbage must fail header validation");
        assert!(matches!(err, OnnxError::Validation(_)));
    }

    #[test]
    fn from_path_rejects_valid_header_invalid_proto() {
        // Passes the structural header check (0x08 protobuf tag + >= 64 bytes)
        // but is not a loadable ONNX graph: both optimize strategies must fail.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let mut bytes = vec![0x08u8, 0x01];
        bytes.extend(std::iter::repeat_n(0xFF, 126));
        tmp.write_all(&bytes).unwrap();
        let err = TractSession::from_path(tmp.path(), None)
            .expect_err("invalid proto must fail session build");
        match err {
            OnnxError::SessionBuild { detail, .. } => {
                assert!(
                    detail.contains("failed to decode Protobuf"),
                    "unexpected detail: {detail}"
                );
            }
            other => panic!("expected SessionBuild, got: {other}"),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn run_rejects_unknown_input_name() {
        let Some(mut session) = cam_pp_session() else {
            eprintln!("skip: models/cam_pp_fp32.onnx missing");
            return;
        };
        let input = cam_pp_input();
        let err = session
            .run(&[NamedTensor::new("no_such_input", &input)])
            .expect_err("unknown name must fail");
        let msg = err.to_string();
        assert!(msg.contains("unknown input name"), "unexpected: {msg}");
        assert!(msg.contains("no_such_input"), "unexpected: {msg}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn run_rejects_duplicate_input_name() {
        let Some(mut session) = cam_pp_session() else {
            eprintln!("skip: models/cam_pp_fp32.onnx missing");
            return;
        };
        let name = session.input_names()[0].clone();
        let input = cam_pp_input();
        let err = session
            .run(&[
                NamedTensor::new(&name, &input),
                NamedTensor::new(&name, &input),
            ])
            .expect_err("duplicate name must fail");
        assert!(
            err.to_string().contains("duplicate input name"),
            "unexpected: {err}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn run_rejects_missing_input() {
        let Some(mut session) = cam_pp_session() else {
            eprintln!("skip: models/cam_pp_fp32.onnx missing");
            return;
        };
        let err = session.run(&[]).expect_err("empty inputs must fail");
        assert!(
            err.to_string().contains("missing inputs for run"),
            "unexpected: {err}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn run_ordered_rejects_wrong_arity() {
        let Some(mut session) = cam_pp_session() else {
            eprintln!("skip: models/cam_pp_fp32.onnx missing");
            return;
        };
        let err = session.run_ordered(&[]).expect_err("wrong arity must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("expected 1 inputs, got 0"),
            "unexpected: {msg}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn run_ordered_rejects_shape_mismatch() {
        let Some(mut session) = cam_pp_session() else {
            eprintln!("skip: models/cam_pp_fp32.onnx missing");
            return;
        };
        // Shape product (6) does not match data length (5).
        let bad = InferenceTensor::f32(vec![1, 2, 3], vec![0.0f32; 5]);
        let err = session
            .run_ordered(&[&bad])
            .expect_err("shape/data mismatch must fail");
        assert!(
            err.to_string().contains("tract f32 tensor"),
            "unexpected: {err}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn named_run_matches_ordered_run() {
        let Some(mut session) = cam_pp_session() else {
            eprintln!("skip: models/cam_pp_fp32.onnx missing");
            return;
        };
        let name = session.input_names()[0].clone();
        let input = cam_pp_input();
        let named_out = session
            .run(&[NamedTensor::new(&name, &input)])
            .expect("named run");
        let ordered_out = session.run_ordered(&[&input]).expect("ordered run");
        assert_eq!(named_out.len(), ordered_out.len());
        for (a, b) in named_out.iter().zip(ordered_out.iter()) {
            assert_eq!(a, b, "named and ordered runs must agree");
        }
        // Real inference happened: f32 output with non-zero extent.
        let first = &named_out[0];
        assert!(first.shape.iter().product::<usize>() > 0);
        assert!(matches!(first.data, TensorData::F32(_)));
    }

    #[test]
    fn to_tract_tvalue_i64_ok() {
        let t = InferenceTensor::i64(vec![2], vec![7, -3]);
        let tv = to_tract_tvalue(&t).expect("i64 conversion");
        let tensor = tv.into_tensor();
        assert_eq!(tensor.datum_type(), DatumType::I64);
        assert_eq!(tensor.shape(), &[2]);
    }

    #[test]
    fn to_tract_tvalue_i64_shape_mismatch() {
        let t = InferenceTensor::i64(vec![3], vec![1, 2]);
        let err = to_tract_tvalue(&t).expect_err("shape/data mismatch must fail");
        assert!(err.contains("tract i64 tensor"), "unexpected: {err}");
    }

    #[test]
    fn from_tract_tvalue_f32_round_trip() {
        let tv = Tensor::from_shape(&[2], &[1.5f32, -2.5])
            .unwrap()
            .into_tvalue();
        let out = from_tract_tvalue(tv, 0).expect("f32 output");
        assert_eq!(out.shape, vec![2]);
        assert_eq!(out.as_f32_slice().unwrap(), &[1.5, -2.5]);
    }

    #[test]
    fn from_tract_tvalue_i64_round_trip() {
        let tv = Tensor::from_shape(&[2], &[42i64, -1])
            .unwrap()
            .into_tvalue();
        let out = from_tract_tvalue(tv, 1).expect("i64 output");
        assert_eq!(out.shape, vec![2]);
        assert_eq!(out.data, TensorData::I64(vec![42, -1]));
    }

    #[test]
    fn from_tract_tvalue_rejects_unsupported_datum_type() {
        let tv = Tensor::from_shape(&[1], &[3u8]).unwrap().into_tvalue();
        let err = from_tract_tvalue(tv, 2).expect_err("u8 output must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported tract datum type"),
            "unexpected: {msg}"
        );
        assert!(msg.contains("output 2"), "unexpected: {msg}");
    }
}
