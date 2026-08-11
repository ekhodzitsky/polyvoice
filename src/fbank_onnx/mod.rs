//! Shared fbank + ONNX speaker embedding engine (WeSpeaker, CAM++, ERes2Net, …).
//!
//! Loads an ONNX model and runs log-mel filterbank + CMVN preprocessing
//! before inference. Implements the canonical [`crate::Embedder`] trait.
//! Model-specific wrappers live in [`crate::embedder`] (`ResNet34Adapter`,
//! `CamPlusPlusExtractor`, `ERes2NetV2Extractor`); prefer those when the
//! architecture is fixed. This type remains public for BYO model paths and
//! the CLI `--legacy` stack.
//!
//! Expected ONNX I/O:
//! - Input: `[batch, time, n_mels]` f32 (typically `n_mels = 80`)
//! - Output: `[batch, embedding_dim]` f32
//!
//! Inference goes through [`crate::onnx::InferenceRuntime`]; this module does
//! not import `ort::`.

use crate::embedder::{Embedder, EmbedderError};
use crate::features::{FbankExtractor, apply_cmvn};
use crate::onnx::{InferenceRuntime, InferenceTensor, RuntimeSession};
use crate::utils::l2_normalize;
use std::path::Path;

/// Pooled fbank → ONNX speaker embedder.
///
/// First-class [`Embedder`] implementation. Prefer architecture-specific
/// adapters in [`crate::embedder`] when targeting a known model family.
#[cfg(feature = "onnx")]
pub struct FbankOnnxExtractor {
    pool: crate::utils::ObjectPool<RuntimeSession>,
    embedding_dim: usize,
    fbank: FbankExtractor,
}

/// Errors from [`FbankOnnxExtractor`] construction.
///
/// Distinguishes a caller configuration error (`pool_size == 0`) from a
/// backend session-build failure so adapters can map each cause onto their
/// own error surface instead of flattening everything into one message.
#[cfg(feature = "onnx")]
#[derive(Clone, thiserror::Error, Debug)]
pub enum FbankExtractorError {
    /// `pool_size` was 0 — the session pool must hold at least one session.
    #[error("pool_size must be > 0")]
    EmptyPool,

    /// A pooled inference session failed to build (missing/invalid model file
    /// or backend error); `index` is the pool slot being constructed.
    #[error("session {index}: {source}")]
    SessionBuild {
        index: usize,
        #[source]
        source: crate::onnx::OnnxError,
    },
}

#[cfg(feature = "onnx")]
impl FbankOnnxExtractor {
    /// { pool_size > 0 }
    /// `fn new(model_path: &Path, embedding_dim: usize, pool_size: usize, ep: ExecutionProvider) -> Result<Self, FbankExtractorError>`
    /// { true }
    pub fn new(
        model_path: &Path,
        embedding_dim: usize,
        pool_size: usize,
        ep: crate::onnx::ExecutionProvider,
    ) -> Result<Self, FbankExtractorError> {
        if pool_size == 0 {
            return Err(FbankExtractorError::EmptyPool);
        }
        // Env can raise/lower fan-out without a rebuild (CPU tuning).
        let pool_size = crate::onnx::resolve_session_pool_size(pool_size);
        let mut sessions = Vec::with_capacity(pool_size);
        // Each pool session gets a fair share of the machine's cores: a
        // single-session extractor (the common CLI case) uses all of them,
        // while a loaded pool does not oversubscribe.
        // Overridable via POLYVOICE_INTRA_THREADS.
        let intra = crate::onnx::resolve_intra_threads(pool_size);
        for i in 0..pool_size {
            let session =
                crate::onnx::build_session_with_ep(model_path, ep, Some(intra)).map_err(|e| {
                    FbankExtractorError::SessionBuild {
                        index: i,
                        source: e,
                    }
                })?;
            sessions.push(session);
        }
        Ok(Self {
            pool: crate::utils::ObjectPool::new(sessions),
            embedding_dim,
            fbank: FbankExtractor::new(crate::features::FbankConfig::default()),
        })
    }

    /// Number of sessions in the pool — the maximum useful concurrency for
    /// batch embedding; spawning more threads than this just burns cores in
    /// the pool's spin-checkout. Called by the embedder adapters; silent when
    /// the crate is built without the `embedder` feature.
    #[cfg_attr(not(any(test, feature = "embedder")), allow(dead_code))]
    pub(crate) fn pool_size(&self) -> usize {
        self.pool.capacity()
    }
}

#[cfg(feature = "onnx")]
impl Embedder for FbankOnnxExtractor {
    fn dim(&self) -> usize {
        self.embedding_dim
    }

    fn embed(&self, samples: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        let mut session = self.pool.checkout();

        // Zero-pad short inputs to the minimum window length required by fbank.
        let min_samples = self.fbank.config.win_length;
        let padded: Vec<f32>;
        let samples = if samples.len() < min_samples {
            padded = {
                let mut v = vec![0.0_f32; min_samples];
                v[..samples.len()].copy_from_slice(samples);
                v
            };
            &padded
        } else {
            samples
        };

        let fbank = self
            .fbank
            .extract(samples)
            .map_err(|e| EmbedderError::InferenceFailed {
                detail: e.to_string(),
            })?;

        if fbank.is_empty() {
            let sample_rate = self.fbank.config.sample_rate as f32;
            return Err(EmbedderError::AudioTooShort {
                actual_secs: samples.len() as f32 / sample_rate,
                min_secs: min_samples as f32 / sample_rate,
            });
        }

        let fbank = apply_cmvn(&fbank);

        let n_frames = fbank.len();
        let n_mels = fbank[0].len();
        let flat: Vec<f32> = fbank.into_iter().flatten().collect();

        let input = InferenceTensor::f32(vec![1, n_frames, n_mels], flat);
        let outputs =
            session
                .run_ordered(&[&input])
                .map_err(|e| EmbedderError::InferenceFailed {
                    detail: e.to_string(),
                })?;

        let first = outputs
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::InferenceFailed {
                detail: "ONNX model produced no outputs".to_string(),
            })?;
        let data = first
            .into_f32()
            .map_err(|e| EmbedderError::InferenceFailed {
                detail: e.to_string(),
            })?;

        let data_len = data.len();
        if data_len != self.embedding_dim {
            return Err(EmbedderError::DimMismatch {
                expected: self.embedding_dim,
                actual: data_len,
            });
        }
        let mut embedding = data;
        l2_normalize(&mut embedding);

        Ok(embedding)
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::{ExecutionProvider, InferenceBackend};
    use std::path::PathBuf;

    /// Self-contained fbank embedder shipped in the repo (256-d output).
    const RESNET34: &str = "models/wespeaker_resnet34.onnx";
    const RESNET34_DIM: usize = 256;

    fn resnet34_path() -> Option<PathBuf> {
        let p = Path::new(RESNET34);
        if p.is_file() {
            Some(p.to_path_buf())
        } else {
            None
        }
    }

    /// `secs` seconds of a 300 Hz sine at sample rate `sr`, amplitude 0.3.
    fn sine_pcm(secs: f32, sr: u32) -> Vec<f32> {
        let n = (secs * sr as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
            })
            .collect()
    }

    /// Extract the construction error without requiring `Debug` on the
    /// extractor itself (it holds ONNX sessions, so it does not derive it).
    fn build_err(r: Result<FbankOnnxExtractor, FbankExtractorError>) -> FbankExtractorError {
        match r {
            Err(e) => e,
            Ok(_) => panic!("expected construction to fail"),
        }
    }

    #[test]
    fn new_rejects_zero_pool_size() {
        let err = build_err(FbankOnnxExtractor::new(
            Path::new("models/__missing__.onnx"),
            RESNET34_DIM,
            0,
            ExecutionProvider::Cpu,
        ));
        assert!(matches!(err, FbankExtractorError::EmptyPool));
        assert_eq!(err.to_string(), "pool_size must be > 0");
    }

    #[test]
    fn new_reports_session_build_error_for_missing_model() {
        let err = build_err(FbankOnnxExtractor::new(
            Path::new("models/__definitely_missing__.onnx"),
            RESNET34_DIM,
            2,
            ExecutionProvider::Cpu,
        ));
        match err {
            FbankExtractorError::SessionBuild { index, source } => {
                assert_eq!(index, 0);
                let msg = FbankExtractorError::SessionBuild { index, source }.to_string();
                assert!(msg.starts_with("session 0:"), "unexpected: {msg}");
            }
            other => panic!("expected SessionBuild, got {other:?}"),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn new_builds_pool_and_reports_size() {
        let Some(path) = resnet34_path() else {
            eprintln!("skip: {RESNET34} missing");
            return;
        };
        // Pin ort: the checked-in fp32 models are validated against ort.
        InferenceBackend::force(Some(InferenceBackend::Ort));
        let ext = FbankOnnxExtractor::new(&path, RESNET34_DIM, 2, ExecutionProvider::Cpu).unwrap();
        assert_eq!(ext.pool_size(), 2);
        assert_eq!(ext.dim(), RESNET34_DIM);
        InferenceBackend::force(None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn embed_returns_unit_norm_embedding() {
        let Some(path) = resnet34_path() else {
            eprintln!("skip: {RESNET34} missing");
            return;
        };
        InferenceBackend::force(Some(InferenceBackend::Ort));
        let ext = FbankOnnxExtractor::new(&path, RESNET34_DIM, 1, ExecutionProvider::Cpu).unwrap();
        let pcm = sine_pcm(1.0, 16_000);
        let emb = ext.embed(&pcm).unwrap();
        assert_eq!(emb.len(), RESNET34_DIM);
        assert!(emb.iter().all(|v| v.is_finite()));
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "expected unit norm, got {norm}");
        // Same input through the pool checkout again → same embedding.
        let emb2 = ext.embed(&pcm).unwrap();
        assert_eq!(emb, emb2);
        InferenceBackend::force(None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn embed_zero_pads_short_input() {
        let Some(path) = resnet34_path() else {
            eprintln!("skip: {RESNET34} missing");
            return;
        };
        InferenceBackend::force(Some(InferenceBackend::Ort));
        let ext = FbankOnnxExtractor::new(&path, RESNET34_DIM, 1, ExecutionProvider::Cpu).unwrap();
        // Shorter than one fbank window (400 samples) → zero-padded internally.
        let pcm = sine_pcm(0.005, 16_000);
        assert!(pcm.len() < 400);
        let emb = ext.embed(&pcm).unwrap();
        assert_eq!(emb.len(), RESNET34_DIM);
        assert!(emb.iter().all(|v| v.is_finite()));
        InferenceBackend::force(None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn embed_detects_dim_mismatch() {
        let Some(path) = resnet34_path() else {
            eprintln!("skip: {RESNET34} missing");
            return;
        };
        InferenceBackend::force(Some(InferenceBackend::Ort));
        // Declare the wrong dim: the model emits 256 values per utterance.
        let ext = FbankOnnxExtractor::new(&path, 192, 1, ExecutionProvider::Cpu).unwrap();
        let err = ext.embed(&sine_pcm(1.0, 16_000)).unwrap_err();
        match err {
            EmbedderError::DimMismatch { expected, actual } => {
                assert_eq!(expected, 192);
                assert_eq!(actual, RESNET34_DIM);
            }
            other => panic!("expected DimMismatch, got {other:?}"),
        }
        InferenceBackend::force(None);
    }
}
