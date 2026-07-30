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
        let mut sessions = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            // intra_threads(1): this extractor parallelises across the session
            // pool, so each session stays single-threaded.
            let session =
                crate::onnx::build_session_with_ep(model_path, ep, Some(1)).map_err(|e| {
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
