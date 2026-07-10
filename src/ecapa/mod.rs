#![allow(deprecated)] // legacy embedding API; see polyvoice::embedder
//! ONNX speaker embedding extractor (WeSpeaker, ECAPA-TDNN, etc.).
//!
//! Loads an ONNX model and runs log-mel filterbank + CMVN preprocessing
//! before inference.
//!
//! Expected ONNX I/O:
//! - Input: `[batch, time, n_mels]` f32 (typically `n_mels = 80`)
//! - Output: `[batch, embedding_dim]` f32

use crate::embedding::{EmbeddingError, EmbeddingExtractor};
use crate::features::{FbankExtractor, apply_cmvn};
use crate::types::DiarizationConfig;
use crate::utils::l2_normalize;
use std::path::Path;

#[cfg(feature = "onnx")]
#[deprecated(
    since = "0.7.0",
    note = "use the v1.0 Embedder trait in polyvoice::embedder"
)]
pub struct FbankOnnxExtractor {
    pool: crossbeam_queue::ArrayQueue<ort::session::Session>,
    embedding_dim: usize,
    fbank: FbankExtractor,
}

#[cfg(feature = "onnx")]
impl FbankOnnxExtractor {
    /// { pool_size > 0 }
    /// `fn new(model_path: &Path, embedding_dim: usize, pool_size: usize, ep: ExecutionProvider) -> Result<Self, anyhow::Error>`
    /// { ret.pool.len() == pool_size }
    pub fn new(
        model_path: &Path,
        embedding_dim: usize,
        pool_size: usize,
        ep: crate::onnx::ExecutionProvider,
    ) -> anyhow::Result<Self> {
        if pool_size == 0 {
            anyhow::bail!("pool_size must be > 0");
        }
        let pool = crossbeam_queue::ArrayQueue::new(pool_size);
        for i in 0..pool_size {
            // intra_threads(1): this extractor parallelises across the session
            // pool, so each session stays single-threaded.
            let session = crate::onnx::build_session_with_ep(model_path, ep, Some(1))
                .map_err(|e| EmbeddingError::InferenceFailed(format!("session {i}: {e}")))?;
            pool.push(session)
                .map_err(|_| anyhow::anyhow!("failed to push session into pool"))?;
        }
        Ok(Self {
            pool,
            embedding_dim,
            fbank: FbankExtractor::new(crate::features::FbankConfig::default()),
        })
    }

    fn checkout(&self) -> Option<PooledSession<'_>> {
        self.pool.pop().map(|s| PooledSession {
            session: Some(s),
            pool: &self.pool,
        })
    }
}

#[cfg(feature = "onnx")]
impl EmbeddingExtractor for FbankOnnxExtractor {
    fn extract(
        &self,
        samples: &[f32],
        _config: &DiarizationConfig,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let mut guard = self.checkout().ok_or_else(|| {
            EmbeddingError::InferenceFailed("ONNX session pool exhausted".to_string())
        })?;

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
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        if fbank.is_empty() {
            return Err(EmbeddingError::InvalidInput {
                expected: self.fbank.config.win_length,
                got: samples.len(),
            });
        }

        let fbank = apply_cmvn(&fbank);

        let n_frames = fbank.len();
        let n_mels = fbank[0].len();
        let flat: Vec<f32> = fbank.into_iter().flatten().collect();

        let array = ndarray::Array3::from_shape_vec((1, n_frames, n_mels), flat)
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;
        let tensor = ort::value::TensorRef::from_array_view(&array)
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        let session = guard
            .session
            .as_mut()
            .ok_or_else(|| EmbeddingError::InferenceFailed("session not available".to_string()))?;
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        if outputs.iter().next().is_none() {
            return Err(EmbeddingError::InferenceFailed(
                "ONNX model produced no outputs".to_string(),
            ));
        }
        let (_, data) = &outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        let data_len = data.len();
        if data_len != self.embedding_dim {
            return Err(EmbeddingError::InferenceFailed(format!(
                "expected embedding dim {}, got {}",
                self.embedding_dim, data_len
            )));
        }
        let mut embedding = vec![0.0f32; self.embedding_dim];
        embedding.copy_from_slice(data);
        l2_normalize(&mut embedding);

        Ok(embedding)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(feature = "onnx")]
struct PooledSession<'a> {
    session: Option<ort::session::Session>,
    pool: &'a crossbeam_queue::ArrayQueue<ort::session::Session>,
}

#[cfg(feature = "onnx")]
impl Drop for PooledSession<'_> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            let _ = self.pool.push(session);
        }
    }
}

#[cfg(not(feature = "onnx"))]
#[derive(Debug)]
#[deprecated(
    since = "0.7.0",
    note = "use the v1.0 Embedder trait in polyvoice::embedder"
)]
pub struct FbankOnnxExtractor;

#[cfg(not(feature = "onnx"))]
impl FbankOnnxExtractor {
    /// { false }
    /// `fn new(_model_path: &Path, _embedding_dim: usize, _pool_size: usize) -> Result<Self, anyhow::Error>`
    /// { false }
    pub fn new(
        _model_path: &Path,
        _embedding_dim: usize,
        _pool_size: usize,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("the `onnx` feature is not enabled")
    }
}

#[cfg(all(test, not(feature = "onnx")))]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn fbank_onnx_extractor_new_without_onnx_fails() {
        let result = FbankOnnxExtractor::new(Path::new("dummy.onnx"), 256, 1);
        assert!(result.is_err());
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        };
        assert!(
            err.contains("onnx") || err.contains("not enabled"),
            "expected onnx-related error, got: {err}"
        );
    }

    #[test]
    fn fbank_onnx_extractor_stub_exists() {
        // Ensure the stub type is constructible (even if useless).
        let _ = FbankOnnxExtractor;
    }
}
