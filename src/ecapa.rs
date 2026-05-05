//! ECAPA-TDNN speaker embedding extractor.
//!
//! Loads an ONNX-exported ECAPA-TDNN model (e.g. from SpeechBrain) and runs
//! log-mel filterbank preprocessing before inference.
//!
//! Expected ONNX I/O:
//! - Input: `[batch, time, n_mels]` f32 (typically `n_mels = 80`)
//! - Output: `[batch, embedding_dim]` f32

use crate::embedding::{EmbeddingError, EmbeddingExtractor};
use crate::features::FbankExtractor;
use crate::types::DiarizationConfig;
use crate::utils::l2_normalize;
use std::path::Path;

#[cfg(feature = "onnx")]
pub struct EcapaTdnnExtractor {
    pool: crossbeam_queue::ArrayQueue<ort::session::Session>,
    embedding_dim: usize,
    fbank: FbankExtractor,
}

#[cfg(feature = "onnx")]
impl EcapaTdnnExtractor {
    /// { pool_size > 0 }
    /// `fn new(model_path: &Path, embedding_dim: usize, pool_size: usize) -> Result<Self, anyhow::Error>`
    /// { ret.pool.len() == pool_size }
    pub fn new(model_path: &Path, embedding_dim: usize, pool_size: usize) -> anyhow::Result<Self> {
        let pool = crossbeam_queue::ArrayQueue::new(pool_size);
        for i in 0..pool_size {
            let session = ort::session::Session::builder()
                .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?
                .commit_from_file(model_path)
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
impl EmbeddingExtractor for EcapaTdnnExtractor {
    fn extract(&self, samples: &[f32], _config: &DiarizationConfig) -> Result<Vec<f32>, EmbeddingError> {
        let mut guard = self.checkout().ok_or_else(|| {
            EmbeddingError::InferenceFailed("ONNX session pool exhausted".to_string())
        })?;

        let fbank = self.fbank.extract(samples)
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        if fbank.is_empty() {
            return Err(EmbeddingError::InvalidInput {
                expected: self.fbank.config.win_length,
                got: samples.len(),
            });
        }

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
pub struct EcapaTdnnExtractor;

#[cfg(not(feature = "onnx"))]
impl EcapaTdnnExtractor {
    /// { false }
    /// `fn new(_model_path: &Path, _embedding_dim: usize, _pool_size: usize) -> Result<Self, anyhow::Error>`
    /// { false }
    pub fn new(_model_path: &Path, _embedding_dim: usize, _pool_size: usize) -> anyhow::Result<Self> {
        anyhow::bail!("the `onnx` feature is not enabled")
    }
}
