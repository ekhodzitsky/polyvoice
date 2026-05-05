//! ONNX-based speaker embedding extractor with a session pool.

use crate::embedding::{EmbeddingError, EmbeddingExtractor};
use crate::types::DiarizationConfig;
use crate::utils::l2_normalize;
use std::path::Path;

/// A pooled ONNX session for speaker embedding extraction.
///
/// Wraps `ort::session::Session` in a [`crossbeam_queue::ArrayQueue`]
/// so that multiple threads can extract embeddings concurrently without lock contention.
#[cfg(feature = "onnx")]
pub struct OnnxEmbeddingExtractor {
    pool: crossbeam_queue::ArrayQueue<ort::session::Session>,
    embedding_dim: usize,
    window_samples: usize,
}

#[cfg(feature = "onnx")]
impl OnnxEmbeddingExtractor {
    /// Load an ONNX model and create a pool of `pool_size` sessions.
    ///
    /// The model is expected to take a single input of shape `[1, window_samples]`
    /// (mono f32 audio) and produce an output of shape `[1, embedding_dim]`.
    pub fn new(
        model_path: &Path,
        embedding_dim: usize,
        window_samples: usize,
        pool_size: usize,
    ) -> anyhow::Result<Self> {
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
            window_samples,
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
impl EmbeddingExtractor for OnnxEmbeddingExtractor {
    fn extract(&self, samples: &[f32], _config: &DiarizationConfig) -> Result<Vec<f32>, EmbeddingError> {
        let mut guard = self.checkout().ok_or_else(|| {
            EmbeddingError::InferenceFailed("ONNX session pool exhausted".to_string())
        })?;

        if samples.len() != self.window_samples {
            return Err(EmbeddingError::InvalidInput {
                expected: self.window_samples,
                got: samples.len(),
            });
        }

        let input_tensor = ort::value::TensorRef::from_array_view(
            ([1_usize, self.window_samples], samples),
        )
        .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        let session = guard
            .session
            .as_mut()
            .ok_or_else(|| EmbeddingError::InferenceFailed("session not available".to_string()))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        let mut embedding = vec![0.0f32; self.embedding_dim];
        embedding.copy_from_slice(&data[..self.embedding_dim.min(data.len())]);
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

/// Stub when the `onnx` feature is disabled.
#[cfg(not(feature = "onnx"))]
pub struct OnnxEmbeddingExtractor;

#[cfg(not(feature = "onnx"))]
impl OnnxEmbeddingExtractor {
    pub fn new(
        _model_path: &Path,
        _embedding_dim: usize,
        _window_samples: usize,
        _pool_size: usize,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("the `onnx` feature is not enabled")
    }
}
