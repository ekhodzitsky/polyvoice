//! Speaker embedding extraction trait.
//!
//! Use this module to turn audio windows into fixed-dimension speaker vectors
//! for clustering. See [`EmbeddingExtractor`] for the trait and
//! [`DummyExtractor`] for a test-time stand-in.

use crate::types::DiarizationConfig;

/// Error type for embedding extraction failures.
#[derive(thiserror::Error, Debug)]
pub enum EmbeddingError {
    #[error("model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("inference failed: {0}")]
    InferenceFailed(String),
    #[error("invalid input: expected {expected} samples, got {got}")]
    InvalidInput { expected: usize, got: usize },
}

/// Trait for speaker embedding extractors.
///
/// Implementors are expected to be thread-safe (either internally synchronized
/// or cheaply clonable), so that they can be shared across concurrent diarizers.
///
/// ```rust
/// use polyvoice::{EmbeddingExtractor, DummyExtractor, DiarizationConfig};
/// let extractor = DummyExtractor::new(256);
/// let config = DiarizationConfig::default();
/// let samples = vec![0.0f32; config.window_samples()];
/// let emb = extractor.extract(&samples, &config).unwrap();
/// assert_eq!(emb.len(), 256);
/// ```
pub trait EmbeddingExtractor: Send + Sync {
    /// Extract an embedding from raw 16 kHz (or `config.sample_rate`) mono f32 samples.
    ///
    /// The caller is responsible for ensuring the buffer length matches the model
    /// expectations (usually `config.window_samples()`). Implementations may pad
    /// or truncate, but should prefer returning an error when the input is unusable.
    fn extract(
        &self,
        samples: &[f32],
        config: &DiarizationConfig,
    ) -> Result<Vec<f32>, EmbeddingError>;

    /// Dimensionality of the produced embedding vectors.
    fn embedding_dim(&self) -> usize;
}

/// A no-op extractor that returns random-ish unit vectors.
/// Useful for tests and benchmarks where the real model is not available.
pub struct DummyExtractor {
    dim: usize,
    seed: std::sync::atomic::AtomicU64,
}

impl DummyExtractor {
    /// Create a dummy extractor that returns deterministic pseudo-random embeddings.
    ///
    /// Useful for tests and benchmarks where a real ONNX model is not available.
    ///
    /// ```rust
    /// use polyvoice::{DummyExtractor, EmbeddingExtractor};
    /// let extractor = DummyExtractor::new(256);
    /// assert_eq!(extractor.embedding_dim(), 256);
    /// ```
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            seed: std::sync::atomic::AtomicU64::new(1),
        }
    }
}

impl EmbeddingExtractor for DummyExtractor {
    fn extract(
        &self,
        samples: &[f32],
        _config: &DiarizationConfig,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let _ = samples; // unused
        let mut seed = self.seed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut vec = vec![0.0f32; self.dim];
        for v in &mut vec {
            // Simple LCG for deterministic "randomness".
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((seed % 1000) as f32 / 1000.0) - 0.5;
        }
        crate::utils::l2_normalize(&mut vec);
        Ok(vec)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}
