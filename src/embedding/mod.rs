#![allow(deprecated)] // EmbeddingExtractor / EmbeddingError remain soft-deprecated
//! Speaker embedding extraction trait (legacy).
//!
//! Prefer [`crate::Embedder`] for new code — offline [`crate::Pipeline`] and
//! online [`crate::streaming::StreamingPipeline`] accept `E: Embedder`.
//! Types that still implement [`EmbeddingExtractor`] automatically satisfy
//! [`Embedder`] via a blanket bridge in `polyvoice::embedder`.
//!
//! [`DummyExtractor`] implements [`Embedder`] directly (the supported test mock).

use crate::types::DiarizationConfig;

/// Error type for legacy embedding extraction failures.
///
/// Prefer [`crate::EmbedderError`] with [`crate::Embedder`].
#[derive(thiserror::Error, Debug)]
#[deprecated(
    since = "0.7.0",
    note = "use polyvoice::EmbedderError with the Embedder trait"
)]
pub enum EmbeddingError {
    #[error("model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("inference failed: {0}")]
    InferenceFailed(String),
    #[error("invalid input: expected {expected} samples, got {got}")]
    InvalidInput { expected: usize, got: usize },
}

/// Legacy trait for speaker embedding extractors.
///
/// Prefer implementing [`crate::Embedder`] directly. Existing
/// `EmbeddingExtractor` implementors continue to work with
/// [`crate::Pipeline`] / [`crate::streaming::StreamingPipeline`] through an
/// automatic bridge.
///
/// ```rust
/// use polyvoice::{DummyExtractor, Embedder, DiarizationConfig};
/// let extractor = DummyExtractor::new(256);
/// let samples = vec![0.0f32; DiarizationConfig::default().window_samples()];
/// let emb = extractor.embed(&samples).unwrap();
/// assert_eq!(emb.len(), 256);
/// ```
#[deprecated(
    since = "0.7.0",
    note = "implement polyvoice::Embedder instead; EmbeddingExtractor still works via an automatic bridge"
)]
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

/// Deterministic pseudo-random unit-vector embedder for tests and benchmarks.
///
/// Implements [`crate::Embedder`] directly — pass it to
/// [`crate::Pipeline`] / [`crate::streaming::StreamingPipeline`] without
/// going through the legacy [`EmbeddingExtractor`] bridge.
///
/// ```rust
/// use polyvoice::{DummyExtractor, Embedder};
/// let extractor = DummyExtractor::new(256);
/// assert_eq!(extractor.dim(), 256);
/// ```
pub struct DummyExtractor {
    dim: usize,
    seed: std::sync::atomic::AtomicU64,
}

impl DummyExtractor {
    /// { true }
    /// pub fn new(dim: usize) -> Self
    /// { true }
    /// Create a dummy extractor that returns deterministic pseudo-random embeddings.
    ///
    /// Useful for tests and benchmarks where a real ONNX model is not available.
    ///
    /// ```rust
    /// use polyvoice::{DummyExtractor, Embedder};
    /// let extractor = DummyExtractor::new(256);
    /// assert_eq!(extractor.dim(), 256);
    /// ```
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            seed: std::sync::atomic::AtomicU64::new(1),
        }
    }

    fn next_unit_vector(&self) -> Vec<f32> {
        let mut seed = self.seed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut vec = vec![0.0f32; self.dim];
        for v in &mut vec {
            // Simple LCG for deterministic "randomness".
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((seed % 1000) as f32 / 1000.0) - 0.5;
        }
        crate::utils::l2_normalize(&mut vec);
        vec
    }
}

impl crate::Embedder for DummyExtractor {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, crate::EmbedderError> {
        Ok(self.next_unit_vector())
    }
}
