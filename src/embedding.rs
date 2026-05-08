//! Speaker embedding extraction trait.

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
pub trait EmbeddingExtractor: Send + Sync {
    /// Extract an embedding from raw 16 kHz mono f32 samples.
    ///
    /// The caller is responsible for ensuring the buffer length matches the model
    /// expectations. Implementations may pad or truncate, but should prefer
    /// returning an error when the input is unusable.
    fn extract(&self, samples: &[f32]) -> Result<Vec<f32>, EmbeddingError>;

    /// Dimensionality of the produced embedding vectors.
    fn embedding_dim(&self) -> usize;
}
