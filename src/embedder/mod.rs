//! v1.0 `Embedder` trait + concrete extractors (CAM++, ResNet34, ERes2NetV2) +
//! overlap-mask helper.
//!
//! `Embedder` is the supported bring-your-own embedder contract for offline
//! [`crate::pipeline::LegacyPipeline`] and online
//! [`crate::streaming::StreamingPipeline`]. The pure-Rust trait and overlap
//! mask are always available (no `onnx` required). ONNX-backed adapters still
//! need `features = ["infer", "embedder"]`. The generic `EmbedderPool` is a
//! test-only helper, not public API.
//!
//! Shared fbank+ONNX engine: `crate::fbank_onnx::FbankOnnxExtractor` (feature
//! `onnx`; implements [`Embedder`] directly). The architecture adapters share
//! one generic wrapper with per-model named constructors.

/// Speaker embedding extractor — turns a slice of 16 kHz mono audio into a
/// fixed-dimension embedding vector. Implementations are expected to L2-normalize
/// their output so cosine similarity is a meaningful metric downstream.
///
/// This is the **supported library injection API** for
/// [`crate::pipeline::LegacyPipeline`] and
/// [`crate::streaming::StreamingPipeline`]. Implement it on an external
/// encoder (Candle, tract, custom) without enabling `onnx`:
///
/// ```rust
/// use polyvoice::{Embedder, EmbedderError};
///
/// struct ConstantEmbedder { dim: usize }
///
/// impl Embedder for ConstantEmbedder {
///     fn dim(&self) -> usize { self.dim }
///     fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
///         let mut v = vec![0.0f32; self.dim];
///         if let Some(first) = v.first_mut() { *first = 1.0; }
///         Ok(v)
///     }
/// }
/// ```
pub trait Embedder: Send + Sync {
    /// Output dimension of this embedder. Constant per instance.
    fn dim(&self) -> usize;

    /// Compute an embedding for one audio segment.
    ///
    /// **Requires:** `audio` is 16 kHz mono PCM.
    /// **Guarantees on Ok:** `result.len() == self.dim()` and the vector is L2-normalized
    /// (`|sum(x²)¹ᐟ² − 1.0| < 1e-3`).
    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError>;

    /// Compute embeddings for a batch of audio segments. Default implementation
    /// is sequential; impls may override with a true batched ONNX call.
    fn embed_batch(&self, audios: &[&[f32]]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        audios.iter().map(|a| self.embed(a)).collect()
    }
}

/// Errors from `Embedder` implementations.
///
/// Marked `#[non_exhaustive]` so new variants (e.g. back-pressure) can land in
/// minor releases without forcing every consumer match to update.
#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum EmbedderError {
    #[error("audio too short for this embedder: {actual_secs:.3}s < {min_secs:.3}s")]
    AudioTooShort { actual_secs: f32, min_secs: f32 },

    #[error("ONNX inference failed: {detail}")]
    InferenceFailed { detail: String },

    /// Encoder concurrency / session pool exhausted (or equivalent back-pressure).
    ///
    /// Prefer this variant over stuffing the marker into
    /// [`EmbedderError::InferenceFailed`] so serving layers can classify metrics
    /// with `downcast` / [`EmbedderError::is_resource_exhausted`] instead of
    /// substring-matching English messages.
    #[error("resource exhausted: {detail}")]
    ResourceExhausted { detail: String },

    #[error("expected embedding dim {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    #[error("model file io error on {path}: {detail}")]
    ModelIo {
        path: std::path::PathBuf,
        detail: String,
    },

    /// An ONNX-backed extractor failed to construct: invalid pool size, an
    /// unloadable model file, or a backend session-build failure. The typed
    /// cause is preserved as the [`std::error::Error::source`].
    #[cfg(feature = "infer")]
    #[error("failed to build embedder for {path}: {source}")]
    SessionBuild {
        path: std::path::PathBuf,
        #[source]
        source: crate::fbank_onnx::FbankExtractorError,
    },

    #[error("legacy adapter error: {0}")]
    Legacy(String),
}

impl EmbedderError {
    /// True when this error reports encoder resource exhaustion.
    ///
    /// Matches the typed [`EmbedderError::ResourceExhausted`] variant and, for
    /// transitional consumers, legacy strings that still embed
    /// `"pool exhausted"` in [`EmbedderError::InferenceFailed`] or
    /// [`EmbedderError::Legacy`].
    pub fn is_resource_exhausted(&self) -> bool {
        match self {
            Self::ResourceExhausted { .. } => true,
            Self::InferenceFailed { detail } | Self::Legacy(detail) => {
                detail_looks_exhausted(detail)
            }
            _ => false,
        }
    }
}

/// Substring still used by historical extractors / metrics classifiers.
fn detail_looks_exhausted(detail: &str) -> bool {
    detail.contains("pool exhausted")
}

/// Deterministic pseudo-random unit-vector embedder for tests and benchmarks.
///
/// Implements [`Embedder`] directly — pass it to
/// [`crate::pipeline::LegacyPipeline`] / [`crate::streaming::StreamingPipeline`].
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

impl Embedder for DummyExtractor {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        Ok(self.next_unit_vector())
    }
}

/// { true }
/// `pub fn apply_overlap_mask( audio: &[f32], overlap_regions: &[(f32, f32)], sample_rate: u32, ) -> Vec<f32>`
/// { ret.len() == audio.len() }
/// Zero-fill audio samples in regions where the segmenter flagged a 2-speaker
/// overlap. The returned `Vec<f32>` is a copy of `audio` with zeros in the
/// `(start_secs, end_secs)` ranges listed in `overlap_regions`.
///
/// Out-of-bounds and inverted (end < start) regions are silently clamped or
/// skipped — never panics.
///
/// **Pure Rust, no allocations beyond the output Vec, wasm32-clean.**
pub fn apply_overlap_mask(
    audio: &[f32],
    overlap_regions: &[(f32, f32)],
    sample_rate: u32,
) -> Vec<f32> {
    let mut out = audio.to_vec();
    if out.is_empty() {
        return out;
    }
    let sr = sample_rate as f32;
    for &(start_s, end_s) in overlap_regions {
        if !end_s.is_finite() || !start_s.is_finite() || end_s <= start_s {
            continue;
        }
        let start = (start_s * sr).max(0.0).floor() as usize;
        let end = (end_s * sr).max(0.0).ceil() as usize;
        let end = end.min(out.len());
        if start >= end || start >= out.len() {
            continue;
        }
        for v in &mut out[start..end] {
            *v = 0.0;
        }
    }
    out
}

/// Pool of `Embedder` instances for concurrent extraction.
///
/// Test-only helper: production pipelines hold a `Box<dyn Embedder>` (and
/// `FbankOnnxExtractor` pools ONNX sessions internally), so this type is
/// compiled only for unit tests and is not part of the public API.
///
/// Generic over `E: Embedder` so the same pool implementation works for
/// `CamPlusPlusExtractor`, `ResNet34Adapter`, or any user-provided embedder.
/// All embedders in a pool must share the same output dimension.
///
/// Backed by a blocking object pool (`Mutex<Vec<E>>`): checkout waits until an
/// embedder is free; Drop returns it.
#[cfg(test)]
pub(crate) struct EmbedderPool<E: Embedder> {
    pool: crate::utils::ObjectPool<E>,
    dim: usize,
    capacity: usize,
}

#[cfg(test)]
impl<E: Embedder> EmbedderPool<E> {
    /// { true }
    /// `pub fn new(embedders: Vec<E>) -> Result<Self, EmbedderError>`
    /// { ret.is_ok() => ret.as_ref().unwrap().dim() == embedders.first().map_or(0, |e| e.dim()) }
    /// Build a pool from a list of embedders. All must share the same `dim()`.
    /// An empty list is allowed and constructs an empty pool: [`Self::is_empty`]
    /// is true and every [`Self::embed`] call fails with
    /// `EmbedderError::ResourceExhausted`.
    pub fn new(embedders: Vec<E>) -> Result<Self, EmbedderError> {
        let dim = embedders.first().map(|e| e.dim()).unwrap_or(0);
        for e in embedders.iter().skip(1) {
            let actual = e.dim();
            if actual != dim {
                return Err(EmbedderError::DimMismatch {
                    expected: dim,
                    actual,
                });
            }
        }
        let capacity = embedders.len();
        Ok(Self {
            pool: crate::utils::ObjectPool::new(embedders),
            dim,
            capacity,
        })
    }

    /// { true }
    /// pub fn dim(&self) -> usize
    /// { ret == self.dim }
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// { true }
    /// pub fn is_empty(&self) -> bool
    /// { ret == (self.capacity == 0) }
    /// True when the pool holds no embedders; every `embed` call then fails
    /// instead of blocking forever on an empty pool.
    pub fn is_empty(&self) -> bool {
        self.capacity == 0
    }

    /// { true }
    /// `pub fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError>`
    /// { ret.as_ref().map_or(true, |v| v.len() == self.dim) }
    /// Extract a single embedding using the next-available pooled embedder.
    /// Blocks until one is free.
    pub fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        if self.is_empty() {
            return Err(EmbedderError::ResourceExhausted {
                detail: "empty embedder pool".to_owned(),
            });
        }
        let embedder = self.pool.checkout();
        embedder.embed(audio)
    }
}

/// Parallel batch embedding using `std::thread::scope`.
/// Spawns up to `max_threads` threads (capped by `available_parallelism`),
/// each processing a chunk of the input via `embedder.embed()`. Callers pass
/// their session-pool size as `max_threads`: extra threads would just spin in
/// the pool's blocking checkout and compete with the workers for cores.
///
/// Only referenced by the shared ONNX adapter backing the per-model wrappers
/// (`ResNet34`, CAM++, ERes2NetV2).
#[cfg(all(feature = "infer", feature = "embedder"))]
fn parallel_embed_batch<E: Embedder>(
    embedder: &E,
    audios: &[&[f32]],
    max_threads: usize,
) -> Result<Vec<Vec<f32>>, EmbedderError> {
    let n = audios.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(max_threads.max(1))
        .min(n);

    let chunk_size = n.div_ceil(num_threads);
    let chunks: Vec<&[&[f32]]> = audios.chunks(chunk_size).collect();

    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                s.spawn(move || {
                    chunk
                        .iter()
                        .map(|audio| embedder.embed(audio))
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        let mut all_results = Vec::with_capacity(n);
        for h in handles {
            let chunk_results = h
                .join()
                .map_err(|_| EmbedderError::Legacy("embed_batch thread panicked".to_string()))?;
            all_results.extend(chunk_results);
        }
        all_results.into_iter().collect::<Result<Vec<_>, _>>()
    })
}

#[cfg(all(feature = "infer", feature = "embedder"))]
mod onnx_adapters {
    use super::*;
    use crate::fbank_onnx::FbankOnnxExtractor;
    use std::path::Path;

    /// Generic fbank+ONNX embedder adapter: owns the shared engine and the
    /// output dim, maps construction failures to [`EmbedderError::SessionBuild`]
    /// with the model path attached, and forwards the [`Embedder`] contract
    /// (batches fan out across threads via `parallel_embed_batch`). The public
    /// per-model adapters below are thin named wrappers over this one
    /// implementation.
    struct FbankAdapter {
        inner: FbankOnnxExtractor,
        dim: usize,
    }

    impl FbankAdapter {
        /// Load an fbank+ONNX model with the given output dim, session pool
        /// size, and execution provider.
        fn new(
            path: impl AsRef<Path>,
            dim: usize,
            pool_size: usize,
            ep: crate::onnx::ExecutionProvider,
        ) -> Result<Self, EmbedderError> {
            let inner =
                FbankOnnxExtractor::new(path.as_ref(), dim, pool_size, ep).map_err(|e| {
                    EmbedderError::SessionBuild {
                        path: path.as_ref().to_path_buf(),
                        source: e,
                    }
                })?;
            Ok(Self { inner, dim })
        }
    }

    impl Embedder for FbankAdapter {
        fn dim(&self) -> usize {
            self.dim
        }

        fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            self.inner.embed(audio)
        }

        fn embed_batch(&self, audios: &[&[f32]]) -> Result<Vec<Vec<f32>>, EmbedderError> {
            parallel_embed_batch(self, audios, self.inner.pool_size())
        }
    }

    /// Declare a public per-model adapter as a named wrapper over
    /// [`FbankAdapter`]: the tuple struct plus the delegating [`Embedder`]
    /// impl. Constructor conventions differ per model and are written
    /// explicitly next to each invocation.
    macro_rules! named_fbank_adapter {
        ($(#[$meta:meta])* $name:ident) => {
            $(#[$meta])*
            pub struct $name(FbankAdapter);

            impl Embedder for $name {
                fn dim(&self) -> usize {
                    self.0.dim()
                }

                fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
                    self.0.embed(audio)
                }

                fn embed_batch(
                    &self,
                    audios: &[&[f32]],
                ) -> Result<Vec<Vec<f32>>, EmbedderError> {
                    self.0.embed_batch(audios)
                }
            }
        };
    }

    named_fbank_adapter! {
        /// WeSpeaker ResNet34 embedder (256-d) via the shared fbank+ONNX engine.
        ResNet34Adapter
    }

    impl ResNet34Adapter {
        /// { true }
        /// `pub fn new(path: impl AsRef<Path>, pool_size: usize, ep: ExecutionProvider) -> Result<Self, EmbedderError>`
        /// { ret.as_ref().map_or(true, |e| e.dim() == 256) }
        /// Load the WeSpeaker ResNet34 ONNX model with the given execution provider.
        pub fn new(
            path: impl AsRef<Path>,
            pool_size: usize,
            ep: crate::onnx::ExecutionProvider,
        ) -> Result<Self, EmbedderError> {
            FbankAdapter::new(path, 256, pool_size, ep).map(Self)
        }
    }

    named_fbank_adapter! {
        /// CAM++ embedder (Channel-Attentive Multi-scale Pooling). Dim is supplied
        /// explicitly because WeSpeaker ships several CAM++ variants:
        /// `voxceleb_CAM++.onnx` is 512-d; smaller variants exist at 192-d.
        /// Uses the same 80-bin log-mel fbank pipeline as ResNet34.
        CamPlusPlusExtractor
    }

    impl CamPlusPlusExtractor {
        /// { true }
        /// `pub fn new( path: impl AsRef<Path>, dim: usize, pool_size: usize, ep: ExecutionProvider, ) -> Result<Self, EmbedderError>`
        /// { ret.as_ref().map_or(true, |e| e.dim() == dim) }
        /// Load a CAM++ ONNX model. `dim` must match the model's output
        /// dimension (e.g. 192 or 512 depending on the variant). Pool size
        /// controls the number of concurrent ONNX sessions held internally
        /// (canonical: `num_cpus().min(4)`).
        pub fn new(
            path: impl AsRef<Path>,
            dim: usize,
            pool_size: usize,
            ep: crate::onnx::ExecutionProvider,
        ) -> Result<Self, EmbedderError> {
            FbankAdapter::new(path, dim, pool_size, ep).map(Self)
        }
    }

    named_fbank_adapter! {
        /// ERes2NetV2 speaker embedder (Interspeech 2024): 192-d output, same
        /// 80-bin log-mel fbank path as CAM++. Tuned for short (1–3 s) utterances.
        /// Weights are optional downloads (Apache-2.0); never bundled or default.
        ERes2NetV2Extractor
    }

    impl ERes2NetV2Extractor {
        /// Output embedding dimension for the common zh-cn 16 kHz ONNX export.
        pub const DIM: usize = 192;

        /// Load an ERes2NetV2 ONNX model. Default dim is [`Self::DIM`] (192).
        pub fn new(
            path: impl AsRef<Path>,
            pool_size: usize,
            ep: crate::onnx::ExecutionProvider,
        ) -> Result<Self, EmbedderError> {
            Self::with_dim(path, Self::DIM, pool_size, ep)
        }

        /// Load with an explicit output dimension (for non-standard exports).
        pub fn with_dim(
            path: impl AsRef<Path>,
            dim: usize,
            pool_size: usize,
            ep: crate::onnx::ExecutionProvider,
        ) -> Result<Self, EmbedderError> {
            FbankAdapter::new(path, dim, pool_size, ep).map(Self)
        }
    }
}

#[cfg(all(feature = "infer", feature = "embedder"))]
pub use onnx_adapters::{CamPlusPlusExtractor, ERes2NetV2Extractor, ResNet34Adapter};

#[cfg(feature = "embedder-native")]
mod native;
#[cfg(feature = "embedder-native")]
pub use native::ResNet34Native;
#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "overlap_mask_tests.rs"]
mod overlap_mask_tests;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "trait_tests.rs"]
mod trait_tests;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "pool_tests.rs"]
mod pool_tests;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "error_display_tests.rs"]
mod error_display_tests;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "dummy_extractor_tests.rs"]
mod dummy_extractor_tests;

#[allow(clippy::unwrap_used)]
#[cfg(all(test, feature = "infer", feature = "embedder"))]
#[path = "onnx_adapter_tests.rs"]
mod onnx_adapter_tests;
