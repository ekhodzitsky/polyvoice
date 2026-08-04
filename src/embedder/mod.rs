//! v1.0 `Embedder` trait + concrete extractors (CAM++, ResNet34, ERes2NetV2) +
//! overlap-mask helper.
//!
//! `Embedder` is the supported bring-your-own embedder contract for offline
//! [`crate::pipeline::LegacyPipeline`] and online
//! [`crate::streaming::StreamingPipeline`]. The pure-Rust trait and overlap
//! mask are always available (no `onnx` required). ONNX-backed adapters still
//! need `features = ["onnx", "embedder"]`. The generic `EmbedderPool` is a
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
    #[cfg(feature = "onnx")]
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
#[cfg(all(feature = "onnx", feature = "embedder"))]
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

#[cfg(all(feature = "onnx", feature = "embedder"))]
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

#[cfg(all(feature = "onnx", feature = "embedder"))]
pub use onnx_adapters::{CamPlusPlusExtractor, ERes2NetV2Extractor, ResNet34Adapter};

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod overlap_mask_tests {
    use super::*;

    #[test]
    fn no_overlap_regions_pass_through() {
        let audio = vec![1.0_f32; 16_000];
        let masked = apply_overlap_mask(&audio, &[], 16_000);
        assert_eq!(masked, audio);
    }

    #[test]
    fn single_overlap_region_is_zeroed() {
        let audio = vec![1.0_f32; 16_000];
        let masked = apply_overlap_mask(&audio, &[(0.5, 0.7)], 16_000);
        for (i, &v) in masked.iter().enumerate() {
            if (8000..11200).contains(&i) {
                assert_eq!(v, 0.0, "sample {i} should be zeroed");
            } else {
                assert_eq!(v, 1.0, "sample {i} should pass through");
            }
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let masked = apply_overlap_mask(&[], &[(0.0, 1.0)], 16_000);
        assert!(masked.is_empty());
    }

    #[test]
    fn out_of_bounds_overlap_is_clamped() {
        let audio = vec![1.0_f32; 100];
        let masked = apply_overlap_mask(&audio, &[(0.5, 1.0)], 16_000);
        assert_eq!(masked, audio, "out-of-bounds overlap is a no-op");
    }

    #[test]
    fn negative_overlap_start_is_clamped_to_zero() {
        let audio = vec![1.0_f32; 16_000];
        let masked = apply_overlap_mask(&audio, &[(-1.0, 0.5)], 16_000);
        for &v in masked.iter().take(8000) {
            assert_eq!(v, 0.0);
        }
        for &v in masked.iter().skip(8000) {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn multiple_overlap_regions_all_zeroed() {
        let audio = vec![1.0_f32; 16_000];
        let masked = apply_overlap_mask(&audio, &[(0.1, 0.2), (0.5, 0.6), (0.9, 1.0)], 16_000);
        let zero_ranges = [(1600..3200), (8000..9600), (14_400..16_000)];
        for (i, &v) in masked.iter().enumerate() {
            let in_zero = zero_ranges.iter().any(|r| r.contains(&i));
            if in_zero {
                assert_eq!(v, 0.0, "sample {i} should be zeroed");
            } else {
                assert_eq!(v, 1.0, "sample {i} should pass through");
            }
        }
    }

    #[test]
    fn invalid_overlap_with_end_before_start_is_no_op() {
        let audio = vec![1.0_f32; 16_000];
        let masked = apply_overlap_mask(&audio, &[(0.7, 0.5)], 16_000);
        assert_eq!(masked, audio, "end<start is silently skipped");
    }

    #[test]
    fn non_finite_overlap_bounds_are_skipped() {
        let audio = vec![1.0_f32; 16_000];
        let masked = apply_overlap_mask(
            &audio,
            &[(f32::NAN, 0.5), (0.1, f32::INFINITY), (0.2, f32::NAN)],
            16_000,
        );
        assert_eq!(masked, audio, "NaN/infinite bounds are silently skipped");
    }

    #[test]
    fn zero_length_overlap_is_no_op() {
        let audio = vec![1.0_f32; 16_000];
        let masked = apply_overlap_mask(&audio, &[(0.5, 0.5)], 16_000);
        assert_eq!(masked, audio, "end==start is silently skipped");
    }

    #[test]
    fn overlap_extending_past_end_is_clamped_to_len() {
        let audio = vec![1.0_f32; 100];
        let masked = apply_overlap_mask(&audio, &[(0.0, 10.0)], 16_000);
        assert!(
            masked.iter().all(|&v| v == 0.0),
            "region past the end zeroes through the final sample"
        );
        assert_eq!(masked.len(), audio.len());
    }

    #[test]
    fn overlap_starting_past_end_is_no_op() {
        let audio = vec![1.0_f32; 100];
        let masked = apply_overlap_mask(&audio, &[(5.0, 6.0)], 16_000);
        assert_eq!(masked, audio, "region fully past the end is a no-op");
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod trait_tests {
    use super::*;

    /// In-memory dummy used by trait tests.
    struct ConstantEmbedder {
        values: Vec<f32>,
    }

    impl Embedder for ConstantEmbedder {
        fn dim(&self) -> usize {
            self.values.len()
        }
        fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            Ok(self.values.clone())
        }
    }

    #[test]
    fn embedder_trait_object_is_dyn_compatible() {
        let e = ConstantEmbedder {
            values: vec![0.1, 0.2, 0.3],
        };
        let _b: Box<dyn Embedder> = Box::new(e);
    }

    #[test]
    fn embedder_default_batch_is_serial() {
        let e = ConstantEmbedder {
            values: vec![0.5; 4],
        };
        let inputs: Vec<&[f32]> = vec![&[][..], &[][..], &[][..]];
        let out = e.embed_batch(&inputs).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|v| v.len() == 4 && v[0] == 0.5));
    }

    #[test]
    fn embedder_dim_matches_output() {
        let e = ConstantEmbedder {
            values: vec![1.0; 192],
        };
        assert_eq!(e.dim(), 192);
        assert_eq!(e.embed(&[]).unwrap().len(), 192);
    }

    #[test]
    fn embedder_error_audio_too_short_displays() {
        let err = EmbedderError::AudioTooShort {
            actual_secs: 0.05,
            min_secs: 0.25,
        };
        let msg = format!("{err}");
        assert!(msg.contains("0.05"));
        assert!(msg.contains("0.25"));
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod pool_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts how many times `embed` was called.
    struct CountingEmbedder {
        counter: Arc<AtomicUsize>,
        dim: usize,
    }

    impl Embedder for CountingEmbedder {
        fn dim(&self) -> usize {
            self.dim
        }
        fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0.0; self.dim])
        }
    }

    fn make_pool(n: usize) -> (EmbedderPool<CountingEmbedder>, Arc<AtomicUsize>) {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut embedders = Vec::with_capacity(n);
        for _ in 0..n {
            embedders.push(CountingEmbedder {
                counter: counter.clone(),
                dim: 192,
            });
        }
        let pool = EmbedderPool::new(embedders).unwrap();
        (pool, counter)
    }

    #[test]
    fn pool_with_single_embedder_round_trip() {
        let (pool, counter) = make_pool(1);
        let result = pool.embed(&[0.0_f32; 100]).unwrap();
        assert_eq!(result.len(), 192);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn pool_dim_is_consistent() {
        let (pool, _) = make_pool(4);
        assert_eq!(pool.dim(), 192);
    }

    #[test]
    fn pool_serial_embed_increments_counter_per_call() {
        let (pool, counter) = make_pool(2);
        for _ in 0..5 {
            pool.embed(&[0.0_f32; 100]).unwrap();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn pool_with_zero_embedders_errors() {
        let pool: EmbedderPool<CountingEmbedder> = EmbedderPool::new(Vec::new()).unwrap();
        let err = pool
            .embed(&[0.0_f32; 100])
            .expect_err("empty pool must fail");
        assert!(
            matches!(err, EmbedderError::ResourceExhausted { .. }),
            "empty pool is resource exhaustion, got {err}"
        );
        assert!(err.is_resource_exhausted());
    }

    #[test]
    fn pool_rejects_mismatched_embedder_dims() {
        let counter = Arc::new(AtomicUsize::new(0));
        let embedders = vec![
            CountingEmbedder {
                counter: counter.clone(),
                dim: 192,
            },
            CountingEmbedder {
                counter: counter.clone(),
                dim: 256,
            },
        ];
        let err = match EmbedderPool::new(embedders) {
            Err(e) => e,
            Ok(_) => panic!("mismatched dims must fail"),
        };
        assert!(
            matches!(
                err,
                EmbedderError::DimMismatch {
                    expected: 192,
                    actual: 256
                }
            ),
            "expected DimMismatch(192, 256), got {err}"
        );
    }

    #[test]
    fn resource_exhausted_classifier() {
        let typed = EmbedderError::ResourceExhausted {
            detail: "speaker sessions busy".into(),
        };
        assert!(typed.is_resource_exhausted());

        let legacy_string = EmbedderError::InferenceFailed {
            detail: "onnx session pool exhausted".into(),
        };
        assert!(legacy_string.is_resource_exhausted());

        let other = EmbedderError::DimMismatch {
            expected: 1,
            actual: 2,
        };
        assert!(!other.is_resource_exhausted());
    }

    #[test]
    fn pool_propagates_inner_embedder_error() {
        struct FailingEmbedder;

        impl Embedder for FailingEmbedder {
            fn dim(&self) -> usize {
                8
            }
            fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
                Err(EmbedderError::InferenceFailed {
                    detail: "boom".into(),
                })
            }
        }

        let pool = EmbedderPool::new(vec![FailingEmbedder]).unwrap();
        let err = pool
            .embed(&[0.0_f32; 16])
            .expect_err("inner error surfaces");
        assert!(
            matches!(err, EmbedderError::InferenceFailed { ref detail } if detail == "boom"),
            "expected the inner InferenceFailed error, got {err}"
        );
    }

    #[test]
    fn pipeline_and_streaming_error_helpers() {
        use crate::pipeline::LegacyPipelineError;
        use crate::streaming::StreamingError;

        let emb = EmbedderError::ResourceExhausted {
            detail: "busy".into(),
        };
        let pe = LegacyPipelineError::Embedding(emb.clone());
        let se = StreamingError::Embedding(emb);
        assert!(pe.is_resource_exhausted());
        assert!(se.is_resource_exhausted());
        let non_embedding = LegacyPipelineError::AudioTooLong {
            actual_secs: 2.0,
            max_secs: 1.0,
        };
        assert!(!non_embedding.is_resource_exhausted());
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod error_display_tests {
    use super::*;

    #[test]
    fn inference_failed_display_includes_detail() {
        let err = EmbedderError::InferenceFailed {
            detail: "session crashed".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("ONNX inference failed"));
        assert!(msg.contains("session crashed"));
    }

    #[test]
    fn resource_exhausted_display_includes_detail() {
        let err = EmbedderError::ResourceExhausted {
            detail: "all sessions busy".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("resource exhausted"));
        assert!(msg.contains("all sessions busy"));
    }

    #[test]
    fn dim_mismatch_display_includes_both_dims() {
        let err = EmbedderError::DimMismatch {
            expected: 192,
            actual: 256,
        };
        let msg = format!("{err}");
        assert!(msg.contains("192"));
        assert!(msg.contains("256"));
    }

    #[test]
    fn model_io_display_includes_path_and_detail() {
        let err = EmbedderError::ModelIo {
            path: std::path::PathBuf::from("/tmp/missing.onnx"),
            detail: "no such file".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("/tmp/missing.onnx"));
        assert!(msg.contains("no such file"));
    }

    #[test]
    fn legacy_display_includes_message() {
        let err = EmbedderError::Legacy("old adapter broke".into());
        assert!(format!("{err}").contains("old adapter broke"));
    }

    #[test]
    fn legacy_variant_with_pool_exhausted_string_classifies() {
        let err = EmbedderError::Legacy("onnx session pool exhausted".into());
        assert!(err.is_resource_exhausted());
    }

    #[test]
    fn legacy_variant_without_marker_does_not_classify() {
        let err = EmbedderError::Legacy("some other failure".into());
        assert!(!err.is_resource_exhausted());
    }

    #[test]
    fn inference_failed_without_marker_does_not_classify() {
        let err = EmbedderError::InferenceFailed {
            detail: "shape mismatch".into(),
        };
        assert!(!err.is_resource_exhausted());
    }

    #[test]
    fn unrelated_variants_do_not_classify_as_exhausted() {
        let too_short = EmbedderError::AudioTooShort {
            actual_secs: 0.1,
            min_secs: 0.5,
        };
        assert!(!too_short.is_resource_exhausted());

        let io = EmbedderError::ModelIo {
            path: std::path::PathBuf::from("m.onnx"),
            detail: "pool exhausted on disk".into(),
        };
        assert!(
            !io.is_resource_exhausted(),
            "ModelIo must not substring-match the legacy marker"
        );
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod dummy_extractor_tests {
    use super::*;

    fn tone_1s() -> Vec<f32> {
        vec![0.1_f32; 16_000]
    }

    #[test]
    fn dim_is_reported() {
        let e = DummyExtractor::new(192);
        assert_eq!(e.dim(), 192);
    }

    #[test]
    fn embed_returns_l2_normalized_vector_of_dim() {
        let e = DummyExtractor::new(256);
        let v = e.embed(&tone_1s()).unwrap();
        assert_eq!(v.len(), 256);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "expected unit vector, |v|={norm}"
        );
    }

    #[test]
    fn successive_embeds_differ() {
        let e = DummyExtractor::new(64);
        let a = e.embed(&tone_1s()).unwrap();
        let b = e.embed(&tone_1s()).unwrap();
        assert_ne!(a, b, "the internal seed advances between calls");
    }

    #[test]
    fn fresh_instances_reproduce_the_same_sequence() {
        let e1 = DummyExtractor::new(32);
        let e2 = DummyExtractor::new(32);
        for _ in 0..3 {
            assert_eq!(e1.embed(&[]).unwrap(), e2.embed(&[]).unwrap());
        }
    }

    #[test]
    fn zero_dim_extractor_returns_empty_embedding() {
        let e = DummyExtractor::new(0);
        assert_eq!(e.dim(), 0);
        assert!(e.embed(&tone_1s()).unwrap().is_empty());
    }

    #[test]
    fn default_batch_embeds_each_input() {
        let e = DummyExtractor::new(16);
        let audio = tone_1s();
        let inputs: Vec<&[f32]> = vec![&audio, &audio];
        let out = e.embed_batch(&inputs).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.len() == 16));
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(all(test, feature = "onnx", feature = "embedder"))]
mod onnx_adapter_tests {
    use super::*;
    use std::error::Error as _;
    use std::path::{Path, PathBuf};

    fn local_model(name: &str) -> Option<PathBuf> {
        let p = Path::new("models").join(name);
        if p.is_file() { Some(p) } else { None }
    }

    /// 1 second of synthetic 16 kHz mono audio (220 Hz tone).
    fn synthetic_audio_1s() -> Vec<f32> {
        use std::f32::consts::PI;
        let sr = 16_000_usize;
        (0..sr)
            .map(|i| (2.0 * PI * 220.0 * (i as f32 / sr as f32)).sin() * 0.3)
            .collect()
    }

    /// `expect_err` without requiring `Debug` on the adapter types.
    fn unwrap_err<T>(r: Result<T, EmbedderError>) -> EmbedderError {
        match r {
            Err(e) => e,
            Ok(_) => panic!("expected Err"),
        }
    }

    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn dim(&self) -> usize {
            4
        }
        fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            Err(EmbedderError::Legacy("synthetic failure".into()))
        }
    }

    #[test]
    fn resnet34_missing_model_reports_session_build_with_path() {
        let path = Path::new("models/definitely-not-a-real-model.onnx");
        let err = unwrap_err(ResNet34Adapter::new(
            path,
            1,
            crate::onnx::ExecutionProvider::Cpu,
        ));
        match &err {
            EmbedderError::SessionBuild { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected SessionBuild, got {other}"),
        }
        let msg = format!("{err}");
        assert!(msg.contains("definitely-not-a-real-model.onnx"));
        assert!(
            err.source().is_some(),
            "typed cause is preserved as the error source"
        );
    }

    #[test]
    fn cam_pp_zero_pool_size_fails_construction() {
        let err = unwrap_err(CamPlusPlusExtractor::new(
            "models/cam_pp_fp32.onnx",
            512,
            0,
            crate::onnx::ExecutionProvider::Cpu,
        ));
        assert!(
            matches!(err, EmbedderError::SessionBuild { .. }),
            "pool-size validation maps to SessionBuild, got {err}"
        );
    }

    #[test]
    fn eres2netv2_dim_constant_is_192() {
        assert_eq!(ERes2NetV2Extractor::DIM, 192);
    }

    #[test]
    fn eres2netv2_missing_model_reports_session_build() {
        let path = Path::new("models/definitely-not-eres2netv2.onnx");
        let err = unwrap_err(ERes2NetV2Extractor::new(
            path,
            1,
            crate::onnx::ExecutionProvider::Cpu,
        ));
        assert!(
            matches!(err, EmbedderError::SessionBuild { .. }),
            "got {err}"
        );

        let err = unwrap_err(ERes2NetV2Extractor::with_dim(
            path,
            256,
            1,
            crate::onnx::ExecutionProvider::Cpu,
        ));
        assert!(
            matches!(err, EmbedderError::SessionBuild { .. }),
            "got {err}"
        );
    }

    #[test]
    fn resnet34_real_model_embeds_256d_unit_vector() {
        let Some(path) = local_model("wespeaker_resnet34.onnx") else {
            eprintln!("skip resnet34_real_model: models/wespeaker_resnet34.onnx missing");
            return;
        };
        let extractor = ResNet34Adapter::new(&path, 1, crate::onnx::ExecutionProvider::Cpu)
            .expect("local model loads");
        assert_eq!(extractor.dim(), 256);

        let embedding = extractor.embed(&synthetic_audio_1s()).expect("embed");
        assert_eq!(embedding.len(), 256);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-2, "L2 norm not 1.0: {norm}");
    }

    #[test]
    fn cam_pp_real_model_embeds_and_batches_512d() {
        let Some(path) = local_model("cam_pp_fp32.onnx") else {
            eprintln!("skip cam_pp_real_model: models/cam_pp_fp32.onnx missing");
            return;
        };
        let extractor =
            CamPlusPlusExtractor::new(&path, 512, 2, crate::onnx::ExecutionProvider::Cpu)
                .expect("local model loads");
        assert_eq!(extractor.dim(), 512);

        let audio = synthetic_audio_1s();
        let embedding = extractor.embed(&audio).expect("embed");
        assert_eq!(embedding.len(), 512);

        // Batches fan out across the session pool via parallel_embed_batch.
        let batch = extractor
            .embed_batch(&[&audio, &audio, &audio])
            .expect("batch");
        assert_eq!(batch.len(), 3);
        for v in &batch {
            assert_eq!(v.len(), 512);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-2, "L2 norm not 1.0: {norm}");
        }
        // Deterministic model: same input yields the same embedding.
        assert_eq!(batch[0], batch[1]);
    }

    #[test]
    fn parallel_embed_batch_empty_input_returns_empty() {
        let e = DummyExtractor::new(8);
        let out = parallel_embed_batch(&e, &[], 4).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn parallel_embed_batch_collects_all_results() {
        let e = DummyExtractor::new(8);
        let audio = synthetic_audio_1s();
        let inputs: Vec<&[f32]> = (0..16).map(|_| &audio[..]).collect();
        let out = parallel_embed_batch(&e, &inputs, 4).unwrap();
        assert_eq!(out.len(), 16);
        for v in &out {
            assert_eq!(v.len(), 8);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "L2 norm not 1.0: {norm}");
        }
    }

    #[test]
    fn parallel_embed_batch_zero_max_threads_still_runs() {
        let e = DummyExtractor::new(8);
        let audio = synthetic_audio_1s();
        let out = parallel_embed_batch(&e, &[&audio, &audio], 0).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn parallel_embed_batch_propagates_inner_error() {
        let e = FailingEmbedder;
        let audio = synthetic_audio_1s();
        let err = parallel_embed_batch(&e, &[&audio, &audio], 2)
            .expect_err("inner failure must propagate");
        assert!(
            matches!(err, EmbedderError::Legacy(ref d) if d == "synthetic failure"),
            "got {err}"
        );
    }
}
