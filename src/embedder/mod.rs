#![allow(deprecated)] // blanket bridge keeps EmbeddingExtractor → Embedder working
//! v1.0 `Embedder` trait + concrete extractors (CAM++, ResNet34) + pool +
//! overlap-mask helper.
//!
//! `Embedder` is the supported bring-your-own embedder contract for offline
//! [`crate::Pipeline`] and online [`crate::streaming::StreamingPipeline`].
//! The pure-Rust trait, pool, and overlap mask are always available (no
//! `onnx` required). ONNX-backed adapters still need `features = ["onnx", "embedder"]`.
//!
//! Shared fbank+ONNX engine: [`crate::ecapa::FbankOnnxExtractor`] (implements
//! [`Embedder`] directly). Architecture adapters wrap it without the legacy
//! `EmbeddingExtractor` path. External types that still implement
//! [`crate::embedding::EmbeddingExtractor`] automatically satisfy [`Embedder`]
//! via a blanket bridge.

/// Speaker embedding extractor — turns a slice of 16 kHz mono audio into a
/// fixed-dimension embedding vector. Implementations are expected to L2-normalize
/// their output so cosine similarity is a meaningful metric downstream.
///
/// This is the **supported library injection API** for [`crate::Pipeline`] and
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

/// Map a legacy [`crate::embedding::EmbeddingError`] into [`EmbedderError`],
/// preserving exhaustion as a typed variant when possible.
#[allow(deprecated)]
fn from_legacy_embedding_error(err: crate::embedding::EmbeddingError) -> EmbedderError {
    use crate::embedding::EmbeddingError as Legacy;
    match err {
        Legacy::ResourceExhausted(detail) => EmbedderError::ResourceExhausted { detail },
        Legacy::InferenceFailed(detail) if detail_looks_exhausted(&detail) => {
            EmbedderError::ResourceExhausted { detail }
        }
        other => EmbedderError::Legacy(other.to_string()),
    }
}

/// Bridge: any legacy [`crate::embedding::EmbeddingExtractor`] is an [`Embedder`].
///
/// Pipeline-owned [`crate::types::DiarizationConfig`] is not part of the
/// `Embedder` surface; the bridge supplies `DiarizationConfig::default()`.
/// In-tree legacy extractors ignore that config (window length is enforced by
/// the caller / model). External extractors that read config fields should
/// implement [`Embedder`] directly.
impl<T> Embedder for T
where
    T: crate::embedding::EmbeddingExtractor + ?Sized,
{
    fn dim(&self) -> usize {
        self.embedding_dim()
    }

    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        let config = crate::types::DiarizationConfig::default();
        self.extract(audio, &config)
            .map_err(from_legacy_embedding_error)
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
/// Generic over `E: Embedder` so the same pool implementation works for
/// `CamPlusPlusExtractor`, `ResNet34Adapter`, or any user-provided embedder.
/// All embedders in a pool must share the same output dimension.
///
/// Backed by a blocking object pool (`Mutex<Vec<E>>`): checkout waits until an
/// embedder is free; Drop returns it.
pub struct EmbedderPool<E: Embedder> {
    pool: crate::utils::ObjectPool<E>,
    dim: usize,
    capacity: usize,
}

impl<E: Embedder> EmbedderPool<E> {
    /// { true }
    /// `pub fn new(embedders: Vec<E>) -> Result<Self, EmbedderError>`
    /// { ret.is_ok() => ret.as_ref().unwrap().dim() == embedders.first().map_or(0, |e| e.dim()) }
    /// Build a pool from a list of embedders. All must share the same `dim()`.
    /// An empty list constructs a pool that fails on every call (returns
    /// `EmbedderError::Legacy("empty pool")`).
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
        let capacity = embedders.len().max(1);
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
    /// pub fn capacity(&self) -> usize
    /// { ret == self.capacity }
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// { true }
    /// `pub fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError>`
    /// { ret.as_ref().map_or(true, |v| v.len() == self.dim) }
    /// Extract a single embedding using the next-available pooled embedder.
    /// Blocks until one is free.
    pub fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        if self.dim == 0 {
            // Empty-construction case (no items in the pool).
            return Err(EmbedderError::ResourceExhausted {
                detail: "empty embedder pool".to_owned(),
            });
        }
        let embedder = self.pool.checkout();
        embedder.embed(audio)
    }
}

/// Parallel batch embedding using `std::thread::scope`.
/// Spawns up to `available_parallelism` threads, each processing a chunk
/// of the input via `embedder.embed()`.
///
/// Only referenced by ONNX-backed adapters (`ResNet34`, CAM++, ERes2NetV2).
#[cfg(all(feature = "onnx", feature = "embedder"))]
fn parallel_embed_batch<E: Embedder>(
    embedder: &E,
    audios: &[&[f32]],
) -> Result<Vec<Vec<f32>>, EmbedderError> {
    let n = audios.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
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
    use crate::ecapa::FbankOnnxExtractor;
    use std::path::Path;

    /// WeSpeaker ResNet34 embedder (256-d) via the shared fbank+ONNX engine.
    pub struct ResNet34Adapter {
        inner: FbankOnnxExtractor,
        dim: usize,
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
            let inner =
                FbankOnnxExtractor::new(path.as_ref(), 256, pool_size, ep).map_err(|e| {
                    EmbedderError::ModelIo {
                        path: path.as_ref().to_path_buf(),
                        detail: format!("{e}"),
                    }
                })?;
            Ok(Self { inner, dim: 256 })
        }
    }

    impl Embedder for ResNet34Adapter {
        fn dim(&self) -> usize {
            self.dim
        }

        fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            self.inner.embed(audio)
        }

        fn embed_batch(&self, audios: &[&[f32]]) -> Result<Vec<Vec<f32>>, EmbedderError> {
            parallel_embed_batch(self, audios)
        }
    }

    /// CAM++ embedder (Channel-Attentive Multi-scale Pooling). Dim is supplied
    /// explicitly because WeSpeaker ships several CAM++ variants:
    /// `voxceleb_CAM++.onnx` is 512-d; smaller variants exist at 192-d.
    /// Uses the same 80-bin log-mel fbank pipeline as ResNet34.
    pub struct CamPlusPlusExtractor {
        inner: FbankOnnxExtractor,
        dim: usize,
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
            let inner =
                FbankOnnxExtractor::new(path.as_ref(), dim, pool_size, ep).map_err(|e| {
                    EmbedderError::ModelIo {
                        path: path.as_ref().to_path_buf(),
                        detail: format!("{e}"),
                    }
                })?;
            Ok(Self { inner, dim })
        }
    }

    impl Embedder for CamPlusPlusExtractor {
        fn dim(&self) -> usize {
            self.dim
        }

        fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            self.inner.embed(audio)
        }

        fn embed_batch(&self, audios: &[&[f32]]) -> Result<Vec<Vec<f32>>, EmbedderError> {
            parallel_embed_batch(self, audios)
        }
    }

    /// ERes2NetV2 speaker embedder (Interspeech 2024): 192-d output, same
    /// 80-bin log-mel fbank path as CAM++. Tuned for short (1–3 s) utterances.
    /// Weights are optional downloads (Apache-2.0); never bundled or default.
    pub struct ERes2NetV2Extractor {
        inner: FbankOnnxExtractor,
        dim: usize,
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
            let inner =
                FbankOnnxExtractor::new(path.as_ref(), dim, pool_size, ep).map_err(|e| {
                    EmbedderError::ModelIo {
                        path: path.as_ref().to_path_buf(),
                        detail: format!("{e}"),
                    }
                })?;
            Ok(Self { inner, dim })
        }
    }

    impl Embedder for ERes2NetV2Extractor {
        fn dim(&self) -> usize {
            self.dim
        }

        fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            self.inner.embed(audio)
        }

        fn embed_batch(&self, audios: &[&[f32]]) -> Result<Vec<Vec<f32>>, EmbedderError> {
            parallel_embed_batch(self, audios)
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
    #[allow(deprecated)]
    fn legacy_bridge_preserves_resource_exhausted() {
        use crate::embedding::{EmbeddingError, EmbeddingExtractor};
        use crate::types::DiarizationConfig;

        struct ExhaustedExtractor;
        impl EmbeddingExtractor for ExhaustedExtractor {
            fn extract(
                &self,
                _samples: &[f32],
                _config: &DiarizationConfig,
            ) -> Result<Vec<f32>, EmbeddingError> {
                Err(EmbeddingError::ResourceExhausted("sessions busy".into()))
            }
            fn embedding_dim(&self) -> usize {
                4
            }
        }

        let err = ExhaustedExtractor
            .embed(&[0.0; 16])
            .expect_err("must fail");
        assert!(
            matches!(
                err,
                EmbedderError::ResourceExhausted { ref detail } if detail == "sessions busy"
            ),
            "bridge must not collapse exhaustion into Legacy, got {err}"
        );
        assert!(err.is_resource_exhausted());
    }

    #[test]
    fn pipeline_and_streaming_error_helpers() {
        use crate::pipeline::PipelineError;
        use crate::streaming::StreamingError;

        let emb = EmbedderError::ResourceExhausted {
            detail: "busy".into(),
        };
        let pe = PipelineError::Embedding(emb.clone());
        let se = StreamingError::Embedding(emb);
        assert!(pe.is_resource_exhausted());
        assert!(se.is_resource_exhausted());
        assert!(!PipelineError::NoSpeech.is_resource_exhausted());
    }
}
