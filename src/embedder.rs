//! v1.0 `Embedder` trait + concrete extractors (CAM++, ResNet34) + pool +
//! overlap-mask helper.
//!
//! Added in v0.6 (M2). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §3.1, §5.3.

/// Speaker embedding extractor — turns a slice of 16 kHz mono audio into a
/// fixed-dimension embedding vector. Implementations are expected to L2-normalize
/// their output so cosine similarity is a meaningful metric downstream.
///
/// In v1.0 (M2) the polyvoice crate introduces `Embedder` as the canonical
/// trait. The legacy `EmbeddingExtractor` trait and its implementations
/// (`FbankOnnxExtractor`, `OnnxEmbeddingExtractor`, `DummyExtractor`) remain
/// available unchanged — M6 will deprecate them.
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
#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    #[error("audio too short for this embedder: {actual_secs:.3}s < {min_secs:.3}s")]
    AudioTooShort { actual_secs: f32, min_secs: f32 },

    #[error("ONNX inference failed: {detail}")]
    InferenceFailed { detail: String },

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

use crossbeam_queue::ArrayQueue;
use std::sync::Arc;

/// Lock-free pool of `Embedder` instances for concurrent extraction.
///
/// Generic over `E: Embedder` so the same pool implementation works for
/// `CamPlusPlusExtractor`, `ResNet34Adapter`, or any user-provided embedder.
/// All embedders in a pool must share the same output dimension.
pub struct EmbedderPool<E: Embedder> {
    queue: Arc<ArrayQueue<E>>,
    dim: usize,
    capacity: usize,
}

impl<E: Embedder> EmbedderPool<E> {
    /// Build a pool from a list of embedders. All must share the same `dim()`.
    /// An empty list constructs a pool that fails on every call (returns
    /// `EmbedderError::Legacy("empty pool")`).
    pub fn new(embedders: Vec<E>) -> Self {
        let dim = embedders.first().map(|e| e.dim()).unwrap_or(0);
        let capacity = embedders.len().max(1);
        let queue = Arc::new(ArrayQueue::new(capacity));
        for e in embedders {
            // ArrayQueue::push only fails if full; capacity == count, so push always succeeds.
            let _ = queue.push(e);
        }
        Self {
            queue,
            dim,
            capacity,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Extract a single embedding using the next-available pooled embedder.
    /// Blocks (busy-spins) until one is free.
    pub fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        if self.dim == 0 {
            // Empty-construction case.
            return Err(EmbedderError::Legacy("empty pool".to_owned()));
        }
        // Acquire (busy-wait fallback for simplicity; real Pipeline use is
        // through `rayon::par_iter` which already throttles concurrency).
        let embedder = loop {
            if let Some(e) = self.queue.pop() {
                break e;
            }
            std::hint::spin_loop();
        };
        let result = embedder.embed(audio);
        // Always return the embedder.
        let _ = self.queue.push(embedder);
        result
    }
}

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
        for i in 0..audio.len() {
            if (8000..11200).contains(&i) {
                assert_eq!(masked[i], 0.0, "sample {i} should be zeroed");
            } else {
                assert_eq!(masked[i], 1.0, "sample {i} should pass through");
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
        for i in 0..8000 {
            assert_eq!(masked[i], 0.0);
        }
        for i in 8000..16_000 {
            assert_eq!(masked[i], 1.0);
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
        let pool = EmbedderPool::new(embedders);
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
        let pool: EmbedderPool<CountingEmbedder> = EmbedderPool::new(Vec::new());
        let err = pool
            .embed(&[0.0_f32; 100])
            .expect_err("empty pool must fail");
        assert!(matches!(err, EmbedderError::Legacy(_)));
    }
}
