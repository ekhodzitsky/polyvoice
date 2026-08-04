//! Math utilities for diarization.
//!
//! Shared vector math (cosine similarity, L2 normalization, pairwise
//! similarity matrices, mean centroids, segment merging) used by clustering,
//! embedding, and overlap modules. See [`cosine_similarity`].

/// { true }
/// pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32
/// { ret >= -1.0 && ret <= 1.0 }
/// Compute cosine similarity between two vectors.
///
/// Returns `0.0` for zero vectors or length mismatches (with a `tracing::warn`).
///
/// ```rust
/// use polyvoice::utils::cosine_similarity;
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![0.0, 1.0, 0.0];
/// assert!(cosine_similarity(&a, &b).abs() < 1e-5);
///
/// let c = vec![1.0, 2.0, 3.0];
/// assert!((cosine_similarity(&c, &c) - 1.0).abs() < 1e-5);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        tracing::warn!(
            "cosine_similarity length mismatch: {} vs {}, returning 0.0",
            a.len(),
            b.len()
        );
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if !norm_a.is_finite() || !norm_b.is_finite() || norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    let sim = dot / (norm_a.sqrt() * norm_b.sqrt());
    if sim.is_finite() { sim } else { 0.0 }
}

/// { true }
/// pub fn l2_normalize(vec: &mut [f32])
/// { true }
/// L2-normalize a vector in-place.
///
/// If the vector norm is below `1e-8`, it is left unchanged (all zeros).
///
/// ```rust
/// use polyvoice::utils::l2_normalize;
/// let mut v = vec![3.0, 4.0];
/// l2_normalize(&mut v);
/// assert!((v[0] - 0.6).abs() < 1e-5);
/// assert!((v[1] - 0.8).abs() < 1e-5);
/// ```
pub fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() {
        // NaN/inf norm: zero the vector so downstream cosine math stays finite.
        vec.fill(0.0);
    } else if norm > 1e-8 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

/// { true }
/// pub fn cosine_similarity_f32_f64(a: &[f32], b: &[f64]) -> f32
/// { ret >= -1.0 && ret <= 1.0 }
/// Compute cosine similarity between an f32 slice and an f64 slice.
///
/// Returns `0.0` for zero vectors or length mismatches.
pub fn cosine_similarity_f32_f64(a: &[f32], b: &[f64]) -> f32 {
    if a.len() != b.len() {
        tracing::warn!(
            "cosine_similarity_f32_f64 length mismatch: {} vs {}, returning 0.0",
            a.len(),
            b.len()
        );
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let y = y as f32;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if !norm_a.is_finite() || !norm_b.is_finite() || norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    let sim = dot / (norm_a.sqrt() * norm_b.sqrt());
    if sim.is_finite() { sim } else { 0.0 }
}

/// { true }
/// `pub fn mean_vector(vectors: &[Vec<f32>]) -> Option<Vec<f32>>`
/// { ret.as_ref().map_or(true, |v| vectors.iter().all(|u| u.len() == v.len())) }
/// Compute the element-wise mean of a list of vectors.
///
/// Returns `None` if the input slice is empty.
///
/// ```rust
/// use polyvoice::utils::mean_vector;
/// let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// let mean = mean_vector(&vectors).unwrap();
/// assert!((mean[0] - 2.0).abs() < 1e-5);
/// assert!((mean[1] - 3.0).abs() < 1e-5);
/// ```
pub fn mean_vector(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    if vectors.is_empty() {
        return None;
    }
    let dim = vectors[0].len();
    let mut sum = vec![0.0f32; dim];
    for v in vectors {
        for (s, &x) in sum.iter_mut().zip(v.iter()) {
            *s += x;
        }
    }
    let n = vectors.len() as f32;
    for s in &mut sum {
        *s /= n;
    }
    Some(sum)
}

/// { embeddings.is_empty() || embeddings.iter().all(|e| e.len() == embeddings`[0]`.len()) }
/// `pub(crate) fn pairwise_cosine_similarity_matrix(embeddings: &[Vec<f32>]) -> Vec<f32>`
/// { ret.len() == embeddings.len() * embeddings.len() }
/// Full symmetric pairwise cosine-similarity matrix as a flat row-major
/// `n * n` buffer; the diagonal is exactly `1.0`.
///
/// Each off-diagonal entry is [`cosine_similarity`] of that pair (f32).
/// Callers needing `f64` cast entry-wise (there is no extra precision to
/// keep); callers needing a nested layout chunk the flat rows.
pub(crate) fn pairwise_cosine_similarity_matrix(embeddings: &[Vec<f32>]) -> Vec<f32> {
    let n = embeddings.len();
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let s = cosine_similarity(&embeddings[i], &embeddings[j]);
            m[i * n + j] = s;
            m[j * n + i] = s;
        }
    }
    m
}

/// { indices.iter().all(|&i| i < embeddings.len()) && labels.len() == indices.len() }
/// `pub(crate) fn normalized_mean_centroids(embeddings, indices, labels, k) -> Vec<Vec<f32>>`
/// { ret.len() == k }
/// Per-slot L2-normalized mean centroid over selected members.
///
/// `indices[i]` is an embedding index and `labels[i]` its centroid slot in
/// `0..k`; embedding `embeddings[indices[i]]` contributes to centroid
/// `labels[i]`. Labels `>= k` (and out-of-range indices) are skipped. Slots
/// with no members stay zero vectors. The mean divides by the member count
/// (same as [`mean_vector`]) and the result is normalized with
/// [`l2_normalize`].
pub(crate) fn normalized_mean_centroids(
    embeddings: &[Vec<f32>],
    indices: &[usize],
    labels: &[usize],
    k: usize,
) -> Vec<Vec<f32>> {
    let dim = embeddings.first().map(Vec::len).unwrap_or(0);
    let mut sums = vec![vec![0.0f32; dim]; k];
    let mut counts = vec![0usize; k];
    for (&i, &l) in indices.iter().zip(labels.iter()) {
        if l >= k {
            continue;
        }
        if let Some(emb) = embeddings.get(i) {
            for (a, &x) in sums[l].iter_mut().zip(emb.iter()) {
                *a += x;
            }
            counts[l] += 1;
        }
    }
    for (c, &n) in sums.iter_mut().zip(counts.iter()) {
        if n > 0 {
            for v in c.iter_mut() {
                *v /= n as f32;
            }
        }
        l2_normalize(c);
    }
    sums
}

use crate::types::Segment;

/// { true }
/// `pub fn merge_segments(segments: Vec<Segment>, max_gap_secs: f64) -> Vec<Segment>`
/// { ret.len() <= segments.len() }
/// Merge adjacent segments with the same speaker if the gap between them
/// is less than `max_gap_secs`.
///
/// The merged confidence is the arithmetic mean of the present (`Some`)
/// confidences across the whole run — order-independent; `None` values are not
/// counted, and a run with no confidences stays `None`.
///
/// ```rust
/// use polyvoice::{merge_segments, Segment, SpeakerId, TimeRange};
/// let segs = vec![
///     Segment { time: TimeRange { start: 0.0, end: 1.0 }, speaker: Some(SpeakerId(0)), confidence: Some(0.8) },
///     Segment { time: TimeRange { start: 1.2, end: 2.0 }, speaker: Some(SpeakerId(0)), confidence: Some(0.9) },
///     Segment { time: TimeRange { start: 2.5, end: 3.0 }, speaker: Some(SpeakerId(1)), confidence: None },
/// ];
/// let merged = merge_segments(segs, 0.5);
/// assert_eq!(merged.len(), 2);
/// assert!((merged[0].time.end - 2.0).abs() < 1e-5);
/// ```
pub fn merge_segments(segments: Vec<Segment>, max_gap_secs: f64) -> Vec<Segment> {
    if segments.is_empty() {
        return segments;
    }
    let mut merged = Vec::new();
    let mut current = segments[0].clone();
    // Accumulate confidence over the whole run and take the arithmetic mean once
    // at flush: order-independent (vs the old pairwise (c1+c2)/2 that recency-
    // weighted earlier segments by 2^-(n-1)) and not poisoned by a single `None`
    // (a None segment is simply not counted instead of forcing the run to None).
    let (mut conf_sum, mut conf_count) = match current.confidence {
        Some(c) => (c, 1u32),
        None => (0.0, 0u32),
    };

    for next in segments.into_iter().skip(1) {
        if current.speaker == next.speaker && next.time.start - current.time.end <= max_gap_secs {
            current.time.end = next.time.end;
            if let Some(c) = next.confidence {
                conf_sum += c;
                conf_count += 1;
            }
        } else {
            current.confidence = mean_confidence(conf_sum, conf_count);
            merged.push(current);
            current = next;
            (conf_sum, conf_count) = match current.confidence {
                Some(c) => (c, 1u32),
                None => (0.0, 0u32),
            };
        }
    }
    current.confidence = mean_confidence(conf_sum, conf_count);
    merged.push(current);
    merged
}

/// Arithmetic mean of the accumulated `Some` confidences in a merged run, or
/// `None` when the run carried no confidence values.
fn mean_confidence(sum: f32, count: u32) -> Option<f32> {
    if count > 0 {
        Some(sum / count as f32)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Object pool — blocking Mutex<Vec<T>> checkout (ONNX sessions, embedders)
// ---------------------------------------------------------------------------

/// Blocking object pool backed by `Mutex<Vec<T>>`.
///
/// Checkout waits until an item is available; [`Drop`] returns it. Pool
/// checkout is not a contention-hot path in this crate, so a small mutex pool
/// is preferred over a dedicated lock-free queue crate.
///
/// Compiled for the `onnx` session pool (`fbank_onnx::FbankOnnxExtractor`) and for
/// unit tests; unused in the default ort-free build.
#[cfg(any(test, feature = "onnx"))]
pub(crate) struct ObjectPool<T> {
    items: std::sync::Mutex<Vec<T>>,
}

/// RAII guard: the pooled item is returned on drop.
#[cfg(any(test, feature = "onnx"))]
pub(crate) struct PooledGuard<'a, T> {
    item: Option<T>,
    pool: &'a ObjectPool<T>,
}

#[cfg(any(test, feature = "onnx"))]
impl<T> ObjectPool<T> {
    pub(crate) fn new(items: Vec<T>) -> Self {
        Self {
            items: std::sync::Mutex::new(items),
        }
    }

    /// Blocking checkout. Spins with yield until an item is free.
    ///
    /// **Caller must ensure the pool is non-empty** (or that empty is handled
    /// before calling); an empty pool spins forever.
    pub(crate) fn checkout(&self) -> PooledGuard<'_, T> {
        loop {
            {
                let mut guard = self.items.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(item) = guard.pop() {
                    return PooledGuard {
                        item: Some(item),
                        pool: self,
                    };
                }
            }
            std::thread::yield_now();
        }
    }
}

#[cfg(any(test, feature = "onnx"))]
impl<T> std::ops::Deref for PooledGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // Invariant: item is Some until Drop takes it.
        match self.item.as_ref() {
            Some(item) => item,
            None => unreachable!("pooled item missing before Drop"),
        }
    }
}

#[cfg(any(test, feature = "onnx"))]
impl<T> std::ops::DerefMut for PooledGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        match self.item.as_mut() {
            Some(item) => item,
            None => unreachable!("pooled item missing before Drop"),
        }
    }
}

#[cfg(any(test, feature = "onnx"))]
impl<T> Drop for PooledGuard<'_, T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            self.pool
                .items
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(item);
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic PRNG — xorshift64* for k-means++ seeding
// ---------------------------------------------------------------------------

/// Marsaglia/Vigna xorshift64* — tiny deterministic PRNG for k-means++ draws.
///
/// Replaces an external RNG dependency for a handful of samples per clustering
/// call. Sequence is fully determined by the seed (non-zero state).
#[derive(Clone, Debug)]
pub(crate) struct XorShift64Star {
    state: u64,
}

impl XorShift64Star {
    /// Construct from a seed. Seed `0` is remapped so the generator is usable.
    pub(crate) fn new(seed: u64) -> Self {
        // xorshift state must be non-zero; splitmix-style constant as fallback.
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub(crate) fn f64(&mut self) -> f64 {
        // Top 53 bits → IEEE-754 double mantissa.
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform integer in `0..upper` (exclusive). `upper` must be > 0.
    pub(crate) fn usize(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        // Multiply-high mapping: nearly unbiased for any upper << 2^64.
        let upper = upper as u64;
        ((self.next_u64() as u128 * upper as u128) >> 64) as usize
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_mean_vector() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mean = mean_vector(&vectors).unwrap();
        assert!((mean[0] - 2.0).abs() < 1e-5);
        assert!((mean[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_nan_input_returns_finite_zero() {
        let a = vec![f32::NAN, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.is_finite());
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn cosine_similarity_inf_input_returns_finite_zero() {
        let a = vec![f32::INFINITY, 1.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.is_finite());
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn cosine_similarity_f32_f64_nan_input_returns_finite_zero() {
        let a = vec![f32::NAN, 1.0, 0.0];
        let b = vec![1.0_f64, 0.0, 0.0];
        let sim = cosine_similarity_f32_f64(&a, &b);
        assert!(sim.is_finite());
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn l2_normalize_nan_input_yields_finite_vector() {
        let mut v = vec![f32::NAN, 1.0, 2.0];
        l2_normalize(&mut v);
        assert!(v.iter().all(|x| x.is_finite()));
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn l2_normalize_inf_input_yields_finite_vector() {
        let mut v = vec![f32::INFINITY, 1.0, 2.0];
        l2_normalize(&mut v);
        assert!(v.iter().all(|x| x.is_finite()));
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn pairwise_cosine_similarity_matrix_is_symmetric_with_unit_diagonal() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let n = embeddings.len();
        let m = pairwise_cosine_similarity_matrix(&embeddings);
        assert_eq!(m.len(), n * n);
        for i in 0..n {
            assert_eq!(m[i * n + i], 1.0);
            for j in 0..n {
                assert_eq!(m[i * n + j], m[j * n + i]);
                if i != j {
                    assert_eq!(
                        m[i * n + j],
                        cosine_similarity(&embeddings[i], &embeddings[j])
                    );
                }
            }
        }
    }

    #[test]
    fn pairwise_cosine_similarity_matrix_empty() {
        assert!(pairwise_cosine_similarity_matrix(&[]).is_empty());
    }

    #[test]
    fn normalized_mean_centroids_basic() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 2.0],
            vec![0.0, 1.0],
        ];
        let indices = [0, 1, 2, 3];
        let labels = [0, 0, 1, 1];
        let centroids = normalized_mean_centroids(&embeddings, &indices, &labels, 2);
        assert_eq!(centroids.len(), 2);
        for c in &centroids {
            let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "centroid not unit norm");
        }
        assert!((centroids[0][0] - 1.0).abs() < 1e-5);
        assert!((centroids[1][1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normalized_mean_centroids_empty_slot_stays_zero() {
        let embeddings = vec![vec![1.0, 0.0]];
        let centroids = normalized_mean_centroids(&embeddings, &[0], &[0], 3);
        assert_eq!(centroids.len(), 3);
        assert_eq!(centroids[1], vec![0.0, 0.0]);
        assert_eq!(centroids[2], vec![0.0, 0.0]);
    }

    #[test]
    fn normalized_mean_centroids_skips_out_of_range() {
        let embeddings = vec![vec![1.0, 0.0]];
        // Label >= k and index >= len are ignored.
        let centroids = normalized_mean_centroids(&embeddings, &[0, 5], &[7, 0], 2);
        assert_eq!(centroids.len(), 2);
        assert_eq!(centroids[0], vec![0.0, 0.0]);
        assert_eq!(centroids[1], vec![0.0, 0.0]);
    }

    fn seg(start: f64, end: f64, spk: u32, conf: Option<f32>) -> Segment {
        Segment {
            time: crate::types::TimeRange { start, end },
            speaker: Some(crate::types::SpeakerId(spk)),
            confidence: conf,
        }
    }

    #[test]
    fn merge_confidence_is_order_independent_mean() {
        // Three same-speaker segments merge into one run. Confidence must be the
        // arithmetic mean (0.8), not the old recency-weighted pairwise fold
        // ((0.6+0.9)/2 + 0.9)/2 = 0.825.
        let segs = vec![
            seg(0.0, 1.0, 0, Some(0.6)),
            seg(1.0, 2.0, 0, Some(0.9)),
            seg(2.0, 3.0, 0, Some(0.9)),
        ];
        let merged = merge_segments(segs, 0.5);
        assert_eq!(merged.len(), 1);
        let c = merged[0].confidence.expect("merged run has confidence");
        assert!(
            (c - 0.8).abs() < 1e-6,
            "expected arithmetic mean 0.8, got {c}"
        );
    }

    #[test]
    fn merge_confidence_ignores_none_no_poisoning() {
        // First segment has no confidence; the run mean must come from the
        // present values (0.7), not be poisoned to None by the leading None.
        let segs = vec![
            seg(0.0, 1.0, 0, None),
            seg(1.0, 2.0, 0, Some(0.8)),
            seg(2.0, 3.0, 0, Some(0.6)),
        ];
        let merged = merge_segments(segs, 0.5);
        assert_eq!(merged.len(), 1);
        let c = merged[0]
            .confidence
            .expect("present values must yield a mean");
        assert!((c - 0.7).abs() < 1e-6, "expected 0.7, got {c}");
    }

    #[test]
    fn merge_confidence_all_none_stays_none() {
        let segs = vec![seg(0.0, 1.0, 0, None), seg(1.0, 2.0, 0, None)];
        let merged = merge_segments(segs, 0.5);
        assert_eq!(merged.len(), 1);
        assert!(merged[0].confidence.is_none());
    }

    #[test]
    fn cosine_similarity_length_mismatch_and_zero_norm_return_zero() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_similarity_f32_f64_paths() {
        // Length mismatch.
        assert_eq!(cosine_similarity_f32_f64(&[1.0, 2.0], &[1.0_f64]), 0.0);
        // Zero-norm input.
        assert_eq!(cosine_similarity_f32_f64(&[0.0, 0.0], &[1.0_f64, 2.0]), 0.0);
        // Matching direction across precisions.
        let sim = cosine_similarity_f32_f64(&[1.0, 0.0], &[1.0_f64, 0.0]);
        assert!((sim - 1.0).abs() < 1e-5);
        // Non-finite input stays finite.
        let sim = cosine_similarity_f32_f64(&[f32::INFINITY, 1.0], &[1.0_f64, 2.0]);
        assert!(sim.is_finite());
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn l2_normalize_zero_vector_unchanged() {
        let mut v = vec![0.0f32; 3];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0; 3]);
    }

    #[test]
    fn mean_vector_empty_returns_none() {
        assert!(mean_vector(&[]).is_none());
    }

    #[test]
    fn merge_segments_empty_input() {
        assert!(merge_segments(Vec::new(), 0.5).is_empty());
    }

    #[test]
    fn merge_segments_flushes_on_speaker_change_and_gap() {
        let segs = vec![
            seg(0.0, 1.0, 0, Some(0.8)),
            seg(1.1, 2.0, 1, Some(0.5)), // different speaker → flush
            seg(2.1, 3.0, 1, Some(0.7)), // same speaker, small gap → extend
            seg(5.0, 6.0, 1, Some(1.0)), // gap above max_gap_secs → flush
        ];
        let merged = merge_segments(segs, 0.5);
        assert_eq!(merged.len(), 3);
        assert!((merged[0].confidence.expect("confidence") - 0.8).abs() < 1e-6);
        assert!((merged[1].time.start - 1.1).abs() < 1e-9);
        assert!((merged[1].time.end - 3.0).abs() < 1e-9);
        assert!((merged[1].confidence.expect("confidence") - 0.6).abs() < 1e-6);
        assert!((merged[2].confidence.expect("confidence") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn object_pool_checkout_deref_and_return_on_drop() {
        let pool = ObjectPool::new(vec![1u32, 2, 3]);
        {
            let mut guard = pool.checkout();
            *guard += 10;
            assert!(*guard >= 11);
        }
        // The mutated item is back in the pool after the guard dropped.
        let guard = pool.checkout();
        assert!(*guard >= 1);
    }

    #[test]
    fn object_pool_checkout_blocks_until_item_returned() {
        use std::sync::Arc;
        let pool = Arc::new(ObjectPool::new(vec![7u32]));
        let guard = pool.checkout();
        let pool2 = Arc::clone(&pool);
        let handle = std::thread::spawn(move || *pool2.checkout());
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!handle.is_finished());
        drop(guard);
        assert_eq!(handle.join().expect("worker panicked"), 7);
    }

    #[test]
    fn xorshift_seed_zero_remapped_and_deterministic() {
        let mut a = XorShift64Star::new(0);
        let mut b = XorShift64Star::new(0);
        assert_eq!(a.next_u64(), b.next_u64());
        let mut c = XorShift64Star::new(1);
        assert_ne!(a.next_u64(), c.next_u64());
    }

    #[test]
    fn xorshift_f64_and_usize_ranges() {
        let mut rng = XorShift64Star::new(42);
        for _ in 0..1000 {
            let f = rng.f64();
            assert!((0.0..1.0).contains(&f), "f64 out of range: {f}");
        }
        let mut rng = XorShift64Star::new(42);
        for _ in 0..1000 {
            assert!(rng.usize(7) < 7);
        }
    }
}
