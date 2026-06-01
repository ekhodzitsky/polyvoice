//! Math utilities for diarization.
//!
//! Shared vector math (cosine similarity, L2 normalization, segment merging)
//! used by clustering, embedding, and overlap modules. See [`cosine_similarity`].

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

use crate::types::Segment;

/// { true }
/// `pub fn merge_segments(segments: Vec<Segment>, max_gap_secs: f64) -> Vec<Segment>`
/// { ret.len() <= segments.len() }
/// Merge adjacent segments with the same speaker if the gap between them
/// is less than `max_gap_secs`.
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

    for next in segments.into_iter().skip(1) {
        if current.speaker == next.speaker && next.time.start - current.time.end <= max_gap_secs {
            current.time.end = next.time.end;
            // Average confidence when both sides have it.
            if let (Some(c1), Some(c2)) = (current.confidence, next.confidence) {
                current.confidence = Some((c1 + c2) / 2.0);
            }
        } else {
            merged.push(current);
            current = next;
        }
    }
    merged.push(current);
    merged
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
}
