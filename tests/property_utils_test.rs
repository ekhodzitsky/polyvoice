#![allow(clippy::unwrap_used)]
//! Property tests for utility function invariants.

use polyvoice::utils::{cosine_similarity, l2_normalize, mean_vector};
use polyvoice::{Segment, SpeakerId, TimeRange, merge_segments};
use proptest::prelude::*;

fn non_zero_vec() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1.0f32..=1.0f32, 1..=256)
        .prop_filter("non-zero vector", |v| v.iter().any(|&x| x != 0.0))
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        ..ProptestConfig::default()
    })]

    /// cosine_similarity always returns a value in [-1.0, 1.0].
    #[test]
    fn cosine_similarity_range(a in non_zero_vec(), b in non_zero_vec()) {
        let sim = cosine_similarity(&a, &b);
        prop_assert!(
            (-1.0..=1.0).contains(&sim),
            "cosine_similarity must be in [-1, 1], got {} for vectors of len {} and {}",
            sim, a.len(), b.len()
        );
    }

    /// cosine_similarity of identical vectors is exactly 1.0.
    #[test]
    fn cosine_similarity_identical_is_one(v in non_zero_vec()) {
        let sim = cosine_similarity(&v, &v);
        prop_assert!(
            (sim - 1.0).abs() < 1e-5,
            "cosine_similarity of identical vectors should be 1.0, got {}",
            sim
        );
    }

    /// After l2_normalize, vector norm is 1.0 (for non-zero vectors).
    #[test]
    fn l2_normalize_produces_unit_vector(mut v in non_zero_vec()) {
        l2_normalize(&mut v);
        let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        prop_assert!(
            (norm - 1.0).abs() < 1e-5,
            "l2_normalize should produce unit vector, got norm {}",
            norm
        );
    }

    /// mean_vector output length equals input dimension.
    #[test]
    fn mean_vector_dimension(vecs in prop::collection::vec(non_zero_vec(), 1..=16)) {
        let dim = vecs[0].len();
        let mean = mean_vector(&vecs);
        prop_assert!(
            mean.is_some(),
            "mean_vector should return Some for non-empty input"
        );
        prop_assert_eq!(
            mean.unwrap().len(),
            dim,
            "mean_vector output dimension must match input dimension"
        );
    }

    /// merge_segments collapses a same-speaker contiguous run into one segment
    /// whose confidence is the arithmetic mean of the Some() members (None
    /// ignored), or None if no member carries a confidence — independent of how
    /// many fold steps the merge performs. Pins the order-independent mean-confidence fix.
    #[test]
    fn merge_confidence_is_mean_of_some_members(
        items in prop::collection::vec(
            (0.0f64..=0.5f64, prop_oneof![Just(None::<f32>), (0.0f32..1.0f32).prop_map(Some)]),
            2..8usize,
        )
    ) {
        const MAX_GAP: f64 = 0.5;
        // Build a contiguous, same-speaker run: each segment starts within
        // MAX_GAP of the previous end, so the whole run merges into one.
        let mut segs = Vec::new();
        let mut t = 0.0f64;
        for (gap, conf) in &items {
            let start = t + *gap;
            let end = start + 1.0;
            segs.push(Segment {
                time: TimeRange { start, end },
                speaker: Some(SpeakerId(0)),
                confidence: *conf,
            });
            t = end;
        }
        let merged = merge_segments(segs, MAX_GAP);
        prop_assert_eq!(merged.len(), 1, "contiguous same-speaker run must merge to one");

        let some: Vec<f32> = items.iter().filter_map(|(_, c)| *c).collect();
        match merged[0].confidence {
            None => prop_assert!(some.is_empty(), "None only when no member had a confidence"),
            Some(c) => {
                prop_assert!(!some.is_empty(), "Some implies at least one member had a confidence");
                let mean = some.iter().sum::<f32>() / some.len() as f32;
                prop_assert!((c - mean).abs() < 1e-5, "confidence {} != arithmetic mean {}", c, mean);
            }
        }
    }
}
