#![allow(clippy::unwrap_used)]
//! Property tests for utility function invariants.

use polyvoice::utils::{cosine_similarity, l2_normalize, mean_vector};
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
}
