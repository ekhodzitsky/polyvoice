//! Property tests for core math utilities.
//!
//! Verified invariants:
//! - cosine_similarity ∈ [-1, 1]
//! - l2_normalize produces unit-norm vectors (or zero vector)
//! - mean_vector length equals input dimension

use polyvoice::utils::{cosine_similarity, l2_normalize, mean_vector};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        ..ProptestConfig::default()
    })]

    /// Cosine similarity of any two finite f32 vectors is in [-1, 1].
    #[test]
    fn cosine_similarity_range(
        (len, a, b) in (1usize..=256)
            .prop_flat_map(|len| {
                (
                    Just(len),
                    prop::collection::vec(-10.0f32..=10.0f32, len),
                    prop::collection::vec(-10.0f32..=10.0f32, len),
                )
            }),
    ) {
        let _ = len;
        let sim = cosine_similarity(&a, &b);
        prop_assert!(
            sim >= -1.0 - 1e-6 && sim <= 1.0 + 1e-6,
            "cosine_similarity out of range: {}",
            sim
        );
    }

    /// L2-normalized vector has norm 1 (or 0 if input was zero).
    #[test]
    fn l2_normalize_produces_unit_norm(
        mut v in prop::collection::vec(-10.0f32..=10.0f32, 1..=256),
    ) {
        let input_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        l2_normalize(&mut v);
        let computed: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if input_norm > 1e-6 {
            prop_assert!(
                (computed - 1.0).abs() < 1e-4,
                "norm after l2_normalize should be 1, got {}",
                computed
            );
        } else {
            prop_assert!(
                computed.abs() < 1e-6,
                "zero input should produce zero output, got norm {}",
                computed
            );
        }
    }

    /// Mean vector has the same dimension as inputs and components are averages.
    #[test]
    fn mean_vector_preserves_dimension(
        (dim, vectors) in (1usize..=64)
            .prop_flat_map(|dim| {
                prop::collection::vec(
                    prop::collection::vec(-10.0f32..=10.0f32, dim),
                    1..=16,
                )
                .prop_map(move |v| (dim, v))
            }),
    ) {
        let mean = mean_vector(&vectors).expect("non-empty input");
        prop_assert_eq!(mean.len(), dim, "mean vector dimension mismatch");
        for (i, &m) in mean.iter().enumerate() {
            let expected: f32 =
                vectors.iter().map(|v| v[i]).sum::<f32>() / vectors.len() as f32;
            prop_assert!(
                (m - expected).abs() < 1e-4,
                "mean[{}] = {} != expected {}",
                i,
                m,
                expected
            );
        }
    }
}
