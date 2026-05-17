//! Property tests for k-means clustering invariants.

use polyvoice::kmeans::kmeans_pp;
use proptest::prelude::*;

fn embedding_vec() -> impl Strategy<Value = Vec<Vec<f32>>> {
    (4usize..=16, 1usize..=32).prop_flat_map(|(dim, n)| {
        prop::collection::vec(prop::collection::vec(-1.0f32..=1.0f32, dim..=dim), n..=n)
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        ..ProptestConfig::default()
    })]

    /// kmeans_pp labels are always in 0..k for non-empty input.
    #[test]
    fn labels_in_range(embeddings in embedding_vec(), k in 1usize..=8) {
        let labels = kmeans_pp(&embeddings, k, 20);
        prop_assert_eq!(labels.len(), embeddings.len());
        for &l in &labels {
            prop_assert!(
                l < k.min(embeddings.len()),
                "label {} must be < k={}, n={}",
                l, k, embeddings.len()
            );
        }
    }

    /// Empty embeddings produce empty labels regardless of k.
    #[test]
    fn empty_input_empty_output(k in 1usize..=8) {
        let labels = kmeans_pp(&[], k, 20);
        prop_assert!(labels.is_empty());
    }

    /// k=1 assigns all points to label 0.
    #[test]
    fn k_one_all_zero(embeddings in embedding_vec()) {
        let labels = kmeans_pp(&embeddings, 1, 20);
        prop_assert!(labels.iter().all(|&l| l == 0));
    }
}
