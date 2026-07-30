#![allow(clippy::unwrap_used)]
//! Property tests for Agglomerative Hierarchical Clustering.
//!
//! Verified invariants:
//! - agglomerative_cluster never panics on finite inputs
//! - output length matches input length
//! - labels are contiguous starting at 0
//! - number of clusters ≤ number of embeddings

use polyvoice::ahc::agglomerative_cluster;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        ..ProptestConfig::default()
    })]

    /// AHC on random embeddings never panics and produces valid labels.
    #[test]
    fn ahc_random_embeddings_never_panics(
        (count, dim, threshold, embeddings) in (1usize..=32, 1usize..=16, 0.0f32..=1.0f32)
            .prop_flat_map(|(count, dim, threshold)| {
                prop::collection::vec(
                    prop::collection::vec(-1.0f32..=1.0f32, dim),
                    count,
                )
                .prop_map(move |embeddings| (count, dim, threshold, embeddings))
            }),
    ) {
        let _ = (count, dim, threshold);
        let labels = agglomerative_cluster(&embeddings, threshold);
        prop_assert_eq!(
            labels.len(),
            count,
            "label count must match embedding count"
        );
        if !labels.is_empty() {
            let max_label = *labels.iter().max().unwrap();
            let num_clusters = max_label + 1;
            prop_assert!(
                num_clusters <= count,
                "clusters {} > embeddings {}",
                num_clusters,
                count
            );
            // Labels must be contiguous 0..num_clusters
            let unique: std::collections::HashSet<usize> =
                labels.iter().copied().collect();
            prop_assert_eq!(
                unique.len(),
                num_clusters,
                "labels must be contiguous"
            );
        }
    }

    /// AHC auto mode never panics and returns a valid threshold.
    #[test]
    fn ahc_auto_mode_never_panics(
        (count, dim, embeddings) in (1usize..=16, 1usize..=16)
            .prop_flat_map(|(count, dim)| {
                prop::collection::vec(
                    prop::collection::vec(-1.0f32..=1.0f32, dim),
                    count,
                )
                .prop_map(move |embeddings| (count, dim, embeddings))
            }),
    ) {
        let _ = (count, dim);
        let (labels, threshold) =
            polyvoice::ahc::agglomerative_cluster_auto_max_clusters(&embeddings, 0);
        prop_assert_eq!(labels.len(), count);
        prop_assert!((0.0..=1.0).contains(&threshold),
            "auto threshold must be in [0, 1], got {}", threshold);
    }
}
