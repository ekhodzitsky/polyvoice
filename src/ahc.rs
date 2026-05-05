//! Agglomerative Hierarchical Clustering (AHC) for speaker diarization.
//!
//! Bottom-up clustering: each embedding starts as its own cluster, then the
//! two most similar clusters are merged iteratively until no pair exceeds
//! the cosine similarity threshold.

use crate::utils::{cosine_similarity, l2_normalize};

/// Run agglomerative hierarchical clustering on a set of embeddings.
///
/// Returns a label vector of the same length as `embeddings`, where each
/// element is the cluster index (0-based, contiguous) for that embedding.
///
/// `threshold` is the minimum cosine similarity to merge two clusters.
/// Higher threshold → more clusters (stricter merging).
pub fn agglomerative_cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }

    let mut labels: Vec<usize> = (0..n).collect();
    let mut centroids: Vec<Vec<f32>> = embeddings.to_vec();
    let mut cluster_sizes: Vec<usize> = vec![1; n];
    let mut active: Vec<bool> = vec![true; n];

    loop {
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;

        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                let sim = cosine_similarity(&centroids[i], &centroids[j]);
                if sim > best_sim {
                    best_sim = sim;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_sim < threshold {
            break;
        }

        // Merge j into i.
        let total = cluster_sizes[best_i] + cluster_sizes[best_j];
        let w_i = cluster_sizes[best_i] as f32 / total as f32;
        let w_j = cluster_sizes[best_j] as f32 / total as f32;
        let mut new_centroid = vec![0.0f32; centroids[best_i].len()];
        for (k, v) in new_centroid.iter_mut().enumerate() {
            *v = centroids[best_i][k] * w_i + centroids[best_j][k] * w_j;
        }
        l2_normalize(&mut new_centroid);

        centroids[best_i] = new_centroid;
        cluster_sizes[best_i] = total;
        active[best_j] = false;

        for label in &mut labels {
            if *label == best_j {
                *label = best_i;
            }
        }
    }

    // Make labels contiguous (0, 1, 2, ...).
    let mut label_map = std::collections::HashMap::new();
    let mut next_label = 0usize;
    for label in &mut labels {
        let entry = label_map.entry(*label).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        *label = *entry;
    }

    labels
}
