//! Agglomerative Hierarchical Clustering (AHC) for speaker diarization.
//!
//! Bottom-up clustering: each embedding starts as its own cluster, then the
//! two most similar clusters are merged iteratively until no pair exceeds
//! the cosine similarity threshold.

use crate::utils::{cosine_similarity, l2_normalize};
use std::collections::HashMap;

/// Run agglomerative hierarchical clustering on a set of embeddings.
///
/// Returns a label vector of the same length as `embeddings`, where each
/// element is the cluster index (0-based, contiguous) for that embedding.
///
/// `threshold` is the minimum cosine similarity to merge two clusters.
/// Higher threshold → more clusters (stricter merging).
pub fn agglomerative_cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    ahc_impl(embeddings, Some(threshold)).0
}

/// Run AHC with automatic threshold selection via largest-merge-gap heuristic.
///
/// Returns labels and the automatically selected threshold.
pub fn agglomerative_cluster_auto(embeddings: &[Vec<f32>]) -> (Vec<usize>, f32) {
    ahc_impl(embeddings, None)
}

fn ahc_impl(embeddings: &[Vec<f32>], threshold: Option<f32>) -> (Vec<usize>, f32) {
    let n = embeddings.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }

    let mut labels: Vec<usize> = (0..n).collect();
    let mut centroids: Vec<Vec<f32>> = embeddings.to_vec();
    let mut cluster_sizes: Vec<usize> = vec![1; n];
    let mut active: Vec<bool> = vec![true; n];
    let mut merge_history: Vec<f32> = Vec::with_capacity(n.saturating_sub(1));

    loop {
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;

        let active_indices: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
        for (ii, &i) in active_indices.iter().enumerate() {
            for &j in &active_indices[ii + 1..] {
                let sim = cosine_similarity(&centroids[i], &centroids[j]);
                if sim > best_sim {
                    best_sim = sim;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if let Some(th) = threshold {
            if best_sim < th {
                break;
            }
        } else if active.iter().filter(|&&a| a).count() <= 1 {
            // Only one cluster left in auto mode.
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

        merge_history.push(best_sim);

        // In auto mode, stop when only one cluster remains.
        if threshold.is_none() && active_indices.len() <= 1 {
            break;
        }
    }

    // If auto mode, estimate threshold from pairwise similarity distribution.
    let effective_threshold = match threshold {
        Some(th) => th,
        None => estimate_threshold_from_similarities(embeddings),
    };

    // If auto mode, re-run with the estimated threshold for clean labels.
    if threshold.is_none() {
        return ahc_impl(embeddings, Some(effective_threshold));
    }

    // Make labels contiguous (0, 1, 2, ...).
    let mut label_map = HashMap::new();
    let mut next_label = 0usize;
    for label in &mut labels {
        let entry = label_map.entry(*label).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        *label = *entry;
    }

    (labels, effective_threshold)
}

/// Estimate a good AHC threshold from the distribution of pairwise similarities.
///
/// Computes all pairwise cosine similarities, sorts them, and finds the largest
/// gap in the lower half of the distribution (between 0.0 and median).
/// This tends to separate within-speaker from between-speaker pairs.
fn estimate_threshold_from_similarities(embeddings: &[Vec<f32>]) -> f32 {
    let n = embeddings.len();
    if n < 2 {
        return 0.5;
    }

    let mut sims: Vec<f32> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            sims.push(cosine_similarity(&embeddings[i], &embeddings[j]));
        }
    }
    sims.sort_by(|a, b| a.total_cmp(b));

    if sims.is_empty() {
        return 0.5;
    }

    let median_idx = sims.len() / 2;
    // Search for the largest gap in the range [0.0, median].
    let mut best_gap = 0.0f32;
    let mut best_idx = 0usize;
    for i in 0..median_idx.saturating_sub(1) {
        let gap = sims[i + 1] - sims[i];
        if gap > best_gap {
            best_gap = gap;
            best_idx = i;
        }
    }

    // Threshold is the similarity value after the gap.
    if sims.len() <= 1 {
        return 0.5;
    }
    let th = sims[best_idx + 1];
    // Clamp to a reasonable range.
    th.clamp(0.2, 0.7)
}

/// Refine cluster assignments by reassigning each embedding to the nearest centroid.
///
/// After AHC, some embeddings may be misassigned due to greedy merging.
/// This function recomputes centroids and reassigns embeddings in a k-means-like
/// refinement step (single iteration).  Returns a new label vector.
pub fn refine_clusters(embeddings: &[Vec<f32>], labels: &[usize]) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }

    let num_clusters = labels.iter().copied().max().unwrap_or(0) + 1;
    let dim = embeddings[0].len();

    // Compute centroids.
    let mut centroids = vec![vec![0.0f32; dim]; num_clusters];
    let mut counts = vec![0usize; num_clusters];
    for (i, emb) in embeddings.iter().enumerate() {
        let c = labels[i];
        for (centroid, &v) in centroids[c].iter_mut().zip(emb.iter()) {
            *centroid += v;
        }
        counts[c] += 1;
    }
    for (c, centroid) in centroids.iter_mut().enumerate() {
        if counts[c] > 0 {
            for v in centroid.iter_mut() {
                *v /= counts[c] as f32;
            }
            l2_normalize(centroid);
        }
    }

    // Reassign each embedding to the nearest centroid.
    let mut new_labels = vec![0usize; n];
    for (i, emb) in embeddings.iter().enumerate() {
        let mut best = 0usize;
        let mut best_sim = f32::NEG_INFINITY;
        for (c_idx, centroid) in centroids.iter().enumerate() {
            let sim = cosine_similarity(emb, centroid);
            if sim > best_sim {
                best_sim = sim;
                best = c_idx;
            }
        }
        new_labels[i] = best;
    }

    new_labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agglomerative_cluster_basic() {
        // Two clear clusters.
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        let labels = agglomerative_cluster(&embeddings, 0.5);
        assert_eq!(labels.len(), 4);
        assert_eq!(labels.iter().copied().max(), Some(1));
        // First two should be same cluster, last two should be same cluster.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_agglomerative_cluster_empty() {
        let labels = agglomerative_cluster(&[], 0.5);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_agglomerative_cluster_single() {
        let embeddings = vec![vec![1.0, 0.0, 0.0]];
        let labels = agglomerative_cluster(&embeddings, 0.5);
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn test_agglomerative_cluster_auto_basic() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        let (labels, th) = agglomerative_cluster_auto(&embeddings);
        assert_eq!(labels.len(), 4);
        assert!((0.2..=0.7).contains(&th), "threshold {} out of range", th);
    }
}
