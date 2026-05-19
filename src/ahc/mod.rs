//! Agglomerative Hierarchical Clustering (AHC) for speaker diarization.
//!
//! Bottom-up clustering: each embedding starts as its own cluster, then the
//! two most similar clusters are merged iteratively until no pair exceeds
//! the cosine similarity threshold.

use crate::utils::{cosine_similarity, l2_normalize};
use std::collections::HashMap;

/// { embeddings.is_empty() || embeddings.iter().all(|e| e.len() == embeddings`[0]`.len()) }
/// `pub fn agglomerative_cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize>`
/// { ret.len() == embeddings.len() && ret.iter().all(|&l| l < embeddings.len()) }
/// Run agglomerative hierarchical clustering on a set of embeddings.
///
/// Returns a label vector of the same length as `embeddings`, where each
/// element is the cluster index (0-based, contiguous) for that embedding.
///
/// `threshold` is the minimum cosine similarity to merge two clusters.
/// Higher threshold → more clusters (stricter merging).
pub fn agglomerative_cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    ahc_impl(embeddings, threshold, 0).0
}

/// Run AHC with a fixed threshold and a hard ceiling on the number of clusters.
pub fn agglomerative_cluster_max_clusters(
    embeddings: &[Vec<f32>],
    threshold: f32,
    max_clusters: usize,
) -> Vec<usize> {
    ahc_impl(embeddings, threshold, max_clusters).0
}

/// { embeddings.is_empty() || embeddings.iter().all(|e| e.len() == embeddings`[0]`.len()) }
/// `pub fn agglomerative_cluster_auto(embeddings: &[Vec<f32>]) -> (Vec<usize>, f32)`
/// { ret.0.len() == embeddings.len() && ret.0.iter().all(|&l| l < embeddings.len()) && ret.1 >= 0.0 }
/// Run AHC with automatic threshold selection via largest-merge-gap heuristic.
///
/// Returns labels and the automatically selected threshold.
pub fn agglomerative_cluster_auto(embeddings: &[Vec<f32>]) -> (Vec<usize>, f32) {
    agglomerative_cluster_auto_max_clusters(embeddings, 0)
}

/// Run AHC with automatic threshold selection and a hard ceiling on the number
/// of clusters.
pub fn agglomerative_cluster_auto_max_clusters(
    embeddings: &[Vec<f32>],
    max_clusters: usize,
) -> (Vec<usize>, f32) {
    let n = embeddings.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    let threshold = estimate_threshold_from_similarities(embeddings);
    ahc_impl(embeddings, threshold, max_clusters)
}

#[allow(clippy::needless_range_loop)]
fn ahc_impl(embeddings: &[Vec<f32>], threshold: f32, max_clusters: usize) -> (Vec<usize>, f32) {
    let n = embeddings.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }

    let mut labels: Vec<usize> = (0..n).collect();
    let mut centroids: Vec<Vec<f32>> = embeddings.to_vec();
    let mut cluster_sizes: Vec<usize> = vec![1; n];
    let mut active: Vec<bool> = vec![true; n];

    // Precompute similarity matrix. sim_matrix[i][j] holds the similarity
    // between centroids i and j. Inactive rows/columns are kept at NEG_INFINITY.
    let neg_inf = f32::NEG_INFINITY;
    let mut sim_matrix: Vec<Vec<f32>> = vec![vec![neg_inf; n]; n];
    for i in 0..n {
        sim_matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let sim = cosine_similarity(&centroids[i], &centroids[j]);
            sim_matrix[i][j] = sim;
            sim_matrix[j][i] = sim;
        }
    }

    loop {
        let mut best_sim = neg_inf;
        let mut best_i = 0;
        let mut best_j = 0;

        // Find the best pair among active clusters.
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                let sim = sim_matrix[i][j];
                if sim > best_sim {
                    best_sim = sim;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let active_count = active.iter().filter(|&&a| a).count();
        let above_ceiling = max_clusters > 0 && max_clusters < n && active_count > max_clusters;
        if !above_ceiling && best_sim < threshold {
            break;
        }
        if above_ceiling && best_sim == neg_inf {
            break;
        }

        // Merge j into i.
        let total = cluster_sizes[best_i] + cluster_sizes[best_j];
        let w_i = cluster_sizes[best_i] as f32 / total as f32;
        let w_j = cluster_sizes[best_j] as f32 / total as f32;
        let dim = centroids[best_i].len();
        let mut new_centroid = vec![0.0f32; dim];
        for k in 0..dim {
            new_centroid[k] = centroids[best_i][k] * w_i + centroids[best_j][k] * w_j;
        }
        l2_normalize(&mut new_centroid);

        centroids[best_i] = new_centroid;
        cluster_sizes[best_i] = total;
        active[best_j] = false;

        // Invalidate best_j from the similarity matrix.
        for k in 0..n {
            sim_matrix[best_j][k] = neg_inf;
            sim_matrix[k][best_j] = neg_inf;
        }

        // Update similarities for best_i against all other active clusters.
        for k in 0..n {
            if k == best_i || !active[k] {
                continue;
            }
            let sim = cosine_similarity(&centroids[best_i], &centroids[k]);
            sim_matrix[best_i][k] = sim;
            sim_matrix[k][best_i] = sim;
        }

        for label in &mut labels {
            if *label == best_j {
                *label = best_i;
            }
        }
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

    (labels, threshold)
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

    #[test]
    fn test_agglomerative_cluster_auto_max_clusters_caps_count() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        let (labels, _th) = agglomerative_cluster_auto_max_clusters(&embeddings, 2);
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(
            unique.len(),
            2,
            "max_clusters=2 must produce exactly 2 clusters"
        );
    }
}
