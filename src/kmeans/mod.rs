//! K-Means++ clustering with automatic k selection via silhouette score.

/// { embeddings.is_empty() || embeddings.iter().all(|e| e.len() == embeddings`[0]`.len()) }
/// `pub fn kmeans_pp(embeddings: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<usize>`
/// { ret.len() == embeddings.len() }
/// K-means++ initialization + Lloyd's algorithm with deterministic seed.
pub fn kmeans_pp(embeddings: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<usize> {
    kmeans_pp_with_seed(embeddings, k, max_iter, 42)
}

fn kmeans_pp_with_seed(
    embeddings: &[Vec<f32>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let dim = embeddings[0].len();
    debug_assert!(
        embeddings.iter().all(|e| e.len() == dim),
        "kmeans_pp precondition violated: all embeddings must have the same dimension"
    );

    // K-means++ initialization.
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut rng = fastrand::Rng::with_seed(seed);
    let first_idx = rng.usize(0..n);
    centroids.push(embeddings[first_idx].iter().map(|&v| v as f64).collect());

    let mut dists = vec![f64::INFINITY; n];
    for _ in 1..k {
        for (i, emb) in embeddings.iter().enumerate() {
            let d = cosine_distance_f32_f64(emb, &centroids[centroids.len() - 1]);
            if d < dists[i] {
                dists[i] = d;
            }
        }
        let total: f64 = dists.iter().sum();
        let target = rng.f64() * total;
        let mut cumsum = 0.0;
        let mut chosen = 0;
        for (i, &d) in dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= target {
                chosen = i;
                break;
            }
        }
        centroids.push(embeddings[chosen].iter().map(|&v| v as f64).collect());
    }

    // Lloyd's algorithm.
    let mut labels = vec![0usize; n];
    for _ in 0..max_iter {
        let mut changed = false;
        // Assign.
        for (i, emb) in embeddings.iter().enumerate() {
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for (c_idx, c) in centroids.iter().enumerate() {
                let dist = cosine_distance_f32_f64(emb, c);
                if dist < best_dist {
                    best_dist = dist;
                    best = c_idx;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update.
        let mut new_centroids = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, emb) in embeddings.iter().enumerate() {
            let c = labels[i];
            for (d, &v) in emb.iter().enumerate() {
                new_centroids[c][d] += v as f64;
            }
            counts[c] += 1;
        }
        for (c, new_centroid) in new_centroids.iter_mut().enumerate().take(k) {
            if counts[c] > 0 {
                for v in new_centroid.iter_mut().take(dim) {
                    *v /= counts[c] as f64;
                }
            }
        }
        centroids = new_centroids;
    }

    labels
}

fn cosine_distance_f32_f64(a: &[f32], b: &[f64]) -> f64 {
    let sim = crate::utils::cosine_similarity_f32_f64(a, b);
    (1.0 - sim).max(0.0) as f64
}

/// Compute the average silhouette score for a clustering using a precomputed
/// pairwise distance matrix.
/// Higher is better (range: -1 to 1).
fn silhouette_score_with_dists(n: usize, labels: &[usize], dists: &[f64]) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let k = labels.iter().copied().max().unwrap_or(0) + 1;
    if k < 2 {
        return 0.0;
    }

    let mut total = 0.0f64;
    for i in 0..n {
        let label_i = labels[i];
        let mut a = 0.0f64;
        let mut a_count = 0usize;
        let mut b = vec![0.0f64; k];
        let mut b_count = vec![0usize; k];
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = dists[i * n + j];
            if labels[j] == label_i {
                a += d;
                a_count += 1;
            } else {
                b[labels[j]] += d;
                b_count[labels[j]] += 1;
            }
        }
        let a_avg = if a_count > 0 { a / a_count as f64 } else { 0.0 };
        let b_min = b
            .iter()
            .zip(b_count.iter())
            .filter(|(_, c)| **c > 0)
            .map(|(sum, c)| sum / *c as f64)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let s = if a_avg == 0.0 && b_min == 0.0 {
            0.0
        } else {
            (b_min - a_avg) / a_avg.max(b_min)
        };
        total += s;
    }
    total / n as f64
}

/// K-means++ with automatic k selection via silhouette score.
/// Searches k in [k_min, k_max] and returns the best clustering.
///
/// * `embeddings` — input vectors (assumed L2-normalized).
/// * `k_min` — minimum number of clusters (at least 2).
/// * `k_max` — maximum number of clusters.
/// * `max_iter` — Lloyd iterations per k.
pub fn kmeans_auto_k(
    embeddings: &[Vec<f32>],
    k_min: usize,
    k_max: usize,
    max_iter: usize,
) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    let k_min = k_min.max(2).min(n);
    let k_max = k_max.max(k_min).min(n);

    // Precompute pairwise cosine distances once.
    let mut dists = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = 1.0 - crate::utils::cosine_similarity(&embeddings[i], &embeddings[j]);
            let d = d.max(0.0) as f64;
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }

    // Run multiple trials per k with different seeds and pick the best by
    // average silhouette score (most stable metric).
    const TRIALS: usize = 3;

    let mut best_labels = vec![0usize; n];
    let mut best_score = f64::NEG_INFINITY;

    for k in k_min..=k_max {
        let mut trial_best_score = f64::NEG_INFINITY;
        let mut trial_best_labels = vec![0usize; n];
        for t in 0..TRIALS {
            let labels = kmeans_pp_with_seed(embeddings, k, max_iter, 42 + t as u64);
            let score = silhouette_score_with_dists(n, &labels, &dists);
            if score > trial_best_score {
                trial_best_score = score;
                trial_best_labels = labels;
            }
        }
        if trial_best_score > best_score {
            best_score = trial_best_score;
            best_labels = trial_best_labels;
        }
    }

    best_labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_empty() {
        let labels = kmeans_pp(&[], 3, 10);
        assert!(labels.is_empty());
    }

    #[test]
    fn well_separated_clusters() {
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
            vec![-1.0, 0.0],
            vec![-0.9, 0.1],
        ];
        let labels = kmeans_pp(&embeddings, 3, 20);
        assert_eq!(labels.len(), 6);
        for &l in &labels {
            assert!(l < 3);
        }
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_eq!(labels[4], labels[5]);
    }

    #[test]
    fn k_larger_than_n() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let labels = kmeans_pp(&embeddings, 10, 10);
        assert_eq!(labels.len(), 2);
        for &l in &labels {
            assert!(l < 2);
        }
    }
}
