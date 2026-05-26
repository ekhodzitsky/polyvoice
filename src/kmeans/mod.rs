//! K-Means++ clustering with automatic k selection via silhouette score.

/// K-means++ result: labels + centroids + WCSS.
pub struct KmeansResult {
    pub labels: Vec<usize>,
    pub centroids: Vec<Vec<f64>>,
    pub wcss: f64,
}

/// { embeddings.is_empty() || embeddings.iter().all(|e| e.len() == embeddings`[0]`.len()) }
/// `pub fn kmeans_pp(embeddings: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<usize>`
/// { ret.len() == embeddings.len() }
/// K-means++ initialization + Lloyd's algorithm with deterministic seed.
pub fn kmeans_pp(embeddings: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<usize> {
    kmeans_pp_with_seed(embeddings, k, max_iter, 42).labels
}

fn kmeans_pp_with_seed(
    embeddings: &[Vec<f32>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> KmeansResult {
    let n = embeddings.len();
    if n == 0 {
        return KmeansResult {
            labels: Vec::new(),
            centroids: Vec::new(),
            wcss: 0.0,
        };
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

    // Compute WCSS.
    let wcss: f64 = embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| {
            let d = cosine_distance_f32_f64(emb, &centroids[labels[i]]);
            d * d
        })
        .sum();

    KmeansResult {
        labels,
        centroids,
        wcss,
    }
}

fn cosine_distance_f32_f64(a: &[f32], b: &[f64]) -> f64 {
    let sim = crate::utils::cosine_similarity_f32_f64(a, b);
    (1.0 - sim).max(0.0) as f64
}

/// Compute average silhouette score using precomputed pairwise distances.
/// O(n²) per call, but uses cached dists matrix.
fn silhouette_score_with_dists(n: usize, labels: &[usize], dists: &[f64]) -> f64 {
    if n <= 2 {
        return 0.0;
    }
    let k = *labels.iter().max().unwrap_or(&0) + 1;
    if k < 2 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut count = 0;
    for i in 0..n {
        let label = labels[i];
        // a(i): average distance to same cluster.
        let mut a = 0.0;
        let mut a_count = 0usize;
        for j in 0..n {
            if i != j && labels[j] == label {
                a += dists[i * n + j];
                a_count += 1;
            }
        }
        if a_count == 0 {
            continue;
        }
        a /= a_count as f64;

        // b(i): minimum average distance to other clusters.
        let mut b = f64::INFINITY;
        for c in 0..k {
            if c == label {
                continue;
            }
            let mut b_c = 0.0;
            let mut b_c_count = 0usize;
            for j in 0..n {
                if labels[j] == c {
                    b_c += dists[i * n + j];
                    b_c_count += 1;
                }
            }
            if b_c_count == 0 {
                continue;
            }
            b_c /= b_c_count as f64;
            if b_c < b {
                b = b_c;
            }
        }
        if b.is_infinite() {
            continue;
        }
        let s = (b - a) / a.max(b);
        total += s;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// K-means++ with automatic k selection via silhouette score.
/// Searches k in [k_min, k_max], runs `trials` per k, picks best by silhouette.
///
/// * `embeddings` — input vectors (assumed L2-normalized).
/// * `k_min` — minimum number of clusters (at least 2).
/// * `k_max` — maximum number of clusters.
/// * `max_iter` — Lloyd iterations for the final accurate run.
/// * `trials` — number of random initializations for the final run.
pub fn kmeans_auto_k(
    embeddings: &[Vec<f32>],
    k_min: usize,
    k_max: usize,
    max_iter: usize,
    trials: usize,
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

    // Precompute pairwise cosine distances once for silhouette.
    let mut dists = vec![0.0f64; n * n];
    let mut total_dist = 0.0;
    let mut dist_count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = 1.0 - crate::utils::cosine_similarity(&embeddings[i], &embeddings[j]);
            let d = d.max(0.0) as f64;
            dists[i * n + j] = d;
            dists[j * n + i] = d;
            total_dist += d;
            dist_count += 1;
        }
    }

    // Single-speaker detection: if embeddings are very homogeneous, force k=1.
    const HOMOGENEITY_THRESHOLD: f64 = 0.15;
    if dist_count > 0 && (total_dist / dist_count as f64) < HOMOGENEITY_THRESHOLD {
        return vec![0; n];
    }

    // Search all k, running multiple trials per k. Pick best by silhouette.
    let mut best_labels = vec![0usize; n];
    let mut best_score = f64::NEG_INFINITY;
    for k in k_min..=k_max {
        let mut trial_best_score = f64::NEG_INFINITY;
        let mut trial_best_labels = vec![0usize; n];
        for t in 0..trials.max(1) {
            let result = kmeans_pp_with_seed(embeddings, k, max_iter, 42 + t as u64);
            let score = silhouette_score_with_dists(n, &result.labels, &dists);
            if score > trial_best_score {
                trial_best_score = score;
                trial_best_labels = result.labels;
            }
        }
        if trial_best_score > best_score {
            best_score = trial_best_score;
            best_labels = trial_best_labels;
        }
    }

    best_labels
}

#[allow(clippy::unwrap_used)]
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
