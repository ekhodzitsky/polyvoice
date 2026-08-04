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
    if k == 0 {
        // Defensive fallback: caller asked for zero clusters, treat as one.
        return vec![0; n];
    }
    let dim = embeddings[0].len();
    if !embeddings.iter().all(|e| e.len() == dim) {
        // Defensive fallback: mixed dimensions would break distance math.
        // Return a single cluster rather than panicking.
        return vec![0; n];
    }
    let k = k.min(n);

    // K-means++ initialization.
    // PRNG: in-tree xorshift64* (not fastrand). Seed inputs are unchanged;
    // the draw sequence differs from fastrand, so exact labels on a fixed seed
    // may shift — quality tests and silhouette selection still apply.
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut rng = crate::utils::XorShift64Star::new(seed);
    let first_idx = rng.usize(n);
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
        // Degenerate-seeding guard: when every
        // remaining point already sits on a chosen centroid (collapsed /
        // homogeneous embeddings) the cumulative-sum sampler would fall through
        // and pick duplicate centroids; a non-finite total would do the same.
        // Stop seeding and let Lloyd's run on the centroids gathered so far.
        if !total.is_finite() || total <= 0.0 {
            break;
        }
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
    let sims = crate::utils::pairwise_cosine_similarity_matrix(embeddings);
    let mut dists = vec![0.0f64; n * n];
    let mut total_dist = 0.0;
    let mut dist_count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = 1.0 - sims[i * n + j];
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

    #[test]
    fn k_zero_returns_single_cluster() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let labels = kmeans_pp(&embeddings, 0, 10);
        assert_eq!(labels, vec![0, 0]);
    }

    #[test]
    fn mismatched_dimensions_returns_single_cluster() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0, 0.0]];
        let labels = kmeans_pp(&embeddings, 2, 10);
        assert_eq!(labels, vec![0, 0]);
    }

    #[test]
    fn identical_embeddings_yield_single_cluster() {
        // All embeddings collapse to one point: the degenerate-seeding
        // guard must keep this to a single effective cluster, with no panic.
        let embeddings: Vec<Vec<f32>> = vec![vec![1.0, 0.0]; 12];
        let labels = kmeans_pp(&embeddings, 4, 20);
        assert_eq!(labels.len(), 12);
        let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            1,
            "identical embeddings must form one cluster"
        );
    }

    #[test]
    fn fewer_distinct_points_than_k_caps_effective_clusters() {
        // Only two distinct values but k=4 requested: the guard prevents
        // duplicate centroids, so exactly two effective clusters result.
        let mut embeddings: Vec<Vec<f32>> = vec![vec![1.0, 0.0]; 6];
        embeddings.extend(vec![vec![0.0, 1.0]; 6]);
        let labels = kmeans_pp(&embeddings, 4, 20);
        assert_eq!(labels.len(), 12);
        let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            2,
            "two distinct points must form two clusters"
        );
    }

    #[test]
    fn seeding_is_deterministic() {
        // Two calls with the same implicit seed must yield identical labels.
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
            vec![-1.0, 0.0],
            vec![-0.9, 0.1],
            vec![0.5, 0.5],
            vec![-0.5, 0.5],
        ];
        let first = kmeans_pp(&embeddings, 3, 20);
        let second = kmeans_pp(&embeddings, 3, 20);
        assert_eq!(first, second);
    }

    #[test]
    fn explicit_seeds_produce_valid_groupings() {
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        for seed in [0u64, 1, 7, 123_456] {
            let labels = kmeans_pp_with_seed(&embeddings, 2, 20, seed);
            assert_eq!(labels.len(), embeddings.len());
            for &l in &labels {
                assert!(l < 2);
            }
            assert_eq!(labels[0], labels[1]);
            assert_eq!(labels[2], labels[3]);
        }
    }

    #[test]
    fn zero_max_iter_returns_valid_labels_without_panic() {
        // Lloyd's loop never runs; labels stay at their initial value.
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let labels = kmeans_pp(&embeddings, 2, 0);
        assert_eq!(labels.len(), 3);
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn converges_before_max_iter_on_separated_data() {
        // Trivially separated points stabilize almost immediately; running
        // many more iterations must not change the labels.
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.95, 0.05],
            vec![0.0, 1.0],
            vec![0.05, 0.95],
        ];
        let short = kmeans_pp(&embeddings, 2, 2);
        let long = kmeans_pp(&embeddings, 2, 100);
        assert_eq!(short, long);
        assert_eq!(short[0], short[1]);
        assert_eq!(short[2], short[3]);
        assert_ne!(short[0], short[2]);
    }

    #[test]
    fn k_equal_to_n_gives_each_point_its_own_cluster() {
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        let labels = kmeans_pp(&embeddings, 4, 20);
        assert_eq!(labels.len(), 4);
        let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(distinct.len(), 4);
    }

    #[test]
    fn cosine_distance_behaves_on_known_geometry() {
        let a = vec![1.0f32, 0.0];
        let same = vec![1.0f64, 0.0];
        let orth = vec![0.0f64, 1.0];
        let opposite = vec![-1.0f64, 0.0];
        assert!(cosine_distance_f32_f64(&a, &same).abs() < 1e-6);
        assert!((cosine_distance_f32_f64(&a, &orth) - 1.0).abs() < 1e-6);
        assert!((cosine_distance_f32_f64(&a, &opposite) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn silhouette_tiny_input_scores_zero() {
        // n <= 2 short-circuits to 0 regardless of labels.
        let dists = vec![0.0f64; 4];
        assert_eq!(silhouette_score_with_dists(0, &[], &dists), 0.0);
        assert_eq!(silhouette_score_with_dists(1, &[0], &dists), 0.0);
        assert_eq!(silhouette_score_with_dists(2, &[0, 1], &dists), 0.0);
    }

    #[test]
    fn silhouette_single_cluster_scores_zero() {
        // All points in one cluster: k < 2 short-circuits to 0.
        let n = 4;
        let dists = vec![0.5f64; n * n];
        let labels = vec![0usize; n];
        assert_eq!(silhouette_score_with_dists(n, &labels, &dists), 0.0);
    }

    #[test]
    fn silhouette_skips_singleton_clusters() {
        // Point 0 is a singleton cluster: it has no intra-cluster neighbors
        // and must be skipped rather than counted with a bogus score.
        let n = 3;
        let mut dists = vec![0.0f64; n * n];
        // d(1,2) small (same cluster), d(0,*) large.
        dists[n + 2] = 0.1;
        dists[2 * n + 1] = 0.1;
        dists[1] = 1.0;
        dists[n] = 1.0;
        dists[2] = 1.0;
        dists[2 * n] = 1.0;
        let labels = vec![0usize, 1, 1];
        let score = silhouette_score_with_dists(n, &labels, &dists);
        // Only points 1 and 2 count; both have a=0.1, b=1.0 → s = 0.9.
        assert!((score - 0.9).abs() < 1e-9, "score was {score}");
    }

    #[test]
    fn silhouette_perfect_separation_scores_near_one() {
        let n = 4;
        let mut dists = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let same_cluster = (i < 2) == (j < 2);
                    dists[i * n + j] = if same_cluster { 0.01 } else { 1.0 };
                }
            }
        }
        let labels = vec![0usize, 0, 1, 1];
        let score = silhouette_score_with_dists(n, &labels, &dists);
        assert!(score > 0.9, "score was {score}");
        assert!(score <= 1.0);
    }

    #[test]
    fn auto_k_empty_input_returns_empty() {
        let labels = kmeans_auto_k(&[], 2, 5, 10, 3);
        assert!(labels.is_empty());
    }

    #[test]
    fn auto_k_single_point_returns_single_label() {
        let embeddings = vec![vec![1.0f32, 0.0]];
        let labels = kmeans_auto_k(&embeddings, 2, 5, 10, 3);
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn auto_k_homogeneous_embeddings_force_single_cluster() {
        // Nearly identical embeddings fall below the homogeneity threshold,
        // so the k search is skipped and a single cluster is returned.
        let embeddings: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                let mut v = vec![1.0f32, 0.0];
                v[1] = i as f32 * 1e-4;
                crate::utils::l2_normalize(&mut v);
                v
            })
            .collect();
        let labels = kmeans_auto_k(&embeddings, 2, 6, 20, 3);
        assert_eq!(labels, vec![0; 8]);
    }

    #[test]
    fn auto_k_finds_two_well_separated_clusters() {
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.95, 0.05],
            vec![0.9, 0.1],
            vec![1.0, 0.02],
            vec![0.0, 1.0],
            vec![0.05, 0.95],
            vec![0.1, 0.9],
            vec![0.02, 1.0],
        ];
        let labels = kmeans_auto_k(&embeddings, 2, 4, 20, 3);
        assert_eq!(labels.len(), 8);
        let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(distinct.len(), 2, "expected two clusters, got {labels:?}");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn auto_k_zero_trials_still_runs_once() {
        // trials=0 is clamped to a single run per k.
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let labels = kmeans_auto_k(&embeddings, 2, 2, 20, 0);
        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn auto_k_clamps_k_range_to_point_count() {
        // k_min/k_max beyond n must not panic or produce invalid labels.
        let embeddings: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let labels = kmeans_auto_k(&embeddings, 10, 20, 10, 2);
        assert_eq!(labels.len(), 3);
        for &l in &labels {
            assert!(l < 3);
        }
    }

    #[test]
    fn auto_k_k_min_below_two_is_raised() {
        // k_min below 2 is clamped up; the search still returns valid labels.
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let labels = kmeans_auto_k(&embeddings, 0, 2, 20, 2);
        assert_eq!(labels.len(), 4);
        for &l in &labels {
            assert!(l < 2);
        }
    }

    #[test]
    fn auto_k_is_deterministic() {
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
            vec![-1.0, 0.0],
            vec![-0.9, 0.1],
        ];
        let first = kmeans_auto_k(&embeddings, 2, 3, 20, 3);
        let second = kmeans_auto_k(&embeddings, 2, 3, 20, 3);
        assert_eq!(first, second);
    }
}
