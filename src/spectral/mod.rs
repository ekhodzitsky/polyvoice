//! Spectral clustering for speaker diarization.
//!
//! Uses normalized graph Laplacian + k-means on eigenvectors.
//! Auto-selects k via eigengap heuristic.
//!
//! Shared graph construction ([`SpectralGraph`]) is the single path used by
//! [`spectral_cluster`] (BIC k-selection) and
//! `crate::clusterer::NmeScClusterer` (pure eigengap k).

use crate::utils::cosine_similarity;
use faer::Side;
use faer::prelude::*;

/// Select the number of clusters via the NME-SC normalized-maximum eigengap
/// heuristic (Park et al., "Auto-Tuning Spectral Clustering for Speaker
/// Diarization Using Normalized Maximum Eigengap", 2020): pick the `k` in
/// `1..max_k` that maximizes `(eig_asc[k] - eig_asc[k-1]) / |eig_asc[k]|`, where
/// `eig_asc` are the normalized-Laplacian eigenvalues sorted ascending. A large
/// normalized gap at position `k` means the first `k` eigenvalues are the
/// near-zero "cluster" eigenvalues, hence `k` clusters. Returns `>= 1`.
///
/// This is the single source of truth for the eigengap convention shared by
/// [`spectral_cluster`] (where it only seeds a BIC search) and
/// `crate::clusterer::NmeScClusterer` (where it drives `k` directly), so the two
/// paths cannot silently diverge.
pub(crate) fn select_k_by_normalized_eigengap(eig_asc: &[f64], max_k: usize) -> usize {
    let max_k = max_k.min(eig_asc.len()).min(20);
    let mut best_k = 1usize;
    let mut best_gap = 0.0f64;
    for k in 1..max_k {
        let lam_k = eig_asc[k - 1];
        let lam_k1 = eig_asc[k];
        let gap = if lam_k1.abs() > 1e-10 {
            (lam_k1 - lam_k) / lam_k1.abs()
        } else {
            0.0
        };
        if gap > best_gap {
            best_gap = gap;
            best_k = k;
        }
    }
    best_k
}

/// k-NN cosine affinity → normalized Laplacian → sorted eigenspectrum.
///
/// Shared by [`spectral_cluster`] and `NmeScClusterer` so graph construction
/// cannot silently diverge.
pub(crate) struct SpectralGraph {
    n: usize,
    /// Eigenvalues ascending with original eigenvector column index.
    eig_pairs: Vec<(f64, usize)>,
    /// Eigenvector matrix `U` from the self-adjoint decomposition (`u[(row, col)]`).
    u: Mat<f64>,
}

impl SpectralGraph {
    /// Build the graph from L2-ish embeddings. Returns `None` if the Laplacian
    /// eigendecomposition fails (caller should fall back to a single cluster).
    pub(crate) fn from_embeddings(embeddings: &[Vec<f32>]) -> Option<Self> {
        let n = embeddings.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            // Degenerate: one point — trivial spectrum not needed by callers
            // that special-case n < 2, but keep constructible.
            let mut u = Mat::zeros(1, 1);
            u[(0, 0)] = 1.0;
            return Some(Self {
                n: 1,
                eig_pairs: vec![(0.0, 0)],
                u,
            });
        }

        let k_nn = (n / 10).clamp(2, 10);
        let mut aff = vec![0.0f64; n * n];
        for i in 0..n {
            aff[i * n + i] = 1.0;
            let mut neighbors: Vec<(f64, usize)> = Vec::with_capacity(n);
            for j in 0..n {
                if i != j {
                    let sim = cosine_similarity(&embeddings[i], &embeddings[j]) as f64;
                    neighbors.push((sim, j));
                }
            }
            neighbors.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(sim, j) in neighbors.iter().take(k_nn) {
                if sim > 0.0 {
                    aff[i * n + j] = sim;
                    aff[j * n + i] = sim;
                }
            }
        }

        let deg: Vec<f64> = (0..n).map(|i| aff[i * n..i * n + n].iter().sum()).collect();

        // Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        let mut lap = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let val = if i == j {
                    1.0 - aff[i * n + j] / deg[i].max(1e-10)
                } else {
                    -aff[i * n + j] / (deg[i].sqrt() * deg[j].sqrt()).max(1e-10)
                };
                lap[(i, j)] = val;
            }
        }

        let eig = lap.self_adjoint_eigen(Side::Lower).ok()?;
        let s = eig.S();
        let u = eig.U().cloned();

        let mut eig_pairs: Vec<(f64, usize)> = (0..n).map(|i| (s[i], i)).collect();
        eig_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        Some(Self { n, eig_pairs, u })
    }

    pub(crate) fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn eig_asc(&self) -> Vec<f64> {
        self.eig_pairs.iter().map(|p| p.0).collect()
    }

    /// Row-normalized spectral embedding using the first `k` eigenvectors
    /// (smallest eigenvalues), as `f64` features for BIC k-means.
    pub(crate) fn embedding_f64(&self, k: usize) -> Vec<Vec<f64>> {
        let k = k.min(self.n).max(1);
        let mut features = vec![vec![0.0f64; k]; self.n];
        for (i, feat) in features.iter_mut().enumerate() {
            for (col, &(_, idx)) in self.eig_pairs.iter().take(k).enumerate() {
                feat[col] = self.u[(i, idx)];
            }
            let norm: f64 = feat.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for v in feat.iter_mut() {
                    *v /= norm;
                }
            }
        }
        features
    }

    /// Same embedding as [`Self::embedding_f64`] in `f32` for `kmeans_pp`.
    pub(crate) fn embedding_f32(&self, k: usize) -> Vec<Vec<f32>> {
        self.embedding_f64(k)
            .into_iter()
            .map(|row| row.into_iter().map(|v| v as f32).collect())
            .collect()
    }
}

/// { true }
/// `pub fn spectral_cluster(embeddings: &[Vec<f32>], max_k: usize) -> Vec<usize>`
/// { ret.len() == embeddings.len() }
/// Run spectral clustering with automatic k selection.
pub fn spectral_cluster(embeddings: &[Vec<f32>], max_k: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let Some(graph) = SpectralGraph::from_embeddings(embeddings) else {
        return vec![0; n];
    };

    // Determine k via eigengap heuristic, then validate with BIC on spectral features.
    let max_k = max_k.min(n).min(20);
    // NME-SC normalized-maximum eigengap (Park et al. 2020) — shared with
    // NmeScClusterer so both spectral paths agree. Here it only SEEDS the
    // BIC search below, which makes the final k decision.
    let eigengap_k = select_k_by_normalized_eigengap(&graph.eig_asc(), max_k);

    // Extract spectral features for a range of k values and pick best via BIC.
    let mut best_k = eigengap_k.max(2).min(max_k);
    let mut best_bic = f64::INFINITY;

    for k in 2..=max_k.min(10) {
        let features = graph.embedding_f64(k);
        let labels = kmeans_on_features(&features, k, 20);
        // A singleton cluster has no definable variance — the spherical
        // Gaussian behind this BIC is degenerate there, and on row-normalized
        // spectral features k == n is always a "perfect" fit, so without this
        // guard the trivial everyone-is-their-own-speaker split wins. Require
        // at least 2 points per cluster for a k to be a valid candidate.
        let mut counts = vec![0usize; k];
        for &l in &labels {
            counts[l] += 1;
        }
        if counts.iter().any(|&c| c < 2) {
            continue;
        }
        let bic = compute_bic(&features, &labels, k);
        if bic < best_bic {
            best_bic = bic;
            best_k = k;
        }
    }

    let features = graph.embedding_f64(best_k);
    kmeans_on_features(&features, best_k, 20)
}

/// Simple Lloyd's k-means on pre-computed feature vectors.
fn kmeans_on_features(features: &[Vec<f64>], k: usize, max_iter: usize) -> Vec<usize> {
    let n = features.len();
    let k = k.min(n).max(1);
    let dim = features[0].len();

    if k == 1 {
        return vec![0; n];
    }

    // K-means++ initialization. Deterministic seed: this legacy path values
    // reproducibility (stable exact-k tests, identical runs on identical
    // input) over stochastic restarts; the production NME-SC path does not
    // come through here. Uses in-tree xorshift64* (same seed constant).
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut rng = crate::utils::XorShift64Star::new(0x504f_4c59_564f_4943);
    let first_idx = rng.usize(n);
    centroids.push(features[first_idx].clone());

    let mut dists = vec![f64::INFINITY; n];
    for _ in 1..k {
        for (i, feat) in features.iter().enumerate() {
            let d = euclidean_distance(feat, &centroids[centroids.len() - 1]);
            if d < dists[i] {
                dists[i] = d;
            }
        }
        let total: f64 = dists.iter().sum();
        if total <= 0.0 {
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
        centroids.push(features[chosen].clone());
    }

    let k = centroids.len();
    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;
        // Assign.
        for (i, feat) in features.iter().enumerate() {
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for (c_idx, c) in centroids.iter().enumerate() {
                let dist = euclidean_distance(feat, c);
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
        for (i, feat) in features.iter().enumerate() {
            let c = labels[i];
            for (new_centroid, &v) in new_centroids[c].iter_mut().zip(feat.iter()) {
                *new_centroid += v;
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

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn compute_bic(features: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
    let n = features.len();
    if n == 0 {
        return f64::INFINITY;
    }
    let dim = features[0].len();

    // Compute centroids.
    let mut centroids = vec![vec![0.0f64; dim]; k];
    let mut counts = vec![0usize; k];
    for (i, feat) in features.iter().enumerate() {
        let c = labels[i];
        for (d, &v) in feat.iter().enumerate() {
            centroids[c][d] += v;
        }
        counts[c] += 1;
    }
    for (c, centroid) in centroids.iter_mut().enumerate().take(k) {
        if counts[c] > 0 {
            for v in centroid.iter_mut().take(dim) {
                *v /= counts[c] as f64;
            }
        }
    }

    // Compute inertia (sum of squared distances).
    let mut inertia = 0.0f64;
    for (i, feat) in features.iter().enumerate() {
        let c = labels[i];
        inertia += euclidean_distance(feat, &centroids[c]).powi(2);
    }

    // BIC for spherical Gaussian: -2*log(L) + p*log(n)
    // where p = k * (dim + 1)  (centroids + 1 variance parameter per cluster)
    let p = k * (dim + 1);
    // Floor the inertia RELATIVE to the total feature variance so the
    // log-likelihood stays finite and saturates on (near-)perfect fits: every
    // k that explains the data down to the floor shares the same maximal
    // likelihood term, and the p*ln(n) complexity penalty then picks the
    // SMALLEST such k. That fixes both failure modes at once: the old branch
    // returned a bare positive penalty for perfect fits (a near-perfect true
    // k lost to an imperfect lower k with negative BIC — under-selection on
    // clean data), while an ABSOLUTE floor would let the trivial k == n
    // perfect fit out-likelihood a merely near-perfect true k (over-selection).
    let mean: Vec<f64> = (0..dim)
        .map(|d| features.iter().map(|f| f[d]).sum::<f64>() / n as f64)
        .collect();
    let total_ss: f64 = features
        .iter()
        .map(|f| euclidean_distance(f, &mean).powi(2))
        .sum();
    let floor = (total_ss * 1e-6).max(1e-12);
    let inertia = inertia.max(floor);
    let log_likelihood = -(n as f64) * (inertia / n as f64).ln() / 2.0;
    -2.0 * log_likelihood + p as f64 * (n as f64).ln()
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_cluster_basic() {
        // Two clear clusters in 3D.
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        let labels = spectral_cluster(&embeddings, 10);
        assert_eq!(labels.len(), 4);
        // Should find 2 clusters.
        let num_clusters = labels.iter().copied().max().unwrap_or(0) + 1;
        assert_eq!(num_clusters, 2);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_spectral_cluster_four_blocks_exact() {
        // Four tight clusters of 3 points each in 4D — exact k must be 4
        // (the singleton guard rejects k > 4 splits, the BIC floor keeps the
        // perfect k=4 from losing to an imperfect lower k).
        let mut embeddings = Vec::new();
        for axis in 0..4usize {
            for jitter in [0.0f32, 0.03, -0.03] {
                let mut v = vec![0.0f32; 4];
                v[axis] = 1.0;
                v[(axis + 1) % 4] = 0.05 + jitter;
                embeddings.push(v);
            }
        }
        let labels = spectral_cluster(&embeddings, 10);
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 4, "expected exactly 4 clusters");
    }

    #[test]
    fn test_spectral_cluster_noisy_three_blocks_does_not_explode() {
        // Three clusters with visible jitter: k must stay 3 — neither collapse
        // below (the historical under-selection) nor climb toward k == n (the
        // failure mode of a naive absolute inertia floor).
        let jitters = [0.00f32, 0.06, -0.06, 0.11, -0.11];
        let mut embeddings = Vec::new();
        for axis in 0..3usize {
            for (j, &jit) in jitters.iter().enumerate() {
                let mut v = vec![0.0f32; 3];
                v[axis] = 1.0 - 0.02 * j as f32;
                v[(axis + 1) % 3] = (0.08 + jit).abs();
                embeddings.push(v);
            }
        }
        let labels = spectral_cluster(&embeddings, 15);
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 3, "expected exactly 3 clusters on noisy data");
    }

    #[test]
    fn test_spectral_cluster_empty() {
        let labels = spectral_cluster(&[], 10);
        assert!(labels.is_empty());
    }

    #[test]
    fn eigengap_selects_k_on_known_sequence() {
        // Three near-zero "cluster" eigenvalues then a jump → k = 3.
        assert_eq!(
            select_k_by_normalized_eigengap(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 6),
            3
        );
        // A single near-zero eigenvalue then a jump → k = 1.
        assert_eq!(select_k_by_normalized_eigengap(&[0.0, 1.0, 1.0, 1.0], 4), 1);
        // Two clusters.
        assert_eq!(select_k_by_normalized_eigengap(&[0.0, 0.0, 1.0, 1.0], 4), 2);
    }

    #[test]
    fn test_spectral_cluster_three_blocks() {
        // Three tight, well-separated clusters (9 points); mirrors
        // NmeScClusterer's synthetic. The eigengap SEEDS k=3 and the BIC search
        // confirms it: with the perfect-fit floor in compute_bic (and the
        // deterministic k-means seed) the exact k is stable, so this asserts
        // unique == 3 — the historical under-selection to 2 would fail here.
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.98, 0.05, 0.0],
            vec![0.97, 0.0, 0.05],
            vec![0.0, 1.0, 0.0],
            vec![0.05, 0.98, 0.0],
            vec![0.0, 0.97, 0.05],
            vec![0.0, 0.0, 1.0],
            vec![0.05, 0.0, 0.98],
            vec![0.0, 0.05, 0.97],
        ];
        let labels = spectral_cluster(&embeddings, 10);
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(
            unique.len(),
            3,
            "spectral_cluster must find exactly 3 clusters, got {}",
            unique.len()
        );
    }

    #[test]
    fn test_spectral_cluster_single() {
        let labels = spectral_cluster(&[vec![1.0, 0.0]], 10);
        assert_eq!(labels, vec![0]);
    }
}
