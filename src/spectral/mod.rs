//! Spectral clustering math for speaker diarization.
//!
//! Normalized graph Laplacian + eigenspectrum over a k-NN cosine affinity
//! graph (`SpectralGraph`, crate-private), plus the NME-SC normalized-maximum
//! eigengap k selection. The single consumer is
//! `crate::clusterer::NmeScClusterer`, which runs k-means (`crate::kmeans`)
//! on the spectral embedding.

use faer::Side;
use faer::prelude::*;

/// Hard cap on the eigengap search width in
/// [`select_k_by_normalized_eigengap`]: NME-SC never considers more than this
/// many candidate cluster counts, regardless of `max_k`.
#[cfg_attr(not(feature = "clusterer"), allow(dead_code))]
pub(crate) const MAX_EIGENGAP_CANDIDATES: usize = 20;

/// Select the number of clusters via the NME-SC normalized-maximum eigengap
/// heuristic (Park et al., "Auto-Tuning Spectral Clustering for Speaker
/// Diarization Using Normalized Maximum Eigengap", 2020): pick the `k` in
/// `1..max_k` that maximizes `(eig_asc[k] - eig_asc[k-1]) / |eig_asc[k]|`, where
/// `eig_asc` are the normalized-Laplacian eigenvalues sorted ascending. A large
/// normalized gap at position `k` means the first `k` eigenvalues are the
/// near-zero "cluster" eigenvalues, hence `k` clusters. Returns `>= 1`.
///
/// Single source of truth for the eigengap convention; drives `k` directly in
/// `crate::clusterer::NmeScClusterer`.
#[cfg_attr(not(feature = "clusterer"), allow(dead_code))]
pub(crate) fn select_k_by_normalized_eigengap(eig_asc: &[f64], max_k: usize) -> usize {
    let max_k = max_k.min(eig_asc.len()).min(MAX_EIGENGAP_CANDIDATES);
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
/// Crate-private single graph-construction path behind
/// `crate::clusterer::NmeScClusterer`.
#[cfg_attr(not(feature = "clusterer"), allow(dead_code))]
pub(crate) struct SpectralGraph {
    n: usize,
    /// Eigenvalues ascending with original eigenvector column index.
    eig_pairs: Vec<(f64, usize)>,
    /// Eigenvector matrix `U` from the self-adjoint decomposition (`u[(row, col)]`).
    u: Mat<f64>,
}

#[cfg_attr(not(feature = "clusterer"), allow(dead_code))]
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
        let sims = crate::utils::pairwise_cosine_similarity_matrix(embeddings);
        let mut aff = vec![0.0f64; n * n];
        for i in 0..n {
            aff[i * n + i] = 1.0;
            let mut neighbors: Vec<(f64, usize)> = Vec::with_capacity(n);
            for j in 0..n {
                if i != j {
                    let sim = sims[i * n + j] as f64;
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
    /// (smallest eigenvalues).
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

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

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
    fn eigengap_clamps_max_k_and_skips_zero_eigenvalue_gaps() {
        // max_k beyond the spectrum length is clamped; a zero |lam_k1|
        // contributes gap 0, so k stays 1 on an all-zero spectrum.
        assert_eq!(select_k_by_normalized_eigengap(&[0.0, 0.0, 0.0], 100), 1);
        // max_k = 1 leaves no candidates.
        assert_eq!(select_k_by_normalized_eigengap(&[0.0, 1.0], 1), 1);
        // Empty spectrum.
        assert_eq!(select_k_by_normalized_eigengap(&[], 5), 1);
    }

    #[test]
    fn spectral_graph_empty_and_singleton() {
        assert!(SpectralGraph::from_embeddings(&[]).is_none());

        let g = SpectralGraph::from_embeddings(&[vec![1.0, 2.0]]).unwrap();
        assert_eq!(g.n(), 1);
        assert_eq!(g.eig_asc(), vec![0.0]);
        // k is clamped into 1..=n.
        assert_eq!(g.embedding_f64(5), vec![vec![1.0]]);
        assert_eq!(g.embedding_f32(1), vec![vec![1.0f32]]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn spectral_graph_two_clusters_recovers_k2() {
        // Two tight, well-separated clusters of near-parallel embeddings.
        let mut embeddings = Vec::new();
        for i in 0..10 {
            embeddings.push(vec![1.0, 0.01 * i as f32, 0.0]);
        }
        for i in 0..10 {
            embeddings.push(vec![0.0, 0.01 * i as f32, 1.0]);
        }
        let g = SpectralGraph::from_embeddings(&embeddings).unwrap();
        assert_eq!(g.n(), 20);

        let eig = g.eig_asc();
        assert_eq!(eig.len(), 20);
        assert!(eig.windows(2).all(|w| w[0] <= w[1]));
        assert!(eig[0].abs() < 1e-6, "first eigenvalue should be ~0");

        let k = select_k_by_normalized_eigengap(&eig, 5);
        assert_eq!(k, 2);

        let emb = g.embedding_f32(2);
        assert_eq!(emb.len(), 20);
        assert!(emb.iter().all(|row| row.len() == 2));
        // Row-normalized: non-degenerate rows are unit norm.
        for row in &emb {
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                norm < 1e-10 || (norm - 1.0).abs() < 1e-4,
                "row norm {norm} neither degenerate nor ~1"
            );
        }
    }
}
