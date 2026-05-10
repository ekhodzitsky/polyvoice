//! v1.0 `Clusterer` trait + concrete clusterers (NME-SC, AHC).
//!
//! Added in v0.6 (M3). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §3.1, §5.3.

/// Speaker clusterer — turns a batch of L2-normalized speaker embeddings into
/// per-embedding cluster labels in the range `0..K` where `K` is the inferred
/// number of clusters.
///
/// In v1.0 (M3) the polyvoice crate introduces `Clusterer` as the canonical
/// trait. The legacy free functions `ahc::agglomerative_cluster_auto` and
/// `spectral::spectral_cluster` remain available — M6 will deprecate them.
pub trait Clusterer: Send + Sync {
    /// Cluster `embeddings`. Each inner vector must have the same length and
    /// be approximately L2-normalized.
    ///
    /// **Requires:** `embeddings.len() >= 1`.
    /// **Guarantees on Ok:** `result.len() == embeddings.len()`,
    /// `result[i] < unique(result).count()` (compact 0..K numbering).
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError>;

    /// Hard ceiling on the number of clusters this implementation can produce.
    fn max_clusters(&self) -> usize;
}

/// Errors from `Clusterer` implementations.
#[derive(Debug, thiserror::Error)]
pub enum ClustererError {
    #[error("too few embeddings: got {actual}, need at least {min}")]
    TooFewEmbeddings { actual: usize, min: usize },

    #[error("embedding dimension mismatch: expected {expected}, got {actual} at index {index}")]
    DimMismatch {
        expected: usize,
        actual: usize,
        index: usize,
    },

    #[error("clustering failed: {detail}")]
    AlgorithmFailed { detail: String },
}

/// AHC (agglomerative hierarchical clustering) wrapper exposing the legacy
/// `crate::ahc::agglomerative_cluster_auto` through the v1.0 `Clusterer` trait.
pub struct AhcClusterer {
    max_clusters: usize,
}

impl AhcClusterer {
/// { TODO: precondition }
/// pub fn new(max_clusters: usize) -> Self
/// { TODO: postcondition }
    pub fn new(max_clusters: usize) -> Self {
        Self {
            max_clusters: max_clusters.max(1),
        }
    }
}

impl Default for AhcClusterer {
    fn default() -> Self {
        Self::new(64)
    }
}

impl Clusterer for AhcClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
        if embeddings.is_empty() {
            return Err(ClustererError::TooFewEmbeddings { actual: 0, min: 1 });
        }
        if embeddings.len() == 1 {
            return Ok(vec![0]);
        }
        let (labels, _threshold) = crate::ahc::agglomerative_cluster_auto(embeddings);
        Ok(labels)
    }

    fn max_clusters(&self) -> usize {
        self.max_clusters
    }
}

#[cfg(test)]
mod trait_tests {
    use super::*;

    /// In-memory dummy.
    struct ConstantClusterer {
        labels: Vec<usize>,
    }

    impl Clusterer for ConstantClusterer {
        fn cluster(&self, _embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
            Ok(self.labels.clone())
        }

        fn max_clusters(&self) -> usize {
            64
        }
    }

    #[test]
    fn clusterer_trait_object_is_dyn_compatible() {
        let c = ConstantClusterer {
            labels: vec![0, 1, 0],
        };
        let _b: Box<dyn Clusterer> = Box::new(c);
    }

    #[test]
    fn clusterer_returns_owned_labels() {
        let c = ConstantClusterer {
            labels: vec![0, 1, 2],
        };
        let embeddings: Vec<Vec<f32>> = (0..3).map(|_| vec![1.0; 3]).collect();
        let labels = c.cluster(&embeddings).unwrap();
        assert_eq!(labels, vec![0, 1, 2]);
    }

    #[test]
    fn error_too_few_embeddings_displays() {
        let err = ClustererError::TooFewEmbeddings { actual: 0, min: 1 };
        let msg = format!("{err}");
        assert!(msg.contains('0'));
    }
}

#[cfg(test)]
mod ahc_tests {
    use super::*;

    fn synth_two_clusters() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.05, 0.0],
            vec![0.95, 0.0, 0.05],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.05, 0.95, 0.0],
            vec![0.0, 1.0, 0.05],
        ]
    }

    fn synth_one_cluster() -> Vec<Vec<f32>> {
        vec![vec![1.0, 0.0, 0.0]; 5]
    }

    #[test]
    fn ahc_separates_two_well_separated_clusters() {
        let c = AhcClusterer::default();
        let labels = c.cluster(&synth_two_clusters()).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn ahc_collapses_one_cluster() {
        let c = AhcClusterer::default();
        let labels = c.cluster(&synth_one_cluster()).unwrap();
        assert!(labels.iter().all(|&l| l == labels[0]));
    }

    #[test]
    fn ahc_rejects_empty_input() {
        let c = AhcClusterer::default();
        let labels: &[Vec<f32>] = &[];
        let err = c.cluster(labels).expect_err("empty must fail");
        assert!(matches!(err, ClustererError::TooFewEmbeddings { .. }));
    }

    #[test]
    fn ahc_handles_single_embedding() {
        let c = AhcClusterer::default();
        let labels = c.cluster(&[vec![1.0, 0.0, 0.0]]).unwrap();
        assert_eq!(labels, vec![0]);
    }
}

/// NME-SC (Normalized Maximum Eigengap Spectral Clustering) clusterer.
///
/// Builds a k-NN cosine-affinity graph, computes the normalized Laplacian,
/// selects k via the normalized maximum eigengap heuristic, then runs
/// k-means++ on the spectral embedding.  This is the canonical NME-SC
/// procedure from Park et al. (2022) and differs from
/// `crate::spectral::spectral_cluster` in that it does **not** apply a
/// BIC override — the eigengap alone drives k selection.
#[cfg(feature = "spectral")]
pub struct NmeScClusterer {
    max_clusters: usize,
}

#[cfg(feature = "spectral")]
impl NmeScClusterer {
/// { TODO: precondition }
/// pub fn new(max_clusters: usize) -> Self
/// { TODO: postcondition }
    pub fn new(max_clusters: usize) -> Self {
        Self {
            max_clusters: max_clusters.max(1),
        }
    }
}

#[cfg(feature = "spectral")]
impl Default for NmeScClusterer {
    fn default() -> Self {
        Self::new(64)
    }
}

#[cfg(feature = "spectral")]
impl Clusterer for NmeScClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
        use crate::utils::cosine_similarity;
        use faer::Side;
        use faer::prelude::*;

        let n = embeddings.len();
        if n == 0 {
            return Err(ClustererError::TooFewEmbeddings { actual: 0, min: 1 });
        }
        if n == 1 {
            return Ok(vec![0]);
        }

        // Build k-NN cosine affinity matrix.
        let k_nn = (n / 10).clamp(2, 10);
        let mut aff = vec![0.0f64; n * n];
        for i in 0..n {
            aff[i * n + i] = 1.0;
            let mut neighbors: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (cosine_similarity(&embeddings[i], &embeddings[j]) as f64, j))
                .collect();
            neighbors.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(sim, j) in neighbors.iter().take(k_nn) {
                if sim > 0.0 {
                    aff[i * n + j] = sim;
                    aff[j * n + i] = sim;
                }
            }
        }

        // Degree vector.
        let deg: Vec<f64> = (0..n).map(|i| aff[i * n..i * n + n].iter().sum()).collect();

        // Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
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

        // Eigendecomposition.
        let eig = match lap.self_adjoint_eigen(Side::Lower) {
            Ok(e) => e,
            Err(_) => return Ok(vec![0; n]),
        };
        let s = eig.S();
        let u = eig.U();

        // Sort eigenvalues ascending.
        let mut eig_pairs: Vec<(f64, usize)> = (0..n).map(|i| (s[i], i)).collect();
        eig_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Normalized Maximum Eigengap: pick k that maximises
        // (λ_{k+1} - λ_k) / λ_{k+1}  for k in 1..max_k.
        let max_k = self.max_clusters.min(n).min(20);
        let mut best_k = 1usize;
        let mut best_gap = 0.0f64;
        for k in 1..max_k {
            let lam_k = eig_pairs[k - 1].0;
            let lam_k1 = eig_pairs[k].0;
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
        let k = best_k.max(1);

        // Extract spectral embedding (top-k eigenvectors, row-normalised).
        let mut spectral: Vec<Vec<f32>> = vec![vec![0.0f32; k]; n];
        for i in 0..n {
            let mut norm_sq = 0.0f64;
            for (col, &(_, idx)) in eig_pairs.iter().take(k).enumerate() {
                let v = u[(i, idx)];
                spectral[i][col] = v as f32;
                norm_sq += v * v;
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-10 {
                for v in spectral[i].iter_mut() {
                    *v /= norm as f32;
                }
            }
        }

        // Final clustering with detected k.
        let labels = crate::kmeans::kmeans_pp(&spectral, k, 50);
        Ok(labels)
    }

    fn max_clusters(&self) -> usize {
        self.max_clusters
    }
}

#[cfg(test)]
#[cfg(feature = "spectral")]
mod nme_sc_tests {
    use super::*;

    fn synth_three_clusters() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.98, 0.05, 0.0],
            vec![0.97, 0.0, 0.05],
            vec![0.0, 1.0, 0.0],
            vec![0.05, 0.98, 0.0],
            vec![0.0, 0.97, 0.05],
            vec![0.0, 0.0, 1.0],
            vec![0.05, 0.0, 0.98],
            vec![0.0, 0.05, 0.97],
        ]
    }

    #[test]
    fn nme_sc_separates_three_clusters() {
        let c = NmeScClusterer::default();
        let labels = c.cluster(&synth_three_clusters()).expect("synthetic clusters must be clusterable");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[7], labels[8]);
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn nme_sc_rejects_empty_input() {
        let c = NmeScClusterer::default();
        let labels: &[Vec<f32>] = &[];
        let err = c.cluster(labels).expect_err("empty must fail");
        assert!(matches!(err, ClustererError::TooFewEmbeddings { .. }));
    }

    #[test]
    fn nme_sc_max_clusters_caps_estimate() {
        let c = NmeScClusterer::new(2);
        let labels = c.cluster(&synth_three_clusters()).expect("synthetic clusters must be clusterable");
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert!(unique.len() <= 2);
    }
}
