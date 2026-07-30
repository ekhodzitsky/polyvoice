//! v1.0 `Clusterer` trait + concrete clusterers (NME-SC, AHC).
//!
//! Added in v0.6.

#[cfg(feature = "vbx")]
pub mod plda;
#[cfg(feature = "vbx")]
pub mod vbx;

pub mod assign;
pub mod short_filter;

pub use assign::{
    LocalGlobalDuration, build_cooccurrence, hungarian_local_to_global, majority_local_to_global,
};
pub use short_filter::{
    partition_by_min_duration, reassign_short_by_cosine, reassign_short_by_features,
};

/// Inputs smaller than this many embeddings make k-means / NME-SC unstable,
/// so the clusterers delegate to AHC instead.
const AHC_FALLBACK_N: usize = 8;

/// Speaker clusterer — turns a batch of L2-normalized speaker embeddings into
/// per-embedding cluster labels in the range `0..K` where `K` is the inferred
/// number of clusters.
///
/// `Clusterer` is the production clustering surface: pipeline v2, CLI, FFI,
/// Python, and MCP all go through it. The free functions in `ahc`, `kmeans`,
/// and `spectral` are the math layer underneath — used directly by the
/// BYO/legacy `LegacyPipeline` and wrapped here by the `Clusterer` adapters.
#[cfg_attr(test, mockall::automock)]
pub trait Clusterer: Send + Sync {
    /// Cluster `embeddings`. Each inner vector must have the same length and
    /// be approximately L2-normalized.
    ///
    /// **Requires:** `embeddings.len() >= 1`.
    /// **Guarantees on Ok:** `result.len() == embeddings.len()`,
    /// `result[i] < unique(result).count()` (compact 0..K numbering).
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError>;

    /// Cluster with per-embedding durations (seconds). Defaults to ignoring
    /// durations and calling [`Self::cluster`]. Backends that filter short embeddings
    /// (cVBx short-segment exclusion) override this so unreliable short windows
    /// are kept out of AHC/VB and reassigned afterward.
    ///
    /// When `durations_secs.len() != embeddings.len()` the durations are ignored
    /// (treated as absent) rather than erroring — callers without timing info
    /// may pass an empty slice.
    fn cluster_with_durations(
        &self,
        embeddings: &[Vec<f32>],
        durations_secs: &[f64],
    ) -> Result<Vec<usize>, ClustererError> {
        let _ = durations_secs;
        self.cluster(embeddings)
    }

    /// Hard ceiling on the number of clusters this implementation can produce.
    fn max_clusters(&self) -> usize;

    /// Whether this clusterer wants raw (non-L2-normalized) embeddings. Cosine
    /// clusterers (AHC, NME-SC) are scale-invariant and default to `false`; a PLDA
    /// backend needs the original embedding scale for its mean-centering and
    /// overrides to `true`.
    fn wants_raw_embeddings(&self) -> bool {
        false
    }
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

    /// PLDA model load/transform failure (VBx backend).
    #[cfg(feature = "vbx")]
    #[error("PLDA model error: {0}")]
    Plda(#[from] crate::clusterer::plda::PldaError),
}

/// Verifies that every embedding in `embeddings` has the same dimension.
///
/// { !embeddings.is_empty() }
/// fn uniform_dim(embeddings: &[Vec<f32>]) -> Result<(), ClustererError>
/// { ret.is_ok() -> embeddings.iter().all(|e| e.len() == embeddings[0].len()) }
fn uniform_dim(embeddings: &[Vec<f32>]) -> Result<(), ClustererError> {
    let expected = embeddings[0].len();
    for (index, emb) in embeddings.iter().enumerate().skip(1) {
        let actual = emb.len();
        if actual != expected {
            return Err(ClustererError::DimMismatch {
                expected,
                actual,
                index,
            });
        }
    }
    Ok(())
}

/// AHC (agglomerative hierarchical clustering) wrapper exposing the
/// `crate::ahc` free functions through the v1.0 `Clusterer` trait.
pub struct AhcClusterer {
    max_clusters: usize,
    /// Fixed cosine-similarity threshold. When `Some`, `agglomerative_cluster`
    /// is used (legacy behaviour). When `None`, automatic threshold selection
    /// via `agglomerative_cluster_auto_max_clusters` is used.
    threshold: Option<f32>,
}

impl AhcClusterer {
    /// { true }
    /// pub fn new(max_clusters: usize) -> Self
    /// { ret.max_clusters >= 1 }
    /// Create with automatic threshold selection.
    pub fn new(max_clusters: usize) -> Self {
        Self {
            max_clusters: max_clusters.max(1),
            threshold: None,
        }
    }

    /// Create with a fixed merge threshold (legacy behaviour).
    ///
    /// `max_clusters == 0` means **no ceiling** — same as
    /// [`crate::ahc::agglomerative_cluster`]. Non-zero values hard-cap the
    /// active cluster count (see `agglomerative_cluster_max_clusters`).
    pub fn with_threshold(max_clusters: usize, threshold: f32) -> Self {
        Self {
            max_clusters,
            threshold: Some(threshold),
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
        uniform_dim(embeddings)?;
        let labels = match self.threshold {
            Some(t) => {
                crate::ahc::agglomerative_cluster_max_clusters(embeddings, t, self.max_clusters)
            }
            None => {
                crate::ahc::agglomerative_cluster_auto_max_clusters(embeddings, self.max_clusters).0
            }
        };
        Ok(labels)
    }

    fn max_clusters(&self) -> usize {
        self.max_clusters
    }
}

/// Decorator clusterer that prunes spurious *small* clusters.
///
/// Runs an `inner` clusterer, then reassigns every member of a cluster with
/// fewer than `min_size` members to the nearest **surviving** (large) cluster
/// centroid by cosine similarity, finally recompacting labels to `0..K`.
///
/// This targets over-clustering — the dominant diarization error where genuine
/// speakers fragment into many tiny clusters, inflating the speaker count while
/// barely moving frame-DER. Unlike lowering the global AHC merge threshold
/// (which over-merges *large* genuine speakers and raises confusion), this only
/// dissolves sub-`min_size` clusters; large clusters are never merged into each
/// other.
pub struct MinClusterSizeClusterer {
    inner: Box<dyn Clusterer>,
    min_size: usize,
}

impl MinClusterSizeClusterer {
    /// Wrap `inner`, pruning any cluster with fewer than `min_size` members.
    /// `min_size <= 1` makes this a transparent pass-through.
    pub fn new(inner: Box<dyn Clusterer>, min_size: usize) -> Self {
        Self { inner, min_size }
    }
}

impl Clusterer for MinClusterSizeClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
        let labels = self.inner.cluster(embeddings)?;
        Ok(crate::ahc::prune_small_clusters(
            embeddings,
            labels,
            self.min_size,
        ))
    }

    fn cluster_with_durations(
        &self,
        embeddings: &[Vec<f32>],
        durations_secs: &[f64],
    ) -> Result<Vec<usize>, ClustererError> {
        let labels = self
            .inner
            .cluster_with_durations(embeddings, durations_secs)?;
        Ok(crate::ahc::prune_small_clusters(
            embeddings,
            labels,
            self.min_size,
        ))
    }

    fn max_clusters(&self) -> usize {
        self.inner.max_clusters()
    }

    fn wants_raw_embeddings(&self) -> bool {
        self.inner.wants_raw_embeddings()
    }
}

/// K-Means++ clusterer with automatic k selection via silhouette score.
pub struct KmeansClusterer {
    max_clusters: usize,
    max_iter: usize,
    trials: usize,
    fast_mode: bool,
}

/// Deprecated alias for the pre-rename name; use [`KmeansClusterer`].
#[deprecated(note = "renamed to KmeansClusterer")]
pub type KMeansClusterer = KmeansClusterer;

impl KmeansClusterer {
    /// Create a new K-means clusterer with automatic k selection.
    /// `max_clusters` is the upper bound on the number of clusters.
    pub fn new(max_clusters: usize) -> Self {
        Self {
            max_clusters: max_clusters.max(2),
            max_iter: 50,
            trials: 3,
            fast_mode: false,
        }
    }

    /// Enable fast mode: fewer k candidates, fewer iterations, 1 trial.
    /// ~10× faster than default, with minor quality trade-off.
    pub fn fast_mode(mut self) -> Self {
        self.fast_mode = true;
        self
    }

    /// Set the maximum number of Lloyd iterations (default 50).
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the number of random initializations per k (default 3).
    pub fn with_trials(mut self, trials: usize) -> Self {
        self.trials = trials.max(1);
        self
    }
}

impl Default for KmeansClusterer {
    fn default() -> Self {
        Self::new(64)
    }
}

impl Clusterer for KmeansClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
        if embeddings.is_empty() {
            return Err(ClustererError::TooFewEmbeddings { actual: 0, min: 1 });
        }
        if embeddings.len() == 1 {
            return Ok(vec![0]);
        }
        uniform_dim(embeddings)?;
        // Fallback to AHC for tiny inputs where k-means is unstable.
        if embeddings.len() < AHC_FALLBACK_N {
            return AhcClusterer::new(self.max_clusters).cluster(embeddings);
        }
        let n = embeddings.len();
        let (k_max, max_iter, trials) = if self.fast_mode {
            // Adaptive k_max: fewer candidates for small n, capped at 12.
            let adaptive_k = (n / 20).clamp(5, 12).min(self.max_clusters);
            (adaptive_k, 20, 1)
        } else {
            (self.max_clusters.min(n), self.max_iter, self.trials)
        };
        let labels = crate::kmeans::kmeans_auto_k(embeddings, 2, k_max, max_iter, trials);
        Ok(labels)
    }

    fn max_clusters(&self) -> usize {
        self.max_clusters
    }
}

#[allow(clippy::unwrap_used)]
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

#[allow(clippy::unwrap_used)]
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

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod min_cluster_size_tests {
    use super::*;

    /// Inner clusterer that returns a fixed label vector, so the decorator's
    /// pruning logic is tested in isolation from any real clustering.
    struct PresetClusterer {
        labels: Vec<usize>,
    }
    impl Clusterer for PresetClusterer {
        fn cluster(&self, _embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
            Ok(self.labels.clone())
        }
        fn max_clusters(&self) -> usize {
            64
        }
    }

    fn unique(labels: &[usize]) -> std::collections::HashSet<usize> {
        labels.iter().copied().collect()
    }

    /// Build n embeddings near a 3-D axis (`axis` in 0..3).
    fn near(axis: usize, n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| {
                let mut v = vec![0.02f32, 0.02, 0.02];
                v[axis] = 1.0;
                v
            })
            .collect()
    }

    #[test]
    fn small_cluster_collapses_into_large() {
        // 12 embeddings (label 0) + 3 embeddings (label 1), min_size 12.
        let mut embs = near(0, 12);
        embs.extend(near(0, 3)); // the 3 are near the same axis -> merge into 0
        let labels: Vec<usize> = std::iter::repeat_n(0, 12)
            .chain(std::iter::repeat_n(1, 3))
            .collect();
        let c = MinClusterSizeClusterer::new(Box::new(PresetClusterer { labels }), 12);
        let out = c.cluster(&embs).unwrap();
        assert_eq!(out.len(), 15, "every member retained");
        assert_eq!(unique(&out).len(), 1, "the 3-member cluster is dissolved");
        assert!(unique(&out).iter().all(|&l| l < 1), "labels compact 0..K");
    }

    #[test]
    fn spurious_split_reassigns_to_correct_survivor() {
        // Two real clusters (12 each) on axes 0 and 1, plus a 3-member spurious
        // split of cluster A (near axis 0) as label 2. min_size 6.
        let mut embs = near(0, 12);
        embs.extend(near(1, 12));
        embs.extend(near(0, 3)); // spurious, belongs with axis-0 cluster
        let labels: Vec<usize> = std::iter::repeat_n(0, 12)
            .chain(std::iter::repeat_n(1, 12))
            .chain(std::iter::repeat_n(2, 3))
            .collect();
        let c = MinClusterSizeClusterer::new(Box::new(PresetClusterer { labels }), 6);
        let out = c.cluster(&embs).unwrap();
        assert_eq!(unique(&out).len(), 2, "two survivors remain");
        // The last 3 (spurious) must share a label with the axis-0 cluster (first 12).
        let axis0_label = out[0];
        assert!(out[24..27].iter().all(|&l| l == axis0_label));
        // and NOT with the axis-1 cluster.
        assert_ne!(out[12], axis0_label);
    }

    #[test]
    fn min_size_one_is_identity() {
        let embs = near(0, 3);
        let labels = vec![0, 1, 2];
        let c = MinClusterSizeClusterer::new(
            Box::new(PresetClusterer {
                labels: labels.clone(),
            }),
            1,
        );
        assert_eq!(c.cluster(&embs).unwrap(), labels);
    }

    #[test]
    fn no_small_clusters_passes_through() {
        let mut embs = near(0, 6);
        embs.extend(near(1, 6));
        let labels: Vec<usize> = std::iter::repeat_n(0, 6)
            .chain(std::iter::repeat_n(1, 6))
            .collect();
        let c = MinClusterSizeClusterer::new(
            Box::new(PresetClusterer {
                labels: labels.clone(),
            }),
            6,
        );
        assert_eq!(c.cluster(&embs).unwrap(), labels);
    }

    #[test]
    fn all_small_keeps_one_cluster_no_panic() {
        let embs = near(0, 3);
        let labels = vec![0, 1, 2];
        let c = MinClusterSizeClusterer::new(Box::new(PresetClusterer { labels }), 12);
        let out = c.cluster(&embs).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(
            unique(&out).len(),
            1,
            "degenerate all-small collapses to 1 survivor"
        );
        assert!(out.iter().all(|&l| l == 0), "compact single label");
    }

    #[test]
    fn output_labels_are_compact() {
        // inner emits non-compact-after-prune labels; ensure 0..K with no gaps.
        let mut embs = near(0, 8);
        embs.extend(near(1, 8));
        embs.extend(near(2, 2)); // small, dissolves
        let labels: Vec<usize> = std::iter::repeat_n(0, 8)
            .chain(std::iter::repeat_n(5, 8)) // deliberately non-contiguous label
            .chain(std::iter::repeat_n(9, 2))
            .collect();
        let c = MinClusterSizeClusterer::new(Box::new(PresetClusterer { labels }), 6);
        let out = c.cluster(&embs).unwrap();
        let u = unique(&out);
        assert_eq!(u.len(), 2);
        let max = out.iter().copied().max().unwrap();
        assert_eq!(max, u.len() - 1, "labels are compact 0..K");
    }
}

/// NME-SC (Normalized Maximum Eigengap Spectral Clustering) clusterer.
///
/// Thin adapter over the shared spectral graph (k-NN affinity + Laplacian
/// spectrum in `crate::spectral`): the normalized-maximum eigengap picks `k`
/// directly, then `kmeans_pp` runs on the spectral embedding.
#[cfg(feature = "spectral")]
pub struct NmeScClusterer {
    max_clusters: usize,
}

#[cfg(feature = "spectral")]
impl NmeScClusterer {
    /// { true }
    /// pub fn new(max_clusters: usize) -> Self
    /// { ret.max_clusters >= 1 }
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
        let n = embeddings.len();
        if n == 0 {
            return Err(ClustererError::TooFewEmbeddings { actual: 0, min: 1 });
        }
        if n == 1 {
            return Ok(vec![0]);
        }
        uniform_dim(embeddings)?;
        // Fallback: tiny k-NN graphs collapse to 1 cluster; delegate to AHC.
        if n < AHC_FALLBACK_N {
            return AhcClusterer::new(self.max_clusters).cluster(embeddings);
        }

        // Shared k-NN affinity → Laplacian → eigenspectrum from crate::spectral.
        let Some(graph) = crate::spectral::SpectralGraph::from_embeddings(embeddings) else {
            return Ok(vec![0; n]);
        };

        // Normalized Maximum Eigengap (Park et al. 2020) drives k directly.
        let max_k = self
            .max_clusters
            .min(graph.n())
            .min(crate::spectral::MAX_EIGENGAP_CANDIDATES);
        let k = crate::spectral::select_k_by_normalized_eigengap(&graph.eig_asc(), max_k).max(1);

        let spectral = graph.embedding_f32(k);
        let labels = crate::kmeans::kmeans_pp(&spectral, k, 50);
        Ok(labels)
    }

    fn max_clusters(&self) -> usize {
        self.max_clusters
    }
}

#[allow(clippy::unwrap_used)]
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
        let labels = c
            .cluster(&synth_three_clusters())
            .expect("synthetic clusters must be clusterable");
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
        let labels = c
            .cluster(&synth_three_clusters())
            .expect("synthetic clusters must be clusterable");
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert!(unique.len() <= 2);
    }

    #[test]
    fn nme_sc_fallback_to_ahc_on_small_n() {
        let c = NmeScClusterer::default();
        // 3 well-separated embeddings — below AHC_FALLBACK_N, NME-SC delegates to AHC.
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let labels = c.cluster(&embeddings).unwrap();
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 3, "AHC fallback should preserve 3 clusters");
    }
}

#[cfg(test)]
mod dim_uniformity_tests {
    use super::*;

    fn mismatched_embeddings() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0], // dimension 4 at index 2
        ]
    }

    #[test]
    fn ahc_rejects_dim_mismatch() {
        let c = AhcClusterer::default();
        let err = c
            .cluster(&mismatched_embeddings())
            .expect_err("mismatched dims must fail");
        assert!(matches!(
            err,
            ClustererError::DimMismatch {
                expected: 3,
                actual: 4,
                index: 2,
            }
        ));
    }

    #[test]
    fn kmeans_rejects_dim_mismatch() {
        let c = KmeansClusterer::default();
        let err = c
            .cluster(&mismatched_embeddings())
            .expect_err("mismatched dims must fail");
        assert!(matches!(
            err,
            ClustererError::DimMismatch {
                expected: 3,
                actual: 4,
                index: 2,
            }
        ));
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn nme_sc_rejects_dim_mismatch() {
        let c = NmeScClusterer::default();
        let err = c
            .cluster(&mismatched_embeddings())
            .expect_err("mismatched dims must fail");
        assert!(matches!(
            err,
            ClustererError::DimMismatch {
                expected: 3,
                actual: 4,
                index: 2,
            }
        ));
    }
}
