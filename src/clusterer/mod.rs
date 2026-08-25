//! v1.0 `Clusterer` trait + concrete clusterers (NME-SC, AHC).
//!
//! Added in v0.6.

#[cfg(feature = "vbx")]
pub mod plda;
#[cfg(feature = "vbx")]
pub mod vbx;

pub mod asnorm;
pub mod assign;
pub mod domain;
pub mod short_filter;

pub use asnorm::{
    AsNormClusterer, AsNormCohort, AsNormConfig, AsNormError, CohortSource, DEFAULT_AS_NORM_TOP_N,
    DEFAULT_ASNORM_COHORT_MODEL_ID,
};
pub use assign::{
    LocalGlobalDuration, build_cooccurrence, hungarian_local_to_global, majority_local_to_global,
};
pub use domain::{
    AMI, CALLHOME, DEFAULT_DOMAIN_PROFILE, DOMAIN_PROFILES, DomainProfile, VOXCONVERSE,
    domain_profile,
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
#[path = "trait_tests.rs"]
mod trait_tests;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "ahc_tests.rs"]
mod ahc_tests;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "min_cluster_size_tests.rs"]
mod min_cluster_size_tests;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "kmeans_tests.rs"]
mod kmeans_tests;

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
#[path = "nme_sc_tests.rs"]
mod nme_sc_tests;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "dim_uniformity_tests.rs"]
mod dim_uniformity_tests;
