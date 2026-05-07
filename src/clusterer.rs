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
    pub fn new(max_clusters: usize) -> Self {
        Self { max_clusters: max_clusters.max(1) }
    }
}

impl Default for AhcClusterer {
    fn default() -> Self { Self::new(64) }
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

    fn max_clusters(&self) -> usize { self.max_clusters }
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
            vec![1.0, 0.05, 0.0], vec![0.95, 0.0, 0.05], vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0], vec![0.05, 0.95, 0.0], vec![0.0, 1.0, 0.05],
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
