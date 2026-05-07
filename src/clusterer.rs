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
