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

#[test]
fn error_dim_mismatch_displays_details() {
    let err = ClustererError::DimMismatch {
        expected: 3,
        actual: 4,
        index: 2,
    };
    let msg = format!("{err}");
    assert!(msg.contains('3'));
    assert!(msg.contains('4'));
    assert!(msg.contains('2'));
}

#[test]
fn error_algorithm_failed_displays_detail() {
    let err = ClustererError::AlgorithmFailed {
        detail: "eigengap blew up".to_string(),
    };
    assert!(format!("{err}").contains("eigengap blew up"));
}

#[test]
fn cluster_with_durations_default_ignores_durations() {
    let c = ConstantClusterer {
        labels: vec![0, 1, 0],
    };
    let embeddings: Vec<Vec<f32>> = (0..3).map(|_| vec![1.0; 3]).collect();
    // Matching and mismatched duration lengths both delegate to `cluster`.
    let with = c
        .cluster_with_durations(&embeddings, &[2.0, 0.5, 1.0])
        .unwrap();
    let without = c.cluster_with_durations(&embeddings, &[]).unwrap();
    assert_eq!(with, vec![0, 1, 0]);
    assert_eq!(without, vec![0, 1, 0]);
}

#[test]
fn wants_raw_embeddings_defaults_to_false() {
    let c = ConstantClusterer { labels: vec![0] };
    assert!(!c.wants_raw_embeddings());
}

#[test]
fn mock_clusterer_satisfies_trait() {
    let mut mock = MockClusterer::new();
    mock.expect_cluster()
        .returning(|embs| Ok(vec![0; embs.len()]));
    mock.expect_max_clusters().returning(|| 4);
    let c: Box<dyn Clusterer> = Box::new(mock);
    let labels = c.cluster(&[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
    assert_eq!(labels, vec![0, 0]);
    assert_eq!(c.max_clusters(), 4);
}
