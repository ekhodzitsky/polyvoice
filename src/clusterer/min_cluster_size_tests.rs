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

#[test]
fn cluster_with_durations_also_prunes() {
    // 2 large members (label 0) + 1 small member (label 1), min_size 2:
    // the duration-aware entry point must prune exactly like `cluster`.
    let embs = near(0, 3);
    let labels = vec![0, 0, 1];
    let c = MinClusterSizeClusterer::new(Box::new(PresetClusterer { labels }), 2);
    let out = c.cluster_with_durations(&embs, &[2.0, 2.0, 0.4]).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(unique(&out).len(), 1, "singleton cluster is dissolved");
}

#[test]
fn delegates_max_clusters_and_raw_embedding_preference() {
    struct RawPresetClusterer {
        labels: Vec<usize>,
    }
    impl Clusterer for RawPresetClusterer {
        fn cluster(&self, _embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
            Ok(self.labels.clone())
        }
        fn max_clusters(&self) -> usize {
            7
        }
        fn wants_raw_embeddings(&self) -> bool {
            true
        }
    }

    let plain = MinClusterSizeClusterer::new(Box::new(PresetClusterer { labels: vec![0] }), 2);
    assert_eq!(plain.max_clusters(), 64);
    assert!(!plain.wants_raw_embeddings());

    let raw = MinClusterSizeClusterer::new(Box::new(RawPresetClusterer { labels: vec![0] }), 2);
    assert_eq!(raw.max_clusters(), 7);
    assert!(raw.wants_raw_embeddings());
}
