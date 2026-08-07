use super::*;

/// Two tight, well-separated blobs of 8 embeddings each on axes 0 and 1.
/// A small per-point jitter keeps the points distinct so higher-k
/// candidates don't hit degenerate seeding.
fn synth_two_blobs() -> Vec<Vec<f32>> {
    let mut embs = Vec::new();
    for axis in [0usize, 1] {
        for i in 0..8 {
            let mut v = vec![0.02f32, 0.02, 0.02];
            v[axis] = 1.0;
            v[2] += i as f32 * 0.001;
            embs.push(v);
        }
    }
    embs
}

#[test]
fn kmeans_rejects_empty_input() {
    let c = KmeansClusterer::default();
    let labels: &[Vec<f32>] = &[];
    let err = c.cluster(labels).expect_err("empty must fail");
    assert!(matches!(err, ClustererError::TooFewEmbeddings { .. }));
}

#[test]
fn kmeans_handles_single_embedding() {
    let c = KmeansClusterer::default();
    let labels = c.cluster(&[vec![1.0, 0.0, 0.0]]).unwrap();
    assert_eq!(labels, vec![0]);
}

#[test]
fn kmeans_new_clamps_max_clusters_to_two() {
    assert_eq!(KmeansClusterer::new(0).max_clusters(), 2);
    assert_eq!(KmeansClusterer::new(8).max_clusters(), 8);
    assert_eq!(KmeansClusterer::default().max_clusters(), 64);
}

#[test]
fn kmeans_falls_back_to_ahc_below_eight_embeddings() {
    // Below the stability floor k-means delegates to AHC; two clear groups
    // must still be separated.
    let c = KmeansClusterer::default();
    let embeddings = vec![
        vec![1.0, 0.02, 0.02],
        vec![0.98, 0.02, 0.02],
        vec![1.0, 0.02, 0.02],
        vec![0.02, 1.0, 0.02],
        vec![0.02, 0.98, 0.02],
        vec![0.02, 1.0, 0.02],
    ];
    let labels = c.cluster(&embeddings).unwrap();
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[3], labels[4]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn kmeans_separates_two_blobs() {
    let c = KmeansClusterer::new(4);
    let labels = c.cluster(&synth_two_blobs()).unwrap();
    assert_eq!(labels.len(), 16);
    let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
    assert_eq!(
        unique.len(),
        2,
        "two well-separated blobs give two clusters"
    );
    assert!(labels[..8].iter().all(|&l| l == labels[0]));
    assert!(labels[8..].iter().all(|&l| l == labels[8]));
    assert_ne!(labels[0], labels[8]);
}

#[test]
fn kmeans_fast_mode_separates_two_blobs() {
    let c = KmeansClusterer::new(4).fast_mode();
    let labels = c.cluster(&synth_two_blobs()).unwrap();
    let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
    assert_eq!(unique.len(), 2);
    assert_ne!(labels[0], labels[8]);
}

#[test]
fn kmeans_builder_overrides_are_applied() {
    // with_trials(0) clamps to 1; custom iteration cap still converges.
    let c = KmeansClusterer::new(4).with_max_iter(10).with_trials(0);
    let labels = c.cluster(&synth_two_blobs()).unwrap();
    assert_eq!(labels.len(), 16);
    let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
    assert_eq!(unique.len(), 2);
}
