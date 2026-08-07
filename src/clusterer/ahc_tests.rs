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

#[test]
fn ahc_new_clamps_zero_max_clusters_to_one() {
    assert_eq!(AhcClusterer::new(0).max_clusters(), 1);
    assert_eq!(AhcClusterer::new(64).max_clusters(), 64);
    assert_eq!(AhcClusterer::default().max_clusters(), 64);
}

#[test]
fn ahc_with_threshold_separates_well_separated_clusters() {
    let c = AhcClusterer::with_threshold(0, 0.5);
    let labels = c.cluster(&synth_two_clusters()).unwrap();
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[3], labels[4]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn ahc_with_threshold_respects_max_clusters_cap() {
    // Ceiling of 1 forces every embedding into a single cluster even
    // though the data clearly holds two groups.
    let c = AhcClusterer::with_threshold(1, 0.5);
    let labels = c.cluster(&synth_two_clusters()).unwrap();
    assert_eq!(labels.len(), 6);
    assert!(
        labels.iter().all(|&l| l == labels[0]),
        "max_clusters = 1 must collapse to one cluster"
    );
}
