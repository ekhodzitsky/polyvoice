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
