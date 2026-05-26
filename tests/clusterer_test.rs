#![allow(clippy::unwrap_used)]
//! Integration test for `AhcClusterer` and `NmeScClusterer` on synthetic
//! clusters. Pure-CPU; runs in normal `cargo test` (no network or model required).

#![cfg(feature = "clusterer")]

use polyvoice::clusterer::{AhcClusterer, Clusterer};

#[cfg(feature = "spectral")]
use polyvoice::clusterer::NmeScClusterer;

fn synth_clusters_4(d: usize) -> Vec<Vec<f32>> {
    let centers: Vec<Vec<f32>> = (0..4)
        .map(|c| {
            let mut v = vec![0.0_f32; d];
            v[c] = 1.0;
            v
        })
        .collect();
    let mut all = Vec::new();
    for _ in 0..6 {
        for c in &centers {
            let mut perturbed = c.clone();
            perturbed[0] += 0.01;
            let n: f32 = perturbed.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut perturbed {
                *x /= n;
            }
            all.push(perturbed);
        }
    }
    all
}

#[test]
fn ahc_finds_four_clusters() {
    let c = AhcClusterer::default();
    let labels = c.cluster(&synth_clusters_4(8)).unwrap();
    let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
    assert!(
        (3..=5).contains(&unique.len()),
        "got {} clusters: {:?}",
        unique.len(),
        labels
    );
}

#[cfg(feature = "spectral")]
#[test]
fn nme_sc_finds_four_clusters() {
    let c = NmeScClusterer::default();
    let labels = c.cluster(&synth_clusters_4(8)).unwrap();
    let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
    assert!(
        (3..=5).contains(&unique.len()),
        "got {} clusters: {:?}",
        unique.len(),
        labels
    );
}
