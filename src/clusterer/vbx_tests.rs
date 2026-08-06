use super::*;
use ndarray::{Array2, array};

#[test]
fn from_dir_missing_returns_error() {
    // A non-existent PLDA dir must surface a clear error, not panic.
    // (VbxClusterer is not Debug, so match instead of expect_err.)
    let err = match VbxClusterer::from_dir(std::path::Path::new("/no/such/plda/dir"), 20) {
        Ok(_) => panic!("missing PLDA dir must error"),
        Err(e) => e,
    };
    assert!(
        matches!(err, ClustererError::Plda(_)),
        "expected the typed PLDA error, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("/no/such/plda/dir"),
        "error should name the PLDA dir: {msg}"
    );
}

#[test]
fn two_clusters_with_vbx() {
    // Two well-separated clusters; VBx must keep them distinct from the seed.
    let features = array![
        [10.0, 0.0],
        [10.1, 0.1],
        [9.9, -0.1],
        [-10.0, 0.0],
        [-10.1, 0.1],
        [-9.9, -0.1],
    ];
    let phi = array![1.0, 1.0];
    let mut gamma_init = Array2::zeros((6, 2));
    for t in 0..3 {
        gamma_init[[t, 0]] = 0.999;
        gamma_init[[t, 1]] = 0.001;
    }
    for t in 3..6 {
        gamma_init[[t, 0]] = 0.001;
        gamma_init[[t, 1]] = 0.999;
    }
    let (gamma, _pi) = vbx(
        &features.view(),
        &phi.view(),
        &gamma_init,
        &VbxConfig::default(),
    );
    let labels = hard_labels(&gamma);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[3], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn vbx_hmm_keeps_two_clusters_and_valid_posteriors() {
    // Same two well-separated clusters, but with the temporal HMM path on.
    let features = array![
        [10.0, 0.0],
        [10.1, 0.1],
        [9.9, -0.1],
        [-10.0, 0.0],
        [-10.1, 0.1],
        [-9.9, -0.1],
    ];
    let phi = array![1.0, 1.0];
    let mut gamma_init = Array2::zeros((6, 2));
    for t in 0..3 {
        gamma_init[[t, 0]] = 0.999;
        gamma_init[[t, 1]] = 0.001;
    }
    for t in 3..6 {
        gamma_init[[t, 0]] = 0.001;
        gamma_init[[t, 1]] = 0.999;
    }
    let cfg = VbxConfig {
        loop_prob: 0.9,
        ..VbxConfig::default()
    };
    let (gamma, _pi) = vbx(&features.view(), &phi.view(), &gamma_init, &cfg);
    // Posteriors per frame are a valid distribution (sum ~ 1, finite).
    for row in gamma.rows() {
        let s: f32 = row.sum();
        assert!(
            s.is_finite() && (s - 1.0).abs() < 1e-3,
            "row sum {s} not ~1"
        );
    }
    let labels = hard_labels(&gamma);
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[3], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn gamma_init_is_smoothed_one_hot() {
    let gamma = build_gamma_init(&[0, 0, 1], 7.0);
    assert_eq!(gamma.dim(), (3, 2));
    assert!(gamma[[0, 0]] > gamma[[0, 1]]);
    assert!(gamma[[2, 1]] > gamma[[2, 0]]);
}

#[test]
fn vbx_prunes_redundant_seed_speaker() {
    // One true cluster split across two seed speakers — VBx's prior should
    // collapse the hard labels to a single speaker (auto-count downward).
    let features = array![[5.0, 0.0], [5.1, 0.1], [4.9, -0.1], [5.05, 0.05]];
    let phi = array![1.0, 1.0];
    let mut gamma_init = Array2::zeros((4, 2));
    gamma_init[[0, 0]] = 0.99;
    gamma_init[[0, 1]] = 0.01;
    gamma_init[[1, 0]] = 0.99;
    gamma_init[[1, 1]] = 0.01;
    gamma_init[[2, 0]] = 0.01;
    gamma_init[[2, 1]] = 0.99;
    gamma_init[[3, 0]] = 0.01;
    gamma_init[[3, 1]] = 0.99;
    let cfg = VbxConfig::default();
    let (gamma, _pi) = vbx(&features.view(), &phi.view(), &gamma_init, &cfg);
    let labels = hard_labels(&gamma);
    let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        1,
        "one acoustic cluster must collapse to one speaker"
    );
}

#[test]
fn hard_labels_are_compact() {
    let gamma = array![[0.1, 0.9, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
    // Used columns are 1 and 2 → compacted to 0 and 1.
    let labels = hard_labels(&gamma);
    assert_eq!(labels, vec![0, 1, 1]);
}

#[test]
fn gmm_mode_helpers() {
    let cfg = VbxConfig::default();
    assert!(cfg.is_gmm(), "default fixture config is GMM (loop_prob=0)");
    let hmm = cfg.hmm(0.9);
    assert!(!hmm.is_gmm());
    assert!((hmm.loop_prob - 0.9).abs() < 1e-12);
    let back = hmm.gmm();
    assert!(back.is_gmm());
    assert_eq!(back.loop_prob, 0.0);
}

#[test]
fn gmm_and_hmm_agree_on_shuffled_order_for_separated_clusters() {
    // When frames of two speakers are interleaved (non-contiguous), GMM-VBx
    // is the correct model; HMM with high loop_prob may over-smooth. Both
    // must still recover two clusters on strongly separated features.
    let features = array![
        [10.0, 0.0],
        [-10.0, 0.0],
        [10.1, 0.1],
        [-10.1, 0.1],
        [9.9, -0.1],
        [-9.9, -0.1],
    ];
    let phi = array![1.0, 1.0];
    let mut gamma_init = Array2::zeros((6, 2));
    for t in [0, 2, 4] {
        gamma_init[[t, 0]] = 0.999;
        gamma_init[[t, 1]] = 0.001;
    }
    for t in [1, 3, 5] {
        gamma_init[[t, 0]] = 0.001;
        gamma_init[[t, 1]] = 0.999;
    }
    let gmm_cfg = VbxConfig::default().gmm();
    let (gamma_gmm, _) = vbx(&features.view(), &phi.view(), &gamma_init, &gmm_cfg);
    let labels_gmm = hard_labels(&gamma_gmm);
    assert_eq!(labels_gmm[0], labels_gmm[2]);
    assert_eq!(labels_gmm[0], labels_gmm[4]);
    assert_eq!(labels_gmm[1], labels_gmm[3]);
    assert_eq!(labels_gmm[1], labels_gmm[5]);
    assert_ne!(labels_gmm[0], labels_gmm[1]);
}

/// Checked-in PLDA fixtures (256-d → 128-d) for offline clusterer tests.
fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/vbx-plda")
}

fn fixture_clusterer(max_speakers: usize) -> VbxClusterer {
    VbxClusterer::from_dir(&fixture_dir(), max_speakers).expect("load fixture plda")
}

/// Deterministic 256-d pseudo-embedding: ones on the `parity` slots plus a
/// small fixed jitter, zeros elsewhere. Two parities give two well-separated
/// directions that any sane clustering must keep apart.
fn synth_emb(parity: usize, seed: usize) -> Vec<f32> {
    (0..256)
        .map(|i| {
            let jitter = (((i * 31 + seed * 17) % 13) as f32 - 6.0) * 0.005;
            if i % 2 == parity {
                1.0 + jitter
            } else {
                jitter
            }
        })
        .collect()
}

/// Two groups of `per_group` embeddings each (parity 0 then parity 1).
fn two_group_embeddings(per_group: usize) -> Vec<Vec<f32>> {
    let mut embs = Vec::new();
    for s in 0..per_group {
        embs.push(synth_emb(0, s));
    }
    for s in 0..per_group {
        embs.push(synth_emb(1, 100 + s));
    }
    embs
}

#[test]
fn clusterer_handles_empty_single_and_dim_mismatch() {
    let c = fixture_clusterer(4);
    assert_eq!(c.cluster(&[]).unwrap(), Vec::<usize>::new());
    assert_eq!(c.cluster(&[synth_emb(0, 0)]).unwrap(), vec![0]);

    let err = c.cluster(&[synth_emb(0, 0), vec![0.0; 128]]).unwrap_err();
    match err {
        ClustererError::DimMismatch {
            expected,
            actual,
            index,
        } => {
            assert_eq!(expected, 256);
            assert_eq!(actual, 128);
            assert_eq!(index, 1);
        }
        other => panic!("expected DimMismatch, got {other:?}"),
    }
}

#[test]
fn clusterer_trait_surface() {
    let c = fixture_clusterer(8);
    assert_eq!(c.max_clusters(), 8);
    assert!(
        c.wants_raw_embeddings(),
        "PLDA mean-centering needs the original embedding scale"
    );
    // max_speakers is clamped to at least one.
    let plda = PldaModel::from_dir(&fixture_dir()).unwrap();
    let one = VbxClusterer::new(plda, VbxConfig::default(), 0.5, 0, 128, 4.88);
    assert_eq!(one.max_clusters(), 1);
}

#[test]
fn clusterer_separates_two_groups_gmm_and_hmm() {
    let embs = two_group_embeddings(3);
    for gmm in [true, false] {
        let c = fixture_clusterer(4).with_gmm_mode(gmm);
        let labels = c.cluster(&embs).unwrap();
        assert_eq!(labels.len(), embs.len());
        let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(distinct.len(), 2, "gmm={gmm}: labels {labels:?}");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }
}

#[test]
fn clusterer_reassigns_short_embeddings_by_nearest_centroid() {
    // Default config filters embeddings shorter than 1.6 s out of AHC/VBx
    // and reassigns them afterward to the nearest PLDA-feature centroid.
    let mut embs = two_group_embeddings(2);
    embs.push(synth_emb(0, 50)); // short, group A
    embs.push(synth_emb(1, 51)); // short, group B
    let durations = vec![5.0, 5.0, 5.0, 5.0, 0.5, 0.5];
    let c = fixture_clusterer(4);
    let labels = c.cluster_with_durations(&embs, &durations).unwrap();
    assert_eq!(labels.len(), 6);
    assert_eq!(labels[4], labels[0], "short A embedding joins group A");
    assert_eq!(labels[5], labels[2], "short B embedding joins group B");
}

#[test]
fn clusterer_ignores_misaligned_durations() {
    // A durations slice that does not align 1:1 with the embeddings disables
    // the short-segment filter entirely.
    let embs = two_group_embeddings(2);
    let durations = vec![0.1, 0.1]; // too short to align with 4 embeddings
    let c = fixture_clusterer(4);
    let labels = c.cluster_with_durations(&embs, &durations).unwrap();
    assert_eq!(labels.len(), 4);
    let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
    assert_eq!(distinct.len(), 2);
}

#[test]
fn clusterer_all_short_falls_back_to_kept_path() {
    // Every embedding below the duration floor: nothing left for AHC/VBx.
    let embs = two_group_embeddings(2);
    let durations = vec![0.1; 4];
    let c = fixture_clusterer(4);
    let labels = c.cluster_with_durations(&embs, &durations).unwrap();
    assert_eq!(labels.len(), 4);
}

#[test]
fn clusterer_with_ahc_established_min_members() {
    // cAHC-ASC stop: refuse merges between two established clusters.
    let embs = two_group_embeddings(3);
    let c = fixture_clusterer(4).with_ahc_established_min_members(2);
    let labels = c.cluster(&embs).unwrap();
    assert_eq!(labels.len(), 6);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[3], labels[4]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn auto_gmm_for_windowed_forces_gmm_only_when_windowed() {
    let c = fixture_clusterer(4);
    assert!(!c.config.is_gmm(), "default clusterer config is HMM-VBx");
    let windowed = c.auto_gmm_for_windowed(true);
    assert!(windowed.config.is_gmm());
    let contiguous = fixture_clusterer(4).auto_gmm_for_windowed(false);
    assert!(!contiguous.config.is_gmm());
}

#[test]
fn builder_overrides_apply() {
    let cfg = VbxConfig {
        fa: 0.11,
        ..VbxConfig::default()
    };
    let c = fixture_clusterer(4)
        .with_config(cfg)
        .with_min_embedding_secs(-3.0)
        .with_ahc_established_min_members(0);
    assert!((c.config.fa - 0.11).abs() < 1e-12);
    // Negative durations floor clamps to 0 = filtering disabled.
    assert_eq!(c.min_embedding_secs, 0.0);
    assert_eq!(c.ahc_established_min_members, 0);
}

#[test]
fn from_dir_with_config_overrides_defaults() {
    let config = VbxClustererConfig {
        ahc_threshold: 0.9,
        min_embedding_secs: 0.0,
        ..VbxClustererConfig::default()
    };
    let c = VbxClusterer::from_dir_with_config(&fixture_dir(), 3, config).unwrap();
    assert_eq!(c.max_clusters(), 3);
    assert!((c.ahc_threshold - 0.9).abs() < 1e-6);
    assert_eq!(c.min_embedding_secs, 0.0);
}

#[test]
fn clusterer_config_defaults_are_the_dev_tuning() {
    let d = VbxClustererConfig::default();
    assert!((d.vbx.fa - 0.3).abs() < 1e-12);
    assert!((d.vbx.loop_prob - 0.9).abs() < 1e-12);
    assert!((d.ahc_threshold - 0.5).abs() < 1e-6);
    assert!((d.emb_scale - 4.88).abs() < 1e-6);
    assert!((d.min_embedding_secs - 1.6).abs() < 1e-12);
    assert_eq!(d.ahc_established_min_members, 0);
}

#[test]
fn from_env_overlays_valid_values_and_ignores_malformed() {
    // Edition 2024 marks env mutation unsafe; this test is the only one in
    // the crate touching these variables, so there is nothing to race with.
    unsafe {
        std::env::set_var("POLYVOICE_VBX_FA", "0.42");
        std::env::set_var("POLYVOICE_VBX_FB", "not-a-float");
        std::env::set_var("POLYVOICE_VBX_LOOP_PROB", "0.5");
        std::env::set_var("POLYVOICE_VBX_AHC_THRESHOLD", "0.7");
        std::env::set_var("POLYVOICE_VBX_EMB_SCALE", "2.5");
        std::env::set_var("POLYVOICE_VBX_MIN_EMB_SECS", "2.0");
        std::env::set_var("POLYVOICE_VBX_AHC_ASC_MEMBERS", "3");
    }
    let c = VbxClustererConfig::from_env();
    unsafe {
        for k in [
            "POLYVOICE_VBX_FA",
            "POLYVOICE_VBX_FB",
            "POLYVOICE_VBX_LOOP_PROB",
            "POLYVOICE_VBX_AHC_THRESHOLD",
            "POLYVOICE_VBX_EMB_SCALE",
            "POLYVOICE_VBX_MIN_EMB_SECS",
            "POLYVOICE_VBX_AHC_ASC_MEMBERS",
        ] {
            std::env::remove_var(k);
        }
    }
    let d = VbxClustererConfig::default();
    assert!((c.vbx.fa - 0.42).abs() < 1e-12);
    assert!(
        (c.vbx.fb - d.vbx.fb).abs() < 1e-12,
        "malformed fb keeps default"
    );
    assert!((c.vbx.loop_prob - 0.5).abs() < 1e-12);
    assert!((c.ahc_threshold - 0.7).abs() < 1e-6);
    assert!((c.emb_scale - 2.5).abs() < 1e-6);
    assert!((c.min_embedding_secs - 2.0).abs() < 1e-12);
    assert_eq!(c.ahc_established_min_members, 3);

    // With nothing set, from_env reproduces the defaults.
    let c2 = VbxClustererConfig::from_env();
    assert!((c2.vbx.fa - d.vbx.fa).abs() < 1e-12);
    assert!((c2.emb_scale - d.emb_scale).abs() < 1e-6);
}

#[test]
fn logsumexp_handles_all_negative_infinity() {
    let v = Array1::from_vec(vec![f64::NEG_INFINITY; 3]);
    assert_eq!(logsumexp_f64(&v.view()), f64::NEG_INFINITY);
    let mixed = Array1::from_vec(vec![f64::NEG_INFINITY, 0.0]);
    assert!((logsumexp_f64(&mixed.view()) - 0.0).abs() < 1e-12);
}

#[test]
fn gamma_init_negative_smoothing_is_hard_one_hot() {
    let gamma = build_gamma_init(&[1, 0, 1], -1.0);
    assert_eq!(gamma.dim(), (3, 2));
    assert_eq!(gamma[[0, 1]], 1.0);
    assert_eq!(gamma[[0, 0]], 0.0);
    assert_eq!(gamma[[1, 0]], 1.0);
}

#[test]
fn gamma_init_empty_labels_yields_empty_seed() {
    let gamma = build_gamma_init(&[], 7.0);
    assert_eq!(gamma.nrows(), 0);
}

#[test]
fn hard_labels_single_column_is_all_zero() {
    let gamma = array![[0.3], [0.9], [0.1]];
    assert_eq!(hard_labels(&gamma), vec![0, 0, 0]);
}

#[test]
fn vbx_stops_early_with_loose_epsilon() {
    // A huge epsilon forces the convergence break on the second iteration;
    // the output must still be a valid responsibility matrix.
    let features = array![[5.0, 0.0], [5.1, 0.1], [-5.0, 0.0], [-5.1, 0.1]];
    let phi = array![1.0, 1.0];
    let gamma_init = build_gamma_init(&[0, 0, 1, 1], 7.0);
    let cfg = VbxConfig {
        epsilon: 1e9,
        ..VbxConfig::default()
    };
    let (gamma, pi) = vbx(&features.view(), &phi.view(), &gamma_init, &cfg);
    assert!(gamma.iter().all(|v| v.is_finite()));
    assert!(pi.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn vbx_single_iteration_still_returns_distribution() {
    let features = array![[5.0, 0.0], [-5.0, 0.0]];
    let phi = array![1.0, 1.0];
    let gamma_init = build_gamma_init(&[0, 1], 7.0);
    let cfg = VbxConfig {
        max_iters: 1,
        ..VbxConfig::default()
    };
    let (gamma, _pi) = vbx(&features.view(), &phi.view(), &gamma_init, &cfg);
    for row in gamma.rows() {
        let s: f32 = row.sum();
        assert!((s - 1.0).abs() < 1e-3, "row sum {s} not ~1");
    }
}

#[test]
fn vbx_hmm_prunes_redundant_speaker_prior() {
    // HMM path: one acoustic cluster seeded as two speakers — the hard
    // labels must collapse to a single speaker even though the temporal
    // prior keeps both pi entries well above zero.
    let features = array![[5.0, 0.0], [5.1, 0.1], [4.9, -0.1], [5.05, 0.05]];
    let phi = array![1.0, 1.0];
    let gamma_init = build_gamma_init(&[0, 0, 1, 1], 7.0);
    let cfg = VbxConfig {
        loop_prob: 0.9,
        ..VbxConfig::default()
    };
    let (gamma, pi) = vbx(&features.view(), &phi.view(), &gamma_init, &cfg);
    assert!(
        pi.iter().all(|v| v.is_finite() && *v >= 0.0),
        "pi must stay a valid distribution: {pi:?}"
    );
    let labels = hard_labels(&gamma);
    let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        1,
        "one acoustic cluster must collapse to one speaker: {labels:?}"
    );
}

/// Offline: seed the registry cache from the checked-in fixtures and load
/// via [`VbxClusterer::from_registry`] (no network).
#[cfg(feature = "download")]
#[test]
fn from_registry_loads_fixture_cache() {
    use crate::models::{ModelRegistry, VBX_PLDA_ARTIFACTS};
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let fixture_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/vbx-plda");
    for art in VBX_PLDA_ARTIFACTS {
        std::fs::copy(
            fixture_dir.join(art.filename),
            tmp.path().join(art.filename),
        )
        .unwrap();
    }
    let registry = ModelRegistry::with_cache_dir(tmp.path()).unwrap();
    let clusterer = VbxClusterer::from_registry(&registry, 8)
        .expect("from_registry must load fixture-seeded cache");
    assert_eq!(clusterer.max_clusters(), 8);
}
