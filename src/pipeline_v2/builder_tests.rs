use super::*;
use crate::models::Manifest;
use crate::pipeline_v2::mocks::{
    MockClusterer, MockEmbedder, MockSegmenter, PassThroughResegmenter,
};
use std::path::PathBuf;

fn fresh() -> PipelineBuilder {
    PipelineBuilder::new()
}

fn repo_file(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[test]
fn execution_provider_setter_overrides_config() {
    let b = fresh().execution_provider(crate::pipeline_v2::ExecutionProvider::Cpu);
    assert_eq!(
        b.config.execution_provider,
        crate::pipeline_v2::ExecutionProvider::Cpu
    );
}

#[test]
fn builder_default_profile_balanced() {
    let b = fresh();
    assert_eq!(b.config.profile, Profile::Balanced);
}

#[test]
fn builder_profile_setter() {
    let b = fresh().profile(Profile::Mobile);
    assert_eq!(b.config.profile, Profile::Mobile);
}

fn with_config(mutate: impl FnOnce(&mut PipelineConfig)) -> PipelineBuilder {
    let mut cfg = PipelineConfig::default();
    mutate(&mut cfg);
    fresh().config(cfg)
}

fn assert_invalid(mutate: impl FnOnce(&mut PipelineConfig), field: &str) {
    let err = with_config(mutate).validate().unwrap_err();
    match err {
        ConfigError::InvalidSetting { field: got, detail } => {
            assert_eq!(got, field, "{detail}");
            assert!(!detail.is_empty());
        }
        other => panic!("expected InvalidSetting for {field}, got {other:?}"),
    }
}

#[test]
fn validate_rejects_non_finite_and_out_of_range_before_registry() {
    assert_invalid(|c| c.min_speech_secs = f32::NAN, "min_speech_secs");
    assert_invalid(|c| c.max_gap_secs = f32::INFINITY, "max_gap_secs");
    assert_invalid(|c| c.min_speech_secs = -0.1, "min_speech_secs");
    assert_invalid(|c| c.max_speakers = 0, "max_speakers");
    assert_invalid(|c| c.min_cluster_size = 0, "min_cluster_size");
    assert_invalid(|c| c.embedder_pool_size = 0, "embedder_pool_size");
    assert_invalid(|c| c.embed_window_secs = Some(0.0), "embed_window_secs");
    assert_invalid(
        |c| c.embed_window_secs = Some(f32::NAN),
        "embed_window_secs",
    );
    assert_invalid(
        |c| {
            c.clusterer = ClustererKind::Ahc {
                threshold: f32::NAN,
            }
        },
        "clusterer.threshold",
    );
    assert_invalid(
        |c| c.clusterer = ClustererKind::Ahc { threshold: 1.5 },
        "clusterer.threshold",
    );
    assert_invalid(
        |c| c.sample_rate = crate::SampleRate::new(8_000).expect("8 kHz is a valid SampleRate"),
        "sample_rate",
    );
    assert_invalid(
        |c| c.execution_provider = crate::pipeline_v2::ExecutionProvider::Cuda,
        "execution_provider",
    );
    assert_invalid(
        |c| {
            c.experimental.binarization = Some(crate::segmentation::BinarizationConfig {
                onset: f32::NAN,
                ..crate::segmentation::BinarizationConfig::default()
            })
        },
        "binarization.onset",
    );
    assert_invalid(
        |c| {
            c.as_norm = Some(crate::clusterer::AsNormConfig {
                top_n: 1,
                cohort: crate::clusterer::CohortSource::ModelId("unused".into()),
            })
        },
        "as_norm.top_n",
    );
}

#[test]
fn validate_accepts_documented_boundaries_without_loading_models() {
    // No registry: a numeric pass still stops at MissingRegistry, never Load.
    let err = with_config(|c| {
        c.min_speech_secs = 0.0;
        c.max_gap_secs = 0.0;
        c.max_speakers = 1;
        c.min_cluster_size = 1;
        c.embed_window_secs = Some(0.5);
        c.clusterer = ClustererKind::Ahc { threshold: -1.0 };
        c.execution_provider = crate::pipeline_v2::ExecutionProvider::Cpu;
    })
    .validate()
    .unwrap_err();
    assert!(matches!(err, ConfigError::MissingRegistry { .. }));

    let err = with_config(|c| {
        c.profile = Profile::Custom;
        c.sample_rate = crate::SampleRate::new(8_000).expect("8 kHz");
    })
    .validate()
    .unwrap_err();
    assert!(
        matches!(err, ConfigError::MissingCustomComponent { .. }),
        "custom may use a non-16 kHz rate; got {err:?}"
    );
}

#[test]
fn validate_mobile_without_registry_errors() {
    let err = fresh().profile(Profile::Mobile).validate().unwrap_err();
    assert!(matches!(
        err,
        ConfigError::MissingRegistry {
            profile: Profile::Mobile
        }
    ));
}

#[test]
fn validate_custom_without_components_errors() {
    let err = fresh().profile(Profile::Custom).validate().unwrap_err();
    match err {
        ConfigError::MissingCustomComponent { missing } => {
            assert!(missing.contains(&"segmenter"));
            assert!(missing.contains(&"embedder"));
            assert!(missing.contains(&"clusterer"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn validate_custom_with_full_components_succeeds() {
    let b = fresh()
        .profile(Profile::Custom)
        .with_segmenter(Box::new(MockSegmenter::default()))
        .with_embedder(Box::new(MockEmbedder::default()))
        .with_clusterer(Box::new(MockClusterer::default()));
    b.validate().expect("custom + 3 components must validate");
}

#[test]
fn validate_balanced_with_custom_segmenter_errors() {
    let b = fresh()
        .profile(Profile::Balanced)
        .with_segmenter(Box::new(MockSegmenter::default()));
    let err = b.validate().unwrap_err();
    assert!(matches!(
        err,
        ConfigError::CustomComponentInProfile {
            offending: "segmenter",
            ..
        }
    ));
}

#[test]
fn validate_custom_with_registry_errors() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let registry = ModelRegistry::with_local_dir(tmp.path()).expect("local registry");
    let b = fresh()
        .profile(Profile::Custom)
        .with_segmenter(Box::new(MockSegmenter::default()))
        .with_embedder(Box::new(MockEmbedder::default()))
        .with_clusterer(Box::new(MockClusterer::default()))
        .with_models_from(registry);
    let err = b.validate().unwrap_err();
    assert!(matches!(err, ConfigError::RegistryInCustomProfile));
}

#[test]
fn embedder_pool_size_clamps_to_1() {
    let b = fresh().embedder_pool_size(0);
    assert_eq!(b.config.embedder_pool_size, 1);
}

#[test]
fn config_setter_replaces_config() {
    let cfg = PipelineConfig {
        max_speakers: 7,
        min_speech_secs: 0.5,
        ..PipelineConfig::default()
    };
    let b = fresh().config(cfg);
    assert_eq!(b.config.max_speakers, 7);
    assert!((b.config.min_speech_secs - 0.5).abs() < f32::EPSILON);
}

#[test]
fn resegment_overlap_setter() {
    let b = fresh().resegment_overlap(false);
    assert!(!b.config.resegment_overlap);
}

#[test]
fn max_speakers_setter() {
    let b = fresh().max_speakers(4);
    assert_eq!(b.config.max_speakers, 4);
}

#[test]
fn validate_fast_without_registry_errors() {
    let err = fresh().profile(Profile::Fast).validate().unwrap_err();
    assert!(matches!(
        err,
        ConfigError::MissingRegistry {
            profile: Profile::Fast
        }
    ));
}

#[test]
fn validate_balanced_with_custom_embedder_errors() {
    let b = fresh()
        .profile(Profile::Balanced)
        .with_embedder(Box::new(MockEmbedder::default()));
    let err = b.validate().unwrap_err();
    assert!(matches!(
        err,
        ConfigError::CustomComponentInProfile {
            profile: Profile::Balanced,
            offending: "embedder",
        }
    ));
}

#[test]
fn validate_balanced_with_custom_clusterer_errors() {
    let b = fresh()
        .profile(Profile::Balanced)
        .with_clusterer(Box::new(MockClusterer::default()));
    let err = b.validate().unwrap_err();
    assert!(matches!(
        err,
        ConfigError::CustomComponentInProfile {
            profile: Profile::Balanced,
            offending: "clusterer",
        }
    ));
}

#[test]
fn validate_custom_reports_only_missing_components() {
    let b = fresh()
        .profile(Profile::Custom)
        .with_segmenter(Box::new(MockSegmenter::default()));
    let err = b.validate().unwrap_err();
    match err {
        ConfigError::MissingCustomComponent { missing } => {
            assert_eq!(missing, vec!["embedder", "clusterer"]);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn config_error_display_messages() {
    let missing = ConfigError::MissingRegistry {
        profile: Profile::Mobile,
    };
    assert_eq!(
        missing.to_string(),
        "profile Mobile requires .with_models_from() call"
    );

    let custom_in_profile = ConfigError::CustomComponentInProfile {
        profile: Profile::Balanced,
        offending: "embedder",
    };
    assert_eq!(
        custom_in_profile.to_string(),
        "profile Balanced cannot accept .with_embedder() — Custom only"
    );

    let registry_in_custom = ConfigError::RegistryInCustomProfile;
    assert_eq!(
        registry_in_custom.to_string(),
        "Custom profile cannot accept .with_models_from() — supply components individually"
    );

    let missing_custom = ConfigError::MissingCustomComponent {
        missing: vec!["embedder"],
    };
    assert_eq!(
        missing_custom.to_string(),
        "Custom profile missing required components: [\"embedder\"]"
    );

    let unknown = ConfigError::UnknownModel {
        model_id: "vbx".to_owned(),
    };
    assert_eq!(unknown.to_string(), "ONNX model not found in registry: vbx");

    let load = ConfigError::Load {
        model_id: "powerset",
        source: Box::new(std::io::Error::other("boom")),
    };
    assert_eq!(load.to_string(), "failed to load model powerset: boom");
    assert!(std::error::Error::source(&load).is_some());

    let registry = ConfigError::Registry(RegistryError::CustomProfileUnresolvable);
    assert!(
        registry
            .to_string()
            .starts_with("registry resolution failed:")
    );
}

#[test]
fn build_custom_with_mocks_succeeds() {
    let p = fresh()
        .profile(Profile::Custom)
        .with_segmenter(Box::new(MockSegmenter::default()))
        .with_embedder(Box::new(MockEmbedder::default()))
        .with_clusterer(Box::new(MockClusterer::default()))
        .build()
        .expect("custom profile with all components builds");
    assert_eq!(p.config().profile, Profile::Custom);
}

#[test]
fn build_custom_propagates_optional_setters() {
    let p = fresh()
        .profile(Profile::Custom)
        .with_segmenter(Box::new(MockSegmenter::default()))
        .with_embedder(Box::new(MockEmbedder::default()))
        .with_clusterer(Box::new(MockClusterer::default()))
        .with_resegmenter(Box::new(PassThroughResegmenter))
        .resegment_overlap(false)
        .max_speakers(3)
        .embedder_pool_size(2)
        .build()
        .expect("custom build with explicit resegmenter");
    assert!(!p.config().resegment_overlap);
    assert_eq!(p.config().max_speakers, 3);
    assert_eq!(p.config().embedder_pool_size, 2);
}

#[test]
fn build_balanced_without_registry_errors() {
    let err = fresh()
        .profile(Profile::Balanced)
        .build()
        .err()
        .expect("build without registry must fail");
    assert!(matches!(
        err,
        ConfigError::MissingRegistry {
            profile: Profile::Balanced
        }
    ));
}

#[cfg(all(feature = "segmenter-native", feature = "embedder-native"))]
#[test]
fn build_native_unknown_embedder_model_errors() {
    let models = repo_file("models");
    let cache = if models.join("int8/powerset_int8.onnx").is_file() {
        models.join("int8")
    } else {
        models.clone()
    };
    if !cache.join("powerset_int8.onnx").is_file() || !cache.join("resnet34_int8.onnx").is_file() {
        eprintln!("skip: INT8 powerset/resnet missing under models/int8");
        return;
    }
    let registry = ModelRegistry::with_cache_dir(&cache).expect("registry");
    let mut cfg = crate::pipeline_v2::PipelineConfig::default();
    cfg.experimental.embedder_model = Some("not_a_real_embedder".into());
    let err = fresh()
        .config(cfg)
        .with_models_from(registry)
        .build()
        .err()
        .expect("unknown embedder id must fail");
    match err {
        ConfigError::UnknownModel { model_id } => {
            assert_eq!(model_id, "not_a_real_embedder");
        }
        other => panic!("expected UnknownModel, got {other:?}"),
    }
}

#[cfg(all(feature = "segmenter-native", feature = "embedder-native"))]
#[test]
fn build_native_with_local_models_succeeds() {
    let models = repo_file("models");
    let cache = if models.join("int8/powerset_int8.onnx").is_file() {
        models.join("int8")
    } else {
        models.clone()
    };
    if !cache.join("powerset_int8.onnx").is_file() || !cache.join("resnet34_int8.onnx").is_file() {
        eprintln!("skip: INT8 powerset/resnet missing under models/int8");
        return;
    }
    let registry = ModelRegistry::with_cache_dir(&cache).expect("registry");
    let p = fresh()
        .config(PipelineConfig {
            vbx_plda_dir: Some(repo_file("fixtures/vbx-plda")),
            ..PipelineConfig::default()
        })
        .with_models_from(registry)
        .build()
        .expect("native kernels build from local INT8 models");
    assert_eq!(p.config().profile, Profile::Balanced);
}

#[cfg(all(feature = "segmenter-native", feature = "embedder-native"))]
#[test]
fn native_pipeline_runs_short_sine() {
    let models = repo_file("models");
    let cache = if models.join("int8/powerset_int8.onnx").is_file() {
        models.join("int8")
    } else {
        models.clone()
    };
    if !cache.join("powerset_int8.onnx").is_file() || !cache.join("resnet34_int8.onnx").is_file() {
        eprintln!("skip: INT8 models missing");
        return;
    }
    let registry = ModelRegistry::with_cache_dir(&cache).expect("registry");
    let p = fresh()
        .config(PipelineConfig {
            vbx_plda_dir: Some(repo_file("fixtures/vbx-plda")),
            ..PipelineConfig::default()
        })
        .with_models_from(registry)
        .build()
        .expect("build");
    let n = 32_000usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin())
        .collect();
    let sr = crate::types::SampleRate::new(16_000).expect("sr");
    let result = p.run(&pcm, sr).expect("native pipeline run");
    eprintln!(
        "native run: turns={} speakers={}",
        result.turns.len(),
        result.num_speakers
    );
}

#[test]
fn build_manifest_without_profile_reports_registry_error() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    // A manifest with no [profiles.balanced] entry: profile resolution
    // fails before any model file is consulted.
    let manifest = Manifest::from_toml_str(
        r#"
        schema = "polyvoice-models-v2"
        [models.local_powerset]
        url      = "https://example.invalid/powerset_fp32.onnx"
        sha256   = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
        size     = 5992913
        filename = "powerset_fp32.onnx"
        "#,
    )
    .expect("manifest parses");
    let registry = ModelRegistry::with_manifest(manifest, tmp.path()).expect("registry");
    let err = fresh()
        .profile(Profile::Balanced)
        .with_models_from(registry)
        .build()
        .err()
        .expect("build must fail");
    // Kernels ask for `powerset_int8`; tract asks for `powerset_fp32_tract`.
    // Neither id is in this manifest, so resolution fails before a file read.
    let ConfigError::Load { source, .. } = &err else {
        panic!("expected stage-load failure, got {err:?}");
    };
    assert!(matches!(
        source.downcast_ref::<RegistryError>(),
        Some(RegistryError::ModelNotFound { .. })
    ));
}

// --- AS-norm / domain-profile wiring ---

/// Write a `(rows, dim) <f4` NPY cohort file for the AS-norm tests.
fn write_test_cohort(dir: &std::path::Path, rows: &[Vec<f32>]) -> PathBuf {
    let cols = rows[0].len();
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {cols}), }}",
        rows.len()
    );
    let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
    let header = format!("{dict}{}{}", " ".repeat(pad), "\n");
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for row in rows {
        for v in row {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let path = dir.join("cohort.npy");
    std::fs::write(&path, &bytes).expect("write test cohort");
    path
}

#[test]
fn resolve_clusterer_kind_domain_profile_overrides_ahc_threshold() {
    use crate::clusterer::domain;

    let resolve = |clusterer: ClustererKind, domain: Option<crate::clusterer::DomainProfile>| {
        resolve_clusterer_kind(&PipelineConfig {
            clusterer,
            domain,
            ..PipelineConfig::default()
        })
    };
    let ahc = ClustererKind::Ahc { threshold: 0.5 };

    assert_eq!(
        resolve(ahc, Some(domain::AMI)),
        ClustererKind::Ahc {
            threshold: domain::AMI.ahc_threshold
        },
        "AMI profile replaces the configured threshold"
    );
    // Deterministic: same config, same resolution.
    assert_eq!(
        resolve(ahc, Some(domain::AMI)),
        resolve(ahc, Some(domain::AMI))
    );

    assert_eq!(
        resolve(ahc, Some(domain::VOXCONVERSE)),
        ClustererKind::Ahc {
            threshold: domain::VOXCONVERSE.ahc_threshold
        },
        "VoxConverse profile replaces the configured threshold too"
    );
    // The VoxConverse raw threshold IS the shipped CLI default.
    assert_eq!(
        domain::VOXCONVERSE.ahc_threshold,
        crate::types::DEFAULT_AHC_THRESHOLD
    );
    assert_eq!(
        resolve(ahc, Some(domain::CALLHOME)),
        ClustererKind::Ahc {
            threshold: domain::CALLHOME.ahc_threshold
        }
    );

    // No domain → configured threshold preserved.
    assert_eq!(
        resolve(ClustererKind::Ahc { threshold: 0.42 }, None),
        ClustererKind::Ahc { threshold: 0.42 }
    );

    // Non-AHC kinds are never rewritten by a domain profile.
    assert_eq!(
        resolve(ClustererKind::NmeSc, Some(domain::AMI)),
        ClustererKind::NmeSc
    );
}

#[test]
fn resolve_clusterer_kind_picks_z_threshold_when_as_norm_enabled() {
    use crate::clusterer::{AsNormConfig, CohortSource, domain};

    let mut config = PipelineConfig {
        clusterer: ClustererKind::Ahc { threshold: 0.5 },
        domain: Some(domain::VOXCONVERSE),
        as_norm: Some(AsNormConfig {
            top_n: 100,
            cohort: CohortSource::Path(std::path::PathBuf::from("unused")),
        }),
        ..PipelineConfig::default()
    };
    assert_eq!(
        resolve_clusterer_kind(&config),
        ClustererKind::Ahc {
            threshold: domain::VOXCONVERSE.as_norm_threshold.unwrap()
        },
        "AS-norm runs on z-scores, so the profile's z-threshold applies"
    );

    // A profile without a calibrated z-threshold keeps the configured value.
    config.domain = Some(domain::CALLHOME);
    assert_eq!(
        resolve_clusterer_kind(&config),
        ClustererKind::Ahc { threshold: 0.5 }
    );

    // Same domain without AS-norm resolves to the raw-cosine threshold.
    config.as_norm = None;
    config.domain = Some(domain::AMI);
    assert_eq!(
        resolve_clusterer_kind(&config),
        ClustererKind::Ahc {
            threshold: domain::AMI.ahc_threshold
        }
    );
}

/// Scene where raw cosine and AS-norm z-scores disagree decisively: two
/// tight pairs (within-pair cosine ≈ 0.98, cross-pair ≈ 0), with a cohort
/// anti-aligned with both groups so every cohort stat mean is negative.
/// Then z(within) ≫ z(cross) ≫ 1, so a z-scale threshold above 1 merges
/// nothing under raw cosine but splits the pairs under AS-norm.
fn as_norm_discriminating_scene() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let embeddings = vec![
        vec![1.0, 0.1, 0.0, 0.0, 0.0],
        vec![1.0, -0.1, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.1, 0.0],
        vec![0.0, 0.0, 1.0, -0.1, 0.0],
    ];
    // Cohort rows ≈ -0.7·(axis0 + axis2) with jitter on axis 4, orthogonal
    // to every embedding, so each embedding sees a tight cluster of
    // negative cohort scores (std driven by the jitter only).
    let cohort: Vec<Vec<f32>> = (0..12)
        .map(|k| {
            let o = 0.05 * (k as f32 - 5.5);
            vec![-0.7, 0.0, -0.7, 0.0, o]
        })
        .collect();
    (embeddings, cohort)
}

#[test]
fn build_profile_clusterer_wraps_ahc_with_as_norm_only_when_enabled() {
    use crate::clusterer::asnorm::AsNormScorer;
    use crate::clusterer::{AsNormCohort, AsNormConfig, CohortSource};

    let tmp = tempfile::TempDir::new().expect("temp dir");
    let registry = ModelRegistry::with_cache_dir(tmp.path()).expect("registry");
    let (embeddings, cohort_rows) = as_norm_discriminating_scene();
    let cohort_path = write_test_cohort(tmp.path(), &cohort_rows);

    // Derive the z-scale decision threshold from the scene itself, then
    // confirm it sits above every raw cosine in the scene.
    use crate::ahc::AhcScorer;
    let cohort = AsNormCohort::from_rows(cohort_rows).expect("uniform test cohort");
    let scorer = AsNormScorer::new(&cohort, &embeddings, 10);
    let z_within = scorer.score(&embeddings[0], 0, &embeddings[1], 1);
    let z_cross = scorer.score(&embeddings[0], 0, &embeddings[2], 2);
    assert!(
        z_within > z_cross + 1.0,
        "scene must separate on the z-scale: within={z_within} cross={z_cross}"
    );
    let threshold = (z_within + z_cross) / 2.0;
    assert!(
        threshold > 1.0,
        "threshold {threshold} must exceed every raw cosine for the contrast to bite"
    );

    // Disabled: plain fixed-threshold AHC — nothing reaches a >1 cosine
    // threshold, so every embedding stays its own cluster.
    let plain_cfg = PipelineConfig {
        clusterer: ClustererKind::Ahc { threshold },
        ..PipelineConfig::default()
    };
    let plain = build_profile_clusterer(&plain_cfg, &registry).expect("plain ahc");
    let plain_labels = plain.cluster(&embeddings).expect("cluster");
    assert_eq!(plain_labels, vec![0, 1, 2, 3], "raw cosine merges nothing");

    // Enabled: the same numeric threshold now sits between the within- and
    // cross-speaker z-scores, so the two pairs merge into two speakers.
    let as_norm_cfg = PipelineConfig {
        clusterer: ClustererKind::Ahc { threshold },
        as_norm: Some(AsNormConfig {
            top_n: 10,
            cohort: CohortSource::Path(cohort_path),
        }),
        ..PipelineConfig::default()
    };
    let wrapped = build_profile_clusterer(&as_norm_cfg, &registry).expect("as-norm ahc");
    let labels = wrapped.cluster(&embeddings).expect("cluster");
    assert_eq!(labels, vec![0, 0, 1, 1], "as-norm recovers the two pairs");
}

#[test]
fn load_as_norm_cohort_missing_model_id_guides_to_explicit_path() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let manifest =
        Manifest::from_toml_str(r#"schema = "polyvoice-models-v2""#).expect("manifest parses");
    let registry = ModelRegistry::with_manifest(manifest, tmp.path()).expect("registry");
    let cfg = crate::clusterer::AsNormConfig {
        top_n: 10,
        cohort: crate::clusterer::CohortSource::ModelId(
            crate::clusterer::DEFAULT_ASNORM_COHORT_MODEL_ID.to_owned(),
        ),
    };
    // The env override would win over the empty manifest, so clear it under
    // the lock: a concurrent override test or the caller's shell must not
    // turn this offline failure into a success.
    let mut env = crate::test_env::lock();
    env.remove("POLYVOICE_ASNORM_COHORT");
    let err = load_as_norm_cohort(&cfg, &registry).expect_err("must fail offline");
    let msg = err.to_string();
    assert!(msg.contains("asnorm_cohort"), "{msg}");
    assert!(msg.contains("--cohort"), "{msg}");
}

#[test]
fn load_as_norm_cohort_env_override_wins_over_registry() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let cohort_path = write_test_cohort(tmp.path(), &[vec![1.0, 0.0], vec![0.0, 1.0]]);
    let manifest =
        Manifest::from_toml_str(r#"schema = "polyvoice-models-v2""#).expect("manifest parses");
    let registry = ModelRegistry::with_manifest(manifest, tmp.path()).expect("registry");
    let cfg = crate::clusterer::AsNormConfig {
        top_n: 2,
        // A model id the manifest does not contain: only the env override
        // can make this load succeed.
        cohort: crate::clusterer::CohortSource::ModelId("absent_cohort".to_owned()),
    };
    let mut env = crate::test_env::lock();
    env.set("POLYVOICE_ASNORM_COHORT", &cohort_path);
    let cohort = load_as_norm_cohort(&cfg, &registry).expect("env override supplies the cohort");
    assert_eq!(cohort.rows().len(), 2);
}

#[test]
fn load_as_norm_cohort_bad_file_reports_load_error() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let registry = ModelRegistry::with_cache_dir(tmp.path()).expect("registry");
    let cfg = crate::clusterer::AsNormConfig {
        top_n: 10,
        cohort: crate::clusterer::CohortSource::Path(tmp.path().join("missing.npy")),
    };
    let err = load_as_norm_cohort(&cfg, &registry).expect_err("missing file must fail");
    assert!(matches!(
        err,
        ConfigError::Load {
            model_id: "asnorm_cohort",
            ..
        }
    ));
}

#[test]
fn local_dir_and_registry_agree_on_the_same_artifacts() {
    let models = repo_file("models/int8");
    let plda = repo_file("fixtures/vbx-plda");
    if !models.join("powerset_int8.onnx").is_file()
        || !models.join("resnet34_int8.onnx").is_file()
        || !plda.join("plda_transform.npy").is_file()
    {
        eprintln!("skip: local INT8 pair or PLDA fixture missing");
        return;
    }
    let local = ModelRegistry::with_local_dir(&models).expect("local dir");
    assert!(!local.allows_download());
    let cache = ModelRegistry::with_cache_dir(&models).expect("cache dir");
    let lp = local.ensure("powerset_int8").expect("local powerset");
    let cp = cache.ensure("powerset_int8").expect("cache powerset");
    assert_eq!(lp, cp);
    let cfg = PipelineConfig {
        vbx_plda_dir: Some(plda),
        // One embedder session: the pool must not reorder reductions.
        embedder_pool_size: 1,
        ..PipelineConfig::default()
    };
    let run = |registry: ModelRegistry| {
        let pipeline = fresh()
            .config(cfg.clone())
            .with_models_from(registry)
            .build()
            .expect("pipeline from verified local artifacts");
        let sr = crate::SampleRate::new(16_000).expect("16 kHz");
        let samples: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.2).collect();
        pipeline.run(&samples, sr).expect("run")
    };
    let turns = |r: &crate::DiarizationResult| {
        r.turns
            .iter()
            .map(|t| (t.speaker.0, t.time.start, t.time.end))
            .collect::<Vec<_>>()
    };
    let local_again = ModelRegistry::with_local_dir(&models).expect("local dir");
    let a = run(local);
    let a2 = run(local_again);
    assert_eq!(turns(&a), turns(&a2), "local path must be deterministic");
    let b = run(cache);
    assert_eq!(turns(&a), turns(&b));
}
