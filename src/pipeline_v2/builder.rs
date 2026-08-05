//! `PipelineBuilder` + `ConfigError`: validates a [`PipelineConfig`] and the
//! injected segmenter/embedder/clusterer components before building a
//! `Pipeline`.

use crate::clusterer::Clusterer;
use crate::embedder::Embedder;
use crate::models::{ModelRegistry, RegistryError};
use crate::pipeline_v2::config::PipelineConfig;
use crate::resegmentation::Resegmenter;
use crate::segmentation::Segmenter;
use crate::types::Profile;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("profile {profile:?} requires .with_models_from() call")]
    MissingRegistry { profile: Profile },

    #[error("profile {profile:?} cannot accept .with_{offending}() — Custom only")]
    CustomComponentInProfile {
        profile: Profile,
        offending: &'static str,
    },

    #[error("Custom profile cannot accept .with_models_from() — supply components individually")]
    RegistryInCustomProfile,

    #[error("Custom profile missing required components: {missing:?}")]
    MissingCustomComponent { missing: Vec<&'static str> },

    #[error("ONNX model not found in registry: {model_id}")]
    UnknownModel { model_id: String },

    #[error("failed to load model {model_id}: {source}")]
    Load {
        model_id: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("registry resolution failed: {0}")]
    Registry(#[from] RegistryError),
}

pub struct PipelineBuilder {
    pub(crate) config: PipelineConfig,
    pub(crate) registry: Option<ModelRegistry>,
    pub(crate) custom_segmenter: Option<Box<dyn Segmenter>>,
    pub(crate) custom_embedder: Option<Box<dyn Embedder>>,
    pub(crate) custom_clusterer: Option<Box<dyn Clusterer>>,
    pub(crate) custom_resegmenter: Option<Box<dyn Resegmenter>>,
}

impl PipelineBuilder {
    pub(crate) fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
            registry: None,
            custom_segmenter: None,
            custom_embedder: None,
            custom_clusterer: None,
            custom_resegmenter: None,
        }
    }

    pub fn config(mut self, cfg: PipelineConfig) -> Self {
        self.config = cfg;
        self
    }

    pub fn profile(mut self, p: Profile) -> Self {
        self.config.profile = p;
        self
    }

    pub fn with_models_from(mut self, r: ModelRegistry) -> Self {
        self.registry = Some(r);
        self
    }

    pub fn with_segmenter(mut self, s: Box<dyn Segmenter>) -> Self {
        self.custom_segmenter = Some(s);
        self
    }

    pub fn with_embedder(mut self, e: Box<dyn Embedder>) -> Self {
        self.custom_embedder = Some(e);
        self
    }

    pub fn with_clusterer(mut self, c: Box<dyn Clusterer>) -> Self {
        self.custom_clusterer = Some(c);
        self
    }

    pub fn with_resegmenter(mut self, r: Box<dyn Resegmenter>) -> Self {
        self.custom_resegmenter = Some(r);
        self
    }

    pub fn resegment_overlap(mut self, on: bool) -> Self {
        self.config.resegment_overlap = on;
        self
    }

    pub fn embedder_pool_size(mut self, n: usize) -> Self {
        self.config.embedder_pool_size = n.max(1);
        self
    }

    pub fn max_speakers(mut self, n: u8) -> Self {
        self.config.max_speakers = n;
        self
    }

    /// Override the execution provider (defaults to
    /// `ExecutionProvider::auto()` via `PipelineConfig::default`).
    pub fn execution_provider(mut self, ep: crate::onnx::ExecutionProvider) -> Self {
        self.config.execution_provider = ep;
        self
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        match self.config.profile {
            Profile::Mobile | Profile::Balanced | Profile::Fast => {
                if self.custom_segmenter.is_some() {
                    return Err(ConfigError::CustomComponentInProfile {
                        profile: self.config.profile,
                        offending: "segmenter",
                    });
                }
                if self.custom_embedder.is_some() {
                    return Err(ConfigError::CustomComponentInProfile {
                        profile: self.config.profile,
                        offending: "embedder",
                    });
                }
                if self.custom_clusterer.is_some() {
                    return Err(ConfigError::CustomComponentInProfile {
                        profile: self.config.profile,
                        offending: "clusterer",
                    });
                }
                if self.registry.is_none() {
                    return Err(ConfigError::MissingRegistry {
                        profile: self.config.profile,
                    });
                }
            }
            Profile::Custom => {
                if self.registry.is_some() {
                    return Err(ConfigError::RegistryInCustomProfile);
                }
                let mut missing: Vec<&'static str> = Vec::new();
                if self.custom_segmenter.is_none() {
                    missing.push("segmenter");
                }
                if self.custom_embedder.is_none() {
                    missing.push("embedder");
                }
                if self.custom_clusterer.is_none() {
                    missing.push("clusterer");
                }
                if !missing.is_empty() {
                    return Err(ConfigError::MissingCustomComponent { missing });
                }
            }
        }
        Ok(())
    }
}

use crate::pipeline_v2::Pipeline;
use crate::pipeline_v2::config::ClustererKind;
use crate::resegmentation::OverlapResegmenter;

/// Resolve the effective clusterer kind: a per-domain profile replaces the
/// configured AHC merge threshold with the profile's calibrated value. This
/// is the library contract — `PipelineConfig.domain` always wins over
/// `PipelineConfig.clusterer`'s threshold (the CLI inverts the precedence for
/// an explicit `--threshold` by simply not setting `domain`). Profiles are
/// data, so this stays a pure lookup; other clusterer kinds are unaffected.
/// The raw and AS-norm z scales differ, so the profile picks the field
/// matching the active scorer; a profile without a calibrated z-threshold
/// keeps the configured threshold unchanged.
pub(crate) fn resolve_clusterer_kind(config: &PipelineConfig) -> ClustererKind {
    match (config.clusterer, config.domain) {
        (ClustererKind::Ahc { threshold }, Some(domain)) => {
            let threshold = if config.as_norm.is_some() {
                domain.as_norm_threshold.unwrap_or(threshold)
            } else {
                domain.ahc_threshold
            };
            ClustererKind::Ahc { threshold }
        }
        (kind, _) => kind,
    }
}

/// Load the AS-norm imposter cohort. Resolution mirrors the VBx PLDA chain:
/// explicit path → `POLYVOICE_ASNORM_COHORT` env → registry download. The
/// builder is the library's single env-resolution point.
fn load_as_norm_cohort(
    as_norm: &crate::clusterer::AsNormConfig,
    registry: &ModelRegistry,
) -> Result<crate::clusterer::AsNormCohort, ConfigError> {
    use crate::clusterer::CohortSource;
    let path = match &as_norm.cohort {
        CohortSource::Path(p) => p.clone(),
        CohortSource::ModelId(id) => match std::env::var_os("POLYVOICE_ASNORM_COHORT") {
            Some(p) => std::path::PathBuf::from(p),
            None => registry.ensure(id).map_err(|e| match e {
                RegistryError::ModelNotFound { .. } => ConfigError::Load {
                    model_id: "asnorm_cohort",
                    source: std::io::Error::other(format!(
                        "cohort model '{id}' is not in the manifest; pass an explicit cohort \
                         file (CLI: --cohort) or set POLYVOICE_ASNORM_COHORT"
                    ))
                    .into(),
                },
                other => ConfigError::Registry(other),
            })?,
        },
    };
    crate::clusterer::AsNormCohort::from_npy(&path).map_err(|e| ConfigError::Load {
        model_id: "asnorm_cohort",
        source: Box::new(e),
    })
}

/// Construct the clusterer for the profile path: domain-profile threshold
/// resolution, optional AS-norm decoration (fixed-threshold AHC only), and
/// the VBx PLDA fallback chain.
fn build_profile_clusterer(
    config: &PipelineConfig,
    registry: &ModelRegistry,
) -> Result<Box<dyn Clusterer>, ConfigError> {
    match resolve_clusterer_kind(config) {
        ClustererKind::Ahc { threshold } => {
            let max = config.max_speakers as usize;
            match &config.as_norm {
                Some(as_norm) => {
                    // AS-norm decorates the fixed-threshold AHC scoring only;
                    // the auto-threshold path derives its threshold from the
                    // raw matrix and is never wrapped.
                    let cohort = load_as_norm_cohort(as_norm, registry)?;
                    Ok(Box::new(crate::clusterer::AsNormClusterer::new(
                        max,
                        threshold,
                        cohort,
                        as_norm.top_n,
                    )))
                }
                None => Ok(Box::new(crate::clusterer::AhcClusterer::with_threshold(
                    max, threshold,
                ))),
            }
        }
        #[cfg(feature = "spectral")]
        ClustererKind::NmeSc => Ok(Box::new(crate::clusterer::NmeScClusterer::new(
            config.max_speakers as usize,
        ))),
        #[cfg(not(feature = "spectral"))]
        ClustererKind::NmeSc => Ok(Box::new(crate::clusterer::AhcClusterer::with_threshold(
            config.max_speakers as usize,
            config.profile.default_threshold(),
        ))),
        #[cfg(feature = "vbx")]
        ClustererKind::Vbx => {
            let max = config.max_speakers as usize;
            // PLDA resolution order: explicit `vbx_plda_dir` →
            // `POLYVOICE_VBX_PLDA_DIR` env → registry download.
            // This is the library's single env-resolution point;
            // `pipeline_v2` always has `download`, so the registry
            // fallback is available.
            let mut vbx = match &config.vbx_plda_dir {
                Some(dir) => crate::clusterer::vbx::VbxClusterer::from_dir(dir, max),
                None => match std::env::var_os("POLYVOICE_VBX_PLDA_DIR") {
                    Some(dir) => crate::clusterer::vbx::VbxClusterer::from_dir(
                        std::path::Path::new(&dir),
                        max,
                    ),
                    None => crate::clusterer::vbx::VbxClusterer::from_registry(registry, max),
                },
            }
            .map_err(|e| ConfigError::Load {
                model_id: "vbx",
                source: Box::new(e),
            })?;
            // Dense windowed embeddings are non-contiguous: the HMM
            // self-loop assumption is invalid → auto GMM-VBx.
            // `loop_prob` is an explicit `VbxConfig` knob; windowed
            // mode always forces GMM.
            let windowed = config.embed_window_secs.is_some_and(|w| w > 0.0);
            vbx = vbx.auto_gmm_for_windowed(windowed);
            Ok(Box::new(vbx))
        }
        #[cfg(not(feature = "vbx"))]
        ClustererKind::Vbx => Err(ConfigError::UnknownModel {
            model_id: "vbx (requires the `vbx` feature)".to_owned(),
        }),
    }
}

impl PipelineBuilder {
    /// Validate + construct the inner `Pipeline`.
    pub fn build(self) -> Result<Pipeline, ConfigError> {
        self.validate()?;
        let resegmenter = self
            .custom_resegmenter
            .unwrap_or_else(|| Box::new(OverlapResegmenter::default()));

        match self.config.profile {
            Profile::Custom => {
                let segmenter =
                    self.custom_segmenter
                        .ok_or_else(|| ConfigError::MissingCustomComponent {
                            missing: vec!["segmenter"],
                        })?;
                let embedder =
                    self.custom_embedder
                        .ok_or_else(|| ConfigError::MissingCustomComponent {
                            missing: vec!["embedder"],
                        })?;
                let clusterer =
                    self.custom_clusterer
                        .ok_or_else(|| ConfigError::MissingCustomComponent {
                            missing: vec!["clusterer"],
                        })?;
                Ok(Pipeline::from_components(
                    self.config,
                    segmenter,
                    embedder,
                    clusterer,
                    resegmenter,
                ))
            }
            Profile::Mobile | Profile::Balanced | Profile::Fast => {
                let registry = self.registry.ok_or(ConfigError::MissingRegistry {
                    profile: self.config.profile,
                })?;
                let profile_models = registry.ensure_for_profile(self.config.profile)?;
                let ep = self.config.execution_provider;
                tracing::info!("pipeline v2 execution provider: {ep:?}");
                let mut seg_cfg = crate::segmentation::PowersetConfig::default();
                seg_cfg.aggregation.binarization = self.config.binarization;
                let segmenter: Box<dyn Segmenter> = Box::new(
                    crate::segmentation::PowersetSegmenter::with_config(
                        &profile_models.segmenter_path,
                        seg_cfg,
                        ep,
                    )
                    .map_err(|e| ConfigError::Load {
                        model_id: "powerset",
                        source: Box::new(e),
                    })?,
                );
                let embedder: Box<dyn Embedder> = Box::new(
                    crate::embedder::ResNet34Adapter::new(
                        &profile_models.embedder_path,
                        self.config.embedder_pool_size,
                        ep,
                    )
                    .map_err(|e| ConfigError::Load {
                        model_id: "resnet34",
                        source: Box::new(e),
                    })?,
                );
                let clusterer: Box<dyn Clusterer> =
                    build_profile_clusterer(&self.config, &registry)?;
                // Activate min_cluster_size pruning (this config field was
                // previously dead — never read by any clusterer). Dissolves
                // spurious sub-min clusters into the nearest large speaker: the
                // over-clustering fix that a global threshold cannot achieve
                // without over-merging real speakers. Profile path only — Custom
                // callers own their clusterer and opt in via with_clusterer.
                // VBx determines the speaker count itself (prior-driven pruning),
                // so post-hoc min-size pruning would dissolve its own clusters —
                // skip the wrap for VBx.
                let min_size = self.config.min_cluster_size;
                let clusterer: Box<dyn Clusterer> =
                    if min_size > 1 && self.config.clusterer != ClustererKind::Vbx {
                        Box::new(crate::clusterer::MinClusterSizeClusterer::new(
                            clusterer, min_size,
                        ))
                    } else {
                        clusterer
                    };
                // Store the effective (post-domain-profile) clusterer kind so
                // `Pipeline::config()` reports the threshold the clusterer was
                // actually built with, not the pre-resolution configured one.
                let mut config = self.config;
                config.clusterer = resolve_clusterer_kind(&config);
                Ok(Pipeline::from_components(
                    config,
                    segmenter,
                    embedder,
                    clusterer,
                    resegmenter,
                ))
            }
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
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

    /// Every shipping profile resolves to the local FP32 model pair checked
    /// into `models/`, so profile builds serve registry cache hits and never
    /// touch the network.
    const LOCAL_PROFILE_MANIFEST: &str = r#"
        schema = "polyvoice-models-v2"
        [profiles.mobile]
        segmenter = "local_powerset"
        embedder  = "local_resnet34"
        [profiles.balanced]
        segmenter = "local_powerset"
        embedder  = "local_resnet34"
        [profiles.fast]
        segmenter = "local_powerset"
        embedder  = "local_resnet34"
        [models.local_powerset]
        url      = "https://example.invalid/powerset_fp32.onnx"
        sha256   = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
        size     = 5992913
        filename = "powerset_fp32.onnx"
        [models.local_resnet34]
        url      = "https://example.invalid/wespeaker_resnet34.onnx"
        sha256   = "9fea6516d7ad6bf0a76c7689f5a49b65d330fad6dde96c91bb4435ffbfe056a1"
        size     = 26534127
        filename = "wespeaker_resnet34.onnx"
    "#;

    /// Bytes that hash-match a manifest entry but are not a loadable ONNX model.
    const GARBAGE_BYTES: &[u8] = b"not an onnx model, just garbage bytes";

    /// Manifest whose balanced profile can be pointed at the garbage segmenter
    /// or embedder to drive the `ConfigError::Load` paths offline.
    const GARBAGE_SEGMENTER_MANIFEST: &str = r#"
        schema = "polyvoice-models-v2"
        [profiles.balanced]
        segmenter = "garbage"
        embedder  = "local_resnet34"
        [models.garbage]
        url      = "https://example.invalid/garbage.onnx"
        sha256   = "018eb9afb44b357df9c828ffe49f87b2e023768ce4585f41cb26835be5a148ec"
        size     = 37
        filename = "garbage.onnx"
        [models.local_resnet34]
        url      = "https://example.invalid/wespeaker_resnet34.onnx"
        sha256   = "9fea6516d7ad6bf0a76c7689f5a49b65d330fad6dde96c91bb4435ffbfe056a1"
        size     = 26534127
        filename = "wespeaker_resnet34.onnx"
    "#;

    const GARBAGE_EMBEDDER_MANIFEST: &str = r#"
        schema = "polyvoice-models-v2"
        [profiles.balanced]
        segmenter = "local_powerset"
        embedder  = "garbage"
        [models.local_powerset]
        url      = "https://example.invalid/powerset_fp32.onnx"
        sha256   = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
        size     = 5992913
        filename = "powerset_fp32.onnx"
        [models.garbage]
        url      = "https://example.invalid/garbage.onnx"
        sha256   = "018eb9afb44b357df9c828ffe49f87b2e023768ce4585f41cb26835be5a148ec"
        size     = 37
        filename = "garbage.onnx"
    "#;

    /// Same local FP32 pair plus the six VBx PLDA artifacts (hashes of the
    /// checked-in `fixtures/vbx-plda/*.npy`), so the registry fallback for the
    /// VBx clusterer resolves offline.
    #[cfg(feature = "vbx")]
    const LOCAL_VBX_MANIFEST: &str = r#"
        schema = "polyvoice-models-v2"
        [profiles.balanced]
        segmenter = "local_powerset"
        embedder  = "local_resnet34"
        [models.local_powerset]
        url      = "https://example.invalid/powerset_fp32.onnx"
        sha256   = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
        size     = 5992913
        filename = "powerset_fp32.onnx"
        [models.local_resnet34]
        url      = "https://example.invalid/wespeaker_resnet34.onnx"
        sha256   = "9fea6516d7ad6bf0a76c7689f5a49b65d330fad6dde96c91bb4435ffbfe056a1"
        size     = 26534127
        filename = "wespeaker_resnet34.onnx"
        [models.vbx_plda_transform]
        url      = "https://example.invalid/plda_transform.npy"
        sha256   = "90261469714415743f4b8a86ee6b89466db858bde3c5944367cccfb7abd34f14"
        size     = 131200
        filename = "plda_transform.npy"
        [models.vbx_plda_phi_computed]
        url      = "https://example.invalid/plda_phi_computed.npy"
        sha256   = "6ef7cf2f5a23a45b66f440f9a996a4cf5c047b369829af695d50ef18aa0a35e3"
        size     = 1152
        filename = "plda_phi_computed.npy"
        [models.vbx_plda_mean1]
        url      = "https://example.invalid/plda_mean1.npy"
        sha256   = "e424c0c352182aa8e0f555dec1f3b30e29a20b9ed6b25d339f112af92e51e36f"
        size     = 2176
        filename = "plda_mean1.npy"
        [models.vbx_plda_mean2]
        url      = "https://example.invalid/plda_mean2.npy"
        sha256   = "6f6fb708a2037197b5b84ffeaa8f140cb878088fbecd6ab042ad26a7691bd2cf"
        size     = 640
        filename = "plda_mean2.npy"
        [models.vbx_plda_lda]
        url      = "https://example.invalid/plda_lda.npy"
        sha256   = "e20c9b012bebd1aabda5a38a127e63a43cf35debdc502715fc143e2fb6bc3c4b"
        size     = 131200
        filename = "plda_lda.npy"
        [models.vbx_plda_mu]
        url      = "https://example.invalid/plda_mu.npy"
        sha256   = "d286d48acf99bbc1ed1502fed0a3e361ae5626ce1870c8be9f7397c5e47886c6"
        size     = 1152
        filename = "plda_mu.npy"
    "#;

    /// `None` (test skips) when the gitignored model blobs are not present
    /// locally — they exist only after a local model download.
    fn registry_with_local_models() -> Option<(tempfile::TempDir, ModelRegistry)> {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        for f in ["powerset_fp32.onnx", "wespeaker_resnet34.onnx"] {
            let src = repo_file(&format!("models/{f}"));
            if !src.exists() {
                eprintln!("skip: models/{f} missing");
                return None;
            }
            std::fs::copy(src, tmp.path().join(f)).expect("copy local model into cache");
        }
        let manifest =
            Manifest::from_toml_str(LOCAL_PROFILE_MANIFEST).expect("local manifest parses");
        let registry = ModelRegistry::with_manifest(manifest, tmp.path()).expect("registry");
        Some((tmp, registry))
    }

    #[test]
    fn execution_provider_setter_overrides_config() {
        let b = fresh().execution_provider(crate::onnx::ExecutionProvider::Cpu);
        assert_eq!(
            b.config.execution_provider,
            crate::onnx::ExecutionProvider::Cpu
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
        let registry = match ModelRegistry::default() {
            Ok(r) => r,
            Err(_) => return,
        };
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

    #[test]
    fn build_balanced_with_local_models_succeeds() {
        let Some((_tmp, registry)) = registry_with_local_models() else {
            return;
        };
        let p = fresh()
            .profile(Profile::Balanced)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .expect("balanced profile builds from cached local models");
        assert_eq!(p.config().profile, Profile::Balanced);
    }

    #[test]
    fn build_mobile_with_local_models_succeeds() {
        let Some((_tmp, registry)) = registry_with_local_models() else {
            return;
        };
        let p = fresh()
            .profile(Profile::Mobile)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .expect("mobile profile builds from cached local models");
        assert_eq!(p.config().profile, Profile::Mobile);
    }

    #[test]
    fn build_fast_with_local_models_succeeds() {
        let Some((_tmp, registry)) = registry_with_local_models() else {
            return;
        };
        let p = fresh()
            .profile(Profile::Fast)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .expect("fast profile resolves through the same local pair");
        assert_eq!(p.config().profile, Profile::Fast);
    }

    #[test]
    fn build_with_nme_sc_clusterer_succeeds() {
        let Some((_tmp, registry)) = registry_with_local_models() else {
            return;
        };
        let cfg = PipelineConfig {
            clusterer: ClustererKind::NmeSc,
            ..PipelineConfig::default()
        };
        let p = fresh()
            .config(cfg)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .expect("NME-SC clusterer selection builds");
        assert!(matches!(p.config().clusterer, ClustererKind::NmeSc));
    }

    #[test]
    fn build_with_min_cluster_size_pruning_succeeds() {
        let Some((_tmp, registry)) = registry_with_local_models() else {
            return;
        };
        let cfg = PipelineConfig {
            min_cluster_size: 4,
            ..PipelineConfig::default()
        };
        let p = fresh()
            .config(cfg)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .expect("min-cluster-size pruning wraps the AHC clusterer");
        assert_eq!(p.config().min_cluster_size, 4);
    }

    #[test]
    fn build_garbage_segmenter_reports_load_error() {
        let embedder_src = repo_file("models/wespeaker_resnet34.onnx");
        if !embedder_src.exists() {
            eprintln!("skip: models/wespeaker_resnet34.onnx missing");
            return;
        }
        let tmp = tempfile::TempDir::new().expect("temp dir");
        std::fs::write(tmp.path().join("garbage.onnx"), GARBAGE_BYTES).expect("write garbage");
        std::fs::copy(embedder_src, tmp.path().join("wespeaker_resnet34.onnx"))
            .expect("copy embedder model");
        let manifest =
            Manifest::from_toml_str(GARBAGE_SEGMENTER_MANIFEST).expect("manifest parses");
        let registry = ModelRegistry::with_manifest(manifest, tmp.path()).expect("registry");
        let err = fresh()
            .profile(Profile::Balanced)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .err()
            .expect("build must fail");
        assert!(matches!(
            err,
            ConfigError::Load {
                model_id: "powerset",
                ..
            }
        ));
    }

    #[test]
    fn build_garbage_embedder_reports_load_error() {
        let segmenter_src = repo_file("models/powerset_fp32.onnx");
        if !segmenter_src.exists() {
            eprintln!("skip: models/powerset_fp32.onnx missing");
            return;
        }
        let tmp = tempfile::TempDir::new().expect("temp dir");
        std::fs::write(tmp.path().join("garbage.onnx"), GARBAGE_BYTES).expect("write garbage");
        std::fs::copy(segmenter_src, tmp.path().join("powerset_fp32.onnx"))
            .expect("copy segmenter model");
        let manifest = Manifest::from_toml_str(GARBAGE_EMBEDDER_MANIFEST).expect("manifest parses");
        let registry = ModelRegistry::with_manifest(manifest, tmp.path()).expect("registry");
        let err = fresh()
            .profile(Profile::Balanced)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .err()
            .expect("build must fail");
        assert!(matches!(
            err,
            ConfigError::Load {
                model_id: "resnet34",
                ..
            }
        ));
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
        assert!(matches!(
            err,
            ConfigError::Registry(RegistryError::ProfileNotFound { .. })
        ));
    }

    #[cfg(feature = "vbx")]
    #[test]
    fn build_vbx_from_explicit_plda_dir_succeeds() {
        let Some((_tmp, registry)) = registry_with_local_models() else {
            return;
        };
        let cfg = PipelineConfig {
            clusterer: ClustererKind::Vbx,
            vbx_plda_dir: Some(repo_file("fixtures/vbx-plda")),
            // Windowed mode forces the GMM-VBx variant; min_cluster_size must
            // not wrap VBx (it prunes its own clusters).
            embed_window_secs: Some(2.0),
            min_cluster_size: 4,
            ..PipelineConfig::default()
        };
        let p = fresh()
            .config(cfg)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .expect("VBx builds from an explicit PLDA dir");
        assert!(matches!(p.config().clusterer, ClustererKind::Vbx));
    }

    #[cfg(feature = "vbx")]
    #[test]
    fn build_vbx_from_env_plda_dir_succeeds() {
        let Some((_tmp, registry)) = registry_with_local_models() else {
            return;
        };
        let cfg = PipelineConfig {
            clusterer: ClustererKind::Vbx,
            ..PipelineConfig::default()
        };
        // The builder is the library's single env-resolution point; nextest
        // isolates each test in its own process, so mutating the environment
        // here cannot leak into other tests.
        unsafe {
            std::env::set_var("POLYVOICE_VBX_PLDA_DIR", repo_file("fixtures/vbx-plda"));
        }
        let built = fresh()
            .config(cfg)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build();
        unsafe {
            std::env::remove_var("POLYVOICE_VBX_PLDA_DIR");
        }
        built.expect("VBx builds from the PLDA dir named by the env var");
    }

    #[cfg(feature = "vbx")]
    #[test]
    fn build_vbx_from_registry_cache_succeeds() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        for f in ["powerset_fp32.onnx", "wespeaker_resnet34.onnx"] {
            let src = repo_file(&format!("models/{f}"));
            if !src.exists() {
                eprintln!("skip: models/{f} missing");
                return;
            }
            std::fs::copy(src, tmp.path().join(f)).expect("copy local model into cache");
        }
        for entry in std::fs::read_dir(repo_file("fixtures/vbx-plda")).expect("fixture dir") {
            let entry = entry.expect("dir entry");
            let name = entry.file_name();
            if name.to_string_lossy().ends_with(".npy") {
                std::fs::copy(entry.path(), tmp.path().join(&name)).expect("copy PLDA artifact");
            }
        }
        let manifest = Manifest::from_toml_str(LOCAL_VBX_MANIFEST).expect("vbx manifest parses");
        let registry = ModelRegistry::with_manifest(manifest, tmp.path()).expect("registry");
        let cfg = PipelineConfig {
            clusterer: ClustererKind::Vbx,
            ..PipelineConfig::default()
        };
        let p = fresh()
            .config(cfg)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .expect("VBx falls back to the registry PLDA artifacts");
        assert!(matches!(p.config().clusterer, ClustererKind::Vbx));
    }

    #[cfg(feature = "vbx")]
    #[test]
    fn build_vbx_missing_plda_dir_reports_load_error() {
        let Some((_tmp, registry)) = registry_with_local_models() else {
            return;
        };
        let cfg = PipelineConfig {
            clusterer: ClustererKind::Vbx,
            vbx_plda_dir: Some(repo_file("fixtures/does-not-exist")),
            ..PipelineConfig::default()
        };
        let err = fresh()
            .config(cfg)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .err()
            .expect("build must fail");
        assert!(matches!(
            err,
            ConfigError::Load {
                model_id: "vbx",
                ..
            }
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

        let resolve = |clusterer: ClustererKind,
                       domain: Option<crate::clusterer::DomainProfile>| {
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
        // nextest runs each test in its own process, so the env var cannot
        // leak into other tests.
        unsafe {
            std::env::set_var("POLYVOICE_ASNORM_COHORT", &cohort_path);
        }
        let loaded = load_as_norm_cohort(&cfg, &registry);
        unsafe {
            std::env::remove_var("POLYVOICE_ASNORM_COHORT");
        }
        let cohort = loaded.expect("env override supplies the cohort");
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
    fn build_ahc_with_as_norm_cohort_path_succeeds() {
        let Some((_tmp, registry)) = registry_with_local_models() else {
            return;
        };
        let cohort_tmp = tempfile::TempDir::new().expect("temp dir");
        let (_, cohort_rows) = as_norm_discriminating_scene();
        let cohort_path = write_test_cohort(cohort_tmp.path(), &cohort_rows);
        let cfg = PipelineConfig {
            clusterer: ClustererKind::Ahc { threshold: 0.5 },
            as_norm: Some(crate::clusterer::AsNormConfig {
                top_n: 10,
                cohort: crate::clusterer::CohortSource::Path(cohort_path),
            }),
            domain: Some(crate::clusterer::domain::AMI),
            ..PipelineConfig::default()
        };
        let p = fresh()
            .config(cfg)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .expect("AHC + AS-norm + domain profile builds");
        // config() reports the EFFECTIVE threshold: the AMI profile's
        // z-threshold replaced the configured 0.5 at build time.
        match p.config().clusterer {
            ClustererKind::Ahc { threshold } => assert_eq!(
                threshold,
                crate::clusterer::domain::AMI.as_norm_threshold.unwrap()
            ),
            other => panic!("expected Ahc, got {other:?}"),
        }
    }

    #[cfg(feature = "vbx")]
    #[test]
    fn build_vbx_never_touches_as_norm_config() {
        let Some((_tmp, registry)) = registry_with_local_models() else {
            return;
        };
        let cfg = PipelineConfig {
            clusterer: ClustererKind::Vbx,
            vbx_plda_dir: Some(repo_file("fixtures/vbx-plda")),
            // A cohort source that would fail if the VBx path resolved it:
            // success below proves AS-norm decorates AHC only.
            as_norm: Some(crate::clusterer::AsNormConfig {
                top_n: 10,
                cohort: crate::clusterer::CohortSource::ModelId("absent_cohort".to_owned()),
            }),
            domain: Some(crate::clusterer::domain::AMI),
            ..PipelineConfig::default()
        };
        let p = fresh()
            .config(cfg)
            .with_models_from(registry)
            .execution_provider(crate::onnx::ExecutionProvider::Cpu)
            .build()
            .expect("VBx path ignores AS-norm and domain config");
        assert!(matches!(p.config().clusterer, ClustererKind::Vbx));
    }
}
