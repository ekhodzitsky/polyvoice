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
            Profile::Mobile | Profile::Balanced => {
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
            Profile::Mobile | Profile::Balanced => {
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
                let clusterer: Box<dyn Clusterer> = match self.config.clusterer {
                    ClustererKind::Ahc { threshold } => {
                        Box::new(crate::clusterer::AhcClusterer::with_threshold(
                            self.config.max_speakers as usize,
                            threshold,
                        ))
                    }
                    #[cfg(feature = "spectral")]
                    ClustererKind::NmeSc => Box::new(crate::clusterer::NmeScClusterer::new(
                        self.config.max_speakers as usize,
                    )),
                    #[cfg(not(feature = "spectral"))]
                    ClustererKind::NmeSc => {
                        Box::new(crate::clusterer::AhcClusterer::with_threshold(
                            self.config.max_speakers as usize,
                            self.config.profile.default_threshold(),
                        ))
                    }
                    #[cfg(feature = "vbx")]
                    ClustererKind::Vbx => {
                        let max = self.config.max_speakers as usize;
                        // PLDA resolution order: explicit `vbx_plda_dir` →
                        // `POLYVOICE_VBX_PLDA_DIR` env → registry download.
                        // This is the library's single env-resolution point;
                        // `pipeline_v2` always has `download`, so the registry
                        // fallback is available.
                        let mut vbx = match &self.config.vbx_plda_dir {
                            Some(dir) => crate::clusterer::vbx::VbxClusterer::from_dir(dir, max),
                            None => match std::env::var_os("POLYVOICE_VBX_PLDA_DIR") {
                                Some(dir) => crate::clusterer::vbx::VbxClusterer::from_dir(
                                    std::path::Path::new(&dir),
                                    max,
                                ),
                                None => crate::clusterer::vbx::VbxClusterer::from_registry(
                                    &registry, max,
                                ),
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
                        let windowed = self.config.embed_window_secs.is_some_and(|w| w > 0.0);
                        vbx = vbx.auto_gmm_for_windowed(windowed);
                        Box::new(vbx)
                    }
                    #[cfg(not(feature = "vbx"))]
                    ClustererKind::Vbx => {
                        return Err(ConfigError::UnknownModel {
                            model_id: "vbx (requires the `vbx` feature)".to_owned(),
                        });
                    }
                };
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
                Ok(Pipeline::from_components(
                    self.config,
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
    use crate::pipeline_v2::mocks::{MockClusterer, MockEmbedder, MockSegmenter};

    fn fresh() -> PipelineBuilder {
        PipelineBuilder::new()
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
}
