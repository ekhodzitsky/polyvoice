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

mod native_stages;
#[cfg(feature = "infer")]
mod tract_stages;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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

    /// A public numeric or backend setting is outside its documented range.
    /// Raised by [`PipelineBuilder::validate`] before any model download.
    #[error("invalid pipeline setting {field}: {detail}")]
    InvalidSetting { field: &'static str, detail: String },
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

    /// Set the maximum PCM length `run` accepts, in samples.
    ///
    /// Defaults to [`MAX_AUDIO_SAMPLES`](crate::pipeline_v2::MAX_AUDIO_SAMPLES),
    /// one hour at 16 kHz. Raise it to diarize longer recordings whose
    /// provenance the caller controls.
    pub fn max_audio_samples(mut self, samples: usize) -> Self {
        self.config.max_audio_samples = samples;
        self
    }

    /// Override the execution provider (defaults to
    /// `ExecutionProvider::auto()` via `PipelineConfig::default`). Product
    /// kernels execute on the CPU only: [`validate`](Self::validate) rejects
    /// every other provider instead of falling back.
    pub fn execution_provider(mut self, ep: crate::pipeline_v2::ExecutionProvider) -> Self {
        self.config.execution_provider = ep;
        self
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        self.config.validate_settings()?;
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

use crate::pipeline_v2::config::ClustererKind;
use crate::pipeline_v2::{Pipeline, StageModelIds};
use crate::resegmentation::OverlapResegmenter;

// Clusterer construction lives in `clusterer_factory`; re-export so `build()`
// and this module's unit tests share one path (`super::*` in tests).
pub(crate) use crate::pipeline_v2::clusterer_factory::{
    build_profile_clusterer, resolve_clusterer_kind,
};
// Only referenced from builder_tests — keep out of non-test lib graphs (bins).
#[cfg(test)]
pub(crate) use crate::pipeline_v2::clusterer_factory::load_as_norm_cohort;

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
                    StageModelIds::default(),
                ))
            }
            Profile::Mobile | Profile::Balanced | Profile::Fast => {
                let registry = self.registry.ok_or(ConfigError::MissingRegistry {
                    profile: self.config.profile,
                })?;
                let ep = self.config.execution_provider;
                tracing::info!("pipeline v2 execution provider: {ep:?}");
                let stages = load_profile_stages(&registry, &self.config)?;
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
                    stages.segmenter,
                    stages.embedder,
                    clusterer,
                    resegmenter,
                    StageModelIds::new(stages.segmenter_id, stages.embedder_id),
                ))
            }
        }
    }
}

struct StagePair {
    segmenter: Box<dyn Segmenter>,
    embedder: Box<dyn Embedder>,
    segmenter_id: &'static str,
    embedder_id: &'static str,
}

/// True when native powerset + ResNet34 kernels are compiled in.
///
/// `backend-tract` may also be on (measurement builds that swap in CAM++);
/// native still wins so the product segmenter stays bit-identical. Tract-only
/// builds (`cli-tract`) do not enable the native features, so they keep the
/// tract stage loader.
fn use_native_kernels() -> bool {
    cfg!(all(
        feature = "segmenter-native",
        feature = "embedder-native"
    ))
}

fn load_profile_stages(
    registry: &ModelRegistry,
    #[cfg_attr(not(feature = "infer"), allow(unused_variables))] config: &PipelineConfig,
) -> Result<StagePair, ConfigError> {
    if use_native_kernels() {
        native_stages::build_native_stages(registry, config)
    } else {
        #[cfg(feature = "infer")]
        {
            tract_stages::build_onnx_stages(registry, config)
        }
        #[cfg(not(feature = "infer"))]
        {
            unreachable!("pipeline_v2 without infer always uses native kernels")
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "builder_tests.rs"]
mod tests;
