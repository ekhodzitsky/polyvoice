//! Profile-path clusterer construction for [`super::builder::PipelineBuilder`].
//!
//! Keeps domain-threshold resolution, AS-norm cohort loading, and VBx PLDA
//! fallback out of the builder façade so `build()` stays a thin orchestrator.

use crate::clusterer::Clusterer;
use crate::models::{ModelRegistry, RegistryError};
use crate::pipeline_v2::builder::ConfigError;
use crate::pipeline_v2::config::{ClustererKind, PipelineConfig};

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
/// builder path is the library's single env-resolution point.
pub(crate) fn load_as_norm_cohort(
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
pub(crate) fn build_profile_clusterer(
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
        ClustererKind::NmeSc => Err(ConfigError::UnknownModel {
            model_id: "nme-sc (requires the `spectral` feature)".to_owned(),
        }),
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

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline_v2::config::{ClustererKind, PipelineConfig};

    #[cfg(not(feature = "spectral"))]
    #[test]
    fn nme_sc_without_spectral_is_an_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = ModelRegistry::with_cache_dir(tmp.path()).unwrap();
        let cfg = PipelineConfig {
            clusterer: ClustererKind::NmeSc,
            ..PipelineConfig::default()
        };
        let err = match build_profile_clusterer(&cfg, &registry) {
            Err(e) => e,
            Ok(_) => panic!("NME-SC without spectral must fail"),
        };
        match err {
            ConfigError::UnknownModel { model_id } => {
                assert!(
                    model_id.contains("spectral"),
                    "error should name the feature, got {model_id}"
                );
            }
            other => panic!("expected UnknownModel, got {other:?}"),
        }
    }
}
