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
            //
            // VBx knobs stay `VbxClustererConfig::default()` unless the
            // caller opts in with `POLYVOICE_VBX_FROM_ENV=1` (offline
            // calibration). Production construction is otherwise env-free.
            let vbx_cfg = vbx_config_for_pipeline();
            let mut vbx = match &config.vbx_plda_dir {
                Some(dir) => {
                    crate::clusterer::vbx::VbxClusterer::from_dir_with_config(dir, max, vbx_cfg)
                }
                None => match std::env::var_os("POLYVOICE_VBX_PLDA_DIR") {
                    Some(dir) => crate::clusterer::vbx::VbxClusterer::from_dir_with_config(
                        std::path::Path::new(&dir),
                        max,
                        vbx_cfg,
                    ),
                    None => crate::clusterer::vbx::VbxClusterer::from_registry_with_config(
                        registry, max, vbx_cfg,
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

/// Production VBx knobs are the compiled defaults. Set
/// `POLYVOICE_VBX_FROM_ENV=1` (or `true`) to overlay
/// `POLYVOICE_VBX_{FA,FB,LOOP_PROB,AHC_THRESHOLD,EMB_SCALE,MIN_EMB_SECS,AHC_ASC_MEMBERS}`
/// for offline calibration. Any other value, or an unset variable, keeps
/// the defaults — stray `POLYVOICE_VBX_FA=…` in the environment must not
/// move shipped DER.
#[cfg(feature = "vbx")]
fn vbx_config_for_pipeline() -> crate::clusterer::vbx::VbxClustererConfig {
    match std::env::var("POLYVOICE_VBX_FROM_ENV") {
        Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => {
            crate::clusterer::vbx::VbxClustererConfig::from_env()
        }
        _ => crate::clusterer::vbx::VbxClustererConfig::default(),
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(all(test, not(feature = "spectral")))]
mod tests {
    use super::*;
    use crate::pipeline_v2::config::{ClustererKind, PipelineConfig};

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

#[allow(clippy::unwrap_used)]
#[cfg(all(test, feature = "vbx"))]
mod vbx_env_tests {
    use super::vbx_config_for_pipeline;
    use crate::clusterer::vbx::VbxClustererConfig;

    fn clear_vbx_env() {
        unsafe {
            for k in [
                "POLYVOICE_VBX_FROM_ENV",
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
    }

    #[test]
    fn from_env_gate_is_required_to_overlay_knobs() {
        // Sequential: cargo test shares a process across this module's tests.
        clear_vbx_env();
        unsafe {
            std::env::set_var("POLYVOICE_VBX_FA", "0.99");
        }
        let c = vbx_config_for_pipeline();
        let d = VbxClustererConfig::default();
        assert!(
            (c.vbx.fa - d.vbx.fa).abs() < 1e-12,
            "FA overlay must require POLYVOICE_VBX_FROM_ENV"
        );
        unsafe {
            std::env::set_var("POLYVOICE_VBX_FROM_ENV", "1");
            std::env::set_var("POLYVOICE_VBX_FA", "0.42");
        }
        let c = vbx_config_for_pipeline();
        assert!((c.vbx.fa - 0.42).abs() < 1e-12);
        clear_vbx_env();
        unsafe {
            std::env::set_var("POLYVOICE_VBX_FROM_ENV", "true");
            std::env::set_var("POLYVOICE_VBX_EMB_SCALE", "3.5");
        }
        let c = vbx_config_for_pipeline();
        assert!((c.emb_scale - 3.5).abs() < 1e-6);
        clear_vbx_env();
    }
}
