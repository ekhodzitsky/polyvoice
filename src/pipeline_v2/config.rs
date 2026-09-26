//! `PipelineConfig`, `ClustererKind`, `ExecutionProvider`: the top-level
//! configuration surface of the v1.0 pipeline.

use crate::types::{Profile, SampleRate};
use std::path::PathBuf;

/// Top-level configuration for the v1.0 Pipeline. Mirrors spec §5.2 verbatim.
///
/// The struct is `#[non_exhaustive]`: start from [`PipelineConfig::default`]
/// and assign the fields you need. Fields may be added in minor releases
/// without breaking that pattern. Everything here except
/// [`experimental`](Self::experimental) is part of the frozen contract
/// (`docs/semver.md`).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PipelineConfig {
    pub profile: Profile,
    /// Sample rate of `run` audio. Shipping profiles (`Mobile`, `Balanced`,
    /// `Fast`) accept only 16 kHz — that is the rate of the bundled models.
    /// `Custom` may use any [`SampleRate`] (8–192 kHz); the caller owns the
    /// components. A mismatched `run` rate still fails at execution.
    pub sample_rate: SampleRate,
    pub clusterer: ClustererKind,
    /// Hard cap on global speakers. Range `1..=255`.
    pub max_speakers: u8,
    /// Dissolve AHC clusters smaller than this into the nearest large
    /// speaker. `1` disables pruning (the shipped default). Range `>= 1`.
    /// Ignored for VBx, which chooses its own speaker count.
    pub min_cluster_size: usize,
    pub resegment_overlap: bool,
    /// Drop speech regions shorter than this. Finite and `>= 0` (`0` keeps
    /// every region). NaN and infinity are rejected.
    pub min_speech_secs: f32,
    /// Gap-filling: merge same-speaker segments separated by at most this many
    /// seconds (cVBx Δ=0.5 s default). Finite and `>= 0` (`0` disables the
    /// merge). One global value — never per-dataset.
    pub max_gap_secs: f32,
    /// Parallel embedder sessions. Range `>= 1`. The builder setter clamps
    /// `0` up to `1`; a raw [`.config()`](crate::pipeline_v2::PipelineBuilder::config)
    /// value of `0` is rejected.
    pub embedder_pool_size: usize,
    /// Maximum PCM length `Pipeline::run` accepts, in samples.
    ///
    /// Defaults to [`MAX_AUDIO_SAMPLES`](crate::pipeline_v2::MAX_AUDIO_SAMPLES),
    /// one hour at 16 kHz. That default guards callers who hand the pipeline a
    /// buffer they did not produce; a caller that controls the audio's
    /// provenance, such as one diarizing a file it recorded itself, can raise
    /// it. Zero accepts only empty input; the limit is inclusive.
    pub max_audio_samples: usize,
    /// Where the ONNX/tract session would run. Product kernels ignore this
    /// and always execute on CPU. Only [`ExecutionProvider::Cpu`] (including
    /// [`ExecutionProvider::auto`], which resolves to CPU) is accepted.
    /// `CoreMl`, `Nnapi`, `Cuda`, and `XnnPack` fail at `validate` — they are
    /// not a silent CPU fallback.
    pub execution_provider: ExecutionProvider,
    /// Directory with the precomputed VBx PLDA params, used only when
    /// `clusterer == ClustererKind::Vbx`. `None` resolves through the
    /// `POLYVOICE_VBX_PLDA_DIR` env var, then the model-registry download
    /// (the builder is the library's single env-resolution point). Has no
    /// effect for other clusterers.
    pub vbx_plda_dir: Option<PathBuf>,
    /// Dense embedding window (seconds). `None` embeds each primary segment once
    /// (sparse). `Some(w)` slides a `w`-second window with `w/2` hop inside each
    /// segment, yielding several embeddings per speaker run — like the legacy
    /// pipeline's dense windows — for more robust centroids / lower confusion at
    /// the cost of more embedder calls. Sub-`w` segments still embed once.
    /// `Some` must be finite and `> 0`.
    pub embed_window_secs: Option<f32>,
    /// Optional AS-norm score normalization for the fixed-threshold AHC
    /// clusterer: pairwise cosine scores are z-normalized against an imposter
    /// cohort before merging, so one threshold generalizes across recording
    /// domains. `None` keeps raw cosine scoring. Only applies to
    /// `ClustererKind::Ahc`; other clusterers ignore it.
    /// `top_n` must be `>= 2` (`0` and `1` would silently skip normalization).
    pub as_norm: Option<crate::clusterer::AsNormConfig>,
    /// Optional per-domain scoring profile. With `ClustererKind::Ahc` the
    /// profile's calibrated threshold replaces the configured one at build
    /// time; `None` keeps the configured threshold. Profiles are data (see
    /// [`crate::clusterer::domain`]) — never code branching.
    pub domain: Option<crate::clusterer::DomainProfile>,
    /// Measurement and ablation switches. Outside the stability contract:
    /// see [`ExperimentalConfig`]. The defaults are the shipped behavior.
    pub experimental: ExperimentalConfig,
}

/// Measurement-only and ablation switches of the v2 pipeline.
///
/// **Not covered by the API freeze.** Fields may be added, renamed, or
/// removed in any minor release, and none of them changes the shipped
/// defaults. They exist so `polyvoice-bench` can A/B alternative paths in one
/// binary. Construct via [`Default`] and assign fields; the struct is
/// `#[non_exhaustive]`.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct ExperimentalConfig {
    /// Ablation: empty the local→global speaker map so every overlap region
    /// takes the mixed-embedding fallback instead of the segmenter's own
    /// two-speaker assignment. Ships `false` (the overlap-accuracy win).
    pub disable_seg_overlap: bool,
    /// Ablation: majority vote instead of Hungarian assignment for the
    /// local→global speaker map. Ships `false`.
    pub majority_local_map: bool,
    /// Optional calibrated binarization of segmentation posteriors (onset/offset
    /// hysteresis + min-duration smoothing) instead of per-frame argmax.
    /// `None` keeps the shipped argmax behavior.
    pub binarization: Option<crate::segmentation::BinarizationConfig>,
    /// Measurement-only embedder override on the native powerset path.
    /// `None` keeps the profile embedder (`resnet34_int8` kernels).
    /// `Some("cam_pp_int8")` / `Some("cam_pp_fp32")` keeps native powerset and
    /// loads CAM++ via tract. Not a shipping profile; product CLI never sets it.
    pub embedder_model: Option<String>,
    /// Cluster per-(window, local-speaker) masked embeddings and reconstruct
    /// turns from mapped masks (no Hungarian window stitch). Ships `false`
    /// until the held-out DER / RTFx gate passes.
    pub reconstruct: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Balanced,
            sample_rate: SampleRate::new(16000).unwrap_or_default(),
            // CLI / FFI / Python / MCP all set VBx. `vbx` off (rare library
            // combo) falls back to AHC so `build()` still constructs.
            clusterer: default_clusterer(),
            max_speakers: 20,
            // v2 ships unpruned: min-cluster pruning is net-negative for the
            // powerset pipeline and collapses short clips (a 26 s clip has every
            // cluster below 12 members → all dissolved into one speaker → DER
            // ~49%). 1 = no pruning; tune per-call if a split-heavy file needs it.
            min_cluster_size: 1,
            resegment_overlap: true,
            min_speech_secs: 0.25,
            max_gap_secs: 0.5,
            embedder_pool_size: default_pool_size(),
            max_audio_samples: crate::pipeline_v2::MAX_AUDIO_SAMPLES,
            execution_provider: ExecutionProvider::auto(),
            vbx_plda_dir: None,
            embed_window_secs: None,
            as_norm: None,
            domain: None,
            experimental: ExperimentalConfig::default(),
        }
    }
}

/// Clustering backend selection. `#[non_exhaustive]`: match with a wildcard
/// arm; new backends may appear in minor releases.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ClustererKind {
    NmeSc,
    /// Fixed-threshold agglomerative clustering. `threshold` is cosine
    /// similarity in `[-1.0, 1.0]` (finite).
    Ahc {
        threshold: f32,
    },
    /// VBx (Variational Bayes HMM + PLDA) with automatic speaker-count selection.
    /// Requires the `vbx` feature; the PLDA params are resolved at construction.
    Vbx,
}

// Tract owns the live EP type (session construction). Kernel-only builds
// (`pipeline-native`) have no `onnx` module — same variants, never executed.
#[cfg(feature = "infer")]
pub use crate::onnx::ExecutionProvider;

/// Where a session would run. Product kernels execute on the CPU only:
/// [`ExecutionProvider::Cpu`] and [`ExecutionProvider::auto`] (which resolves
/// to CPU) are the only values [`PipelineBuilder::validate`] accepts. The
/// other variants exist for tract builds and are rejected, never silently
/// downgraded. `#[non_exhaustive]`: match with a wildcard arm.
///
/// [`PipelineBuilder::validate`]: crate::pipeline_v2::PipelineBuilder::validate
#[cfg(not(feature = "infer"))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ExecutionProvider {
    Cpu,
    CoreMl,
    Nnapi,
    Cuda,
    XnnPack,
}

#[cfg(not(feature = "infer"))]
impl ExecutionProvider {
    pub fn auto() -> Self {
        Self::Cpu
    }

    pub fn is_available(self) -> bool {
        matches!(self, Self::Cpu)
    }
}

fn default_clusterer() -> ClustererKind {
    #[cfg(feature = "vbx")]
    {
        ClustererKind::Vbx
    }
    #[cfg(not(feature = "vbx"))]
    {
        ClustererKind::Ahc {
            threshold: crate::types::DEFAULT_AHC_THRESHOLD,
        }
    }
}

fn default_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 4)
}

impl PipelineConfig {
    /// Reject public settings that would otherwise download models and then
    /// mis-index audio, silently disable a feature, or pretend a backend ran.
    /// Called from [`crate::pipeline_v2::PipelineBuilder::validate`] before
    /// any registry fetch.
    pub(crate) fn validate_settings(&self) -> Result<(), crate::pipeline_v2::ConfigError> {
        use crate::pipeline_v2::ConfigError;

        let bad =
            |field: &'static str, detail: String| ConfigError::InvalidSetting { field, detail };

        if !matches!(self.profile, Profile::Custom) && self.sample_rate.get() != 16_000 {
            return Err(bad(
                "sample_rate",
                format!(
                    "shipping profiles require 16000 Hz (bundled models); got {} Hz. Use Profile::Custom with your own components for another rate",
                    self.sample_rate.get()
                ),
            ));
        }
        if self.max_speakers == 0 {
            return Err(bad("max_speakers", "must be in 1..=255, got 0".into()));
        }
        if self.min_cluster_size == 0 {
            return Err(bad(
                "min_cluster_size",
                "must be >= 1 (1 disables pruning), got 0".into(),
            ));
        }
        if !(self.min_speech_secs.is_finite() && self.min_speech_secs >= 0.0) {
            return Err(bad(
                "min_speech_secs",
                format!("must be finite and >= 0, got {}", self.min_speech_secs),
            ));
        }
        if !(self.max_gap_secs.is_finite() && self.max_gap_secs >= 0.0) {
            return Err(bad(
                "max_gap_secs",
                format!("must be finite and >= 0, got {}", self.max_gap_secs),
            ));
        }
        if self.embedder_pool_size == 0 {
            return Err(bad("embedder_pool_size", "must be >= 1, got 0".into()));
        }
        if let Some(w) = self.embed_window_secs
            && !(w.is_finite() && w > 0.0)
        {
            return Err(bad(
                "embed_window_secs",
                format!("must be finite and > 0 when set, got {w}"),
            ));
        }
        if let ClustererKind::Ahc { threshold } = self.clusterer
            && !(-1.0..=1.0).contains(&threshold)
        {
            return Err(bad(
                "clusterer.threshold",
                format!("cosine similarity must be in [-1.0, 1.0], got {threshold}"),
            ));
        }
        if !self.execution_provider.is_available() {
            return Err(bad(
                "execution_provider",
                format!(
                    "{:?} is not available; only Cpu is executed (auto resolves to Cpu)",
                    self.execution_provider
                ),
            ));
        }
        if let Some(bin) = self.experimental.binarization {
            if !(bin.onset.is_finite() && (0.0..=1.0).contains(&bin.onset)) {
                return Err(bad(
                    "binarization.onset",
                    format!("must be finite and in [0.0, 1.0], got {}", bin.onset),
                ));
            }
            if !(bin.offset.is_finite() && (0.0..=1.0).contains(&bin.offset)) {
                return Err(bad(
                    "binarization.offset",
                    format!("must be finite and in [0.0, 1.0], got {}", bin.offset),
                ));
            }
            if !(bin.min_duration_on.is_finite() && bin.min_duration_on >= 0.0) {
                return Err(bad(
                    "binarization.min_duration_on",
                    format!("must be finite and >= 0, got {}", bin.min_duration_on),
                ));
            }
            if !(bin.min_duration_off.is_finite() && bin.min_duration_off >= 0.0) {
                return Err(bad(
                    "binarization.min_duration_off",
                    format!("must be finite and >= 0, got {}", bin.min_duration_off),
                ));
            }
        }
        if let Some(norm) = &self.as_norm
            && norm.top_n < 2
        {
            return Err(bad(
                "as_norm.top_n",
                format!(
                    "must be >= 2 (1 silently disables normalization), got {}",
                    norm.top_n
                ),
            ));
        }
        Ok(())
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Profile;

    #[test]
    fn pipeline_config_default_is_balanced() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.profile, Profile::Balanced);
        assert_eq!(cfg.sample_rate.get(), 16000);
        #[cfg(feature = "vbx")]
        assert_eq!(cfg.clusterer, ClustererKind::Vbx);
        #[cfg(not(feature = "vbx"))]
        assert!(matches!(
            cfg.clusterer,
            ClustererKind::Ahc {
                threshold
            } if (threshold - crate::types::DEFAULT_AHC_THRESHOLD).abs() < f32::EPSILON
        ));
        assert_eq!(cfg.max_speakers, 20);
        assert_eq!(cfg.min_cluster_size, 1);
        assert!(cfg.resegment_overlap);
        assert!(!cfg.experimental.disable_seg_overlap);
        assert!(!cfg.experimental.majority_local_map);
        assert!((cfg.min_speech_secs - 0.25).abs() < f32::EPSILON);
        assert!((cfg.max_gap_secs - 0.5).abs() < f32::EPSILON);
        assert!(cfg.embedder_pool_size >= 1);
        assert!(cfg.embedder_pool_size <= 4);
        assert!(cfg.as_norm.is_none());
        assert!(cfg.domain.is_none());
        assert!(cfg.experimental.embedder_model.is_none());
        assert!(!cfg.experimental.reconstruct);
        assert!(cfg.experimental.binarization.is_none());
    }

    #[test]
    fn default_clusterer_matches_front_doors() {
        let cfg = PipelineConfig::default();
        #[cfg(feature = "vbx")]
        assert_eq!(cfg.clusterer, ClustererKind::Vbx);
        #[cfg(not(feature = "vbx"))]
        match cfg.clusterer {
            ClustererKind::Ahc { threshold } => {
                assert!((threshold - crate::types::DEFAULT_AHC_THRESHOLD).abs() < f32::EPSILON);
                assert!((threshold - Profile::Balanced.default_threshold()).abs() < f32::EPSILON);
            }
            other => panic!("expected AHC fallback without vbx, got {other:?}"),
        }
    }

    #[test]
    fn clusterer_kind_ahc_with_threshold() {
        let k = ClustererKind::Ahc { threshold: 0.7 };
        if let ClustererKind::Ahc { threshold } = k {
            assert!((threshold - 0.7).abs() < f32::EPSILON);
        } else {
            panic!("expected Ahc variant");
        }
    }

    #[test]
    fn execution_provider_auto_returns_some_variant() {
        let ep = ExecutionProvider::auto();
        let _ = ep;
    }

    #[test]
    fn clusterer_kind_nme_sc_and_vbx_variants_are_distinct() {
        assert_eq!(ClustererKind::NmeSc, ClustererKind::NmeSc);
        assert_eq!(ClustererKind::Vbx, ClustererKind::Vbx);
        assert_ne!(ClustererKind::NmeSc, ClustererKind::Vbx);
        assert_ne!(ClustererKind::NmeSc, ClustererKind::Ahc { threshold: 0.5 });
    }

    #[test]
    fn default_pool_size_stays_within_clamp() {
        let n = default_pool_size();
        assert!((1..=4).contains(&n));
    }
}
