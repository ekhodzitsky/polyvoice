//! `PipelineConfig`, `ClustererKind`, `ExecutionProvider`: the top-level
//! configuration surface of the v1.0 pipeline.

use crate::types::{Profile, SampleRate};
use std::path::PathBuf;

/// Top-level configuration for the v1.0 Pipeline. Mirrors spec §5.2 verbatim.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub profile: Profile,
    pub sample_rate: SampleRate,
    pub clusterer: ClustererKind,
    pub max_speakers: u8,
    pub min_cluster_size: usize,
    pub resegment_overlap: bool,
    /// Ablation: empty the local→global speaker map so every overlap region
    /// takes the mixed-embedding fallback instead of the segmenter's own
    /// two-speaker assignment. Ships `false` (the overlap-accuracy win).
    pub disable_seg_overlap: bool,
    /// Ablation: majority vote instead of Hungarian assignment for the
    /// local→global speaker map. Ships `false`.
    pub majority_local_map: bool,
    pub min_speech_secs: f32,
    /// Gap-filling: merge same-speaker segments separated by at most this many
    /// seconds (cVBx Δ=0.5 s default). One global value — never per-dataset.
    pub max_gap_secs: f32,
    pub embedder_pool_size: usize,
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
    pub embed_window_secs: Option<f32>,
    /// Optional calibrated binarization of segmentation posteriors (onset/offset
    /// hysteresis + min-duration smoothing) instead of per-frame argmax.
    /// `None` keeps the shipped argmax behavior.
    pub binarization: Option<crate::segmentation::BinarizationConfig>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Balanced,
            sample_rate: SampleRate::new(16000).unwrap_or_default(),
            clusterer: ClustererKind::Ahc { threshold: 0.5 },
            max_speakers: 20,
            // v2 ships unpruned: min-cluster pruning is net-negative for the
            // powerset pipeline and collapses short clips (a 26 s clip has every
            // cluster below 12 members → all dissolved into one speaker → DER
            // ~49%). 1 = no pruning; tune per-call if a split-heavy file needs it.
            min_cluster_size: 1,
            resegment_overlap: true,
            disable_seg_overlap: false,
            majority_local_map: false,
            min_speech_secs: 0.25,
            max_gap_secs: 0.5,
            embedder_pool_size: default_pool_size(),
            execution_provider: ExecutionProvider::auto(),
            vbx_plda_dir: None,
            embed_window_secs: None,
            binarization: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClustererKind {
    NmeSc,
    Ahc {
        threshold: f32,
    },
    /// VBx (Variational Bayes HMM + PLDA) with automatic speaker-count selection.
    /// Requires the `vbx` feature; the PLDA params are resolved at construction.
    Vbx,
}

// Canonical home moved to `crate::onnx` (the module that owns session
// creation) so low-level constructors can name the EP without a cyclic dep on
// pipeline_v2; re-exported here so existing imports keep compiling.
pub use crate::onnx::ExecutionProvider;

fn default_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 4)
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
        assert!(matches!(
            cfg.clusterer,
            ClustererKind::Ahc { threshold: 0.5 }
        ));
        assert_eq!(cfg.max_speakers, 20);
        assert_eq!(cfg.min_cluster_size, 1);
        assert!(cfg.resegment_overlap);
        assert!(!cfg.disable_seg_overlap);
        assert!(!cfg.majority_local_map);
        assert!((cfg.min_speech_secs - 0.25).abs() < f32::EPSILON);
        assert!((cfg.max_gap_secs - 0.5).abs() < f32::EPSILON);
        assert!(cfg.embedder_pool_size >= 1);
        assert!(cfg.embedder_pool_size <= 4);
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
