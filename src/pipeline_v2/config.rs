//! M6a — `PipelineConfig`, `ClustererKind`, `ExecutionProvider`.

use crate::types::{Profile, SampleRate};

/// Top-level configuration for the v1.0 Pipeline. Mirrors spec §5.2 verbatim.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub profile: Profile,
    pub sample_rate: SampleRate,
    pub seg_window_secs: f32,
    pub seg_hop_secs: f32,
    pub clusterer: ClustererKind,
    pub max_speakers: u8,
    pub min_cluster_size: usize,
    pub resegment_overlap: bool,
    pub resegment_min_overlap_secs: f32,
    pub min_speech_secs: f32,
    pub max_gap_secs: f32,
    pub embedder_pool_size: usize,
    pub execution_provider: ExecutionProvider,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Balanced,
            sample_rate: SampleRate::new(16000).unwrap_or_default(),
            seg_window_secs: 10.0,
            seg_hop_secs: 0.5,
            clusterer: ClustererKind::Ahc { threshold: 0.5 },
            max_speakers: 20,
            min_cluster_size: 12,
            resegment_overlap: true,
            resegment_min_overlap_secs: 0.1,
            min_speech_secs: 0.25,
            max_gap_secs: 0.5,
            embedder_pool_size: default_pool_size(),
            execution_provider: ExecutionProvider::auto(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClustererKind {
    NmeSc,
    Ahc { threshold: f32 },
    /// VBx (Variational Bayes HMM + PLDA) with automatic speaker-count selection.
    /// Requires the `vbx` feature; the PLDA params are resolved at construction.
    Vbx,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExecutionProvider {
    Cpu,
    CoreMl,
    Nnapi,
    Cuda,
    XnnPack,
}

impl ExecutionProvider {
    pub fn auto() -> Self {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        return Self::CoreMl;
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        return Self::XnnPack;
        #[cfg(not(any(
            all(target_os = "macos", target_arch = "aarch64"),
            all(target_os = "linux", target_arch = "aarch64"),
        )))]
        return Self::Cpu;
    }
}

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
        assert!((cfg.seg_window_secs - 10.0).abs() < f32::EPSILON);
        assert!((cfg.seg_hop_secs - 0.5).abs() < f32::EPSILON);
        assert!(matches!(
            cfg.clusterer,
            ClustererKind::Ahc { threshold: 0.5 }
        ));
        assert_eq!(cfg.max_speakers, 20);
        assert_eq!(cfg.min_cluster_size, 12);
        assert!(cfg.resegment_overlap);
        assert!((cfg.resegment_min_overlap_secs - 0.1).abs() < f32::EPSILON);
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
}
