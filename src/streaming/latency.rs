//! Named streaming latency presets.
//!
//! Presets bundle window geometry, speaker-cache size, and label-stability
//! knobs. Input-buffer latency is a **configuration** number (not measured RTF):
//!
//! ```text
//! input_buffer_latency ≈ window_secs + right_context_secs + vad_frame_secs
//! ```
//!
//! At 16 kHz with EnergyVad `frame_size = 512`, `vad_frame_secs ≈ 0.032 s`.
//! Report latency, RTF, and DER as **separate** numbers (diart / NeMo convention).

use crate::types::{ClusterConfig, DiarizationConfig, WindowConfig};

/// Named latency / accuracy trade-off for [`crate::streaming::StreamingPipeline`].
///
/// `Balanced` matches [`DiarizationConfig::default`] window geometry so existing
/// callers see no change when they keep using `StreamingPipeline::new` without a
/// preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum LatencyPreset {
    /// Low input-buffer latency (short window / hop).
    Realtime,
    /// Current production defaults (`window_secs = 1.5`, `hop_secs = 0.75`).
    #[default]
    Balanced,
    /// Larger window and cache for higher label quality at higher latency.
    Accurate,
}

/// Concrete streaming parameters produced by a [`LatencyPreset`] (or set manually).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingParams {
    /// Embedding window length in seconds.
    pub window_secs: f32,
    /// Hop between consecutive embedding windows in seconds.
    pub hop_secs: f32,
    /// Extra right-context held before treating a label as final (seconds).
    /// Contributes to the documented input-buffer latency budget; emission still
    /// follows window readiness.
    pub right_context_secs: f32,
    /// Maximum number of speaker entries in the arrival-order cache.
    ///
    /// **Overflow:** when the cache is full and a frame does not match any entry
    /// above threshold, it is force-merged into the closest existing speaker
    /// (AWS-style overflow merge). No new ID is allocated.
    pub speaker_cache_cap: usize,
    /// Number of confident hits required before a speaker label is `stable`.
    pub min_hits_to_stable: usize,
    /// Cosine margin for [`super::stability::prefer_current_speaker`] hysteresis.
    pub prefer_current_margin: f32,
    /// Cosine similarity threshold to join an existing cache entry.
    pub match_threshold: f32,
}

impl LatencyPreset {
    /// Resolve this preset to concrete streaming parameters.
    pub fn params(self) -> StreamingParams {
        match self {
            Self::Realtime => StreamingParams {
                window_secs: 1.0,
                hop_secs: 0.5,
                right_context_secs: 0.0,
                speaker_cache_cap: 16,
                min_hits_to_stable: 2,
                prefer_current_margin: 0.05,
                match_threshold: 0.45,
            },
            Self::Balanced => StreamingParams {
                window_secs: 1.5,
                hop_secs: 0.75,
                right_context_secs: 0.0,
                speaker_cache_cap: 32,
                min_hits_to_stable: 3,
                prefer_current_margin: 0.08,
                match_threshold: 0.45,
            },
            Self::Accurate => StreamingParams {
                window_secs: 2.0,
                hop_secs: 1.0,
                right_context_secs: 0.25,
                speaker_cache_cap: 64,
                min_hits_to_stable: 4,
                prefer_current_margin: 0.10,
                match_threshold: 0.45,
            },
        }
    }

    /// Nominal input-buffer latency in seconds for EnergyVad at `sample_rate`
    /// with a 512-sample frame (32 ms @ 16 kHz).
    pub fn input_buffer_latency_secs(self, sample_rate: u32, vad_frame_samples: usize) -> f32 {
        let p = self.params();
        let vad_frame_secs = vad_frame_samples as f32 / sample_rate as f32;
        p.window_secs + p.right_context_secs + vad_frame_secs
    }

    /// Apply window geometry from this preset onto a [`DiarizationConfig`].
    ///
    /// Cluster threshold is set from the preset's match threshold; `max_speakers`
    /// is capped to the speaker-cache capacity (bounded state).
    pub fn apply(self, config: &mut DiarizationConfig) {
        let p = self.params();
        config.window = WindowConfig {
            window_secs: p.window_secs,
            hop_secs: p.hop_secs,
            sample_rate: config.window.sample_rate,
        };
        config.cluster = ClusterConfig {
            threshold: p.match_threshold,
            max_speakers: p.speaker_cache_cap,
            ..config.cluster
        };
    }

    /// Parse a CLI / config name (`realtime`, `balanced`, `accurate`).
    pub fn parse_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "realtime" | "real-time" | "low" | "low-latency" => Some(Self::Realtime),
            "balanced" | "default" => Some(Self::Balanced),
            "accurate" | "accuracy" | "high" => Some(Self::Accurate),
            _ => None,
        }
    }

    /// Canonical lowercase name for CLI / docs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Realtime => "realtime",
            Self::Balanced => "balanced",
            Self::Accurate => "accurate",
        }
    }
}

impl StreamingParams {
    /// Build params from a preset (same as [`LatencyPreset::params`]).
    pub fn from_preset(preset: LatencyPreset) -> Self {
        preset.params()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_matches_default_window_geometry() {
        let p = LatencyPreset::Balanced.params();
        let d = DiarizationConfig::default();
        assert!((p.window_secs - d.window.window_secs).abs() < 1e-6);
        assert!((p.hop_secs - d.window.hop_secs).abs() < 1e-6);
        assert!((p.match_threshold - d.cluster.threshold).abs() < 1e-6);
    }

    #[test]
    fn apply_mutates_config() {
        let mut cfg = DiarizationConfig::default();
        LatencyPreset::Realtime.apply(&mut cfg);
        assert!((cfg.window.window_secs - 1.0).abs() < 1e-6);
        assert_eq!(cfg.cluster.max_speakers, 16);
    }

    #[test]
    fn parse_names() {
        assert_eq!(
            LatencyPreset::parse_name("realtime"),
            Some(LatencyPreset::Realtime)
        );
        assert_eq!(
            LatencyPreset::parse_name("Balanced"),
            Some(LatencyPreset::Balanced)
        );
        assert_eq!(
            LatencyPreset::parse_name("accurate"),
            Some(LatencyPreset::Accurate)
        );
        assert_eq!(LatencyPreset::parse_name("nope"), None);
    }

    #[test]
    fn latency_budget_includes_vad_frame() {
        // 512 / 16000 = 0.032; realtime window 1.0 → ≈ 1.032
        let lat = LatencyPreset::Realtime.input_buffer_latency_secs(16000, 512);
        assert!((lat - 1.032).abs() < 1e-3);
        let bal = LatencyPreset::Balanced.input_buffer_latency_secs(16000, 512);
        assert!((bal - 1.532).abs() < 1e-3);
        let acc = LatencyPreset::Accurate.input_buffer_latency_secs(16000, 512);
        // 2.0 + 0.25 + 0.032
        assert!((acc - 2.282).abs() < 1e-3);
    }
}
