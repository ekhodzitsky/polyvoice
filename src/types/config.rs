//! Pipeline configuration bags (cluster / window / speech filter / diarization).
use super::measures::SampleRate;

/// Default cosine-similarity threshold for the fixed-threshold AHC clusterer:
/// clusters whose centroids are at least this similar are merged. Shared by
/// [`ClusterConfig::default`], the Balanced profile default, and every CLI /
/// FFI / SDK surface that exposes a `--threshold`-style knob, so the shipped
/// defaults cannot drift apart.
pub const DEFAULT_AHC_THRESHOLD: f32 = 0.45;

/// Configuration for speaker clustering.
#[derive(Debug, Clone, Copy)]
pub struct ClusterConfig {
    /// Cosine similarity threshold: clusters whose centroids are at least this
    /// similar are merged by the agglomerative clusterer. Higher = stricter =
    /// more (smaller) clusters.
    pub threshold: f32,
    /// Maximum number of speakers to track.
    pub max_speakers: usize,
    /// Minimum members a cluster must have to survive. After clustering, any
    /// cluster smaller than this is dissolved and its frames reassigned to the
    /// nearest large speaker centroid. This prunes spurious tiny clusters that
    /// inflate the speaker count without hurting frame-DER. `1` disables pruning.
    /// Ignored when `min_cluster_secs > 0` (duration pruning takes precedence).
    pub min_cluster_size: usize,
    /// Minimum total speech duration (seconds) a cluster must have to survive —
    /// the length-invariant alternative to `min_cluster_size`. When `> 0`, a
    /// cluster whose overlap-merged window duration is below this is dissolved.
    /// `0.0` disables it (the member-count rule applies instead).
    pub min_cluster_secs: f64,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_AHC_THRESHOLD,
            max_speakers: 64,
            // Pruning singleton clusters (size < 2) cuts over-clustering and
            // lowers DER on real-length audio (VoxConverse-dev collar
            // 7.97%→7.22%, speaker-count off-by-2+ 58→20 on the dev-80 sweep)
            // while staying safe on short clips: a fixed min of 3-4 wins more on
            // long files but wrongly dissolves real minority speakers on short
            // ones (the bundled 26 s clip regresses 6.62%→9.54% at min 3). A
            // length-aware / duration-based prune for the larger gain is future
            // work (see `min_cluster_secs`). `1` disables pruning.
            min_cluster_size: 2,
            // Duration pruning off by default until calibrated; the validated
            // shipped default is the member-count rule above.
            min_cluster_secs: 0.0,
        }
    }
}

/// Configuration for sliding-window embedding extraction.
#[derive(Debug, Clone, Copy)]
pub struct WindowConfig {
    /// Window size for embedding extraction, in seconds.
    pub window_secs: f32,
    /// Hop length between consecutive windows, in seconds.
    pub hop_secs: f32,
    /// Sample rate expected by the embedding model (usually 16000).
    pub sample_rate: SampleRate,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_secs: 1.5,
            hop_secs: 0.75,
            sample_rate: SampleRate::default(),
        }
    }
}

impl WindowConfig {
    /// { self.window_secs >= 0.0 }
    /// `fn window_samples(&self) -> usize`
    /// { ret == (self.window_secs * self.sample_rate.get() as f32) as usize }
    pub fn window_samples(&self) -> usize {
        (self.window_secs * self.sample_rate.get() as f32) as usize
    }

    /// { self.hop_secs >= 0.0 }
    /// `fn hop_samples(&self) -> usize`
    /// { ret == (self.hop_secs * self.sample_rate.get() as f32) as usize }
    pub fn hop_samples(&self) -> usize {
        (self.hop_secs * self.sample_rate.get() as f32) as usize
    }
}

/// Configuration for post-clustering speech filtering.
#[derive(Debug, Clone, Copy)]
pub struct SpeechFilterConfig {
    /// Minimum speech duration to consider for clustering, in seconds.
    pub min_speech_secs: f32,
    /// Maximum gap between same-speaker segments to merge, in seconds.
    pub max_gap_secs: f32,
}

impl Default for SpeechFilterConfig {
    fn default() -> Self {
        Self {
            min_speech_secs: 0.25,
            max_gap_secs: 0.5,
        }
    }
}

/// Configuration shared between online and offline diarizers.
#[derive(Debug, Clone, Copy)]
pub struct DiarizationConfig {
    pub cluster: ClusterConfig,
    pub window: WindowConfig,
    pub speech_filter: SpeechFilterConfig,
    /// Maximum allowed audio duration in seconds (DoS guard).
    pub max_duration_secs: f32,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            cluster: ClusterConfig::default(),
            window: WindowConfig::default(),
            speech_filter: SpeechFilterConfig::default(),
            max_duration_secs: 3600.0,
        }
    }
}

impl DiarizationConfig {
    /// { self.window.window_secs >= 0.0 }
    /// `fn window_samples(&self) -> usize`
    /// { ret == self.window.window_samples() }
    pub fn window_samples(&self) -> usize {
        self.window.window_samples()
    }

    /// { self.window.hop_secs >= 0.0 }
    /// `fn hop_samples(&self) -> usize`
    /// { ret == self.window.hop_samples() }
    pub fn hop_samples(&self) -> usize {
        self.window.hop_samples()
    }

    /// { true }
    /// `fn validate(&self) -> Result<(), ConfigError>`
    /// { ret.is_ok() -> (self.window.window_secs > 0.0 && self.window.hop_secs > 0.0 && self.window.hop_secs <= self.window.window_secs) }
    /// Validate field ranges that downstream code relies on.
    ///
    /// Window geometry is the strict part: `window_secs <= 0`, `hop_secs <= 0`,
    /// or `hop_secs > window_secs` panic the window iterator once speech is
    /// found. The remaining checks reject values that silently produce
    /// garbage: an out-of-range cosine threshold never or always merges, and
    /// non-finite durations confuse segment filtering.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let window_secs = self.window.window_secs;
        if !(window_secs.is_finite() && window_secs > 0.0) {
            return Err(ConfigError::InvalidWindowSecs(window_secs));
        }
        let hop_secs = self.window.hop_secs;
        if !(hop_secs.is_finite() && hop_secs > 0.0) {
            return Err(ConfigError::InvalidHopSecs(hop_secs));
        }
        if hop_secs > window_secs {
            return Err(ConfigError::HopExceedsWindow {
                hop_secs,
                window_secs,
            });
        }
        // Positive-but-tiny geometry quantizes to zero samples (e.g. 1e-9 s at
        // 16 kHz) and would panic the window iterator once speech is found.
        if self.window_samples() == 0 || self.hop_samples() == 0 {
            return Err(ConfigError::SubSampleWindow {
                window_secs,
                hop_secs,
            });
        }
        let threshold = self.cluster.threshold;
        if !(-1.0..=1.0).contains(&threshold) {
            return Err(ConfigError::InvalidThreshold(threshold));
        }
        let min_speech_secs = self.speech_filter.min_speech_secs;
        if !(min_speech_secs.is_finite() && min_speech_secs >= 0.0) {
            return Err(ConfigError::InvalidMinSpeechSecs(min_speech_secs));
        }
        let max_gap_secs = self.speech_filter.max_gap_secs;
        if !(max_gap_secs.is_finite() && max_gap_secs >= 0.0) {
            return Err(ConfigError::InvalidMaxGapSecs(max_gap_secs));
        }
        let max_duration_secs = self.max_duration_secs;
        if !(max_duration_secs.is_finite() && max_duration_secs > 0.0) {
            return Err(ConfigError::InvalidMaxDurationSecs(max_duration_secs));
        }
        Ok(())
    }
}

/// Validation failure for [`DiarizationConfig`] field ranges.
///
/// Produced by [`DiarizationConfig::validate`]; covers values that would
/// otherwise panic downstream (zero or inverted window geometry reaches the
/// window iterator) or silently degrade the output (out-of-range cosine
/// threshold, negative or non-finite durations).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ConfigError {
    /// `window.window_secs` must be finite and > 0.
    #[error("window.window_secs must be finite and > 0, got {0}")]
    InvalidWindowSecs(f32),
    /// `window.hop_secs` must be finite and > 0.
    #[error("window.hop_secs must be finite and > 0, got {0}")]
    InvalidHopSecs(f32),
    /// `window.hop_secs` must not exceed `window.window_secs`.
    #[error("window.hop_secs ({hop_secs}) must be <= window.window_secs ({window_secs})")]
    HopExceedsWindow { hop_secs: f32, window_secs: f32 },
    /// Window geometry must quantize to at least one sample per window and hop
    /// at the configured sample rate.
    #[error(
        "window geometry too small: window_secs={window_secs}, hop_secs={hop_secs} quantize to zero samples"
    )]
    SubSampleWindow { window_secs: f32, hop_secs: f32 },
    /// `cluster.threshold` is a cosine similarity and must lie in [-1, 1].
    #[error("cluster.threshold must be in [-1.0, 1.0], got {0}")]
    InvalidThreshold(f32),
    /// `speech_filter.min_speech_secs` must be finite and >= 0.
    #[error("speech_filter.min_speech_secs must be finite and >= 0, got {0}")]
    InvalidMinSpeechSecs(f32),
    /// `speech_filter.max_gap_secs` must be finite and >= 0.
    #[error("speech_filter.max_gap_secs must be finite and >= 0, got {0}")]
    InvalidMaxGapSecs(f32),
    /// `max_duration_secs` must be finite and > 0.
    #[error("max_duration_secs must be finite and > 0, got {0}")]
    InvalidMaxDurationSecs(f32),
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        DiarizationConfig::default().validate().unwrap();
    }

    #[test]
    fn non_positive_or_non_finite_window_secs_rejected() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let config = DiarizationConfig {
                window: WindowConfig {
                    window_secs: bad,
                    ..WindowConfig::default()
                },
                ..DiarizationConfig::default()
            };
            assert!(
                matches!(config.validate(), Err(ConfigError::InvalidWindowSecs(_))),
                "window_secs={bad}"
            );
        }
    }

    #[test]
    fn non_positive_or_non_finite_hop_secs_rejected() {
        for bad in [0.0f32, -0.5, f32::NAN] {
            let config = DiarizationConfig {
                window: WindowConfig {
                    hop_secs: bad,
                    ..WindowConfig::default()
                },
                ..DiarizationConfig::default()
            };
            assert!(
                matches!(config.validate(), Err(ConfigError::InvalidHopSecs(_))),
                "hop_secs={bad}"
            );
        }
    }

    #[test]
    fn hop_greater_than_window_rejected() {
        let config = DiarizationConfig {
            window: WindowConfig {
                window_secs: 1.0,
                hop_secs: 1.5,
                ..WindowConfig::default()
            },
            ..DiarizationConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::HopExceedsWindow {
                hop_secs: 1.5,
                window_secs: 1.0,
            })
        ));
    }

    #[test]
    fn sub_sample_window_geometry_rejected() {
        // 1e-9 s at 16 kHz quantizes to zero samples: it passes the raw
        // seconds checks but would panic the window iterator downstream.
        let config = DiarizationConfig {
            window: WindowConfig {
                window_secs: 1e-9,
                hop_secs: 1e-9,
                ..WindowConfig::default()
            },
            ..DiarizationConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::SubSampleWindow { .. })
        ));
    }

    #[test]
    fn out_of_range_threshold_rejected() {
        for bad in [-1.5f32, 1.5, f32::NAN] {
            let config = DiarizationConfig {
                cluster: ClusterConfig {
                    threshold: bad,
                    ..ClusterConfig::default()
                },
                ..DiarizationConfig::default()
            };
            assert!(
                matches!(config.validate(), Err(ConfigError::InvalidThreshold(_))),
                "threshold={bad}"
            );
        }
        // Bounds are inclusive: pure merge-everything / merge-nothing configs
        // are extreme but coherent.
        for ok in [-1.0f32, 1.0] {
            let config = DiarizationConfig {
                cluster: ClusterConfig {
                    threshold: ok,
                    ..ClusterConfig::default()
                },
                ..DiarizationConfig::default()
            };
            assert!(config.validate().is_ok(), "threshold={ok}");
        }
    }

    #[test]
    fn negative_speech_filter_values_rejected() {
        let config = DiarizationConfig {
            speech_filter: SpeechFilterConfig {
                min_speech_secs: -0.1,
                ..SpeechFilterConfig::default()
            },
            ..DiarizationConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMinSpeechSecs(_))
        ));
        let config = DiarizationConfig {
            speech_filter: SpeechFilterConfig {
                max_gap_secs: f32::NAN,
                ..SpeechFilterConfig::default()
            },
            ..DiarizationConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMaxGapSecs(_))
        ));
    }

    #[test]
    fn non_positive_max_duration_rejected() {
        for bad in [0.0f32, -10.0, f32::NAN] {
            let config = DiarizationConfig {
                max_duration_secs: bad,
                ..DiarizationConfig::default()
            };
            assert!(
                matches!(
                    config.validate(),
                    Err(ConfigError::InvalidMaxDurationSecs(_))
                ),
                "max_duration_secs={bad}"
            );
        }
    }

    #[test]
    fn infinite_hop_and_max_duration_rejected() {
        let config = DiarizationConfig {
            window: WindowConfig {
                hop_secs: f32::INFINITY,
                ..WindowConfig::default()
            },
            ..DiarizationConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidHopSecs(_))
        ));
        let config = DiarizationConfig {
            max_duration_secs: f32::INFINITY,
            ..DiarizationConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMaxDurationSecs(_))
        ));
    }

    #[test]
    fn sub_sample_hop_alone_rejected() {
        // Window quantizes fine but the hop alone collapses to zero samples.
        let config = DiarizationConfig {
            window: WindowConfig {
                window_secs: 1.0,
                hop_secs: 1e-9,
                ..WindowConfig::default()
            },
            ..DiarizationConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::SubSampleWindow { .. })
        ));
    }

    #[test]
    fn sample_counts_defer_to_window_config() {
        let cfg = DiarizationConfig::default();
        assert_eq!(cfg.window_samples(), cfg.window.window_samples());
        assert_eq!(cfg.hop_samples(), cfg.window.hop_samples());
        assert_eq!(cfg.window_samples(), 24000); // 1.5 s at 16 kHz
        assert_eq!(cfg.hop_samples(), 12000); // 0.75 s at 16 kHz

        let window = WindowConfig {
            window_secs: 2.0,
            hop_secs: 1.0,
            sample_rate: SampleRate::new(8000).unwrap(),
        };
        assert_eq!(window.window_samples(), 16000);
        assert_eq!(window.hop_samples(), 8000);
    }

    #[test]
    fn config_error_display_names_the_field() {
        let cases = [
            (
                ConfigError::InvalidWindowSecs(0.0).to_string(),
                "window.window_secs",
            ),
            (
                ConfigError::InvalidHopSecs(-1.0).to_string(),
                "window.hop_secs",
            ),
            (
                ConfigError::HopExceedsWindow {
                    hop_secs: 2.0,
                    window_secs: 1.0,
                }
                .to_string(),
                "must be <=",
            ),
            (
                ConfigError::SubSampleWindow {
                    window_secs: 1e-9,
                    hop_secs: 1e-9,
                }
                .to_string(),
                "zero samples",
            ),
            (
                ConfigError::InvalidThreshold(2.0).to_string(),
                "cluster.threshold",
            ),
            (
                ConfigError::InvalidMinSpeechSecs(-0.1).to_string(),
                "speech_filter.min_speech_secs",
            ),
            (
                ConfigError::InvalidMaxGapSecs(-0.1).to_string(),
                "speech_filter.max_gap_secs",
            ),
            (
                ConfigError::InvalidMaxDurationSecs(0.0).to_string(),
                "max_duration_secs",
            ),
        ];
        for (msg, needle) in cases {
            assert!(msg.contains(needle), "{msg} missing {needle}");
        }
    }
}
