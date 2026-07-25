//! Pipeline configuration bags (cluster / window / speech filter / diarization).
use super::measures::SampleRate;

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
            threshold: 0.45,
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
}
