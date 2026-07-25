//! Configuration for the NVIDIA Streaming Sortformer v2 adapter.
//!
//! The model has a **hard architectural cap of 4 speakers** (four sigmoid
//! heads). Configurations that request more speakers are rejected up front.

/// Hard maximum speakers supported by Sortformer v2.
pub const MAX_SPEAKERS: usize = 4;

/// Sample rate expected by the ONNX export (Hz).
pub const SAMPLE_RATE: u32 = 16_000;

/// Model frame duration in seconds (80 ms).
pub const FRAME_DURATION_SECS: f32 = 0.08;

/// Default streaming geometry (overridable via ONNX metadata).
pub const DEFAULT_CHUNK_LEN: usize = 124;
pub const DEFAULT_FIFO_LEN: usize = 124;
pub const DEFAULT_SPKCACHE_LEN: usize = 188;
pub const DEFAULT_RIGHT_CONTEXT: usize = 1;
pub const SUBSAMPLING: usize = 8;
pub const EMB_DIM: usize = 512;
pub const N_MELS: usize = 128;

/// Errors from Sortformer configuration and inference.
#[derive(Debug, thiserror::Error)]
pub enum SortformerError {
    #[error(
        "Sortformer hard-caps max_speakers at {MAX_SPEAKERS} (four sigmoid heads); \
         got {requested}. Prefer the VBx clusterer path for meetings with more speakers"
    )]
    MaxSpeakersExceeded { requested: usize },
    #[error("Sortformer requires 16 kHz mono audio; got sample_rate={sample_rate}")]
    UnsupportedSampleRate { sample_rate: u32 },
    #[error("feature extraction failed: {0}")]
    Features(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("model load failed: {0}")]
    Load(String),
    #[error("missing ONNX output '{name}' (available: {available:?})")]
    MissingOutput {
        name: &'static str,
        available: Vec<String>,
    },
    #[error("invalid tensor shape: {0}")]
    Shape(String),
}

/// Post-processing thresholds for turning frame-level speaker activity into
/// turns. Defaults match NVIDIA CallHome v2 YAML
/// (`diar_streaming_sortformer_4spk-v2_callhome-part1.yaml`).
#[derive(Debug, Clone, PartialEq)]
pub struct PostProcessConfig {
    pub onset: f32,
    pub offset: f32,
    pub pad_onset: f32,
    pub pad_offset: f32,
    pub min_duration_on: f32,
    pub min_duration_off: f32,
    pub median_window: usize,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self::callhome()
    }
}

impl PostProcessConfig {
    /// CallHome-tuned thresholds (NVIDIA default for v2).
    pub fn callhome() -> Self {
        Self {
            onset: 0.641,
            offset: 0.561,
            pad_onset: 0.229,
            pad_offset: 0.079,
            min_duration_on: 0.511,
            min_duration_off: 0.296,
            median_window: 11,
        }
    }

    /// DIHARD3-tuned thresholds.
    pub fn dihard3() -> Self {
        Self {
            onset: 0.56,
            offset: 1.0,
            pad_onset: 0.063,
            pad_offset: 0.002,
            min_duration_on: 0.007,
            min_duration_off: 0.151,
            median_window: 11,
        }
    }
}

/// Top-level Sortformer adapter configuration.
#[derive(Debug, Clone)]
pub struct SortformerConfig {
    /// Maximum speakers to emit. Must be in `1..=MAX_SPEAKERS`.
    pub max_speakers: usize,
    pub post: PostProcessConfig,
    pub chunk_len: usize,
    pub fifo_len: usize,
    pub spkcache_len: usize,
    pub right_context: usize,
}

impl Default for SortformerConfig {
    fn default() -> Self {
        Self {
            max_speakers: MAX_SPEAKERS,
            post: PostProcessConfig::default(),
            chunk_len: DEFAULT_CHUNK_LEN,
            fifo_len: DEFAULT_FIFO_LEN,
            spkcache_len: DEFAULT_SPKCACHE_LEN,
            right_context: DEFAULT_RIGHT_CONTEXT,
        }
    }
}

impl SortformerConfig {
    /// Validate hard constraints. Callers should run this before loading a
    /// session so misconfiguration fails closed without network/IO work.
    pub fn validate(&self) -> Result<(), SortformerError> {
        if self.max_speakers == 0 || self.max_speakers > MAX_SPEAKERS {
            return Err(SortformerError::MaxSpeakersExceeded {
                requested: self.max_speakers,
            });
        }
        Ok(())
    }

    /// Builder-style setter that enforces the hard speaker cap.
    pub fn with_max_speakers(mut self, n: usize) -> Result<Self, SortformerError> {
        if n == 0 || n > MAX_SPEAKERS {
            return Err(SortformerError::MaxSpeakersExceeded { requested: n });
        }
        self.max_speakers = n;
        Ok(self)
    }

    /// Nominal streaming latency in seconds:
    /// `(chunk_len + right_context) * 80 ms`.
    pub fn latency_secs(&self) -> f32 {
        (self.chunk_len + self.right_context) as f32 * FRAME_DURATION_SECS
    }
}

/// Stable adapter type id registered with [`crate::models::AdapterRegistry`].
pub const ADAPTER_TYPE: &str = "sortformer-v2";

/// Manifest model id for optional download (`src/models/manifest.toml`).
pub const MODEL_ID: &str = "sortformer_v2";

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        SortformerConfig::default().validate().unwrap();
    }

    #[test]
    fn max_speakers_above_four_is_config_error() {
        let err = SortformerConfig::default()
            .with_max_speakers(5)
            .expect_err("must reject");
        match err {
            SortformerError::MaxSpeakersExceeded { requested } => assert_eq!(requested, 5),
            other => panic!("unexpected: {other}"),
        }
        let msg = format!("{err}");
        assert!(msg.contains("4"), "{msg}");
        assert!(
            msg.contains("VBx") || msg.contains("vbx") || msg.contains("speakers"),
            "{msg}"
        );
    }

    #[test]
    fn max_speakers_zero_rejected() {
        let err = SortformerConfig {
            max_speakers: 0,
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert!(matches!(
            err,
            SortformerError::MaxSpeakersExceeded { requested: 0 }
        ));
    }

    #[test]
    fn latency_for_default_chunk() {
        let cfg = SortformerConfig::default();
        // (124 + 1) * 0.08 = 10.0
        assert!((cfg.latency_secs() - 10.0).abs() < 1e-6);
    }
}
