//! Speaker segmentation: powerset-classifier + sliding-window aggregator.
//!
//! Added in v0.6 (M1). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §3.1, §5.3.

mod aggregator;
mod decoder;
mod hungarian;

#[cfg(all(feature = "onnx", feature = "segmentation"))]
mod powerset;

pub use aggregator::{AggregationConfig, Aggregator, WindowOutput};
pub use decoder::{FrameLabel, PowersetClass, PowersetDecoder};

#[cfg(all(feature = "onnx", feature = "segmentation"))]
pub use powerset::{PowersetConfig, PowersetSegmenter};

use crate::types::{Confidence, TimeRange};

/// One contiguous segment attributed to a single local speaker index.
///
/// "Local" means consistent within a single `segment()` call's output (same person ↔
/// same `local_speaker_idx` across all frames of the file). Cross-file global IDs
/// are assigned later by the clusterer (see M3).
#[derive(Debug, Clone, PartialEq)]
pub struct RawSegment {
    /// The temporal span of this segment in seconds, audio-relative.
    pub time: TimeRange,
    /// Speaker index local to this segmentation result. `0..=2` for `powerset-3.0`.
    pub local_speaker_idx: u8,
    /// True if the segmenter classified this region as a 2-speaker overlap.
    /// In that case a *second* segment for the other speaker covers the same
    /// time range with `local_speaker_idx` set to that other speaker.
    pub is_overlap: bool,
    /// Mean per-frame confidence: max-softmax averaged over the frames.
    pub confidence: Confidence,
}

/// A speaker segmentation engine — turns raw audio into spans of speech attributed
/// to local speaker indices, with overlap detection.
#[cfg_attr(test, mockall::automock)]
pub trait Segmenter: Send + Sync {
    /// Segment `audio`. Audio must be 16 kHz mono `f32` PCM.
    ///
    /// **Requires:** `audio.len() >= MIN_AUDIO_SAMPLES` (1600 samples = 0.1s).
    /// **Guarantees on Ok:** segments are sorted by `time.start`; every
    /// `local_speaker_idx < self.max_local_speakers()`; timestamps lie within
    /// `[0, audio.len() / 16000]`.
    fn segment(&self, audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError>;

    /// Max number of distinct local speakers this implementation can output.
    /// `powerset-3.0` ⇒ 3.
    fn max_local_speakers(&self) -> usize;

    /// True if the implementation can detect overlap (two simultaneous speakers).
    /// `powerset-3.0` ⇒ true.
    fn supports_overlap(&self) -> bool;
}

/// Minimum audio length (16 kHz samples) accepted by `Segmenter::segment`.
pub const MIN_AUDIO_SAMPLES: usize = 1600;

/// Errors from `Segmenter` implementations.
#[derive(Debug, thiserror::Error)]
pub enum SegmentationError {
    #[error("audio too short: {actual_secs:.3}s < {min_secs:.3}s required")]
    AudioTooShort { actual_secs: f32, min_secs: f32 },

    #[error("ONNX inference failed at window {window_idx}: {detail}")]
    InferenceFailed { window_idx: usize, detail: String },

    #[error(
        "powerset decoder produced invalid output shape: expected (_, 7), got {actual_shape:?}"
    )]
    InvalidOutputShape { actual_shape: Vec<usize> },

    #[error("speaker permutation matching failed across windows {prev_idx}->{next_idx}: {detail}")]
    PermutationFailed {
        prev_idx: usize,
        next_idx: usize,
        detail: String,
    },

    #[error("model file io error on {path}: {detail}")]
    ModelIo {
        path: std::path::PathBuf,
        detail: String,
    },
}

#[cfg(test)]
mod trait_tests {
    use super::*;
    use crate::types::{Confidence, TimeRange};

    /// A minimal in-memory segmenter used for trait conformance tests.
    struct ConstantSegmenter {
        segments: Vec<RawSegment>,
        max_speakers: usize,
        overlap: bool,
    }

    impl Segmenter for ConstantSegmenter {
        fn segment(&self, _audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError> {
            Ok(self.segments.clone())
        }
        fn max_local_speakers(&self) -> usize {
            self.max_speakers
        }
        fn supports_overlap(&self) -> bool {
            self.overlap
        }
    }

    #[test]
    fn raw_segment_roundtrip() {
        let s = RawSegment {
            time: TimeRange {
                start: 0.5,
                end: 1.5,
            },
            local_speaker_idx: 1,
            is_overlap: true,
            confidence: Confidence::new(0.85).unwrap(),
        };
        assert_eq!(s.local_speaker_idx, 1);
        assert!(s.is_overlap);
        assert!((s.confidence.get() - 0.85).abs() < 1e-6);
    }

    #[test]
    fn segmenter_trait_object_is_dyn_compatible() {
        let cs = ConstantSegmenter {
            segments: vec![],
            max_speakers: 3,
            overlap: true,
        };
        let _boxed: Box<dyn Segmenter> = Box::new(cs);
    }

    #[test]
    fn segmenter_segment_returns_owned_vec() {
        let cs = ConstantSegmenter {
            segments: vec![RawSegment {
                time: TimeRange {
                    start: 0.0,
                    end: 1.0,
                },
                local_speaker_idx: 0,
                is_overlap: false,
                confidence: Confidence::new(1.0).unwrap(),
            }],
            max_speakers: 3,
            overlap: true,
        };
        let out = cs.segment(&[]).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn error_audio_too_short_displays_required_thresholds() {
        let err = SegmentationError::AudioTooShort {
            actual_secs: 0.05,
            min_secs: 0.1,
        };
        let msg = format!("{err}");
        assert!(msg.contains("0.05"));
        assert!(msg.contains("0.1"));
    }
}
