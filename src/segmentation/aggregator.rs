//! Sliding-window aggregator for powerset segmentation outputs.
//!
//! Combines per-window 7-class logits into file-globally-consistent
//! `RawSegment` outputs. Implementation builds on top of `decoder` and
//! `hungarian` modules (added in Tasks 4 and 2). Pure Rust, wasm-clean.

use crate::segmentation::{RawSegment, SegmentationError};

/// One window's segmentation output: when the window starts and the per-frame logits.
#[derive(Debug, Clone)]
pub struct WindowOutput {
    /// Audio start time of this window, in seconds.
    pub start_time: f32,
    /// Audio end time of this window, in seconds.
    pub end_time: f32,
    /// Flat row-major buffer of `(num_frames, 7)` logits.
    pub logits: Vec<f32>,
    /// Number of frames in this window.
    pub num_frames: usize,
}

impl WindowOutput {
    /// Create a window output, validating shape.
    pub fn new(
        start_time: f32,
        end_time: f32,
        logits: Vec<f32>,
        num_frames: usize,
    ) -> Result<Self, SegmentationError> {
        if logits.len() != num_frames * 7 {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: vec![logits.len()],
            });
        }
        Ok(Self {
            start_time,
            end_time,
            logits,
            num_frames,
        })
    }

    /// Frame stride in seconds (window duration ÷ frame count).
    pub fn frame_stride(&self) -> f32 {
        if self.num_frames == 0 {
            0.0
        } else {
            (self.end_time - self.start_time) / self.num_frames as f32
        }
    }

    /// Convert a per-window frame index to its absolute audio time (seconds).
    pub fn frame_time(&self, frame_idx: usize) -> f32 {
        self.start_time + frame_idx as f32 * self.frame_stride()
    }
}

/// Configuration for aggregation.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Drop run-length-encoded segments shorter than this duration (seconds).
    pub min_segment_secs: f32,
    /// Maximum number of local speakers any single window can produce.
    /// Should match the underlying model's `max_local_speakers`.
    pub max_local_speakers: usize,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            min_segment_secs: 0.0,
            max_local_speakers: 3,
        }
    }
}

/// The sliding-window aggregator. Holds configuration; operates on borrowed
/// window outputs.
pub struct Aggregator {
    config: AggregationConfig,
}

impl Aggregator {
    pub fn new(config: AggregationConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &AggregationConfig {
        &self.config
    }

    /// Aggregate `windows` into file-globally-consistent `RawSegment`s.
    /// Real implementation lands in Task 6.
    pub fn stitch(&self, windows: &[WindowOutput]) -> Result<Vec<RawSegment>, SegmentationError> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        // Placeholder: Task 6 will implement Hungarian-driven stitching.
        Err(SegmentationError::PermutationFailed {
            prev_idx: 0,
            next_idx: 0,
            detail: "not yet implemented; lands in Task 6".to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_output_validates_shape() {
        let ok = WindowOutput::new(0.0, 10.0, vec![0.0; 7 * 5], 5);
        assert!(ok.is_ok());

        let bad = WindowOutput::new(0.0, 10.0, vec![0.0; 13], 5);
        assert!(bad.is_err());
    }

    #[test]
    fn window_frame_stride_matches_duration() {
        let w = WindowOutput::new(0.0, 10.0, vec![0.0; 7 * 100], 100).unwrap();
        assert!((w.frame_stride() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn window_frame_time_is_anchored_at_start() {
        let w = WindowOutput::new(2.5, 12.5, vec![0.0; 7 * 100], 100).unwrap();
        assert!((w.frame_time(0) - 2.5).abs() < 1e-6);
        assert!((w.frame_time(50) - 7.5).abs() < 1e-6);
    }

    #[test]
    fn empty_windows_yields_empty_segments() {
        let agg = Aggregator::new(AggregationConfig::default());
        let segments = agg.stitch(&[]).unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn stitch_returns_not_yet_implemented_error_for_non_empty_input() {
        // Skeleton commit: Task 6 replaces the placeholder.
        let agg = Aggregator::new(AggregationConfig::default());
        let w = WindowOutput::new(0.0, 10.0, vec![0.0; 7 * 100], 100).unwrap();
        let result = agg.stitch(&[w]);
        assert!(result.is_err());
    }

    #[test]
    fn config_default_is_3_speakers() {
        let c = AggregationConfig::default();
        assert_eq!(c.max_local_speakers, 3);
        assert_eq!(c.min_segment_secs, 0.0);
    }
}
