//! Powerset 7-class decoder for `pyannote/segmentation-3.0`.
//!
//! Each frame's 7-vector of logits is interpreted as one of:
//!
//! | Class | Set | Is overlap |
//! |---|---|---|
//! | 0 | ∅ (silence) | no |
//! | 1 | {0} | no |
//! | 2 | {1} | no |
//! | 3 | {2} | no |
//! | 4 | {0, 1} | yes |
//! | 5 | {0, 2} | yes |
//! | 6 | {1, 2} | yes |
//!
//! The decoder takes argmax over softmax, returning a `FrameLabel`.

use crate::segmentation::SegmentationError;
use crate::types::Confidence;

/// One of the seven powerset classes, identifying which speakers are active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowersetClass {
    Silence,
    Speaker(u8),
    Pair(u8, u8),
}

impl PowersetClass {
    /// True for classes 4–6 (two speakers active simultaneously).
    pub const fn is_overlap(self) -> bool {
        matches!(self, PowersetClass::Pair(_, _))
    }

/// { TODO: precondition }
/// pub fn speakers(self) -> Vec<u8>
/// { TODO: postcondition }
    /// Local speaker indices active in this class.
    pub fn speakers(self) -> Vec<u8> {
        match self {
            PowersetClass::Silence => Vec::new(),
            PowersetClass::Speaker(s) => vec![s],
            PowersetClass::Pair(a, b) => vec![a, b],
        }
    }
}

/// Decoded label for a single audio frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameLabel {
    pub class: PowersetClass,
    /// Maximum-class softmax probability (∈ [0, 1]). Useful for confidence reporting.
    pub max_softmax: f32,
}

/// Stateless decoder; methods are associated functions because no per-instance
/// configuration is needed.
pub struct PowersetDecoder;

impl PowersetDecoder {
    /// Convert a 7-class index (0..=6) to its `PowersetClass`.
    pub const fn class_for_index(idx: usize) -> Option<PowersetClass> {
        match idx {
            0 => Some(PowersetClass::Silence),
            1 => Some(PowersetClass::Speaker(0)),
            2 => Some(PowersetClass::Speaker(1)),
            3 => Some(PowersetClass::Speaker(2)),
            4 => Some(PowersetClass::Pair(0, 1)),
            5 => Some(PowersetClass::Pair(0, 2)),
            6 => Some(PowersetClass::Pair(1, 2)),
            _ => None,
        }
    }

/// { TODO: precondition }
/// pub fn decode_frame(logits: &[f32]) -> Result<FrameLabel, SegmentationError>
/// { TODO: postcondition }
    /// Decode one frame given its 7-vector of logits.
    pub fn decode_frame(logits: &[f32]) -> Result<FrameLabel, SegmentationError> {
        if logits.len() != 7 {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: vec![logits.len()],
            });
        }
        // Stable softmax: subtract max for numerical stability.
        let mut max_logit = f32::NEG_INFINITY;
        for &l in logits {
            if l > max_logit {
                max_logit = l;
            }
        }
        let mut exps = [0.0_f32; 7];
        let mut sum = 0.0_f32;
        for (i, &l) in logits.iter().enumerate() {
            exps[i] = (l - max_logit).exp();
            sum += exps[i];
        }
        // Guard against degenerate sum (sum=0 would only happen with NaN logits).
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        let mut argmax = 0_usize;
        let mut max_softmax = 0.0_f32;
        for (i, &e) in exps.iter().enumerate() {
            let p = e * inv_sum;
            if p > max_softmax {
                max_softmax = p;
                argmax = i;
            }
        }
        let class = Self::class_for_index(argmax).ok_or(SegmentationError::InvalidOutputShape {
            actual_shape: vec![argmax],
        })?;
        Ok(FrameLabel { class, max_softmax })
    }

/// { TODO: precondition }
/// pub fn decode_window( logits_flat: &[f32], num_frames: usize, ) -> Result<Vec<FrameLabel>, SegmentationError>
/// { TODO: postcondition }
    /// Decode every frame in a flat row-major `[num_frames, 7]` buffer.
    pub fn decode_window(
        logits_flat: &[f32],
        num_frames: usize,
    ) -> Result<Vec<FrameLabel>, SegmentationError> {
        if logits_flat.len() != num_frames * 7 {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: vec![logits_flat.len()],
            });
        }
        let mut out = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let frame = &logits_flat[i * 7..(i + 1) * 7];
            out.push(Self::decode_frame(frame)?);
        }
        Ok(out)
    }

/// { TODO: precondition }
/// pub fn frame_confidence(softmax: f32) -> Confidence
/// { TODO: postcondition }
    /// Convert a softmax probability into a `Confidence`. Clamps tiny over-/underflows
    /// to the valid `[0, 1]` range so we never panic on numerical artifacts.
    pub fn frame_confidence(softmax: f32) -> Confidence {
        let clamped = softmax.clamp(0.0, 1.0);
        // `Confidence::new` validates the closed range; clamped is guaranteed valid.
        Confidence::new(clamped).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn class_0_is_silence() {
        let logits = [10.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Silence);
        assert!(!label.class.is_overlap());
    }

    #[test]
    fn class_1_is_speaker_0() {
        let logits = [1.0_f32, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Speaker(0));
    }

    #[test]
    fn class_3_is_speaker_2() {
        let logits = [1.0_f32, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Speaker(2));
    }

    #[test]
    fn class_4_is_overlap_pair_0_1() {
        let logits = [1.0_f32, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Pair(0, 1));
        assert!(label.class.is_overlap());
    }

    #[test]
    fn class_5_is_overlap_pair_0_2() {
        let logits = [1.0_f32, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Pair(0, 2));
    }

    #[test]
    fn class_6_is_overlap_pair_1_2() {
        let logits = [1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Pair(1, 2));
    }

    #[test]
    fn rejects_wrong_logit_count() {
        let logits = [1.0_f32, 2.0, 3.0];
        assert!(PowersetDecoder::decode_frame(&logits).is_err());
    }

    #[test]
    fn max_softmax_is_softmax_of_argmax_class() {
        let logits = [0.0_f32; 7];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert!(approx(label.max_softmax, 1.0 / 7.0));
    }

    #[test]
    fn confidence_clamps_to_valid_range() {
        let logits = [-1e6_f32, -1e6, -1e6, -1e6, -1e6, -1e6, 0.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert!(label.max_softmax > 0.99);
        assert!(label.max_softmax <= 1.0 + 1e-6);
    }

    #[test]
    fn class_method_returns_speaker_set() {
        assert_eq!(PowersetClass::Silence.speakers(), Vec::<u8>::new());
        assert_eq!(PowersetClass::Speaker(0).speakers(), vec![0]);
        assert_eq!(PowersetClass::Pair(0, 2).speakers(), vec![0, 2]);
        assert_eq!(PowersetClass::Pair(1, 2).speakers(), vec![1, 2]);
    }

    #[test]
    fn decode_window_iterates_over_frames() {
        let logits_flat: Vec<f32> = vec![
            10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let labels = PowersetDecoder::decode_window(&logits_flat, 2).unwrap();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].class, PowersetClass::Silence);
        assert_eq!(labels[1].class, PowersetClass::Speaker(1));
    }

    #[test]
    fn decode_window_rejects_misshaped_buffer() {
        let logits_flat = vec![1.0_f32; 8];
        assert!(PowersetDecoder::decode_window(&logits_flat, 1).is_err());
    }

    #[test]
    fn confidence_construction_via_helper() {
        let c = PowersetDecoder::frame_confidence(1.0_f32 + 1e-7);
        assert!((c.get() - 1.0).abs() < 1e-5);

        let c = PowersetDecoder::frame_confidence(-1e-7);
        assert!(c.get() >= 0.0);
    }
}
