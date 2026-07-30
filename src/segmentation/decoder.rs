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
//! The decoder takes argmax over softmax, returning a `FrameLabel` that also
//! carries the full softmax vector, so the aggregator can average and remap
//! probabilities without recomputing the softmax from the logits.

use crate::segmentation::SegmentationError;
use crate::types::Confidence;

/// Number of local speakers the powerset scheme addresses: the solo classes
/// `{0}`, `{1}`, `{2}` and the three pair classes built from them.
pub const MAX_LOCAL_SPEAKERS: usize = 3;

/// Number of powerset classes: silence + 3 solo + 3 pairs.
pub const NUM_POWERSET_CLASSES: usize = 7;

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

    /// { true }
    /// `pub fn speakers(self) -> Vec<u8>`
    /// { ret.len() <= 2 }
    /// Local speaker indices active in this class.
    pub fn speakers(self) -> Vec<u8> {
        match self {
            PowersetClass::Silence => Vec::new(),
            PowersetClass::Speaker(s) => vec![s],
            PowersetClass::Pair(a, b) => vec![a, b],
        }
    }

    /// Class index in the 7-class powerset scheme — the inverse of
    /// [`PowersetDecoder::class_for_index`] for the classes it can produce.
    /// Pair order is normalized (`Pair(1, 0)` indexes like `Pair(0, 1)`).
    /// Values outside the scheme (an out-of-range solo speaker or an
    /// unexpressible pair) return 0, matching the historical remap fallback.
    pub(crate) const fn index(self) -> usize {
        match self {
            PowersetClass::Silence => 0,
            PowersetClass::Speaker(s) => 1 + s as usize,
            PowersetClass::Pair(a, b) => {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                match (lo, hi) {
                    (0, 1) => 4,
                    (0, 2) => 5,
                    (1, 2) => 6,
                    _ => 0,
                }
            }
        }
    }

    /// The class whose speaker set is exactly `speakers` — the checked inverse
    /// of [`Self::index`]. Pair order is normalized, so `[1, 0]` yields
    /// `Pair(0, 1)`. Returns `None` for sets the powerset scheme cannot
    /// express (more than two speakers, a duplicated speaker, or an
    /// out-of-range local index) instead of silently mapping to silence.
    pub(crate) fn from_speakers(speakers: &[u8]) -> Option<PowersetClass> {
        match speakers {
            [] => Some(PowersetClass::Silence),
            [s] if (*s as usize) < MAX_LOCAL_SPEAKERS => Some(PowersetClass::Speaker(*s)),
            [a, b] => {
                let (lo, hi) = if a < b { (*a, *b) } else { (*b, *a) };
                match (lo, hi) {
                    (0, 1) => Some(PowersetClass::Pair(0, 1)),
                    (0, 2) => Some(PowersetClass::Pair(0, 2)),
                    (1, 2) => Some(PowersetClass::Pair(1, 2)),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// Decoded label for a single audio frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameLabel {
    pub class: PowersetClass,
    /// Maximum-class softmax probability (∈ [0, 1]). Useful for confidence reporting.
    pub max_softmax: f32,
    /// Full softmax vector over the powerset classes (sums to 1). Carried so
    /// the aggregator can average and permute probabilities without
    /// recomputing the softmax from the logits.
    pub probs: [f32; NUM_POWERSET_CLASSES],
}

/// { true }
/// `pub(crate) fn softmax(logits: &[f32; NUM_POWERSET_CLASSES]) -> [f32; NUM_POWERSET_CLASSES]`
/// { ret.iter().all(|p| p.is_finite()) }
/// Stable softmax over one frame's class logits: subtract the max logit for
/// numerical stability, then normalize. A degenerate zero sum (only possible
/// with NaN logits) falls back to a unit denominator so the output stays finite.
pub(crate) fn softmax(logits: &[f32; NUM_POWERSET_CLASSES]) -> [f32; NUM_POWERSET_CLASSES] {
    let mut max_logit = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_logit {
            max_logit = l;
        }
    }
    let mut probs = [0.0_f32; NUM_POWERSET_CLASSES];
    let mut sum = 0.0_f32;
    for (p, &l) in probs.iter_mut().zip(logits.iter()) {
        *p = (l - max_logit).exp();
        sum += *p;
    }
    // Guard against degenerate sum (sum=0 would only happen with NaN logits).
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }
    probs
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

    /// { true }
    /// `pub fn decode_frame(logits: &[f32]) -> Result<FrameLabel, SegmentationError>`
    /// { true }
    /// Decode one frame given its 7-vector of logits.
    pub fn decode_frame(logits: &[f32]) -> Result<FrameLabel, SegmentationError> {
        if logits.len() != NUM_POWERSET_CLASSES {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: vec![logits.len()],
            });
        }
        let logits: &[f32; NUM_POWERSET_CLASSES] =
            logits
                .first_chunk()
                .ok_or(SegmentationError::InvalidOutputShape {
                    actual_shape: vec![logits.len()],
                })?;
        let probs = softmax(logits);
        let mut argmax = 0_usize;
        let mut max_softmax = 0.0_f32;
        for (i, &p) in probs.iter().enumerate() {
            if p > max_softmax {
                max_softmax = p;
                argmax = i;
            }
        }
        let class = Self::class_for_index(argmax).ok_or(SegmentationError::InvalidOutputShape {
            actual_shape: vec![argmax],
        })?;
        Ok(FrameLabel {
            class,
            max_softmax,
            probs,
        })
    }

    /// { true }
    /// `pub fn decode_window( logits_flat: &[f32], num_frames: usize, ) -> Result<Vec<FrameLabel>, SegmentationError>`
    /// { ret.as_ref().map_or(true, |v| v.len() == num_frames) }
    /// Decode every frame in a flat row-major `[num_frames, 7]` buffer.
    pub fn decode_window(
        logits_flat: &[f32],
        num_frames: usize,
    ) -> Result<Vec<FrameLabel>, SegmentationError> {
        if logits_flat.len() != num_frames * NUM_POWERSET_CLASSES {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: vec![logits_flat.len()],
            });
        }
        let mut out = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let frame = &logits_flat[i * NUM_POWERSET_CLASSES..(i + 1) * NUM_POWERSET_CLASSES];
            out.push(Self::decode_frame(frame)?);
        }
        Ok(out)
    }

    /// { true }
    /// pub fn frame_confidence(softmax: f32) -> Confidence
    /// { ret.get() >= 0.0 && ret.get() <= 1.0 }
    /// Convert a softmax probability into a `Confidence`. Clamps tiny over-/underflows
    /// to the valid `[0, 1]` range so we never panic on numerical artifacts.
    pub fn frame_confidence(softmax: f32) -> Confidence {
        let clamped = softmax.clamp(0.0, 1.0);
        // `Confidence::new` validates the closed range; clamped is guaranteed valid.
        Confidence::new(clamped).unwrap_or_default()
    }
}

#[allow(clippy::unwrap_used)]
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

    #[test]
    fn probs_sum_to_one_and_match_max_softmax() {
        let logits = [0.5_f32, 2.0, -1.0, 0.3, 1.0, -0.2, 0.7];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        let sum: f32 = label.probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs must sum to 1, got {sum}");
        let argmax = label
            .probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert!(approx(label.max_softmax, label.probs[argmax]));
        assert_eq!(PowersetDecoder::class_for_index(argmax), Some(label.class));
    }

    #[test]
    fn class_index_round_trips_through_class_for_index() {
        for idx in 0..NUM_POWERSET_CLASSES {
            let class = PowersetDecoder::class_for_index(idx).unwrap();
            assert_eq!(class.index(), idx, "round-trip failed for {idx}");
            assert_eq!(
                PowersetClass::from_speakers(&class.speakers()),
                Some(class),
                "from_speakers round-trip failed for {idx}"
            );
        }
    }

    #[test]
    fn from_speakers_normalizes_pair_order() {
        assert_eq!(
            PowersetClass::from_speakers(&[1, 0]),
            Some(PowersetClass::Pair(0, 1))
        );
        assert_eq!(
            PowersetClass::from_speakers(&[2, 1]),
            Some(PowersetClass::Pair(1, 2))
        );
    }

    #[test]
    fn from_speakers_rejects_unexpressible_sets() {
        assert_eq!(PowersetClass::from_speakers(&[0, 1, 2]), None);
        assert_eq!(PowersetClass::from_speakers(&[3]), None);
        assert_eq!(PowersetClass::from_speakers(&[1, 1]), None);
        assert_eq!(PowersetClass::from_speakers(&[0, 3]), None);
    }
}
