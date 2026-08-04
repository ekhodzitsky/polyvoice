//! Calibrated binarization of powerset posteriors (pyannote-style hysteresis).

use crate::segmentation::decoder::{
    MAX_LOCAL_SPEAKERS, NUM_POWERSET_CLASSES, PowersetClass, PowersetDecoder,
};
use crate::vad::hysteresis::{HysteresisGate, RegionEvent, RegionTracker, TailPolicy};

/// Calibrated binarization of segmentation posteriors, pyannote-style: each
/// speaker's activity probability (sum of the powerset classes containing the
/// speaker) is thresholded with onset/offset hysteresis, then short active
/// blips are dropped and short gaps bridged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinarizationConfig {
    /// Enter-speech threshold: a speaker turns ON when its probability
    /// reaches `onset`.
    pub onset: f32,
    /// Leave-speech threshold: an ON speaker turns OFF only when its
    /// probability drops below `offset` (set `offset < onset` for hysteresis).
    pub offset: f32,
    /// Active runs shorter than this many seconds are dropped.
    pub min_duration_on: f32,
    /// Gaps shorter than this many seconds between active runs are bridged.
    pub min_duration_off: f32,
}

impl Default for BinarizationConfig {
    fn default() -> Self {
        Self {
            onset: 0.5,
            offset: 0.5,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
        }
    }
}

/// Binarize averaged powerset posteriors into per-frame classes.
///
/// `avg_probs[g]` are the 7 mean class probabilities for global frame `g`;
/// `has_data[g]` is false where no window covered the frame (emits `None`).
/// Returns per-frame classes + confidences with the same conventions as the
/// argmax path (`None` = uncovered, `Empty` = silence).
// Index loops are deliberate: three parallel per-frame arrays are read and
// written by frame/speaker index, including range writes for region marking.
#[allow(clippy::needless_range_loop)]
pub fn binarize_frames(
    avg_probs: &[[f32; NUM_POWERSET_CLASSES]],
    has_data: &[bool],
    stride: f32,
    cfg: &BinarizationConfig,
) -> (Vec<Option<PowersetClass>>, Vec<f32>) {
    let n = avg_probs.len();
    // Per-speaker activity probability: sum of classes containing the speaker.
    let mut speaker_probs = vec![[0.0_f32; MAX_LOCAL_SPEAKERS]; n];
    for g in 0..n {
        if !has_data[g] {
            continue;
        }
        for c in 0..NUM_POWERSET_CLASSES {
            if let Some(class) = PowersetDecoder::class_for_index(c) {
                for s in class.speakers() {
                    speaker_probs[g][s as usize] += avg_probs[g][c];
                }
            }
        }
    }

    // Hysteresis + duration smoothing per speaker, pyannote order: short gaps
    // bridge first, then short active blips drop. The region tracker bridges
    // inactive runs shorter than `min_off` frames while the region is open,
    // `keeps` drops closed regions shorter than `min_on` frames, and a
    // coverage hole (`has_data == false`) hard-closes the region and resets
    // the gate instead of being bridged. Frame counts round to the nearest
    // whole frame.
    let min_on = (cfg.min_duration_on / stride).round() as usize;
    let min_off = (cfg.min_duration_off / stride).round() as usize;
    let mut active = vec![[false; MAX_LOCAL_SPEAKERS]; n];
    for s in 0..MAX_LOCAL_SPEAKERS {
        let mut gate = HysteresisGate::new(cfg.onset, cfg.offset);
        let mut tracker = RegionTracker::new(min_off, min_on, TailPolicy::Trim);
        for g in 0..n {
            if !has_data[g] {
                gate.reset();
                let event = tracker.reset();
                mark_region(&mut active, s, &tracker, event);
                continue;
            }
            let on = gate.update(speaker_probs[g][s]);
            let event = tracker.advance(on, g);
            mark_region(&mut active, s, &tracker, event);
        }
        let event = tracker.flush(n);
        mark_region(&mut active, s, &tracker, event);
    }

    // Rebuild per-frame classes: 0 active -> Silence, 1 -> solo speaker,
    // 2 -> pair, 3 -> top-2 by probability (powerset expresses at most two
    // speakers).
    let mut classes = Vec::with_capacity(n);
    let mut confidences = Vec::with_capacity(n);
    for g in 0..n {
        if !has_data[g] {
            classes.push(None);
            confidences.push(0.0);
            continue;
        }
        let mut on: Vec<u8> = (0..MAX_LOCAL_SPEAKERS as u8)
            .filter(|&s| active[g][s as usize])
            .collect();
        if on.len() > 2 {
            on.sort_by(|a, b| {
                speaker_probs[g][*b as usize].total_cmp(&speaker_probs[g][*a as usize])
            });
            on.truncate(2);
            on.sort_unstable();
        }
        debug_assert!(
            matches!(on.as_slice(), [] | [_] | [_, _]),
            "at most two speakers survive the top-2 truncation"
        );
        // Checked reverse mapping (speakers -> class): sets the powerset
        // scheme cannot express yield `None` instead of being silently
        // relabeled Silence (they cannot occur after the top-2 truncation).
        let class = PowersetClass::from_speakers(&on);
        classes.push(class);
        let conf = if on.is_empty() {
            avg_probs[g][0]
        } else {
            on.iter()
                .map(|s| speaker_probs[g][*s as usize])
                .sum::<f32>()
                / on.len() as f32
        };
        confidences.push(conf.clamp(0.0, 1.0));
    }
    (classes, confidences)
}

/// Rasterize a closed region that survives the minimum-duration filter into
/// the per-speaker activity track.
fn mark_region(
    active: &mut [[bool; MAX_LOCAL_SPEAKERS]],
    speaker: usize,
    tracker: &RegionTracker,
    event: Option<RegionEvent>,
) {
    if let Some(RegionEvent::End {
        start_frame,
        end_frame,
    }) = event
        && tracker.keeps(start_frame, end_frame)
    {
        for frame in active.iter_mut().take(end_frame).skip(start_frame) {
            frame[speaker] = true;
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    /// One frame of class probabilities with `p` on the solo class of speaker
    /// `s` and the rest on silence.
    fn solo(s: usize, p: f32) -> [f32; NUM_POWERSET_CLASSES] {
        let mut row = [0.0; NUM_POWERSET_CLASSES];
        row[0] = 1.0 - p;
        row[1 + s] = p;
        row
    }

    fn silence() -> [f32; NUM_POWERSET_CLASSES] {
        solo(0, 0.0)
    }

    #[test]
    fn default_config_is_plain_thresholding() {
        let cfg = BinarizationConfig::default();
        assert!((cfg.onset - 0.5).abs() < 1e-6);
        assert!((cfg.offset - 0.5).abs() < 1e-6);
        assert!((cfg.min_duration_on - 0.0).abs() < 1e-6);
        assert!((cfg.min_duration_off - 0.0).abs() < 1e-6);
    }

    #[test]
    fn uncovered_frames_emit_none_and_zero_confidence() {
        let avg = vec![silence(), silence(), silence()];
        let has_data = vec![true, false, true];
        let (classes, confs) =
            binarize_frames(&avg, &has_data, 0.01, &BinarizationConfig::default());
        assert_eq!(classes.len(), 3);
        assert_eq!(confs.len(), 3);
        assert_eq!(classes[0], Some(PowersetClass::Silence));
        assert_eq!(classes[1], None);
        assert_eq!(classes[2], Some(PowersetClass::Silence));
        assert!((confs[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn empty_input_returns_empty_tracks() {
        let (classes, confs) = binarize_frames(&[], &[], 0.01, &BinarizationConfig::default());
        assert!(classes.is_empty());
        assert!(confs.is_empty());
    }

    #[test]
    fn silence_frame_confidence_is_silence_probability() {
        let avg = vec![silence()];
        let has_data = vec![true];
        let (classes, confs) =
            binarize_frames(&avg, &has_data, 0.01, &BinarizationConfig::default());
        assert_eq!(classes[0], Some(PowersetClass::Silence));
        assert!((confs[0] - 1.0).abs() < 1e-6, "got {}", confs[0]);
    }

    #[test]
    fn single_active_speaker_maps_to_solo_class_with_mean_confidence() {
        let avg = vec![silence(), solo(1, 0.9), solo(1, 0.7), silence()];
        let has_data = vec![true; 4];
        let (classes, confs) =
            binarize_frames(&avg, &has_data, 0.01, &BinarizationConfig::default());
        assert_eq!(
            classes,
            vec![
                Some(PowersetClass::Silence),
                Some(PowersetClass::Speaker(1)),
                Some(PowersetClass::Speaker(1)),
                Some(PowersetClass::Silence),
            ]
        );
        assert!((confs[1] - 0.9).abs() < 1e-6, "got {}", confs[1]);
        assert!((confs[2] - 0.7).abs() < 1e-6, "got {}", confs[2]);
    }

    #[test]
    fn two_active_speakers_map_to_pair_class() {
        // Classes {0} and {1} both above onset on the middle frame.
        let mut frame = [0.0; NUM_POWERSET_CLASSES];
        frame[0] = 0.1;
        frame[1] = 0.5; // {0}
        frame[2] = 0.4; // {1}
        let avg = vec![silence(), frame, silence()];
        let has_data = vec![true; 3];
        let cfg = BinarizationConfig {
            onset: 0.2,
            ..BinarizationConfig::default()
        };
        let (classes, confs) = binarize_frames(&avg, &has_data, 0.01, &cfg);
        assert_eq!(classes[1], Some(PowersetClass::Pair(0, 1)));
        assert!((confs[1] - 0.45).abs() < 1e-6, "got {}", confs[1]);
    }

    #[test]
    fn three_active_speakers_truncate_to_top_two_by_probability() {
        // All three solo classes above onset: powerset expresses at most two,
        // so the weakest speaker is dropped.
        let mut frame = [0.0; NUM_POWERSET_CLASSES];
        frame[1] = 0.4; // {0}
        frame[2] = 0.35; // {1}
        frame[3] = 0.25; // {2}
        let avg = vec![frame];
        let has_data = vec![true];
        let cfg = BinarizationConfig {
            onset: 0.2,
            offset: 0.2,
            ..Default::default()
        };
        let (classes, confs) = binarize_frames(&avg, &has_data, 0.01, &cfg);
        assert_eq!(classes[0], Some(PowersetClass::Pair(0, 1)));
        assert!((confs[0] - 0.375).abs() < 1e-6, "got {}", confs[0]);
    }

    #[test]
    fn hysteresis_holds_speaker_on_through_dip_above_offset() {
        let avg = vec![
            silence(),
            solo(0, 0.7),  // crosses onset
            solo(0, 0.45), // between offset and onset: stays ON
            solo(0, 0.1),  // below offset: OFF
        ];
        let has_data = vec![true; 4];
        let cfg = BinarizationConfig {
            onset: 0.6,
            offset: 0.4,
            ..Default::default()
        };
        let (classes, _) = binarize_frames(&avg, &has_data, 1.0, &cfg);
        assert_eq!(
            classes,
            vec![
                Some(PowersetClass::Silence),
                Some(PowersetClass::Speaker(0)),
                Some(PowersetClass::Speaker(0)),
                Some(PowersetClass::Silence),
            ]
        );
    }

    #[test]
    fn short_gap_is_bridged_by_min_duration_off() {
        // One inactive frame between two active runs, min_off = 2 frames.
        let avg = vec![
            silence(),
            solo(0, 0.9),
            solo(0, 0.0),
            solo(0, 0.9),
            silence(),
        ];
        let has_data = vec![true; 5];
        let cfg = BinarizationConfig {
            min_duration_off: 2.0, // stride 1.0 -> 2 frames
            ..Default::default()
        };
        let (classes, _) = binarize_frames(&avg, &has_data, 1.0, &cfg);
        assert_eq!(
            classes,
            vec![
                Some(PowersetClass::Silence),
                Some(PowersetClass::Speaker(0)),
                Some(PowersetClass::Speaker(0)), // bridged
                Some(PowersetClass::Speaker(0)),
                Some(PowersetClass::Silence),
            ]
        );
    }

    #[test]
    fn short_active_blip_is_dropped_by_min_duration_on() {
        // Two active frames, min_on = 3 frames: the run is too short to keep.
        let avg = vec![silence(), solo(0, 0.9), solo(0, 0.9), silence()];
        let has_data = vec![true; 4];
        let cfg = BinarizationConfig {
            min_duration_on: 3.0, // stride 1.0 -> 3 frames
            ..Default::default()
        };
        let (classes, _) = binarize_frames(&avg, &has_data, 1.0, &cfg);
        assert_eq!(classes, vec![Some(PowersetClass::Silence); 4]);
    }

    #[test]
    fn coverage_hole_hard_closes_region_instead_of_bridging() {
        // min_off large enough to bridge any gap, but the uncovered frame must
        // still split the two active runs and emit None.
        let avg = vec![
            solo(0, 0.9),
            solo(0, 0.9),
            silence(),
            solo(1, 0.9),
            solo(1, 0.9),
        ];
        let has_data = vec![true, true, false, true, true];
        let cfg = BinarizationConfig {
            min_duration_off: 10.0, // stride 1.0 -> 10 frames
            ..Default::default()
        };
        let (classes, confs) = binarize_frames(&avg, &has_data, 1.0, &cfg);
        assert_eq!(
            classes,
            vec![
                Some(PowersetClass::Speaker(0)),
                Some(PowersetClass::Speaker(0)),
                None,
                Some(PowersetClass::Speaker(1)),
                Some(PowersetClass::Speaker(1)),
            ]
        );
        assert!((confs[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn speakers_are_independent_tracks() {
        // Speaker 0 active on frame 1, speaker 2 active on frame 2.
        let avg = vec![silence(), solo(0, 0.9), solo(2, 0.9), silence()];
        let has_data = vec![true; 4];
        let (classes, _) = binarize_frames(&avg, &has_data, 0.01, &BinarizationConfig::default());
        assert_eq!(
            classes,
            vec![
                Some(PowersetClass::Silence),
                Some(PowersetClass::Speaker(0)),
                Some(PowersetClass::Speaker(2)),
                Some(PowersetClass::Silence),
            ]
        );
    }
}
