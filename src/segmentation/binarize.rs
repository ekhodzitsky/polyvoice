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
