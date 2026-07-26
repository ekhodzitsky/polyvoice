//! Calibrated binarization of powerset posteriors (pyannote-style hysteresis).

use crate::segmentation::decoder::{PowersetClass, PowersetDecoder};

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
// written by frame/speaker index, including range writes for gap bridging.
#[allow(clippy::needless_range_loop)]
pub fn binarize_frames(
    avg_probs: &[[f32; 7]],
    has_data: &[bool],
    stride: f32,
    cfg: &BinarizationConfig,
) -> (Vec<Option<PowersetClass>>, Vec<f32>) {
    let n = avg_probs.len();
    // Per-speaker activity probability: sum of classes containing the speaker.
    let mut speaker_probs = vec![[0.0_f32; 3]; n];
    for g in 0..n {
        if !has_data[g] {
            continue;
        }
        for c in 0..7 {
            if let Some(class) = PowersetDecoder::class_for_index(c) {
                for s in class.speakers() {
                    speaker_probs[g][s as usize] += avg_probs[g][c];
                }
            }
        }
    }

    // Hysteresis per speaker: ON at prob >= onset, OFF only below offset.
    let mut active = vec![[false; 3]; n];
    for s in 0..3 {
        let mut on = false;
        for g in 0..n {
            if !has_data[g] {
                on = false;
                continue;
            }
            let prob = speaker_probs[g][s];
            if on {
                on = prob >= cfg.offset;
            } else {
                on = prob >= cfg.onset;
            }
            active[g][s] = on;
        }
    }

    // Duration smoothing, pyannote order: bridge short gaps first, then drop
    // short blips. Frame counts round to the nearest whole frame.
    let min_on = (cfg.min_duration_on / stride).round() as usize;
    let min_off = (cfg.min_duration_off / stride).round() as usize;
    for s in 0..3 {
        if min_off > 1 {
            let mut last_on: Option<usize> = None;
            for g in 0..n {
                if active[g][s] {
                    if let Some(prev) = last_on {
                        let gap = g - prev - 1;
                        if gap > 0 && gap < min_off && (prev + 1..g).all(|k| has_data[k]) {
                            for k in prev + 1..g {
                                active[k][s] = true;
                            }
                        }
                    }
                    last_on = Some(g);
                }
            }
        }
        if min_on > 1 {
            let mut run_start: Option<usize> = None;
            for g in 0..=n {
                let is_on = g < n && active[g][s];
                match (run_start, is_on) {
                    (None, true) => run_start = Some(g),
                    (Some(start), false) => {
                        if g - start < min_on {
                            for k in start..g {
                                active[k][s] = false;
                            }
                        }
                        run_start = None;
                    }
                    _ => {}
                }
            }
        }
    }

    // Rebuild per-frame classes: 0 active -> Empty, 1 -> solo, 2 -> pair,
    // 3 -> top-2 by probability (powerset expresses at most two speakers).
    let mut classes = Vec::with_capacity(n);
    let mut confidences = Vec::with_capacity(n);
    for g in 0..n {
        if !has_data[g] {
            classes.push(None);
            confidences.push(0.0);
            continue;
        }
        let mut on: Vec<u8> = (0..3u8).filter(|&s| active[g][s as usize]).collect();
        if on.len() > 2 {
            on.sort_by(|a, b| {
                speaker_probs[g][*b as usize].total_cmp(&speaker_probs[g][*a as usize])
            });
            on.truncate(2);
            on.sort_unstable();
        }
        let idx = match on.as_slice() {
            [] => 0,
            [s] => 1 + *s as usize,
            [a, b] => match (*a, *b) {
                (0, 1) => 4,
                (0, 2) => 5,
                _ => 6,
            },
            _ => 0,
        };
        classes.push(PowersetDecoder::class_for_index(idx));
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
