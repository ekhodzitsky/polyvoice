//! Per-(window, local-speaker) embedding + reconstruct.
//!
//! Community-1 / speakrs never stitch windows into file-global tracks before
//! clustering. Each 10 s window keeps its own local speakers; a masked
//! embedding is extracted per (window, speaker); clustering assigns global
//! IDs; reconstruct overlap-adds the mapped masks and takes per-frame top-k
//! from the instantaneous speaker count.

use crate::embedder::Embedder;
use crate::segmentation::{
    FrameLabel, MAX_LOCAL_SPEAKERS, PowersetDecoder, SegmentationError, WindowOutput,
};
use crate::types::{SpeakerId, SpeakerTurn, TimeRange};
use crate::utils::{cosine_similarity, l2_normalize};

/// Minimum unmasked speech (seconds) before a (window, speaker) unit is
/// embedded. Matches the WeSpeaker NaN floor; VBx then drops units shorter
/// than 5 s on the reconstruct path and reassigns them.
const MIN_EMBED_SECS: f64 = 0.20;

/// Use the overlap-excluded mask when it still has at least this much speech;
/// otherwise fall back to the raw (overlap-inclusive) speaker mask.
const CLEAN_FALLBACK_SECS: f64 = 2.0;

/// Local-speaker activity inside one sliding window.
#[derive(Debug, Clone)]
pub struct WindowMasks {
    pub start: f64,
    pub end: f64,
    pub num_frames: usize,
    /// `masks[s][f]` — local speaker `s` is on at frame `f`.
    pub masks: [Vec<bool>; MAX_LOCAL_SPEAKERS],
}

impl WindowMasks {
    fn stride(&self) -> f64 {
        if self.num_frames == 0 {
            0.0
        } else {
            (self.end - self.start) / self.num_frames as f64
        }
    }

    fn active_secs(&self, speaker: usize) -> f64 {
        if speaker >= MAX_LOCAL_SPEAKERS {
            return 0.0;
        }
        self.masks[speaker].iter().filter(|on| **on).count() as f64 * self.stride()
    }

    fn clean_secs(&self, speaker: usize) -> f64 {
        if speaker >= MAX_LOCAL_SPEAKERS {
            return 0.0;
        }
        let n = self.num_frames;
        let mut frames = 0usize;
        for f in 0..n {
            if self.masks[speaker][f]
                && (0..MAX_LOCAL_SPEAKERS)
                    .filter(|&s| self.masks[s][f])
                    .count()
                    == 1
            {
                frames += 1;
            }
        }
        frames as f64 * self.stride()
    }
}

/// One embedding unit: a local speaker inside one window.
#[derive(Debug, Clone)]
pub struct WindowSpeaker {
    pub window_idx: usize,
    pub local: u8,
    pub duration: f64,
}

/// Decode each window's powerset logits into per-local-speaker frame masks.
pub fn window_masks(windows: &[WindowOutput]) -> Result<Vec<WindowMasks>, SegmentationError> {
    let mut out = Vec::with_capacity(windows.len());
    for w in windows {
        let labels = PowersetDecoder::decode_window(&w.logits, w.num_frames)?;
        out.push(masks_from_labels(
            w.start_time as f64,
            w.end_time as f64,
            &labels,
        ));
    }
    Ok(out)
}

fn masks_from_labels(start: f64, end: f64, labels: &[FrameLabel]) -> WindowMasks {
    let n = labels.len();
    let mut masks: [Vec<bool>; MAX_LOCAL_SPEAKERS] = std::array::from_fn(|_| vec![false; n]);
    for (f, lab) in labels.iter().enumerate() {
        for s in lab.class.speakers() {
            if (s as usize) < MAX_LOCAL_SPEAKERS {
                masks[s as usize][f] = true;
            }
        }
    }
    WindowMasks {
        start,
        end,
        num_frames: n,
        masks,
    }
}

/// Crop `chunk` to the bounding box of `mask`, then zero frames that are off.
/// Full-window PCM zeros pollute WeSpeaker stats-pooling; cropping keeps the
/// same (window, speaker) unit but only the speech that the mask selected.
pub fn crop_masked(chunk: &[f32], mask: &[bool], sample_rate: u32, window_secs: f64) -> Vec<f32> {
    let n_frames = mask.len();
    if n_frames == 0 || chunk.is_empty() {
        return Vec::new();
    }
    let first = match mask.iter().position(|&on| on) {
        Some(i) => i,
        None => return Vec::new(),
    };
    let last = mask.iter().rposition(|&on| on).unwrap_or(first);
    let stride = window_secs / n_frames as f64;
    let sr = sample_rate as f64;
    let start = ((first as f64 * stride) * sr).floor() as usize;
    let end = (((last + 1) as f64 * stride) * sr).ceil() as usize;
    let end = end.min(chunk.len());
    if end <= start {
        return Vec::new();
    }
    let cropped = &chunk[start..end];
    let sub_secs = (end - start) as f64 / sr;
    let sub_mask: Vec<bool> = mask[first..=last].to_vec();
    mask_pcm(cropped, &sub_mask, sample_rate, sub_secs)
}

/// PCM-multiply `chunk` by a frame mask covering the same time span.
pub fn mask_pcm(chunk: &[f32], mask: &[bool], sample_rate: u32, window_secs: f64) -> Vec<f32> {
    let n_frames = mask.len();
    if n_frames == 0 || chunk.is_empty() {
        return chunk.to_vec();
    }
    let stride = window_secs / n_frames as f64;
    let sr = sample_rate as f64;
    let mut out = chunk.to_vec();
    for (i, sample) in out.iter_mut().enumerate() {
        let t = i as f64 / sr;
        let f = ((t / stride).floor() as usize).min(n_frames - 1);
        if !mask[f] {
            *sample = 0.0;
        }
    }
    out
}

fn speaker_mask(wm: &WindowMasks, speaker: usize, clean: bool) -> Vec<bool> {
    let n = wm.num_frames;
    let mut out = vec![false; n];
    if speaker >= MAX_LOCAL_SPEAKERS {
        return out;
    }
    for (f, slot) in out.iter_mut().enumerate() {
        if !wm.masks[speaker][f] {
            continue;
        }
        *slot = if clean {
            wm.masks.iter().filter(|m| m[f]).count() == 1
        } else {
            true
        };
    }
    out
}

/// Collect embeddable (window, speaker) units and their masked PCM.
pub fn collect_units(
    masks: &[WindowMasks],
    samples: &[f32],
    sample_rate: u32,
) -> (Vec<WindowSpeaker>, Vec<Vec<f32>>) {
    let sr = sample_rate as f64;
    let mut units = Vec::new();
    let mut chunks = Vec::new();
    for (w_idx, wm) in masks.iter().enumerate() {
        let start_i = (wm.start * sr).floor() as usize;
        let end_i = ((wm.end * sr).ceil() as usize).min(samples.len());
        if end_i <= start_i {
            continue;
        }
        let chunk = &samples[start_i..end_i];
        let window_secs = wm.end - wm.start;
        for s in 0..MAX_LOCAL_SPEAKERS {
            let raw_secs = wm.active_secs(s);
            if raw_secs < MIN_EMBED_SECS {
                continue;
            }
            let clean_secs = wm.clean_secs(s);
            let use_clean = clean_secs >= CLEAN_FALLBACK_SECS;
            let mask = speaker_mask(wm, s, use_clean);
            let masked = crop_masked(chunk, &mask, sample_rate, window_secs);
            if masked.len() < (MIN_EMBED_SECS * sample_rate as f64) as usize {
                continue;
            }
            units.push(WindowSpeaker {
                window_idx: w_idx,
                local: s as u8,
                duration: if use_clean { clean_secs } else { raw_secs },
            });
            chunks.push(masked);
        }
    }
    (units, chunks)
}

/// Embed the masked chunks. Empty input yields empty embeddings.
pub fn embed_units(
    embedder: &dyn Embedder,
    chunks: &[Vec<f32>],
    wants_raw: bool,
) -> Result<Vec<Vec<f32>>, crate::embedder::EmbedderError> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    let refs: Vec<&[f32]> = chunks.iter().map(Vec::as_slice).collect();
    let mut embs = embedder.embed_batch(&refs)?;
    if !wants_raw {
        for e in &mut embs {
            l2_normalize(e);
        }
    }
    Ok(embs)
}

/// Same-window locals must not share a global cluster. On a clash, the
/// embedding farther from the taken centroid is reassigned to the next-best
/// unused centroid.
pub fn repair_window_conflicts(
    units: &[WindowSpeaker],
    embeddings: &[Vec<f32>],
    labels: &mut [usize],
) {
    if units.len() != labels.len() || units.len() != embeddings.len() {
        return;
    }
    let k = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if k <= 1 {
        return;
    }
    let mut sums = vec![vec![0.0f32; embeddings[0].len()]; k];
    let mut counts = vec![0usize; k];
    for (e, &lab) in embeddings.iter().zip(labels.iter()) {
        if lab >= k || e.len() != sums[lab].len() {
            continue;
        }
        counts[lab] += 1;
        for (s, &v) in sums[lab].iter_mut().zip(e.iter()) {
            *s += v;
        }
    }
    let mut centroids = Vec::with_capacity(k);
    for (sum, n) in sums.into_iter().zip(counts) {
        let mut c = sum;
        if n > 0 {
            let inv = 1.0 / n as f32;
            for v in &mut c {
                *v *= inv;
            }
            l2_normalize(&mut c);
        }
        centroids.push(c);
    }

    let mut by_window: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, u) in units.iter().enumerate() {
        by_window.entry(u.window_idx).or_default().push(i);
    }
    for idxs in by_window.values() {
        if idxs.len() < 2 {
            continue;
        }
        let mut taken: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &i in idxs {
            let g = labels[i];
            if let Some(&prev) = taken.get(&g) {
                // Keep the closer of the two on `g`; move the other.
                let sim_i = cosine_similarity(&embeddings[i], &centroids[g]);
                let sim_p = cosine_similarity(&embeddings[prev], &centroids[g]);
                let mover = if sim_i < sim_p { i } else { prev };
                let stay = if mover == i { prev } else { i };
                taken.insert(g, stay);
                let used: std::collections::HashSet<usize> =
                    idxs.iter().map(|&j| labels[j]).collect();
                let mut best = g;
                let mut best_sim = f32::NEG_INFINITY;
                for (cidx, c) in centroids.iter().enumerate() {
                    if used.contains(&cidx) {
                        continue;
                    }
                    let s = cosine_similarity(&embeddings[mover], c);
                    if s > best_sim {
                        best_sim = s;
                        best = cidx;
                    }
                }
                if best != g {
                    labels[mover] = best;
                    taken.insert(best, mover);
                }
            } else {
                taken.insert(g, i);
            }
        }
    }
}

/// Overlap-add mapped masks, take per-frame top-`count` speakers, RLE to turns.
pub fn reconstruct_turns(
    masks: &[WindowMasks],
    units: &[WindowSpeaker],
    labels: &[usize],
    audio_secs: f64,
    min_speech_secs: f64,
    max_gap_secs: f64,
) -> Vec<SpeakerTurn> {
    if masks.is_empty() || units.is_empty() {
        return Vec::new();
    }
    let stride = masks
        .iter()
        .find(|w| w.num_frames > 0)
        .map(WindowMasks::stride)
        .unwrap_or(0.016);
    if stride <= 0.0 || audio_secs <= 0.0 {
        return Vec::new();
    }
    let n_frames = ((audio_secs / stride).ceil() as usize).max(1);
    let n_spk = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if n_spk == 0 {
        return Vec::new();
    }

    let mut activations = vec![vec![0.0f32; n_spk]; n_frames];
    let mut count_sum = vec![0.0f32; n_frames];
    let mut count_n = vec![0u32; n_frames];

    let mut global_of: Vec<Vec<Option<usize>>> = vec![vec![None; MAX_LOCAL_SPEAKERS]; masks.len()];
    for (u, &lab) in units.iter().zip(labels.iter()) {
        if u.window_idx < global_of.len() && (u.local as usize) < MAX_LOCAL_SPEAKERS {
            global_of[u.window_idx][u.local as usize] = Some(lab);
        }
    }

    for (w_idx, wm) in masks.iter().enumerate() {
        let w_stride = wm.stride();
        if w_stride <= 0.0 {
            continue;
        }
        for f in 0..wm.num_frames {
            let t = wm.start + (f as f64 + 0.5) * w_stride;
            let fi = ((t / stride).floor() as usize).min(n_frames - 1);
            let mut n_on = 0u32;
            for (s, local_mask) in wm.masks.iter().enumerate() {
                if local_mask.get(f).copied().unwrap_or(false) {
                    n_on += 1;
                    if let Some(g) = global_of[w_idx].get(s).copied().flatten()
                        && g < n_spk
                    {
                        activations[fi][g] += 1.0;
                    }
                }
            }
            count_sum[fi] += n_on as f32;
            count_n[fi] += 1;
        }
    }

    let mut on = vec![vec![false; n_frames]; n_spk];
    for t in 0..n_frames {
        let c = if count_n[t] == 0 {
            0usize
        } else {
            (count_sum[t] / count_n[t] as f32)
                .round()
                .clamp(0.0, n_spk as f32) as usize
        };
        if c == 0 {
            continue;
        }
        let mut order: Vec<usize> = (0..n_spk).collect();
        order.sort_by(|&a, &b| {
            activations[t][b]
                .total_cmp(&activations[t][a])
                .then(a.cmp(&b))
        });
        for &g in order.iter().take(c) {
            if activations[t][g] > 0.0 {
                on[g][t] = true;
            }
        }
    }

    let mut turns = Vec::new();
    for (g, track) in on.iter().enumerate() {
        let mut i = 0usize;
        while i < n_frames {
            if !track[i] {
                i += 1;
                continue;
            }
            let start = i;
            while i < n_frames && track[i] {
                i += 1;
            }
            let t0 = start as f64 * stride;
            let t1 = (i as f64 * stride).min(audio_secs);
            if t1 - t0 >= min_speech_secs {
                turns.push(SpeakerTurn {
                    speaker: SpeakerId(g as u32),
                    time: TimeRange { start: t0, end: t1 },
                    text: None,
                    stable: true,
                });
            }
        }
    }

    if max_gap_secs > 0.0 {
        turns.sort_by(|a, b| {
            a.speaker
                .0
                .cmp(&b.speaker.0)
                .then(a.time.start.total_cmp(&b.time.start))
        });
        let mut merged: Vec<SpeakerTurn> = Vec::new();
        for t in turns {
            if let Some(last) = merged.last_mut()
                && last.speaker == t.speaker
                && t.time.start - last.time.end <= max_gap_secs
            {
                last.time.end = last.time.end.max(t.time.end);
            } else {
                merged.push(t);
            }
        }
        turns = merged;
    }
    turns.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));
    turns
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    const NUM_POWERSET_CLASSES: usize = 7;

    fn window(start: f32, end: f32, class_per_frame: &[usize]) -> WindowOutput {
        let n = class_per_frame.len();
        let mut logits = vec![0.0f32; n * NUM_POWERSET_CLASSES];
        for (f, &c) in class_per_frame.iter().enumerate() {
            logits[f * NUM_POWERSET_CLASSES + c] = 10.0;
        }
        WindowOutput::new(start, end, logits, n).unwrap()
    }

    #[test]
    fn window_masks_decode_solo_and_overlap() {
        // class 1 = speaker 0, class 4 = pair 0+1
        let w = window(0.0, 1.0, &[1, 4, 1]);
        let masks = window_masks(&[w]).unwrap();
        assert_eq!(masks[0].num_frames, 3);
        assert_eq!(masks[0].masks[0], vec![true, true, true]);
        assert_eq!(masks[0].masks[1], vec![false, true, false]);
        assert_eq!(masks[0].masks[2], vec![false, false, false]);
    }

    #[test]
    fn crop_masked_drops_leading_and_trailing_silence() {
        let chunk = vec![1.0f32; 8];
        let mask = vec![false, true];
        let out = crop_masked(&chunk, &mask, 4, 2.0);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|&v| v != 0.0));
    }

    #[test]
    fn mask_pcm_zeros_off_frames() {
        let chunk = vec![1.0f32; 8];
        let mask = vec![true, false];
        let out = mask_pcm(&chunk, &mask, 4, 2.0);
        assert!(out[0] != 0.0 && out[1] != 0.0 && out[2] != 0.0 && out[3] != 0.0);
        assert_eq!(&out[4..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn reconstruct_two_windows_one_speaker() {
        let w0 = window(0.0, 2.0, &[1, 1]);
        let w1 = window(1.0, 3.0, &[1, 1]);
        let masks = window_masks(&[w0, w1]).unwrap();
        let units = vec![
            WindowSpeaker {
                window_idx: 0,
                local: 0,
                duration: 2.0,
            },
            WindowSpeaker {
                window_idx: 1,
                local: 0,
                duration: 2.0,
            },
        ];
        let turns = reconstruct_turns(&masks, &units, &[0, 0], 3.0, 0.0, 0.0);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker, SpeakerId(0));
        assert!(turns[0].time.start < 0.2);
        assert!(turns[0].time.end > 2.8);
    }

    #[test]
    fn reconstruct_overlap_keeps_two_speakers() {
        let w = window(0.0, 1.0, &[4, 4]); // pair 0+1
        let masks = window_masks(&[w]).unwrap();
        let units = vec![
            WindowSpeaker {
                window_idx: 0,
                local: 0,
                duration: 1.0,
            },
            WindowSpeaker {
                window_idx: 0,
                local: 1,
                duration: 1.0,
            },
        ];
        let turns = reconstruct_turns(&masks, &units, &[0, 1], 1.0, 0.0, 0.0);
        let speakers: std::collections::HashSet<_> = turns.iter().map(|t| t.speaker.0).collect();
        assert_eq!(speakers.len(), 2);
    }

    #[test]
    fn repair_splits_same_window_clash() {
        let units = vec![
            WindowSpeaker {
                window_idx: 0,
                local: 0,
                duration: 1.0,
            },
            WindowSpeaker {
                window_idx: 0,
                local: 1,
                duration: 1.0,
            },
        ];
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut labels = vec![0, 0];
        repair_window_conflicts(&units, &embeddings, &mut labels);
        // Only one cluster exists, so the clash cannot be repaired — labels stay.
        assert_eq!(labels, vec![0, 0]);

        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.9, 0.1]];
        let units = vec![
            WindowSpeaker {
                window_idx: 0,
                local: 0,
                duration: 1.0,
            },
            WindowSpeaker {
                window_idx: 0,
                local: 1,
                duration: 1.0,
            },
            WindowSpeaker {
                window_idx: 1,
                local: 0,
                duration: 1.0,
            },
        ];
        let mut labels = vec![0, 0, 1];
        repair_window_conflicts(&units, &embeddings, &mut labels);
        assert_ne!(
            labels[0], labels[1],
            "same-window locals must not share a cluster: {labels:?}"
        );
    }
}
