//! Sliding-window aggregator for powerset segmentation outputs.
//!
//! Combines per-window 7-class logits into file-globally-consistent
//! `RawSegment` outputs.
//!
//! Algorithm:
//! 1. For each adjacent window pair (i, i+1), build the 3×3 IoU matrix between
//!    speaker masks in the temporal overlap region. Each speaker mask is the
//!    union of frames where the window's argmax label includes that speaker.
//! 2. Use Kuhn-Munkres on `-IoU` to find the assignment that maps window i+1's
//!    local indices onto window i's. Apply the permutation so the same person
//!    has the same index file-wide.
//! 3. For every audio frame, average the per-class logits across every window
//!    that contains that frame.
//! 4. Argmax each averaged logit vector → frame label.
//! 5. Run-length encode consecutive identical labels into `RawSegment`s.

use crate::segmentation::decoder::{FrameLabel, PowersetClass, PowersetDecoder};
use crate::segmentation::hungarian;
use crate::segmentation::{RawSegment, SegmentationError};
use crate::types::TimeRange;

/// One window's segmentation output.
#[derive(Debug, Clone)]
pub struct WindowOutput {
    pub start_time: f32,
    pub end_time: f32,
    /// Row-major `(num_frames, 7)` logits.
    pub logits: Vec<f32>,
    pub num_frames: usize,
}

impl WindowOutput {
    /// { true }
    /// `pub fn new( start_time: f32, end_time: f32, logits: Vec<f32>, num_frames: usize, ) -> Result<Self, SegmentationError>`
    /// { ret.as_ref().map_or(true, |w| w.logits.len() == w.num_frames * 7) }
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

    /// { true }
    /// pub fn frame_stride(&self) -> f32
    /// { ret >= 0.0 || self.num_frames == 0 }
    pub fn frame_stride(&self) -> f32 {
        if self.num_frames == 0 {
            0.0
        } else {
            (self.end_time - self.start_time) / self.num_frames as f32
        }
    }

    /// { true }
    /// pub fn frame_time(&self, frame_idx: usize) -> f32
    /// { ret == self.start_time + frame_idx as f32 * self.frame_stride() }
    pub fn frame_time(&self, frame_idx: usize) -> f32 {
        self.start_time + frame_idx as f32 * self.frame_stride()
    }
}

/// Configuration for aggregation.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    pub min_segment_secs: f32,
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

/// Aggregator over sliding-window powerset outputs.
pub struct Aggregator {
    config: AggregationConfig,
}

impl Aggregator {
    /// { true }
    /// pub fn new(config: AggregationConfig) -> Self
    /// { true }
    pub fn new(config: AggregationConfig) -> Self {
        Self { config }
    }

    /// { true }
    /// pub fn config(&self) -> &AggregationConfig
    /// { ret == &self.config }
    pub fn config(&self) -> &AggregationConfig {
        &self.config
    }

    /// { true }
    /// `pub fn stitch(&self, windows: &[WindowOutput]) -> Result<Vec<RawSegment>, SegmentationError>`
    /// { true }
    /// Stitch overlapping windows into file-consistent `RawSegment`s.
    pub fn stitch(&self, windows: &[WindowOutput]) -> Result<Vec<RawSegment>, SegmentationError> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }

        // 1) For each window, compute per-frame argmax labels.
        let mut window_labels: Vec<Vec<FrameLabel>> = Vec::with_capacity(windows.len());
        for w in windows {
            let labels = PowersetDecoder::decode_window(&w.logits, w.num_frames)?;
            window_labels.push(labels);
        }

        // 2) Hungarian-align each adjacent window pair: permute window i+1's local
        // speaker indices onto window i's reference frame.
        let mut permutations: Vec<[u8; 3]> =
            std::iter::repeat_n([0u8, 1u8, 2u8], windows.len()).collect();

        for i in 1..windows.len() {
            let perm = self.window_permutation(
                &windows[i - 1],
                &window_labels[i - 1],
                &windows[i],
                &window_labels[i],
                &permutations[i - 1],
            )?;
            // Compose: window i's labels are *first* permuted by `perm`, *then*
            // by the cumulative permutation up to window i-1.
            let prev = permutations[i - 1];
            let composed: [u8; 3] = [
                prev[perm[0] as usize],
                prev[perm[1] as usize],
                prev[perm[2] as usize],
            ];
            permutations[i] = composed;
        }

        // 3-5) For every audio frame across the file, average per-class logits over
        // every window that contains it. Argmax + run-length encode.
        self.average_and_run_length_encode(windows, &window_labels, &permutations)
    }

    /// Compute the permutation that maps window B's local indices onto A's frame.
    fn window_permutation(
        &self,
        a: &WindowOutput,
        a_labels: &[FrameLabel],
        b: &WindowOutput,
        b_labels: &[FrameLabel],
        a_perm_so_far: &[u8; 3],
    ) -> Result<[u8; 3], SegmentationError> {
        let n = self.config.max_local_speakers.min(3);
        let overlap_start = a.start_time.max(b.start_time);
        let overlap_end = a.end_time.min(b.end_time);
        if overlap_end <= overlap_start || n == 0 {
            return Ok([0, 1, 2]);
        }

        let stride = a.frame_stride().max(1e-6);
        let grid_len = ((overlap_end - overlap_start) / stride).ceil() as usize;
        if grid_len == 0 {
            return Ok([0, 1, 2]);
        }

        let mut a_masks = vec![vec![false; grid_len]; 3];
        let mut b_masks = vec![vec![false; grid_len]; 3];
        for k in 0..grid_len {
            let t = overlap_start + k as f32 * stride;
            if let Some(idx_a) = self.frame_index_at(a, t)
                && idx_a < a_labels.len()
            {
                for s in a_labels[idx_a].class.speakers() {
                    if (s as usize) < 3 {
                        let permuted = a_perm_so_far[s as usize] as usize;
                        if permuted < 3 {
                            a_masks[permuted][k] = true;
                        }
                    }
                }
            }
            if let Some(idx_b) = self.frame_index_at(b, t)
                && idx_b < b_labels.len()
            {
                for s in b_labels[idx_b].class.speakers() {
                    if (s as usize) < 3 {
                        b_masks[s as usize][k] = true;
                    }
                }
            }
        }

        // Build cost matrix: C[a][b] = -IoU(a_mask, b_mask).
        let mut cost: Vec<Vec<f32>> = vec![vec![0.0_f32; n]; n];
        let a_active_count = a_masks
            .iter()
            .take(n)
            .filter(|m| m.iter().any(|&x| x))
            .count();
        let b_active_count = b_masks
            .iter()
            .take(n)
            .filter(|m| m.iter().any(|&x| x))
            .count();

        for ai in 0..n {
            for bi in 0..n {
                let mut inter = 0_usize;
                let mut uni = 0_usize;
                for k in 0..grid_len {
                    let ax = a_masks[ai][k];
                    let bx = b_masks[bi][k];
                    if ax && bx {
                        inter += 1;
                    }
                    if ax || bx {
                        uni += 1;
                    }
                }
                let iou = if uni == 0 {
                    0.0
                } else {
                    inter as f32 / uni as f32
                };
                cost[ai][bi] = -iou;
            }
        }

        // If fewer than 2 speakers are active on either side in the overlap, we
        // cannot reliably determine the full permutation — return identity.
        if a_active_count < 2 || b_active_count < 2 {
            return Ok([0, 1, 2]);
        }

        let assignment =
            hungarian::solve(&cost).ok_or_else(|| SegmentationError::PermutationFailed {
                prev_idx: 0,
                next_idx: 0,
                detail: "non-square cost matrix".to_owned(),
            })?;

        // assignment[i] = j means: row i (A's global speaker i) best matches column j
        // (B's local speaker j). We want perm[b_local] = a_global, i.e. the direct
        // mapping: perm[j] = i.
        let mut perm = [0_u8, 1_u8, 2_u8];
        for (i, &j) in assignment.iter().enumerate() {
            if j < 3 && i < 3 {
                perm[j] = i as u8;
            }
        }
        Ok(perm)
    }

    /// Find the frame index in `w` whose center is closest to time `t`. Returns
    /// `None` if `t` is outside the window's span.
    fn frame_index_at(&self, w: &WindowOutput, t: f32) -> Option<usize> {
        if t < w.start_time || t > w.end_time || w.num_frames == 0 {
            return None;
        }
        let stride = w.frame_stride();
        if stride <= 0.0 {
            return None;
        }
        let idx = ((t - w.start_time) / stride).floor() as usize;
        Some(idx.min(w.num_frames - 1))
    }

    /// Average per-class logits across windows that contain each global frame,
    /// then argmax + run-length encode into `RawSegment`s.
    fn average_and_run_length_encode(
        &self,
        windows: &[WindowOutput],
        window_labels: &[Vec<FrameLabel>],
        permutations: &[[u8; 3]],
    ) -> Result<Vec<RawSegment>, SegmentationError> {
        let stride = windows[0].frame_stride().max(1e-6);
        let global_start = windows
            .iter()
            .map(|w| w.start_time)
            .fold(f32::INFINITY, f32::min);
        let global_end = windows
            .iter()
            .map(|w| w.end_time)
            .fold(f32::NEG_INFINITY, f32::max);
        let global_frames = ((global_end - global_start) / stride).ceil() as usize;

        let mut summed_probs = vec![[0.0_f32; 7]; global_frames];
        let mut counts = vec![0_u32; global_frames];

        for (wi, w) in windows.iter().enumerate() {
            let perm = permutations[wi];
            for f in 0..w.num_frames {
                let t_center = w.frame_time(f) + 0.5 * stride;
                let g_idx_f = (t_center - global_start) / stride;
                if g_idx_f < 0.0 {
                    continue;
                }
                let g_idx = g_idx_f.floor() as usize;
                if g_idx >= global_frames {
                    continue;
                }
                if window_labels[wi].get(f).is_none() {
                    continue;
                }

                let frame_logits = &w.logits[f * 7..(f + 1) * 7];

                let mut max_logit = f32::NEG_INFINITY;
                for &l in frame_logits {
                    if l > max_logit {
                        max_logit = l;
                    }
                }
                let mut exps = [0.0_f32; 7];
                let mut sum = 0.0_f32;
                for (i, &l) in frame_logits.iter().enumerate() {
                    exps[i] = (l - max_logit).exp();
                    sum += exps[i];
                }
                let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };

                // Apply permutation: rebuild the softmax vector under the file-global
                // speaker ordering by remapping each class's speaker set through `perm`.
                let mut remapped = [0.0_f32; 7];
                for (c, _) in exps.iter().enumerate() {
                    if let Some(class) = PowersetDecoder::class_for_index(c) {
                        let speakers = class.speakers();
                        let remapped_speakers: Vec<u8> = speakers
                            .iter()
                            .map(|s| {
                                if (*s as usize) < 3 {
                                    perm[*s as usize]
                                } else {
                                    *s
                                }
                            })
                            .collect();
                        let new_class = match remapped_speakers.as_slice() {
                            [] => 0,
                            [s] => 1 + (*s as usize),
                            [a, b] => {
                                let (lo, hi) = if a < b {
                                    (*a as usize, *b as usize)
                                } else {
                                    (*b as usize, *a as usize)
                                };
                                match (lo, hi) {
                                    (0, 1) => 4,
                                    (0, 2) => 5,
                                    (1, 2) => 6,
                                    _ => 0,
                                }
                            }
                            _ => 0,
                        };
                        remapped[new_class] += exps[c] * inv_sum;
                    }
                }

                for (i, &p) in remapped.iter().enumerate() {
                    summed_probs[g_idx][i] += p;
                }
                counts[g_idx] += 1;
            }
        }

        let mut frame_classes: Vec<Option<PowersetClass>> = Vec::with_capacity(global_frames);
        let mut frame_confidences: Vec<f32> = Vec::with_capacity(global_frames);
        for g in 0..global_frames {
            if counts[g] == 0 {
                frame_classes.push(None);
                frame_confidences.push(0.0);
                continue;
            }
            let inv = 1.0 / counts[g] as f32;
            let mut argmax = 0_usize;
            let mut maxp = 0.0_f32;
            for (c, &sp) in summed_probs[g].iter().enumerate() {
                let p = sp * inv;
                if p > maxp {
                    maxp = p;
                    argmax = c;
                }
            }
            frame_classes.push(PowersetDecoder::class_for_index(argmax));
            frame_confidences.push(maxp);
        }

        // Run-length encode per speaker.
        let mut segments: Vec<RawSegment> = Vec::new();
        // (start_global_frame, confidence_sum, confidence_count)
        let mut active: [Option<(usize, f32, f32)>; 3] = [None, None, None];

        for g in 0..global_frames {
            let frame_class = frame_classes[g];
            let conf = frame_confidences[g];
            let active_speakers: Vec<u8> = match frame_class {
                Some(c) => c.speakers(),
                None => Vec::new(),
            };

            for (s, slot) in active.iter_mut().enumerate() {
                let s_active_now = active_speakers.iter().any(|x| *x as usize == s);
                match (*slot, s_active_now) {
                    (None, true) => {
                        *slot = Some((g, conf, 1.0));
                    }
                    (Some((start_g, conf_sum, conf_count)), true) => {
                        *slot = Some((start_g, conf_sum + conf, conf_count + 1.0));
                    }
                    (Some((start_g, conf_sum, conf_count)), false) => {
                        let start_t = global_start + start_g as f32 * stride;
                        let end_t = global_start + g as f32 * stride;
                        let dur = end_t - start_t;
                        if dur >= self.config.min_segment_secs {
                            let mean_conf = (conf_sum / conf_count.max(1.0)).clamp(0.0, 1.0);
                            let had_overlap = (start_g..g).any(|gg| {
                                frame_classes[gg]
                                    .map(|c| {
                                        c.is_overlap()
                                            && c.speakers().iter().any(|x| *x as usize == s)
                                    })
                                    .unwrap_or(false)
                            });
                            segments.push(RawSegment {
                                time: TimeRange {
                                    start: start_t as f64,
                                    end: end_t as f64,
                                },
                                local_speaker_idx: s as u8,
                                is_overlap: had_overlap,
                                confidence: PowersetDecoder::frame_confidence(mean_conf),
                            });
                        }
                        *slot = None;
                    }
                    (None, false) => {}
                }
            }
        }

        // Flush trailing active runs.
        for (s, slot) in active.iter().enumerate() {
            if let Some((start_g, conf_sum, conf_count)) = *slot {
                let start_t = global_start + start_g as f32 * stride;
                let end_t = global_start + global_frames as f32 * stride;
                let dur = end_t - start_t;
                if dur >= self.config.min_segment_secs {
                    let mean_conf = (conf_sum / conf_count.max(1.0)).clamp(0.0, 1.0);
                    let had_overlap = (start_g..global_frames).any(|gg| {
                        frame_classes[gg]
                            .map(|c| {
                                c.is_overlap() && c.speakers().iter().any(|x| *x as usize == s)
                            })
                            .unwrap_or(false)
                    });
                    segments.push(RawSegment {
                        time: TimeRange {
                            start: start_t as f64,
                            end: end_t as f64,
                        },
                        local_speaker_idx: s as u8,
                        is_overlap: had_overlap,
                        confidence: PowersetDecoder::frame_confidence(mean_conf),
                    });
                }
            }
        }

        segments.sort_by(|a, b| {
            a.time
                .start
                .partial_cmp(&b.time.start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(segments)
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a window where every frame is a single class (like 0=silence,
    /// 1=speaker 0, etc.) with the listed class as logit 10 and others as logit 0.
    fn synthetic_window(
        start: f32,
        end: f32,
        num_frames: usize,
        classes: &[usize],
    ) -> WindowOutput {
        assert_eq!(classes.len(), num_frames);
        let mut logits = Vec::with_capacity(num_frames * 7);
        for &c in classes {
            for k in 0..7 {
                logits.push(if k == c { 10.0 } else { 0.0 });
            }
        }
        WindowOutput::new(start, end, logits, num_frames).unwrap()
    }

    #[test]
    fn empty_returns_empty() {
        let agg = Aggregator::new(AggregationConfig::default());
        assert!(agg.stitch(&[]).unwrap().is_empty());
    }

    #[test]
    fn single_window_silence_yields_no_segments() {
        let agg = Aggregator::new(AggregationConfig::default());
        let w = synthetic_window(0.0, 1.0, 10, &[0; 10]);
        let segs = agg.stitch(&[w]).unwrap();
        assert!(segs.is_empty());
    }

    #[test]
    fn single_window_one_speaker_yields_one_segment() {
        let agg = Aggregator::new(AggregationConfig::default());
        let w = synthetic_window(0.0, 1.0, 10, &[1; 10]);
        let segs = agg.stitch(&[w]).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].local_speaker_idx, 0);
        assert!(!segs[0].is_overlap);
    }

    #[test]
    fn single_window_overlap_yields_two_segments_same_time() {
        let agg = Aggregator::new(AggregationConfig::default());
        let w = synthetic_window(0.0, 1.0, 10, &[4; 10]);
        let segs = agg.stitch(&[w]).unwrap();
        assert_eq!(segs.len(), 2);
        assert!((segs[0].time.start - segs[1].time.start).abs() < 1e-3);
        assert!((segs[0].time.end - segs[1].time.end).abs() < 1e-3);
        assert!(segs.iter().all(|s| s.is_overlap));
        let speakers: Vec<u8> = segs.iter().map(|s| s.local_speaker_idx).collect();
        assert!(speakers.contains(&0));
        assert!(speakers.contains(&1));
    }

    #[test]
    fn two_windows_with_consistent_speakers_remain_consistent() {
        let a = synthetic_window(0.0, 5.0, 50, &[1; 50]);
        let b = synthetic_window(4.0, 9.0, 50, &[1; 50]);
        let agg = Aggregator::new(AggregationConfig::default());
        let segs = agg.stitch(&[a, b]).unwrap();
        assert!(segs.iter().all(|s| s.local_speaker_idx == 0));
        assert!(segs.iter().all(|s| !s.is_overlap));
    }

    #[test]
    fn two_windows_requiring_permutation_get_aligned() {
        let a = synthetic_window(
            0.0,
            5.0,
            50,
            &[
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            ],
        );
        let b = synthetic_window(
            4.0,
            9.0,
            50,
            &[
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            ],
        );
        let agg = Aggregator::new(AggregationConfig::default());
        let segs = agg.stitch(&[a, b]).unwrap();

        let mut idx_set = std::collections::HashSet::new();
        for s in &segs {
            idx_set.insert(s.local_speaker_idx);
        }
        assert_eq!(idx_set.len(), 2);

        let mut sorted = segs.clone();
        sorted.sort_by(|a, b| a.time.start.partial_cmp(&b.time.start).unwrap());
        let first = sorted.first().unwrap();
        let last = sorted.last().unwrap();
        assert_ne!(first.local_speaker_idx, last.local_speaker_idx);
    }

    #[test]
    fn min_segment_filter_drops_tiny_runs() {
        let w = synthetic_window(0.0, 1.0, 100, &{
            let mut v = vec![0; 100];
            v[50] = 1;
            v
        });
        let config = AggregationConfig {
            min_segment_secs: 0.1,
            ..AggregationConfig::default()
        };
        let agg = Aggregator::new(config);
        let segs = agg.stitch(&[w]).unwrap();
        assert!(segs.is_empty());
    }

    #[test]
    fn output_segments_are_sorted_by_start_time() {
        let mut classes = vec![0; 100];
        for c in &mut classes[10..20] {
            *c = 1;
        }
        for c in &mut classes[50..60] {
            *c = 1;
        }
        let w = synthetic_window(0.0, 1.0, 100, &classes);
        let agg = Aggregator::new(AggregationConfig::default());
        let segs = agg.stitch(&[w]).unwrap();
        assert!(segs.len() >= 2);
        for pair in segs.windows(2) {
            assert!(pair[0].time.start <= pair[1].time.start);
        }
    }

    #[test]
    fn confidence_is_within_unit_interval() {
        let w = synthetic_window(0.0, 1.0, 10, &[1; 10]);
        let agg = Aggregator::new(AggregationConfig::default());
        let segs = agg.stitch(&[w]).unwrap();
        for s in segs {
            assert!(s.confidence.get() >= 0.0);
            assert!(s.confidence.get() <= 1.0);
        }
    }
}
