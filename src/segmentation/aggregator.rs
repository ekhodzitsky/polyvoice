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
//! 3. For every audio frame, average the per-class softmax probabilities
//!    (decoded once by `PowersetDecoder`) across every window that contains
//!    that frame, remapping each window's classes through its permutation.
//! 4. Argmax each averaged probability vector → frame label.
//! 5. Run-length encode consecutive identical labels into `RawSegment`s.

use crate::hungarian;
use crate::segmentation::binarize::{BinarizationConfig, binarize_frames};
use crate::segmentation::decoder::{
    FrameLabel, MAX_LOCAL_SPEAKERS, NUM_POWERSET_CLASSES, PowersetClass, PowersetDecoder,
};
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
        if logits.len() != num_frames * NUM_POWERSET_CLASSES {
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
    /// Optional calibrated binarization of the averaged posteriors (hysteresis
    /// + min-duration smoothing) instead of the plain per-frame argmax.
    ///
    /// `None` keeps the historical argmax behavior.
    pub binarization: Option<BinarizationConfig>,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            min_segment_secs: 0.0,
            max_local_speakers: 3,
            binarization: None,
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
        let mut permutations: Vec<[u8; MAX_LOCAL_SPEAKERS]> =
            std::iter::repeat_n([0u8, 1u8, 2u8], windows.len()).collect();

        for i in 1..windows.len() {
            let perm = self.window_permutation(
                &windows[i - 1],
                &window_labels[i - 1],
                &windows[i],
                &window_labels[i],
                &permutations[i - 1],
            )?;
            // `window_permutation` already built A's masks using
            // `a_perm_so_far`, so `perm` maps window i's local indices directly
            // onto the file-global speaker frame. No extra composition needed.
            permutations[i] = perm;
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
        a_perm_so_far: &[u8; MAX_LOCAL_SPEAKERS],
    ) -> Result<[u8; MAX_LOCAL_SPEAKERS], SegmentationError> {
        let n = self.config.max_local_speakers.min(MAX_LOCAL_SPEAKERS);
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

        let mut a_masks = vec![vec![false; grid_len]; MAX_LOCAL_SPEAKERS];
        let mut b_masks = vec![vec![false; grid_len]; MAX_LOCAL_SPEAKERS];
        for k in 0..grid_len {
            let t = overlap_start + k as f32 * stride;
            if let Some(idx_a) = self.frame_index_at(a, t)
                && idx_a < a_labels.len()
            {
                for s in a_labels[idx_a].class.speakers() {
                    if (s as usize) < MAX_LOCAL_SPEAKERS {
                        let permuted = a_perm_so_far[s as usize] as usize;
                        if permuted < MAX_LOCAL_SPEAKERS {
                            a_masks[permuted][k] = true;
                        }
                    }
                }
            }
            if let Some(idx_b) = self.frame_index_at(b, t)
                && idx_b < b_labels.len()
            {
                for s in b_labels[idx_b].class.speakers() {
                    if (s as usize) < MAX_LOCAL_SPEAKERS {
                        b_masks[s as usize][k] = true;
                    }
                }
            }
        }

        // Active speakers only — never put inactive (never-on) speakers into the
        // cost matrix. Including them is the pyannote "inactive speakers in the
        // similarity matrix" bug: zero-IoU rows compete for columns and corrupt
        // the permutation among the speakers that actually speak in the overlap.
        let a_active: Vec<usize> = (0..n).filter(|&i| a_masks[i].iter().any(|&x| x)).collect();
        let b_active: Vec<usize> = (0..n).filter(|&i| b_masks[i].iter().any(|&x| x)).collect();

        // If fewer than 2 speakers are active on either side in the overlap, we
        // cannot reliably determine the full permutation — return identity.
        if a_active.len() < 2 || b_active.len() < 2 {
            return Ok([0, 1, 2]);
        }

        // Build cost over the active×active submatrix, padded to square.
        let m = a_active.len().max(b_active.len());
        let mut cost: Vec<Vec<f32>> = vec![vec![0.0_f32; m]; m];
        for (ai, &a_idx) in a_active.iter().enumerate() {
            for (bi, &b_idx) in b_active.iter().enumerate() {
                let mut inter = 0_usize;
                let mut uni = 0_usize;
                for k in 0..grid_len {
                    let ax = a_masks[a_idx][k];
                    let bx = b_masks[b_idx][k];
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

        let assignment =
            hungarian::solve(&cost).ok_or_else(|| SegmentationError::PermutationFailed {
                prev_idx: 0,
                next_idx: 0,
                detail: "non-square cost matrix".to_owned(),
            })?;

        // assignment[ai] = bi means: active-A row ai best matches active-B col bi.
        // Map back onto the original local indices: perm[b_local] = a_global.
        // Inactive locals keep identity.
        let mut perm = [0_u8, 1_u8, 2_u8];
        for (ai, &a_idx) in a_active.iter().enumerate() {
            let bi = assignment[ai];
            if bi < b_active.len() {
                let b_idx = b_active[bi];
                if b_idx < MAX_LOCAL_SPEAKERS && a_idx < MAX_LOCAL_SPEAKERS {
                    perm[b_idx] = a_idx as u8;
                }
            }
        }
        Ok(perm)
    }

    /// Find the frame index in `w` whose center is closest to time `t`. Returns
    /// `None` if `t` is outside the window's span.
    ///
    /// Uses `floor((t - start)/stride)`. This already IS the nearest-center frame:
    /// `round((t - start)/stride - 0.5)` equals this `floor` for every in-span `t`
    /// once clamped to `[0, num_frames-1]`, so it matches the center convention used
    /// by `average_and_run_length_encode` (proven by
    /// `frame_index_floor_equals_nearest_center`). Do NOT "fix" this to `round`; it
    /// is a no-op that only changes the out-of-span `t == start - ε` corner.
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

    /// Average per-class probabilities across windows that contain each global
    /// frame, then argmax + run-length encode into `RawSegment`s.
    fn average_and_run_length_encode(
        &self,
        windows: &[WindowOutput],
        window_labels: &[Vec<FrameLabel>],
        permutations: &[[u8; MAX_LOCAL_SPEAKERS]],
    ) -> Result<Vec<RawSegment>, SegmentationError> {
        let grid = GlobalGrid::from_windows(windows);
        let (summed_probs, counts) = accumulate_grid(windows, window_labels, permutations, &grid);
        let (frame_classes, frame_confidences) =
            self.classify_frames(&summed_probs, &counts, grid.stride);

        let mut encoder = RleEncoder::new(grid.start, grid.stride, self.config.min_segment_secs);
        for g in 0..grid.frames {
            encoder.push_frame(g, frame_classes[g], frame_confidences[g]);
        }
        Ok(encoder.finish(grid.frames))
    }

    /// Reduce the accumulated probability sums to per-frame classes +
    /// confidences, either via the calibrated binarization or the plain
    /// per-frame argmax.
    fn classify_frames(
        &self,
        summed_probs: &[[f32; NUM_POWERSET_CLASSES]],
        counts: &[u32],
        stride: f32,
    ) -> (Vec<Option<PowersetClass>>, Vec<f32>) {
        let global_frames = summed_probs.len();
        if let Some(bin) = &self.config.binarization {
            let mut avg_probs = vec![[0.0_f32; NUM_POWERSET_CLASSES]; global_frames];
            let mut has_data = vec![false; global_frames];
            for g in 0..global_frames {
                if counts[g] == 0 {
                    continue;
                }
                let inv = 1.0 / counts[g] as f32;
                for c in 0..NUM_POWERSET_CLASSES {
                    avg_probs[g][c] = summed_probs[g][c] * inv;
                }
                has_data[g] = true;
            }
            binarize_frames(&avg_probs, &has_data, stride, bin)
        } else {
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
            (frame_classes, frame_confidences)
        }
    }
}

/// Global frame-grid geometry shared by probability accumulation and the
/// run-length encoder.
struct GlobalGrid {
    start: f32,
    stride: f32,
    frames: usize,
}

impl GlobalGrid {
    fn from_windows(windows: &[WindowOutput]) -> Self {
        let stride = windows[0].frame_stride().max(1e-6);
        let start = windows
            .iter()
            .map(|w| w.start_time)
            .fold(f32::INFINITY, f32::min);
        let end = windows
            .iter()
            .map(|w| w.end_time)
            .fold(f32::NEG_INFINITY, f32::max);
        let frames = ((end - start) / stride).ceil() as usize;
        Self {
            start,
            stride,
            frames,
        }
    }
}

/// Accumulate permutation-remapped frame probabilities onto the global grid:
/// every window frame contributes its softmax vector (decoded once by
/// `PowersetDecoder`) to the global frame covering its center. Returns the
/// per-global-frame probability sums and contribution counts.
fn accumulate_grid(
    windows: &[WindowOutput],
    window_labels: &[Vec<FrameLabel>],
    permutations: &[[u8; MAX_LOCAL_SPEAKERS]],
    grid: &GlobalGrid,
) -> (Vec<[f32; NUM_POWERSET_CLASSES]>, Vec<u32>) {
    let mut summed_probs = vec![[0.0_f32; NUM_POWERSET_CLASSES]; grid.frames];
    let mut counts = vec![0_u32; grid.frames];

    for (wi, w) in windows.iter().enumerate() {
        let perm = permutations[wi];
        for f in 0..w.num_frames {
            let t_center = w.frame_time(f) + 0.5 * grid.stride;
            let g_idx_f = (t_center - grid.start) / grid.stride;
            if g_idx_f < 0.0 {
                continue;
            }
            let g_idx = g_idx_f.floor() as usize;
            if g_idx >= grid.frames {
                continue;
            }
            let Some(label) = window_labels[wi].get(f) else {
                continue;
            };

            let remapped = remap_probs(&label.probs, &perm);
            for (i, &p) in remapped.iter().enumerate() {
                summed_probs[g_idx][i] += p;
            }
            counts[g_idx] += 1;
        }
    }
    (summed_probs, counts)
}

/// Remap one frame's softmax vector into the file-global speaker ordering:
/// each class's speaker set is mapped through `perm` and its probability mass
/// added to the remapped class. Classes whose remapped set the powerset scheme
/// cannot express fall back to class 0 (silence), matching the historical
/// behavior; such sets are unreachable from decoded classes and a valid
/// permutation.
fn remap_probs(
    probs: &[f32; NUM_POWERSET_CLASSES],
    perm: &[u8; MAX_LOCAL_SPEAKERS],
) -> [f32; NUM_POWERSET_CLASSES] {
    let mut remapped = [0.0_f32; NUM_POWERSET_CLASSES];
    for (c, &p) in probs.iter().enumerate() {
        if let Some(class) = PowersetDecoder::class_for_index(c) {
            let speakers = class.speakers();
            let remapped_speakers: Vec<u8> = speakers
                .iter()
                .map(|s| {
                    if (*s as usize) < MAX_LOCAL_SPEAKERS {
                        perm[*s as usize]
                    } else {
                        *s
                    }
                })
                .collect();
            let new_class =
                PowersetClass::from_speakers(&remapped_speakers).map_or(0, PowersetClass::index);
            remapped[new_class] += p;
        }
    }
    remapped
}

/// One speaker's open run: start frame, accumulated confidence, and whether
/// the run is an overlap run.
#[derive(Clone, Copy)]
struct RunState {
    start_g: usize,
    conf_sum: f32,
    conf_count: f32,
    is_overlap: bool,
}

/// Run-length encoder over per-frame classes, holding one open run per local
/// speaker. A run is broken not only when the speaker falls silent but also
/// when its overlap status flips: a speaker talking across a brief overlap
/// emits separate solo and overlap segments rather than one run tainted by the
/// overlap. This keeps `is_overlap` precise (a segment is overlap iff *every*
/// frame in it was an overlap frame) and makes two simultaneously-active
/// speakers emit time-equal overlap segments that
/// `extract_overlap_time_ranges` pairs.
struct RleEncoder {
    global_start: f32,
    stride: f32,
    min_segment_secs: f32,
    active: [Option<RunState>; MAX_LOCAL_SPEAKERS],
    segments: Vec<RawSegment>,
}

impl RleEncoder {
    fn new(global_start: f32, stride: f32, min_segment_secs: f32) -> Self {
        Self {
            global_start,
            stride,
            min_segment_secs,
            active: [None; MAX_LOCAL_SPEAKERS],
            segments: Vec::new(),
        }
    }

    /// Feed one global frame: open, extend, split, or close each speaker's run.
    fn push_frame(&mut self, g: usize, frame_class: Option<PowersetClass>, conf: f32) {
        let is_overlap_frame = frame_class.map(|c| c.is_overlap()).unwrap_or(false);
        let active_speakers: Vec<u8> = match frame_class {
            Some(c) => c.speakers(),
            None => Vec::new(),
        };

        for s in 0..MAX_LOCAL_SPEAKERS {
            let s_active_now = active_speakers.iter().any(|x| *x as usize == s);
            let ov_now = s_active_now && is_overlap_frame;
            match (self.active[s], s_active_now) {
                (None, true) => {
                    self.active[s] = Some(RunState {
                        start_g: g,
                        conf_sum: conf,
                        conf_count: 1.0,
                        is_overlap: ov_now,
                    });
                }
                (Some(run), true) if run.is_overlap == ov_now => {
                    self.active[s] = Some(RunState {
                        conf_sum: run.conf_sum + conf,
                        conf_count: run.conf_count + 1.0,
                        ..run
                    });
                }
                (Some(run), true) => {
                    // Overlap status flipped — close the current run, open a new one at g.
                    self.emit_segment(s, run, g);
                    self.active[s] = Some(RunState {
                        start_g: g,
                        conf_sum: conf,
                        conf_count: 1.0,
                        is_overlap: ov_now,
                    });
                }
                (Some(run), false) => {
                    self.emit_segment(s, run, g);
                    self.active[s] = None;
                }
                (None, false) => {}
            }
        }
    }

    /// Flush the trailing open runs and return the segments sorted by start time.
    fn finish(mut self, global_frames: usize) -> Vec<RawSegment> {
        for s in 0..MAX_LOCAL_SPEAKERS {
            if let Some(run) = self.active[s] {
                self.emit_segment(s, run, global_frames);
            }
        }
        self.segments.sort_by(|a, b| {
            a.time
                .start
                .partial_cmp(&b.time.start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.segments
    }

    /// Emit one run as a `RawSegment`, skipping runs shorter than
    /// `min_segment_secs`. `end_g` is the exclusive end global frame index; the
    /// run carries a single `is_overlap` flag because the RLE splits runs at
    /// every overlap-status change.
    fn emit_segment(&mut self, speaker: usize, run: RunState, end_g: usize) {
        let start_t = self.global_start + run.start_g as f32 * self.stride;
        let end_t = self.global_start + end_g as f32 * self.stride;
        if end_t - start_t < self.min_segment_secs {
            return;
        }
        let mean_conf = (run.conf_sum / run.conf_count.max(1.0)).clamp(0.0, 1.0);
        self.segments.push(RawSegment {
            time: TimeRange {
                start: start_t as f64,
                end: end_t as f64,
            },
            local_speaker_idx: speaker as u8,
            is_overlap: run.is_overlap,
            confidence: PowersetDecoder::frame_confidence(mean_conf),
        });
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "aggregator_tests.rs"]
mod tests;
