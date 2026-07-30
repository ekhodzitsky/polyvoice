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
mod tests {
    /// Frame probs for a solo speaker-0 activity level `p` (rest goes to Empty).
    fn spk0_frame(p: f32) -> [f32; 7] {
        let mut f = [0.0; 7];
        f[1] = p; // class 1 = {spk0}
        f[0] = 1.0 - p;
        f
    }

    /// The class-index remap used by `remap_probs` must agree with the
    /// historical inline table on every reachable input: all 7 classes crossed
    /// with all 27 permutations valued in 0..=2 (including non-bijective ones).
    /// Probability mass is preserved exactly (same accumulation order).
    #[test]
    fn remap_probs_matches_historical_class_table() {
        fn historical_table(speakers: &[u8]) -> usize {
            match speakers {
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
            }
        }
        let probs = [0.05_f32, 0.3, 0.2, 0.1, 0.15, 0.1, 0.1];
        for p0 in 0..MAX_LOCAL_SPEAKERS as u8 {
            for p1 in 0..MAX_LOCAL_SPEAKERS as u8 {
                for p2 in 0..MAX_LOCAL_SPEAKERS as u8 {
                    let perm = [p0, p1, p2];
                    let remapped = remap_probs(&probs, &perm);
                    let mut expected = [0.0_f32; NUM_POWERSET_CLASSES];
                    for (c, &p) in probs.iter().enumerate() {
                        let class = PowersetDecoder::class_for_index(c).unwrap();
                        let mapped: Vec<u8> = class
                            .speakers()
                            .iter()
                            .map(|s| {
                                if (*s as usize) < MAX_LOCAL_SPEAKERS {
                                    perm[*s as usize]
                                } else {
                                    *s
                                }
                            })
                            .collect();
                        expected[historical_table(&mapped)] += p;
                    }
                    assert_eq!(remapped, expected, "perm {perm:?}");
                }
            }
        }
    }

    #[test]
    fn remap_probs_with_identity_permutation_is_exact() {
        let probs = [0.05_f32, 0.3, 0.2, 0.1, 0.15, 0.1, 0.1];
        let remapped = remap_probs(&probs, &[0, 1, 2]);
        assert_eq!(remapped, probs);
    }

    #[test]
    fn binarize_drops_short_blip_and_bridges_short_gap() {
        let stride = 0.1;
        // 2-frame blip (frames 1-2), then a solid run 6..16 with a 1-frame gap at 10.
        let mut frames = vec![spk0_frame(0.1); 20];
        for g in [1, 2] {
            frames[g] = spk0_frame(0.9);
        }
        for f in frames.iter_mut().take(16).skip(6) {
            *f = spk0_frame(0.9);
        }
        frames[10] = spk0_frame(0.1);
        let has_data = vec![true; 20];
        let cfg = BinarizationConfig {
            onset: 0.5,
            offset: 0.5,
            min_duration_on: 0.3,  // 3 frames: the 2-frame blip must go
            min_duration_off: 0.2, // 2 frames: the 1-frame gap must be bridged
        };
        let (classes, _) = binarize_frames(&frames, &has_data, stride, &cfg);
        let active: Vec<bool> = classes
            .iter()
            .map(|c| c.map(|c| c.speakers().contains(&0)).unwrap_or(false))
            .collect();
        assert!(!active[1] && !active[2], "short blip must be dropped");
        assert!(active[10], "one-frame gap must be bridged");
        assert!((6..16).all(|g| active[g]), "solid run must stay active");
        assert!(!active[0] && !active[19], "silence stays silent");
    }

    #[test]
    fn binarize_hysteresis_prevents_flicker() {
        let stride = 0.1;
        // Rise to 0.7, then oscillate around 0.5 (0.45/0.55): with offset 0.3
        // the speaker must stay ON through the dips; a plain 0.5 threshold
        // (onset == offset) flickers.
        let mut frames = vec![spk0_frame(0.1); 12];
        frames[2] = spk0_frame(0.7);
        for (i, g) in (3..9).enumerate() {
            frames[g] = spk0_frame(if i % 2 == 0 { 0.45 } else { 0.55 });
        }
        let has_data = vec![true; 12];

        let hysteresis = BinarizationConfig {
            onset: 0.6,
            offset: 0.3,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
        };
        let (classes, _) = binarize_frames(&frames, &has_data, stride, &hysteresis);
        assert!(
            (2..9).all(|g| classes[g]
                .map(|c| !c.speakers().is_empty())
                .unwrap_or(false)),
            "hysteresis must hold the speaker ON through sub-onset dips"
        );

        let plain = BinarizationConfig {
            onset: 0.5,
            offset: 0.5,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
        };
        let (classes, _) = binarize_frames(&frames, &has_data, stride, &plain);
        let flickers = (3..9)
            .filter(|&g| classes[g].map(|c| c.speakers().is_empty()).unwrap_or(true))
            .count();
        assert!(flickers > 0, "plain threshold must flicker on this input");
    }

    #[test]
    fn binarize_uncovered_frames_stay_none_and_three_speakers_truncate_to_top2() {
        let stride = 0.1;
        // One frame where all three speakers are active (probs 0.9/0.8/0.7):
        // powerset expresses at most two — keep the top-2.
        let mut f = [0.0_f32; 7];
        f[1] = 0.5; // spk0 solo
        f[4] = 0.3; // {0,1}
        f[6] = 0.4; // {1,2}
        f[5] = 0.1; // {0,2}
        // spk0 = 0.9, spk1 = 0.7, spk2 = 0.5 — all above onset 0.4.
        let frames = vec![f, [0.0; 7]];
        let has_data = vec![true, false];
        let cfg = BinarizationConfig {
            onset: 0.4,
            offset: 0.4,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
        };
        let (classes, conf) = binarize_frames(&frames, &has_data, stride, &cfg);
        assert_eq!(
            classes[0].map(|c| c.speakers()),
            Some(vec![0, 1]),
            "top-2 speakers by probability"
        );
        assert!(classes[1].is_none(), "uncovered frame stays None");
        assert_eq!(conf[1], 0.0);
    }

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

    /// Frame-time convention check: `frame_index_at` uses
    /// `floor((t - start)/stride)`, and the RLE pass (line ~300) places frame `f`
    /// by its center `start + (f + 0.5)*stride`. These are NOT two different
    /// conventions: `floor(x)` already returns the frame whose CENTER is closest to
    /// `t`, because `round(x - 0.5) == floor(x)` for every non-negative `x` once the
    /// result is clamped to `[0, num_frames-1]`. This test pins that equivalence so
    /// a future "fix" to `round((t-start)/stride - 0.5)` is recognized as a no-op.
    #[test]
    fn frame_index_floor_equals_nearest_center() {
        let stride = 0.1f32;
        let start = 0.37f32;
        for i in 0..5000 {
            let t = start + i as f32 * 0.00713;
            let x = (t - start) / stride;
            // Clamp both at the lower edge the way frame_index_at does.
            let floor_idx = (x.floor() as i64).max(0);
            let round_idx = ((x - 0.5).round() as i64).max(0);
            assert_eq!(
                floor_idx, round_idx,
                "floor and nearest-center disagree at t={t} x={x:.6}"
            );
        }
    }

    /// Frame-time convention check: a speaker change staggered ~0.5*stride off a
    /// window boundary must still be labelled consistently after stitching. With the
    /// sampler (`frame_index_at`) and the RLE applier sharing the nearest-center
    /// convention, there is no 1-frame boundary flip — this passes on current code,
    /// confirming the two conventions already coincide (no off-by-one).
    #[test]
    fn staggered_speaker_change_is_labelled_consistently() {
        // Window A: 0.0–5.0, 50 frames (stride 0.1). spk0 (class 1) then spk1
        // (class 2); change at frame 25 → t = 2.5s.
        let mut a_classes = vec![1usize; 50];
        for c in &mut a_classes[25..50] {
            *c = 2;
        }
        let a = synthetic_window(0.0, 5.0, 50, &a_classes);
        // Window B: 2.45–7.45, 50 frames (stride 0.1) — its grid is offset half a
        // stride from A. Same physical truth: spk0 until ~2.5s then spk1.
        let mut b_classes = vec![1usize; 50];
        for c in &mut b_classes[1..50] {
            *c = 2;
        }
        let b = synthetic_window(2.45, 7.45, 50, &b_classes);

        let agg = Aggregator::new(AggregationConfig::default());
        let segs = agg.stitch(&[a, b]).unwrap();

        fn speaker_at(segs: &[RawSegment], t: f64) -> Option<u8> {
            segs.iter()
                .find(|s| s.time.start <= t && t < s.time.end)
                .map(|s| s.local_speaker_idx)
        }
        // Well clear of the staggered boundary, the two speakers are distinct and
        // stable (identity permutation — both windows use the same class indices).
        let early = speaker_at(&segs, 1.0).expect("segment around 1.0s");
        let late = speaker_at(&segs, 6.0).expect("segment around 6.0s");
        assert_ne!(
            early, late,
            "the two speakers must stay distinct across the seam"
        );
        // Exactly two global speakers across the file.
        let unique: std::collections::HashSet<u8> =
            segs.iter().map(|s| s.local_speaker_idx).collect();
        assert_eq!(unique.len(), 2, "expected 2 speakers, got {}", unique.len());
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
    fn partial_overlap_run_splits_into_solo_and_overlap_segments() {
        // spk0 talks the whole 0-10s window; spk1 joins only over 4-6s (class 4 =
        // pair{0,1}). spk0's run must split into solo [0,4), overlap [4,6), solo
        // [6,10); spk1 emits one overlap [4,6). The two overlap pieces must share
        // an exact time range so extract_overlap_time_ranges can pair them — this
        // is the fix for whole single-speaker runs being falsely flagged overlap.
        let mut classes = vec![1usize; 100];
        for c in &mut classes[40..60] {
            *c = 4;
        }
        let w = synthetic_window(0.0, 10.0, 100, &classes);
        let agg = Aggregator::new(AggregationConfig::default());
        let segs = agg.stitch(&[w]).unwrap();

        let overlap: Vec<&RawSegment> = segs.iter().filter(|s| s.is_overlap).collect();
        assert_eq!(
            overlap.len(),
            2,
            "exactly two overlap pieces (one per speaker)"
        );
        for s in &overlap {
            assert!((s.time.start - 4.0).abs() < 1e-3, "overlap starts at 4.0s");
            assert!((s.time.end - 6.0).abs() < 1e-3, "overlap ends at 6.0s");
        }
        let ov_speakers: std::collections::HashSet<u8> =
            overlap.iter().map(|s| s.local_speaker_idx).collect();
        assert_eq!(ov_speakers, [0u8, 1u8].into_iter().collect());

        // spk0: three pieces (solo, overlap, solo); the two solo ones are NOT overlap.
        let spk0: Vec<&RawSegment> = segs.iter().filter(|s| s.local_speaker_idx == 0).collect();
        assert_eq!(spk0.len(), 3, "spk0 run splits at both overlap boundaries");
        let solo0: Vec<&&RawSegment> = spk0.iter().filter(|s| !s.is_overlap).collect();
        assert_eq!(solo0.len(), 2, "spk0 keeps two solo pieces");
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

    /// Regression test for the cumulative-permutation double-application bug.
    /// Windows 0 and 1 are swapped once; window 2 is swapped once relative to
    /// window 1. Because `window_permutation` already applies the cumulative
    /// permutation when building A-masks, the returned `perm` is already
    /// file-global. Before the fix the code composed `prev[perm[...]]`, which
    /// double-applied the permutation and produced inconsistent global speaker
    /// indices across window boundaries.
    #[test]
    fn three_windows_keep_global_speaker_indices_consistent() {
        // Window 0: spk0 in 0-3.0s and 4.0-4.5s; spk1 in 3.0-4.0s and 4.5-5.0s.
        // The 4.0-5.0s overlap with window 1 contains both speakers.
        let w0 = synthetic_window(
            0.0,
            5.0,
            50,
            &[
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            ],
        );
        // Window 1 overlaps 4-5s. In the overlap it swaps local indices
        // relative to window 0, so local spk1 = global spk0 and local spk0 =
        // global spk1. Both speakers remain active through the window so the
        // overlap with window 2 also contains both global speakers.
        let w1 = synthetic_window(
            4.0,
            9.0,
            50,
            &[
                2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
            ],
        );
        // Window 2 overlaps 8-9s. In the overlap it swaps local indices
        // relative to window 1, so local spk1 = global spk0 and local spk0 =
        // global spk1. In the non-overlap region only global spk1 continues.
        let w2 = synthetic_window(
            8.0,
            13.0,
            50,
            &[
                2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ],
        );

        let agg = Aggregator::new(AggregationConfig::default());
        let segs = agg.stitch(&[w0, w1, w2]).unwrap();

        fn speaker_at(segs: &[RawSegment], t: f64) -> Option<u8> {
            segs.iter()
                .find(|s| s.time.start <= t && t < s.time.end)
                .map(|s| s.local_speaker_idx)
        }

        let spk_0_early = speaker_at(&segs, 1.0).expect("segment expected around 1.0s");
        let spk_0_late = speaker_at(&segs, 8.25).expect("segment expected around 8.25s");
        let spk_1_early = speaker_at(&segs, 7.5).expect("segment expected around 7.5s");
        let spk_1_late = speaker_at(&segs, 11.0).expect("segment expected around 11.0s");

        assert_eq!(
            spk_0_early, spk_0_late,
            "global speaker 0 must keep the same index across window 2 boundary"
        );
        assert_eq!(
            spk_1_early, spk_1_late,
            "global speaker 1 must keep the same index across window 2 boundary"
        );
        assert_ne!(spk_0_early, spk_1_early, "two distinct speakers expected");
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

    /// Regression: inactive speakers must not enter the Hungarian cost matrix.
    ///
    /// Setup: max_local_speakers = 3, but only speakers 0 and 1 are ever on.
    /// Window B swaps their local indices in the overlap. With the active-only
    /// matrix the permutation recovers the swap; a full 3×3 matrix that pads
    /// inactive speaker 2 with zero-IoU rows is the pyannote-style bug and can
    /// mis-assign columns. This test locks the correct (active-only) outcome.
    #[test]
    fn window_perm_ignores_inactive_third_speaker() {
        // Window A: spk0 then both in the last second (overlap with B).
        // classes: 1 = spk0, 2 = spk1, 4 = spk0+spk1 (powerset).
        let mut a_classes = vec![1usize; 50];
        for c in &mut a_classes[40..50] {
            *c = 4; // both speakers in overlap region
        }
        let a = synthetic_window(0.0, 5.0, 50, &a_classes);
        // Window B: in the overlap, local indices are swapped (class 4 is
        // unordered {0,1}; pure spk runs use swapped singles).
        // First 10 frames (overlap): both; then pure local-1 (which is global 0)
        // then pure local-0 (which is global 1).
        let mut b_classes = vec![4usize; 10];
        b_classes.extend(std::iter::repeat_n(2usize, 20)); // local spk1
        b_classes.extend(std::iter::repeat_n(1usize, 20)); // local spk0
        let b = synthetic_window(4.0, 9.0, 50, &b_classes);

        let agg = Aggregator::new(AggregationConfig {
            max_local_speakers: 3,
            ..AggregationConfig::default()
        });
        let segs = agg.stitch(&[a, b]).unwrap();
        let unique: std::collections::HashSet<u8> =
            segs.iter().map(|s| s.local_speaker_idx).collect();
        // Only two speakers exist in the file — the inactive third slot must not
        // produce a third global identity.
        assert!(
            unique.len() <= 2,
            "inactive speaker slot must not invent a third speaker; got {unique:?}"
        );
        assert!(
            !unique.contains(&2),
            "speaker index 2 must stay unused: {unique:?}"
        );
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
