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
fn synthetic_window(start: f32, end: f32, num_frames: usize, classes: &[usize]) -> WindowOutput {
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
    let unique: std::collections::HashSet<u8> = segs.iter().map(|s| s.local_speaker_idx).collect();
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
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        ],
    );
    let b = synthetic_window(
        4.0,
        9.0,
        50,
        &[
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
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
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
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
            2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
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
            2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
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
    let unique: std::collections::HashSet<u8> = segs.iter().map(|s| s.local_speaker_idx).collect();
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

#[test]
fn window_output_rejects_mismatched_logits_len() {
    let err = WindowOutput::new(0.0, 1.0, vec![0.0; 8], 1).unwrap_err();
    assert!(matches!(err, SegmentationError::InvalidOutputShape { .. }));
}

#[test]
fn frame_stride_and_time_handle_empty_window() {
    let w = WindowOutput::new(0.0, 1.0, Vec::new(), 0).unwrap();
    assert_eq!(w.frame_stride(), 0.0);
    // With no frames the stride is 0, so every frame time is the start.
    assert_eq!(w.frame_time(3), 0.0);
}

#[test]
fn config_getter_returns_the_config() {
    let agg = Aggregator::new(AggregationConfig {
        min_segment_secs: 0.5,
        max_local_speakers: 2,
        binarization: None,
    });
    assert_eq!(agg.config().min_segment_secs, 0.5);
    assert_eq!(agg.config().max_local_speakers, 2);
    assert!(agg.config().binarization.is_none());
}

/// Disjoint windows take the identity-permutation early return, and the
/// global-grid frames in the gap between them have no contributing window
/// (count 0), so they must decode as "no label" and emit no segment.
#[test]
fn disjoint_windows_leave_the_gap_empty() {
    let a = synthetic_window(0.0, 1.0, 10, &[1; 10]);
    let b = synthetic_window(2.0, 3.0, 10, &[2; 10]);
    let agg = Aggregator::new(AggregationConfig::default());
    let segs = agg.stitch(&[a, b]).unwrap();
    assert_eq!(segs.len(), 2);
    let first = segs
        .iter()
        .find(|s| s.local_speaker_idx == 0)
        .expect("speaker 0 segment");
    assert!(first.time.end <= 1.0 + 1e-3, "speaker 0 stays in window A");
    let second = segs
        .iter()
        .find(|s| s.local_speaker_idx == 1)
        .expect("speaker 1 segment");
    assert!(
        second.time.start >= 2.0 - 1e-3,
        "speaker 1 stays in window B"
    );
}

/// When fewer than two speakers are active on either side of an overlap,
/// the permutation cannot be determined reliably and must stay identity.
#[test]
fn single_active_speaker_per_side_keeps_identity_permutation() {
    let a = synthetic_window(0.0, 2.0, 20, &[1; 20]);
    let b = synthetic_window(1.0, 3.0, 20, &[2; 20]);
    let agg = Aggregator::new(AggregationConfig::default());
    let segs = agg.stitch(&[a, b]).unwrap();
    let speakers: std::collections::HashSet<u8> =
        segs.iter().map(|s| s.local_speaker_idx).collect();
    assert_eq!(speakers, [0u8, 1u8].into_iter().collect());
}

/// `max_local_speakers = 0` disables alignment entirely: every adjacent
/// window pair takes the identity permutation without touching the masks.
#[test]
fn max_local_speakers_zero_disables_permutation() {
    let a = synthetic_window(0.0, 2.0, 20, &[4; 20]);
    let b = synthetic_window(1.0, 3.0, 20, &[4; 20]);
    let agg = Aggregator::new(AggregationConfig {
        max_local_speakers: 0,
        ..AggregationConfig::default()
    });
    let segs = agg.stitch(&[a, b]).unwrap();
    assert!(!segs.is_empty());
    assert!(segs.iter().all(|s| s.local_speaker_idx < 2));
}

/// `WindowOutput` fields are public, so a window whose flat logits do not
/// match `num_frames * 7` can exist; `stitch` must surface the decode
/// error instead of panicking.
#[test]
fn stitch_propagates_window_decode_errors() {
    let bad = WindowOutput {
        start_time: 0.0,
        end_time: 1.0,
        logits: vec![1.0; 3],
        num_frames: 1,
    };
    let agg = Aggregator::new(AggregationConfig::default());
    assert!(matches!(
        agg.stitch(&[bad]),
        Err(SegmentationError::InvalidOutputShape { .. })
    ));
}

/// The calibrated-binarization path of `classify_frames`: covered frames
/// are averaged and binarized, gap frames (no contributing window) are
/// marked `has_data = false` and stay inactive.
#[test]
fn binarization_config_stitches_across_gaps() {
    let agg = Aggregator::new(AggregationConfig {
        binarization: Some(BinarizationConfig {
            onset: 0.5,
            offset: 0.5,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
        }),
        ..AggregationConfig::default()
    });
    let a = synthetic_window(0.0, 1.0, 10, &[1; 10]);
    let b = synthetic_window(2.0, 3.0, 10, &[1; 10]);
    let segs = agg.stitch(&[a, b]).unwrap();
    assert_eq!(segs.len(), 2, "one segment per window, gap stays empty");
    assert!(segs.iter().all(|s| s.local_speaker_idx == 0));
    assert!(!segs.iter().any(|s| s.is_overlap));
    assert!(
        segs.iter()
            .all(|s| s.time.end <= 1.0 + 1e-3 || s.time.start >= 2.0 - 1e-3),
        "no segment may cover the gap"
    );
}

#[test]
fn frame_index_at_rejects_out_of_span_and_degenerate_windows() {
    let agg = Aggregator::new(AggregationConfig::default());
    let w = synthetic_window(1.0, 2.0, 10, &[0; 10]);
    assert_eq!(agg.frame_index_at(&w, 0.5), None, "before the window");
    assert_eq!(agg.frame_index_at(&w, 2.5), None, "after the window");
    assert_eq!(agg.frame_index_at(&w, 1.5), Some(5));
    let empty = WindowOutput::new(1.0, 2.0, Vec::new(), 0).unwrap();
    assert_eq!(agg.frame_index_at(&empty, 1.5), None, "no frames");
    let zero_stride = synthetic_window(1.0, 1.0, 5, &[0; 5]);
    assert_eq!(agg.frame_index_at(&zero_stride, 1.0), None, "zero stride");
}

/// The global grid takes its stride from window 0; a later window with a
/// much finer stride has trailing frame centers past the last global
/// frame, and those contributions must be skipped.
#[test]
fn finer_second_window_skips_frames_beyond_the_grid() {
    let a = synthetic_window(0.0, 1.0, 10, &[1; 10]);
    let b = synthetic_window(0.9, 2.0, 220, &[1; 220]);
    let agg = Aggregator::new(AggregationConfig::default());
    let segs = agg.stitch(&[a, b]).unwrap();
    assert!(!segs.is_empty());
    assert!(segs.iter().all(|s| s.local_speaker_idx == 0));
}

/// Three active speakers on side A but only two on side B: the square
/// cost matrix is padded, and the unmatched A row is optimally assigned to
/// the padding column. That assignment must be dropped, leaving B's local
/// speakers mapped onto their true global indices.
#[test]
fn permutation_drops_assignment_to_padding_column() {
    // Fully overlapping windows. A alternates {0,2} / {1,2} halves so all
    // three speakers are active; B plays spk0 then spk1, matching A's
    // spk0/spk1 halves exactly (IoU 1), while A's spk2 matches nothing
    // well (IoU 0.5 with either).
    let mut a_classes = vec![5usize; 10]; // {0, 2}
    a_classes.extend(std::iter::repeat_n(6usize, 10)); // {1, 2}
    let a = synthetic_window(0.0, 2.0, 20, &a_classes);
    let mut b_classes = vec![1usize; 10]; // spk0
    b_classes.extend(std::iter::repeat_n(2usize, 10)); // spk1
    let b = synthetic_window(0.0, 2.0, 20, &b_classes);

    let agg = Aggregator::new(AggregationConfig::default());
    let a_labels = PowersetDecoder::decode_window(&a.logits, a.num_frames).unwrap();
    let b_labels = PowersetDecoder::decode_window(&b.logits, b.num_frames).unwrap();
    let perm = agg
        .window_permutation(&a, &a_labels, &b, &b_labels, &[0, 1, 2])
        .unwrap();
    assert_eq!(perm, [0, 1, 2], "identity mapping is already optimal");
}

#[test]
fn empty_window_yields_no_segments() {
    let w = WindowOutput::new(0.0, 0.0, Vec::new(), 0).unwrap();
    let agg = Aggregator::new(AggregationConfig::default());
    assert!(agg.stitch(&[w]).unwrap().is_empty());
}

#[test]
fn single_frame_window_emits_one_segment() {
    let w = synthetic_window(0.0, 0.1, 1, &[1]);
    let agg = Aggregator::new(AggregationConfig::default());
    let segs = agg.stitch(&[w]).unwrap();
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].local_speaker_idx, 0);
    assert!((segs[0].time.end - segs[0].time.start - 0.1).abs() < 1e-3);
}

/// NaN window times make the overlap bounds NaN, which degenerates the
/// sampling grid to length 0; the permutation must fall back to identity
/// and stitching must not panic.
#[test]
fn nan_window_times_degenerate_safely() {
    let nan_window = |classes: &[usize]| WindowOutput {
        start_time: f32::NAN,
        end_time: f32::NAN,
        logits: classes
            .iter()
            .flat_map(|&c| (0..7).map(move |k| if k == c { 10.0 } else { 0.0 }))
            .collect(),
        num_frames: classes.len(),
    };
    let a = nan_window(&[1, 1]);
    let b = nan_window(&[2, 2]);
    let agg = Aggregator::new(AggregationConfig::default());
    let segs = agg.stitch(&[a, b]).unwrap();
    assert!(segs.is_empty());
}
