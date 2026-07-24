//! Diarization Error Rate (DER) and word-level attribution metrics.
//!
//! Frame-based DER with forgiveness collar and optimal speaker mapping, plus
//! WDER (Word Diarization Error Rate) for who-said-what evaluation.

use crate::types::{SpeakerTurn, TimeRange, WordAlignment};
use std::collections::HashMap;

/// DER evaluation result.
#[derive(Debug, Clone, Copy)]
pub struct DerResult {
    pub der: f64,
    pub miss_rate: f64,
    pub false_alarm_rate: f64,
    pub confusion_rate: f64,
    pub total_speech: f64,
    /// Raw frame counts (10 ms frames, collar-excluded) behind the ratios above.
    /// Expose them so callers can compute a correct duration-weighted
    /// micro-average across files (sum of error frames / sum of reference
    /// frames) — an average of per-file ratios cannot.
    pub total_ref_frames: u64,
    pub missed_frames: u64,
    pub false_alarm_frames: u64,
    pub confusion_frames: u64,
}

impl std::fmt::Display for DerResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DER={:.1}% (miss={:.1}%, fa={:.1}%, conf={:.1}%, speech={:.1}s)",
            self.der * 100.0,
            self.miss_rate * 100.0,
            self.false_alarm_rate * 100.0,
            self.confusion_rate * 100.0,
            self.total_speech,
        )
    }
}

/// { collar >= 0.0 }
/// pub fn compute_der( reference: &[SpeakerTurn], hypothesis: &[SpeakerTurn], collar: f64, ) -> DerResult
/// { ret.der >= 0.0 && ret.der <= 1.0 }
/// Compute DER between reference and hypothesis annotations.
///
/// `collar` is the forgiveness window (in seconds) around each reference
/// boundary. Standard value is 0.25s. Frames within the collar are ignored.
///
/// Speaker IDs between ref and hyp are mapped optimally via max-weight bipartite
/// (Hungarian / Kuhn-Munkres) matching on co-occurrence counts.
///
/// **Approximate DER.** Frame-based at 10 ms resolution with a forgiveness
/// boundary collar: frames within `collar` of any reference boundary are
/// excluded from BOTH the numerator and the denominator. For UEM scoring (restrict
/// to a scored timeline) use [`compute_der_with_uem`]. It is **not bit-identical
/// to `pyannote.metrics`** — always quote it alongside the collar value used. Raw
/// frame counts are exposed on [`DerResult`] for duration-weighted micro-averaging.
///
/// # Defensive behaviour
///
/// Returns an all-zero result if `collar` is non-finite or negative, or if any
/// turn end time is non-finite or negative. This prevents panics/DoS on
/// malformed input rather than propagating NaN/Infinity.
pub fn compute_der(
    reference: &[SpeakerTurn],
    hypothesis: &[SpeakerTurn],
    collar: f64,
) -> DerResult {
    der_core(reference, hypothesis, collar, Region::All, None)
}

/// { collar >= 0.0 }
/// pub fn compute_der_single_speaker_regions( reference: &[SpeakerTurn], hypothesis: &[SpeakerTurn], collar: f64, ) -> DerResult
/// { ret.der >= 0.0 && ret.der <= 1.0 }
/// Overlap-excluded DER: DER computed only over reference frames where exactly
/// ONE speaker is active.
///
/// Reference frames whose label set has `>= 2` speakers (overlapping speech) are
/// excluded from BOTH the speaker mapping and the error counts, on top of the
/// usual forgiveness collar. This removes the overlap-miss term that pins total
/// DER near ~88% on high-overlap audio (e.g. AMI EN2002a, ~79% overlap), giving
/// a numeric quality floor that discriminates healthy vs collapsed diarization on
/// long-form recordings — where total [`compute_der`] cannot (the miss term holds
/// DER near 88% whether diarization is healthy or collapsed).
///
/// **Never conflate this with the headline DER.** [`compute_der`] is
/// overlap-inclusive; this metric is a single-speaker-region subset. Always
/// report it under a distinct name.
pub fn compute_der_single_speaker_regions(
    reference: &[SpeakerTurn],
    hypothesis: &[SpeakerTurn],
    collar: f64,
) -> DerResult {
    der_core(reference, hypothesis, collar, Region::SingleSpeaker, None)
}

/// { collar >= 0.0 }
/// pub fn compute_der_with_uem( reference: &[SpeakerTurn], hypothesis: &[SpeakerTurn], collar: f64, scored: &[TimeRange], ) -> DerResult
/// { ret.der >= 0.0 && ret.der <= 1.0 }
/// DER restricted to the UEM (Un-partitioned Evaluation Map) scored regions.
///
/// Frames whose center falls outside every `scored` region are excluded from BOTH
/// the speaker mapping and the error counts (on top of the forgiveness collar),
/// matching `pyannote.metrics` UEM semantics. An empty `scored` slice scores
/// nothing (all-zero result); use [`compute_der`] when there is no UEM. Parse a
/// `.uem` file into the per-file `scored` regions with [`parse_uem`].
pub fn compute_der_with_uem(
    reference: &[SpeakerTurn],
    hypothesis: &[SpeakerTurn],
    collar: f64,
    scored: &[TimeRange],
) -> DerResult {
    der_core(reference, hypothesis, collar, Region::All, Some(scored))
}

/// Reference-region selector for [`der_core`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Region {
    /// Score every non-collar frame (standard overlap-inclusive DER).
    All,
    /// Score only frames where the reference has exactly one active speaker.
    SingleSpeaker,
    /// Score only frames where the reference has >= 2 concurrent speakers.
    Overlap,
}

/// Shared DER core. `region` selects which non-collar reference frames are scored:
/// all of them, single-speaker regions only, or overlap regions only. The excluded
/// frames are dropped from BOTH the speaker mapping and the error counts so each
/// metric is self-consistent on its scored subset.
fn der_core(
    reference: &[SpeakerTurn],
    hypothesis: &[SpeakerTurn],
    collar: f64,
    region: Region,
    uem: Option<&[TimeRange]>,
) -> DerResult {
    if reference.is_empty() {
        return DerResult {
            der: 0.0,
            miss_rate: 0.0,
            false_alarm_rate: 0.0,
            confusion_rate: 0.0,
            total_speech: 0.0,
            total_ref_frames: 0,
            missed_frames: 0,
            false_alarm_frames: 0,
            confusion_frames: 0,
        };
    }

    if !collar.is_finite() || collar < 0.0 {
        return DerResult {
            der: 0.0,
            miss_rate: 0.0,
            false_alarm_rate: 0.0,
            confusion_rate: 0.0,
            total_speech: 0.0,
            total_ref_frames: 0,
            missed_frames: 0,
            false_alarm_frames: 0,
            confusion_frames: 0,
        };
    }

    let resolution = 0.01; // 10ms frames
    const MAX_FRAMES: usize = 24 * 3600 * 100; // 24 hours at 10ms resolution

    let max_time = reference
        .iter()
        .chain(hypothesis.iter())
        .map(|t| t.time.end)
        .fold(0.0f64, f64::max);

    if !max_time.is_finite() || max_time < 0.0 {
        return DerResult {
            der: 0.0,
            miss_rate: 0.0,
            false_alarm_rate: 0.0,
            confusion_rate: 0.0,
            total_speech: 0.0,
            total_ref_frames: 0,
            missed_frames: 0,
            false_alarm_frames: 0,
            confusion_frames: 0,
        };
    }

    let n_frames = ((max_time / resolution).ceil() as usize + 1).min(MAX_FRAMES);

    // Frames to ignore: always those inside the forgiveness collar.
    let mut ignore_mask = build_collar_mask(reference, collar, resolution, n_frames);

    // Build frame-level speaker labels.
    let ref_frames = build_speaker_frames(reference, resolution, n_frames);
    let hyp_frames = build_speaker_frames(hypothesis, resolution, n_frames);

    // Restrict the scored subset by region. SingleSpeaker drops overlap frames
    // (removing the overlap-miss term); Overlap drops single/zero-speaker frames
    // (isolating the overlap-region error). Either way the dropped frames leave
    // both the mapping and the counts so each metric is self-consistent.
    match region {
        Region::All => {}
        Region::SingleSpeaker => {
            for (i, frame) in ref_frames.iter().enumerate() {
                if frame.len() >= 2 {
                    ignore_mask[i] = true;
                }
            }
        }
        Region::Overlap => {
            for (i, frame) in ref_frames.iter().enumerate() {
                if frame.len() < 2 {
                    ignore_mask[i] = true;
                }
            }
        }
    }

    // UEM (Un-partitioned Evaluation Map): when scored regions are supplied, any
    // frame whose center falls outside every scored region is ignored — dropped
    // from BOTH the mapping and the counts, exactly like a collar frame. `None`
    // (or an empty mask) leaves the whole file scored, so the no-UEM path is
    // byte-identical to before.
    if let Some(scored) = uem {
        for (i, slot) in ignore_mask.iter_mut().enumerate() {
            if *slot {
                continue;
            }
            let center = (i as f64 + 0.5) * resolution;
            let in_scope = scored.iter().any(|r| center >= r.start && center < r.end);
            if !in_scope {
                *slot = true;
            }
        }
    }

    // Optimal (Hungarian) speaker mapping based on co-occurrence.
    let mapping = optimal_speaker_mapping(&ref_frames, &hyp_frames, &ignore_mask);

    let mut total_ref = 0u64;
    let mut missed = 0u64;
    let mut false_alarm = 0u64;
    let mut confusion = 0u64;

    for i in 0..n_frames {
        if ignore_mask[i] {
            continue;
        }

        let ref_spk = &ref_frames[i];
        let hyp_spk = &hyp_frames[i];
        let n_ref = ref_spk.len() as u64;
        let n_hyp = hyp_spk.len() as u64;

        total_ref += n_ref;

        // Count correctly matched pairs
        let mut n_correct = 0u64;
        for h in hyp_spk {
            if let Some(&mapped_ref) = mapping.get(h) {
                if ref_spk.contains(&mapped_ref) {
                    n_correct += 1;
                }
            }
        }
        n_correct = n_correct.min(n_ref);

        // Standard DER decomposition (pyannote-metrics formulation)
        missed += n_ref.saturating_sub(n_hyp);
        false_alarm += n_hyp.saturating_sub(n_ref);
        confusion += n_ref.min(n_hyp) - n_correct;
    }

    let total_ref_f = total_ref as f64;
    if total_ref == 0 {
        return DerResult {
            der: 0.0,
            miss_rate: 0.0,
            false_alarm_rate: 0.0,
            confusion_rate: 0.0,
            total_speech: 0.0,
            total_ref_frames: 0,
            missed_frames: 0,
            false_alarm_frames: 0,
            confusion_frames: 0,
        };
    }

    let total_speech_secs = total_ref as f64 * resolution;

    DerResult {
        der: (missed + false_alarm + confusion) as f64 / total_ref_f,
        miss_rate: missed as f64 / total_ref_f,
        false_alarm_rate: false_alarm as f64 / total_ref_f,
        confusion_rate: confusion as f64 / total_ref_f,
        total_speech: total_speech_secs,
        total_ref_frames: total_ref,
        missed_frames: missed,
        false_alarm_frames: false_alarm,
        confusion_frames: confusion,
    }
}

fn build_collar_mask(
    reference: &[SpeakerTurn],
    collar: f64,
    resolution: f64,
    n_frames: usize,
) -> Vec<bool> {
    let mut mask = vec![false; n_frames];
    if collar <= 0.0 {
        return mask;
    }

    for turn in reference {
        for boundary in [turn.time.start, turn.time.end] {
            let start_frame = ((boundary - collar).max(0.0) / resolution) as usize;
            let end_frame = ((boundary + collar) / resolution).ceil() as usize;
            for item in mask
                .iter_mut()
                .take(end_frame.min(n_frames))
                .skip(start_frame)
            {
                *item = true;
            }
        }
    }

    mask
}

fn build_speaker_frames(turns: &[SpeakerTurn], resolution: f64, n_frames: usize) -> Vec<Vec<u32>> {
    let mut frames: Vec<Vec<u32>> = vec![Vec::new(); n_frames];
    for turn in turns {
        let start_frame = (turn.time.start / resolution) as usize;
        let end_frame = (turn.time.end / resolution).ceil() as usize;
        for frame in frames
            .iter_mut()
            .take(end_frame.min(n_frames))
            .skip(start_frame)
        {
            if !frame.contains(&turn.speaker.0) {
                frame.push(turn.speaker.0);
            }
        }
    }
    frames
}

/// Optimal 1-to-1 mapping from hypothesis speaker IDs to reference speaker IDs.
///
/// Maximizes total frame co-occurrence via Kuhn-Munkres (Hungarian) assignment,
/// matching pyannote.metrics semantics. Greedy 1-to-1 assignment is provably
/// suboptimal — e.g. co-occurrence (X,A)=10,(X,B)=9,(Y,A)=8 yields 10 correct
/// frames greedily vs 17 optimally (X→B, Y→A) — which inflated confusion/DER on
/// cross-talk and fragmented files.
fn optimal_speaker_mapping(
    ref_frames: &[Vec<u32>],
    hyp_frames: &[Vec<u32>],
    collar_mask: &[bool],
) -> HashMap<u32, u32> {
    let mut cooccurrence: HashMap<(u32, u32), u64> = HashMap::new();

    for i in 0..ref_frames.len().min(hyp_frames.len()) {
        if collar_mask[i] {
            continue;
        }
        for &r in &ref_frames[i] {
            for &h in &hyp_frames[i] {
                *cooccurrence.entry((h, r)).or_insert(0) += 1;
            }
        }
    }

    if cooccurrence.is_empty() {
        return HashMap::new();
    }

    // Distinct hyp ids (rows) and ref ids (cols), sorted for deterministic output.
    let mut hyp_ids: Vec<u32> = cooccurrence.keys().map(|&(h, _)| h).collect();
    hyp_ids.sort_unstable();
    hyp_ids.dedup();
    let mut ref_ids: Vec<u32> = cooccurrence.keys().map(|&(_, r)| r).collect();
    ref_ids.sort_unstable();
    ref_ids.dedup();

    // Square cost matrix: cost = -co-occurrence so minimizing cost maximizes
    // agreement; padding cells stay 0.0. Counts cast to f32 are exact below
    // ~16.7M frames (f32 has a 24-bit mantissa); 10ms frames capped at 24h
    // (MAX_FRAMES) stay within that range.
    let n = hyp_ids.len().max(ref_ids.len());
    let mut cost = vec![vec![0.0_f32; n]; n];
    for (&(h, r), &count) in &cooccurrence {
        if let (Ok(i), Ok(j)) = (hyp_ids.binary_search(&h), ref_ids.binary_search(&r)) {
            cost[i][j] = -(count as f32);
        }
    }

    let assignment = match crate::hungarian::solve(&cost) {
        Some(a) => a,
        None => return HashMap::new(),
    };

    let mut mapping: HashMap<u32, u32> = HashMap::new();
    for (row, &col) in assignment.iter().enumerate() {
        // Map only real (non-padding) speakers that actually co-occur — the
        // solver may pair leftover rows/cols through zero-cost padding cells.
        if let (Some(&h), Some(&r)) = (hyp_ids.get(row), ref_ids.get(col)) {
            if cooccurrence.get(&(h, r)).copied().unwrap_or(0) > 0 {
                mapping.insert(h, r);
            }
        }
    }

    mapping
}

/// { collar >= 0.0 }
/// pub fn compute_der_from_rttm( reference: &[(f64, f64, &str)], hypothesis: &[SpeakerTurn], collar: f64, ) -> DerResult
/// { ret.der >= 0.0 && ret.der <= 1.0 }
/// Convenience: compute DER from RTTM segments (string speaker labels).
pub fn compute_der_from_rttm(
    reference: &[(f64, f64, &str)],
    hypothesis: &[SpeakerTurn],
    collar: f64,
) -> DerResult {
    let mut speaker_map: HashMap<&str, u32> = HashMap::new();
    let mut next_id = 1000u32; // offset to avoid collision with hyp IDs

    let ref_turns: Vec<SpeakerTurn> = reference
        .iter()
        .map(|&(start, end, speaker)| {
            let id = *speaker_map.entry(speaker).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            SpeakerTurn {
                speaker: crate::types::SpeakerId(id),
                time: TimeRange { start, end },
                text: None,
                stable: true,
            }
        })
        .collect();

    compute_der(&ref_turns, hypothesis, collar)
}

/// Parse a UEM (Un-partitioned Evaluation Map) file body into per-file scored
/// regions, keyed by file id. Lines are `<file-id> <channel> <start> <end>`;
/// blank lines and `;`/`#` comments are skipped, and malformed/degenerate lines
/// are ignored. Pure-Rust and wasm-clean — callers read the file and pass the
/// text here, then feed the per-file `Vec<TimeRange>` to [`compute_der_with_uem`].
pub fn parse_uem(text: &str) -> HashMap<String, Vec<TimeRange>> {
    let mut out: HashMap<String, Vec<TimeRange>> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(file), Some(_channel), Some(start), Some(end)) =
            (it.next(), it.next(), it.next(), it.next())
        else {
            continue;
        };
        let (Ok(start), Ok(end)) = (start.parse::<f64>(), end.parse::<f64>()) else {
            continue;
        };
        if !start.is_finite() || !end.is_finite() || end <= start {
            continue;
        }
        out.entry(file.to_owned())
            .or_default()
            .push(TimeRange { start, end });
    }
    out
}

/// Per-speaker recall: how much of one reference speaker's speech the mapped
/// hypothesis speaker recovered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerRecall {
    /// Reference speaker id.
    pub speaker: u32,
    /// Reference frames (10 ms, collar-excluded) for this speaker.
    pub ref_frames: u64,
    /// Of those, frames also covered by the mapped hypothesis speaker.
    pub recalled_frames: u64,
    /// `recalled_frames / ref_frames`, in [0, 1].
    pub recall: f64,
}

/// Overlap-aware DER decomposition: the headline DER plus single-speaker- and
/// overlap-region DERs and per-speaker recall.
///
/// Headline DER hides where error comes from — on overlap-heavy audio the miss
/// term dominates, so a total-DER ceiling cannot tell healthy diarization from
/// collapse. This split makes accuracy targets interpretable.
#[derive(Debug, Clone)]
pub struct DerDecomposition {
    /// Headline overlap-inclusive DER (== [`compute_der`]).
    pub total: DerResult,
    /// DER over single-speaker reference regions only
    /// (== [`compute_der_single_speaker_regions`]).
    pub single_speaker: DerResult,
    /// DER over overlap reference regions only (>= 2 concurrent reference speakers).
    pub overlap: DerResult,
    /// Per-speaker recall, sorted by reference speaker id.
    pub per_speaker_recall: Vec<SpeakerRecall>,
}

/// { collar >= 0.0 }
/// pub fn compute_der_decomposition( reference: &[SpeakerTurn], hypothesis: &[SpeakerTurn], collar: f64, ) -> DerDecomposition
/// { ret.total.der >= 0.0 && ret.total.der <= 1.0 }
/// Compute the overlap-aware DER decomposition (total / single-speaker / overlap
/// DER + per-speaker recall) in one call. Intended for bench artifacts and the
/// long-form AMI gate; the headline path stays on [`compute_der`].
pub fn compute_der_decomposition(
    reference: &[SpeakerTurn],
    hypothesis: &[SpeakerTurn],
    collar: f64,
) -> DerDecomposition {
    DerDecomposition {
        total: der_core(reference, hypothesis, collar, Region::All, None),
        single_speaker: der_core(reference, hypothesis, collar, Region::SingleSpeaker, None),
        overlap: der_core(reference, hypothesis, collar, Region::Overlap, None),
        per_speaker_recall: compute_per_speaker_recall(reference, hypothesis, collar),
    }
}

/// Per-reference-speaker recall over non-collar frames, using the same optimal
/// hyp->ref mapping as [`compute_der`].
fn compute_per_speaker_recall(
    reference: &[SpeakerTurn],
    hypothesis: &[SpeakerTurn],
    collar: f64,
) -> Vec<SpeakerRecall> {
    if reference.is_empty() || !collar.is_finite() || collar < 0.0 {
        return Vec::new();
    }

    let resolution = 0.01;
    const MAX_FRAMES: usize = 24 * 3600 * 100;
    let max_time = reference
        .iter()
        .chain(hypothesis.iter())
        .map(|t| t.time.end)
        .fold(0.0f64, f64::max);
    if !max_time.is_finite() || max_time < 0.0 {
        return Vec::new();
    }
    let n_frames = ((max_time / resolution).ceil() as usize + 1).min(MAX_FRAMES);

    let collar_mask = build_collar_mask(reference, collar, resolution, n_frames);
    let ref_frames = build_speaker_frames(reference, resolution, n_frames);
    let hyp_frames = build_speaker_frames(hypothesis, resolution, n_frames);
    let mapping = optimal_speaker_mapping(&ref_frames, &hyp_frames, &collar_mask);

    // Invert the 1-to-1 hyp->ref mapping to ref->hyp.
    let mut ref_to_hyp: HashMap<u32, u32> = HashMap::new();
    for (&h, &r) in &mapping {
        ref_to_hyp.insert(r, h);
    }

    let mut ref_count: HashMap<u32, u64> = HashMap::new();
    let mut recalled: HashMap<u32, u64> = HashMap::new();
    for i in 0..n_frames {
        if collar_mask[i] {
            continue;
        }
        for &r in &ref_frames[i] {
            *ref_count.entry(r).or_insert(0) += 1;
            if let Some(&h) = ref_to_hyp.get(&r) {
                if hyp_frames[i].contains(&h) {
                    *recalled.entry(r).or_insert(0) += 1;
                }
            }
        }
    }

    let mut out: Vec<SpeakerRecall> = ref_count
        .into_iter()
        .map(|(speaker, ref_frames)| {
            let recalled_frames = recalled.get(&speaker).copied().unwrap_or(0);
            SpeakerRecall {
                speaker,
                ref_frames,
                recalled_frames,
                recall: recalled_frames as f64 / ref_frames as f64,
            }
        })
        .collect();
    out.sort_by_key(|s| s.speaker);
    out
}

/// Word Diarization Error Rate (WDER) result.
///
/// WDER is the fraction of reference words whose speaker label is wrong after
/// optimal 1-to-1 speaker mapping (DiarizationLM / standard attribution metric).
/// Words with no reference speaker (`speaker: None`) are excluded from both
/// numerator and denominator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WderResult {
    /// `speaker_errors / total_words` in [0, 1], or 0 when no scored words.
    pub wder: f64,
    /// Reference words with a speaker label that were scored.
    pub total_words: u64,
    /// Scored words whose mapped hypothesis speaker disagrees with reference.
    pub speaker_errors: u64,
}

impl std::fmt::Display for WderResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WDER={:.1}% ({}/{} words)",
            self.wder * 100.0,
            self.speaker_errors,
            self.total_words,
        )
    }
}

/// { true }
/// pub fn compute_wder( reference: &[WordAlignment], hypothesis: &[WordAlignment], ) -> WderResult
/// { ret.wder >= 0.0 && ret.wder <= 1.0 }
/// Compute Word Diarization Error Rate between reference and hypothesis word
/// alignments.
///
/// **Alignment:** when both slices have the same length, words are paired by
/// index (the usual case when the hypothesis is speaker tags on the same ASR
/// word sequence as the reference). When lengths differ, each reference word is
/// matched to the hypothesis word with the greatest temporal midpoint proximity
/// among those sharing the same lower-cased text; unmatched refs count as
/// speaker errors.
///
/// **Speaker mapping:** optimal 1-to-1 mapping maximizing co-occurrence counts
/// over scored word pairs (Hungarian), so absolute speaker ids need not match.
///
/// Reference words with `speaker: None` are skipped. Empty reference yields
/// `wder = 0`.
pub fn compute_wder(reference: &[WordAlignment], hypothesis: &[WordAlignment]) -> WderResult {
    // Pair (ref_spk, hyp_spk_opt) for each scored reference word.
    let pairs: Vec<(u32, Option<u32>)> = if reference.len() == hypothesis.len() {
        reference
            .iter()
            .zip(hypothesis.iter())
            .filter_map(|(r, h)| {
                let ref_spk = r.speaker?.0;
                Some((ref_spk, h.speaker.map(|s| s.0)))
            })
            .collect()
    } else {
        reference
            .iter()
            .filter_map(|r| {
                let ref_spk = r.speaker?.0;
                let r_mid = (r.time.start + r.time.end) / 2.0;
                let r_word = r.word.to_ascii_lowercase();
                let mut best: Option<(usize, f64)> = None;
                for (i, h) in hypothesis.iter().enumerate() {
                    if h.word.to_ascii_lowercase() != r_word {
                        continue;
                    }
                    let h_mid = (h.time.start + h.time.end) / 2.0;
                    let dist = (r_mid - h_mid).abs();
                    if best.is_none_or(|(_, d)| dist < d) {
                        best = Some((i, dist));
                    }
                }
                let hyp_spk = best.and_then(|(i, _)| hypothesis[i].speaker.map(|s| s.0));
                Some((ref_spk, hyp_spk))
            })
            .collect()
    };

    if pairs.is_empty() {
        return WderResult {
            wder: 0.0,
            total_words: 0,
            speaker_errors: 0,
        };
    }

    // Co-occurrence for Hungarian mapping (hyp -> ref), only pairs with both sides.
    let mut cooccurrence: HashMap<(u32, u32), u64> = HashMap::new();
    for &(r, h) in &pairs {
        if let Some(h) = h {
            *cooccurrence.entry((h, r)).or_insert(0) += 1;
        }
    }

    let mapping = if cooccurrence.is_empty() {
        HashMap::new()
    } else {
        // Reuse the same Hungarian path as frame DER via a one-frame co-occurrence
        // matrix: build tiny "frames" of co-occurring pairs.
        let mut hyp_ids: Vec<u32> = cooccurrence.keys().map(|&(h, _)| h).collect();
        hyp_ids.sort_unstable();
        hyp_ids.dedup();
        let mut ref_ids: Vec<u32> = cooccurrence.keys().map(|&(_, r)| r).collect();
        ref_ids.sort_unstable();
        ref_ids.dedup();
        let n = hyp_ids.len().max(ref_ids.len());
        let mut cost = vec![vec![0.0_f32; n]; n];
        for (&(h, r), &count) in &cooccurrence {
            if let (Ok(i), Ok(j)) = (hyp_ids.binary_search(&h), ref_ids.binary_search(&r)) {
                cost[i][j] = -(count as f32);
            }
        }
        let assignment = crate::hungarian::solve(&cost).unwrap_or_default();
        let mut mapping: HashMap<u32, u32> = HashMap::new();
        for (row, &col) in assignment.iter().enumerate() {
            if let (Some(&h), Some(&r)) = (hyp_ids.get(row), ref_ids.get(col))
                && cooccurrence.get(&(h, r)).copied().unwrap_or(0) > 0
            {
                mapping.insert(h, r);
            }
        }
        mapping
    };

    let total_words = pairs.len() as u64;
    let mut speaker_errors = 0u64;
    for &(ref_spk, hyp_spk) in &pairs {
        let ok = match hyp_spk {
            Some(h) => mapping.get(&h).copied() == Some(ref_spk),
            None => false,
        };
        if !ok {
            speaker_errors += 1;
        }
    }

    WderResult {
        wder: speaker_errors as f64 / total_words as f64,
        total_words,
        speaker_errors,
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SpeakerId;

    fn turn(speaker: u32, start: f64, end: f64) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(speaker),
            time: TimeRange { start, end },
            text: None,
            stable: true,
        }
    }

    fn w(word: &str, start: f64, end: f64, spk: Option<u32>) -> crate::types::WordAlignment {
        crate::types::WordAlignment {
            word: word.to_owned(),
            time: TimeRange { start, end },
            speaker: spk.map(SpeakerId),
            confidence: 1.0,
            interpolated: false,
        }
    }

    #[test]
    fn perfect_match() {
        let reference = vec![turn(0, 0.0, 3.0), turn(1, 3.5, 6.0), turn(0, 6.5, 10.0)];
        let hypothesis = vec![turn(0, 0.0, 3.0), turn(1, 3.5, 6.0), turn(0, 6.5, 10.0)];
        let result = compute_der(&reference, &hypothesis, 0.0);
        assert!(
            result.der < 0.01,
            "perfect match DER should be ~0, got {}",
            result.der
        );
    }

    #[test]
    fn swapped_ids_still_maps() {
        let reference = vec![turn(0, 0.0, 3.0), turn(1, 3.5, 6.0)];
        let hypothesis = vec![turn(5, 0.0, 3.0), turn(9, 3.5, 6.0)];
        let result = compute_der(&reference, &hypothesis, 0.0);
        assert!(
            result.der < 0.01,
            "swapped IDs should map correctly, got DER={}",
            result.der
        );
    }

    #[test]
    fn full_miss() {
        let reference = vec![turn(0, 0.0, 5.0)];
        let hypothesis = vec![];
        let result = compute_der(&reference, &hypothesis, 0.0);
        assert!((result.miss_rate - 1.0).abs() < 0.01);
        assert!((result.der - 1.0).abs() < 0.01);
    }

    #[test]
    fn full_false_alarm() {
        let reference = vec![turn(0, 0.0, 5.0)];
        let hypothesis = vec![turn(0, 0.0, 5.0), turn(1, 0.0, 5.0)];
        let result = compute_der(&reference, &hypothesis, 0.0);
        assert!(result.false_alarm_rate > 0.5);
    }

    #[test]
    fn speaker_confusion() {
        let reference = vec![turn(0, 0.0, 3.0), turn(1, 3.0, 6.0)];
        // Both segments attributed to same speaker
        let hypothesis = vec![turn(0, 0.0, 6.0)];
        let result = compute_der(&reference, &hypothesis, 0.0);
        assert!(
            result.confusion_rate > 0.3,
            "should have confusion, got {}",
            result
        );
    }

    #[test]
    fn collar_reduces_error() {
        let reference = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        // Hypothesis has 0.2s boundary error
        let hypothesis = vec![turn(0, 0.0, 5.2), turn(1, 5.2, 10.0)];
        let no_collar = compute_der(&reference, &hypothesis, 0.0);
        let with_collar = compute_der(&reference, &hypothesis, 0.25);
        assert!(with_collar.der < no_collar.der, "collar should reduce DER");
    }

    #[test]
    fn empty_reference() {
        let result = compute_der(&[], &[turn(0, 0.0, 5.0)], 0.0);
        assert_eq!(result.der, 0.0);
    }

    #[test]
    fn non_finite_collar_returns_zero() {
        let reference = vec![turn(0, 0.0, 5.0)];
        let hypothesis = vec![turn(0, 0.0, 5.0)];
        let result = compute_der(&reference, &hypothesis, f64::NAN);
        assert_eq!(result.der, 0.0);
        let result = compute_der(&reference, &hypothesis, f64::NEG_INFINITY);
        assert_eq!(result.der, 0.0);
    }

    #[test]
    fn huge_max_time_is_capped() {
        let reference = vec![turn(0, 0.0, 1e12)];
        let hypothesis = vec![turn(0, 0.0, 1e12)];
        // Should not panic or allocate unbounded memory.
        let result = compute_der(&reference, &hypothesis, 0.0);
        assert_eq!(result.der, 0.0);
    }

    #[test]
    fn der_result_frame_counts_are_consistent() {
        // Two reference speakers, hypothesis covers only the first → real miss
        // frames, so the counts are non-trivial.
        let reference = vec![turn(0, 0.0, 3.0), turn(1, 3.0, 6.0)];
        let hypothesis = vec![turn(0, 0.0, 3.0)];
        let r = compute_der(&reference, &hypothesis, 0.0);
        assert!(
            r.total_ref_frames > 0,
            "expected non-empty reference frames"
        );
        // der == sum(error frames) / total_ref_frames
        let expected = (r.missed_frames + r.false_alarm_frames + r.confusion_frames) as f64
            / r.total_ref_frames as f64;
        assert!(
            (r.der - expected).abs() < 1e-9,
            "der {} != error-frames/ref-frames {expected}",
            r.der
        );
        // total_ref_frames are 10 ms frames, so * 0.01 == total_speech seconds.
        assert!(
            (r.total_ref_frames as f64 * 0.01 - r.total_speech).abs() < 1e-9,
            "frame count * 0.01 ({}) != total_speech ({})",
            r.total_ref_frames as f64 * 0.01,
            r.total_speech
        );
    }

    #[test]
    fn single_speaker_der_excludes_overlap_frames() {
        // ref: spk0 [0,4), spk1 [2,6) → [2,4) is a 2-speaker overlap region.
        let reference = vec![turn(0, 0.0, 4.0), turn(1, 2.0, 6.0)];
        // Empty hypothesis → everything in the scored subset is a miss.
        let hypothesis: Vec<SpeakerTurn> = vec![];
        let full = compute_der(&reference, &hypothesis, 0.0);
        let single = compute_der_single_speaker_regions(&reference, &hypothesis, 0.0);
        // Overlap frames contribute 2 ref speakers/frame to the headline metric but
        // are entirely excluded from the single-speaker metric.
        assert!(
            single.total_ref_frames < full.total_ref_frames,
            "overlap frames must be excluded: single={} full={}",
            single.total_ref_frames,
            full.total_ref_frames
        );
        // Single-speaker regions are [0,2) and [4,6) ≈ 400 frames at 10 ms.
        assert!(
            (380..=420).contains(&single.total_ref_frames),
            "expected ~400 single-speaker frames, got {}",
            single.total_ref_frames
        );
        // Still a full miss over the single-speaker subset.
        assert!(
            (single.miss_rate - 1.0).abs() < 1e-9,
            "miss={}",
            single.miss_rate
        );
    }

    #[test]
    fn single_speaker_der_ignores_overlap_mismatch() {
        // ref: spk0 [0,6) with spk1 also active on [4,6) → [4,6) is the overlap.
        let reference = vec![turn(0, 0.0, 6.0), turn(1, 4.0, 6.0)];
        // hyp: spk0 over the whole span. It is correct on the single-speaker region
        // [0,4) and only "wrong" (misses spk1) inside the excluded overlap [4,6).
        let hypothesis = vec![turn(0, 0.0, 6.0)];
        let single = compute_der_single_speaker_regions(&reference, &hypothesis, 0.0);
        assert!(
            single.der < 0.01,
            "single-speaker DER must ignore the overlap-region mismatch, got {single}"
        );
    }

    #[test]
    fn decomposition_splits_overlap_and_recall() {
        // ref: spk0 [0,6), spk1 [3,6) → [3,6) is overlap, [0,3) is single (spk0).
        let reference = vec![turn(0, 0.0, 6.0), turn(1, 3.0, 6.0)];
        // hyp: spk0 over the whole span — never recovers spk1.
        let hypothesis = vec![turn(0, 0.0, 6.0)];
        let d = compute_der_decomposition(&reference, &hypothesis, 0.0);

        // Total DER: spk1 missed across the overlap half → ~1/3.
        assert!((d.total.der - 1.0 / 3.0).abs() < 0.02, "total {}", d.total);
        // The single-speaker region [0,3) is perfectly diarized.
        assert!(d.single_speaker.der < 0.02, "single {}", d.single_speaker);
        // Overlap region [3,6): one of two speakers is missed every frame → ~0.5.
        assert!((d.overlap.der - 0.5).abs() < 0.02, "overlap {}", d.overlap);

        // Per-speaker recall: spk0 fully recovered, spk1 entirely missed.
        let r0 = d
            .per_speaker_recall
            .iter()
            .find(|s| s.speaker == 0)
            .expect("spk0 recall");
        let r1 = d
            .per_speaker_recall
            .iter()
            .find(|s| s.speaker == 1)
            .expect("spk1 recall");
        assert!((r0.recall - 1.0).abs() < 0.02, "spk0 recall {}", r0.recall);
        assert!(r1.recall < 0.02, "spk1 recall {}", r1.recall);
    }

    #[test]
    fn uem_excludes_out_of_scope_frames() {
        // ref: one speaker over [0,10); hyp empty → full miss.
        let reference = vec![turn(0, 0.0, 10.0)];
        let hypothesis: Vec<SpeakerTurn> = vec![];
        let full = compute_der(&reference, &hypothesis, 0.0);
        let scoped = compute_der_with_uem(
            &reference,
            &hypothesis,
            0.0,
            &[TimeRange {
                start: 0.0,
                end: 5.0,
            }],
        );
        // Only the [0,5) half is scored → ~half the reference frames count.
        assert!(
            scoped.total_ref_frames < full.total_ref_frames,
            "UEM must drop out-of-scope frames: scoped={} full={}",
            scoped.total_ref_frames,
            full.total_ref_frames
        );
        assert!(
            (480..=520).contains(&scoped.total_ref_frames),
            "expected ~500 scored frames, got {}",
            scoped.total_ref_frames
        );
        assert!((scoped.miss_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn uem_ignores_error_outside_scope() {
        // ref: spk0 [0,10). hyp: correct spk0 on [0,5), wrong speaker on [5,10).
        let reference = vec![turn(0, 0.0, 10.0)];
        let hypothesis = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let full = compute_der(&reference, &hypothesis, 0.0);
        let scoped = compute_der_with_uem(
            &reference,
            &hypothesis,
            0.0,
            &[TimeRange {
                start: 0.0,
                end: 5.0,
            }],
        );
        assert!(
            full.der > 0.4,
            "headline DER should see the [5,10) error, got {full}"
        );
        assert!(
            scoped.der < 0.01,
            "UEM-scoped DER must ignore the out-of-scope error, got {scoped}"
        );
    }

    #[test]
    fn uem_full_scope_matches_no_uem() {
        let reference = vec![turn(0, 0.0, 3.0), turn(1, 3.0, 6.0)];
        let hypothesis = vec![turn(0, 0.0, 3.0)];
        let plain = compute_der(&reference, &hypothesis, 0.0);
        let scoped = compute_der_with_uem(
            &reference,
            &hypothesis,
            0.0,
            &[TimeRange {
                start: 0.0,
                end: 100.0,
            }],
        );
        assert_eq!(plain.total_ref_frames, scoped.total_ref_frames);
        assert!(
            (plain.der - scoped.der).abs() < 1e-12,
            "full UEM must equal no-UEM"
        );
    }

    #[test]
    fn uem_empty_scope_scores_nothing() {
        let reference = vec![turn(0, 0.0, 5.0)];
        let hypothesis = vec![turn(0, 0.0, 5.0)];
        let scoped = compute_der_with_uem(&reference, &hypothesis, 0.0, &[]);
        assert_eq!(scoped.total_ref_frames, 0);
        assert_eq!(scoped.der, 0.0);
    }

    #[test]
    fn parse_uem_reads_regions_and_skips_junk() {
        let text = "\
; a comment
# another comment

EN2002a 1 0.00 1234.56
EN2002a 1 1300.0 1400.0
fuzfh 1 0.5 25.9
bad line with too few
EN2002a 1 50.0 10.0
";
        let map = parse_uem(text);
        let en = map.get("EN2002a").expect("EN2002a present");
        // two valid regions (the 50.0->10.0 degenerate line is dropped)
        assert_eq!(en.len(), 2);
        assert!((en[0].start - 0.0).abs() < 1e-9 && (en[0].end - 1234.56).abs() < 1e-9);
        let fz = map.get("fuzfh").expect("fuzfh present");
        assert_eq!(fz.len(), 1);
        assert!(!map.contains_key("bad"));
    }

    #[test]
    fn optimal_mapping_beats_greedy_on_counterexample() {
        // Co-occurrence (hyp, ref): (0,0)=10, (0,1)=9, (1,0)=8.
        // Greedy picks 0->0 (10 correct) and leaves hyp 1 unmapped.
        // Optimal picks 0->1 (9) + 1->0 (8) = 17 correct.
        let mut ref_frames: Vec<Vec<u32>> = Vec::new();
        let mut hyp_frames: Vec<Vec<u32>> = Vec::new();
        for _ in 0..10 {
            ref_frames.push(vec![0]);
            hyp_frames.push(vec![0]);
        }
        for _ in 0..9 {
            ref_frames.push(vec![1]);
            hyp_frames.push(vec![0]);
        }
        for _ in 0..8 {
            ref_frames.push(vec![0]);
            hyp_frames.push(vec![1]);
        }
        let collar_mask = vec![false; ref_frames.len()];
        let mapping = optimal_speaker_mapping(&ref_frames, &hyp_frames, &collar_mask);
        assert_eq!(
            mapping.get(&0),
            Some(&1),
            "hyp 0 must map to ref 1 (optimal), not ref 0 (greedy)"
        );
        assert_eq!(mapping.get(&1), Some(&0), "hyp 1 must map to ref 0");
    }

    #[test]
    fn wder_perfect_match_is_zero() {
        let reference = vec![
            w("hello", 0.0, 0.5, Some(0)),
            w("world", 0.5, 1.0, Some(0)),
            w("hi", 1.0, 1.5, Some(1)),
        ];
        // Swapped absolute ids — optimal mapping should still yield WDER 0.
        let hypothesis = vec![
            w("hello", 0.0, 0.5, Some(7)),
            w("world", 0.5, 1.0, Some(7)),
            w("hi", 1.0, 1.5, Some(3)),
        ];
        let r = compute_wder(&reference, &hypothesis);
        assert_eq!(r.total_words, 3);
        assert_eq!(r.speaker_errors, 0);
        assert!((r.wder - 0.0).abs() < 1e-12, "got {r}");
    }

    #[test]
    fn wder_hand_crafted_one_of_four_wrong() {
        // Four words; hyp mislabels the third after identity mapping.
        let reference = vec![
            w("a", 0.0, 0.5, Some(0)),
            w("b", 0.5, 1.0, Some(0)),
            w("c", 1.0, 1.5, Some(1)),
            w("d", 1.5, 2.0, Some(1)),
        ];
        let hypothesis = vec![
            w("a", 0.0, 0.5, Some(0)),
            w("b", 0.5, 1.0, Some(0)),
            w("c", 1.0, 1.5, Some(0)), // wrong: should be 1
            w("d", 1.5, 2.0, Some(1)),
        ];
        let r = compute_wder(&reference, &hypothesis);
        assert_eq!(r.total_words, 4);
        assert_eq!(r.speaker_errors, 1);
        assert!((r.wder - 0.25).abs() < 1e-12, "got {r}");
    }

    #[test]
    fn wder_empty_reference_is_zero() {
        let r = compute_wder(&[], &[w("x", 0.0, 1.0, Some(0))]);
        assert_eq!(r.total_words, 0);
        assert_eq!(r.wder, 0.0);
    }

    #[test]
    fn wder_skips_unlabeled_reference_words() {
        let reference = vec![w("a", 0.0, 0.5, None), w("b", 0.5, 1.0, Some(0))];
        let hypothesis = vec![w("a", 0.0, 0.5, Some(0)), w("b", 0.5, 1.0, Some(0))];
        let r = compute_wder(&reference, &hypothesis);
        assert_eq!(r.total_words, 1);
        assert_eq!(r.speaker_errors, 0);
    }
}
