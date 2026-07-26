//! Frame-based Diarization Error Rate (DER).
use crate::types::{SpeakerTurn, TimeRange};
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
pub(crate) enum Region {
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
pub(crate) fn der_core(
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
            if let Some(&mapped_ref) = mapping.get(h)
                && ref_spk.contains(&mapped_ref)
            {
                n_correct += 1;
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

pub(crate) fn build_collar_mask(
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

pub(crate) fn build_speaker_frames(
    turns: &[SpeakerTurn],
    resolution: f64,
    n_frames: usize,
) -> Vec<Vec<u32>> {
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
pub(crate) fn optimal_speaker_mapping(
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
        if let (Some(&h), Some(&r)) = (hyp_ids.get(row), ref_ids.get(col))
            && cooccurrence.get(&(h, r)).copied().unwrap_or(0) > 0
        {
            mapping.insert(h, r);
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
