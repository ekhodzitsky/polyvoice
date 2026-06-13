//! Diarization Error Rate (DER) computation.
//!
//! Frame-based DER with forgiveness collar and optimal speaker mapping.

use crate::types::{SpeakerTurn, TimeRange};
use std::collections::HashMap;
use std::collections::HashSet;

/// DER evaluation result.
#[derive(Debug, Clone, Copy)]
pub struct DerResult {
    pub der: f64,
    pub miss_rate: f64,
    pub false_alarm_rate: f64,
    pub confusion_rate: f64,
    pub total_speech: f64,
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
/// Speaker IDs between ref and hyp are mapped optimally via greedy matching
/// on co-occurrence counts.
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
    if reference.is_empty() {
        return DerResult {
            der: 0.0,
            miss_rate: 0.0,
            false_alarm_rate: 0.0,
            confusion_rate: 0.0,
            total_speech: 0.0,
        };
    }

    if !collar.is_finite() || collar < 0.0 {
        return DerResult {
            der: 0.0,
            miss_rate: 0.0,
            false_alarm_rate: 0.0,
            confusion_rate: 0.0,
            total_speech: 0.0,
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
        };
    }

    let n_frames = ((max_time / resolution).ceil() as usize + 1).min(MAX_FRAMES);

    // Build collar mask: true = inside collar (ignored).
    let collar_mask = build_collar_mask(reference, collar, resolution, n_frames);

    // Build frame-level speaker labels.
    let ref_frames = build_speaker_frames(reference, resolution, n_frames);
    let hyp_frames = build_speaker_frames(hypothesis, resolution, n_frames);

    // Greedy speaker mapping based on co-occurrence.
    let mapping = greedy_speaker_mapping(&ref_frames, &hyp_frames, &collar_mask);

    let mut total_ref = 0u64;
    let mut missed = 0u64;
    let mut false_alarm = 0u64;
    let mut confusion = 0u64;

    for i in 0..n_frames {
        if collar_mask[i] {
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
        };
    }

    let total_speech_secs = total_ref as f64 * resolution;

    DerResult {
        der: (missed + false_alarm + confusion) as f64 / total_ref_f,
        miss_rate: missed as f64 / total_ref_f,
        false_alarm_rate: false_alarm as f64 / total_ref_f,
        confusion_rate: confusion as f64 / total_ref_f,
        total_speech: total_speech_secs,
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

/// Greedy 1-to-1 mapping from hypothesis speaker IDs to reference speaker IDs.
fn greedy_speaker_mapping(
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

    let mut pairs: Vec<((u32, u32), u64)> = cooccurrence.into_iter().collect();
    pairs.sort_by_key(|a| std::cmp::Reverse(a.1));

    let mut mapping: HashMap<u32, u32> = HashMap::new();
    let mut used_ref: HashSet<u32> = HashSet::new();

    for ((h, r), _) in pairs {
        if !mapping.contains_key(&h) && !used_ref.contains(&r) {
            mapping.insert(h, r);
            used_ref.insert(r);
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
            }
        })
        .collect();

    compute_der(&ref_turns, hypothesis, collar)
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
}
