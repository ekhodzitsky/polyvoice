//! Overlap-aware DER decomposition and per-speaker recall.
use super::frame::{
    DerResult, Region, build_collar_mask, build_speaker_frames, der_core, optimal_speaker_mapping,
};
use crate::types::SpeakerTurn;
use std::collections::HashMap;

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
            if let Some(&h) = ref_to_hyp.get(&r)
                && hyp_frames[i].contains(&h)
            {
                *recalled.entry(r).or_insert(0) += 1;
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
