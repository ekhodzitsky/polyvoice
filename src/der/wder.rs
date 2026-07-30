//! Word Diarization Error Rate (WDER).
use crate::types::WordAlignment;
use std::collections::HashMap;

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
                let r_mid = r.time.midpoint();
                let r_word = r.word.to_ascii_lowercase();
                let mut best: Option<(usize, f64)> = None;
                for (i, h) in hypothesis.iter().enumerate() {
                    if h.word.to_ascii_lowercase() != r_word {
                        continue;
                    }
                    let h_mid = h.time.midpoint();
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

    // Optimal (Hungarian) speaker mapping based on co-occurrence — the same
    // max-co-occurrence path as frame DER.
    let mapping = crate::hungarian::map_max_cooccurrence(&cooccurrence);

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
