//! Label stability helpers for streaming diarization.
//!
//! - [`prefer_current_speaker`]: Speechmatics-style hysteresis that suppresses
//!   single-frame speaker flicker when scores are close.
//! - [`label_flip_rate`]: frame-level flip rate of final labels vs first-emitted
//!   labels (AssemblyAI-style stability metric).

use crate::types::SpeakerId;

/// Prefer the currently active speaker when its score is within `margin` of the
/// best alternative (hysteresis).
///
/// `candidates` is a list of `(speaker, score)` pairs for the current frame —
/// higher score is better (cosine similarity in the streaming pipeline).
///
/// Returns the speaker to use:
/// - If `current` is among the candidates and `best_score - current_score <= margin`,
///   keep `current`.
/// - Otherwise return the highest-scoring candidate (or `None` if empty).
pub fn prefer_current_speaker(
    current: Option<SpeakerId>,
    candidates: &[(SpeakerId, f32)],
    margin: f32,
) -> Option<SpeakerId> {
    if candidates.is_empty() {
        return None;
    }
    let (best_id, best_score) = candidates
        .iter()
        .copied()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;

    let Some(cur) = current else {
        return Some(best_id);
    };
    if cur == best_id {
        return Some(cur);
    }
    let cur_score = candidates
        .iter()
        .find(|(id, _)| *id == cur)
        .map(|(_, s)| *s);
    match cur_score {
        Some(cs) if best_score - cs <= margin => Some(cur),
        _ => Some(best_id),
    }
}

/// Fraction of positions where `final_labels[i] != first_emitted[i]`.
///
/// Both slices are compared over `min(len)` positions. Empty input → `0.0`.
/// Values are in `[0.0, 1.0]`. Lower is more stable.
pub fn label_flip_rate(first_emitted: &[SpeakerId], final_labels: &[SpeakerId]) -> f32 {
    let n = first_emitted.len().min(final_labels.len());
    if n == 0 {
        return 0.0;
    }
    let flips = first_emitted
        .iter()
        .zip(final_labels.iter())
        .take(n)
        .filter(|(a, b)| a != b)
        .count();
    flips as f32 / n as f32
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn hysteresis_keeps_current_within_margin() {
        let cur = SpeakerId(0);
        let cands = [(SpeakerId(0), 0.70), (SpeakerId(1), 0.74)];
        // margin 0.05: best-current = 0.04 ≤ 0.05 → keep 0
        let got = prefer_current_speaker(Some(cur), &cands, 0.05);
        assert_eq!(got, Some(SpeakerId(0)));
    }

    #[test]
    fn hysteresis_switches_when_margin_exceeded() {
        let cur = SpeakerId(0);
        let cands = [(SpeakerId(0), 0.60), (SpeakerId(1), 0.80)];
        let got = prefer_current_speaker(Some(cur), &cands, 0.05);
        assert_eq!(got, Some(SpeakerId(1)));
    }

    #[test]
    fn hysteresis_no_current_picks_best() {
        let cands = [(SpeakerId(2), 0.5), (SpeakerId(3), 0.9)];
        assert_eq!(
            prefer_current_speaker(None, &cands, 0.1),
            Some(SpeakerId(3))
        );
    }

    #[test]
    fn hysteresis_suppresses_single_flicker() {
        // Sequence: speaker 0 dominant, one frame slightly favors 1, then 0 again.
        let margin = 0.08;
        let mut current = Some(SpeakerId(0));
        let frames = [
            vec![(SpeakerId(0), 0.85), (SpeakerId(1), 0.40)],
            vec![(SpeakerId(0), 0.72), (SpeakerId(1), 0.78)], // flicker within margin
            vec![(SpeakerId(0), 0.88), (SpeakerId(1), 0.35)],
        ];
        let mut labels = Vec::new();
        for f in &frames {
            current = prefer_current_speaker(current, f, margin);
            labels.push(current.unwrap());
        }
        assert_eq!(
            labels,
            vec![SpeakerId(0), SpeakerId(0), SpeakerId(0)],
            "single near-tie flicker must not flip the label"
        );
    }

    #[test]
    fn flip_rate_zero_when_identical() {
        let a = [SpeakerId(0), SpeakerId(1), SpeakerId(1)];
        assert_eq!(label_flip_rate(&a, &a), 0.0);
    }

    #[test]
    fn flip_rate_counts_disagreements() {
        let first = [SpeakerId(0), SpeakerId(0), SpeakerId(1), SpeakerId(2)];
        let final_ = [SpeakerId(0), SpeakerId(1), SpeakerId(1), SpeakerId(1)];
        // flips at indices 1 and 3 → 2/4 = 0.5
        assert!((label_flip_rate(&first, &final_) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn flip_rate_empty_is_zero() {
        assert_eq!(label_flip_rate(&[], &[]), 0.0);
    }

    #[test]
    fn hysteresis_empty_candidates_returns_none() {
        assert_eq!(prefer_current_speaker(Some(SpeakerId(0)), &[], 0.1), None);
        assert_eq!(prefer_current_speaker(None, &[], 0.1), None);
    }

    #[test]
    fn hysteresis_current_absent_from_candidates_picks_best() {
        let cands = [(SpeakerId(0), 0.5), (SpeakerId(1), 0.9)];
        assert_eq!(
            prefer_current_speaker(Some(SpeakerId(7)), &cands, 0.5),
            Some(SpeakerId(1)),
            "a current speaker with no score cannot be kept"
        );
    }

    #[test]
    fn hysteresis_current_already_best_is_kept() {
        let cands = [(SpeakerId(0), 0.9), (SpeakerId(1), 0.5)];
        assert_eq!(
            prefer_current_speaker(Some(SpeakerId(0)), &cands, 0.0),
            Some(SpeakerId(0))
        );
    }

    #[test]
    fn flip_rate_compares_over_shorter_input() {
        let first = [SpeakerId(0), SpeakerId(1), SpeakerId(1), SpeakerId(9)];
        let final_ = [SpeakerId(0), SpeakerId(0)];
        // Only the first two positions are compared: one flip out of two.
        assert!((label_flip_rate(&first, &final_) - 0.5).abs() < 1e-6);
        assert_eq!(label_flip_rate(&first, &[]), 0.0);
    }
}
