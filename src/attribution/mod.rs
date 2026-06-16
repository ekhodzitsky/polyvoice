//! Word→speaker attribution: join raw ASR words to diarization turns.
//!
//! Pure-Rust and wasm-clean, behind the opt-in `attribution` feature — no models,
//! no `ort`, no I/O. Just interval arithmetic on [`TimeRange`]/[`SpeakerId`],
//! reusing the same overlap definition as `der::compute_der`.

use crate::types::{SpeakerId, SpeakerTurn, TimeRange, Word, WordAlignment};

/// Overlap of two time intervals in seconds (0 when disjoint).
fn overlap(a: &TimeRange, b: &TimeRange) -> f64 {
    (a.end.min(b.end) - a.start.max(b.start)).max(0.0)
}

/// Gap between two intervals in seconds (0 when they overlap/touch).
fn gap(a: &TimeRange, b: &TimeRange) -> f64 {
    if a.end <= b.start {
        b.start - a.end
    } else if b.end <= a.start {
        a.start - b.end
    } else {
        0.0
    }
}

/// Attribute each ASR word to a diarization speaker turn, returning a
/// [`WordAlignment`] per input word in the **same order and length**.
///
/// Rules:
/// - **Overlap:** the word goes to the turn with the greatest temporal overlap;
///   `speaker = turn.speaker`. Confidence is scaled by the fraction of the word
///   covered by that turn, so a word straddling two turns (or partly in silence)
///   gets a **lowered** confidence while a fully-covered word keeps its ASR score.
/// - **No overlap:** the word is attributed to the nearest turn by interval gap.
/// - **Empty `turns`:** words pass through with `speaker: None`.
///
/// Ties (equal overlap, or equal gap) are broken deterministically by the smaller
/// `SpeakerId`, then the earlier turn.
pub fn attribute_words(words: &[Word], turns: &[SpeakerTurn]) -> Vec<WordAlignment> {
    words.iter().map(|w| attribute_one(w, turns)).collect()
}

fn attribute_one(word: &Word, turns: &[SpeakerTurn]) -> WordAlignment {
    let make = |speaker: Option<SpeakerId>, confidence: f32| WordAlignment {
        word: word.word.clone(),
        time: word.time,
        speaker,
        confidence,
    };

    if turns.is_empty() {
        return make(None, word.confidence);
    }

    // Best-overlap turn (tie-break: smaller SpeakerId, then earlier turn). `turns`
    // is non-empty here, so seed with turn 0 and refine — no Option/expect needed.
    let mut bi = 0usize;
    let mut bov = overlap(&word.time, &turns[0].time);
    for (i, t) in turns.iter().enumerate().skip(1) {
        let ov = overlap(&word.time, &t.time);
        if ov > bov || (ov == bov && t.speaker.0 < turns[bi].speaker.0) {
            bi = i;
            bov = ov;
        }
    }

    if bov > 0.0 {
        let word_dur = (word.time.end - word.time.start).max(0.0);
        // Coverage share in [0, 1]: 1.0 when fully inside one turn, < 1.0 when the
        // word straddles a boundary or spills into silence → confidence drops.
        let conf = if word_dur > 0.0 {
            (word.confidence as f64 * (bov / word_dur).min(1.0)) as f32
        } else {
            word.confidence
        };
        return make(Some(turns[bi].speaker), conf);
    }

    // No overlap: nearest turn by gap (tie-break smaller SpeakerId, then earlier).
    let mut nearest = 0usize;
    let mut min_gap = f64::INFINITY;
    for (i, t) in turns.iter().enumerate() {
        let g = gap(&word.time, &t.time);
        if g < min_gap || (g == min_gap && t.speaker.0 < turns[nearest].speaker.0) {
            min_gap = g;
            nearest = i;
        }
    }
    make(Some(turns[nearest].speaker), word.confidence)
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64, end: f64, conf: f32) -> Word {
        Word {
            word: text.to_owned(),
            time: TimeRange { start, end },
            confidence: conf,
        }
    }
    fn turn(id: u32, start: f64, end: f64) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(id),
            time: TimeRange { start, end },
            text: None,
        }
    }

    #[test]
    fn word_fully_inside_turn_keeps_confidence() {
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let out = attribute_words(&[word("hi", 1.0, 2.0, 0.9)], &turns);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, Some(SpeakerId(0)));
        assert!((out[0].confidence - 0.9).abs() < 1e-6, "conf {}", out[0].confidence);
    }

    #[test]
    fn straddling_word_picks_dominant_and_lowers_confidence() {
        // [4.0, 6.0): 1.0s in spk0 [0,5), 1.0s in spk1 [5,10) — exact tie on overlap
        // → tie-break to smaller SpeakerId (0); confidence halved (50% coverage).
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let out = attribute_words(&[word("x", 4.0, 6.0, 1.0)], &turns);
        assert_eq!(out[0].speaker, Some(SpeakerId(0)));
        assert!(out[0].confidence < 1.0, "straddle conf must drop, got {}", out[0].confidence);
        assert!((out[0].confidence - 0.5).abs() < 1e-6);

        // Dominant share: [4.0, 8.0) is 1.0s in spk0, 3.0s in spk1 → spk1 wins.
        let out2 = attribute_words(&[word("y", 4.0, 8.0, 1.0)], &turns);
        assert_eq!(out2[0].speaker, Some(SpeakerId(1)));
        assert!((out2[0].confidence - 0.75).abs() < 1e-6); // 3/4 covered
    }

    #[test]
    fn word_in_silence_goes_to_nearest_turn() {
        // turns at [0,2) spk0 and [8,10) spk1; word at [4.5,5.0) → nearer to spk0
        // (gap 2.5) than spk1 (gap 3.0).
        let turns = vec![turn(0, 0.0, 2.0), turn(1, 8.0, 10.0)];
        let out = attribute_words(&[word("z", 4.5, 5.0, 0.8)], &turns);
        assert_eq!(out[0].speaker, Some(SpeakerId(0)));
        assert!((out[0].confidence - 0.8).abs() < 1e-6); // unchanged for nearest
    }

    #[test]
    fn empty_turns_yield_none_speaker() {
        let out = attribute_words(&[word("a", 0.0, 1.0, 0.7)], &[]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, None);
        assert!((out[0].confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn empty_words_yield_empty() {
        let turns = vec![turn(0, 0.0, 5.0)];
        assert!(attribute_words(&[], &turns).is_empty());
    }

    #[test]
    fn order_and_length_preserved_incl_before_after() {
        let turns = vec![turn(0, 2.0, 4.0), turn(1, 6.0, 8.0)];
        let words = vec![
            word("before", 0.0, 1.0, 1.0), // before first turn → nearest spk0
            word("in0", 2.5, 3.0, 1.0),    // in spk0
            word("after", 9.0, 9.5, 1.0),  // after last turn → nearest spk1
        ];
        let out = attribute_words(&words, &turns);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].word, "before");
        assert_eq!(out[2].word, "after");
        assert_eq!(out[0].speaker, Some(SpeakerId(0)));
        assert_eq!(out[1].speaker, Some(SpeakerId(0)));
        assert_eq!(out[2].speaker, Some(SpeakerId(1)));
        // every word attributed
        assert!(out.iter().all(|w| w.speaker.is_some()));
    }
}
