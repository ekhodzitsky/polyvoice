//! Word→speaker attribution: join raw ASR words to diarization turns.
//!
//! Pure-Rust and wasm-clean, behind the opt-in `attribution` feature — no models,
//! no `ort`, no I/O. Just interval arithmetic on [`TimeRange`]/[`SpeakerId`],
//! reusing the same overlap definition as `der::compute_der`.

use crate::asr::{Asr, AsrError};
use crate::types::{SampleRate, SpeakerId, SpeakerTurn, TimeRange, Word, WordAlignment};

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

/// Result of the who-said-what cascade: ASR words tagged with speakers, plus the
/// diarization turns with [`SpeakerTurn::text`] filled from those words.
#[derive(Debug, Clone, PartialEq)]
pub struct WhoSaidWhat {
    /// Every ASR word with its attributed speaker (same order/length as the ASR
    /// output).
    pub words: Vec<WordAlignment>,
    /// Diarization turns, each with `text` assembled from its words in time order.
    pub turns: Vec<SpeakerTurn>,
}

/// Assemble [`SpeakerTurn::text`] for each turn from `aligned` words. A word
/// belongs to a turn when it was attributed to that turn's speaker and its
/// midpoint falls within the turn's span; words are joined in time order. Turns
/// with no words keep `text: None`.
pub fn fill_turn_text(turns: &[SpeakerTurn], aligned: &[WordAlignment]) -> Vec<SpeakerTurn> {
    turns
        .iter()
        .map(|turn| {
            let mut words: Vec<&WordAlignment> = aligned
                .iter()
                .filter(|w| {
                    let mid = (w.time.start + w.time.end) / 2.0;
                    w.speaker == Some(turn.speaker) && mid >= turn.time.start && mid < turn.time.end
                })
                .collect();
            words.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));
            let text = if words.is_empty() {
                None
            } else {
                Some(
                    words
                        .iter()
                        .map(|w| w.word.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                )
            };
            SpeakerTurn {
                speaker: turn.speaker,
                time: turn.time,
                text,
            }
        })
        .collect()
}

/// Join raw ASR `words` to diarization `turns`: attribute each word to a speaker
/// (overlap-region words go to the **dominant** speaker only — see
/// [`attribute_words`]), then fill each turn's text. Pure — no ASR, no I/O.
pub fn attribute_and_fill(words: &[Word], turns: &[SpeakerTurn]) -> WhoSaidWhat {
    let aligned = attribute_words(words, turns);
    let turns = fill_turn_text(turns, &aligned);
    WhoSaidWhat {
        words: aligned,
        turns,
    }
}

/// Cascaded who-said-what: run **one** ASR pass over the whole audio, then join
/// its word timestamps to the already-computed diarization `turns`.
///
/// Diarizer-agnostic — pass `turns` from any pipeline. Diarization must run
/// FIRST (the caller supplies `turns`); ASR is a single pass over the full
/// `samples`. Per-segment ASR is intentionally NOT done: it loses cross-boundary
/// language context and multiplies cost.
///
/// Known limitation: words inside overlapped speech are attributed to the single
/// dominant speaker only.
pub fn who_said_what(
    turns: &[SpeakerTurn],
    asr: &dyn Asr,
    samples: &[f32],
    sample_rate: SampleRate,
) -> Result<WhoSaidWhat, AsrError> {
    let words = asr.transcribe(samples, sample_rate)?;
    Ok(attribute_and_fill(&words, turns))
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
        assert!(
            (out[0].confidence - 0.9).abs() < 1e-6,
            "conf {}",
            out[0].confidence
        );
    }

    #[test]
    fn straddling_word_picks_dominant_and_lowers_confidence() {
        // [4.0, 6.0): 1.0s in spk0 [0,5), 1.0s in spk1 [5,10) — exact tie on overlap
        // → tie-break to smaller SpeakerId (0); confidence halved (50% coverage).
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let out = attribute_words(&[word("x", 4.0, 6.0, 1.0)], &turns);
        assert_eq!(out[0].speaker, Some(SpeakerId(0)));
        assert!(
            out[0].confidence < 1.0,
            "straddle conf must drop, got {}",
            out[0].confidence
        );
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

    // --- cascade (who-said-what) ---

    struct MockAsr(Vec<Word>);
    impl Asr for MockAsr {
        fn transcribe(&self, _a: &[f32], _sr: SampleRate) -> Result<Vec<Word>, AsrError> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn cascade_assigns_speakers_and_fills_turn_text() {
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let words = vec![
            word("hello", 0.5, 1.0, 1.0),
            word("there", 1.5, 2.0, 1.0),
            word("hi", 6.0, 6.5, 1.0),
        ];
        let wsw = attribute_and_fill(&words, &turns);
        assert_eq!(wsw.words[0].speaker, Some(SpeakerId(0)));
        assert_eq!(wsw.words[2].speaker, Some(SpeakerId(1)));
        assert_eq!(wsw.turns[0].text.as_deref(), Some("hello there"));
        assert_eq!(wsw.turns[1].text.as_deref(), Some("hi"));
    }

    #[test]
    fn cascade_overlap_word_goes_to_dominant_turn() {
        // [4.0,6.0) overlaps spk0 [0,5) by 1s and spk1 [5,10) by 1s — tie →
        // dominant by tie-break is the smaller SpeakerId (0); the word lands there.
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let wsw = attribute_and_fill(&[word("x", 4.0, 6.0, 1.0)], &turns);
        assert_eq!(wsw.words[0].speaker, Some(SpeakerId(0)));
        // midpoint 5.0 is NOT < turn0.end (5.0), so it isn't placed in turn0's
        // text — text-fill is midpoint-in-span; attribution still tags the speaker.
        assert_eq!(wsw.turns[1].text, None);
    }

    #[test]
    fn cascade_turn_text_is_time_ordered() {
        let turns = vec![turn(0, 0.0, 10.0)];
        // deliberately out of time order on input
        let words = vec![word("world", 3.0, 3.5, 1.0), word("hello", 1.0, 1.5, 1.0)];
        let wsw = attribute_and_fill(&words, &turns);
        assert_eq!(wsw.turns[0].text.as_deref(), Some("hello world"));
    }

    #[test]
    fn cascade_empty_asr_yields_turns_without_text() {
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let wsw = attribute_and_fill(&[], &turns);
        assert!(wsw.words.is_empty());
        assert!(wsw.turns.iter().all(|t| t.text.is_none()));
        assert_eq!(wsw.turns.len(), 2);
    }

    #[test]
    fn who_said_what_runs_one_asr_pass() {
        let turns = vec![turn(0, 0.0, 5.0)];
        let asr = MockAsr(vec![word("one", 1.0, 1.5, 1.0), word("two", 2.0, 2.5, 1.0)]);
        let sr = SampleRate::new(16_000).unwrap();
        let wsw = who_said_what(&turns, &asr, &[0.0_f32; 16], sr).unwrap();
        assert_eq!(wsw.words.len(), 2);
        assert_eq!(wsw.turns[0].text.as_deref(), Some("one two"));
    }
}
