//! Word→speaker attribution: join raw ASR words to diarization turns.
//!
//! Pure-Rust and wasm-clean, behind the opt-in `attribution` feature — no models,
//! no `ort`, no I/O. Just interval arithmetic on [`TimeRange`]/[`SpeakerId`],
//! reusing the same overlap definition as `der::compute_der`.
//!
//! The join is an O(W+T) two-pointer sweep over time-sorted words and turns
//! (max-overlap tagging, bit-identical to the historical linear scan). Optional
//! extras: missing-timestamp interpolation, sentence-level speaker smoothing,
//! and a configurable word anchor for turn-text placement.

use crate::asr::{Asr, AsrError};
use crate::types::{
    SampleRate, SpeakerId, SpeakerTurn, TimeRange, Word, WordAlignment, mean_speaker_embeddings,
};

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

/// Which point on a word interval is used for turn-text placement.
///
/// Tagging itself always uses max temporal overlap of the full word span
/// (historical behavior). The anchor only affects [`fill_turn_text`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WordAnchor {
    /// Word start time.
    Start,
    /// Midpoint `(start + end) / 2` — historical `fill_turn_text` default.
    #[default]
    Mid,
    /// Word end time.
    End,
}

impl WordAnchor {
    /// Resolve the anchor point on `time`.
    pub fn point(self, time: &TimeRange) -> f64 {
        match self {
            WordAnchor::Start => time.start,
            WordAnchor::Mid => (time.start + time.end) / 2.0,
            WordAnchor::End => time.end,
        }
    }
}

/// Configuration for the word→speaker join.
///
/// Defaults preserve historical tagging (max-overlap, no sentence smoothing)
/// and historical turn-text placement (midpoint anchor). Missing/zero-duration
/// timestamps are interpolated by default so attribution stays total.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributionConfig {
    /// Point of each word used by [`fill_turn_text`] membership tests.
    pub word_anchor: WordAnchor,
    /// When true, apply sentence-level speaker smoothing after tagging.
    /// Default `false` — opt-in until measured on a real cascade.
    pub sentence_smoothing: bool,
    /// Dominant-speaker fraction required to relabel a whole sentence.
    /// Compared with `>`; default `0.5` means strictly more than half the words.
    pub smoothing_threshold: f32,
    /// When true (default), fill missing/zero-duration word timestamps via
    /// nearest-neighbor interpolation before tagging.
    pub interpolate_timestamps: bool,
}

impl Default for AttributionConfig {
    fn default() -> Self {
        Self {
            word_anchor: WordAnchor::Mid,
            sentence_smoothing: false,
            smoothing_threshold: 0.5,
            interpolate_timestamps: true,
        }
    }
}

fn make_alignment(
    word: &Word,
    speaker: Option<SpeakerId>,
    confidence: f32,
    interpolated: bool,
) -> WordAlignment {
    WordAlignment {
        word: word.word.clone(),
        time: word.time,
        speaker,
        confidence,
        interpolated,
    }
}

/// True when a word's timestamps are unusable for interval join.
fn needs_timestamp_interpolation(w: &Word) -> bool {
    !w.time.start.is_finite()
        || !w.time.end.is_finite()
        || w.time
            .end
            .partial_cmp(&w.time.start)
            .is_none_or(|o| !matches!(o, std::cmp::Ordering::Greater))
}

/// Fill missing/zero-duration word timestamps with nearest-neighbor values.
///
/// For each unusable word, `start` comes from the previous valid word's `end`
/// and `end` from the next valid word's `start` (clamped to a single edge when
/// only one neighbor exists). Words that already have positive finite duration
/// are left unchanged. Output length always equals input length; the parallel
/// `interpolated` flags mark which entries were rewritten.
///
/// When both neighbors collapse to a non-positive span, a 1 ms epsilon duration
/// is used so the word still participates in max-overlap tagging.
pub fn interpolate_word_timestamps(words: &[Word]) -> (Vec<Word>, Vec<bool>) {
    let n = words.len();
    let mut out = words.to_vec();
    let mut interpolated = vec![false; n];
    if n == 0 {
        return (out, interpolated);
    }

    let valid: Vec<bool> = words
        .iter()
        .map(|w| !needs_timestamp_interpolation(w))
        .collect();

    const EPS: f64 = 1e-3;

    for i in 0..n {
        if valid[i] {
            continue;
        }
        interpolated[i] = true;

        let prev = (0..i).rev().find(|&j| valid[j]);
        let next = ((i + 1)..n).find(|&j| valid[j]);

        let (mut start, mut end) = match (prev, next) {
            (Some(p), Some(nx)) => (words[p].time.end, words[nx].time.start),
            (Some(p), None) => {
                let s = words[p].time.end;
                (s, s + EPS)
            }
            (None, Some(nx)) => {
                let e = words[nx].time.start;
                ((e - EPS).max(0.0), e)
            }
            (None, None) => (0.0, EPS),
        };

        if end
            .partial_cmp(&start)
            .is_none_or(|o| !matches!(o, std::cmp::Ordering::Greater))
        {
            // Neighbors meet or cross: place a tiny interval at the boundary.
            start = start.max(0.0);
            end = start + EPS;
        }

        out[i].time = TimeRange { start, end };
    }

    (out, interpolated)
}

/// Whether `cand` beats `cur` under the documented tie-break: smaller
/// [`SpeakerId`], then earlier original turn index.
fn better_turn(cand: usize, cur: usize, turns: &[SpeakerTurn]) -> bool {
    let cs = turns[cand].speaker.0;
    let us = turns[cur].speaker.0;
    cs < us || (cs == us && cand < cur)
}

/// Historical O(W·T) scan — kept for equivalence property tests only.
#[cfg(test)]
fn attribute_one_reference(word: &Word, turns: &[SpeakerTurn]) -> (Option<SpeakerId>, f32) {
    if turns.is_empty() {
        return (None, word.confidence);
    }

    let mut bi = 0usize;
    let mut bov = overlap(&word.time, &turns[0].time);
    for (i, t) in turns.iter().enumerate().skip(1) {
        let ov = overlap(&word.time, &t.time);
        if ov > bov || (ov == bov && better_turn(i, bi, turns)) {
            bi = i;
            bov = ov;
        }
    }

    if bov > 0.0 {
        let word_dur = (word.time.end - word.time.start).max(0.0);
        let conf = if word_dur > 0.0 {
            (word.confidence as f64 * (bov / word_dur).min(1.0)) as f32
        } else {
            word.confidence
        };
        return (Some(turns[bi].speaker), conf);
    }

    let mut nearest = 0usize;
    let mut min_gap = f64::INFINITY;
    for (i, t) in turns.iter().enumerate() {
        let g = gap(&word.time, &t.time);
        if g < min_gap || (g == min_gap && better_turn(i, nearest, turns)) {
            min_gap = g;
            nearest = i;
        }
    }
    (Some(turns[nearest].speaker), word.confidence)
}

/// Tag one word against sorted turns via a candidate window starting at `left`.
///
/// `turn_order` is turn indices sorted by start time. `left` is the first index
/// into `turn_order` whose turn may still overlap this word (turns before it
/// end at or before the word start). Returns the attributed speaker + confidence.
fn attribute_one_sweep(
    word: &Word,
    turns: &[SpeakerTurn],
    turn_order: &[usize],
    left: usize,
    best_left: Option<usize>,
) -> (Option<SpeakerId>, f32) {
    debug_assert!(!turns.is_empty());

    let mut bi: Option<usize> = None;
    let mut bov = 0.0f64;
    let mut j = left;
    while j < turn_order.len() {
        let ti = turn_order[j];
        let t = &turns[ti];
        // Turns are sorted by start: once start >= word.end, no further overlap.
        if t.time.start >= word.time.end {
            break;
        }
        let ov = overlap(&word.time, &t.time);
        if ov > 0.0 {
            let take = match bi {
                None => true,
                Some(cur) => ov > bov || (ov == bov && better_turn(ti, cur, turns)),
            };
            if take {
                bi = Some(ti);
                bov = ov;
            }
        }
        j += 1;
    }

    if let Some(ti) = bi.filter(|_| bov > 0.0) {
        let word_dur = (word.time.end - word.time.start).max(0.0);
        let conf = if word_dur > 0.0 {
            (word.confidence as f64 * (bov / word_dur).min(1.0)) as f32
        } else {
            word.confidence
        };
        return (Some(turns[ti].speaker), conf);
    }

    // No overlap: nearest turn by gap. Left candidate is the best (max end)
    // turn completely to the left; right candidates are turns that start at or
    // after the word end (and any non-overlapping turns still in the window).
    let mut nearest: Option<usize> = best_left;
    let mut min_gap = best_left
        .map(|ti| gap(&word.time, &turns[ti].time))
        .unwrap_or(f64::INFINITY);

    // Consider every remaining turn from `left` (none overlapped with ov > 0).
    for &ti in turn_order.iter().skip(left) {
        let g = gap(&word.time, &turns[ti].time);
        let take = match nearest {
            None => true,
            Some(cur) => g < min_gap || (g == min_gap && better_turn(ti, cur, turns)),
        };
        if take {
            nearest = Some(ti);
            min_gap = g;
        }
    }

    // If best_left was None and left == turn_order.len(), also scan nothing —
    // but then every turn was to the left and best_left should have been set.
    // Fall back to a full scan only if the window produced nothing (defensive).
    let nearest = nearest.unwrap_or(0);
    (Some(turns[nearest].speaker), word.confidence)
}

/// Attribute each ASR word to a diarization speaker turn, returning a
/// [`WordAlignment`] per input word in the **same order and length**.
///
/// Equivalent to [`attribute_words_with_config`] with [`AttributionConfig::default`].
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
/// `SpeakerId`, then the earlier turn (lower original index).
pub fn attribute_words(words: &[Word], turns: &[SpeakerTurn]) -> Vec<WordAlignment> {
    attribute_words_with_config(words, turns, &AttributionConfig::default())
}

/// Like [`attribute_words`], but with explicit [`AttributionConfig`].
///
/// Pipeline:
/// 1. Optionally interpolate missing/zero-duration timestamps (totality-preserving).
/// 2. O(W+T) two-pointer max-overlap join (bit-identical to the historical scan).
/// 3. Optionally smooth mid-sentence speaker flips.
pub fn attribute_words_with_config(
    words: &[Word],
    turns: &[SpeakerTurn],
    config: &AttributionConfig,
) -> Vec<WordAlignment> {
    let (owned, interp_flags) = if config.interpolate_timestamps {
        interpolate_word_timestamps(words)
    } else {
        (words.to_vec(), vec![false; words.len()])
    };
    let words = owned.as_slice();

    let mut aligned = attribute_words_sweep(words, turns, &interp_flags);

    if config.sentence_smoothing {
        apply_sentence_smoothing(&mut aligned, config.smoothing_threshold);
    }

    aligned
}

/// O(W+T) two-pointer sweep implementing max-overlap / nearest-gap tagging.
fn attribute_words_sweep(
    words: &[Word],
    turns: &[SpeakerTurn],
    interp_flags: &[bool],
) -> Vec<WordAlignment> {
    let n = words.len();
    if n == 0 {
        return Vec::new();
    }
    if turns.is_empty() {
        return words
            .iter()
            .enumerate()
            .map(|(i, w)| make_alignment(w, None, w.confidence, interp_flags[i]))
            .collect();
    }

    // Sort turns by start (stable via original index) for the sweep.
    let mut turn_order: Vec<usize> = (0..turns.len()).collect();
    turn_order.sort_by(|&a, &b| {
        turns[a]
            .time
            .start
            .total_cmp(&turns[b].time.start)
            .then_with(|| a.cmp(&b))
    });

    // Process words in start-time order; write results at original indices.
    let mut word_order: Vec<usize> = (0..n).collect();
    word_order.sort_by(|&a, &b| {
        words[a]
            .time
            .start
            .total_cmp(&words[b].time.start)
            .then_with(|| a.cmp(&b))
    });

    // Pre-fill; every index is overwritten exactly once below.
    let mut out: Vec<WordAlignment> = words
        .iter()
        .enumerate()
        .map(|(i, w)| make_alignment(w, None, w.confidence, interp_flags[i]))
        .collect();
    let mut left = 0usize; // into turn_order
    // Best turn completely to the left of the current word (max end, tie-break).
    let mut best_left: Option<usize> = None;

    for &wi in &word_order {
        let word = &words[wi];

        // Advance past turns that end at or before the word start.
        while left < turn_order.len() {
            let ti = turn_order[left];
            if turns[ti].time.end <= word.time.start {
                // Update best_left: prefer larger end (closer), then tie-break.
                let take = match best_left {
                    None => true,
                    Some(cur) => {
                        let te = turns[ti].time.end;
                        let ce = turns[cur].time.end;
                        te > ce || (te == ce && better_turn(ti, cur, turns))
                    }
                };
                if take {
                    best_left = Some(ti);
                }
                left += 1;
            } else {
                break;
            }
        }

        let (speaker, conf) = attribute_one_sweep(word, turns, &turn_order, left, best_left);
        out[wi] = make_alignment(word, speaker, conf, interp_flags[wi]);
    }

    out
}

/// Detect a sentence-ending token via trailing `.`, `?`, or `!` (after common
/// closing quotes/brackets). Intentionally tiny — no external NLP.
fn ends_sentence(token: &str) -> bool {
    let trimmed = token.trim_end_matches(|c: char| {
        matches!(
            c,
            '"' | '\'' | '\u{201D}' | '\u{2019}' | ')' | ']' | '\u{00BB}'
        )
    });
    matches!(trimmed.chars().last(), Some('.' | '?' | '!'))
}

/// Relabel mid-sentence speaker changes when one speaker holds more than
/// `threshold` of the sentence's attributed words.
fn apply_sentence_smoothing(aligned: &mut [WordAlignment], threshold: f32) {
    let n = aligned.len();
    let mut start = 0usize;
    while start < n {
        let mut end = start;
        while end < n {
            let boundary = ends_sentence(&aligned[end].word);
            end += 1;
            if boundary {
                break;
            }
        }
        smooth_sentence_range(&mut aligned[start..end], threshold);
        start = end;
    }
}

fn smooth_sentence_range(words: &mut [WordAlignment], threshold: f32) {
    if words.len() < 2 {
        return;
    }

    // Count speakers among attributed words.
    let mut counts: Vec<(SpeakerId, usize)> = Vec::new();
    let mut attributed = 0usize;
    for w in words.iter() {
        if let Some(spk) = w.speaker {
            attributed += 1;
            if let Some(slot) = counts.iter_mut().find(|(s, _)| *s == spk) {
                slot.1 += 1;
            } else {
                counts.push((spk, 1));
            }
        }
    }
    if attributed == 0 || counts.len() <= 1 {
        return;
    }

    // Dominant: highest count, then smaller SpeakerId.
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.0.cmp(&b.0.0)));
    let (dom, dom_count) = counts[0];
    let share = dom_count as f32 / attributed as f32;
    if share > threshold {
        for w in words.iter_mut() {
            if w.speaker.is_some() {
                w.speaker = Some(dom);
            }
        }
    }
}

/// L2-normalized mean embedding for one speaker (opt-in attribution export).
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerEmbedding {
    pub speaker: SpeakerId,
    /// Unit-norm embedding vector (embedder dimension).
    pub embedding: Vec<f32>,
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
    /// Optional per-speaker embeddings (WhisperX `return_embeddings` pattern).
    /// `None` unless filled via [`WhoSaidWhat::with_speaker_embeddings`].
    pub speaker_embeddings: Option<Vec<SpeakerEmbedding>>,
}

/// Assemble [`SpeakerTurn::text`] for each turn from `aligned` words. A word
/// belongs to a turn when it was attributed to that turn's speaker and its
/// midpoint falls within the turn's span; words are joined in time order. Turns
/// with no words keep `text: None`.
///
/// Uses [`WordAnchor::Mid`] (historical behavior). See
/// [`fill_turn_text_with_config`] to choose start/mid/end.
pub fn fill_turn_text(turns: &[SpeakerTurn], aligned: &[WordAlignment]) -> Vec<SpeakerTurn> {
    fill_turn_text_with_config(turns, aligned, &AttributionConfig::default())
}

/// Like [`fill_turn_text`], but the word point tested for turn membership is
/// selected by `config.word_anchor`.
pub fn fill_turn_text_with_config(
    turns: &[SpeakerTurn],
    aligned: &[WordAlignment],
    config: &AttributionConfig,
) -> Vec<SpeakerTurn> {
    let anchor = config.word_anchor;
    turns
        .iter()
        .map(|turn| {
            let mut words: Vec<&WordAlignment> = aligned
                .iter()
                .filter(|w| {
                    let pt = anchor.point(&w.time);
                    w.speaker == Some(turn.speaker) && pt >= turn.time.start && pt < turn.time.end
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
                stable: turn.stable,
            }
        })
        .collect()
}

/// Join raw ASR `words` to diarization `turns`: attribute each word to a speaker
/// (overlap-region words go to the **dominant** speaker only — see
/// [`attribute_words`]), then fill each turn's text. Pure — no ASR, no I/O.
pub fn attribute_and_fill(words: &[Word], turns: &[SpeakerTurn]) -> WhoSaidWhat {
    attribute_and_fill_with_config(words, turns, &AttributionConfig::default())
}

/// Like [`attribute_and_fill`] with an explicit [`AttributionConfig`].
pub fn attribute_and_fill_with_config(
    words: &[Word],
    turns: &[SpeakerTurn],
    config: &AttributionConfig,
) -> WhoSaidWhat {
    let aligned = attribute_words_with_config(words, turns, config);
    let turns = fill_turn_text_with_config(turns, &aligned, config);
    WhoSaidWhat {
        words: aligned,
        turns,
        speaker_embeddings: None,
    }
}

impl WhoSaidWhat {
    /// Attach L2-normalized per-speaker embeddings (opt-in).
    ///
    /// Each input vector is re-normalized. Speakers are stored sorted by numeric
    /// id. Pass the output of [`mean_speaker_embeddings`] or any
    /// `(SpeakerId, Vec<f32>)` list from the diarization stage.
    pub fn with_speaker_embeddings(mut self, embeddings: &[(SpeakerId, Vec<f32>)]) -> Self {
        let mut out: Vec<SpeakerEmbedding> = embeddings
            .iter()
            .map(|(spk, emb)| {
                let mut v = emb.clone();
                crate::utils::l2_normalize(&mut v);
                SpeakerEmbedding {
                    speaker: *spk,
                    embedding: v,
                }
            })
            .collect();
        out.sort_by_key(|e| e.speaker.0);
        self.speaker_embeddings = Some(out);
        self
    }
}

/// Average and L2-normalize embeddings per speaker label.
///
/// Thin wrapper around [`mean_speaker_embeddings`] for attribution callers.
pub fn speaker_embeddings_from_segments(
    labels: &[SpeakerId],
    embeddings: &[Vec<f32>],
) -> Vec<(SpeakerId, Vec<f32>)> {
    mean_speaker_embeddings(labels, embeddings)
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
    who_said_what_with_config(
        turns,
        asr,
        samples,
        sample_rate,
        &AttributionConfig::default(),
    )
}

/// Like [`who_said_what`] with an explicit [`AttributionConfig`].
pub fn who_said_what_with_config(
    turns: &[SpeakerTurn],
    asr: &dyn Asr,
    samples: &[f32],
    sample_rate: SampleRate,
    config: &AttributionConfig,
) -> Result<WhoSaidWhat, AsrError> {
    let words = asr.transcribe(samples, sample_rate)?;
    Ok(attribute_and_fill_with_config(&words, turns, config))
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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
            stable: true,
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
        assert!(!out[0].interpolated);
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
        assert!(wsw.speaker_embeddings.is_none());
    }

    #[test]
    fn speaker_embeddings_opt_in_are_l2_normalized() {
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let wsw = attribute_and_fill(&[word("hi", 1.0, 1.5, 1.0)], &turns);
        let labels = [SpeakerId(0), SpeakerId(0), SpeakerId(1)];
        let embs = vec![vec![3.0, 0.0], vec![0.0, 4.0], vec![0.0, 2.0]];
        let means = speaker_embeddings_from_segments(&labels, &embs);
        let wsw = wsw.with_speaker_embeddings(&means);
        let se = wsw
            .speaker_embeddings
            .as_ref()
            .expect("embeddings attached");
        assert_eq!(se.len(), 2);
        for e in se {
            let n: f32 = e.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-5, "norm {n}");
        }
        // Deterministic.
        let again = attribute_and_fill(&[word("hi", 1.0, 1.5, 1.0)], &turns)
            .with_speaker_embeddings(&means);
        assert_eq!(wsw.speaker_embeddings, again.speaker_embeddings);
    }

    // --- timestamp interpolation ---

    #[test]
    fn interpolate_zero_duration_uses_neighbors() {
        let words = vec![
            word("a", 0.0, 1.0, 1.0),
            word("b", 0.0, 0.0, 1.0), // missing
            word("c", 2.0, 3.0, 1.0),
        ];
        let (fixed, flags) = interpolate_word_timestamps(&words);
        assert_eq!(fixed.len(), words.len());
        assert!(!flags[0] && flags[1] && !flags[2]);
        assert!((fixed[1].time.start - 1.0).abs() < 1e-9);
        assert!((fixed[1].time.end - 2.0).abs() < 1e-9);
        assert!(fixed[1].time.end > fixed[1].time.start);
    }

    #[test]
    fn interpolate_preserves_length_all_missing() {
        let words = vec![word("x", 0.0, 0.0, 1.0), word("y", 5.0, 5.0, 1.0)];
        let (fixed, flags) = interpolate_word_timestamps(&words);
        assert_eq!(fixed.len(), 2);
        assert!(flags.iter().all(|&f| f));
        assert!(fixed.iter().all(|w| w.time.end > w.time.start));
    }

    #[test]
    fn attribution_totality_with_missing_timestamps() {
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let words = vec![
            word("ok", 1.0, 2.0, 1.0),
            word("hole", 0.0, 0.0, 1.0),
            word("ok2", 6.0, 7.0, 1.0),
        ];
        let out = attribute_words(&words, &turns);
        assert_eq!(out.len(), words.len());
        assert!(out.iter().all(|w| w.speaker.is_some()));
        assert!(out[1].interpolated);
        assert!(!out[0].interpolated);
    }

    // --- sentence smoothing ---

    #[test]
    fn sentence_smoothing_relabels_mid_sentence_minority() {
        // "Hello world from me." — first three spk0, last spk1 → relabel to spk0.
        let turns = vec![turn(0, 0.0, 10.0), turn(1, 10.0, 20.0)];
        let words = vec![
            word("Hello", 0.0, 1.0, 1.0),
            word("world", 1.0, 2.0, 1.0),
            word("from", 2.0, 3.0, 1.0),
            word("me.", 11.0, 12.0, 1.0), // lands in spk1 by time
        ];
        let cfg = AttributionConfig {
            sentence_smoothing: true,
            interpolate_timestamps: false,
            ..AttributionConfig::default()
        };
        let out = attribute_words_with_config(&words, &turns, &cfg);
        assert!(out.iter().all(|w| w.speaker == Some(SpeakerId(0))));
    }

    #[test]
    fn sentence_smoothing_skips_boundary_between_sentences() {
        // Two sentences, different speakers — must not bleed across boundary.
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let words = vec![
            word("Hi.", 1.0, 2.0, 1.0),  // spk0
            word("Hey.", 6.0, 7.0, 1.0), // spk1
        ];
        let cfg = AttributionConfig {
            sentence_smoothing: true,
            interpolate_timestamps: false,
            ..AttributionConfig::default()
        };
        let out = attribute_words_with_config(&words, &turns, &cfg);
        assert_eq!(out[0].speaker, Some(SpeakerId(0)));
        assert_eq!(out[1].speaker, Some(SpeakerId(1)));
    }

    #[test]
    fn sentence_smoothing_respects_threshold() {
        // 2 vs 2 exactly at 50% with threshold 0.5 → no relabel (need share > 0.5).
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let words = vec![
            word("a", 1.0, 1.5, 1.0),
            word("b", 2.0, 2.5, 1.0),
            word("c", 6.0, 6.5, 1.0),
            word("d.", 7.0, 7.5, 1.0),
        ];
        let cfg = AttributionConfig {
            sentence_smoothing: true,
            smoothing_threshold: 0.5,
            interpolate_timestamps: false,
            ..AttributionConfig::default()
        };
        let out = attribute_words_with_config(&words, &turns, &cfg);
        assert_eq!(out[0].speaker, Some(SpeakerId(0)));
        assert_eq!(out[2].speaker, Some(SpeakerId(1)));
    }

    // --- word anchor ---

    #[test]
    fn word_anchor_start_places_word_in_earlier_turn() {
        // Word [4.5, 5.5], midpoint 5.0 is not < 5.0 so mid puts it outside turn0;
        // start 4.5 is inside turn0.
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let words = vec![word("x", 4.5, 5.5, 1.0)];
        let aligned = attribute_words(&words, &turns);
        // max-overlap: 0.5 each → tie → spk0
        assert_eq!(aligned[0].speaker, Some(SpeakerId(0)));

        let mid = fill_turn_text(&turns, &aligned);
        // mid point 5.0: not in [0,5), not placed in turn0 text
        assert!(mid[0].text.is_none());

        let cfg = AttributionConfig {
            word_anchor: WordAnchor::Start,
            interpolate_timestamps: false,
            ..AttributionConfig::default()
        };
        let start_fill = fill_turn_text_with_config(&turns, &aligned, &cfg);
        assert_eq!(start_fill[0].text.as_deref(), Some("x"));
    }

    #[test]
    fn word_anchor_end_places_word_in_later_turn() {
        let turns = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let words = vec![word("x", 4.5, 5.5, 1.0)];
        let aligned = attribute_words(&words, &turns);
        let cfg = AttributionConfig {
            word_anchor: WordAnchor::End,
            interpolate_timestamps: false,
            ..AttributionConfig::default()
        };
        let end_fill = fill_turn_text_with_config(&turns, &aligned, &cfg);
        // end 5.5 is in turn1, but speaker is spk0 — membership requires speaker match,
        // so no text in either turn.
        assert!(end_fill[0].text.is_none());
        assert!(end_fill[1].text.is_none());
    }

    // --- property: sweep == reference scan ---

    fn arb_time_range() -> impl Strategy<Value = TimeRange> {
        (0.0f64..50.0, 0.01f64..5.0).prop_map(|(start, dur)| TimeRange {
            start,
            end: start + dur,
        })
    }

    fn arb_word() -> impl Strategy<Value = Word> {
        (arb_time_range(), 0.0f32..1.0).prop_map(|(time, confidence)| Word {
            word: "w".into(),
            time,
            confidence,
        })
    }

    fn arb_turn() -> impl Strategy<Value = SpeakerTurn> {
        (0u32..8, arb_time_range()).prop_map(|(id, time)| SpeakerTurn {
            speaker: SpeakerId(id),
            time,
            text: None,
            stable: true,
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            ..ProptestConfig::default()
        })]

        /// Sweep join matches the historical O(W·T) max-overlap scan bit-for-bit
        /// on speaker id and confidence (no interpolation, no smoothing).
        #[test]
        fn sweep_matches_reference_scan(
            words in prop::collection::vec(arb_word(), 0..40),
            turns in prop::collection::vec(arb_turn(), 0..20),
        ) {
            let cfg = AttributionConfig {
                interpolate_timestamps: false,
                sentence_smoothing: false,
                ..AttributionConfig::default()
            };
            let got = attribute_words_with_config(&words, &turns, &cfg);

            prop_assert_eq!(got.len(), words.len());
            for (w, g) in words.iter().zip(got.iter()) {
                let (spk, conf) = attribute_one_reference(w, &turns);
                prop_assert_eq!(g.speaker, spk, "speaker mismatch for {:?}", w.time);
                prop_assert!(
                    (g.confidence - conf).abs() < 1e-5,
                    "conf mismatch: got {} want {} for {:?}",
                    g.confidence, conf, w.time
                );
                prop_assert!(!g.interpolated);
            }
        }

        /// Attribution is total: output length always equals input length.
        #[test]
        fn attribution_length_equals_input(
            words in prop::collection::vec(arb_word(), 0..30),
            turns in prop::collection::vec(arb_turn(), 0..15),
            // Sprinkle some zero-duration holes.
            holes in prop::collection::vec(0usize..30, 0..5),
        ) {
            let mut words = words;
            for h in holes {
                if h < words.len() {
                    words[h].time.end = words[h].time.start;
                }
            }
            let out = attribute_words(&words, &turns);
            prop_assert_eq!(out.len(), words.len());
            if !turns.is_empty() {
                prop_assert!(out.iter().all(|w| w.speaker.is_some()));
            }
        }
    }

    /// Unsorted turns still yield correct (reference-equivalent) tags.
    #[test]
    fn unsorted_turns_still_match_reference() {
        let turns = vec![
            turn(1, 5.0, 10.0),
            turn(0, 0.0, 5.0),
            turn(2, 2.0, 3.0), // nested/overlapping
        ];
        let words = vec![
            word("a", 1.0, 1.5, 1.0),
            word("b", 2.2, 2.8, 1.0),
            word("c", 6.0, 7.0, 1.0),
            word("d", 4.0, 6.0, 1.0),
        ];
        let cfg = AttributionConfig {
            interpolate_timestamps: false,
            ..AttributionConfig::default()
        };
        let got = attribute_words_with_config(&words, &turns, &cfg);
        for (w, g) in words.iter().zip(got.iter()) {
            let (spk, conf) = attribute_one_reference(w, &turns);
            assert_eq!(g.speaker, spk);
            assert!((g.confidence - conf).abs() < 1e-5);
        }
    }
}
