//! Lightweight word → speaker labeling for STT product consumers.
//!
//! Always available (no feature flag). Pure midpoint coverage on
//! [`SpeakerTurn`] ranges — no ASR trait, no models, no I/O.
//!
//! This is the thin join streaming/file STT stacks need after diarization
//! turns are known. The richer max-overlap join, sentence smoothing, and
//! turn-text fill live behind the opt-in `attribution` feature.
//!
//! ## Policies
//!
//! | Path | Typical call |
//! |------|----------------|
//! | Offline / file | `UncoveredPolicy::None`, `stable_only: false` |
//! | Streaming / live | `UncoveredPolicy::LastTurn`, optionally `stable_only: true` for UI |
//!
//! `stable_only` filters **covering** turns only. [`UncoveredPolicy::LastTurn`]
//! always uses the last turn in the slice (including provisional), matching
//! live servers that label the trailing “now” region before a turn finalizes.

use crate::types::{SpeakerId, SpeakerTurn, TimeRange, Word, WordAlignment};

/// What to do when a word's midpoint is not covered by any turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UncoveredPolicy {
    /// Leave the speaker unset (typical offline / file path).
    #[default]
    None,
    /// Use the last turn's speaker when history is non-empty.
    ///
    /// Streaming consumers use this for the trailing region at "now", where
    /// speech may not have produced a finalized turn yet. With an empty turn
    /// list every label stays unset.
    LastTurn,
}

/// Speaker of the first turn whose time range covers `t` (seconds), if any.
///
/// Coverage is inclusive on both ends ([`TimeRange::contains_instant`]). On
/// overlap the first covering turn in slice order wins.
pub fn speaker_at(turns: &[SpeakerTurn], t: f64) -> Option<SpeakerId> {
    turns
        .iter()
        .find(|turn| turn.time.contains_instant(t))
        .map(|turn| turn.speaker)
}

/// Like [`speaker_at`], but only turns with [`SpeakerTurn::stable`] `== true`
/// participate. Provisional streaming labels are ignored so product UIs can
/// wait for immutable speaker IDs.
pub fn speaker_at_stable(turns: &[SpeakerTurn], t: f64) -> Option<SpeakerId> {
    turns
        .iter()
        .find(|turn| turn.stable && turn.time.contains_instant(t))
        .map(|turn| turn.speaker)
}

/// Midpoint of a time range in seconds.
///
/// Thin alias of [`TimeRange::midpoint`] for call sites that already import
/// this module.
#[inline]
pub fn midpoint(time: &TimeRange) -> f64 {
    time.midpoint()
}

fn resolve_at(
    turns: &[SpeakerTurn],
    t: f64,
    stable_only: bool,
    policy: UncoveredPolicy,
) -> Option<SpeakerId> {
    let hit = if stable_only {
        speaker_at_stable(turns, t)
    } else {
        speaker_at(turns, t)
    };
    hit.or_else(|| match policy {
        UncoveredPolicy::None => None,
        // Last turn is intentional even when stable_only: streaming tails need
        // a label before the cache marks the speaker stable.
        UncoveredPolicy::LastTurn => turns.last().map(|turn| turn.speaker),
    })
}

/// Assign a speaker to each word interval by midpoint coverage.
///
/// Returns one `Option<SpeakerId>` per input range, in the same order.
///
/// * `stable_only` — when true, only stable turns may **cover** a midpoint.
/// * `policy` — [`UncoveredPolicy::LastTurn`] for streaming tails;
///   [`UncoveredPolicy::None`] for offline files.
pub fn assign_speakers_by_midpoint(
    word_times: impl IntoIterator<Item = TimeRange>,
    turns: &[SpeakerTurn],
    policy: UncoveredPolicy,
    stable_only: bool,
) -> Vec<Option<SpeakerId>> {
    word_times
        .into_iter()
        .map(|time| resolve_at(turns, time.midpoint(), stable_only, policy))
        .collect()
}

/// Label raw ASR [`Word`]s into [`WordAlignment`]s using midpoint coverage.
///
/// Confidence is `1.0` when a covering turn is found (or last-turn fallback
/// applied), else `0.0`. Does not interpolate timestamps — pass already-timed
/// words (or use the `attribution` feature for richer joins).
pub fn label_words(
    words: &[Word],
    turns: &[SpeakerTurn],
    policy: UncoveredPolicy,
    stable_only: bool,
) -> Vec<WordAlignment> {
    words
        .iter()
        .map(|w| {
            let speaker = resolve_at(turns, w.time.midpoint(), stable_only, policy);
            WordAlignment {
                word: w.word.clone(),
                time: w.time,
                speaker,
                confidence: if speaker.is_some() { 1.0 } else { 0.0 },
                interpolated: false,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(id: u32, start: f64, end: f64) -> SpeakerTurn {
        SpeakerTurn::new(SpeakerId(id), TimeRange { start, end })
    }

    fn turn_unstable(id: u32, start: f64, end: f64) -> SpeakerTurn {
        SpeakerTurn::with_stability(SpeakerId(id), TimeRange { start, end }, false)
    }

    #[test]
    fn speaker_at_first_covering_turn_wins() {
        let turns = vec![turn(0, 0.0, 2.0), turn(1, 1.5, 3.0)];
        assert_eq!(speaker_at(&turns, 1.7), Some(SpeakerId(0)));
        assert_eq!(speaker_at(&turns, 2.5), Some(SpeakerId(1)));
        assert_eq!(speaker_at(&turns, 4.0), None);
    }

    #[test]
    fn streaming_last_turn_fallback() {
        let turns = vec![turn(0, 0.0, 1.0), turn(1, 1.0, 2.0)];
        let times = [TimeRange {
            start: 2.5,
            end: 2.7,
        }];
        let labels = assign_speakers_by_midpoint(times, &turns, UncoveredPolicy::LastTurn, false);
        assert_eq!(labels, vec![Some(SpeakerId(1))]);
    }

    #[test]
    fn last_turn_fallback_ignores_stable_only_for_tail() {
        // Covering turns are filtered by stable_only, but LastTurn still uses
        // the slice tail (including provisional) for live labeling.
        let turns = vec![turn_unstable(7, 0.0, 1.0)];
        let times = [TimeRange {
            start: 5.0,
            end: 5.1,
        }];
        let labels = assign_speakers_by_midpoint(times, &turns, UncoveredPolicy::LastTurn, true);
        assert_eq!(labels, vec![Some(SpeakerId(7))]);
    }

    #[test]
    fn offline_none_leaves_uncovered_unset() {
        let turns = vec![turn(0, 0.0, 1.0)];
        let times = [TimeRange {
            start: 5.0,
            end: 5.2,
        }];
        let labels = assign_speakers_by_midpoint(times, &turns, UncoveredPolicy::None, false);
        assert_eq!(labels, vec![None]);
    }

    #[test]
    fn empty_turns_clear_all_with_last_turn_policy() {
        let times = [TimeRange {
            start: 0.0,
            end: 0.1,
        }];
        let labels = assign_speakers_by_midpoint(times, &[], UncoveredPolicy::LastTurn, false);
        assert_eq!(labels, vec![None]);
    }

    #[test]
    fn stable_only_skips_provisional() {
        let turns = vec![turn_unstable(0, 0.0, 2.0), turn(1, 1.0, 3.0)];
        assert_eq!(speaker_at_stable(&turns, 0.5), None);
        assert_eq!(speaker_at_stable(&turns, 1.5), Some(SpeakerId(1)));
    }

    #[test]
    fn label_words_midpoint() {
        let turns = vec![turn(0, 0.0, 1.0), turn(1, 1.0, 2.0)];
        let words = vec![
            Word {
                word: "hello".into(),
                time: TimeRange {
                    start: 0.1,
                    end: 0.3,
                },
                confidence: 0.9,
            },
            Word {
                word: "there".into(),
                time: TimeRange {
                    start: 1.2,
                    end: 1.4,
                },
                confidence: 0.8,
            },
        ];
        let out = label_words(&words, &turns, UncoveredPolicy::None, false);
        assert_eq!(out[0].speaker, Some(SpeakerId(0)));
        assert_eq!(out[1].speaker, Some(SpeakerId(1)));
        assert!((out[0].confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn midpoint_matches_time_range() {
        let tr = TimeRange {
            start: 1.0,
            end: 3.0,
        };
        assert!((midpoint(&tr) - 2.0).abs() < f64::EPSILON);
        assert!((tr.midpoint() - 2.0).abs() < f64::EPSILON);
    }
}
