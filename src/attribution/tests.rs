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
    let again =
        attribute_and_fill(&[word("hi", 1.0, 1.5, 1.0)], &turns).with_speaker_embeddings(&means);
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

// --- interval helpers ---

#[test]
fn gap_is_zero_for_overlapping_intervals() {
    let a = TimeRange {
        start: 0.0,
        end: 2.0,
    };
    let b = TimeRange {
        start: 1.0,
        end: 3.0,
    };
    assert_eq!(gap(&a, &b), 0.0);
    assert_eq!(gap(&b, &a), 0.0);
    // Containment also counts as overlap.
    let inner = TimeRange {
        start: 0.5,
        end: 1.0,
    };
    assert_eq!(gap(&a, &inner), 0.0);
    // Disjoint intervals keep the true gap, in either direction.
    let later = TimeRange {
        start: 5.0,
        end: 6.0,
    };
    assert!((gap(&a, &later) - 3.0).abs() < 1e-12);
    assert!((gap(&later, &a) - 3.0).abs() < 1e-12);
}

#[test]
fn sentence_smoothing_noop_for_single_speaker_sentence() {
    // A whole sentence attributed to one speaker: nothing to relabel.
    let turns = vec![turn(0, 0.0, 10.0)];
    let words = vec![word("all", 1.0, 1.5, 1.0), word("mine.", 2.0, 2.5, 1.0)];
    let cfg = AttributionConfig {
        sentence_smoothing: true,
        interpolate_timestamps: false,
        ..AttributionConfig::default()
    };
    let out = attribute_words_with_config(&words, &turns, &cfg);
    assert!(out.iter().all(|w| w.speaker == Some(SpeakerId(0))));
}

#[test]
fn sentence_smoothing_noop_without_attributed_words() {
    // No turns → no attributed speakers; smoothing must leave words alone.
    let words = vec![word("nobody", 1.0, 1.5, 0.9), word("here.", 2.0, 2.5, 0.8)];
    let cfg = AttributionConfig {
        sentence_smoothing: true,
        interpolate_timestamps: false,
        ..AttributionConfig::default()
    };
    let out = attribute_words_with_config(&words, &[], &cfg);
    assert!(out.iter().all(|w| w.speaker.is_none()));
    assert!((out[0].confidence - 0.9).abs() < 1e-6);
}
