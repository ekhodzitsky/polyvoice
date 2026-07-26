//! Diarization Error Rate (DER) and word-level attribution metrics.
//!
//! Frame-based DER with forgiveness collar and optimal speaker mapping, plus
//! WDER (Word Diarization Error Rate) for who-said-what evaluation.

mod decompose;
mod frame;
mod wder;

pub use decompose::{DerDecomposition, SpeakerRecall, compute_der_decomposition};
pub use frame::{
    DerResult, compute_der, compute_der_from_rttm, compute_der_single_speaker_regions,
    compute_der_with_uem, parse_uem,
};
pub use wder::{WderResult, compute_wder};

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::frame::optimal_speaker_mapping;
    use super::*;
    use crate::types::{SpeakerId, SpeakerTurn, TimeRange};

    fn turn(speaker: u32, start: f64, end: f64) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(speaker),
            time: TimeRange { start, end },
            text: None,
            stable: true,
        }
    }

    fn w(word: &str, start: f64, end: f64, spk: Option<u32>) -> crate::types::WordAlignment {
        crate::types::WordAlignment {
            word: word.to_owned(),
            time: TimeRange { start, end },
            speaker: spk.map(SpeakerId),
            confidence: 1.0,
            interpolated: false,
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

    #[test]
    fn der_result_frame_counts_are_consistent() {
        // Two reference speakers, hypothesis covers only the first → real miss
        // frames, so the counts are non-trivial.
        let reference = vec![turn(0, 0.0, 3.0), turn(1, 3.0, 6.0)];
        let hypothesis = vec![turn(0, 0.0, 3.0)];
        let r = compute_der(&reference, &hypothesis, 0.0);
        assert!(
            r.total_ref_frames > 0,
            "expected non-empty reference frames"
        );
        // der == sum(error frames) / total_ref_frames
        let expected = (r.missed_frames + r.false_alarm_frames + r.confusion_frames) as f64
            / r.total_ref_frames as f64;
        assert!(
            (r.der - expected).abs() < 1e-9,
            "der {} != error-frames/ref-frames {expected}",
            r.der
        );
        // total_ref_frames are 10 ms frames, so * 0.01 == total_speech seconds.
        assert!(
            (r.total_ref_frames as f64 * 0.01 - r.total_speech).abs() < 1e-9,
            "frame count * 0.01 ({}) != total_speech ({})",
            r.total_ref_frames as f64 * 0.01,
            r.total_speech
        );
    }

    #[test]
    fn single_speaker_der_excludes_overlap_frames() {
        // ref: spk0 [0,4), spk1 [2,6) → [2,4) is a 2-speaker overlap region.
        let reference = vec![turn(0, 0.0, 4.0), turn(1, 2.0, 6.0)];
        // Empty hypothesis → everything in the scored subset is a miss.
        let hypothesis: Vec<SpeakerTurn> = vec![];
        let full = compute_der(&reference, &hypothesis, 0.0);
        let single = compute_der_single_speaker_regions(&reference, &hypothesis, 0.0);
        // Overlap frames contribute 2 ref speakers/frame to the headline metric but
        // are entirely excluded from the single-speaker metric.
        assert!(
            single.total_ref_frames < full.total_ref_frames,
            "overlap frames must be excluded: single={} full={}",
            single.total_ref_frames,
            full.total_ref_frames
        );
        // Single-speaker regions are [0,2) and [4,6) ≈ 400 frames at 10 ms.
        assert!(
            (380..=420).contains(&single.total_ref_frames),
            "expected ~400 single-speaker frames, got {}",
            single.total_ref_frames
        );
        // Still a full miss over the single-speaker subset.
        assert!(
            (single.miss_rate - 1.0).abs() < 1e-9,
            "miss={}",
            single.miss_rate
        );
    }

    #[test]
    fn single_speaker_der_ignores_overlap_mismatch() {
        // ref: spk0 [0,6) with spk1 also active on [4,6) → [4,6) is the overlap.
        let reference = vec![turn(0, 0.0, 6.0), turn(1, 4.0, 6.0)];
        // hyp: spk0 over the whole span. It is correct on the single-speaker region
        // [0,4) and only "wrong" (misses spk1) inside the excluded overlap [4,6).
        let hypothesis = vec![turn(0, 0.0, 6.0)];
        let single = compute_der_single_speaker_regions(&reference, &hypothesis, 0.0);
        assert!(
            single.der < 0.01,
            "single-speaker DER must ignore the overlap-region mismatch, got {single}"
        );
    }

    #[test]
    fn decomposition_splits_overlap_and_recall() {
        // ref: spk0 [0,6), spk1 [3,6) → [3,6) is overlap, [0,3) is single (spk0).
        let reference = vec![turn(0, 0.0, 6.0), turn(1, 3.0, 6.0)];
        // hyp: spk0 over the whole span — never recovers spk1.
        let hypothesis = vec![turn(0, 0.0, 6.0)];
        let d = compute_der_decomposition(&reference, &hypothesis, 0.0);

        // Total DER: spk1 missed across the overlap half → ~1/3.
        assert!((d.total.der - 1.0 / 3.0).abs() < 0.02, "total {}", d.total);
        // The single-speaker region [0,3) is perfectly diarized.
        assert!(d.single_speaker.der < 0.02, "single {}", d.single_speaker);
        // Overlap region [3,6): one of two speakers is missed every frame → ~0.5.
        assert!((d.overlap.der - 0.5).abs() < 0.02, "overlap {}", d.overlap);

        // Per-speaker recall: spk0 fully recovered, spk1 entirely missed.
        let r0 = d
            .per_speaker_recall
            .iter()
            .find(|s| s.speaker == 0)
            .expect("spk0 recall");
        let r1 = d
            .per_speaker_recall
            .iter()
            .find(|s| s.speaker == 1)
            .expect("spk1 recall");
        assert!((r0.recall - 1.0).abs() < 0.02, "spk0 recall {}", r0.recall);
        assert!(r1.recall < 0.02, "spk1 recall {}", r1.recall);
    }

    #[test]
    fn uem_excludes_out_of_scope_frames() {
        // ref: one speaker over [0,10); hyp empty → full miss.
        let reference = vec![turn(0, 0.0, 10.0)];
        let hypothesis: Vec<SpeakerTurn> = vec![];
        let full = compute_der(&reference, &hypothesis, 0.0);
        let scoped = compute_der_with_uem(
            &reference,
            &hypothesis,
            0.0,
            &[TimeRange {
                start: 0.0,
                end: 5.0,
            }],
        );
        // Only the [0,5) half is scored → ~half the reference frames count.
        assert!(
            scoped.total_ref_frames < full.total_ref_frames,
            "UEM must drop out-of-scope frames: scoped={} full={}",
            scoped.total_ref_frames,
            full.total_ref_frames
        );
        assert!(
            (480..=520).contains(&scoped.total_ref_frames),
            "expected ~500 scored frames, got {}",
            scoped.total_ref_frames
        );
        assert!((scoped.miss_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn uem_ignores_error_outside_scope() {
        // ref: spk0 [0,10). hyp: correct spk0 on [0,5), wrong speaker on [5,10).
        let reference = vec![turn(0, 0.0, 10.0)];
        let hypothesis = vec![turn(0, 0.0, 5.0), turn(1, 5.0, 10.0)];
        let full = compute_der(&reference, &hypothesis, 0.0);
        let scoped = compute_der_with_uem(
            &reference,
            &hypothesis,
            0.0,
            &[TimeRange {
                start: 0.0,
                end: 5.0,
            }],
        );
        assert!(
            full.der > 0.4,
            "headline DER should see the [5,10) error, got {full}"
        );
        assert!(
            scoped.der < 0.01,
            "UEM-scoped DER must ignore the out-of-scope error, got {scoped}"
        );
    }

    #[test]
    fn uem_full_scope_matches_no_uem() {
        let reference = vec![turn(0, 0.0, 3.0), turn(1, 3.0, 6.0)];
        let hypothesis = vec![turn(0, 0.0, 3.0)];
        let plain = compute_der(&reference, &hypothesis, 0.0);
        let scoped = compute_der_with_uem(
            &reference,
            &hypothesis,
            0.0,
            &[TimeRange {
                start: 0.0,
                end: 100.0,
            }],
        );
        assert_eq!(plain.total_ref_frames, scoped.total_ref_frames);
        assert!(
            (plain.der - scoped.der).abs() < 1e-12,
            "full UEM must equal no-UEM"
        );
    }

    #[test]
    fn uem_empty_scope_scores_nothing() {
        let reference = vec![turn(0, 0.0, 5.0)];
        let hypothesis = vec![turn(0, 0.0, 5.0)];
        let scoped = compute_der_with_uem(&reference, &hypothesis, 0.0, &[]);
        assert_eq!(scoped.total_ref_frames, 0);
        assert_eq!(scoped.der, 0.0);
    }

    #[test]
    fn parse_uem_reads_regions_and_skips_junk() {
        let text = "\
; a comment
# another comment

EN2002a 1 0.00 1234.56
EN2002a 1 1300.0 1400.0
fuzfh 1 0.5 25.9
bad line with too few
EN2002a 1 50.0 10.0
";
        let map = parse_uem(text);
        let en = map.get("EN2002a").expect("EN2002a present");
        // two valid regions (the 50.0->10.0 degenerate line is dropped)
        assert_eq!(en.len(), 2);
        assert!((en[0].start - 0.0).abs() < 1e-9 && (en[0].end - 1234.56).abs() < 1e-9);
        let fz = map.get("fuzfh").expect("fuzfh present");
        assert_eq!(fz.len(), 1);
        assert!(!map.contains_key("bad"));
    }

    #[test]
    fn optimal_mapping_beats_greedy_on_counterexample() {
        // Co-occurrence (hyp, ref): (0,0)=10, (0,1)=9, (1,0)=8.
        // Greedy picks 0->0 (10 correct) and leaves hyp 1 unmapped.
        // Optimal picks 0->1 (9) + 1->0 (8) = 17 correct.
        let mut ref_frames: Vec<Vec<u32>> = Vec::new();
        let mut hyp_frames: Vec<Vec<u32>> = Vec::new();
        for _ in 0..10 {
            ref_frames.push(vec![0]);
            hyp_frames.push(vec![0]);
        }
        for _ in 0..9 {
            ref_frames.push(vec![1]);
            hyp_frames.push(vec![0]);
        }
        for _ in 0..8 {
            ref_frames.push(vec![0]);
            hyp_frames.push(vec![1]);
        }
        let collar_mask = vec![false; ref_frames.len()];
        let mapping = optimal_speaker_mapping(&ref_frames, &hyp_frames, &collar_mask);
        assert_eq!(
            mapping.get(&0),
            Some(&1),
            "hyp 0 must map to ref 1 (optimal), not ref 0 (greedy)"
        );
        assert_eq!(mapping.get(&1), Some(&0), "hyp 1 must map to ref 0");
    }

    #[test]
    fn wder_perfect_match_is_zero() {
        let reference = vec![
            w("hello", 0.0, 0.5, Some(0)),
            w("world", 0.5, 1.0, Some(0)),
            w("hi", 1.0, 1.5, Some(1)),
        ];
        // Swapped absolute ids — optimal mapping should still yield WDER 0.
        let hypothesis = vec![
            w("hello", 0.0, 0.5, Some(7)),
            w("world", 0.5, 1.0, Some(7)),
            w("hi", 1.0, 1.5, Some(3)),
        ];
        let r = compute_wder(&reference, &hypothesis);
        assert_eq!(r.total_words, 3);
        assert_eq!(r.speaker_errors, 0);
        assert!((r.wder - 0.0).abs() < 1e-12, "got {r}");
    }

    #[test]
    fn wder_hand_crafted_one_of_four_wrong() {
        // Four words; hyp mislabels the third after identity mapping.
        let reference = vec![
            w("a", 0.0, 0.5, Some(0)),
            w("b", 0.5, 1.0, Some(0)),
            w("c", 1.0, 1.5, Some(1)),
            w("d", 1.5, 2.0, Some(1)),
        ];
        let hypothesis = vec![
            w("a", 0.0, 0.5, Some(0)),
            w("b", 0.5, 1.0, Some(0)),
            w("c", 1.0, 1.5, Some(0)), // wrong: should be 1
            w("d", 1.5, 2.0, Some(1)),
        ];
        let r = compute_wder(&reference, &hypothesis);
        assert_eq!(r.total_words, 4);
        assert_eq!(r.speaker_errors, 1);
        assert!((r.wder - 0.25).abs() < 1e-12, "got {r}");
    }

    #[test]
    fn wder_empty_reference_is_zero() {
        let r = compute_wder(&[], &[w("x", 0.0, 1.0, Some(0))]);
        assert_eq!(r.total_words, 0);
        assert_eq!(r.wder, 0.0);
    }

    #[test]
    fn wder_skips_unlabeled_reference_words() {
        let reference = vec![w("a", 0.0, 0.5, None), w("b", 0.5, 1.0, Some(0))];
        let hypothesis = vec![w("a", 0.0, 0.5, Some(0)), w("b", 0.5, 1.0, Some(0))];
        let r = compute_wder(&reference, &hypothesis);
        assert_eq!(r.total_words, 1);
        assert_eq!(r.speaker_errors, 0);
    }
}
