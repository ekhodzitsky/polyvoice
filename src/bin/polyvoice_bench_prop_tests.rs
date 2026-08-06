use super::*;
use proptest::prelude::*;

/// Synthetic DerResult: `errors` error frames over `ref_frames` (all miss).
fn synth(errors: u64, ref_frames: u64) -> DerResult {
    DerResult {
        der: errors as f64 / ref_frames as f64,
        miss_rate: errors as f64 / ref_frames as f64,
        false_alarm_rate: 0.0,
        confusion_rate: 0.0,
        total_speech: ref_frames as f64 * 0.01,
        total_ref_frames: ref_frames,
        missed_frames: errors,
        false_alarm_frames: 0,
        confusion_frames: 0,
    }
}

#[test]
fn aggregate_macro_diverges_from_micro_and_micro_is_frame_weighted() {
    // A tiny 1s file at 50% DER and a long 60s file at 1% DER: the mean of
    // ratios (macro) must NOT equal the frame-weighted micro, and micro
    // must equal summed error frames / summed reference frames exactly.
    let short = synth(50, 100);
    let long = synth(60, 6000);
    let agg = aggregate_der(&[(short, short), (long, long)]);
    assert!(
        (agg.collar_macro - 25.5).abs() < 1e-9,
        "{}",
        agg.collar_macro
    );
    let expected_micro = (50 + 60) as f64 / (100 + 6000) as f64 * 100.0;
    assert!((agg.collar_micro - expected_micro).abs() < 1e-9);
    assert!((agg.collar_macro - agg.collar_micro).abs() > 10.0);
    // Same inputs on both passes => identical aggregates per pass.
    assert_eq!(agg.collar_micro, agg.no_collar_micro);
}

#[test]
fn aggregate_no_collar_at_least_collar_on_boundary_errors() {
    use polyvoice::types::{SpeakerId, SpeakerTurn, TimeRange};
    let turn = |s: u32, a: f64, b: f64| SpeakerTurn {
        speaker: SpeakerId(s),
        time: TimeRange { start: a, end: b },
        text: None,
        stable: true,
    };
    // Hypothesis shifted 0.3s off every reference boundary: the collar
    // forgives part of that error, no-collar must not.
    let reference = vec![turn(0, 0.0, 10.0), turn(1, 12.0, 20.0)];
    let hypothesis = vec![turn(0, 0.3, 10.3), turn(1, 12.3, 20.3)];
    let collar = compute_der(&reference, &hypothesis, 0.25);
    let no_collar = compute_der(&reference, &hypothesis, 0.0);
    let agg = aggregate_der(&[(collar, no_collar)]);
    assert!(
        agg.no_collar_micro >= agg.collar_micro,
        "no-collar {} < collar {}",
        agg.no_collar_micro,
        agg.collar_micro
    );
    assert!(agg.no_collar_macro >= agg.collar_macro);
    assert!(agg.no_collar_micro > 0.0, "boundary errors must be scored");
}

proptest! {
    #[test]
    fn bench_args_parses_with_valid_args(
        profile in "(mobile|balanced|fast)",
        collar in 0.0f64..1.0f64,
        threshold in 0.0f32..1.0f32,
        max_files in 0usize..100usize,
    ) {
        let args = vec![
            "polyvoice-bench".to_string(),
            "/tmp/dataset".to_string(),
            "--profile".to_string(), profile,
            "--collar".to_string(), collar.to_string(),
            "--threshold".to_string(), threshold.to_string(),
            "--max-files".to_string(), max_files.to_string(),
        ];
        let result = Args::try_parse_from(&args);
        prop_assert!(result.is_ok());
    }

    #[test]
    fn profile_from_str_accepts_only_known_names(s in "[a-zA-Z0-9_-]{1,20}") {
        let result = s.parse::<Profile>();
        let lower = s.to_ascii_lowercase();
        if lower == "mobile" || lower == "balanced" || lower == "custom" {
            prop_assert!(result.is_ok());
        } else {
            prop_assert!(result.is_err());
        }
    }
}
