//! Property tests for core type invariants.

use polyvoice::types::{Confidence, SampleRate, TimeRange};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        ..ProptestConfig::default()
    })]

    /// SampleRate constructor accepts all valid rates and rejects invalid ones.
    #[test]
    fn sample_rate_invariant(rate in 0u32..=200_000) {
        let result = SampleRate::new(rate);
        if (8000..=192_000).contains(&rate) {
            prop_assert!(result.is_some(), "valid rate {} should be accepted", rate);
        } else {
            prop_assert!(result.is_none(), "invalid rate {} should be rejected", rate);
        }
    }

    /// Confidence constructor accepts values in [0.0, 1.0] and rejects others.
    #[test]
    fn confidence_invariant(value in -1.0f32..=2.0) {
        let result = Confidence::new(value);
        if (0.0..=1.0).contains(&value) {
            prop_assert!(result.is_some(), "valid confidence {} should be accepted", value);
        } else {
            prop_assert!(result.is_none(), "invalid confidence {} should be rejected", value);
        }
    }

    /// TimeRange duration is non-negative for valid ranges.
    #[test]
    fn time_range_duration_non_negative(start in 0.0f64..=100.0, duration in 0.0f64..=100.0) {
        let end = start + duration;
        let tr = TimeRange { start, end };
        prop_assert!(
            tr.duration() >= 0.0,
            "duration should be non-negative for valid range [{}..{}], got {}",
            start, end, tr.duration()
        );
        prop_assert!(
            (tr.duration() - duration).abs() < 1e-9,
            "duration should equal end - start, expected {} got {}",
            duration, tr.duration()
        );
    }
}
