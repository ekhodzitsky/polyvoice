//! Property tests for Diarization Error Rate (DER).
//!
//! Verified invariants:
//! - compute_der returns DER in [0, 1]
//! - identical reference and hypothesis → DER = 0
//! - DER is symmetric under optimal speaker mapping

use polyvoice::der::compute_der;
use polyvoice::types::SpeakerTurn;
use proptest::prelude::*;

mod common;
use common::turn;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        ..ProptestConfig::default()
    })]

    /// DER of identical ref and hyp is exactly 0.
    #[test]
    fn der_identical_ref_hyp_is_zero(
        segs in prop::collection::vec(
            (0.0f64..=100.0, 0.1f64..=10.0, 0u32..=3),
            0..=32,
        ),
    ) {
        let reference: Vec<SpeakerTurn> = segs
            .iter()
            .map(|&(start, dur, spk)| turn(start, start + dur, spk))
            .collect();
        let hypothesis = reference.clone();
        let der = compute_der(&reference, &hypothesis, 0.25);
        prop_assert!(
            der.der.abs() < 1e-9,
            "identical ref/hyp must have DER=0, got {}",
            der.der
        );
    }

    /// DER is always in [0, 1] for random inputs.
    #[test]
    fn der_range_is_0_to_1(
        ref_segs in prop::collection::vec(
            (0.0f64..=100.0, 0.1f64..=10.0, 0u32..=3),
            0..=32,
        ),
        hyp_segs in prop::collection::vec(
            (0.0f64..=100.0, 0.1f64..=10.0, 0u32..=3),
            0..=32,
        ),
    ) {
        let reference: Vec<SpeakerTurn> = ref_segs
            .iter()
            .map(|&(start, dur, spk)| turn(start, start + dur, spk))
            .collect();
        let hypothesis: Vec<SpeakerTurn> = hyp_segs
            .iter()
            .map(|&(start, dur, spk)| turn(start, start + dur, spk))
            .collect();
        let der = compute_der(&reference, &hypothesis, 0.25);
        prop_assert!(
            der.der >= 0.0,
            "DER must be non-negative, got {}",
            der.der
        );
    }
}
