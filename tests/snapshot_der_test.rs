//! Snapshot tests for DER computation output.

#![allow(clippy::unwrap_used)]

use polyvoice::der::compute_der;
use polyvoice::types::{SpeakerId, SpeakerTurn};

fn turn(start: f64, end: f64, speaker: u32) -> SpeakerTurn {
    SpeakerTurn {
        time: polyvoice::types::TimeRange { start, end },
        speaker: SpeakerId(speaker),
        text: None,
    }
}

#[test]
fn snapshot_der_perfect_match() {
    let turns = vec![turn(0.0, 1.0, 0), turn(1.0, 2.0, 1)];
    let der = compute_der(&turns, &turns, 0.25);
    insta::assert_snapshot!(format!("{:.6?}", der));
}

#[test]
fn snapshot_der_full_miss() {
    let ref_turns = vec![turn(0.0, 1.0, 0)];
    let hyp_turns = vec![turn(2.0, 3.0, 0)];
    let der = compute_der(&ref_turns, &hyp_turns, 0.25);
    insta::assert_snapshot!(format!("{:.6?}", der));
}
