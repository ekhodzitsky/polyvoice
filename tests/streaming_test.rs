//! Integration test for the streaming pipeline: drives feed()/flush() over a
//! many-small-chunk stream (microphone-style) and asserts the cumulative
//! turns() contract end-to-end.
//!
//! Regression guard: on the pre-fix streaming bug (turns() stayed empty while
//! per-call returns were non-empty), the `turns() == concatenation` assertion
//! below fails — this test would have caught it.

mod common;

use polyvoice::streaming::StreamingPipeline;
use polyvoice::types::SpeakerTurn;
use polyvoice::{DiarizationConfig, DummyExtractor, EnergyVad, VadConfig};
use std::collections::HashSet;

fn loud_samples(seconds: f32) -> Vec<f32> {
    let n = (seconds * 16000.0) as usize;
    (0..n)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect()
}

fn assert_cumulative_contract(
    turns: &[SpeakerTurn],
    expected: &[SpeakerTurn],
    num_speakers: usize,
) {
    assert_eq!(
        turns, expected,
        "turns() must equal the in-order concatenation of every feed()/flush() return"
    );
    assert!(!turns.is_empty(), "stream with speech must emit turns");
    assert!(
        turns.windows(2).all(|w| w[0].time.start <= w[1].time.start),
        "turns() must be monotonic by start time"
    );
    let distinct: HashSet<u32> = turns.iter().map(|t| t.speaker.0).collect();
    assert!(
        num_speakers >= distinct.len(),
        "num_speakers() ({num_speakers}) must cover the {} distinct ids in turns()",
        distinct.len()
    );
}

#[test]
fn feed_flush_round_trip_over_small_chunks() {
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    let mut pipeline = StreamingPipeline::new(
        vad,
        extractor,
        DiarizationConfig::default(),
        VadConfig::default(),
    )
    .expect("streaming pipeline");

    // loud / silent / loud — fed in 320-sample chunks (sub-frame sized, so the
    // internal VAD buffering across feed() boundaries is exercised).
    let mut stream = loud_samples(5.0);
    stream.extend(std::iter::repeat_n(0.0f32, 16000));
    stream.extend(loud_samples(5.0));

    let mut expected: Vec<SpeakerTurn> = Vec::new();
    for chunk in stream.chunks(320) {
        expected.extend(pipeline.feed(chunk).expect("feed"));
    }
    expected.extend(pipeline.flush().expect("flush"));

    assert_cumulative_contract(pipeline.turns(), &expected, pipeline.num_speakers());
}

/// Same contract through the real ONNX embedder (higher fidelity, model-gated).
#[cfg(all(feature = "onnx", feature = "download"))]
#[test]
#[ignore = "requires downloaded models"]
fn feed_flush_round_trip_with_real_embedder() {
    let extractor = common::balanced_onnx_extractor();

    let vad = EnergyVad::new(-40.0, 16000, 512);
    let mut pipeline = StreamingPipeline::new(
        vad,
        extractor,
        DiarizationConfig::default(),
        VadConfig::default(),
    )
    .expect("streaming pipeline");

    let stream = loud_samples(6.0);
    let mut expected: Vec<SpeakerTurn> = Vec::new();
    for chunk in stream.chunks(320) {
        expected.extend(pipeline.feed(chunk).expect("feed"));
    }
    expected.extend(pipeline.flush().expect("flush"));

    assert_cumulative_contract(pipeline.turns(), &expected, pipeline.num_speakers());
}
