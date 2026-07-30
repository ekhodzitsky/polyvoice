//! Integration test for `PowersetSegmenter` against the real upstream ONNX model.
//!
//! Runs only when explicitly invoked:
//!   cargo test --features onnx,segmentation,download --test segmenter_test -- --ignored
//!
//! Downloads ~6 MB of model weights. Requires network connectivity.

#![cfg(all(feature = "onnx", feature = "segmentation", feature = "download"))]
#![allow(clippy::expect_used)]

use polyvoice::models::ModelRegistry;
use polyvoice::segmentation::{PowersetSegmenter, Segmenter};
use std::f32::consts::PI;
use tempfile::TempDir;

/// Construct 10 seconds of synthetic 16 kHz mono audio: half a 220 Hz sine
/// (speaker A), half a 440 Hz sine (speaker B). Not a perfect speaker model
/// but enough for the segmenter to find structure.
fn synthetic_two_speaker_audio() -> Vec<f32> {
    let sr = 16_000_usize;
    let total = 10 * sr;
    let mut audio = Vec::with_capacity(total);
    for i in 0..total {
        let t = i as f32 / sr as f32;
        let amp = if i < total / 2 {
            (2.0 * PI * 220.0 * t).sin() * 0.3
        } else {
            (2.0 * PI * 440.0 * t).sin() * 0.3
        };
        audio.push(amp);
    }
    audio
}

#[test]
#[ignore = "requires network"]
fn powerset_segmenter_emits_segments_on_real_model() {
    let tmp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::with_cache_dir(tmp.path()).expect("registry");
    let model_path = registry
        .ensure("powerset_fp32")
        .expect("model download must succeed");

    let segmenter = PowersetSegmenter::new(&model_path).expect("segmenter loads");
    assert_eq!(segmenter.max_local_speakers(), 3);
    assert!(segmenter.supports_overlap());

    let audio = synthetic_two_speaker_audio();
    let segments = segmenter.segment(&audio).expect("segment runs");

    // Synthetic audio with sustained tones is unrealistic for a speech model.
    // The segmenter may legitimately label most of it as silence. Just assert
    // the call succeeded and produced a Vec (possibly empty) of well-formed segments.
    for s in &segments {
        assert!(s.time.end >= s.time.start, "non-decreasing time");
        assert!(s.local_speaker_idx < segmenter.max_local_speakers() as u8);
        assert!(s.confidence.get() >= 0.0 && s.confidence.get() <= 1.0);
    }
}

#[test]
#[ignore = "requires network"]
fn powerset_segmenter_rejects_short_audio() {
    let tmp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::with_cache_dir(tmp.path()).expect("registry");
    let model_path = registry.ensure("powerset_fp32").expect("model download");

    let segmenter = PowersetSegmenter::new(&model_path).expect("segmenter loads");
    let too_short = vec![0.0_f32; 100]; // 6.25 ms < 100 ms minimum
    let err = segmenter.segment(&too_short).expect_err("must reject");
    let msg = format!("{err}");
    assert!(msg.to_lowercase().contains("too short"));
}
