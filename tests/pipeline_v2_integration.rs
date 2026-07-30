#![allow(clippy::unwrap_used)]
//! Integration test for Pipeline v2 with ResNet34 + AHC.
//!
//! Ensures the v2 pipeline achieves DER < 10% on the e2e-smoke test audio.
//!
//! Run with:
//!   cargo test --test pipeline_v2_integration --features "onnx,segmentation,embedder,clusterer,resegmentation,download" -- --ignored --nocapture

#![cfg(all(
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
    feature = "download",
))]

mod common;

use polyvoice::der::compute_der;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::{Pipeline, PipelineConfig};
use polyvoice::types::{Profile, SampleRate};
use std::path::Path;

fn load_test_audio(wav_path: &Path) -> Vec<f32> {
    let (samples, sr) = polyvoice::wav::read_wav(wav_path).expect("read wav");
    assert_eq!(sr, 16000, "expected 16kHz mono");
    samples
}

fn load_ground_truth() -> Vec<polyvoice::types::SpeakerTurn> {
    common::load_ref_turns(Path::new("tests/data/e2e-smoke/rttm/fuzfh.rttm"), "fuzfh")
}

#[test]
#[ignore = "requires downloaded models"]
fn pipeline_v2_balanced_resnet34_ahc_der_under_10_percent() {
    let wav_path = Path::new("tests/data/e2e-smoke/audio/fuzfh.wav");
    if !common::require_wav(wav_path) {
        return;
    }
    let samples = load_test_audio(wav_path);
    let ground_truth = load_ground_truth();

    let registry = ModelRegistry::default().expect("model registry");

    let config = PipelineConfig {
        profile: Profile::Balanced,
        sample_rate: SampleRate::new(16000).unwrap(),
        resegment_overlap: false,
        ..PipelineConfig::default()
    };

    let pipeline = Pipeline::builder()
        .config(config)
        .profile(Profile::Balanced)
        .with_models_from(registry)
        .build()
        .expect("pipeline build");

    let result = pipeline
        .run(&samples, SampleRate::new(16000).unwrap())
        .expect("pipeline run");

    let der = compute_der(&ground_truth, &result.turns, 0.25);

    println!(
        "Pipeline v2 (Balanced / ResNet34 / AHC) DER: {:.2}%",
        der.der * 100.0
    );
    println!("  num_speakers: {}", result.num_speakers);
    println!("  num_turns: {}", result.turns.len());

    assert!(
        der.der < 0.10,
        "DER must be < 10%, got {:.2}%",
        der.der * 100.0
    );
}

#[test]
#[ignore = "requires downloaded models"]
fn pipeline_v2_mobile_resnet34_ahc_der_under_10_percent() {
    let wav_path = Path::new("tests/data/e2e-smoke/audio/fuzfh.wav");
    if !common::require_wav(wav_path) {
        return;
    }
    let samples = load_test_audio(wav_path);
    let ground_truth = load_ground_truth();

    let registry = ModelRegistry::default().expect("model registry");

    let config = PipelineConfig {
        profile: Profile::Mobile,
        sample_rate: SampleRate::new(16000).unwrap(),
        resegment_overlap: false,
        ..PipelineConfig::default()
    };

    let pipeline = Pipeline::builder()
        .config(config)
        .profile(Profile::Mobile)
        .with_models_from(registry)
        .build()
        .expect("pipeline build");

    let result = pipeline
        .run(&samples, SampleRate::new(16000).unwrap())
        .expect("pipeline run");

    let der = compute_der(&ground_truth, &result.turns, 0.25);

    println!(
        "Pipeline v2 (Mobile / ResNet34 / AHC) DER: {:.2}%",
        der.der * 100.0
    );
    println!("  num_speakers: {}", result.num_speakers);
    println!("  num_turns: {}", result.turns.len());

    assert!(
        der.der < 0.10,
        "DER must be < 10%, got {:.2}%",
        der.der * 100.0
    );
}
