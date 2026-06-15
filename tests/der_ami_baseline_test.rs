#![allow(deprecated)] // legacy embedding API; see polyvoice::embedder
//! AMI DER baseline — long-running, ignored by default.
//!
//! Run manually after downloading models:
//!   cargo test --all-features --test der_ami_baseline_test -- --ignored --nocapture
//!
//! Full AMI meetings are 30–60 min; expect 10–30 min runtime on CPU.
//!
//! The DER bound is read from `tests/der_baseline.json` (the single source of
//! truth) rather than hard-coded, so this test cannot drift from the recorded
//! baseline.

#![cfg(all(feature = "onnx", feature = "download"))]

use polyvoice::der::compute_der;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::Pipeline;
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::types::{DiarizationConfig, Profile};
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
use polyvoice::{FbankOnnxExtractor, SileroVad};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Baseline {
    ami_test_single: DatasetBaseline,
}

#[derive(Deserialize)]
struct DatasetBaseline {
    #[serde(rename = "der_collar_0_25")]
    der_collar_0_25: f64,
    tolerance: f64,
}

fn load_baseline() -> Baseline {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json");
    let raw = std::fs::read_to_string(&path).expect("read der_baseline.json");
    serde_json::from_str(&raw).expect("parse der_baseline.json")
}

#[ignore = "requires ONNX models + ~30 min runtime"]
#[test]
fn ami_single_file_der_within_baseline() {
    let wav_path = Path::new("data/ami-test-single/audio/EN2002a.Mix-Headset.wav");
    let rttm_path = Path::new("data/ami-test-single/rttm/EN2002a.Mix-Headset.rttm");

    let (samples, sr_hz) = read_wav(wav_path).expect("WAV read");
    assert_eq!(sr_hz, 16000);

    let registry = ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(Profile::Balanced)
        .expect("models");

    let extractor =
        FbankOnnxExtractor::new(&models.embedder_path, Profile::Balanced.embedding_dim(), 1)
            .expect("embedder");
    let vad_path = registry.ensure("silero_vad").expect("silero_vad model");
    let mut vad = SileroVad::new(&vad_path, 512).expect("vad");
    let pipeline = Pipeline::new(DiarizationConfig::default(), VadConfig::default());

    let result = pipeline
        .run(&samples, &extractor, &mut vad)
        .expect("pipeline");

    let raw = parse_rttm_file(rttm_path).expect("rttm");
    let grouped = group_by_file(&raw);
    let segs: Vec<_> = grouped
        .get("EN2002a")
        .map(|v| v.iter().map(|s| (*s).clone()).collect())
        .unwrap_or_default();
    let (ref_turns, _map) = to_speaker_turns(&segs);

    let der = compute_der(&ref_turns, &result.turns, 0.25);

    // Bound comes from tests/der_baseline.json ami_test_single (collar 0.25),
    // not a hard-coded threshold — keeps this test consistent with the gate.
    let baseline = load_baseline();
    let expected = baseline.ami_test_single.der_collar_0_25 / 100.0;
    let tolerance = baseline.ami_test_single.tolerance / 100.0;
    let bound = expected + tolerance;

    println!(
        "AMI EN2002a: DER={:.2}% (baseline {:.2}% + tol {:.2}% => bound {:.2}%)",
        der.der * 100.0,
        expected * 100.0,
        tolerance * 100.0,
        bound * 100.0
    );
    assert!(
        der.der <= bound,
        "DER regression: expected <= {:.2}%, got {:.2}% (baseline {:.2}% + tolerance {:.2}%)",
        bound * 100.0,
        der.der * 100.0,
        expected * 100.0,
        tolerance * 100.0,
    );
}
