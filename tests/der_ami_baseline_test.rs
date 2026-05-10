//! AMI DER baseline — long-running, ignored by default.
//!
//! Run manually after downloading models:
//!   cargo test --all-features --test der_ami_baseline_test -- --ignored --nocapture
//!
//! Full AMI meetings are 30–60 min; expect 10–30 min runtime on CPU.

#![cfg(all(feature = "onnx", feature = "download"))]

use polyvoice::der::compute_der;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::Pipeline;
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::types::{DiarizationConfig, Profile};
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
use polyvoice::{FbankOnnxExtractor, SileroVad};
use std::path::Path;

#[ignore = "requires ONNX models + ~30 min runtime"]
#[test]
fn ami_single_file_der_below_30_percent() {
    let wav_path = Path::new("data/ami-test-single/audio/EN2002a.Mix-Headset.wav");
    let rttm_path = Path::new("data/ami-test-single/rttm/EN2002a.Mix-Headset.rttm");

    let (samples, sr_hz) = read_wav(&wav_path).expect("WAV read");
    assert_eq!(sr_hz, 16000);

    let registry = ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(Profile::Balanced)
        .expect("models");

    let extractor =
        FbankOnnxExtractor::new(&models.embedder_path, Profile::Balanced.embedding_dim(), 1)
            .expect("embedder");
    let mut vad = SileroVad::new(&models.segmenter_path, 512).expect("vad");
    let pipeline = Pipeline::new(DiarizationConfig::default(), VadConfig::default());

    let result = pipeline.run(&samples, &extractor, &mut vad).expect("pipeline");

    let raw = parse_rttm_file(&rttm_path).expect("rttm");
    let grouped = group_by_file(&raw);
    let segs: Vec<_> = grouped
        .get("EN2002a")
        .map(|v| v.iter().map(|s| (*s).clone()).collect())
        .unwrap_or_default();
    let (ref_turns, _map) = to_speaker_turns(&segs);

    let der = compute_der(&ref_turns, &result.turns, 0.25);
    println!("AMI EN2002a: DER={:.2}%", der.der * 100.0);
    assert!(
        der.der < 0.30,
        "expected DER < 30%, got {:.2}%",
        der.der * 100.0
    );
}
