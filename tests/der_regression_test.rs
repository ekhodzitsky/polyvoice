//! DER regression test against committed `tests/der_baseline.json`.
//!
//! Uses the legacy v0.5 pipeline. Must stay within `tolerance` of the baseline
//! DER for each dataset. If a change legitimately improves DER, update the
//! baseline JSON — never silence the test.
//!
//! Run with:
//!   cargo test --test der_regression_test --features "onnx,download" -- --ignored

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
    #[serde(rename = "voxconverse_test_10files")]
    voxconverse_test_10files: DatasetBaseline,
    e2e_smoke: DatasetBaseline,
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

fn run_legacy_pipeline(wav_path: &Path, rttm_path: &Path) -> (f64, String) {
    let stem = wav_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    let (samples, sr_hz) = read_wav(wav_path).expect("WAV read failure");
    assert_eq!(sr_hz, 16000, "only 16 kHz WAVs supported");

    let registry = ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(Profile::Balanced)
        .expect("models");

    let embedding_dim = Profile::Balanced.embedding_dim();
    let extractor =
        FbankOnnxExtractor::new(&models.embedder_path, embedding_dim, 1).expect("embedder");
    let mut vad = SileroVad::new(Path::new("models/silero_vad.onnx"), 512).expect("vad");

    let config = DiarizationConfig::default();
    let vad_config = VadConfig::default();
    let pipeline = Pipeline::new(config, vad_config);

    let result = pipeline
        .run(&samples, &extractor, &mut vad)
        .expect("pipeline.run");

    let ref_turns = {
        let raw = parse_rttm_file(rttm_path).expect("parse rttm");
        let grouped = group_by_file(&raw);
        // AMI files use basename like EN2002a.Mix-Headset.wav but RTTM key is EN2002a
        let rttm_key = if stem.contains(".Mix-Headset") {
            stem.trim_end_matches(".Mix-Headset")
        } else {
            &stem
        };
        let segs: Vec<_> = grouped
            .get(rttm_key)
            .map(|v| v.iter().map(|s| (*s).clone()).collect())
            .unwrap_or_default();
        let (turns, _map) = to_speaker_turns(&segs);
        turns
    };

    let der = compute_der(&ref_turns, &result.turns, 0.25);
    (der.der, stem)
}

const SUBSET_10: &[&str] = &[
    "aepyx", "aggyz", "aiqwk", "aorju", "auzru", "bgvvt", "bidnq", "bjruf", "bmsyn", "bpzsc",
];

#[ignore = "requires cached ONNX bundle + wav/rttm files under data/voxconverse-test/"]
#[test]
fn der_regression_voxconverse_10_file_subset() {
    let baseline = load_baseline();
    let audio_dir = Path::new("data/voxconverse-test/audio");
    let rttm_dir = Path::new("data/voxconverse-test/rttm");

    let mut total_der = 0.0_f64;
    let mut count = 0_usize;

    for stem in SUBSET_10 {
        let wav_path = audio_dir.join(format!("{stem}.wav"));
        let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
        assert!(wav_path.is_file(), "WAV not found: {}", wav_path.display());
        assert!(
            rttm_path.is_file(),
            "RTTM not found: {}",
            rttm_path.display()
        );

        let (der, _stem) = run_legacy_pipeline(&wav_path, &rttm_path);
        println!("{stem}: DER={:.2}%", der * 100.0);
        total_der += der;
        count += 1;
    }

    assert!(count > 0, "no files processed");
    let avg_der = total_der / count as f64;
    println!("Average DER over {count} files: {:.2}%", avg_der * 100.0);

    let expected = baseline.voxconverse_test_10files.der_collar_0_25 / 100.0;
    let tolerance = baseline.voxconverse_test_10files.tolerance / 100.0;
    assert!(
        avg_der <= expected + tolerance,
        "DER regression: expected <= {:.2}%, got {:.2}% (baseline {:.2}% + tolerance {:.2}%)",
        (expected + tolerance) * 100.0,
        avg_der * 100.0,
        expected * 100.0,
        tolerance * 100.0,
    );
}

#[ignore = "requires cached ONNX bundle + wav/rttm files under tests/data/e2e-smoke/"]
#[test]
fn der_regression_e2e_smoke() {
    let baseline = load_baseline();
    let wav_path = Path::new("tests/data/e2e-smoke/audio/fuzfh.wav");
    let rttm_path = Path::new("tests/data/e2e-smoke/rttm/fuzfh.rttm");

    if !wav_path.is_file() {
        println!("e2e-smoke WAV not found — skipping");
        return;
    }

    let (der, stem) = run_legacy_pipeline(wav_path, rttm_path);
    println!("{stem}: DER={:.2}%", der * 100.0);

    let expected = baseline.e2e_smoke.der_collar_0_25 / 100.0;
    let tolerance = baseline.e2e_smoke.tolerance / 100.0;
    assert!(
        der <= expected + tolerance,
        "DER regression: expected <= {:.2}%, got {:.2}% (baseline {:.2}% + tolerance {:.2}%)",
        (expected + tolerance) * 100.0,
        der * 100.0,
        expected * 100.0,
        tolerance * 100.0,
    );
}

#[ignore = "requires cached ONNX bundle + wav/rttm files under data/ami-test-single/"]
#[test]
fn der_regression_ami_test_single() {
    let baseline = load_baseline();
    let audio_dir = Path::new("data/ami-test-single/audio");
    let rttm_dir = Path::new("data/ami-test-single/rttm");

    let wav_path = audio_dir.join("EN2002a.Mix-Headset.wav");
    let rttm_path = rttm_dir.join("EN2002a.Mix-Headset.rttm");
    let rttm_path_alt = rttm_dir.join("EN2002a.rttm");

    let wav_path = if wav_path.is_file() {
        wav_path
    } else {
        audio_dir.join("EN2002a.wav")
    };
    let rttm_path = if rttm_path.is_file() {
        rttm_path
    } else {
        rttm_path_alt
    };

    if !wav_path.is_file() {
        println!("AMI WAV not found — skipping");
        return;
    }

    let (der, stem) = run_legacy_pipeline(&wav_path, &rttm_path);
    println!("{stem}: DER={:.2}%", der * 100.0);

    let expected = baseline.ami_test_single.der_collar_0_25 / 100.0;
    let tolerance = baseline.ami_test_single.tolerance / 100.0;
    assert!(
        der <= expected + tolerance,
        "DER regression: expected <= {:.2}%, got {:.2}% (baseline {:.2}% + tolerance {:.2}%)",
        (expected + tolerance) * 100.0,
        der * 100.0,
        expected * 100.0,
        tolerance * 100.0,
    );
}
