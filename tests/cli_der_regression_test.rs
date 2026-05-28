//! DER regression test for the CLI using pipeline v2 (default as of v0.6.8+).
//!
//! Runs `polyvoice diarize` via `cargo run` and asserts DER stays within
//! tolerance of the v2 baseline. This prevents a repeat of the 0.6.1 incident
//! where pipeline v2 shipped as default without long-form audio validation.
//!
//! Run with:
//!   cargo test --test cli_der_regression_test --features "cli,download" -- --ignored

#![cfg(all(feature = "cli", feature = "download"))]

use polyvoice::der::compute_der;
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Baseline {
    #[serde(rename = "v2_e2e_smoke")]
    v2_e2e_smoke: DatasetBaseline,
    #[serde(rename = "hybrid_ami_test_single")]
    hybrid_ami_test_single: DatasetBaseline,
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

/// Run CLI `polyvoice diarize` on a WAV and return (DER, stem).
fn run_cli_diarize(wav_path: &Path, rttm_path: &Path) -> (f64, String) {
    let stem = wav_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    let output_rttm = tempfile::NamedTempFile::with_suffix(".rttm").expect("create temp rttm");
    let output_path = output_rttm.path().to_path_buf();

    let mut cmd = std::process::Command::new("cargo");
    cmd.args([
        "run",
        "--quiet",
        "--features",
        "cli",
        "--bin",
        "polyvoice",
        "--",
        "diarize",
        wav_path.to_str().expect("wav path is valid utf-8"),
        "--profile",
        "balanced",
        "--v2",
        "--output",
        output_path.to_str().expect("output path is valid utf-8"),
    ]);

    let output = cmd.output().expect("spawn cargo run");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("CLI diarize failed for {stem}: {stderr}");
    }

    let hyp_turns = {
        let raw = parse_rttm_file(&output_path).expect("parse CLI output rttm");
        let grouped = group_by_file(&raw);
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

    let ref_turns = {
        let raw = parse_rttm_file(rttm_path).expect("parse ground-truth rttm");
        let grouped = group_by_file(&raw);
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

    let der = compute_der(&ref_turns, &hyp_turns, 0.25);
    (der.der, stem)
}

#[ignore = "requires cached ONNX bundle + tests/data/e2e-smoke/"]
#[test]
fn cli_der_regression_v2_e2e_smoke() {
    let baseline = load_baseline();
    let wav_path = Path::new("tests/data/e2e-smoke/audio/fuzfh.wav");
    let rttm_path = Path::new("tests/data/e2e-smoke/rttm/fuzfh.rttm");

    if !wav_path.is_file() {
        println!("e2e-smoke WAV not found — skipping");
        return;
    }

    let (der, stem) = run_cli_diarize(wav_path, rttm_path);
    println!("{stem}: DER={:.2}%", der * 100.0);

    let expected = baseline.v2_e2e_smoke.der_collar_0_25 / 100.0;
    let tolerance = baseline.v2_e2e_smoke.tolerance / 100.0;
    assert!(
        der <= expected + tolerance,
        "DER regression: expected <= {:.2}%, got {:.2}% (baseline {:.2}% + tolerance {:.2}%)",
        (expected + tolerance) * 100.0,
        der * 100.0,
        expected * 100.0,
        tolerance * 100.0,
    );
}

#[ignore = "requires cached ONNX bundle + data/ami-test-single/"]
#[test]
fn cli_der_regression_v2_ami_single() {
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

    let (der, stem) = run_cli_diarize(&wav_path, &rttm_path);
    println!("{stem}: DER={:.2}%", der * 100.0);

    let expected = baseline.hybrid_ami_test_single.der_collar_0_25 / 100.0;
    let tolerance = baseline.hybrid_ami_test_single.tolerance / 100.0;
    assert!(
        der <= expected + tolerance,
        "DER regression: expected <= {:.2}%, got {:.2}% (baseline {:.2}% + tolerance {:.2}%)",
        (expected + tolerance) * 100.0,
        der * 100.0,
        expected * 100.0,
        tolerance * 100.0,
    );
}
