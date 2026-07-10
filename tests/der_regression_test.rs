#![allow(deprecated)] // legacy embedding API; see polyvoice::embedder
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

/// Release-gate signal: when `POLYVOICE_REQUIRE_DATA=1` is set (the release gate
/// exports it), missing test data is a hard failure instead of a silent skip —
/// so a partial cache/download miss can never green-light a release without
/// actually running DER. Unset (local dev) keeps the soft-skip ergonomics.
fn require_data() -> bool {
    std::env::var("POLYVOICE_REQUIRE_DATA")
        .map(|v| v == "1")
        .unwrap_or(false)
}

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
    /// No-collar (collar=0) DER baseline — the headline metric, micro-averaged
    /// (frame-weighted) on multi-file sets. `None` (JSON null) = not yet
    /// measured → the no-collar gate stays inactive for that dataset.
    der_no_collar: Option<f64>,
    tolerance: f64,
}

/// Gate `measured` (a 0..1 ratio) against an optional percent baseline. Inactive
/// (prints the value to record) while the baseline is null in der_baseline.json.
fn assert_no_collar(dataset: &str, measured: f64, baseline_pct: Option<f64>, tolerance_pct: f64) {
    match baseline_pct {
        Some(expected_pct) => {
            let bound = (expected_pct + tolerance_pct) / 100.0;
            assert!(
                measured <= bound,
                "no-collar DER regression on {dataset}: expected <= {:.2}%, got {:.2}% (baseline {:.2}% + tolerance {:.2}%)",
                bound * 100.0,
                measured * 100.0,
                expected_pct,
                tolerance_pct,
            );
        }
        None => println!(
            "{dataset}: no-collar baseline not yet measured — record {:.2}% as der_no_collar in tests/der_baseline.json to activate the gate",
            measured * 100.0
        ),
    }
}

fn load_baseline() -> Baseline {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json");
    let raw = std::fs::read_to_string(&path).expect("read der_baseline.json");
    serde_json::from_str(&raw).expect("parse der_baseline.json")
}

fn run_legacy_pipeline(
    wav_path: &Path,
    rttm_path: &Path,
) -> (polyvoice::der::DerResult, polyvoice::der::DerResult, String) {
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
    let vad_path = registry.ensure("silero_vad").expect("silero_vad model");
    let mut vad = SileroVad::new(&vad_path, 512).expect("vad");

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

    // Same hypothesis scored at both collars: 0.25 for the historical gate,
    // 0 (no-collar) for the headline like-for-like metric.
    let der_collar = compute_der(&ref_turns, &result.turns, 0.25);
    let der_no_collar = compute_der(&ref_turns, &result.turns, 0.0);
    (der_collar, der_no_collar, stem)
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
    // Frame accumulators for the duration-weighted micro average (sum of error
    // frames / sum of reference frames) — an average of per-file ratios cannot
    // produce it, and micro is what speakrs/pyannote headline numbers use.
    let (mut nc_err, mut nc_ref) = (0_u64, 0_u64);
    let (mut c_err, mut c_ref) = (0_u64, 0_u64);

    for stem in SUBSET_10 {
        let wav_path = audio_dir.join(format!("{stem}.wav"));
        let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
        assert!(wav_path.is_file(), "WAV not found: {}", wav_path.display());
        assert!(
            rttm_path.is_file(),
            "RTTM not found: {}",
            rttm_path.display()
        );

        let (der_collar, der_no_collar, _stem) = run_legacy_pipeline(&wav_path, &rttm_path);
        println!(
            "{stem}: DER(collar 0.25)={:.2}% DER(no collar)={:.2}%",
            der_collar.der * 100.0,
            der_no_collar.der * 100.0
        );
        total_der += der_collar.der;
        count += 1;
        c_err +=
            der_collar.missed_frames + der_collar.false_alarm_frames + der_collar.confusion_frames;
        c_ref += der_collar.total_ref_frames;
        nc_err += der_no_collar.missed_frames
            + der_no_collar.false_alarm_frames
            + der_no_collar.confusion_frames;
        nc_ref += der_no_collar.total_ref_frames;
    }

    assert!(count > 0, "no files processed");
    assert!(nc_ref > 0 && c_ref > 0, "no reference frames scored");
    let avg_der = total_der / count as f64;
    let micro_collar = c_err as f64 / c_ref as f64;
    let micro_no_collar = nc_err as f64 / nc_ref as f64;
    println!(
        "Over {count} files: macro(collar 0.25)={:.2}% micro(collar 0.25)={:.2}% micro(no collar)={:.2}%",
        avg_der * 100.0,
        micro_collar * 100.0,
        micro_no_collar * 100.0
    );

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
    assert_no_collar(
        "voxconverse_test_10files (micro)",
        micro_no_collar,
        baseline.voxconverse_test_10files.der_no_collar,
        baseline.voxconverse_test_10files.tolerance,
    );
}

#[ignore = "requires cached ONNX bundle + wav/rttm files under tests/data/e2e-smoke/"]
#[test]
fn der_regression_e2e_smoke() {
    let baseline = load_baseline();
    let wav_path = Path::new("tests/data/e2e-smoke/audio/fuzfh.wav");
    let rttm_path = Path::new("tests/data/e2e-smoke/rttm/fuzfh.rttm");

    if !wav_path.is_file() {
        assert!(
            !require_data(),
            "release gate requires e2e-smoke data but it is missing: {}",
            wav_path.display()
        );
        println!("e2e-smoke WAV not found — skipping (set POLYVOICE_REQUIRE_DATA=1 to require it)");
        return;
    }

    let (der_collar, der_no_collar, stem) = run_legacy_pipeline(wav_path, rttm_path);
    println!(
        "{stem}: DER(collar 0.25)={:.2}% DER(no collar)={:.2}%",
        der_collar.der * 100.0,
        der_no_collar.der * 100.0
    );

    let expected = baseline.e2e_smoke.der_collar_0_25 / 100.0;
    let tolerance = baseline.e2e_smoke.tolerance / 100.0;
    assert!(
        der_collar.der <= expected + tolerance,
        "DER regression: expected <= {:.2}%, got {:.2}% (baseline {:.2}% + tolerance {:.2}%)",
        (expected + tolerance) * 100.0,
        der_collar.der * 100.0,
        expected * 100.0,
        tolerance * 100.0,
    );
    assert_no_collar(
        "e2e_smoke",
        der_no_collar.der,
        baseline.e2e_smoke.der_no_collar,
        baseline.e2e_smoke.tolerance,
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
        assert!(
            !require_data(),
            "release gate requires AMI test data but it is missing: {}",
            wav_path.display()
        );
        println!("AMI WAV not found — skipping (set POLYVOICE_REQUIRE_DATA=1 to require it)");
        return;
    }

    let (der_collar, der_no_collar, stem) = run_legacy_pipeline(&wav_path, &rttm_path);
    println!(
        "{stem}: DER(collar 0.25)={:.2}% DER(no collar)={:.2}%",
        der_collar.der * 100.0,
        der_no_collar.der * 100.0
    );

    let expected = baseline.ami_test_single.der_collar_0_25 / 100.0;
    let tolerance = baseline.ami_test_single.tolerance / 100.0;
    assert!(
        der_collar.der <= expected + tolerance,
        "DER regression: expected <= {:.2}%, got {:.2}% (baseline {:.2}% + tolerance {:.2}%)",
        (expected + tolerance) * 100.0,
        der_collar.der * 100.0,
        expected * 100.0,
        tolerance * 100.0,
    );
    assert_no_collar(
        "ami_test_single",
        der_no_collar.der,
        baseline.ami_test_single.der_no_collar,
        baseline.ami_test_single.tolerance,
    );
}
