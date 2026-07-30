//! DER regression test against committed `tests/der_baseline.json`.
//!
//! Uses the legacy v0.5 pipeline. Must stay within `tolerance` of the baseline
//! DER for each dataset. If a change legitimately improves DER, update the
//! baseline JSON — never silence the test.
//!
//! Run with:
//!   cargo test --test der_regression_test --features "onnx,download" -- --ignored

#![cfg(all(feature = "onnx", feature = "download"))]

mod common;

use polyvoice::SileroVad;
use polyvoice::der::compute_der;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::LegacyPipeline;
use polyvoice::types::DiarizationConfig;
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
use std::path::Path;

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

    let extractor = common::balanced_onnx_extractor();
    let registry = ModelRegistry::default().expect("registry");
    let vad_path = registry.ensure("silero_vad").expect("silero_vad model");
    let mut vad = SileroVad::new(&vad_path, 512).expect("vad");

    let config = DiarizationConfig::default();
    let vad_config = VadConfig::default();
    let pipeline = LegacyPipeline::new(config, vad_config);

    let result = pipeline
        .run(&samples, &extractor, &mut vad)
        .expect("pipeline.run");

    let ref_turns = common::load_ref_turns(rttm_path, &stem);

    // Same hypothesis scored at both collars: 0.25 for the historical gate,
    // 0 (no-collar) for the headline like-for-like metric.
    let der_collar = compute_der(&ref_turns, &result.turns, 0.25);
    let der_no_collar = compute_der(&ref_turns, &result.turns, 0.0);
    (der_collar, der_no_collar, stem)
}

#[ignore = "requires downloaded models and dataset"]
#[test]
fn der_regression_voxconverse_10_file_subset() {
    let baseline = common::load_baseline(&common::der_baseline_path());
    let audio_dir = Path::new("data/voxconverse-test/audio");
    let rttm_dir = Path::new("data/voxconverse-test/rttm");
    if !common::require_wav(audio_dir) {
        return;
    }

    let mut total_der = 0.0_f64;
    let mut count = 0_usize;
    // Frame accumulators for the duration-weighted micro average (sum of error
    // frames / sum of reference frames) — an average of per-file ratios cannot
    // produce it, and micro is what speakrs/pyannote headline numbers use.
    let (mut nc_err, mut nc_ref) = (0_u64, 0_u64);
    let (mut c_err, mut c_ref) = (0_u64, 0_u64);

    for stem in common::VOXCONVERSE_SUBSET_10 {
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

    common::gate_against_baseline(
        "voxconverse_test_10files",
        avg_der,
        &baseline.voxconverse_test_10files,
    );
    common::assert_no_collar(
        "voxconverse_test_10files (micro)",
        micro_no_collar,
        &baseline.voxconverse_test_10files,
    );
}

#[ignore = "requires downloaded models"]
#[test]
fn der_regression_e2e_smoke() {
    let baseline = common::load_baseline(&common::der_baseline_path());
    let wav_path = Path::new("tests/data/e2e-smoke/audio/fuzfh.wav");
    let rttm_path = Path::new("tests/data/e2e-smoke/rttm/fuzfh.rttm");

    if !common::require_wav(wav_path) {
        return;
    }

    let (der_collar, der_no_collar, stem) = run_legacy_pipeline(wav_path, rttm_path);
    println!(
        "{stem}: DER(collar 0.25)={:.2}% DER(no collar)={:.2}%",
        der_collar.der * 100.0,
        der_no_collar.der * 100.0
    );

    common::gate_against_baseline("e2e_smoke", der_collar.der, &baseline.e2e_smoke);
    common::assert_no_collar("e2e_smoke", der_no_collar.der, &baseline.e2e_smoke);
}

#[ignore = "requires downloaded models and dataset"]
#[test]
fn der_regression_ami_test_single() {
    let baseline = common::load_baseline(&common::der_baseline_path());
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

    if !common::require_wav(&wav_path) {
        return;
    }

    let (der_collar, der_no_collar, stem) = run_legacy_pipeline(&wav_path, &rttm_path);
    println!(
        "{stem}: DER(collar 0.25)={:.2}% DER(no collar)={:.2}%",
        der_collar.der * 100.0,
        der_no_collar.der * 100.0
    );

    common::gate_against_baseline("ami_test_single", der_collar.der, &baseline.ami_test_single);
    common::assert_no_collar(
        "ami_test_single",
        der_no_collar.der,
        &baseline.ami_test_single,
    );
}
