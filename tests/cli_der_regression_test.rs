//! DER regression test for the CLI using pipeline v2 (default as of v0.6.8+).
//!
//! Runs the `polyvoice diarize` binary built alongside the tests (assert_cmd)
//! and asserts DER stays within tolerance of the v2 baseline. This prevents a
//! repeat of the 0.6.1 incident where pipeline v2 shipped as default without
//! long-form audio validation.
//!
//! Run with:
//!   cargo test --test cli_der_regression_test --features "cli,download" -- --ignored

#![cfg(all(feature = "cli", feature = "download"))]

mod common;

use polyvoice::der::{DerDecomposition, compute_der_decomposition};
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use std::path::Path;

/// Run CLI `polyvoice diarize --v2` and return
/// (DER decomposition at collar 0.25, at collar 0, num_speakers, stem).
///
/// Uses the binary cargo built for this test invocation (assert_cmd), so the
/// features come from the outer `cargo test --features ...` call — no nested
/// `cargo run` rebuild or target-lock wait inside the test.
fn run_cli_diarize(
    wav_path: &Path,
    rttm_path: &Path,
) -> (DerDecomposition, DerDecomposition, usize, String) {
    let stem = wav_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    let output_rttm = tempfile::NamedTempFile::with_suffix(".rttm").expect("create temp rttm");
    let output_path = output_rttm.path().to_path_buf();

    let mut cmd = common::polyvoice_cmd();
    cmd.args([
        "diarize",
        wav_path.to_str().expect("wav path is valid utf-8"),
        "--profile",
        "balanced",
        // Default path is v2 + VBx (0.11+). `--v2` stays accepted for scripts.
        "--v2",
        "--output",
        output_path.to_str().expect("output path is valid utf-8"),
    ]);
    // Prefer env (release-check exports it); else the checked-in fixtures so
    // local `cargo test -- --ignored` exercises the real default clusterer.
    let fixture_plda = common::vbx_plda_fixture_dir();
    if let Ok(dir) = std::env::var("POLYVOICE_VBX_PLDA_DIR") {
        cmd.args(["--vbx-plda-dir", &dir]);
    } else if fixture_plda.join("plda_transform.npy").is_file() {
        cmd.args(["--vbx-plda-dir", fixture_plda.to_str().expect("utf-8 path")]);
    }

    let output = cmd.output().expect("spawn polyvoice diarize");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("CLI diarize failed for {stem}: {stderr}");
    }

    let hyp_turns = {
        let raw = parse_rttm_file(&output_path).expect("parse CLI output rttm");
        let grouped = group_by_file(&raw);
        // The CLI output RTTM holds a single file's segments, but writes the file
        // id as the input stem ("EN2002a.Mix-Headset"), which differs from the ref
        // key ("EN2002a") — so collect every segment regardless of id.
        let segs: Vec<_> = grouped
            .values()
            .flat_map(|v| v.iter().map(|s| (*s).clone()))
            .collect();
        let (turns, _map) = to_speaker_turns(&segs);
        turns
    };

    let ref_turns = common::load_ref_turns(rttm_path, &stem);

    // Same hypothesis scored at both collars: 0.25 for the historical gate,
    // 0 (no-collar) for the headline like-for-like metric.
    let decomp = compute_der_decomposition(&ref_turns, &hyp_turns, 0.25);
    let decomp_no_collar = compute_der_decomposition(&ref_turns, &hyp_turns, 0.0);
    let num_speakers = hyp_turns
        .iter()
        .map(|t| t.speaker.0)
        .collect::<std::collections::HashSet<_>>()
        .len();
    (decomp, decomp_no_collar, num_speakers, stem)
}

#[ignore = "requires downloaded models"]
#[test]
fn cli_der_regression_v2_e2e_smoke() {
    let baseline = common::load_baseline(&common::der_baseline_path());
    let wav_path = Path::new("tests/data/e2e-smoke/audio/fuzfh.wav");
    let rttm_path = Path::new("tests/data/e2e-smoke/rttm/fuzfh.rttm");

    if !common::require_wav(wav_path) {
        return;
    }

    let (decomp, decomp_no_collar, _num_speakers, stem) = run_cli_diarize(wav_path, rttm_path);
    let der = decomp.total.der;
    let der_no_collar = decomp_no_collar.total.der;
    println!(
        "{stem}: DER(collar 0.25)={:.2}% DER(no collar)={:.2}%",
        der * 100.0,
        der_no_collar * 100.0
    );

    common::gate_against_baseline("v2_e2e_smoke", der, &baseline.v2_e2e_smoke);
    common::assert_no_collar("v2_e2e_smoke", der_no_collar, &baseline.v2_e2e_smoke);
}

#[ignore = "requires downloaded models and dataset"]
#[test]
fn cli_der_regression_v2_ami_single() {
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

    // No-collar decomposition is not gated here — on ~79%-overlap audio total DER
    // is miss-bound at any collar (see common::gate_ami_longform).
    let (decomp, _decomp_no_collar, num_speakers, stem) = run_cli_diarize(&wav_path, &rttm_path);
    let der = decomp.total.der;
    let single_der = decomp.single_speaker.der;
    let confusion = decomp.total.confusion_rate;
    // Overlap-aware decomposition: the AMI gate references the split so
    // a regression is interpretable — total DER hides where the error comes from.
    println!(
        "{stem}: DER={:.2}% overlap-excluded DER={:.2}% overlap-region DER={:.2}% confusion={:.2}% speakers={}",
        der * 100.0,
        single_der * 100.0,
        decomp.overlap.der * 100.0,
        confusion * 100.0,
        num_speakers
    );
    for r in &decomp.per_speaker_recall {
        println!(
            "  ref spk {} recall={:.1}% ({}/{} frames)",
            r.speaker,
            r.recall * 100.0,
            r.recalled_frames,
            r.ref_frames
        );
    }
    // Shared AMI long-form gate (speaker-count collapse + clustering confusion
    // + overlap-excluded DER floor); mirrors der_v2_baseline_test::v2_der_ami_test_single.
    let baseline = common::load_baseline(&common::der_baseline_path());
    common::gate_ami_longform(
        num_speakers,
        confusion,
        single_der,
        &baseline.hybrid_ami_test_single,
    );
}
