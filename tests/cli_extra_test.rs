#![allow(clippy::unwrap_used)]
//! Extended CLI coverage: flag-validation error paths, `completions` / `models`
//! subcommands, and offline end-to-end diarization runs against a model cache
//! seeded from the checked-in ONNX files (SHA-256-verified cache hits — no
//! network). Model-backed tests soft-skip when the local models are absent.

#![cfg(feature = "cli")]

use predicates::prelude::*;
use std::path::{Path, PathBuf};

mod common;
use common::polyvoice_cmd;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Write a 16 kHz mono 16-bit WAV holding `secs` seconds of a two-tone signal.
fn write_tone_wav(path: &Path, secs: f32) {
    let sr = 16_000_u32;
    let n = (secs * sr as f32) as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let f = if i < n / 2 { 220.0 } else { 440.0 };
            (t * std::f32::consts::TAU * f).sin() * 0.3
        })
        .collect();
    common::write_pcm16_mono(path, sr, &samples);
}

const SEEDED_MODELS: [(&str, &str); 3] = [
    ("silero_vad.onnx", "silero_vad.onnx"),
    ("int8/powerset_int8.onnx", "powerset_int8.onnx"),
    ("int8/resnet34_int8.onnx", "resnet34_int8.onnx"),
];

/// A model cache dir populated from the checked-in `models/` files (hard link,
/// falling back to copy). The registry treats them as verified cache hits, so
/// diarization runs fully offline. `None` when the local models are absent.
fn seeded_models_cache() -> Option<tempfile::TempDir> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    if SEEDED_MODELS
        .iter()
        .any(|(rel, _)| !src.join(rel).is_file())
    {
        eprintln!("checked-in ONNX models not found under models/ — skipping e2e");
        return None;
    }
    let tmp = tempfile::tempdir().unwrap();
    for (rel, dest_name) in SEEDED_MODELS {
        let (from, to) = (src.join(rel), tmp.path().join(dest_name));
        if std::fs::hard_link(&from, &to).is_err() {
            std::fs::copy(&from, &to).unwrap();
        }
    }
    Some(tmp)
}

/// Small real-speech fixture used by the e2e runs; soft-gated via require_wav.
fn smoke_wav() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/e2e-smoke/audio/fuzfh.wav")
}

// ---------------------------------------------------------------------------
// Flag-validation error paths (no models, no network)
// ---------------------------------------------------------------------------

#[test]
fn diarize_without_input_errors() {
    polyvoice_cmd()
        .arg("--quiet")
        .assert()
        .failure()
        .stderr(predicate::str::contains("no input"));
}

#[test]
fn diarize_rejects_models_cache_traversal() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("in.wav");
    std::fs::write(&wav, b"x").unwrap();
    polyvoice_cmd()
        .args([
            "diarize",
            wav.to_str().unwrap(),
            "--models-cache",
            "../escape",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("path traversal"));
}

#[test]
fn diarize_rejects_bad_latency_preset() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("in.wav");
    std::fs::write(&wav, b"x").unwrap();
    polyvoice_cmd()
        .args([
            "diarize",
            wav.to_str().unwrap(),
            "--models-cache",
            dir.path().join("cache").to_str().unwrap(),
            "--latency-preset",
            "warp-speed",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid --latency-preset"));
}

#[test]
fn diarize_rejects_unknown_clusterer() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("in.wav");
    std::fs::write(&wav, b"x").unwrap();
    polyvoice_cmd()
        .args([
            "diarize",
            wav.to_str().unwrap(),
            "--models-cache",
            dir.path().join("cache").to_str().unwrap(),
            "--clusterer",
            "kmeans",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown --clusterer"));
}

#[test]
fn diarize_rejects_unknown_execution_provider() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("in.wav");
    std::fs::write(&wav, b"x").unwrap();
    polyvoice_cmd()
        .args([
            "diarize",
            wav.to_str().unwrap(),
            "--models-cache",
            dir.path().join("cache").to_str().unwrap(),
            "--clusterer",
            "ahc",
            "--execution-provider",
            "tpu",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown --execution-provider"));
}

#[test]
fn diarize_rejects_oversized_max_speakers() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("in.wav");
    std::fs::write(&wav, b"x").unwrap();
    polyvoice_cmd()
        .args([
            "diarize",
            wav.to_str().unwrap(),
            "--models-cache",
            dir.path().join("cache").to_str().unwrap(),
            "--clusterer",
            "ahc",
            "--max-speakers",
            "300",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("max_speakers must be in 1..=255"));
}

// ---------------------------------------------------------------------------
// Subcommands that run offline
// ---------------------------------------------------------------------------

#[test]
fn completions_generate_for_every_shell() {
    for shell in ["bash", "zsh", "fish", "powershell", "elvish"] {
        let assert = polyvoice_cmd()
            .args(["completions", shell])
            .assert()
            .success();
        let stdout = String::from_utf8_lossy(&assert.get_output().stdout);
        assert!(!stdout.is_empty(), "empty completions for {shell}");
        assert!(
            stdout.contains("polyvoice"),
            "no binary name in {shell} completions"
        );
    }
}

#[test]
fn completions_reject_unknown_shell() {
    polyvoice_cmd()
        .args(["completions", "tcsh"])
        .assert()
        .failure();
}

#[test]
fn models_info_resolves_stage_alias() {
    polyvoice_cmd()
        .args(["models", "info", "latest"])
        .assert()
        .success()
        .stdout(predicate::str::contains("latest ->"))
        .stdout(predicate::str::contains("sha256"));
}

#[test]
fn models_info_prints_calibration_metadata() {
    polyvoice_cmd()
        .args(["models", "info", "powerset_int8"])
        .assert()
        .success()
        .stdout(predicate::str::contains("calibration"))
        .stdout(predicate::str::contains("provenance"))
        .stdout(predicate::str::contains("license"));
}

#[test]
fn models_info_unknown_model_errors() {
    polyvoice_cmd()
        .args(["models", "info", "no_such_model_zzz"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found in manifest"));
}

#[test]
fn schema_stdout_is_the_committed_schema_file() {
    let assert = polyvoice_cmd().arg("schema").assert().success();
    let committed = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("schema/diarization-result-v1.json"),
    )
    .unwrap();
    assert_eq!(assert.get_output().stdout, committed);
}

// ---------------------------------------------------------------------------
// Offline end-to-end diarization (seeded model cache)
// ---------------------------------------------------------------------------

#[test]
fn e2e_bare_wav_json_to_stdout() {
    let Some(cache) = seeded_models_cache() else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("tones.wav");
    write_tone_wav(&wav, 12.0);
    // Bare `polyvoice <wav>` (no subcommand) is the implicit diarize form.
    let assert = polyvoice_cmd()
        .args([
            wav.to_str().unwrap(),
            "--models-cache",
            cache.path().to_str().unwrap(),
            "--clusterer",
            "ahc",
            "--format",
            "json",
            "--quiet",
        ])
        .assert()
        .success();
    let json: serde_json::Value = serde_json::from_slice(&assert.get_output().stdout)
        .expect("stdout must be the JSON result");
    assert_eq!(json["schema_version"], "diarization-result-v1");
    assert!(json["turns"].is_array());
    assert!(json["num_speakers"].is_number());
}

#[test]
fn e2e_v2_vbx_rttm_on_real_speech() {
    let Some(cache) = seeded_models_cache() else {
        return;
    };
    let wav = smoke_wav();
    if !common::require_wav(&wav) {
        return;
    }
    let plda = common::vbx_plda_fixture_dir();
    // Default clusterer is vbx; pass the checked-in PLDA fixtures explicitly.
    polyvoice_cmd()
        .args([
            "diarize",
            wav.to_str().unwrap(),
            "--models-cache",
            cache.path().to_str().unwrap(),
            "--vbx-plda-dir",
            plda.to_str().unwrap(),
            "--format",
            "rttm",
            "--quiet",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("SPEAKER fuzfh 1"));
}

// --legacy runs the Silero-based legacy pipeline, which native/tract builds
// reject by design (see legacy_is_rejected in cli_native_smoke.rs).
#[cfg(feature = "onnx")]
#[test]
fn e2e_legacy_writes_output_file_and_keeps_stdout_clean() {
    let Some(cache) = seeded_models_cache() else {
        return;
    };
    let wav = smoke_wav();
    if !common::require_wav(&wav) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.rttm");
    let assert = polyvoice_cmd()
        .args([
            "diarize",
            wav.to_str().unwrap(),
            "--models-cache",
            cache.path().to_str().unwrap(),
            "--legacy",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();
    // STDOUT discipline: with --output, nothing goes to stdout.
    assert!(assert.get_output().stdout.is_empty());
    let rttm = std::fs::read_to_string(&out).unwrap();
    assert!(rttm.contains("SPEAKER fuzfh 1"), "unexpected RTTM:\n{rttm}");
}

#[test]
fn e2e_machine_mode_json_flag() {
    let Some(cache) = seeded_models_cache() else {
        return;
    };
    let wav = smoke_wav();
    if !common::require_wav(&wav) {
        return;
    }
    let assert = polyvoice_cmd()
        .args([
            "diarize",
            wav.to_str().unwrap(),
            "--models-cache",
            cache.path().to_str().unwrap(),
            "--clusterer",
            "ahc",
            "--json",
        ])
        .assert()
        .success();
    let json: serde_json::Value = serde_json::from_slice(&assert.get_output().stdout)
        .expect("--json stdout must be the JSON result");
    assert!(json["turns"].is_array());
}

#[test]
fn e2e_exclusive_srt_projection() {
    let Some(cache) = seeded_models_cache() else {
        return;
    };
    let wav = smoke_wav();
    if !common::require_wav(&wav) {
        return;
    }
    polyvoice_cmd()
        .args([
            "diarize",
            wav.to_str().unwrap(),
            "--models-cache",
            cache.path().to_str().unwrap(),
            "--clusterer",
            "ahc",
            "--exclusive",
            "--format",
            "srt",
            "--quiet",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("-->"));
}

#[test]
fn e2e_latency_preset_and_embed_window_run() {
    let Some(cache) = seeded_models_cache() else {
        return;
    };
    let wav = smoke_wav();
    if !common::require_wav(&wav) {
        return;
    }
    for extra in [
        vec!["--latency-preset", "realtime"],
        vec!["--embed-window", "1.5"],
    ] {
        polyvoice_cmd()
            .args([
                "diarize",
                wav.to_str().unwrap(),
                "--models-cache",
                cache.path().to_str().unwrap(),
                "--clusterer",
                "ahc",
                "--format",
                "txt",
                "--quiet",
            ])
            .args(&extra)
            .assert()
            .success();
    }
}
