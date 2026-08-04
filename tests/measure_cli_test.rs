//! CLI-level integration tests for the `polyvoice-measure` binary.
//!
//! Argument-handling tests run unconditionally. Tests that drive real ONNX
//! inference are soft-gated on the local model cache already holding the
//! required artifacts: the binary resolves models through the default
//! `ModelRegistry`, which downloads on a cache miss — these tests never want
//! the network, so a missing cache entry skips the test instead.

#![cfg(feature = "cli")]

mod common;

use assert_cmd::Command;
use predicates::prelude::*;
use std::path::Path;

const SILERO_VAD_FILE: &str = "silero_vad.onnx";
const WESPEAKER_FILE: &str = "wespeaker_resnet34.onnx";
const ERES2NETV2_FILE: &str = "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx";

fn measure_cmd() -> Command {
    let mut cmd = Command::cargo_bin("polyvoice-measure").expect("polyvoice-measure binary");
    cmd.env("RUST_BACKTRACE", "0");
    cmd
}

/// Soft gate: `true` when every required model file is already in the default
/// registry cache (so the binary will not attempt a download).
fn models_cached(files: &[&str]) -> bool {
    let Ok(registry) = polyvoice::models::ModelRegistry::default() else {
        eprintln!("no default model registry cache dir — skipping");
        return false;
    };
    let missing: Vec<&str> = files
        .iter()
        .copied()
        .filter(|f| !registry.cache_dir().join(f).is_file())
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "model cache missing {missing:?} under {} — skipping",
            registry.cache_dir().display()
        );
        return false;
    }
    true
}

fn write_wav_16k(path: &Path, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("create wav");
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .expect("write sample");
    }
    w.finalize().expect("finalize wav");
}

/// Minimal diarization dataset: one 6 s file, two speakers, speaker A with two
/// segments (so short-segment pair construction finds a positive).
fn make_rttm_dataset() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir(dir.path().join("audio")).expect("audio dir");
    std::fs::create_dir(dir.path().join("rttm")).expect("rttm dir");
    write_wav_16k(
        &dir.path().join("audio/f1.wav"),
        &common::speech_pcm(6.0, 16_000),
    );
    std::fs::write(
        dir.path().join("rttm/f1.rttm"),
        "SPEAKER f1 1 0.0 1.5 <NA> <NA> A <NA> <NA>\n\
         SPEAKER f1 1 2.0 1.5 <NA> <NA> A <NA> <NA>\n\
         SPEAKER f1 1 4.0 1.5 <NA> <NA> B <NA> <NA>\n",
    )
    .expect("write rttm");
    dir
}

fn read_json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path).expect("read report json");
    serde_json::from_str(&text).expect("parse report json")
}

#[test]
fn help_lists_all_subcommands() {
    measure_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("streaming"))
        .stdout(predicate::str::contains("vad-parity"))
        .stdout(predicate::str::contains("embedder-short"));
}

#[test]
fn missing_subcommand_fails() {
    measure_cmd().assert().failure();
}

#[test]
fn unknown_subcommand_fails() {
    measure_cmd().arg("nope").assert().failure();
}

#[test]
fn streaming_rejects_missing_dataset_dir() {
    if !models_cached(&[WESPEAKER_FILE, SILERO_VAD_FILE]) {
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    measure_cmd()
        .args([
            "streaming",
            "--dataset",
            tmp.path().join("no-such-dir").to_str().expect("utf-8"),
        ])
        .assert()
        .failure();
}

#[test]
fn streaming_empty_dataset_emits_zero_file_rows() {
    if !models_cached(&[WESPEAKER_FILE, SILERO_VAD_FILE]) {
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir(tmp.path().join("audio")).expect("audio dir");
    let out = tmp.path().join("report.json");
    measure_cmd()
        .args([
            "streaming",
            "--dataset",
            tmp.path().to_str().expect("utf-8"),
            "--max-files",
            "5",
            "--output",
            out.to_str().expect("utf-8"),
        ])
        .assert()
        .success();
    let v = read_json(&out);
    assert_eq!(v["schema"], "polyvoice-streaming-latency-v1");
    assert_eq!(v["max_files"], 5);
    let rows = v["rows"].as_array().expect("rows array");
    assert_eq!(rows.len(), 3);
    for row in rows {
        assert_eq!(row["files"], 0);
        assert_eq!(row["mean_rtf"], 0.0);
    }
    let names: Vec<&str> = rows
        .iter()
        .map(|r| r["preset"].as_str().expect("preset name"))
        .collect();
    assert_eq!(names, ["realtime", "balanced", "accurate"]);
}

#[test]
fn streaming_scores_fixture_file() {
    if !models_cached(&[WESPEAKER_FILE, SILERO_VAD_FILE]) {
        return;
    }
    let ds = make_rttm_dataset();
    let out = ds.path().join("report.json");
    measure_cmd()
        .args([
            "streaming",
            "--dataset",
            ds.path().to_str().expect("utf-8"),
            "--max-files",
            "5",
            "--output",
            out.to_str().expect("utf-8"),
        ])
        .assert()
        .success();
    let v = read_json(&out);
    let rows = v["rows"].as_array().expect("rows array");
    assert_eq!(rows.len(), 3);
    for row in rows {
        assert_eq!(row["files"], 1);
        assert!(row["total_audio_secs"].as_f64().expect("audio secs") > 5.0);
        assert!(row["macro_der_collar_0"].as_f64().expect("der") >= 0.0);
        assert!(row["input_buffer_latency_secs"].as_f64().expect("lat") > 0.0);
    }
}

#[cfg(feature = "vad-earshot")]
#[test]
fn vad_parity_empty_dataset_passes_gate() {
    if !models_cached(&[WESPEAKER_FILE, SILERO_VAD_FILE]) {
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir(tmp.path().join("audio")).expect("audio dir");
    let out = tmp.path().join("parity.json");
    measure_cmd()
        .args([
            "vad-parity",
            "--dataset",
            tmp.path().to_str().expect("utf-8"),
            "--output",
            out.to_str().expect("utf-8"),
        ])
        .assert()
        .success();
    let v = read_json(&out);
    assert_eq!(v["schema"], "polyvoice-vad-parity-v1");
    assert_eq!(v["silero"]["files"], 0);
    assert_eq!(v["earshot"]["files"], 0);
    assert_eq!(v["parity_pass_collar_0"], true);
    assert_eq!(v["parity_pass_collar_025"], true);
    assert_eq!(v["parity_gate_abs_pp"], 0.3);
}

#[cfg(feature = "vad-earshot")]
#[test]
fn vad_parity_scores_fixture_file() {
    if !models_cached(&[WESPEAKER_FILE, SILERO_VAD_FILE]) {
        return;
    }
    let ds = make_rttm_dataset();
    let out = ds.path().join("parity.json");
    measure_cmd()
        .args([
            "vad-parity",
            "--dataset",
            ds.path().to_str().expect("utf-8"),
            "--output",
            out.to_str().expect("utf-8"),
        ])
        .assert()
        .success();
    let v = read_json(&out);
    assert_eq!(v["silero"]["files"], 1);
    assert_eq!(v["earshot"]["files"], 1);
    assert_eq!(v["silero"]["frame_size"], 512);
    assert!(v["earshot"]["frame_size"].as_u64().expect("frame") > 0);
}

#[test]
fn embedder_short_from_rttm_dataset() {
    if !models_cached(&[WESPEAKER_FILE, SILERO_VAD_FILE, ERES2NETV2_FILE]) {
        return;
    }
    let ds = make_rttm_dataset();
    let out = ds.path().join("embedder.json");
    measure_cmd()
        .args([
            "embedder-short",
            "--veri-list",
            ds.path().join("veri.txt").to_str().expect("utf-8"),
            "--wav-root",
            ds.path().to_str().expect("utf-8"),
            "--durations",
            "0.5,1.0",
            "--max-pairs",
            "4",
            "--output",
            out.to_str().expect("utf-8"),
        ])
        .assert()
        .success();
    let v = read_json(&out);
    assert_eq!(v["schema"], "polyvoice-embedder-short-v1");
    assert_eq!(v["max_pairs"], 4);
    assert_eq!(v["default_embedder"]["model_id"], "wespeaker_resnet34");
    assert_eq!(v["default_embedder"]["dim"], 256);
    assert_eq!(v["eres2netv2"]["model_id"], "eres2netv2");
    let buckets = v["default_embedder"]["short_seg_eer"]
        .as_array()
        .expect("eer buckets");
    assert_eq!(buckets.len(), 2);
    assert_eq!(buckets[0]["duration_secs"], 0.5);
    assert!(buckets[0]["pairs"].as_u64().expect("pairs") >= 1);
    // No --der-dataset: DER fields stay null.
    assert!(v["default_embedder"]["der_macro_collar_0"].is_null());
    assert!(v["eres2netv2"]["der_files"].is_null());
}

#[test]
fn embedder_short_with_der_comparison() {
    if !models_cached(&[WESPEAKER_FILE, SILERO_VAD_FILE, ERES2NETV2_FILE]) {
        return;
    }
    let ds = make_rttm_dataset();
    let out = ds.path().join("embedder_der.json");
    measure_cmd()
        .args([
            "embedder-short",
            "--veri-list",
            ds.path().join("veri.txt").to_str().expect("utf-8"),
            "--wav-root",
            ds.path().to_str().expect("utf-8"),
            "--durations",
            "0.5",
            "--max-pairs",
            "2",
            "--der-dataset",
            ds.path().to_str().expect("utf-8"),
            "--der-max-files",
            "5",
            "--output",
            out.to_str().expect("utf-8"),
        ])
        .assert()
        .success();
    let v = read_json(&out);
    assert_eq!(v["default_embedder"]["der_files"], 1);
    assert_eq!(v["eres2netv2"]["der_files"], 1);
    assert!(
        v["default_embedder"]["der_macro_collar_0"]
            .as_f64()
            .expect("der value")
            >= 0.0
    );
}

#[test]
fn embedder_short_rejects_empty_durations() {
    let tmp = tempfile::tempdir().expect("tempdir");
    measure_cmd()
        .args([
            "embedder-short",
            "--veri-list",
            tmp.path().join("veri.txt").to_str().expect("utf-8"),
            "--wav-root",
            tmp.path().to_str().expect("utf-8"),
            "--durations",
            ",,",
        ])
        .assert()
        .failure();
}

#[test]
fn embedder_short_fails_without_any_pair_source() {
    let tmp = tempfile::tempdir().expect("tempdir");
    measure_cmd()
        .args([
            "embedder-short",
            "--veri-list",
            tmp.path().join("veri.txt").to_str().expect("utf-8"),
            "--wav-root",
            tmp.path().to_str().expect("utf-8"),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no VoxCeleb pairs"));
}
