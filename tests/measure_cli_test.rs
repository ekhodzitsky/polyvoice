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

const SILERO_VAD_FILE: &str = "silero_vad.onnx";
const WESPEAKER_FILE: &str = "wespeaker_resnet34.onnx";

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
    let assert = measure_cmd()
        .args([
            "embedder-short",
            "--veri-list",
            tmp.path().join("veri.txt").to_str().expect("utf-8"),
            "--wav-root",
            tmp.path().to_str().expect("utf-8"),
        ])
        .assert()
        .failure();
    // Product `cli` builds stub the subcommand; the live path needs tract CAM++.
    #[cfg(all(
        feature = "embedder-native",
        feature = "backend-tract",
        feature = "embedder"
    ))]
    assert.stderr(predicate::str::contains("no VoxCeleb pairs"));
    #[cfg(not(all(
        feature = "embedder-native",
        feature = "backend-tract",
        feature = "embedder"
    )))]
    assert.stderr(predicate::str::contains("backend-tract"));
}
