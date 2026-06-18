#![allow(clippy::unwrap_used)]
//! M6b — smoke tests for the polyvoice CLI using assert_cmd + predicates.

#![cfg(feature = "cli")]

use assert_cmd::Command;
use predicates::prelude::*;

fn polyvoice() -> Command {
    let mut cmd = Command::cargo_bin("polyvoice").unwrap();
    cmd.env("RUST_BACKTRACE", "0");
    cmd
}

fn polyvoice_bench() -> Command {
    let mut cmd = Command::cargo_bin("polyvoice-bench").unwrap();
    cmd.env("RUST_BACKTRACE", "0");
    cmd
}

#[test]
fn help_top_level() {
    polyvoice()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("diarize"))
        .stdout(predicate::str::contains("download-models"))
        .stdout(predicate::str::contains("models"));
}

#[test]
fn help_diarize() {
    polyvoice()
        .args(["diarize", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--profile"))
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--format"));
}

#[test]
fn version_prints_correctly() {
    polyvoice()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "polyvoice {}",
            env!("CARGO_PKG_VERSION")
        )));
}

#[test]
fn diarize_invalid_profile_errors() {
    polyvoice()
        .args(["diarize", "/nonexistent/file.wav", "--profile", "garbage"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("invalid profile").or(predicate::str::contains("garbage")),
        );
}

#[test]
fn diarize_missing_file_errors() {
    polyvoice()
        .args([
            "diarize",
            "/nonexistent/directory/file.wav",
            "--profile",
            "balanced",
        ])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("No such file").or(predicate::str::contains("cannot find")),
        );
}

#[test]
fn models_list_runs_without_panic() {
    polyvoice()
        .args(["models", "list"])
        .assert()
        .stderr(predicate::str::contains("panicked at").not());
}

#[test]
fn models_info_shows_metadata() {
    polyvoice()
        .args(["models", "info", "silero_vad"])
        .assert()
        .success()
        .stdout(predicate::str::contains("silero_vad").or(predicate::str::contains("sha256")));
}

#[test]
fn download_models_help_shows_profiles() {
    polyvoice()
        .args(["download-models", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--profile"));
}

#[test]
fn bench_help_shows_args() {
    polyvoice_bench()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--profile"))
        .stdout(predicate::str::contains("--collar"))
        .stdout(predicate::str::contains("--output"));
}

#[test]
fn schema_outputs_valid_json_contract() {
    let assert = polyvoice().arg("schema").assert().success();
    let json: serde_json::Value =
        serde_json::from_slice(&assert.get_output().stdout).expect("schema must be valid JSON");
    assert_eq!(json["title"], "DiarizationResult");
    // Core contract fields the agent relies on.
    let required = json["required"].as_array().unwrap();
    for f in ["segments", "turns", "num_speakers"] {
        assert!(required.iter().any(|v| v == f), "missing required: {f}");
    }
    // Additive who-said-what fields are documented in the schema.
    assert!(json["$defs"]["turn"]["properties"]["text"].is_object());
    assert!(json["$defs"]["word"].is_object());
}
