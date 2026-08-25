//! Smoke tests for the tract-only CLI (`--features cli-tract`).
//!
//! Compiled only when `onnx` is off so the assertions describe the no-dylib
//! front door. Product `cli` tests live in `cli_smoke_test.rs`.

#![allow(clippy::unwrap_used)]
#![cfg(all(feature = "cli-tract", not(feature = "onnx")))]

use assert_cmd::Command;
use predicates::prelude::*;

fn polyvoice() -> Command {
    let mut cmd = Command::cargo_bin("polyvoice").unwrap();
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
        .stdout(predicate::str::contains("download-models"));
}

#[test]
fn version_prints() {
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
fn legacy_is_rejected_without_onnx() {
    polyvoice()
        .args(["--legacy", "/nonexistent.wav"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("onnx"))
        .stderr(predicate::str::contains("cli-tract"));
}

#[test]
fn bench_help() {
    Command::cargo_bin("polyvoice-bench")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("pipeline"));
}

#[test]
fn measure_help() {
    Command::cargo_bin("polyvoice-measure")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("streaming"));
}
