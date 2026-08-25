//! Smoke tests for the kernel-only CLI (`--features cli` / `cli-native`).

#![allow(clippy::unwrap_used)]
#![cfg(all(
    any(feature = "cli", feature = "cli-native"),
    not(feature = "onnx"),
    not(feature = "backend-tract")
))]

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
        .stdout(predicate::str::contains("diarize"));
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
fn legacy_is_rejected() {
    polyvoice()
        .args(["--legacy", "/nonexistent.wav"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("onnx"));
}
