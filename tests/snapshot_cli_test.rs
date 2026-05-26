//! Snapshot tests for CLI output stability.

#![cfg(feature = "cli")]
#![allow(clippy::unwrap_used)]

use assert_cmd::Command;

fn polyvoice() -> Command {
    let mut cmd = Command::cargo_bin("polyvoice").unwrap();
    cmd.env("RUST_BACKTRACE", "0");
    cmd
}

#[test]
fn snapshot_help_top_level() {
    let output = polyvoice().arg("--help").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    insta::assert_snapshot!(stdout);
}

#[test]
fn snapshot_version() {
    let output = polyvoice().arg("--version").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Version changes with releases — snapshot only the prefix.
    assert!(stdout.starts_with("polyvoice "));
}

