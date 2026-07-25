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
    let stdout = String::from_utf8_lossy(&output.stdout).replace("polyvoice.exe", "polyvoice");
    // Help text for the input path depends on the optional `audio-io` feature
    // (multi-format decode). Matrix CI jobs use `cli` only; release-check and
    // ubuntu all-features enable `audio-io` — keep a snap per variant.
    #[cfg(feature = "audio-io")]
    insta::assert_snapshot!("help_top_level_audio_io", stdout);
    #[cfg(not(feature = "audio-io"))]
    insta::assert_snapshot!("help_top_level", stdout);
}

#[test]
fn snapshot_version() {
    let output = polyvoice().arg("--version").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Version changes with releases — snapshot only the prefix.
    assert!(stdout.starts_with("polyvoice "));
}
