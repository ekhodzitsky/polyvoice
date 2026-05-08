//! M6b — smoke tests for the polyvoice CLI.

#![cfg(feature = "cli")]

use std::process::Command;

fn cli() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_polyvoice"));
    c.env("RUST_BACKTRACE", "0");
    c
}

#[test]
fn help_top_level() {
    let out = cli().arg("--help").output().expect("spawn polyvoice");
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("diarize"), "help missing 'diarize' subcommand: {s}");
    assert!(s.contains("download-models"), "help missing 'download-models': {s}");
    assert!(s.contains("models"), "help missing 'models': {s}");
}

#[test]
fn help_diarize() {
    let out = cli().args(["diarize", "--help"]).output().expect("spawn");
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("--profile"));
    assert!(s.contains("--output"));
    assert!(s.contains("--format"));
}

#[test]
fn diarize_invalid_profile_errors() {
    let out = cli()
        .args(["diarize", "/nonexistent/file.wav", "--profile", "garbage"])
        .output()
        .expect("spawn");
    assert!(!out.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("invalid profile") || stderr.contains("garbage"), "stderr: {stderr}");
}

#[test]
fn models_list_runs() {
    let out = cli().args(["models", "list"]).output().expect("spawn");
    // May fail if registry can't write to home dir in CI sandbox — accept either success or
    // a known cache-dir error; we only assert the binary doesn't crash with internal panic.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked at"), "binary panicked: {stderr}");
}
