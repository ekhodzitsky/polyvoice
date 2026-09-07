//! File-parallel bench determinism: `--jobs 3` on one shared v2 pipeline must
//! produce the same per-file scores and aggregates as `--jobs 1`, bit for bit
//! (timing fields excluded).
//!
//! ```bash
//! cargo test --test bench_jobs_determinism --features cli -- --nocapture
//! ```

#![allow(clippy::unwrap_used)]
#![cfg(all(
    any(feature = "cli", feature = "cli-native"),
    not(feature = "onnx"),
    not(feature = "backend-tract")
))]

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

fn vox3() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let p = root.join("tests/data/native-vox3");
    if p.join("audio").is_dir() && p.join("rttm").is_dir() {
        Some(p)
    } else {
        None
    }
}

fn require_data() -> bool {
    std::env::var("POLYVOICE_REQUIRE_DATA")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn run_bench(bench: &Path, dataset: &Path, jobs: usize) -> Value {
    let out_json = tempfile::NamedTempFile::with_suffix(".json").unwrap();
    let plda = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/vbx-plda");
    let output = Command::new(bench)
        .env("POLYVOICE_VBX_PLDA_DIR", &plda)
        .args([
            dataset.to_str().unwrap(),
            "--profile",
            "balanced",
            "--pipeline",
            "v2",
            "--clusterer",
            "vbx",
            "--collar",
            "0.0",
            "--jobs",
            &jobs.to_string(),
            "--output",
            out_json.path().to_str().unwrap(),
        ])
        .output()
        .expect("run bench");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "polyvoice-bench failed: {stderr}");
    serde_json::from_slice(&std::fs::read(out_json.path()).unwrap()).unwrap()
}

/// Per-file score keys that must match bit for bit across job counts. Timing
/// (`rt_factor*`, `runtime_secs`, `stage_timings`) legitimately differs.
const FILE_KEYS: &[&str] = &[
    "der_collar",
    "der_no_collar",
    "miss_rate",
    "false_alarm_rate",
    "confusion_rate",
];
const AGG_KEYS: &[&str] = &[
    "der_collar_macro",
    "der_no_collar_macro",
    "der_collar_micro",
    "der_no_collar_micro",
    "miss",
    "false_alarm",
    "confusion",
];

#[test]
fn jobs3_matches_jobs1_bitwise() {
    let Some(dataset) = vox3() else {
        assert!(
            !require_data(),
            "POLYVOICE_REQUIRE_DATA=1 but tests/data/native-vox3 is missing"
        );
        eprintln!("skip: Vox-3 smoke dataset missing");
        return;
    };
    let release_bench =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release/polyvoice-bench");
    let bench = if release_bench.is_file() {
        release_bench
    } else {
        assert_cmd::cargo::cargo_bin("polyvoice-bench")
    };
    let one = run_bench(&bench, &dataset, 1);
    let three = run_bench(&bench, &dataset, 3);

    for key in AGG_KEYS {
        assert_eq!(one[key], three[key], "aggregate {key} differs jobs=1 vs 3");
    }
    let files1 = one["per_file"].as_array().unwrap();
    let files3 = three["per_file"].as_array().unwrap();
    assert_eq!(files1.len(), files3.len(), "per_file count differs");
    for f1 in files1 {
        let name = f1["filename"].as_str().unwrap();
        let f3 = files3
            .iter()
            .find(|f| f["filename"].as_str() == Some(name))
            .unwrap_or_else(|| panic!("{name} missing from jobs=3 report"));
        for key in FILE_KEYS {
            assert_eq!(f1[key], f3[key], "{name}.{key} differs jobs=1 vs 3");
        }
    }
}
