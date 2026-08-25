//! Native INT8 scoreboard: DER, RTF, on-disk model size, peak RSS.
//!
//! ```bash
//! cargo test --test native_scoreboard --features cli -- --nocapture
//! ```
//!
//! Vox-3 and RSS/RTF gates skip when the smoke dataset or `polyvoice-bench`
//! is missing. Model-size always runs if `models/int8/` is present.

#![allow(clippy::unwrap_used)]
#![cfg(all(
    any(feature = "cli", feature = "cli-native"),
    not(feature = "onnx"),
    not(feature = "backend-tract")
))]

use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Deserialize)]
struct Floors {
    der_no_collar_micro_max: f64,
    der_no_collar_macro_max: f64,
    rt_factor_min: f64,
    model_bytes_max: u64,
    rss_mib_max: f64,
}

#[derive(Deserialize)]
struct BenchOut {
    der_no_collar_micro: f64,
    der_no_collar_macro: f64,
    rt_factor_avg: f64,
}

fn floors() -> Floors {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/native_scoreboard.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn int8_paths() -> Option<(PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    let a = root.join("int8/powerset_int8.onnx");
    let b = root.join("int8/resnet34_int8.onnx");
    if a.is_file() && b.is_file() {
        Some((a, b))
    } else {
        None
    }
}

fn vox3() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks/results/powerset-tract-rtf-der-2026-08-12/smoke-vox3");
    (p.join("audio").is_dir() && p.join("rttm").is_dir()).then_some(p)
}

struct Run {
    out: BenchOut,
    stderr: String,
}

fn run_bench(bench: &Path, dataset: &Path) -> Run {
    let out_json = tempfile::NamedTempFile::with_suffix(".json").unwrap();
    #[cfg(target_os = "macos")]
    let output = Command::new("/usr/bin/time")
        .args([
            "-l",
            bench.to_str().unwrap(),
            dataset.to_str().unwrap(),
            "--profile",
            "balanced",
            "--pipeline",
            "v2",
            "--clusterer",
            "vbx",
            "--collar",
            "0.0",
            "--output",
            out_json.path().to_str().unwrap(),
        ])
        .output()
        .expect("run timed bench");
    #[cfg(not(target_os = "macos"))]
    let output = Command::new(bench)
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
            "--output",
            out_json.path().to_str().unwrap(),
        ])
        .output()
        .expect("run bench");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "polyvoice-bench failed: {stderr}");
    let out: BenchOut = serde_json::from_slice(&std::fs::read(out_json.path()).unwrap()).unwrap();
    Run { out, stderr }
}

fn assert_der(out: &BenchOut, floors: &Floors) {
    eprintln!(
        "scoreboard DER_micro={:.4} DER_macro={:.4} RTFx={:.2}",
        out.der_no_collar_micro, out.der_no_collar_macro, out.rt_factor_avg
    );
    assert!(
        out.der_no_collar_micro <= floors.der_no_collar_micro_max + 1e-6,
        "DER micro {:.4} > floor {}",
        out.der_no_collar_micro,
        floors.der_no_collar_micro_max
    );
    assert!(
        out.der_no_collar_macro <= floors.der_no_collar_macro_max + 1e-6,
        "DER macro {:.4} > floor {}",
        out.der_no_collar_macro,
        floors.der_no_collar_macro_max
    );
}

#[test]
fn int8_pair_stays_under_size_floor() {
    let Some((seg, emb)) = int8_paths() else {
        eprintln!("skip: models/int8 pair missing");
        return;
    };
    let n = std::fs::metadata(&seg).unwrap().len() + std::fs::metadata(&emb).unwrap().len();
    let max = floors().model_bytes_max;
    assert!(
        n <= max,
        "INT8 pair is {n} bytes; floor is {max} (powerset+resnet)"
    );
}

#[cfg(target_os = "macos")]
fn parse_max_rss_mib(time_stderr: &str) -> Option<f64> {
    for line in time_stderr.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_suffix("maximum resident set size")
            && let Ok(bytes) = rest.split_whitespace().last()?.parse::<f64>()
        {
            return Some(bytes / 1024.0 / 1024.0);
        }
    }
    None
}

#[test]
fn vox3_holds_der_rtf_rss_floors() {
    let Some(dataset) = vox3() else {
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
    let is_release = bench.components().any(|c| c.as_os_str() == "release");
    let floors = floors();
    let run = run_bench(&bench, &dataset);
    assert_der(&run.out, &floors);

    #[cfg(target_os = "macos")]
    if is_release {
        // Isolated idle-host runs sit at 117–121×. A cold BNNS cache or a
        // busy workstation can land a hair under; one retry is the gate.
        let run = if run.out.rt_factor_avg + 1e-6 < floors.rt_factor_min {
            eprintln!(
                "RTFx {:.2} < {}; retrying once",
                run.out.rt_factor_avg, floors.rt_factor_min
            );
            let retry = run_bench(&bench, &dataset);
            assert_der(&retry.out, &floors);
            retry
        } else {
            run
        };
        assert!(
            run.out.rt_factor_avg + 1e-6 >= floors.rt_factor_min,
            "RTFx {:.2} < floor {}",
            run.out.rt_factor_avg,
            floors.rt_factor_min
        );
        let rss = parse_max_rss_mib(&run.stderr).expect("parse max RSS from /usr/bin/time -l");
        eprintln!("scoreboard RSS={rss:.2} MiB");
        assert!(
            rss <= floors.rss_mib_max + 0.5,
            "peak RSS {rss:.2} MiB > floor {} MiB",
            floors.rss_mib_max
        );
    } else {
        let _ = run;
        eprintln!("skip RTF/RSS floors: using debug polyvoice-bench");
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (is_release, run);
}
