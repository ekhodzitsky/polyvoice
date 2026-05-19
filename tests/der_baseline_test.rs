//! DER baseline validation. Numbers are verified against committed benchmark artifacts.

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
#[allow(dead_code)]
struct Baseline {
    schema: String,
    #[serde(rename = "crate_version")]
    crate_version: Option<String>,
    voxconverse_test: DatasetBaseline,
    #[serde(default)]
    voxconverse_test_10files: Option<DatasetBaseline>,
    e2e_smoke: DatasetBaseline,
    ami_test_single: DatasetBaseline,
    v2_e2e_smoke: DatasetBaseline,
}

#[derive(Deserialize)]
struct DatasetBaseline {
    files: usize,
    profile: String,
    #[serde(rename = "der_collar_0_25")]
    der_collar_0_25: f64,
    tolerance: f64,
    #[serde(rename = "_status")]
    status: String,
}

#[test]
fn der_baseline_json_parses() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json");
    let raw = std::fs::read_to_string(&path).expect("read der_baseline.json");
    let parsed: Baseline = serde_json::from_str(&raw).expect("parse der_baseline.json");
    assert_eq!(parsed.schema, "polyvoice-der-baseline-v2");
    assert_eq!(parsed.voxconverse_test.profile, "balanced");
    assert_eq!(parsed.voxconverse_test.tolerance, 1.0);
    assert!(parsed.voxconverse_test.status.contains("operational"));
    assert!(parsed.e2e_smoke.status.contains("operational"));
    assert!(parsed.ami_test_single.status.contains("operational"));
    assert!(parsed.v2_e2e_smoke.status.contains("operational"));
}

#[test]
fn der_baseline_voxconverse_has_verified_numbers() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json");
    let raw = std::fs::read_to_string(&path).expect("read");
    let parsed: Baseline = serde_json::from_str(&raw).expect("parse");

    assert_eq!(
        parsed.voxconverse_test.files, 232,
        "must cover full VoxConverse-test"
    );
    assert!(
        parsed.voxconverse_test.der_collar_0_25 > 0.0
            && parsed.voxconverse_test.der_collar_0_25 < 100.0,
        "DER must be a sane percentage"
    );
}

#[test]
fn der_baseline_ami_has_verified_numbers() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json");
    let raw = std::fs::read_to_string(&path).expect("read");
    let parsed: Baseline = serde_json::from_str(&raw).expect("parse");

    assert_eq!(parsed.ami_test_single.files, 1);
    assert!(
        parsed.ami_test_single.der_collar_0_25 > 0.0
            && parsed.ami_test_single.der_collar_0_25 < 100.0,
        "AMI DER must be a sane percentage"
    );
}

#[test]
fn der_baseline_v2_e2e_smoke_has_verified_numbers() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json");
    let raw = std::fs::read_to_string(&path).expect("read");
    let parsed: Baseline = serde_json::from_str(&raw).expect("parse");

    assert_eq!(parsed.v2_e2e_smoke.files, 1);
    assert!(
        parsed.v2_e2e_smoke.der_collar_0_25 > 0.0 && parsed.v2_e2e_smoke.der_collar_0_25 < 10.0,
        "Pipeline v2 e2e_smoke DER must be < 10%"
    );
}
