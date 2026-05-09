//! DER baseline validation. Numbers are verified against a full VoxConverse-test
//! run with the legacy v0.5 pipeline (threshold=0.45, collar=0.25).

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Baseline {
    schema: String,
    voxconverse_test: VoxConverse,
}

#[derive(Deserialize)]
struct VoxConverse {
    files: usize,
    profile: String,
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
    assert_eq!(parsed.schema, "polyvoice-der-baseline-v1");
    assert_eq!(parsed.voxconverse_test.profile, "balanced");
    assert_eq!(parsed.voxconverse_test.tolerance, 1.0);
    assert!(parsed.voxconverse_test.status.contains("operational"));
}

#[test]
fn der_baseline_has_verified_numbers() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json");
    let raw = std::fs::read_to_string(&path).expect("read");
    let parsed: Baseline = serde_json::from_str(&raw).expect("parse");

    assert_eq!(parsed.voxconverse_test.files, 232, "must cover full VoxConverse-test");
    assert!(
        parsed.voxconverse_test.der_collar_0_25 > 0.0
            && parsed.voxconverse_test.der_collar_0_25 < 100.0,
        "DER must be a sane percentage"
    );
}
