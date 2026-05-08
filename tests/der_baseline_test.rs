//! M6b — DER baseline schema validity tests. Numbers are deferred to an
//! operational follow-up after M5 INT8 publish closes.

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Baseline {
    schema: String,
    voxconverse_test: VoxConverse,
}

#[derive(Deserialize)]
struct VoxConverse {
    files: Option<usize>,
    profile: String,
    der_collar_0_25: Option<f64>,
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
    assert!(parsed.voxconverse_test.status.contains("schema-only"));
}

#[test]
fn der_baseline_acknowledges_deferred_numbers() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json");
    let raw = std::fs::read_to_string(&path).expect("read");
    let parsed: Baseline = serde_json::from_str(&raw).expect("parse");
    assert!(
        parsed.voxconverse_test.files.is_none()
            && parsed.voxconverse_test.der_collar_0_25.is_none(),
        "numbers must remain null until operational baseline closure run"
    );
}
