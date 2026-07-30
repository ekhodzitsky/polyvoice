//! DER baseline validation. Numbers are verified against committed benchmark artifacts.

mod common;

fn load() -> common::DerBaseline {
    common::load_baseline(&common::der_baseline_path())
}

#[test]
fn der_baseline_json_parses() {
    let parsed = load();
    assert_eq!(parsed.schema, "polyvoice-der-baseline-v2");
    assert_eq!(parsed.voxconverse_test.profile, "balanced");
    assert_eq!(parsed.voxconverse_test.tolerance, Some(1.0));
    assert!(parsed.voxconverse_test.status.contains("operational"));
    assert!(parsed.e2e_smoke.status.contains("operational"));
    assert!(parsed.ami_test_single.status.contains("operational"));
    assert!(parsed.v2_e2e_smoke.status.contains("operational"));
}

#[test]
fn der_baseline_voxconverse_has_verified_numbers() {
    let parsed = load();

    assert_eq!(
        parsed.voxconverse_test.files,
        Some(232),
        "must cover full VoxConverse-test"
    );
    let der = parsed
        .voxconverse_test
        .der_collar_0_25
        .expect("der_collar_0_25 must be set");
    assert!(der > 0.0 && der < 100.0, "DER must be a sane percentage");
}

#[test]
fn der_baseline_ami_has_verified_numbers() {
    let parsed = load();

    assert_eq!(parsed.ami_test_single.files, Some(1));
    let der = parsed
        .ami_test_single
        .der_collar_0_25
        .expect("der_collar_0_25 must be set");
    assert!(
        der > 0.0 && der < 100.0,
        "AMI DER must be a sane percentage"
    );
}

#[test]
fn der_baseline_v2_e2e_smoke_has_verified_numbers() {
    let parsed = load();

    assert_eq!(parsed.v2_e2e_smoke.files, Some(1));
    let der = parsed
        .v2_e2e_smoke
        .der_collar_0_25
        .expect("der_collar_0_25 must be set");
    assert!(
        der > 0.0 && der < 10.0,
        "Pipeline v2 e2e_smoke DER must be < 10%"
    );
}
