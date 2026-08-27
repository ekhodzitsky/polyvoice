//! DER baseline validation. Numbers are verified against committed benchmark artifacts.

mod common;

fn load() -> common::DerBaseline {
    common::load_baseline(&common::der_baseline_path())
}

#[test]
fn der_baseline_json_parses() {
    let parsed = load();
    assert_eq!(parsed.schema, "polyvoice-der-baseline-v2");
    assert_eq!(
        parsed.crate_version.as_deref(),
        Some(env!("CARGO_PKG_VERSION")),
        "der_baseline.json crate_version must match the crate"
    );
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

#[test]
fn der_baseline_linux_cpu_product_rows() {
    let parsed = load();

    let vox = &parsed.voxconverse_test_linux_cpu;
    assert_eq!(vox.files, Some(232));
    assert_eq!(vox.execution_provider.as_deref(), Some("cpu"));
    assert_eq!(vox.powerset_batch, Some(8));
    assert_eq!(vox.miss_fa_conf_collar_secs, Some(0.25));
    let der0 = vox
        .der_no_collar_micro
        .or(vox.der_no_collar)
        .expect("linux vox DER0");
    assert!(
        (14.0..16.0).contains(&der0),
        "linux vox DER0 micro should be ~14.9%, got {der0}"
    );
    assert!(vox.status.contains("operational"));
    assert!(vox.rt_factor_avg.is_some_and(|r| r > 1.0));

    let ami = &parsed.ami_test_linux_cpu;
    assert_eq!(ami.files, Some(16));
    assert_eq!(ami.execution_provider.as_deref(), Some("cpu"));
    assert_eq!(ami.powerset_batch, Some(8));
    assert_eq!(ami.miss_fa_conf_collar_secs, Some(0.25));
    let ami_der0 = ami
        .der_no_collar_micro
        .or(ami.der_no_collar)
        .expect("linux ami DER0");
    assert!(
        (22.0..28.0).contains(&ami_der0),
        "linux ami DER0 micro should be ~24.2%, got {ami_der0}"
    );
    assert!(ami.status.contains("operational"));
}

#[test]
fn der_baseline_linux_native_rows_are_unmeasured_ceilings() {
    let parsed = load();
    let vox = &parsed.voxconverse_test_linux_cpu_native;
    assert_eq!(vox.files, Some(232));
    assert_eq!(vox.engine.as_deref(), Some("cli-native"));
    assert!(
        vox.status.contains("unmeasured"),
        "native Vox row must not look operational, got {}",
        vox.status
    );
    assert!(
        vox.filled_by.is_none(),
        "unmeasured ceiling must not claim an artifact"
    );
    let ami = &parsed.ami_test_linux_cpu_native;
    assert_eq!(ami.files, Some(16));
    assert!(
        ami.status.contains("unmeasured"),
        "native AMI row must not look operational, got {}",
        ami.status
    );
    assert!(
        ami.filled_by.is_none(),
        "unmeasured ceiling must not claim an artifact"
    );
}

/// Artifact lock: committed linux-cpu full-split JSON must match baseline rows.
#[test]
fn der_baseline_linux_cpu_matches_committed_artifacts() {
    let parsed = load();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks/results/linux-cpu-der-2026-08-11");

    let vox_art: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("voxconverse-test.json")).expect("linux vox artifact"),
    )
    .expect("parse linux vox artifact JSON");
    let ami_art: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("ami-test.json")).expect("linux ami artifact"),
    )
    .expect("parse linux ami artifact JSON");

    let vox_bl = parsed
        .voxconverse_test_linux_cpu
        .der_no_collar_micro
        .expect("baseline");
    let ami_bl = parsed
        .ami_test_linux_cpu
        .der_no_collar_micro
        .expect("baseline");
    let vox_m = vox_art["der_no_collar_micro"]
        .as_f64()
        .expect("vox der_no_collar_micro f64");
    let ami_m = ami_art["der_no_collar_micro"]
        .as_f64()
        .expect("ami der_no_collar_micro f64");
    assert!(
        (vox_m - vox_bl).abs() < 0.02,
        "vox artifact {vox_m} vs baseline {vox_bl}"
    );
    assert!(
        (ami_m - ami_bl).abs() < 0.02,
        "ami artifact {ami_m} vs baseline {ami_bl}"
    );
    assert_eq!(vox_art["resolved_execution_provider"], "Cpu");
    assert_eq!(ami_art["resolved_execution_provider"], "Cpu");
    assert_eq!(vox_art["files_processed"], 232);
    assert_eq!(ami_art["files_processed"], 16);
}
