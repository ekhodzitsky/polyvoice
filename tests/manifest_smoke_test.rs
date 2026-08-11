#![allow(clippy::unwrap_used)]
//! Manifest smoke tests over the production `src/models/manifest.toml`.
//!
//! Verifies that after the model publish step:
//!   - `[profiles.mobile]` / `[profiles.balanced]` resolve to the FP32 pair.
//!   - `[profiles.fast]` resolves to the recalibrated INT8 pair.
//!   - Every INT8 sha256 is a real 64-char hex digest (not a placeholder).
//!   - Mobile profile total bundle is ≤ MOBILE_BUNDLE_BUDGET_BYTES.
//!   - Balanced profile total bundle is ≤ 35 MB.

#![cfg(feature = "download")]

use polyvoice::models::Manifest;

const MANIFEST_TOML: &str = include_str!("../src/models/manifest.toml");

/// Bundle ceiling for legacy FP32 profiles (Mobile + Balanced).
const BUNDLE_BUDGET_BYTES: u64 = 35_000_000;

fn parse() -> Manifest {
    Manifest::from_toml_str(MANIFEST_TOML).expect("manifest.toml must parse cleanly")
}

#[test]
fn manifest_contains_all_three_int8_entries() {
    let m = parse();
    for id in ["powerset_int8", "cam_pp_int8", "resnet34_int8"] {
        assert!(m.model(id).is_some(), "missing model entry: {id}");
    }
}

#[test]
fn int8_sha256_is_real_not_placeholder() {
    let m = parse();
    for id in ["powerset_int8", "cam_pp_int8", "resnet34_int8"] {
        let entry = m.model(id).expect(id);
        assert_eq!(entry.sha256.len(), 64, "{id} sha256 must be 64 hex chars");
        assert!(
            entry
                .sha256
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "{id} sha256 must be lowercase hex"
        );
        assert_ne!(
            entry.sha256, "0000000000000000000000000000000000000000000000000000000000000000",
            "{id} sha256 must not be all-zero placeholder"
        );
    }
}

#[test]
fn mobile_profile_resolves_to_int8() {
    // Since 0.17 every shipping profile uses the INT8 pair.
    let m = parse();
    let prof = m.profile("mobile").expect("mobile profile present");
    assert_eq!(prof.segmenter, "powerset_int8");
    assert_eq!(prof.embedder, "resnet34_int8");
}

#[test]
fn balanced_profile_resolves_to_int8() {
    let m = parse();
    let prof = m.profile("balanced").expect("balanced profile present");
    assert_eq!(prof.segmenter, "powerset_int8");
    assert_eq!(prof.embedder, "resnet34_int8");
}

#[test]
fn fast_profile_resolves_to_int8() {
    let m = parse();
    let prof = m.profile("fast").expect("fast profile present");
    assert_eq!(prof.segmenter, "powerset_int8");
    assert_eq!(prof.embedder, "resnet34_int8");
}

#[test]
fn mobile_bundle_under_35mb_budget() {
    let m = parse();
    let prof = m.profile("mobile").unwrap();
    let seg = m.model(&prof.segmenter).unwrap();
    let emb = m.model(&prof.embedder).unwrap();
    let total = seg.size.unwrap_or(0) + emb.size.unwrap_or(0);
    assert!(
        total <= BUNDLE_BUDGET_BYTES,
        "mobile bundle {} bytes > {} budget",
        total,
        BUNDLE_BUDGET_BYTES
    );
}

#[test]
fn balanced_bundle_under_35mb_budget() {
    let m = parse();
    let prof = m.profile("balanced").unwrap();
    let seg = m.model(&prof.segmenter).unwrap();
    let emb = m.model(&prof.embedder).unwrap();
    let total = seg.size.unwrap_or(0) + emb.size.unwrap_or(0);
    assert!(
        total <= BUNDLE_BUDGET_BYTES,
        "balanced bundle {} bytes > {} budget",
        total,
        BUNDLE_BUDGET_BYTES
    );
}

#[test]
fn int8_entries_have_calibration_descriptor() {
    let m = parse();
    // powerset_int8 is weights-only dynamic quantization — no calibration
    // data is involved, the field must say so instead of naming a dataset.
    let powerset = m.model("powerset_int8").expect("powerset_int8");
    let calib = powerset.calibration.as_deref().unwrap_or("");
    assert!(
        calib.contains("dynamic"),
        "powerset_int8 calibration field must document dynamic quantization (got '{calib}')"
    );
    // Static-QDQ embedders must name their calibration set.
    for id in ["cam_pp_int8", "resnet34_int8"] {
        let entry = m.model(id).expect(id);
        let calib = entry.calibration.as_deref().unwrap_or("");
        assert!(
            calib.contains("voxconverse_dev"),
            "{id} calibration field must reference voxconverse_dev (got '{calib}')"
        );
    }
}
