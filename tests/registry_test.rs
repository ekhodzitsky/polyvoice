#![allow(clippy::unwrap_used)]
//! Integration test for ModelRegistry against the real upstream URLs.
//!
//! Runs only when explicitly invoked:
//!   cargo test --features download --test registry_test -- --ignored
//!
//! The download is ~28 MB total. Requires network connectivity.

#![cfg(feature = "download")]
#![allow(clippy::expect_used)]

use polyvoice::models::ModelRegistry;
use polyvoice::types::Profile;
use tempfile::TempDir;

#[test]
#[ignore = "real network — run with --ignored"]
fn ensure_for_profile_mobile_downloads_and_verifies() {
    let tmp = TempDir::new().unwrap();
    let r = ModelRegistry::with_cache_dir(tmp.path()).unwrap();

    let bundle = r
        .ensure_for_profile(Profile::Mobile)
        .expect("download must succeed");

    assert!(bundle.segmenter_path.exists());
    assert!(bundle.embedder_path.exists());

    let seg_size = std::fs::metadata(&bundle.segmenter_path).unwrap().len();
    let emb_size = std::fs::metadata(&bundle.embedder_path).unwrap().len();
    assert!(seg_size > 1_000_000, "silero ~2.3MB");
    assert!(emb_size > 20_000_000, "wespeaker ~26MB");

    // Second call should be a no-op (idempotent cache hit, no download).
    let bundle2 = r.ensure_for_profile(Profile::Mobile).unwrap();
    assert_eq!(bundle2.segmenter_path, bundle.segmenter_path);
}

#[test]
#[ignore = "real network — run with --ignored"]
fn ensure_for_profile_custom_returns_explicit_error() {
    let tmp = TempDir::new().unwrap();
    let r = ModelRegistry::with_cache_dir(tmp.path()).unwrap();
    let err = r
        .ensure_for_profile(Profile::Custom)
        .expect_err("must reject");
    let msg = format!("{err}");
    assert!(msg.contains("custom"), "got: {msg}");
}
