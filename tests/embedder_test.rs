#![allow(deprecated)] // legacy embedding API (F09); see polyvoice::embedder
//! Integration test for `CamPlusPlusExtractor` and `ResNet34Adapter` against
//! real upstream ONNX models.
//!
//! Runs only when explicitly invoked:
//!   cargo test --features onnx,embedder,download --test embedder_test -- --ignored
//!
//! Downloads ~55 MB of models. Requires network connectivity.

#![cfg(all(feature = "onnx", feature = "embedder", feature = "download"))]
#![allow(clippy::expect_used)]

use polyvoice::embedder::{CamPlusPlusExtractor, Embedder, ResNet34Adapter};
use polyvoice::models::ModelRegistry;
use tempfile::TempDir;

/// 1 second of synthetic 16 kHz mono audio (220 Hz tone).
fn synthetic_audio_1s() -> Vec<f32> {
    use std::f32::consts::PI;
    let sr = 16_000_usize;
    let mut audio = Vec::with_capacity(sr);
    for i in 0..sr {
        let t = i as f32 / sr as f32;
        audio.push((2.0 * PI * 220.0 * t).sin() * 0.3);
    }
    audio
}

#[test]
#[ignore = "real network — run with --ignored"]
fn cam_plus_plus_extractor_produces_512d_normalized_embedding() {
    let tmp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::with_cache_dir(tmp.path()).expect("registry");
    let model_path = registry
        .ensure("cam_pp_fp32")
        .expect("download must succeed");

    // The WeSpeaker voxceleb_CAM++ ONNX outputs 512-d.
    let extractor = CamPlusPlusExtractor::new(&model_path, 512, 1).expect("loads");
    assert_eq!(extractor.dim(), 512);

    let embedding = extractor.embed(&synthetic_audio_1s()).expect("embed runs");
    assert_eq!(embedding.len(), 512);

    // L2 norm should be ~1.0 (the underlying FbankOnnxExtractor L2-normalizes).
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-2, "L2 norm not 1.0: {norm}");
}

#[test]
#[ignore = "real network — run with --ignored"]
fn resnet34_adapter_produces_256d_normalized_embedding() {
    let tmp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::with_cache_dir(tmp.path()).expect("registry");
    let model_path = registry.ensure("wespeaker_resnet34").expect("download");

    let extractor = ResNet34Adapter::new(&model_path, 1).expect("loads");
    assert_eq!(extractor.dim(), 256);

    let embedding = extractor.embed(&synthetic_audio_1s()).expect("embed");
    assert_eq!(embedding.len(), 256);

    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-2, "L2 norm not 1.0: {norm}");
}
