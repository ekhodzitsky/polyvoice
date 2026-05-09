//! E2E smoke test for the legacy v0.5 pipeline.
//!
//! Requires the Balanced ONNX bundle to be cached (run
//! `cargo run --features cli --bin polyvoice -- download-models --profile balanced`
//! once before invoking with `cargo test -- --ignored e2e`). Picks one WAV
//! from `data/voxconverse-test/audio/`, runs the legacy pipeline, and asserts
//! DER < 50%.

#![cfg(all(feature = "onnx", feature = "download"))]

use polyvoice::der::compute_der;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::Pipeline;
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::types::{DiarizationConfig, Profile};
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
use polyvoice::{FbankOnnxExtractor, SileroVad};
use std::path::Path;

fn first_voxconverse_wav() -> Option<std::path::PathBuf> {
    let dir = Path::new("data/voxconverse-test/audio");
    if !dir.is_dir() {
        return None;
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "wav"))
        .map(|e| e.path())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

#[ignore = "requires cached ONNX bundle + a wav file under data/voxconverse-test/audio/"]
#[test]
fn e2e_smoke_single_file_der_below_50_percent() {
    let wav_path = match first_voxconverse_wav() {
        Some(p) => p,
        None => panic!(
            "data/voxconverse-test/audio/ is empty — run scripts/download-voxconverse-test.sh first"
        ),
    };
    let stem = wav_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

    let (samples, sr_hz) =
        read_wav(&wav_path).expect("WAV read failure — check the file is 16 kHz mono");
    assert_eq!(sr_hz, 16000, "only 16 kHz WAVs supported");

    let registry = ModelRegistry::default()
        .expect("default ModelRegistry should resolve a writable cache dir");
    let models = registry.ensure_for_profile(Profile::Balanced).expect(
        "Balanced profile models should be available — \
             run `polyvoice download-models --profile balanced` first",
    );

    let embedding_dim = Profile::Balanced.embedding_dim();
    let extractor =
        FbankOnnxExtractor::new(&models.embedder_path, embedding_dim, 1).expect("load embedder");
    let mut vad = SileroVad::new(&models.segmenter_path, 512).expect("load vad");

    let config = DiarizationConfig::default();
    let vad_config = VadConfig::default();
    let pipeline = Pipeline::new(config, vad_config);

    let result = pipeline
        .run(&samples, &extractor, &mut vad)
        .expect("pipeline.run on a real VoxConverse clip should succeed");

    let rttm_path = Path::new("data/voxconverse-test/rttm").join(format!("{stem}.rttm"));
    let ref_turns = {
        let raw = parse_rttm_file(&rttm_path).expect("parse rttm");
        let grouped = group_by_file(&raw);
        let segs: Vec<_> = grouped
            .get(stem)
            .map(|v| v.iter().map(|s| (*s).clone()).collect())
            .unwrap_or_default();
        let (turns, _map) = to_speaker_turns(&segs);
        turns
    };

    let der = compute_der(&ref_turns, &result.turns, 0.25);
    println!("{stem}: DER={:.2}%", der.der * 100.0);
    assert!(
        der.der < 0.50,
        "expected DER < 50%, got {:.2}% for {stem}",
        der.der * 100.0,
    );
}
