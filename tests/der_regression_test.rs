//! DER regression test on a fixed 10-file subset using the legacy v0.5 pipeline.
//!
//! Requires the Balanced ONNX bundle to be cached (run
//! `cargo run --features cli --bin polyvoice -- download-models --profile balanced`
//! once before invoking with `cargo test -- --ignored der`). Computes average
//! DER across 10 alphabetically-first VoxConverse-test files and asserts it
//! stays below 25%.

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

const SUBSET_10: &[&str] = &[
    "aepyx", "aggyz", "aiqwk", "aorju", "auzru", "bgvvt", "bidnq", "bjruf", "bmsyn", "bpzsc",
];

#[ignore = "requires cached ONNX bundle + wav/rttm files under data/voxconverse-test/"]
#[test]
fn der_regression_10_file_subset_below_25_percent() {
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

    let audio_dir = Path::new("data/voxconverse-test/audio");
    let rttm_dir = Path::new("data/voxconverse-test/rttm");

    let mut total_der = 0.0_f64;
    let mut count = 0_usize;

    for stem in SUBSET_10 {
        let wav_path = audio_dir.join(format!("{stem}.wav"));
        let rttm_path = rttm_dir.join(format!("{stem}.rttm"));

        assert!(wav_path.is_file(), "WAV not found: {}", wav_path.display());
        assert!(
            rttm_path.is_file(),
            "RTTM not found: {}",
            rttm_path.display()
        );

        let (samples, sr_hz) = read_wav(&wav_path).expect("WAV read failure");
        assert_eq!(sr_hz, 16000, "only 16 kHz WAVs supported");

        let result = pipeline
            .run(&samples, &extractor, &mut vad)
            .expect("pipeline.run should succeed");

        let ref_turns = {
            let raw = parse_rttm_file(&rttm_path).expect("parse rttm");
            let grouped = group_by_file(&raw);
            let segs: Vec<_> = grouped
                .get(*stem)
                .map(|v| v.iter().map(|s| (*s).clone()).collect())
                .unwrap_or_default();
            let (turns, _map) = to_speaker_turns(&segs);
            turns
        };

        let der = compute_der(&ref_turns, &result.turns, 0.25);
        println!("{stem}: DER={:.2}%", der.der * 100.0);
        total_der += der.der;
        count += 1;
    }

    assert!(count > 0, "no files processed");
    let avg_der = total_der / count as f64;
    println!("Average DER over {count} files: {:.2}%", avg_der * 100.0);
    assert!(
        avg_der < 0.25,
        "expected average DER < 25%, got {:.2}%",
        avg_der * 100.0,
    );
}
