#![allow(clippy::unwrap_used)]
//! Full VoxConverse-test regression for Hybrid pipeline (232 files).
//!
//! Run with:
//!   cargo test --release --test hybrid_voxconverse_full \
//!     --features "onnx,segmentation,embedder,clusterer,resegmentation,download" \
//!     -- --ignored --nocapture

#![cfg(all(
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
    feature = "download",
))]

use polyvoice::clusterer::KMeansClusterer;
use polyvoice::embedder::ResNet34Adapter;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::hybrid::HybridPipeline;
use polyvoice::segmentation::PowersetSegmenter;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use std::fs;
use std::path::Path;

#[test]
#[ignore = "requires ONNX models + full VoxConverse-test dataset (~7 hours in release)"]
fn hybrid_voxconverse_full_regression() {
    let audio_dir = Path::new("data/voxconverse-test/audio");
    let rttm_dir = Path::new("data/voxconverse-test/rttm");

    let registry = ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(Profile::Balanced)
        .expect("models");

    let segmenter = PowersetSegmenter::new(&models.segmenter_path).expect("segmenter");
    let pool_size = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let embedder = ResNet34Adapter::new(&models.embedder_path, pool_size).expect("embedder");
    let clusterer = KMeansClusterer::new(20);

    let pipeline =
        HybridPipeline::new(Box::new(segmenter), Box::new(embedder), Box::new(clusterer));

    let mut total_der = 0.0_f64;
    let mut total_miss = 0.0_f64;
    let mut total_fa = 0.0_f64;
    let mut total_conf = 0.0_f64;
    let mut count = 0_usize;

    let entries: Vec<_> = fs::read_dir(audio_dir)
        .expect("read audio dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "wav"))
        .collect();

    for entry in entries {
        let wav_path = entry.path();
        let stem = wav_path.file_stem().unwrap().to_str().unwrap();
        let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
        if !rttm_path.is_file() {
            println!("{stem}: skipping (no RTTM)");
            continue;
        }

        let (samples, sr) = read_wav(&wav_path).expect("read wav");
        assert_eq!(sr, 16000);

        let result = pipeline
            .run(&samples, SampleRate::new(16000).unwrap())
            .expect("run");

        let ref_turns = {
            let raw = polyvoice::rttm::parse_rttm_file(&rttm_path).expect("parse rttm");
            let grouped = polyvoice::rttm::group_by_file(&raw);
            let segs: Vec<_> = grouped
                .get(stem)
                .map(|v| v.iter().map(|s| (*s).clone()).collect())
                .unwrap_or_default();
            let (turns, _map) = polyvoice::rttm::to_speaker_turns(&segs);
            turns
        };

        let der = polyvoice::der::compute_der(&ref_turns, &result.turns, 0.25);
        println!(
            "{}: DER={:.2}% miss={:.2}% fa={:.2}% conf={:.2}% speakers={} ref={}",
            stem,
            der.der * 100.0,
            der.miss_rate * 100.0,
            der.false_alarm_rate * 100.0,
            der.confusion_rate * 100.0,
            result.num_speakers,
            ref_turns
                .iter()
                .map(|t| t.speaker.0)
                .max()
                .map_or(0, |m| m + 1),
        );

        total_der += der.der;
        total_miss += der.miss_rate;
        total_fa += der.false_alarm_rate;
        total_conf += der.confusion_rate;
        count += 1;
    }

    assert!(count > 0, "No files processed");
    let avg_der = total_der / count as f64;
    let avg_miss = total_miss / count as f64;
    let avg_fa = total_fa / count as f64;
    let avg_conf = total_conf / count as f64;

    println!("\n=== Summary ===");
    println!(
        "Files: {} | Avg DER: {:.2}% | miss: {:.2}% | fa: {:.2}% | conf: {:.2}%",
        count,
        avg_der * 100.0,
        avg_miss * 100.0,
        avg_fa * 100.0,
        avg_conf * 100.0,
    );

    // Baseline: legacy VoxConverse full = ~14% DER.
    // K-means auto-k with hop=1.0s: 14.12% DER (232 files).
    assert!(
        avg_der < 0.16,
        "Full VoxConverse avg DER must be < 16%, got {:.2}%",
        avg_der * 100.0
    );
}
