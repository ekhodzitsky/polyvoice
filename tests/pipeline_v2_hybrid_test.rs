//! Hybrid pipeline integration test.
//!
//! PowersetSegmenter as VAD → sliding window → ResNet34 → AHC.
//!
//! Run with:
//!   cargo test --test pipeline_v2_hybrid_test --features "onnx,segmentation,embedder,clusterer,resegmentation,download" -- --ignored --nocapture

#![cfg(all(
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
    feature = "download",
))]

use polyvoice::clusterer::AhcClusterer;
use polyvoice::der::compute_der;
use polyvoice::embedder::ResNet34Adapter;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::hybrid::HybridPipeline;
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::segmentation::PowersetSegmenter;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use std::path::Path;

#[allow(clippy::needless_borrow)]
fn run_hybrid_on_file(stem: &str, audio_dir: &Path, rttm_dir: &Path) -> (f64, usize, usize) {
    let registry = ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(Profile::Balanced)
        .expect("models");

    let segmenter = PowersetSegmenter::new(&models.segmenter_path).expect("segmenter");
    let pool_size = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let embedder = ResNet34Adapter::new(&models.embedder_path, pool_size).expect("embedder");
    let clusterer = AhcClusterer::with_threshold(20, 0.35);

    let pipeline =
        HybridPipeline::new(Box::new(segmenter), Box::new(embedder), Box::new(clusterer));

    let wav_path = audio_dir.join(format!("{stem}.wav"));
    let rttm_path = rttm_dir.join(format!("{stem}.rttm"));

    let (samples, sr) = read_wav(&wav_path).expect("read wav");
    assert_eq!(sr, 16000);

    let result = pipeline
        .run(&samples, SampleRate::new(16000).unwrap())
        .expect("run");

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
    let ref_speakers = ref_turns
        .iter()
        .map(|t| t.speaker.0)
        .collect::<std::collections::HashSet<_>>()
        .len();
    (der.der, result.num_speakers, ref_speakers)
}

#[test]
#[ignore = "requires ONNX models + wav/rttm"]
fn hybrid_e2e_smoke() {
    let (der, num_speakers, ref_speakers) = run_hybrid_on_file(
        "fuzfh",
        Path::new("tests/data/e2e-smoke/audio"),
        Path::new("tests/data/e2e-smoke/rttm"),
    );
    println!(
        "Hybrid e2e_smoke: DER={:.2}% speakers={} ref={}",
        der * 100.0,
        num_speakers,
        ref_speakers
    );
    assert!(der < 0.10, "DER must be < 10%, got {:.2}%", der * 100.0);
}

#[test]
#[ignore = "requires ONNX models + wav/rttm under data/voxconverse-test/"]
fn hybrid_voxconverse_3_file_subset() {
    const SUBSET_3: &[&str] = &["aepyx", "aggyz", "aiqwk"];
    let audio_dir = Path::new("data/voxconverse-test/audio");
    let rttm_dir = Path::new("data/voxconverse-test/rttm");

    let mut total_der = 0.0_f64;
    let mut count = 0_usize;

    for stem in SUBSET_3 {
        let (der, num_speakers, ref_speakers) = run_hybrid_on_file(stem, audio_dir, rttm_dir);
        println!(
            "{}: DER={:.2}% speakers={} ref={}",
            stem,
            der * 100.0,
            num_speakers,
            ref_speakers
        );
        total_der += der;
        count += 1;
    }

    assert!(count > 0);
    let avg = total_der / count as f64;
    println!("Average DER over {} files: {:.2}%", count, avg * 100.0);
    assert!(
        avg < 0.25,
        "Average DER must be < 25%, got {:.2}%",
        avg * 100.0
    );
}

#[test]
#[ignore = "requires ONNX models + wav/rttm under data/voxconverse-test/"]
fn hybrid_voxconverse_10_file_subset() {
    const SUBSET_10: &[&str] = &[
        "aepyx", "aggyz", "aiqwk", "aorju", "auzru", "bgvvt", "bidnq", "bjruf", "bmsyn", "bpzsc",
    ];
    let audio_dir = Path::new("data/voxconverse-test/audio");
    let rttm_dir = Path::new("data/voxconverse-test/rttm");

    let mut total_der = 0.0_f64;
    let mut count = 0_usize;

    for stem in SUBSET_10 {
        let (der, num_speakers, ref_speakers) = run_hybrid_on_file(stem, audio_dir, rttm_dir);
        println!(
            "{}: DER={:.2}% speakers={} ref={}",
            stem,
            der * 100.0,
            num_speakers,
            ref_speakers
        );
        total_der += der;
        count += 1;
    }

    assert!(count > 0);
    let avg = total_der / count as f64;
    println!("Average DER over {} files: {:.2}%", count, avg * 100.0);
    assert!(
        avg < 0.25,
        "Average DER must be < 25%, got {:.2}%",
        avg * 100.0
    );
}

#[test]
#[ignore = "requires ONNX models + wav/rttm under data/ami-test-single/"]
#[allow(clippy::needless_borrow)]
fn hybrid_ami_test_single() {
    let wav_path = Path::new("data/ami-test-single/audio/EN2002a.Mix-Headset.wav");
    let rttm_path = Path::new("data/ami-test-single/rttm/EN2002a.Mix-Headset.rttm");
    let rttm_path_alt = Path::new("data/ami-test-single/rttm/EN2002a.rttm");
    let wav_path = if wav_path.is_file() {
        wav_path
    } else {
        Path::new("data/ami-test-single/audio/EN2002a.wav")
    };
    let rttm_path = if rttm_path.is_file() {
        rttm_path
    } else {
        rttm_path_alt
    };

    let registry = ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(Profile::Balanced)
        .expect("models");

    let segmenter = PowersetSegmenter::new(&models.segmenter_path).expect("segmenter");
    let pool_size = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let embedder = ResNet34Adapter::new(&models.embedder_path, pool_size).expect("embedder");
    let clusterer = AhcClusterer::with_threshold(20, 0.35);

    let pipeline =
        HybridPipeline::new(Box::new(segmenter), Box::new(embedder), Box::new(clusterer));

    let (samples, sr) = read_wav(&wav_path).expect("read wav");
    assert_eq!(sr, 16000);

    let result = pipeline
        .run(&samples, SampleRate::new(16000).unwrap())
        .expect("run");

    let ref_turns = {
        let raw = parse_rttm_file(&rttm_path).expect("parse rttm");
        let grouped = group_by_file(&raw);
        let segs: Vec<_> = grouped
            .get("EN2002a")
            .map(|v| v.iter().map(|s| (*s).clone()).collect())
            .unwrap_or_default();
        let (turns, _map) = to_speaker_turns(&segs);
        turns
    };

    let der = compute_der(&ref_turns, &result.turns, 0.25);
    println!(
        "Hybrid AMI: DER={:.2}% speakers={} ref={}",
        der.der * 100.0,
        result.num_speakers,
        4
    );
    assert!(
        der.der < 0.50,
        "DER must be < 50%, got {:.2}%",
        der.der * 100.0
    );
}
