#![allow(clippy::unwrap_used)]
//! DER baseline measurement for Pipeline v2.
//!
//! Run with:
//!   cargo test --test der_v2_baseline_test --features "onnx,segmentation,embedder,clusterer,resegmentation,download" -- --ignored --nocapture

#![cfg(all(
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
    feature = "download",
))]

use polyvoice::der::compute_der;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::{Pipeline, PipelineConfig};
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use std::path::Path;

const SUBSET_10: &[&str] = &[
    "aepyx", "aggyz", "aiqwk", "aorju", "auzru", "bgvvt", "bidnq", "bjruf", "bmsyn", "bpzsc",
];

fn run_v2_pipeline_on_file(stem: &str, audio_dir: &Path, rttm_dir: &Path) -> (f64, usize, usize) {
    let registry = ModelRegistry::default().expect("registry");
    let config = PipelineConfig {
        profile: Profile::Balanced,
        sample_rate: SampleRate::new(16000).unwrap(),
        resegment_overlap: false,
        ..PipelineConfig::default()
    };
    let pipeline = Pipeline::builder()
        .config(config)
        .profile(Profile::Balanced)
        .with_models_from(registry)
        .build()
        .expect("pipeline build");

    let wav_path = audio_dir.join(format!("{stem}.wav"));
    let wav_path_alt = audio_dir.join(format!("{stem}.Mix-Headset.wav"));
    let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
    let rttm_path_alt = rttm_dir.join(format!("{stem}.Mix-Headset.rttm"));
    let wav_path = if wav_path.is_file() {
        wav_path
    } else {
        wav_path_alt
    };
    let rttm_path = if rttm_path.is_file() {
        rttm_path
    } else {
        rttm_path_alt
    };

    let (samples, sr_hz) = read_wav(&wav_path).expect("WAV read failure");
    assert_eq!(sr_hz, 16000, "only 16 kHz WAVs supported");

    let result = pipeline
        .run(&samples, SampleRate::new(16000).unwrap())
        .expect("pipeline.run should succeed");

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
    (
        der.der,
        result.num_speakers,
        ref_turns
            .iter()
            .map(|t| t.speaker.0)
            .collect::<std::collections::HashSet<_>>()
            .len(),
    )
}

#[test]
#[ignore = "requires cached ONNX bundle + wav/rttm files"]
fn v2_der_e2e_smoke() {
    let (der, num_speakers, ref_speakers) = run_v2_pipeline_on_file(
        "fuzfh",
        Path::new("tests/data/e2e-smoke/audio"),
        Path::new("tests/data/e2e-smoke/rttm"),
    );
    println!(
        "e2e_smoke: DER={:.2}% speakers={} ref_speakers={}",
        der * 100.0,
        num_speakers,
        ref_speakers
    );
}

#[test]
#[ignore = "requires cached ONNX bundle + wav/rttm files under data/voxconverse-test/"]
fn v2_der_voxconverse_10_file_subset() {
    let audio_dir = Path::new("data/voxconverse-test/audio");
    let rttm_dir = Path::new("data/voxconverse-test/rttm");

    let mut total_der = 0.0_f64;
    let mut count = 0_usize;

    for stem in SUBSET_10 {
        let (der, num_speakers, ref_speakers) = run_v2_pipeline_on_file(stem, audio_dir, rttm_dir);
        println!(
            "{stem}: DER={:.2}% speakers={} ref_speakers={}",
            der * 100.0,
            num_speakers,
            ref_speakers
        );
        total_der += der;
        count += 1;
    }

    assert!(count > 0, "no files processed");
    let avg_der = total_der / count as f64;
    println!("Average DER over {count} files: {:.2}%", avg_der * 100.0);
}

#[test]
#[ignore = "requires cached ONNX bundle + wav/rttm files under data/ami-test-single/"]
fn v2_der_ami_test_single() {
    let (der, num_speakers, ref_speakers) = run_v2_pipeline_on_file(
        "EN2002a",
        Path::new("data/ami-test-single/audio"),
        Path::new("data/ami-test-single/rttm"),
    );
    println!(
        "ami_test_single: DER={:.2}% speakers={} ref_speakers={}",
        der * 100.0,
        num_speakers,
        ref_speakers
    );
}
