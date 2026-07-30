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

mod common;

use polyvoice::der::{DerDecomposition, compute_der_decomposition};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::{Pipeline, PipelineConfig};
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use std::path::Path;

/// Returns (DER decomposition, num_speakers, ref_speakers).
fn run_v2_pipeline_on_file(
    stem: &str,
    audio_dir: &Path,
    rttm_dir: &Path,
) -> (DerDecomposition, usize, usize) {
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

    let ref_turns = common::load_ref_turns(&rttm_path, stem);

    let decomp = compute_der_decomposition(&ref_turns, &result.turns, 0.25);
    (
        decomp,
        result.num_speakers,
        ref_turns
            .iter()
            .map(|t| t.speaker.0)
            .collect::<std::collections::HashSet<_>>()
            .len(),
    )
}

#[test]
#[ignore = "requires downloaded models"]
fn v2_der_e2e_smoke() {
    let audio_dir = Path::new("tests/data/e2e-smoke/audio");
    let rttm_dir = Path::new("tests/data/e2e-smoke/rttm");
    if !common::require_wav(&audio_dir.join("fuzfh.wav")) {
        return;
    }
    let (decomp, num_speakers, ref_speakers) =
        run_v2_pipeline_on_file("fuzfh", audio_dir, rttm_dir);
    println!(
        "e2e_smoke: DER={:.2}% speakers={} ref_speakers={}",
        decomp.total.der * 100.0,
        num_speakers,
        ref_speakers
    );
}

#[test]
#[ignore = "requires downloaded models and dataset"]
fn v2_der_voxconverse_10_file_subset() {
    let audio_dir = Path::new("data/voxconverse-test/audio");
    let rttm_dir = Path::new("data/voxconverse-test/rttm");
    if !common::require_wav(audio_dir) {
        return;
    }

    let mut total_der = 0.0_f64;
    let mut count = 0_usize;

    for stem in common::VOXCONVERSE_SUBSET_10 {
        let (decomp, num_speakers, ref_speakers) =
            run_v2_pipeline_on_file(stem, audio_dir, rttm_dir);
        println!(
            "{stem}: DER={:.2}% speakers={} ref_speakers={}",
            decomp.total.der * 100.0,
            num_speakers,
            ref_speakers
        );
        total_der += decomp.total.der;
        count += 1;
    }

    assert!(count > 0, "no files processed");
    let avg_der = total_der / count as f64;
    println!("Average DER over {count} files: {:.2}%", avg_der * 100.0);
}

#[test]
#[ignore = "requires downloaded models and dataset"]
fn v2_der_ami_test_single() {
    let audio_dir = Path::new("data/ami-test-single/audio");
    let rttm_dir = Path::new("data/ami-test-single/rttm");
    if !common::require_wav(audio_dir) {
        return;
    }
    let (decomp, num_speakers, ref_speakers) =
        run_v2_pipeline_on_file("EN2002a", audio_dir, rttm_dir);
    let single_der = decomp.single_speaker.der;
    let confusion = decomp.total.confusion_rate;
    // Overlap-aware decomposition: the AMI gate references the split so a
    // regression is interpretable — total DER alone hides where the error comes from.
    println!(
        "ami_test_single: DER={:.2}% overlap-excluded DER={:.2}% overlap-region DER={:.2}% confusion={:.2}% speakers={} ref_speakers={}",
        decomp.total.der * 100.0,
        single_der * 100.0,
        decomp.overlap.der * 100.0,
        confusion * 100.0,
        num_speakers,
        ref_speakers
    );
    for r in &decomp.per_speaker_recall {
        println!(
            "  ref spk {} recall={:.1}% ({}/{} frames)",
            r.speaker,
            r.recall * 100.0,
            r.recalled_frames,
            r.ref_frames
        );
    }
    // Shared AMI long-form gate (speaker-count collapse + clustering confusion
    // + overlap-excluded DER floor); mirrors cli_der_regression_test::cli_der_regression_v2_ami_single.
    let baseline = common::load_baseline(&common::der_baseline_path());
    common::gate_ami_longform(
        num_speakers,
        confusion,
        single_der,
        &baseline.hybrid_ami_test_single,
    );
}
