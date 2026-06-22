#![allow(clippy::unwrap_used)]
//! Overlap-path measurement for Pipeline v2 (resegment_overlap = ON).
//!
//! Isolates the overlap-region DER so the segmentation-derived overlap path can
//! be A/B'd against the legacy mixed-embedding path:
//!
//!   # new (segmentation-derived) path
//!   cargo test --test v2_overlap_measure --features "onnx,segmentation,embedder,clusterer,resegmentation,download" -- --ignored --nocapture
//!   # legacy (mixed-embedding) path
//!   POLYVOICE_V2_DISABLE_SEG_OVERLAP=1 cargo test --test v2_overlap_measure --features "onnx,segmentation,embedder,clusterer,resegmentation,download" -- --ignored --nocapture

#![cfg(all(
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
    feature = "download",
))]

use polyvoice::der::{DerDecomposition, compute_der_decomposition};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::{Pipeline, PipelineConfig};
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use std::path::Path;

/// Runs the v2 pipeline with overlap reconstruction ON and returns the
/// overlap-aware DER decomposition plus (hyp_speakers, ref_speakers).
fn run_overlap_on(
    stem: &str,
    audio_dir: &Path,
    rttm_dir: &Path,
) -> (DerDecomposition, usize, usize) {
    let registry = ModelRegistry::default().expect("registry");
    let config = PipelineConfig {
        profile: Profile::Balanced,
        sample_rate: SampleRate::new(16000).unwrap(),
        resegment_overlap: true,
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

    let (samples, sr_hz) = read_wav(&wav_path).expect("WAV read");
    assert_eq!(sr_hz, 16000, "only 16 kHz WAVs supported");

    let result = pipeline
        .run(&samples, SampleRate::new(16000).unwrap())
        .expect("pipeline.run");

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

    let decomp = compute_der_decomposition(&ref_turns, &result.turns, 0.25);
    let ref_speakers = ref_turns
        .iter()
        .map(|t| t.speaker.0)
        .collect::<std::collections::HashSet<_>>()
        .len();
    (decomp, result.num_speakers, ref_speakers)
}

fn report(label: &str, d: &DerDecomposition, hyp: usize, refn: usize) {
    let path = if std::env::var_os("POLYVOICE_V2_DISABLE_SEG_OVERLAP").is_some() {
        "LEGACY mixed-embedding"
    } else {
        "segmentation-derived"
    };
    println!(
        "[{path}] {label}: DER={:.2}% | overlap-region DER={:.2}% | overlap-excluded DER={:.2}% | confusion={:.2}% | speakers={hyp} ref={refn}",
        d.total.der * 100.0,
        d.overlap.der * 100.0,
        d.single_speaker.der * 100.0,
        d.total.confusion_rate * 100.0,
    );
}

const VOX_5: &[&str] = &["aepyx", "aggyz", "aiqwk", "aorju", "auzru"];

#[test]
#[ignore = "requires cached ONNX bundle + wav/rttm under data/voxconverse-test/"]
fn measure_overlap_vox5() {
    let audio_dir = Path::new("data/voxconverse-test/audio");
    let rttm_dir = Path::new("data/voxconverse-test/rttm");
    let (mut sum_total, mut sum_overlap, mut sum_single) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut n = 0_usize;
    for stem in VOX_5 {
        let (d, hyp, refn) = run_overlap_on(stem, audio_dir, rttm_dir);
        report(stem, &d, hyp, refn);
        sum_total += d.total.der;
        sum_overlap += d.overlap.der;
        sum_single += d.single_speaker.der;
        n += 1;
    }
    println!(
        "VOX5 mean: DER={:.2}% | overlap-region DER={:.2}% | overlap-excluded DER={:.2}% (n={n})",
        sum_total / n as f64 * 100.0,
        sum_overlap / n as f64 * 100.0,
        sum_single / n as f64 * 100.0,
    );
}

#[test]
#[ignore = "requires cached ONNX bundle + AMI EN2002a under data/ami-test-single/"]
fn measure_overlap_ami_en2002a() {
    let (d, hyp, refn) = run_overlap_on(
        "EN2002a",
        Path::new("data/ami-test-single/audio"),
        Path::new("data/ami-test-single/rttm"),
    );
    report("AMI/EN2002a", &d, hyp, refn);
    for r in &d.per_speaker_recall {
        println!(
            "  ref spk {} recall={:.1}% ({}/{} frames)",
            r.speaker,
            r.recall * 100.0,
            r.recalled_frames,
            r.ref_frames
        );
    }
}
