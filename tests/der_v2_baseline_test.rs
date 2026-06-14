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

use polyvoice::der::{DerDecomposition, compute_der_decomposition};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::{Pipeline, PipelineConfig};
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use serde::Deserialize;
use std::path::Path;

const SUBSET_10: &[&str] = &[
    "aepyx", "aggyz", "aiqwk", "aorju", "auzru", "bgvvt", "bidnq", "bjruf", "bmsyn", "bpzsc",
];

/// v2-family AMI baseline entry; hosts the overlap-excluded long-form floor (task 110/F37).
#[derive(Deserialize)]
struct Baseline {
    #[serde(rename = "hybrid_ami_test_single")]
    ami_v2: AmiBaseline,
}

#[derive(Deserialize)]
struct AmiBaseline {
    /// Overlap-excluded DER floor. `None` (JSON null) = not yet measured → gate inactive.
    der_single_speaker: Option<f64>,
    der_single_speaker_tolerance: Option<f64>,
}

fn load_baseline() -> Baseline {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json");
    let raw = std::fs::read_to_string(&path).expect("read der_baseline.json");
    serde_json::from_str(&raw).expect("parse der_baseline.json")
}

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
#[ignore = "requires cached ONNX bundle + wav/rttm files"]
fn v2_der_e2e_smoke() {
    let (decomp, num_speakers, ref_speakers) = run_v2_pipeline_on_file(
        "fuzfh",
        Path::new("tests/data/e2e-smoke/audio"),
        Path::new("tests/data/e2e-smoke/rttm"),
    );
    println!(
        "e2e_smoke: DER={:.2}% speakers={} ref_speakers={}",
        decomp.total.der * 100.0,
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
#[ignore = "requires cached ONNX bundle + wav/rttm files under data/ami-test-single/"]
fn v2_der_ami_test_single() {
    let (decomp, num_speakers, ref_speakers) = run_v2_pipeline_on_file(
        "EN2002a",
        Path::new("data/ami-test-single/audio"),
        Path::new("data/ami-test-single/rttm"),
    );
    let single_der = decomp.single_speaker.der;
    let confusion = decomp.total.confusion_rate;
    // Overlap-aware decomposition (task 111/F37): the AMI gate references the split so a
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
    // The NaN-embedding collapse manifested as num_speakers=1 — every segment merged
    // into a single cluster. Total DER is deliberately NOT gated here: AMI EN2002a is
    // ~79% overlapping speech, and with single-speaker-per-frame output the miss term
    // alone holds DER near 88% whether the bug is present or fixed, so a DER ceiling
    // cannot tell the two apart. Gate instead on the signals that actually move when
    // the bug regresses: the speaker count must not collapse, and clustering confusion
    // must stay low (post-fix it is ~11%).
    assert!(
        num_speakers >= 2,
        "pipeline_v2 collapsed to {num_speakers} speaker(s) on EN2002a (NaN-embedding regression?)"
    );
    assert!(
        confusion < 0.25,
        "pipeline_v2 clustering regressed on EN2002a: confusion={:.1}% exceeds 25%",
        confusion * 100.0
    );
    // Numeric long-form floor (task 110/F37): the overlap-excluded DER DOES discriminate
    // healthy vs collapsed diarization on high-overlap audio, unlike total DER. The gate
    // activates only once a baseline is measured and recorded in tests/der_baseline.json
    // (hybrid_ami_test_single.der_single_speaker); until then the printed value is the
    // measurement to record.
    let baseline = load_baseline();
    match (
        baseline.ami_v2.der_single_speaker,
        baseline.ami_v2.der_single_speaker_tolerance,
    ) {
        (Some(floor), Some(tol)) => {
            let ceiling = (floor + tol) / 100.0;
            assert!(
                single_der <= ceiling,
                "long-form floor regressed: overlap-excluded DER={:.2}% exceeds {:.2}% (baseline {:.2}% + tol {:.2}%)",
                single_der * 100.0,
                ceiling * 100.0,
                floor,
                tol,
            );
        }
        _ => println!(
            "  overlap-excluded DER baseline not yet measured — record {:.2}% in tests/der_baseline.json to activate the long-form floor",
            single_der * 100.0
        ),
    }
}
