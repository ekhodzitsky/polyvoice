#![allow(clippy::unwrap_used)]
//! Compare ResNet34 vs CAM++ embedders on aorju via Hybrid pipeline.
//!
//! Run with:
//!   cargo test --release --test hybrid_embedder_sweep \
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

use polyvoice::clusterer::AhcClusterer;
use polyvoice::der::compute_der;
use polyvoice::embedder::{CamPlusPlusExtractor, ResNet34Adapter};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::hybrid::HybridPipeline;
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::segmentation::PowersetSegmenter;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use std::path::Path;

#[test]
#[ignore = "requires ONNX models + wav/rttm under data/voxconverse-test/"]
fn hybrid_embedder_sweep_on_aorju() {
    let stem = "aorju";
    let audio_dir = Path::new("data/voxconverse-test/audio");
    let rttm_dir = Path::new("data/voxconverse-test/rttm");

    let wav_path = audio_dir.join(format!("{stem}.wav"));
    let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
    let (samples, sr) = read_wav(&wav_path).expect("read wav");
    assert_eq!(sr, 16000);

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

    let registry = ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(Profile::Balanced)
        .expect("models");

    let pool_size = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("\nEmbedder sweep on aorju:");
    println!(
        "{:>20} {:>10} {:>10} {:>8} {:>8} {:>10}",
        "embedder", "chunks", "speakers", "DER%", "miss%", "conf%"
    );

    // ResNet34 (256-d).
    {
        let embedder = ResNet34Adapter::new(
            &models.embedder_path,
            pool_size,
            polyvoice::onnx::ExecutionProvider::Cpu,
        )
        .expect("resnet");
        let pipeline = HybridPipeline::new(
            Box::new(PowersetSegmenter::new(&models.segmenter_path).unwrap()),
            Box::new(embedder),
            Box::new(AhcClusterer::with_threshold(20, 0.40)),
        );
        let result = pipeline
            .run(&samples, SampleRate::new(16000).unwrap())
            .expect("run");
        let der = compute_der(&ref_turns, &result.turns, 0.25);
        println!(
            "{:>20} {:>10} {:>10} {:>8.2} {:>8.2} {:>10.2}",
            "ResNet34",
            result.turns.len(),
            result.num_speakers,
            der.der * 100.0,
            der.miss_rate * 100.0,
            der.confusion_rate * 100.0,
        );
    }

    // CAM++ 512-d.
    {
        let cam_path = Path::new("models/cam_pp_fp32.onnx");
        if cam_path.is_file() {
            let embedder = CamPlusPlusExtractor::new(
                cam_path,
                512,
                pool_size,
                polyvoice::onnx::ExecutionProvider::Cpu,
            )
            .expect("cam++");
            let pipeline = HybridPipeline::new(
                Box::new(PowersetSegmenter::new(&models.segmenter_path).unwrap()),
                Box::new(embedder),
                Box::new(AhcClusterer::with_threshold(20, 0.40)),
            );
            let result = pipeline
                .run(&samples, SampleRate::new(16000).unwrap())
                .expect("run");
            let der = compute_der(&ref_turns, &result.turns, 0.25);
            println!(
                "{:>20} {:>10} {:>10} {:>8.2} {:>8.2} {:>10.2}",
                "CAM++512",
                result.turns.len(),
                result.num_speakers,
                der.der * 100.0,
                der.miss_rate * 100.0,
                der.confusion_rate * 100.0,
            );
        } else {
            println!("{:>20} {:>10}", "CAM++512", "N/A (no model)");
        }
    }
}
