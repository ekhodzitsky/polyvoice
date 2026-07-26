#![allow(deprecated)] // HybridPipeline is soft-deprecated research surface
#![allow(clippy::unwrap_used)]
//! Compare Pipeline v2 (overlap-aware) vs Hybrid on long-form audio.
//!
//! Run with:
//!   cargo test --test pipeline_v2_vs_hybrid \
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
use polyvoice::embedder::ResNet34Adapter;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::Pipeline;
use polyvoice::pipeline_v2::hybrid::HybridPipeline;
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::segmentation::PowersetSegmenter;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use std::path::Path;

#[test]
#[ignore = "requires ONNX models + wav/rttm under data/voxconverse-test/"]
fn compare_pipeline_v2_vs_hybrid_on_aorju() {
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
    let ref_speakers = ref_turns
        .iter()
        .map(|t| t.speaker.0)
        .collect::<std::collections::HashSet<_>>()
        .len();

    let registry = ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(Profile::Balanced)
        .expect("models");

    let pool_size = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let embedder_hybrid = ResNet34Adapter::new(
        &models.embedder_path,
        pool_size,
        polyvoice::onnx::ExecutionProvider::Cpu,
    )
    .expect("embedder");
    let embedder_v2 = ResNet34Adapter::new(
        &models.embedder_path,
        pool_size,
        polyvoice::onnx::ExecutionProvider::Cpu,
    )
    .expect("embedder");
    let clusterer_hybrid = AhcClusterer::with_threshold(20, 0.40);
    let clusterer_v2 = AhcClusterer::with_threshold(20, 0.40);

    // Hybrid pipeline.
    let hybrid = HybridPipeline::new(
        Box::new(PowersetSegmenter::new(&models.segmenter_path).unwrap()),
        Box::new(embedder_hybrid),
        Box::new(clusterer_hybrid),
    );
    let hybrid_result = hybrid
        .run(&samples, SampleRate::new(16000).unwrap())
        .expect("hybrid run");
    let hybrid_der = compute_der(&ref_turns, &hybrid_result.turns, 0.25);

    // Pipeline v2 (overlap-aware).
    let v2 = Pipeline::builder()
        .profile(Profile::Custom)
        .with_segmenter(Box::new(
            PowersetSegmenter::new(&models.segmenter_path).unwrap(),
        ))
        .with_embedder(Box::new(embedder_v2))
        .with_clusterer(Box::new(clusterer_v2))
        .build()
        .expect("v2 build");
    let v2_result = v2
        .run(&samples, SampleRate::new(16000).unwrap())
        .expect("v2 run");
    let v2_der = compute_der(&ref_turns, &v2_result.turns, 0.25);

    println!(
        "Hybrid:   DER={:.2}% speakers={} ref={}",
        hybrid_der.der * 100.0,
        hybrid_result.num_speakers,
        ref_speakers
    );
    println!(
        "PipelineV2: DER={:.2}% speakers={} ref={}",
        v2_der.der * 100.0,
        v2_result.num_speakers,
        ref_speakers
    );
}
