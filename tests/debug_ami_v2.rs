//! Debug pipeline v2 on AMI single.

#![allow(clippy::unwrap_used)]
#![cfg(all(
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
    feature = "download",
))]

use polyvoice::clusterer::{AhcClusterer, Clusterer};
use polyvoice::embedder::{Embedder, ResNet34Adapter};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::{Pipeline, PipelineConfig};
use polyvoice::segmentation::{PowersetSegmenter, Segmenter};
use polyvoice::types::{Profile, SampleRate};

#[test]
#[ignore = "debug only"]
fn debug_ami_v2_components() {
    let wav_path = std::path::Path::new("data/ami-test-single/audio/EN2002a.Mix-Headset.wav");
    let (samples, sr) = polyvoice::wav::read_wav(wav_path).unwrap();
    assert_eq!(sr, 16000);

    let registry = ModelRegistry::default().unwrap();
    let models = registry.ensure_for_profile(Profile::Balanced).unwrap();

    // 1. Segmentation
    let segmenter = PowersetSegmenter::new(&models.segmenter_path).unwrap();
    let raw_segments = segmenter.segment(&samples).unwrap();
    println!("raw_segments: {}", raw_segments.len());
    let primary: Vec<_> = raw_segments
        .iter()
        .filter(|s| !s.is_overlap)
        .cloned()
        .collect();
    println!("primary segments: {}", primary.len());

    // 2. Embeddings
    let embedder = ResNet34Adapter::new(
        &models.embedder_path,
        1,
        polyvoice::onnx::ExecutionProvider::Cpu,
    )
    .unwrap();
    let sr_f = 16000_f64;
    let mut embeddings = Vec::new();
    let mut nan_durs: Vec<f64> = Vec::new();
    let mut clean_durs: Vec<f64> = Vec::new();
    for seg in &primary {
        let start_idx = (seg.time.start * sr_f) as usize;
        let end_idx = ((seg.time.end * sr_f) as usize).min(samples.len());
        if end_idx <= start_idx {
            continue;
        }
        let chunk = &samples[start_idx..end_idx];
        let dur = (end_idx - start_idx) as f64 / sr_f;
        match embedder.embed(chunk) {
            Ok(mut emb) => {
                let nonfinite = emb.iter().filter(|v| !v.is_finite()).count();
                if nonfinite > 0 {
                    nan_durs.push(dur);
                } else {
                    clean_durs.push(dur);
                }
                polyvoice::utils::l2_normalize(&mut emb);
                embeddings.push(emb);
            }
            Err(e) => println!("embed error: {e}"),
        }
    }
    println!(
        "embeddings: {} / {} attempted",
        embeddings.len(),
        primary.len()
    );
    let stats = |label: &str, v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        if v.is_empty() {
            println!("  {label}: count=0");
        } else {
            println!(
                "  {label}: count={} dur_secs[min={:.3} median={:.3} max={:.3}]",
                v.len(),
                v[0],
                v[v.len() / 2],
                v[v.len() - 1]
            );
        }
    };
    println!(
        "NaN embeddings (non-finite at embedder output, before normalize): {} / {}",
        nan_durs.len(),
        nan_durs.len() + clean_durs.len()
    );
    stats("NaN-producing segments", &mut nan_durs);
    stats("clean segments", &mut clean_durs);

    if embeddings.len() >= 2 {
        // Compute pairwise similarities
        let mut sims: Vec<f32> = Vec::new();
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                sims.push(polyvoice::utils::cosine_similarity(
                    &embeddings[i],
                    &embeddings[j],
                ));
            }
        }
        sims.sort_by(|a, b| a.total_cmp(b));
        println!(
            "similarities: min={:.3} max={:.3} median={:.3}",
            sims.first().unwrap_or(&0.0),
            sims.last().unwrap_or(&0.0),
            sims[sims.len() / 2]
        );
    }

    // 3. Clustering with auto threshold
    let clusterer = AhcClusterer::new(20);
    let labels = clusterer.cluster(&embeddings).unwrap();
    let num_speakers = labels.iter().copied().max().map_or(0, |m| m + 1);
    println!("AhcAuto speakers: {}", num_speakers);

    // 4. Full pipeline v2
    let config = PipelineConfig {
        profile: Profile::Balanced,
        ..PipelineConfig::default()
    };
    let pipeline = Pipeline::builder()
        .config(config)
        .with_models_from(registry)
        .build()
        .unwrap();
    let result = pipeline
        .run(&samples, SampleRate::new(16000).unwrap())
        .unwrap();
    println!(
        "Full pipeline v2: turns={}, speakers={}",
        result.turns.len(),
        result.num_speakers
    );
}
