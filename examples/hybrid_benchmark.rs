//! Standalone benchmark for Hybrid pipeline with checkpointing.
//!
//! Run with:
//!   cargo run --example hybrid_benchmark --release \
//!     --features "onnx,segmentation,embedder,clusterer,resegmentation,download"
//!
//! Results are appended to `benchmark_kmeans.json` after each file.
//! If interrupted, rerun — already-processed files are skipped.

use polyvoice::clusterer::KMeansClusterer;
use polyvoice::embedder::ResNet34Adapter;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::hybrid::HybridPipeline;
use polyvoice::segmentation::PowersetSegmenter;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Default)]
struct Checkpoint {
    files: Vec<FileResult>,
}

#[derive(Serialize, Deserialize)]
struct FileResult {
    stem: String,
    der: f64,
    miss: f64,
    fa: f64,
    conf: f64,
    speakers: usize,
    ref_speakers: usize,
}

fn main() {
    let audio_dir = Path::new("data/voxconverse-test/audio");
    let rttm_dir = Path::new("data/voxconverse-test/rttm");
    let checkpoint_path = Path::new("benchmark_kmeans.json");

    let mut checkpoint: Checkpoint = if checkpoint_path.is_file() {
        let raw = fs::read_to_string(checkpoint_path).expect("read checkpoint");
        serde_json::from_str(&raw).unwrap_or_default()
    } else {
        Checkpoint::default()
    };

    let done: HashSet<String> = checkpoint.files.iter().map(|f| f.stem.clone()).collect();

    let registry = ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(Profile::Balanced)
        .expect("models");

    let segmenter = PowersetSegmenter::with_config(
        &models.segmenter_path,
        polyvoice::segmentation::PowersetConfig {
            window_secs: 10.0,
            hop_secs: 1.0,
            sample_rate: 16000,
            aggregation: Default::default(),
        },
    ).expect("segmenter");
    let pool_size = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let embedder = ResNet34Adapter::new(&models.embedder_path, pool_size).expect("embedder");
    let clusterer = KMeansClusterer::new(20);

    let pipeline =
        HybridPipeline::new(Box::new(segmenter), Box::new(embedder), Box::new(clusterer));

    let entries: Vec<_> = fs::read_dir(audio_dir)
        .expect("read audio dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "wav"))
        .collect();

    let total = entries.len();
    let remaining = entries
        .iter()
        .filter(|e| {
            let path = e.path();
            let stem = path.file_stem().unwrap().to_str().unwrap();
            !done.contains(stem)
        })
        .count();

    println!(
        "Total files: {total}, already done: {}, remaining: {remaining}",
        total - remaining
    );

    for (idx, entry) in entries.iter().enumerate() {
        let wav_path = entry.path();
        let stem = wav_path.file_stem().unwrap().to_str().unwrap().to_string();

        if done.contains(&stem) {
            continue;
        }

        let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
        if !rttm_path.is_file() {
            println!("[{}/{}] {stem}: skipping (no RTTM)", idx + 1, total);
            continue;
        }

        let (samples, sr) = read_wav(&wav_path).expect("read wav");
        assert_eq!(sr, 16000);

        let t0 = std::time::Instant::now();
        let result = pipeline
            .run(&samples, SampleRate::new(16000).unwrap())
            .expect("run");
        let elapsed = t0.elapsed().as_secs_f64();

        let ref_turns = {
            let raw = polyvoice::rttm::parse_rttm_file(&rttm_path).expect("parse rttm");
            let grouped = polyvoice::rttm::group_by_file(&raw);
            let segs: Vec<_> = grouped
                .get(stem.as_str())
                .map(|v| v.iter().map(|s| (*s).clone()).collect())
                .unwrap_or_default();
            let (turns, _map) = polyvoice::rttm::to_speaker_turns(&segs);
            turns
        };

        let der = polyvoice::der::compute_der(&ref_turns, &result.turns, 0.25);
        let ref_speakers = ref_turns
            .iter()
            .map(|t| t.speaker.0)
            .max()
            .map_or(0, |m| m + 1) as usize;

        println!(
            "[{}/{}] {}: DER={:.2}% miss={:.2}% fa={:.2}% conf={:.2}% spk={} ref={} time={:.1}s",
            idx + 1,
            total,
            stem,
            der.der * 100.0,
            der.miss_rate * 100.0,
            der.false_alarm_rate * 100.0,
            der.confusion_rate * 100.0,
            result.num_speakers,
            ref_speakers,
            elapsed,
        );

        checkpoint.files.push(FileResult {
            stem: stem.clone(),
            der: der.der,
            miss: der.miss_rate,
            fa: der.false_alarm_rate,
            conf: der.confusion_rate,
            speakers: result.num_speakers,
            ref_speakers,
        });

        // Write checkpoint after every file.
        let raw = serde_json::to_string_pretty(&checkpoint).expect("serialize");
        fs::write(checkpoint_path, raw).expect("write checkpoint");
    }

    let n = checkpoint.files.len();
    let avg_der = checkpoint.files.iter().map(|f| f.der).sum::<f64>() / n as f64;
    let avg_miss = checkpoint.files.iter().map(|f| f.miss).sum::<f64>() / n as f64;
    let avg_fa = checkpoint.files.iter().map(|f| f.fa).sum::<f64>() / n as f64;
    let avg_conf = checkpoint.files.iter().map(|f| f.conf).sum::<f64>() / n as f64;

    println!("\n=== Summary ===");
    println!(
        "Files: {} | Avg DER: {:.2}% | miss: {:.2}% | fa: {:.2}% | conf: {:.2}%",
        n,
        avg_der * 100.0,
        avg_miss * 100.0,
        avg_fa * 100.0,
        avg_conf * 100.0,
    );
}
