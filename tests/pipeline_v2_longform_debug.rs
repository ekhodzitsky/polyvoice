//! Pipeline v2 long-form audio diagnostics.
//!
//! Run with:
//!   cargo test --test pipeline_v2_longform_debug --features "onnx,segmentation,embedder,clusterer,resegmentation,download" -- --ignored --nocapture

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
use polyvoice::segmentation::{PowersetSegmenter, Segmenter};

use polyvoice::utils::cosine_similarity;
use polyvoice::wav::read_wav;
use std::path::Path;

#[ignore = "requires cached ONNX bundle + wav file in data/voxconverse-test/"]
#[test]
fn debug_v2_segments_on_aepyx() {
    let wav_path = Path::new("data/voxconverse-test/audio/aepyx.wav");
    let _rttm_path = Path::new("data/voxconverse-test/rttm/aepyx.rttm");
    if !wav_path.is_file() {
        println!("WAV not found — skipping");
        return;
    }

    let (samples, sr) = read_wav(wav_path).expect("read wav");
    assert_eq!(sr, 16000);

    let registry = ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(polyvoice::types::Profile::Balanced)
        .expect("models");

    let segmenter = PowersetSegmenter::new(&models.segmenter_path).expect("segmenter");
    let embedder = ResNet34Adapter::new(&models.embedder_path, 1).expect("embedder");

    println!("=== SEGMENTATION ===");
    let segments = segmenter.segment(&samples).expect("segment");
    println!("Total segments: {}", segments.len());

    let durations: Vec<f64> = segments.iter().map(|s| s.time.duration()).collect();
    let avg_dur = durations.iter().sum::<f64>() / durations.len() as f64;
    let min_dur = durations.iter().copied().fold(f64::INFINITY, f64::min);
    let max_dur = durations.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "Duration: avg={:.2}s min={:.2}s max={:.2}s",
        avg_dur, min_dur, max_dur
    );

    let mut short = 0;
    let mut medium = 0;
    let mut long = 0;
    for d in &durations {
        if *d < 0.5 {
            short += 1;
        } else if *d < 2.0 {
            medium += 1;
        } else {
            long += 1;
        }
    }
    println!(
        "Distribution: <0.5s={} 0.5-2s={} >2s={}",
        short, medium, long
    );

    let primary: Vec<_> = segments.iter().filter(|s| !s.is_overlap).cloned().collect();
    println!("Primary (non-overlap) segments: {}", primary.len());

    // Simulate pre-embedding aggregation strategies
    for (max_gap, min_dur) in [(0.5, 0.0), (2.0, 0.0), (2.0, 0.5)] {
        let mut aggregated: Vec<polyvoice::segmentation::RawSegment> = Vec::new();
        for seg in &primary {
            if seg.time.duration() < min_dur && !aggregated.is_empty() {
                continue;
            }
            if let Some(last) = aggregated.last_mut()
                && last.local_speaker_idx == seg.local_speaker_idx
                && seg.time.start - last.time.end < max_gap
            {
                last.time.end = seg.time.end.max(last.time.end);
                continue;
            }
            aggregated.push(seg.clone());
        }
        let agg_durations: Vec<f64> = aggregated.iter().map(|s| s.time.duration()).collect();
        let agg_avg = agg_durations.iter().sum::<f64>() / agg_durations.len() as f64;
        let agg_short = agg_durations.iter().filter(|&&d| d < 0.5).count();
        println!(
            "Agg (gap<{:.1}s, min>{:.1}s): {} segs, avg={:.2}s, short={}",
            max_gap,
            min_dur,
            aggregated.len(),
            agg_avg,
            agg_short
        );
    }

    // Compare with legacy SileroVAD segmentation
    println!("\n=== LEGACY SILEROVAD SEGMENTATION ===");
    let silero_path = Path::new("models/silero_vad.onnx");
    let mut vad = if silero_path.is_file() {
        polyvoice::SileroVad::new(silero_path, 512).expect("vad")
    } else {
        println!("  silero_vad.onnx not found — skipping comparison");
        return;
    };
    let speech_regions = polyvoice::vad::segment_speech(
        &mut vad,
        &samples,
        &polyvoice::types::DiarizationConfig::default(),
        &polyvoice::vad::VadConfig::default(),
    )
    .expect("segment_speech");
    println!("SileroVAD speech regions: {}", speech_regions.len());
    let vad_durations: Vec<f64> = speech_regions
        .iter()
        .map(|(s, e)| (*e - *s) as f64 / sr as f64)
        .collect();
    let vad_avg = vad_durations.iter().sum::<f64>() / vad_durations.len() as f64;
    println!("  avg region duration: {:.2}s", vad_avg);

    // Quick check: what if we just use local_speaker_idx directly (no embedding/clustering)?
    println!("\n=== DER USING LOCAL_SPEAKER_IDX DIRECTLY ===");
    {
        let ref_turns = {
            let raw = polyvoice::rttm::parse_rttm_file(_rttm_path).expect("parse rttm");
            let grouped = polyvoice::rttm::group_by_file(&raw);
            let segs: Vec<_> = grouped
                .get("aepyx")
                .map(|v| v.iter().map(|s| (*s).clone()).collect())
                .unwrap_or_default();
            let (turns, _map) = polyvoice::rttm::to_speaker_turns(&segs);
            turns
        };
        let turns: Vec<polyvoice::types::SpeakerTurn> = primary
            .iter()
            .map(|s| polyvoice::types::SpeakerTurn {
                speaker: polyvoice::types::SpeakerId(s.local_speaker_idx as u32),
                time: s.time,
                text: None,
            })
            .collect();
        let der = polyvoice::der::compute_der(&ref_turns, &turns, 0.25);
        println!(
            "  DER = {:.2}% (speakers={})",
            der.der * 100.0,
            primary
                .iter()
                .map(|s| s.local_speaker_idx)
                .collect::<std::collections::HashSet<_>>()
                .len()
        );
    }

    println!("\n=== EMBEDDINGS ===");
    let sr_f64 = sr as f64;
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(primary.len());
    for seg in &primary {
        let start = (seg.time.start * sr_f64) as usize;
        let end = ((seg.time.end * sr_f64) as usize).min(samples.len());
        if end <= start {
            continue;
        }
        let chunk = &samples[start..end];
        let emb = embedder.embed(chunk).expect("embed");
        embeddings.push(emb);
    }
    println!("Valid embeddings: {}", embeddings.len());

    if embeddings.len() >= 2 {
        let mut sims: Vec<f32> = Vec::new();
        for i in 0..embeddings.len().min(50) {
            for j in (i + 1)..embeddings.len().min(50) {
                sims.push(cosine_similarity(&embeddings[i], &embeddings[j]));
            }
        }
        sims.sort_by(|a, b| a.total_cmp(b));
        println!("Pairwise cosine similarity (first 50 embeddings):");
        println!(
            "  min={:.3} p5={:.3} p25={:.3} median={:.3} p75={:.3} p95={:.3} max={:.3}",
            sims.first().unwrap(),
            sims[sims.len() * 5 / 100],
            sims[sims.len() * 25 / 100],
            sims[sims.len() * 50 / 100],
            sims[sims.len() * 75 / 100],
            sims[sims.len() * 95 / 100],
            sims.last().unwrap(),
        );
    }

    println!("\n=== CLUSTERING (AHC auto) ===");
    let clusterer = AhcClusterer::new(20);
    let labels = clusterer.cluster(&embeddings).expect("cluster");
    let num_clusters = labels
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len();
    println!(
        "Inferred clusters: {} (max_clusters={})",
        num_clusters,
        clusterer.max_clusters()
    );

    println!("\n=== CLUSTERING (AHC fixed threshold 0.45) ===");
    let clusterer_fixed = AhcClusterer::with_threshold(20, 0.45);
    let labels_fixed = clusterer_fixed.cluster(&embeddings).expect("cluster");
    let num_fixed = labels_fixed
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len();
    println!("Inferred clusters: {} (threshold=0.45)", num_fixed);

    // Print first 20 segments with their cluster assignments
    println!("\n=== SEGMENT → CLUSTER MAP (first 20) ===");
    for (i, seg) in primary.iter().take(20).enumerate() {
        println!(
            "  {:2}: {:.2}s - {:.2}s  cluster={}",
            i, seg.time.start, seg.time.end, labels[i]
        );
    }
}
