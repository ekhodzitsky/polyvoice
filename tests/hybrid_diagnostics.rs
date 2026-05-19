//! Hybrid pipeline diagnostics: print embeddings stats and similarity matrix
//! to understand over-clustering on long-form audio.
//!
//! Run with:
//!   cargo test --test hybrid_diagnostics \
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

use polyvoice::embedder::ResNet34Adapter;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::hybrid::HybridPipeline;
use polyvoice::segmentation::{PowersetSegmenter, Segmenter};
use polyvoice::types::{Profile, SampleRate};
use polyvoice::utils::cosine_similarity;
use polyvoice::wav::read_wav;
use std::path::Path;

fn percentile_sorted(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn analyze_file(stem: &str, audio_dir: &Path, rttm_dir: Option<&Path>) {
    let registry = ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(Profile::Balanced)
        .expect("models");

    let wav_path = audio_dir.join(format!("{stem}.wav"));
    let (samples, sr) = read_wav(&wav_path).expect("read wav");
    assert_eq!(sr, 16000);

    let segmenter = PowersetSegmenter::new(&models.segmenter_path).expect("segmenter");

    // Check overlap in raw segments before moving segmenter into pipeline.
    let raw_segments = segmenter.segment(&samples).unwrap();

    let pool_size = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let embedder = ResNet34Adapter::new(&models.embedder_path, pool_size).expect("embedder");

    // Use a dummy clusterer that we will override later.
    // HybridPipeline requires a Clusterer at construction, but run_diagnostics
    // only calls clusterer.cluster() at the end. We can swap the clusterer
    // later by building a new pipeline with the same components but different
    // clusterer.  However, run_diagnostics uses *self* clusterer.
    // Simpler: build pipeline with AhcClusterer(20, 0.35), call run_diagnostics,
    // ignore labels, then re-cluster manually with different thresholds.
    use polyvoice::clusterer::AhcClusterer;
    let clusterer = AhcClusterer::with_threshold(20, 0.35);

    let pipeline =
        HybridPipeline::new(Box::new(segmenter), Box::new(embedder), Box::new(clusterer))
            .with_exclude_overlap(true);

    let diag = pipeline
        .run_diagnostics(&samples, SampleRate::new(16000).unwrap())
        .expect("run_diagnostics");
    let overlap_time: f64 = raw_segments
        .iter()
        .filter(|s| s.is_overlap)
        .map(|s| s.time.duration())
        .sum();
    let total_time: f64 = raw_segments.iter().map(|s| s.time.duration()).sum();
    println!(
        "Overlap: {:.1}s / {:.1}s ({:.1}%)",
        overlap_time,
        total_time,
        if total_time > 0.0 {
            overlap_time / total_time * 100.0
        } else {
            0.0
        }
    );

    let n = diag.embeddings.len();
    println!("\n=== {stem} ===");
    println!("Chunks (embeddings): {n}");
    println!(
        "Total speech time: {:.1} s",
        diag.time_ranges.iter().map(|t| t.duration()).sum::<f64>()
    );

    // NaN / Inf check
    let mut nan_count = 0_usize;
    let mut inf_count = 0_usize;
    let mut zero_count = 0_usize;
    for (i, emb) in diag.embeddings.iter().enumerate() {
        if emb.iter().any(|v| v.is_nan()) {
            nan_count += 1;
            if nan_count <= 5 {
                println!(
                    "  WARNING: embedding {i} contains NaN (chunk {} audio samples, time {:.2}-{:.2})",
                    diag.raw_chunk_lengths[i], diag.time_ranges[i].start, diag.time_ranges[i].end,
                );
            }
        }
        if emb.iter().any(|v| v.is_infinite()) {
            inf_count += 1;
        }
        if emb.iter().all(|v| *v == 0.0) {
            zero_count += 1;
        }
    }
    if nan_count > 0 {
        println!("  NaN embeddings: {nan_count}");
    }
    if inf_count > 0 {
        println!("  Inf embeddings: {inf_count}");
    }
    if zero_count > 0 {
        println!("  All-zero embeddings: {zero_count}");
    }

    if n < 2 {
        println!("Too few chunks for similarity analysis");
        return;
    }

    // Pairwise similarities (upper triangle only, no diagonal).
    let mut sims: Vec<f32> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            sims.push(cosine_similarity(&diag.embeddings[i], &diag.embeddings[j]));
        }
    }
    sims.sort_by(|a, b| a.total_cmp(b));

    let sum: f32 = sims.iter().sum();
    let mean = sum / sims.len() as f32;
    let min = sims.first().copied().unwrap_or(0.0);
    let max = sims.last().copied().unwrap_or(0.0);

    println!(
        "Similarities (n={}): min={:.3} max={:.3} mean={:.3}",
        sims.len(),
        min,
        max,
        mean
    );
    println!(
        "Percentiles: p10={:.3} p25={:.3} p50={:.3} p75={:.3} p90={:.3} p99={:.3}",
        percentile_sorted(&sims, 10.0),
        percentile_sorted(&sims, 25.0),
        percentile_sorted(&sims, 50.0),
        percentile_sorted(&sims, 75.0),
        percentile_sorted(&sims, 90.0),
        percentile_sorted(&sims, 99.0),
    );

    // Speaker count sweep across thresholds.
    // Auto-threshold AHC.
    let (auto_labels, auto_thr) = polyvoice::ahc::agglomerative_cluster_auto(&diag.embeddings);
    let auto_speakers = auto_labels.iter().copied().max().map_or(0, |m| m + 1);
    println!(
        "\nAuto-threshold AHC: threshold={:.3}, speakers={}",
        auto_thr, auto_speakers
    );

    // Oracle clustering analysis (if RTTM available).
    if let Some(rttm_dir) = rttm_dir {
        let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
        if rttm_path.is_file() {
            use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
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

            // Assign each chunk the dominant speaker from ref_turns.
            let mut oracle_labels: Vec<Option<u32>> = vec![None; n];
            for (i, time) in diag.time_ranges.iter().enumerate() {
                let mid = (time.start + time.end) / 2.0;
                // Find which speaker covers the midpoint.
                for turn in &ref_turns {
                    if turn.time.start <= mid && mid < turn.time.end {
                        oracle_labels[i] = Some(turn.speaker.0);
                        break;
                    }
                }
            }

            fn compute_oracle_stats(
                embeddings: &[Vec<f32>],
                labels: &[Option<u32>],
                valid: &[bool],
            ) -> (Vec<f32>, Vec<f32>) {
                let mut within_sims: Vec<f32> = Vec::new();
                let mut between_sims: Vec<f32> = Vec::new();
                let n = embeddings.len();
                for i in 0..n {
                    if !valid[i] {
                        continue;
                    }
                    for j in (i + 1)..n {
                        if !valid[j] {
                            continue;
                        }
                        let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
                        match (labels[i], labels[j]) {
                            (Some(a), Some(b)) if a == b => within_sims.push(sim),
                            (Some(_), Some(_)) => between_sims.push(sim),
                            _ => {}
                        }
                    }
                }
                (within_sims, between_sims)
            }

            // All chunks.
            let valid_all = vec![true; n];
            let (within_all, between_all) =
                compute_oracle_stats(&diag.embeddings, &oracle_labels, &valid_all);
            if !within_all.is_empty() && !between_all.is_empty() {
                let mut w = within_all.clone();
                let mut b = between_all.clone();
                w.sort_by(|a, b| a.total_cmp(b));
                b.sort_by(|a, b| a.total_cmp(b));
                println!("\nOracle analysis (all chunks):");
                println!(
                    "  Within  pairs={} mean={:.3} p50={:.3} p90={:.3}",
                    w.len(),
                    w.iter().sum::<f32>() / w.len() as f32,
                    percentile_sorted(&w, 50.0),
                    percentile_sorted(&w, 90.0),
                );
                println!(
                    "  Between pairs={} mean={:.3} p50={:.3} p90={:.3}",
                    b.len(),
                    b.iter().sum::<f32>() / b.len() as f32,
                    percentile_sorted(&b, 50.0),
                    percentile_sorted(&b, 90.0),
                );
            }

            // Exclude zero-padded partial chunks (< 2s window).
            let valid_full: Vec<bool> =
                diag.raw_chunk_lengths.iter().map(|&l| l >= 32000).collect();
            let n_full = valid_full.iter().filter(|&&v| v).count();
            let (within_full, between_full) =
                compute_oracle_stats(&diag.embeddings, &oracle_labels, &valid_full);
            if !within_full.is_empty() && !between_full.is_empty() {
                let mut w = within_full.clone();
                let mut b = between_full.clone();
                w.sort_by(|a, b| a.total_cmp(b));
                b.sort_by(|a, b| a.total_cmp(b));
                println!("\nOracle analysis (full-window chunks only, n={n_full}):");
                println!(
                    "  Within  pairs={} mean={:.3} p50={:.3} p90={:.3}",
                    w.len(),
                    w.iter().sum::<f32>() / w.len() as f32,
                    percentile_sorted(&w, 50.0),
                    percentile_sorted(&w, 90.0),
                );
                println!(
                    "  Between pairs={} mean={:.3} p50={:.3} p90={:.3}",
                    b.len(),
                    b.iter().sum::<f32>() / b.len() as f32,
                    percentile_sorted(&b, 50.0),
                    percentile_sorted(&b, 90.0),
                );
            }
        }
    }

    // K-means with oracle K.
    if let Some(rttm_dir) = rttm_dir {
        let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
        if rttm_path.is_file() {
            use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
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
            let oracle_k = ref_turns
                .iter()
                .map(|t| t.speaker.0)
                .collect::<std::collections::HashSet<_>>()
                .len();
            if oracle_k >= 2 {
                let kmeans_labels = polyvoice::kmeans::kmeans_pp(&diag.embeddings, oracle_k, 100);
                let kmeans_speakers = kmeans_labels.iter().copied().max().map_or(0, |m| m + 1);
                println!("\nK-means (k={oracle_k}): speakers={kmeans_speakers}");

                // Compute DER for k-means.
                let segments: Vec<polyvoice::types::Segment> = kmeans_labels
                    .iter()
                    .zip(diag.time_ranges.iter())
                    .map(|(&label, &time)| polyvoice::types::Segment {
                        time,
                        speaker: Some(polyvoice::types::SpeakerId(label as u32)),
                        confidence: None,
                    })
                    .collect();
                let merged = polyvoice::utils::merge_segments(segments, 0.5);
                let turns: Vec<polyvoice::types::SpeakerTurn> = merged
                    .iter()
                    .filter_map(|s| {
                        s.speaker.map(|spk| polyvoice::types::SpeakerTurn {
                            speaker: spk,
                            time: s.time,
                            text: None,
                        })
                    })
                    .collect();
                let der = polyvoice::der::compute_der(&ref_turns, &turns, 0.25);
                println!(
                    "  DER={:.2}% miss={:.2}% fa={:.2}% conf={:.2}%",
                    der.der * 100.0,
                    der.miss_rate * 100.0,
                    der.false_alarm_rate * 100.0,
                    der.confusion_rate * 100.0,
                );
            }
        }
    }

    // Max-clusters sweep at fixed threshold 0.40.
    println!("\nMax-clusters sweep (threshold=0.40):");
    println!("{:>12} {:>10} {:>12}", "max_clust", "speakers", "avg_sim");
    for max_clust in [5, 8, 10, 12, 15, 20, 30, 50] {
        let labels =
            polyvoice::ahc::agglomerative_cluster_max_clusters(&diag.embeddings, 0.40, max_clust);
        let num_speakers = labels.iter().copied().max().map_or(0, |m| m + 1);
        let mut intra_sims: Vec<f32> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if labels[i] == labels[j] {
                    intra_sims.push(cosine_similarity(&diag.embeddings[i], &diag.embeddings[j]));
                }
            }
        }
        let avg_intra = if intra_sims.is_empty() {
            0.0
        } else {
            intra_sims.iter().sum::<f32>() / intra_sims.len() as f32
        };
        println!("{:>12} {:>10} {:>12.3}", max_clust, num_speakers, avg_intra);
    }

    println!("\nThreshold sweep (max_clusters=20):");
    println!("{:>8} {:>10} {:>12}", "thr", "speakers", "avg_sim");
    for thr in [0.25f32, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60] {
        let labels = polyvoice::ahc::agglomerative_cluster_max_clusters(&diag.embeddings, thr, 20);
        let num_speakers = labels.iter().copied().max().map_or(0, |m| m + 1);
        // Average within-cluster similarity.
        let mut intra_sims: Vec<f32> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if labels[i] == labels[j] {
                    intra_sims.push(cosine_similarity(&diag.embeddings[i], &diag.embeddings[j]));
                }
            }
        }
        let avg_intra = if intra_sims.is_empty() {
            0.0
        } else {
            intra_sims.iter().sum::<f32>() / intra_sims.len() as f32
        };
        println!("{:>8.2} {:>10} {:>12.3}", thr, num_speakers, avg_intra);
    }

    // If we have reference RTTM, compute DER for a few thresholds.
    if let Some(rttm_dir) = rttm_dir {
        let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
        if rttm_path.is_file() {
            use polyvoice::der::compute_der;
            use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
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
            println!("\nReference speakers: {ref_speakers}");
            println!(
                "{:>8} {:>10} {:>10} {:>8} {:>8} {:>10}",
                "thr", "speakers", "DER%", "miss%", "fa%", "conf%"
            );
            for thr in [0.25f32, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60] {
                let labels =
                    polyvoice::ahc::agglomerative_cluster_max_clusters(&diag.embeddings, thr, 20);
                let num_speakers = labels.iter().copied().max().map_or(0, |m| m + 1);

                // Reconstruct turns from labels + time_ranges.
                let segments: Vec<polyvoice::types::Segment> = labels
                    .iter()
                    .zip(diag.time_ranges.iter())
                    .map(|(&label, &time)| polyvoice::types::Segment {
                        time,
                        speaker: Some(polyvoice::types::SpeakerId(label as u32)),
                        confidence: None,
                    })
                    .collect();
                let merged = polyvoice::utils::merge_segments(segments, 0.5);
                let turns: Vec<polyvoice::types::SpeakerTurn> = merged
                    .iter()
                    .filter_map(|s| {
                        s.speaker.map(|spk| polyvoice::types::SpeakerTurn {
                            speaker: spk,
                            time: s.time,
                            text: None,
                        })
                    })
                    .collect();

                let der = compute_der(&ref_turns, &turns, 0.25);
                println!(
                    "{:>8.2} {:>10} {:>10.2} {:>8.2} {:>8.2} {:>10.2}",
                    thr,
                    num_speakers,
                    der.der * 100.0,
                    der.miss_rate * 100.0,
                    der.false_alarm_rate * 100.0,
                    der.confusion_rate * 100.0,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires ONNX models + wav/rttm"]
fn diagnose_aorju() {
    analyze_file(
        "aorju",
        Path::new("data/voxconverse-test/audio"),
        Some(Path::new("data/voxconverse-test/rttm")),
    );
}

#[test]
#[ignore = "requires ONNX models + wav/rttm"]
fn diagnose_e2e_smoke() {
    analyze_file(
        "fuzfh",
        Path::new("tests/data/e2e-smoke/audio"),
        Some(Path::new("tests/data/e2e-smoke/rttm")),
    );
}

#[test]
#[ignore = "requires ONNX models + wav/rttm"]
fn diagnose_3_file_subset() {
    for stem in ["aepyx", "aggyz", "aiqwk"] {
        analyze_file(
            stem,
            Path::new("data/voxconverse-test/audio"),
            Some(Path::new("data/voxconverse-test/rttm")),
        );
    }
}
