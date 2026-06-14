#![allow(deprecated)] // legacy embedding API (F09); see polyvoice::embedder
//! Pipeline v2 debug test — compares legacy pipeline vs v1.0 components on real audio.
//!
//! Requires ONNX models in `models/`:
//!   - legacy: silero_vad.onnx + wespeaker_resnet34.onnx
//!   - v2:     powerset_fp32.onnx + cam_pp_fp32.onnx (or resnet34)
//!
//! Run with:
//!   cargo test --test pipeline_v2_debug --features "onnx segmentation embedder clusterer resegmentation spectral download" -- --ignored --nocapture

#![cfg(all(
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
    feature = "spectral",
))]

use polyvoice::EmbeddingExtractor;
use polyvoice::clusterer::{AhcClusterer, Clusterer, NmeScClusterer};
use polyvoice::der::compute_der;
use polyvoice::embedder::{CamPlusPlusExtractor, Embedder, ResNet34Adapter};
use polyvoice::resegmentation::compute_centroids;
use polyvoice::segmentation::{PowersetSegmenter, Segmenter};
use polyvoice::types::{DiarizationResult, SpeakerId, SpeakerTurn};
use polyvoice::utils::cosine_similarity;
use std::path::Path;

fn load_test_audio() -> Vec<f32> {
    let wav_path = Path::new("tests/data/e2e-smoke/audio/fuzfh.wav");
    if !wav_path.exists() {
        panic!(
            "Test WAV not found at {} — run scripts/download-ami-test-single.sh",
            wav_path.display()
        );
    }
    let (samples, sr) = polyvoice::wav::read_wav(wav_path).expect("read wav");
    assert_eq!(sr, 16000, "expected 16kHz mono");
    samples
}

fn load_ground_truth() -> Vec<SpeakerTurn> {
    use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
    let rttm_path = Path::new("tests/data/e2e-smoke/rttm/fuzfh.rttm");
    let raw = parse_rttm_file(rttm_path).expect("parse rttm");
    let grouped = group_by_file(&raw);
    let segs: Vec<_> = grouped
        .get("fuzfh")
        .map(|v| v.iter().map(|s| (*s).clone()).collect())
        .unwrap_or_default();
    let (turns, _map) = to_speaker_turns(&segs);
    turns
}

fn run_legacy_pipeline(samples: &[f32]) -> DiarizationResult {
    use polyvoice::pipeline::Pipeline;
    use polyvoice::types::DiarizationConfig;
    use polyvoice::vad::VadConfig;
    use polyvoice::{FbankOnnxExtractor, SileroVad};

    let extractor = FbankOnnxExtractor::new(Path::new("models/wespeaker_resnet34.onnx"), 256, 1)
        .expect("legacy embedder load");
    let mut vad =
        SileroVad::new(Path::new("models/silero_vad.onnx"), 512).expect("legacy vad load");

    let config = DiarizationConfig::default();
    let vad_config = VadConfig::default();
    let pipeline = Pipeline::new(config, vad_config);
    pipeline
        .run(samples, &extractor, &mut vad)
        .expect("legacy pipeline run")
}

fn extract_segments_from_v2(samples: &[f32]) -> Vec<polyvoice::segmentation::RawSegment> {
    let segmenter =
        PowersetSegmenter::new(Path::new("models/powerset_fp32.onnx")).expect("powerset load");
    segmenter.segment(samples).expect("segment")
}

fn compare_embeddings_on_segments(
    samples: &[f32],
    segments: &[polyvoice::segmentation::RawSegment],
) {
    use polyvoice::types::DiarizationConfig;

    let primary_segments: Vec<_> = segments.iter().filter(|s| !s.is_overlap).cloned().collect();
    let sr = 16000_f64;

    // Legacy ResNet34
    let legacy_ext =
        polyvoice::FbankOnnxExtractor::new(Path::new("models/wespeaker_resnet34.onnx"), 256, 1)
            .expect("legacy load");
    let config = DiarizationConfig::default();
    let mut legacy_embs: Vec<Vec<f32>> = Vec::new();
    for seg in &primary_segments {
        let start_idx = (seg.time.start * sr) as usize;
        let end_idx = ((seg.time.end * sr) as usize).min(samples.len());
        if end_idx <= start_idx {
            continue;
        }
        let chunk = &samples[start_idx..end_idx];
        match legacy_ext.extract(chunk, &config) {
            Ok(emb) => legacy_embs.push(emb),
            Err(e) => println!("legacy embed error: {}", e),
        }
    }

    // CAM++
    let campp_ext = CamPlusPlusExtractor::new(Path::new("models/cam_pp_fp32.onnx"), 512, 1)
        .expect("cam++ load");
    let mut campp_embs: Vec<Vec<f32>> = Vec::new();
    for seg in &primary_segments {
        let start_idx = (seg.time.start * sr) as usize;
        let end_idx = ((seg.time.end * sr) as usize).min(samples.len());
        if end_idx <= start_idx {
            continue;
        }
        let chunk = &samples[start_idx..end_idx];
        match campp_ext.embed(chunk) {
            Ok(emb) => campp_embs.push(emb),
            Err(e) => println!("cam++ embed error: {}", e),
        }
    }

    // ResNet34 via v1.0 Embedder trait
    let resnet34_ext = ResNet34Adapter::new(Path::new("models/wespeaker_resnet34.onnx"), 1)
        .expect("resnet34 load");
    let mut resnet34_embs: Vec<Vec<f32>> = Vec::new();
    for seg in &primary_segments {
        let start_idx = (seg.time.start * sr) as usize;
        let end_idx = ((seg.time.end * sr) as usize).min(samples.len());
        if end_idx <= start_idx {
            continue;
        }
        let chunk = &samples[start_idx..end_idx];
        match resnet34_ext.embed(chunk) {
            Ok(emb) => resnet34_embs.push(emb),
            Err(e) => println!("resnet34 v2 embed error: {}", e),
        }
    }

    println!("\n=== EMBEDDING COMPARISON ===");
    println!(
        "Segments: {} | Legacy embs: {} | CAM++ embs: {} | ResNet34 v2 embs: {}",
        primary_segments.len(),
        legacy_embs.len(),
        campp_embs.len(),
        resnet34_embs.len()
    );

    if legacy_embs.len() >= 2 {
        println!("Legacy (ResNet34 via old trait) pairwise cosine similarities:");
        for i in 0..legacy_embs.len() {
            for j in (i + 1)..legacy_embs.len() {
                let sim = cosine_similarity(&legacy_embs[i], &legacy_embs[j]);
                println!("  emb[{}] x emb[{}] = {:.4}", i, j, sim);
            }
        }
    }

    if campp_embs.len() >= 2 {
        println!("CAM++ pairwise cosine similarities:");
        for i in 0..campp_embs.len() {
            for j in (i + 1)..campp_embs.len() {
                let sim = cosine_similarity(&campp_embs[i], &campp_embs[j]);
                println!("  emb[{}] x emb[{}] = {:.4}", i, j, sim);
            }
        }
    }

    if resnet34_embs.len() >= 2 {
        println!("ResNet34 (v1.0 Embedder trait) pairwise cosine similarities:");
        for i in 0..resnet34_embs.len() {
            for j in (i + 1)..resnet34_embs.len() {
                let sim = cosine_similarity(&resnet34_embs[i], &resnet34_embs[j]);
                println!("  emb[{}] x emb[{}] = {:.4}", i, j, sim);
            }
        }
    }

    if let Some(e) = campp_embs.first() {
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("CAM++ first embedding L2 norm: {:.4} (expected ~1.0)", norm);
    }
    if let Some(e) = legacy_embs.first() {
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "Legacy first embedding L2 norm: {:.4} (expected ~1.0)",
            norm
        );
    }
    if let Some(e) = resnet34_embs.first() {
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "ResNet34 v2 first embedding L2 norm: {:.4} (expected ~1.0)",
            norm
        );
    }
}

fn run_v2_pipeline<C: Clusterer>(
    samples: &[f32],
    embedder_name: &str,
    clusterer: &C,
    clusterer_name: &str,
) -> Result<Vec<SpeakerTurn>, Box<dyn std::error::Error>> {
    let segmenter =
        PowersetSegmenter::new(Path::new("models/powerset_fp32.onnx")).expect("powerset load");
    let raw_segments = segmenter.segment(samples)?;
    println!("\n=== V2 STAGE 1: Segmentation ===");
    println!("raw_segments count: {}", raw_segments.len());
    for seg in &raw_segments[..raw_segments.len().min(10)] {
        println!(
            "  spk={} overlap={} conf={:.2} time={:.2}-{:.2}",
            seg.local_speaker_idx,
            seg.is_overlap,
            seg.confidence.get(),
            seg.time.start,
            seg.time.end
        );
    }

    let primary_segments: Vec<_> = raw_segments
        .iter()
        .filter(|s| !s.is_overlap)
        .cloned()
        .collect();
    println!("primary_segments (non-overlap): {}", primary_segments.len());

    // Select embedder
    let embedder: Box<dyn Embedder> = match embedder_name {
        "cam++" => Box::new(
            CamPlusPlusExtractor::new(Path::new("models/cam_pp_fp32.onnx"), 512, 1)
                .expect("cam++ load"),
        ),
        "resnet34" => Box::new(
            ResNet34Adapter::new(Path::new("models/wespeaker_resnet34.onnx"), 1)
                .expect("resnet34 load"),
        ),
        other => panic!("unknown embedder: {}", other),
    };

    let sr = 16000_f64;
    let mut embeddings = Vec::new();
    let mut valid_segments = Vec::new();
    for seg in &primary_segments {
        let start_idx = (seg.time.start * sr) as usize;
        let end_idx = ((seg.time.end * sr) as usize).min(samples.len());
        if end_idx <= start_idx {
            continue;
        }
        let chunk = &samples[start_idx..end_idx];
        match embedder.embed(chunk) {
            Ok(emb) => {
                embeddings.push(emb);
                valid_segments.push(seg.clone());
            }
            Err(e) => {
                println!(
                    "  embed error for segment {:.2}-{:.2}: {}",
                    seg.time.start, seg.time.end, e
                );
            }
        }
    }

    println!("\n=== V2 STAGE 2: Embeddings ({}) ===", embedder_name);
    println!(
        "valid segments with embeddings: {} / {} attempted",
        embeddings.len(),
        primary_segments.len()
    );
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }

    let labels = clusterer.cluster(&embeddings)?;
    let num_speakers = labels.iter().copied().max().map_or(0, |m| m + 1);
    println!("\n=== V2 STAGE 3: Clustering ({}) ===", clusterer_name);
    println!("inferred speakers: {}", num_speakers);
    println!("label distribution: {:?}", {
        let mut dist: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &l in &labels {
            *dist.entry(l).or_insert(0) += 1;
        }
        dist
    });

    let mut primary_turns: Vec<SpeakerTurn> = valid_segments
        .iter()
        .zip(labels.iter())
        .map(|(seg, &lbl)| SpeakerTurn {
            speaker: SpeakerId(lbl as u32),
            time: seg.time,
            text: None,
        })
        .collect();
    primary_turns.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));

    let centroids = compute_centroids(&embeddings, &labels);
    let overlap_regions = polyvoice::resegmentation::extract_overlap_time_ranges(&raw_segments);
    println!("\n=== V2 STAGE 4: Resegmentation ===");
    println!("overlap_regions: {}", overlap_regions.len());
    println!("centroids: {}", centroids.len());

    Ok(primary_turns)
}

#[ignore = "requires ONNX models in models/ directory"]
#[test]
fn debug_pipeline_v2_vs_legacy() {
    let samples = load_test_audio();
    let ref_turns = load_ground_truth();
    println!("\nGround truth turns: {}", ref_turns.len());
    for t in &ref_turns[..ref_turns.len().min(5)] {
        println!(
            "  {}: {:.2}s - {:.2}s",
            t.speaker.0, t.time.start, t.time.end
        );
    }

    // Extract segments once for embedding comparison
    let v2_segments = extract_segments_from_v2(&samples);
    compare_embeddings_on_segments(&samples, &v2_segments);

    // Legacy
    let legacy_result = run_legacy_pipeline(&samples);
    let legacy_der = compute_der(&ref_turns, &legacy_result.turns, 0.25);
    println!("\n=== LEGACY PIPELINE ===");
    println!(
        "speakers={} turns={} DER={:.2}%",
        legacy_result.num_speakers,
        legacy_result.turns.len(),
        legacy_der.der * 100.0
    );
    for t in &legacy_result.turns[..legacy_result.turns.len().min(10)] {
        println!(
            "  {}: {:.2}s - {:.2}s",
            t.speaker.0, t.time.start, t.time.end
        );
    }

    // V2 with CAM++ + NME-SC
    let v2_campp_nme = run_v2_pipeline(&samples, "cam++", &NmeScClusterer::new(64), "NME-SC")
        .expect("v2 cam++ nme run");
    let v2_campp_nme_der = compute_der(&ref_turns, &v2_campp_nme, 0.25);
    let v2_campp_nme_spk: std::collections::HashSet<u32> =
        v2_campp_nme.iter().map(|t| t.speaker.0).collect();

    // V2 with ResNet34 + NME-SC
    let v2_res34_nme = run_v2_pipeline(&samples, "resnet34", &NmeScClusterer::new(64), "NME-SC")
        .expect("v2 resnet34 nme run");
    let v2_res34_nme_der = compute_der(&ref_turns, &v2_res34_nme, 0.25);
    let v2_res34_nme_spk: std::collections::HashSet<u32> =
        v2_res34_nme.iter().map(|t| t.speaker.0).collect();

    // V2 with ResNet34 + AHC
    let v2_res34_ahc = run_v2_pipeline(&samples, "resnet34", &AhcClusterer::new(64), "AHC")
        .expect("v2 resnet34 ahc run");
    let v2_res34_ahc_der = compute_der(&ref_turns, &v2_res34_ahc, 0.25);
    let v2_res34_ahc_spk: std::collections::HashSet<u32> =
        v2_res34_ahc.iter().map(|t| t.speaker.0).collect();

    println!("\n=== DER COMPARISON ===");
    println!("Legacy DER:        {:.2}%", legacy_der.der * 100.0);
    println!(
        "V2 + CAM++ + NME:  speakers={} turns={} DER={:.2}%",
        v2_campp_nme_spk.len(),
        v2_campp_nme.len(),
        v2_campp_nme_der.der * 100.0
    );
    println!(
        "V2 + Res34 + NME:  speakers={} turns={} DER={:.2}%",
        v2_res34_nme_spk.len(),
        v2_res34_nme.len(),
        v2_res34_nme_der.der * 100.0
    );
    println!(
        "V2 + Res34 + AHC:  speakers={} turns={} DER={:.2}%",
        v2_res34_ahc_spk.len(),
        v2_res34_ahc.len(),
        v2_res34_ahc_der.der * 100.0
    );

    assert!(
        legacy_der.der < 0.50,
        "legacy DER should be < 50% on real data, got {:.2}%",
        legacy_der.der * 100.0
    );
}
