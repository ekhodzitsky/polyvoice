//! DER benchmark runner for polyvoice.
//!
//! Usage:
//!   cargo run --release --features cli --bin polyvoice-bench -- data/ami-test
//!   cargo run --release --features cli --bin polyvoice-bench -- data/ami-test --collar 0.25

use clap::Parser;
use polyvoice::der::{DerResult, compute_der};
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::silero_vad::SileroVad;
use polyvoice::vad::{VadConfig, VoiceActivityDetector};
use polyvoice::{DiarizationConfig, FbankOnnxExtractor, Pipeline, SpeakerTurn};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "polyvoice-bench",
    about = "DER benchmark on annotated datasets"
)]
struct Args {
    /// Path to dataset directory (expects audio/ and rttm/ subdirs)
    data_dir: PathBuf,

    /// Path to ONNX models directory
    #[arg(long, env = "POLYVOICE_MODEL_DIR", default_value = "models")]
    model_dir: PathBuf,

    /// Forgiveness collar in seconds around reference boundaries
    #[arg(long, default_value = "0.25")]
    collar: f64,

    /// Cosine similarity threshold for AHC
    #[arg(long, default_value = "0.5")]
    threshold: f32,

    /// Print per-file results
    #[arg(long, default_value = "true")]
    verbose: bool,

    /// Clustering backend (ahc, kmeans, or spectral)
    #[arg(long, default_value = "ahc")]
    backend: String,

    /// Minimum turn duration in seconds (filters short turns)
    #[arg(long, default_value = "0.0")]
    min_turn_duration: f32,

    /// Maximum gap between same-speaker segments to merge (seconds)
    #[arg(long, default_value = "0.5")]
    max_gap: f32,

    /// Window size for embedding extraction (seconds)
    #[arg(long, default_value = "1.5")]
    window_secs: f32,

    /// Hop length between consecutive windows (seconds)
    #[arg(long, default_value = "0.75")]
    hop_secs: f32,
}

struct FileResult {
    file_id: String,
    der: DerResult,
    num_ref_speakers: usize,
    num_hyp_speakers: usize,
    duration_secs: f64,
    processing_time: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let audio_dir = args.data_dir.join("audio");
    let rttm_dir = args.data_dir.join("rttm");

    if !audio_dir.exists() {
        eprintln!("Error: audio directory not found: {}", audio_dir.display());
        eprintln!("Run: bash scripts/download-ami-test.sh");
        std::process::exit(1);
    }

    // Parse ground truth from all RTTM files in directory
    let all_segments = load_all_rttm(&rttm_dir)?;
    let grouped = group_by_file(&all_segments);
    eprintln!("Files in RTTM: {}", grouped.len());

    // Load models
    eprintln!("Loading models from {}...", args.model_dir.display());
    let extractor =
        FbankOnnxExtractor::new(&args.model_dir.join("wespeaker_resnet34.onnx"), 256, 4)?;

    let config = DiarizationConfig {
        threshold: args.threshold,
        clustering_backend: match args.backend.as_str() {
            "kmeans" => polyvoice::ClusteringBackend::KMeans,
            "spectral" => polyvoice::ClusteringBackend::Spectral,
            "auto" => polyvoice::ClusteringBackend::Auto,
            _ => polyvoice::ClusteringBackend::Ahc,
        },
        min_turn_duration_secs: args.min_turn_duration,
        max_gap_secs: args.max_gap,
        window_secs: args.window_secs,
        hop_secs: args.hop_secs,
        ..Default::default()
    };
    let vad_config = VadConfig::default();
    let pipeline = Pipeline::new(config, vad_config);
    let mut vad = SileroVad::new(&args.model_dir.join("silero_vad.onnx"), 512)?;

    eprintln!("Threshold: {}, Collar: {}s", args.threshold, args.collar);
    eprintln!("---");

    let mut results: Vec<FileResult> = Vec::new();
    let mut skipped = 0;

    let mut file_ids: Vec<&str> = grouped.keys().copied().collect();
    file_ids.sort();

    for file_id in &file_ids {
        let wav_path = find_audio(&audio_dir, file_id);
        let wav_path = match wav_path {
            Some(p) => p,
            None => {
                if args.verbose {
                    eprintln!("[skip] {} — WAV not found", file_id);
                }
                skipped += 1;
                continue;
            }
        };

        let ref_segments = &grouped[file_id];
        let ref_rttm: Vec<_> = ref_segments.iter().copied().cloned().collect();
        let (ref_turns, ref_speaker_map) = to_speaker_turns(&ref_rttm);

        // Load and process audio
        let start = Instant::now();
        let (samples, sr) = match polyvoice::wav::read_wav(&wav_path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[error] {} — {}", file_id, e);
                skipped += 1;
                continue;
            }
        };
        let duration_secs = samples.len() as f64 / sr as f64;

        vad.reset();
        let result = match pipeline.run(&samples, &extractor, &mut vad) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[error] {} — {}", file_id, e);
                skipped += 1;
                continue;
            }
        };
        let processing_time = start.elapsed().as_secs_f64();

        let hyp_turns: Vec<SpeakerTurn> = result.turns;
        let der = compute_der(&ref_turns, &hyp_turns, args.collar);

        let file_result = FileResult {
            file_id: file_id.to_string(),
            der,
            num_ref_speakers: ref_speaker_map.len(),
            num_hyp_speakers: result.num_speakers,
            duration_secs,
            processing_time,
        };

        if args.verbose {
            let rtf = processing_time / duration_secs;
            eprintln!(
                "{:20} | {} | ref_spk={} hyp_spk={} | {:.0}s audio in {:.1}s (RTF={:.2})",
                file_result.file_id,
                file_result.der,
                file_result.num_ref_speakers,
                file_result.num_hyp_speakers,
                file_result.duration_secs,
                file_result.processing_time,
                rtf,
            );
        }

        results.push(file_result);
    }

    eprintln!("---");

    if results.is_empty() {
        eprintln!("No files processed. Download a test set first:");
        eprintln!("  bash scripts/download-ami-test.sh");
        eprintln!("  bash scripts/download-voxconverse-test.sh");
        std::process::exit(1);
    }

    // Aggregate results
    let total_speech: f64 = results.iter().map(|r| r.der.total_speech).sum();
    let total_miss: f64 = results
        .iter()
        .map(|r| r.der.miss_rate * r.der.total_speech)
        .sum();
    let total_fa: f64 = results
        .iter()
        .map(|r| r.der.false_alarm_rate * r.der.total_speech)
        .sum();
    let total_conf: f64 = results
        .iter()
        .map(|r| r.der.confusion_rate * r.der.total_speech)
        .sum();

    let avg_der = if total_speech > 0.0 {
        (total_miss + total_fa + total_conf) / total_speech
    } else {
        0.0
    };

    let total_audio: f64 = results.iter().map(|r| r.duration_secs).sum();
    let total_proc: f64 = results.iter().map(|r| r.processing_time).sum();

    println!();
    println!("=== DER Benchmark Results ===");
    println!("Dataset:     {}", args.data_dir.display());
    println!(
        "Files:       {} processed, {} skipped",
        results.len(),
        skipped
    );
    println!("Collar:      {:.2}s", args.collar);
    println!("Threshold:   {:.2}", args.threshold);
    println!("Backend:     {}", args.backend);
    println!("Min turn:    {:.2}s", args.min_turn_duration);
    println!("Max gap:     {:.2}s", args.max_gap);
    println!("Window:      {:.2}s", args.window_secs);
    println!("Hop:         {:.2}s", args.hop_secs);
    println!();
    println!("  DER:           {:.1}%", avg_der * 100.0);
    println!(
        "  Miss rate:     {:.1}%",
        (total_miss / total_speech) * 100.0
    );
    println!("  False alarm:   {:.1}%", (total_fa / total_speech) * 100.0);
    println!(
        "  Confusion:     {:.1}%",
        (total_conf / total_speech) * 100.0
    );
    println!();
    println!(
        "  Total speech:  {:.0}s ({:.1} min)",
        total_speech,
        total_speech / 60.0
    );
    println!(
        "  Total audio:   {:.0}s ({:.1} min)",
        total_audio,
        total_audio / 60.0
    );
    println!(
        "  Processing:    {:.1}s ({:.1} min)",
        total_proc,
        total_proc / 60.0
    );
    println!("  RTF:           {:.3}", total_proc / total_audio);
    println!();

    // JSON output for CI
    println!("{{");
    println!("  \"der\": {:.4},", avg_der);
    println!("  \"miss_rate\": {:.4},", total_miss / total_speech);
    println!("  \"false_alarm_rate\": {:.4},", total_fa / total_speech);
    println!("  \"confusion_rate\": {:.4},", total_conf / total_speech);
    println!("  \"files_processed\": {},", results.len());
    println!("  \"total_audio_secs\": {:.1},", total_audio);
    println!("  \"rtf\": {:.4}", total_proc / total_audio);
    println!("}}");

    Ok(())
}

fn load_all_rttm(
    rttm_dir: &Path,
) -> Result<Vec<polyvoice::rttm::RttmSegment>, Box<dyn std::error::Error>> {
    let mut all_segments = Vec::new();
    if rttm_dir.is_file() {
        eprintln!("RTTM: {}", rttm_dir.display());
        return Ok(parse_rttm_file(rttm_dir)?);
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(rttm_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "rttm"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("No .rttm files found in {}", rttm_dir.display()).into());
    }
    if paths.len() == 1 {
        eprintln!("RTTM: {}", paths[0].display());
    } else {
        eprintln!("RTTM: {} files in {}", paths.len(), rttm_dir.display());
    }
    for path in &paths {
        all_segments.extend(parse_rttm_file(path)?);
    }
    Ok(all_segments)
}

fn find_audio(audio_dir: &Path, file_id: &str) -> Option<PathBuf> {
    // Try common naming patterns
    let candidates = [
        audio_dir.join(format!("{}.wav", file_id)),
        audio_dir.join(format!("{}.Mix-Headset.wav", file_id)),
        // AMI uses file_id like "EN2002a.Mix-Headset" in RTTM
        audio_dir.join(format!("{}.wav", file_id.replace(".Mix-Headset", ""))),
    ];
    candidates.into_iter().find(|p| p.exists())
}
