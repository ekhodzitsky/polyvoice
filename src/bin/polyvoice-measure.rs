//! Measurement harness: streaming latency presets, VAD parity, short-segment embedder EER.
//!
//! ```text
//! cargo run --features "cli,vad-earshot" --bin polyvoice-measure -- streaming \
//!   --dataset data/voxconverse-test --max-files 30 --output benchmarks/results/streaming-latency-measured.json
//! ```

#![allow(deprecated)]

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use polyvoice::der::compute_der;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::Pipeline;
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::streaming::{LatencyPreset, StreamingPipeline};
use polyvoice::types::{ClusterConfig, DiarizationConfig, SpeakerTurn};
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
use polyvoice::{FbankOnnxExtractor, SileroVad};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "polyvoice-measure", about = "Parity / latency measurement harness")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Streaming presets: input-buffer latency (config), measured RTF, DER @ collar 0 and 0.25.
    Streaming {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, default_value = "30")]
        max_files: usize,
        #[arg(long, default_value = "3200")]
        chunk_samples: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Legacy pipeline DER: Silero vs Earshot VAD (same embedder/cluster).
    VadParity {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, default_value = "30")]
        max_files: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Short-segment speaker verification EER + optional DER with ERes2NetV2 vs default embedder.
    EmbedderShort {
        /// VoxCeleb1-style verification list (label enroll test).
        #[arg(long)]
        veri_list: PathBuf,
        /// Root containing `wav/` (or flat id paths from the list).
        #[arg(long)]
        wav_root: PathBuf,
        #[arg(long, default_value = "0.5,1.0,2.0,3.0")]
        durations: String,
        #[arg(long, default_value = "500")]
        max_pairs: usize,
        #[arg(long)]
        der_dataset: Option<PathBuf>,
        #[arg(long, default_value = "30")]
        der_max_files: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Serialize)]
struct Hardware {
    cpu: String,
    arch: String,
    cores: usize,
}

#[derive(Serialize)]
struct StreamingPresetRow {
    preset: String,
    window_secs: f32,
    hop_secs: f32,
    right_context_secs: f32,
    cache_cap: usize,
    input_buffer_latency_secs: f32,
    mean_rtf: f64,
    macro_der_collar_0: f64,
    macro_der_collar_025: f64,
    files: usize,
    total_audio_secs: f64,
    total_wall_secs: f64,
}

#[derive(Serialize)]
struct StreamingReport {
    schema: String,
    hardware: Hardware,
    chunk_samples: usize,
    max_files: usize,
    dataset: String,
    rows: Vec<StreamingPresetRow>,
}

#[cfg(feature = "vad-earshot")]
#[derive(Serialize)]
struct VadArm {
    name: String,
    frame_size: usize,
    macro_der_collar_0: f64,
    macro_der_collar_025: f64,
    mean_rtf: f64,
    files: usize,
}

#[cfg(feature = "vad-earshot")]
#[derive(Serialize)]
struct VadParityReport {
    schema: String,
    hardware: Hardware,
    max_files: usize,
    dataset: String,
    silero: VadArm,
    earshot: VadArm,
    delta_der_collar_0_pp: f64,
    delta_der_collar_025_pp: f64,
    parity_gate_abs_pp: f64,
    parity_pass_collar_0: bool,
    parity_pass_collar_025: bool,
}

#[derive(Serialize)]
struct EerBucket {
    duration_secs: f32,
    pairs: usize,
    eer: f64,
}

#[derive(Serialize)]
struct EmbedderArm {
    name: String,
    model_id: String,
    dim: usize,
    short_seg_eer: Vec<EerBucket>,
    der_macro_collar_0: Option<f64>,
    der_macro_collar_025: Option<f64>,
    der_files: Option<usize>,
}

#[derive(Serialize)]
struct EmbedderReport {
    schema: String,
    hardware: Hardware,
    max_pairs: usize,
    default_embedder: EmbedderArm,
    eres2netv2: EmbedderArm,
}

fn hardware() -> Hardware {
    let cpu = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    Hardware {
        cpu,
        arch: std::env::consts::ARCH.into(),
        cores,
    }
}

fn list_wavs(dataset: &Path, max_files: usize) -> Result<Vec<PathBuf>> {
    let audio_dir = dataset.join("audio");
    let mut wavs: Vec<PathBuf> = std::fs::read_dir(&audio_dir)
        .with_context(|| format!("read_dir {}", audio_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
        .map(|e| e.path())
        .collect();
    wavs.sort();
    wavs.truncate(max_files);
    Ok(wavs)
}

fn load_ref_turns(rttm_dir: &Path, stem: &str) -> Result<Vec<SpeakerTurn>> {
    let rttm = rttm_dir.join(format!("{stem}.rttm"));
    let raw = parse_rttm_file(&rttm).with_context(|| format!("parse {}", rttm.display()))?;
    let grouped = group_by_file(&raw);
    let segs: Vec<_> = grouped
        .get(stem)
        .or_else(|| stem.split('.').next().and_then(|s| grouped.get(s)))
        .map(|v| v.iter().map(|s| (*s).clone()).collect())
        .unwrap_or_default();
    let (turns, _) = to_speaker_turns(&segs);
    Ok(turns)
}

fn macro_der(pairs: &[(f64, f64)]) -> (f64, f64) {
    if pairs.is_empty() {
        return (0.0, 0.0);
    }
    let n = pairs.len() as f64;
    let c0 = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let c025 = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    (c0, c025)
}

fn der_pair(ref_t: &[SpeakerTurn], hyp: &[SpeakerTurn]) -> (f64, f64) {
    let d0 = compute_der(ref_t, hyp, 0.0);
    let d25 = compute_der(ref_t, hyp, 0.25);
    (d0.der * 100.0, d25.der * 100.0)
}

fn run_streaming(
    dataset: PathBuf,
    max_files: usize,
    chunk_samples: usize,
    output: Option<PathBuf>,
) -> Result<()> {
    let registry = ModelRegistry::default()?;
    let emb_path = registry.ensure("wespeaker_resnet34")?;
    let vad_path = registry.ensure("silero_vad")?;
    let wavs = list_wavs(&dataset, max_files)?;
    let rttm_dir = dataset.join("rttm");
    let presets = [
        LatencyPreset::Realtime,
        LatencyPreset::Balanced,
        LatencyPreset::Accurate,
    ];
    let mut rows = Vec::new();

    for preset in presets {
        let mut ders = Vec::new();
        let mut audio_secs = 0.0_f64;
        let mut wall_secs = 0.0_f64;
        let mut n_ok = 0_usize;
        let params = preset.params();
        let input_lat = preset.input_buffer_latency_secs(16_000, 512);

        for wav in &wavs {
            let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let rttm = rttm_dir.join(format!("{stem}.rttm"));
            if !rttm.is_file() {
                continue;
            }
            let (samples, sr_hz) = read_wav(wav)?;
            if sr_hz != 16_000 {
                eprintln!("[SKIP] {stem}: sample rate {sr_hz}");
                continue;
            }
            let ref_t = load_ref_turns(&rttm_dir, stem)?;
            let extractor = FbankOnnxExtractor::new(
                &emb_path,
                256,
                1,
                polyvoice::onnx::ExecutionProvider::Cpu,
            )?;
            let vad = SileroVad::new(&vad_path, 512)?;
            let vad_config = VadConfig {
                frame_size: 512,
                threshold: 0.5,
                ..VadConfig::default()
            };
            let mut stream =
                StreamingPipeline::with_latency_preset(vad, extractor, preset, vad_config)?;

            let t0 = Instant::now();
            let mut off = 0;
            while off < samples.len() {
                let end = (off + chunk_samples).min(samples.len());
                let _ = stream.feed(&samples[off..end])?;
                off = end;
            }
            let _ = stream.flush()?;
            let wall = t0.elapsed().as_secs_f64();
            let audio = samples.len() as f64 / sr_hz as f64;
            let hyp = stream.turns().to_vec();
            ders.push(der_pair(&ref_t, &hyp));
            audio_secs += audio;
            wall_secs += wall;
            n_ok += 1;
            eprint!(".");
        }
        eprintln!();
        let (c0, c025) = macro_der(&ders);
        let rtf = if audio_secs > 0.0 {
            wall_secs / audio_secs
        } else {
            0.0
        };
        rows.push(StreamingPresetRow {
            preset: match preset {
                LatencyPreset::Realtime => "realtime",
                LatencyPreset::Balanced => "balanced",
                LatencyPreset::Accurate => "accurate",
                _ => "other",
            }
            .into(),
            window_secs: params.window_secs,
            hop_secs: params.hop_secs,
            right_context_secs: params.right_context_secs,
            cache_cap: params.speaker_cache_cap,
            input_buffer_latency_secs: input_lat,
            mean_rtf: rtf,
            macro_der_collar_0: c0,
            macro_der_collar_025: c025,
            files: n_ok,
            total_audio_secs: audio_secs,
            total_wall_secs: wall_secs,
        });
        eprintln!(
            "[{}] files={n_ok} RTF={rtf:.4} DER0={c0:.2}% DER0.25={c025:.2}% lat={input_lat:.3}s",
            rows.last().unwrap().preset
        );
    }

    let report = StreamingReport {
        schema: "polyvoice-streaming-latency-v1".into(),
        hardware: hardware(),
        chunk_samples,
        max_files,
        dataset: dataset.display().to_string(),
        rows,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output {
        std::fs::write(&path, &json)?;
        eprintln!("wrote {}", path.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

#[cfg(feature = "vad-earshot")]
fn run_legacy_arm_silero(
    name: &str,
    wavs: &[PathBuf],
    rttm_dir: &Path,
    emb_path: &Path,
    vad_path: &Path,
    frame_size: usize,
) -> Result<VadArm> {
    let mut ders = Vec::new();
    let mut audio_secs = 0.0_f64;
    let mut wall_secs = 0.0_f64;
    let mut n_ok = 0_usize;

    for wav in wavs {
        let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if !rttm_dir.join(format!("{stem}.rttm")).is_file() {
            continue;
        }
        let (samples, sr_hz) = read_wav(wav)?;
        let ref_t = load_ref_turns(rttm_dir, stem)?;
        let extractor = FbankOnnxExtractor::new(
            emb_path,
            256,
            1,
            polyvoice::onnx::ExecutionProvider::Cpu,
        )?;
        let mut vad = SileroVad::new(vad_path, frame_size)?;
        let vad_config = VadConfig {
            frame_size,
            threshold: 0.5,
            ..VadConfig::default()
        };
        let pipeline = Pipeline::new(
            DiarizationConfig {
                cluster: ClusterConfig {
                    threshold: 0.45,
                    ..Default::default()
                },
                ..DiarizationConfig::default()
            },
            vad_config,
        );
        let t0 = Instant::now();
        let result = pipeline.run(&samples, &extractor, &mut vad)?;
        let wall = t0.elapsed().as_secs_f64();
        let audio = samples.len() as f64 / sr_hz as f64;
        ders.push(der_pair(&ref_t, &result.turns));
        audio_secs += audio;
        wall_secs += wall;
        n_ok += 1;
        eprint!(".");
    }
    eprintln!();
    let (c0, c025) = macro_der(&ders);
    let rtf = if audio_secs > 0.0 {
        wall_secs / audio_secs
    } else {
        0.0
    };
    Ok(VadArm {
        name: name.into(),
        frame_size,
        macro_der_collar_0: c0,
        macro_der_collar_025: c025,
        mean_rtf: rtf,
        files: n_ok,
    })
}

#[cfg(feature = "vad-earshot")]
fn run_legacy_arm_earshot(
    name: &str,
    wavs: &[PathBuf],
    rttm_dir: &Path,
    emb_path: &Path,
) -> Result<VadArm> {
    let mut ders = Vec::new();
    let mut audio_secs = 0.0_f64;
    let mut wall_secs = 0.0_f64;
    let mut n_ok = 0_usize;
    let frame_size = 256_usize;

    for wav in wavs {
        let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if !rttm_dir.join(format!("{stem}.rttm")).is_file() {
            continue;
        }
        let (samples, sr_hz) = read_wav(wav)?;
        let ref_t = load_ref_turns(rttm_dir, stem)?;
        let extractor = FbankOnnxExtractor::new(
            emb_path,
            256,
            1,
            polyvoice::onnx::ExecutionProvider::Cpu,
        )?;
        let mut vad = polyvoice::EarshotVad::new();
        let vad_config = VadConfig {
            frame_size,
            threshold: 0.5,
            ..VadConfig::default()
        };
        let pipeline = Pipeline::new(
            DiarizationConfig {
                cluster: ClusterConfig {
                    threshold: 0.45,
                    ..Default::default()
                },
                ..DiarizationConfig::default()
            },
            vad_config,
        );
        let t0 = Instant::now();
        let result = pipeline.run(&samples, &extractor, &mut vad)?;
        let wall = t0.elapsed().as_secs_f64();
        let audio = samples.len() as f64 / sr_hz as f64;
        ders.push(der_pair(&ref_t, &result.turns));
        audio_secs += audio;
        wall_secs += wall;
        n_ok += 1;
        eprint!(".");
    }
    eprintln!();
    let (c0, c025) = macro_der(&ders);
    let rtf = if audio_secs > 0.0 {
        wall_secs / audio_secs
    } else {
        0.0
    };
    Ok(VadArm {
        name: name.into(),
        frame_size,
        macro_der_collar_0: c0,
        macro_der_collar_025: c025,
        mean_rtf: rtf,
        files: n_ok,
    })
}

fn run_vad_parity(dataset: PathBuf, max_files: usize, output: Option<PathBuf>) -> Result<()> {
    #[cfg(not(feature = "vad-earshot"))]
    {
        let _ = (dataset, max_files, output);
        anyhow::bail!("rebuild with --features vad-earshot");
    }
    #[cfg(feature = "vad-earshot")]
    {
        let registry = ModelRegistry::default()?;
        let emb_path = registry.ensure("wespeaker_resnet34")?;
        let vad_path = registry.ensure("silero_vad")?;
        let wavs = list_wavs(&dataset, max_files)?;
        let rttm_dir = dataset.join("rttm");

        eprintln!("Silero arm…");
        let silero =
            run_legacy_arm_silero("silero", &wavs, &rttm_dir, &emb_path, &vad_path, 512)?;
        eprintln!(
            "silero DER0={:.2}% DER0.25={:.2}% RTF={:.4}",
            silero.macro_der_collar_0, silero.macro_der_collar_025, silero.mean_rtf
        );

        eprintln!("Earshot arm…");
        let earshot = run_legacy_arm_earshot("earshot", &wavs, &rttm_dir, &emb_path)?;
        eprintln!(
            "earshot DER0={:.2}% DER0.25={:.2}% RTF={:.4}",
            earshot.macro_der_collar_0, earshot.macro_der_collar_025, earshot.mean_rtf
        );

        let d0 = earshot.macro_der_collar_0 - silero.macro_der_collar_0;
        let d25 = earshot.macro_der_collar_025 - silero.macro_der_collar_025;
        let gate = 0.3_f64;
        let report = VadParityReport {
            schema: "polyvoice-vad-parity-v1".into(),
            hardware: hardware(),
            max_files,
            dataset: dataset.display().to_string(),
            silero,
            earshot,
            delta_der_collar_0_pp: d0,
            delta_der_collar_025_pp: d25,
            parity_gate_abs_pp: gate,
            parity_pass_collar_0: d0.abs() <= gate,
            parity_pass_collar_025: d25.abs() <= gate,
        };
        let json = serde_json::to_string_pretty(&report)?;
        if let Some(path) = output {
            std::fs::write(&path, &json)?;
            eprintln!("wrote {}", path.display());
        } else {
            println!("{json}");
        }
        Ok(())
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Equal-error-rate from sorted scores: label 1 = same speaker.
fn eer_from_scores(mut pairs: Vec<(f32, bool)>) -> f64 {
    if pairs.is_empty() {
        return 1.0;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_pos = pairs.iter().filter(|p| p.1).count() as f64;
    let n_neg = pairs.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 1.0;
    }
    // Sweep threshold from high to low (accept if score >= thr).
    let mut best = 1.0_f64;
    for thr in pairs.iter().map(|p| p.0) {
        let mut fa = 0.0;
        let mut fr = 0.0;
        for &(s, same) in &pairs {
            if same && s < thr {
                fr += 1.0;
            }
            if !same && s >= thr {
                fa += 1.0;
            }
        }
        let far = fa / n_neg;
        let frr = fr / n_pos;
        let err = (far + frr) / 2.0;
        // Track min |FAR-FRR| point
        let gap = (far - frr).abs();
        if gap < best || (gap - best).abs() < 1e-12 && err < best {
            // Use average of FAR/FRR at best-balance threshold
            best = err.max((far - frr).abs()); // placeholder
        }
        let _ = best;
    }
    // Recompute properly: find thr minimizing |FAR-FRR|
    let mut best_gap = f64::INFINITY;
    let mut best_eer = 1.0_f64;
    for thr in pairs.iter().map(|p| p.0) {
        let mut fa = 0.0_f64;
        let mut fr = 0.0_f64;
        for &(s, same) in &pairs {
            if same && (s as f64) < thr as f64 {
                fr += 1.0;
            }
            if !same && (s as f64) >= thr as f64 {
                fa += 1.0;
            }
        }
        let far = fa / n_neg;
        let frr = fr / n_pos;
        let gap = (far - frr).abs();
        if gap < best_gap {
            best_gap = gap;
            best_eer = (far + frr) / 2.0;
        }
    }
    best_eer * 100.0
}

fn crop_center(samples: &[f32], sr: u32, duration_secs: f32) -> Vec<f32> {
    let n = (duration_secs * sr as f32).round() as usize;
    if samples.len() <= n {
        return samples.to_vec();
    }
    let start = (samples.len() - n) / 2;
    samples[start..start + n].to_vec()
}

/// In-memory verification pair: (same_speaker, enroll_pcm, test_pcm) at 16 kHz.
type MemPair = (bool, Vec<f32>, Vec<f32>);

/// Build short-segment verification pairs from VoxConverse-style RTTMs when
/// VoxCeleb audio is not present. Same-speaker positives from one speaker's
/// segments; negatives from different speakers (same file when possible).
fn pairs_from_rttm_dataset(dataset: &Path, max_files: usize, max_pairs: usize) -> Result<Vec<MemPair>> {
    let wavs = list_wavs(dataset, max_files)?;
    let rttm_dir = dataset.join("rttm");
    let mut out: Vec<MemPair> = Vec::new();
    let mut by_spk: Vec<(String, Vec<f32>)> = Vec::new(); // (file_spk, crop) for cross-file negs

    for wav in &wavs {
        let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let rttm = rttm_dir.join(format!("{stem}.rttm"));
        if !rttm.is_file() {
            continue;
        }
        let (samples, sr) = read_wav(wav)?;
        if sr != 16_000 {
            continue;
        }
        let raw = parse_rttm_file(&rttm)?;
        let grouped = group_by_file(&raw);
        let segs = grouped
            .get(stem)
            .or_else(|| stem.split('.').next().and_then(|s| grouped.get(s)));
        let Some(segs) = segs else { continue };

        // Collect per-speaker slices (≥0.6 s so 0.5 s crop works).
        let mut spk_slices: std::collections::HashMap<String, Vec<Vec<f32>>> =
            std::collections::HashMap::new();
        for s in segs {
            let start = (s.start * sr as f64).floor() as usize;
            let end = (s.end() * sr as f64).ceil() as usize;
            if end <= start || end > samples.len() {
                continue;
            }
            let slice = samples[start..end].to_vec();
            if slice.len() < (0.6 * sr as f32) as usize {
                continue;
            }
            spk_slices
                .entry(s.speaker.clone())
                .or_default()
                .push(slice);
        }

        let speakers: Vec<String> = spk_slices.keys().cloned().collect();
        // Same-speaker pairs within file
        for spk in &speakers {
            let slices = &spk_slices[spk];
            for i in 0..slices.len() {
                for j in (i + 1)..slices.len() {
                    out.push((true, slices[i].clone(), slices[j].clone()));
                    if out.len() >= max_pairs {
                        return Ok(out);
                    }
                }
            }
            if let Some(first) = slices.first() {
                by_spk.push((format!("{stem}:{spk}"), first.clone()));
            }
        }
        // Different-speaker pairs within file
        for i in 0..speakers.len() {
            for j in (i + 1)..speakers.len() {
                let a = &spk_slices[&speakers[i]][0];
                let b = &spk_slices[&speakers[j]][0];
                out.push((false, a.clone(), b.clone()));
                if out.len() >= max_pairs {
                    return Ok(out);
                }
            }
        }
    }
    // Cross-file negatives if we still need pairs
    for i in 0..by_spk.len().min(50) {
        for j in (i + 1)..by_spk.len().min(50) {
            if by_spk[i].0.split(':').next() == by_spk[j].0.split(':').next() {
                continue;
            }
            out.push((false, by_spk[i].1.clone(), by_spk[j].1.clone()));
            if out.len() >= max_pairs {
                break;
            }
        }
    }
    Ok(out)
}

fn run_embedder_short(
    veri_list: PathBuf,
    wav_root: PathBuf,
    durations: String,
    max_pairs: usize,
    der_dataset: Option<PathBuf>,
    der_max_files: usize,
    output: Option<PathBuf>,
) -> Result<()> {
    use polyvoice::embedder::{Embedder, ERes2NetV2Extractor, ResNet34Adapter};

    let registry = ModelRegistry::default()?;
    let durs: Vec<f32> = durations
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if durs.is_empty() {
        anyhow::bail!("empty --durations");
    }

    // Prefer VoxCeleb-style list; fall back to RTTM-derived pairs from der_dataset or wav_root.
    let mut mem_pairs: Vec<MemPair> = Vec::new();
    if veri_list.is_file() {
        let list_text = std::fs::read_to_string(&veri_list)?;
        for line in list_text.lines() {
            let mut parts = line.split_whitespace();
            let Some(lab) = parts.next() else { continue };
            let Some(a) = parts.next() else { continue };
            let Some(b) = parts.next() else { continue };
            let same = lab == "1";
            let pa = wav_root.join("wav").join(a);
            let pb = wav_root.join("wav").join(b);
            if pa.is_file() && pb.is_file() {
                let (sa, sra) = read_wav(&pa)?;
                let (sb, srb) = read_wav(&pb)?;
                if sra == 16_000 && srb == 16_000 {
                    mem_pairs.push((same, sa, sb));
                }
            }
            if mem_pairs.len() >= max_pairs {
                break;
            }
        }
    }
    if mem_pairs.is_empty() {
        let ds = der_dataset
            .clone()
            .filter(|p| p.join("audio").is_dir())
            .or_else(|| {
                if wav_root.join("audio").is_dir() {
                    Some(wav_root.clone())
                } else {
                    None
                }
            })
            .context("no VoxCeleb pairs and no diarization dataset for RTTM-derived pairs")?;
        eprintln!(
            "no VoxCeleb audio; building short-seg pairs from RTTM under {}",
            ds.display()
        );
        mem_pairs = pairs_from_rttm_dataset(&ds, der_max_files.max(10), max_pairs)?;
    }
    eprintln!("verification pairs available: {}", mem_pairs.len());
    if mem_pairs.is_empty() {
        anyhow::bail!("no verification pairs constructed");
    }

    let default_path = registry.ensure("wespeaker_resnet34")?;
    let eres_path = registry
        .ensure("eres2netv2")
        .context("download eres2netv2 (optional model; needs network once)")?;

    let default_emb = ResNet34Adapter::new(
        &default_path,
        2,
        polyvoice::onnx::ExecutionProvider::Cpu,
    )?;
    let eres_emb = ERes2NetV2Extractor::new(
        &eres_path,
        2,
        polyvoice::onnx::ExecutionProvider::Cpu,
    )?;

    fn score_arm(emb: &dyn Embedder, pairs: &[MemPair], durs: &[f32]) -> Result<Vec<EerBucket>> {
        let mut out = Vec::new();
        for &dur in durs {
            let mut scores = Vec::new();
            for (same, sa, sb) in pairs {
                let ca = crop_center(sa, 16_000, dur);
                let cb = crop_center(sb, 16_000, dur);
                if ca.len() < 4000 || cb.len() < 4000 {
                    continue;
                }
                let ea = match emb.embed(&ca) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let eb = match emb.embed(&cb) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                scores.push((cosine(&ea, &eb), *same));
            }
            let eer = eer_from_scores(scores.clone());
            eprintln!(
                "  duration={dur:.1}s pairs={} EER={eer:.2}%",
                scores.len()
            );
            out.push(EerBucket {
                duration_secs: dur,
                pairs: scores.len(),
                eer,
            });
        }
        Ok(out)
    }

    eprintln!("default ResNet34 short-seg EER…");
    let def_eer = score_arm(&default_emb, &mem_pairs, &durs)?;
    eprintln!("ERes2NetV2 short-seg EER…");
    let eres_eer = score_arm(&eres_emb, &mem_pairs, &durs)?;

    // Optional DER on diarization dataset with each embedder via legacy pipeline.
    let mut def_der0 = None;
    let mut def_der25 = None;
    let mut eres_der0 = None;
    let mut eres_der25 = None;
    let mut der_files = None;
    if let Some(ds) = der_dataset {
        let wavs = list_wavs(&ds, der_max_files)?;
        let rttm_dir = ds.join("rttm");
        let vad_path = registry.ensure("silero_vad")?;
        let mut d_pairs = Vec::new();
        let mut e_pairs = Vec::new();
        let mut n = 0_usize;
        for wav in &wavs {
            let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if !rttm_dir.join(format!("{stem}.rttm")).is_file() {
                continue;
            }
            let (samples, sr_hz) = read_wav(wav)?;
            let ref_t = load_ref_turns(&rttm_dir, stem)?;
            let config = DiarizationConfig {
                cluster: ClusterConfig {
                    threshold: 0.45,
                    ..Default::default()
                },
                ..DiarizationConfig::default()
            };
            let vad_config = VadConfig {
                frame_size: 512,
                threshold: 0.5,
                ..VadConfig::default()
            };
            let pipeline = Pipeline::new(config, vad_config);

            let mut vad_d = SileroVad::new(&vad_path, 512)?;
            let ext_d = FbankOnnxExtractor::new(
                &default_path,
                256,
                1,
                polyvoice::onnx::ExecutionProvider::Cpu,
            )?;
            let res_d = pipeline.run(&samples, &ext_d, &mut vad_d)?;
            d_pairs.push(der_pair(&ref_t, &res_d.turns));

            let mut vad_e = SileroVad::new(&vad_path, 512)?;
            // ERes2Net uses same fbank front-end path via FbankOnnxExtractor with dim 192
            let ext_e = FbankOnnxExtractor::new(
                &eres_path,
                192,
                1,
                polyvoice::onnx::ExecutionProvider::Cpu,
            )?;
            let res_e = pipeline.run(&samples, &ext_e, &mut vad_e)?;
            e_pairs.push(der_pair(&ref_t, &res_e.turns));
            n += 1;
            eprint!(".");
            let _ = sr_hz;
        }
        eprintln!();
        let (d0, d25) = macro_der(&d_pairs);
        let (e0, e25) = macro_der(&e_pairs);
        def_der0 = Some(d0);
        def_der25 = Some(d25);
        eres_der0 = Some(e0);
        eres_der25 = Some(e25);
        der_files = Some(n);
        eprintln!("DER default ResNet34: 0={d0:.2}% 0.25={d25:.2}% files={n}");
        eprintln!("DER ERes2NetV2:       0={e0:.2}% 0.25={e25:.2}% files={n}");
    }

    let report = EmbedderReport {
        schema: "polyvoice-embedder-short-v1".into(),
        hardware: hardware(),
        max_pairs,
        default_embedder: EmbedderArm {
            name: "wespeaker-resnet34".into(),
            model_id: "wespeaker_resnet34".into(),
            dim: default_emb.dim(),
            short_seg_eer: def_eer,
            der_macro_collar_0: def_der0,
            der_macro_collar_025: def_der25,
            der_files,
        },
        eres2netv2: EmbedderArm {
            name: "eres2netv2".into(),
            model_id: "eres2netv2".into(),
            dim: eres_emb.dim(),
            short_seg_eer: eres_eer,
            der_macro_collar_0: eres_der0,
            der_macro_collar_025: eres_der25,
            der_files,
        },
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output {
        std::fs::write(&path, &json)?;
        eprintln!("wrote {}", path.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Streaming {
            dataset,
            max_files,
            chunk_samples,
            output,
        } => run_streaming(dataset, max_files, chunk_samples, output),
        Cmd::VadParity {
            dataset,
            max_files,
            output,
        } => run_vad_parity(dataset, max_files, output),
        Cmd::EmbedderShort {
            veri_list,
            wav_root,
            durations,
            max_pairs,
            der_dataset,
            der_max_files,
            output,
        } => run_embedder_short(
            veri_list,
            wav_root,
            durations,
            max_pairs,
            der_dataset,
            der_max_files,
            output,
        ),
    }
}
