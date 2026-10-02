//! Measurement harness: streaming latency presets, VAD parity, short-segment embedder EER.
//!
//! ```text
//! cargo run --features "cli,vad-earshot" --bin polyvoice-measure -- streaming \
//!   --dataset data/voxconverse-test --max-files 30 --output benchmarks/results/streaming-latency-measured.json
//! ```
//!
//! `streaming` and `vad-parity` always fail: those paths needed ONNX Runtime.
//! `embedder-short` runs only with native ResNet34 plus tract CAM++; the helpers
//! below stay in the binary so that build can use them.

#![cfg_attr(
    not(all(
        feature = "embedder-native",
        feature = "backend-tract",
        feature = "embedder"
    )),
    allow(dead_code, unused_imports)
)]

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use polyvoice::cli_common;
use polyvoice::embedder::Embedder;
#[cfg(all(
    feature = "embedder-native",
    feature = "backend-tract",
    feature = "embedder"
))]
use polyvoice::embedder::{CamPlusPlusExtractor, ResNet34Native};
use polyvoice::models::ModelRegistry;
use polyvoice::wav::read_wav;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "polyvoice-measure",
    about = "Parity / latency measurement harness"
)]
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
    /// Short-segment speaker verification EER: ResNet34Native vs CAM++.
    EmbedderShort {
        /// VoxCeleb1-style verification list (label enroll test). Omit to
        /// build pairs from RTTM under `--wav-root` / `--der-dataset`.
        #[arg(long)]
        veri_list: Option<PathBuf>,
        /// Root containing `wav/` (or a diarization `{audio,rttm}` dataset).
        #[arg(long)]
        wav_root: PathBuf,
        #[arg(long, default_value = "0.5,1.0,2.0,3.0")]
        durations: String,
        #[arg(long, default_value = "500")]
        max_pairs: usize,
        /// Diarization `{audio,rttm}` dataset used as the RTTM pair source
        /// when VoxCeleb audio is absent. Full-file DER is
        /// `polyvoice-bench --embedder`, not this subcommand.
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
    resnet34: EmbedderArm,
    cam_pp: EmbedderArm,
}

fn cpu_brand() -> String {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".into())
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| {
                text.lines().find_map(|line| {
                    line.strip_prefix("model name")
                        .map(|v| v.trim().trim_start_matches(':').trim().to_owned())
                        .filter(|s| !s.is_empty())
                })
            })
            .unwrap_or_else(|| "unknown".into())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unknown".into()
    }
}

fn hardware() -> Hardware {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    Hardware {
        cpu: cpu_brand(),
        arch: std::env::consts::ARCH.into(),
        cores,
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
    // Sweep thresholds; keep the point minimizing |FAR-FRR|.
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
fn pairs_from_rttm_dataset(
    dataset: &Path,
    max_files: usize,
    max_pairs: usize,
) -> Result<Vec<MemPair>> {
    let wavs = cli_common::list_wavs(dataset, Some(max_files))?;
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
        let segs = cli_common::load_rttm_segments(&rttm_dir, stem)?;
        if segs.is_empty() {
            continue;
        }

        // Collect per-speaker slices (≥0.6 s so 0.5 s crop works).
        let mut spk_slices: std::collections::HashMap<String, Vec<Vec<f32>>> =
            std::collections::HashMap::new();
        for s in &segs {
            let start = (s.start * sr as f64).floor() as usize;
            let end = (s.end() * sr as f64).ceil() as usize;
            if end <= start || end > samples.len() {
                continue;
            }
            let slice = samples[start..end].to_vec();
            if slice.len() < (0.6 * sr as f32) as usize {
                continue;
            }
            spk_slices.entry(s.speaker.clone()).or_default().push(slice);
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

fn parse_durations(s: &str) -> Result<Vec<f32>> {
    let durs: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    if durs.is_empty() {
        anyhow::bail!("empty --durations");
    }
    Ok(durs)
}

/// Build the in-memory verification pairs: the VoxCeleb-style list when the
/// audio is present, otherwise RTTM-derived pairs from `der_dataset` (or
/// `wav_root` when it itself is a dataset directory).
fn load_verification_pairs(
    veri_list: Option<&Path>,
    wav_root: &Path,
    max_pairs: usize,
    der_dataset: Option<&Path>,
    der_max_files: usize,
) -> Result<Vec<MemPair>> {
    let mut mem_pairs: Vec<MemPair> = Vec::new();
    if let Some(veri_list) = veri_list.filter(|p| p.is_file()) {
        let list_text = std::fs::read_to_string(veri_list)?;
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
            .filter(|p| p.join("audio").is_dir())
            .or_else(|| {
                if wav_root.join("audio").is_dir() {
                    Some(wav_root)
                } else {
                    None
                }
            })
            .context("no VoxCeleb pairs and no diarization dataset for RTTM-derived pairs")?;
        eprintln!(
            "no VoxCeleb audio; building short-seg pairs from RTTM under {}",
            ds.display()
        );
        mem_pairs = pairs_from_rttm_dataset(ds, der_max_files.max(10), max_pairs)?;
    }
    eprintln!("verification pairs available: {}", mem_pairs.len());
    if mem_pairs.is_empty() {
        anyhow::bail!("no verification pairs constructed");
    }
    Ok(mem_pairs)
}

/// Product ResNet34 kernels plus the CAM++ ONNX (INT8 preferred, FP32 fallback).
#[cfg(all(
    feature = "embedder-native",
    feature = "backend-tract",
    feature = "embedder"
))]
struct EmbedderModels {
    resnet: ResNet34Native,
    cam: CamPlusPlusExtractor,
    resnet_id: String,
    cam_id: String,
}

#[cfg(all(
    feature = "embedder-native",
    feature = "backend-tract",
    feature = "embedder"
))]
fn load_cam_pp(registry: &ModelRegistry) -> Result<(String, CamPlusPlusExtractor)> {
    let mut last_err: Option<anyhow::Error> = None;
    for id in ["cam_pp_int8", "cam_pp_fp32"] {
        match try_cam_pp(registry, id) {
            Ok(emb) => return Ok((id.to_string(), emb)),
            Err(e) => {
                eprintln!("CAM++ {id} unavailable: {e:#}");
                last_err = Some(e);
            }
        }
    }
    match last_err {
        Some(e) => Err(e),
        None => anyhow::bail!("CAM++ INT8 and FP32 both failed to load"),
    }
}

#[cfg(all(
    feature = "embedder-native",
    feature = "backend-tract",
    feature = "embedder"
))]
fn try_cam_pp(registry: &ModelRegistry, id: &str) -> Result<CamPlusPlusExtractor> {
    let path = registry.ensure(id)?;
    let emb = CamPlusPlusExtractor::new(&path, 512, 2, polyvoice::onnx::ExecutionProvider::Cpu)?;
    // Session build can succeed on unsupported INT8 ops that then fail at run.
    let smoke = vec![0.0_f32; 8_000];
    emb.embed(&smoke)
        .with_context(|| format!("{id} session built but embed failed"))?;
    Ok(emb)
}

#[cfg(all(
    feature = "embedder-native",
    feature = "backend-tract",
    feature = "embedder"
))]
fn load_embedder_models(registry: &ModelRegistry) -> Result<EmbedderModels> {
    let resnet_path = registry.ensure("resnet34_int8")?;
    let resnet = ResNet34Native::from_onnx_path(&resnet_path)?;
    let (cam_id, cam) = load_cam_pp(registry)?;
    Ok(EmbedderModels {
        resnet,
        cam,
        resnet_id: "resnet34_int8".into(),
        cam_id,
    })
}

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
        eprintln!("  duration={dur:.1}s pairs={} EER={eer:.2}%", scores.len());
        out.push(EerBucket {
            duration_secs: dur,
            pairs: scores.len(),
            eer,
        });
    }
    Ok(out)
}

/// Macro DER at collar 0 and 0.25 for each embedder, plus the scored file count.
///
/// The shipping subcommand passes `None`. Tests construct the struct to check
/// that the report fields round-trip.
#[cfg_attr(not(test), allow(dead_code))]
struct DerComparison {
    default_der: (f64, f64),
    eres_der: (f64, f64),
    files: usize,
}

#[allow(clippy::too_many_arguments)]
fn build_embedder_report(
    max_pairs: usize,
    resnet_id: &str,
    resnet_dim: usize,
    cam_id: &str,
    cam_dim: usize,
    resnet_eer: Vec<EerBucket>,
    cam_eer: Vec<EerBucket>,
    der: Option<DerComparison>,
) -> EmbedderReport {
    let (resnet_der, cam_der, der_files) = match &der {
        Some(d) => (Some(d.default_der), Some(d.eres_der), Some(d.files)),
        None => (None, None, None),
    };
    let (resnet_der0, resnet_der25) = resnet_der.unzip();
    let (cam_der0, cam_der25) = cam_der.unzip();
    EmbedderReport {
        schema: "polyvoice-embedder-short-v2".into(),
        hardware: hardware(),
        max_pairs,
        resnet34: EmbedderArm {
            name: "wespeaker-resnet34-native".into(),
            model_id: resnet_id.into(),
            dim: resnet_dim,
            short_seg_eer: resnet_eer,
            der_macro_collar_0: resnet_der0,
            der_macro_collar_025: resnet_der25,
            der_files,
        },
        cam_pp: EmbedderArm {
            name: "cam++".into(),
            model_id: cam_id.into(),
            dim: cam_dim,
            short_seg_eer: cam_eer,
            der_macro_collar_0: cam_der0,
            der_macro_collar_025: cam_der25,
            der_files,
        },
    }
}

#[cfg(not(all(
    feature = "embedder-native",
    feature = "backend-tract",
    feature = "embedder"
)))]
fn run_embedder_short(
    _veri_list: Option<PathBuf>,
    _wav_root: PathBuf,
    _durations: String,
    _max_pairs: usize,
    _der_dataset: Option<PathBuf>,
    _der_max_files: usize,
    _output: Option<PathBuf>,
) -> Result<()> {
    anyhow::bail!(
        "polyvoice-measure embedder-short requires `--features cli,backend-tract` \
         (native ResNet34 + tract CAM++)"
    )
}

#[cfg(all(
    feature = "embedder-native",
    feature = "backend-tract",
    feature = "embedder"
))]
fn run_embedder_short(
    veri_list: Option<PathBuf>,
    wav_root: PathBuf,
    durations: String,
    max_pairs: usize,
    der_dataset: Option<PathBuf>,
    der_max_files: usize,
    output: Option<PathBuf>,
) -> Result<()> {
    let registry = ModelRegistry::default()?;
    let durs = parse_durations(&durations)?;
    let mem_pairs = load_verification_pairs(
        veri_list.as_deref(),
        &wav_root,
        max_pairs,
        der_dataset.as_deref(),
        der_max_files,
    )?;
    let models = load_embedder_models(&registry)?;

    eprintln!("ResNet34Native ({}) short-seg EER…", models.resnet_id);
    let resnet_eer = score_arm(&models.resnet, &mem_pairs, &durs)?;
    eprintln!("CAM++ ({}) short-seg EER…", models.cam_id);
    let cam_eer = score_arm(&models.cam, &mem_pairs, &durs)?;

    let report = build_embedder_report(
        max_pairs,
        &models.resnet_id,
        models.resnet.dim(),
        &models.cam_id,
        models.cam.dim(),
        resnet_eer,
        cam_eer,
        None,
    );
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
    cli_common::limit_malloc_arenas();
    let args = Args::parse();
    match args.cmd {
        Cmd::Streaming { .. } => cli_common::require_onnx("polyvoice-measure streaming"),
        Cmd::VadParity { .. } => cli_common::require_onnx("polyvoice-measure vad-parity"),
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

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "polyvoice_measure_tests.rs"]
mod tests;
