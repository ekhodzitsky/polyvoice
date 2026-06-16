#![allow(deprecated)] // legacy embedding API; see polyvoice::embedder
//! polyvoice-bench — DER on a {audio,rttm} dataset directory using the legacy v0.5 Pipeline.

use anyhow::{Context, Result};
use clap::Parser;
use polyvoice::der::{compute_der, compute_der_decomposition, compute_der_with_uem, parse_uem};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::Pipeline;
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::types::{ClusterConfig, DiarizationConfig, Profile, SampleRate, TimeRange};
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
use polyvoice::{FbankOnnxExtractor, SileroVad};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "polyvoice-bench", about = "Run DER on a {audio,rttm} dataset")]
struct Args {
    dataset: PathBuf,
    #[arg(long, default_value = "balanced")]
    profile: String,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value = "0.25")]
    collar: f64,
    #[arg(long, default_value = "false")]
    skip_overlap: bool,
    #[arg(long)]
    max_files: Option<usize>,
    #[arg(long, default_value = "0.45")]
    threshold: f32,
    /// Optional .uem file. Restricts DER to the scored regions per file (frames
    /// outside the UEM are dropped from both mapping and counts).
    #[arg(long)]
    uem: Option<PathBuf>,
}

#[derive(Serialize)]
struct ModelHash {
    model_id: String,
    sha256: String,
}

#[derive(Serialize)]
struct PerSpeakerRecall {
    speaker: u32,
    recall: f64,
}

#[derive(Serialize)]
struct PerFileResult {
    filename: String,
    der_collar: f64,
    der_no_collar: f64,
    miss_rate: f64,
    false_alarm_rate: f64,
    confusion_rate: f64,
    /// Overlap-aware decomposition: DER over single-speaker reference regions
    /// only, DER over overlap regions only (>= 2 ref speakers), and per-speaker
    /// recall. All at the requested collar. Makes overlap-heavy DER interpretable.
    der_single_speaker: f64,
    der_overlap: f64,
    per_speaker_recall: Vec<PerSpeakerRecall>,
    rt_factor: f64,
    ref_speakers: usize,
    hyp_speakers: usize,
    num_turns: usize,
    audio_duration_secs: f64,
    runtime_secs: f64,
}

#[derive(Serialize)]
struct SpeakerCountDiagnostics {
    exact: usize,
    plus_minus_1: usize,
    off_by_2_or_more: usize,
}

#[derive(Serialize)]
struct BenchReport {
    schema: &'static str,
    crate_version: &'static str,
    git_sha: String,
    host_arch: String,
    host_os: String,
    command_line: String,
    dataset_name: String,
    profile: String,
    files_processed: usize,
    files_skipped: usize,
    /// Mean of per-file DER (macro) at the requested collar and at collar=0.
    der_collar_macro: f64,
    der_no_collar_macro: f64,
    /// Duration-weighted DER (micro): sum of error frames / sum of reference
    /// frames — comparable to pyannote/speakrs headline numbers.
    der_collar_micro: f64,
    der_no_collar_micro: f64,
    collar_secs: f64,
    averaging_policy: &'static str,
    miss: f64,
    false_alarm: f64,
    confusion: f64,
    rt_factor_avg: f64,
    speaker_count: SpeakerCountDiagnostics,
    model_hashes: Vec<ModelHash>,
    per_file: Vec<PerFileResult>,
}

fn parse_profile(name: &str) -> Result<Profile> {
    match name {
        "mobile" => Ok(Profile::Mobile),
        "balanced" => Ok(Profile::Balanced),
        other => anyhow::bail!("invalid profile: {other}"),
    }
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn model_hashes(registry: &ModelRegistry, profile: Profile) -> Vec<ModelHash> {
    let mut out = Vec::new();
    let manifest = registry.manifest();
    let prof = match manifest.profile(profile.manifest_id()) {
        Some(p) => p,
        None => return out,
    };
    // The legacy bench pipeline segments with Silero VAD (not the profile's
    // powerset segmenter, which only the v2/hybrid path consumes) and embeds with
    // the profile embedder. Report exactly those two so the integrity record names
    // the models actually loaded for this DER number.
    for model_id in ["silero_vad", prof.embedder.as_str()] {
        if let Some(entry) = manifest.model(model_id) {
            out.push(ModelHash {
                model_id: model_id.to_string(),
                sha256: entry.sha256.clone(),
            });
        }
    }
    out
}

/// Hard-fail unless the on-disk embedder + VAD match the manifest sha256, so a DER
/// number can never be silently attributed to a swapped/corrupted/non-FP32 model.
fn verify_model_integrity(
    registry: &ModelRegistry,
    profile: Profile,
    embedder_path: &Path,
    vad_path: &Path,
) -> Result<()> {
    let manifest = registry.manifest();
    let prof = manifest
        .profile(profile.manifest_id())
        .ok_or_else(|| anyhow::anyhow!("profile {} not in manifest", profile.manifest_id()))?;
    check_model_sha256(registry, &prof.embedder, embedder_path)?;
    check_model_sha256(registry, "silero_vad", vad_path)?;
    Ok(())
}

fn check_model_sha256(registry: &ModelRegistry, model_id: &str, path: &Path) -> Result<()> {
    let manifest = registry.manifest();
    let entry = manifest
        .model(model_id)
        .ok_or_else(|| anyhow::anyhow!("model {model_id} not in manifest"))?;
    let bytes = std::fs::read(path).with_context(|| format!("read model {}", path.display()))?;
    let got = hex_lower(&Sha256::digest(&bytes));
    if !got.eq_ignore_ascii_case(&entry.sha256) {
        anyhow::bail!(
            "model integrity FAIL for {model_id}: on-disk sha256 {got} != manifest {}",
            entry.sha256
        );
    }
    Ok(())
}

fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn main() -> Result<()> {
    let args = Args::parse();
    let profile = parse_profile(&args.profile)?;
    let registry = ModelRegistry::default().context("registry")?;
    let models = registry
        .ensure_for_profile(profile)
        .context("ensure models")?;

    let embedding_dim = profile.embedding_dim();
    let extractor = FbankOnnxExtractor::new(&models.embedder_path, embedding_dim, 1)
        .context("load embedder")?;
    let vad_path = registry.ensure("silero_vad").context("silero_vad model")?;
    let mut vad = SileroVad::new(&vad_path, 512).context("load vad")?;

    // Integrity gate: a DER number is only trustworthy if produced by
    // the EXACT shipped artifact — hard-fail if the on-disk embedder/VAD sha256
    // disagrees with the manifest (swapped, corrupted, or non-FP32 cache).
    verify_model_integrity(&registry, profile, &models.embedder_path, &vad_path)?;

    // Optional UEM scoped regions, keyed by file id.
    let uem_map: Option<HashMap<String, Vec<TimeRange>>> = match &args.uem {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read uem {}", path.display()))?;
            Some(parse_uem(&text))
        }
        None => None,
    };

    let config = DiarizationConfig {
        cluster: ClusterConfig {
            threshold: args.threshold,
            ..Default::default()
        },
        ..DiarizationConfig::default()
    };
    let vad_config = VadConfig::default();
    let pipeline = Pipeline::new(config, vad_config);

    let audio_dir = args.dataset.join("audio");
    let rttm_dir = args.dataset.join("rttm");
    let mut wavs: Vec<PathBuf> = std::fs::read_dir(&audio_dir)
        .with_context(|| format!("read_dir {}", audio_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
        .map(|e| e.path())
        .collect();
    wavs.sort();
    if let Some(n) = args.max_files {
        wavs.truncate(n);
    }

    let dataset_name = args
        .dataset
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_owned();

    let mut totals = Aggregate::default();
    let mut total_audio_secs = 0.0_f64;
    let mut total_runtime_secs = 0.0_f64;
    let mut speaker_count_exact = 0_usize;
    let mut speaker_count_pm1 = 0_usize;
    let mut speaker_count_off = 0_usize;
    let mut files_skipped = 0_usize;
    let mut per_file = Vec::with_capacity(wavs.len());

    for wav in &wavs {
        let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let rttm = rttm_dir.join(format!("{stem}.rttm"));
        if !rttm.is_file() {
            eprintln!("[SKIP] {stem}: no rttm");
            files_skipped += 1;
            continue;
        }
        let (samples, sr_hz) = read_wav(wav)?;
        let _sr = SampleRate::new(sr_hz)
            .ok_or_else(|| anyhow::anyhow!("invalid sample rate: {sr_hz}"))?;
        let audio_secs = samples.len() as f64 / sr_hz as f64;

        let t0 = Instant::now();
        let result = pipeline.run(&samples, &extractor, &mut vad)?;
        let runtime_secs = t0.elapsed().as_secs_f64();

        let ref_turns = {
            let raw = parse_rttm_file(&rttm).context("parse rttm")?;
            let grouped = group_by_file(&raw);
            // AMI-style fallback: EN2002a.Mix-Headset.wav → EN2002a
            let segs: Vec<_> = grouped
                .get(stem)
                .or_else(|| stem.split('.').next().and_then(|s| grouped.get(s)))
                .map(|v| v.iter().map(|s| (*s).clone()).collect())
                .unwrap_or_default();
            let (turns, _map) = to_speaker_turns(&segs);
            turns
        };

        // Headline collar + no-collar DER, restricted to the UEM scope when present
        // (AMI-style id fallback like the RTTM lookup). The overlap decomposition
        // below is a diagnostic and stays over the full file.
        let scored: Option<&[TimeRange]> = uem_map.as_ref().and_then(|m| {
            m.get(stem)
                .or_else(|| stem.split('.').next().and_then(|s| m.get(s)))
                .map(|v| v.as_slice())
        });
        let (der, der_no_collar) = match scored {
            Some(s) => (
                compute_der_with_uem(&ref_turns, &result.turns, args.collar, s),
                compute_der_with_uem(&ref_turns, &result.turns, 0.0, s),
            ),
            None => (
                compute_der(&ref_turns, &result.turns, args.collar),
                compute_der(&ref_turns, &result.turns, 0.0),
            ),
        };
        let decomp = compute_der_decomposition(&ref_turns, &result.turns, args.collar);

        let ref_speakers: HashSet<_> = ref_turns.iter().map(|t| t.speaker.0).collect();
        let hyp_speakers: HashSet<_> = result.turns.iter().map(|t| t.speaker.0).collect();
        let ref_count = ref_speakers.len();
        let hyp_count = hyp_speakers.len();
        let diff = ref_count.abs_diff(hyp_count);
        match diff {
            0 => speaker_count_exact += 1,
            1 => speaker_count_pm1 += 1,
            _ => speaker_count_off += 1,
        }

        totals.der_total += der.der;
        totals.der_no_collar_total += der_no_collar.der;
        totals.miss += der.miss_rate;
        totals.false_alarm += der.false_alarm_rate;
        totals.confusion += der.confusion_rate;
        totals.count += 1;
        // Frame sums for duration-weighted micro-averages — kept strictly
        // separate per collar pass (collar and no-collar frames must not mix).
        totals.collar_missed_frames += der.missed_frames;
        totals.collar_fa_frames += der.false_alarm_frames;
        totals.collar_confusion_frames += der.confusion_frames;
        totals.collar_ref_frames += der.total_ref_frames;
        totals.no_collar_missed_frames += der_no_collar.missed_frames;
        totals.no_collar_fa_frames += der_no_collar.false_alarm_frames;
        totals.no_collar_confusion_frames += der_no_collar.confusion_frames;
        totals.no_collar_ref_frames += der_no_collar.total_ref_frames;
        total_audio_secs += audio_secs;
        total_runtime_secs += runtime_secs;

        let rt_factor = audio_secs / runtime_secs.max(1e-6);

        println!(
            "{stem}\t DER={:.3}%\t miss={:.3}%\t fa={:.3}%\t conf={:.3}%\t rt={:.1}x\t spk={}\t turns={}",
            der.der * 100.0,
            der.miss_rate * 100.0,
            der.false_alarm_rate * 100.0,
            der.confusion_rate * 100.0,
            rt_factor,
            result.num_speakers,
            result.turns.len(),
        );

        per_file.push(PerFileResult {
            filename: stem.to_owned(),
            der_collar: der.der * 100.0,
            der_no_collar: der_no_collar.der * 100.0,
            miss_rate: der.miss_rate * 100.0,
            false_alarm_rate: der.false_alarm_rate * 100.0,
            confusion_rate: der.confusion_rate * 100.0,
            der_single_speaker: decomp.single_speaker.der * 100.0,
            der_overlap: decomp.overlap.der * 100.0,
            per_speaker_recall: decomp
                .per_speaker_recall
                .iter()
                .map(|s| PerSpeakerRecall {
                    speaker: s.speaker,
                    recall: s.recall,
                })
                .collect(),
            rt_factor,
            ref_speakers: ref_count,
            hyp_speakers: hyp_count,
            num_turns: result.turns.len(),
            audio_duration_secs: audio_secs,
            runtime_secs,
        });
    }

    let n = totals.count.max(1) as f64;
    let der_collar_macro = (totals.der_total / n) * 100.0;
    let der_no_collar_macro = (totals.der_no_collar_total / n) * 100.0;
    let der_collar_micro = micro_der(
        totals.collar_missed_frames,
        totals.collar_fa_frames,
        totals.collar_confusion_frames,
        totals.collar_ref_frames,
    );
    let der_no_collar_micro = micro_der(
        totals.no_collar_missed_frames,
        totals.no_collar_fa_frames,
        totals.no_collar_confusion_frames,
        totals.no_collar_ref_frames,
    );

    println!(
        "\n=== Aggregate DER over {} files (collar={:.2}s) ===",
        totals.count, args.collar
    );
    println!("  der_collar    : macro={der_collar_macro:.2}%  micro={der_collar_micro:.2}%");
    println!("  der_no_collar : macro={der_no_collar_macro:.2}%  micro={der_no_collar_micro:.2}%");

    let report = BenchReport {
        schema: "polyvoice-bench-v0.8",
        crate_version: env!("CARGO_PKG_VERSION"),
        git_sha: git_sha(),
        host_arch: std::env::consts::ARCH.to_owned(),
        host_os: std::env::consts::OS.to_owned(),
        command_line: std::env::args().collect::<Vec<_>>().join(" "),
        dataset_name,
        profile: args.profile.clone(),
        files_processed: totals.count,
        files_skipped,
        der_collar_macro,
        der_no_collar_macro,
        der_collar_micro,
        der_no_collar_micro,
        collar_secs: args.collar,
        averaging_policy: "macro = mean of per-file DER; micro = frame-weighted (sum error frames / sum ref frames)",
        miss: (totals.miss / n) * 100.0,
        false_alarm: (totals.false_alarm / n) * 100.0,
        confusion: (totals.confusion / n) * 100.0,
        rt_factor_avg: total_audio_secs / total_runtime_secs.max(1e-6),
        speaker_count: SpeakerCountDiagnostics {
            exact: speaker_count_exact,
            plus_minus_1: speaker_count_pm1,
            off_by_2_or_more: speaker_count_off,
        },
        model_hashes: model_hashes(&registry, profile),
        per_file,
    };
    let json = serde_json::to_string_pretty(&report)?;
    match args.output {
        Some(p) => std::fs::write(&p, json)?,
        None => println!("{json}"),
    }
    Ok(())
}

#[derive(Default)]
struct Aggregate {
    der_total: f64,
    der_no_collar_total: f64,
    miss: f64,
    false_alarm: f64,
    confusion: f64,
    count: usize,
    // Frame sums for duration-weighted micro-averages (collar pass).
    collar_missed_frames: u64,
    collar_fa_frames: u64,
    collar_confusion_frames: u64,
    collar_ref_frames: u64,
    // Frame sums (no-collar pass).
    no_collar_missed_frames: u64,
    no_collar_fa_frames: u64,
    no_collar_confusion_frames: u64,
    no_collar_ref_frames: u64,
}

/// Duration-weighted micro-average DER as a percentage: total error frames over
/// total reference frames (not a mean of per-file ratios). Returns 0.0 when no
/// reference frames were seen.
fn micro_der(missed: u64, false_alarm: u64, confusion: u64, ref_frames: u64) -> f64 {
    if ref_frames == 0 {
        0.0
    } else {
        (missed + false_alarm + confusion) as f64 / ref_frames as f64 * 100.0
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn bench_args_parses_with_valid_args(
            profile in "(mobile|balanced)",
            collar in 0.0f64..1.0f64,
            threshold in 0.0f32..1.0f32,
            max_files in 0usize..100usize,
        ) {
            let args = vec![
                "polyvoice-bench".to_string(),
                "/tmp/dataset".to_string(),
                "--profile".to_string(), profile,
                "--collar".to_string(), collar.to_string(),
                "--threshold".to_string(), threshold.to_string(),
                "--max-files".to_string(), max_files.to_string(),
            ];
            let result = Args::try_parse_from(&args);
            prop_assert!(result.is_ok());
        }

        #[test]
        fn parse_profile_accepts_only_valid(s in "[a-zA-Z0-9_-]{1,20}") {
            let result = parse_profile(&s);
            if s == "mobile" || s == "balanced" {
                prop_assert!(result.is_ok());
            } else {
                prop_assert!(result.is_err());
            }
        }
    }
}
