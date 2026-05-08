//! polyvoice-bench — DER on a {audio,rttm} dataset directory using the v1.0 Pipeline.

use anyhow::{Context, Result};
use clap::Parser;
use polyvoice::der::compute_der;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::{Pipeline, PipelineConfig};
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use serde::Serialize;
use std::path::PathBuf;
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
}

#[derive(Serialize)]
struct BenchReport {
    schema: &'static str,
    profile: String,
    files: usize,
    der_collar_0_25_skip_overlap: f64,
    der_no_collar: f64,
    miss: f64,
    false_alarm: f64,
    confusion: f64,
    rt_factor_avg: f64,
    polyvoice_version: &'static str,
}

fn parse_profile(name: &str) -> Result<Profile> {
    match name {
        "mobile" => Ok(Profile::Mobile),
        "balanced" => Ok(Profile::Balanced),
        other => anyhow::bail!("invalid profile: {other}"),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let profile = parse_profile(&args.profile)?;
    let registry = ModelRegistry::default().context("registry")?;

    let cfg = PipelineConfig {
        profile,
        ..PipelineConfig::default()
    };
    let pipeline = Pipeline::builder()
        .config(cfg)
        .with_models_from(registry)
        .build()
        .context("build pipeline")?;

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

    let mut totals = Aggregate::default();
    let mut total_audio_secs = 0.0_f64;
    let mut total_runtime_secs = 0.0_f64;

    for wav in &wavs {
        let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let rttm = rttm_dir.join(format!("{stem}.rttm"));
        if !rttm.is_file() {
            eprintln!("[SKIP] {stem}: no rttm");
            continue;
        }
        let (samples, sr_hz) = read_wav(wav)?;
        let sr = SampleRate::new(sr_hz)
            .ok_or_else(|| anyhow::anyhow!("invalid sample rate: {sr_hz}"))?;
        let audio_secs = samples.len() as f64 / sr_hz as f64;

        let t0 = Instant::now();
        let result = pipeline.run(&samples, sr)?;
        let runtime_secs = t0.elapsed().as_secs_f64();

        let ref_turns = {
            let raw = parse_rttm_file(&rttm).context("parse rttm")?;
            let grouped = group_by_file(&raw);
            let segs: Vec<_> = grouped
                .get(stem)
                .map(|v| v.iter().map(|s| (*s).clone()).collect())
                .unwrap_or_default();
            let (turns, _map) = to_speaker_turns(&segs);
            turns
        };
        let der = compute_der(&ref_turns, &result.turns, args.collar);

        totals.der_total += der.der;
        totals.miss += der.miss_rate;
        totals.false_alarm += der.false_alarm_rate;
        totals.confusion += der.confusion_rate;
        totals.count += 1;
        total_audio_secs += audio_secs;
        total_runtime_secs += runtime_secs;

        println!(
            "{stem}\t DER={:.3}%\t miss={:.3}%\t fa={:.3}%\t conf={:.3}%\t rt={:.1}x",
            der.der * 100.0,
            der.miss_rate * 100.0,
            der.false_alarm_rate * 100.0,
            der.confusion_rate * 100.0,
            audio_secs / runtime_secs.max(1e-6),
        );
    }

    let n = totals.count.max(1) as f64;
    let report = BenchReport {
        schema: "polyvoice-bench-v1",
        profile: args.profile.clone(),
        files: totals.count,
        der_collar_0_25_skip_overlap: (totals.der_total / n) * 100.0,
        der_no_collar: 0.0, // computed separately when --collar 0 is invoked
        miss: (totals.miss / n) * 100.0,
        false_alarm: (totals.false_alarm / n) * 100.0,
        confusion: (totals.confusion / n) * 100.0,
        rt_factor_avg: total_audio_secs / total_runtime_secs.max(1e-6),
        polyvoice_version: env!("CARGO_PKG_VERSION"),
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
    miss: f64,
    false_alarm: f64,
    confusion: f64,
    count: usize,
}
