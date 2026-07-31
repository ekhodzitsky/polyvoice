//! polyvoice-bench — DER on a {audio,rttm} dataset directory.
//!
//! Default pipeline matches the shipped CLI (**v2 + VBx** since 0.11). Pass
//! `--pipeline legacy` for the pre-0.11 Silero + AHC path, or `--clusterer ahc`
//! to keep v2 segmentation with fixed-threshold AHC.

use anyhow::{Context, Result};
use clap::Parser;
use polyvoice::cli_common;
use polyvoice::der::{
    DerResult, compute_der, compute_der_decomposition, compute_der_single_speaker_regions,
    compute_der_with_uem, parse_uem,
};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::LegacyPipeline;
use polyvoice::pipeline_v2::{Pipeline as V2Pipeline, PipelineConfig, StageTimings};
use polyvoice::types::{DiarizationResult, Profile, SampleRate, TimeRange};
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
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
    /// Score the headline DER over single-speaker reference regions only
    /// (md-eval skip-overlap semantics): overlap frames are excluded from both
    /// the speaker mapping and the error counts, per file and in all
    /// aggregates. Incompatible with --uem.
    #[arg(long, default_value = "false")]
    skip_overlap: bool,
    #[arg(long)]
    max_files: Option<usize>,
    #[arg(long, default_value_t = polyvoice::DEFAULT_AHC_THRESHOLD)]
    threshold: f32,
    /// Which pipeline to benchmark: `v2` (powerset segmentation + embeddings +
    /// clusterer + overlap resegmentation — the shipped CLI default) or
    /// `legacy` (Silero VAD + sliding-window embeddings + AHC).
    #[arg(long, default_value = "v2")]
    pipeline: String,
    /// Min cluster size (members): clusters smaller than this are dissolved into
    /// the nearest large speaker. Applies to both pipelines.
    #[arg(long)]
    min_cluster_size: Option<usize>,
    /// v2 clusterer: `vbx` (Variational Bayes HMM + PLDA with automatic speaker
    /// count — matches CLI default; PLDA from env/dir/registry) or `ahc`
    /// (fixed-threshold AHC). Ignored with `--pipeline legacy`.
    #[arg(long, default_value = "vbx")]
    clusterer: String,
    /// Min cluster duration in seconds (length-invariant pruning). When > 0 it
    /// takes precedence over --min-cluster-size on the legacy pipeline.
    #[arg(long)]
    min_cluster_secs: Option<f64>,
    /// Optional .uem file. Restricts DER to the scored regions per file (frames
    /// outside the UEM are dropped from both mapping and counts).
    #[arg(long)]
    uem: Option<PathBuf>,
    /// v2 dense embedding window (seconds): split segments into `w`-sec windows
    /// (hop w/2) for more embeddings per speaker. Omit for one embedding/segment.
    #[arg(long)]
    embed_window: Option<f32>,
    /// ONNX execution provider: auto|cpu|coreml|nnapi|cuda|xnnpack. Omitted =
    /// each pipeline's shipped default (legacy embedder: cpu; v2: auto), so
    /// committed DER baselines stay reproducible. The resolved provider is
    /// recorded in the report for per-backend RTFx comparison.
    #[arg(long)]
    execution_provider: Option<String>,
    /// v2 binarization: enter-speech (onset) threshold. Setting ANY
    /// --binarize-* flag enables calibrated hysteresis binarization of the
    /// segmentation posteriors (defaults for unset knobs: onset/offset 0.5,
    /// min durations 0).
    #[arg(long)]
    binarize_onset: Option<f32>,
    /// v2 binarization: leave-speech (offset) threshold (< onset = hysteresis).
    #[arg(long)]
    binarize_offset: Option<f32>,
    /// v2 binarization: drop active runs shorter than this many seconds.
    #[arg(long)]
    binarize_min_on: Option<f32>,
    /// v2 binarization: bridge gaps shorter than this many seconds.
    #[arg(long)]
    binarize_min_off: Option<f32>,
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
    /// Per-stage wall-clock seconds (v2 pipeline only; absent on legacy).
    #[serde(skip_serializing_if = "Option::is_none")]
    stage_timings: Option<StageTimings>,
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
    /// True when --skip-overlap was active: the headline DER (per file and in
    /// all aggregates) is computed over single-speaker reference regions only.
    skip_overlap: bool,
    averaging_policy: &'static str,
    /// Debug-formatted resolved execution provider (e.g. "CoreMl", "Cpu") —
    /// labels every report for per-backend RTFx comparison.
    resolved_execution_provider: String,
    host_cpus: usize,
    /// Sum of per-stage wall-clock seconds across files (v2 only).
    #[serde(skip_serializing_if = "Option::is_none")]
    stage_totals: Option<StageTimings>,
    miss: f64,
    false_alarm: f64,
    confusion: f64,
    rt_factor_avg: f64,
    speaker_count: SpeakerCountDiagnostics,
    model_hashes: Vec<ModelHash>,
    per_file: Vec<PerFileResult>,
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

fn model_hashes(registry: &ModelRegistry, profile: Profile, segmenter_id: &str) -> Vec<ModelHash> {
    let mut out = Vec::new();
    let manifest = registry.manifest();
    let prof = match manifest.profile(profile.manifest_id()) {
        Some(p) => p,
        None => return out,
    };
    // Report exactly the models the chosen pipeline actually loads: the legacy
    // path segments with Silero VAD, the v2 path with the profile's powerset
    // segmenter — `segmenter_id` carries the right one. Both embed with the
    // profile embedder. This keeps the integrity record honest about what
    // produced the DER number.
    for model_id in [segmenter_id, prof.embedder.as_str()] {
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

/// Legacy pipeline + its ONNX sessions (Silero VAD + sliding-window embedder).
struct LegacyRunner {
    pipeline: LegacyPipeline,
    stack: cli_common::LegacyStack,
}

/// The pipeline under benchmark. Both arms produce a `DiarizationResult` so all
/// downstream DER / speaker-count reporting is shared. Both payloads are boxed
/// so the variants are the same (pointer) size.
enum Runner {
    Legacy(Box<LegacyRunner>),
    V2(Box<V2Pipeline>),
}

impl Runner {
    fn run(
        &mut self,
        samples: &[f32],
        sr: SampleRate,
    ) -> Result<(DiarizationResult, Option<StageTimings>)> {
        match self {
            Runner::Legacy(l) => Ok((
                l.pipeline
                    .run(samples, &l.stack.extractor, &mut l.stack.vad)?,
                None,
            )),
            Runner::V2(p) => {
                let (result, timings) = p.run_with_timings(samples, sr)?;
                Ok((result, Some(timings)))
            }
        }
    }
}

/// The runner plus everything the report needs that is file-independent.
struct BenchRunner {
    runner: Runner,
    segmenter_id: String,
    resolved_ep: polyvoice::onnx::ExecutionProvider,
    profile: Profile,
    registry: ModelRegistry,
}

/// Build the requested pipeline. Each arm verifies the integrity of exactly
/// the models it loads and yields the segmenter id for the report.
fn build_runner(args: &Args) -> Result<BenchRunner> {
    let profile: Profile = args.profile.parse()?;
    let registry = ModelRegistry::default().context("registry")?;
    let models = registry
        .ensure_for_profile(profile)
        .context("ensure models")?;

    // Resolve the execution provider: an explicit flag applies to the selected
    // pipeline; omitted keeps each pipeline's shipped default (legacy embedder
    // cpu, v2 auto) so committed DER baselines stay reproducible.
    let explicit_ep = args
        .execution_provider
        .as_deref()
        .map(cli_common::parse_execution_provider)
        .transpose()?;
    let resolved_ep = match args.pipeline.as_str() {
        "v2" => explicit_ep.unwrap_or_else(polyvoice::onnx::ExecutionProvider::auto),
        _ => explicit_ep.unwrap_or(polyvoice::onnx::ExecutionProvider::Cpu),
    };

    let (runner, segmenter_id): (Runner, String) = match args.pipeline.as_str() {
        "v2" => {
            let clusterer = cli_common::parse_clusterer_kind(&args.clusterer, args.threshold)?;
            let binarization = if args.binarize_onset.is_some()
                || args.binarize_offset.is_some()
                || args.binarize_min_on.is_some()
                || args.binarize_min_off.is_some()
            {
                let d = polyvoice::segmentation::BinarizationConfig::default();
                Some(polyvoice::segmentation::BinarizationConfig {
                    onset: args.binarize_onset.unwrap_or(d.onset),
                    offset: args.binarize_offset.unwrap_or(d.offset),
                    min_duration_on: args.binarize_min_on.unwrap_or(d.min_duration_on),
                    min_duration_off: args.binarize_min_off.unwrap_or(d.min_duration_off),
                })
            } else {
                None
            };
            let mut cfg = PipelineConfig {
                profile,
                clusterer,
                embed_window_secs: args.embed_window,
                execution_provider: resolved_ep,
                binarization,
                ..PipelineConfig::default()
            };
            if let Some(mcs) = args.min_cluster_size {
                cfg.min_cluster_size = mcs;
            }
            // v2 segments with the profile's powerset model — verify it + embedder.
            let seg_id = registry
                .manifest()
                .profile(profile.manifest_id())
                .map(|p| p.segmenter.clone())
                .unwrap_or_else(|| "powerset_fp32".to_owned());
            let emb_id = registry
                .manifest()
                .profile(profile.manifest_id())
                .map(|p| p.embedder.clone())
                .unwrap_or_default();
            check_model_sha256(&registry, &seg_id, &models.segmenter_path)?;
            check_model_sha256(&registry, &emb_id, &models.embedder_path)?;
            let pipeline = cli_common::build_v2_pipeline(cfg, registry.clone())?;
            (Runner::V2(Box::new(pipeline)), seg_id)
        }
        other => {
            if other != "legacy" {
                anyhow::bail!("unknown --pipeline '{other}' (expected 'legacy' or 'v2')");
            }
            let vad_path = registry.ensure("silero_vad").context("silero_vad model")?;
            let stack = cli_common::load_legacy_stack(
                &models.embedder_path,
                profile.embedding_dim(),
                resolved_ep,
                &vad_path,
                512,
            )?;

            // Integrity gate: a DER number is only trustworthy if produced by
            // the EXACT shipped artifact — hard-fail if the on-disk embedder/VAD
            // sha256 disagrees with the manifest (swapped/corrupted/non-FP32).
            verify_model_integrity(&registry, profile, &models.embedder_path, &vad_path)?;

            let mut config = cli_common::legacy_diarization_config(args.threshold);
            config.cluster.min_cluster_size = args.min_cluster_size.unwrap_or(1);
            config.cluster.min_cluster_secs = args.min_cluster_secs.unwrap_or(0.0);
            let pipeline = LegacyPipeline::new(config, VadConfig::default());
            (
                Runner::Legacy(Box::new(LegacyRunner { pipeline, stack })),
                "silero_vad".to_owned(),
            )
        }
    };
    Ok(BenchRunner {
        runner,
        segmenter_id,
        resolved_ep,
        profile,
        registry,
    })
}

/// One scored file: the report row plus the inputs the aggregates need.
struct FileOutcome {
    row: PerFileResult,
    der_pair: (DerResult, DerResult),
    ref_count: usize,
    hyp_count: usize,
    audio_secs: f64,
    runtime_secs: f64,
}

/// Run one wav through the pipeline and score it. `Ok(None)` means the file
/// was skipped (no reference RTTM).
fn run_file(
    runner: &mut Runner,
    wav: &Path,
    rttm_dir: &Path,
    uem_map: Option<&HashMap<String, Vec<TimeRange>>>,
    args: &Args,
) -> Result<Option<FileOutcome>> {
    let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let rttm = rttm_dir.join(format!("{stem}.rttm"));
    if !rttm.is_file() {
        eprintln!("[SKIP] {stem}: no rttm");
        return Ok(None);
    }
    let (samples, sr_hz) = read_wav(wav)?;
    let sr =
        SampleRate::new(sr_hz).ok_or_else(|| anyhow::anyhow!("invalid sample rate: {sr_hz}"))?;
    let audio_secs = samples.len() as f64 / sr_hz as f64;

    let t0 = Instant::now();
    let (result, stage_timings) = runner.run(&samples, sr)?;
    let runtime_secs = t0.elapsed().as_secs_f64();

    let ref_turns = cli_common::load_ref_turns(rttm_dir, stem)?;

    // Headline collar + no-collar DER, restricted to the UEM scope when present
    // (AMI-style id fallback like the RTTM lookup). With --skip-overlap the
    // headline is the single-speaker-regions DER (md-eval skip-overlap
    // semantics); that mode rejects --uem at startup. The overlap
    // decomposition below is a diagnostic and stays over the full file.
    let scored: Option<&[TimeRange]> = uem_map.and_then(|m| {
        m.get(stem)
            .or_else(|| stem.split('.').next().and_then(|s| m.get(s)))
            .map(|v| v.as_slice())
    });
    let (der, der_no_collar) = if args.skip_overlap {
        (
            compute_der_single_speaker_regions(&ref_turns, &result.turns, args.collar),
            compute_der_single_speaker_regions(&ref_turns, &result.turns, 0.0),
        )
    } else {
        match scored {
            Some(s) => (
                compute_der_with_uem(&ref_turns, &result.turns, args.collar, s),
                compute_der_with_uem(&ref_turns, &result.turns, 0.0, s),
            ),
            None => (
                compute_der(&ref_turns, &result.turns, args.collar),
                compute_der(&ref_turns, &result.turns, 0.0),
            ),
        }
    };
    let decomp = compute_der_decomposition(&ref_turns, &result.turns, args.collar);

    let ref_speakers: HashSet<_> = ref_turns.iter().map(|t| t.speaker.0).collect();
    let hyp_speakers: HashSet<_> = result.turns.iter().map(|t| t.speaker.0).collect();
    let ref_count = ref_speakers.len();
    let hyp_count = hyp_speakers.len();

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

    let row = PerFileResult {
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
        stage_timings,
    };
    Ok(Some(FileOutcome {
        row,
        der_pair: (der, der_no_collar),
        ref_count,
        hyp_count,
        audio_secs,
        runtime_secs,
    }))
}

/// Per-run accumulators, folded one [`FileOutcome`] at a time.
#[derive(Default)]
struct Accum {
    totals: Aggregate,
    /// Per-file (collar, no-collar) DER pairs — the four report aggregates are
    /// computed from these by the unit-tested aggregate_der helper.
    der_pairs: Vec<(DerResult, DerResult)>,
    speaker_exact: usize,
    speaker_pm1: usize,
    speaker_off: usize,
    files_skipped: usize,
    total_audio_secs: f64,
    total_runtime_secs: f64,
    stage_totals: Option<StageTimings>,
    per_file: Vec<PerFileResult>,
}

impl Accum {
    fn record(&mut self, outcome: FileOutcome) {
        let FileOutcome {
            row,
            der_pair,
            ref_count,
            hyp_count,
            audio_secs,
            runtime_secs,
        } = outcome;
        let (der, der_no_collar) = der_pair;
        self.totals.miss += der.miss_rate;
        self.totals.false_alarm += der.false_alarm_rate;
        self.totals.confusion += der.confusion_rate;
        self.totals.count += 1;
        match ref_count.abs_diff(hyp_count) {
            0 => self.speaker_exact += 1,
            1 => self.speaker_pm1 += 1,
            _ => self.speaker_off += 1,
        }
        self.der_pairs.push((der, der_no_collar));
        self.total_audio_secs += audio_secs;
        self.total_runtime_secs += runtime_secs;
        if let Some(t) = &row.stage_timings {
            let acc = self.stage_totals.get_or_insert_with(StageTimings::default);
            acc.segmentation_secs += t.segmentation_secs;
            acc.embedding_secs += t.embedding_secs;
            acc.clustering_secs += t.clustering_secs;
            acc.resegmentation_secs += t.resegmentation_secs;
        }
        self.per_file.push(row);
    }
}

/// Print the aggregate summary and assemble the JSON report.
fn build_report(args: &Args, b: &BenchRunner, dataset_name: String, acc: Accum) -> BenchReport {
    let n = acc.totals.count.max(1) as f64;
    let agg = aggregate_der(&acc.der_pairs);

    println!(
        "\n=== Aggregate DER over {} files (collar={:.2}s) ===",
        acc.totals.count, args.collar
    );
    if args.skip_overlap {
        println!("  skip-overlap  : ON (single-speaker reference regions only)");
    }
    println!(
        "  der_collar    : macro={:.2}%  micro={:.2}%",
        agg.collar_macro, agg.collar_micro
    );
    println!(
        "  der_no_collar : macro={:.2}%  micro={:.2}%",
        agg.no_collar_macro, agg.no_collar_micro
    );

    BenchReport {
        schema: "polyvoice-bench-v0.10",
        crate_version: env!("CARGO_PKG_VERSION"),
        git_sha: git_sha(),
        host_arch: std::env::consts::ARCH.to_owned(),
        host_os: std::env::consts::OS.to_owned(),
        command_line: std::env::args().collect::<Vec<_>>().join(" "),
        dataset_name,
        profile: args.profile.clone(),
        files_processed: acc.totals.count,
        files_skipped: acc.files_skipped,
        der_collar_macro: agg.collar_macro,
        der_no_collar_macro: agg.no_collar_macro,
        der_collar_micro: agg.collar_micro,
        der_no_collar_micro: agg.no_collar_micro,
        collar_secs: args.collar,
        skip_overlap: args.skip_overlap,
        averaging_policy: "macro = mean of per-file DER; micro = frame-weighted (sum error frames / sum ref frames)",
        resolved_execution_provider: format!("{:?}", b.resolved_ep),
        host_cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        stage_totals: acc.stage_totals,
        miss: (acc.totals.miss / n) * 100.0,
        false_alarm: (acc.totals.false_alarm / n) * 100.0,
        confusion: (acc.totals.confusion / n) * 100.0,
        rt_factor_avg: acc.total_audio_secs / acc.total_runtime_secs.max(1e-6),
        speaker_count: SpeakerCountDiagnostics {
            exact: acc.speaker_exact,
            plus_minus_1: acc.speaker_pm1,
            off_by_2_or_more: acc.speaker_off,
        },
        model_hashes: model_hashes(&b.registry, b.profile, &b.segmenter_id),
        per_file: acc.per_file,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    // The DER library has no single-speaker-regions + UEM scorer, so this
    // combination cannot be honoured — fail loudly instead of scoring the
    // wrong thing.
    if args.skip_overlap && args.uem.is_some() {
        anyhow::bail!("--skip-overlap cannot be combined with --uem");
    }
    if args.skip_overlap {
        println!("skip-overlap: headline DER over single-speaker reference regions only");
    }
    let mut b = build_runner(&args)?;

    // Optional UEM scoped regions, keyed by file id.
    let uem_map: Option<HashMap<String, Vec<TimeRange>>> = match &args.uem {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read uem {}", path.display()))?;
            Some(parse_uem(&text))
        }
        None => None,
    };

    let wavs = cli_common::list_wavs(&args.dataset, args.max_files)?;
    let rttm_dir = args.dataset.join("rttm");
    let dataset_name = args
        .dataset
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_owned();

    let mut acc = Accum::default();
    for wav in &wavs {
        match run_file(&mut b.runner, wav, &rttm_dir, uem_map.as_ref(), &args)? {
            Some(outcome) => acc.record(outcome),
            None => acc.files_skipped += 1,
        }
    }

    let report = build_report(&args, &b, dataset_name, acc);
    let json = serde_json::to_string_pretty(&report)?;
    match args.output {
        Some(p) => std::fs::write(&p, json)?,
        None => println!("{json}"),
    }
    Ok(())
}

#[derive(Default)]
struct Aggregate {
    miss: f64,
    false_alarm: f64,
    confusion: f64,
    count: usize,
}

/// The four report aggregates, as percentages.
struct DerAggregates {
    collar_macro: f64,
    no_collar_macro: f64,
    collar_micro: f64,
    no_collar_micro: f64,
}

/// Compute collar/no-collar x macro/micro DER from per-file result pairs.
/// Macro = mean of per-file ratios; micro = duration-weighted (summed error
/// frames / summed reference frames), with collar and no-collar frame sums
/// kept strictly separate. This is THE aggregation the report publishes —
/// unit-tested so a refactor cannot silently revert micro to a ratio-average
/// or swap the collar passes.
fn aggregate_der(pairs: &[(DerResult, DerResult)]) -> DerAggregates {
    let n = pairs.len().max(1) as f64;
    let (mut cm, mut cf, mut cc, mut cr) = (0u64, 0u64, 0u64, 0u64);
    let (mut nm, mut nf, mut nc, mut nr) = (0u64, 0u64, 0u64, 0u64);
    for (c, n_) in pairs {
        cm += c.missed_frames;
        cf += c.false_alarm_frames;
        cc += c.confusion_frames;
        cr += c.total_ref_frames;
        nm += n_.missed_frames;
        nf += n_.false_alarm_frames;
        nc += n_.confusion_frames;
        nr += n_.total_ref_frames;
    }
    DerAggregates {
        collar_macro: pairs.iter().map(|(c, _)| c.der).sum::<f64>() / n * 100.0,
        no_collar_macro: pairs.iter().map(|(_, x)| x.der).sum::<f64>() / n * 100.0,
        collar_micro: micro_der(cm, cf, cc, cr),
        no_collar_micro: micro_der(nm, nf, nc, nr),
    }
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

    /// Synthetic DerResult: `errors` error frames over `ref_frames` (all miss).
    fn synth(errors: u64, ref_frames: u64) -> DerResult {
        DerResult {
            der: errors as f64 / ref_frames as f64,
            miss_rate: errors as f64 / ref_frames as f64,
            false_alarm_rate: 0.0,
            confusion_rate: 0.0,
            total_speech: ref_frames as f64 * 0.01,
            total_ref_frames: ref_frames,
            missed_frames: errors,
            false_alarm_frames: 0,
            confusion_frames: 0,
        }
    }

    #[test]
    fn aggregate_macro_diverges_from_micro_and_micro_is_frame_weighted() {
        // A tiny 1s file at 50% DER and a long 60s file at 1% DER: the mean of
        // ratios (macro) must NOT equal the frame-weighted micro, and micro
        // must equal summed error frames / summed reference frames exactly.
        let short = synth(50, 100);
        let long = synth(60, 6000);
        let agg = aggregate_der(&[(short, short), (long, long)]);
        assert!(
            (agg.collar_macro - 25.5).abs() < 1e-9,
            "{}",
            agg.collar_macro
        );
        let expected_micro = (50 + 60) as f64 / (100 + 6000) as f64 * 100.0;
        assert!((agg.collar_micro - expected_micro).abs() < 1e-9);
        assert!((agg.collar_macro - agg.collar_micro).abs() > 10.0);
        // Same inputs on both passes => identical aggregates per pass.
        assert_eq!(agg.collar_micro, agg.no_collar_micro);
    }

    #[test]
    fn aggregate_no_collar_at_least_collar_on_boundary_errors() {
        use polyvoice::types::{SpeakerId, SpeakerTurn, TimeRange};
        let turn = |s: u32, a: f64, b: f64| SpeakerTurn {
            speaker: SpeakerId(s),
            time: TimeRange { start: a, end: b },
            text: None,
            stable: true,
        };
        // Hypothesis shifted 0.3s off every reference boundary: the collar
        // forgives part of that error, no-collar must not.
        let reference = vec![turn(0, 0.0, 10.0), turn(1, 12.0, 20.0)];
        let hypothesis = vec![turn(0, 0.3, 10.3), turn(1, 12.3, 20.3)];
        let collar = compute_der(&reference, &hypothesis, 0.25);
        let no_collar = compute_der(&reference, &hypothesis, 0.0);
        let agg = aggregate_der(&[(collar, no_collar)]);
        assert!(
            agg.no_collar_micro >= agg.collar_micro,
            "no-collar {} < collar {}",
            agg.no_collar_micro,
            agg.collar_micro
        );
        assert!(agg.no_collar_macro >= agg.collar_macro);
        assert!(agg.no_collar_micro > 0.0, "boundary errors must be scored");
    }

    proptest! {
        #[test]
        fn bench_args_parses_with_valid_args(
            profile in "(mobile|balanced|fast)",
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
        fn profile_from_str_accepts_only_known_names(s in "[a-zA-Z0-9_-]{1,20}") {
            let result = s.parse::<Profile>();
            let lower = s.to_ascii_lowercase();
            if lower == "mobile" || lower == "balanced" || lower == "custom" {
                prop_assert!(result.is_ok());
            } else {
                prop_assert!(result.is_err());
            }
        }
    }
}
