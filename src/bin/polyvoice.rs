//! polyvoice — speaker diarization CLI.
//!
//! `polyvoice meeting.wav` diarizes a file (implicit `diarize`); subcommands
//! `models` / `download-models` / `completions` are still available. Default
//! pipeline (since 0.11): **v2 + VBx** (powerset segmentation, ResNet34
//! embeddings, VB-HMM + PLDA clustering). PLDA weights come from
//! `--vbx-plda-dir` / `POLYVOICE_VBX_PLDA_DIR`, or are auto-downloaded via the
//! model registry when neither is set (or pass `--clusterer ahc`).
//! Use `--legacy` for the pre-0.11 Silero + AHC path.
//!
//! Audio input: without the `audio-io` build feature, only mono 16 kHz WAV is
//! accepted. Rebuild with `--features "cli,audio-io"` to decode mp3/flac/ogg/
//! m4a/aac (and other containers) and resample any rate → 16 kHz mono.
//!
//! STDOUT discipline: only the diarization result goes to stdout (so `--format
//! json` / `--json` and downstream pipes stay clean); all progress and info go to
//! stderr.

use anyhow::{Context, Result};
use clap::{Args, CommandFactory, Parser, Subcommand};
use polyvoice::cli_common;
use polyvoice::format::{write_srt, write_txt, write_vtt};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::LegacyPipeline;
use polyvoice::pipeline_v2::PipelineConfig;
use polyvoice::rttm::write_rttm;
use polyvoice::types::{DiarizationResult, Profile, SampleRate};
use polyvoice::vad::VadConfig;
use polyvoice::wav::load_audio;
use std::io::Write;
use std::path::{Path, PathBuf};

#[cfg(feature = "audio-io")]
const INPUT_HELP: &str = "Audio file to diarize (mp3/flac/ogg/m4a/aac/wav at any sample rate; decoded and resampled to 16 kHz mono)";
#[cfg(not(feature = "audio-io"))]
const INPUT_HELP: &str = "WAV file to diarize (mono 16 kHz). Rebuild with --features audio-io for mp3/flac/ogg/m4a and any-rate resampling";

#[derive(Parser, Debug)]
#[command(
    name = "polyvoice",
    version,
    about = "Speaker diarization toolkit",
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    /// Default (implicit-`diarize`) arguments, used when no subcommand is given.
    #[command(flatten)]
    diarize: DiarizeArgs,
}

/// Diarization arguments, shared by `polyvoice <wav>` and `polyvoice diarize <wav>`.
#[derive(Args, Debug)]
struct DiarizeArgs {
    /// Input audio path. Bare `polyvoice meeting.wav` is implicit diarize.
    #[arg(help = INPUT_HELP)]
    wav: Option<PathBuf>,
    #[arg(long, default_value = "balanced")]
    profile: String,
    /// Write output to a file instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Output format.
    #[arg(long, default_value = "rttm")]
    format: OutputFormat,
    #[arg(long)]
    models_cache: Option<PathBuf>,
    /// AHC merge threshold on the active scorer's scale: raw cosine (default
    /// 0.45), or AS-norm z-score when `--as-norm` is set (default 4.0).
    /// An explicit value wins over `--domain-profile`'s calibrated threshold.
    #[arg(long)]
    threshold: Option<f32>,
    /// Target speaker count (caps clustering at N speakers).
    #[arg(long)]
    speakers: Option<usize>,
    /// Maximum number of speakers (clustering ceiling).
    #[arg(long)]
    max_speakers: Option<usize>,
    /// Suppress progress/info on stderr.
    #[arg(long)]
    quiet: bool,
    /// Machine mode: emit only the structured JSON result on stdout; route all
    /// human-readable output to stderr. Implies `--format json --quiet`.
    #[arg(long)]
    json: bool,
    /// Use the pre-0.11 legacy pipeline (Silero VAD + sliding-window embeddings
    /// + AHC) instead of the default v2 + VBx path.
    #[arg(long)]
    legacy: bool,
    /// Deprecated no-op: pipeline v2 is the default since 0.11. Kept so scripts
    /// that still pass `--v2` keep working.
    #[arg(long, hide = true)]
    v2: bool,
    /// Clusterer for the default (v2) path: `vbx` (PLDA + VB-HMM, automatic
    /// speaker count — the accuracy gate default) or `ahc` (fixed-threshold
    /// cosine AHC). `vbx` loads PLDA from `--vbx-plda-dir` /
    /// `POLYVOICE_VBX_PLDA_DIR`, or auto-downloads via the model registry.
    /// Ignored with `--legacy`.
    #[arg(long, default_value = "vbx")]
    clusterer: String,
    /// Directory with the precomputed VBx PLDA params (overrides
    /// `POLYVOICE_VBX_PLDA_DIR` and the registry auto-download). Used when
    /// `--clusterer vbx` (the default).
    #[arg(long)]
    vbx_plda_dir: Option<PathBuf>,
    /// AS-norm score normalization for the AHC clusterer: pairwise cosine
    /// scores are z-normalized against an imposter cohort before merging, so
    /// one threshold generalizes across recording domains. Requires
    /// `--clusterer ahc`. Ignored with `--legacy`.
    #[arg(long)]
    as_norm: bool,
    /// Imposter cohort for --as-norm: (N, 256) '<f4' .npy of speaker
    /// embeddings. Omitted = model-registry cohort, with the
    /// POLYVOICE_ASNORM_COHORT env override.
    #[arg(long)]
    cohort: Option<PathBuf>,
    /// Per-domain scoring profile: voxconverse | ami | callhome. Replaces the
    /// default AHC threshold with the profile's calibrated value (an explicit
    /// --threshold wins) and sets the AS-norm cohort size. Requires
    /// `--clusterer ahc`. Ignored with `--legacy`.
    #[arg(long)]
    domain_profile: Option<String>,
    /// v2 dense embedding window in seconds (e.g. `1.5`): split segments into
    /// overlapping sub-windows for more embeddings per speaker — lower confusion
    /// on clean audio at the cost of more embedder calls. Omit for one
    /// embedding/segment. Ignored with `--legacy`.
    #[arg(long)]
    embed_window: Option<f32>,
    /// ONNX execution provider: `auto` (CoreML on Apple Silicon, XNNPACK on
    /// aarch64 Linux, else CPU), `cpu`, `coreml`, `nnapi`, `cuda`, `xnnpack`.
    /// Providers not compiled into this build log a warning and run on CPU.
    /// Applies to the default v2 path; legacy keeps its built-in per-session
    /// defaults.
    #[arg(long, default_value = "auto")]
    execution_provider: String,
    /// Also emit a single-speaker (exclusive) timeline. In JSON this is the
    /// additive `exclusive_turns` field beside the overlap-aware `turns`. For
    /// RTTM/SRT/VTT/TXT the exclusive timeline is written instead (ASR-
    /// reconciliation surface — concurrent speakers are collapsed per frame).
    #[arg(long)]
    exclusive: bool,
    /// Streaming-aligned window geometry preset: `realtime` | `balanced` |
    /// `accurate`. Sets embedding window/hop (and max speakers ceiling) to match
    /// `polyvoice::streaming::LatencyPreset`. Default leaves config unchanged
    /// (balanced geometry is already the DiarizationConfig default).
    #[arg(long, value_name = "PRESET")]
    latency_preset: Option<String>,
}

#[derive(Subcommand, Debug)]
// One instance for the process lifetime, parsed once: boxing the big
// DiarizeArgs variant would buy nothing but indirection.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Run diarization on an audio file (same as the bare `polyvoice <file>` form).
    Diarize(DiarizeArgs),
    /// Download Mobile/Balanced ONNX models.
    DownloadModels {
        #[arg(long, default_value = "balanced")]
        profile: String,
    },
    /// Inspect models registry.
    Models {
        #[command(subcommand)]
        sub: ModelsCommand,
    },
    /// Generate shell completions to stdout.
    Completions {
        /// Shell to generate completions for (bash, zsh, fish, powershell, elvish).
        shell: clap_complete::Shell,
    },
    /// Print the JSON Schema of the diarization result to stdout (agent contract).
    Schema,
}

#[derive(Subcommand, Debug)]
enum ModelsCommand {
    /// Print available profiles + model bundle sizes.
    List,
    /// Print URL/sha256/calibration metadata for a single model.
    Info { name: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum OutputFormat {
    Rttm,
    Json,
    Srt,
    Vtt,
    Txt,
}

fn cmd_diarize(args: DiarizeArgs) -> Result<()> {
    let DiarizeArgs {
        wav,
        profile,
        output,
        format,
        models_cache,
        threshold,
        speakers,
        max_speakers,
        quiet,
        json,
        legacy,
        v2: _v2_deprecated,
        clusterer,
        vbx_plda_dir,
        as_norm,
        cohort,
        domain_profile,
        embed_window,
        execution_provider,
        exclusive,
        latency_preset,
    } = args;
    // v2 is the default; --legacy opts into the pre-0.11 Silero+AHC path.
    // `--v2` remains accepted (hidden) for script compatibility.
    let use_legacy = legacy;

    let wav = wav.ok_or_else(|| {
        anyhow::anyhow!(
            "no input: provide an audio file (e.g. `polyvoice meeting.wav`) or a subcommand (see --help)"
        )
    })?;
    // Machine mode forces JSON to stdout and silences human chatter on stderr.
    let format = if json { OutputFormat::Json } else { format };
    let quiet = quiet || json;

    let profile: Profile = profile.parse()?;
    if !wav.is_file() {
        anyhow::bail!("No such file: {}", wav.display());
    }
    let registry = match models_cache {
        Some(p) => {
            if p.to_str().is_some_and(|s| s.contains("..")) {
                anyhow::bail!("models_cache path contains '..' (path traversal rejected)");
            }
            ModelRegistry::with_cache_dir(&p).context("failed to open models cache")?
        }
        None => ModelRegistry::default().context("failed to resolve default models cache")?,
    };

    if !quiet {
        eprintln!(
            "Loading {profile:?} profile from registry (models auto-download on first run)..."
        );
    }

    // `--speakers N` is an exact target; both it and `--max-speakers` cap the
    // clusterer ceiling (`--speakers` wins when both are set).
    let max_clusters = speakers.or(max_speakers);

    let latency = match latency_preset.as_deref() {
        None => None,
        Some(name) => Some(
            polyvoice::streaming::LatencyPreset::parse_name(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "invalid --latency-preset '{name}' (expected realtime|balanced|accurate)"
                )
            })?,
        ),
    };

    let mut result = if use_legacy {
        if as_norm || cohort.is_some() || domain_profile.is_some() {
            anyhow::bail!(
                "--as-norm/--cohort/--domain-profile apply to the default v2 pipeline only"
            );
        }
        run_legacy_pipeline(
            &wav,
            profile,
            &registry,
            threshold,
            max_clusters,
            latency,
            quiet,
        )?
    } else {
        run_v2_pipeline(
            &wav,
            profile,
            &registry,
            threshold,
            max_clusters,
            &clusterer,
            vbx_plda_dir,
            as_norm,
            cohort,
            domain_profile,
            // Streaming latency presets map onto v2 dense embed windows when the
            // user did not pass an explicit --embed-window.
            embed_window.or_else(|| latency.map(|p| p.params().window_secs)),
            &execution_provider,
            quiet,
        )?
    };

    if exclusive {
        result = result.with_exclusive();
    }

    write_output(&result, &wav, format, exclusive, output)
}

/// Project the result into the requested format and write it to a file or stdout.
/// The bytes are built in a buffer first, so stdout receives ONLY the result.
///
/// When `exclusive` is set and the format is not JSON, the exclusive single-
/// speaker timeline is projected (RTTM/SRT/VTT/TXT). JSON always serializes the
/// full result, so both `turns` and `exclusive_turns` appear side by side.
fn write_output(
    result: &DiarizationResult,
    wav: &Path,
    format: OutputFormat,
    exclusive: bool,
    output: Option<PathBuf>,
) -> Result<()> {
    let file_id = wav
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio")
        .to_string();

    let project_turns = if exclusive && !result.exclusive_turns.is_empty() {
        &result.exclusive_turns
    } else {
        &result.turns
    };

    let mut buf: Vec<u8> = Vec::new();
    match format {
        OutputFormat::Rttm => {
            write_rttm(&mut buf, &file_id, project_turns).context("write RTTM")?
        }
        OutputFormat::Srt => write_srt(&mut buf, project_turns).context("write SRT")?,
        OutputFormat::Vtt => write_vtt(&mut buf, project_turns).context("write VTT")?,
        OutputFormat::Txt => write_txt(&mut buf, project_turns).context("write TXT")?,
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(result).context("serialize JSON")?;
            buf.extend_from_slice(json.as_bytes());
            buf.push(b'\n');
        }
    }

    match output {
        Some(path) => {
            std::fs::write(&path, &buf).with_context(|| format!("write {}", path.display()))?
        }
        None => std::io::stdout()
            .lock()
            .write_all(&buf)
            .context("write to stdout")?,
    }
    Ok(())
}

fn run_legacy_pipeline(
    wav: &Path,
    profile: Profile,
    registry: &ModelRegistry,
    threshold: Option<f32>,
    max_clusters: Option<usize>,
    latency: Option<polyvoice::streaming::LatencyPreset>,
    quiet: bool,
) -> Result<DiarizationResult> {
    let models = registry
        .ensure_for_profile(profile)
        .context("ensure models")?;
    let vad_path = registry.ensure("silero_vad").context("silero_vad model")?;
    let mut stack = cli_common::load_legacy_stack(
        &models.embedder_path,
        profile.embedding_dim(),
        polyvoice::onnx::ExecutionProvider::Cpu,
        &vad_path,
        512,
    )?;

    let mut config = cli_common::legacy_diarization_config(
        threshold.unwrap_or(polyvoice::DEFAULT_AHC_THRESHOLD),
    );
    if let Some(n) = max_clusters {
        config.cluster.max_speakers = n;
    }
    if let Some(preset) = latency {
        // Apply window/hop/cap from the streaming latency preset; keep the
        // CLI --threshold / --max-speakers overrides when the user set them.
        let saved_threshold = config.cluster.threshold;
        let saved_max = max_clusters;
        preset.apply(&mut config);
        config.cluster.threshold = saved_threshold;
        if let Some(n) = saved_max {
            config.cluster.max_speakers = n;
        }
    }
    let pipeline = LegacyPipeline::new(config, VadConfig::default());

    if !quiet {
        eprintln!("Reading {}...", wav.display());
    }
    let (samples, sr_hz) =
        load_audio(wav).with_context(|| format!("load audio {}", wav.display()))?;
    let _sr = SampleRate::new(sr_hz).with_context(|| format!("invalid sample rate {sr_hz} Hz"))?;

    if !quiet {
        eprintln!(
            "Running diarization on {} samples ({} Hz)...",
            samples.len(),
            sr_hz
        );
    }
    let result = pipeline
        .run(&samples, &stack.extractor, &mut stack.vad)
        .context("pipeline.run failed")?;
    if !quiet {
        eprintln!(
            "Done — {} turns, {} speakers",
            result.turns.len(),
            result.num_speakers
        );
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn run_v2_pipeline(
    wav: &Path,
    profile: Profile,
    registry: &ModelRegistry,
    threshold: Option<f32>,
    max_clusters: Option<usize>,
    clusterer: &str,
    vbx_plda_dir: Option<PathBuf>,
    as_norm: bool,
    cohort: Option<PathBuf>,
    domain_profile: Option<String>,
    embed_window: Option<f32>,
    execution_provider: &str,
    quiet: bool,
) -> Result<DiarizationResult> {
    let (clusterer_kind, as_norm_config, domain) = cli_common::resolve_clusterer_flags(
        clusterer,
        threshold,
        as_norm,
        cohort,
        domain_profile.as_deref(),
    )?;
    let ep = cli_common::parse_execution_provider(execution_provider)?;
    let mut config = PipelineConfig {
        profile,
        clusterer: clusterer_kind,
        vbx_plda_dir,
        as_norm: as_norm_config,
        domain,
        embed_window_secs: embed_window,
        execution_provider: ep,
        ..PipelineConfig::default()
    };
    if let Some(n) = max_clusters {
        config.max_speakers = cli_common::max_speakers_u8(n)?;
    }
    let pipeline = cli_common::build_v2_pipeline(config, registry.clone())?;

    if !quiet {
        eprintln!("Reading {}...", wav.display());
    }
    let (samples, sr_hz) =
        load_audio(wav).with_context(|| format!("load audio {}", wav.display()))?;
    let sr = SampleRate::new(sr_hz).with_context(|| format!("invalid sample rate {sr_hz} Hz"))?;

    if !quiet {
        eprintln!(
            "Running diarization on {} samples ({} Hz)...",
            samples.len(),
            sr_hz
        );
    }
    let result = pipeline
        .run(&samples, sr)
        .context("pipeline v2 run failed")?;
    if !quiet {
        eprintln!(
            "Done — {} turns, {} speakers",
            result.turns.len(),
            result.num_speakers
        );
    }
    Ok(result)
}

fn cmd_completions(shell: clap_complete::Shell) -> Result<()> {
    let mut cmd = Cli::command();
    clap_complete::generate(shell, &mut cmd, "polyvoice", &mut std::io::stdout());
    Ok(())
}

/// Print the canonical diarization-result JSON Schema to stdout — the
/// machine-readable contract agents can read to know the `--json` shape.
fn cmd_schema() -> Result<()> {
    const SCHEMA: &str = include_str!("../../schema/diarization-result-v1.json");
    print!("{SCHEMA}");
    Ok(())
}

fn cmd_download_models(profile: String) -> Result<()> {
    let registry = ModelRegistry::default()?;
    match profile.as_str() {
        "all" => {
            let _ = registry.ensure_for_profile(Profile::Mobile)?;
            let _ = registry.ensure_for_profile(Profile::Balanced)?;
        }
        other => {
            let p: Profile = other.parse()?;
            let _ = registry.ensure_for_profile(p)?;
        }
    }
    eprintln!("Models cached at {}", registry.cache_dir().display());
    Ok(())
}

fn cmd_models_list() -> Result<()> {
    let registry = ModelRegistry::default()?;
    let manifest = registry.manifest();
    println!("Profiles:");
    for (name, prof) in &manifest.profiles {
        let seg = manifest
            .model(&prof.segmenter)
            .map(|m| {
                format!(
                    "{} ({:.1} MB)",
                    m.filename,
                    m.size.unwrap_or(0) as f64 / 1_048_576.0
                )
            })
            .unwrap_or_else(|| "(missing)".to_string());
        let emb = manifest
            .model(&prof.embedder)
            .map(|m| {
                format!(
                    "{} ({:.1} MB)",
                    m.filename,
                    m.size.unwrap_or(0) as f64 / 1_048_576.0
                )
            })
            .unwrap_or_else(|| "(missing)".to_string());
        println!("  {name}: segmenter={seg}, embedder={emb}");
    }
    println!("\nModels:");
    for (id, entry) in &manifest.models {
        let size_mb = entry.size.unwrap_or(0) as f64 / 1_048_576.0;
        println!(
            "  {id}: {} ({size_mb:.1} MB) sha256={}",
            entry.filename, entry.sha256
        );
    }
    Ok(())
}

fn cmd_models_info(name: String) -> Result<()> {
    let registry = ModelRegistry::default()?;
    let manifest = registry.manifest();
    // Accept direct model ids and stage-scoped aliases (e.g. embedder/latest).
    let resolved = manifest.model(&name).map(|_| name.as_str()).or_else(|| {
        for stage in ["segmenter", "embedder", "vad"] {
            if let Some(id) = manifest.resolve_model_ref(stage, &name) {
                return Some(id);
            }
        }
        None
    });
    let Some(model_id) = resolved else {
        anyhow::bail!("model '{name}' not found in manifest");
    };
    let entry = manifest.model(model_id).expect("resolved id must exist");
    if model_id != name {
        println!("{name} -> {model_id}:");
    } else {
        println!("{name}:");
    }
    println!("  filename: {}", entry.filename);
    println!("  url: {}", entry.url);
    println!("  sha256: {}", entry.sha256);
    println!("  size: {} bytes", entry.size.unwrap_or(0));
    if let Some(cal) = &entry.calibration {
        println!("  calibration: {cal}");
    }
    if let Some(v) = &entry.version {
        println!("  version: {v}");
    }
    if let Some(a) = &entry.adapter_type {
        println!("  adapter_type: {a}");
    }
    if let Some(l) = &entry.license {
        println!("  license: {l}");
    }
    if let Some(u) = &entry.license_url {
        println!("  license_url: {u}");
    }
    if let Some(p) = &entry.provenance {
        println!("  provenance: {p}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Diarize(d)) => cmd_diarize(d),
        Some(Command::DownloadModels { profile }) => cmd_download_models(profile),
        Some(Command::Models { sub }) => match sub {
            ModelsCommand::List => cmd_models_list(),
            ModelsCommand::Info { name } => cmd_models_info(name),
        },
        Some(Command::Completions { shell }) => cmd_completions(shell),
        Some(Command::Schema) => cmd_schema(),
        None => cmd_diarize(cli.diarize),
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn bare_wav_is_implicit_diarize() {
        let cli = Cli::try_parse_from(["polyvoice", "meeting.wav", "--format", "srt"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.diarize.wav.as_deref(), Some(Path::new("meeting.wav")));
        assert_eq!(cli.diarize.format, OutputFormat::Srt);
    }

    #[test]
    fn subcommands_are_not_shadowed_by_default_diarize() {
        assert!(matches!(
            Cli::try_parse_from(["polyvoice", "models", "list"])
                .unwrap()
                .command,
            Some(Command::Models { .. })
        ));
        assert!(matches!(
            Cli::try_parse_from(["polyvoice", "download-models"])
                .unwrap()
                .command,
            Some(Command::DownloadModels { .. })
        ));
        assert!(matches!(
            Cli::try_parse_from(["polyvoice", "completions", "bash"])
                .unwrap()
                .command,
            Some(Command::Completions { .. })
        ));
        assert!(matches!(
            Cli::try_parse_from(["polyvoice", "diarize", "x.wav"])
                .unwrap()
                .command,
            Some(Command::Diarize(_))
        ));
    }

    proptest! {
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

        #[test]
        fn cli_diarize_parses_all_formats(
            profile in "(mobile|balanced|fast)",
            format in "(rttm|json|srt|vtt|txt)",
            threshold in 0.0f32..2.0f32,
            v2 in prop::bool::ANY,
        ) {
            let mut args = vec![
                "polyvoice".to_string(),
                "diarize".to_string(),
                "/tmp/test.wav".to_string(),
                "--profile".to_string(), profile,
                "--format".to_string(), format,
                "--threshold".to_string(), threshold.to_string(),
            ];
            if v2 {
                args.push("--v2".to_string());
            }
            prop_assert!(Cli::try_parse_from(&args).is_ok());
        }

        #[test]
        fn cli_models_info_parses(name in "[a-zA-Z0-9_][a-zA-Z0-9_-]{0,29}") {
            let args = vec![
                "polyvoice".to_string(),
                "models".to_string(),
                "info".to_string(),
                name,
            ];
            prop_assert!(Cli::try_parse_from(&args).is_ok());
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod unit_tests {
    use super::*;
    use polyvoice::types::{SpeakerId, SpeakerTurn, TimeRange};

    /// DiarizeArgs with every flag at its CLI default; tests override the one
    /// knob they exercise.
    fn base_args() -> DiarizeArgs {
        DiarizeArgs {
            wav: None,
            profile: "balanced".to_string(),
            output: None,
            format: OutputFormat::Rttm,
            models_cache: None,
            threshold: Some(polyvoice::DEFAULT_AHC_THRESHOLD),
            speakers: None,
            max_speakers: None,
            quiet: true,
            json: false,
            legacy: false,
            v2: false,
            clusterer: "vbx".to_string(),
            vbx_plda_dir: None,
            as_norm: false,
            cohort: None,
            domain_profile: None,
            embed_window: None,
            execution_provider: "auto".to_string(),
            exclusive: false,
            latency_preset: None,
        }
    }

    /// Two overlapping turns so the exclusive projection differs from the
    /// overlap-aware timeline.
    fn sample_result() -> DiarizationResult {
        let turns = vec![
            SpeakerTurn {
                time: TimeRange {
                    start: 0.0,
                    end: 1.5,
                },
                speaker: SpeakerId(0),
                text: None,
                stable: true,
            },
            SpeakerTurn {
                time: TimeRange {
                    start: 1.0,
                    end: 2.5,
                },
                speaker: SpeakerId(1),
                text: None,
                stable: true,
            },
        ];
        DiarizationResult::new(Vec::new(), turns, 2).with_exclusive()
    }

    fn temp_wav_file(dir: &tempfile::TempDir) -> PathBuf {
        let p = dir.path().join("in.wav");
        std::fs::write(&p, b"not really audio").unwrap();
        p
    }

    #[test]
    fn write_output_writes_every_format_to_file() {
        let result = sample_result();
        let dir = tempfile::tempdir().unwrap();
        let wav = Path::new("meeting.wav");
        for (format, needle) in [
            (OutputFormat::Rttm, "SPEAKER meeting 1"),
            (OutputFormat::Json, "\"turns\""),
            (OutputFormat::Srt, "-->"),
            (OutputFormat::Vtt, "WEBVTT"),
            (OutputFormat::Txt, "SPEAKER_00"),
        ] {
            let out = dir.path().join(format!("out.{format:?}"));
            write_output(&result, wav, format, false, Some(out.clone())).unwrap();
            let content = std::fs::read_to_string(&out).unwrap();
            assert!(
                content.contains(needle),
                "{format:?} missing '{needle}':\n{content}"
            );
        }
    }

    #[test]
    fn write_output_exclusive_projects_exclusive_timeline() {
        let result = sample_result();
        assert!(!result.exclusive_turns.is_empty());
        let dir = tempfile::tempdir().unwrap();
        let wav = Path::new("meeting.wav");
        let shared = dir.path().join("shared.rttm");
        let exclusive = dir.path().join("exclusive.rttm");
        write_output(
            &result,
            wav,
            OutputFormat::Rttm,
            false,
            Some(shared.clone()),
        )
        .unwrap();
        write_output(
            &result,
            wav,
            OutputFormat::Rttm,
            true,
            Some(exclusive.clone()),
        )
        .unwrap();
        // Overlapping turns collapse in the exclusive projection, so the two
        // timelines must serialize differently.
        assert_ne!(
            std::fs::read_to_string(shared).unwrap(),
            std::fs::read_to_string(exclusive).unwrap()
        );
    }

    #[test]
    fn write_output_json_keeps_both_timelines_when_exclusive() {
        let result = sample_result();
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.json");
        write_output(
            &result,
            Path::new("a.wav"),
            OutputFormat::Json,
            true,
            Some(out.clone()),
        )
        .unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert!(json["turns"].is_array());
        assert!(json["exclusive_turns"].is_array());
    }

    #[test]
    fn write_output_falls_back_to_audio_file_id() {
        let result = sample_result();
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.rttm");
        // A path without a file stem exercises the "audio" fallback id.
        write_output(
            &result,
            Path::new("/"),
            OutputFormat::Rttm,
            false,
            Some(out.clone()),
        )
        .unwrap();
        assert!(
            std::fs::read_to_string(&out)
                .unwrap()
                .contains("SPEAKER audio 1")
        );
    }

    #[test]
    fn write_output_to_stdout_succeeds() {
        // The None branch streams to stdout (captured by the test harness).
        write_output(
            &sample_result(),
            Path::new("a.wav"),
            OutputFormat::Txt,
            false,
            None,
        )
        .unwrap();
    }

    #[test]
    fn write_output_report_write_failure() {
        let result = sample_result();
        let err = write_output(
            &result,
            Path::new("a.wav"),
            OutputFormat::Rttm,
            false,
            Some(PathBuf::from("/nonexistent/dir/out.rttm")),
        )
        .err()
        .unwrap();
        assert!(format!("{err:#}").contains("write"));
    }

    #[test]
    fn cmd_diarize_without_input_errors() {
        let err = cmd_diarize(base_args()).err().unwrap();
        assert!(format!("{err:#}").contains("no input"));
    }

    #[test]
    fn cmd_diarize_missing_file_errors() {
        let mut args = base_args();
        args.wav = Some(PathBuf::from("/nonexistent/dir/file.wav"));
        let err = cmd_diarize(args).err().unwrap();
        assert!(format!("{err:#}").contains("No such file"));
    }

    #[test]
    fn cmd_diarize_invalid_profile_errors() {
        let mut args = base_args();
        args.wav = Some(PathBuf::from("/nonexistent/dir/file.wav"));
        args.profile = "garbage".to_string();
        assert!(cmd_diarize(args).is_err());
    }

    #[test]
    fn cmd_diarize_rejects_models_cache_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let mut args = base_args();
        args.wav = Some(temp_wav_file(&dir));
        args.models_cache = Some(PathBuf::from("../escape"));
        let err = cmd_diarize(args).err().unwrap();
        assert!(format!("{err:#}").contains("path traversal"));
    }

    #[test]
    fn cmd_diarize_rejects_bad_latency_preset() {
        let dir = tempfile::tempdir().unwrap();
        let mut args = base_args();
        args.wav = Some(temp_wav_file(&dir));
        args.models_cache = Some(dir.path().join("cache"));
        args.latency_preset = Some("warp-speed".to_string());
        let err = cmd_diarize(args).err().unwrap();
        assert!(format!("{err:#}").contains("invalid --latency-preset"));
    }

    #[test]
    fn cmd_diarize_accepts_known_latency_preset_then_fails_later() {
        let dir = tempfile::tempdir().unwrap();
        let mut args = base_args();
        args.wav = Some(temp_wav_file(&dir));
        args.models_cache = Some(dir.path().join("cache"));
        args.latency_preset = Some("realtime".to_string());
        // The preset parses; the run then stops at the unknown clusterer,
        // before any model resolution.
        args.clusterer = "kmeans".to_string();
        let err = cmd_diarize(args).err().unwrap();
        assert!(format!("{err:#}").contains("unknown --clusterer"));
    }

    #[test]
    fn cmd_diarize_rejects_unknown_clusterer() {
        let dir = tempfile::tempdir().unwrap();
        let mut args = base_args();
        args.wav = Some(temp_wav_file(&dir));
        args.models_cache = Some(dir.path().join("cache"));
        args.clusterer = "kmeans".to_string();
        let err = cmd_diarize(args).err().unwrap();
        assert!(format!("{err:#}").contains("unknown --clusterer"));
    }

    #[test]
    fn cmd_diarize_rejects_unknown_execution_provider() {
        let dir = tempfile::tempdir().unwrap();
        let mut args = base_args();
        args.wav = Some(temp_wav_file(&dir));
        args.models_cache = Some(dir.path().join("cache"));
        args.clusterer = "ahc".to_string();
        args.execution_provider = "tpu".to_string();
        let err = cmd_diarize(args).err().unwrap();
        assert!(format!("{err:#}").contains("unknown --execution-provider"));
    }

    #[test]
    fn cmd_diarize_rejects_oversized_max_speakers() {
        let dir = tempfile::tempdir().unwrap();
        let mut args = base_args();
        args.wav = Some(temp_wav_file(&dir));
        args.models_cache = Some(dir.path().join("cache"));
        args.clusterer = "ahc".to_string();
        args.max_speakers = Some(300);
        let err = cmd_diarize(args).err().unwrap();
        assert!(format!("{err:#}").contains("max_speakers must be in 1..=255"));
    }

    #[test]
    fn cmd_models_list_runs() {
        cmd_models_list().unwrap();
    }

    #[test]
    fn cmd_models_info_direct_id_runs() {
        cmd_models_info("silero_vad".to_string()).unwrap();
    }

    #[test]
    fn cmd_models_info_resolves_stage_alias() {
        // "latest" is not a model id; the stage-alias fallback resolves it.
        cmd_models_info("latest".to_string()).unwrap();
    }

    #[test]
    fn cmd_models_info_prints_calibration_when_present() {
        cmd_models_info("powerset_int8".to_string()).unwrap();
    }

    #[test]
    fn cmd_models_info_unknown_model_errors() {
        let err = cmd_models_info("no_such_model_zzz".to_string())
            .err()
            .unwrap();
        assert!(format!("{err:#}").contains("not found in manifest"));
    }

    #[test]
    fn cmd_download_models_rejects_unknown_profile() {
        assert!(cmd_download_models("garbage".to_string()).is_err());
    }

    #[test]
    fn cmd_download_models_custom_profile_is_unresolvable() {
        // "custom" parses as a profile but the registry cannot resolve it —
        // this fails before any network access.
        let err = cmd_download_models("custom".to_string()).err().unwrap();
        assert!(format!("{err:#}").contains("custom"));
    }

    #[test]
    fn cmd_completions_generates_for_every_shell() {
        for shell in [
            clap_complete::Shell::Bash,
            clap_complete::Shell::Zsh,
            clap_complete::Shell::Fish,
            clap_complete::Shell::PowerShell,
            clap_complete::Shell::Elvish,
        ] {
            cmd_completions(shell).unwrap();
        }
    }

    #[test]
    fn cmd_schema_prints_committed_schema() {
        cmd_schema().unwrap();
    }
}
