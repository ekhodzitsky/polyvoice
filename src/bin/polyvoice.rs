#![allow(deprecated)] // legacy embedding API; see polyvoice::embedder
//! polyvoice — speaker diarization CLI.
//!
//! `polyvoice meeting.wav` diarizes a file (implicit `diarize`); subcommands
//! `models` / `download-models` / `completions` are still available. Default
//! pipeline: legacy v0.5 (Silero VAD + sliding-window embeddings + AHC). Use
//! `--v2` to opt into pipeline v2 (Powerset segmentation + overlap masking +
//! resegmentation); v2 is not yet validated as default on long-form audio — see
//! PRODUCTION-READINESS.md.
//!
//! STDOUT discipline: only the diarization result goes to stdout (so `--format
//! json` / `--json` and downstream pipes stay clean); all progress and info go to
//! stderr.

use anyhow::{Context, Result};
use clap::{Args, CommandFactory, Parser, Subcommand};
use polyvoice::format::{write_srt, write_txt, write_vtt};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::Pipeline as LegacyPipeline;
use polyvoice::pipeline_v2::{ClustererKind, Pipeline as V2Pipeline, PipelineConfig};
use polyvoice::rttm::write_rttm;
use polyvoice::types::{ClusterConfig, DiarizationConfig, DiarizationResult, Profile, SampleRate};
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
use polyvoice::{FbankOnnxExtractor, SileroVad};
use std::io::Write;
use std::path::{Path, PathBuf};

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
    /// WAV file to diarize. `polyvoice meeting.wav` diarizes it directly.
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
    #[arg(long, default_value = "0.45")]
    threshold: f32,
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
    /// Use pipeline v2 (experimental; not recommended for long-form audio).
    #[arg(long)]
    v2: bool,
    /// v2 clusterer: `ahc` (default cosine AHC) or `vbx` (PLDA + VB-HMM — best for
    /// overlap-heavy / meeting audio; needs the `vbx` build feature and a PLDA dir
    /// via `--vbx-plda-dir` or `POLYVOICE_VBX_PLDA_DIR`). Only affects `--v2`.
    #[arg(long, default_value = "ahc")]
    clusterer: String,
    /// Directory with the precomputed VBx PLDA params (overrides
    /// `POLYVOICE_VBX_PLDA_DIR`). Only used with `--v2 --clusterer vbx`.
    #[arg(long)]
    vbx_plda_dir: Option<PathBuf>,
    /// v2 dense embedding window in seconds (e.g. `1.5`): split segments into
    /// overlapping sub-windows for more embeddings per speaker — lower confusion
    /// on clean audio at the cost of more embedder calls. Omit for one
    /// embedding/segment. Only affects `--v2`.
    #[arg(long)]
    embed_window: Option<f32>,
    /// ONNX execution provider: `auto` (CoreML on Apple Silicon, XNNPACK on
    /// aarch64 Linux, else CPU), `cpu`, `coreml`, `nnapi`, `cuda`, `xnnpack`.
    /// Providers not compiled into this build log a warning and run on CPU.
    /// Only affects `--v2`; the legacy default pipeline keeps its built-in
    /// per-session defaults.
    #[arg(long, default_value = "auto")]
    execution_provider: String,
    /// Also emit a single-speaker (exclusive) timeline. In JSON this is the
    /// additive `exclusive_turns` field beside the overlap-aware `turns`. For
    /// RTTM/SRT/VTT/TXT the exclusive timeline is written instead (ASR-
    /// reconciliation surface — concurrent speakers are collapsed per frame).
    #[arg(long)]
    exclusive: bool,
}

#[derive(Subcommand, Debug)]
// One instance for the process lifetime, parsed once: boxing the big
// DiarizeArgs variant would buy nothing but indirection.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Run diarization on a WAV file (same as the bare `polyvoice <wav>` form).
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

fn parse_profile(name: &str) -> Result<Profile> {
    match name {
        "mobile" => Ok(Profile::Mobile),
        "balanced" => Ok(Profile::Balanced),
        other => anyhow::bail!("invalid profile: {other} (expected mobile|balanced)"),
    }
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
        v2,
        clusterer,
        vbx_plda_dir,
        embed_window,
        execution_provider,
        exclusive,
    } = args;

    let wav = wav.ok_or_else(|| {
        anyhow::anyhow!(
            "no input: provide a WAV file (e.g. `polyvoice meeting.wav`) or a subcommand (see --help)"
        )
    })?;
    // Machine mode forces JSON to stdout and silences human chatter on stderr.
    let format = if json { OutputFormat::Json } else { format };
    let quiet = quiet || json;

    let profile = parse_profile(&profile)?;
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

    let mut result = if v2 {
        run_v2_pipeline(
            &wav,
            profile,
            &registry,
            threshold,
            &clusterer,
            vbx_plda_dir,
            embed_window,
            &execution_provider,
            quiet,
        )?
    } else {
        run_legacy_pipeline(&wav, profile, &registry, threshold, max_clusters, quiet)?
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
    threshold: f32,
    max_clusters: Option<usize>,
    quiet: bool,
) -> Result<DiarizationResult> {
    let models = registry
        .ensure_for_profile(profile)
        .context("ensure models")?;
    let embedding_dim = profile.embedding_dim();
    let extractor = FbankOnnxExtractor::new(
        &models.embedder_path,
        embedding_dim,
        1,
        polyvoice::onnx::ExecutionProvider::Cpu,
    )
    .context("load embedder")?;
    let vad_path = registry.ensure("silero_vad").context("silero_vad model")?;
    let mut vad = SileroVad::new(&vad_path, 512).context("load vad")?;

    let mut cluster = ClusterConfig {
        threshold,
        ..Default::default()
    };
    if let Some(n) = max_clusters {
        cluster.max_speakers = n;
    }
    let config = DiarizationConfig {
        cluster,
        ..DiarizationConfig::default()
    };
    let vad_config = VadConfig::default();
    let pipeline = LegacyPipeline::new(config, vad_config);

    if !quiet {
        eprintln!("Reading {}...", wav.display());
    }
    let (samples, sr_hz) = read_wav(wav).with_context(|| format!("read WAV {}", wav.display()))?;
    let _sr = SampleRate::new(sr_hz).with_context(|| format!("invalid sample rate {sr_hz} Hz"))?;

    if !quiet {
        eprintln!(
            "Running diarization on {} samples ({} Hz)...",
            samples.len(),
            sr_hz
        );
    }
    let result = pipeline
        .run(&samples, &extractor, &mut vad)
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
    threshold: f32,
    clusterer: &str,
    vbx_plda_dir: Option<PathBuf>,
    embed_window: Option<f32>,
    execution_provider: &str,
    quiet: bool,
) -> Result<DiarizationResult> {
    let clusterer_kind = match clusterer {
        "ahc" => ClustererKind::Ahc { threshold },
        "vbx" => ClustererKind::Vbx,
        other => anyhow::bail!("unknown --clusterer '{other}' (expected 'ahc' or 'vbx')"),
    };
    let ep = match execution_provider {
        "auto" => polyvoice::onnx::ExecutionProvider::auto(),
        "cpu" => polyvoice::onnx::ExecutionProvider::Cpu,
        "coreml" => polyvoice::onnx::ExecutionProvider::CoreMl,
        "nnapi" => polyvoice::onnx::ExecutionProvider::Nnapi,
        "cuda" => polyvoice::onnx::ExecutionProvider::Cuda,
        "xnnpack" => polyvoice::onnx::ExecutionProvider::XnnPack,
        other => anyhow::bail!(
            "unknown --execution-provider '{other}' (expected auto|cpu|coreml|nnapi|cuda|xnnpack)"
        ),
    };
    let config = PipelineConfig {
        profile,
        clusterer: clusterer_kind,
        vbx_plda_dir,
        embed_window_secs: embed_window,
        execution_provider: ep,
        ..PipelineConfig::default()
    };
    let pipeline = V2Pipeline::builder()
        .config(config)
        .with_models_from(registry.clone())
        .build()
        .context("build pipeline v2")?;

    if !quiet {
        eprintln!("Reading {}...", wav.display());
    }
    let (samples, sr_hz) = read_wav(wav).with_context(|| format!("read WAV {}", wav.display()))?;
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
            let p = parse_profile(other)?;
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
    if let Some(entry) = manifest.model(&name) {
        println!("{name}:");
        println!("  filename: {}", entry.filename);
        println!("  url: {}", entry.url);
        println!("  sha256: {}", entry.sha256);
        println!("  size: {} bytes", entry.size.unwrap_or(0));
        if let Some(cal) = &entry.calibration {
            println!("  calibration: {cal}");
        }
    } else {
        anyhow::bail!("model '{name}' not found in manifest");
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
        fn parse_profile_accepts_only_valid(s in "[a-zA-Z0-9_-]{1,20}") {
            let result = parse_profile(&s);
            if s == "mobile" || s == "balanced" {
                prop_assert!(result.is_ok());
            } else {
                prop_assert!(result.is_err());
            }
        }

        #[test]
        fn cli_diarize_parses_all_formats(
            profile in "(mobile|balanced)",
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
