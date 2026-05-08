//! polyvoice — speaker diarization CLI.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::{ExecutionProvider, Pipeline, PipelineConfig};
use polyvoice::rttm::write_rttm;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "polyvoice", version, about = "Speaker diarization toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run diarization on a WAV file.
    Diarize {
        wav: PathBuf,
        #[arg(long, default_value = "balanced")]
        profile: String,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "rttm")]
        format: OutputFormat,
        #[arg(long)]
        models_cache: Option<PathBuf>,
        #[arg(long, default_value = "auto")]
        execution_provider: String,
        #[arg(long, default_value = "true")]
        resegment_overlap: bool,
        #[arg(long, default_value = "20")]
        max_speakers: u8,
        #[arg(long)]
        quiet: bool,
    },
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
}

fn parse_profile(name: &str) -> Result<Profile> {
    match name {
        "mobile" => Ok(Profile::Mobile),
        "balanced" => Ok(Profile::Balanced),
        other => anyhow::bail!("invalid profile: {other} (expected mobile|balanced)"),
    }
}

fn parse_execution_provider(name: &str) -> Result<ExecutionProvider> {
    match name {
        "auto" => Ok(ExecutionProvider::auto()),
        "cpu" => Ok(ExecutionProvider::Cpu),
        "coreml" => Ok(ExecutionProvider::CoreMl),
        "nnapi" => Ok(ExecutionProvider::Nnapi),
        "cuda" => Ok(ExecutionProvider::Cuda),
        "xnnpack" => Ok(ExecutionProvider::XnnPack),
        other => anyhow::bail!(
            "invalid --execution-provider: {other} (expected auto|cpu|coreml|nnapi|cuda|xnnpack)"
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_diarize(
    wav: PathBuf,
    profile: String,
    output: Option<PathBuf>,
    format: OutputFormat,
    models_cache: Option<PathBuf>,
    execution_provider: String,
    resegment_overlap: bool,
    max_speakers: u8,
    quiet: bool,
) -> Result<()> {
    let profile = parse_profile(&profile)?;
    let ep = parse_execution_provider(&execution_provider)?;
    let registry = match models_cache {
        Some(p) => ModelRegistry::with_cache_dir(&p).context("failed to open models cache")?,
        None => ModelRegistry::default().context("failed to resolve default models cache")?,
    };

    if !quiet {
        eprintln!("Loading {profile:?} profile from registry...");
    }

    let cfg = PipelineConfig {
        profile,
        execution_provider: ep,
        resegment_overlap,
        max_speakers,
        ..PipelineConfig::default()
    };

    let pipeline = Pipeline::builder()
        .config(cfg)
        .with_models_from(registry)
        .build()
        .context("failed to build pipeline")?;

    if !quiet {
        eprintln!("Reading {}...", wav.display());
    }
    let (samples, sr_hz) = read_wav(&wav).with_context(|| format!("read WAV {}", wav.display()))?;
    let sr = SampleRate::new(sr_hz)
        .with_context(|| format!("invalid sample rate {sr_hz} Hz"))?;

    if !quiet {
        eprintln!("Running diarization on {} samples ({} Hz)...", samples.len(), sr_hz);
    }
    let result = pipeline.run(&samples, sr).context("pipeline.run failed")?;
    if !quiet {
        eprintln!("Done — {} turns, {} speakers", result.turns.len(), result.num_speakers);
    }

    match format {
        OutputFormat::Rttm => {
            let file_id = wav
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("audio")
                .to_string();
            match output {
                Some(path) => {
                    let mut f = std::fs::File::create(&path)
                        .with_context(|| format!("create {}", path.display()))?;
                    write_rttm(&mut f, &file_id, &result.turns).context("rttm write")?;
                }
                None => {
                    let mut stdout = std::io::stdout().lock();
                    write_rttm(&mut stdout, &file_id, &result.turns).context("rttm write")?;
                }
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&result).context("serialize JSON")?;
            match output {
                Some(path) => std::fs::write(&path, json)
                    .with_context(|| format!("write JSON to {}", path.display()))?,
                None => println!("{json}"),
            }
        }
    }

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
        let seg = manifest.models.get(&prof.segmenter);
        let emb = manifest.models.get(&prof.embedder);
        let total: u64 = seg.and_then(|m| m.size).unwrap_or(0)
            + emb.and_then(|m| m.size).unwrap_or(0);
        println!(
            "  {name}: segmenter={} embedder={} total={} bytes",
            prof.segmenter, prof.embedder, total
        );
    }
    Ok(())
}

fn cmd_models_info(name: String) -> Result<()> {
    let registry = ModelRegistry::default()?;
    let manifest = registry.manifest();
    match manifest.models.get(&name) {
        Some(m) => {
            println!("name: {name}");
            println!("url: {}", m.url);
            println!("sha256: {}", m.sha256);
            if let Some(size) = m.size {
                println!("size: {size}");
            }
            if let Some(calib) = &m.calibration {
                println!("calibration: {calib}");
            }
            Ok(())
        }
        None => anyhow::bail!("unknown model: {name}"),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Diarize {
            wav,
            profile,
            output,
            format,
            models_cache,
            execution_provider,
            resegment_overlap,
            max_speakers,
            quiet,
        } => cmd_diarize(
            wav,
            profile,
            output,
            format,
            models_cache,
            execution_provider,
            resegment_overlap,
            max_speakers,
            quiet,
        ),
        Command::DownloadModels { profile } => cmd_download_models(profile),
        Command::Models { sub } => match sub {
            ModelsCommand::List => cmd_models_list(),
            ModelsCommand::Info { name } => cmd_models_info(name),
        },
    }
}
