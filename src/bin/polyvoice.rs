//! polyvoice — speaker diarization CLI (legacy v0.5 pipeline).

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::Pipeline;
use polyvoice::rttm::write_rttm;
use polyvoice::types::{DiarizationConfig, Profile, SampleRate};
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
use polyvoice::{FbankOnnxExtractor, SileroVad};
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
        #[arg(long, default_value = "0.45")]
        threshold: f32,
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

fn cmd_diarize(
    wav: PathBuf,
    profile: String,
    output: Option<PathBuf>,
    format: OutputFormat,
    models_cache: Option<PathBuf>,
    threshold: f32,
    quiet: bool,
) -> Result<()> {
    let profile = parse_profile(&profile)?;
    let registry = match models_cache {
        Some(p) => ModelRegistry::with_cache_dir(&p).context("failed to open models cache")?,
        None => ModelRegistry::default().context("failed to resolve default models cache")?,
    };

    if !quiet {
        eprintln!("Loading {profile:?} profile from registry...");
    }

    let models = registry
        .ensure_for_profile(profile)
        .context("ensure models")?;
    let embedding_dim = profile.embedding_dim();
    let extractor = FbankOnnxExtractor::new(&models.embedder_path, embedding_dim, 1)
        .context("load embedder")?;
    let mut vad = SileroVad::new(&models.segmenter_path, 512).context("load vad")?;

    let config = DiarizationConfig {
        threshold,
        ..DiarizationConfig::default()
    };
    let vad_config = VadConfig::default();
    let pipeline = Pipeline::new(config, vad_config);

    if !quiet {
        eprintln!("Reading {}...", wav.display());
    }
    let (samples, sr_hz) = read_wav(&wav).with_context(|| format!("read WAV {}", wav.display()))?;
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
        Command::Diarize {
            wav,
            profile,
            output,
            format,
            models_cache,
            threshold,
            quiet,
        } => cmd_diarize(wav, profile, output, format, models_cache, threshold, quiet),
        Command::DownloadModels { profile } => cmd_download_models(profile),
        Command::Models { sub } => match sub {
            ModelsCommand::List => cmd_models_list(),
            ModelsCommand::Info { name } => cmd_models_info(name),
        },
    }
}
