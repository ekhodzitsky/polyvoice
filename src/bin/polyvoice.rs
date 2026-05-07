use clap::{Parser, Subcommand, ValueEnum};
use polyvoice::{
    ClusteringBackend, DiarizationConfig, FbankOnnxExtractor, Pipeline, SampleRate, SileroVad,
    VadConfig,
};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "polyvoice",
    version,
    about = "Speaker diarization — who spoke when"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run speaker diarization on a WAV file
    Diarize {
        /// Path to 16 kHz mono WAV file
        file: PathBuf,

        /// Model directory (default: ~/.cache/polyvoice/models)
        #[arg(long, env = "POLYVOICE_MODEL_DIR")]
        model_dir: Option<PathBuf>,

        /// Maximum number of speakers
        #[arg(long, default_value = "64")]
        max_speakers: usize,

        /// Cosine similarity threshold for clustering
        #[arg(long, default_value = "0.45")]
        threshold: f32,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,

        /// Clustering backend (ahc | kmeans)
        #[arg(long, default_value = "ahc")]
        backend: String,
    },
    /// Download ONNX models for a profile (or all profiles)
    DownloadModels {
        /// Target directory (default: ~/.cache/polyvoice/models)
        #[arg(long)]
        dir: Option<PathBuf>,

        /// Profile to fetch models for.
        ///
        /// In M0 both `mobile` and `balanced` map to the legacy v0.5 model pair
        /// (silero_vad + wespeaker_resnet34). Future milestones (M2/M5) will
        /// diverge them. Use `all` to fetch every distinct model in the manifest.
        #[arg(long, default_value = "all", value_parser = ["mobile", "balanced", "all"])]
        profile: String,
    },
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Rttm,
}

fn default_model_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("polyvoice")
        .join("models")
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Diarize {
            file,
            model_dir,
            max_speakers,
            threshold,
            format,
            backend,
        } => {
            let model_dir = model_dir.unwrap_or_else(default_model_dir);
            if let Err(e) = run_diarize(&file, &model_dir, max_speakers, threshold, format, backend)
            {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::DownloadModels { dir, profile } => {
            let dir = dir.unwrap_or_else(default_model_dir);
            if let Err(e) = run_download(&dir, &profile) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn run_diarize(
    file: &Path,
    model_dir: &Path,
    max_speakers: usize,
    threshold: f32,
    format: OutputFormat,
    backend: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let wespeaker_path = model_dir.join("wespeaker_resnet34.onnx");
    let vad_path = model_dir.join("silero_vad.onnx");

    if !wespeaker_path.exists() || !vad_path.exists() {
        return Err(format!(
            "models not found in {}\nRun: polyvoice download-models --dir {}",
            model_dir.display(),
            model_dir.display()
        )
        .into());
    }

    eprintln!("Loading models...");
    let extractor = FbankOnnxExtractor::new(&wespeaker_path, 256, 4)?;
    let mut vad = SileroVad::new(&vad_path, 512)?;

    let backend = match backend.as_str() {
        "kmeans" => ClusteringBackend::KMeans,
        "spectral" => ClusteringBackend::Spectral,
        "auto" => ClusteringBackend::Auto,
        _ => ClusteringBackend::Ahc,
    };
    let config = DiarizationConfig {
        threshold,
        max_speakers,
        sample_rate: SampleRate::default(),
        clustering_backend: backend,
        ..Default::default()
    };
    let vad_config = VadConfig::default();
    let pipeline = Pipeline::new(config, vad_config);

    eprintln!("Reading {}...", file.display());
    let (samples, sr) = polyvoice::wav::read_wav(file)?;
    eprintln!(
        "Audio: {:.1}s, {} Hz, {} samples",
        samples.len() as f64 / sr as f64,
        sr,
        samples.len()
    );

    eprintln!("Running diarization...");
    let result = pipeline.run(&samples, &extractor, &mut vad)?;
    eprintln!(
        "Found {} speaker(s), {} turn(s)\n",
        result.num_speakers,
        result.turns.len()
    );

    match format {
        OutputFormat::Text => {
            for turn in &result.turns {
                println!(
                    "{}\t{:.2}s\t{:.2}s",
                    turn.speaker, turn.time.start, turn.time.end
                );
            }
        }
        OutputFormat::Json => {
            let entries: Vec<serde_json::Value> = result
                .turns
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "speaker": format!("{}", t.speaker),
                        "start": (t.time.start * 100.0).round() / 100.0,
                        "end": (t.time.end * 100.0).round() / 100.0,
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&entries)?);
        }
        OutputFormat::Rttm => {
            let file_id = file.file_stem().and_then(|s| s.to_str()).unwrap_or("audio");
            for turn in &result.turns {
                let dur = turn.time.end - turn.time.start;
                println!(
                    "SPEAKER {} 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>",
                    file_id, turn.time.start, dur, turn.speaker
                );
            }
        }
    }

    Ok(())
}

fn run_download(dir: &Path, profile: &str) -> Result<(), Box<dyn std::error::Error>> {
    use polyvoice::ModelRegistry;
    use polyvoice::Profile;

    std::fs::create_dir_all(dir)?;
    let registry = ModelRegistry::with_cache_dir(dir)?;

    let profiles_to_fetch: Vec<Profile> = match profile {
        "mobile" => vec![Profile::Mobile],
        "balanced" => vec![Profile::Balanced],
        "all" => vec![Profile::Mobile, Profile::Balanced],
        other => return Err(format!("unknown profile '{other}'").into()),
    };

    for prof in profiles_to_fetch {
        eprintln!("Resolving profile '{}'…", prof.manifest_id());
        let bundle = registry.ensure_for_profile(prof)?;
        eprintln!("  segmenter: {}", bundle.segmenter_path.display());
        eprintln!("  embedder : {}", bundle.embedder_path.display());
    }

    eprintln!("\nModels saved to {}", dir.display());
    Ok(())
}
