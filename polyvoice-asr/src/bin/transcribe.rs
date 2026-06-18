#![allow(deprecated)] // legacy embedding API (FbankOnnxExtractor); matches the core CLI/bench.
//! polyvoice-transcribe — who-said-what CLI: diarize -> one ASR pass -> join.
//!
//! Lives in polyvoice-asr (not the core `polyvoice` CLI) because the core crate
//! cannot depend on this companion — that would be a package cycle. Diarizes with
//! the validated legacy pipeline, runs a single Parakeet TDT pass, and joins words
//! to speakers. stdout carries only the result; all progress goes to stderr.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use polyvoice::format::{write_srt, write_txt, write_vtt};
use polyvoice::models::ModelRegistry;
use polyvoice::types::{
    ClusterConfig, DiarizationConfig, Profile, SampleRate, SpeakerTurn, WordAlignment,
};
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
use polyvoice::{FbankOnnxExtractor, Pipeline, SileroVad, who_said_what};
use polyvoice_asr::ParakeetAsr;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "polyvoice-transcribe",
    about = "Who-said-what: diarize a WAV, transcribe it, and attribute words to speakers"
)]
struct Args {
    /// Input WAV (mono 16 kHz recommended).
    wav: PathBuf,
    /// Directory with the Parakeet TDT model files (encoder-model.onnx +
    /// encoder-model.onnx.data, decoder_joint-model.onnx, vocab.txt).
    #[arg(long)]
    asr_model: PathBuf,
    /// Diarization profile (mobile | balanced).
    #[arg(long, default_value = "balanced")]
    profile: String,
    /// Output format.
    #[arg(long, value_enum, default_value_t = OutFormat::Json)]
    format: OutFormat,
    /// AHC cosine-similarity threshold.
    #[arg(long, default_value = "0.45")]
    threshold: f32,
    /// Suppress progress on stderr.
    #[arg(long)]
    quiet: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutFormat {
    Json,
    Srt,
    Vtt,
    Txt,
}

/// who-said-what JSON payload: turns carry `text`, plus a per-word alignment array.
/// Field shapes match schema/diarization-result-v1.json.
#[derive(Serialize)]
struct TranscriptOutput {
    schema_version: &'static str,
    num_speakers: usize,
    turns: Vec<SpeakerTurn>,
    words: Vec<WordAlignment>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let profile: Profile = args.profile.parse().context("invalid --profile")?;
    let registry = ModelRegistry::default().context("model registry")?;

    // --- diarize with the validated legacy pipeline ---
    let models = registry
        .ensure_for_profile(profile)
        .context("ensure diarization models")?;
    let extractor = FbankOnnxExtractor::new(&models.embedder_path, profile.embedding_dim(), 1)
        .context("load embedder")?;
    let vad_path = registry.ensure("silero_vad").context("silero_vad model")?;
    let mut vad = SileroVad::new(&vad_path, 512).context("load VAD")?;
    let config = DiarizationConfig {
        cluster: ClusterConfig {
            threshold: args.threshold,
            ..Default::default()
        },
        ..DiarizationConfig::default()
    };
    let pipeline = Pipeline::new(config, VadConfig::default());

    if !args.quiet {
        eprintln!("Reading {}...", args.wav.display());
    }
    let (samples, sr_hz) =
        read_wav(&args.wav).with_context(|| format!("read WAV {}", args.wav.display()))?;
    let sr = SampleRate::new(sr_hz).with_context(|| format!("invalid sample rate {sr_hz} Hz"))?;
    if !args.quiet {
        eprintln!("Diarizing {} samples ({sr_hz} Hz)...", samples.len());
    }
    let diar = pipeline
        .run(&samples, &extractor, &mut vad)
        .context("diarization failed")?;

    // --- one ASR pass, then join words to speakers ---
    if !args.quiet {
        eprintln!("Loading ASR model from {}...", args.asr_model.display());
    }
    let asr = ParakeetAsr::from_dir(&args.asr_model).context("load ASR model")?;
    if !args.quiet {
        eprintln!("Transcribing + attributing...");
    }
    let wsw =
        who_said_what(&diar.turns, &asr, &samples, sr).context("who-said-what cascade failed")?;
    if !args.quiet {
        eprintln!(
            "Done — {} turns, {} words, {} speakers",
            wsw.turns.len(),
            wsw.words.len(),
            diar.num_speakers
        );
    }

    // --- emit result on stdout only ---
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    match args.format {
        OutFormat::Json => {
            let payload = TranscriptOutput {
                schema_version: "diarization-result-v1",
                num_speakers: diar.num_speakers,
                turns: wsw.turns,
                words: wsw.words,
            };
            serde_json::to_writer_pretty(&mut out, &payload).context("write JSON")?;
            writeln!(out).context("write JSON newline")?;
        }
        OutFormat::Srt => write_srt(&mut out, &wsw.turns).context("write SRT")?,
        OutFormat::Vtt => write_vtt(&mut out, &wsw.turns).context("write VTT")?,
        OutFormat::Txt => write_txt(&mut out, &wsw.turns).context("write TXT")?,
    }
    Ok(())
}
