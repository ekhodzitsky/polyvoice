//! Native diarization from local ONNX and PLDA files. No downloader.
//!
//! ```bash
//! cargo run --no-default-features --features pipeline-local --example local_native -- \
//!   models/int8 fixtures/vbx-plda audio.wav
//! ```
//!
//! The model directory must contain the manifest filenames
//! `powerset_int8.onnx` and `resnet34_int8.onnx`. The PLDA directory is the
//! six `.npy` files (this repo ships `fixtures/vbx-plda`). Each ONNX file is
//! checked against the embedded manifest (SHA-256, and minisign when the
//! entry is signed). Nothing is downloaded.

use polyvoice::{ModelRegistry, Pipeline, PipelineConfig, SampleRate};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(models) = args.next() else {
        eprintln!(
            "usage: local_native <model-dir> <plda-dir> <wav>\n\
             build: cargo run --no-default-features --features pipeline-local --example local_native"
        );
        return ExitCode::from(2);
    };
    let Some(plda) = args.next() else {
        eprintln!("missing <plda-dir>");
        return ExitCode::from(2);
    };
    let Some(wav) = args.next() else {
        eprintln!("missing <wav>");
        return ExitCode::from(2);
    };
    match run(
        PathBuf::from(models),
        PathBuf::from(plda),
        PathBuf::from(wav),
    ) {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(1)
        }
    }
}

fn run(models: PathBuf, plda: PathBuf, wav: PathBuf) -> Result<String, Box<dyn std::error::Error>> {
    let registry = ModelRegistry::with_local_dir(&models)?;
    let config = PipelineConfig {
        vbx_plda_dir: Some(plda),
        ..PipelineConfig::default()
    };
    let pipeline = Pipeline::builder()
        .config(config)
        .with_models_from(registry)
        .build()?;
    let (samples, rate) = polyvoice::wav::load_audio(&wav)?;
    let sr = SampleRate::new(rate).ok_or("wav sample rate is outside 8000..=192000")?;
    let result = pipeline.run(&samples, sr)?;
    let mut text = String::new();
    for turn in &result.turns {
        text.push_str(&format!(
            "{:.3}\t{:.3}\t{}\n",
            turn.time.start, turn.time.end, turn.speaker.0
        ));
    }
    Ok(text)
}
