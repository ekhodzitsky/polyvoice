//! Emit RTTM for one WAV via speakrs.
//!
//! ```text
//! speakrs-rttm --mode cpu input.wav
//! speakrs-rttm --mode coreml --models-dir /path/to/models input.wav -o out.rttm
//! ```
//!
//! Models: `--models-dir` / `SPEAKRS_MODELS_DIR`, else `from_pretrained` (online).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use speakrs::{ExecutionMode, OwnedDiarizationPipeline};

fn usage() -> ! {
    eprintln!(
        "Usage: speakrs-rttm [--mode cpu|coreml|coreml-fast|cuda|cuda-fast|migraphx] \
         [--models-dir DIR] [-o OUT.rttm] [--file-id ID] <audio.wav>\n\
         Env: SPEAKRS_MODE, SPEAKRS_MODELS_DIR"
    );
    process::exit(2);
}

fn parse_mode(s: &str) -> ExecutionMode {
    match s.to_ascii_lowercase().as_str() {
        "cpu" => ExecutionMode::Cpu,
        "coreml" => ExecutionMode::CoreMl,
        "coreml-fast" | "coreml_fast" => ExecutionMode::CoreMlFast,
        "cuda" => ExecutionMode::Cuda,
        "cuda-fast" | "cuda_fast" => ExecutionMode::CudaFast,
        "migraphx" => ExecutionMode::MiGraphX,
        other => {
            eprintln!("unknown mode {other:?}");
            usage();
        }
    }
}

fn load_wav_mono_16k(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 {
        return Err(format!(
            "expected 16 kHz WAV, got {} Hz ({})",
            spec.sample_rate,
            path.display()
        ));
    }
    let channels = spec.channels.max(1) as usize;
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample.min(31) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
    };
    let interleaved = samples.map_err(|e| e.to_string())?;
    if channels == 1 {
        return Ok(interleaved);
    }
    // Average channels → mono.
    Ok(interleaved
        .chunks(channels)
        .map(|c| c.iter().sum::<f32>() / channels as f32)
        .collect())
}

fn file_id_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio")
        .to_owned()
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut mode = env::var("SPEAKRS_MODE").unwrap_or_else(|_| "cpu".into());
    let mut models_dir: Option<PathBuf> = env::var_os("SPEAKRS_MODELS_DIR").map(PathBuf::from);
    let mut out_path: Option<PathBuf> = None;
    let mut file_id: Option<String> = None;
    let mut wav_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                mode = args.get(i).cloned().unwrap_or_else(|| usage());
            }
            "--models-dir" => {
                i += 1;
                models_dir = Some(PathBuf::from(args.get(i).cloned().unwrap_or_else(|| usage())));
            }
            "-o" | "--output" => {
                i += 1;
                out_path = Some(PathBuf::from(args.get(i).cloned().unwrap_or_else(|| usage())));
            }
            "--file-id" => {
                i += 1;
                file_id = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "-h" | "--help" => usage(),
            s if s.starts_with('-') => {
                eprintln!("unknown flag {s}");
                usage();
            }
            s => {
                if wav_path.is_some() {
                    eprintln!("unexpected argument {s}");
                    usage();
                }
                wav_path = Some(PathBuf::from(s));
            }
        }
        i += 1;
    }

    let wav_path = wav_path.unwrap_or_else(|| usage());
    let mode = parse_mode(&mode);
    let file_id = file_id.unwrap_or_else(|| file_id_from_path(&wav_path));

    let audio = match load_wav_mono_16k(&wav_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("wav load failed: {e}");
            process::exit(1);
        }
    };

    let pipeline = match models_dir {
        Some(dir) => OwnedDiarizationPipeline::from_dir(&dir, mode),
        None => OwnedDiarizationPipeline::from_pretrained(mode),
    };
    let mut pipeline = match pipeline {
        Ok(p) => p,
        Err(e) => {
            eprintln!("pipeline init failed: {e}");
            process::exit(1);
        }
    };

    let result = match pipeline.run(&audio) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("diarize failed: {e}");
            process::exit(1);
        }
    };

    let rttm = result.rttm(&file_id);
    if let Some(path) = out_path {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Err(e) = fs::write(&path, &rttm) {
            eprintln!("write {}: {e}", path.display());
            process::exit(1);
        }
    } else {
        print!("{rttm}");
    }
}
