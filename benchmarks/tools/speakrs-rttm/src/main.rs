//! Emit RTTM for one or many WAVs via speakrs (warm pipeline in batch mode).
//!
//! ```text
//! speakrs-rttm --mode coreml file.wav
//! speakrs-rttm --mode coreml --hyp-dir out/ --models-dir M data/voxconverse-test/audio
//! speakrs-rttm --mode cpu --hyp-dir out/ a.wav b.wav
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use speakrs::{ExecutionMode, OwnedDiarizationPipeline};

fn usage() -> ! {
    eprintln!(
        "Usage: speakrs-rttm [--mode MODE] [--models-dir DIR] [--hyp-dir DIR] \
         [-o OUT.rttm] [--file-id ID] <audio.wav|audio-dir> [more.wav ...]\n\
         Modes: cpu|coreml|coreml-fast|cuda|cuda-fast|migraphx\n\
         Env: SPEAKRS_MODE, SPEAKRS_MODELS_DIR\n\
         Batch: pass a directory of .wav or multiple files; requires --hyp-dir.\n\
         Single file: prints RTTM to stdout or -o path."
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

fn collect_wavs(inputs: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in inputs {
        if p.is_dir() {
            let mut kids: Vec<_> = fs::read_dir(p)
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|q| {
                    q.extension()
                        .and_then(|x| x.to_str())
                        .map(|x| x.eq_ignore_ascii_case("wav"))
                        .unwrap_or(false)
                })
                .collect();
            kids.sort();
            out.extend(kids);
        } else {
            out.push(p.clone());
        }
    }
    out
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut mode = env::var("SPEAKRS_MODE").unwrap_or_else(|_| "cpu".into());
    let mut models_dir: Option<PathBuf> = env::var_os("SPEAKRS_MODELS_DIR").map(PathBuf::from);
    let mut out_path: Option<PathBuf> = None;
    let mut hyp_dir: Option<PathBuf> = None;
    let mut file_id: Option<String> = None;
    let mut inputs: Vec<PathBuf> = Vec::new();

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
            "--hyp-dir" => {
                i += 1;
                hyp_dir = Some(PathBuf::from(args.get(i).cloned().unwrap_or_else(|| usage())));
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
            s => inputs.push(PathBuf::from(s)),
        }
        i += 1;
    }

    if inputs.is_empty() {
        usage();
    }

    let wavs = collect_wavs(&inputs);
    if wavs.is_empty() {
        eprintln!("no .wav inputs found");
        process::exit(1);
    }

    let batch = wavs.len() > 1 || inputs.iter().any(|p| p.is_dir());
    if batch && hyp_dir.is_none() {
        eprintln!("batch mode requires --hyp-dir DIR");
        process::exit(2);
    }

    let mode = parse_mode(&mode);
    let t0 = Instant::now();
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
    eprintln!(
        "[speakrs-rttm] mode={mode:?} load={:.1}s files={}",
        t0.elapsed().as_secs_f64(),
        wavs.len()
    );

    if let Some(ref dir) = hyp_dir {
        let _ = fs::create_dir_all(dir);
    }

    let mut audio_secs = 0.0f64;
    let mut wall = 0.0f64;
    let mut ok = 0usize;
    let mut fail = 0usize;

    for (idx, wav_path) in wavs.iter().enumerate() {
        let fid = if wavs.len() == 1 {
            file_id.clone().unwrap_or_else(|| file_id_from_path(wav_path))
        } else {
            file_id_from_path(wav_path)
        };

        if let Some(ref dir) = hyp_dir {
            let dest = dir.join(format!("{fid}.rttm"));
            if dest.is_file() && dest.metadata().map(|m| m.len() > 0).unwrap_or(false) {
                eprintln!("[skip] {fid} (cached)");
                ok += 1;
                continue;
            }
        }

        let audio = match load_wav_mono_16k(wav_path) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("[fail] {fid}: wav {e}");
                fail += 1;
                continue;
            }
        };
        let dur = audio.len() as f64 / 16_000.0;
        let start = Instant::now();
        let result = match pipeline.run(&audio) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[fail] {fid}: diarize {e}");
                fail += 1;
                continue;
            }
        };
        let elapsed = start.elapsed().as_secs_f64();
        audio_secs += dur;
        wall += elapsed;
        let rttm = result.rttm(&fid);

        if let Some(ref dir) = hyp_dir {
            let dest = dir.join(format!("{fid}.rttm"));
            if let Err(e) = fs::write(&dest, &rttm) {
                eprintln!("[fail] {fid}: write {e}");
                fail += 1;
                continue;
            }
        } else if let Some(ref path) = out_path {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if let Err(e) = fs::write(path, &rttm) {
                eprintln!("write {}: {e}", path.display());
                process::exit(1);
            }
        } else {
            print!("{rttm}");
        }

        ok += 1;
        eprintln!(
            "[{}/{}] {fid} audio={dur:.1}s wall={elapsed:.2}s RTFx={:.1}",
            idx + 1,
            wavs.len(),
            if elapsed > 0.0 { dur / elapsed } else { 0.0 }
        );
    }

    if batch || hyp_dir.is_some() {
        let rtf = if audio_secs > 0.0 {
            wall / audio_secs
        } else {
            0.0
        };
        eprintln!(
            "[speakrs-rttm] done ok={ok} fail={fail} audio={audio_secs:.1}s wall={wall:.1}s RTF={rtf:.4} RTFx={:.1}",
            if rtf > 0.0 { 1.0 / rtf } else { 0.0 }
        );
    }
}
