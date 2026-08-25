//! Build the AS-norm imposter cohort from VoxConverse-dev speaker embeddings.
//!
//! ```bash
//! cargo run --release --features cli-ort --example build_asnorm_cohort -- \
//!     [--data-dir data/voxconverse-dev] [--max-speakers 96] [--out fixtures/asnorm/cohort_voxdev.npy]
//! ```
//!
//! Embeds up to N distinct VoxConverse-dev speakers — the DEV split only,
//! never the evaluation/test split — with the local WeSpeaker ResNet34
//! (`models/wespeaker_resnet34.onnx`, 256-d), averages up to 3 segment
//! embeddings per speaker into a centroid, L2-normalizes each row, and writes
//! the cohort as an NPY v1.0 file (`'<f4'`, shape `(N, 256)`, C-order) — the
//! format `clusterer::asnorm::AsNormCohort::from_npy` loads.
//!
//! Embedding on CPU keeps the fixture byte-reproducible across hosts (the
//! CoreML EP is allowed to drift numerically). After regenerating, record
//! `shasum -a 256 <out>` plus the byte size in the manifest entry.

use polyvoice::FbankOnnxExtractor;
use polyvoice::embedder::Embedder;
use polyvoice::onnx::ExecutionProvider;
use polyvoice::rttm::parse_rttm_file;
use polyvoice::utils::l2_normalize;
use polyvoice::wav::read_wav;
use std::io::Write;
use std::path::{Path, PathBuf};

const EMBEDDING_DIM: usize = 256;
const MIN_SEGMENT_SECS: f64 = 1.0;
const SEGMENTS_PER_SPEAKER: usize = 3;

fn write_npy_f4(path: &Path, rows: &[Vec<f32>]) -> std::io::Result<()> {
    let n = rows.len();
    let d = rows.first().map_or(0, Vec::len);
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({n}, {d}), }}");
    // npy v1.0: magic(6) + ver(2) + header_len(2) + header, padded to 64 B.
    let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
    let header = format!("{dict}{}{}", " ".repeat(pad), "\n");
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for row in rows {
        for &v in row {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = arg_value(&args, "--data-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/voxconverse-dev"));
    let max_speakers: usize = arg_value(&args, "--max-speakers")
        .and_then(|v| v.parse().ok())
        .unwrap_or(96);
    let out = arg_value(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("fixtures/asnorm/cohort_voxdev.npy"));

    let model = Path::new("models/wespeaker_resnet34.onnx");
    anyhow::ensure!(
        model.is_file(),
        "models/wespeaker_resnet34.onnx missing (run the model download first)"
    );
    let extractor = FbankOnnxExtractor::new(model, EMBEDDING_DIM, 1, ExecutionProvider::Cpu)?;

    let audio_dir = data_dir.join("audio");
    let rttm_dir = data_dir.join("rttm");
    let mut wavs: Vec<_> = std::fs::read_dir(&audio_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "wav"))
        .collect();
    wavs.sort();

    let mut cohort: Vec<Vec<f32>> = Vec::new();
    'files: for wav_path in &wavs {
        let stem = wav_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
        if !rttm_path.is_file() {
            continue;
        }
        let segments = parse_rttm_file(&rttm_path)?;
        let (samples, sr) = read_wav(wav_path)?;
        anyhow::ensure!(
            sr == 16000,
            "{}: expected 16 kHz, got {sr}",
            wav_path.display()
        );

        // First-seen speaker order per file keeps the cohort deterministic.
        let mut speakers: Vec<String> = Vec::new();
        for seg in &segments {
            if !speakers.contains(&seg.speaker) {
                speakers.push(seg.speaker.clone());
            }
        }
        for speaker in speakers {
            if cohort.len() >= max_speakers {
                break 'files;
            }
            let mut acc = vec![0.0f32; EMBEDDING_DIM];
            let mut used = 0usize;
            for seg in segments.iter().filter(|s| s.speaker == speaker) {
                if used >= SEGMENTS_PER_SPEAKER {
                    break;
                }
                if seg.duration < MIN_SEGMENT_SECS {
                    continue;
                }
                let start = (seg.start * 16000.0) as usize;
                let end = ((seg.start + seg.duration) * 16000.0) as usize;
                if start >= samples.len() {
                    continue;
                }
                let end = end.min(samples.len());
                let emb = extractor.embed(&samples[start..end])?;
                for (a, e) in acc.iter_mut().zip(&emb) {
                    *a += e;
                }
                used += 1;
            }
            if used > 0 {
                for a in &mut acc {
                    *a /= used as f32;
                }
                l2_normalize(&mut acc);
                cohort.push(acc);
            }
        }
    }

    anyhow::ensure!(
        cohort.len() >= 8,
        "only {} speakers embedded — a usable cohort needs many more",
        cohort.len()
    );
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_npy_f4(&out, &cohort)?;
    let size = std::fs::metadata(&out)?.len();
    eprintln!(
        "wrote {} ({} speakers x {EMBEDDING_DIM} dims, {size} bytes)",
        out.display(),
        cohort.len()
    );
    eprintln!("for the manifest entry: shasum -a 256 {}", out.display());
    Ok(())
}
