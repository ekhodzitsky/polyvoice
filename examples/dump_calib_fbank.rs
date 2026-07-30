//! One-off: dump fbank+CMVN calibration windows (npy) for static QDQ
//! quantization of the embedder models. Windows are real speech segments
//! (0.5-5s) taken from VoxConverse-dev RTTM annotations, matching the
//! production pipeline's feature path (FbankExtractor::default + apply_cmvn).

use polyvoice::features::{FbankConfig, FbankExtractor, apply_cmvn};
use polyvoice::rttm::parse_rttm_file;
use polyvoice::wav::read_wav;
use std::io::Write;
use std::path::Path;

fn write_npy(path: &Path, frames: &[Vec<f32>]) -> std::io::Result<()> {
    let t = frames.len();
    let m = frames.first().map_or(0, Vec::len);
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({t}, {m}), }}");
    // npy v1.0: magic(6) + ver(2) + header_len(2) + header, padded to 64 B.
    let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
    let header = format!("{dict}{}{}", " ".repeat(pad), "\n");
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for frame in frames {
        for &v in frame {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let out_dir = Path::new("/tmp/calib_fbank_dev");
    std::fs::create_dir_all(out_dir)?;
    let audio_dir = Path::new("data/voxconverse-dev/audio");
    let rttm_dir = Path::new("data/voxconverse-dev/rttm");

    const MAX_FILES: usize = 50;
    const WINDOWS_PER_FILE: usize = 40;
    const MIN_DUR: f64 = 0.5;
    const MAX_DUR: f64 = 5.0;

    let extractor = FbankExtractor::new(FbankConfig::default());
    let mut total = 0usize;

    let mut entries: Vec<_> = std::fs::read_dir(audio_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "wav"))
        .collect();
    entries.sort();
    // Deterministic stride sampling across the 216 files.
    let stride = (entries.len() / MAX_FILES).max(1);
    let picked: Vec<_> = entries.iter().step_by(stride).take(MAX_FILES).collect();

    for wav_path in picked {
        if total >= MAX_FILES * WINDOWS_PER_FILE {
            break;
        }
        let stem = wav_path.file_stem().unwrap().to_string_lossy().to_string();
        let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
        if !rttm_path.exists() {
            continue;
        }
        let segs = parse_rttm_file(&rttm_path)?;
        let (samples, sr) = read_wav(&wav_path)?;
        assert_eq!(sr, 16000);

        let mut n = 0usize;
        for seg in &segs {
            if n >= WINDOWS_PER_FILE {
                break;
            }
            if seg.duration < MIN_DUR {
                continue;
            }
            // Chunk long segments into <= MAX_DUR windows (keeps real speech,
            // adds length diversity for the dynamic T axis).
            let mut off = 0.0f64;
            while off + MIN_DUR <= seg.duration && n < WINDOWS_PER_FILE {
                let dur = (seg.duration - off).min(MAX_DUR);
                let start = ((seg.start + off) * 16000.0) as usize;
                let len = (dur * 16000.0) as usize;
                off += dur;
                if start + len > samples.len() {
                    continue;
                }
                let frames = extractor.extract(&samples[start..start + len])?;
                if frames.is_empty() {
                    continue;
                }
                let frames = apply_cmvn(&frames);
                write_npy(&out_dir.join(format!("{stem}_{n:03}.npy")), &frames)?;
                n += 1;
                total += 1;
            }
        }
        println!("{stem}: {n} windows");
    }
    println!("wrote {total} calibration windows to {out_dir:?}");
    Ok(())
}
