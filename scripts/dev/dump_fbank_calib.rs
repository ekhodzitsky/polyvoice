//! One-off: dump fbank+CMVN calibration inputs (npy) for static QDQ quantization.

use polyvoice::features::{FbankConfig, FbankExtractor, apply_cmvn};
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
    let out_dir = Path::new("/tmp/calib_fbank");
    std::fs::create_dir_all(out_dir)?;
    let extractor = FbankExtractor::new(FbankConfig::default());
    let mut n = 0usize;
    for stem in ["aepyx", "aggyz", "aiqwk"] {
        let (samples, sr) = read_wav(Path::new(&format!(
            "data/voxconverse-test/audio/{stem}.wav"
        )))?;
        assert_eq!(sr, 16000);
        let mut off = 0usize;
        while off + 24000 <= samples.len() && n < 600 {
            let frames = extractor.extract(&samples[off..off + 24000])?;
            let frames = apply_cmvn(&frames);
            write_npy(&out_dir.join(format!("{n:04}.npy")), &frames)?;
            n += 1;
            off += 24000;
        }
    }
    println!("wrote {n} calibration windows to {out_dir:?}");
    Ok(())
}
