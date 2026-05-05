//! WAV file I/O via the `hound` crate.

use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum WavError {
    #[error("failed to read WAV: {0}")]
    Read(#[from] hound::Error),
    #[error("unsupported sample format: {0}")]
    UnsupportedFormat(String),
}

/// Read a WAV file and return mono f32 samples normalized to [-1.0, 1.0] and its sample rate.
///
/// Stereo files are downmixed by averaging channels. 16-bit and 32-bit float
/// formats are supported.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), WavError> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<f32>, _>>()?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()?,
    };

    let mono = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok((mono, sample_rate))
}
