//! WAV file I/O via the `hound` crate.
//!
//! Read mono f32 samples from WAV files with size and duration guards.
//! See [`read_wav`] for the main entry point.

use std::path::Path;

const MAX_WAV_FILE_SIZE: u64 = 1_073_741_824; // 1 GiB

#[derive(thiserror::Error, Debug)]
pub enum WavError {
    #[error("failed to read WAV: {0}")]
    Read(#[from] hound::Error),
    #[error("unsupported sample format: {0}")]
    UnsupportedFormat(String),
    #[error("WAV file too large: {size} bytes (max {max} bytes)")]
    FileTooLarge { size: u64, max: u64 },
    #[error("WAV duration too long: {duration_secs:.1}s (max {max_secs:.1}s)")]
    DurationTooLong { duration_secs: f64, max_secs: f64 },
    #[error("failed to get file metadata: {0}")]
    Metadata(#[from] std::io::Error),
}

/// { TODO: precondition }
/// `pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), WavError>`
/// { TODO: postcondition }
/// Read a WAV file and return mono f32 samples normalized to [-1.0, 1.0] and its sample rate.
///
/// Stereo files are downmixed by averaging channels. 16-bit and 32-bit float
/// formats are supported.
///
/// # Guards
///
/// - If the file size exceeds 1 GiB, returns [`WavError::FileTooLarge`].
/// - If the declared duration in the WAV header exceeds 1 hour, returns
///   [`WavError::DurationTooLong`] before reading any samples.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), WavError> {
    let metadata = std::fs::metadata(path)?;
    let file_size = metadata.len();
    if file_size > MAX_WAV_FILE_SIZE {
        return Err(WavError::FileTooLarge {
            size: file_size,
            max: MAX_WAV_FILE_SIZE,
        });
    }

    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let duration = reader.duration();
    let duration_secs = duration as f64 / sample_rate as f64;
    const MAX_DURATION_SECS: f64 = 3600.0;
    if duration_secs > MAX_DURATION_SECS {
        return Err(WavError::DurationTooLong {
            duration_secs,
            max_secs: MAX_DURATION_SECS,
        });
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_error() {
        let result = read_wav(Path::new("/nonexistent/path/file.wav"));
        assert!(result.is_err());
    }

    #[test]
    fn wav_error_display() {
        let e = WavError::FileTooLarge {
            size: 2_000_000_000,
            max: MAX_WAV_FILE_SIZE,
        };
        let msg = format!("{e}");
        assert!(msg.contains("too large"));
    }
}
