//! Offline mono resampling via rubato (FFT synchronous resampler).
//!
//! Only compiled with the `audio-io` feature.

use super::WavError;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{FixedSync, Fft, Resampler};

/// Resample mono f32 PCM from `from_hz` to `to_hz`.
///
/// Uses rubato's synchronous FFT resampler with fixed-rate whole-clip
/// processing (`process_all`), which trims filter startup delay so duration
/// stays close to the source.
pub(super) fn resample_mono(
    samples: &[f32],
    from_hz: u32,
    to_hz: u32,
) -> Result<Vec<f32>, WavError> {
    if from_hz == to_hz {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    if from_hz == 0 || to_hz == 0 {
        return Err(WavError::Resample(format!(
            "invalid sample rates: {from_hz} → {to_hz}"
        )));
    }

    // Chunk size is a hint; FixedSync::Both picks sizes that fit the ratio.
    const CHUNK_HINT: usize = 1024;
    let mut resampler = Fft::<f32>::new(
        from_hz as usize,
        to_hz as usize,
        CHUNK_HINT,
        1, // mono
        FixedSync::Both,
    )
    .map_err(|e| WavError::Resample(format!("construct resampler {from_hz}→{to_hz}: {e}")))?;

    let frames = samples.len();
    let input = InterleavedSlice::new(samples, 1, frames)
        .map_err(|e| WavError::Resample(format!("input adapter: {e}")))?;

    let output = resampler
        .process_all(&input, frames, None)
        .map_err(|e| WavError::Resample(format!("process_all: {e}")))?;

    Ok(output.take_data())
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::TARGET_SAMPLE_RATE;

    #[test]
    fn identity_rate_is_clone() {
        let s = vec![0.1, 0.2, 0.3];
        let out = resample_mono(&s, TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE).unwrap();
        assert_eq!(out, s);
    }

    #[test]
    fn empty_stays_empty() {
        let out = resample_mono(&[], 48_000, TARGET_SAMPLE_RATE).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn forty_eight_to_sixteen_length() {
        // 1 second of 48 kHz silence → ~16k samples.
        let input = vec![0.0f32; 48_000];
        let out = resample_mono(&input, 48_000, TARGET_SAMPLE_RATE).unwrap();
        let secs = out.len() as f64 / TARGET_SAMPLE_RATE as f64;
        assert!(
            (secs - 1.0).abs() < 0.02,
            "got {secs}s ({} samples)",
            out.len()
        );
    }
}
