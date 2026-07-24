//! Audio file I/O for the loading layer.
//!
//! - [`read_wav`] — raw WAV decode via `hound` (any sample rate; caller decides).
//! - [`load_audio`] — pipeline-ready mono f32 at 16 kHz.
//!
//! Without the `audio-io` feature, [`load_audio`] accepts only 16 kHz WAV and
//! returns a clear rebuild hint for other rates/formats. With `audio-io`,
//! symphonia decodes mp3/flac/ogg/m4a/aac (and other supported containers) and
//! rubato resamples any rate → 16 kHz mono (multi-channel is averaged).

use std::path::Path;

const MAX_WAV_FILE_SIZE: u64 = 1_073_741_824; // 1 GiB
const MAX_DURATION_SECS: f64 = 3600.0;

/// Sample rate expected by the diarization / ASR pipelines (mono PCM).
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

#[cfg(feature = "audio-io")]
mod decode;
#[cfg(feature = "audio-io")]
mod resample;

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
    /// Non-16 kHz input without the `audio-io` feature (resampling unavailable).
    #[error(
        "unsupported sample rate {actual} Hz (expected {expected} Hz mono). \
         Rebuild with `--features audio-io` to enable resampling of any rate to 16 kHz"
    )]
    UnsupportedSampleRate { actual: u32, expected: u32 },
    /// Non-WAV input (or unreadable path) without the `audio-io` feature.
    #[error(
        "cannot decode '{path}': multi-format decoding requires the `audio-io` cargo feature \
         (rebuild with `--features audio-io`). With that feature: mp3, flac, ogg, m4a/aac, wav \
         at any sample rate are decoded and resampled to 16 kHz mono"
    )]
    FeatureRequired { path: String },
    /// Decoder failure (symphonia) or unsupported codec (e.g. Opus).
    #[error("failed to decode audio: {0}")]
    Decode(String),
    /// Resampler construction or processing failure.
    #[error("failed to resample audio: {0}")]
    Resample(String),
}

/// Read a WAV file and return mono f32 samples normalized to [-1.0, 1.0] and its sample rate.
///
/// Stereo (and multi-channel) files are downmixed by averaging channels. 16-bit
/// integer and 32-bit float formats are supported. The returned sample rate is
/// whatever the WAV header declares — this function does **not** resample.
///
/// # Guards
///
/// - If the file size exceeds 1 GiB, returns [`WavError::FileTooLarge`].
/// - If the declared duration in the WAV header exceeds 1 hour, returns
///   [`WavError::DurationTooLong`] before reading any samples.
///
/// For pipeline-ready 16 kHz mono, prefer [`load_audio`].
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
    if duration_secs > MAX_DURATION_SECS {
        return Err(WavError::DurationTooLong {
            duration_secs,
            max_secs: MAX_DURATION_SECS,
        });
    }

    let bps = spec.bits_per_sample;
    if bps == 0 || bps > 32 {
        return Err(WavError::UnsupportedFormat(format!(
            "bits_per_sample {bps} out of supported range 1..=32"
        )));
    }

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (bps - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<f32>, _>>()?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()?,
    };

    let mono = downmix_to_mono(interleaved, channels);
    Ok((mono, sample_rate))
}

/// Load audio for the diarization pipeline: mono f32 at [`TARGET_SAMPLE_RATE`] (16 kHz).
///
/// # Without `audio-io`
///
/// Accepts 16 kHz mono/stereo WAV only (via [`read_wav`]). Other sample rates or
/// formats return [`WavError::UnsupportedSampleRate`] / [`WavError::FeatureRequired`]
/// with a rebuild hint.
///
/// # With `audio-io`
///
/// Decodes common formats (mp3, flac, ogg/vorbis, m4a/aac, wav, aiff, caf, …)
/// via symphonia, downmixes multi-channel to mono by averaging, and resamples
/// any rate to 16 kHz with rubato (FFT synchronous resampler). WAV continues
/// to use hound. Opus is not supported (no native libopus dep); you get a
/// named decode error.
///
/// Multi-channel downmix discards spatial information (e.g. stereo telephony).
pub fn load_audio(path: &Path) -> Result<(Vec<f32>, u32), WavError> {
    #[cfg(feature = "audio-io")]
    {
        load_audio_any(path)
    }
    #[cfg(not(feature = "audio-io"))]
    {
        load_audio_wav_only(path)
    }
}

#[cfg(not(feature = "audio-io"))]
fn load_audio_wav_only(path: &Path) -> Result<(Vec<f32>, u32), WavError> {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        let ext = ext.to_ascii_lowercase();
        if !matches!(ext.as_str(), "wav" | "wave") {
            return Err(WavError::FeatureRequired {
                path: path.display().to_string(),
            });
        }
    }

    let (samples, sr) = read_wav(path)?;
    if sr != TARGET_SAMPLE_RATE {
        return Err(WavError::UnsupportedSampleRate {
            actual: sr,
            expected: TARGET_SAMPLE_RATE,
        });
    }
    Ok((samples, TARGET_SAMPLE_RATE))
}

#[cfg(feature = "audio-io")]
fn load_audio_any(path: &Path) -> Result<(Vec<f32>, u32), WavError> {
    let (samples, sr) = decode_path(path)?;
    if samples.is_empty() {
        return Ok((samples, TARGET_SAMPLE_RATE));
    }
    if sr == TARGET_SAMPLE_RATE {
        return Ok((samples, TARGET_SAMPLE_RATE));
    }
    if sr == 0 {
        return Err(WavError::Decode("sample rate is zero".into()));
    }
    let resampled = resample::resample_mono(&samples, sr, TARGET_SAMPLE_RATE)?;
    Ok((resampled, TARGET_SAMPLE_RATE))
}

#[cfg(feature = "audio-io")]
fn decode_path(path: &Path) -> Result<(Vec<f32>, u32), WavError> {
    let is_wav = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "wav" | "wave"));

    if is_wav {
        // Prefer hound for the existing WAV path (stable, zero media-codec deps on
        // the hot path when the file really is WAV).
        return read_wav(path);
    }

    decode::decode_with_symphonia(path)
}

fn downmix_to_mono(interleaved: Vec<f32>, channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved;
    }
    interleaved
        .chunks(channels)
        .map(|ch| ch.iter().sum::<f32>() / channels as f32)
        .collect()
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_sine_wav(path: &Path, sample_rate: u32, duration_secs: f32, freq_hz: f32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let n = (sample_rate as f32 * duration_secs) as usize;
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..n {
            let t = i as f32 / sample_rate as f32;
            let sample = (t * std::f32::consts::TAU * freq_hz).sin();
            writer
                .write_sample((sample * 32767.0 * 0.5) as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
    }

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

    #[test]
    fn load_audio_accepts_16k_wav() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ok.wav");
        write_sine_wav(&path, 16_000, 0.25, 440.0);
        let (samples, sr) = load_audio(&path).unwrap();
        assert_eq!(sr, TARGET_SAMPLE_RATE);
        assert_eq!(samples.len(), 4_000);
    }

    #[test]
    fn load_audio_rejects_non_wav_without_feature() {
        #[cfg(not(feature = "audio-io"))]
        {
            let err = load_audio(Path::new("meeting.mp3")).unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("audio-io"),
                "expected rebuild hint, got: {msg}"
            );
        }
    }

    #[test]
    fn load_audio_rejects_wrong_rate_without_feature() {
        #[cfg(not(feature = "audio-io"))]
        {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("48k.wav");
            write_sine_wav(&path, 48_000, 0.1, 440.0);
            let err = load_audio(&path).unwrap_err();
            match err {
                WavError::UnsupportedSampleRate {
                    actual: 48_000,
                    expected: TARGET_SAMPLE_RATE,
                } => {}
                other => panic!("unexpected error: {other}"),
            }
            let msg = format!("{err}");
            assert!(msg.contains("audio-io"));
        }
    }

    #[cfg(feature = "audio-io")]
    #[test]
    fn load_audio_resamples_multi_rate_wav_duration() {
        // Synthetic tones at common rates; duration after resample ≈ input duration.
        // Allow a small filter-tail tolerance (FFT resampler delay trim is not exact).
        let cases: &[(u32, f32)] = &[
            (8_000, 1.0),
            (22_050, 1.0),
            (44_100, 1.0),
            (48_000, 1.0),
        ];
        let dir = tempfile::tempdir().unwrap();
        for &(rate, secs) in cases {
            let path = dir.path().join(format!("tone_{rate}.wav"));
            write_sine_wav(&path, rate, secs, 220.0);
            let (samples, sr) = load_audio(&path).unwrap();
            assert_eq!(sr, TARGET_SAMPLE_RATE, "rate={rate}");
            let got_secs = samples.len() as f64 / TARGET_SAMPLE_RATE as f64;
            let err = (got_secs - secs as f64).abs();
            assert!(
                err < 0.02,
                "rate={rate}: duration {got_secs:.4}s vs {secs}s (err={err:.4})"
            );
            // Non-silent output.
            let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
            assert!(energy > 1e-6, "rate={rate}: near-silent after resample");
        }
    }

    #[cfg(feature = "audio-io")]
    #[test]
    fn load_audio_16k_wav_bypasses_resampler_byte_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("native16k.wav");
        write_sine_wav(&path, 16_000, 0.5, 330.0);
        let (via_load, sr) = load_audio(&path).unwrap();
        let (via_read, sr2) = read_wav(&path).unwrap();
        assert_eq!(sr, 16_000);
        assert_eq!(sr2, 16_000);
        assert_eq!(via_load.len(), via_read.len());
        assert_eq!(via_load, via_read);
    }

    #[test]
    fn read_wav_stereo_roundtrip_via_cursor_buffer() {
        // Smoke that hound path still works when writing via Cursor (mirrors
        // integration tests that build synthetic WAVs in memory).
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
            for _ in 0..100 {
                writer.write_sample(0i16).unwrap();
            }
            writer.finalize().unwrap();
        }
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &buf).unwrap();
        let (samples, sr) = read_wav(tmp.path()).unwrap();
        assert_eq!(sr, 16_000);
        assert_eq!(samples.len(), 100);
    }
}
