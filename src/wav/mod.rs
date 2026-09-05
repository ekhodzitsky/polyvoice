//! Audio file I/O for the loading layer.
//!
//! - [`read_wav`] — raw WAVE-family decode via `ryf` (any sample rate; caller decides).
//! - [`load_audio`] — pipeline-ready mono f32 at 16 kHz.
//!
//! Without the `audio-io` feature, [`load_audio`] accepts only 16 kHz WAV and
//! returns a clear rebuild hint for other rates/formats. With `audio-io`,
//! symphonia decodes mp3/flac/ogg/m4a/aac (and other supported containers) and
//! rubato resamples any rate → 16 kHz mono (multi-channel is averaged).

use std::io::{Seek, SeekFrom};
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
    Read(String),
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

fn decode_opts() -> ryf::DecodeOptions {
    // Mix-to-mono + speech ingest. Keep speech's 192 kHz sample-rate ceiling
    // and 4 GiB planar-f32 budget; do not inherit the 48 kHz *frame-budget*
    // clamp, or a 96 kHz file would hit DurationTooLong at 30 minutes.
    let speech = ryf::DecodeOptions::speech();
    let rate_ceiling = speech.max_sample_rate;
    speech
        .with_max_duration_secs(MAX_DURATION_SECS)
        .with_max_decode_sample_rate(rate_ceiling)
}

fn map_ryf(err: ryf::WavError) -> WavError {
    match err {
        ryf::WavError::TooLong {
            observed_secs,
            max_secs,
        } => WavError::DurationTooLong {
            duration_secs: observed_secs,
            max_secs,
        },
        ryf::WavError::UnsupportedCodec { tag } => {
            WavError::UnsupportedFormat(format!("WAVE codec tag {tag}"))
        }
        ryf::WavError::FeatureDisabled { feature } => {
            WavError::UnsupportedFormat(format!("WAVE codec requires `{feature}`"))
        }
        ryf::WavError::Format(
            kind @ (ryf::FormatKind::UnsupportedWaveFormat | ryf::FormatKind::InvalidSize),
        ) => WavError::UnsupportedFormat(kind.to_string()),
        other => WavError::Read(other.to_string()),
    }
}

/// Read a WAV file and return mono f32 samples normalized to [-1.0, 1.0] and its sample rate.
///
/// Stereo (and multi-channel) files are downmixed by averaging channels. PCM,
/// IEEE float, G.711, G.722, GSM, and MS/IMA ADPCM WAVE containers are
/// supported. The
/// returned sample rate is whatever the WAV header declares (up to 192 kHz) —
/// this function does **not** resample.
///
/// # Guards
///
/// - If the file size exceeds 1 GiB, returns [`WavError::FileTooLarge`].
/// - If duration (clamped to the file, not a lying `data` size) exceeds
///   1 hour, returns [`WavError::DurationTooLong`] before reading samples.
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

    let opts = decode_opts();
    let file = std::fs::File::open(path)?;
    let mut src = ryf::ByteSource::from_file(file);
    let probe = ryf::probe_with(&mut src, &opts).map_err(map_ryf)?;

    if let Some(frames) = probe.declared_frames
        && probe.sample_rate > 0
    {
        let duration_secs = frames as f64 / f64::from(probe.sample_rate);
        if duration_secs > MAX_DURATION_SECS {
            return Err(WavError::DurationTooLong {
                duration_secs,
                max_secs: MAX_DURATION_SECS,
            });
        }
    }

    src.seek(SeekFrom::Start(0))
        .map_err(|e| WavError::Read(e.to_string()))?;
    match ryf::decode_with(&mut src, &opts) {
        Ok(wav) => {
            let samples = wav.channels.into_iter().next().unwrap_or_default();
            Ok((samples, wav.sample_rate))
        }
        Err(ryf::WavError::Empty) => Ok((Vec::new(), probe.sample_rate)),
        Err(e) => Err(map_ryf(e)),
    }
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
/// to use ryf. Opus is not supported (no native libopus dep); you get a
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
        // Prefer ryf for WAVE (zero media-codec deps on the hot path when the
        // file really is WAV, including telephony / RF64 containers).
        return read_wav(path);
    }

    decode::decode_with_symphonia(path)
}

#[cfg(feature = "audio-io")]
fn downmix_to_mono(interleaved: Vec<f32>, channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved;
    }
    interleaved
        .chunks(channels)
        .map(|ch| ch.iter().sum::<f32>() / channels as f32)
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(crate) fn write_pcm16(path: &Path, sample_rate: u32, channels: u16, interleaved: &[f32]) {
    let pcm = ryf::f32_to_s16le(interleaved);
    ryf::write(path, ryf::WriteSpec::s16(sample_rate, channels), &pcm).expect("write pcm16 wav");
}

#[cfg(test)]
pub(crate) fn write_pcm16_mono(path: &Path, sample_rate: u32, samples: &[f32]) {
    write_pcm16(path, sample_rate, 1, samples);
}

/// Minimal PCM WAV header with caller-controlled fields (invalid widths, lying
/// `data` sizes) that a well-formed writer will not emit.
#[cfg(test)]
pub(crate) fn crafted_pcm_wav(
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    data_len: u32,
) -> Vec<u8> {
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * u32::from(block_align);
    let mut b = Vec::new();
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36u32 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVE");
    b.extend_from_slice(b"fmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&channels.to_le_bytes());
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&byte_rate.to_le_bytes());
    b.extend_from_slice(&block_align.to_le_bytes());
    b.extend_from_slice(&bits_per_sample.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    b
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn write_sine_wav(path: &Path, sample_rate: u32, duration_secs: f32, freq_hz: f32) {
        let n = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * std::f32::consts::TAU * freq_hz).sin() * 0.5
            })
            .collect();
        write_pcm16_mono(path, sample_rate, &samples);
    }

    #[test]
    fn missing_file_error() {
        match read_wav(Path::new("/nonexistent/path/file.wav")) {
            Err(WavError::Metadata(_)) => {}
            other => panic!("expected Metadata error, got: {other:?}"),
        }
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
        let cases: &[(u32, f32)] = &[(8_000, 1.0), (22_050, 1.0), (44_100, 1.0), (48_000, 1.0)];
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
    fn read_wav_rejects_oversized_file() {
        // Sparse file: large logical length without writing real data.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.wav");
        let f = std::fs::File::create(&path).unwrap();
        f.set_len(MAX_WAV_FILE_SIZE + 1).unwrap();
        drop(f);
        match read_wav(&path) {
            Err(WavError::FileTooLarge { size, max }) => {
                assert_eq!(size, MAX_WAV_FILE_SIZE + 1);
                assert_eq!(max, MAX_WAV_FILE_SIZE);
            }
            other => panic!("expected FileTooLarge, got: {other:?}"),
        }
    }

    #[test]
    fn read_wav_rejects_declared_duration_over_limit() {
        // Sparse file whose payload length really is > 1 h of PCM16 @ 16 kHz.
        // ryf 0.7 clamps probe frames to the file, so a lying-small file no
        // longer inflates duration; the product 1 h cap still fires here.
        let data_len = (MAX_DURATION_SECS as u64 * 16_000 * 2 + 2) as u32;
        let bytes = crafted_pcm_wav(1, 16_000, 16, data_len);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long.wav");
        std::fs::write(&path, &bytes).unwrap();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(u64::from(44u32 + data_len)).unwrap();
        drop(f);
        match read_wav(&path) {
            Err(WavError::DurationTooLong {
                duration_secs,
                max_secs,
            }) => {
                assert!(duration_secs > MAX_DURATION_SECS);
                assert!((max_secs - MAX_DURATION_SECS).abs() < f64::EPSILON);
            }
            other => panic!("expected DurationTooLong, got: {other:?}"),
        }
    }

    #[test]
    fn read_wav_rejects_truncated_riff() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.wav");
        std::fs::write(&path, b"RIFF\x00\x00").unwrap();
        match read_wav(&path) {
            Err(WavError::Read(_)) => {}
            other => panic!("expected Read error, got: {other:?}"),
        }
    }

    #[test]
    fn read_wav_rejects_bits_per_sample_above_32() {
        // A 16-byte PCM fmt chunk with bits_per_sample = 40 is a valid-looking
        // RIFF header but an unsupported PCM width.
        let bytes = crafted_pcm_wav(1, 16_000, 40, 5);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("forty_bit.wav");
        std::fs::write(&path, &bytes).unwrap();
        match read_wav(&path) {
            Err(WavError::UnsupportedFormat(_)) => {}
            other => panic!("expected UnsupportedFormat, got: {other:?}"),
        }
    }

    #[test]
    fn decode_opts_does_not_clamp_frame_budget_below_rate_ceiling() {
        let opts = decode_opts();
        assert_eq!(opts.max_duration_secs, MAX_DURATION_SECS);
        assert!(
            opts.max_decode_sample_rate >= opts.max_sample_rate,
            "frame-budget rate {} < sample-rate ceiling {}",
            opts.max_decode_sample_rate,
            opts.max_sample_rate
        );
    }

    #[test]
    fn read_wav_empty_header_only_returns_empty_samples() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty16k.wav");
        write_pcm16_mono(&path, 16_000, &[]);
        let (samples, sr) = read_wav(&path).unwrap();
        assert!(samples.is_empty());
        assert_eq!(sr, 16_000);
    }

    #[cfg(feature = "audio-io")]
    #[test]
    fn load_audio_empty_non_target_wav_returns_empty_at_target_rate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty44k.wav");
        write_sine_wav(&path, 44_100, 0.0, 440.0);
        let (samples, sr) = load_audio(&path).unwrap();
        assert!(samples.is_empty());
        assert_eq!(sr, TARGET_SAMPLE_RATE);
    }

    #[test]
    fn wav_error_display_covers_all_variants() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "nope");
        let cases: Vec<(WavError, &str)> = vec![
            (WavError::Read("truncated riff".into()), "truncated riff"),
            (
                WavError::FileTooLarge {
                    size: 2_000_000_000,
                    max: MAX_WAV_FILE_SIZE,
                },
                "too large",
            ),
            (
                WavError::UnsupportedFormat("bits_per_sample 40".into()),
                "unsupported sample format",
            ),
            (
                WavError::DurationTooLong {
                    duration_secs: 7200.0,
                    max_secs: MAX_DURATION_SECS,
                },
                "too long",
            ),
            (WavError::Metadata(io_err), "metadata"),
            (
                WavError::UnsupportedSampleRate {
                    actual: 48_000,
                    expected: TARGET_SAMPLE_RATE,
                },
                "48000",
            ),
            (
                WavError::FeatureRequired {
                    path: "clip.mp3".into(),
                },
                "audio-io",
            ),
            (WavError::Decode("bad stream".into()), "decode"),
            (WavError::Resample("bad ratio".into()), "resample"),
        ];
        for (err, needle) in cases {
            let msg = format!("{err}");
            assert!(msg.contains(needle), "expected '{needle}' in: {msg}");
        }
    }

    #[test]
    fn read_wav_stereo_roundtrip_via_cursor_buffer() {
        // Smoke that the WAVE path still works when encoding in memory (mirrors
        // integration tests that build synthetic WAVs without a filesystem writer).
        let pcm = vec![0u8; 200];
        let buf = ryf::encode_s16(&pcm, 16_000).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &buf).unwrap();
        let (samples, sr) = read_wav(tmp.path()).unwrap();
        assert_eq!(sr, 16_000);
        assert_eq!(samples.len(), 100);
    }
}
