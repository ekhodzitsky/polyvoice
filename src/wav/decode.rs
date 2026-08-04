//! Multi-format decode via symphonia (feature `audio-io` only).
//!
//! Unmodified MPL-2.0 dependency — do not patch or vendor sources. See NOTICE.

use super::{MAX_DURATION_SECS, MAX_WAV_FILE_SIZE, WavError, downmix_to_mono};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Decode any symphonia-supported file to mono f32 samples + native sample rate.
pub(super) fn decode_with_symphonia(path: &Path) -> Result<(Vec<f32>, u32), WavError> {
    let metadata = std::fs::metadata(path).map_err(WavError::Metadata)?;
    let file_size = metadata.len();
    if file_size > MAX_WAV_FILE_SIZE {
        return Err(WavError::FileTooLarge {
            size: file_size,
            max: MAX_WAV_FILE_SIZE,
        });
    }

    let file = File::open(path).map_err(WavError::Metadata)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| {
            WavError::Decode(format!(
                "unsupported or unreadable format for '{}': {e} \
                 (supported with audio-io: mp3, flac, ogg/vorbis, m4a/aac, wav, aiff, caf, mkv; \
                 Opus is not supported)",
                path.display()
            ))
        })?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| WavError::Decode("no decodable audio track found".into()))?
        .clone();

    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| WavError::Decode("track has no audio codec parameters".into()))?;

    let sample_rate = audio_params
        .sample_rate
        .ok_or_else(|| WavError::Decode("codec did not report a sample rate".into()))?;
    if sample_rate == 0 {
        return Err(WavError::Decode("codec reported sample rate 0".into()));
    }

    let channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1);

    // Prefer duration from container/track when available.
    if let Some(n_frames) = track.num_frames {
        let duration_secs = n_frames as f64 / sample_rate as f64;
        if duration_secs > MAX_DURATION_SECS {
            return Err(WavError::DurationTooLong {
                duration_secs,
                max_secs: MAX_DURATION_SECS,
            });
        }
    }

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|e| {
            WavError::Decode(format!(
                "unsupported codec for '{}': {e} (Opus/libopus is not bundled)",
                path.display()
            ))
        })?;

    let track_id = track.id;
    let mut mono_out: Vec<f32> = Vec::new();
    let mut packet_buf: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymphoniaError::ResetRequired) => {
                // Chained OGG etc. — treat as end of the primary stream for
                // offline diarization rather than restarting decoders.
                break;
            }
            Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::DecodeError(_)) => {
                // Soft errors: skip.
                continue;
            }
            Err(e) => return Err(WavError::Decode(format!("demux: {e}"))),
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                packet_buf.resize(audio_buf.samples_interleaved(), f32::MID);
                audio_buf.copy_to_slice_interleaved(&mut packet_buf);
                let chunk_mono = downmix_to_mono(std::mem::take(&mut packet_buf), channels);
                mono_out.extend_from_slice(&chunk_mono);
            }
            Err(SymphoniaError::DecodeError(_)) | Err(SymphoniaError::IoError(_)) => continue,
            Err(e) => return Err(WavError::Decode(format!("decode: {e}"))),
        }

        let secs_so_far = mono_out.len() as f64 / sample_rate as f64;
        if secs_so_far > MAX_DURATION_SECS {
            return Err(WavError::DurationTooLong {
                duration_secs: secs_so_far,
                max_secs: MAX_DURATION_SECS,
            });
        }
    }

    Ok((mono_out, sample_rate))
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn write_sine_wav(path: &Path, sample_rate: u32, channels: u16, secs: f32) {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let n = (sample_rate as f32 * secs) as usize;
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..n {
            let t = i as f32 / sample_rate as f32;
            let sample = (t * std::f32::consts::TAU * 440.0).sin();
            let v = (sample * 32767.0 * 0.5) as i16;
            for _ in 0..channels {
                writer.write_sample(v).unwrap();
            }
        }
        writer.finalize().unwrap();
    }

    /// Minimal PCM WAV header with a caller-controlled declared data length.
    fn crafted_pcm_wav(sample_rate: u32, bits_per_sample: u16, data_len: u32) -> Vec<u8> {
        let channels: u16 = 1;
        let block_align = channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * u32::from(block_align);
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36u32 + data_len).to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&sample_rate.to_le_bytes());
        b.extend_from_slice(&byte_rate.to_le_bytes());
        b.extend_from_slice(&block_align.to_le_bytes());
        b.extend_from_slice(&bits_per_sample.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&data_len.to_le_bytes());
        b
    }

    #[test]
    fn missing_file_is_metadata_error() {
        match decode_with_symphonia(Path::new("/nonexistent/dir/clip.mp3")) {
            Err(WavError::Metadata(_)) => {}
            other => panic!("expected Metadata error, got: {other:?}"),
        }
    }

    #[test]
    fn oversized_file_rejected_before_open() {
        // Sparse file: large logical length without writing real data.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.mp3");
        let f = std::fs::File::create(&path).unwrap();
        f.set_len(MAX_WAV_FILE_SIZE + 1).unwrap();
        drop(f);
        match decode_with_symphonia(&path) {
            Err(WavError::FileTooLarge { size, max }) => {
                assert_eq!(size, MAX_WAV_FILE_SIZE + 1);
                assert_eq!(max, MAX_WAV_FILE_SIZE);
            }
            other => panic!("expected FileTooLarge, got: {other:?}"),
        }
    }

    #[test]
    fn garbage_bytes_fail_probe_with_named_error() {
        let dir = tempfile::tempdir().unwrap();
        for ext in ["mp3", "flac", "ogg", "m4a"] {
            let path = dir.path().join(format!("junk.{ext}"));
            std::fs::write(&path, b"this is not audio data at all, just text").unwrap();
            match decode_with_symphonia(&path) {
                Err(WavError::Decode(msg)) => {
                    assert!(
                        msg.contains("unsupported or unreadable format"),
                        "ext={ext}: {msg}"
                    );
                }
                other => panic!("ext={ext}: expected Decode error, got: {other:?}"),
            }
        }
    }

    #[test]
    fn empty_file_fails_probe() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.mp3");
        std::fs::write(&path, b"").unwrap();
        match decode_with_symphonia(&path) {
            Err(WavError::Decode(msg)) => {
                assert!(
                    msg.contains("unsupported or unreadable format"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Decode error, got: {other:?}"),
        }
    }

    #[test]
    fn wav_decodes_through_symphonia_downmixing_stereo() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tone.wav");
        write_sine_wav(&path, 44_100, 2, 0.25);
        let (samples, sr) = decode_with_symphonia(&path).unwrap();
        assert_eq!(sr, 44_100);
        let expected = (44_100.0f32 * 0.25) as usize;
        let diff = samples.len().abs_diff(expected);
        assert!(
            diff <= 2,
            "got {} mono samples, expected ~{expected}",
            samples.len()
        );
        let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32;
        assert!(energy > 1e-6, "near-silent after decode");
    }

    #[test]
    fn wav_without_extension_sniffs_content() {
        // No extension → hint stays empty; probe must fall back to content sniffing.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("noext");
        write_sine_wav(&path, 16_000, 1, 0.1);
        let (samples, sr) = decode_with_symphonia(&path).unwrap();
        assert_eq!(sr, 16_000);
        assert_eq!(samples.len(), 1_600);
    }

    #[test]
    fn declared_duration_over_limit_rejected() {
        // Header declares far more frames than the file holds; the guard must
        // fire on the container-reported duration before decoding packets.
        let data_len = (MAX_DURATION_SECS as u64 * 16_000 * 2 + 2) as u32;
        let bytes = crafted_pcm_wav(16_000, 16, data_len);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long.wav");
        std::fs::write(&path, &bytes).unwrap();
        match decode_with_symphonia(&path) {
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
}
