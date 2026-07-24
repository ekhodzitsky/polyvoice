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
