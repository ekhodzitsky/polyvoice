#![allow(clippy::unwrap_used)]
use polyvoice::wav;
use std::io::Cursor;
use std::path::Path;

fn write_tone_wav(path: &Path, sample_rate: u32, n_samples: usize) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        let s = (t * std::f32::consts::TAU * 440.0).sin();
        writer.write_sample((s * 16000.0) as i16).unwrap();
    }
    writer.finalize().unwrap();
}

#[test]
fn test_read_wav_mono_16bit() {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Vec::new();
    {
        let cursor = Cursor::new(&mut buf);
        let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
        for i in 0..16000 {
            let sample = ((i as f32 / 16000.0) * std::f32::consts::TAU * 440.0).sin();
            writer.write_sample((sample * 32767.0) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &buf).unwrap();

    let (samples, sample_rate) = wav::read_wav(tmp.path()).unwrap();
    assert_eq!(sample_rate, 16000);
    assert_eq!(samples.len(), 16000);
    assert!(samples.iter().all(|s| *s >= -1.0 && *s <= 1.0));
}

#[test]
fn test_read_wav_stereo_downmix() {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Vec::new();
    {
        let cursor = Cursor::new(&mut buf);
        let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
        for _ in 0..8000 {
            writer.write_sample(16383i16).unwrap(); // left
            writer.write_sample(-16383i16).unwrap(); // right
        }
        writer.finalize().unwrap();
    }

    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &buf).unwrap();

    let (samples, sample_rate) = wav::read_wav(tmp.path()).unwrap();
    assert_eq!(sample_rate, 16000);
    assert_eq!(samples.len(), 8000);
    assert!(samples.iter().all(|s| s.abs() < 0.01));
}

#[test]
fn test_read_wav_not_found() {
    let result = wav::read_wav(std::path::Path::new("/nonexistent/audio.wav"));
    assert!(result.is_err());
}

#[test]
fn test_load_audio_16k_wav() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ok.wav");
    write_tone_wav(&path, 16_000, 8_000);
    let (samples, sr) = wav::load_audio(&path).unwrap();
    assert_eq!(sr, wav::TARGET_SAMPLE_RATE);
    assert_eq!(samples.len(), 8_000);
}

#[test]
fn test_load_audio_wrong_rate_or_format_without_audio_io() {
    #[cfg(not(feature = "audio-io"))]
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("48k.wav");
        write_tone_wav(&path, 48_000, 4_800);
        let err = wav::load_audio(&path).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("audio-io"), "hint missing: {msg}");

        let err2 = wav::load_audio(Path::new("clip.mp3")).unwrap_err();
        assert!(format!("{err2}").contains("audio-io"));
    }
}

#[cfg(feature = "audio-io")]
#[test]
fn test_load_audio_resamples_common_rates() {
    let dir = tempfile::tempdir().unwrap();
    for rate in [8_000_u32, 22_050, 44_100, 48_000] {
        let path = dir.path().join(format!("{rate}.wav"));
        // 0.5 seconds at source rate
        let n = (rate as f64 * 0.5) as usize;
        write_tone_wav(&path, rate, n);
        let (samples, sr) = wav::load_audio(&path).unwrap();
        assert_eq!(sr, 16_000);
        let secs = samples.len() as f64 / 16_000.0;
        assert!(
            (secs - 0.5).abs() < 0.02,
            "rate={rate}: duration {secs:.4}s"
        );
    }
}

/// Decode checked-in flac/mp3 fixtures (any rate) → 16 kHz mono.
#[cfg(feature = "audio-io")]
#[test]
fn test_load_audio_flac_and_mp3_fixtures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/audio-io");
    let fixtures = [("tone_440_44k1.flac", 0.25), ("tone_440_48k.mp3", 0.25)];
    for (name, expected_secs) in fixtures {
        let path = root.join(name);
        assert!(path.is_file(), "missing fixture {}", path.display());
        let (samples, sr) = wav::load_audio(&path).unwrap_or_else(|e| {
            panic!("load_audio({}) failed: {e}", path.display());
        });
        assert_eq!(sr, 16_000, "{name}");
        let secs = samples.len() as f64 / 16_000.0;
        // MP3 framing can pad a few ms; allow a looser bound than WAV.
        assert!(
            (secs - expected_secs).abs() < 0.08,
            "{name}: duration {secs:.4}s (expected ~{expected_secs})"
        );
        let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32;
        assert!(energy > 1e-6, "{name}: near-silent");
    }
}
