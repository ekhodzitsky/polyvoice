use polyvoice::wav;
use std::io::Cursor;

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
