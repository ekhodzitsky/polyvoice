use polyvoice::pipeline::Pipeline;
use polyvoice::{DiarizationConfig, DummyExtractor, EnergyVad, VadConfig};

#[test]
fn test_pipeline_basic() {
    let config = DiarizationConfig {
        window_secs: 0.5,
        hop_secs: 0.25,
        min_speech_secs: 0.1,
        ..Default::default()
    };
    let vad_config = VadConfig {
        threshold: 0.1,
        ..Default::default()
    };
    let extractor = DummyExtractor::new(256);
    let pipeline = Pipeline::new(config, vad_config);

    let samples: Vec<f32> = (0..16000 * 5)
        .map(|i| ((i as f32 / 16000.0) * std::f32::consts::TAU * 440.0).sin() * 0.5)
        .collect();

    let result = pipeline
        .run(&samples, &extractor, &mut EnergyVad::new(-60.0, 16000, 512))
        .unwrap();
    assert!(!result.segments.is_empty());
    assert!(!result.turns.is_empty());
    assert!(result.num_speakers >= 1);
}

#[test]
fn test_pipeline_silence() {
    let config = DiarizationConfig::default();
    let vad_config = VadConfig::default();
    let extractor = DummyExtractor::new(256);
    let mut vad = EnergyVad::new(-20.0, 16000, 512);
    let pipeline = Pipeline::new(config, vad_config);

    let samples = vec![0.0f32; 16000 * 3];
    let result = pipeline.run(&samples, &extractor, &mut vad).unwrap();
    assert!(result.turns.is_empty());
    assert_eq!(result.num_speakers, 0);
}

#[test]
fn test_pipeline_from_wav() {
    use std::io::Cursor;

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
        for i in 0..16000 * 3 {
            let sample = ((i as f32 / 16000.0) * std::f32::consts::TAU * 300.0).sin();
            writer.write_sample((sample * 16000.0) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &buf).unwrap();

    let config = DiarizationConfig {
        window_secs: 0.5,
        hop_secs: 0.25,
        min_speech_secs: 0.1,
        ..Default::default()
    };
    let vad_config = VadConfig {
        threshold: 0.1,
        ..Default::default()
    };
    let extractor = DummyExtractor::new(256);
    let pipeline = Pipeline::new(config, vad_config);
    let mut vad = polyvoice::EnergyVad::new(-60.0, 16000, 512);

    let result = pipeline
        .run_from_wav(tmp.path(), &extractor, &mut vad)
        .unwrap();
    assert!(!result.turns.is_empty());
}
