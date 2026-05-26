#![allow(clippy::unwrap_used)]
//! Mock-based E2E integration tests for the pipeline.
//!
//! These tests run the full diarization pipeline without any ONNX models,
//! using `DummyExtractor` + `EnergyVad` on synthetic audio.  They execute in
//! every PR (no `#[ignore]`).
//!
//! Chaos tests (empty input, all silence) live in `tests/chaos_test.rs` and are
//! not duplicated here.

use polyvoice::{
    embedding::DummyExtractor,
    pipeline::Pipeline,
    types::DiarizationConfig,
    vad::{EnergyVad, VadConfig, segment_speech},
};

fn default_config() -> DiarizationConfig {
    DiarizationConfig::default()
}

fn default_vad_config() -> VadConfig {
    VadConfig::default()
}

/// Generate a sine wave of the given frequency, duration and sample rate.
fn sine_wave(freq: f32, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let n = (duration_secs * sample_rate as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5f32 * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// Generate silence.
fn silence(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let n = (duration_secs * sample_rate as f32) as usize;
    vec![0.0f32; n]
}

// ---------------------------------------------------------------------------
// 1. Multi-region synthetic audio (two tones separated by silence)
// ---------------------------------------------------------------------------

#[test]
fn pipeline_mock_multi_region_synthetic_audio() {
    let sr = 16000;
    // Speaker A: 300 Hz for 2 seconds
    let speaker_a = sine_wave(300.0, 2.0, sr);
    // 1 second silence
    let gap = silence(1.0, sr);
    // Speaker B: 800 Hz for 2 seconds
    let speaker_b = sine_wave(800.0, 2.0, sr);

    let mut samples = Vec::new();
    samples.extend_from_slice(&speaker_a);
    samples.extend_from_slice(&gap);
    samples.extend_from_slice(&speaker_b);

    let pipeline = Pipeline::new(default_config(), default_vad_config());
    let extractor = DummyExtractor::new(256);
    let mut vad = EnergyVad::new(-40.0, sr, 512);

    let result = pipeline.run(&samples, &extractor, &mut vad).unwrap();

    // Basic sanity checks — DummyExtractor is pseudo-random, so we cannot
    // assert an exact speaker count.  We only verify the pipeline runs
    // without panic and produces structurally valid output.
    assert!(
        result.num_speakers >= 1,
        "expected at least 1 speaker, got {}",
        result.num_speakers
    );
    assert!(
        !result.turns.is_empty(),
        "expected non-empty turns for multi-region synthetic audio"
    );
    for turn in &result.turns {
        assert!(
            turn.time.start < turn.time.end,
            "turn must have start < end: {:?}",
            turn.time
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Single sustained speaker
// ---------------------------------------------------------------------------

#[test]
fn pipeline_mock_single_speaker_sustained() {
    let sr = 16000;
    let samples = sine_wave(400.0, 5.0, sr);

    let pipeline = Pipeline::new(default_config(), default_vad_config());
    let extractor = DummyExtractor::new(256);
    let mut vad = EnergyVad::new(-40.0, sr, 512);

    let result = pipeline.run(&samples, &extractor, &mut vad).unwrap();

    assert!(
        result.num_speakers >= 1,
        "expected at least 1 speaker for sustained audio"
    );
    assert!(
        !result.turns.is_empty(),
        "expected non-empty turns for sustained audio"
    );
    for turn in &result.turns {
        assert!(
            turn.time.start < turn.time.end,
            "turn must have start < end"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. segment_speech via EnergyVad on synthetic multi-region audio
// ---------------------------------------------------------------------------

#[test]
fn segment_speech_mock_detects_speech_regions() {
    let sr = 16000;
    let mut samples = Vec::new();
    samples.extend_from_slice(&sine_wave(300.0, 2.0, sr));
    samples.extend_from_slice(&silence(1.0, sr));
    samples.extend_from_slice(&sine_wave(500.0, 2.0, sr));

    let mut vad = EnergyVad::new(-40.0, sr, 512);
    let config = DiarizationConfig::default();
    let vad_config = VadConfig::default();

    let segs = segment_speech(&mut vad, &samples, &config, &vad_config).unwrap();

    // EnergyVad on loud sine waves should detect at least one speech region.
    // (The exact count depends on energy threshold and silence duration.)
    assert!(
        !segs.is_empty(),
        "expected at least one speech region for loud synthetic audio"
    );
    for (start, end) in &segs {
        assert!(
            start < end,
            "segment must have start < end: {} < {}",
            start,
            end
        );
    }
}
