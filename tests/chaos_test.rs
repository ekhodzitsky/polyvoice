#![allow(clippy::unwrap_used)]
//! Chaos / malformed-input tests for the legacy polyvoice pipeline.
//!
//! These tests verify that the pipeline, VAD, AHC, and WAV I/O handle
//! edge-case or malicious inputs gracefully — returning `Err` or empty
//! results rather than panicking.
//!
//! No ONNX models are required; all tests use `DummyExtractor` and `EnergyVad`.

use polyvoice::{
    ahc::agglomerative_cluster,
    embedder::DummyExtractor,
    pipeline::LegacyPipeline,
    types::{DiarizationConfig, SampleRate},
    vad::{EnergyVad, VadConfig, VadError, VoiceActivityDetector, segment_speech},
    wav::{WavError, read_wav},
};
use std::io::Write;
use tempfile::NamedTempFile;

fn default_config() -> DiarizationConfig {
    DiarizationConfig::default()
}

fn default_vad_config() -> VadConfig {
    VadConfig::default()
}

// ---------------------------------------------------------------------------
// 1. Empty samples
// ---------------------------------------------------------------------------

#[test]
fn segment_speech_with_empty_samples_returns_empty() {
    let mut vad = EnergyVad::new(-40.0, 16000, 512);
    let segments = segment_speech(&mut vad, &[], &default_config(), &default_vad_config()).unwrap();
    assert!(segments.is_empty(), "expected no segments for empty input");
}

#[test]
fn pipeline_run_with_empty_samples_returns_ok_zero_speakers() {
    let pipeline = LegacyPipeline::new(default_config(), default_vad_config());
    let extractor = DummyExtractor::new(256);
    let mut vad = EnergyVad::new(-40.0, 16000, 512);
    let result = pipeline.run(&[], &extractor, &mut vad).unwrap();
    assert_eq!(result.num_speakers, 0);
    assert!(result.segments.is_empty());
    assert!(result.turns.is_empty());
}

// ---------------------------------------------------------------------------
// 2. Very short audio (100 samples @ 16 kHz = 6.25 ms)
// ---------------------------------------------------------------------------

#[test]
fn pipeline_run_with_very_short_audio_returns_zero_speakers() {
    let pipeline = LegacyPipeline::new(default_config(), default_vad_config());
    let extractor = DummyExtractor::new(256);
    let mut vad = EnergyVad::new(-40.0, 16000, 512);
    let samples = vec![0.0f32; 100];
    let result = pipeline.run(&samples, &extractor, &mut vad).unwrap();
    assert_eq!(result.num_speakers, 0);
    assert!(result.segments.is_empty());
    assert!(result.turns.is_empty());
}

// ---------------------------------------------------------------------------
// 3. All-silence audio (1 second of zeros)
// ---------------------------------------------------------------------------

#[test]
fn pipeline_run_with_all_silence_returns_zero_speakers() {
    let pipeline = LegacyPipeline::new(default_config(), default_vad_config());
    let extractor = DummyExtractor::new(256);
    let mut vad = EnergyVad::new(-40.0, 16000, 512);
    let samples = vec![0.0f32; 16000];
    let result = pipeline.run(&samples, &extractor, &mut vad).unwrap();
    assert_eq!(result.num_speakers, 0);
    assert!(result.segments.is_empty());
    assert!(result.turns.is_empty());
}

// ---------------------------------------------------------------------------
// 4. Very loud audio (1 second of 1.0 amplitude square wave)
// ---------------------------------------------------------------------------

#[test]
fn pipeline_run_with_very_loud_audio_returns_at_least_one_speaker() {
    let pipeline = LegacyPipeline::new(default_config(), default_vad_config());
    let extractor = DummyExtractor::new(256);
    let mut vad = EnergyVad::new(-40.0, 16000, 512);
    let samples = vec![1.0f32; 16000];
    let result = pipeline.run(&samples, &extractor, &mut vad).unwrap();
    assert!(
        result.num_speakers >= 1,
        "expected at least 1 speaker for loud audio, got {}",
        result.num_speakers
    );
}

// ---------------------------------------------------------------------------
// 5. Corrupted / truncated WAV file
// ---------------------------------------------------------------------------

#[test]
fn read_wav_on_truncated_file_returns_error() {
    let mut temp = NamedTempFile::new().expect("failed to create temp file");
    temp.write_all(b"RIFFxxxxxx")
        .expect("failed to write temp file");
    let result = read_wav(temp.path());
    assert!(
        result.is_err(),
        "expected Err for truncated WAV, got {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// 6. Crafted WAV header declaring huge duration
// ---------------------------------------------------------------------------

#[test]
fn read_wav_on_oversized_duration_header_returns_error() {
    let mut temp = NamedTempFile::new().expect("failed to create temp file");

    // Build a minimal WAV header declaring 2 hours of 16-bit mono @ 16 kHz
    // but only write a few bytes of actual data.
    let sample_rate = 16000u32;
    let channels = 1u16;
    let bits_per_sample = 16u16;
    let bytes_per_sample = (bits_per_sample / 8) as u32;
    let block_align = channels * bits_per_sample / 8;
    let byte_rate = sample_rate * bytes_per_sample;

    let declared_samples = 2u64 * 3600 * sample_rate as u64;
    let data_chunk_size = (declared_samples * bytes_per_sample as u64) as u32;
    let riff_chunk_size = 36u32 + data_chunk_size;

    let mut header = Vec::new();
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&riff_chunk_size.to_le_bytes());
    header.extend_from_slice(b"WAVE");
    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&channels.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    header.extend_from_slice(&byte_rate.to_le_bytes());
    header.extend_from_slice(&block_align.to_le_bytes());
    header.extend_from_slice(&bits_per_sample.to_le_bytes());
    header.extend_from_slice(b"data");
    header.extend_from_slice(&data_chunk_size.to_le_bytes());
    header.extend_from_slice(&[0u8; 4]);

    temp.write_all(&header).expect("failed to write temp file");

    let result = read_wav(temp.path());
    match result {
        Err(WavError::DurationTooLong {
            duration_secs,
            max_secs,
        }) => {
            assert!(
                duration_secs > 3600.0,
                "expected duration > 3600, got {}",
                duration_secs
            );
            assert!(
                (max_secs - 3600.0).abs() < 0.01,
                "expected max_secs = 3600.0, got {}",
                max_secs
            );
        }
        other => panic!("expected DurationTooLong error, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 7. AHC clustering with empty embeddings
// ---------------------------------------------------------------------------

#[test]
fn ahc_cluster_empty_embeddings_returns_empty() {
    let labels = agglomerative_cluster(&[], 0.45);
    assert!(
        labels.is_empty(),
        "expected empty labels for empty embeddings"
    );
}

// ---------------------------------------------------------------------------
// 7. AHC clustering with single embedding
// ---------------------------------------------------------------------------

#[test]
fn ahc_cluster_single_embedding_returns_zero() {
    let embeddings = vec![vec![1.0f32, 0.0, 0.0]];
    let labels = agglomerative_cluster(&embeddings, 0.45);
    assert_eq!(labels, vec![0]);
}

// ---------------------------------------------------------------------------
// 8. VAD with wrong chunk size
// ---------------------------------------------------------------------------

#[test]
fn vad_process_with_invalid_chunk_size_returns_err() {
    let mut vad = EnergyVad::new(-40.0, 16000, 512);
    let samples = vec![0.0f32; 513];
    let result = vad.process(&samples);
    match result {
        Err(VadError::InvalidChunkSize { expected, got }) => {
            assert_eq!(expected, 512);
            assert_eq!(got, 513);
        }
        other => panic!("expected InvalidChunkSize error, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 9. Pipeline with mismatched sample rate config
// ---------------------------------------------------------------------------

#[test]
fn pipeline_mismatched_sample_rate_does_not_crash() {
    let mut config = default_config();
    config.window.sample_rate = SampleRate::new(8000).unwrap();
    let pipeline = LegacyPipeline::new(config, default_vad_config());
    let extractor = DummyExtractor::new(256);
    let mut vad = EnergyVad::new(-40.0, 16000, 512);
    // Pass 16000 samples while config claims 8000 Hz — should not panic.
    let samples = vec![0.0f32; 16000];
    let _result = pipeline.run(&samples, &extractor, &mut vad);
}

// ---------------------------------------------------------------------------
// 10. Audio exceeding max_duration_secs is rejected
// ---------------------------------------------------------------------------

#[test]
fn pipeline_run_with_audio_too_long_returns_audio_too_long_error() {
    let mut config = default_config();
    config.max_duration_secs = 1.0;
    let pipeline = LegacyPipeline::new(config, default_vad_config());
    let extractor = DummyExtractor::new(256);
    let mut vad = EnergyVad::new(-40.0, 16000, 512);
    // 2 seconds at 16 kHz = 32000 samples
    let samples = vec![0.0f32; 32000];
    let result = pipeline.run(&samples, &extractor, &mut vad);
    match result {
        Err(polyvoice::pipeline::LegacyPipelineError::AudioTooLong {
            actual_secs,
            max_secs,
        }) => {
            assert!(
                (actual_secs - 2.0).abs() < 0.01,
                "expected actual_secs ≈ 2.0, got {}",
                actual_secs
            );
            assert!(
                (max_secs - 1.0).abs() < 0.01,
                "expected max_secs = 1.0, got {}",
                max_secs
            );
        }
        other => panic!("expected AudioTooLong error, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 11. Property test: random samples of various lengths never panic
// ---------------------------------------------------------------------------

use proptest::prelude::*;

proptest! {
    #[test]
    fn pipeline_random_samples_never_panic(samples in prop::collection::vec(-1.0f32..=1.0f32, 0..=32000)) {
        let pipeline = LegacyPipeline::new(default_config(), default_vad_config());
        let extractor = DummyExtractor::new(256);
        let mut vad = EnergyVad::new(-40.0, 16000, 512);
        // We only care that this does not panic; Ok/Err are both acceptable.
        let _ = pipeline.run(&samples, &extractor, &mut vad);
    }
}
