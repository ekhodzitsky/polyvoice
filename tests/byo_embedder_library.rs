//! Library-mode BYO embedder path: no onnx feature, no network, no model files.

use polyvoice::{
    ClusterConfig, DiarizationConfig, Embedder, EmbedderError, EnergyVad, VadConfig,
    pipeline::LegacyPipeline,
    streaming::{LatencyPreset, StreamingParams, StreamingPipeline},
    types::WindowConfig,
};

mod common;
use common::{AxisEmbedder, speech_pcm};

fn short_window_config() -> DiarizationConfig {
    DiarizationConfig {
        cluster: ClusterConfig {
            threshold: 0.45,
            ..ClusterConfig::default()
        },
        window: WindowConfig {
            window_secs: 0.5,
            hop_secs: 0.25,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn offline_custom_embedder_finds_multiple_speakers() {
    let sr = 16_000u32;
    let samples = speech_pcm(4.0, sr);
    let config = short_window_config();
    let vad_config = VadConfig::default();
    let mut vad = EnergyVad::new(-40.0, sr, vad_config.frame_size);
    let result = LegacyPipeline::new(config, vad_config)
        .run(&samples, &AxisEmbedder::new(32), &mut vad)
        .expect("pipeline");
    assert!(
        result.num_speakers >= 2,
        "expected >= 2 speakers from alternating axes, got {}",
        result.num_speakers
    );
    assert!(!result.turns.is_empty());
}

#[test]
fn streaming_custom_embedder_emits_turns_after_flush() {
    let sr = 16_000u32;
    let samples = speech_pcm(3.0, sr);
    let config = short_window_config();
    let vad_config = VadConfig::default();
    let params = StreamingParams {
        window_secs: config.window.window_secs,
        hop_secs: config.window.hop_secs,
        right_context_secs: 0.0,
        speaker_cache_cap: config.cluster.max_speakers.max(1),
        min_hits_to_stable: LatencyPreset::Realtime.params().min_hits_to_stable,
        prefer_current_margin: LatencyPreset::Realtime.params().prefer_current_margin,
        match_threshold: config.cluster.threshold,
    };
    let mut stream = StreamingPipeline::with_params(
        EnergyVad::new(-40.0, sr, vad_config.frame_size),
        AxisEmbedder::new(32),
        config,
        vad_config,
        params,
    )
    .expect("stream init");
    for chunk in samples.chunks(1600) {
        stream.feed(chunk).expect("feed");
    }
    stream.flush().expect("flush");
    assert!(
        !stream.turns().is_empty(),
        "expected non-empty turns after 3s speech + flush"
    );
    assert!(
        stream.num_speakers() >= 1,
        "expected at least one speaker after flush"
    );
}

#[test]
fn empty_audio_returns_embedder_error() {
    let err = AxisEmbedder::new(32)
        .embed(&[])
        .expect_err("empty audio must fail");
    assert!(matches!(err, EmbedderError::AudioTooShort { .. }));
}
