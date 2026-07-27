//! Library-mode BYO embedder path: no onnx feature, no network, no model files.

use polyvoice::{
    ClusterConfig, DiarizationConfig, Embedder, EmbedderError, EnergyVad, Pipeline, VadConfig,
    streaming::{LatencyPreset, StreamingParams, StreamingPipeline},
    types::WindowConfig,
};

/// Alternating unit axes — deterministic two-speaker clustering without a model.
struct AxisEmbedder {
    dim: usize,
    flip: std::sync::atomic::AtomicUsize,
}

impl AxisEmbedder {
    fn new(dim: usize) -> Self {
        assert!(dim >= 2);
        Self {
            dim,
            flip: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Embedder for AxisEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        if audio.is_empty() {
            return Err(EmbedderError::AudioTooShort {
                actual_secs: 0.0,
                min_secs: 0.01,
            });
        }
        let n = self.flip.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut v = vec![0.0f32; self.dim];
        if n.is_multiple_of(2) {
            v[0] = 1.0;
        } else {
            v[1] = 1.0;
        }
        Ok(v)
    }
}

fn speech_pcm(secs: f32, sr: u32) -> Vec<f32> {
    let n = (secs * sr as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
        })
        .collect()
}

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
    let result = Pipeline::new(config, vad_config)
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
