use super::*;
use crate::embedder::DummyExtractor;
use crate::types::SpeakerId;
use crate::{EnergyVad, VadConfig};

fn default_config() -> DiarizationConfig {
    DiarizationConfig::default()
}

fn default_vad_config() -> VadConfig {
    VadConfig::default()
}

fn pipeline() -> StreamingPipeline<EnergyVad, DummyExtractor> {
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    StreamingPipeline::new(vad, extractor, default_config(), default_vad_config()).unwrap()
}

fn pipeline_preset(preset: LatencyPreset) -> StreamingPipeline<EnergyVad, DummyExtractor> {
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    StreamingPipeline::with_latency_preset(vad, extractor, preset, default_vad_config())
        .unwrap()
}

/// Loud samples that should trigger speech detection.
fn loud_samples(seconds: f32) -> Vec<f32> {
    let n = (seconds * 16000.0) as usize;
    vec![0.5f32; n]
}

/// Silent samples that should not trigger speech.
fn silent_samples(seconds: f32) -> Vec<f32> {
    let n = (seconds * 16000.0) as usize;
    vec![0.0f32; n]
}

#[test]
fn streaming_pipeline_new_is_empty() {
    // Empty case: a fresh pipeline (no feed yet) has empty cumulative history.
    let p = pipeline();
    assert_eq!(p.num_speakers(), 0);
    assert!(p.turns().is_empty());
    assert_eq!(p.cache_len(), 0);
    assert!(p.speaker_cache_cap() >= 1);
}

#[test]
fn feed_silence_returns_no_turns() {
    // No-speech case: silence emits nothing and leaves the cumulative
    // history empty (the populated path is covered below).
    let mut p = pipeline();
    let turns = p.feed(&silent_samples(2.0)).unwrap();
    assert!(turns.is_empty());
    assert!(p.turns().is_empty());
}

#[test]
fn feed_loud_audio_returns_at_least_one_turn() {
    let mut p = pipeline();
    // 5 seconds of loud audio guarantees at least one full window (1.5 s)
    let turns = p.feed(&loud_samples(5.0)).unwrap();
    assert!(
        !turns.is_empty(),
        "expected at least one turn for 5 s of speech"
    );
}

#[test]
fn feed_rejects_vad_with_mismatched_native_frame() {
    // Native VAD frame 256 < VadConfig::frame_size 512: each pipeline frame
    // yields two probabilities, which would silently stretch every
    // timestamp — the pipeline must fail loudly on the first frame instead.
    let vad = EnergyVad::new(-40.0, 16000, 256);
    let extractor = DummyExtractor::new(256);
    let mut p =
        StreamingPipeline::new(vad, extractor, default_config(), default_vad_config()).unwrap();
    let err = p.feed(&loud_samples(1.0)).unwrap_err();
    assert!(!err.is_resource_exhausted());
    match err {
        StreamingError::VadFrameMismatch { frame_samples, got } => {
            assert_eq!(frame_samples, 512);
            assert_eq!(got, 2);
        }
        other => panic!("expected VadFrameMismatch, got {other:?}"),
    }
}

#[test]
fn flush_after_speech_emits_remaining_turn() {
    let mut p = pipeline();
    // Feed just under one window — no turn emitted yet.
    let _ = p.feed(&loud_samples(1.0)).unwrap();
    let turns = p.flush().unwrap();
    assert!(
        !turns.is_empty(),
        "flush should emit the trailing partial window"
    );
}

#[test]
fn turns_are_monotonically_ordered() {
    let mut p = pipeline();
    let mut emitted: Vec<SpeakerTurn> = Vec::new();
    emitted.extend(p.feed(&loud_samples(5.0)).unwrap());
    emitted.extend(p.flush().unwrap());
    // turns() must expose the cumulative history, not an always-empty slice
    // Regression: assert it is populated and equals what feed()/flush() emitted
    // BEFORE the ordering loop, so the loop is never vacuous.
    assert!(
        !p.turns().is_empty(),
        "turns() must be populated after feeding speech"
    );
    assert_eq!(
        p.turns(),
        emitted.as_slice(),
        "turns() must equal the concatenation of feed()/flush() returns"
    );
    let turns = p.turns();
    for i in 1..turns.len() {
        assert!(
            turns[i].time.start >= turns[i - 1].time.start,
            "turns must be monotonically ordered"
        );
    }
}

#[test]
fn turns_accumulates_across_feed_and_flush() {
    let mut p = pipeline();
    let mut emitted: Vec<SpeakerTurn> = Vec::new();
    emitted.extend(p.feed(&loud_samples(3.0)).unwrap());
    emitted.extend(p.feed(&loud_samples(3.0)).unwrap());
    emitted.extend(p.flush().unwrap());
    assert!(
        !emitted.is_empty(),
        "expected turns across two feeds plus a flush"
    );
    assert_eq!(
        p.turns(),
        emitted.as_slice(),
        "turns() must accumulate every feed()/flush() return in order"
    );
}

#[test]
fn balanced_preset_matches_default_window() {
    let p = pipeline_preset(LatencyPreset::Balanced);
    let d = DiarizationConfig::default();
    assert!((p.params().window_secs - d.window.window_secs).abs() < 1e-6);
    assert!((p.params().hop_secs - d.window.hop_secs).abs() < 1e-6);
    assert_eq!(p.latency_preset(), Some(LatencyPreset::Balanced));
}

#[test]
fn realtime_preset_has_shorter_window() {
    let p = pipeline_preset(LatencyPreset::Realtime);
    assert!((p.params().window_secs - 1.0).abs() < 1e-6);
    assert_eq!(p.speaker_cache_cap(), 16);
}

#[test]
fn cache_never_exceeds_cap_under_long_feed() {
    // Tight cap: force overflow merge path under continuous speech.
    let params = StreamingParams {
        window_secs: 1.0,
        hop_secs: 0.5,
        right_context_secs: 0.0,
        speaker_cache_cap: 2,
        min_hits_to_stable: 2,
        prefer_current_margin: 0.05,
        match_threshold: 0.45,
    };
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    let mut p = StreamingPipeline::with_params(
        vad,
        extractor,
        DiarizationConfig::default(),
        default_vad_config(),
        params,
    )
    .unwrap();
    let _ = p.feed(&loud_samples(20.0)).unwrap();
    let _ = p.flush().unwrap();
    assert!(
        p.cache_len() <= p.speaker_cache_cap(),
        "cache len {} > cap {}",
        p.cache_len(),
        p.speaker_cache_cap()
    );
    assert!(p.cache_len() <= 2);
}

#[test]
fn with_params_rejects_non_positive_window_secs() {
    let params = StreamingParams {
        window_secs: 0.0,
        ..LatencyPreset::Balanced.params()
    };
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    let res = StreamingPipeline::with_params(
        vad,
        extractor,
        DiarizationConfig::default(),
        default_vad_config(),
        params,
    );
    assert!(matches!(res, Err(StreamingError::InvalidParams { .. })));
}

#[test]
fn with_params_rejects_hop_larger_than_window() {
    let params = StreamingParams {
        window_secs: 0.5,
        hop_secs: 1.0,
        ..LatencyPreset::Balanced.params()
    };
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    let res = StreamingPipeline::with_params(
        vad,
        extractor,
        DiarizationConfig::default(),
        default_vad_config(),
        params,
    );
    match res {
        Err(StreamingError::InvalidParams { detail }) => {
            assert!(detail.contains("hop_secs"), "got: {detail}");
        }
        Err(other) => panic!("expected InvalidParams, got {other:?}"),
        Ok(_) => panic!("expected InvalidParams error"),
    }
}

#[test]
fn with_params_rejects_sub_sample_window() {
    // Positive but so small it truncates to zero samples at 16 kHz.
    let params = StreamingParams {
        window_secs: 1e-9,
        hop_secs: 1e-9,
        ..LatencyPreset::Balanced.params()
    };
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    let res = StreamingPipeline::with_params(
        vad,
        extractor,
        DiarizationConfig::default(),
        default_vad_config(),
        params,
    );
    assert!(matches!(res, Err(StreamingError::InvalidParams { .. })));
}

#[test]
fn with_params_clamps_zero_speaker_cache_cap() {
    let params = StreamingParams {
        speaker_cache_cap: 0,
        ..LatencyPreset::Balanced.params()
    };
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    let p = StreamingPipeline::with_params(
        vad,
        extractor,
        DiarizationConfig::default(),
        default_vad_config(),
        params,
    )
    .unwrap();
    assert_eq!(p.speaker_cache_cap(), 1);
}

#[test]
fn new_rejects_zero_window_secs_in_config() {
    let mut config = default_config();
    config.window.window_secs = 0.0;
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    let res = StreamingPipeline::new(vad, extractor, config, default_vad_config());
    assert!(matches!(res, Err(StreamingError::InvalidParams { .. })));
}

#[test]
fn emitted_turns_carry_stable_flag() {
    // DummyExtractor yields a fresh pseudo-random unit vector per call, so
    // use a tiny cache + high match threshold so force-merge reuses speakers
    // and stability can latch.
    let params = StreamingParams {
        window_secs: 1.0,
        hop_secs: 0.5,
        right_context_secs: 0.0,
        speaker_cache_cap: 2,
        min_hits_to_stable: 2,
        prefer_current_margin: 0.05,
        match_threshold: 0.99,
    };
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    let mut p = StreamingPipeline::with_params(
        vad,
        extractor,
        DiarizationConfig::default(),
        default_vad_config(),
        params,
    )
    .unwrap();
    let turns = p.feed(&loud_samples(6.0)).unwrap();
    assert!(!turns.is_empty());
    let all = p.turns();
    // First hits for a speaker are provisional; overflow re-hits latch stability.
    assert!(
        all.iter().any(|t| !t.stable),
        "first hits for a speaker are provisional"
    );
    assert!(
        all.iter().any(|t| t.stable),
        "expected at least one stable turn after repeated overflow hits"
    );
}

#[test]
fn model_long_stream_cache_stays_bounded() {
    // Model-level stand-in for the ≥1 h bench: many assign steps must not
    // grow cache past cap (O(cap) state, no O(t) centroid list growth).
    let mut cache = ArrivalOrderSpeakerCache::new(8, 0.5, 3, 0.05);
    let dim = 32;
    for i in 0..5_000 {
        let mut emb = vec![0.0f32; dim];
        emb[i % dim] = 1.0;
        emb[(i * 3) % dim] += 0.1;
        crate::utils::l2_normalize(&mut emb);
        cache.assign(&emb);
        assert!(cache.len() <= cache.cap());
    }
    assert_eq!(cache.cap(), 8);
    assert!(cache.len() <= 8);
}

#[test]
fn flip_rate_helper_exported() {
    let first = [SpeakerId(0), SpeakerId(1)];
    let final_ = [SpeakerId(0), SpeakerId(0)];
    assert!((label_flip_rate(&first, &final_) - 0.5).abs() < 1e-6);
}

/// VAD that fails on the first frame, to exercise the `vad.process`
/// error-propagation path.
struct FailingVad;

impl VoiceActivityDetector for FailingVad {
    fn reset(&mut self) {}

    fn process(&mut self, _samples: &[f32]) -> Result<Vec<f32>, VadError> {
        Err(VadError::Model("synthetic vad failure".into()))
    }

    fn sample_rate(&self) -> u32 {
        16000
    }
}

/// Embedder that always fails with resource exhaustion, to exercise the
/// embedding error paths in window extraction and flush.
struct FailingEmbedder;

impl Embedder for FailingEmbedder {
    fn dim(&self) -> usize {
        4
    }

    fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        Err(EmbedderError::ResourceExhausted {
            detail: "synthetic pool exhaustion".into(),
        })
    }
}

#[test]
fn feed_propagates_vad_error() {
    let extractor = DummyExtractor::new(256);
    let mut p = StreamingPipeline::new(
        FailingVad,
        extractor,
        default_config(),
        default_vad_config(),
    )
    .unwrap();
    let err = p.feed(&loud_samples(1.0)).unwrap_err();
    assert!(!err.is_resource_exhausted());
    match err {
        StreamingError::Vad(VadError::Model(msg)) => {
            assert!(msg.contains("synthetic vad failure"));
        }
        other => panic!("expected Vad error, got {other:?}"),
    }
}

#[test]
fn feed_propagates_embed_error_as_resource_exhausted() {
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let mut p =
        StreamingPipeline::new(vad, FailingEmbedder, default_config(), default_vad_config())
            .unwrap();
    // 5 s of speech guarantees a full window is extracted during feed.
    let err = p.feed(&loud_samples(5.0)).unwrap_err();
    assert!(matches!(err, StreamingError::Embedding(_)));
    assert!(
        err.is_resource_exhausted(),
        "embedder back-pressure must classify as resource exhaustion"
    );
}

#[test]
fn flush_propagates_embed_error() {
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let mut p =
        StreamingPipeline::new(vad, FailingEmbedder, default_config(), default_vad_config())
            .unwrap();
    // Under one window: nothing extracted during feed, so the error
    // surfaces from the trailing-window flush path instead.
    let _ = p.feed(&loud_samples(1.0)).unwrap();
    let err = p.flush().unwrap_err();
    assert!(matches!(err, StreamingError::Embedding(_)));
}

#[test]
fn short_speech_blip_is_dropped_by_min_duration_filter() {
    // Region closes after the min-silence hangover but is far shorter than
    // the configured minimum speech duration, so the buffered audio is
    // discarded instead of producing a turn.
    let mut config = default_config();
    config.speech_filter.min_speech_secs = 5.0;
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    let mut p = StreamingPipeline::new(vad, extractor, config, default_vad_config()).unwrap();
    assert!(p.feed(&loud_samples(0.1)).unwrap().is_empty());
    // 1 s of silence exceeds the 300 ms min-silence hangover and closes
    // the region inside feed().
    assert!(p.feed(&silent_samples(1.0)).unwrap().is_empty());
    assert!(p.turns().is_empty());
    assert_eq!(p.num_speakers(), 0);
}

#[test]
fn short_in_flight_speech_is_dropped_on_flush() {
    let mut config = default_config();
    config.speech_filter.min_speech_secs = 5.0;
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    let mut p = StreamingPipeline::new(vad, extractor, config, default_vad_config()).unwrap();
    let _ = p.feed(&loud_samples(0.5)).unwrap();
    let turns = p.flush().unwrap();
    assert!(turns.is_empty(), "sub-minimum region must be discarded");
    assert!(p.turns().is_empty());
}

#[test]
fn flush_without_any_speech_is_empty() {
    let mut p = pipeline();
    assert!(p.flush().unwrap().is_empty(), "no region was ever opened");
    let _ = p.feed(&silent_samples(1.0)).unwrap();
    assert!(p.flush().unwrap().is_empty(), "silence opens no region");
    assert!(p.turns().is_empty());
}

#[test]
fn sub_frame_chunks_are_buffered_until_a_full_frame() {
    let mut p = pipeline();
    // 160 + 160 samples < one 512-sample VAD frame: nothing processed yet.
    assert!(p.feed(&silent_samples(0.01)).unwrap().is_empty());
    assert!(p.feed(&silent_samples(0.01)).unwrap().is_empty());
    // Crossing the frame boundary processes normally.
    assert!(p.feed(&silent_samples(1.0)).unwrap().is_empty());
    assert!(p.turns().is_empty());
}

#[test]
fn new_pipeline_reports_no_preset_and_config_derived_params() {
    let mut config = default_config();
    config.cluster.max_speakers = 5;
    config.cluster.threshold = 0.7;
    let vad = EnergyVad::new(-40.0, 16000, 512);
    let extractor = DummyExtractor::new(256);
    let p = StreamingPipeline::new(vad, extractor, config, default_vad_config()).unwrap();
    assert_eq!(p.latency_preset(), None);
    assert_eq!(p.speaker_cache_cap(), 5);
    assert!((p.params().match_threshold - 0.7).abs() < 1e-6);
}

#[test]
fn accurate_preset_reports_preset_and_geometry() {
    let p = pipeline_preset(LatencyPreset::Accurate);
    assert_eq!(p.latency_preset(), Some(LatencyPreset::Accurate));
    assert!((p.params().window_secs - 2.0).abs() < 1e-6);
    assert_eq!(p.speaker_cache_cap(), 64);
}

#[test]
fn streaming_error_display_mentions_variant_details() {
    let vad_err = StreamingError::Vad(VadError::InvalidChunkSize {
        expected: 512,
        got: 100,
    });
    let s = vad_err.to_string();
    assert!(s.contains("VAD error") && s.contains("512") && s.contains("100"));
    assert!(!vad_err.is_resource_exhausted());

    let emb = StreamingError::Embedding(EmbedderError::AudioTooShort {
        actual_secs: 0.1,
        min_secs: 1.0,
    });
    assert!(emb.to_string().contains("embedding error"));
    assert!(!emb.is_resource_exhausted());

    let mismatch = StreamingError::VadFrameMismatch {
        frame_samples: 512,
        got: 2,
    };
    let s = mismatch.to_string();
    assert!(s.contains("512") && s.contains('2'));
    assert!(!mismatch.is_resource_exhausted());

    let invalid = StreamingError::InvalidParams {
        detail: "window_secs must be positive".into(),
    };
    assert!(invalid.to_string().contains("invalid streaming params"));
    assert!(!invalid.is_resource_exhausted());
}
