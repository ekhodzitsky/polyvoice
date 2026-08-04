//! Real-time streaming diarization pipeline.
//!
//! Processes audio incrementally chunk-by-chunk with bounded latency.
//! Unlike the offline [`LegacyPipeline`](crate::pipeline::LegacyPipeline),
//! `StreamingPipeline`
//! emits [`SpeakerTurn`]s as soon as each embedding window is processed.
//!
//! Generic over [`crate::Embedder`] — bring-your-own encoders work without the
//! `onnx` feature (see module example).
//!
//! # Latency
//!
//! Input-buffer latency is bounded by the active [`LatencyPreset`]:
//!
//! ```text
//! input_buffer_latency ≈ window_secs + right_context_secs + vad_frame_secs
//! ```
//!
//! At 16 kHz with EnergyVad frame size 512, `vad_frame_secs ≈ 0.032 s`.
//! Report **latency**, **RTF**, and **DER** as separate numbers (see
//! `docs/BENCHMARKS.md`).
//!
//! | Preset     | window | hop  | right ctx | cache cap | budget @16 kHz |
//! |------------|--------|------|-----------|-----------|----------------|
//! | `realtime` | 1.0 s  | 0.5  | 0.0       | 16        | ≈ 1.03 s       |
//! | `balanced` | 1.5 s  | 0.75 | 0.0       | 32        | ≈ 1.53 s       |
//! | `accurate` | 2.0 s  | 1.0  | 0.25      | 64        | ≈ 2.28 s       |
//!
//! `balanced` matches [`DiarizationConfig::default`] window geometry.
//!
//! # Provisional labels
//!
//! Turns may be emitted with [`SpeakerTurn::stable`]` == false` while a speaker
//! is still gathering hits in the arrival-order cache. Until the cache entry
//! reaches `min_hits_to_stable`, the label is **provisional** (Unknown-class):
//! subsequent windows for that talker may still flip under hysteresis. Once
//! `stable` is `true`, the speaker ID for that cache entry is immutable.
//! Already-emitted history is not rewritten — callers that need only final
//! labels should wait for `stable: true` turns (Azure DiarizeIntermediateResults
//! pattern).
//!
//! # Speaker cap / overflow
//!
//! The arrival-order cache is hard-capped (`speaker_cache_cap`). When full,
//! unmatched embeddings are **force-merged** into the closest existing speaker
//! (AWS-style overflow). Per-chunk work stays O(cap).
//!
//! # Example
//! ```rust,no_run
//! use polyvoice::streaming::{LatencyPreset, StreamingPipeline};
//! use polyvoice::{DummyExtractor, EnergyVad, VadConfig};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let vad = EnergyVad::new(-40.0, 16000, 512);
//!     // DummyExtractor implements Embedder directly.
//!     let extractor = DummyExtractor::new(256);
//!     let mut pipeline = StreamingPipeline::with_latency_preset(
//!         vad,
//!         extractor,
//!         LatencyPreset::Balanced,
//!         VadConfig::default(),
//!     )?;
//!     let chunk = vec![0.0f32; 16000];
//!     let _turns = pipeline.feed(&chunk)?;
//!     Ok(())
//! }
//! ```

mod cache;
mod latency;
mod stability;

pub use cache::{ArrivalOrderSpeakerCache, AssignResult};
pub use latency::{LatencyPreset, LatencyPresetParseError, StreamingParams};
pub use stability::{label_flip_rate, prefer_current_speaker};

use crate::VadConfig;
use crate::embedder::{Embedder, EmbedderError};
use crate::types::{DiarizationConfig, SpeakerTurn, TimeRange};
use crate::vad::{VadError, VadEvent, VadStateMachine, VoiceActivityDetector};
use crate::window::WindowBuffer;

/// Errors from streaming pipeline operations.
#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    #[error("VAD error: {0}")]
    Vad(#[from] VadError),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbedderError),
    #[error(
        "VAD returned {got} probabilities for one {frame_samples}-sample frame; \
         StreamingPipeline requires exactly one probability per VadConfig::frame_size \
         samples, so VadConfig::frame_size must equal the detector's native frame size"
    )]
    VadFrameMismatch { frame_samples: usize, got: usize },
    #[error("invalid streaming params: {detail}")]
    InvalidParams { detail: String },
}

impl StreamingError {
    /// True when the failure is encoder resource exhaustion (pool / back-pressure).
    pub fn is_resource_exhausted(&self) -> bool {
        match self {
            Self::Embedding(e) => e.is_resource_exhausted(),
            Self::Vad(_) | Self::VadFrameMismatch { .. } | Self::InvalidParams { .. } => false,
        }
    }
}

/// Stateful streaming diarization pipeline.
///
/// Generic over a [`VoiceActivityDetector`] `V` and an [`Embedder`] `E`.
/// Speaker assignment uses an AOSC-style [`ArrivalOrderSpeakerCache`] (bounded,
/// arrival-order IDs, provisional→stable labels, prefer-current hysteresis).
pub struct StreamingPipeline<V, E> {
    vad: V,
    extractor: E,
    cache: ArrivalOrderSpeakerCache,
    params: StreamingParams,
    preset: Option<LatencyPreset>,
    frame_size: usize,
    sample_rate: u32,
    // VAD buffering
    vad_buffer: Vec<f32>,
    // Speech detection state
    vad_state: VadStateMachine,
    // Embedding state (active speech region)
    window_buffer: WindowBuffer,
    // Output
    turns: Vec<SpeakerTurn>,
    total_frames: usize,
}

impl<V, E> StreamingPipeline<V, E>
where
    V: VoiceActivityDetector,
    E: Embedder,
{
    /// Create a new streaming pipeline with explicit diarization + VAD config.
    ///
    /// Uses balanced-equivalent cache defaults derived from `config.cluster`
    /// (`max_speakers` as cache cap, `threshold` as match threshold) and the
    /// balanced stability knobs (`min_hits_to_stable = 3`, prefer-current margin
    /// `0.08`). Prefer [`Self::with_latency_preset`] for named latency modes.
    ///
    /// # Errors
    /// Returns `VadError::InvalidChunkSize` if the VAD `frame_size` is zero and
    /// [`StreamingError::InvalidParams`] if the config's window geometry is not
    /// positive and ordered (`0 < hop_secs <= window_secs`).
    pub fn new(
        vad: V,
        extractor: E,
        config: DiarizationConfig,
        vad_config: VadConfig,
    ) -> Result<Self, StreamingError> {
        let params = StreamingParams {
            window_secs: config.window.window_secs,
            hop_secs: config.window.hop_secs,
            right_context_secs: 0.0,
            speaker_cache_cap: config.cluster.max_speakers.max(1),
            min_hits_to_stable: LatencyPreset::Balanced.params().min_hits_to_stable,
            prefer_current_margin: LatencyPreset::Balanced.params().prefer_current_margin,
            match_threshold: config.cluster.threshold,
        };
        Self::from_parts(vad, extractor, config, vad_config, params, None)
    }

    /// Construct a pipeline from a named [`LatencyPreset`].
    ///
    /// Applies the preset's window geometry onto a default [`DiarizationConfig`]
    /// and installs the matching cache / stability parameters.
    pub fn with_latency_preset(
        vad: V,
        extractor: E,
        preset: LatencyPreset,
        vad_config: VadConfig,
    ) -> Result<Self, StreamingError> {
        let mut config = DiarizationConfig::default();
        preset.apply(&mut config);
        let params = preset.params();
        Self::from_parts(vad, extractor, config, vad_config, params, Some(preset))
    }

    /// Construct with full control over diarization config and streaming params.
    ///
    /// `params.speaker_cache_cap == 0` is clamped to 1, matching the
    /// `max_speakers.max(1)` policy of [`Self::new`].
    ///
    /// # Errors
    /// Returns `VadError::InvalidChunkSize` if the VAD `frame_size` is zero and
    /// [`StreamingError::InvalidParams`] if the window geometry is not positive
    /// and ordered (`0 < hop_secs <= window_secs`, yielding at least one sample
    /// each at the configured sample rate).
    pub fn with_params(
        vad: V,
        extractor: E,
        mut config: DiarizationConfig,
        vad_config: VadConfig,
        mut params: StreamingParams,
    ) -> Result<Self, StreamingError> {
        params.speaker_cache_cap = params.speaker_cache_cap.max(1);
        // Keep window geometry on the diarization config aligned with params.
        config.window.window_secs = params.window_secs;
        config.window.hop_secs = params.hop_secs;
        config.cluster.threshold = params.match_threshold;
        config.cluster.max_speakers = params.speaker_cache_cap;
        Self::from_parts(vad, extractor, config, vad_config, params, None)
    }

    fn from_parts(
        vad: V,
        extractor: E,
        config: DiarizationConfig,
        vad_config: VadConfig,
        params: StreamingParams,
        preset: Option<LatencyPreset>,
    ) -> Result<Self, StreamingError> {
        let frame_size = vad_config.frame_size;
        let sample_rate = config.window.sample_rate.get();
        let geometry =
            vad_config.frame_geometry(sample_rate, config.speech_filter.min_speech_secs)?;
        Self::validate_window_geometry(&config, &params)?;

        let cache = ArrivalOrderSpeakerCache::new(
            params.speaker_cache_cap,
            params.match_threshold,
            params.min_hits_to_stable,
            params.prefer_current_margin,
        );

        let vad_state = VadStateMachine::new(
            vad_config.threshold,
            geometry.min_silence_frames,
            geometry.min_speech_frames,
        );

        Ok(Self {
            vad,
            extractor,
            cache,
            params,
            preset,
            frame_size,
            sample_rate,
            vad_buffer: Vec::new(),
            vad_state,
            window_buffer: WindowBuffer::new(config.window_samples(), config.hop_samples()),
            turns: Vec::new(),
            total_frames: 0,
        })
    }

    /// Reject window geometry that would otherwise panic inside
    /// [`WindowBuffer`]: non-positive or non-finite durations, a hop larger
    /// than the window, or durations too small to yield even one sample at
    /// the configured sample rate.
    fn validate_window_geometry(
        config: &DiarizationConfig,
        params: &StreamingParams,
    ) -> Result<(), StreamingError> {
        let window_secs = params.window_secs;
        let hop_secs = params.hop_secs;
        if !window_secs.is_finite() || window_secs <= 0.0 {
            return Err(StreamingError::InvalidParams {
                detail: format!("window_secs must be finite and > 0, got {window_secs}"),
            });
        }
        if !hop_secs.is_finite() || hop_secs <= 0.0 {
            return Err(StreamingError::InvalidParams {
                detail: format!("hop_secs must be finite and > 0, got {hop_secs}"),
            });
        }
        if hop_secs > window_secs {
            return Err(StreamingError::InvalidParams {
                detail: format!("hop_secs ({hop_secs}) must be <= window_secs ({window_secs})"),
            });
        }
        if config.window_samples() == 0 || config.hop_samples() == 0 {
            return Err(StreamingError::InvalidParams {
                detail: format!(
                    "window_secs ({window_secs}) / hop_secs ({hop_secs}) must each yield at \
                     least one sample at sample_rate {}",
                    config.window.sample_rate.get()
                ),
            });
        }
        Ok(())
    }

    /// Active streaming parameters (window, cache cap, stability knobs).
    pub fn params(&self) -> StreamingParams {
        self.params
    }

    /// Named preset if the pipeline was built via [`Self::with_latency_preset`].
    pub fn latency_preset(&self) -> Option<LatencyPreset> {
        self.preset
    }

    /// Hard cap on the speaker cache (`params.speaker_cache_cap`).
    pub fn speaker_cache_cap(&self) -> usize {
        self.cache.cap()
    }

    /// Current number of cache entries (always `<= speaker_cache_cap()`).
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Feed a chunk of audio samples and return any newly finalized speaker turns.
    ///
    /// The pipeline internally buffers samples until a full VAD frame is available,
    /// then runs the frame through VAD, speech detection, and — during active speech —
    /// extracts embeddings and assigns speakers incrementally.
    ///
    /// Callers should feed chunks as they arrive from the audio source (e.g. microphone).
    /// There is no minimum chunk size; sub-frame chunks are buffered transparently.
    ///
    /// Returned turns may have `stable: false` (provisional); see module docs.
    ///
    /// # VAD frame contract
    ///
    /// The detector's native frame size must equal [`VadConfig::frame_size`],
    /// so each buffered frame yields exactly one probability (see the trait's
    /// [frame contract](VoiceActivityDetector#frame-contract)). A mismatch is
    /// rejected with [`StreamingError::VadFrameMismatch`] on the first frame
    /// instead of silently shifting every derived timestamp.
    pub fn feed(&mut self, samples: &[f32]) -> Result<Vec<SpeakerTurn>, StreamingError> {
        let mut new_turns = Vec::new();
        self.vad_buffer.extend_from_slice(samples);

        let frame_size = self.frame_size;
        while self.vad_buffer.len() >= frame_size {
            let frame: Vec<f32> = self.vad_buffer.drain(..frame_size).collect();
            let probs = self.vad.process(&frame)?;

            // Frame-numbering guard: frame indices are converted to sample
            // offsets as `frame_index * frame_size`, which is only exact when
            // each `frame_size` block yields exactly one probability. A
            // detector whose native frame differs from `VadConfig::frame_size`
            // would silently shift every timestamp, so reject it loudly on
            // the first offending frame.
            if probs.len() != 1 {
                return Err(StreamingError::VadFrameMismatch {
                    frame_samples: frame_size,
                    got: probs.len(),
                });
            }

            let prob = probs[0];
            let current_frame = self.total_frames;
            self.total_frames += 1;

            if let Some(event) = self.vad_state.advance(prob, current_frame) {
                match event {
                    VadEvent::SpeechStart { start_frame } => {
                        self.window_buffer.clear();
                        self.window_buffer.set_next_start(start_frame * frame_size);
                    }
                    VadEvent::SpeechEnd {
                        start_frame,
                        end_frame,
                    } => {
                        let seg_end_sample = end_frame * frame_size;
                        if self
                            .vad_state
                            .meets_min_speech_duration(start_frame, end_frame)
                        {
                            new_turns.extend(self.flush_window_buffer(seg_end_sample)?);
                        } else {
                            self.window_buffer.clear();
                        }
                    }
                }
            }

            if self.vad_state.in_speech() {
                self.window_buffer.extend(&frame);
                new_turns.extend(self.try_extract_windows()?);
            }
        }

        // Accumulate into the cumulative history exposed by `turns()`. Turns are
        // produced in increasing start-time order (windows pop sequentially) and
        // feed() is called in stream order, so global monotonicity is preserved.
        self.turns.extend(new_turns.iter().cloned());
        Ok(new_turns)
    }

    /// Flush any pending audio and return final speaker turns.
    ///
    /// This finalizes an in-flight speech region (if any), extracts the last
    /// embedding window, and clears all internal buffers. After `flush` the
    /// pipeline is ready to process a new stream (or the same stream after a
    /// gap) via subsequent `feed` calls.
    pub fn flush(&mut self) -> Result<Vec<SpeakerTurn>, StreamingError> {
        let mut new_turns = Vec::new();

        // Discard any trailing sub-frame samples.
        self.vad_buffer.clear();

        if let Some(VadEvent::SpeechEnd {
            start_frame,
            end_frame,
        }) = self.vad_state.flush(self.total_frames)
        {
            if self
                .vad_state
                .meets_min_speech_duration(start_frame, end_frame)
            {
                let seg_end_sample = end_frame * self.frame_size;
                new_turns.extend(self.flush_window_buffer(seg_end_sample)?);
            } else {
                self.window_buffer.clear();
            }
        }

        // Accumulate into the cumulative history exposed by `turns()` (same
        // monotonicity reasoning as feed()). We deliberately do NOT clear
        // self.turns here: turns() promises cumulative history; callers wanting a
        // fresh history construct a new pipeline.
        self.turns.extend(new_turns.iter().cloned());
        Ok(new_turns)
    }

    /// Return the number of distinct speakers observed so far.
    pub fn num_speakers(&self) -> usize {
        self.cache.len()
    }

    /// Return all turns emitted so far (including those from prior `feed` calls).
    ///
    /// History is cumulative across `feed`/`flush`; `flush` does not reset it.
    /// Construct a new pipeline for a fresh history.
    pub fn turns(&self) -> &[SpeakerTurn] {
        &self.turns
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Extract as many full windows as possible from `window_buffer`.
    fn try_extract_windows(&mut self) -> Result<Vec<SpeakerTurn>, StreamingError> {
        let mut turns = Vec::new();
        let sr_f = self.sample_rate as f64;

        while let Some((start, chunk)) = self.window_buffer.try_pop() {
            let embedding = self.extractor.embed(&chunk)?;
            let assigned = self.cache.assign(&embedding);
            debug_assert!(self.cache.len() <= self.cache.cap());
            let end = start + chunk.len();
            turns.push(SpeakerTurn::with_stability(
                assigned.speaker,
                TimeRange {
                    start: start as f64 / sr_f,
                    end: end as f64 / sr_f,
                },
                assigned.stable,
            ));
        }

        Ok(turns)
    }

    /// Zero-pad the trailing `window_buffer`, extract one final embedding, and clear the buffer.
    fn flush_window_buffer(
        &mut self,
        seg_end_sample: usize,
    ) -> Result<Vec<SpeakerTurn>, StreamingError> {
        let mut turns = Vec::new();
        let sr_f = self.sample_rate as f64;

        if let Some((start, padded)) = self.window_buffer.flush() {
            let embedding = self.extractor.embed(&padded)?;
            let assigned = self.cache.assign(&embedding);
            debug_assert!(self.cache.len() <= self.cache.cap());
            let end = seg_end_sample.min(start + padded.len());
            turns.push(SpeakerTurn::with_stability(
                assigned.speaker,
                TimeRange {
                    start: start as f64 / sr_f,
                    end: end as f64 / sr_f,
                },
                assigned.stable,
            ));
        }

        Ok(turns)
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
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
}
