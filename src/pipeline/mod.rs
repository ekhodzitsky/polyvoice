//! High-level offline diarization pipeline (legacy v1).
//!
//! Wires together VAD, embedding extraction, and AHC clustering into a
//! single `run()` call that takes audio and returns `DiarizationResult`.
//! This is the BYO / ort-free path; the production ONNX pipeline is
//! `pipeline_v2::Pipeline` (feature-gated, re-exported at the crate root).
//!
//! # Bring-your-own embedder
//!
//! `LegacyPipeline` is generic over [`crate::Embedder`]. Implement that trait on an
//! external encoder (Candle, tract, custom) — no `onnx` feature required:
//!
//! ```rust
//! use polyvoice::pipeline::LegacyPipeline;
//! use polyvoice::{DiarizationConfig, Embedder, EmbedderError, EnergyVad, VadConfig};
//!
//! struct FixedEmbedder;
//!
//! impl Embedder for FixedEmbedder {
//!     fn dim(&self) -> usize { 4 }
//!     fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
//!         Ok(vec![1.0, 0.0, 0.0, 0.0])
//!     }
//! }
//!
//! let pipeline = LegacyPipeline::new(DiarizationConfig::default(), VadConfig::default());
//! let mut vad = EnergyVad::new(-40.0, 16_000, 512);
//! // Silence → no speech; exercise the type bounds only.
//! let _ = pipeline.run(&vec![0.0f32; 16_000], &FixedEmbedder, &mut vad);
//! ```

use crate::embedder::{Embedder, EmbedderError};
use crate::types::{
    ConfigError, DiarizationConfig, DiarizationResult, Segment, SpeakerId, SpeakerTurn, TimeRange,
};
use crate::vad::{VadConfig, VadError, VoiceActivityDetector, segment_speech};
use crate::wav;
use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum LegacyPipelineError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(#[from] ConfigError),
    #[error("VAD error: {0}")]
    Vad(#[from] VadError),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbedderError),
    #[error("WAV error: {0}")]
    Wav(#[from] wav::WavError),
    #[error("unsupported WAV sample rate: {actual}, expected: {expected}")]
    UnsupportedSampleRate { expected: u32, actual: u32 },
    #[error("audio too long: {actual_secs:.1}s > max {max_secs:.1}s")]
    AudioTooLong { actual_secs: f32, max_secs: f32 },
    /// Pluggable [`crate::clusterer::Clusterer`] failure
    /// ([`LegacyPipeline::run_with_clusterer`], VBx loading, or the internal
    /// `AhcClusterer` path of [`LegacyPipeline::run`]).
    ///
    /// Gated on the `clusterer` feature because
    /// [`crate::clusterer::ClustererError`] only exists behind it; without the
    /// feature no code path produces this variant.
    #[cfg(feature = "clusterer")]
    #[error("clustering failed: {0}")]
    Clustering(#[from] crate::clusterer::ClustererError),
}

impl LegacyPipelineError {
    /// True when the failure is encoder resource exhaustion (pool / back-pressure).
    ///
    /// Walks the typed [`EmbedderError`] payload so metrics code can avoid
    /// substring-matching display strings.
    pub fn is_resource_exhausted(&self) -> bool {
        match self {
            Self::Embedding(e) => e.is_resource_exhausted(),
            _ => false,
        }
    }
}

/// Backwards-compatible alias for the pre-rename v1 pipeline. Not used by
/// in-crate code; new code should name [`LegacyPipeline`] explicitly (or the
/// crate-root `Pipeline` from `pipeline_v2` when the ONNX features are on).
#[deprecated(note = "renamed to LegacyPipeline; the crate-root `Pipeline` is now pipeline v2")]
pub type Pipeline = LegacyPipeline;

/// Backwards-compatible alias for the pre-rename v1 pipeline error.
#[deprecated(
    note = "renamed to LegacyPipelineError; the crate-root `PipelineError` is now pipeline v2"
)]
pub type PipelineError = LegacyPipelineError;

pub struct LegacyPipeline {
    config: DiarizationConfig,
    vad_config: VadConfig,
}

impl LegacyPipeline {
    /// { true }
    /// pub fn new(config: DiarizationConfig, vad_config: VadConfig) -> Self
    /// { true }
    pub fn new(config: DiarizationConfig, vad_config: VadConfig) -> Self {
        Self { config, vad_config }
    }

    /// { true }
    /// `pub fn run<E: Embedder, V: VoiceActivityDetector>( &self, samples: &[f32], extractor: &E, vad: &mut V, ) -> Result<DiarizationResult, LegacyPipelineError>`
    /// { ret.as_ref().map_or(true, |r| r.num_speakers <= r.segments.len()) }
    /// Run the full diarization pipeline on raw f32 samples.
    ///
    /// `extractor` must implement [`Embedder`] (the supported BYO surface).
    ///
    /// Clustering defaults to free AHC (cosine threshold from
    /// [`DiarizationConfig::cluster`]). For a pluggable backend (VBx from a
    /// local PLDA directory, NME-SC, …) use `Self::run_with_clusterer`
    /// (feature `clusterer`).
    ///
    /// Returns [`LegacyPipelineError::InvalidConfig`] if
    /// [`DiarizationConfig::validate`] rejects the configuration (a zero or
    /// inverted window geometry previously panicked inside the window
    /// iterator).
    ///
    /// Returns [`LegacyPipelineError::AudioTooLong`] if the input exceeds
    /// `config.max_duration_secs` (default 1 hour).
    pub fn run<E: Embedder, V: VoiceActivityDetector>(
        &self,
        samples: &[f32],
        extractor: &E,
        vad: &mut V,
    ) -> Result<DiarizationResult, LegacyPipelineError> {
        self.config.validate()?;
        let (embeddings, time_ranges) = self.embed_windows(samples, extractor, vad)?;
        if embeddings.is_empty() {
            return Ok(self.empty_result(samples.len()));
        }

        // Same free-AHC semantics as before: fixed threshold, no cluster
        // ceiling (`max_clusters = 0`). Routed through `AhcClusterer` so the
        // BYO path shares the Clusterer surface with pipeline_v2. A dim
        // mismatch between windows surfaces as a typed clustering error
        // instead of silently degrading into one cluster per window.
        let labels = {
            #[cfg(feature = "clusterer")]
            {
                use crate::clusterer::{AhcClusterer, Clusterer};
                AhcClusterer::with_threshold(0, self.config.cluster.threshold)
                    .cluster(&embeddings)?
            }
            #[cfg(not(feature = "clusterer"))]
            {
                crate::ahc::agglomerative_cluster(&embeddings, self.config.cluster.threshold)
            }
        };
        self.assemble_result(samples.len(), embeddings, time_ranges, labels)
    }

    /// Run offline diarization with an injected [`crate::clusterer::Clusterer`].
    ///
    /// Use this for BYO accuracy paths that stay ort-free, for example VBx with
    /// PLDA weights loaded from disk:
    ///
    /// ```rust,ignore
    /// use polyvoice::{
    ///     EnergyVad, VadConfig, DiarizationConfig,
    ///     clusterer::vbx::VbxClusterer,
    ///     pipeline::LegacyPipeline,
    /// };
    ///
    /// let vbx = VbxClusterer::from_dir(std::path::Path::new("plda/"), 20)?;
    /// let result = pipeline.run_with_clusterer(&samples, &embedder, &mut vad, &vbx)?;
    /// ```
    ///
    /// Requires features `clusterer` (and `vbx` when using
    /// [`crate::clusterer::vbx::VbxClusterer`]).
    /// Embeddings still come from the caller's [`Embedder`] (typically
    /// L2-normalized); VBx restores scale via its configured `emb_scale`.
    ///
    /// Durations for each embedding window are passed to
    /// [`crate::clusterer::Clusterer::cluster_with_durations`] so short-segment
    /// filtering works.
    ///
    /// Validates the configuration first, like [`Self::run`]
    /// ([`LegacyPipelineError::InvalidConfig`]).
    #[cfg(feature = "clusterer")]
    pub fn run_with_clusterer<E, V, C>(
        &self,
        samples: &[f32],
        extractor: &E,
        vad: &mut V,
        clusterer: &C,
    ) -> Result<DiarizationResult, LegacyPipelineError>
    where
        E: Embedder,
        V: VoiceActivityDetector,
        C: crate::clusterer::Clusterer + ?Sized,
    {
        self.config.validate()?;
        let (embeddings, time_ranges) = self.embed_windows(samples, extractor, vad)?;
        if embeddings.is_empty() {
            return Ok(self.empty_result(samples.len()));
        }

        let durations: Vec<f64> = time_ranges.iter().map(|t| t.duration()).collect();
        let labels = clusterer.cluster_with_durations(&embeddings, &durations)?;
        self.assemble_result(samples.len(), embeddings, time_ranges, labels)
    }

    /// Convenience: [`Self::run_with_clusterer`] with
    /// [`crate::clusterer::vbx::VbxClusterer::from_dir`].
    ///
    /// `plda_dir` must contain the six precomputed `plda_*.npy` files (see
    /// `fixtures/vbx-plda/`). No network and no `download` feature.
    #[cfg(feature = "vbx")]
    pub fn run_with_vbx_from_dir<E, V>(
        &self,
        samples: &[f32],
        extractor: &E,
        vad: &mut V,
        plda_dir: &Path,
        max_speakers: usize,
    ) -> Result<DiarizationResult, LegacyPipelineError>
    where
        E: Embedder,
        V: VoiceActivityDetector,
    {
        let vbx = crate::clusterer::vbx::VbxClusterer::from_dir(plda_dir, max_speakers)?;
        self.run_with_clusterer(samples, extractor, vad, &vbx)
    }

    /// VAD → sliding windows → embeddings (and matching time ranges).
    fn embed_windows<E: Embedder, V: VoiceActivityDetector>(
        &self,
        samples: &[f32],
        extractor: &E,
        vad: &mut V,
    ) -> Result<(Vec<Vec<f32>>, Vec<TimeRange>), LegacyPipelineError> {
        let actual_secs = samples.len() as f32 / self.config.window.sample_rate.get() as f32;
        if actual_secs > self.config.max_duration_secs {
            return Err(LegacyPipelineError::AudioTooLong {
                actual_secs,
                max_secs: self.config.max_duration_secs,
            });
        }
        let speech_regions = segment_speech(vad, samples, &self.config, &self.vad_config)?;
        if speech_regions.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let sr = self.config.window.sample_rate.get() as f64;
        let window = self.config.window_samples();
        let hop = self.config.hop_samples();
        let mut embeddings = Vec::new();
        let mut time_ranges = Vec::new();

        for &(start, end) in &speech_regions {
            let region = &samples[start..end];

            if region.len() < window {
                let mut padded = vec![0.0f32; window];
                padded[..region.len()].copy_from_slice(region);
                let emb = extractor.embed(&padded)?;
                embeddings.push(emb);
                time_ranges.push(TimeRange {
                    start: start as f64 / sr,
                    end: end as f64 / sr,
                });
            } else {
                for (offset, offset_end) in
                    crate::window::WindowIter::new(region.len(), window, hop)
                {
                    let chunk = &region[offset..offset_end];
                    let emb = extractor.embed(chunk)?;
                    embeddings.push(emb);
                    time_ranges.push(TimeRange {
                        start: (start + offset) as f64 / sr,
                        end: (start + offset_end) as f64 / sr,
                    });
                }
            }
        }
        Ok((embeddings, time_ranges))
    }

    fn empty_result(&self, n_samples: usize) -> DiarizationResult {
        let sr_hz = self.config.window.sample_rate.get();
        DiarizationResult::new(Vec::new(), Vec::new(), 0)
            .with_audio(n_samples as f64 / sr_hz as f64, sr_hz)
    }

    /// Labels → prune → segments/turns.
    fn assemble_result(
        &self,
        n_samples: usize,
        embeddings: Vec<Vec<f32>>,
        time_ranges: Vec<TimeRange>,
        labels: Vec<usize>,
    ) -> Result<DiarizationResult, LegacyPipelineError> {
        // Dissolve spurious tiny clusters into the nearest large speaker — the
        // over-clustering fix. Duration pruning (length-invariant) takes
        // precedence when configured; otherwise the member-count rule applies.
        let labels = if self.config.cluster.min_cluster_secs > 0.0 {
            crate::ahc::prune_small_clusters_by_duration(
                &time_ranges,
                &embeddings,
                labels,
                self.config.cluster.min_cluster_secs,
            )
        } else {
            crate::ahc::prune_small_clusters(
                &embeddings,
                labels,
                self.config.cluster.min_cluster_size,
            )
        };
        let num_speakers = labels.iter().copied().max().map_or(0, |m| m + 1);

        // Per-window confidence from cosine similarity to the speaker centroid
        // (logistic map — ranking score, not a calibrated probability).
        let speaker_ids: Vec<SpeakerId> = labels.iter().map(|&l| SpeakerId(l as u32)).collect();
        let confidences =
            crate::types::segment_confidences_from_embeddings(&speaker_ids, &embeddings);

        let mut segments: Vec<Segment> = labels
            .iter()
            .zip(time_ranges.iter())
            .enumerate()
            .map(|(i, (&label, &time))| Segment {
                time,
                speaker: Some(SpeakerId(label as u32)),
                confidence: confidences.get(i).copied(),
            })
            .collect();

        segments =
            crate::utils::merge_segments(segments, self.config.speech_filter.max_gap_secs as f64);
        segments.retain(|s| s.time.duration() >= self.config.speech_filter.min_speech_secs as f64);

        let turns: Vec<SpeakerTurn> = segments
            .iter()
            .filter_map(|s| {
                s.speaker.map(|spk| SpeakerTurn {
                    speaker: spk,
                    time: s.time,
                    text: None,
                    stable: true,
                })
            })
            .collect();

        let sr_hz = self.config.window.sample_rate.get();
        Ok(DiarizationResult::new(segments, turns, num_speakers)
            .with_audio(n_samples as f64 / sr_hz as f64, sr_hz))
    }

    /// { true }
    /// `pub fn run_from_wav<E: Embedder, V: VoiceActivityDetector>( &self, path: &Path, extractor: &E, vad: &mut V, ) -> Result<DiarizationResult, LegacyPipelineError>`
    /// { ret.as_ref().map_or(true, |r| r.num_speakers <= r.segments.len()) }
    /// Run the pipeline from a WAV file path.
    ///
    /// Returns [`LegacyPipelineError::UnsupportedSampleRate`] if the WAV sample rate
    /// does not match [`crate::types::WindowConfig::sample_rate`] in
    /// [`DiarizationConfig::window`].
    pub fn run_from_wav<E: Embedder, V: VoiceActivityDetector>(
        &self,
        path: &Path,
        extractor: &E,
        vad: &mut V,
    ) -> Result<DiarizationResult, LegacyPipelineError> {
        let (samples, sample_rate) = wav::read_wav(path)?;
        let expected = self.config.window.sample_rate.get();
        if sample_rate != expected {
            return Err(LegacyPipelineError::UnsupportedSampleRate {
                expected,
                actual: sample_rate,
            });
        }
        self.run(&samples, extractor, vad)
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Embedder;
    use std::io::Cursor;

    #[test]
    fn pipeline_new_with_defaults() {
        let config = DiarizationConfig::default();
        let vad_config = VadConfig::default();
        let pipeline = LegacyPipeline::new(config, vad_config);
        // LegacyPipeline exists; basic sanity check via debug print would require
        // accessing private fields, so we just verify construction succeeds.
        assert!(std::mem::size_of_val(&pipeline) > 0);
    }

    #[test]
    fn audio_too_long_error() {
        let config = DiarizationConfig {
            max_duration_secs: 1.0,
            ..Default::default()
        };
        let vad_config = VadConfig::default();
        let pipeline = LegacyPipeline::new(config, vad_config);

        // Create 2 seconds of silence at 16kHz
        let samples = vec![0.0f32; 32000];
        let extractor = crate::embedder::DummyExtractor::new(256);
        let mut vad = crate::vad::EnergyVad::new(-40.0, 16000, 512);
        let result = pipeline.run(&samples, &extractor, &mut vad);
        assert!(
            matches!(result, Err(LegacyPipelineError::AudioTooLong { .. })),
            "expected AudioTooLong error, got {:?}",
            result
        );
    }

    #[test]
    fn wav_sample_rate_mismatch_error() {
        // Create a 1-second mono WAV at 22050 Hz while the pipeline expects 16000 Hz.
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 22050,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
            for i in 0..22050 {
                let sample = ((i as f32 / 22050.0) * std::f32::consts::TAU * 440.0).sin();
                writer.write_sample((sample * 32767.0) as i16).unwrap();
            }
            writer.finalize().unwrap();
        }

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &buf).unwrap();

        let config = DiarizationConfig::default();
        let pipeline = LegacyPipeline::new(config, VadConfig::default());
        let extractor = crate::embedder::DummyExtractor::new(256);
        let mut vad = crate::vad::EnergyVad::new(-40.0, 16000, 512);
        let result = pipeline.run_from_wav(tmp.path(), &extractor, &mut vad);
        assert!(
            matches!(
                result,
                Err(LegacyPipelineError::UnsupportedSampleRate {
                    expected: 16000,
                    actual: 22050,
                })
            ),
            "expected UnsupportedSampleRate error, got {:?}",
            result
        );
    }

    /// Deterministic two-prototype embedder: high zero-crossing rate → speaker A,
    /// low ZCR → speaker B. Orthogonal unit vectors so AHC must yield ≥2 clusters.
    struct TwoSpeakerEmbedder;

    impl Embedder for TwoSpeakerEmbedder {
        fn dim(&self) -> usize {
            4
        }

        fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            let mut zcr = 0usize;
            for w in audio.windows(2) {
                if w[0].signum() != w[1].signum() {
                    zcr += 1;
                }
            }
            let rate = zcr as f32 / audio.len().max(1) as f32;
            // 300 Hz @ 16 kHz ≈ 0.0375 ZCR; 800 Hz ≈ 0.1 ZCR.
            let mut v = if rate > 0.06 {
                vec![1.0, 0.0, 0.0, 0.0]
            } else {
                vec![0.0, 1.0, 0.0, 0.0]
            };
            crate::utils::l2_normalize(&mut v);
            Ok(v)
        }
    }

    fn sine_wave(freq: f32, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
        let n = (duration_secs * sample_rate as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn custom_embedder_two_speakers_without_onnx() {
        let sr = 16_000u32;
        let mut samples = sine_wave(300.0, 2.0, sr);
        samples.extend(std::iter::repeat_n(0.0, sr as usize)); // 1 s silence
        samples.extend(sine_wave(800.0, 2.0, sr));

        let mut config = DiarizationConfig::default();
        // Orthogonal prototypes must never merge.
        config.cluster.threshold = 0.9;
        config.cluster.min_cluster_size = 1;
        config.cluster.min_cluster_secs = 0.0;

        let pipeline = LegacyPipeline::new(config, VadConfig::default());
        let embedder = TwoSpeakerEmbedder;
        assert_eq!(embedder.dim(), 4);
        let mut vad = crate::vad::EnergyVad::new(-40.0, sr, 512);
        let result = pipeline.run(&samples, &embedder, &mut vad).unwrap();

        assert!(
            result.num_speakers >= 2,
            "expected ≥2 speakers from deterministic two-prototype embedder, got {}",
            result.num_speakers
        );
        assert!(!result.turns.is_empty());
    }

    #[cfg(feature = "clusterer")]
    #[test]
    fn run_with_clusterer_ahc_matches_run_shape() {
        use crate::clusterer::{AhcClusterer, Clusterer};

        let sr = 16_000u32;
        let mut samples = sine_wave(300.0, 2.0, sr);
        samples.extend(std::iter::repeat_n(0.0, sr as usize));
        samples.extend(sine_wave(800.0, 2.0, sr));

        let mut config = DiarizationConfig::default();
        config.cluster.threshold = 0.9;
        config.cluster.min_cluster_size = 1;
        config.cluster.min_cluster_secs = 0.0;

        let pipeline = LegacyPipeline::new(config, VadConfig::default());
        let embedder = TwoSpeakerEmbedder;
        let mut vad = crate::vad::EnergyVad::new(-40.0, sr, 512);
        let ahc = AhcClusterer::with_threshold(0, config.cluster.threshold);
        let result = pipeline
            .run_with_clusterer(&samples, &embedder, &mut vad, &ahc)
            .expect("run_with_clusterer");
        assert!(result.num_speakers >= 2);
        assert!(!result.turns.is_empty());
        // Trait object path also compiles.
        let boxed: Box<dyn Clusterer> = Box::new(AhcClusterer::with_threshold(0, 0.9));
        let mut vad2 = crate::vad::EnergyVad::new(-40.0, sr, 512);
        let _ = pipeline
            .run_with_clusterer(&samples, &embedder, &mut vad2, boxed.as_ref())
            .expect("dyn Clusterer");
    }

    #[cfg(feature = "vbx")]
    #[test]
    fn run_with_vbx_from_dir_loads_fixtures() {
        let plda = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/vbx-plda");
        assert!(
            plda.join("plda_transform.npy").is_file(),
            "checked-in PLDA fixtures required"
        );

        let sr = 16_000u32;
        // Enough speech for VBx min_embedding_secs (default ~1.6s) after windows.
        let mut samples = sine_wave(300.0, 3.0, sr);
        samples.extend(std::iter::repeat_n(0.0, (sr / 2) as usize));
        samples.extend(sine_wave(800.0, 3.0, sr));

        let mut config = DiarizationConfig::default();
        config.cluster.min_cluster_size = 1;
        config.cluster.min_cluster_secs = 0.0;
        config.window.window_secs = 1.5;
        config.window.hop_secs = 0.75;

        let pipeline = LegacyPipeline::new(config, VadConfig::default());
        // 256-d matches WeSpeaker / PLDA fixture dimensionality.
        let embedder = crate::embedder::DummyExtractor::new(256);
        let mut vad = crate::vad::EnergyVad::new(-40.0, sr, 512);
        let result = pipeline
            .run_with_vbx_from_dir(&samples, &embedder, &mut vad, &plda, 8)
            .expect("VBx from fixtures must run offline");
        // Dummy embeddings are weak; only require a successful offline run.
        assert!(result.num_speakers >= 1 || result.turns.is_empty());
    }

    #[cfg(feature = "vbx")]
    #[test]
    fn run_with_vbx_missing_dir_errors() {
        let pipeline = LegacyPipeline::new(DiarizationConfig::default(), VadConfig::default());
        let embedder = crate::embedder::DummyExtractor::new(256);
        let mut vad = crate::vad::EnergyVad::new(-40.0, 16_000, 512);
        let err = pipeline
            .run_with_vbx_from_dir(
                &[0.1f32; 16_000],
                &embedder,
                &mut vad,
                std::path::Path::new("/no/such/plda"),
                8,
            )
            .expect_err("missing PLDA dir");
        assert!(
            matches!(err, LegacyPipelineError::Clustering(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn run_rejects_invalid_window_geometry() {
        // Zero-length windows used to panic inside WindowIter once speech was
        // found; the config guard must turn that into a typed error.
        for (window_secs, hop_secs) in [(0.0f32, 0.75), (1.5, 0.0), (1.0, 1.5)] {
            let mut config = DiarizationConfig::default();
            config.window.window_secs = window_secs;
            config.window.hop_secs = hop_secs;
            let pipeline = LegacyPipeline::new(config, VadConfig::default());
            let extractor = crate::embedder::DummyExtractor::new(256);
            let mut vad = crate::vad::EnergyVad::new(-40.0, 16_000, 512);
            let result = pipeline.run(&sine_wave(440.0, 1.0, 16_000), &extractor, &mut vad);
            assert!(
                matches!(result, Err(LegacyPipelineError::InvalidConfig(_))),
                "window={window_secs} hop={hop_secs}: expected InvalidConfig, got {result:?}"
            );
        }
    }

    #[test]
    fn run_rejects_out_of_range_threshold() {
        let mut config = DiarizationConfig::default();
        config.cluster.threshold = 1.5;
        let pipeline = LegacyPipeline::new(config, VadConfig::default());
        let extractor = crate::embedder::DummyExtractor::new(256);
        let mut vad = crate::vad::EnergyVad::new(-40.0, 16_000, 512);
        let result = pipeline.run(&sine_wave(440.0, 1.0, 16_000), &extractor, &mut vad);
        assert!(
            matches!(
                result,
                Err(LegacyPipelineError::InvalidConfig(
                    crate::types::ConfigError::InvalidThreshold(_)
                ))
            ),
            "got {result:?}"
        );
    }

    #[cfg(feature = "clusterer")]
    #[test]
    fn run_surfaces_dim_mismatch_as_clustering_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// Returns a longer vector on every call — a buggy BYO embedder whose
        /// inconsistent dims previously degraded silently into one cluster per
        /// window via the free-AHC fallback.
        struct InconsistentDimEmbedder(AtomicUsize);

        impl Embedder for InconsistentDimEmbedder {
            fn dim(&self) -> usize {
                4
            }

            fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
                let n = self.0.fetch_add(1, Ordering::Relaxed);
                Ok(vec![1.0; 4 + n])
            }
        }

        let sr = 16_000u32;
        let samples = sine_wave(300.0, 4.0, sr);
        let pipeline = LegacyPipeline::new(DiarizationConfig::default(), VadConfig::default());
        let embedder = InconsistentDimEmbedder(AtomicUsize::new(0));
        assert_eq!(embedder.dim(), 4);
        let mut vad = crate::vad::EnergyVad::new(-40.0, sr, 512);
        let err = pipeline
            .run(&samples, &embedder, &mut vad)
            .expect_err("inconsistent embedding dims must fail");
        assert!(
            matches!(
                err,
                LegacyPipelineError::Clustering(
                    crate::clusterer::ClustererError::DimMismatch { .. }
                )
            ),
            "got {err:?}"
        );
    }

    #[cfg(feature = "clusterer")]
    #[test]
    fn run_with_clusterer_silence_returns_empty_result() {
        use crate::clusterer::AhcClusterer;

        let pipeline = LegacyPipeline::new(DiarizationConfig::default(), VadConfig::default());
        let embedder = crate::embedder::DummyExtractor::new(256);
        let mut vad = crate::vad::EnergyVad::new(-40.0, 16_000, 512);
        let ahc = AhcClusterer::with_threshold(0, 0.5);
        // Pure silence → no speech regions → no embeddings: the clusterer is
        // never invoked and the empty-result shortcut is returned.
        let result = pipeline
            .run_with_clusterer(&vec![0.0f32; 16_000], &embedder, &mut vad, &ahc)
            .expect("silence run");
        assert_eq!(result.num_speakers, 0);
        assert!(result.segments.is_empty());
        assert!(result.turns.is_empty());
        assert!((result.audio.duration_secs - 1.0).abs() < 1e-9);
    }

    #[test]
    fn run_prunes_small_clusters_by_duration_when_configured() {
        let sr = 16_000u32;
        let mut samples = sine_wave(300.0, 2.0, sr);
        samples.extend(std::iter::repeat_n(0.0, sr as usize));
        samples.extend(sine_wave(800.0, 2.0, sr));

        let mut config = DiarizationConfig::default();
        config.cluster.threshold = 0.9;
        config.cluster.min_cluster_size = 1;
        // Non-zero min_cluster_secs selects the duration-based pruning rule.
        config.cluster.min_cluster_secs = 0.5;

        let pipeline = LegacyPipeline::new(config, VadConfig::default());
        let embedder = TwoSpeakerEmbedder;
        let mut vad = crate::vad::EnergyVad::new(-40.0, sr, 512);
        let result = pipeline.run(&samples, &embedder, &mut vad).unwrap();

        assert!(
            result.num_speakers >= 1,
            "duration pruning must keep at least one speaker"
        );
        assert!(!result.turns.is_empty());
        assert!(result.num_speakers <= result.segments.len());
    }

    #[test]
    fn run_from_wav_succeeds_on_matching_sample_rate() {
        let sr = 16_000u32;
        let samples = sine_wave(440.0, 1.0, sr);

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
            for &s in &samples {
                writer.write_sample((s * 32767.0) as i16).unwrap();
            }
            writer.finalize().unwrap();
        }
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &buf).unwrap();

        let pipeline = LegacyPipeline::new(DiarizationConfig::default(), VadConfig::default());
        let embedder = crate::embedder::DummyExtractor::new(256);
        let mut vad = crate::vad::EnergyVad::new(-40.0, sr, 512);
        let result = pipeline
            .run_from_wav(tmp.path(), &embedder, &mut vad)
            .expect("matching sample rate must run");
        assert!(!result.segments.is_empty());
        assert!(result.num_speakers >= 1);
    }
}
