//! Core types for speaker diarization.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Opaque identifier for a speaker cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpeakerId(pub u32);

/// A remapping table produced by [`SpeakerCluster::merge`](crate::cluster::SpeakerCluster::merge).
///
/// When two speaker centroids are merged, all indices after the removed one shift
/// left by one. This struct captures the old → new mapping so that callers can
/// update any stored [`SpeakerId`]s (e.g. in [`Segment`]s or [`SpeakerTurn`]s).
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerIdRemap {
    /// Mapping from old SpeakerId to new SpeakerId.
    mapping: Vec<(SpeakerId, SpeakerId)>,
}

impl SpeakerIdRemap {
    /// Create a remap from a raw vector of (old, new) pairs.
    ///
    /// { mapping.iter().all(|(old, new)| old != new) }
    /// `fn from_mapping(mapping: Vec<(SpeakerId, SpeakerId)>) -> Self`
    /// { ret.mapping.len() == mapping.len() }
    pub fn from_mapping(mapping: Vec<(SpeakerId, SpeakerId)>) -> Self {
        Self { mapping }
    }

    /// Apply the remap to a single [`SpeakerId`].
    ///
    /// Returns the new ID if the old ID was remapped, otherwise returns `id` unchanged.
    pub fn remap(&self, id: SpeakerId) -> SpeakerId {
        self.mapping
            .iter()
            .find(|(old, _)| *old == id)
            .map(|(_, new)| *new)
            .unwrap_or(id)
    }

    /// Returns true if no IDs were changed.
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

    /// Returns the number of remapped IDs.
    pub fn len(&self) -> usize {
        self.mapping.len()
    }
}

/// Remap speaker IDs in a slice of [`Segment`]s in-place.
///
/// { true }
/// `fn remap_segments(segments: &mut [Segment], remap: &SpeakerIdRemap)`
/// { segments.iter().all(|s| s.speaker.map_or(true, |spk| remap.remap(spk) == s.speaker.unwrap())) || !remap.is_empty() }
pub fn remap_segments(segments: &mut [Segment], remap: &SpeakerIdRemap) {
    for seg in segments.iter_mut() {
        if let Some(spk) = seg.speaker {
            seg.speaker = Some(remap.remap(spk));
        }
    }
}

/// Remap speaker IDs in a slice of [`SpeakerTurn`]s in-place.
///
/// { true }
/// `fn remap_turns(turns: &mut [SpeakerTurn], remap: &SpeakerIdRemap)`
/// { turns.iter().all(|t| remap.remap(t.speaker) == t.speaker) || !remap.is_empty() }
pub fn remap_turns(turns: &mut [SpeakerTurn], remap: &SpeakerIdRemap) {
    for turn in turns.iter_mut() {
        turn.speaker = remap.remap(turn.speaker);
    }
}

impl fmt::Display for SpeakerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SPEAKER_{:02}", self.0)
    }
}

/// Pre-configured model bundles trading off accuracy and footprint.
///
/// `Mobile` targets weak/embedded ARM CPUs (≤10 MB total models, ≤200 MB peak RAM).
/// `Balanced` targets modern phone/laptop ARM CPUs (≤35 MB total models, ≤400 MB peak RAM).
/// `Custom` defers all model selection to the caller and is used by `PipelineBuilder`
/// when individual `Segmenter`/`Embedder`/`Clusterer` instances are supplied directly.
///
/// Added in v0.6 (M0). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md`
/// §5.1 for the full motivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Profile {
    Mobile,
    Balanced,
    Custom,
}

impl Profile {
    /// Embedding dimension produced by the embedder for this profile.
    /// Returns 0 for `Custom` (caller must resolve dimension explicitly).
    pub const fn embedding_dim(self) -> usize {
        match self {
            Profile::Mobile => 512,   // CAM++ output dim (voxceleb_CAM++.onnx)
            Profile::Balanced => 256, // WeSpeaker ResNet34 output dim
            Profile::Custom => 0,
        }
    }

    /// Default cosine similarity threshold tuned to the embedding space of this profile.
    pub const fn default_threshold(self) -> f32 {
        match self {
            Profile::Mobile => 0.55,
            Profile::Balanced => 0.45,
            Profile::Custom => 0.5,
        }
    }

    /// Stable identifier used in the manifest TOML and CLI flags.
    pub const fn manifest_id(self) -> &'static str {
        match self {
            Profile::Mobile => "mobile",
            Profile::Balanced => "balanced",
            Profile::Custom => "custom",
        }
    }
}

impl std::str::FromStr for Profile {
    type Err = ProfileParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "mobile" => Ok(Profile::Mobile),
            "balanced" => Ok(Profile::Balanced),
            "custom" => Ok(Profile::Custom),
            other => Err(ProfileParseError(other.to_owned())),
        }
    }
}

/// Returned by `Profile::from_str` when the input doesn't match a known variant.
#[derive(Debug, Clone)]
pub struct ProfileParseError(pub String);

impl std::fmt::Display for ProfileParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown profile '{}': expected mobile|balanced|custom",
            self.0
        )
    }
}

impl std::error::Error for ProfileParseError {}

/// A validated sample rate (8000–192000 Hz).
///
/// Invariant: 8000 <= inner <= 192000.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SampleRate(u32);

impl SampleRate {
    /// Create a validated sample rate.
    ///
    /// Returns `None` if the rate is outside the supported range (8000–192000 Hz).
    ///
    /// ```rust
    /// use polyvoice::SampleRate;
    /// let sr = SampleRate::new(16000).expect("valid rate");
    /// assert_eq!(sr.get(), 16000);
    /// assert!(SampleRate::new(7000).is_none());
    /// ```
    pub fn new(rate: u32) -> Option<Self> {
        (8000..=192000).contains(&rate).then_some(Self(rate))
    }

    /// Return the raw sample rate value in Hz.
    ///
    /// ```rust
    /// use polyvoice::SampleRate;
    /// let sr = SampleRate::new(44100).unwrap();
    /// assert_eq!(sr.get(), 44100);
    /// ```
    pub fn get(&self) -> u32 {
        self.0
    }
}

impl Default for SampleRate {
    fn default() -> Self {
        Self(16000)
    }
}

/// A validated confidence score in [0.0, 1.0].
///
/// Invariant: 0.0 <= inner <= 1.0.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Confidence(f32);

impl Confidence {
    /// Create a validated confidence score.
    ///
    /// Returns `None` if `v` is outside `[0.0, 1.0]`.
    ///
    /// ```rust
    /// use polyvoice::Confidence;
    /// assert!(Confidence::new(0.75).is_some());
    /// assert!(Confidence::new(1.5).is_none());
    /// ```
    pub fn new(v: f32) -> Option<Self> {
        (0.0..=1.0).contains(&v).then_some(Self(v))
    }

    /// Return the raw confidence value.
    ///
    /// ```rust
    /// use polyvoice::Confidence;
    /// let c = Confidence::new(0.9).unwrap();
    /// assert_eq!(c.get(), 0.9);
    /// ```
    pub fn get(&self) -> f32 {
        self.0
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self(1.0)
    }
}

/// A non-negative duration in seconds.
///
/// Invariant: inner >= 0.0.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Seconds(f32);

impl Seconds {
    /// Create a validated non-negative duration in seconds.
    ///
    /// Returns `None` if `v` is negative.
    ///
    /// ```rust
    /// use polyvoice::Seconds;
    /// assert!(Seconds::new(3.5).is_some());
    /// assert!(Seconds::new(-1.0).is_none());
    /// ```
    pub fn new(v: f32) -> Option<Self> {
        (v >= 0.0).then_some(Self(v))
    }

    /// Return the raw duration value in seconds.
    ///
    /// ```rust
    /// use polyvoice::Seconds;
    /// let s = Seconds::new(2.0).unwrap();
    /// assert_eq!(s.get(), 2.0);
    /// ```
    pub fn get(&self) -> f32 {
        self.0
    }
}

impl Default for Seconds {
    fn default() -> Self {
        Self(0.0)
    }
}

/// A time interval in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
}

impl TimeRange {
    /// Return the duration of this time range in seconds.
    ///
    /// ```rust
    /// use polyvoice::TimeRange;
    /// let tr = TimeRange { start: 1.0, end: 3.5 };
    /// assert_eq!(tr.duration(), 2.5);
    /// ```
    pub fn duration(&self) -> f64 {
        debug_assert!(
            self.end >= self.start,
            "TimeRange invariant violated: end < start"
        );
        self.end - self.start
    }
}

/// A speech segment with a speaker label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    /// Time range of the segment.
    pub time: TimeRange,
    /// Assigned speaker (None if not yet clustered).
    pub speaker: Option<SpeakerId>,
    /// Confidence of the speaker assignment (cosine similarity or posterior).
    pub confidence: Option<f32>,
}

/// A speaker turn: continuous stretch of speech by one speaker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerTurn {
    pub speaker: SpeakerId,
    pub time: TimeRange,
    /// Transcript text, if available from an ASR downstream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Alignment of a single word to a speaker and time range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordAlignment {
    pub word: String,
    pub time: TimeRange,
    pub speaker: Option<SpeakerId>,
    pub confidence: f32,
}

/// Result of offline diarization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizationResult {
    pub segments: Vec<Segment>,
    pub turns: Vec<SpeakerTurn>,
    pub num_speakers: usize,
}

/// Configuration shared between online and offline diarizers.
#[derive(Debug, Clone, Copy)]
pub struct DiarizationConfig {
    /// Cosine similarity threshold for assigning to an existing speaker.
    pub threshold: f32,
    /// Maximum number of speakers to track.
    pub max_speakers: usize,
    /// Window size for embedding extraction, in seconds.
    pub window_secs: f32,
    /// Hop length between consecutive windows, in seconds.
    pub hop_secs: f32,
    /// Minimum speech duration to consider for clustering, in seconds.
    pub min_speech_secs: f32,
    /// Maximum gap between same-speaker segments to merge, in seconds.
    pub max_gap_secs: f32,
    /// Sample rate expected by the embedding model (usually 16000).
    pub sample_rate: SampleRate,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            threshold: 0.4,
            max_speakers: 64,
            window_secs: 1.5,
            hop_secs: 0.75,
            min_speech_secs: 0.25,
            max_gap_secs: 0.5,
            sample_rate: SampleRate(16000),
        }
    }
}

impl DiarizationConfig {
    /// { self.window_secs >= 0.0 }
    /// `fn window_samples(&self) -> usize`
    /// { ret == (self.window_secs * self.sample_rate as f32) as usize }
    pub fn window_samples(&self) -> usize {
        (self.window_secs * self.sample_rate.get() as f32) as usize
    }

    /// { self.hop_secs >= 0.0 }
    /// `fn hop_samples(&self) -> usize`
    /// { ret == (self.hop_secs * self.sample_rate as f32) as usize }
    pub fn hop_samples(&self) -> usize {
        (self.hop_secs * self.sample_rate.get() as f32) as usize
    }

    /// { self.min_speech_secs >= 0.0 }
    /// `fn min_speech_samples(&self) -> usize`
    /// { ret == (self.min_speech_secs * self.sample_rate as f32) as usize }
    pub fn min_speech_samples(&self) -> usize {
        (self.min_speech_secs * self.sample_rate.get() as f32) as usize
    }
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn mobile_profile_uses_cam_pp_dim() {
        assert_eq!(Profile::Mobile.embedding_dim(), 512);
    }

    #[test]
    fn balanced_profile_uses_resnet34_dim() {
        assert_eq!(Profile::Balanced.embedding_dim(), 256);
    }

    #[test]
    fn custom_profile_dim_is_unresolved() {
        assert_eq!(Profile::Custom.embedding_dim(), 0);
    }

    #[test]
    fn default_thresholds_match_spec() {
        // §5.1 of v1.0 design spec
        assert!((Profile::Mobile.default_threshold() - 0.55).abs() < 1e-6);
        assert!((Profile::Balanced.default_threshold() - 0.45).abs() < 1e-6);
        assert!((Profile::Custom.default_threshold() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn manifest_id_for_each_variant() {
        assert_eq!(Profile::Mobile.manifest_id(), "mobile");
        assert_eq!(Profile::Balanced.manifest_id(), "balanced");
        assert_eq!(Profile::Custom.manifest_id(), "custom");
    }

    #[test]
    fn from_str_parses_kebab_and_lowercase() {
        assert_eq!("mobile".parse::<Profile>().unwrap(), Profile::Mobile);
        assert_eq!("Mobile".parse::<Profile>().unwrap(), Profile::Mobile);
        assert_eq!("balanced".parse::<Profile>().unwrap(), Profile::Balanced);
        assert!("nope".parse::<Profile>().is_err());
    }
}
