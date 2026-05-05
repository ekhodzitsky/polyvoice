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

/// A validated embedding dimension (> 0).
///
/// Invariant: inner > 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingDim(usize);

impl EmbeddingDim {
    /// Create a validated embedding dimension.
    ///
    /// Returns `None` if `dim` is zero.
    ///
    /// ```rust
    /// use polyvoice::EmbeddingDim;
    /// assert!(EmbeddingDim::new(256).is_some());
    /// assert!(EmbeddingDim::new(0).is_none());
    /// ```
    pub fn new(dim: usize) -> Option<Self> {
        (dim > 0).then_some(Self(dim))
    }

    /// Return the raw dimension value.
    ///
    /// ```rust
    /// use polyvoice::EmbeddingDim;
    /// let d = EmbeddingDim::new(192).unwrap();
    /// assert_eq!(d.get(), 192);
    /// ```
    pub fn get(&self) -> usize {
        self.0
    }
}

impl Default for EmbeddingDim {
    fn default() -> Self {
        Self(256)
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
            threshold: 0.5,
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
