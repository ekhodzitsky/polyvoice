//! Core types for speaker diarization.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Opaque identifier for a speaker cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpeakerId(pub u32);

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
    /// { 8000 <= rate && rate <= 192000 }
    /// `fn new(rate: u32) -> Option<SampleRate>`
    /// { ret.is_some() => ret.unwrap().0 == rate }
    pub fn new(rate: u32) -> Option<Self> {
        (8000..=192000).contains(&rate).then_some(Self(rate))
    }

    /// { true }
    /// `fn get(&self) -> u32`
    /// { ret == self.0 }
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
    /// { 0.0 <= v && v <= 1.0 }
    /// `fn new(v: f32) -> Option<Confidence>`
    /// { ret.is_some() => ret.unwrap().0 == v }
    pub fn new(v: f32) -> Option<Self> {
        (0.0..=1.0).contains(&v).then_some(Self(v))
    }

    /// { true }
    /// `fn get(&self) -> f32`
    /// { ret == self.0 }
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
    /// { dim > 0 }
    /// `fn new(dim: usize) -> Option<EmbeddingDim>`
    /// { ret.is_some() => ret.unwrap().0 == dim }
    pub fn new(dim: usize) -> Option<Self> {
        (dim > 0).then_some(Self(dim))
    }

    /// { true }
    /// `fn get(&self) -> usize`
    /// { ret == self.0 }
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
    /// { v >= 0.0 }
    /// `fn new(v: f32) -> Option<Seconds>`
    /// { ret.is_some() => ret.unwrap().0 == v }
    pub fn new(v: f32) -> Option<Self> {
        (v >= 0.0).then_some(Self(v))
    }

    /// { true }
    /// `fn get(&self) -> f32`
    /// { ret == self.0 }
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
    /// { self.end >= self.start }
    /// `fn duration(&self) -> f64`
    /// { ret >= 0.0 }
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
