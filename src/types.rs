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

/// A time interval in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
}

impl TimeRange {
    /// Duration in seconds.
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
    /// Sample rate expected by the embedding model (usually 16000).
    pub sample_rate: u32,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            max_speakers: 64,
            window_secs: 1.5,
            hop_secs: 0.75,
            min_speech_secs: 0.25,
            sample_rate: 16000,
        }
    }
}

impl DiarizationConfig {
    /// Number of samples per analysis window.
    pub fn window_samples(&self) -> usize {
        (self.window_secs * self.sample_rate as f32) as usize
    }

    /// Number of samples per hop.
    pub fn hop_samples(&self) -> usize {
        (self.hop_secs * self.sample_rate as f32) as usize
    }

    /// Number of samples for minimum speech duration.
    pub fn min_speech_samples(&self) -> usize {
        (self.min_speech_secs * self.sample_rate as f32) as usize
    }
}
