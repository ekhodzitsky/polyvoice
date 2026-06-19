//! Core types for speaker diarization.
//!
//! These types are shared across the offline pipeline, online diarizer, and
//! evaluation code. Start with [`DiarizationResult`] and [`SpeakerId`].

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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
    /// { true }
    /// `fn from_mapping(mapping: Vec<(SpeakerId, SpeakerId)>) -> Option<Self>`
    /// { ret.is_some() == (mapping.iter().map(|(old, _)| old).collect::<HashSet<_>>().len() == mapping.len()) }
    pub fn from_mapping(mapping: Vec<(SpeakerId, SpeakerId)>) -> Option<Self> {
        let mut seen = HashSet::with_capacity(mapping.len());
        for (old, _) in &mapping {
            if !seen.insert(old) {
                return None;
            }
        }
        Some(Self { mapping })
    }

    /// { true }
    /// pub fn remap(&self, id: SpeakerId) -> SpeakerId
    /// { ret == self.mapping.iter().find(|(old, _)| *old == id).map(|(_, new)| *new).unwrap_or(id) }
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

    /// { true }
    /// pub fn is_empty(&self) -> bool
    /// { ret == (self.mapping.len() == 0) }
    /// Returns true if no IDs were changed.
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

    /// { true }
    /// pub fn len(&self) -> usize
    /// { ret == self.mapping.len() }
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
/// Added in v0.6 (M0).
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
    /// { true }
    /// `pub fn new(rate: u32) -> Option<Self>`
    /// { ret.is_some() == (8000..=192000).contains(&rate) }
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

    /// { true }
    /// pub fn get(&self) -> u32
    /// { ret == self.0 && 8000 <= ret && ret <= 192000 }
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
    /// { true }
    /// `pub fn new(v: f32) -> Option<Self>`
    /// { ret.is_some() == (0.0..=1.0).contains(&v) }
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

    /// { true }
    /// pub fn get(&self) -> f32
    /// { ret == self.0 && 0.0 <= ret && ret <= 1.0 }
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
    /// { true }
    /// `pub fn new(v: f32) -> Option<Self>`
    /// { ret.is_some() == (v >= 0.0) }
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

    /// { true }
    /// pub fn get(&self) -> f32
    /// { ret == self.0 && ret >= 0.0 }
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
    /// { true }
    /// pub fn duration(&self) -> f64
    /// { ret >= 0.0 }
    /// Return the duration of this time range in seconds.
    ///
    /// ```rust
    /// use polyvoice::TimeRange;
    /// let tr = TimeRange { start: 1.0, end: 3.5 };
    /// assert_eq!(tr.duration(), 2.5);
    /// ```
    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
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

/// A single transcribed word with its time span and ASR confidence.
///
/// This is the raw ASR output (no speaker yet). The word→speaker join attributes
/// each `Word` to a [`SpeakerTurn`], producing a [`WordAlignment`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    /// The recognized word text.
    pub word: String,
    /// Time span of the word.
    pub time: TimeRange,
    /// ASR confidence in [0.0, 1.0].
    pub confidence: f32,
}

/// A raw ASR transcript: an ordered list of [`Word`]s (no speaker labels yet).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Transcript {
    pub words: Vec<Word>,
}

/// Audio metadata for a [`DiarizationResult`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AudioMeta {
    /// Audio duration in seconds.
    pub duration_secs: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

/// Provenance for a [`DiarizationResult`]: how it was produced.
///
/// `version` is always set by [`DiarizationResult::new`]; `profile` is set when the
/// producing pipeline knows it. The model-id fields (`segmenter`/`embedder`/
/// `clusterer`) are populated when the model registry is threaded through — an
/// empty string means "not recorded".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Provenance {
    /// Crate version that produced the result.
    pub version: String,
    /// Profile id (e.g. "balanced"), or empty if not recorded.
    pub profile: String,
    /// Segmentation/VAD model id, or empty.
    pub segmenter: String,
    /// Embedding model id, or empty.
    pub embedder: String,
    /// Clustering backend id, or empty.
    pub clusterer: String,
}

/// Per-speaker rollup for a [`DiarizationResult`], exposing the speaker both as a
/// numeric `id` and the canonical `SPEAKER_NN` `label`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerSummary {
    /// Canonical string label, e.g. "SPEAKER_00".
    pub label: String,
    /// Numeric speaker id.
    pub id: u32,
    /// Total speech attributed to this speaker, in seconds.
    pub total_speech_s: f64,
    /// Number of turns for this speaker.
    pub turn_count: usize,
}

fn default_schema_version() -> String {
    "diarization-result-v1".to_owned()
}

/// Result of offline diarization — the canonical v1 result type that every
/// surface (RTTM/JSON/SRT/VTT/TXT, CLI, MCP) projects from.
///
/// The metadata fields (`schema_version`, `audio`, `provenance`, `speakers`) are
/// additive and `#[serde(default)]`, so older JSON without them still
/// deserializes. Construct via [`DiarizationResult::new`] so `speakers` and
/// `schema_version` are always populated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizationResult {
    pub segments: Vec<Segment>,
    pub turns: Vec<SpeakerTurn>,
    pub num_speakers: usize,
    /// Schema identifier for downstream consumers.
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    /// Audio metadata (duration, sample rate).
    #[serde(default)]
    pub audio: AudioMeta,
    /// How this result was produced.
    #[serde(default)]
    pub provenance: Provenance,
    /// Per-speaker rollup (sorted by id); exposes id AND the SPEAKER_NN label.
    #[serde(default)]
    pub speakers: Vec<SpeakerSummary>,
}

impl DiarizationResult {
    /// Build a v1 result from the core fields, computing the per-speaker
    /// `speakers` rollup from `turns` and stamping `schema_version` +
    /// `provenance.version`. Refine `audio`/`provenance` with
    /// [`DiarizationResult::with_audio`] / [`DiarizationResult::with_provenance`].
    pub fn new(segments: Vec<Segment>, turns: Vec<SpeakerTurn>, num_speakers: usize) -> Self {
        let speakers = speaker_summaries(&turns);
        Self {
            segments,
            turns,
            num_speakers,
            schema_version: default_schema_version(),
            audio: AudioMeta::default(),
            provenance: Provenance {
                version: env!("CARGO_PKG_VERSION").to_owned(),
                ..Provenance::default()
            },
            speakers,
        }
    }

    /// Attach audio metadata (builder).
    pub fn with_audio(mut self, duration_secs: f64, sample_rate: u32) -> Self {
        self.audio = AudioMeta {
            duration_secs,
            sample_rate,
        };
        self
    }

    /// Attach producer provenance (builder). Keeps the existing `version` when the
    /// supplied `provenance.version` is empty.
    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        let version = if provenance.version.is_empty() {
            self.provenance.version.clone()
        } else {
            provenance.version.clone()
        };
        self.provenance = Provenance {
            version,
            ..provenance
        };
        self
    }
}

/// Per-speaker rollup (total speech + turn count) from turns, sorted by numeric
/// speaker id, labelling each via the `SpeakerId` `Display`.
fn speaker_summaries(turns: &[SpeakerTurn]) -> Vec<SpeakerSummary> {
    use std::collections::BTreeMap;
    let mut agg: BTreeMap<u32, (f64, usize)> = BTreeMap::new();
    for t in turns {
        let e = agg.entry(t.speaker.0).or_insert((0.0, 0));
        e.0 += t.time.duration();
        e.1 += 1;
    }
    agg.into_iter()
        .map(|(id, (total, count))| SpeakerSummary {
            label: SpeakerId(id).to_string(),
            id,
            total_speech_s: total,
            turn_count: count,
        })
        .collect()
}

/// Configuration for speaker clustering.
#[derive(Debug, Clone, Copy)]
pub struct ClusterConfig {
    /// Cosine similarity threshold: clusters whose centroids are at least this
    /// similar are merged by the agglomerative clusterer. Higher = stricter =
    /// more (smaller) clusters.
    pub threshold: f32,
    /// Maximum number of speakers to track.
    pub max_speakers: usize,
    /// Minimum members a cluster must have to survive. After clustering, any
    /// cluster smaller than this is dissolved and its frames reassigned to the
    /// nearest large speaker centroid. This prunes spurious tiny clusters that
    /// inflate the speaker count without hurting frame-DER. `1` disables pruning.
    /// Ignored when `min_cluster_secs > 0` (duration pruning takes precedence).
    pub min_cluster_size: usize,
    /// Minimum total speech duration (seconds) a cluster must have to survive —
    /// the length-invariant alternative to `min_cluster_size`. When `> 0`, a
    /// cluster whose overlap-merged window duration is below this is dissolved.
    /// `0.0` disables it (the member-count rule applies instead).
    pub min_cluster_secs: f64,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            threshold: 0.45,
            max_speakers: 64,
            // Pruning singleton clusters (size < 2) cuts over-clustering and
            // lowers DER on real-length audio (VoxConverse-dev collar
            // 7.97%→7.22%, speaker-count off-by-2+ 58→20 on the dev-80 sweep)
            // while staying safe on short clips: a fixed min of 3-4 wins more on
            // long files but wrongly dissolves real minority speakers on short
            // ones (the bundled 26 s clip regresses 6.62%→9.54% at min 3). A
            // length-aware / duration-based prune for the larger gain is future
            // work (see `min_cluster_secs`). `1` disables pruning.
            min_cluster_size: 2,
            // Duration pruning off by default until calibrated; the validated
            // shipped default is the member-count rule above.
            min_cluster_secs: 0.0,
        }
    }
}

/// Configuration for sliding-window embedding extraction.
#[derive(Debug, Clone, Copy)]
pub struct WindowConfig {
    /// Window size for embedding extraction, in seconds.
    pub window_secs: f32,
    /// Hop length between consecutive windows, in seconds.
    pub hop_secs: f32,
    /// Sample rate expected by the embedding model (usually 16000).
    pub sample_rate: SampleRate,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_secs: 1.5,
            hop_secs: 0.75,
            sample_rate: SampleRate(16000),
        }
    }
}

impl WindowConfig {
    /// { self.window_secs >= 0.0 }
    /// `fn window_samples(&self) -> usize`
    /// { ret == (self.window_secs * self.sample_rate.get() as f32) as usize }
    pub fn window_samples(&self) -> usize {
        (self.window_secs * self.sample_rate.get() as f32) as usize
    }

    /// { self.hop_secs >= 0.0 }
    /// `fn hop_samples(&self) -> usize`
    /// { ret == (self.hop_secs * self.sample_rate.get() as f32) as usize }
    pub fn hop_samples(&self) -> usize {
        (self.hop_secs * self.sample_rate.get() as f32) as usize
    }
}

/// Configuration for post-clustering speech filtering.
#[derive(Debug, Clone, Copy)]
pub struct SpeechFilterConfig {
    /// Minimum speech duration to consider for clustering, in seconds.
    pub min_speech_secs: f32,
    /// Maximum gap between same-speaker segments to merge, in seconds.
    pub max_gap_secs: f32,
}

impl Default for SpeechFilterConfig {
    fn default() -> Self {
        Self {
            min_speech_secs: 0.25,
            max_gap_secs: 0.5,
        }
    }
}

/// Configuration shared between online and offline diarizers.
#[derive(Debug, Clone, Copy)]
pub struct DiarizationConfig {
    pub cluster: ClusterConfig,
    pub window: WindowConfig,
    pub speech_filter: SpeechFilterConfig,
    /// Maximum allowed audio duration in seconds (DoS guard).
    pub max_duration_secs: f32,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            cluster: ClusterConfig::default(),
            window: WindowConfig::default(),
            speech_filter: SpeechFilterConfig::default(),
            max_duration_secs: 3600.0,
        }
    }
}

impl DiarizationConfig {
    /// { self.window.window_secs >= 0.0 }
    /// `fn window_samples(&self) -> usize`
    /// { ret == self.window.window_samples() }
    pub fn window_samples(&self) -> usize {
        self.window.window_samples()
    }

    /// { self.window.hop_secs >= 0.0 }
    /// `fn hop_samples(&self) -> usize`
    /// { ret == self.window.hop_samples() }
    pub fn hop_samples(&self) -> usize {
        self.window.hop_samples()
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod speaker_id_remap_tests {
    use super::*;

    #[test]
    fn from_mapping_accepts_unique_old_ids() {
        let mapping = vec![
            (SpeakerId(0), SpeakerId(0)),
            (SpeakerId(1), SpeakerId(0)),
            (SpeakerId(2), SpeakerId(1)),
        ];
        let remap = SpeakerIdRemap::from_mapping(mapping).unwrap();
        assert_eq!(remap.len(), 3);
        assert_eq!(remap.remap(SpeakerId(0)), SpeakerId(0));
        assert_eq!(remap.remap(SpeakerId(1)), SpeakerId(0));
        assert_eq!(remap.remap(SpeakerId(2)), SpeakerId(1));
        assert_eq!(remap.remap(SpeakerId(99)), SpeakerId(99));
    }

    #[test]
    fn from_mapping_rejects_duplicate_old_ids() {
        let mapping = vec![(SpeakerId(0), SpeakerId(1)), (SpeakerId(0), SpeakerId(2))];
        assert!(SpeakerIdRemap::from_mapping(mapping).is_none());
    }
}

#[allow(clippy::unwrap_used)]
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

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod diarization_result_tests {
    use super::*;

    fn turn(id: u32, start: f64, end: f64) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(id),
            time: TimeRange { start, end },
            text: None,
        }
    }

    #[test]
    fn new_stamps_schema_version_and_provenance_version() {
        let r = DiarizationResult::new(vec![], vec![], 0);
        assert_eq!(r.schema_version, "diarization-result-v1");
        assert_eq!(r.provenance.version, env!("CARGO_PKG_VERSION"));
        assert!(r.speakers.is_empty());
    }

    #[test]
    fn speakers_rollup_matches_turns_with_dual_id() {
        let turns = vec![turn(0, 0.0, 2.0), turn(1, 2.0, 5.0), turn(0, 6.0, 7.0)];
        let r = DiarizationResult::new(vec![], turns, 2);
        assert_eq!(r.speakers.len(), 2);
        // Dual representation: numeric id AND canonical string label.
        assert_eq!(r.speakers[0].id, 0);
        assert_eq!(r.speakers[0].label, "SPEAKER_00");
        assert_eq!(r.speakers[0].turn_count, 2);
        assert!((r.speakers[0].total_speech_s - 3.0).abs() < 1e-9); // 2.0 + 1.0
        assert_eq!(r.speakers[1].id, 1);
        assert_eq!(r.speakers[1].label, "SPEAKER_01");
        assert_eq!(r.speakers[1].turn_count, 1);
        assert!((r.speakers[1].total_speech_s - 3.0).abs() < 1e-9);
    }

    #[test]
    fn old_json_without_metadata_deserializes() {
        // JSON shaped like the pre-v1 result (no metadata fields).
        let json = r#"{"segments":[],"turns":[],"num_speakers":0}"#;
        let r: DiarizationResult = serde_json::from_str(json).unwrap();
        assert_eq!(r.num_speakers, 0);
        assert_eq!(r.schema_version, "diarization-result-v1"); // serde default
        assert_eq!(r.audio, AudioMeta::default());
        assert_eq!(r.provenance, Provenance::default());
        assert!(r.speakers.is_empty());
    }

    #[test]
    fn round_trips_through_json_with_builders() {
        let r = DiarizationResult::new(vec![], vec![turn(0, 0.0, 1.0)], 1)
            .with_audio(12.5, 16000)
            .with_provenance(Provenance {
                profile: "balanced".to_owned(),
                ..Provenance::default()
            });
        let json = serde_json::to_string(&r).unwrap();
        let back: DiarizationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
        assert_eq!(back.audio.sample_rate, 16000);
        assert_eq!(back.provenance.profile, "balanced");
        // version preserved by the builder when the supplied one is empty.
        assert_eq!(back.provenance.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn word_and_transcript_round_trip() {
        let t = Transcript {
            words: vec![
                Word { word: "hello".into(), time: TimeRange { start: 0.0, end: 0.4 }, confidence: 0.95 },
                Word { word: "world".into(), time: TimeRange { start: 0.4, end: 0.9 }, confidence: 0.88 },
            ],
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
        assert_eq!(back.words.len(), 2);
        assert_eq!(back.words[0].word, "hello");
        assert_eq!(Transcript::default().words.len(), 0);
    }
}

#[cfg(kani)]
mod kani_proofs;
