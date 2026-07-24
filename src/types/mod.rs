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
    /// True when the word's timestamps were filled in by nearest-neighbor
    /// interpolation because the ASR left them missing or zero-duration.
    /// Downstream consumers can ignore or down-weight these if desired.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub interpolated: bool,
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
    /// Optional L2-normalized mean embedding for this speaker (opt-in export).
    ///
    /// Additive: absent when embeddings were not requested. Shape is the
    /// embedder dimension (e.g. 256 for ResNet34). Intended for downstream
    /// identification / voiceprint consumers — not identification itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

fn default_schema_version() -> String {
    "diarization-result-v1".to_owned()
}

/// Result of offline diarization — the canonical v1 result type that every
/// surface (RTTM/JSON/SRT/VTT/TXT, CLI, MCP) projects from.
///
/// The metadata fields (`schema_version`, `audio`, `provenance`, `speakers`,
/// `exclusive_turns`) are additive and `#[serde(default)]`, so older JSON
/// without them still deserializes. Construct via [`DiarizationResult::new`] so
/// `speakers` and `schema_version` are always populated.
///
/// Schema family stays `diarization-result-v1`: new fields are optional and
/// omitted when empty, so consumers that ignore unknown keys remain compatible.
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
    /// Single-speaker (exclusive) timeline derived from [`Self::turns`].
    ///
    /// At most one speaker is active at every frame — the ASR-reconciliation
    /// surface (overlap is collapsed by a deterministic per-frame argmax). Empty
    /// unless filled via [`DiarizationResult::with_exclusive`]. The overlap-aware
    /// [`Self::turns`] field is always the primary diarization output.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclusive_turns: Vec<SpeakerTurn>,
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
            exclusive_turns: Vec::new(),
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

    /// Fill [`Self::exclusive_turns`] from the overlap-aware [`Self::turns`].
    ///
    /// Does not modify `turns` / `segments`. Safe to call multiple times.
    pub fn with_exclusive(mut self) -> Self {
        self.exclusive_turns = exclusive_turns(&self.turns);
        self
    }

    /// Attach L2-normalized per-speaker embeddings onto [`Self::speakers`].
    ///
    /// Speakers with no matching entry keep `embedding: None`. Vectors are
    /// cloned and re-normalized so callers need not pre-normalize.
    pub fn with_speaker_embeddings(mut self, embeddings: &[(SpeakerId, Vec<f32>)]) -> Self {
        for sp in &mut self.speakers {
            if let Some((_, emb)) = embeddings.iter().find(|(id, _)| id.0 == sp.id) {
                let mut v = emb.clone();
                crate::utils::l2_normalize(&mut v);
                sp.embedding = Some(v);
            }
        }
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
            embedding: None,
        })
        .collect()
}

/// Frame resolution (seconds) used by exclusive-mode conversion — matches DER.
const EXCLUSIVE_FRAME_SECS: f64 = 0.01;

/// Derive a single-speaker (exclusive) timeline from overlap-aware turns.
///
/// For every 10 ms frame with any active speech, exactly one speaker is chosen:
/// among speakers covering the frame, pick the one whose covering turn is
/// longest (dominant claim); ties break to the smaller [`SpeakerId`]. Consecutive
/// frames with the same speaker are collapsed into turns. Silence frames produce
/// no output.
///
/// This is the ASR-reconciliation surface (cf. pyannote exclusive diarization):
/// on overlap-heavy audio the second concurrent speaker is dropped by design.
///
/// Coverage of the exclusive timeline equals the union of the input speech
/// frames (every speech frame keeps one speaker).
pub fn exclusive_turns(turns: &[SpeakerTurn]) -> Vec<SpeakerTurn> {
    if turns.is_empty() {
        return Vec::new();
    }
    let max_time = turns.iter().map(|t| t.time.end).fold(0.0f64, f64::max);
    if !max_time.is_finite() || max_time <= 0.0 {
        return Vec::new();
    }
    // Cap at 24 h of frames (same guard as DER) to avoid unbounded allocation.
    const MAX_FRAMES: usize = 24 * 3600 * 100;
    let n_frames = ((max_time / EXCLUSIVE_FRAME_SECS).ceil() as usize + 1).min(MAX_FRAMES);

    // Per frame: best (speaker, covering_turn_duration). None = silence.
    let mut best: Vec<Option<(u32, f64)>> = vec![None; n_frames];
    for turn in turns {
        if !turn.time.start.is_finite()
            || !turn.time.end.is_finite()
            || turn.time.end <= turn.time.start
        {
            continue;
        }
        let dur = turn.time.duration();
        let start_f = (turn.time.start / EXCLUSIVE_FRAME_SECS).max(0.0) as usize;
        let end_f = (turn.time.end / EXCLUSIVE_FRAME_SECS).ceil().max(0.0) as usize;
        for frame in best.iter_mut().take(end_f.min(n_frames)).skip(start_f) {
            match frame {
                None => *frame = Some((turn.speaker.0, dur)),
                Some((spk, best_dur)) => {
                    // Longer covering turn wins; tie-break smaller speaker id.
                    if dur > *best_dur + f64::EPSILON
                        || ((dur - *best_dur).abs() <= f64::EPSILON && turn.speaker.0 < *spk)
                    {
                        *frame = Some((turn.speaker.0, dur));
                    }
                }
            }
        }
    }

    // Collapse consecutive same-speaker frames into turns.
    let mut out: Vec<SpeakerTurn> = Vec::new();
    let mut i = 0usize;
    while i < n_frames {
        let Some((spk, _)) = best[i] else {
            i += 1;
            continue;
        };
        let start = i;
        i += 1;
        while i < n_frames {
            match best[i] {
                Some((s, _)) if s == spk => i += 1,
                _ => break,
            }
        }
        out.push(SpeakerTurn {
            speaker: SpeakerId(spk),
            time: TimeRange {
                start: start as f64 * EXCLUSIVE_FRAME_SECS,
                end: i as f64 * EXCLUSIVE_FRAME_SECS,
            },
            text: None,
        });
    }
    out
}

/// Default midpoint for [`confidence_from_similarity`] (cosine similarity scale).
pub const CONFIDENCE_SIM_MIDPOINT: f32 = 0.5;
/// Default steepness for the logistic confidence curve.
pub const CONFIDENCE_SIM_STEEPNESS: f32 = 10.0;

/// Map a cosine similarity in `[-1, 1]` to a confidence score in `(0, 1]`.
///
/// Uses a fixed logistic curve centered at [`CONFIDENCE_SIM_MIDPOINT`] with
/// slope [`CONFIDENCE_SIM_STEEPNESS`]. **Monotone increasing** in similarity
/// (equivalently monotone decreasing in cosine distance `1 − sim`).
///
/// This is a cheap heuristic for ranking / low-confidence triage, **not** a
/// calibrated probability of label correctness. Full isotonic calibration needs
/// labeled dev data and is intentionally not hard-coded here.
///
/// Non-finite inputs map to confidence near 0.
pub fn confidence_from_similarity(sim: f32) -> f32 {
    confidence_from_similarity_params(sim, CONFIDENCE_SIM_MIDPOINT, CONFIDENCE_SIM_STEEPNESS)
}

/// Logistic confidence from cosine similarity with explicit midpoint/steepness.
///
/// `steepness` must be positive for the intended "higher sim → higher conf"
/// direction; non-positive values are treated as [`CONFIDENCE_SIM_STEEPNESS`].
pub fn confidence_from_similarity_params(sim: f32, midpoint: f32, steepness: f32) -> f32 {
    let s = if sim.is_finite() {
        sim.clamp(-1.0, 1.0)
    } else {
        -1.0
    };
    let k = if steepness.is_finite() && steepness > 0.0 {
        steepness
    } else {
        CONFIDENCE_SIM_STEEPNESS
    };
    let m = if midpoint.is_finite() {
        midpoint
    } else {
        CONFIDENCE_SIM_MIDPOINT
    };
    let x = k * (s - m);
    // sigmoid(x) = 1 / (1 + e^{-x}); clamp for numerical safety.
    let conf = if x >= 20.0 {
        1.0
    } else if x <= -20.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    };
    conf.clamp(0.0, 1.0)
}

/// Confidence from cosine distance `d = 1 − sim` (L2-normalized embeddings).
///
/// Monotone **decreasing** in `distance`. Equivalent to
/// [`confidence_from_similarity`]`(1 − distance)`.
pub fn confidence_from_distance(distance: f32) -> f32 {
    let d = if distance.is_finite() { distance } else { 2.0 };
    confidence_from_similarity(1.0 - d)
}

/// Mean L2-normalized embedding per speaker from parallel label/embedding slices.
///
/// Speakers appear sorted by numeric id. Empty / mismatched input yields an empty
/// vec. Each output vector is L2-normalized.
pub fn mean_speaker_embeddings(
    labels: &[SpeakerId],
    embeddings: &[Vec<f32>],
) -> Vec<(SpeakerId, Vec<f32>)> {
    use std::collections::BTreeMap;
    if labels.is_empty() || embeddings.is_empty() {
        return Vec::new();
    }
    let n = labels.len().min(embeddings.len());
    let mut sums: BTreeMap<u32, (Vec<f32>, usize)> = BTreeMap::new();
    for i in 0..n {
        let emb = &embeddings[i];
        if emb.is_empty() || emb.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let id = labels[i].0;
        let entry = sums.entry(id).or_insert_with(|| (vec![0.0; emb.len()], 0));
        if entry.0.len() != emb.len() {
            continue; // dimension mismatch — skip
        }
        for (s, &v) in entry.0.iter_mut().zip(emb.iter()) {
            *s += v;
        }
        entry.1 += 1;
    }
    sums.into_iter()
        .filter_map(|(id, (mut sum, count))| {
            if count == 0 {
                return None;
            }
            let inv = 1.0 / count as f32;
            for v in &mut sum {
                *v *= inv;
            }
            crate::utils::l2_normalize(&mut sum);
            Some((SpeakerId(id), sum))
        })
        .collect()
}

/// Per-embedding confidence from cosine similarity to the speaker's mean centroid.
///
/// Returns one score per pair in `labels.zip(embeddings)` (length
/// `min(labels.len(), embeddings.len())`). Embeddings whose label has no usable
/// centroid get confidence `0.0`.
pub fn segment_confidences_from_embeddings(
    labels: &[SpeakerId],
    embeddings: &[Vec<f32>],
) -> Vec<f32> {
    let centroids = mean_speaker_embeddings(labels, embeddings);
    let n = labels.len().min(embeddings.len());
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let Some((_, centroid)) = centroids.iter().find(|(id, _)| *id == labels[i]) else {
            continue;
        };
        let sim = crate::utils::cosine_similarity(&embeddings[i], centroid);
        out[i] = confidence_from_similarity(sim);
    }
    out
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
                Word {
                    word: "hello".into(),
                    time: TimeRange {
                        start: 0.0,
                        end: 0.4,
                    },
                    confidence: 0.95,
                },
                Word {
                    word: "world".into(),
                    time: TimeRange {
                        start: 0.4,
                        end: 0.9,
                    },
                    confidence: 0.88,
                },
            ],
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
        assert_eq!(back.words.len(), 2);
        assert_eq!(back.words[0].word, "hello");
        assert_eq!(Transcript::default().words.len(), 0);
    }

    #[test]
    fn exclusive_collapses_overlap_to_one_speaker_per_frame() {
        // Overlap on [2, 4): both spk0 and spk1 active. spk0 turn is longer (0-4)
        // than spk1 (2-6), so exclusive should keep spk0 on the overlap.
        let turns = vec![turn(0, 0.0, 4.0), turn(1, 2.0, 6.0)];
        let ex = exclusive_turns(&turns);
        // Frame check: never two speakers.
        assert_exclusive_one_speaker(&ex, 6.0);
        // Speech coverage equals the union [0, 6).
        let speech: f64 = ex.iter().map(|t| t.time.duration()).sum();
        assert!(
            (speech - 6.0).abs() < 0.02,
            "exclusive speech coverage should match union, got {speech}"
        );
    }

    #[test]
    fn exclusive_no_overlap_is_identity_up_to_frame_quantize() {
        let turns = vec![turn(0, 0.0, 2.0), turn(1, 2.0, 4.0)];
        let ex = exclusive_turns(&turns);
        assert_exclusive_one_speaker(&ex, 4.0);
        assert_eq!(ex.len(), 2);
        assert_eq!(ex[0].speaker, SpeakerId(0));
        assert_eq!(ex[1].speaker, SpeakerId(1));
    }

    #[test]
    fn with_exclusive_populates_field_without_touching_turns() {
        let turns = vec![turn(0, 0.0, 3.0), turn(1, 2.0, 5.0)];
        let r = DiarizationResult::new(vec![], turns.clone(), 2).with_exclusive();
        assert_eq!(r.turns, turns);
        assert!(!r.exclusive_turns.is_empty());
        assert_exclusive_one_speaker(&r.exclusive_turns, 5.0);
        // Empty exclusive_turns is omitted from JSON.
        let bare = DiarizationResult::new(vec![], turns, 2);
        let json = serde_json::to_string(&bare).unwrap();
        assert!(!json.contains("exclusive_turns"));
        let with = bare.with_exclusive();
        let json2 = serde_json::to_string(&with).unwrap();
        assert!(json2.contains("exclusive_turns"));
    }

    #[test]
    fn confidence_from_similarity_is_monotone() {
        let sims = [-1.0f32, -0.5, 0.0, 0.3, 0.5, 0.7, 0.9, 1.0];
        let mut prev = -1.0f32;
        for &s in &sims {
            let c = confidence_from_similarity(s);
            assert!(
                (0.0..=1.0).contains(&c),
                "conf {c} out of range for sim {s}"
            );
            assert!(
                c + 1e-6 >= prev,
                "not monotone: sim {s} conf {c} < prev {prev}"
            );
            prev = c;
        }
        // Larger distance → lower confidence.
        let d_small = confidence_from_distance(0.1);
        let d_large = confidence_from_distance(0.8);
        assert!(d_small > d_large, "{d_small} should beat {d_large}");
    }

    #[test]
    fn mean_speaker_embeddings_are_l2_normalized_and_deterministic() {
        let labels = [SpeakerId(0), SpeakerId(0), SpeakerId(1), SpeakerId(1)];
        let embeddings = vec![
            vec![3.0, 0.0],
            vec![0.0, 4.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
        ];
        let a = mean_speaker_embeddings(&labels, &embeddings);
        let b = mean_speaker_embeddings(&labels, &embeddings);
        assert_eq!(a, b, "must be deterministic");
        assert_eq!(a.len(), 2);
        for (_, emb) in &a {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
        }
        // Speaker 1 mean is [1,0] already unit.
        assert!((a[1].1[0] - 1.0).abs() < 1e-5);
        // Attach to result.
        let r = DiarizationResult::new(vec![], vec![turn(0, 0.0, 1.0), turn(1, 1.0, 2.0)], 2)
            .with_speaker_embeddings(&a);
        assert!(r.speakers[0].embedding.is_some());
        assert!(r.speakers[1].embedding.is_some());
        let confs = segment_confidences_from_embeddings(&labels, &embeddings);
        assert_eq!(confs.len(), 4);
        assert!(confs.iter().all(|&c| (0.0..=1.0).contains(&c)));
    }

    fn assert_exclusive_one_speaker(turns: &[SpeakerTurn], max_time: f64) {
        let n = ((max_time / EXCLUSIVE_FRAME_SECS).ceil() as usize) + 1;
        let mut counts = vec![0u32; n];
        for t in turns {
            let s = (t.time.start / EXCLUSIVE_FRAME_SECS) as usize;
            let e = (t.time.end / EXCLUSIVE_FRAME_SECS).ceil() as usize;
            for c in counts.iter_mut().take(e.min(n)).skip(s) {
                *c += 1;
            }
        }
        assert!(
            counts.iter().all(|&c| c <= 1),
            "exclusive timeline has dual-speaker frames"
        );
    }
}

#[cfg(kani)]
mod kani_proofs;
