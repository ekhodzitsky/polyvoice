//! Segments, turns, and diarization result schema.
use super::ids::{SpeakerId, SpeakerIdRemap};
use super::measures::TimeRange;
use serde::{Deserialize, Serialize};

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

fn default_turn_stable() -> bool {
    true
}

fn is_true(v: &bool) -> bool {
    *v
}

/// A speaker turn: continuous stretch of speech by one speaker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerTurn {
    pub speaker: SpeakerId,
    pub time: TimeRange,
    /// Transcript text, if available from an ASR downstream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Whether the speaker label has stabilized.
    ///
    /// Offline pipeline turns are always `true`. Streaming may emit provisional
    /// labels (`stable: false`) that can still change until the speaker cache
    /// reaches its stability threshold; once `true`, the label for that speaker
    /// identity is treated as immutable. Absent in older JSON (defaults to
    /// `true`); omitted from serialization when `true` so offline payloads stay
    /// unchanged.
    #[serde(default = "default_turn_stable", skip_serializing_if = "is_true")]
    pub stable: bool,
}

impl SpeakerTurn {
    /// Construct a turn with no transcript text and `stable: true` (offline default).
    pub fn new(speaker: SpeakerId, time: TimeRange) -> Self {
        Self {
            speaker,
            time,
            text: None,
            stable: true,
        }
    }

    /// Construct a turn with an explicit stability flag (used by streaming).
    pub fn with_stability(speaker: SpeakerId, time: TimeRange, stable: bool) -> Self {
        Self {
            speaker,
            time,
            text: None,
            stable,
        }
    }
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

/// Frame-grid evaluation support: a uniform 10 ms grid over the timeline,
/// shared by DER scoring and the exclusive-timeline projection below.
///
/// The associated items live next to the grid's heaviest consumer
/// ([`exclusive_turns`]); `TimeRange` itself is defined in `super::measures`.
impl TimeRange {
    /// Frame-grid resolution in seconds (10 ms).
    pub(crate) const FRAME_GRID_RESOLUTION_SECS: f64 = 0.01;

    /// Hard cap on grid frames: 24 hours at [`Self::FRAME_GRID_RESOLUTION_SECS`],
    /// guarding against unbounded allocation on malformed or huge timelines.
    pub(crate) const MAX_GRID_FRAMES: usize = 24 * 3600 * 100;

    /// Number of grid frames covering `max_time` seconds:
    /// `ceil(max_time / resolution) + 1`, capped at [`Self::MAX_GRID_FRAMES`].
    /// Callers must validate `max_time` (finite, non-negative) beforehand.
    pub(crate) fn grid_frame_count(max_time: f64) -> usize {
        ((max_time / Self::FRAME_GRID_RESOLUTION_SECS).ceil() as usize + 1)
            .min(Self::MAX_GRID_FRAMES)
    }

    /// Frame-index range `[start, end)` this range covers on the grid. The end
    /// index is ceiled so the range covers every frame its end timestamp
    /// touches. Negative or NaN coordinates saturate to frame 0 (Rust
    /// float→int cast semantics), matching the historical behavior.
    pub(crate) fn grid_frame_range(&self) -> (usize, usize) {
        (
            (self.start / Self::FRAME_GRID_RESOLUTION_SECS) as usize,
            (self.end / Self::FRAME_GRID_RESOLUTION_SECS).ceil() as usize,
        )
    }
}

/// Frame resolution (seconds) used by exclusive-mode conversion — alias for
/// the shared evaluation-grid resolution ([`TimeRange::FRAME_GRID_RESOLUTION_SECS`]).
pub(crate) const EXCLUSIVE_FRAME_SECS: f64 = TimeRange::FRAME_GRID_RESOLUTION_SECS;

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
    // Capped at 24 h of frames (same guard as DER) to avoid unbounded allocation.
    let n_frames = TimeRange::grid_frame_count(max_time);

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
        let (start_f, end_f) = turn.time.grid_frame_range();
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
        out.push(SpeakerTurn::new(
            SpeakerId(spk),
            TimeRange {
                start: start as f64 * EXCLUSIVE_FRAME_SECS,
                end: i as f64 * EXCLUSIVE_FRAME_SECS,
            },
        ));
    }
    out
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, speaker: Option<u32>) -> Segment {
        Segment {
            time: TimeRange { start, end },
            speaker: speaker.map(SpeakerId),
            confidence: Some(0.9),
        }
    }

    fn turn_range(id: u32, start: f64, end: f64) -> SpeakerTurn {
        SpeakerTurn::new(SpeakerId(id), TimeRange { start, end })
    }

    /// Two speakers: id 0 speaks [0,1) and [1,2), id 1 speaks [2,3.5).
    fn sample_result() -> DiarizationResult {
        let turns = vec![
            turn_range(1, 2.0, 3.5),
            turn_range(0, 0.0, 1.0),
            turn_range(0, 1.0, 2.0),
        ];
        DiarizationResult::new(vec![seg(0.0, 1.0, Some(0))], turns, 2)
    }

    #[test]
    fn remap_segments_and_turns_apply_mapping() {
        let remap = SpeakerIdRemap::from_mapping(vec![(SpeakerId(1), SpeakerId(0))]).unwrap();

        let mut segments = vec![
            seg(0.0, 1.0, Some(1)),
            seg(1.0, 2.0, None),
            seg(2.0, 3.0, Some(5)),
        ];
        remap_segments(&mut segments, &remap);
        assert_eq!(segments[0].speaker, Some(SpeakerId(0)));
        assert_eq!(segments[1].speaker, None); // unassigned stays unassigned
        assert_eq!(segments[2].speaker, Some(SpeakerId(5))); // unmapped passes through

        let mut turns = vec![turn_range(1, 0.0, 1.0), turn_range(5, 1.0, 2.0)];
        remap_turns(&mut turns, &remap);
        assert_eq!(turns[0].speaker, SpeakerId(0));
        assert_eq!(turns[1].speaker, SpeakerId(5));
    }

    #[test]
    fn speaker_turn_constructors() {
        let t = SpeakerTurn::new(
            SpeakerId(1),
            TimeRange {
                start: 0.0,
                end: 1.0,
            },
        );
        assert!(t.stable);
        assert!(t.text.is_none());
        let p = SpeakerTurn::with_stability(
            SpeakerId(1),
            TimeRange {
                start: 0.0,
                end: 1.0,
            },
            false,
        );
        assert!(!p.stable);
    }

    #[test]
    fn speaker_turn_serde_stability_compatibility() {
        // `stable: true` and absent text are omitted from the payload.
        let t = turn_range(1, 0.0, 1.0);
        let json = serde_json::to_string(&t).unwrap();
        assert!(!json.contains("stable"), "{json}");
        assert!(!json.contains("text"), "{json}");
        assert_eq!(serde_json::from_str::<SpeakerTurn>(&json).unwrap(), t);

        // Provisional turns keep the flag in the payload.
        let p = SpeakerTurn::with_stability(
            SpeakerId(1),
            TimeRange {
                start: 0.0,
                end: 1.0,
            },
            false,
        );
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"stable\":false"), "{json}");
        assert_eq!(serde_json::from_str::<SpeakerTurn>(&json).unwrap(), p);

        // Legacy payloads without `stable` default to true.
        let legacy = r#"{"speaker":2,"time":{"start":1.0,"end":2.0}}"#;
        let back: SpeakerTurn = serde_json::from_str(legacy).unwrap();
        assert!(back.stable);
        assert_eq!(back.speaker, SpeakerId(2));
        assert!(back.text.is_none());
    }

    #[test]
    fn word_alignment_interpolated_omitted_when_false() {
        let wa = WordAlignment {
            word: "hi".into(),
            time: TimeRange {
                start: 0.0,
                end: 0.4,
            },
            speaker: Some(SpeakerId(0)),
            confidence: 0.9,
            interpolated: false,
        };
        let json = serde_json::to_string(&wa).unwrap();
        assert!(!json.contains("interpolated"), "{json}");
        assert_eq!(serde_json::from_str::<WordAlignment>(&json).unwrap(), wa);

        let wa = WordAlignment {
            interpolated: true,
            ..wa
        };
        let json = serde_json::to_string(&wa).unwrap();
        assert!(json.contains("\"interpolated\":true"), "{json}");
        assert_eq!(serde_json::from_str::<WordAlignment>(&json).unwrap(), wa);
    }

    #[test]
    fn segment_and_transcript_serde_roundtrips() {
        let s = seg(0.1, 0.9, Some(3));
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Segment>(&json).unwrap(), s);
        let s = Segment {
            speaker: None,
            confidence: None,
            ..s
        };
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Segment>(&json).unwrap(), s);

        let t = Transcript {
            words: vec![Word {
                word: "hi".into(),
                time: TimeRange {
                    start: 0.0,
                    end: 0.3,
                },
                confidence: 0.8,
            }],
        };
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(serde_json::from_str::<Transcript>(&json).unwrap(), t);
        assert!(Transcript::default().words.is_empty());
    }

    #[test]
    fn result_new_builds_sorted_speaker_rollup() {
        let r = sample_result();
        assert_eq!(r.schema_version, "diarization-result-v1");
        assert!(!r.provenance.version.is_empty());
        assert_eq!(r.speakers.len(), 2);
        // Sorted by numeric id with the canonical label attached.
        assert_eq!(r.speakers[0].id, 0);
        assert_eq!(r.speakers[0].label, "SPEAKER_00");
        assert!((r.speakers[0].total_speech_s - 2.0).abs() < 1e-9);
        assert_eq!(r.speakers[0].turn_count, 2);
        assert_eq!(r.speakers[1].id, 1);
        assert_eq!(r.speakers[1].label, "SPEAKER_01");
        assert!((r.speakers[1].total_speech_s - 1.5).abs() < 1e-9);
        assert_eq!(r.speakers[1].turn_count, 1);
        assert!(r.speakers.iter().all(|s| s.embedding.is_none()));
    }

    #[test]
    fn result_builders_attach_audio_and_provenance() {
        let r = sample_result().with_audio(10.0, 16000);
        assert_eq!(r.audio.duration_secs, 10.0);
        assert_eq!(r.audio.sample_rate, 16000);

        // An empty version keeps the version stamped by `new`.
        let keep = sample_result().with_provenance(Provenance {
            profile: "balanced".into(),
            embedder: "wespeaker".into(),
            ..Provenance::default()
        });
        assert_eq!(keep.provenance.profile, "balanced");
        assert_eq!(keep.provenance.embedder, "wespeaker");
        assert!(!keep.provenance.version.is_empty());

        // A non-empty version overrides.
        let over = sample_result().with_provenance(Provenance {
            version: "9.9.9".into(),
            ..Provenance::default()
        });
        assert_eq!(over.provenance.version, "9.9.9");
    }

    #[test]
    fn speaker_summary_embedding_omitted_when_absent() {
        let r = sample_result();
        let json = serde_json::to_string(&r.speakers[0]).unwrap();
        assert!(!json.contains("embedding"), "{json}");
    }

    #[test]
    fn with_speaker_embeddings_normalizes_and_matches_by_id() {
        let r = sample_result().with_speaker_embeddings(&[
            (SpeakerId(0), vec![3.0, 4.0]), // not pre-normalized
            (SpeakerId(9), vec![1.0, 0.0]), // no matching speaker — ignored
        ]);
        let e0 = r.speakers[0].embedding.as_ref().unwrap();
        let norm = e0.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
        assert!((e0[0] - 0.6).abs() < 1e-6);
        assert!(r.speakers[1].embedding.is_none());
    }

    #[test]
    fn result_serde_roundtrip_and_legacy_defaults() {
        let r = sample_result().with_audio(5.0, 16000).with_exclusive();
        let json = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<DiarizationResult>(&json).unwrap(), r);

        // Older payloads without the additive metadata still parse.
        let legacy = r#"{"segments":[],"turns":[],"num_speakers":0}"#;
        let back: DiarizationResult = serde_json::from_str(legacy).unwrap();
        assert_eq!(back.schema_version, "diarization-result-v1");
        assert_eq!(back.audio, AudioMeta::default());
        assert_eq!(back.provenance, Provenance::default());
        assert!(back.speakers.is_empty());
        assert!(back.exclusive_turns.is_empty());
    }

    #[test]
    fn grid_frame_helpers() {
        assert_eq!(TimeRange::grid_frame_count(1.0), 101);
        // Huge timelines are capped at the 24 h guard.
        assert_eq!(
            TimeRange::grid_frame_count(1e12),
            TimeRange::MAX_GRID_FRAMES
        );

        let tr = TimeRange {
            start: 0.0,
            end: 0.03,
        };
        assert_eq!(tr.grid_frame_range(), (0, 3));
        // Negative coordinates saturate to frame 0.
        let neg = TimeRange {
            start: -1.0,
            end: 0.02,
        };
        assert_eq!(neg.grid_frame_range().0, 0);
    }

    #[test]
    fn exclusive_turns_empty_and_degenerate_inputs() {
        assert!(exclusive_turns(&[]).is_empty());
        // Non-finite maximum end → no grid to score.
        let inf = turn_range(0, 0.0, f64::INFINITY);
        assert!(exclusive_turns(&[inf]).is_empty());
        // Zero-length, inverted, and NaN turns are all skipped.
        let zero = turn_range(0, 1.0, 1.0);
        let neg = turn_range(1, 2.0, 1.0);
        let nan = turn_range(2, f64::NAN, 3.0);
        assert!(exclusive_turns(&[zero, neg, nan]).is_empty());
    }

    #[test]
    fn exclusive_turns_single_turn_collapses_frames() {
        let out = exclusive_turns(&[turn_range(0, 0.0, 0.03)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, SpeakerId(0));
        assert!(out[0].time.start.abs() < 1e-9);
        // Frame quantization keeps the end within one frame of the input.
        assert!((out[0].time.end - 0.03).abs() <= EXCLUSIVE_FRAME_SECS + 1e-9);
    }

    #[test]
    fn exclusive_turns_overlap_picks_longer_covering_turn() {
        let turns = vec![
            turn_range(0, 0.0, 1.0), // long — dominates everywhere
            turn_range(1, 0.2, 0.4), // short concurrent speaker dropped by design
        ];
        let out = exclusive_turns(&turns);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, SpeakerId(0));
    }

    #[test]
    fn exclusive_turns_tie_breaks_to_smaller_speaker_id() {
        // Equal-duration fully-overlapping turns → smaller id wins.
        let turns = vec![turn_range(2, 0.0, 0.5), turn_range(1, 0.0, 0.5)];
        let out = exclusive_turns(&turns);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, SpeakerId(1));
    }

    #[test]
    fn exclusive_turns_silence_splits_same_speaker() {
        let turns = vec![turn_range(0, 0.0, 0.1), turn_range(0, 0.5, 0.7)];
        let out = exclusive_turns(&turns);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].speaker, SpeakerId(0));
        assert_eq!(out[1].speaker, SpeakerId(0));
        assert!((out[1].time.start - 0.5).abs() <= EXCLUSIVE_FRAME_SECS + 1e-9);
    }

    #[test]
    fn exclusive_turns_adjacent_speakers_stay_contiguous() {
        let turns = vec![turn_range(0, 0.0, 0.2), turn_range(1, 0.2, 0.4)];
        let out = exclusive_turns(&turns);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].speaker, SpeakerId(0));
        assert_eq!(out[1].speaker, SpeakerId(1));
        assert!((out[0].time.end - out[1].time.start).abs() < 1e-9);
    }

    #[test]
    fn with_exclusive_fills_and_is_idempotent() {
        let r = sample_result().with_exclusive();
        assert!(!r.exclusive_turns.is_empty());
        // `turns` is untouched by the projection.
        assert_eq!(r.turns.len(), 3);
        let again = r.clone().with_exclusive();
        assert_eq!(r.exclusive_turns, again.exclusive_turns);
    }
}
