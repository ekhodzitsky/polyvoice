//! polyvoice-mcp — MCP (Model Context Protocol) stdio server: the agent front door.
//!
//! Exposes `polyvoice.diarize` (+ `transcribe`/`diarize_and_transcribe` stubbed
//! until the opt-in `polyvoice-asr` crate exists, and `capabilities`) over stdio.
//! Diarization uses the same production path as the CLI (**pipeline v2 + VBx
//! kernels** by default). Tools project the canonical `DiarizationResult` v1. **stdout is
//! reserved for JSON-RPC** — nothing else is ever printed to it (no `println!`,
//! no tracing subscriber installed, pipeline runs quietly), so the protocol
//! stream stays clean. Errors carry the polyvoice FFI numeric codes as
//! `{code, message}`.

use anyhow::Result;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::model::{ErrorData, ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

use polyvoice::cli_common;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::PipelineConfig;
use polyvoice::types::{DEFAULT_AHC_THRESHOLD, DiarizationResult, Profile, SampleRate};
use polyvoice::wav::read_wav;

// Numeric error codes mirror include/polyvoice.h (do not invent new ones).
const ERR_INVALID_ARG: i32 = 1;
const ERR_AUDIO_TOO_LONG: i32 = 3;
const ERR_MODEL_LOAD: i32 = 10;
const ERR_INFERENCE: i32 = 11;
const ERR_REGISTRY: i32 = 30;
const ERR_INTERNAL: i32 = 99;

/// Build a structured MCP error carrying the FFI `{code, message}` payload.
/// Only genuine bad-input failures map to JSON-RPC invalid params; server-side
/// failures (model load, inference, registry, internal) are internal errors.
fn err(code: i32, message: impl Into<String>) -> ErrorData {
    let message = message.into();
    let data = Some(serde_json::json!({ "code": code, "message": message }));
    if code == ERR_INVALID_ARG {
        ErrorData::invalid_params(message, data)
    } else {
        ErrorData::internal_error(message, data)
    }
}

// ----- tool input/output DTOs (strict schemas; additionalProperties: false) -----

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct DiarizeInput {
    /// Path to a mono 16 kHz WAV file to diarize.
    path: String,
    /// Model profile: "balanced" (default) or "mobile".
    #[serde(default)]
    profile: Option<String>,
    /// Clusterer: "vbx" (default, PLDA + VB-HMM, matches CLI) or "ahc"
    /// (fixed-threshold cosine AHC).
    #[serde(default)]
    clusterer: Option<String>,
    /// AHC cosine-similarity threshold when clusterer is "ahc" (default 0.45).
    /// Ignored for "vbx".
    #[serde(default)]
    threshold: Option<f32>,
    /// Cap the number of speakers (clustering ceiling, 1..=255).
    #[serde(default)]
    max_speakers: Option<usize>,
    /// Optional directory with VBx PLDA `.npy` params (overrides env/registry).
    #[serde(default)]
    vbx_plda_dir: Option<String>,
    /// Accepted for compatibility. Both values return the full result, including turns.
    #[serde(default)]
    verbosity: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `path` is part of the tool input schema; the stub errors without reading it
struct TranscribeInput {
    /// Path to a mono 16 kHz WAV file to transcribe.
    path: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct SpeakerRollup {
    /// Canonical speaker label, e.g. "SPEAKER_00".
    label: String,
    /// Numeric speaker id.
    id: u32,
    /// Total speech attributed to this speaker, in seconds.
    total_speech_s: f64,
    /// Number of turns for this speaker.
    turn_count: usize,
}

#[derive(Debug, Serialize, JsonSchema)]
struct TimeDto {
    /// Start time in seconds.
    start: f64,
    /// End time in seconds.
    end: f64,
}

#[derive(Debug, Serialize, JsonSchema)]
struct SegmentDto {
    /// Time range of the segment.
    time: TimeDto,
    /// Numeric speaker id, or null if unassigned.
    speaker: Option<u32>,
    /// Assignment confidence, or null.
    confidence: Option<f32>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct TurnDto {
    /// Numeric speaker id.
    speaker: u32,
    /// Canonical speaker label, e.g. "SPEAKER_00".
    label: String,
    /// Numeric speaker id (same value as `speaker`).
    speaker_id: u32,
    /// Turn start, seconds from the beginning of the audio.
    start: f64,
    /// Turn end, seconds from the beginning of the audio.
    end: f64,
    /// Turn time range. `start`/`end` match the flat keys above.
    time: TimeDto,
}

#[derive(Debug, Serialize, JsonSchema)]
struct AudioDto {
    /// Audio duration in seconds.
    duration_secs: f64,
    /// Sample rate in Hz.
    sample_rate: u32,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ProvenanceDto {
    /// Crate version that produced the result.
    version: String,
    /// Profile id, or empty if not recorded.
    profile: String,
    /// Segmentation/VAD model id, or empty.
    segmenter: String,
    /// Embedding model id, or empty.
    embedder: String,
    /// Clustering backend id, or empty.
    clusterer: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct DiarizeOutput {
    /// Result schema identifier (canonical DiarizationResult v1).
    schema_version: String,
    /// Number of distinct speakers detected.
    num_speakers: usize,
    /// Per-window segments.
    segments: Vec<SegmentDto>,
    /// Ordered speaker turns. Always present.
    turns: Vec<TurnDto>,
    /// Audio metadata.
    audio: AudioDto,
    /// How this result was produced.
    provenance: ProvenanceDto,
    /// Per-speaker rollup.
    speakers: Vec<SpeakerRollup>,
    /// Audio duration, in seconds. Same value as `audio.duration_secs`.
    duration_s: f64,
}

#[derive(Debug, Serialize, JsonSchema)]
struct Capabilities {
    /// Server name.
    name: String,
    /// Server (crate) version.
    version: String,
    /// Tool names this server exposes.
    tools: Vec<String>,
    /// Whether speech-to-text is available (requires the opt-in polyvoice-asr crate).
    asr_available: bool,
    /// Output formats the diarize CLI/library can project to.
    output_formats: Vec<String>,
    /// Model profiles available.
    profiles: Vec<String>,
}

#[derive(Clone)]
struct PolyvoiceMcp;

#[tool_router]
impl PolyvoiceMcp {
    fn new() -> Self {
        Self
    }

    #[tool(
        name = "polyvoice.capabilities",
        description = "List the tools, version, ASR availability, and output formats of this server."
    )]
    fn capabilities(&self) -> Json<Capabilities> {
        Json(Capabilities {
            name: "polyvoice-mcp".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            tools: vec![
                "polyvoice.diarize".to_owned(),
                "polyvoice.transcribe".to_owned(),
                "polyvoice.diarize_and_transcribe".to_owned(),
                "polyvoice.capabilities".to_owned(),
            ],
            asr_available: false,
            output_formats: vec![
                "rttm".to_owned(),
                "json".to_owned(),
                "srt".to_owned(),
                "vtt".to_owned(),
                "txt".to_owned(),
            ],
            profiles: vec!["balanced".to_owned(), "mobile".to_owned()],
        })
    }

    #[tool(
        name = "polyvoice.diarize",
        description = "Diarize a WAV file (who spoke when). Returns diarization-result-v1 fields (segments, turns with numeric speaker and time, num_speakers) plus the speaker rollup."
    )]
    fn diarize(
        &self,
        Parameters(input): Parameters<DiarizeInput>,
    ) -> Result<Json<DiarizeOutput>, ErrorData> {
        let result = run_diarize(&input)?;
        let detailed = input.verbosity.as_deref() == Some("detailed");
        Ok(Json(project(&result, detailed)))
    }

    #[tool(
        name = "polyvoice.transcribe",
        description = "Transcribe a WAV file. Requires the optional polyvoice-asr crate, which is not installed."
    )]
    fn transcribe(
        &self,
        Parameters(_input): Parameters<TranscribeInput>,
    ) -> Result<Json<DiarizeOutput>, ErrorData> {
        Err(asr_unavailable())
    }

    #[tool(
        name = "polyvoice.diarize_and_transcribe",
        description = "Diarize + transcribe (who said what). Requires the optional polyvoice-asr crate, which is not installed."
    )]
    fn diarize_and_transcribe(
        &self,
        Parameters(_input): Parameters<DiarizeInput>,
    ) -> Result<Json<DiarizeOutput>, ErrorData> {
        // Transcription is unavailable without polyvoice-asr; fail as a whole and
        // tell the agent to call `polyvoice.diarize` for diarization-only output.
        Err(asr_unavailable())
    }
}

#[tool_handler]
impl ServerHandler for PolyvoiceMcp {
    fn get_info(&self) -> ServerInfo {
        // ServerInfo is #[non_exhaustive] — can't use a struct literal; mutate a
        // Default instead.
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.instructions = Some(
            "polyvoice speaker diarization (pipeline v2 + VBx by default, same as the CLI). \
             Call polyvoice.diarize with a WAV path to get who-spoke-when; \
             polyvoice.capabilities to discover features. Pass clusterer=ahc for fixed-threshold \
             AHC. Transcription tools require the optional polyvoice-asr crate."
                .to_owned(),
        );
        info
    }
}

fn asr_unavailable() -> ErrorData {
    err(
        ERR_INTERNAL,
        "ASR is unavailable: install the optional `polyvoice-asr` companion crate to enable transcription",
    )
}

/// Project a canonical DiarizationResult v1 onto the MCP output DTO.
///
/// `detailed` is still accepted so existing call sites can pass the verbosity
/// flag; it no longer drops turns.
fn project(result: &DiarizationResult, _detailed: bool) -> DiarizeOutput {
    let speakers = result
        .speakers
        .iter()
        .map(|s| SpeakerRollup {
            label: s.label.clone(),
            id: s.id,
            total_speech_s: s.total_speech_s,
            turn_count: s.turn_count,
        })
        .collect();
    let segments = result
        .segments
        .iter()
        .map(|s| SegmentDto {
            time: TimeDto {
                start: s.time.start,
                end: s.time.end,
            },
            speaker: s.speaker.map(|id| id.0),
            confidence: s.confidence,
        })
        .collect();
    let turns = result
        .turns
        .iter()
        .map(|t| TurnDto {
            speaker: t.speaker.0,
            label: t.speaker.to_string(),
            speaker_id: t.speaker.0,
            start: t.time.start,
            end: t.time.end,
            time: TimeDto {
                start: t.time.start,
                end: t.time.end,
            },
        })
        .collect();
    DiarizeOutput {
        schema_version: result.schema_version.clone(),
        num_speakers: result.num_speakers,
        segments,
        turns,
        audio: AudioDto {
            duration_secs: result.audio.duration_secs,
            sample_rate: result.audio.sample_rate,
        },
        provenance: ProvenanceDto {
            version: result.provenance.version.clone(),
            profile: result.provenance.profile.clone(),
            segmenter: result.provenance.segmenter.clone(),
            embedder: result.provenance.embedder.clone(),
            clusterer: result.provenance.clusterer.clone(),
        },
        speakers,
        duration_s: result.audio.duration_secs,
    }
}

/// Resolve the optional `max_speakers` cap into the pipeline config's `u8`
/// ceiling (shared range check with the CLI).
fn resolve_max_speakers(max_speakers: Option<usize>) -> Result<u8, ErrorData> {
    match max_speakers {
        None => Ok(PipelineConfig::default().max_speakers),
        Some(n) => cli_common::max_speakers_u8(n).map_err(|e| err(ERR_INVALID_ARG, e.to_string())),
    }
}

fn mcp_root() -> Result<Option<std::path::PathBuf>, ErrorData> {
    let Some(raw) = std::env::var_os("POLYVOICE_MCP_ROOT") else {
        return Ok(None);
    };
    std::path::PathBuf::from(raw)
        .canonicalize()
        .map(Some)
        .map_err(|e| err(ERR_INVALID_ARG, format!("POLYVOICE_MCP_ROOT: {e}")))
}

fn reject_parent_dir(p: &Path, label: &str) -> Result<(), ErrorData> {
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(err(
            ERR_INVALID_ARG,
            format!("{label} path traversal rejected"),
        ));
    }
    Ok(())
}

/// Nearest existing ancestor of `path`, canonicalized. A missing leaf stops
/// at the parent that does exist (`.` when nothing relative exists yet), so
/// the caller can confine that ancestor without stating whether the leaf exists.
fn canonical_existing_ancestor(path: &Path) -> std::io::Result<PathBuf> {
    let mut acc = PathBuf::new();
    let mut last_existing: Option<PathBuf> = None;
    for component in path.components() {
        acc.push(component.as_os_str());
        match acc.try_exists() {
            Ok(true) => last_existing = Some(acc.clone()),
            Ok(false) => break,
            Err(e) if last_existing.is_none() => return Err(e),
            Err(_) => break,
        }
    }
    last_existing
        .unwrap_or_else(|| PathBuf::from("."))
        .canonicalize()
}

/// Audio path for [`run_diarize`]. When `root` is set, an ancestor outside
/// that root is rejected before any leaf existence check.
fn open_audio_path(path: &Path, root: Option<&Path>) -> Result<PathBuf, ErrorData> {
    reject_parent_dir(path, "audio")?;
    if let Some(root) = root {
        let ancestor = canonical_existing_ancestor(path)
            .map_err(|e| err(ERR_INVALID_ARG, format!("audio: {e}")))?;
        if !ancestor.starts_with(root) {
            return Err(err(ERR_INVALID_ARG, "audio is outside POLYVOICE_MCP_ROOT"));
        }
        if !path.is_file() {
            return Err(err(
                ERR_INVALID_ARG,
                format!("no such file: {}", path.display()),
            ));
        }
    } else if !path.is_file() {
        return Err(err(
            ERR_INVALID_ARG,
            format!("no such file: {}", path.display()),
        ));
    }
    confine_path(path, root, "audio")
}

fn map_pipeline_run_error(e: polyvoice::pipeline_v2::PipelineError) -> ErrorData {
    use polyvoice::pipeline_v2::PipelineError;
    let code = match &e {
        PipelineError::UnsupportedSampleRate { .. } => ERR_INVALID_ARG,
        PipelineError::AudioTooLong { .. } => ERR_AUDIO_TOO_LONG,
        PipelineError::Registry(_) => ERR_REGISTRY,
        _ => ERR_INFERENCE,
    };
    err(code, e.to_string())
}

fn confine_path(
    p: &Path,
    root: Option<&Path>,
    label: &str,
) -> Result<std::path::PathBuf, ErrorData> {
    reject_parent_dir(p, label)?;
    let canon = p
        .canonicalize()
        .map_err(|e| err(ERR_INVALID_ARG, format!("{label}: {e}")))?;
    if let Some(root) = root
        && !canon.starts_with(root)
    {
        return Err(err(
            ERR_INVALID_ARG,
            format!("{label} is outside POLYVOICE_MCP_ROOT"),
        ));
    }
    Ok(canon)
}

/// Run the production (pipeline v2) diarization path quietly, mapping failures
/// to FFI-coded MCP errors. Defaults match the CLI: VBx clusterer + registry
/// PLDA auto-download when `vbx_plda_dir` is unset. `..` components are
/// always rejected. When `POLYVOICE_MCP_ROOT` is set, the nearest existing
/// ancestor must stay under it — a missing leaf outside the root is "outside",
/// not "no such file".
fn run_diarize(input: &DiarizeInput) -> Result<DiarizationResult, ErrorData> {
    let root = mcp_root()?;
    let path = open_audio_path(Path::new(&input.path), root.as_deref())?;
    let profile: Profile = input
        .profile
        .as_deref()
        .unwrap_or("balanced")
        .parse()
        .map_err(|e: polyvoice::types::ProfileParseError| err(ERR_INVALID_ARG, e.to_string()))?;
    let clusterer_kind = cli_common::parse_clusterer_kind(
        input.clusterer.as_deref().unwrap_or("vbx"),
        input.threshold.unwrap_or(DEFAULT_AHC_THRESHOLD),
    )
    .map_err(|e| err(ERR_INVALID_ARG, e.to_string()))?;
    let max_speakers = resolve_max_speakers(input.max_speakers)?;

    let registry = ModelRegistry::default().map_err(|e| err(ERR_REGISTRY, e.to_string()))?;
    // Ensure profile models exist before build (clearer error mapping).
    let _models = registry
        .ensure_for_profile(profile)
        .map_err(|e| err(ERR_MODEL_LOAD, e.to_string()))?;

    let vbx_plda_dir = match input.vbx_plda_dir.as_ref() {
        Some(s) => Some(confine_path(Path::new(s), root.as_deref(), "vbx_plda_dir")?),
        // Unset: the builder reads POLYVOICE_VBX_PLDA_DIR. No extra root check.
        None => None,
    };
    let config = cli_common::product_pipeline_config(cli_common::ProductParts {
        profile,
        clusterer: clusterer_kind,
        vbx_plda_dir,
        max_speakers: input.max_speakers.is_some().then_some(max_speakers),
        embed_window_secs: None,
        execution_provider: polyvoice::pipeline_v2::ExecutionProvider::auto(),
        as_norm: None,
        domain: None,
    });
    let pipeline = cli_common::build_v2_pipeline(config, registry)
        .map_err(|e| err(ERR_MODEL_LOAD, format!("{e:#}")))?;

    let (samples, sr_hz) = read_wav(&path).map_err(|e| err(ERR_INVALID_ARG, e.to_string()))?;
    let sr = SampleRate::new(sr_hz)
        .ok_or_else(|| err(ERR_INVALID_ARG, format!("invalid sample rate {sr_hz} Hz")))?;

    pipeline.run(&samples, sr).map_err(map_pipeline_run_error)
}

#[tokio::main]
async fn main() -> Result<()> {
    cli_common::limit_malloc_arenas();
    // No tracing subscriber and no stdout writes anywhere — stdout is the JSON-RPC
    // channel. ort emits via the `tracing` crate (dropped without a subscriber).
    let service = PolyvoiceMcp::new()
        .serve(rmcp::transport::io::stdio())
        .await?;
    service.waiting().await?;
    Ok(())
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_lists_four_tools_and_no_asr() {
        let cap = PolyvoiceMcp::new().capabilities().0;
        assert_eq!(cap.tools.len(), 4);
        assert!(!cap.asr_available);
        assert!(cap.tools.iter().any(|t| t == "polyvoice.diarize"));
        assert_eq!(cap.output_formats.len(), 5);
    }

    #[test]
    fn asr_unavailable_error_carries_ffi_code() {
        let e = asr_unavailable();
        let data = e.data.expect("data");
        assert_eq!(data["code"], ERR_INTERNAL);
        assert!(data["message"].as_str().unwrap().contains("polyvoice-asr"));
    }

    #[test]
    fn invalid_arg_maps_to_jsonrpc_invalid_params() {
        let e = err(ERR_INVALID_ARG, "no such file: x.wav");
        assert_eq!(e.code.0, -32602);
        let data = e.data.expect("data");
        assert_eq!(data["code"], ERR_INVALID_ARG);
        assert_eq!(data["message"], "no such file: x.wav");
    }

    #[test]
    fn model_load_maps_to_jsonrpc_internal_error() {
        let e = err(ERR_MODEL_LOAD, "model missing");
        assert_eq!(e.code.0, -32603);
        let data = e.data.expect("data");
        assert_eq!(data["code"], ERR_MODEL_LOAD);
        assert_eq!(data["message"], "model missing");
    }

    #[test]
    fn input_schema_is_strict() {
        // additionalProperties:false comes from #[serde(deny_unknown_fields)].
        let schema = schemars::schema_for!(DiarizeInput);
        let json = serde_json::to_value(&schema).unwrap();
        assert_eq!(json["additionalProperties"], serde_json::json!(false));
        assert!(json["properties"]["path"].is_object());
    }

    #[test]
    fn max_speakers_accepts_default_and_valid_range() {
        assert_eq!(
            resolve_max_speakers(None).unwrap(),
            PipelineConfig::default().max_speakers
        );
        assert_eq!(resolve_max_speakers(Some(1)).unwrap(), 1);
        assert_eq!(resolve_max_speakers(Some(255)).unwrap(), 255);
    }

    #[test]
    fn max_speakers_rejects_out_of_range_with_invalid_arg() {
        for n in [0_usize, 256, 1000] {
            let e = resolve_max_speakers(Some(n)).expect_err("out of range must error");
            assert_eq!(e.code.0, -32602, "n={n} must map to invalid params");
            let data = e.data.expect("data");
            assert_eq!(data["code"], ERR_INVALID_ARG);
            assert!(
                data["message"]
                    .as_str()
                    .unwrap()
                    .contains("max_speakers must be in 1..=255"),
                "message must name the valid range: {data}"
            );
        }
    }

    use polyvoice::types::{Segment, SpeakerId, SpeakerTurn, TimeRange};

    /// Minimal valid input pointing at `path`; every optional knob unset.
    fn input(path: &str) -> DiarizeInput {
        DiarizeInput {
            path: path.to_owned(),
            profile: None,
            clusterer: None,
            threshold: None,
            max_speakers: None,
            vbx_plda_dir: None,
            verbosity: None,
        }
    }

    /// A two-speaker, three-turn result over 3s of audio.
    fn sample_result() -> DiarizationResult {
        let turn = |id: u32, start: f64, end: f64| SpeakerTurn {
            speaker: SpeakerId(id),
            time: TimeRange { start, end },
            text: None,
            stable: true,
        };
        DiarizationResult::new(
            vec![Segment {
                time: TimeRange {
                    start: 0.0,
                    end: 1.0,
                },
                speaker: Some(SpeakerId(0)),
                confidence: None,
            }],
            vec![turn(0, 0.0, 1.0), turn(1, 1.0, 2.5), turn(0, 2.5, 3.0)],
            2,
        )
        .with_audio(3.0, 16000)
    }

    #[test]
    fn project_includes_schema_turns_and_rolls_up_speakers() {
        let out = project(&sample_result(), false);
        assert_eq!(out.turns.len(), 3);
        assert_eq!(out.turns[0].speaker, 0);
        assert_eq!(out.turns[0].speaker_id, 0);
        assert_eq!(out.turns[0].label, "SPEAKER_00");
        assert!((out.turns[0].time.start - 0.0).abs() < 1e-9);
        assert!((out.turns[0].time.end - 1.0).abs() < 1e-9);
        assert!((out.turns[0].start - out.turns[0].time.start).abs() < 1e-9);
        assert!((out.turns[0].end - out.turns[0].time.end).abs() < 1e-9);
        assert_eq!(out.num_speakers, 2);
        assert!((out.duration_s - 3.0).abs() < 1e-9);
        assert!(!out.schema_version.is_empty());
        assert_eq!(out.speakers.len(), 2);
        let s0 = &out.speakers[0];
        assert_eq!(s0.label, "SPEAKER_00");
        assert_eq!(s0.id, 0);
        assert_eq!(s0.turn_count, 2);
        assert!((s0.total_speech_s - 1.5).abs() < 1e-9);
        let s1 = &out.speakers[1];
        assert_eq!(s1.label, "SPEAKER_01");
        assert_eq!(s1.turn_count, 1);
        let json = serde_json::to_value(&out).unwrap();
        assert_eq!(json["turns"].as_array().unwrap().len(), 3);
        assert_eq!(json["turns"][0]["speaker"], 0);
        assert_eq!(json["turns"][0]["label"], "SPEAKER_00");
        assert!(json["segments"].is_array());
        assert_eq!(json["segments"].as_array().unwrap().len(), 1);
        assert_eq!(json["segments"][0]["speaker"], 0);
        assert!(json["segments"][0]["confidence"].is_null());
        assert!(json["speakers"].as_array().unwrap().len() == 2);
        assert!((json["audio"]["duration_secs"].as_f64().unwrap() - 3.0).abs() < 1e-9);
        assert_eq!(json["audio"]["sample_rate"], 16000);
        assert!(json["provenance"]["version"].is_string());
    }

    #[test]
    fn project_detailed_includes_ordered_turns() {
        let out = project(&sample_result(), true);
        let turns = &out.turns;
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].speaker, 0);
        assert_eq!(turns[0].label, "SPEAKER_00");
        assert_eq!(turns[0].speaker_id, 0);
        assert!((turns[0].start - 0.0).abs() < 1e-9);
        assert!((turns[0].end - 1.0).abs() < 1e-9);
        assert!((turns[0].time.start - 0.0).abs() < 1e-9);
        assert!((turns[0].time.end - 1.0).abs() < 1e-9);
        assert_eq!(turns[1].speaker, 1);
        assert_eq!(turns[1].label, "SPEAKER_01");
        assert!((turns[1].end - 2.5).abs() < 1e-9);
        assert_eq!(turns[2].speaker_id, 0);
        let json = serde_json::to_value(&out).unwrap();
        assert_eq!(json["turns"].as_array().unwrap().len(), 3);
        assert_eq!(json["turns"][0]["speaker"], 0);
        assert!(json["segments"].is_array());
    }

    #[test]
    fn run_diarize_missing_file_is_invalid_params() {
        let e = run_diarize(&input("/definitely/not/here.wav")).unwrap_err();
        assert_eq!(e.code.0, -32602);
        let data = e.data.expect("data");
        assert_eq!(data["code"], ERR_INVALID_ARG);
        assert!(data["message"].as_str().unwrap().contains("no such file"));
    }

    #[test]
    fn run_diarize_rejects_parent_dir() {
        let e = run_diarize(&input("../secret.wav")).unwrap_err();
        assert_eq!(e.code.0, -32602);
        let msg = e.data.expect("data")["message"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(msg.contains("traversal"), "{msg}");
    }

    #[test]
    fn reject_parent_dir_allows_plain_paths() {
        reject_parent_dir(Path::new("meeting.wav"), "audio").unwrap();
        reject_parent_dir(Path::new("/abs/meeting.wav"), "audio").unwrap();
    }

    #[test]
    fn confine_path_rejects_outside_root() {
        let root = tempfile::TempDir::new().unwrap();
        let root_canon = root.path().canonicalize().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let e = confine_path(outside.path(), Some(&root_canon), "audio").unwrap_err();
        assert_eq!(e.code.0, -32602);
        let msg = e.data.expect("data")["message"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(msg.contains("outside"), "{msg}");
    }

    #[test]
    fn confine_path_accepts_inside_root() {
        let root = tempfile::TempDir::new().unwrap();
        let root_canon = root.path().canonicalize().unwrap();
        let inside = tempfile::NamedTempFile::new_in(root.path()).unwrap();
        let got = confine_path(inside.path(), Some(&root_canon), "audio").unwrap();
        assert!(got.starts_with(&root_canon));
    }

    #[test]
    fn missing_leaf_outside_root_does_not_report_no_such_file() {
        let root = tempfile::TempDir::new().unwrap();
        let root_canon = root.path().canonicalize().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let missing = outside.path().join("missing.wav");
        assert!(!missing.exists());
        let msg = open_audio_path(&missing, Some(&root_canon))
            .unwrap_err()
            .data
            .expect("data")["message"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(msg.contains("outside POLYVOICE_MCP_ROOT"), "{msg}");
        assert!(!msg.contains("no such file"), "{msg}");

        let inside_missing = root.path().join("missing.wav");
        let msg = open_audio_path(&inside_missing, Some(&root_canon))
            .unwrap_err()
            .data
            .expect("data")["message"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(msg.contains("no such file"), "{msg}");
        assert!(!msg.contains("outside"), "{msg}");
    }

    #[test]
    fn pipeline_run_errors_use_ffi_codes() {
        use polyvoice::models::RegistryError;
        use polyvoice::pipeline_v2::{ConfigError, PipelineError};
        let cases = [
            (
                PipelineError::UnsupportedSampleRate { actual: 8000 },
                ERR_INVALID_ARG,
            ),
            (
                PipelineError::AudioTooLong {
                    actual_samples: 9,
                    max_samples: 8,
                },
                ERR_AUDIO_TOO_LONG,
            ),
            (
                PipelineError::Registry(RegistryError::ModelNotFound {
                    model_id: "x".into(),
                }),
                ERR_REGISTRY,
            ),
            (
                PipelineError::Config(ConfigError::RegistryInCustomProfile),
                ERR_INFERENCE,
            ),
        ];
        for (error, code) in cases {
            let data = map_pipeline_run_error(error).data.expect("data");
            assert_eq!(data["code"], code);
        }
    }

    #[test]
    fn run_diarize_rejects_unknown_profile() {
        let tmp = tempfile::NamedTempFile::with_suffix(".wav").unwrap();
        let mut i = input(tmp.path().to_str().unwrap());
        i.profile = Some("nope".to_owned());
        let e = run_diarize(&i).unwrap_err();
        assert_eq!(e.code.0, -32602);
        assert_eq!(e.data.expect("data")["code"], ERR_INVALID_ARG);
    }

    #[test]
    fn run_diarize_rejects_unknown_clusterer() {
        let tmp = tempfile::NamedTempFile::with_suffix(".wav").unwrap();
        let mut i = input(tmp.path().to_str().unwrap());
        i.clusterer = Some("nope".to_owned());
        let e = run_diarize(&i).unwrap_err();
        assert_eq!(e.code.0, -32602);
        assert_eq!(e.data.expect("data")["code"], ERR_INVALID_ARG);
    }

    #[test]
    fn run_diarize_rejects_out_of_range_max_speakers() {
        let tmp = tempfile::NamedTempFile::with_suffix(".wav").unwrap();
        let mut i = input(tmp.path().to_str().unwrap());
        i.max_speakers = Some(0);
        let e = run_diarize(&i).unwrap_err();
        assert_eq!(e.code.0, -32602);
        assert_eq!(e.data.expect("data")["code"], ERR_INVALID_ARG);
    }

    #[test]
    fn diarize_tool_surfaces_run_diarize_errors() {
        let server = PolyvoiceMcp::new();
        let e = server
            .diarize(Parameters(input("/definitely/not/here.wav")))
            .err()
            .expect("missing file must error");
        assert_eq!(e.code.0, -32602);
        assert_eq!(e.data.expect("data")["code"], ERR_INVALID_ARG);
    }

    #[test]
    fn transcribe_stub_returns_asr_unavailable() {
        let server = PolyvoiceMcp::new();
        let e = server
            .transcribe(Parameters(TranscribeInput {
                path: "x.wav".to_owned(),
            }))
            .err()
            .expect("transcribe is stubbed");
        assert_eq!(e.data.expect("data")["code"], ERR_INTERNAL);
    }

    #[test]
    fn diarize_and_transcribe_stub_fails_as_a_whole() {
        let server = PolyvoiceMcp::new();
        let e = server
            .diarize_and_transcribe(Parameters(input("x.wav")))
            .err()
            .expect("diarize_and_transcribe is stubbed");
        let data = e.data.expect("data");
        assert_eq!(data["code"], ERR_INTERNAL);
        assert!(data["message"].as_str().unwrap().contains("polyvoice-asr"));
    }

    #[test]
    fn server_info_advertises_tools_and_instructions() {
        let info = PolyvoiceMcp::new().get_info();
        assert!(info.capabilities.tools.is_some());
        let instructions = info.instructions.expect("instructions");
        assert!(instructions.contains("polyvoice.diarize"));
        assert!(instructions.contains("polyvoice.capabilities"));
    }
}
