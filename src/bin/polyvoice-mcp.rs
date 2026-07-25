//! polyvoice-mcp — MCP (Model Context Protocol) stdio server: the agent front door.
//!
//! Exposes `polyvoice.diarize` (+ `transcribe`/`diarize_and_transcribe` stubbed
//! until the opt-in `polyvoice-asr` crate exists, and `capabilities`) over stdio.
//! Diarization uses the same production path as the CLI (**pipeline v2 + VBx**
//! by default). Tools project the canonical `DiarizationResult` v1. **stdout is
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
use std::path::Path;

use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::{ClustererKind, Pipeline as V2Pipeline, PipelineConfig};
use polyvoice::types::{DiarizationResult, Profile, SampleRate};
use polyvoice::wav::read_wav;

// Numeric error codes mirror include/polyvoice.h (do not invent new ones).
const ERR_INVALID_ARG: i32 = 1;
const ERR_MODEL_LOAD: i32 = 10;
const ERR_INFERENCE: i32 = 11;
const ERR_REGISTRY: i32 = 30;
const ERR_INTERNAL: i32 = 99;

/// Build a structured MCP error carrying the FFI `{code, message}` payload.
fn err(code: i32, message: impl Into<String>) -> ErrorData {
    let message = message.into();
    ErrorData::invalid_params(
        message.clone(),
        Some(serde_json::json!({ "code": code, "message": message })),
    )
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
    /// Cap the number of speakers (clustering ceiling).
    #[serde(default)]
    max_speakers: Option<usize>,
    /// Optional directory with VBx PLDA `.npy` params (overrides env/registry).
    #[serde(default)]
    vbx_plda_dir: Option<String>,
    /// Response detail: "concise" (per-speaker rollup only, default) or
    /// "detailed" (also the full ordered turns).
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
struct TurnDto {
    /// Canonical speaker label, e.g. "SPEAKER_00".
    speaker: String,
    /// Numeric speaker id.
    speaker_id: u32,
    /// Turn start, seconds from the beginning of the audio.
    start: f64,
    /// Turn end, seconds from the beginning of the audio.
    end: f64,
}

#[derive(Debug, Serialize, JsonSchema)]
struct DiarizeOutput {
    /// Result schema identifier (canonical DiarizationResult v1).
    schema_version: String,
    /// Number of distinct speakers detected.
    num_speakers: usize,
    /// Audio duration, in seconds.
    duration_s: f64,
    /// Per-speaker rollup.
    speakers: Vec<SpeakerRollup>,
    /// Ordered speaker turns. Present only when verbosity = "detailed".
    #[serde(skip_serializing_if = "Option::is_none")]
    turns: Option<Vec<TurnDto>>,
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
        description = "Diarize a WAV file (who spoke when). Returns the canonical DiarizationResult v1 (concise rollup, or full turns with verbosity=detailed)."
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
fn project(result: &DiarizationResult, detailed: bool) -> DiarizeOutput {
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
    let turns = detailed.then(|| {
        result
            .turns
            .iter()
            .map(|t| TurnDto {
                speaker: t.speaker.to_string(),
                speaker_id: t.speaker.0,
                start: t.time.start,
                end: t.time.end,
            })
            .collect()
    });
    DiarizeOutput {
        schema_version: result.schema_version.clone(),
        num_speakers: result.num_speakers,
        duration_s: result.audio.duration_secs,
        speakers,
        turns,
    }
}

/// Run the production (pipeline v2) diarization path quietly, mapping failures
/// to FFI-coded MCP errors. Defaults match the CLI: VBx clusterer + registry
/// PLDA auto-download when `vbx_plda_dir` is unset.
fn run_diarize(input: &DiarizeInput) -> Result<DiarizationResult, ErrorData> {
    let path = Path::new(&input.path);
    if !path.is_file() {
        return Err(err(
            ERR_INVALID_ARG,
            format!("no such file: {}", input.path),
        ));
    }
    let profile = match input.profile.as_deref().unwrap_or("balanced") {
        "balanced" => Profile::Balanced,
        "mobile" => Profile::Mobile,
        other => {
            return Err(err(
                ERR_INVALID_ARG,
                format!("invalid profile: {other} (expected balanced|mobile)"),
            ));
        }
    };
    let clusterer_kind = match input.clusterer.as_deref().unwrap_or("vbx") {
        "vbx" => ClustererKind::Vbx,
        "ahc" => ClustererKind::Ahc {
            threshold: input.threshold.unwrap_or(0.45),
        },
        other => {
            return Err(err(
                ERR_INVALID_ARG,
                format!("invalid clusterer: {other} (expected vbx|ahc)"),
            ));
        }
    };
    let max_speakers = input
        .max_speakers
        .map(|n| u8::try_from(n).unwrap_or(u8::MAX))
        .unwrap_or_else(|| PipelineConfig::default().max_speakers);

    let registry = ModelRegistry::default().map_err(|e| err(ERR_REGISTRY, e.to_string()))?;
    // Ensure profile models exist before build (clearer error mapping).
    let _models = registry
        .ensure_for_profile(profile)
        .map_err(|e| err(ERR_MODEL_LOAD, e.to_string()))?;

    let config = PipelineConfig {
        profile,
        clusterer: clusterer_kind,
        max_speakers,
        vbx_plda_dir: input
            .vbx_plda_dir
            .as_ref()
            .map(|s| Path::new(s).to_path_buf()),
        ..PipelineConfig::default()
    };
    let pipeline = V2Pipeline::builder()
        .config(config)
        .with_models_from(registry)
        .build()
        .map_err(|e| {
            if matches!(clusterer_kind, ClustererKind::Vbx) {
                err(
                    ERR_MODEL_LOAD,
                    format!(
                        "build pipeline v2 (clusterer=vbx): {e}; set vbx_plda_dir / \
                         POLYVOICE_VBX_PLDA_DIR, allow registry PLDA download, or pass clusterer=ahc"
                    ),
                )
            } else {
                err(ERR_MODEL_LOAD, e.to_string())
            }
        })?;

    let (samples, sr_hz) = read_wav(path).map_err(|e| err(ERR_INVALID_ARG, e.to_string()))?;
    let sr = SampleRate::new(sr_hz)
        .ok_or_else(|| err(ERR_INVALID_ARG, format!("invalid sample rate {sr_hz} Hz")))?;

    pipeline
        .run(&samples, sr)
        .map_err(|e| err(ERR_INFERENCE, e.to_string()))
}

#[tokio::main]
async fn main() -> Result<()> {
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
    fn input_schema_is_strict() {
        // additionalProperties:false comes from #[serde(deny_unknown_fields)].
        let schema = schemars::schema_for!(DiarizeInput);
        let json = serde_json::to_value(&schema).unwrap();
        assert_eq!(json["additionalProperties"], serde_json::json!(false));
        assert!(json["properties"]["path"].is_object());
    }
}
