#![allow(clippy::unwrap_used)]
//! CLI-level tests for the `polyvoice-bench` and `polyvoice-mcp` binaries.
//!
//! The full-pipeline runs are gated on the ONNX models already being present
//! in the local registry cache — the tests never touch the network: when the
//! cache is cold they soft-skip exactly like the data-gated DER tests.

#![cfg(any(feature = "cli", feature = "mcp"))]

use assert_cmd::Command;
use predicates::prelude::*;
use std::path::Path;

mod common;

#[cfg(feature = "cli")]
fn bench_cmd() -> Command {
    let mut cmd = Command::cargo_bin("polyvoice-bench").unwrap();
    cmd.env("RUST_BACKTRACE", "0");
    cmd
}

/// True when every listed model file is already in the registry cache, so the
/// run below cannot trigger a download.
fn models_cached(files: &[&str]) -> bool {
    let Ok(registry) = polyvoice::models::ModelRegistry::default() else {
        return false;
    };
    files.iter().all(|f| registry.cache_dir().join(f).exists())
}

/// Skip helper: reports why a model-gated test is not running.
fn require_models(files: &[&str]) -> bool {
    if models_cached(files) {
        return true;
    }
    eprintln!("models {files:?} not in registry cache — skipping (offline test)");
    false
}

/// Write a mono 16-bit WAV of `secs` seconds of `speech_pcm` at rate `sr`.
fn write_wav(path: &Path, secs: f32, sr: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for s in common::speech_pcm(secs, sr) {
        writer
            .write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
}

/// A {audio,rttm} dataset dir: `a.wav` with a reference RTTM, `b.wav` without
/// one (exercises the skip path). Returns the tempdir root.
fn make_dataset() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let audio = dir.path().join("audio");
    let rttm = dir.path().join("rttm");
    std::fs::create_dir_all(&audio).unwrap();
    std::fs::create_dir_all(&rttm).unwrap();
    write_wav(&audio.join("a.wav"), 4.0, 16000);
    write_wav(&audio.join("b.wav"), 4.0, 16000);
    std::fs::write(
        rttm.join("a.rttm"),
        "SPEAKER a 1 0.25 1.5 <NA> <NA> spk0 <NA> <NA>\n\
         SPEAKER a 1 2.00 1.5 <NA> <NA> spk1 <NA> <NA>\n",
    )
    .unwrap();
    dir
}

// ---------------------------------------------------------------------------
// polyvoice-bench
// ---------------------------------------------------------------------------

#[cfg(feature = "cli")]
#[test]
fn bench_skip_overlap_rejects_uem() {
    // Rejected before any model loading, so this runs fully offline.
    bench_cmd()
        .args(["/tmp/whatever", "--skip-overlap", "--uem", "x.uem"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--skip-overlap cannot be combined with --uem",
        ));
}

#[cfg(feature = "cli")]
#[test]
fn bench_rejects_unknown_pipeline() {
    if !require_models(&["powerset_fp32.onnx", "wespeaker_resnet34.onnx"]) {
        return;
    }
    bench_cmd()
        .args(["/tmp/whatever", "--pipeline", "bogus"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unknown --pipeline 'bogus' (expected 'legacy' or 'v2')",
        ));
}

#[cfg(feature = "cli")]
#[test]
fn bench_rejects_unknown_profile() {
    bench_cmd()
        .args(["/tmp/whatever", "--profile", "garbage"])
        .assert()
        .failure();
}

#[cfg(feature = "cli")]
#[test]
fn bench_v2_end_to_end_writes_json_report() {
    if !require_models(&["powerset_fp32.onnx", "wespeaker_resnet34.onnx"]) {
        return;
    }
    let dataset = make_dataset();
    let report_path = dataset.path().join("report.json");
    bench_cmd()
        .args([
            dataset.path().to_str().unwrap(),
            "--pipeline",
            "v2",
            "--clusterer",
            "ahc",
            "--output",
            report_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("a\t DER="))
        .stdout(predicate::str::contains("=== Aggregate DER over 1 files"));

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    assert_eq!(json["schema"], "polyvoice-bench-v0.10");
    assert_eq!(
        json["dataset_name"],
        dataset.path().file_name().unwrap().to_str().unwrap()
    );
    assert_eq!(json["profile"], "balanced");
    assert_eq!(json["files_processed"], 1);
    assert_eq!(
        json["files_skipped"], 1,
        "b.wav has no RTTM and must be skipped"
    );
    assert!(!json["skip_overlap"].as_bool().unwrap());
    assert!((json["collar_secs"].as_f64().unwrap() - 0.25).abs() < 1e-9);
    assert!(
        !json["resolved_execution_provider"]
            .as_str()
            .unwrap()
            .is_empty()
    );
    assert!(json["host_cpus"].as_u64().unwrap() >= 1);
    assert_eq!(json["model_hashes"].as_array().unwrap().len(), 2);
    // v2 records per-stage timings per file and in the aggregate.
    assert!(json["stage_totals"].is_object());
    let per_file = json["per_file"].as_array().unwrap();
    assert_eq!(per_file.len(), 1);
    assert_eq!(per_file[0]["filename"], "a");
    assert!(per_file[0]["stage_timings"].is_object());
    assert!(per_file[0]["der_collar"].is_number());
    assert!(per_file[0]["der_no_collar"].is_number());
    assert!(per_file[0]["der_single_speaker"].is_number());
    assert!(per_file[0]["der_overlap"].is_number());
    assert!(per_file[0]["per_speaker_recall"].is_array());
    assert_eq!(per_file[0]["ref_speakers"], 2);
    assert!((per_file[0]["audio_duration_secs"].as_f64().unwrap() - 4.0).abs() < 0.05);
    // Speaker-count diagnostics add up to the number of processed files.
    let sc = &json["speaker_count"];
    let total = sc["exact"].as_u64().unwrap()
        + sc["plus_minus_1"].as_u64().unwrap()
        + sc["off_by_2_or_more"].as_u64().unwrap();
    assert_eq!(total, 1);
}

#[cfg(feature = "cli")]
#[test]
fn bench_v2_skip_overlap_mode_runs() {
    if !require_models(&["powerset_fp32.onnx", "wespeaker_resnet34.onnx"]) {
        return;
    }
    let dataset = make_dataset();
    let report_path = dataset.path().join("report.json");
    bench_cmd()
        .args([
            dataset.path().to_str().unwrap(),
            "--clusterer",
            "ahc",
            "--skip-overlap",
            "--output",
            report_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "skip-overlap: headline DER over single-speaker reference regions only",
        ));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    assert!(json["skip_overlap"].as_bool().unwrap());
    assert_eq!(json["files_processed"], 1);
}

#[cfg(feature = "cli")]
#[test]
fn bench_legacy_end_to_end_runs() {
    if !require_models(&["silero_vad.onnx", "wespeaker_resnet34.onnx"]) {
        return;
    }
    let dataset = make_dataset();
    let report_path = dataset.path().join("report.json");
    bench_cmd()
        .args([
            dataset.path().to_str().unwrap(),
            "--pipeline",
            "legacy",
            "--min-cluster-size",
            "1",
            "--output",
            report_path.to_str().unwrap(),
        ])
        .assert()
        .success();
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    assert_eq!(json["files_processed"], 1);
    // Legacy pipeline has no per-stage timings: the fields are omitted.
    assert!(json.get("stage_totals").is_none());
    assert!(json["per_file"][0].get("stage_timings").is_none());
    // The legacy arm reports the Silero VAD as its segmenter.
    let hashes = json["model_hashes"].as_array().unwrap();
    assert!(hashes.iter().any(|h| h["model_id"] == "silero_vad"));
    assert!(hashes.iter().any(|h| h["model_id"] == "wespeaker_resnet34"));
}

// ---------------------------------------------------------------------------
// polyvoice-mcp (JSON-RPC over stdio)
// ---------------------------------------------------------------------------

/// Feed newline-delimited JSON-RPC requests to the server and parse the
/// response lines back, keyed by request id. stdin stays open until every
/// expected response id arrived: the server drops in-flight tool calls when
/// stdin closes early, so a fire-and-forget write would race long-running
/// tools like `polyvoice.diarize`.
#[cfg(feature = "mcp")]
fn mcp_roundtrip(
    requests: &str,
    expect_ids: &[u64],
) -> std::collections::HashMap<u64, serde_json::Value> {
    use std::io::{BufRead, BufReader, Write};
    let bin = assert_cmd::cargo::cargo_bin("polyvoice-mcp");
    let mut child = std::process::Command::new(bin)
        .env("RUST_BACKTRACE", "0")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(requests.as_bytes())
        .unwrap();
    let mut responses = std::collections::HashMap::new();
    {
        let stdout = child.stdout.take().unwrap();
        for line in BufReader::new(stdout).lines() {
            let line = line.unwrap();
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("stdout line is not JSON-RPC: {e}: {line}"));
            if let Some(id) = v["id"].as_u64() {
                responses.insert(id, v);
                if expect_ids.iter().all(|id| responses.contains_key(id)) {
                    break;
                }
            }
        }
    }
    // All expected responses in: closing stdin lets the server exit cleanly.
    drop(child.stdin.take());
    let status = child.wait().unwrap();
    assert!(status.success(), "mcp server exited with {status}");
    for id in expect_ids {
        assert!(
            responses.contains_key(id),
            "no response for request id {id}"
        );
    }
    responses
}

#[cfg(feature = "mcp")]
fn initialize_handshake() -> String {
    let mut s = String::new();
    s.push_str(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
    );
    s.push('\n');
    s.push_str(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
    s.push('\n');
    s
}

#[cfg(feature = "mcp")]
#[test]
fn mcp_initialize_and_tools_list() {
    let mut req = initialize_handshake();
    req.push_str(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#);
    req.push('\n');
    let responses = mcp_roundtrip(&req, &[1, 2]);

    let init = &responses[&1]["result"];
    assert!(init["capabilities"]["tools"].is_object());
    assert!(
        init["instructions"]
            .as_str()
            .unwrap()
            .contains("polyvoice.diarize")
    );
    assert!(init["protocolVersion"].is_string());

    let tools = responses[&2]["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 4);
    for name in [
        "polyvoice.diarize",
        "polyvoice.transcribe",
        "polyvoice.diarize_and_transcribe",
        "polyvoice.capabilities",
    ] {
        assert!(
            tools.iter().any(|t| t["name"] == name),
            "tools/list missing {name}"
        );
    }
}

#[cfg(feature = "mcp")]
#[test]
fn mcp_capabilities_tool_lists_formats_and_profiles() {
    let mut req = initialize_handshake();
    req.push_str(
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"polyvoice.capabilities","arguments":{}}}"#,
    );
    req.push('\n');
    let responses = mcp_roundtrip(&req, &[1, 2]);
    let result = &responses[&2]["result"];
    assert!(result["error"].is_null(), "capabilities failed: {result}");
    let text = result["content"][0]["text"].as_str().unwrap();
    let cap: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(cap["name"], "polyvoice-mcp");
    assert_eq!(cap["asr_available"], false);
    assert_eq!(cap["tools"].as_array().unwrap().len(), 4);
    assert!(
        cap["output_formats"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f == "rttm")
    );
    assert!(
        cap["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p == "balanced")
    );
}

#[cfg(feature = "mcp")]
#[test]
fn mcp_transcribe_returns_coded_error() {
    let mut req = initialize_handshake();
    req.push_str(
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"polyvoice.transcribe","arguments":{"path":"x.wav"}}}"#,
    );
    req.push('\n');
    let responses = mcp_roundtrip(&req, &[1, 2]);
    let err = &responses[&2]["error"];
    assert_eq!(err["code"], -32603);
    assert_eq!(err["data"]["code"], 99);
    assert!(
        err["data"]["message"]
            .as_str()
            .unwrap()
            .contains("polyvoice-asr")
    );
}

#[cfg(feature = "mcp")]
#[test]
fn mcp_diarize_missing_file_is_invalid_params() {
    let mut req = initialize_handshake();
    req.push_str(
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"polyvoice.diarize","arguments":{"path":"/definitely/not/here.wav"}}}"#,
    );
    req.push('\n');
    let responses = mcp_roundtrip(&req, &[1, 2]);
    let err = &responses[&2]["error"];
    assert_eq!(err["code"], -32602);
    assert_eq!(err["data"]["code"], 1);
    assert!(
        err["data"]["message"]
            .as_str()
            .unwrap()
            .contains("no such file")
    );
}

#[cfg(feature = "mcp")]
#[test]
fn mcp_unknown_tool_is_method_error() {
    let mut req = initialize_handshake();
    req.push_str(
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"polyvoice.nope","arguments":{}}}"#,
    );
    req.push('\n');
    let responses = mcp_roundtrip(&req, &[1, 2]);
    assert!(responses[&2]["error"].is_object());
}

#[cfg(feature = "mcp")]
#[test]
fn mcp_diarize_end_to_end_ahc() {
    if !require_models(&["powerset_fp32.onnx", "wespeaker_resnet34.onnx"]) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("tone.wav");
    write_wav(&wav, 4.0, 16000);

    let mut req = initialize_handshake();
    req.push_str(&format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{{\"name\":\"polyvoice.diarize\",\"arguments\":{{\"path\":\"{}\",\"clusterer\":\"ahc\",\"verbosity\":\"detailed\"}}}}}}",
        wav.display()
    ));
    req.push('\n');
    let responses = mcp_roundtrip(&req, &[1, 2]);
    let result = &responses[&2]["result"];
    assert!(
        result["error"].is_null() && !result["isError"].as_bool().unwrap_or(false),
        "diarize failed: {result}"
    );
    let text = result["content"][0]["text"].as_str().unwrap();
    let out: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(!out["schema_version"].as_str().unwrap().is_empty());
    assert!((out["duration_s"].as_f64().unwrap() - 4.0).abs() < 0.05);
    assert!(out["num_speakers"].is_number());
    assert!(out["speakers"].is_array());
    // verbosity=detailed includes the ordered turns array.
    assert!(out["turns"].is_array());
}
