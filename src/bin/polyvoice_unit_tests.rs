use super::*;
use polyvoice::types::{SpeakerId, SpeakerTurn, TimeRange};

/// DiarizeArgs with every flag at its CLI default; tests override the one
/// knob they exercise.
fn base_args() -> DiarizeArgs {
    DiarizeArgs {
        wav: None,
        profile: "balanced".to_string(),
        output: None,
        format: OutputFormat::Rttm,
        models_cache: None,
        threshold: Some(polyvoice::DEFAULT_AHC_THRESHOLD),
        speakers: None,
        max_speakers: None,
        quiet: true,
        json: false,
        legacy: false,
        v2: false,
        clusterer: "vbx".to_string(),
        vbx_plda_dir: None,
        as_norm: false,
        cohort: None,
        domain_profile: None,
        embed_window: None,
        execution_provider: "auto".to_string(),
        exclusive: false,
        latency_preset: None,
    }
}

/// Two overlapping turns so the exclusive projection differs from the
/// overlap-aware timeline.
fn sample_result() -> DiarizationResult {
    let turns = vec![
        SpeakerTurn {
            time: TimeRange {
                start: 0.0,
                end: 1.5,
            },
            speaker: SpeakerId(0),
            text: None,
            stable: true,
        },
        SpeakerTurn {
            time: TimeRange {
                start: 1.0,
                end: 2.5,
            },
            speaker: SpeakerId(1),
            text: None,
            stable: true,
        },
    ];
    DiarizationResult::new(Vec::new(), turns, 2).with_exclusive()
}

fn temp_wav_file(dir: &tempfile::TempDir) -> PathBuf {
    let p = dir.path().join("in.wav");
    std::fs::write(&p, b"not really audio").unwrap();
    p
}

#[test]
fn write_output_writes_every_format_to_file() {
    let result = sample_result();
    let dir = tempfile::tempdir().unwrap();
    let wav = Path::new("meeting.wav");
    for (format, needle) in [
        (OutputFormat::Rttm, "SPEAKER meeting 1"),
        (OutputFormat::Json, "\"turns\""),
        (OutputFormat::Srt, "-->"),
        (OutputFormat::Vtt, "WEBVTT"),
        (OutputFormat::Txt, "SPEAKER_00"),
    ] {
        let out = dir.path().join(format!("out.{format:?}"));
        write_output(&result, wav, format, false, Some(out.clone())).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        assert!(
            content.contains(needle),
            "{format:?} missing '{needle}':\n{content}"
        );
    }
}

#[test]
fn write_output_exclusive_projects_exclusive_timeline() {
    let result = sample_result();
    assert!(!result.exclusive_turns.is_empty());
    let dir = tempfile::tempdir().unwrap();
    let wav = Path::new("meeting.wav");
    let shared = dir.path().join("shared.rttm");
    let exclusive = dir.path().join("exclusive.rttm");
    write_output(
        &result,
        wav,
        OutputFormat::Rttm,
        false,
        Some(shared.clone()),
    )
    .unwrap();
    write_output(
        &result,
        wav,
        OutputFormat::Rttm,
        true,
        Some(exclusive.clone()),
    )
    .unwrap();
    // Overlapping turns collapse in the exclusive projection, so the two
    // timelines must serialize differently.
    assert_ne!(
        std::fs::read_to_string(shared).unwrap(),
        std::fs::read_to_string(exclusive).unwrap()
    );
}

#[test]
fn write_output_json_keeps_both_timelines_when_exclusive() {
    let result = sample_result();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.json");
    write_output(
        &result,
        Path::new("a.wav"),
        OutputFormat::Json,
        true,
        Some(out.clone()),
    )
    .unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert!(json["turns"].is_array());
    assert!(json["exclusive_turns"].is_array());
}

#[test]
fn write_output_falls_back_to_audio_file_id() {
    let result = sample_result();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.rttm");
    // A path without a file stem exercises the "audio" fallback id.
    write_output(
        &result,
        Path::new("/"),
        OutputFormat::Rttm,
        false,
        Some(out.clone()),
    )
    .unwrap();
    assert!(
        std::fs::read_to_string(&out)
            .unwrap()
            .contains("SPEAKER audio 1")
    );
}

#[test]
fn write_output_to_stdout_succeeds() {
    // The None branch streams to stdout (captured by the test harness).
    write_output(
        &sample_result(),
        Path::new("a.wav"),
        OutputFormat::Txt,
        false,
        None,
    )
    .unwrap();
}

#[test]
fn write_output_report_write_failure() {
    let result = sample_result();
    let err = write_output(
        &result,
        Path::new("a.wav"),
        OutputFormat::Rttm,
        false,
        Some(PathBuf::from("/nonexistent/dir/out.rttm")),
    )
    .err()
    .unwrap();
    assert!(format!("{err:#}").contains("write"));
}

#[test]
fn cmd_diarize_without_input_errors() {
    let err = cmd_diarize(base_args()).err().unwrap();
    assert!(format!("{err:#}").contains("no input"));
}

#[test]
fn cmd_diarize_missing_file_errors() {
    let mut args = base_args();
    args.wav = Some(PathBuf::from("/nonexistent/dir/file.wav"));
    let err = cmd_diarize(args).err().unwrap();
    assert!(format!("{err:#}").contains("No such file"));
}

#[test]
fn cmd_diarize_invalid_profile_errors() {
    let mut args = base_args();
    args.wav = Some(PathBuf::from("/nonexistent/dir/file.wav"));
    args.profile = "garbage".to_string();
    assert!(cmd_diarize(args).is_err());
}

#[test]
fn cmd_diarize_rejects_models_cache_traversal() {
    let dir = tempfile::tempdir().unwrap();
    let mut args = base_args();
    args.wav = Some(temp_wav_file(&dir));
    args.models_cache = Some(PathBuf::from("../escape"));
    let err = cmd_diarize(args).err().unwrap();
    assert!(format!("{err:#}").contains("path traversal"));
}

#[test]
fn cmd_diarize_rejects_bad_latency_preset() {
    let dir = tempfile::tempdir().unwrap();
    let mut args = base_args();
    args.wav = Some(temp_wav_file(&dir));
    args.models_cache = Some(dir.path().join("cache"));
    args.latency_preset = Some("warp-speed".to_string());
    let err = cmd_diarize(args).err().unwrap();
    assert!(format!("{err:#}").contains("invalid --latency-preset"));
}

#[test]
fn cmd_diarize_accepts_known_latency_preset_then_fails_later() {
    let dir = tempfile::tempdir().unwrap();
    let mut args = base_args();
    args.wav = Some(temp_wav_file(&dir));
    args.models_cache = Some(dir.path().join("cache"));
    args.latency_preset = Some("realtime".to_string());
    // The preset parses; the run then stops at the unknown clusterer,
    // before any model resolution.
    args.clusterer = "kmeans".to_string();
    let err = cmd_diarize(args).err().unwrap();
    assert!(format!("{err:#}").contains("unknown --clusterer"));
}

#[test]
fn cmd_diarize_rejects_unknown_clusterer() {
    let dir = tempfile::tempdir().unwrap();
    let mut args = base_args();
    args.wav = Some(temp_wav_file(&dir));
    args.models_cache = Some(dir.path().join("cache"));
    args.clusterer = "kmeans".to_string();
    let err = cmd_diarize(args).err().unwrap();
    assert!(format!("{err:#}").contains("unknown --clusterer"));
}

#[test]
fn cmd_diarize_rejects_unknown_execution_provider() {
    let dir = tempfile::tempdir().unwrap();
    let mut args = base_args();
    args.wav = Some(temp_wav_file(&dir));
    args.models_cache = Some(dir.path().join("cache"));
    args.clusterer = "ahc".to_string();
    args.execution_provider = "tpu".to_string();
    let err = cmd_diarize(args).err().unwrap();
    assert!(format!("{err:#}").contains("unknown --execution-provider"));
}

#[test]
fn cmd_diarize_rejects_oversized_max_speakers() {
    let dir = tempfile::tempdir().unwrap();
    let mut args = base_args();
    args.wav = Some(temp_wav_file(&dir));
    args.models_cache = Some(dir.path().join("cache"));
    args.clusterer = "ahc".to_string();
    args.max_speakers = Some(300);
    let err = cmd_diarize(args).err().unwrap();
    assert!(format!("{err:#}").contains("max_speakers must be in 1..=255"));
}

#[test]
fn cmd_models_list_runs() {
    cmd_models_list().unwrap();
}

#[test]
fn cmd_models_info_direct_id_runs() {
    cmd_models_info("silero_vad".to_string()).unwrap();
}

#[test]
fn cmd_models_info_resolves_stage_alias() {
    // "latest" is not a model id; the stage-alias fallback resolves it.
    cmd_models_info("latest".to_string()).unwrap();
}

#[test]
fn cmd_models_info_prints_calibration_when_present() {
    cmd_models_info("powerset_int8".to_string()).unwrap();
}

#[test]
fn cmd_models_info_unknown_model_errors() {
    let err = cmd_models_info("no_such_model_zzz".to_string())
        .err()
        .unwrap();
    assert!(format!("{err:#}").contains("not found in manifest"));
}

#[test]
fn cmd_download_models_rejects_unknown_profile() {
    assert!(cmd_download_models("garbage".to_string()).is_err());
}

#[test]
fn cmd_download_models_custom_profile_is_unresolvable() {
    // "custom" parses as a profile but the registry cannot resolve it —
    // this fails before any network access.
    let err = cmd_download_models("custom".to_string()).err().unwrap();
    assert!(format!("{err:#}").contains("custom"));
}

#[test]
fn cmd_completions_generates_for_every_shell() {
    for shell in [
        clap_complete::Shell::Bash,
        clap_complete::Shell::Zsh,
        clap_complete::Shell::Fish,
        clap_complete::Shell::PowerShell,
        clap_complete::Shell::Elvish,
    ] {
        cmd_completions(shell).unwrap();
    }
}

#[test]
fn cmd_schema_prints_committed_schema() {
    cmd_schema().unwrap();
}
