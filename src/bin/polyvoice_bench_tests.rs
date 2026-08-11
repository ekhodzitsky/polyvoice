use super::*;

/// Shipped model file in the repo (content matches the embedded manifest).
/// Paths may be nested (e.g. `int8/resnet34_int8.onnx`).
fn repo_model(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name)
}

/// `false` (test skips) when the gitignored model blob is absent locally.
fn has_model(name: &str) -> bool {
    if repo_model(name).exists() {
        true
    } else {
        eprintln!("skip: models/{name} missing");
        false
    }
}

/// Synthetic DerResult: `errors` miss frames over `ref_frames`.
fn synth_der(errors: u64, ref_frames: u64) -> DerResult {
    DerResult {
        der: errors as f64 / ref_frames as f64,
        miss_rate: errors as f64 / ref_frames as f64,
        false_alarm_rate: 0.0,
        confusion_rate: 0.0,
        total_speech: ref_frames as f64 * 0.01,
        total_ref_frames: ref_frames,
        missed_frames: errors,
        false_alarm_frames: 0,
        confusion_frames: 0,
    }
}

/// A scored-file outcome with all-miss DER and fixed 10s/1s audio/runtime.
fn outcome(
    ref_count: usize,
    hyp_count: usize,
    errors: u64,
    ref_frames: u64,
    stage_timings: Option<StageTimings>,
) -> FileOutcome {
    let der = synth_der(errors, ref_frames);
    let row = PerFileResult {
        filename: "file".to_owned(),
        der_collar: der.der * 100.0,
        der_no_collar: der.der * 100.0,
        miss_rate: der.miss_rate * 100.0,
        false_alarm_rate: 0.0,
        confusion_rate: 0.0,
        der_single_speaker: 0.0,
        der_overlap: 0.0,
        per_speaker_recall: vec![],
        rt_factor: 10.0,
        ref_speakers: ref_count,
        hyp_speakers: hyp_count,
        num_turns: 0,
        audio_duration_secs: 10.0,
        runtime_secs: 1.0,
        stage_timings,
    };
    FileOutcome {
        row,
        der_pair: (der, der),
        ref_count,
        hyp_count,
        audio_secs: 10.0,
        runtime_secs: 1.0,
    }
}

fn default_args() -> Args {
    Args::try_parse_from(["polyvoice-bench", "/tmp/dataset"]).unwrap()
}

#[test]
fn hex_lower_formats_bytes_as_two_digit_hex() {
    assert_eq!(hex_lower(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
    assert_eq!(hex_lower(&[]), "");
}

#[test]
fn git_sha_is_nonempty_in_or_out_of_a_checkout() {
    // Inside the repo this is the 40-char HEAD; in a bare export it falls
    // back to "unknown" — either way never empty.
    assert!(!git_sha().is_empty());
}

#[test]
fn micro_der_is_error_frames_over_reference_frames() {
    assert!((micro_der(10, 5, 5, 200) - 10.0).abs() < 1e-9);
    assert!((micro_der(1, 0, 0, 4) - 25.0).abs() < 1e-9);
}

#[test]
fn micro_der_zero_reference_frames_is_zero() {
    assert_eq!(micro_der(0, 0, 0, 0), 0.0);
    assert_eq!(micro_der(10, 5, 5, 0), 0.0);
}

#[test]
fn model_hashes_reports_segmenter_and_profile_embedder() {
    let registry = ModelRegistry::default().unwrap();
    let hashes = model_hashes(&registry, Profile::Balanced, "powerset_int8");
    assert_eq!(hashes.len(), 2);
    assert_eq!(hashes[0].model_id, "powerset_int8");
    assert_eq!(hashes[1].model_id, "resnet34_int8");
    for h in &hashes {
        assert_eq!(h.sha256.len(), 64, "{} sha256 must be hex", h.model_id);
    }
}

#[test]
fn model_hashes_skips_models_absent_from_manifest() {
    let registry = ModelRegistry::default().unwrap();
    // Unknown segmenter id: only the embedder entry survives the lookup.
    let hashes = model_hashes(&registry, Profile::Balanced, "no_such_model");
    assert_eq!(hashes.len(), 1);
    assert_eq!(hashes[0].model_id, "resnet34_int8");
}

#[test]
fn model_hashes_empty_for_profile_absent_from_manifest() {
    let registry = ModelRegistry::default().unwrap();
    assert!(model_hashes(&registry, Profile::Custom, "powerset_fp32").is_empty());
}

#[test]
fn check_model_sha256_accepts_shipped_model() {
    if !has_model("silero_vad.onnx") {
        return;
    }
    let registry = ModelRegistry::default().unwrap();
    check_model_sha256(&registry, "silero_vad", &repo_model("silero_vad.onnx")).unwrap();
}

#[test]
fn check_model_sha256_rejects_corrupted_model() {
    let registry = ModelRegistry::default().unwrap();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"not an onnx model").unwrap();
    let e = check_model_sha256(&registry, "silero_vad", tmp.path()).unwrap_err();
    let msg = format!("{e:#}");
    assert!(msg.contains("integrity FAIL"), "{msg}");
    assert!(msg.contains("silero_vad"), "{msg}");
}

#[test]
fn check_model_sha256_unknown_model_id_errors() {
    let registry = ModelRegistry::default().unwrap();
    let e =
        check_model_sha256(&registry, "no_such_model", &repo_model("silero_vad.onnx")).unwrap_err();
    assert!(format!("{e:#}").contains("not in manifest"));
}

#[test]
fn check_model_sha256_missing_file_errors() {
    let registry = ModelRegistry::default().unwrap();
    let e = check_model_sha256(
        &registry,
        "silero_vad",
        Path::new("/nonexistent/model.onnx"),
    )
    .unwrap_err();
    assert!(format!("{e:#}").contains("read model"));
}

#[test]
fn verify_model_integrity_accepts_shipped_pair() {
    // Balanced profile embedder is resnet34_int8 (0.17+); VAD is still silero.
    if !has_model("int8/resnet34_int8.onnx") || !has_model("silero_vad.onnx") {
        return;
    }
    let registry = ModelRegistry::default().unwrap();
    verify_model_integrity(
        &registry,
        Profile::Balanced,
        &repo_model("int8/resnet34_int8.onnx"),
        &repo_model("silero_vad.onnx"),
    )
    .unwrap();
}

#[test]
fn verify_model_integrity_rejects_swapped_vad() {
    if !has_model("int8/resnet34_int8.onnx") {
        return;
    }
    let registry = ModelRegistry::default().unwrap();
    // The embedder file standing in as the VAD fails the sha256 gate.
    let e = verify_model_integrity(
        &registry,
        Profile::Balanced,
        &repo_model("int8/resnet34_int8.onnx"),
        &repo_model("int8/resnet34_int8.onnx"),
    )
    .unwrap_err();
    assert!(format!("{e:#}").contains("integrity FAIL"));
}

#[test]
fn accum_record_buckets_speaker_count_accuracy() {
    let mut acc = Accum::default();
    acc.record(outcome(2, 2, 10, 100, None)); // exact
    acc.record(outcome(2, 3, 10, 100, None)); // off by one
    acc.record(outcome(2, 1, 10, 100, None)); // off by one (other side)
    acc.record(outcome(2, 5, 10, 100, None)); // off by >= 2
    assert_eq!(acc.totals.count, 4);
    assert_eq!(acc.speaker_exact, 1);
    assert_eq!(acc.speaker_pm1, 2);
    assert_eq!(acc.speaker_off, 1);
    assert_eq!(acc.der_pairs.len(), 4);
    assert!((acc.total_audio_secs - 40.0).abs() < 1e-9);
    assert!((acc.total_runtime_secs - 4.0).abs() < 1e-9);
    // All-miss DER at 10% per file feeds the error-rate totals.
    assert!((acc.totals.miss - 0.4).abs() < 1e-9);
    // No stage timings recorded -> the report omits stage_totals.
    assert!(acc.stage_totals.is_none());
}

#[test]
fn accum_record_sums_stage_timings_across_files() {
    let t = |s: f64| StageTimings {
        segmentation_secs: s,
        embedding_secs: s * 2.0,
        clustering_secs: s * 3.0,
        resegmentation_secs: s * 4.0,
    };
    let mut acc = Accum::default();
    acc.record(outcome(1, 1, 0, 100, Some(t(1.0))));
    acc.record(outcome(1, 1, 0, 100, None)); // legacy-style row: no timings
    acc.record(outcome(1, 1, 0, 100, Some(t(0.5))));
    let totals = acc.stage_totals.unwrap();
    assert!((totals.segmentation_secs - 1.5).abs() < 1e-9);
    assert!((totals.embedding_secs - 3.0).abs() < 1e-9);
    assert!((totals.clustering_secs - 4.5).abs() < 1e-9);
    assert!((totals.resegmentation_secs - 6.0).abs() < 1e-9);
}

#[test]
fn build_report_assembles_serializable_report() {
    let args = default_args();
    let registry = ModelRegistry::default().unwrap();
    let mut acc = Accum {
        files_skipped: 1,
        ..Accum::default()
    };
    acc.record(outcome(2, 2, 50, 1000, Some(StageTimings::default())));
    acc.record(outcome(3, 2, 25, 500, None));
    let report = build_report(
        &args,
        &registry,
        Profile::Balanced,
        "powerset_fp32",
        polyvoice::onnx::ExecutionProvider::Cpu,
        "dataset".to_owned(),
        acc,
    );
    assert_eq!(report.schema, "polyvoice-bench-v0.10");
    assert_eq!(report.files_processed, 2);
    assert_eq!(report.files_skipped, 1);
    assert_eq!(report.dataset_name, "dataset");
    assert_eq!(report.profile, "balanced");
    assert!((report.collar_secs - 0.25).abs() < 1e-9);
    assert!(!report.skip_overlap);
    assert_eq!(report.resolved_execution_provider, "Cpu");
    assert!(report.host_cpus >= 1);
    assert!(report.stage_totals.is_some());
    assert_eq!(report.speaker_count.exact, 1);
    assert_eq!(report.speaker_count.plus_minus_1, 1);
    assert_eq!(report.speaker_count.off_by_2_or_more, 0);
    assert_eq!(report.model_hashes.len(), 2);
    assert_eq!(report.per_file.len(), 2);
    // Both files at 5% all-miss DER: macro and micro agree.
    assert!((report.der_collar_macro - 5.0).abs() < 1e-9);
    assert!((report.der_collar_micro - 5.0).abs() < 1e-9);
    assert!((report.miss - 5.0).abs() < 1e-9);
    // 20s audio over 2s runtime per run pair.
    assert!((report.rt_factor_avg - 10.0).abs() < 1e-9);
    // The whole report must round-trip as JSON for --output.
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(
        json["averaging_policy"].as_str().unwrap(),
        report.averaging_policy
    );
    assert!(json["per_file"][0]["stage_timings"].is_object());
    assert!(json["per_file"][1].get("stage_timings").is_none());
}

#[test]
fn build_report_empty_accumulator_does_not_panic() {
    let args = default_args();
    let registry = ModelRegistry::default().unwrap();
    let report = build_report(
        &args,
        &registry,
        Profile::Balanced,
        "powerset_fp32",
        polyvoice::onnx::ExecutionProvider::Cpu,
        "empty".to_owned(),
        Accum::default(),
    );
    assert_eq!(report.files_processed, 0);
    assert_eq!(report.der_collar_macro, 0.0);
    assert_eq!(report.miss, 0.0);
    assert!(report.stage_totals.is_none());
}

#[test]
fn args_reject_unknown_flag_and_missing_dataset() {
    assert!(Args::try_parse_from(["polyvoice-bench"]).is_err());
    assert!(Args::try_parse_from(["polyvoice-bench", "/tmp/ds", "--bogus"]).is_err());
    // skip-overlap parses as a plain flag.
    let args = Args::try_parse_from(["polyvoice-bench", "/tmp/ds", "--skip-overlap"]).unwrap();
    assert!(args.skip_overlap);
}
