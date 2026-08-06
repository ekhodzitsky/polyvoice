use super::*;
use polyvoice::types::{SpeakerId, TimeRange};

fn turn(start: f64, end: f64, speaker: u32) -> SpeakerTurn {
    SpeakerTurn {
        time: TimeRange { start, end },
        speaker: SpeakerId(speaker),
        text: None,
        stable: true,
    }
}

/// `secs` seconds of a 300 Hz sine at 16 kHz, amplitude 0.3.
fn sine_pcm(secs: f32) -> Vec<f32> {
    let n = (secs * 16_000.0) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
        })
        .collect()
}

fn write_wav_16k(path: &Path, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .unwrap();
    }
    w.finalize().unwrap();
}

/// Minimal diarization dataset: one 6 s file, two speakers, speaker A with
/// two segments (so one within-file positive pair exists).
fn make_rttm_dataset() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("audio")).unwrap();
    std::fs::create_dir(dir.path().join("rttm")).unwrap();
    write_wav_16k(&dir.path().join("audio/f1.wav"), &sine_pcm(6.0));
    std::fs::write(
        dir.path().join("rttm/f1.rttm"),
        "SPEAKER f1 1 0.0 1.5 <NA> <NA> A <NA> <NA>\n\
         SPEAKER f1 1 2.0 1.5 <NA> <NA> A <NA> <NA>\n\
         SPEAKER f1 1 4.0 1.5 <NA> <NA> B <NA> <NA>\n",
    )
    .unwrap();
    dir
}

#[test]
fn macro_der_empty_is_zero() {
    assert_eq!(macro_der(&[]), (0.0, 0.0));
}

#[test]
fn macro_der_averages_both_collars() {
    let (c0, c025) = macro_der(&[(10.0, 20.0), (30.0, 40.0), (20.0, 0.0)]);
    assert!((c0 - 20.0).abs() < 1e-9);
    assert!((c025 - 20.0).abs() < 1e-9);
}

#[test]
fn der_pair_identical_turns_is_zero() {
    let turns = vec![turn(0.0, 1.0, 0), turn(1.5, 3.0, 1)];
    let (d0, d25) = der_pair(&turns, &turns.clone());
    assert_eq!(d0, 0.0);
    assert_eq!(d25, 0.0);
}

#[test]
fn der_pair_empty_hypothesis_is_full_miss() {
    let ref_t = vec![turn(0.0, 2.0, 0)];
    let (d0, d25) = der_pair(&ref_t, &[]);
    assert!((d0 - 100.0).abs() < 1e-9);
    assert!((d25 - 100.0).abs() < 1e-9);
}

#[test]
fn cosine_known_values() {
    assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    assert!((cosine(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-6);
    assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    // Zero-norm input is defined as 0 rather than NaN.
    assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    // Length mismatch: only the zipped prefix contributes.
    assert!((cosine(&[1.0, 0.0], &[1.0, 0.0, 9.0]) - 1.0).abs() < 1e-6);
}

#[test]
fn eer_empty_or_single_class_returns_degenerate_value() {
    // Degenerate inputs short-circuit to 1.0 (unlike the sweep path, which
    // returns a percentage).
    assert_eq!(eer_from_scores(vec![]), 1.0);
    assert_eq!(eer_from_scores(vec![(0.9, true), (0.8, true)]), 1.0);
    assert_eq!(eer_from_scores(vec![(0.1, false), (0.2, false)]), 1.0);
}

#[test]
fn eer_perfect_separation_is_zero() {
    let pairs = vec![(0.9_f32, true), (0.8, true), (0.2, false), (0.1, false)];
    assert_eq!(eer_from_scores(pairs), 0.0);
}

#[test]
fn eer_overlapping_scores() {
    let pairs = vec![(0.9_f32, true), (0.4, true), (0.6, false), (0.1, false)];
    assert_eq!(eer_from_scores(pairs), 50.0);
}

#[test]
fn crop_center_short_input_returned_whole() {
    let samples = vec![1.0_f32; 8_000];
    let out = crop_center(&samples, 16_000, 1.0);
    assert_eq!(out.len(), 8_000);
}

#[test]
fn crop_center_crops_symmetrically() {
    let mut samples = vec![0.0_f32; 32_000];
    samples[8_000..24_000].fill(1.0);
    let out = crop_center(&samples, 16_000, 1.0);
    assert_eq!(out.len(), 16_000);
    assert!(out.iter().all(|&x| x == 1.0));
}

#[test]
fn parse_durations_valid_list() {
    let d = parse_durations("0.5, 1.0 ,2.0,3.0").unwrap();
    assert_eq!(d, vec![0.5, 1.0, 2.0, 3.0]);
}

#[test]
fn parse_durations_skips_garbage_entries() {
    let d = parse_durations("abc,1.5,,nope").unwrap();
    assert_eq!(d, vec![1.5]);
}

#[test]
fn parse_durations_empty_or_all_garbage_errors() {
    assert!(parse_durations("").is_err());
    assert!(parse_durations(" , , ").is_err());
    assert!(parse_durations("abc").is_err());
}

#[test]
fn pairs_from_rttm_dataset_positives_and_negatives() {
    let ds = make_rttm_dataset();
    let pairs = pairs_from_rttm_dataset(ds.path(), 10, 100).unwrap();
    // One within-file positive (A,A) and one within-file negative (A,B).
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs.iter().filter(|p| p.0).count(), 1);
    assert_eq!(pairs.iter().filter(|p| !p.0).count(), 1);
    for (_, a, b) in &pairs {
        assert!(!a.is_empty());
        assert!(!b.is_empty());
    }
}

#[test]
fn pairs_from_rttm_dataset_respects_max_pairs() {
    let ds = make_rttm_dataset();
    let pairs = pairs_from_rttm_dataset(ds.path(), 10, 1).unwrap();
    assert_eq!(pairs.len(), 1);
}

#[test]
fn pairs_from_rttm_dataset_skips_files_without_rttm() {
    let ds = make_rttm_dataset();
    // Extra wav with no matching RTTM must be skipped, not fail.
    write_wav_16k(&ds.path().join("audio/f2.wav"), &sine_pcm(1.0));
    let pairs = pairs_from_rttm_dataset(ds.path(), 10, 100).unwrap();
    assert_eq!(pairs.len(), 2);
}

#[test]
fn load_verification_pairs_no_source_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let err = load_verification_pairs(&tmp.path().join("veri.txt"), tmp.path(), 10, None, 10)
        .unwrap_err();
    assert!(err.to_string().contains("no VoxCeleb pairs"));
}

#[test]
fn load_verification_pairs_from_voxceleb_list() {
    let tmp = tempfile::tempdir().unwrap();
    let wav_dir = tmp.path().join("wav");
    std::fs::create_dir(&wav_dir).unwrap();
    for name in ["a.wav", "b.wav", "c.wav"] {
        write_wav_16k(&wav_dir.join(name), &sine_pcm(1.0));
    }
    let veri = tmp.path().join("veri.txt");
    std::fs::write(
        &veri,
        "1 a.wav b.wav\n0 a.wav missing.wav\n0 b.wav c.wav\nmalformed\n",
    )
    .unwrap();
    let pairs = load_verification_pairs(&veri, tmp.path(), 10, None, 10).unwrap();
    assert_eq!(pairs.len(), 2);
    assert!(pairs[0].0);
    assert!(!pairs[1].0);
}

#[test]
fn load_verification_pairs_voxceleb_list_respects_max_pairs() {
    let tmp = tempfile::tempdir().unwrap();
    let wav_dir = tmp.path().join("wav");
    std::fs::create_dir(&wav_dir).unwrap();
    for name in ["a.wav", "b.wav"] {
        write_wav_16k(&wav_dir.join(name), &sine_pcm(1.0));
    }
    let veri = tmp.path().join("veri.txt");
    std::fs::write(&veri, "1 a.wav b.wav\n0 b.wav a.wav\n").unwrap();
    let pairs = load_verification_pairs(&veri, tmp.path(), 1, None, 10).unwrap();
    assert_eq!(pairs.len(), 1);
}

#[test]
fn load_verification_pairs_falls_back_to_rttm_dataset() {
    let ds = make_rttm_dataset();
    let tmp = tempfile::tempdir().unwrap();
    // No veri list file at all → RTTM fallback via der_dataset.
    let pairs = load_verification_pairs(
        &tmp.path().join("veri.txt"),
        tmp.path(),
        100,
        Some(ds.path()),
        10,
    )
    .unwrap();
    assert_eq!(pairs.len(), 2);
}

#[test]
fn load_verification_pairs_falls_back_to_wav_root_dataset() {
    let ds = make_rttm_dataset();
    let tmp = tempfile::tempdir().unwrap();
    // wav_root itself is a dataset directory (has audio/) → used directly.
    let pairs = load_verification_pairs(&tmp.path().join("veri.txt"), ds.path(), 100, None, 10)
        .unwrap();
    assert_eq!(pairs.len(), 2);
}

struct SignEmbedder;

impl polyvoice::embedder::Embedder for SignEmbedder {
    fn dim(&self) -> usize {
        2
    }

    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, polyvoice::EmbedderError> {
        if audio.is_empty() {
            return Err(polyvoice::EmbedderError::AudioTooShort {
                actual_secs: 0.0,
                min_secs: 0.01,
            });
        }
        let sum: f32 = audio.iter().sum();
        Ok(if sum >= 0.0 {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        })
    }
}

#[test]
fn score_arm_separable_pairs_give_zero_eer() {
    let pos = vec![1.0_f32; 16_000];
    let neg = vec![-1.0_f32; 16_000];
    let pairs: Vec<MemPair> = vec![
        (true, pos.clone(), pos.clone()),
        (false, pos.clone(), neg.clone()),
    ];
    let buckets = score_arm(&SignEmbedder, &pairs, &[0.5, 1.0]).unwrap();
    assert_eq!(buckets.len(), 2);
    for b in &buckets {
        assert_eq!(b.pairs, 2);
        assert_eq!(b.eer, 0.0);
    }
    assert_eq!(buckets[0].duration_secs, 0.5);
    assert_eq!(buckets[1].duration_secs, 1.0);
}

#[test]
fn score_arm_skips_crops_below_min_length() {
    // 0.2 s of audio: any duration ≥ 0.25 s crops below the 4000-sample floor.
    let short = vec![1.0_f32; 3_200];
    let pairs: Vec<MemPair> = vec![(true, short.clone(), short.clone())];
    let buckets = score_arm(&SignEmbedder, &pairs, &[1.0]).unwrap();
    assert_eq!(buckets.len(), 1);
    assert_eq!(buckets[0].pairs, 0);
    // No scorable pairs → the degenerate EER value.
    assert_eq!(buckets[0].eer, 1.0);
}

#[test]
fn build_embedder_report_with_der() {
    let def_eer = vec![EerBucket {
        duration_secs: 0.5,
        pairs: 10,
        eer: 12.5,
    }];
    let der = DerComparison {
        default_der: (1.0, 2.0),
        eres_der: (3.0, 4.0),
        files: 5,
    };
    let report = build_embedder_report(500, 256, 192, def_eer, vec![], Some(der));
    assert_eq!(report.schema, "polyvoice-embedder-short-v1");
    assert_eq!(report.max_pairs, 500);
    assert_eq!(report.default_embedder.dim, 256);
    assert_eq!(report.eres2netv2.dim, 192);
    assert_eq!(report.default_embedder.der_macro_collar_0, Some(1.0));
    assert_eq!(report.default_embedder.der_macro_collar_025, Some(2.0));
    assert_eq!(report.eres2netv2.der_macro_collar_0, Some(3.0));
    assert_eq!(report.eres2netv2.der_macro_collar_025, Some(4.0));
    assert_eq!(report.default_embedder.der_files, Some(5));
    assert_eq!(report.eres2netv2.der_files, Some(5));
    assert_eq!(report.default_embedder.short_seg_eer.len(), 1);
    assert!(report.eres2netv2.short_seg_eer.is_empty());
}

#[test]
fn build_embedder_report_without_der() {
    let report = build_embedder_report(10, 256, 192, vec![], vec![], None);
    assert_eq!(report.default_embedder.der_macro_collar_0, None);
    assert_eq!(report.default_embedder.der_macro_collar_025, None);
    assert_eq!(report.default_embedder.der_files, None);
    assert_eq!(report.eres2netv2.der_files, None);
}

#[test]
fn embedder_report_serializes_expected_schema() {
    let report = build_embedder_report(10, 256, 192, vec![], vec![], None);
    let v: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
    assert_eq!(v["schema"], "polyvoice-embedder-short-v1");
    assert_eq!(v["default_embedder"]["model_id"], "wespeaker_resnet34");
    assert_eq!(v["eres2netv2"]["model_id"], "eres2netv2");
    assert!(v["hardware"]["cores"].as_u64().unwrap() >= 1);
}

#[test]
fn streaming_report_serializes_expected_schema() {
    let report = StreamingReport {
        schema: "polyvoice-streaming-latency-v1".into(),
        hardware: hardware(),
        chunk_samples: 3200,
        max_files: 30,
        dataset: "data/x".into(),
        rows: vec![StreamingPresetRow {
            preset: "balanced".into(),
            window_secs: 5.0,
            hop_secs: 0.5,
            right_context_secs: 1.0,
            cache_cap: 200,
            input_buffer_latency_secs: 0.2,
            mean_rtf: 0.1,
            macro_der_collar_0: 12.0,
            macro_der_collar_025: 10.0,
            files: 3,
            total_audio_secs: 30.0,
            total_wall_secs: 3.0,
        }],
    };
    let v: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
    assert_eq!(v["schema"], "polyvoice-streaming-latency-v1");
    assert_eq!(v["rows"][0]["preset"], "balanced");
    assert_eq!(v["rows"][0]["files"], 3);
    assert_eq!(v["chunk_samples"], 3200);
}

#[cfg(feature = "vad-earshot")]
#[test]
fn vad_parity_report_serializes_expected_schema() {
    let arm = |name: &str| VadArm {
        name: name.into(),
        frame_size: 512,
        macro_der_collar_0: 10.0,
        macro_der_collar_025: 8.0,
        mean_rtf: 0.05,
        files: 2,
    };
    let report = VadParityReport {
        schema: "polyvoice-vad-parity-v1".into(),
        hardware: hardware(),
        max_files: 30,
        dataset: "data/x".into(),
        silero: arm("silero"),
        earshot: arm("earshot"),
        delta_der_collar_0_pp: 0.1,
        delta_der_collar_025_pp: -0.2,
        parity_gate_abs_pp: 0.3,
        parity_pass_collar_0: true,
        parity_pass_collar_025: true,
    };
    let v: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
    assert_eq!(v["schema"], "polyvoice-vad-parity-v1");
    assert_eq!(v["silero"]["name"], "silero");
    assert_eq!(v["earshot"]["frame_size"], 512);
    assert_eq!(v["parity_pass_collar_0"], true);
}

#[test]
fn hardware_reports_host_arch_and_cores() {
    let hw = hardware();
    assert_eq!(hw.arch, std::env::consts::ARCH);
    assert!(hw.cores >= 1);
    assert!(!hw.cpu.is_empty());
}
