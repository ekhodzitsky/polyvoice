//! Ort vs tract numerical parity harness (feature `backend-tract`).
//!
//! Behaviour when models are missing under `models/`: tests **skip cleanly**
//! (pass with an `eprintln!`), so CI without ONNX blobs stays green.
//!
//! When models are present:
//! - feed-forward embedders (cam++ / resnet34) are expected to load and match
//!   ort within fixed tolerances;
//! - silero / powerset may fail to load on tract — that is recorded as a
//!   documented incompatibility (test still passes; see the verdict report).

#![cfg(all(test, feature = "backend-tract"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::ExecutionProvider;
use super::factory::{InferenceBackend, RuntimeSession};
use super::runtime::{InferenceRuntime, InferenceTensor, NamedTensor, TensorData};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Absolute + relative tolerance for f32 outputs.
#[derive(Clone, Copy)]
struct Tol {
    abs: f32,
    rel: f32,
}

impl Tol {
    const DEFAULT: Self = Self {
        abs: 1e-3,
        rel: 1e-2,
    };
}

fn model_path(name: &str) -> Option<PathBuf> {
    // Prefer `models/<name>`, then `models/int8/<name>` (shipping INT8 pair).
    for base in ["models", "models/int8"] {
        let p = Path::new(base).join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn try_open(path: &Path, backend: InferenceBackend) -> Result<RuntimeSession, String> {
    InferenceBackend::force(Some(backend));
    let result = RuntimeSession::from_path(path, ExecutionProvider::Cpu, Some(1));
    InferenceBackend::force(None);
    result.map_err(|e| e.to_string())
}

fn max_abs_rel(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        let scale = x.abs().max(y.abs()).max(1e-8);
        max_rel = max_rel.max(d / scale);
    }
    (max_abs, max_rel)
}

fn assert_f32_close(label: &str, a: &InferenceTensor, b: &InferenceTensor, tol: Tol) {
    assert_eq!(
        a.shape, b.shape,
        "{label}: shape mismatch {:?} vs {:?}",
        a.shape, b.shape
    );
    let aa = a.as_f32_slice().expect("a f32");
    let bb = b.as_f32_slice().expect("b f32");
    let (max_abs, max_rel) = max_abs_rel(aa, bb);
    assert!(
        max_abs <= tol.abs || max_rel <= tol.rel,
        "{label}: max_abs={max_abs:.6e} (tol {}) max_rel={max_rel:.6e} (tol {})",
        tol.abs,
        tol.rel
    );
}

fn compare_ordered(
    label: &str,
    path: &Path,
    inputs: &[InferenceTensor],
    tol: Tol,
) -> Result<(f64, f64), String> {
    let mut ort = try_open(path, InferenceBackend::Ort)?;
    let mut tract = try_open(path, InferenceBackend::Tract)?;

    let refs: Vec<&InferenceTensor> = inputs.iter().collect();

    let t0 = Instant::now();
    let ort_out = ort
        .run_ordered(&refs)
        .map_err(|e| format!("ort run: {e}"))?;
    let ort_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let tract_out = tract
        .run_ordered(&refs)
        .map_err(|e| format!("tract run: {e}"))?;
    let tract_ms = t1.elapsed().as_secs_f64() * 1e3;

    assert_eq!(
        ort_out.len(),
        tract_out.len(),
        "{label}: output count {} vs {}",
        ort_out.len(),
        tract_out.len()
    );
    for (i, (o, t)) in ort_out.iter().zip(tract_out.iter()).enumerate() {
        match (&o.data, &t.data) {
            (TensorData::F32(_), TensorData::F32(_)) => {
                assert_f32_close(&format!("{label} out[{i}]"), o, t, tol);
            }
            (TensorData::I64(a), TensorData::I64(b)) => {
                assert_eq!(a, b, "{label} out[{i}] i64 mismatch");
            }
            _ => panic!("{label} out[{i}]: type mismatch ort vs tract"),
        }
    }
    eprintln!(
        "parity {label}: ort={ort_ms:.2}ms tract={tract_ms:.2}ms ratio={:.2}x",
        tract_ms / ort_ms.max(1e-9)
    );
    Ok((ort_ms, tract_ms))
}

/// Report load status for every known model; never fails the suite.
#[test]
#[cfg_attr(miri, ignore)]
fn tract_per_model_load_status() {
    let models = [
        "silero_vad.onnx",
        "powerset_fp32.onnx",
        "powerset_int8.onnx",
        "cam_pp_fp32.onnx",
        "wespeaker_resnet34.onnx",
        "resnet34_int8.onnx",
        "ecapa_tdnn_mel.onnx",
    ];
    for name in models {
        let Some(path) = model_path(name) else {
            eprintln!("load-status {name}: MISSING (skip)");
            continue;
        };
        match try_open(&path, InferenceBackend::Tract) {
            Ok(s) => eprintln!(
                "load-status {name}: OK backend={:?} inputs={:?} path={}",
                s.backend(),
                s.input_names(),
                path.display()
            ),
            Err(e) => eprintln!("load-status {name}: FAIL — {e}"),
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn parity_cam_pp_if_present() {
    let Some(path) = model_path("cam_pp_fp32.onnx") else {
        eprintln!("skip parity_cam_pp: models/cam_pp_fp32.onnx missing");
        return;
    };
    let time = 200usize;
    let n_mels = 80usize;
    let input = InferenceTensor::f32(vec![1, time, n_mels], vec![0.05f32; time * n_mels]);
    compare_ordered("cam_pp_fp32", &path, &[input], Tol::DEFAULT)
        .unwrap_or_else(|e| panic!("cam_pp parity failed: {e}"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn parity_wespeaker_resnet34_if_present() {
    let Some(path) = model_path("wespeaker_resnet34.onnx") else {
        eprintln!("skip parity_resnet34: models/wespeaker_resnet34.onnx missing");
        return;
    };
    let time = 200usize;
    let n_mels = 80usize;
    let input = InferenceTensor::f32(vec![1, time, n_mels], vec![0.05f32; time * n_mels]);
    compare_ordered("wespeaker_resnet34", &path, &[input], Tol::DEFAULT)
        .unwrap_or_else(|e| panic!("resnet34 parity failed: {e}"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn silero_tract_load_documents_status() {
    let Some(path) = model_path("silero_vad.onnx") else {
        eprintln!("skip silero status: models/silero_vad.onnx missing");
        return;
    };
    // Known incompatibility in tract 0.23: nested If/Squeeze analysis fails on
    // the shipped Silero ONNX. Document via eprintln; do not fail the suite.
    match try_open(&path, InferenceBackend::Tract) {
        Ok(_) => {
            // If a future tract version loads it, also check a zero-state step.
            let mut tract = try_open(&path, InferenceBackend::Tract).unwrap();
            let chunk = 512usize;
            let context = 64usize;
            let input_len = context + chunk;
            let input = InferenceTensor::f32(vec![1, input_len], vec![0.01f32; input_len]);
            let state = InferenceTensor::f32(vec![2, 1, 128], vec![0.0f32; 2 * 128]);
            let sr = InferenceTensor::i64_scalar(16_000);
            let out = tract
                .run(&[
                    NamedTensor::new("input", &input),
                    NamedTensor::new("state", &state),
                    NamedTensor::new("sr", &sr),
                ])
                .expect("silero run after successful load");
            assert!(out.len() >= 2);
            eprintln!("silero_vad: tract LOAD+RUN OK (unexpected win — update verdict)");
        }
        Err(e) => {
            eprintln!("silero_vad: tract LOAD FAIL (documented): {e}");
            assert!(
                e.contains("If")
                    || e.contains("into_optimized")
                    || e.contains("analyse")
                    || e.contains("tract"),
                "unexpected silero load error shape: {e}"
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn powerset_tract_load_documents_status() {
    for name in ["powerset_fp32.onnx", "powerset_int8.onnx"] {
        let Some(path) = model_path(name) else {
            eprintln!("skip powerset status: {name} missing");
            continue;
        };
        match try_open(&path, InferenceBackend::Tract) {
            Ok(_) => eprintln!("{name}: tract LOAD OK (unexpected win — update verdict)"),
            Err(e) => {
                eprintln!("{name}: tract LOAD FAIL (documented): {e}");
                assert!(
                    e.contains("If")
                        || e.contains("InstanceNorm")
                        || e.contains("into_optimized")
                        || e.contains("analyse")
                        || e.contains("tract"),
                    "unexpected powerset load error shape: {e}"
                );
            }
        }
    }
}

/// Shipping INT8 embedder must **load and run** on pure-Rust tract.
///
/// Bit-parity with ort is **not** required: dynamic INT8 scales differ across
/// runtimes (same reason powerset micro-batch N>1 is not bit-identical). FP32
/// ResNet34 remains the numerical parity reference (`parity_wespeaker_resnet34`).
#[test]
#[cfg_attr(miri, ignore)]
fn resnet34_int8_tract_load_and_run_if_present() {
    let Some(path) = model_path("resnet34_int8.onnx") else {
        eprintln!("skip resnet34_int8 tract: models/int8/resnet34_int8.onnx missing");
        return;
    };
    let mut tract = try_open(&path, InferenceBackend::Tract).unwrap_or_else(|e| {
        panic!("resnet34_int8 must load on tract (zero-deps embedder path): {e}")
    });
    let time = 200usize;
    let n_mels = 80usize;
    let input = InferenceTensor::f32(vec![1, time, n_mels], vec![0.05f32; time * n_mels]);
    let out = tract
        .run_ordered(&[&input])
        .unwrap_or_else(|e| panic!("resnet34_int8 tract run failed: {e}"));
    assert!(!out.is_empty(), "expected at least one output tensor");
    let emb = out[0].as_f32_slice().expect("f32 embedding");
    assert!(
        emb.iter().all(|v| v.is_finite()),
        "resnet34_int8 tract output must be finite"
    );
    eprintln!(
        "resnet34_int8: tract LOAD+RUN OK dim={} (ort bit-parity not required for INT8)",
        emb.len()
    );
}

/// Tract-friendly powerset rewrite (`scripts/export-powerset-tract.py`):
/// InstanceNorm expanded + identical `If` inlined.
///
/// - **Load+run** at product window T=160000 (10 s @ 16 kHz).
/// - **Tight ort parity** on a 1 s window (T=16000): long-window LSTM paths
///   can diverge more under tract; rewrite vs original is verified by the
///   export script's `--verify` (ort-only).
#[test]
#[cfg_attr(miri, ignore)]
fn powerset_fp32_tract_friendly_load_and_parity_if_present() {
    let Some(path) = model_path("powerset_fp32_tract.onnx") else {
        eprintln!("skip powerset_fp32_tract: models/powerset_fp32_tract.onnx missing");
        return;
    };

    // 10 s product window: must load+run (finite logits).
    let t_long = 160_000usize;
    let input_long = InferenceTensor::f32(vec![1, 1, t_long], vec![0.01f32; t_long]);
    let mut tract = try_open(&path, InferenceBackend::Tract)
        .unwrap_or_else(|e| panic!("powerset_fp32_tract load failed: {e}"));
    let long_out = tract
        .run_ordered(&[&input_long])
        .unwrap_or_else(|e| panic!("powerset_fp32_tract run T={t_long} failed: {e}"));
    assert!(!long_out.is_empty());
    let logits = long_out[0].as_f32_slice().expect("f32");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "powerset_fp32_tract long-window logits must be finite"
    );
    eprintln!(
        "powerset_fp32_tract: LOAD+RUN OK T={t_long} out_shape={:?} values={}",
        long_out[0].shape,
        logits.len()
    );

    // Tight parity on 1 s window (export script verifies rewrite vs original on ort).
    let t_short = 16_000usize;
    let input_short = InferenceTensor::f32(vec![1, 1, t_short], vec![0.01f32; t_short]);
    // Re-open: concrete plan may be for T=160k; short window may need T=16k plan.
    let (ort_ms, tract_ms) = compare_ordered(
        "powerset_fp32_tract_1s",
        &path,
        &[input_short],
        Tol::DEFAULT,
    )
    .unwrap_or_else(|e| panic!("powerset_fp32_tract 1s ort/tract parity failed: {e}"));
    eprintln!("powerset_fp32_tract: 1s PARITY OK ort={ort_ms:.1}ms tract={tract_ms:.1}ms");
}

/// Diagnostic: product 10 s window ort vs tract on the rewrite graph.
///
/// Does **not** hard-fail the suite: reports max-abs / argmax agreement so we
/// can track whether full-window drift explains the DER collapse. Run with
/// `--nocapture`.
#[test]
#[cfg_attr(miri, ignore)]
fn powerset_fp32_tract_10s_parity_report() {
    let Some(path) = model_path("powerset_fp32_tract.onnx") else {
        eprintln!("skip powerset 10s report: missing rewrite");
        return;
    };
    let t = 160_000usize;
    // Deterministic non-constant signal (constant 0.01 was 1 s parity input).
    let mut wave = vec![0.0f32; t];
    for (i, s) in wave.iter_mut().enumerate() {
        let x = i as f32 / 16_000.0;
        *s = (2.0 * std::f32::consts::PI * 220.0 * x).sin() * 0.2
            + (2.0 * std::f32::consts::PI * 440.0 * x).sin() * 0.1;
    }
    let input = InferenceTensor::f32(vec![1, 1, t], wave);

    let mut ort = try_open(&path, InferenceBackend::Ort).expect("ort load");
    let mut tract = try_open(&path, InferenceBackend::Tract).expect("tract load");
    eprintln!(
        "powerset 10s inputs: ort={:?} tract={:?}",
        ort.input_names(),
        tract.input_names()
    );

    let ort_out = ort.run_ordered(&[&input]).expect("ort run");
    let tract_out = tract.run_ordered(&[&input]).expect("tract run");
    assert_eq!(ort_out[0].shape, tract_out[0].shape, "shape");
    let o = ort_out[0].as_f32_slice().unwrap();
    let tr = tract_out[0].as_f32_slice().unwrap();
    let (max_abs, max_rel) = max_abs_rel(o, tr);

    // Logits layout [1, F, 7] or [F, 7] — last dim is powerset classes.
    let shape = &ort_out[0].shape;
    let n_classes = *shape.last().unwrap_or(&7);
    assert_eq!(n_classes, 7, "expected 7 powerset classes, shape={shape:?}");
    let n_frames = o.len() / n_classes;
    let mut argmax_agree = 0usize;
    let mut max_frame_abs = 0.0f32;
    for f in 0..n_frames {
        let base = f * n_classes;
        let o_row = &o[base..base + n_classes];
        let t_row = &tr[base..base + n_classes];
        let o_arg = o_row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let t_arg = t_row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if o_arg == t_arg {
            argmax_agree += 1;
        }
        for c in 0..n_classes {
            max_frame_abs = max_frame_abs.max((o_row[c] - t_row[c]).abs());
        }
    }
    let agree_pct = 100.0 * argmax_agree as f64 / n_frames.max(1) as f64;
    eprintln!(
        "powerset 10s ort/tract: shape={shape:?} max_abs={max_abs:.6e} max_rel={max_rel:.6e} \
         argmax_agree={argmax_agree}/{n_frames} ({agree_pct:.1}%) max_frame_abs={max_frame_abs:.6e}"
    );
    // Soft signal: product DER needs high argmax agreement. Keep as report unless
    // both sides are garbage (non-finite already checked by run).
    assert!(o.iter().all(|v| v.is_finite()) && tr.iter().all(|v| v.is_finite()));
}

/// Real-speech 10 s window (first window of a short Vox file if present).
#[test]
#[cfg_attr(miri, ignore)]
fn powerset_fp32_tract_10s_real_audio_report() {
    let Some(path) = model_path("powerset_fp32_tract.onnx") else {
        eprintln!("skip real-audio powerset: missing rewrite");
        return;
    };
    // Prefer committed smoke path, then full corpus, then skip.
    let wav_candidates = [
        "benchmarks/results/powerset-tract-rtf-der-2026-08-12/smoke-vox3/audio/fuzfh.wav",
        "data/voxconverse-test/audio/fuzfh.wav",
    ];
    let mut samples: Option<Vec<f32>> = None;
    for w in wav_candidates {
        let p = Path::new(w);
        if !p.is_file() {
            continue;
        }
        match crate::wav::read_wav(p) {
            Ok((s, sr)) => {
                assert_eq!(sr, 16_000, "expected 16 kHz, got {sr}");
                let n = s.len().min(160_000);
                let mut buf = vec![0.0f32; 160_000];
                buf[..n].copy_from_slice(&s[..n]);
                samples = Some(buf);
                eprintln!("real-audio window from {w} (padded to 10s if short)");
                break;
            }
            Err(e) => eprintln!("read_wav {w}: {e}"),
        }
    }
    let Some(wave) = samples else {
        eprintln!("skip real-audio powerset: no fuzfh.wav");
        return;
    };
    let input = InferenceTensor::f32(vec![1, 1, 160_000], wave);
    let mut ort = try_open(&path, InferenceBackend::Ort).expect("ort");
    let mut tract = try_open(&path, InferenceBackend::Tract).expect("tract");
    let o = ort.run_ordered(&[&input]).expect("ort")[0].clone();
    let t = tract.run_ordered(&[&input]).expect("tract")[0].clone();
    let os = o.as_f32_slice().unwrap();
    let ts = t.as_f32_slice().unwrap();
    let (max_abs, max_rel) = max_abs_rel(os, ts);
    let n_classes = 7usize;
    let n_frames = os.len() / n_classes;
    let mut argmax_agree = 0usize;
    let mut hist_o = [0u32; 7];
    let mut hist_t = [0u32; 7];
    for f in 0..n_frames {
        let base = f * n_classes;
        let o_arg = (0..n_classes)
            .max_by(|&a, &b| os[base + a].partial_cmp(&os[base + b]).unwrap())
            .unwrap();
        let t_arg = (0..n_classes)
            .max_by(|&a, &b| ts[base + a].partial_cmp(&ts[base + b]).unwrap())
            .unwrap();
        hist_o[o_arg] += 1;
        hist_t[t_arg] += 1;
        if o_arg == t_arg {
            argmax_agree += 1;
        }
    }
    let agree_pct = 100.0 * argmax_agree as f64 / n_frames.max(1) as f64;
    eprintln!(
        "powerset 10s REAL ort/tract: max_abs={max_abs:.6e} max_rel={max_rel:.6e} \
         argmax_agree={argmax_agree}/{n_frames} ({agree_pct:.1}%) \
         hist_ort={hist_o:?} hist_tract={hist_t:?}"
    );
}

/// ResNet34 FP32 embedding cosine: ort vs tract on identical fbank-like input.
#[test]
#[cfg_attr(miri, ignore)]
fn resnet34_fp32_tract_cosine_report() {
    let Some(path) = model_path("wespeaker_resnet34.onnx") else {
        eprintln!("skip resnet cosine: missing wespeaker_resnet34.onnx");
        return;
    };
    let time = 200usize;
    let n_mels = 80usize;
    let mut feats = vec![0.0f32; time * n_mels];
    for (i, v) in feats.iter_mut().enumerate() {
        *v = ((i % 97) as f32 * 0.01).sin() * 0.5;
    }
    let input = InferenceTensor::f32(vec![1, time, n_mels], feats);
    let mut ort = try_open(&path, InferenceBackend::Ort).expect("ort");
    let mut tract = try_open(&path, InferenceBackend::Tract).expect("tract");
    let o = ort.run_ordered(&[&input]).expect("ort")[0]
        .as_f32_slice()
        .unwrap()
        .to_vec();
    let t = tract.run_ordered(&[&input]).expect("tract")[0]
        .as_f32_slice()
        .unwrap()
        .to_vec();
    let (max_abs, max_rel) = max_abs_rel(&o, &t);
    let mut dot = 0.0f64;
    let mut no = 0.0f64;
    let mut nt = 0.0f64;
    for (&a, &b) in o.iter().zip(t.iter()) {
        dot += f64::from(a) * f64::from(b);
        no += f64::from(a) * f64::from(a);
        nt += f64::from(b) * f64::from(b);
    }
    let cos = dot / (no.sqrt() * nt.sqrt()).max(1e-12);
    eprintln!(
        "resnet34_fp32 ort/tract: dim={} max_abs={max_abs:.6e} max_rel={max_rel:.6e} cosine={cos:.8}",
        o.len()
    );
    assert!(
        cos > 0.99,
        "FP32 ResNet ort/tract cosine should be high, got {cos}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn tract_rejects_garbage_before_parse() {
    use std::io::Write;
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(&[0xAB; 64]).unwrap();
    InferenceBackend::force(Some(InferenceBackend::Tract));
    let err = RuntimeSession::from_path(tmp.path(), ExecutionProvider::Cpu, None)
        .expect_err("garbage must fail header validation");
    InferenceBackend::force(None);
    assert!(
        err.to_string().contains("ONNX header validation failed"),
        "unexpected: {err}"
    );
}

#[test]
fn backend_resolve_force_override() {
    InferenceBackend::force(Some(InferenceBackend::Tract));
    assert_eq!(InferenceBackend::resolve(), InferenceBackend::Tract);
    InferenceBackend::force(Some(InferenceBackend::Ort));
    assert_eq!(InferenceBackend::resolve(), InferenceBackend::Ort);
    InferenceBackend::force(None);
}

#[test]
fn model_path_returns_none_for_missing_blob() {
    assert!(model_path("no_such_model_blob.onnx").is_none());
}

#[test]
fn max_abs_rel_reports_worst_case() {
    // Dominant absolute error on the second element; relative error uses the
    // max(|a|, |b|) scale with a 1e-8 floor so near-zero pairs stay finite.
    let a = [1.0f32, 2.0, 0.0];
    let b = [1.0f32, 2.5, 0.0];
    let (max_abs, max_rel) = max_abs_rel(&a, &b);
    assert!((max_abs - 0.5).abs() < 1e-6, "max_abs={max_abs}");
    assert!((max_rel - 0.2).abs() < 1e-6, "max_rel={max_rel}");
}

#[test]
fn max_abs_rel_handles_all_zero_pairs() {
    let (max_abs, max_rel) = max_abs_rel(&[0.0f32, 0.0], &[0.0f32, 0.0]);
    assert_eq!(max_abs, 0.0);
    assert_eq!(max_rel, 0.0);
}

#[test]
#[should_panic(expected = "length mismatch 2 vs 3")]
fn max_abs_rel_panics_on_length_mismatch() {
    max_abs_rel(&[1.0f32, 2.0], &[1.0f32, 2.0, 3.0]);
}

#[test]
fn assert_f32_close_accepts_within_tolerance() {
    let a = InferenceTensor::f32(vec![2], vec![1.0, 100.0]);
    let b = InferenceTensor::f32(vec![2], vec![1.0 + 5e-4, 100.5]);
    assert_f32_close("within-tol", &a, &b, Tol::DEFAULT);
}

#[test]
fn assert_f32_close_accepts_looser_tol() {
    let a = InferenceTensor::f32(vec![1], vec![1.0]);
    let b = InferenceTensor::f32(vec![1], vec![1.05]);
    assert_f32_close("loose", &a, &b, Tol { abs: 0.1, rel: 0.1 });
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn assert_f32_close_rejects_shape_mismatch() {
    let a = InferenceTensor::f32(vec![1, 2], vec![1.0, 2.0]);
    let b = InferenceTensor::f32(vec![2], vec![1.0, 2.0]);
    assert_f32_close("shapes", &a, &b, Tol::DEFAULT);
}

#[test]
#[should_panic(expected = "max_abs=")]
fn assert_f32_close_rejects_large_error() {
    let a = InferenceTensor::f32(vec![1], vec![1.0]);
    let b = InferenceTensor::f32(vec![1], vec![2.0]);
    assert_f32_close("values", &a, &b, Tol::DEFAULT);
}

#[test]
#[cfg_attr(miri, ignore)]
fn compare_ordered_reports_ort_run_error() {
    let Some(path) = model_path("cam_pp_fp32.onnx") else {
        eprintln!("skip: models/cam_pp_fp32.onnx missing");
        return;
    };
    // Shape product (6) disagrees with the data length (5): the ort run must
    // fail before the tract side is even attempted.
    let bad = InferenceTensor::f32(vec![1, 2, 3], vec![0.0f32; 5]);
    let err = compare_ordered("cam_pp_bad_input", &path, &[bad], Tol::DEFAULT)
        .expect_err("malformed input must fail");
    assert!(err.contains("ort run:"), "unexpected: {err}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn compare_ordered_reports_tract_load_error() {
    // Silero is a documented tract load failure: compare_ordered must surface
    // the session-build error instead of panicking.
    let Some(path) = model_path("silero_vad.onnx") else {
        eprintln!("skip: models/silero_vad.onnx missing");
        return;
    };
    let input = InferenceTensor::f32(vec![1, 576], vec![0.0f32; 576]);
    match compare_ordered("silero", &path, &[input], Tol::DEFAULT) {
        Err(e) => eprintln!("compare_ordered silero: documented failure: {e}"),
        Ok(_) => {
            // A future tract version may load Silero; the run itself must then
            // have failed on the incomplete input set — unreachable in practice.
            eprintln!("compare_ordered silero: unexpectedly succeeded");
        }
    }
}
