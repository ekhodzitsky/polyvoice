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
    let p = Path::new("models").join(name);
    if p.is_file() { Some(p) } else { None }
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
        "cam_pp_fp32.onnx",
        "wespeaker_resnet34.onnx",
        "ecapa_tdnn_mel.onnx",
    ];
    for name in models {
        let Some(path) = model_path(name) else {
            eprintln!("load-status {name}: MISSING (skip)");
            continue;
        };
        match try_open(&path, InferenceBackend::Tract) {
            Ok(s) => eprintln!(
                "load-status {name}: OK backend={:?} inputs={:?}",
                s.backend(),
                s.input_names()
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
    let Some(path) = model_path("powerset_fp32.onnx") else {
        eprintln!("skip powerset status: models/powerset_fp32.onnx missing");
        return;
    };
    match try_open(&path, InferenceBackend::Tract) {
        Ok(_) => eprintln!("powerset_fp32: tract LOAD OK (unexpected win — update verdict)"),
        Err(e) => {
            eprintln!("powerset_fp32: tract LOAD FAIL (documented): {e}");
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
