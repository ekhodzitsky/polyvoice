use super::*;
use std::error::Error as _;
use std::path::{Path, PathBuf};

fn local_model(name: &str) -> Option<PathBuf> {
    let p = Path::new("models").join(name);
    if p.is_file() { Some(p) } else { None }
}

/// 1 second of synthetic 16 kHz mono audio (220 Hz tone).
fn synthetic_audio_1s() -> Vec<f32> {
    use std::f32::consts::PI;
    let sr = 16_000_usize;
    (0..sr)
        .map(|i| (2.0 * PI * 220.0 * (i as f32 / sr as f32)).sin() * 0.3)
        .collect()
}

/// `expect_err` without requiring `Debug` on the adapter types.
fn unwrap_err<T>(r: Result<T, EmbedderError>) -> EmbedderError {
    match r {
        Err(e) => e,
        Ok(_) => panic!("expected Err"),
    }
}

struct FailingEmbedder;

impl Embedder for FailingEmbedder {
    fn dim(&self) -> usize {
        4
    }
    fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        Err(EmbedderError::Legacy("synthetic failure".into()))
    }
}

#[test]
fn resnet34_missing_model_reports_session_build_with_path() {
    let path = Path::new("models/definitely-not-a-real-model.onnx");
    let err = unwrap_err(ResNet34Adapter::new(
        path,
        1,
        crate::onnx::ExecutionProvider::Cpu,
    ));
    match &err {
        EmbedderError::SessionBuild { path: p, .. } => {
            assert_eq!(p, path);
        }
        other => panic!("expected SessionBuild, got {other}"),
    }
    let msg = format!("{err}");
    assert!(msg.contains("definitely-not-a-real-model.onnx"));
    assert!(
        err.source().is_some(),
        "typed cause is preserved as the error source"
    );
}

#[test]
fn cam_pp_zero_pool_size_fails_construction() {
    let err = unwrap_err(CamPlusPlusExtractor::new(
        "models/cam_pp_fp32.onnx",
        512,
        0,
        crate::onnx::ExecutionProvider::Cpu,
    ));
    assert!(
        matches!(err, EmbedderError::SessionBuild { .. }),
        "pool-size validation maps to SessionBuild, got {err}"
    );
}

#[test]
fn eres2netv2_dim_constant_is_192() {
    assert_eq!(ERes2NetV2Extractor::DIM, 192);
}

#[test]
fn eres2netv2_missing_model_reports_session_build() {
    let path = Path::new("models/definitely-not-eres2netv2.onnx");
    let err = unwrap_err(ERes2NetV2Extractor::new(
        path,
        1,
        crate::onnx::ExecutionProvider::Cpu,
    ));
    assert!(
        matches!(err, EmbedderError::SessionBuild { .. }),
        "got {err}"
    );

    let err = unwrap_err(ERes2NetV2Extractor::with_dim(
        path,
        256,
        1,
        crate::onnx::ExecutionProvider::Cpu,
    ));
    assert!(
        matches!(err, EmbedderError::SessionBuild { .. }),
        "got {err}"
    );
}

#[test]
fn resnet34_real_model_embeds_256d_unit_vector() {
    let Some(path) = local_model("wespeaker_resnet34.onnx") else {
        eprintln!("skip resnet34_real_model: models/wespeaker_resnet34.onnx missing");
        return;
    };
    let extractor = ResNet34Adapter::new(&path, 1, crate::onnx::ExecutionProvider::Cpu)
        .expect("local model loads");
    assert_eq!(extractor.dim(), 256);

    let embedding = extractor.embed(&synthetic_audio_1s()).expect("embed");
    assert_eq!(embedding.len(), 256);
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-2, "L2 norm not 1.0: {norm}");
}

#[test]
fn cam_pp_real_model_embeds_and_batches_512d() {
    let Some(path) = local_model("cam_pp_fp32.onnx") else {
        eprintln!("skip cam_pp_real_model: models/cam_pp_fp32.onnx missing");
        return;
    };
    let extractor = CamPlusPlusExtractor::new(&path, 512, 2, crate::onnx::ExecutionProvider::Cpu)
        .expect("local model loads");
    assert_eq!(extractor.dim(), 512);

    let audio = synthetic_audio_1s();
    let embedding = extractor.embed(&audio).expect("embed");
    assert_eq!(embedding.len(), 512);

    // Batches fan out across the session pool via parallel_embed_batch.
    let batch = extractor
        .embed_batch(&[&audio, &audio, &audio])
        .expect("batch");
    assert_eq!(batch.len(), 3);
    for v in &batch {
        assert_eq!(v.len(), 512);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-2, "L2 norm not 1.0: {norm}");
    }
    // Deterministic model: same input yields the same embedding.
    assert_eq!(batch[0], batch[1]);
}

#[test]
fn parallel_embed_batch_empty_input_returns_empty() {
    let e = DummyExtractor::new(8);
    let out = parallel_embed_batch(&e, &[], 4).unwrap();
    assert!(out.is_empty());
}

#[test]
fn parallel_embed_batch_collects_all_results() {
    let e = DummyExtractor::new(8);
    let audio = synthetic_audio_1s();
    let inputs: Vec<&[f32]> = (0..16).map(|_| &audio[..]).collect();
    let out = parallel_embed_batch(&e, &inputs, 4).unwrap();
    assert_eq!(out.len(), 16);
    for v in &out {
        assert_eq!(v.len(), 8);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "L2 norm not 1.0: {norm}");
    }
}

#[test]
fn parallel_embed_batch_zero_max_threads_still_runs() {
    let e = DummyExtractor::new(8);
    let audio = synthetic_audio_1s();
    let out = parallel_embed_batch(&e, &[&audio, &audio], 0).unwrap();
    assert_eq!(out.len(), 2);
}

#[test]
fn parallel_embed_batch_propagates_inner_error() {
    let e = FailingEmbedder;
    let audio = synthetic_audio_1s();
    let err =
        parallel_embed_batch(&e, &[&audio, &audio], 2).expect_err("inner failure must propagate");
    assert!(
        matches!(err, EmbedderError::Legacy(ref d) if d == "synthetic failure"),
        "got {err}"
    );
}
