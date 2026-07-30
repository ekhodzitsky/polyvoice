//! Full-pipeline CoreML coverage smoke: every ONNX session in the diarization
//! paths (powerset segmentation, fbank embedding, Silero VAD) must accept
//! `ExecutionProvider::CoreMl` and complete end-to-end. CoreML partitions ops
//! it supports and falls back to CPU for the rest, so "accepts the EP and
//! produces a result" is the correct assertion — not "runs 100% on ANE".

#![cfg(all(
    feature = "onnx",
    feature = "download",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
    feature = "coreml",
    target_os = "macos",
    target_arch = "aarch64"
))]

use polyvoice::SileroVad;
use polyvoice::models::ModelRegistry;
use polyvoice::onnx::ExecutionProvider;
use polyvoice::pipeline_v2::Pipeline;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::vad::VoiceActivityDetector;
use polyvoice::wav::read_wav;

#[test]
#[ignore = "requires downloaded models"]
fn coreml_full_pipeline_smoke() {
    let registry = ModelRegistry::default().expect("registry");

    // v2 pipeline (powerset segmentation + fbank embedding) with CoreML.
    let pipeline = Pipeline::builder()
        .profile(Profile::Balanced)
        .execution_provider(ExecutionProvider::CoreMl)
        .with_models_from(registry.clone())
        .build()
        .expect("build v2 pipeline with CoreMl");

    let (samples, sr_hz) =
        read_wav(std::path::Path::new("tests/data/e2e-smoke/audio/fuzfh.wav")).expect("read wav");
    let sr = SampleRate::new(sr_hz).expect("sample rate");
    let result = pipeline.run(&samples, sr).expect("v2 run with CoreMl");
    assert!(
        result.num_speakers >= 1,
        "CoreML v2 run produced no speakers"
    );

    // Silero VAD session (legacy path) with CoreML.
    let vad_path = registry.ensure("silero_vad").expect("silero_vad model");
    let mut vad =
        SileroVad::with_ep(&vad_path, 512, ExecutionProvider::CoreMl).expect("vad with CoreMl");
    // process() requires a multiple of the 512-sample chunk size.
    let probs = vad.process(&samples[..512 * 8]).expect("vad process");
    assert_eq!(probs.len(), 8, "one probability per 512-sample chunk");
    assert!(
        probs.iter().all(|p| (0.0..=1.0).contains(p)),
        "VAD probabilities out of range: {probs:?}"
    );
}
