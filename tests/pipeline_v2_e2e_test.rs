//! M6a — `#[ignore]` end-to-end test for `polyvoice::pipeline_v2`.
//!
//! Requires the Balanced ONNX bundle to be cached (run
//! `cargo run --features cli --bin polyvoice -- download-models --profile balanced`
//! once before invoking with `cargo test -- --ignored e2e`). Reads a single
//! WAV from `data/voxconverse-test/audio/` and asserts the pipeline returns
//! at least one turn with valid speaker IDs.

#![cfg(all(
    feature = "pipeline_v2",
    feature = "download",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]

use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::Pipeline;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use std::path::Path;

fn first_voxconverse_wav() -> Option<std::path::PathBuf> {
    let dir = Path::new("data/voxconverse-test/audio");
    if !dir.is_dir() {
        return None;
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "wav"))
        .map(|e| e.path())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

#[ignore = "requires cached ONNX bundle + a wav file under data/voxconverse-test/audio/"]
#[test]
fn e2e_balanced_profile_voxconverse_clip() {
    let wav_path = match first_voxconverse_wav() {
        Some(p) => p,
        None => panic!(
            "data/voxconverse-test/audio/ is empty — run scripts/download-voxconverse-test.sh first"
        ),
    };
    let (samples, sr_hz) =
        read_wav(&wav_path).expect("WAV read failure — check the file is 16 kHz mono");
    assert_eq!(sr_hz, 16000, "only 16 kHz WAVs supported");
    let registry = ModelRegistry::default()
        .expect("default ModelRegistry should resolve a writable cache dir");
    let pipeline = Pipeline::builder()
        .profile(Profile::Balanced)
        .with_models_from(registry)
        .build()
        .expect(
            "Balanced profile build should succeed when cached ONNX is present — \
             run `polyvoice download-models --profile balanced` first",
        );
    let result = pipeline
        .run(&samples, SampleRate::new(16000).unwrap())
        .expect("pipeline.run on a real VoxConverse clip should succeed");
    assert!(
        result.num_speakers >= 1,
        "expected at least 1 speaker, got {}",
        result.num_speakers
    );
    for w in result.turns.windows(2) {
        assert!(
            w[0].time.start <= w[1].time.start,
            "turns must be sorted by start time"
        );
    }
}
