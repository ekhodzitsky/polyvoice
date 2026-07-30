//! Ort-free library path: BYO embedder + VBx PLDA from a local directory.
//!
//! ```bash
//! cargo test --no-default-features --features clusterer,vbx --test library_vbx_from_dir
//! ```

#![cfg(feature = "vbx")]

mod common;

use common::{AxisEmbedder, speech_pcm};
use polyvoice::{
    ClusterConfig, DiarizationConfig, EnergyVad, VadConfig, pipeline::LegacyPipeline,
    types::WindowConfig,
};

#[test]
fn offline_vbx_from_checked_in_fixtures() {
    let plda = common::vbx_plda_fixture_dir();
    assert!(
        plda.join("plda_transform.npy").is_file(),
        "fixtures/vbx-plda must be present"
    );

    let sr = 16_000u32;
    let samples = speech_pcm(8.0, sr);
    let config = DiarizationConfig {
        cluster: ClusterConfig {
            min_cluster_size: 1,
            min_cluster_secs: 0.0,
            ..ClusterConfig::default()
        },
        window: WindowConfig {
            window_secs: 1.5,
            hop_secs: 0.75,
            ..Default::default()
        },
        ..Default::default()
    };
    let vad_config = VadConfig::default();
    let mut vad = EnergyVad::new(-40.0, sr, vad_config.frame_size);
    // PLDA fixtures are trained for 256-d WeSpeaker-like space.
    let embedder = AxisEmbedder::new(256);
    let result = LegacyPipeline::new(config, vad_config)
        .run_with_vbx_from_dir(&samples, &embedder, &mut vad, &plda, 8)
        .expect("VBx library path must not require onnx or network");
    // Structural success only — mock embeddings are not DER-quality.
    assert_eq!(
        result.turns.len(),
        result
            .segments
            .iter()
            .filter(|s| s.speaker.is_some())
            .count()
            .max(result.turns.len())
    );
}
