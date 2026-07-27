//! Ort-free library path: BYO embedder + VBx PLDA from a local directory.
//!
//! ```bash
//! cargo test --no-default-features --features clusterer,vbx --test library_vbx_from_dir
//! ```

#![cfg(feature = "vbx")]

use polyvoice::{
    ClusterConfig, DiarizationConfig, Embedder, EmbedderError, EnergyVad, Pipeline, VadConfig,
    types::WindowConfig,
};

struct AxisEmbedder {
    dim: usize,
    flip: std::sync::atomic::AtomicUsize,
}

impl AxisEmbedder {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            flip: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Embedder for AxisEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        if audio.is_empty() {
            return Err(EmbedderError::AudioTooShort {
                actual_secs: 0.0,
                min_secs: 0.01,
            });
        }
        let n = self.flip.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut v = vec![0.0f32; self.dim];
        if n.is_multiple_of(2) {
            v[0] = 1.0;
        } else {
            v[1] = 1.0;
        }
        Ok(v)
    }
}

fn speech_pcm(secs: f32, sr: u32) -> Vec<f32> {
    let n = (secs * sr as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
        })
        .collect()
}

#[test]
fn offline_vbx_from_checked_in_fixtures() {
    let plda = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/vbx-plda");
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
    let result = Pipeline::new(config, vad_config)
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
