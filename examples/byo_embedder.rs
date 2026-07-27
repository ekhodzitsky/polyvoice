//! Bring-your-own embedder (library mode) — no ONNX, no network.
//!
//! Pattern used by product STT stacks: the consumer owns speaker embeddings
//! (Candle WeSpeaker, tract, custom); polyvoice owns Energy VAD, offline
//! `Pipeline`, streaming `StreamingPipeline`, and clustering.
//!
//! ```bash
//! cargo run --no-default-features --example byo_embedder
//! ```
//!
//! Replace [`OrthogonalEmbedder`] with your real encoder implementing
//! [`polyvoice::Embedder`].

use polyvoice::{
    ClusterConfig, DiarizationConfig, Embedder, EmbedderError, EnergyVad, Pipeline, VadConfig,
    streaming::{LatencyPreset, StreamingPipeline},
    types::WindowConfig,
};

/// Mock embedder: alternate unit axes so cosine clustering yields two speakers.
///
/// Vectors are L2 unit (`[1,0,…]` / `[0,1,…]`) without a real model.
struct OrthogonalEmbedder {
    dim: usize,
    call: std::sync::atomic::AtomicUsize,
}

impl OrthogonalEmbedder {
    fn new(dim: usize) -> Self {
        assert!(dim >= 2, "need ≥2 dims for two orthogonal axes");
        Self {
            dim,
            call: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Embedder for OrthogonalEmbedder {
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
        let n = self.call.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut v = vec![0.0f32; self.dim];
        if n.is_multiple_of(2) {
            v[0] = 1.0;
        } else {
            v[1] = 1.0;
        }
        Ok(v)
    }
}

/// ~4 s of speech-level energy so Energy VAD keeps frames (not silence).
fn synthetic_speech_pcm(sr: u32, secs: u32) -> Vec<f32> {
    let n = (sr as usize) * (secs as usize);
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr as f32;
        // Mild amplitude / pitch change every half-second — still above −40 dB.
        let band = if (i / (sr as usize / 2)).is_multiple_of(2) {
            0.3
        } else {
            0.25
        };
        let f = if (i / sr as usize).is_multiple_of(2) {
            220.0
        } else {
            440.0
        };
        samples.push(band * (2.0 * std::f32::consts::PI * f * t).sin());
    }
    samples
}

fn short_window_config() -> DiarizationConfig {
    // Short windows so the mock alternates embeddings often enough for 2 clusters.
    DiarizationConfig {
        cluster: ClusterConfig {
            threshold: 0.45,
            ..ClusterConfig::default()
        },
        window: WindowConfig {
            window_secs: 0.5,
            hop_secs: 0.25,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 16_000u32;
    let samples = synthetic_speech_pcm(sr, 4);
    let config = short_window_config();
    let vad_config = VadConfig::default();

    // --- Offline ---
    let mut vad = EnergyVad::new(-40.0, sr, vad_config.frame_size);
    let result = Pipeline::new(config, vad_config).run(
        &samples,
        &OrthogonalEmbedder::new(256),
        &mut vad,
    )?;
    println!(
        "offline: {} speakers, {} turns",
        result.num_speakers,
        result.turns.len()
    );
    for turn in &result.turns {
        println!(
            "  speaker_{}: {:.2}s - {:.2}s (stable={})",
            turn.speaker.0, turn.time.start, turn.time.end, turn.stable
        );
    }

    // --- Streaming (LatencyPreset::Realtime for live STT) ---
    let mut stream = StreamingPipeline::with_latency_preset(
        EnergyVad::new(-40.0, sr, vad_config.frame_size),
        OrthogonalEmbedder::new(256),
        LatencyPreset::Realtime,
        vad_config,
    )?;
    const CHUNK: usize = 1600; // 100 ms @ 16 kHz
    for chunk in samples.chunks(CHUNK) {
        stream.feed(chunk)?;
    }
    stream.flush()?;
    println!(
        "streaming: {} speakers, {} turns (preset={:?})",
        stream.num_speakers(),
        stream.turns().len(),
        stream.latency_preset()
    );
    for turn in stream.turns() {
        println!(
            "  speaker_{}: {:.2}s - {:.2}s (stable={})",
            turn.speaker.0, turn.time.start, turn.time.end, turn.stable
        );
    }

    // After ASR, map word midpoints onto turns (offline: leave uncovered unset;
    // streaming: last-turn fallback). See docs/library-mode.md.
    Ok(())
}
