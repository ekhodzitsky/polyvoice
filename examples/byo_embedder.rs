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
};

/// Mock embedder: odd-energy windows → speaker A axis, even → speaker B.
///
/// Vectors are L2 unit so cosine AHC can separate two clusters without a model.
struct OrthogonalEmbedder {
    dim: usize,
    call: std::sync::atomic::AtomicUsize,
}

impl OrthogonalEmbedder {
    fn new(dim: usize) -> Self {
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
        // Alternate axes so AHC finds two speakers on synthetic two-tone audio.
        if n % 2 == 0 {
            v[0] = 1.0;
        } else if self.dim > 1 {
            v[1] = 1.0;
        } else {
            v[0] = -1.0;
        }
        Ok(v)
    }
}

/// Build ~4 s of alternating loud / quieter frames so Energy VAD keeps speech.
fn synthetic_two_speaker_pcm(sr: u32) -> Vec<f32> {
    let n = (sr as usize) * 4;
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        // Two amplitude bands: still above a −40 dB energy floor after framing.
        let band = if (i / (sr as usize / 2)) % 2 == 0 {
            0.3
        } else {
            0.25
        };
        let t = i as f32 / sr as f32;
        let f = if (i / (sr as usize)) % 2 == 0 {
            220.0
        } else {
            440.0
        };
        samples.push(band * (2.0 * std::f32::consts::PI * f * t).sin());
    }
    samples
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 16_000u32;
    let samples = synthetic_two_speaker_pcm(sr);
    let extractor = OrthogonalEmbedder::new(256);

    let mut config = DiarizationConfig::default();
    config.cluster = ClusterConfig {
        threshold: 0.45,
        ..ClusterConfig::default()
    };
    // Shorter windows so the mock alternates embeddings more often.
    config.window.window_secs = 0.5;
    config.window.hop_secs = 0.25;

    let vad_config = VadConfig::default();
    let mut vad = EnergyVad::new(-40.0, sr, vad_config.frame_size);

    // --- Offline ---
    let offline = Pipeline::new(config.clone(), vad_config.clone());
    let result = offline.run(&samples, &extractor, &mut vad)?;
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

    // --- Streaming (same Embedder; LatencyPreset::Realtime for live STT) ---
    let stream_vad = EnergyVad::new(-40.0, sr, vad_config.frame_size);
    let mut stream = StreamingPipeline::with_latency_preset(
        stream_vad,
        OrthogonalEmbedder::new(256),
        LatencyPreset::Realtime,
        vad_config,
    )?;
    const CHUNK: usize = 1600; // 100 ms @ 16 kHz
    for chunk in samples.chunks(CHUNK) {
        let _ = stream.feed(chunk)?;
    }
    let _ = stream.flush()?;
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

    // Product STT stacks map turns → word speaker labels after ASR.
    // Prefer midpoint coverage; streaming tails often use last-turn fallback
    // (see docs/library-mode.md).
    Ok(())
}
