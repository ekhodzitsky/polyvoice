# Phase 1: Silent Quality Push — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end speaker diarization pipeline on real audio with WeSpeaker embeddings, Silero VAD, and agglomerative clustering — targeting DER ≤ pyannote 3.x + 3% on AMI test set.

**Architecture:** A new `Pipeline` struct wires together: (1) Silero VAD for speech segmentation, (2) WeSpeaker ONNX for speaker embeddings (reusing existing `EcapaTdnnExtractor` + adding CMVN), (3) AHC for offline re-clustering. WAV I/O via `hound` crate. The existing `OfflineDiarizer` stays untouched — `Pipeline` composes on top of the existing building blocks.

**Tech Stack:** Rust, `ort` 2.0.0-rc.12 (existing), `hound` (new), `crossbeam-queue` (existing), WeSpeaker ResNet34 ONNX, Silero VAD v5 ONNX.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/wav.rs` | Read/write mono f32 PCM from WAV files via `hound` |
| Create | `src/ahc.rs` | Agglomerative hierarchical clustering |
| Create | `src/silero_vad.rs` | Silero VAD ONNX integration, implements `VoiceActivityDetector` |
| Create | `src/pipeline.rs` | High-level `Pipeline` that wires VAD + extractor + AHC |
| Modify | `src/features.rs` | Add CMVN (cepstral mean-variance normalization) for WeSpeaker |
| Modify | `src/lib.rs` | Register new modules and re-exports |
| Modify | `Cargo.toml` | Add `hound` dependency |
| Create | `tests/test_wav.rs` | WAV I/O integration tests |
| Create | `tests/test_ahc.rs` | AHC integration tests |
| Create | `tests/test_pipeline.rs` | Pipeline integration tests (with DummyExtractor) |
| Create | `scripts/download-models.sh` | Download WeSpeaker + Silero VAD ONNX models |
| Create | `benches/der_ami.rs` | DER benchmark on AMI test set |

---

### Task 1: WAV I/O Module

**Files:**
- Modify: `Cargo.toml`
- Create: `src/wav.rs`
- Modify: `src/lib.rs`
- Create: `tests/test_wav.rs`

- [ ] **Step 1: Add `hound` dependency**

In `Cargo.toml`, add under `[dependencies]`:

```toml
hound = "3.5"
```

Run: `cargo check`
Expected: compiles with no errors.

- [ ] **Step 2: Write failing test for WAV reading**

Create `tests/test_wav.rs`:

```rust
use polyvoice::wav;
use std::io::Cursor;

#[test]
fn test_read_wav_mono_16bit() {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Vec::new();
    {
        let cursor = Cursor::new(&mut buf);
        let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
        for i in 0..16000 {
            let sample = ((i as f32 / 16000.0) * std::f32::consts::TAU * 440.0).sin();
            writer.write_sample((sample * 32767.0) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &buf).unwrap();

    let (samples, sample_rate) = wav::read_wav(tmp.path()).unwrap();
    assert_eq!(sample_rate, 16000);
    assert_eq!(samples.len(), 16000);
    assert!(samples.iter().all(|s| *s >= -1.0 && *s <= 1.0));
}

#[test]
fn test_read_wav_stereo_downmix() {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Vec::new();
    {
        let cursor = Cursor::new(&mut buf);
        let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
        for _ in 0..8000 {
            writer.write_sample(16383i16).unwrap(); // left
            writer.write_sample(-16383i16).unwrap(); // right
        }
        writer.finalize().unwrap();
    }

    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &buf).unwrap();

    let (samples, sample_rate) = wav::read_wav(tmp.path()).unwrap();
    assert_eq!(sample_rate, 16000);
    assert_eq!(samples.len(), 8000);
    // Stereo average of +0.5 and -0.5 ≈ 0.0
    assert!(samples.iter().all(|s| s.abs() < 0.01));
}

#[test]
fn test_read_wav_not_found() {
    let result = wav::read_wav(std::path::Path::new("/nonexistent/audio.wav"));
    assert!(result.is_err());
}
```

Run: `cargo test --test test_wav`
Expected: FAIL — `wav` module doesn't exist yet.

- [ ] **Step 3: Implement `wav` module**

Create `src/wav.rs`:

```rust
//! WAV file I/O via the `hound` crate.

use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum WavError {
    #[error("failed to read WAV: {0}")]
    Read(#[from] hound::Error),
    #[error("unsupported sample format: {0}")]
    UnsupportedFormat(String),
}

/// Read a WAV file and return mono f32 samples normalized to [-1.0, 1.0] and its sample rate.
///
/// Stereo files are downmixed by averaging channels. 16-bit and 32-bit float
/// formats are supported.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), WavError> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<f32>, _>>()?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()?,
    };

    let mono = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok((mono, sample_rate))
}
```

- [ ] **Step 4: Register module in lib.rs**

In `src/lib.rs`, add after the `pub mod vad;` line:

```rust
pub mod wav;
```

- [ ] **Step 5: Run tests**

Run: `cargo test --test test_wav`
Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/wav.rs src/lib.rs tests/test_wav.rs
git commit -m "feat: add WAV I/O module via hound crate"
```

---

### Task 2: Agglomerative Hierarchical Clustering

**Files:**
- Create: `src/ahc.rs`
- Modify: `src/lib.rs`
- Create: `tests/test_ahc.rs`

- [ ] **Step 1: Write failing tests for AHC**

Create `tests/test_ahc.rs`:

```rust
use polyvoice::ahc::agglomerative_cluster;
use polyvoice::utils::l2_normalize;

fn unit_vec(dim: usize, axis: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[axis] = 1.0;
    v
}

fn noisy_vec(dim: usize, axis: usize, noise: f32) -> Vec<f32> {
    let mut v = unit_vec(dim, axis);
    v[(axis + 1) % dim] = noise;
    l2_normalize(&mut v);
    v
}

#[test]
fn test_ahc_two_speakers() {
    let embeddings = vec![
        unit_vec(256, 0), // speaker A
        unit_vec(256, 1), // speaker B
        noisy_vec(256, 0, 0.05), // speaker A (noisy)
        noisy_vec(256, 1, 0.05), // speaker B (noisy)
    ];

    let labels = agglomerative_cluster(&embeddings, 0.5);
    assert_eq!(labels.len(), 4);
    assert_eq!(labels[0], labels[2]); // A grouped together
    assert_eq!(labels[1], labels[3]); // B grouped together
    assert_ne!(labels[0], labels[1]); // A != B
}

#[test]
fn test_ahc_single_speaker() {
    let mut embeddings = Vec::new();
    for i in 0..5 {
        embeddings.push(noisy_vec(256, 0, 0.01 * i as f32));
    }

    let labels = agglomerative_cluster(&embeddings, 0.5);
    assert!(labels.iter().all(|&l| l == labels[0]));
}

#[test]
fn test_ahc_all_different() {
    let embeddings = vec![
        unit_vec(256, 0),
        unit_vec(256, 1),
        unit_vec(256, 2),
    ];

    // Very high threshold — nothing should merge
    let labels = agglomerative_cluster(&embeddings, 0.99);
    assert_ne!(labels[0], labels[1]);
    assert_ne!(labels[1], labels[2]);
    assert_ne!(labels[0], labels[2]);
}

#[test]
fn test_ahc_empty() {
    let embeddings: Vec<Vec<f32>> = vec![];
    let labels = agglomerative_cluster(&embeddings, 0.5);
    assert!(labels.is_empty());
}

#[test]
fn test_ahc_single_embedding() {
    let embeddings = vec![unit_vec(256, 0)];
    let labels = agglomerative_cluster(&embeddings, 0.5);
    assert_eq!(labels, vec![0]);
}
```

Run: `cargo test --test test_ahc`
Expected: FAIL — `ahc` module doesn't exist.

- [ ] **Step 2: Implement AHC module**

Create `src/ahc.rs`:

```rust
//! Agglomerative Hierarchical Clustering (AHC) for speaker diarization.
//!
//! Bottom-up clustering: each embedding starts as its own cluster, then the
//! two most similar clusters are merged iteratively until no pair exceeds
//! the cosine similarity threshold.

use crate::utils::{cosine_similarity, l2_normalize, mean_vector};

/// Run agglomerative hierarchical clustering on a set of embeddings.
///
/// Returns a label vector of the same length as `embeddings`, where each
/// element is the cluster index (0-based, contiguous) for that embedding.
///
/// `threshold` is the minimum cosine similarity to merge two clusters.
/// Higher threshold → more clusters (stricter merging).
pub fn agglomerative_cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }

    // Each embedding starts in its own cluster.
    let mut labels: Vec<usize> = (0..n).collect();
    let mut centroids: Vec<Vec<f32>> = embeddings.to_vec();
    let mut cluster_sizes: Vec<usize> = vec![1; n];
    let mut active: Vec<bool> = vec![true; n];

    loop {
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;

        // Find the two most similar active clusters.
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                let sim = cosine_similarity(&centroids[i], &centroids[j]);
                if sim > best_sim {
                    best_sim = sim;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_sim < threshold {
            break;
        }

        // Merge j into i.
        let total = cluster_sizes[best_i] + cluster_sizes[best_j];
        let w_i = cluster_sizes[best_i] as f32 / total as f32;
        let w_j = cluster_sizes[best_j] as f32 / total as f32;
        let mut new_centroid = vec![0.0f32; centroids[best_i].len()];
        for (k, v) in new_centroid.iter_mut().enumerate() {
            *v = centroids[best_i][k] * w_i + centroids[best_j][k] * w_j;
        }
        l2_normalize(&mut new_centroid);

        centroids[best_i] = new_centroid;
        cluster_sizes[best_i] = total;
        active[best_j] = false;

        // Relabel all embeddings from cluster j to cluster i.
        for label in &mut labels {
            if *label == best_j {
                *label = best_i;
            }
        }
    }

    // Make labels contiguous (0, 1, 2, ...).
    let mut label_map = std::collections::HashMap::new();
    let mut next_label = 0usize;
    for label in &mut labels {
        let entry = label_map.entry(*label).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        *label = *entry;
    }

    labels
}
```

- [ ] **Step 3: Register module in lib.rs**

In `src/lib.rs`, add after `pub mod cluster;`:

```rust
pub mod ahc;
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test test_ahc`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ahc.rs src/lib.rs tests/test_ahc.rs
git commit -m "feat: add agglomerative hierarchical clustering (AHC)"
```

---

### Task 3: CMVN Normalization for WeSpeaker

**Files:**
- Modify: `src/features.rs`

WeSpeaker models expect sliding-window cepstral mean normalization (CMN) applied to fbank features. This normalizes each mel bin to zero mean over the utterance.

- [ ] **Step 1: Write failing test for CMVN**

Add to the bottom of the `#[cfg(test)] mod tests` block in `src/features.rs`:

```rust
#[test]
fn test_apply_cmvn() {
    let frames = vec![
        vec![1.0, 2.0, 3.0],
        vec![3.0, 4.0, 5.0],
        vec![5.0, 6.0, 7.0],
    ];
    let normalized = apply_cmvn(&frames);
    assert_eq!(normalized.len(), 3);
    // Mean of each bin: [3.0, 4.0, 5.0]
    // Frame 0 after CMN: [-2.0, -2.0, -2.0]
    assert!((normalized[0][0] - (-2.0)).abs() < 1e-5);
    assert!((normalized[1][0] - 0.0).abs() < 1e-5);
    assert!((normalized[2][0] - 2.0).abs() < 1e-5);
}

#[test]
fn test_apply_cmvn_empty() {
    let frames: Vec<Vec<f32>> = vec![];
    let normalized = apply_cmvn(&frames);
    assert!(normalized.is_empty());
}
```

Run: `cargo test --lib features::tests::test_apply_cmvn`
Expected: FAIL — `apply_cmvn` doesn't exist.

- [ ] **Step 2: Implement `apply_cmvn`**

Add this public function in `src/features.rs`, above the `fn mel_filterbank` function:

```rust
/// Apply cepstral mean normalization (CMN) to fbank features.
///
/// Subtracts the per-bin mean across all frames. This is required by WeSpeaker
/// models to normalize channel effects.
pub fn apply_cmvn(frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if frames.is_empty() {
        return Vec::new();
    }
    let n_bins = frames[0].len();
    let n_frames = frames.len() as f32;

    let mut means = vec![0.0f32; n_bins];
    for frame in frames {
        for (i, &v) in frame.iter().enumerate() {
            means[i] += v;
        }
    }
    for m in &mut means {
        *m /= n_frames;
    }

    frames
        .iter()
        .map(|frame| {
            frame
                .iter()
                .zip(means.iter())
                .map(|(&v, &m)| v - m)
                .collect()
        })
        .collect()
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --lib features::tests::test_apply_cmvn`
Expected: PASS.

Run: `cargo test --lib features::tests`
Expected: all features tests pass (no regressions).

- [ ] **Step 4: Commit**

```bash
git add src/features.rs
git commit -m "feat: add CMVN normalization for WeSpeaker compatibility"
```

---

### Task 4: Silero VAD ONNX Integration

**Files:**
- Create: `src/silero_vad.rs`
- Modify: `src/lib.rs`

Silero VAD v5 ONNX model expects:
- Input `input`: `[batch=1, chunk_size]` f32 (chunk_size = 512 at 16kHz)
- Input `sr`: scalar i64 (sample rate, 16000)
- Input `state`: `[2, 1, 128]` f32 (LSTM hidden state, zeros initially)
- Output `output`: `[batch=1, 1]` f32 (speech probability)
- Output `stateN`: `[2, 1, 128]` f32 (updated LSTM state)

- [ ] **Step 1: Write failing test**

Add a unit test at the bottom of the new file (we'll write the struct first, test with mock):

Create `src/silero_vad.rs`:

```rust
//! Silero VAD v5 ONNX integration.
//!
//! Implements `VoiceActivityDetector` using the Silero VAD v5 ONNX model.
//! The model is stateful (LSTM) — hidden state is carried between calls
//! to `process()` and reset via `reset()`.

use crate::vad::{VadError, VoiceActivityDetector};

#[cfg(feature = "onnx")]
pub struct SileroVad {
    session: ort::session::Session,
    state: Vec<f32>,
    sample_rate: u32,
    chunk_size: usize,
}

#[cfg(feature = "onnx")]
impl SileroVad {
    const STATE_SIZE: usize = 2 * 1 * 128;

    /// Load Silero VAD from an ONNX model file.
    ///
    /// `chunk_size` is typically 512 for 16kHz audio (32ms windows).
    pub fn new(model_path: &std::path::Path, chunk_size: usize) -> Result<Self, anyhow::Error> {
        let session = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("load model: {e}"))?;

        Ok(Self {
            session,
            state: vec![0.0f32; Self::STATE_SIZE],
            sample_rate: 16000,
            chunk_size,
        })
    }

    fn run_chunk(&mut self, chunk: &[f32]) -> Result<f32, VadError> {
        let input_array =
            ndarray::Array2::from_shape_vec((1, chunk.len()), chunk.to_vec())
                .map_err(|e| VadError::Model(e.to_string()))?;

        let sr_array = ndarray::Array1::from_vec(vec![self.sample_rate as i64]);

        let state_array =
            ndarray::Array3::from_shape_vec((2, 1, 128), self.state.clone())
                .map_err(|e| VadError::Model(e.to_string()))?;

        let input_tensor = ort::value::TensorRef::from_array_view(&input_array)
            .map_err(|e| VadError::Model(e.to_string()))?;
        let sr_tensor = ort::value::TensorRef::from_array_view(&sr_array)
            .map_err(|e| VadError::Model(e.to_string()))?;
        let state_tensor = ort::value::TensorRef::from_array_view(&state_array)
            .map_err(|e| VadError::Model(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![input_tensor, sr_tensor, state_tensor])
            .map_err(|e| VadError::Model(e.to_string()))?;

        let (_, prob_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::Model(e.to_string()))?;

        let (_, new_state) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::Model(e.to_string()))?;

        self.state = new_state.to_vec();

        Ok(prob_data[0])
    }
}

#[cfg(feature = "onnx")]
impl VoiceActivityDetector for SileroVad {
    fn reset(&mut self) {
        self.state = vec![0.0f32; Self::STATE_SIZE];
    }

    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, VadError> {
        if samples.len() % self.chunk_size != 0 {
            return Err(VadError::InvalidChunkSize {
                expected: self.chunk_size,
                got: samples.len(),
            });
        }

        let mut probs = Vec::with_capacity(samples.len() / self.chunk_size);
        for chunk in samples.chunks(self.chunk_size) {
            let prob = self.run_chunk(chunk)?;
            probs.push(prob);
        }
        Ok(probs)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[cfg(not(feature = "onnx"))]
pub struct SileroVad;

#[cfg(not(feature = "onnx"))]
impl SileroVad {
    pub fn new(
        _model_path: &std::path::Path,
        _chunk_size: usize,
    ) -> Result<Self, anyhow::Error> {
        anyhow::bail!("the `onnx` feature is not enabled")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_silero_vad_stub_without_onnx() {
        // When onnx feature is disabled, constructor returns an error.
        #[cfg(not(feature = "onnx"))]
        {
            let result = super::SileroVad::new(std::path::Path::new("model.onnx"), 512);
            assert!(result.is_err());
        }
    }
}
```

- [ ] **Step 2: Register module in lib.rs**

In `src/lib.rs`, add after `pub mod vad;`:

```rust
pub mod silero_vad;
```

Add to re-exports:

```rust
pub use silero_vad::SileroVad;
```

(The `#[cfg(feature = "onnx")]` guard is inside the module itself.)

- [ ] **Step 3: Run compilation check**

Run: `cargo check --all-features`
Expected: compiles with no errors.

Run: `cargo test --lib silero_vad`
Expected: stub test passes.

- [ ] **Step 4: Commit**

```bash
git add src/silero_vad.rs src/lib.rs
git commit -m "feat: add Silero VAD v5 ONNX integration"
```

---

### Task 5: Pipeline — Wire VAD + Extractor + AHC

**Files:**
- Create: `src/pipeline.rs`
- Modify: `src/lib.rs`
- Create: `tests/test_pipeline.rs`

The `Pipeline` struct is the high-level entry point that composes:
1. VAD → speech regions
2. Embedding extraction per region
3. AHC clustering
4. Post-processing (merge same-speaker gaps, filter short segments)

- [ ] **Step 1: Write failing test for Pipeline**

Create `tests/test_pipeline.rs`:

```rust
use polyvoice::pipeline::Pipeline;
use polyvoice::{DiarizationConfig, DummyExtractor, EnergyVad, VadConfig};

#[test]
fn test_pipeline_basic() {
    let config = DiarizationConfig {
        window_secs: 0.5,
        hop_secs: 0.25,
        min_speech_secs: 0.1,
        ..Default::default()
    };
    let vad_config = VadConfig {
        threshold: 0.1, // low threshold so sine wave triggers speech
        ..Default::default()
    };
    let extractor = DummyExtractor::new(256);
    let vad = EnergyVad::new(-60.0, 16000, 512);

    let pipeline = Pipeline::new(config, vad_config);

    // 5 seconds of "loud" audio
    let samples: Vec<f32> = (0..16000 * 5)
        .map(|i| ((i as f32 / 16000.0) * std::f32::consts::TAU * 440.0).sin() * 0.5)
        .collect();

    let result = pipeline.run(&samples, &extractor, &mut EnergyVad::new(-60.0, 16000, 512)).unwrap();
    assert!(!result.segments.is_empty());
    assert!(!result.turns.is_empty());
    assert!(result.num_speakers >= 1);
}

#[test]
fn test_pipeline_silence() {
    let config = DiarizationConfig::default();
    let vad_config = VadConfig::default();
    let extractor = DummyExtractor::new(256);
    let mut vad = EnergyVad::new(-20.0, 16000, 512);

    let pipeline = Pipeline::new(config, vad_config);

    // Pure silence
    let samples = vec![0.0f32; 16000 * 3];
    let result = pipeline.run(&samples, &extractor, &mut vad).unwrap();
    assert!(result.turns.is_empty());
    assert_eq!(result.num_speakers, 0);
}

#[test]
fn test_pipeline_from_wav() {
    use std::io::Cursor;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Vec::new();
    {
        let cursor = Cursor::new(&mut buf);
        let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
        for i in 0..16000 * 3 {
            let sample = ((i as f32 / 16000.0) * std::f32::consts::TAU * 300.0).sin();
            writer.write_sample((sample * 16000.0) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &buf).unwrap();

    let config = DiarizationConfig {
        window_secs: 0.5,
        hop_secs: 0.25,
        min_speech_secs: 0.1,
        ..Default::default()
    };
    let vad_config = VadConfig {
        threshold: 0.1,
        ..Default::default()
    };
    let extractor = DummyExtractor::new(256);
    let pipeline = Pipeline::new(config, vad_config);
    let mut vad = polyvoice::EnergyVad::new(-60.0, 16000, 512);

    let result = pipeline.run_from_wav(tmp.path(), &extractor, &mut vad).unwrap();
    assert!(!result.turns.is_empty());
}
```

Run: `cargo test --test test_pipeline`
Expected: FAIL — `pipeline` module doesn't exist.

- [ ] **Step 2: Implement Pipeline**

Create `src/pipeline.rs`:

```rust
//! High-level diarization pipeline.
//!
//! Wires together VAD, embedding extraction, and AHC clustering into a
//! single `run()` call that takes audio and returns `DiarizationResult`.

use crate::ahc::agglomerative_cluster;
use crate::embedding::EmbeddingExtractor;
use crate::types::{
    DiarizationConfig, DiarizationResult, Segment, SpeakerId, SpeakerTurn, TimeRange,
};
use crate::vad::{VadConfig, VadError, VoiceActivityDetector, segment_speech};
use crate::wav;
use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    #[error("VAD error: {0}")]
    Vad(#[from] VadError),
    #[error("embedding error: {0}")]
    Embedding(#[from] crate::embedding::EmbeddingError),
    #[error("WAV error: {0}")]
    Wav(#[from] wav::WavError),
    #[error("no speech detected in audio")]
    NoSpeech,
}

pub struct Pipeline {
    config: DiarizationConfig,
    vad_config: VadConfig,
}

impl Pipeline {
    pub fn new(config: DiarizationConfig, vad_config: VadConfig) -> Self {
        Self { config, vad_config }
    }

    /// Run the full diarization pipeline on raw f32 samples.
    pub fn run<E: EmbeddingExtractor, V: VoiceActivityDetector>(
        &self,
        samples: &[f32],
        extractor: &E,
        vad: &mut V,
    ) -> Result<DiarizationResult, PipelineError> {
        // Step 1: VAD — find speech regions.
        let speech_regions = segment_speech(vad, samples, &self.config, &self.vad_config)?;

        if speech_regions.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                turns: Vec::new(),
                num_speakers: 0,
            });
        }

        // Step 2: Extract embeddings per speech region.
        let sr = self.config.sample_rate.get() as f64;
        let window = self.config.window_samples();
        let hop = self.config.hop_samples();
        let mut embeddings = Vec::new();
        let mut time_ranges = Vec::new();

        for &(start, end) in &speech_regions {
            let region = &samples[start..end];

            if region.len() < window {
                // Region too short for a full window — extract one embedding from padded region.
                let mut padded = vec![0.0f32; window];
                padded[..region.len()].copy_from_slice(region);
                let emb = extractor.extract(&padded, &self.config)?;
                embeddings.push(emb);
                time_ranges.push(TimeRange {
                    start: start as f64 / sr,
                    end: end as f64 / sr,
                });
            } else {
                // Slide windows across the region.
                let mut offset = 0;
                while offset + window <= region.len() {
                    let chunk = &region[offset..offset + window];
                    let emb = extractor.extract(chunk, &self.config)?;
                    embeddings.push(emb);
                    time_ranges.push(TimeRange {
                        start: (start + offset) as f64 / sr,
                        end: (start + offset + window) as f64 / sr,
                    });
                    offset += hop;
                }
            }
        }

        if embeddings.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                turns: Vec::new(),
                num_speakers: 0,
            });
        }

        // Step 3: AHC clustering.
        let labels = agglomerative_cluster(&embeddings, self.config.threshold);
        let num_speakers = labels.iter().copied().max().map_or(0, |m| m + 1);

        // Step 4: Build segments.
        let mut segments: Vec<Segment> = labels
            .iter()
            .zip(time_ranges.iter())
            .map(|(&label, &time)| Segment {
                time,
                speaker: Some(SpeakerId(label as u32)),
                confidence: None,
            })
            .collect();

        // Step 5: Merge adjacent same-speaker segments with small gaps.
        segments = merge_segments(segments, self.config.max_gap_secs as f64);
        segments.retain(|s| s.time.duration() >= self.config.min_speech_secs as f64);

        let turns: Vec<SpeakerTurn> = segments
            .iter()
            .filter_map(|s| {
                s.speaker.map(|spk| SpeakerTurn {
                    speaker: spk,
                    time: s.time,
                    text: None,
                })
            })
            .collect();

        Ok(DiarizationResult {
            segments,
            turns,
            num_speakers,
        })
    }

    /// Run the pipeline from a WAV file path.
    pub fn run_from_wav<E: EmbeddingExtractor, V: VoiceActivityDetector>(
        &self,
        path: &Path,
        extractor: &E,
        vad: &mut V,
    ) -> Result<DiarizationResult, PipelineError> {
        let (samples, _sample_rate) = wav::read_wav(path)?;
        self.run(&samples, extractor, vad)
    }
}

fn merge_segments(segments: Vec<Segment>, max_gap_secs: f64) -> Vec<Segment> {
    if segments.is_empty() {
        return segments;
    }
    let mut merged = Vec::new();
    let mut current = segments[0].clone();

    for next in segments.into_iter().skip(1) {
        if current.speaker == next.speaker && next.time.start - current.time.end <= max_gap_secs {
            current.time.end = next.time.end;
        } else {
            merged.push(current);
            current = next;
        }
    }
    merged.push(current);
    merged
}
```

- [ ] **Step 3: Register module and re-exports in lib.rs**

In `src/lib.rs`, add after `pub mod overlap;`:

```rust
pub mod pipeline;
```

Add to re-exports:

```rust
pub use pipeline::{Pipeline, PipelineError};
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test test_pipeline`
Expected: all 3 tests pass.

Run: `cargo test`
Expected: all existing tests still pass (no regressions).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.rs src/lib.rs tests/test_pipeline.rs
git commit -m "feat: add Pipeline — VAD + embedding + AHC end-to-end"
```

---

### Task 6: Model Download Script

**Files:**
- Create: `scripts/download-models.sh`
- Create: `models/.gitkeep`

- [ ] **Step 1: Create download script**

Create `scripts/download-models.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"
mkdir -p "$MODEL_DIR"

WESPEAKER_URL="https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34/voxceleb_resnet34.onnx"
SILERO_URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

echo "Downloading WeSpeaker ResNet34 (VoxCeleb)..."
if [ -f "$MODEL_DIR/wespeaker_resnet34.onnx" ]; then
    echo "  Already exists, skipping."
else
    curl -L --progress-bar -o "$MODEL_DIR/wespeaker_resnet34.onnx" "$WESPEAKER_URL"
    echo "  Done: $(du -h "$MODEL_DIR/wespeaker_resnet34.onnx" | cut -f1)"
fi

echo "Downloading Silero VAD v5..."
if [ -f "$MODEL_DIR/silero_vad.onnx" ]; then
    echo "  Already exists, skipping."
else
    curl -L --progress-bar -o "$MODEL_DIR/silero_vad.onnx" "$SILERO_URL"
    echo "  Done: $(du -h "$MODEL_DIR/silero_vad.onnx" | cut -f1)"
fi

echo ""
echo "Models downloaded to $MODEL_DIR/"
ls -lh "$MODEL_DIR/"*.onnx
```

- [ ] **Step 2: Create models directory placeholder**

```bash
mkdir -p models
touch models/.gitkeep
```

- [ ] **Step 3: Add `models/*.onnx` to `.gitignore`**

Append to `.gitignore`:

```
models/*.onnx
```

- [ ] **Step 4: Make script executable and verify**

```bash
chmod +x scripts/download-models.sh
```

Run: `scripts/download-models.sh`
Expected: both models download successfully. WeSpeaker ~25MB, Silero ~2MB.

- [ ] **Step 5: Commit**

```bash
git add scripts/download-models.sh models/.gitkeep .gitignore
git commit -m "feat: add model download script for WeSpeaker + Silero VAD"
```

---

### Task 7: End-to-End Integration Test with Real Models

**Files:**
- Create: `tests/test_e2e_onnx.rs`

These tests are gated behind an environment variable `POLYVOICE_MODEL_DIR` so they don't run in CI without models present.

- [ ] **Step 1: Write integration test**

Create `tests/test_e2e_onnx.rs`:

```rust
//! End-to-end tests with real ONNX models.
//! Requires: POLYVOICE_MODEL_DIR env var pointing to downloaded models.
//! Run: POLYVOICE_MODEL_DIR=models cargo test --test test_e2e_onnx --features onnx

#[cfg(feature = "onnx")]
mod e2e {
    use polyvoice::{
        DiarizationConfig, EcapaTdnnExtractor, Pipeline, VadConfig,
        silero_vad::SileroVad,
        vad::VoiceActivityDetector,
    };
    use std::path::PathBuf;

    fn model_dir() -> Option<PathBuf> {
        std::env::var("POLYVOICE_MODEL_DIR").ok().map(PathBuf::from)
    }

    fn sine_wave(freq: f32, duration_secs: f32, amplitude: f32) -> Vec<f32> {
        let sample_rate = 16000;
        let n = (duration_secs * sample_rate as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin() * amplitude
            })
            .collect()
    }

    #[test]
    fn test_silero_vad_detects_speech() {
        let dir = match model_dir() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: POLYVOICE_MODEL_DIR not set");
                return;
            }
        };

        let vad_path = dir.join("silero_vad.onnx");
        let mut vad = SileroVad::new(&vad_path, 512).expect("load Silero VAD");

        // Loud sine wave should trigger speech detection.
        let loud = sine_wave(300.0, 1.0, 0.8);
        let probs = vad.process(&loud[..512 * (loud.len() / 512)]).unwrap();
        let max_prob = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // Silero may or may not detect pure sine as speech — we just verify no crash
        // and that probabilities are in [0, 1].
        assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
        eprintln!("Silero VAD max prob on sine: {max_prob:.3}");
    }

    #[test]
    fn test_wespeaker_embedding_shape() {
        let dir = match model_dir() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: POLYVOICE_MODEL_DIR not set");
                return;
            }
        };

        let model_path = dir.join("wespeaker_resnet34.onnx");
        let extractor = EcapaTdnnExtractor::new(&model_path, 256, 2)
            .expect("load WeSpeaker");

        let config = DiarizationConfig::default();
        let samples = sine_wave(440.0, 2.0, 0.5);

        let embedding = polyvoice::EmbeddingExtractor::extract(
            &extractor,
            &samples[..config.window_samples()],
            &config,
        )
        .expect("extract embedding");

        assert_eq!(embedding.len(), 256);
        // L2-normalized: norm ≈ 1.0
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_full_pipeline_onnx() {
        let dir = match model_dir() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: POLYVOICE_MODEL_DIR not set");
                return;
            }
        };

        let model_path = dir.join("wespeaker_resnet34.onnx");
        let vad_path = dir.join("silero_vad.onnx");

        let extractor = EcapaTdnnExtractor::new(&model_path, 256, 4)
            .expect("load WeSpeaker");
        let mut vad = SileroVad::new(&vad_path, 512)
            .expect("load Silero VAD");

        let config = DiarizationConfig {
            window_secs: 1.5,
            hop_secs: 0.75,
            threshold: 0.5,
            ..Default::default()
        };
        let vad_config = VadConfig {
            frame_size: 512,
            threshold: 0.3,
            min_silence_ms: 300.0,
        };

        let pipeline = Pipeline::new(config, vad_config);

        // 10 seconds of audio: two "speakers" at different frequencies.
        let mut samples = sine_wave(200.0, 5.0, 0.7);
        samples.extend_from_slice(&sine_wave(800.0, 5.0, 0.7));

        let result = pipeline.run(&samples, &extractor, &mut vad).unwrap();
        eprintln!("Pipeline result: {} speakers, {} turns",
            result.num_speakers, result.turns.len());
        for turn in &result.turns {
            eprintln!("  {}: {:.2}s - {:.2}s",
                turn.speaker, turn.time.start, turn.time.end);
        }
    }
}
```

- [ ] **Step 2: Run tests with models**

Run: `POLYVOICE_MODEL_DIR=models cargo test --test test_e2e_onnx --features onnx -- --nocapture`
Expected: all 3 tests pass (or skip gracefully if models not downloaded).

- [ ] **Step 3: Debug and fix any model I/O issues**

The WeSpeaker model may have a different embedding dimension (e.g., 256 vs 192) or input name. Use `ort` to inspect the model and fix parameters if needed:

```bash
# Quick inspection via Python if available:
# python3 -c "import onnxruntime as ort; s = ort.InferenceSession('models/wespeaker_resnet34.onnx'); print([i.name for i in s.get_inputs()]); print([o.name for o in s.get_outputs()])"
```

Adjust `EcapaTdnnExtractor::new()` parameters or create a thin `WeSpeakerExtractor` wrapper if the I/O shape differs.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_onnx.rs
git commit -m "test: add end-to-end ONNX integration tests"
```

---

### Task 8: DER Evaluation Benchmark

**Files:**
- Create: `benches/der_ami.rs`
- Modify: `Cargo.toml` (add bench target)

This task creates the infrastructure for DER evaluation. Full AMI dataset evaluation requires downloading the AMI corpus, but the benchmark framework can be validated with synthetic ground truth.

- [ ] **Step 1: Add bench target to Cargo.toml**

Add to `Cargo.toml`:

```toml
[[bench]]
name = "der_ami"
harness = false
```

- [ ] **Step 2: Write DER computation utility**

Create `benches/der_ami.rs`:

```rust
//! DER (Diarization Error Rate) evaluation.
//!
//! Usage:
//!   cargo bench --bench der_ami --features onnx
//!
//! Requires POLYVOICE_MODEL_DIR to be set.

use criterion::{Criterion, criterion_group, criterion_main};
use std::path::PathBuf;

/// A reference annotation: (start_sec, end_sec, speaker_label).
type Annotation = Vec<(f64, f64, &'static str)>;

/// Compute DER between reference and hypothesis annotations.
///
/// DER = (false_alarm + missed_speech + speaker_confusion) / total_reference_speech
///
/// Uses a simplified frame-based approach at 100ms resolution.
fn compute_der(
    reference: &[(f64, f64, &str)],
    hypothesis: &[(f64, f64, u32)],
    collar: f64,
) -> f64 {
    if reference.is_empty() {
        return 0.0;
    }

    let max_time = reference
        .iter()
        .map(|r| r.1)
        .chain(hypothesis.iter().map(|h| h.1))
        .fold(0.0f64, f64::max);

    let resolution = 0.1; // 100ms frames
    let n_frames = (max_time / resolution).ceil() as usize;

    // Build frame-level labels.
    let mut ref_frames: Vec<Option<&str>> = vec![None; n_frames];
    for &(start, end, speaker) in reference {
        let s = ((start + collar) / resolution).ceil() as usize;
        let e = ((end - collar) / resolution).floor() as usize;
        for i in s..e.min(n_frames) {
            ref_frames[i] = Some(speaker);
        }
    }

    let mut hyp_frames: Vec<Option<u32>> = vec![None; n_frames];
    for &(start, end, speaker) in hypothesis {
        let s = (start / resolution).ceil() as usize;
        let e = (end / resolution).floor() as usize;
        for i in s..e.min(n_frames) {
            hyp_frames[i] = Some(speaker);
        }
    }

    // Build optimal ref→hyp speaker mapping based on overlap.
    let ref_speakers: Vec<&str> = reference.iter().map(|r| r.2).collect::<std::collections::HashSet<_>>().into_iter().collect();
    let hyp_speakers: Vec<u32> = hypothesis.iter().map(|h| h.2).collect::<std::collections::HashSet<_>>().into_iter().collect();

    // Count co-occurrences for Hungarian-style mapping.
    let mut overlap = std::collections::HashMap::new();
    for i in 0..n_frames {
        if let (Some(r), Some(h)) = (ref_frames[i], hyp_frames[i]) {
            *overlap.entry((r, h)).or_insert(0usize) += 1;
        }
    }

    // Greedy mapping (good enough for evaluation).
    let mut mapping: std::collections::HashMap<u32, &str> = std::collections::HashMap::new();
    let mut used_ref: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut pairs: Vec<((&str, u32), usize)> = overlap.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    for ((r, h), _) in pairs {
        if !mapping.contains_key(&h) && !used_ref.contains(r) {
            mapping.insert(h, r);
            used_ref.insert(r);
        }
    }

    let mut total_ref = 0usize;
    let mut missed = 0usize;
    let mut false_alarm = 0usize;
    let mut confusion = 0usize;

    for i in 0..n_frames {
        match (ref_frames[i], hyp_frames[i]) {
            (Some(_), None) => {
                total_ref += 1;
                missed += 1;
            }
            (None, Some(_)) => {
                false_alarm += 1;
            }
            (Some(r), Some(h)) => {
                total_ref += 1;
                if mapping.get(&h) != Some(&r) {
                    confusion += 1;
                }
            }
            (None, None) => {}
        }
    }

    if total_ref == 0 {
        return 0.0;
    }

    (missed + false_alarm + confusion) as f64 / total_ref as f64
}

fn bench_der_synthetic(c: &mut Criterion) {
    // Synthetic ground truth: two speakers, clean boundaries.
    let reference: Vec<(f64, f64, &str)> = vec![
        (0.0, 3.0, "A"),
        (3.5, 6.0, "B"),
        (6.5, 10.0, "A"),
    ];

    // Perfect hypothesis.
    let perfect_hyp: Vec<(f64, f64, u32)> = vec![
        (0.0, 3.0, 0),
        (3.5, 6.0, 1),
        (6.5, 10.0, 0),
    ];

    let der = compute_der(&reference, &perfect_hyp, 0.25);
    eprintln!("Synthetic DER (perfect): {:.1}%", der * 100.0);
    assert!(der < 0.05, "perfect hypothesis should have near-zero DER");

    // Imperfect hypothesis: speaker confusion in middle.
    let imperfect_hyp: Vec<(f64, f64, u32)> = vec![
        (0.0, 3.0, 0),
        (3.5, 6.0, 0), // wrong speaker!
        (6.5, 10.0, 0),
    ];

    let der2 = compute_der(&reference, &imperfect_hyp, 0.25);
    eprintln!("Synthetic DER (confused): {:.1}%", der2 * 100.0);
    assert!(der2 > 0.1, "confused hypothesis should have significant DER");

    c.bench_function("der_synthetic", |b| {
        b.iter(|| compute_der(&reference, &perfect_hyp, 0.25));
    });
}

criterion_group!(benches, bench_der_synthetic);
criterion_main!(benches);
```

- [ ] **Step 3: Run benchmark**

Run: `cargo bench --bench der_ami`
Expected: benchmark runs, prints synthetic DER values, no crashes.

- [ ] **Step 4: Commit**

```bash
git add benches/der_ami.rs Cargo.toml
git commit -m "feat: add DER evaluation benchmark framework"
```

---

## Post-plan Verification Checklist

After completing all tasks:

- [ ] `cargo check --all-features` passes
- [ ] `cargo clippy --all-features -- -D warnings` passes
- [ ] `cargo test` passes (all existing + new tests)
- [ ] `cargo test --features onnx` passes
- [ ] `POLYVOICE_MODEL_DIR=models cargo test --test test_e2e_onnx --features onnx` passes with downloaded models
- [ ] `cargo bench --bench der_ami` runs successfully
- [ ] `scripts/download-models.sh` downloads both models
