# M2 — CAM++ Embedder + `Embedder` Trait Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the v1.0 `Embedder` trait and a CAM++ embedder (192-d, ~7M params, ~7 MB FP32). CAM++ becomes the Mobile-tier embedder of choice (M5 ships INT8, M6 wires it into Pipeline). Keep ResNet34 reachable through the new trait via a thin adapter. Add overlap-mask helper for M4's resegmentation.

**Architecture:** New single-file module `src/embedder.rs` (feature-gated `embedder`, default-on). Holds `Embedder` trait + `EmbedderError` + `CamPlusPlusExtractor` (gated `onnx+embedder`) + `ResNet34Adapter` (wraps existing `FbankOnnxExtractor`) + `EmbedderPool` (crossbeam-queue based, generic over `Embedder`) + `apply_overlap_mask` helper. Pure-Rust pieces (`Embedder` trait, `apply_overlap_mask`, `EmbedderPool` with mock backend) compile to wasm32. ONNX-backed extractors gated behind `onnx`. Existing `embedding::EmbeddingExtractor` / `DummyExtractor` / `OnnxEmbeddingExtractor` / `FbankOnnxExtractor` stay untouched — additive only. M6 will rename `embedder.rs` → `embedding/mod.rs` and remove legacy types.

**Tech Stack:** Rust 2024, `ort 2.0.0-rc.12` (existing), `crossbeam-queue` (existing), `realfft` (existing — used by `features.rs`). No new deps.

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `Cargo.toml` | modify | Add `embedder` feature (default-on) |
| `src/embedder.rs` | create | `Embedder` trait, `EmbedderError`, `CamPlusPlusExtractor`, `ResNet34Adapter`, `EmbedderPool`, `apply_overlap_mask` |
| `src/lib.rs` | modify | `pub mod embedder;` gated, re-exports |
| `src/models/manifest.toml` | modify | Add `[models.cam_pp_fp32]` entry |
| `tests/embedder_test.rs` | create | `#[ignore]` integration test against real CAM++ model |
| `CHANGELOG.md` | modify | Unreleased M2 section |

Total roughly 700 lines Rust + 8 lines TOML.

---

## Task 1: Add `embedder` Cargo feature

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1.1: Read current `[features]` block**

```bash
sed -n '14,40p' /Users/ekhodzitsky/Documents/personal/polyvoice-m2/Cargo.toml
```

After M0+M1, the features should be `default = ["spectral", "segmentation"]` plus all the others.

- [ ] **Step 1.2: Add `embedder` feature**

In the `[features]` block, change `default` and append the new feature:

```toml
default = ["spectral", "segmentation", "embedder"]
```

After the `segmentation = []` line, append:

```toml

# v1.0 Embedder trait + CAM++/ResNet34 adapters + EmbedderPool + overlap masking.
# The pure-Rust algorithmic core (trait, overlap mask, pool with mock backend)
# compiles to wasm32-clean. The ONNX-backed `CamPlusPlusExtractor` and
# `ResNet34Adapter` additionally require `onnx`.
embedder = []
```

- [ ] **Step 1.3: Verify all feature combos still build**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m2
cargo check
cargo check --features download
cargo check --features cli
cargo check --features ffi
cargo check --features onnx
cargo check --features segmentation
cargo check --features embedder
cargo check --features onnx,embedder
cargo check --no-default-features
cargo check --target wasm32-unknown-unknown --no-default-features --lib
cargo check --target wasm32-unknown-unknown --no-default-features --features embedder --lib
cargo check --all-features
```

All must exit 0.

- [ ] **Step 1.4: Commit**

```bash
git add Cargo.toml
git commit -m "feat(cargo): add embedder feature flag for v1.0 M2 work"
```

---

## Task 2: `Embedder` trait + `EmbedderError`

**Files:**
- Create: `src/embedder.rs`

- [ ] **Step 2.1: Write the failing tests**

Create `src/embedder.rs`:

```rust
//! v1.0 `Embedder` trait + concrete extractors (CAM++, ResNet34) + pool +
//! overlap-mask helper.
//!
//! Added in v0.6 (M2). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §3.1, §5.3.

#[cfg(test)]
mod trait_tests {
    use super::*;

    /// In-memory dummy used by trait tests.
    struct ConstantEmbedder { values: Vec<f32> }

    impl Embedder for ConstantEmbedder {
        fn dim(&self) -> usize { self.values.len() }
        fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            Ok(self.values.clone())
        }
    }

    #[test]
    fn embedder_trait_object_is_dyn_compatible() {
        let e = ConstantEmbedder { values: vec![0.1, 0.2, 0.3] };
        let _b: Box<dyn Embedder> = Box::new(e);
    }

    #[test]
    fn embedder_default_batch_is_serial() {
        let e = ConstantEmbedder { values: vec![0.5; 4] };
        let inputs: Vec<&[f32]> = vec![&[][..], &[][..], &[][..]];
        let out = e.embed_batch(&inputs).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|v| v.len() == 4 && v[0] == 0.5));
    }

    #[test]
    fn embedder_dim_matches_output() {
        let e = ConstantEmbedder { values: vec![1.0; 192] };
        assert_eq!(e.dim(), 192);
        assert_eq!(e.embed(&[]).unwrap().len(), 192);
    }

    #[test]
    fn embedder_error_audio_too_short_displays() {
        let err = EmbedderError::AudioTooShort { actual_secs: 0.05, min_secs: 0.25 };
        let msg = format!("{err}");
        assert!(msg.contains("0.05"));
        assert!(msg.contains("0.25"));
    }
}
```

- [ ] **Step 2.2: Run tests, confirm compile-failure**

```bash
cargo test --features embedder --lib embedder::trait_tests 2>&1 | head -25
```

Expected: errors about `Embedder`, `EmbedderError` not found.

- [ ] **Step 2.3: Implement the trait + error type**

Replace `src/embedder.rs` with the implementation (keep the test block at the bottom unchanged):

```rust
//! v1.0 `Embedder` trait + concrete extractors (CAM++, ResNet34) + pool +
//! overlap-mask helper.

/// Speaker embedding extractor — turns a slice of 16 kHz mono audio into a
/// fixed-dimension embedding vector. Implementations are expected to L2-normalize
/// their output so cosine similarity is a meaningful metric downstream.
///
/// In v1.0 (M2) the polyvoice crate introduces `Embedder` as the canonical
/// trait. The legacy `EmbeddingExtractor` trait and its implementations
/// (`FbankOnnxExtractor`, `OnnxEmbeddingExtractor`, `DummyExtractor`) remain
/// available unchanged — M6 will deprecate them.
pub trait Embedder: Send + Sync {
    /// Output dimension of this embedder. Constant per instance.
    fn dim(&self) -> usize;

    /// Compute an embedding for one audio segment.
    ///
    /// **Requires:** `audio` is 16 kHz mono PCM.
    /// **Guarantees on Ok:** `result.len() == self.dim()` and the vector is L2-normalized
    /// (`|sum(x²)¹ᐟ² − 1.0| < 1e-3`).
    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError>;

    /// Compute embeddings for a batch of audio segments. Default implementation
    /// is sequential; impls may override with a true batched ONNX call.
    fn embed_batch(&self, audios: &[&[f32]]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        audios.iter().map(|a| self.embed(a)).collect()
    }
}

/// Errors from `Embedder` implementations.
#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    #[error("audio too short for this embedder: {actual_secs:.3}s < {min_secs:.3}s")]
    AudioTooShort { actual_secs: f32, min_secs: f32 },

    #[error("ONNX inference failed: {detail}")]
    InferenceFailed { detail: String },

    #[error("expected embedding dim {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    #[error("model file io error on {path}: {detail}")]
    ModelIo { path: std::path::PathBuf, detail: String },

    #[error("legacy adapter error: {0}")]
    Legacy(String),
}

#[cfg(test)]
mod trait_tests {
    // (test block from Step 2.1 stays unchanged)
}
```

- [ ] **Step 2.4: Wire module into lib.rs**

In `src/lib.rs`, after the existing `#[cfg(feature = "segmentation")] pub mod segmentation;` block (and its re-exports), append:

```rust

#[cfg(feature = "embedder")]
pub mod embedder;
```

- [ ] **Step 2.5: Run tests + clippy + fmt + wasm**

```bash
cargo test --features embedder --lib embedder::trait_tests
cargo fmt
cargo clippy --features embedder --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features embedder --lib
```

Expected: 4 tests pass, all clean.

- [ ] **Step 2.6: Commit**

```bash
git add src/embedder.rs src/lib.rs
git commit -m "feat(embedder): add Embedder trait + EmbedderError"
```

---

## Task 3: `apply_overlap_mask` helper

**Files:**
- Modify: `src/embedder.rs`

This function zero-fills audio samples in regions where the powerset segmenter flagged 2-speaker overlap. Used by the pipeline (M6) before calling `Embedder::embed` to prevent overlap contamination of speaker centroids. Pure Rust, wasm-clean.

- [ ] **Step 3.1: Append failing tests**

Add this `#[cfg(test)] mod overlap_mask_tests` block to `src/embedder.rs` (alongside `trait_tests`):

```rust
#[cfg(test)]
mod overlap_mask_tests {
    use super::*;

    #[test]
    fn no_overlap_regions_pass_through() {
        let audio = vec![1.0_f32; 16_000];
        let masked = apply_overlap_mask(&audio, &[], 16_000);
        assert_eq!(masked, audio);
    }

    #[test]
    fn single_overlap_region_is_zeroed() {
        let audio = vec![1.0_f32; 16_000];
        let masked = apply_overlap_mask(&audio, &[(0.5, 0.7)], 16_000);
        // [0.5, 0.7) seconds at 16 kHz = samples [8000, 11200)
        for i in 0..audio.len() {
            if (8000..11200).contains(&i) {
                assert_eq!(masked[i], 0.0, "sample {i} should be zeroed");
            } else {
                assert_eq!(masked[i], 1.0, "sample {i} should pass through");
            }
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let masked = apply_overlap_mask(&[], &[(0.0, 1.0)], 16_000);
        assert!(masked.is_empty());
    }

    #[test]
    fn out_of_bounds_overlap_is_clamped() {
        let audio = vec![1.0_f32; 100];
        // Audio is 0.00625s; the [0.5, 1.0)s overlap is entirely past the audio.
        let masked = apply_overlap_mask(&audio, &[(0.5, 1.0)], 16_000);
        assert_eq!(masked, audio, "out-of-bounds overlap is a no-op");
    }

    #[test]
    fn negative_overlap_start_is_clamped_to_zero() {
        let audio = vec![1.0_f32; 16_000];
        // Overlap starts before audio (t=-1s), ends mid-audio.
        let masked = apply_overlap_mask(&audio, &[(-1.0, 0.5)], 16_000);
        for i in 0..8000 {
            assert_eq!(masked[i], 0.0);
        }
        for i in 8000..16_000 {
            assert_eq!(masked[i], 1.0);
        }
    }

    #[test]
    fn multiple_overlap_regions_all_zeroed() {
        let audio = vec![1.0_f32; 16_000];
        let masked = apply_overlap_mask(
            &audio,
            &[(0.1, 0.2), (0.5, 0.6), (0.9, 1.0)],
            16_000,
        );
        let zero_ranges = [(1600..3200), (8000..9600), (14_400..16_000)];
        for (i, &v) in masked.iter().enumerate() {
            let in_zero = zero_ranges.iter().any(|r| r.contains(&i));
            if in_zero {
                assert_eq!(v, 0.0, "sample {i} should be zeroed");
            } else {
                assert_eq!(v, 1.0, "sample {i} should pass through");
            }
        }
    }

    #[test]
    fn invalid_overlap_with_end_before_start_is_no_op() {
        let audio = vec![1.0_f32; 16_000];
        let masked = apply_overlap_mask(&audio, &[(0.7, 0.5)], 16_000);
        assert_eq!(masked, audio, "end<start is silently skipped");
    }
}
```

- [ ] **Step 3.2: Confirm compile-failure**

```bash
cargo test --features embedder --lib embedder::overlap_mask_tests 2>&1 | head -10
```

Expected: errors about `apply_overlap_mask` not found.

- [ ] **Step 3.3: Implement the helper**

Add to `src/embedder.rs` (above the test blocks):

```rust
/// Zero-fill audio samples in regions where the segmenter flagged a 2-speaker
/// overlap. The returned `Vec<f32>` is a copy of `audio` with zeros in the
/// `(start_secs, end_secs)` ranges listed in `overlap_regions`.
///
/// Out-of-bounds and inverted (end < start) regions are silently clamped or
/// skipped — never panics.
///
/// **Pure Rust, no allocations beyond the output Vec, wasm32-clean.**
pub fn apply_overlap_mask(
    audio: &[f32],
    overlap_regions: &[(f32, f32)],
    sample_rate: u32,
) -> Vec<f32> {
    let mut out = audio.to_vec();
    if out.is_empty() {
        return out;
    }
    let sr = sample_rate as f32;
    for &(start_s, end_s) in overlap_regions {
        if !end_s.is_finite() || !start_s.is_finite() || end_s <= start_s {
            continue;
        }
        let start = (start_s * sr).max(0.0).floor() as usize;
        let end = (end_s * sr).max(0.0).ceil() as usize;
        let end = end.min(out.len());
        if start >= end || start >= out.len() {
            continue;
        }
        for v in &mut out[start..end] {
            *v = 0.0;
        }
    }
    out
}
```

- [ ] **Step 3.4: Verify**

```bash
cargo test --features embedder --lib embedder::overlap_mask_tests
cargo clippy --features embedder --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features embedder --lib
```

Expected: 7 overlap_mask tests pass, all clean.

- [ ] **Step 3.5: Commit**

```bash
git add src/embedder.rs
git commit -m "feat(embedder): add apply_overlap_mask helper"
```

---

## Task 4: `EmbedderPool`

**Files:**
- Modify: `src/embedder.rs`

`EmbedderPool` is a crossbeam-queue-based pool of `Embedder` instances. Used by Pipeline (M6) for concurrent embedding extraction across many audio segments. Pure Rust (no ONNX dependency). Tests use `Box<dyn Embedder>` with a counting mock to verify the pool returns embedders correctly.

- [ ] **Step 4.1: Write failing tests**

Append to `src/embedder.rs`:

```rust
#[cfg(test)]
mod pool_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts how many times `embed` was called.
    struct CountingEmbedder {
        counter: Arc<AtomicUsize>,
        dim: usize,
    }

    impl Embedder for CountingEmbedder {
        fn dim(&self) -> usize { self.dim }
        fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0.0; self.dim])
        }
    }

    fn make_pool(n: usize) -> (EmbedderPool<CountingEmbedder>, Arc<AtomicUsize>) {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut embedders = Vec::with_capacity(n);
        for _ in 0..n {
            embedders.push(CountingEmbedder { counter: counter.clone(), dim: 192 });
        }
        let pool = EmbedderPool::new(embedders);
        (pool, counter)
    }

    #[test]
    fn pool_with_single_embedder_round_trip() {
        let (pool, counter) = make_pool(1);
        let result = pool.embed(&[0.0_f32; 100]).unwrap();
        assert_eq!(result.len(), 192);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn pool_dim_is_consistent() {
        let (pool, _) = make_pool(4);
        assert_eq!(pool.dim(), 192);
    }

    #[test]
    fn pool_serial_embed_increments_counter_per_call() {
        let (pool, counter) = make_pool(2);
        for _ in 0..5 {
            pool.embed(&[0.0_f32; 100]).unwrap();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn pool_with_zero_embedders_errors() {
        let pool: EmbedderPool<CountingEmbedder> = EmbedderPool::new(Vec::new());
        let err = pool.embed(&[0.0_f32; 100]).expect_err("empty pool must fail");
        assert!(matches!(err, EmbedderError::Legacy(_)));
    }
}
```

- [ ] **Step 4.2: Confirm compile-failure**

```bash
cargo test --features embedder --lib embedder::pool_tests 2>&1 | head -10
```

- [ ] **Step 4.3: Implement `EmbedderPool`**

Add to `src/embedder.rs` (above test blocks):

```rust
use crossbeam_queue::ArrayQueue;
use std::sync::Arc;

/// Lock-free pool of `Embedder` instances for concurrent extraction.
///
/// Generic over `E: Embedder` so the same pool implementation works for
/// `CamPlusPlusExtractor`, `ResNet34Adapter`, or any user-provided embedder.
/// All embedders in a pool must share the same output dimension.
pub struct EmbedderPool<E: Embedder> {
    queue: Arc<ArrayQueue<E>>,
    dim: usize,
    capacity: usize,
}

impl<E: Embedder> EmbedderPool<E> {
    /// Build a pool from a list of embedders. All must share the same `dim()`.
    /// An empty list constructs a pool that fails on every call (returns
    /// `EmbedderError::Legacy("empty pool")`).
    pub fn new(embedders: Vec<E>) -> Self {
        let dim = embedders.first().map(|e| e.dim()).unwrap_or(0);
        let capacity = embedders.len().max(1);
        let queue = Arc::new(ArrayQueue::new(capacity));
        for e in embedders {
            // ArrayQueue::push only fails if full; capacity == count, so push always succeeds.
            let _ = queue.push(e);
        }
        Self { queue, dim, capacity }
    }

    pub fn dim(&self) -> usize { self.dim }
    pub fn capacity(&self) -> usize { self.capacity }

    /// Extract a single embedding using the next-available pooled embedder.
    /// Blocks (busy-spins) until one is free.
    pub fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        if self.queue.is_empty() && self.dim == 0 {
            // Empty-construction case.
            return Err(EmbedderError::Legacy("empty pool".to_owned()));
        }
        // Acquire (busy-wait fallback for simplicity; real Pipeline use is
        // through `rayon::par_iter` which already throttles concurrency).
        let embedder = loop {
            if let Some(e) = self.queue.pop() {
                break e;
            }
            std::hint::spin_loop();
        };
        let result = embedder.embed(audio);
        // Always return the embedder.
        let _ = self.queue.push(embedder);
        result
    }
}
```

- [ ] **Step 4.4: Verify**

```bash
cargo test --features embedder --lib embedder::pool_tests
cargo clippy --features embedder --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features embedder --lib
```

Expected: 4 pool tests pass. wasm32 clean (crossbeam-queue is wasm-friendly).

- [ ] **Step 4.5: Commit**

```bash
git add src/embedder.rs
git commit -m "feat(embedder): add EmbedderPool over Embedder trait"
```

---

## Task 5: `ResNet34Adapter` (wraps existing `FbankOnnxExtractor`)

**Files:**
- Modify: `src/embedder.rs`

Thin adapter: `ResNet34Adapter` wraps the existing `FbankOnnxExtractor` (from `src/ecapa.rs`) so users can use it through the new `Embedder` trait. Gated `onnx + embedder`.

- [ ] **Step 5.1: Append the adapter implementation**

Add this gated block to `src/embedder.rs`:

```rust
#[cfg(feature = "onnx")]
mod onnx_adapters {
    use super::*;
    use crate::ecapa::FbankOnnxExtractor;
    use crate::embedding::EmbeddingExtractor;
    use std::path::Path;

    /// New-trait adapter for the existing `FbankOnnxExtractor` (WeSpeaker ResNet34, 256-d).
    ///
    /// The legacy `FbankOnnxExtractor` already implements the v0.5.x
    /// `EmbeddingExtractor`; this adapter exposes the same model through the
    /// v1.0 `Embedder` trait. M6 will fold this into a unified type.
    pub struct ResNet34Adapter {
        inner: FbankOnnxExtractor,
        dim: usize,
    }

    impl ResNet34Adapter {
        /// Load the WeSpeaker ResNet34 ONNX model.
        pub fn new(path: impl AsRef<Path>, pool_size: usize) -> Result<Self, EmbedderError> {
            let inner = FbankOnnxExtractor::new(path.as_ref(), 256, pool_size).map_err(|e| {
                EmbedderError::ModelIo {
                    path: path.as_ref().to_path_buf(),
                    detail: format!("{e}"),
                }
            })?;
            Ok(Self { inner, dim: 256 })
        }
    }

    impl Embedder for ResNet34Adapter {
        fn dim(&self) -> usize { self.dim }

        fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            self.inner.extract(audio).map_err(|e| EmbedderError::Legacy(format!("{e}")))
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx_adapters::ResNet34Adapter;
```

The `ecapa::FbankOnnxExtractor::new` signature in v0.5.x is `new(path, dim, pool_size) -> Result<Self, _>`. Verify by reading `src/ecapa.rs`:

```bash
grep -A 5 "impl FbankOnnxExtractor" /Users/ekhodzitsky/Documents/personal/polyvoice-m2/src/ecapa.rs | head -10
```

Adjust the call signature to match exactly.

- [ ] **Step 5.2: Build with onnx + embedder**

```bash
cargo check --features onnx,embedder --lib 2>&1 | tail -5
cargo clippy --features onnx,embedder --lib -- -D warnings 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 5.3: Verify wasm32 still builds without onnx**

```bash
cargo check --target wasm32-unknown-unknown --no-default-features --features embedder --lib
```

Expected: clean (the `onnx_adapters` module is gated out).

- [ ] **Step 5.4: Commit**

```bash
git add src/embedder.rs
git commit -m "feat(embedder): add ResNet34Adapter over existing FbankOnnxExtractor"
```

---

## Task 6: `CamPlusPlusExtractor` (CAM++ ONNX wrapper)

**Files:**
- Modify: `src/embedder.rs`

CAM++ uses the SAME 80-bin log-mel fbank features as ResNet34. We can reuse `FbankOnnxExtractor` as the backing implementation — only the output dim differs (192 instead of 256).

- [ ] **Step 6.1: Append the implementation**

Inside the existing `#[cfg(feature = "onnx")] mod onnx_adapters` block in `src/embedder.rs` (next to `ResNet34Adapter`), append:

```rust
    /// CAM++ embedder (Channel-Attentive Multi-scale Pooling, ~7M params, 192-d output).
    ///
    /// Targets the Mobile profile of v1.0. Uses the same 80-bin log-mel fbank
    /// pipeline as ResNet34 (wraps `FbankOnnxExtractor` internally with dim=192).
    /// Models with different fbank parameters or input shapes will produce
    /// shape-mismatch errors at inference time.
    pub struct CamPlusPlusExtractor {
        inner: FbankOnnxExtractor,
    }

    impl CamPlusPlusExtractor {
        /// Load a CAM++ ONNX model. Pool size controls the number of concurrent
        /// ONNX sessions held internally (canonical: `num_cpus().min(4)`).
        pub fn new(path: impl AsRef<Path>, pool_size: usize) -> Result<Self, EmbedderError> {
            let inner = FbankOnnxExtractor::new(path.as_ref(), 192, pool_size).map_err(|e| {
                EmbedderError::ModelIo {
                    path: path.as_ref().to_path_buf(),
                    detail: format!("{e}"),
                }
            })?;
            Ok(Self { inner })
        }
    }

    impl Embedder for CamPlusPlusExtractor {
        fn dim(&self) -> usize { 192 }

        fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            self.inner.extract(audio).map_err(|e| EmbedderError::Legacy(format!("{e}")))
        }
    }
```

And update the `pub use` line to include `CamPlusPlusExtractor`:

```rust
#[cfg(feature = "onnx")]
pub use onnx_adapters::{CamPlusPlusExtractor, ResNet34Adapter};
```

- [ ] **Step 6.2: Verify build**

```bash
cargo check --features onnx,embedder --lib
cargo test --features onnx,embedder --lib embedder::
cargo clippy --features onnx,embedder --lib -- -D warnings
```

Expected: 15+ embedder tests pass (4 trait + 7 overlap_mask + 4 pool + any new ones), clippy clean.

- [ ] **Step 6.3: Commit**

```bash
git add src/embedder.rs
git commit -m "feat(embedder): add CamPlusPlusExtractor (192-d CAM++ wrapper)"
```

---

## Task 7: Add `cam_pp_fp32` manifest entry

**Files:**
- Modify: `src/models/manifest.toml`

Discover the URL+SHA-256 for the CAM++ ONNX model. Likely candidates:
1. `https://huggingface.co/csukuangfj/sherpa-onnx-zh_en_speaker_diarization_cam++/resolve/main/...`
2. `https://github.com/wenet-e2e/wespeaker/releases/...` — WeSpeaker model zoo
3. `https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/...`

- [ ] **Step 7.1: Probe candidate URLs**

```bash
mkdir -p /tmp/polyvoice-m2-cam-pp
cd /tmp/polyvoice-m2-cam-pp

# Candidate A: csukuangfj's HF
curl -sILo /dev/null -w "A: status=%{http_code} size=%{size_download}\n" \
  "https://huggingface.co/csukuangfj/sherpa-onnx-zh-wenet-e2e-tts-cam-plus-plus/resolve/main/model.onnx"

# Candidate B: alternate
curl -sILo /dev/null -w "B: status=%{http_code} size=%{size_download}\n" \
  "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet152/resolve/main/voxceleb_resnet152.onnx"

# Candidate C: search HF for CAM++ ONNX
curl -sILo /dev/null -w "C: status=%{http_code} size=%{size_download}\n" \
  "https://huggingface.co/Wespeaker/cam_plus_plus/resolve/main/cam_plus_plus.onnx"

# Candidate D: sherpa-onnx assets release
curl -sILo /dev/null -w "D: status=%{http_code} size=%{size_download}\n" \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-cam-plus-plus.tar.bz2"

# If none of the above work, search HF for "campplus" or "cam-plus":
echo "=== HF search ==="
curl -s "https://huggingface.co/api/models?search=campplus&limit=5" | head -50
```

If none of the candidates returns a direct .onnx of ~5-10 MB, the implementer must:
- Either find a working URL by searching HF / GitHub releases
- Or self-host (extract from a tar.bz2 if one exists, or upload a converted version)
- Or report `BLOCKED` with diagnostics so the controller can decide

**Important fallback:** the existing `wespeaker_resnet34` ONNX model from the manifest IS a working WeSpeaker FP32 model — if no CAM++ ONNX is reachable, the implementer can document that M2 ships with `cam_pp_fp32` pointing to **the same URL as `wespeaker_resnet34_fp32`** as a temporary placeholder (with a `# TODO(M5): swap to real CAM++ when ONNX is published` comment), so manifest stays valid and the integration test exercises ONNX inference end-to-end. This is a pragmatic choice — flag it explicitly in the report.

- [ ] **Step 7.2: Fetch + checksum the chosen URL**

```bash
URL="<chosen-url>"
curl -sL "$URL" -o /tmp/polyvoice-m2-cam-pp/cam_pp.onnx
file /tmp/polyvoice-m2-cam-pp/cam_pp.onnx
shasum -a 256 /tmp/polyvoice-m2-cam-pp/cam_pp.onnx
ls -la /tmp/polyvoice-m2-cam-pp/cam_pp.onnx | awk '{print "size="$5}'
```

`file` should report binary content. Size should be 5–25 MB depending on the model variant.

- [ ] **Step 7.3: Update `src/models/manifest.toml`**

Append a new entry **at the end of the file**:

```toml

[models.cam_pp_fp32]
url      = "<URL_FROM_STEP_7_2>"
sha256   = "<SHA256_FROM_STEP_7_2>"
size     = <SIZE_FROM_STEP_7_2>
filename = "cam_pp_fp32.onnx"
```

**Do NOT modify the existing `[profiles.mobile]` or `[profiles.balanced]` blocks.** Profile mappings stay on `silero_vad` until M6.

- [ ] **Step 7.4: Verify the manifest still parses**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m2
cargo test --features download --lib models::
```

Expected: 17 tests still pass.

- [ ] **Step 7.5: Cleanup**

```bash
rm -rf /tmp/polyvoice-m2-cam-pp
```

- [ ] **Step 7.6: Commit**

```bash
git add src/models/manifest.toml
git commit -m "feat(models): add cam_pp_fp32 manifest entry for M2"
```

---

## Task 8: Network integration test

**Files:**
- Create: `tests/embedder_test.rs`

- [ ] **Step 8.1: Create the test**

Write `tests/embedder_test.rs`:

```rust
//! Integration test for `CamPlusPlusExtractor` and `ResNet34Adapter` against
//! real upstream ONNX models.
//!
//! Runs only when explicitly invoked:
//!   cargo test --features onnx,embedder,download --test embedder_test -- --ignored
//!
//! Downloads ~30 MB of models. Requires network connectivity.

#![cfg(all(feature = "onnx", feature = "embedder", feature = "download"))]
#![allow(clippy::expect_used)]

use polyvoice::embedder::{CamPlusPlusExtractor, Embedder, ResNet34Adapter};
use polyvoice::models::ModelRegistry;
use tempfile::TempDir;

/// 1 second of synthetic 16 kHz mono audio (220 Hz tone).
fn synthetic_audio_1s() -> Vec<f32> {
    use std::f32::consts::PI;
    let sr = 16_000_usize;
    let mut audio = Vec::with_capacity(sr);
    for i in 0..sr {
        let t = i as f32 / sr as f32;
        audio.push((2.0 * PI * 220.0 * t).sin() * 0.3);
    }
    audio
}

#[test]
#[ignore = "real network — run with --ignored"]
fn cam_plus_plus_extractor_produces_192d_normalized_embedding() {
    let tmp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::with_cache_dir(tmp.path()).expect("registry");
    let model_path = registry.ensure("cam_pp_fp32").expect("download must succeed");

    let extractor = CamPlusPlusExtractor::new(&model_path, 1).expect("loads");
    assert_eq!(extractor.dim(), 192);

    let embedding = extractor.embed(&synthetic_audio_1s()).expect("embed runs");
    assert_eq!(embedding.len(), 192);

    // L2 norm should be ~1.0 (the underlying FbankOnnxExtractor L2-normalizes).
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-2, "L2 norm not 1.0: {norm}");
}

#[test]
#[ignore = "real network — run with --ignored"]
fn resnet34_adapter_produces_256d_normalized_embedding() {
    let tmp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::with_cache_dir(tmp.path()).expect("registry");
    let model_path = registry.ensure("wespeaker_resnet34").expect("download");

    let extractor = ResNet34Adapter::new(&model_path, 1).expect("loads");
    assert_eq!(extractor.dim(), 256);

    let embedding = extractor.embed(&synthetic_audio_1s()).expect("embed");
    assert_eq!(embedding.len(), 256);

    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-2, "L2 norm not 1.0: {norm}");
}
```

- [ ] **Step 8.2: Confirm compile**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m2
cargo test --features onnx,embedder,download --test embedder_test -- --list
```

Expected: 2 tests, both `#[ignore]`.

- [ ] **Step 8.3: Commit**

```bash
git add tests/embedder_test.rs
git commit -m "test(embedder): add network integration tests behind --ignored"
```

---

## Task 9: lib.rs re-exports + CHANGELOG + E2E

**Files:**
- Modify: `src/lib.rs`
- Modify: `CHANGELOG.md`

- [ ] **Step 9.1: Add re-exports**

In `src/lib.rs`, after the existing `#[cfg(feature = "embedder")] pub mod embedder;` line (added in Task 2), append:

```rust

#[cfg(feature = "embedder")]
pub use embedder::{Embedder, EmbedderError, EmbedderPool, apply_overlap_mask};

#[cfg(all(feature = "onnx", feature = "embedder"))]
pub use embedder::{CamPlusPlusExtractor, ResNet34Adapter};
```

- [ ] **Step 9.2: Verify all feature combos build**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m2
cargo check
cargo check --features download
cargo check --features cli
cargo check --features ffi
cargo check --features onnx
cargo check --features segmentation
cargo check --features embedder
cargo check --features onnx,embedder
cargo check --features onnx,segmentation,embedder
cargo check --no-default-features
cargo check --target wasm32-unknown-unknown --no-default-features --lib
cargo check --target wasm32-unknown-unknown --no-default-features --features embedder --lib
cargo check --all-features
```

All must exit 0.

- [ ] **Step 9.3: Run all tests**

```bash
cargo test --all-features --lib 2>&1 | tail -3
cargo test --all-features --doc 2>&1 | tail -3
```

Expected: all green.

- [ ] **Step 9.4: Run integration tests (network)**

```bash
cargo test --features onnx,embedder,download --test embedder_test -- --ignored 2>&1 | tail -10
```

Expected: 2/2 pass.

- [ ] **Step 9.5: Lint**

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```

If `cargo fmt --check` flags M2 files, run `cargo fmt` and commit `chore(fmt): apply rustfmt to M2 files`.

- [ ] **Step 9.6: Update CHANGELOG.md Unreleased section**

After the existing M1 `### Added (M1 — Powerset segmentation)` block, append:

```markdown

### Added (M2 — Embedder trait + CAM++)
- `polyvoice::embedder` module: `Embedder` trait, `EmbedderError`, `EmbedderPool`,
  `apply_overlap_mask` helper.
- `CamPlusPlusExtractor` (192-d output, gated `onnx`+`embedder`) — wraps the
  same fbank pipeline as ResNet34 with the CAM++ ONNX model.
- `ResNet34Adapter` — bridges existing `FbankOnnxExtractor` (256-d, WeSpeaker)
  to the new `Embedder` trait. Legacy `EmbeddingExtractor` trait is unchanged.
- New Cargo feature `embedder` (in default features). Pure-Rust core (trait,
  `apply_overlap_mask`, `EmbedderPool` over a generic `E: Embedder`) is
  wasm32-clean; `CamPlusPlusExtractor` and `ResNet34Adapter` additionally
  require `onnx`.
- New manifest entry `[models.cam_pp_fp32]`. Profiles still resolve to
  `wespeaker_resnet34` until M6 swaps them.
```

- [ ] **Step 9.7: Tag the milestone**

```bash
git tag -a m2-complete -m "M2 complete: Embedder trait + CAM++"
```

(Don't push tag.)

- [ ] **Step 9.8: Commit lib.rs + CHANGELOG**

```bash
git add src/lib.rs CHANGELOG.md
git commit -m "feat(lib): re-export embedder surface + document M2 in changelog"
```

- [ ] **Step 9.9: Final git log**

```bash
git log --oneline c5c17b1..HEAD
```

Should show ~9-10 commits.

---

## Self-review checklist

After all tasks:

1. **Spec coverage:** Every M2 deliverable from spec §10.1 / §3.1 / §5.3 maps to a task:
   - `Embedder` trait + types → Task 2
   - `apply_overlap_mask` → Task 3
   - `EmbedderPool` → Task 4
   - `ResNet34Adapter` → Task 5
   - `CamPlusPlusExtractor` → Task 6
   - manifest entry → Task 7
   - integration test → Task 8
   - re-exports + CHANGELOG → Task 9

2. **Additive guarantee:** No removal/rename of `EmbeddingExtractor`, `DummyExtractor`, `FbankOnnxExtractor`, `OnnxEmbeddingExtractor`. Run `cargo semver-checks check-release` against published 0.5.2 — only additions expected (since `0.5 → 0.6` is the major bump).

3. **Wasm32 cleanness:** trait + `apply_overlap_mask` + `EmbedderPool<E>` (with mock `E`) compile to `wasm32-unknown-unknown`. Verified in steps 2.5, 3.4, 4.4, 5.3, 9.2.

4. **No `unwrap`/`expect`/`panic` in lib non-test code:** verified via clippy `unwrap_used = "deny"` lint.

5. **Test coverage:** every public function has at least one test or doc-test. ~15 unit tests + 2 ignored network tests.

6. **Commits are atomic:** ~10 commits total. Each task ends in exactly one commit.

---

## Out of scope for this plan

- Profile manifest swap (`silero_vad`/`wespeaker_resnet34` → `cam_pp_fp32` in profile entries) — that's M6.
- Pipeline integration (`Pipeline::run` calling the embedder) — M6.
- INT8 quantization of CAM++/ResNet34 — M5.
- Removing `src/embedding.rs`, `src/ecapa.rs`, `src/onnx.rs` — M6 breaking redesign.
- Renaming `src/embedder.rs` → `src/embedding/mod.rs` — M6.
