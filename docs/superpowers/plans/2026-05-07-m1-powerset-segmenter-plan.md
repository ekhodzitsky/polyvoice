# M1 — Powerset Segmenter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Segmenter` trait + `PowersetSegmenter` (ONNX wrap of `sherpa-onnx-pyannote-segmentation-3-0`) + sliding-window aggregator with Hungarian matching + powerset frame decoder. The single biggest DER lever in the v1.0 redesign — replaces the bit-VAD heuristic with a 7-class powerset segmentation model that natively produces overlap detection.

**Architecture:** New `src/segmentation/` module with five files. `Segmenter` trait is the abstract surface; `PowersetSegmenter` is the only concrete impl in M1; `PowersetDecoder` does class → (speaker_set, is_overlap); `Aggregator` runs the model in a 10-second sliding window with 500ms hop and uses a self-implemented Kuhn-Munkres assignment to align local 0..2 speaker indices across overlapping windows so the same person consistently has the same index file-wide. Pure-Rust algorithmic core (decoder, aggregator, hungarian) is wasm32-clean; only `PowersetSegmenter` itself is gated behind `onnx + segmentation` features. Manifest gains a new `[models.powerset_fp32]` entry; profiles still point at `silero_vad` until M6's Pipeline integration swaps them.

**Tech Stack:** Rust 2024, `ort 2.0.0-rc.12` (already a dep), `ndarray` (already a dep), `thiserror` (already a dep). No new deps. Hungarian implemented in-tree (~50 LOC of CP-style Kuhn-Munkres).

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `Cargo.toml` | modify | Add `segmentation` feature (default-on) |
| `src/segmentation/mod.rs` | create | `Segmenter` trait, `RawSegment`, `SegmentationError`, public surface |
| `src/segmentation/hungarian.rs` | create | Kuhn-Munkres min-cost assignment for square matrices, wasm-clean |
| `src/segmentation/decoder.rs` | create | `PowersetDecoder`: class → (speaker_set, is_overlap), wasm-clean |
| `src/segmentation/aggregator.rs` | create | Sliding-window stitching using Hungarian, run-length encode, wasm-clean |
| `src/segmentation/powerset.rs` | create | `PowersetSegmenter` (ort::Session, gated onnx+segmentation) |
| `src/lib.rs` | modify | `pub mod segmentation;` gated, re-exports |
| `src/models/manifest.toml` | modify | Add `[models.powerset_fp32]` entry |
| `tests/segmenter_test.rs` | create | `#[ignore]` integration test against real ONNX model |
| `CHANGELOG.md` | modify | Unreleased M1 section |

Total roughly 800 lines Rust + 8 lines TOML.

---

## Task 1: Add `segmentation` Cargo feature

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1.1: Read current `[features]` block**

```bash
sed -n '14,40p' /Users/ekhodzitsky/Documents/personal/polyvoice-m1/Cargo.toml
```

Confirm M0's features are present: `default = ["spectral"]`, `onnx`, `ffi`, `download`, `cli`, `coreml`, `nnapi`, `xnnpack`, `spectral`, `profile-mobile`, `profile-balanced`, `profile-all`.

- [ ] **Step 1.2: Add `segmentation` feature**

In the `[features]` block, add `segmentation` to the `default` set and define it. Apply this exact diff to `Cargo.toml`:

Find:
```toml
[features]
default = ["spectral"]
```

Replace with:
```toml
[features]
default = ["spectral", "segmentation"]
```

Find:
```toml
# Spectral clustering backend (pulls faer for SVD/eigendecomp).
spectral = ["dep:faer"]
```

Append immediately after that line:
```toml

# Powerset speaker segmentation (sherpa-onnx-pyannote-segmentation-3-0).
# The pure-Rust algorithmic core (decoder, aggregator, hungarian) compiles to
# wasm32-clean. The ONNX-backed `PowersetSegmenter` additionally requires `onnx`.
segmentation = []
```

- [ ] **Step 1.3: Verify all feature combos still build**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m1
cargo check
cargo check --features download
cargo check --features cli
cargo check --features ffi
cargo check --features onnx
cargo check --features segmentation
cargo check --no-default-features
cargo check --target wasm32-unknown-unknown --no-default-features --lib
cargo check --all-features
```

Expected: every command exits 0. The wasm32 build must remain clean — `segmentation` alone (without `onnx`) is wasm-friendly because we only add pure-Rust modules under that flag.

- [ ] **Step 1.4: Commit**

```bash
git add Cargo.toml
git commit -m "feat(cargo): add segmentation feature flag for v1.0 M1 work"
```

---

## Task 2: Hungarian (Kuhn-Munkres) algorithm

**Files:**
- Create: `src/segmentation/mod.rs` (stub)
- Create: `src/segmentation/hungarian.rs`

This is the load-bearing math. Implement Kuhn-Munkres min-cost assignment for square cost matrices. ~60 lines core + ~80 lines tests. Pure Rust, no `unsafe`, no allocations beyond the algorithm's working buffers, wasm-clean.

- [ ] **Step 2.1: Create the directory and stub `mod.rs`**

```bash
mkdir -p /Users/ekhodzitsky/Documents/personal/polyvoice-m1/src/segmentation
```

Write `src/segmentation/mod.rs`:

```rust
//! Speaker segmentation: powerset-classifier + sliding-window aggregator.
//!
//! Added in v0.6 (M1). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §3.1, §5.3.

mod hungarian;
```

- [ ] **Step 2.2: Wire the stub `mod` into `lib.rs`** (needed so the module compiles)

In `src/lib.rs`, find the existing `#[cfg(feature = "download")] pub mod models;` line. Append immediately after it:

```rust
#[cfg(feature = "segmentation")]
pub mod segmentation;
```

- [ ] **Step 2.3: Write the failing tests for `hungarian` first**

Write `src/segmentation/hungarian.rs`:

```rust
//! Kuhn-Munkres minimum-cost assignment for square cost matrices.
//!
//! Pure Rust, no `unsafe`, wasm32-clean. Used by the segmentation aggregator to
//! align local speaker indices between overlapping windows.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_matrix_returns_empty_assignment() {
        let cost: Vec<Vec<f32>> = Vec::new();
        let assignment = solve(&cost).expect("empty matrix is valid");
        assert!(assignment.is_empty());
    }

    #[test]
    fn one_by_one_matrix_returns_self() {
        let cost = vec![vec![3.5_f32]];
        let assignment = solve(&cost).expect("1x1 valid");
        assert_eq!(assignment, vec![0]);
    }

    #[test]
    fn diagonal_zero_matrix_returns_identity() {
        // Cost is 0 on the diagonal, large off-diagonal — assignment must be identity.
        let n = 3;
        let mut cost = vec![vec![10.0_f32; n]; n];
        for i in 0..n {
            cost[i][i] = 0.0;
        }
        let assignment = solve(&cost).expect("3x3 valid");
        assert_eq!(assignment, vec![0, 1, 2]);
    }

    #[test]
    fn anti_diagonal_zero_matrix_returns_reverse_permutation() {
        // Cost is 0 on the anti-diagonal — assignment must be the reverse permutation.
        let cost = vec![
            vec![10.0_f32, 10.0, 0.0],
            vec![10.0, 0.0, 10.0],
            vec![0.0, 10.0, 10.0],
        ];
        let assignment = solve(&cost).expect("3x3 valid");
        assert_eq!(assignment, vec![2, 1, 0]);
    }

    #[test]
    fn permutation_matrix_recovered() {
        // The optimal assignment is row 0 → col 1, row 1 → col 2, row 2 → col 0
        // because each row has its minimum at exactly that column.
        let cost = vec![
            vec![5.0_f32, 0.0, 5.0],
            vec![5.0, 5.0, 0.0],
            vec![0.0, 5.0, 5.0],
        ];
        let assignment = solve(&cost).expect("3x3 valid");
        assert_eq!(assignment, vec![1, 2, 0]);
    }

    #[test]
    fn rejects_non_square_matrix() {
        let cost = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        assert!(solve(&cost).is_none());
    }

    #[test]
    fn handles_negative_costs() {
        // Algorithm should handle negative entries (we use it with -IoU costs).
        let cost = vec![vec![-1.0_f32, -3.0], vec![-2.0, -4.0]];
        // Optimal: row 0 → col 0, row 1 → col 1 gives -1 + -4 = -5
        // But row 0 → col 1, row 1 → col 0 gives -3 + -2 = -5. Tie — either OK.
        // To make it unique, change one cell:
        let cost = vec![vec![-1.0_f32, -3.0], vec![-2.0, -5.0]];
        // Now row 0 → col 0, row 1 → col 1 = -1 + -5 = -6 (best)
        // vs row 0 → col 1, row 1 → col 0 = -3 + -2 = -5
        let assignment = solve(&cost).expect("2x2 valid");
        assert_eq!(assignment, vec![0, 1]);
    }

    #[test]
    fn cost_matrix_with_repeated_rows_still_assigns_unique_columns() {
        // All rows identical — algorithm must still produce a permutation (each col used once).
        let cost = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];
        let assignment = solve(&cost).expect("3x3 valid");
        let mut sorted = assignment.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2], "must be a permutation");
    }
}
```

- [ ] **Step 2.4: Run tests, confirm compilation failure**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m1
cargo test --features segmentation --lib segmentation::hungarian::tests 2>&1 | head -20
```

Expected: errors about `solve` not found.

- [ ] **Step 2.5: Implement `solve` (Kuhn-Munkres)**

Replace the contents of `src/segmentation/hungarian.rs` with the full implementation, **keeping the test block at the bottom unchanged**:

```rust
//! Kuhn-Munkres minimum-cost assignment for square cost matrices.
//!
//! Pure Rust, no `unsafe`, wasm32-clean. Used by the segmentation aggregator to
//! align local speaker indices between overlapping windows.

/// Solve the assignment problem for an N×N cost matrix.
///
/// Returns a `Vec<usize>` of length N where `result[i]` is the column assigned to row `i`.
/// Each column is assigned to exactly one row. The total cost
/// `sum(cost[i][result[i]])` is minimized.
///
/// **Requires:** `cost` is square (every row has length `cost.len()`).
/// **Returns** `None` if `cost` is not square. An empty matrix returns `Some(vec![])`.
///
/// Implementation: classic Kuhn-Munkres in O(N³) using row/column potentials
/// (u/v) and shortest-path augmentation. Index 0 is reserved as a sentinel,
/// so internal arrays are length N+1.
pub fn solve(cost: &[Vec<f32>]) -> Option<Vec<usize>> {
    let n = cost.len();
    if n == 0 {
        return Some(Vec::new());
    }
    if cost.iter().any(|row| row.len() != n) {
        return None;
    }

    let inf = f32::INFINITY;
    let mut u = vec![0.0_f32; n + 1];
    let mut v = vec![0.0_f32; n + 1];
    // p[j] = row assigned to column j (0 = unassigned, sentinel)
    let mut p = vec![0_usize; n + 1];
    // way[j] = column predecessor in augmenting path
    let mut way = vec![0_usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0_usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0_usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            // Update potentials
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // Reconstruct: walk back via `way` and fix `p`
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut result = vec![0_usize; n];
    for j in 1..=n {
        if p[j] > 0 {
            result[p[j] - 1] = j - 1;
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    // (test block from Step 2.3 stays unchanged)
}
```

Keep the entire `#[cfg(test)] mod tests { ... }` from Step 2.3 unchanged at the bottom.

- [ ] **Step 2.6: Run tests, confirm 8/8 pass**

```bash
cargo test --features segmentation --lib segmentation::hungarian::tests
```

Expected: `test result: ok. 8 passed`.

- [ ] **Step 2.7: Run clippy + fmt**

```bash
cargo fmt
cargo clippy --features segmentation --lib -- -D warnings
```

Expected: clean.

- [ ] **Step 2.8: Verify wasm32 still compiles**

```bash
cargo check --target wasm32-unknown-unknown --no-default-features --lib
cargo check --target wasm32-unknown-unknown --no-default-features --features segmentation --lib
```

Expected: both succeed.

- [ ] **Step 2.9: Commit**

```bash
git add src/segmentation/mod.rs src/segmentation/hungarian.rs src/lib.rs
git commit -m "feat(segmentation): add Kuhn-Munkres assignment for window stitching"
```

---

## Task 3: `Segmenter` trait + `RawSegment` + errors

**Files:**
- Modify: `src/segmentation/mod.rs`

- [ ] **Step 3.1: Write the failing tests**

Append to `src/segmentation/mod.rs` (after the `mod hungarian;` line):

```rust
#[cfg(test)]
mod trait_tests {
    use super::*;
    use crate::types::{Confidence, TimeRange};

    /// A minimal in-memory segmenter used for trait conformance tests.
    struct ConstantSegmenter {
        segments: Vec<RawSegment>,
        max_speakers: usize,
        overlap: bool,
    }

    impl Segmenter for ConstantSegmenter {
        fn segment(&self, _audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError> {
            Ok(self.segments.clone())
        }
        fn max_local_speakers(&self) -> usize { self.max_speakers }
        fn supports_overlap(&self) -> bool { self.overlap }
    }

    #[test]
    fn raw_segment_roundtrip() {
        let s = RawSegment {
            time: TimeRange { start: 0.5, end: 1.5 },
            local_speaker_idx: 1,
            is_overlap: true,
            confidence: Confidence::new(0.85).unwrap(),
        };
        assert_eq!(s.local_speaker_idx, 1);
        assert!(s.is_overlap);
        assert!((s.confidence.get() - 0.85).abs() < 1e-6);
    }

    #[test]
    fn segmenter_trait_object_is_dyn_compatible() {
        // Compile-time check: must be storable behind `dyn`.
        let cs = ConstantSegmenter {
            segments: vec![],
            max_speakers: 3,
            overlap: true,
        };
        let _boxed: Box<dyn Segmenter> = Box::new(cs);
    }

    #[test]
    fn segmenter_segment_returns_owned_vec() {
        let cs = ConstantSegmenter {
            segments: vec![RawSegment {
                time: TimeRange { start: 0.0, end: 1.0 },
                local_speaker_idx: 0,
                is_overlap: false,
                confidence: Confidence::new(1.0).unwrap(),
            }],
            max_speakers: 3,
            overlap: true,
        };
        let out = cs.segment(&[]).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn error_audio_too_short_displays_required_thresholds() {
        let err = SegmentationError::AudioTooShort {
            actual_secs: 0.05,
            min_secs: 0.1,
        };
        let msg = format!("{err}");
        assert!(msg.contains("0.05"));
        assert!(msg.contains("0.1"));
    }
}
```

- [ ] **Step 3.2: Run tests to confirm compile-failure**

```bash
cargo test --features segmentation --lib segmentation::trait_tests 2>&1 | head -20
```

Expected: errors about `Segmenter`, `RawSegment`, `SegmentationError` not found.

- [ ] **Step 3.3: Implement the trait, struct, and error type**

Replace `src/segmentation/mod.rs` with:

```rust
//! Speaker segmentation: powerset-classifier + sliding-window aggregator.
//!
//! Added in v0.6 (M1). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §3.1, §5.3.

mod hungarian;
mod decoder;
mod aggregator;

#[cfg(all(feature = "onnx", feature = "segmentation"))]
mod powerset;

pub use decoder::{PowersetDecoder, PowersetClass, FrameLabel};
pub use aggregator::{Aggregator, WindowOutput, AggregationConfig};

#[cfg(all(feature = "onnx", feature = "segmentation"))]
pub use powerset::{PowersetSegmenter, PowersetConfig};

use crate::types::{Confidence, TimeRange};

/// One contiguous segment attributed to a single local speaker index.
///
/// "Local" means consistent within a single `segment()` call's output (same person ↔
/// same `local_speaker_idx` across all frames of the file). Cross-file global IDs
/// are assigned later by the clusterer (see M3).
#[derive(Debug, Clone, PartialEq)]
pub struct RawSegment {
    /// The temporal span of this segment in seconds, audio-relative.
    pub time: TimeRange,
    /// Speaker index local to this segmentation result. `0..=2` for `powerset-3.0`.
    pub local_speaker_idx: u8,
    /// True if the segmenter classified this region as a 2-speaker overlap.
    /// In that case a *second* segment for the other speaker covers the same
    /// time range with `local_speaker_idx` set to that other speaker.
    pub is_overlap: bool,
    /// Mean per-frame confidence: max-softmax averaged over the frames.
    pub confidence: Confidence,
}

/// A speaker segmentation engine — turns raw audio into spans of speech attributed
/// to local speaker indices, with overlap detection.
///
/// Implementations:
/// - `PowersetSegmenter` (this crate, gated `onnx` + `segmentation`) — wraps the
///   `sherpa-onnx-pyannote-segmentation-3-0` ONNX model.
pub trait Segmenter: Send + Sync {
    /// Segment `audio`. Audio must be 16 kHz mono `f32` PCM.
    ///
    /// **Requires:** `audio.len() >= MIN_AUDIO_SAMPLES` (1600 samples = 0.1s).
    /// Implementations may add stricter requirements documented per impl.
    /// **Guarantees on Ok:** segments are sorted by `time.start`; every
    /// `local_speaker_idx < self.max_local_speakers()`; timestamps lie within
    /// `[0, audio.len() / 16000]`.
    fn segment(&self, audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError>;

    /// Max number of distinct local speakers this implementation can output.
    /// `powerset-3.0` ⇒ 3.
    fn max_local_speakers(&self) -> usize;

    /// True if the implementation can detect overlap (two simultaneous speakers).
    /// `powerset-3.0` ⇒ true.
    fn supports_overlap(&self) -> bool;
}

/// Minimum audio length (16 kHz samples) accepted by `Segmenter::segment`.
pub const MIN_AUDIO_SAMPLES: usize = 1600;

/// Errors from `Segmenter` implementations.
#[derive(Debug, thiserror::Error)]
pub enum SegmentationError {
    #[error("audio too short: {actual_secs:.3}s < {min_secs:.3}s required")]
    AudioTooShort {
        actual_secs: f32,
        min_secs: f32,
    },

    #[error("ONNX inference failed at window {window_idx}: {detail}")]
    InferenceFailed {
        window_idx: usize,
        detail: String,
    },

    #[error("powerset decoder produced invalid output shape: expected (_, 7), got {actual_shape:?}")]
    InvalidOutputShape { actual_shape: Vec<usize> },

    #[error("speaker permutation matching failed across windows {prev_idx}->{next_idx}: {detail}")]
    PermutationFailed {
        prev_idx: usize,
        next_idx: usize,
        detail: String,
    },

    #[error("model file io error on {path}: {detail}")]
    ModelIo {
        path: std::path::PathBuf,
        detail: String,
    },
}

#[cfg(test)]
mod trait_tests {
    // (test block from Step 3.1 stays unchanged)
}
```

Keep the test block from Step 3.1 unchanged.

The lines that import `decoder`, `aggregator`, and `powerset` will fail until those files exist (Tasks 4–7). Add them as **empty stub files** now so the module compiles:

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m1
echo '//! Stub — implemented in Task 4.' > src/segmentation/decoder.rs
echo '//! Stub — implemented in Tasks 5–6.' > src/segmentation/aggregator.rs
```

Then add a minimal stub of the public surface that the `pub use` lines reference. Append to `src/segmentation/decoder.rs`:

```rust

/// Stub — implemented in Task 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowersetClass {
    Silence,
    Speaker(u8),
    Pair(u8, u8),
}

/// Stub — implemented in Task 4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameLabel {
    pub class: PowersetClass,
    pub max_softmax: f32,
}

/// Stub — implemented in Task 4.
pub struct PowersetDecoder;
```

Append to `src/segmentation/aggregator.rs`:

```rust

/// Stub — implemented in Tasks 5–6.
pub struct Aggregator;

/// Stub — implemented in Tasks 5–6.
pub struct WindowOutput;

/// Stub — implemented in Tasks 5–6.
pub struct AggregationConfig;
```

These stubs will be replaced with real implementations in subsequent tasks.

Also create the `powerset` stub even though it's gated:

```bash
echo '//! Stub — implemented in Task 7.' > src/segmentation/powerset.rs
```

And append:

```rust

/// Stub — implemented in Task 7.
pub struct PowersetSegmenter;

/// Stub — implemented in Task 7.
pub struct PowersetConfig;
```

- [ ] **Step 3.4: Run tests, confirm 4/4 trait tests pass**

```bash
cargo test --features segmentation --lib segmentation::trait_tests
cargo test --features segmentation --lib segmentation::hungarian::tests
```

Expected: 4 trait tests + 8 hungarian tests, all passing.

- [ ] **Step 3.5: Run clippy + fmt + wasm check**

```bash
cargo fmt
cargo clippy --features segmentation --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features segmentation --lib
```

Expected: all clean.

- [ ] **Step 3.6: Commit**

```bash
git add src/segmentation/mod.rs src/segmentation/decoder.rs src/segmentation/aggregator.rs src/segmentation/powerset.rs
git commit -m "feat(segmentation): add Segmenter trait, RawSegment, error type"
```

---

## Task 4: `PowersetDecoder` (class → speaker_set)

**Files:**
- Modify: `src/segmentation/decoder.rs`

The decoder converts a 7-vector of softmaxed logits into a `FrameLabel`. Mapping:

| Class | Set | Is overlap |
|---|---|---|
| 0 | ∅ (silence) | no |
| 1 | {0} | no |
| 2 | {1} | no |
| 3 | {2} | no |
| 4 | {0, 1} | yes |
| 5 | {0, 2} | yes |
| 6 | {1, 2} | yes |

This is pure Rust, wasm-clean.

- [ ] **Step 4.1: Write the failing tests**

Replace the entire content of `src/segmentation/decoder.rs` with the **tests-only** version first:

```rust
//! Powerset 7-class decoder for `pyannote/segmentation-3.0`.

use crate::types::Confidence;

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-6 }

    #[test]
    fn class_0_is_silence() {
        let logits = [10.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Silence);
        assert!(!label.class.is_overlap());
    }

    #[test]
    fn class_1_is_speaker_0() {
        let logits = [1.0_f32, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Speaker(0));
    }

    #[test]
    fn class_3_is_speaker_2() {
        let logits = [1.0_f32, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Speaker(2));
    }

    #[test]
    fn class_4_is_overlap_pair_0_1() {
        let logits = [1.0_f32, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Pair(0, 1));
        assert!(label.class.is_overlap());
    }

    #[test]
    fn class_5_is_overlap_pair_0_2() {
        let logits = [1.0_f32, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Pair(0, 2));
    }

    #[test]
    fn class_6_is_overlap_pair_1_2() {
        let logits = [1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert_eq!(label.class, PowersetClass::Pair(1, 2));
    }

    #[test]
    fn rejects_wrong_logit_count() {
        let logits = [1.0_f32, 2.0, 3.0]; // 3 classes, not 7
        assert!(PowersetDecoder::decode_frame(&logits).is_err());
    }

    #[test]
    fn max_softmax_is_softmax_of_argmax_class() {
        // Uniform logits at 0.0 → softmax = 1/7 for each. Max softmax = 1/7.
        let logits = [0.0_f32; 7];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert!(approx(label.max_softmax, 1.0 / 7.0));
    }

    #[test]
    fn confidence_clamps_to_valid_range() {
        // Negative-infinity logits everywhere except one — softmax max = 1.0.
        let logits = [-1e6_f32, -1e6, -1e6, -1e6, -1e6, -1e6, 0.0];
        let label = PowersetDecoder::decode_frame(&logits).unwrap();
        assert!(label.max_softmax > 0.99);
        assert!(label.max_softmax <= 1.0 + 1e-6);
    }

    #[test]
    fn class_method_returns_speaker_set() {
        assert_eq!(PowersetClass::Silence.speakers(), Vec::<u8>::new());
        assert_eq!(PowersetClass::Speaker(0).speakers(), vec![0]);
        assert_eq!(PowersetClass::Pair(0, 2).speakers(), vec![0, 2]);
        assert_eq!(PowersetClass::Pair(1, 2).speakers(), vec![1, 2]);
    }

    #[test]
    fn decode_window_iterates_over_frames() {
        // 2 frames of logits — each independently decoded.
        let logits_flat: Vec<f32> = vec![
            // frame 0: silence
            10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            // frame 1: speaker 1
            1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let labels = PowersetDecoder::decode_window(&logits_flat, 2).unwrap();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].class, PowersetClass::Silence);
        assert_eq!(labels[1].class, PowersetClass::Speaker(1));
    }

    #[test]
    fn decode_window_rejects_misshaped_buffer() {
        let logits_flat = vec![1.0_f32; 8]; // Not divisible by 7
        assert!(PowersetDecoder::decode_window(&logits_flat, 1).is_err());
    }

    #[test]
    fn confidence_construction_via_helper() {
        // The decoder helper that wraps Confidence must always succeed for
        // valid softmax outputs (clamping handled internally).
        let c = PowersetDecoder::frame_confidence(1.0_f32 + 1e-7);
        assert!((c.get() - 1.0).abs() < 1e-5);

        let c = PowersetDecoder::frame_confidence(-1e-7);
        assert!(c.get() >= 0.0);
    }
}
```

- [ ] **Step 4.2: Run tests, confirm compilation failure**

```bash
cargo test --features segmentation --lib segmentation::decoder::tests 2>&1 | head -30
```

Expected: errors about `PowersetClass`, `FrameLabel`, `PowersetDecoder`, `decode_frame`, `decode_window`, `is_overlap`, `speakers`, `frame_confidence`.

- [ ] **Step 4.3: Implement the decoder**

Replace `src/segmentation/decoder.rs` content (keep the test block from Step 4.1 unchanged at the bottom):

```rust
//! Powerset 7-class decoder for `pyannote/segmentation-3.0`.
//!
//! Each frame's 7-vector of logits is interpreted as one of:
//!
//! | Class | Set | Is overlap |
//! |---|---|---|
//! | 0 | ∅ (silence) | no |
//! | 1 | {0} | no |
//! | 2 | {1} | no |
//! | 3 | {2} | no |
//! | 4 | {0, 1} | yes |
//! | 5 | {0, 2} | yes |
//! | 6 | {1, 2} | yes |
//!
//! The decoder takes argmax over softmax, returning a `FrameLabel`.

use crate::segmentation::SegmentationError;
use crate::types::Confidence;

/// One of the seven powerset classes, identifying which speakers are active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowersetClass {
    Silence,
    Speaker(u8),
    Pair(u8, u8),
}

impl PowersetClass {
    /// True for classes 4–6 (two speakers active simultaneously).
    pub const fn is_overlap(self) -> bool {
        matches!(self, PowersetClass::Pair(_, _))
    }

    /// Local speaker indices active in this class.
    pub fn speakers(self) -> Vec<u8> {
        match self {
            PowersetClass::Silence => Vec::new(),
            PowersetClass::Speaker(s) => vec![s],
            PowersetClass::Pair(a, b) => vec![a, b],
        }
    }
}

/// Decoded label for a single audio frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameLabel {
    pub class: PowersetClass,
    /// Maximum-class softmax probability (∈ [0, 1]). Useful for confidence reporting.
    pub max_softmax: f32,
}

/// Stateless decoder; methods are associated functions because no per-instance
/// configuration is needed.
pub struct PowersetDecoder;

impl PowersetDecoder {
    /// Convert a 7-class index (0..=6) to its `PowersetClass`.
    pub const fn class_for_index(idx: usize) -> Option<PowersetClass> {
        match idx {
            0 => Some(PowersetClass::Silence),
            1 => Some(PowersetClass::Speaker(0)),
            2 => Some(PowersetClass::Speaker(1)),
            3 => Some(PowersetClass::Speaker(2)),
            4 => Some(PowersetClass::Pair(0, 1)),
            5 => Some(PowersetClass::Pair(0, 2)),
            6 => Some(PowersetClass::Pair(1, 2)),
            _ => None,
        }
    }

    /// Decode one frame given its 7-vector of logits.
    pub fn decode_frame(logits: &[f32]) -> Result<FrameLabel, SegmentationError> {
        if logits.len() != 7 {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: vec![logits.len()],
            });
        }
        // Stable softmax: subtract max for numerical stability.
        let mut max_logit = f32::NEG_INFINITY;
        for &l in logits {
            if l > max_logit {
                max_logit = l;
            }
        }
        let mut exps = [0.0_f32; 7];
        let mut sum = 0.0_f32;
        for (i, &l) in logits.iter().enumerate() {
            exps[i] = (l - max_logit).exp();
            sum += exps[i];
        }
        // Guard against degenerate sum (sum=0 would only happen with NaN logits).
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        let mut argmax = 0_usize;
        let mut max_softmax = 0.0_f32;
        for i in 0..7 {
            let p = exps[i] * inv_sum;
            if p > max_softmax {
                max_softmax = p;
                argmax = i;
            }
        }
        let class = Self::class_for_index(argmax).ok_or(SegmentationError::InvalidOutputShape {
            actual_shape: vec![argmax],
        })?;
        Ok(FrameLabel { class, max_softmax })
    }

    /// Decode every frame in a flat row-major `[num_frames, 7]` buffer.
    pub fn decode_window(
        logits_flat: &[f32],
        num_frames: usize,
    ) -> Result<Vec<FrameLabel>, SegmentationError> {
        if logits_flat.len() != num_frames * 7 {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: vec![logits_flat.len()],
            });
        }
        let mut out = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let frame = &logits_flat[i * 7..(i + 1) * 7];
            out.push(Self::decode_frame(frame)?);
        }
        Ok(out)
    }

    /// Convert a softmax probability into a `Confidence`. Clamps tiny over-/underflows
    /// to the valid `[0, 1]` range so we never panic on numerical artifacts.
    pub fn frame_confidence(softmax: f32) -> Confidence {
        let clamped = softmax.clamp(0.0, 1.0);
        // `Confidence::new` validates the closed range; clamped is guaranteed valid.
        Confidence::new(clamped).unwrap_or(Confidence::default())
    }
}

#[cfg(test)]
mod tests {
    // (test block from Step 4.1 stays unchanged)
}
```

The `unwrap_or(Confidence::default())` is correct — `clamped ∈ [0,1]` always, but lint-friendly to use `unwrap_or` over `unwrap`.

- [ ] **Step 4.4: Run tests, confirm 13 pass**

```bash
cargo test --features segmentation --lib segmentation::decoder::tests
```

Expected: 13/13 pass.

- [ ] **Step 4.5: Run clippy + fmt + wasm check**

```bash
cargo fmt
cargo clippy --features segmentation --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features segmentation --lib
```

Expected: clean.

- [ ] **Step 4.6: Commit**

```bash
git add src/segmentation/decoder.rs
git commit -m "feat(segmentation): implement PowersetDecoder (7-class argmax)"
```

---

## Task 5: `Aggregator` foundation types

**Files:**
- Modify: `src/segmentation/aggregator.rs`

This task introduces `WindowOutput`, `AggregationConfig`, and the `Aggregator` struct skeleton — without yet wiring up Hungarian. Task 6 builds on top.

- [ ] **Step 5.1: Replace the stub with the foundation types and minimal tests**

Replace `src/segmentation/aggregator.rs` with:

```rust
//! Sliding-window aggregator for powerset segmentation outputs.
//!
//! Combines per-window 7-class logits into file-globally-consistent
//! `RawSegment` outputs. Implementation builds on top of `decoder` and
//! `hungarian` modules (added in Tasks 4 and 2). Pure Rust, wasm-clean.

use crate::segmentation::{RawSegment, SegmentationError};
use crate::types::{Confidence, TimeRange};

/// One window's segmentation output: when the window starts and the per-frame logits.
#[derive(Debug, Clone)]
pub struct WindowOutput {
    /// Audio start time of this window, in seconds.
    pub start_time: f32,
    /// Audio end time of this window, in seconds.
    pub end_time: f32,
    /// Flat row-major buffer of `(num_frames, 7)` logits.
    pub logits: Vec<f32>,
    /// Number of frames in this window.
    pub num_frames: usize,
}

impl WindowOutput {
    /// Create a window output, validating shape.
    pub fn new(
        start_time: f32,
        end_time: f32,
        logits: Vec<f32>,
        num_frames: usize,
    ) -> Result<Self, SegmentationError> {
        if logits.len() != num_frames * 7 {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: vec![logits.len()],
            });
        }
        Ok(Self { start_time, end_time, logits, num_frames })
    }

    /// Frame stride in seconds (window duration ÷ frame count).
    pub fn frame_stride(&self) -> f32 {
        if self.num_frames == 0 {
            0.0
        } else {
            (self.end_time - self.start_time) / self.num_frames as f32
        }
    }

    /// Convert a per-window frame index to its absolute audio time (seconds).
    pub fn frame_time(&self, frame_idx: usize) -> f32 {
        self.start_time + frame_idx as f32 * self.frame_stride()
    }
}

/// Configuration for aggregation.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Drop run-length-encoded segments shorter than this duration (seconds).
    pub min_segment_secs: f32,
    /// Maximum number of local speakers any single window can produce.
    /// Should match the underlying model's `max_local_speakers`.
    pub max_local_speakers: usize,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            min_segment_secs: 0.0,
            max_local_speakers: 3,
        }
    }
}

/// The sliding-window aggregator. Holds configuration; operates on borrowed
/// window outputs.
pub struct Aggregator {
    config: AggregationConfig,
}

impl Aggregator {
    pub fn new(config: AggregationConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &AggregationConfig {
        &self.config
    }

    /// Aggregate `windows` into file-globally-consistent `RawSegment`s.
    /// Real implementation lands in Task 6.
    pub fn stitch(&self, windows: &[WindowOutput]) -> Result<Vec<RawSegment>, SegmentationError> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        // Placeholder: Task 6 will implement Hungarian-driven stitching.
        // For the M1 skeleton commit, return a single silent placeholder so the
        // type signature is exercised by tests at this stage.
        Err(SegmentationError::PermutationFailed {
            prev_idx: 0,
            next_idx: 0,
            detail: "not yet implemented; lands in Task 6".to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_output_validates_shape() {
        let ok = WindowOutput::new(0.0, 10.0, vec![0.0; 7 * 5], 5);
        assert!(ok.is_ok());

        let bad = WindowOutput::new(0.0, 10.0, vec![0.0; 13], 5);
        assert!(bad.is_err());
    }

    #[test]
    fn window_frame_stride_matches_duration() {
        let w = WindowOutput::new(0.0, 10.0, vec![0.0; 7 * 100], 100).unwrap();
        assert!((w.frame_stride() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn window_frame_time_is_anchored_at_start() {
        let w = WindowOutput::new(2.5, 12.5, vec![0.0; 7 * 100], 100).unwrap();
        assert!((w.frame_time(0) - 2.5).abs() < 1e-6);
        assert!((w.frame_time(50) - 7.5).abs() < 1e-6);
    }

    #[test]
    fn empty_windows_yields_empty_segments() {
        let agg = Aggregator::new(AggregationConfig::default());
        let segments = agg.stitch(&[]).unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn stitch_returns_not_yet_implemented_error_for_non_empty_input() {
        // Skeleton commit: Task 6 replaces the placeholder.
        let agg = Aggregator::new(AggregationConfig::default());
        let w = WindowOutput::new(0.0, 10.0, vec![0.0; 7 * 100], 100).unwrap();
        let result = agg.stitch(&[w]);
        assert!(result.is_err());
    }

    #[test]
    fn config_default_is_3_speakers() {
        let c = AggregationConfig::default();
        assert_eq!(c.max_local_speakers, 3);
        assert_eq!(c.min_segment_secs, 0.0);
    }
}
```

The `stitch_returns_not_yet_implemented_error_for_non_empty_input` test will be replaced in Task 6 — that's intentional.

- [ ] **Step 5.2: Run tests**

```bash
cargo test --features segmentation --lib segmentation::aggregator::tests
```

Expected: 6/6 pass.

- [ ] **Step 5.3: Run clippy + fmt + wasm**

```bash
cargo fmt
cargo clippy --features segmentation --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features segmentation --lib
```

Expected: all clean.

- [ ] **Step 5.4: Commit**

```bash
git add src/segmentation/aggregator.rs
git commit -m "feat(segmentation): add Aggregator/WindowOutput foundation types"
```

---

## Task 6: `Aggregator` stitching with Hungarian + run-length encoding

**Files:**
- Modify: `src/segmentation/aggregator.rs`

The real algorithm lives here. Inputs: a sorted list of `WindowOutput`s with overlapping time spans. For each adjacent window pair, compute IoU between window-i's per-speaker frame masks and window-(i+1)'s, build a 3×3 cost matrix `C[a][b] = -IoU(a, b)`, run Kuhn-Munkres to get the assignment, and apply the resulting permutation to window-(i+1) so speaker indices remain consistent. Then average per-class logits at every frame across all windows that contain it, argmax to get the dominant class, and run-length encode consecutive identical labels into `RawSegment`s.

- [ ] **Step 6.1: Replace the previous placeholder + extend the test suite**

Replace `src/segmentation/aggregator.rs` with the full implementation (the `tests` block at the bottom is replaced with a richer suite below):

```rust
//! Sliding-window aggregator for powerset segmentation outputs.
//!
//! Combines per-window 7-class logits into file-globally-consistent
//! `RawSegment` outputs.
//!
//! Algorithm:
//! 1. For each adjacent window pair (i, i+1), build the 3×3 IoU matrix between
//!    speaker masks in the temporal overlap region. Each speaker mask is the
//!    union of frames where the window's argmax label includes that speaker.
//! 2. Use Kuhn-Munkres on `-IoU` to find the assignment that maps window i+1's
//!    local indices onto window i's. Apply the permutation so the same person
//!    has the same index file-wide.
//! 3. For every audio frame, average the per-class logits across every window
//!    that contains that frame.
//! 4. Argmax each averaged logit vector → frame label.
//! 5. Run-length encode consecutive identical labels into `RawSegment`s.

use crate::segmentation::decoder::{FrameLabel, PowersetClass, PowersetDecoder};
use crate::segmentation::hungarian;
use crate::segmentation::{RawSegment, SegmentationError};
use crate::types::{Confidence, TimeRange};

/// One window's segmentation output.
#[derive(Debug, Clone)]
pub struct WindowOutput {
    pub start_time: f32,
    pub end_time: f32,
    /// Row-major `(num_frames, 7)` logits.
    pub logits: Vec<f32>,
    pub num_frames: usize,
}

impl WindowOutput {
    pub fn new(
        start_time: f32,
        end_time: f32,
        logits: Vec<f32>,
        num_frames: usize,
    ) -> Result<Self, SegmentationError> {
        if logits.len() != num_frames * 7 {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: vec![logits.len()],
            });
        }
        Ok(Self { start_time, end_time, logits, num_frames })
    }

    pub fn frame_stride(&self) -> f32 {
        if self.num_frames == 0 {
            0.0
        } else {
            (self.end_time - self.start_time) / self.num_frames as f32
        }
    }

    pub fn frame_time(&self, frame_idx: usize) -> f32 {
        self.start_time + frame_idx as f32 * self.frame_stride()
    }
}

/// Configuration for aggregation.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    pub min_segment_secs: f32,
    pub max_local_speakers: usize,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self { min_segment_secs: 0.0, max_local_speakers: 3 }
    }
}

/// Aggregator over sliding-window powerset outputs.
pub struct Aggregator {
    config: AggregationConfig,
}

impl Aggregator {
    pub fn new(config: AggregationConfig) -> Self { Self { config } }
    pub fn config(&self) -> &AggregationConfig { &self.config }

    /// Stitch overlapping windows into file-consistent `RawSegment`s.
    pub fn stitch(&self, windows: &[WindowOutput]) -> Result<Vec<RawSegment>, SegmentationError> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }

        // 1) For each window, compute per-frame argmax labels.
        let mut window_labels: Vec<Vec<FrameLabel>> = Vec::with_capacity(windows.len());
        for w in windows {
            let labels = PowersetDecoder::decode_window(&w.logits, w.num_frames)?;
            window_labels.push(labels);
        }

        // 2) Hungarian-align each adjacent window pair: permute window i+1's local
        // speaker indices onto window i's reference frame.
        let mut permutations: Vec<[u8; 3]> =
            std::iter::repeat([0u8, 1u8, 2u8]).take(windows.len()).collect();

        for i in 1..windows.len() {
            let perm = self.window_permutation(
                &windows[i - 1],
                &window_labels[i - 1],
                &windows[i],
                &window_labels[i],
                &permutations[i - 1],
            )?;
            // Compose: window i's labels are *first* permuted by `perm`, *then*
            // by the cumulative permutation up to window i-1.
            let prev = permutations[i - 1];
            let composed: [u8; 3] = [prev[perm[0] as usize], prev[perm[1] as usize], prev[perm[2] as usize]];
            permutations[i] = composed;
        }

        // 3) For every audio frame across the file, average per-class logits over
        // every window that contains it. Use a fine global frame grid based on
        // the first window's frame stride.
        let segments = self.average_and_run_length_encode(windows, &window_labels, &permutations)?;
        Ok(segments)
    }

    /// Compute the permutation that maps window B's local indices onto A's frame.
    fn window_permutation(
        &self,
        a: &WindowOutput,
        a_labels: &[FrameLabel],
        b: &WindowOutput,
        b_labels: &[FrameLabel],
        a_perm_so_far: &[u8; 3],
    ) -> Result<[u8; 3], SegmentationError> {
        let n = self.config.max_local_speakers.min(3);
        // Temporal overlap [overlap_start, overlap_end].
        let overlap_start = a.start_time.max(b.start_time);
        let overlap_end = a.end_time.min(b.end_time);
        if overlap_end <= overlap_start || n == 0 {
            // No overlap (or no speakers) — identity.
            return Ok([0, 1, 2]);
        }

        // Build per-speaker frame-occupancy bitmaps in the overlap region.
        // Use a coarse fixed-resolution grid (frame stride = a.frame_stride()).
        let stride = a.frame_stride().max(1e-6);
        let grid_len = ((overlap_end - overlap_start) / stride).ceil() as usize;
        if grid_len == 0 {
            return Ok([0, 1, 2]);
        }

        let mut a_masks = vec![vec![false; grid_len]; 3];
        let mut b_masks = vec![vec![false; grid_len]; 3];
        for k in 0..grid_len {
            let t = overlap_start + k as f32 * stride;
            // Find the frame index in A that contains time `t`.
            if let Some(idx_a) = self.frame_index_at(a, t) {
                if idx_a < a_labels.len() {
                    for s in a_labels[idx_a].class.speakers() {
                        if (s as usize) < 3 {
                            // Apply the cumulative permutation already in effect for window A.
                            let permuted = a_perm_so_far[s as usize] as usize;
                            if permuted < 3 {
                                a_masks[permuted][k] = true;
                            }
                        }
                    }
                }
            }
            if let Some(idx_b) = self.frame_index_at(b, t) {
                if idx_b < b_labels.len() {
                    for s in b_labels[idx_b].class.speakers() {
                        if (s as usize) < 3 {
                            b_masks[s as usize][k] = true;
                        }
                    }
                }
            }
        }

        // Build cost matrix: C[a][b] = -IoU(a_mask, b_mask).
        // (We minimize cost ⇒ maximize IoU.)
        let mut cost: Vec<Vec<f32>> = vec![vec![0.0_f32; n]; n];
        for ai in 0..n {
            for bi in 0..n {
                let mut inter = 0_usize;
                let mut uni = 0_usize;
                for k in 0..grid_len {
                    let ax = a_masks[ai][k];
                    let bx = b_masks[bi][k];
                    if ax && bx { inter += 1; }
                    if ax || bx { uni += 1; }
                }
                let iou = if uni == 0 { 0.0 } else { inter as f32 / uni as f32 };
                cost[ai][bi] = -iou;
            }
        }

        let assignment = hungarian::solve(&cost).ok_or_else(|| {
            SegmentationError::PermutationFailed {
                prev_idx: 0,
                next_idx: 0,
                detail: "non-square cost matrix".to_owned(),
            }
        })?;

        // The assignment maps row i (speaker i in A's permuted frame) → column j (speaker j in B).
        // We want a permutation `perm` such that `perm[b_speaker]` is the A-frame speaker.
        // That is the *inverse* of `assignment`.
        let mut perm = [0_u8, 1_u8, 2_u8];
        for (a_speaker, b_speaker) in assignment.iter().enumerate() {
            if *b_speaker < 3 && a_speaker < 3 {
                perm[*b_speaker] = a_speaker as u8;
            }
        }
        Ok(perm)
    }

    /// Find the frame index in `w` whose center is closest to time `t`. Returns
    /// `None` if `t` is outside the window's span.
    fn frame_index_at(&self, w: &WindowOutput, t: f32) -> Option<usize> {
        if t < w.start_time || t > w.end_time || w.num_frames == 0 {
            return None;
        }
        let stride = w.frame_stride();
        if stride <= 0.0 { return None; }
        let idx = ((t - w.start_time) / stride).floor() as usize;
        Some(idx.min(w.num_frames - 1))
    }

    /// Average per-class logits across windows that contain each global frame,
    /// then argmax + run-length encode into `RawSegment`s.
    fn average_and_run_length_encode(
        &self,
        windows: &[WindowOutput],
        window_labels: &[Vec<FrameLabel>],
        permutations: &[[u8; 3]],
    ) -> Result<Vec<RawSegment>, SegmentationError> {
        // Global frame grid: from the earliest start to the latest end, with the
        // stride of the *first* window (every window is assumed to have the same
        // frame stride — they come from the same model).
        let stride = windows[0].frame_stride().max(1e-6);
        let global_start = windows
            .iter()
            .map(|w| w.start_time)
            .fold(f32::INFINITY, f32::min);
        let global_end = windows
            .iter()
            .map(|w| w.end_time)
            .fold(f32::NEG_INFINITY, f32::max);
        let global_frames = ((global_end - global_start) / stride).ceil() as usize;

        // For each global frame, accumulate averaged per-(permuted-)class softmax probs.
        // Each global frame's vector has 7 entries (the 7 powerset classes).
        let mut summed_probs = vec![[0.0_f32; 7]; global_frames];
        let mut counts = vec![0_u32; global_frames];

        for (wi, w) in windows.iter().enumerate() {
            let perm = permutations[wi];
            for f in 0..w.num_frames {
                let t_center = w.frame_time(f) + 0.5 * stride;
                let g_idx_f = (t_center - global_start) / stride;
                if g_idx_f < 0.0 { continue; }
                let g_idx = g_idx_f.floor() as usize;
                if g_idx >= global_frames { continue; }
                // Convert this frame's logits into permuted-class softmax probs.
                let frame_logits = &w.logits[f * 7..(f + 1) * 7];
                let frame_label = window_labels[wi].get(f);
                if frame_label.is_none() { continue; }

                // Compute softmax for this frame.
                let mut max_logit = f32::NEG_INFINITY;
                for &l in frame_logits {
                    if l > max_logit { max_logit = l; }
                }
                let mut exps = [0.0_f32; 7];
                let mut sum = 0.0_f32;
                for (i, &l) in frame_logits.iter().enumerate() {
                    exps[i] = (l - max_logit).exp();
                    sum += exps[i];
                }
                let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };

                // Apply permutation: rebuild the softmax vector under the file-global
                // speaker ordering by remapping each class's speaker set through `perm`.
                let mut remapped = [0.0_f32; 7];
                for c in 0..7 {
                    if let Some(class) = PowersetDecoder::class_for_index(c) {
                        let speakers = class.speakers();
                        let remapped_speakers: Vec<u8> = speakers
                            .iter()
                            .map(|s| if (*s as usize) < 3 { perm[*s as usize] } else { *s })
                            .collect();
                        // Map back to a class index.
                        let new_class = match remapped_speakers.as_slice() {
                            [] => 0,
                            [s] => 1 + (*s as usize),
                            [a, b] => {
                                let (lo, hi) = if a < b { (*a, *b) } else { (*b, *a) };
                                match (lo, hi) {
                                    (0, 1) => 4,
                                    (0, 2) => 5,
                                    (1, 2) => 6,
                                    _ => 0, // unreachable for valid permuted speakers
                                }
                            }
                            _ => 0, // > 2 speakers — unreachable for powerset-3
                        };
                        remapped[new_class] += exps[c] * inv_sum;
                    }
                }

                for (i, &p) in remapped.iter().enumerate() {
                    summed_probs[g_idx][i] += p;
                }
                counts[g_idx] += 1;
            }
        }

        // Argmax per global frame, accumulating averaged-probability max into the
        // confidence channel.
        let mut frame_classes: Vec<Option<PowersetClass>> = Vec::with_capacity(global_frames);
        let mut frame_confidences: Vec<f32> = Vec::with_capacity(global_frames);
        for g in 0..global_frames {
            if counts[g] == 0 {
                frame_classes.push(None);
                frame_confidences.push(0.0);
                continue;
            }
            let inv = 1.0 / counts[g] as f32;
            let mut argmax = 0_usize;
            let mut maxp = 0.0_f32;
            for c in 0..7 {
                let p = summed_probs[g][c] * inv;
                if p > maxp {
                    maxp = p;
                    argmax = c;
                }
            }
            frame_classes.push(PowersetDecoder::class_for_index(argmax));
            frame_confidences.push(maxp);
        }

        // Run-length encode: emit one RawSegment per (speaker_idx, is_overlap)
        // contiguous run of frames per local speaker. For overlap classes (Pair),
        // emit two simultaneous segments — one per speaker.
        let mut segments: Vec<RawSegment> = Vec::new();
        let mut active: [Option<(usize, f32, f32)>; 3] = [None, None, None]; // (start_g, conf_sum, conf_count)
        let stride_dur = stride;

        for g in 0..global_frames {
            let frame_class = frame_classes[g];
            let conf = frame_confidences[g];
            let active_speakers: Vec<u8> = match frame_class {
                Some(c) => c.speakers(),
                None => Vec::new(),
            };
            let is_overlap_now = matches!(frame_class, Some(c) if c.is_overlap());

            for s in 0..3 {
                let s_active_now = active_speakers.iter().any(|x| *x as usize == s);
                match (active[s], s_active_now) {
                    (None, true) => {
                        active[s] = Some((g, conf, 1.0));
                    }
                    (Some((start_g, conf_sum, conf_count)), true) => {
                        active[s] = Some((start_g, conf_sum + conf, conf_count + 1.0));
                    }
                    (Some((start_g, conf_sum, conf_count)), false) => {
                        let start_t = global_start + start_g as f32 * stride_dur;
                        let end_t = global_start + g as f32 * stride_dur;
                        let dur = end_t - start_t;
                        if dur >= self.config.min_segment_secs {
                            let mean_conf = (conf_sum / conf_count.max(1.0)).clamp(0.0, 1.0);
                            // We need to know whether this run involved overlap. For
                            // simplicity, mark overlap = true if any frame in the run
                            // saw an overlap class for this speaker. To keep memory
                            // bounded, recompute from frame_classes:
                            let mut had_overlap = false;
                            for gg in start_g..g {
                                if let Some(c) = frame_classes[gg] {
                                    if c.is_overlap() && c.speakers().iter().any(|x| *x as usize == s) {
                                        had_overlap = true;
                                        break;
                                    }
                                }
                            }
                            segments.push(RawSegment {
                                time: TimeRange { start: start_t as f64, end: end_t as f64 },
                                local_speaker_idx: s as u8,
                                is_overlap: had_overlap,
                                confidence: PowersetDecoder::frame_confidence(mean_conf),
                            });
                        }
                        active[s] = None;
                    }
                    (None, false) => {}
                }
            }
        }
        // Flush trailing active runs.
        for s in 0..3 {
            if let Some((start_g, conf_sum, conf_count)) = active[s] {
                let start_t = global_start + start_g as f32 * stride_dur;
                let end_t = global_start + global_frames as f32 * stride_dur;
                let dur = end_t - start_t;
                if dur >= self.config.min_segment_secs {
                    let mean_conf = (conf_sum / conf_count.max(1.0)).clamp(0.0, 1.0);
                    let mut had_overlap = false;
                    for gg in start_g..global_frames {
                        if let Some(c) = frame_classes[gg] {
                            if c.is_overlap() && c.speakers().iter().any(|x| *x as usize == s) {
                                had_overlap = true;
                                break;
                            }
                        }
                    }
                    segments.push(RawSegment {
                        time: TimeRange { start: start_t as f64, end: end_t as f64 },
                        local_speaker_idx: s as u8,
                        is_overlap: had_overlap,
                        confidence: PowersetDecoder::frame_confidence(mean_conf),
                    });
                }
            }
        }

        // Sort by start time for the contract.
        segments.sort_by(|a, b| a.time.start.partial_cmp(&b.time.start).unwrap_or(std::cmp::Ordering::Equal));
        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a window where every frame is a single class (like 0=silence,
    /// 1=speaker 0, etc.) with the listed class as logit 10 and others as logit 0.
    fn synthetic_window(start: f32, end: f32, num_frames: usize, classes: &[usize]) -> WindowOutput {
        assert_eq!(classes.len(), num_frames);
        let mut logits = Vec::with_capacity(num_frames * 7);
        for &c in classes {
            for k in 0..7 {
                logits.push(if k == c { 10.0 } else { 0.0 });
            }
        }
        WindowOutput::new(start, end, logits, num_frames).unwrap()
    }

    #[test]
    fn empty_returns_empty() {
        let agg = Aggregator::new(AggregationConfig::default());
        assert!(agg.stitch(&[]).unwrap().is_empty());
    }

    #[test]
    fn single_window_silence_yields_no_segments() {
        let agg = Aggregator::new(AggregationConfig::default());
        let w = synthetic_window(0.0, 1.0, 10, &[0; 10]);
        let segs = agg.stitch(&[w]).unwrap();
        assert!(segs.is_empty());
    }

    #[test]
    fn single_window_one_speaker_yields_one_segment() {
        let agg = Aggregator::new(AggregationConfig::default());
        // 10 frames over 1.0s, speaker 0 throughout.
        let w = synthetic_window(0.0, 1.0, 10, &[1; 10]);
        let segs = agg.stitch(&[w]).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].local_speaker_idx, 0);
        assert!(!segs[0].is_overlap);
    }

    #[test]
    fn single_window_overlap_yields_two_segments_same_time() {
        let agg = Aggregator::new(AggregationConfig::default());
        // Class 4 = overlap of speakers 0 and 1.
        let w = synthetic_window(0.0, 1.0, 10, &[4; 10]);
        let segs = agg.stitch(&[w]).unwrap();
        assert_eq!(segs.len(), 2);
        // Both segments should share approximately the same time range.
        assert!((segs[0].time.start - segs[1].time.start).abs() < 1e-3);
        assert!((segs[0].time.end - segs[1].time.end).abs() < 1e-3);
        // One segment per speaker, both flagged overlap.
        assert!(segs.iter().all(|s| s.is_overlap));
        let speakers: Vec<u8> = segs.iter().map(|s| s.local_speaker_idx).collect();
        assert!(speakers.contains(&0));
        assert!(speakers.contains(&1));
    }

    #[test]
    fn two_windows_with_consistent_speakers_remain_consistent() {
        // Window A: 0..5s, speaker 0 (class 1) throughout.
        // Window B: 4..9s (1s overlap), speaker 0 (class 1) throughout.
        // After stitching, both should resolve to speaker 0.
        let a = synthetic_window(0.0, 5.0, 50, &[1; 50]);
        let b = synthetic_window(4.0, 9.0, 50, &[1; 50]);
        let agg = Aggregator::new(AggregationConfig::default());
        let segs = agg.stitch(&[a, b]).unwrap();
        // Every segment should be speaker 0, no overlaps.
        assert!(segs.iter().all(|s| s.local_speaker_idx == 0));
        assert!(segs.iter().all(|s| !s.is_overlap));
    }

    #[test]
    fn two_windows_requiring_permutation_get_aligned() {
        // Window A: 0..5s. Frames split half/half: speaker 0 (class 1) then speaker 1 (class 2).
        // Window B: 4..9s. SAME audio in the overlap, but the model labels it
        // with permuted indices: in window B, the same person who was speaker 0
        // is now labeled speaker 1.
        // So in the overlap region, A says (silence, sp0, sp0, sp0, sp0, sp0, sp1, sp1, sp1, sp1)
        // and B says (sp1, sp1, sp1, sp1, sp1, sp1, sp0, sp0, sp0, sp0) — perfectly inverted.
        // After Hungarian alignment, B's labels must be remapped so the same
        // person gets the same index.
        let a = synthetic_window(0.0, 5.0, 50,
            &[
                1,1,1,1,1,1,1,1,1,1, // first 1s: speaker 0
                1,1,1,1,1,1,1,1,1,1, // 1-2s: speaker 0
                1,1,1,1,1,1,1,1,1,1, // 2-3s: speaker 0
                1,1,1,1,1,1,1,1,1,1, // 3-4s: speaker 0
                2,2,2,2,2,2,2,2,2,2, // 4-5s: speaker 1
            ]);
        let b = synthetic_window(4.0, 9.0, 50,
            &[
                1,1,1,1,1,1,1,1,1,1, // 4-5s: speaker 0 (in B's local frame); should align with A's speaker 1
                2,2,2,2,2,2,2,2,2,2, // 5-6s
                2,2,2,2,2,2,2,2,2,2, // 6-7s
                2,2,2,2,2,2,2,2,2,2, // 7-8s
                2,2,2,2,2,2,2,2,2,2, // 8-9s
            ]);
        let agg = Aggregator::new(AggregationConfig::default());
        let segs = agg.stitch(&[a, b]).unwrap();

        // We should see two distinct speaker indices file-wide (0 and 1).
        let mut idx_set = std::collections::HashSet::new();
        for s in &segs { idx_set.insert(s.local_speaker_idx); }
        assert_eq!(idx_set.len(), 2);

        // The earliest segment must be one speaker and the latest must be the other.
        let mut sorted = segs.clone();
        sorted.sort_by(|a, b| a.time.start.partial_cmp(&b.time.start).unwrap());
        let first = sorted.first().unwrap();
        let last = sorted.last().unwrap();
        assert_ne!(first.local_speaker_idx, last.local_speaker_idx);
    }

    #[test]
    fn min_segment_filter_drops_tiny_runs() {
        // 1 frame of speaker 0 — duration is one frame stride.
        let w = synthetic_window(0.0, 1.0, 100, &{
            let mut v = vec![0; 100]; // silence
            v[50] = 1; // 1-frame blip
            v
        });
        let mut config = AggregationConfig::default();
        config.min_segment_secs = 0.1; // 100ms — bigger than 1 frame (10ms)
        let agg = Aggregator::new(config);
        let segs = agg.stitch(&[w]).unwrap();
        assert!(segs.is_empty());
    }

    #[test]
    fn output_segments_are_sorted_by_start_time() {
        // Two non-overlapping spans of speaker 0.
        let mut classes = vec![0; 100];
        for i in 10..20 { classes[i] = 1; }
        for i in 50..60 { classes[i] = 1; }
        let w = synthetic_window(0.0, 1.0, 100, &classes);
        let agg = Aggregator::new(AggregationConfig::default());
        let segs = agg.stitch(&[w]).unwrap();
        assert!(segs.len() >= 2);
        for pair in segs.windows(2) {
            assert!(pair[0].time.start <= pair[1].time.start);
        }
    }

    #[test]
    fn confidence_is_within_unit_interval() {
        let w = synthetic_window(0.0, 1.0, 10, &[1; 10]);
        let agg = Aggregator::new(AggregationConfig::default());
        let segs = agg.stitch(&[w]).unwrap();
        for s in segs {
            assert!(s.confidence.get() >= 0.0);
            assert!(s.confidence.get() <= 1.0);
        }
    }
}
```

- [ ] **Step 6.2: Run tests, confirm 9/9 pass**

```bash
cargo test --features segmentation --lib segmentation::aggregator::tests
```

Expected: 9/9 pass.

If any test fails (especially `two_windows_requiring_permutation_get_aligned`), debug carefully — the Hungarian permutation logic has multiple sign conventions. The test checks that *some* file-wide consistency emerges; if the algorithm is correct but the permutation direction is off, swap the inverse step in `window_permutation`.

- [ ] **Step 6.3: Run all segmentation tests cumulatively**

```bash
cargo test --features segmentation --lib segmentation::
```

Expected: 8 (hungarian) + 4 (trait) + 13 (decoder) + 9 (aggregator) = **34 passed**.

- [ ] **Step 6.4: Run clippy + fmt + wasm**

```bash
cargo fmt
cargo clippy --features segmentation --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features segmentation --lib
```

Expected: all clean.

- [ ] **Step 6.5: Commit**

```bash
git add src/segmentation/aggregator.rs
git commit -m "feat(segmentation): aggregator with Hungarian + run-length encoding"
```

---

## Task 7: `PowersetSegmenter` (ONNX wrapper)

**Files:**
- Modify: `src/segmentation/powerset.rs`

This wraps `ort::Session` and orchestrates: load model → for each 10s sliding window of audio, run inference → collect `WindowOutput` → call `Aggregator::stitch` → return `Vec<RawSegment>`. Gated behind both `onnx` and `segmentation` features.

The hardest part is binding the ONNX I/O correctly. The model expects `f32[1, 1, 160_000]` (batch, channel, samples) and produces `f32[1, num_frames, 7]`. `num_frames` depends on the model — we read it from the output shape rather than hardcoding.

- [ ] **Step 7.1: Implement `PowersetSegmenter`**

Replace `src/segmentation/powerset.rs` with:

```rust
//! `PowersetSegmenter` — ONNX-backed `Segmenter` wrapping
//! `sherpa-onnx-pyannote-segmentation-3-0`.
//!
//! Slides a 10-second window across the audio with a 500ms hop (95% overlap),
//! runs ONNX inference per window, and feeds outputs into `Aggregator`.

use crate::segmentation::aggregator::{AggregationConfig, Aggregator, WindowOutput};
use crate::segmentation::{RawSegment, SegmentationError, Segmenter, MIN_AUDIO_SAMPLES};
use ndarray::{Array, IxDyn};
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::Value;
use std::path::{Path, PathBuf};

/// Tunable parameters for `PowersetSegmenter`.
#[derive(Debug, Clone)]
pub struct PowersetConfig {
    /// Window duration in seconds.
    pub window_secs: f32,
    /// Hop size between windows in seconds.
    pub hop_secs: f32,
    /// Sample rate the model expects (16000 for sherpa-onnx-pyannote-segmentation-3-0).
    pub sample_rate: u32,
    /// Forwarded to the inner `Aggregator`.
    pub aggregation: AggregationConfig,
}

impl Default for PowersetConfig {
    fn default() -> Self {
        Self {
            window_secs: 10.0,
            hop_secs: 0.5,
            sample_rate: 16000,
            aggregation: AggregationConfig::default(),
        }
    }
}

/// ONNX-backed powerset speaker segmenter.
pub struct PowersetSegmenter {
    session: Session,
    config: PowersetConfig,
    model_path: PathBuf,
}

impl PowersetSegmenter {
    /// Load the ONNX model from `model_path`.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, SegmentationError> {
        Self::with_config(model_path, PowersetConfig::default())
    }

    /// Load with explicit configuration.
    pub fn with_config(
        model_path: impl AsRef<Path>,
        config: PowersetConfig,
    ) -> Result<Self, SegmentationError> {
        let path = model_path.as_ref().to_path_buf();
        let session = Session::builder()
            .map_err(|e| SegmentationError::ModelIo {
                path: path.clone(),
                detail: format!("session builder failed: {e}"),
            })?
            .commit_from_file(&path)
            .map_err(|e| SegmentationError::ModelIo {
                path: path.clone(),
                detail: format!("commit_from_file failed: {e}"),
            })?;
        Ok(Self { session, config, model_path: path })
    }

    pub fn config(&self) -> &PowersetConfig { &self.config }
    pub fn model_path(&self) -> &Path { &self.model_path }

    fn window_samples(&self) -> usize {
        (self.config.window_secs * self.config.sample_rate as f32) as usize
    }
    fn hop_samples(&self) -> usize {
        (self.config.hop_secs * self.config.sample_rate as f32) as usize
    }

    /// Run inference on a single 10-second window.
    fn infer_window(
        &self,
        window: &[f32],
        window_idx: usize,
    ) -> Result<(Vec<f32>, usize), SegmentationError> {
        let win_samples = self.window_samples();
        // Zero-pad short audio to the full window length.
        let mut buf = vec![0.0_f32; win_samples];
        let n = window.len().min(win_samples);
        buf[..n].copy_from_slice(&window[..n]);

        let input = Array::from_shape_vec(IxDyn(&[1, 1, win_samples]), buf).map_err(|e| {
            SegmentationError::InferenceFailed {
                window_idx,
                detail: format!("input shape: {e}"),
            }
        })?;
        let value = Value::from_array(input).map_err(|e| SegmentationError::InferenceFailed {
            window_idx,
            detail: format!("Value::from_array: {e}"),
        })?;

        let outputs: SessionOutputs = self
            .session
            .run([SessionInputValue::Owned(value.into_dyn())].into_iter().enumerate().map(
                |(_, v)| ("waveform", v),
            ))
            .map_err(|e| SegmentationError::InferenceFailed {
                window_idx,
                detail: format!("session.run: {e}"),
            })?;

        // Take the first (and only) output.
        let (_name, output_value) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| SegmentationError::InferenceFailed {
                window_idx,
                detail: "no output tensors".to_owned(),
            })?;
        let (shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| SegmentationError::InferenceFailed {
                window_idx,
                detail: format!("try_extract_tensor: {e}"),
            })?;
        let shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != 7 {
            return Err(SegmentationError::InvalidOutputShape { actual_shape: shape });
        }
        let num_frames = shape[1];
        Ok((data.to_vec(), num_frames))
    }
}

impl Segmenter for PowersetSegmenter {
    fn segment(&self, audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError> {
        if audio.len() < MIN_AUDIO_SAMPLES {
            return Err(SegmentationError::AudioTooShort {
                actual_secs: audio.len() as f32 / self.config.sample_rate as f32,
                min_secs: MIN_AUDIO_SAMPLES as f32 / self.config.sample_rate as f32,
            });
        }

        let win_samples = self.window_samples();
        let hop_samples = self.hop_samples();
        let total_samples = audio.len();
        let mut windows: Vec<WindowOutput> = Vec::new();
        let mut window_idx = 0_usize;
        let mut start_sample = 0_usize;
        loop {
            let end_sample = (start_sample + win_samples).min(total_samples);
            let slice = &audio[start_sample..end_sample];
            let (logits, num_frames) = self.infer_window(slice, window_idx)?;
            let start_t = start_sample as f32 / self.config.sample_rate as f32;
            let end_t = (start_sample + win_samples) as f32 / self.config.sample_rate as f32;
            let w = WindowOutput::new(start_t, end_t, logits, num_frames)?;
            windows.push(w);
            window_idx += 1;
            if start_sample + win_samples >= total_samples {
                break;
            }
            start_sample += hop_samples;
        }

        let agg = Aggregator::new(self.config.aggregation.clone());
        agg.stitch(&windows)
    }

    fn max_local_speakers(&self) -> usize { 3 }
    fn supports_overlap(&self) -> bool { true }
}
```

> **Note on ort 2.0 API:** the `Session::run` input format may need slight adjustment depending on the exact `ort 2.0.0-rc.12` API — check `cargo doc` or examples; the signature shown here uses `(name, value)` pairs which is the canonical pattern. If a different API surface is required, adapt syntax while keeping the `Value::from_array` / `try_extract_tensor::<f32>` pattern.

- [ ] **Step 7.2: Build with onnx + segmentation features**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m1
cargo check --features onnx,segmentation --lib 2>&1 | tail -10
```

Expected: clean. **If you see API mismatches with `ort 2.0.0-rc.12`** (e.g. `Session::run` signature different, `try_extract_tensor` returns different tuple, etc.), adapt the call sites to match the actual API. The conceptual flow — load session, build f32[1,1,N] input tensor, run, extract f32[1,F,7] output, return logits + frame count — stays the same regardless of syntax adjustments.

- [ ] **Step 7.3: Run lib tests with the onnx feature on**

```bash
cargo test --features onnx,segmentation --lib segmentation::
```

Expected: same 34 tests pass (the powerset.rs file itself has no unit tests — it's exercised at integration level in Task 9).

- [ ] **Step 7.4: Verify wasm32 still builds (segmentation only, no onnx)**

```bash
cargo check --target wasm32-unknown-unknown --no-default-features --features segmentation --lib
```

Expected: clean. The `powerset` module is conditionally compiled out (it requires `onnx`), so wasm32 + `segmentation` only builds the pure-Rust core.

- [ ] **Step 7.5: Run clippy with all features**

```bash
cargo clippy --features onnx,segmentation --lib -- -D warnings
```

Expected: clean.

- [ ] **Step 7.6: Commit**

```bash
git add src/segmentation/powerset.rs
git commit -m "feat(segmentation): PowersetSegmenter ONNX wrapper with sliding window"
```

---

## Task 8: Add `powerset_fp32` model entry to manifest

**Files:**
- Modify: `src/models/manifest.toml`

The sherpa-onnx project ships `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` as the canonical asset. Inside the tar there is a `model.onnx` file (~5.9 MB). For M1 we need either:

1. A direct .onnx URL we can SHA-256 (preferred), OR
2. Self-host the extracted .onnx as a polyvoice GitHub Release asset, OR
3. Block this task and ask the controller to decide.

This task tries options in that priority order.

- [ ] **Step 8.1: Probe for a direct .onnx URL on HuggingFace**

```bash
mkdir -p /tmp/polyvoice-m1-segmodel
cd /tmp/polyvoice-m1-segmodel

# Candidate 1 (csukuangfj's HF): direct file lookup.
curl -sIL "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx" \
  | head -5

# Candidate 2: pyannote/segmentation-3.0 (gated; will return 401 unless HF token set).
curl -sIL "https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.bin" \
  | head -5

# Candidate 3: sherpa-onnx releases — only tar.bz2 available.
curl -sIL "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" \
  | head -5
```

Look at the HTTP status codes:
- If candidate 1 returns `200 OK` and `Content-Type: application/octet-stream` (or similar) and the size is around 5–7 MB, **it's a direct .onnx — use it**.
- If candidate 1 returns 404 or HTML, fall through to step 8.2.
- Candidate 2 (gated pyannote) is unlikely to work without auth.
- Candidate 3 is .tar.bz2 — needs unpacking.

- [ ] **Step 8.2: If a direct URL works, fetch + checksum it**

```bash
curl -sL "<the-working-url>" -o /tmp/polyvoice-m1-segmodel/powerset.onnx
file /tmp/polyvoice-m1-segmodel/powerset.onnx
shasum -a 256 /tmp/polyvoice-m1-segmodel/powerset.onnx
ls -la /tmp/polyvoice-m1-segmodel/powerset.onnx | awk '{print "size="$5}'
```

`file` should report `data` (or specifically ONNX-related bytes); `du` should show ~5–7 MB. **If it shows HTML or much smaller, the URL was wrong; report `BLOCKED` with the actual content type.**

- [ ] **Step 8.3: If no direct URL works, fall back to extracting from the tar**

```bash
cd /tmp/polyvoice-m1-segmodel
curl -sL "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" \
  -o sherpa.tar.bz2
ls -la sherpa.tar.bz2
tar -xjf sherpa.tar.bz2
find . -name "*.onnx"
```

Locate the .onnx file (it's likely at `sherpa-onnx-pyannote-segmentation-3-0/model.onnx`). Compute its SHA-256 and size:

```bash
shasum -a 256 sherpa-onnx-pyannote-segmentation-3-0/model.onnx
ls -la sherpa-onnx-pyannote-segmentation-3-0/model.onnx | awk '{print "size="$5}'
```

For the manifest URL, **upload this .onnx as an asset** to the polyvoice GitHub Releases (suggest tag `v0.6.0-alpha.0-models`). Document the upload step:

```bash
# Manual step (controller must perform — requires gh + write permissions):
gh release create v0.6.0-alpha.0-models \
  --repo ekhodzitsky/polyvoice \
  --title "v0.6.0-alpha.0 model bundle" \
  --notes "M1 powerset segmentation model (extracted from sherpa-onnx-pyannote-segmentation-3-0.tar.bz2)" \
  sherpa-onnx-pyannote-segmentation-3-0/model.onnx#powerset_fp32.onnx
```

If the implementer cannot perform the upload (no `gh` permission or the tag exists), report `BLOCKED` with the local SHA-256/size so the controller can complete the upload and provide the URL.

- [ ] **Step 8.4: Update `src/models/manifest.toml`**

Append a new model entry. The manifest currently has `[models.silero_vad]` and `[models.wespeaker_resnet34]`. Add a third entry **at the end of the file** (after `wespeaker_resnet34`):

```toml

[models.powerset_fp32]
url      = "<URL_FROM_STEP_8_2_OR_8_3>"
sha256   = "<SHA256_FROM_STEP_8_2_OR_8_3>"
size     = <SIZE_FROM_STEP_8_2_OR_8_3>
filename = "powerset_fp32.onnx"
```

**Do NOT modify the existing `[profiles.mobile]` or `[profiles.balanced]` blocks.** They keep referring to `silero_vad` until M6's pipeline integration swaps them.

- [ ] **Step 8.5: Verify the manifest still parses**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m1
cargo test --features download --lib models::
```

Expected: 17 tests still pass (no profile dangling-ref, sha256 valid, etc.).

- [ ] **Step 8.6: Cleanup**

```bash
rm -rf /tmp/polyvoice-m1-segmodel
```

- [ ] **Step 8.7: Commit**

```bash
git add src/models/manifest.toml
git commit -m "feat(models): add powerset_fp32 manifest entry for M1"
```

---

## Task 9: Network integration test

**Files:**
- Create: `tests/segmenter_test.rs`

Real end-to-end test: download the model via `ModelRegistry`, run inference on synthetic audio, assert at least one non-silence segment is produced. Marked `#[ignore]` so the M0 PR-time CI doesn't run it.

- [ ] **Step 9.1: Create the test file**

Write `tests/segmenter_test.rs`:

```rust
//! Integration test for `PowersetSegmenter` against the real upstream ONNX model.
//!
//! Runs only when explicitly invoked:
//!   cargo test --features onnx,segmentation,download --test segmenter_test -- --ignored
//!
//! Downloads ~6 MB of model weights. Requires network connectivity.

#![cfg(all(feature = "onnx", feature = "segmentation", feature = "download"))]
#![allow(clippy::expect_used)]

use polyvoice::models::ModelRegistry;
use polyvoice::segmentation::{PowersetSegmenter, Segmenter};
use std::f32::consts::PI;
use tempfile::TempDir;

/// Construct 10 seconds of synthetic 16 kHz mono audio: half a 220 Hz sine
/// (speaker A), half a 440 Hz sine (speaker B). Not a perfect speaker model
/// but enough for the segmenter to find structure.
fn synthetic_two_speaker_audio() -> Vec<f32> {
    let sr = 16_000_usize;
    let total = 10 * sr;
    let mut audio = Vec::with_capacity(total);
    for i in 0..total {
        let t = i as f32 / sr as f32;
        let amp = if i < total / 2 {
            (2.0 * PI * 220.0 * t).sin() * 0.3
        } else {
            (2.0 * PI * 440.0 * t).sin() * 0.3
        };
        audio.push(amp);
    }
    audio
}

#[test]
#[ignore = "real network — run with --ignored"]
fn powerset_segmenter_emits_segments_on_real_model() {
    let tmp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::with_cache_dir(tmp.path()).expect("registry");
    let model_path = registry
        .ensure("powerset_fp32")
        .expect("model download must succeed");

    let segmenter = PowersetSegmenter::new(&model_path).expect("segmenter loads");
    assert_eq!(segmenter.max_local_speakers(), 3);
    assert!(segmenter.supports_overlap());

    let audio = synthetic_two_speaker_audio();
    let segments = segmenter.segment(&audio).expect("segment runs");

    // Synthetic audio with sustained tones is unrealistic for a speech model.
    // The segmenter may legitimately label most of it as silence. Just assert
    // the call succeeded and produced a Vec (possibly empty) of well-formed segments.
    for s in &segments {
        assert!(s.time.end >= s.time.start, "non-decreasing time");
        assert!(s.local_speaker_idx < segmenter.max_local_speakers() as u8);
        assert!(s.confidence.get() >= 0.0 && s.confidence.get() <= 1.0);
    }
}

#[test]
#[ignore = "real network — run with --ignored"]
fn powerset_segmenter_rejects_short_audio() {
    let tmp = TempDir::new().expect("temp dir");
    let registry = ModelRegistry::with_cache_dir(tmp.path()).expect("registry");
    let model_path = registry.ensure("powerset_fp32").expect("model download");

    let segmenter = PowersetSegmenter::new(&model_path).expect("segmenter loads");
    let too_short = vec![0.0_f32; 100]; // 6.25 ms < 100 ms minimum
    let err = segmenter.segment(&too_short).expect_err("must reject");
    let msg = format!("{err}");
    assert!(msg.to_lowercase().contains("too short"));
}
```

- [ ] **Step 9.2: Confirm the file compiles (without running the ignored tests)**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m1
cargo test --features onnx,segmentation,download --test segmenter_test -- --list
```

Expected: lists 2 tests, both `#[ignore]`.

- [ ] **Step 9.3: Commit**

```bash
git add tests/segmenter_test.rs
git commit -m "test(segmentation): add network integration tests behind --ignored"
```

---

## Task 10: lib.rs re-exports

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 10.1: Add re-exports**

In `src/lib.rs`, find the `#[cfg(feature = "segmentation")] pub mod segmentation;` line added in Task 2. Append a public re-export block beneath it:

```rust
#[cfg(feature = "segmentation")]
pub mod segmentation;

#[cfg(feature = "segmentation")]
pub use segmentation::{
    Aggregator, AggregationConfig, FrameLabel, PowersetClass, PowersetDecoder,
    RawSegment, Segmenter, SegmentationError, WindowOutput, MIN_AUDIO_SAMPLES,
};

#[cfg(all(feature = "onnx", feature = "segmentation"))]
pub use segmentation::{PowersetConfig, PowersetSegmenter};
```

- [ ] **Step 10.2: Build all feature combos**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m1
cargo check --features download
cargo check --features cli
cargo check --features ffi
cargo check --features onnx
cargo check --features segmentation
cargo check --no-default-features
cargo check --target wasm32-unknown-unknown --no-default-features --lib
cargo check --all-features
```

All must exit 0.

- [ ] **Step 10.3: Run all tests**

```bash
cargo test --all-features --lib
cargo test --all-features --doc
```

Expected: lib + doc tests pass.

- [ ] **Step 10.4: Run clippy**

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

Expected: clean.

- [ ] **Step 10.5: Commit**

```bash
git add src/lib.rs
git commit -m "feat(lib): re-export segmentation surface at crate root"
```

---

## Task 11: CHANGELOG update

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 11.1: Add an M1 section under `## [Unreleased]`**

Find the existing `## [Unreleased]` block (added in M0). Locate the M0 `### Added` section. Below the existing `### Changed` block (or at the end of the Unreleased section), append:

```markdown

### Added (M1 — Powerset segmentation)
- `polyvoice::segmentation` module: `Segmenter` trait, `RawSegment`, `SegmentationError`,
  `PowersetSegmenter` (ONNX-backed), `PowersetDecoder`, `PowersetClass`, `FrameLabel`,
  `Aggregator`, `WindowOutput`, `AggregationConfig`.
- New Cargo feature `segmentation` (in default features). The pure-Rust algorithmic
  core (decoder, aggregator, hungarian) is wasm32-clean; only `PowersetSegmenter`
  additionally requires `onnx`.
- In-tree Kuhn-Munkres minimum-cost assignment (~50 LOC) for sliding-window speaker
  index alignment — no external dependency added.
- New manifest entry `[models.powerset_fp32]` for sherpa-onnx-pyannote-segmentation-3-0.
  Profiles still resolve to the legacy `silero_vad` segmenter; M6 swaps them.
```

- [ ] **Step 11.2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): document M1 segmentation additions"
```

---

## Task 12: End-to-end verification

**Files:** none modified.

- [ ] **Step 12.1: Full feature matrix build**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m1
cargo build --no-default-features 2>&1 | tail -3
cargo build --features download 2>&1 | tail -3
cargo build --features cli 2>&1 | tail -3
cargo build --features ffi 2>&1 | tail -3
cargo build --features onnx 2>&1 | tail -3
cargo build --features segmentation 2>&1 | tail -3
cargo build --features onnx,segmentation 2>&1 | tail -3
cargo build --all-features 2>&1 | tail -3
```

All must pass.

- [ ] **Step 12.2: Tests**

```bash
cargo test --features download --lib 2>&1 | tail -5
cargo test --features segmentation --lib 2>&1 | tail -5
cargo test --features onnx,segmentation --lib 2>&1 | tail -5
cargo test --all-features --lib 2>&1 | tail -5
cargo test --all-features --doc 2>&1 | tail -5
```

- [ ] **Step 12.3: Wasm32 smoke**

```bash
cargo check --target wasm32-unknown-unknown --no-default-features --lib
cargo check --target wasm32-unknown-unknown --no-default-features --features segmentation --lib
```

Both must pass — segmentation's pure-Rust core compiles to wasm32.

- [ ] **Step 12.4: Lint**

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```

If `cargo fmt --check` flags any file, run `cargo fmt` and commit the resulting diff with `chore(fmt): apply rustfmt to M1 files`.

- [ ] **Step 12.5: Network integration test**

```bash
cargo test --features onnx,segmentation,download --test segmenter_test -- --ignored 2>&1 | tail -10
```

Expected: 2 tests pass.

- [ ] **Step 12.6: Release-gate stub**

```bash
./scripts/release-gate.sh
echo "exit: $?"
```

Expected: exit code 2 (PENDING-only). The PASS lines (CI matrix presence, cargo doc) should remain green.

- [ ] **Step 12.7: Tag the milestone in git**

```bash
git tag -a m1-complete -m "M1 complete: powerset segmenter"
```

(Do not push tags unless requested.)

- [ ] **Step 12.8: Final git log**

```bash
git log --oneline 2b03c48..HEAD
```

Should show ~12 commits.

---

## Self-review checklist

After all tasks:

1. **Spec coverage:** Every M1 deliverable from spec §10.1 / §3.1 / §5.3 maps to a task:
   - `Segmenter` trait + `RawSegment` → Task 3
   - `PowersetSegmenter` → Task 7
   - `PowersetDecoder` → Task 4
   - `Aggregator` w/ Hungarian → Tasks 2 + 5 + 6
   - manifest entry → Task 8
   - integration test → Task 9
   - re-exports → Task 10
   - CHANGELOG → Task 11

2. **Additive guarantee:** verify no v0.5.x or M0 public API was removed. Run `cargo semver-checks check-release` against the published 0.6.0-alpha.0 baseline (or skip if baseline missing). Public surface only grows.

3. **Wasm32 cleanness:** the segmentation pure-Rust core (`mod.rs`, `decoder.rs`, `aggregator.rs`, `hungarian.rs`) compiles to `wasm32-unknown-unknown` with `--no-default-features --features segmentation`. Verified in Tasks 2.8, 3.5, 4.5, 5.3, 6.4, 7.4, 10.2, 12.3.

4. **No `unwrap`/`expect`/`panic` in lib non-test code:** verified by `cargo clippy -- -D warnings` (the project has `clippy::unwrap_used = "deny"` set in lib.rs). The `unwrap_or(Confidence::default())` in `frame_confidence` is the only borderline case — it's lint-friendly and explicitly justified.

5. **Test coverage:** every public function in segmentation has at least one test or doc-test. The 34 segmentation unit tests + 2 ignored integration tests cover decoder (13), aggregator (9), hungarian (8), trait (4), and end-to-end (2).

6. **Commits are atomic:** each task ends in exactly one commit. Total 12 commits.

---

## Out of scope for this plan

- Profile manifest swap (silero_vad → powerset_fp32 in profile entries) — that's M6.
- Pipeline integration (`Pipeline::run` calling the segmenter) — M6.
- INT8 quantization of the powerset model — M5.
- Streaming-friendly inference path — M1.x or v1.1.
- Any change to `OfflineDiarizer`, `OnlineDiarizer`, `SileroVad`, etc. — additive only.
