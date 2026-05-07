# M4 — Overlap Resegmenter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the v1.0 `Resegmenter` trait + `OverlapResegmenter` (pure-Rust post-clustering pass) that, given primary single-speaker turns, speaker centroids, and per-overlap-region embeddings, attaches a **second** speaker label to each overlap region by picking the nearest cosine-cluster ≠ primary. Pure-Rust, wasm-clean, no ONNX dependency. Spec: `docs/superpowers/specs/2026-05-07-m4-overlap-resegmenter-design.md`.

**Architecture:** New single-file module `src/resegmentation.rs` (feature-gated `resegmentation`, default-on, no `onnx` requirement). Holds `Resegmenter` trait, `ResegmentError`, `OverlapResegmenter` struct + impl, helpers `compute_centroids` and `extract_overlap_time_ranges`. M6 will wire it into `Pipeline` and produce overlap embeddings via the existing `EmbedderPool` + `apply_overlap_mask`. Legacy `src/overlap.rs::detect_overlaps` stays untouched — it's interval-only and serves a different code path.

**Tech Stack:** Rust 2024. No new dependencies. Reuses `crate::utils::{cosine_similarity, mean_vector, l2_normalize}` and `crate::types::{SpeakerId, SpeakerTurn, TimeRange}`. M1's `RawSegment` (gated `segmentation`) is consumed by `extract_overlap_time_ranges`.

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `Cargo.toml` | modify | Add `resegmentation` feature (default-on) |
| `src/resegmentation.rs` | create | `Resegmenter` trait, `ResegmentError`, `OverlapResegmenter`, types, `compute_centroids`, `extract_overlap_time_ranges` |
| `src/lib.rs` | modify | `pub mod resegmentation;` gated, re-exports |
| `tests/resegmentation_test.rs` | create | Synthetic-data integration tests (no #[ignore], runs in CI) |
| `tests/miri_resegmentation.rs` | create | Miri-friendly subset (no-overlap, single-overlap, centroid math) |
| `CHANGELOG.md` | modify | Unreleased M4 section |

Total roughly 520 lines new code.

---

## Task 1: Add `resegmentation` Cargo feature

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1.1: Update default + add feature**

In `/Users/ekhodzitsky/Documents/personal/polyvoice/Cargo.toml`, find:

```toml
default = ["spectral", "segmentation", "embedder", "clusterer"]
```

Replace with:

```toml
default = ["spectral", "segmentation", "embedder", "clusterer", "resegmentation"]
```

After the `clusterer = []` line, append:

```toml

# v1.0 Overlap-aware post-clustering resegmentation pass.
# Pure-Rust, wasm32-clean — does not require `onnx`. Operates on already-
# computed speaker centroids and overlap-region embeddings supplied by the
# caller (M6 Pipeline wires the embedder pool into this).
resegmentation = []
```

- [ ] **Step 1.2: Verify build matrix**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo check
cargo check --features resegmentation
cargo check --features resegmentation,segmentation
cargo check --features resegmentation,clusterer
cargo check --features resegmentation,clusterer,spectral,segmentation,embedder
cargo check --no-default-features
cargo check --no-default-features --features resegmentation
cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib
cargo check --all-features
```

All exit 0.

- [ ] **Step 1.3: Commit**

```bash
git add Cargo.toml
git commit -m "feat(cargo): add resegmentation feature flag for v1.0 M4 work"
```

---

## Task 2: `Resegmenter` trait + `ResegmentError` + input types

**Files:**
- Create: `src/resegmentation.rs`
- Modify: `src/lib.rs`

- [ ] **Step 2.1: Write failing tests first**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/src/resegmentation.rs`:

```rust
//! v1.0 OverlapResegmenter — overlap-aware post-clustering pass.
//!
//! Added in v0.6 (M4). See `docs/superpowers/specs/2026-05-07-m4-overlap-resegmenter-design.md`
//! and `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §3.1.

#[cfg(test)]
mod trait_tests {
    use super::*;
    use crate::types::{SpeakerId, SpeakerTurn, TimeRange};

    /// In-memory dummy used by trait conformance tests.
    struct ConstantResegmenter {
        out: Vec<SpeakerTurn>,
    }

    impl Resegmenter for ConstantResegmenter {
        fn resegment(
            &self,
            _inputs: ResegmentInputs<'_>,
        ) -> Result<Vec<SpeakerTurn>, ResegmentError> {
            Ok(self.out.clone())
        }
    }

    fn turn(start: f64, end: f64, spk: u32) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(spk),
            time: TimeRange { start, end },
            text: None,
        }
    }

    #[test]
    fn resegmenter_trait_object_is_dyn_compatible() {
        let r = ConstantResegmenter {
            out: vec![turn(0.0, 1.0, 0)],
        };
        let _b: Box<dyn Resegmenter> = Box::new(r);
    }

    #[test]
    fn resegmenter_returns_owned_turns() {
        let r = ConstantResegmenter {
            out: vec![turn(0.0, 1.0, 0), turn(1.0, 2.0, 1)],
        };
        let inputs = ResegmentInputs {
            primary_turns: &[],
            speaker_centroids: &[],
            overlap_regions: &[],
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].speaker, SpeakerId(0));
    }

    #[test]
    fn error_centroid_dim_mismatch_displays() {
        let err = ResegmentError::CentroidDimMismatch {
            index: 1,
            expected: 192,
            actual: 256,
        };
        let msg = format!("{err}");
        assert!(msg.contains("192"));
        assert!(msg.contains("256"));
        assert!(msg.contains("index 1"));
    }

    #[test]
    fn error_overlap_dim_mismatch_displays() {
        let err = ResegmentError::OverlapDimMismatch {
            index: 0,
            expected: 192,
            actual: 64,
        };
        let msg = format!("{err}");
        assert!(msg.contains("192"));
        assert!(msg.contains("64"));
    }

    #[test]
    fn error_missing_primary_centroid_displays() {
        let err = ResegmentError::MissingPrimaryCentroid {
            index: 2,
            primary: SpeakerId(7),
        };
        let msg = format!("{err}");
        assert!(msg.contains('2'));
        assert!(msg.contains('7'));
    }
}
```

- [ ] **Step 2.2: Wire stub mod into lib.rs**

In `/Users/ekhodzitsky/Documents/personal/polyvoice/src/lib.rs`, after the existing block:

```rust
#[cfg(all(feature = "clusterer", feature = "spectral"))]
pub use clusterer::NmeScClusterer;
```

append:

```rust

#[cfg(feature = "resegmentation")]
pub mod resegmentation;
```

- [ ] **Step 2.3: Confirm compile-failure**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --features resegmentation --lib resegmentation::trait_tests 2>&1 | head -30
```

Expected: errors about undefined `Resegmenter`, `ResegmentInputs`, `ResegmentError`, `OverlapRegionInput`, `SpeakerCentroid`.

- [ ] **Step 2.4: Implement trait + error + input types**

Replace the body of `src/resegmentation.rs` (keep the `#[cfg(test)] mod trait_tests` block at the bottom):

```rust
//! v1.0 OverlapResegmenter — overlap-aware post-clustering pass.
//!
//! Added in v0.6 (M4). See `docs/superpowers/specs/2026-05-07-m4-overlap-resegmenter-design.md`
//! and `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §3.1.
//!
//! Pure Rust, wasm32-clean. Operates on already-computed speaker centroids and
//! overlap-region embeddings supplied by the caller. M6 (`Pipeline`) wires the
//! `EmbedderPool` and `apply_overlap_mask` into this.

use crate::types::{SpeakerId, SpeakerTurn, TimeRange};

/// Speaker resegmenter — given primary single-speaker turns, cluster centroids,
/// and per-overlap-region embeddings, returns a (possibly overlap-aware) flat
/// list of `SpeakerTurn`s where overlap regions may produce two turns over the
/// same time range with different speakers.
///
/// In v1.0 (M4) the polyvoice crate introduces `Resegmenter` as the canonical
/// trait. The legacy `crate::overlap::detect_overlaps` remains as an
/// interval-only helper unrelated to this pass.
pub trait Resegmenter: Send + Sync {
    /// Run the pass.
    ///
    /// **Requires:** all centroid vectors and all overlap embeddings have the
    /// same dimension and are approximately L2-normalized.
    /// **Guarantees on Ok:** every turn in `inputs.primary_turns` is preserved
    /// verbatim; secondary turns (if any) carry an existing `SpeakerId` from
    /// `inputs.speaker_centroids` and never repeat the primary speaker for the
    /// same region; output is sorted by `time.start`.
    fn resegment(
        &self,
        inputs: ResegmentInputs<'_>,
    ) -> Result<Vec<SpeakerTurn>, ResegmentError>;
}

/// All inputs needed by `Resegmenter::resegment`.
#[derive(Debug, Clone)]
pub struct ResegmentInputs<'a> {
    pub primary_turns: &'a [SpeakerTurn],
    pub speaker_centroids: &'a [SpeakerCentroid],
    pub overlap_regions: &'a [OverlapRegionInput],
}

/// L2-normalized centroid for one speaker cluster.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerCentroid {
    pub speaker: SpeakerId,
    pub embedding: Vec<f32>,
}

/// One overlap region with its caller-supplied embedding.
///
/// `embedding` is expected to be L2-normalized; this struct does not enforce
/// it (`OverlapResegmenter` returns `OverlapDimMismatch` only on dimension
/// mismatches, not on norm drift).
#[derive(Debug, Clone, PartialEq)]
pub struct OverlapRegionInput {
    pub time: TimeRange,
    pub primary_speaker: SpeakerId,
    pub embedding: Vec<f32>,
}

/// Errors from `Resegmenter` implementations.
#[derive(Debug, thiserror::Error)]
pub enum ResegmentError {
    #[error("centroid dim mismatch at index {index}: expected {expected}, got {actual}")]
    CentroidDimMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },

    #[error("overlap embedding dim mismatch at index {index}: expected {expected}, got {actual}")]
    OverlapDimMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },

    #[error("primary speaker {primary} for overlap region {index} not present in centroids")]
    MissingPrimaryCentroid { index: usize, primary: SpeakerId },
}

#[cfg(test)]
mod trait_tests {
    use super::*;

    /// In-memory dummy used by trait conformance tests.
    struct ConstantResegmenter {
        out: Vec<SpeakerTurn>,
    }

    impl Resegmenter for ConstantResegmenter {
        fn resegment(
            &self,
            _inputs: ResegmentInputs<'_>,
        ) -> Result<Vec<SpeakerTurn>, ResegmentError> {
            Ok(self.out.clone())
        }
    }

    fn turn(start: f64, end: f64, spk: u32) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(spk),
            time: TimeRange { start, end },
            text: None,
        }
    }

    #[test]
    fn resegmenter_trait_object_is_dyn_compatible() {
        let r = ConstantResegmenter {
            out: vec![turn(0.0, 1.0, 0)],
        };
        let _b: Box<dyn Resegmenter> = Box::new(r);
    }

    #[test]
    fn resegmenter_returns_owned_turns() {
        let r = ConstantResegmenter {
            out: vec![turn(0.0, 1.0, 0), turn(1.0, 2.0, 1)],
        };
        let inputs = ResegmentInputs {
            primary_turns: &[],
            speaker_centroids: &[],
            overlap_regions: &[],
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].speaker, SpeakerId(0));
    }

    #[test]
    fn error_centroid_dim_mismatch_displays() {
        let err = ResegmentError::CentroidDimMismatch {
            index: 1,
            expected: 192,
            actual: 256,
        };
        let msg = format!("{err}");
        assert!(msg.contains("192"));
        assert!(msg.contains("256"));
        assert!(msg.contains("index 1"));
    }

    #[test]
    fn error_overlap_dim_mismatch_displays() {
        let err = ResegmentError::OverlapDimMismatch {
            index: 0,
            expected: 192,
            actual: 64,
        };
        let msg = format!("{err}");
        assert!(msg.contains("192"));
        assert!(msg.contains("64"));
    }

    #[test]
    fn error_missing_primary_centroid_displays() {
        let err = ResegmentError::MissingPrimaryCentroid {
            index: 2,
            primary: SpeakerId(7),
        };
        let msg = format!("{err}");
        assert!(msg.contains('2'));
        assert!(msg.contains('7'));
    }
}
```

- [ ] **Step 2.5: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --features resegmentation --lib resegmentation::trait_tests
cargo fmt
cargo clippy --features resegmentation --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib
```

Expected: 5 trait tests pass, all clean.

- [ ] **Step 2.6: Commit**

```bash
git add src/resegmentation.rs src/lib.rs
git commit -m "feat(resegmentation): add Resegmenter trait + ResegmentError + input types"
```

---

## Task 3: Helpers — `compute_centroids` + `extract_overlap_time_ranges`

**Files:**
- Modify: `src/resegmentation.rs`

- [ ] **Step 3.1: Append failing tests**

Add to `src/resegmentation.rs` (after `mod trait_tests`):

```rust
#[cfg(test)]
mod centroid_tests {
    use super::*;
    use crate::types::SpeakerId;

    fn unit(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[axis] = 1.0;
        v
    }

    #[test]
    fn compute_centroids_l2_normalized() {
        let embeddings = vec![
            unit(3, 0),
            unit(3, 0),
            unit(3, 1),
            unit(3, 1),
        ];
        let labels = vec![0, 0, 1, 1];
        let centroids = compute_centroids(&embeddings, &labels);
        assert_eq!(centroids.len(), 2);
        for c in &centroids {
            let n: f32 = c.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-3, "centroid not L2-normalized: norm={n}");
        }
    }

    #[test]
    fn compute_centroids_drops_empty_clusters() {
        // Labels skip from 0 to 2; cluster 1 has no members.
        let embeddings = vec![unit(3, 0), unit(3, 1), unit(3, 1)];
        let labels = vec![0, 2, 2];
        let centroids = compute_centroids(&embeddings, &labels);
        assert_eq!(centroids.len(), 2);
        let speakers: Vec<u32> = centroids.iter().map(|c| c.speaker.0).collect();
        assert_eq!(speakers, vec![0, 2]);
    }

    #[test]
    fn compute_centroids_sorted_by_speaker_id() {
        let embeddings = vec![unit(3, 0), unit(3, 1), unit(3, 2)];
        let labels = vec![5, 1, 3];
        let centroids = compute_centroids(&embeddings, &labels);
        let speakers: Vec<u32> = centroids.iter().map(|c| c.speaker.0).collect();
        assert_eq!(speakers, vec![1, 3, 5]);
    }

    #[test]
    fn compute_centroids_empty_input_returns_empty() {
        let centroids = compute_centroids(&[], &[]);
        assert!(centroids.is_empty());
    }

    #[test]
    fn compute_centroids_label_mismatch_returns_empty() {
        // Mismatched lengths: caller bug, conservative empty return rather than panic.
        let centroids = compute_centroids(&[unit(3, 0)], &[0, 1]);
        assert!(centroids.is_empty());
    }
}

#[cfg(all(test, feature = "segmentation"))]
mod overlap_extract_tests {
    use super::*;
    use crate::segmentation::RawSegment;
    use crate::types::Confidence;

    fn raw(start: f64, end: f64, spk: u8, overlap: bool) -> RawSegment {
        RawSegment {
            time: TimeRange { start, end },
            local_speaker_idx: spk,
            is_overlap: overlap,
            confidence: Confidence::new(0.9).unwrap(),
        }
    }

    #[test]
    fn extract_returns_pairs_for_simultaneous_overlap_segments() {
        // Two RawSegments with the same time range and is_overlap = true:
        // aggregator's canonical overlap output.
        let segs = vec![
            raw(0.0, 1.0, 0, true),
            raw(0.0, 1.0, 1, true),
        ];
        let pairs = extract_overlap_time_ranges(&segs);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].0.start - 0.0).abs() < 1e-6);
        assert!((pairs[0].0.end - 1.0).abs() < 1e-6);
        // local pair is (lo, hi) where lo < hi.
        assert_eq!(pairs[0].1, 0);
        assert_eq!(pairs[0].2, 1);
    }

    #[test]
    fn extract_ignores_non_overlap_segments() {
        let segs = vec![
            raw(0.0, 1.0, 0, false),
            raw(0.0, 1.0, 1, false),
        ];
        let pairs = extract_overlap_time_ranges(&segs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn extract_ignores_overlap_flag_without_pair() {
        // is_overlap=true but only one local speaker present at this range.
        let segs = vec![raw(0.0, 1.0, 0, true)];
        let pairs = extract_overlap_time_ranges(&segs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn extract_handles_multiple_overlap_regions() {
        let segs = vec![
            raw(0.0, 1.0, 0, true),
            raw(0.0, 1.0, 1, true),
            raw(2.0, 3.0, 1, true),
            raw(2.0, 3.0, 2, true),
        ];
        let pairs = extract_overlap_time_ranges(&segs);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].1, 0);
        assert_eq!(pairs[0].2, 1);
        assert_eq!(pairs[1].1, 1);
        assert_eq!(pairs[1].2, 2);
    }
}
```

- [ ] **Step 3.2: Confirm compile-failure**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --features resegmentation,segmentation --lib resegmentation:: 2>&1 | head -10
```

Expected: undefined `compute_centroids`, `extract_overlap_time_ranges`.

- [ ] **Step 3.3: Implement helpers**

Add to `src/resegmentation.rs` (after the error type, before the test blocks):

```rust
/// Compute per-cluster L2-normalized centroids from clustered embeddings.
///
/// `labels[i]` is the cluster label of `embeddings[i]`. The cluster id stored
/// in the resulting `SpeakerCentroid` is the raw `labels[i]` cast to `SpeakerId`.
/// Empty clusters yield no entry. Output is sorted by `SpeakerId.0` ascending.
///
/// Returns an empty `Vec` if `embeddings.len() != labels.len()` or both are
/// empty — never panics.
///
/// **Pure Rust, wasm32-clean.**
pub fn compute_centroids(
    embeddings: &[Vec<f32>],
    labels: &[usize],
) -> Vec<SpeakerCentroid> {
    if embeddings.len() != labels.len() || embeddings.is_empty() {
        return Vec::new();
    }
    // Bucket by label.
    let mut buckets: std::collections::BTreeMap<usize, Vec<&Vec<f32>>> =
        std::collections::BTreeMap::new();
    for (emb, &lbl) in embeddings.iter().zip(labels.iter()) {
        buckets.entry(lbl).or_default().push(emb);
    }
    let mut out = Vec::with_capacity(buckets.len());
    for (lbl, members) in buckets {
        let owned: Vec<Vec<f32>> = members.iter().map(|e| (*e).clone()).collect();
        if let Some(mut mean) = crate::utils::mean_vector(&owned) {
            crate::utils::l2_normalize(&mut mean);
            // SpeakerId is u32; clamp to its range conservatively.
            let id = SpeakerId(lbl as u32);
            out.push(SpeakerCentroid {
                speaker: id,
                embedding: mean,
            });
        }
    }
    // BTreeMap iterates in label order, but cast to SpeakerId may reorder if
    // u32 truncation happened. Sort explicitly.
    out.sort_by_key(|c| c.speaker.0);
    out
}

/// Find pairs of `RawSegment`s that share a time range, are flagged
/// `is_overlap = true`, and carry two distinct `local_speaker_idx`.
/// Returns `(time_range, lo_local_idx, hi_local_idx)` per detected pair.
///
/// "Same time range" uses an `f64` tolerance of `1e-6`.
///
/// `lo_local_idx < hi_local_idx`. Caller is responsible for the local→global
/// `SpeakerId` mapping (typically from the same clustering pipeline).
///
/// **Pure Rust, wasm32-clean.** Gated `segmentation` because `RawSegment`
/// lives in the segmentation module.
#[cfg(feature = "segmentation")]
pub fn extract_overlap_time_ranges(
    segments: &[crate::segmentation::RawSegment],
) -> Vec<(TimeRange, u8, u8)> {
    let mut pairs: Vec<(TimeRange, u8, u8)> = Vec::new();
    for (i, a) in segments.iter().enumerate() {
        if !a.is_overlap {
            continue;
        }
        for b in segments.iter().skip(i + 1) {
            if !b.is_overlap {
                continue;
            }
            if a.local_speaker_idx == b.local_speaker_idx {
                continue;
            }
            if (a.time.start - b.time.start).abs() > 1e-6
                || (a.time.end - b.time.end).abs() > 1e-6
            {
                continue;
            }
            let (lo, hi) = if a.local_speaker_idx < b.local_speaker_idx {
                (a.local_speaker_idx, b.local_speaker_idx)
            } else {
                (b.local_speaker_idx, a.local_speaker_idx)
            };
            pairs.push((a.time, lo, hi));
        }
    }
    pairs
}
```

- [ ] **Step 3.4: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --features resegmentation --lib resegmentation::centroid_tests
cargo test --features resegmentation,segmentation --lib resegmentation::overlap_extract_tests
cargo clippy --features resegmentation,segmentation --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib
cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation,segmentation --lib
```

Expected:
- 5 centroid tests pass.
- 4 overlap_extract tests pass.
- Clippy clean.
- wasm32 with `resegmentation` only — clean.
- wasm32 with `resegmentation,segmentation` — clean (segmentation pure-Rust core is wasm32-clean per M1).

- [ ] **Step 3.5: Commit**

```bash
git add src/resegmentation.rs
git commit -m "feat(resegmentation): add compute_centroids + extract_overlap_time_ranges helpers"
```

---

## Task 4: `OverlapResegmenter` impl with cosine matching

**Files:**
- Modify: `src/resegmentation.rs`

- [ ] **Step 4.1: Append failing tests**

Add to `src/resegmentation.rs` (after the existing test modules):

```rust
#[cfg(test)]
mod resegmenter_tests {
    use super::*;
    use crate::types::{SpeakerId, SpeakerTurn, TimeRange};

    fn unit(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[axis] = 1.0;
        v
    }

    fn turn(start: f64, end: f64, spk: u32) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(spk),
            time: TimeRange { start, end },
            text: None,
        }
    }

    fn centroid(spk: u32, dim: usize, axis: usize) -> SpeakerCentroid {
        SpeakerCentroid {
            speaker: SpeakerId(spk),
            embedding: unit(dim, axis),
        }
    }

    fn region(start: f64, end: f64, primary: u32, dim: usize, axis: usize) -> OverlapRegionInput {
        OverlapRegionInput {
            time: TimeRange { start, end },
            primary_speaker: SpeakerId(primary),
            embedding: unit(dim, axis),
        }
    }

    #[test]
    fn no_overlap_passes_primary_through() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0), turn(2.0, 3.0, 1)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &[],
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out, primary);
    }

    #[test]
    fn single_cluster_passes_through() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0)];
        let regions = vec![region(0.5, 0.9, 0, 3, 0)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out, primary);
    }

    #[test]
    fn picks_secondary_excluding_primary() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1), centroid(2, 3, 2)];
        // Overlap region embedding lies along axis 1 → nearest to centroid id=1.
        let regions = vec![region(0.0, 1.0, 0, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out.len(), 2);
        // Both turns cover (0.0, 1.0); one is primary (id=0), other is secondary (id=1).
        let speakers: Vec<u32> = out.iter().map(|t| t.speaker.0).collect();
        assert!(speakers.contains(&0));
        assert!(speakers.contains(&1));
        assert!(!speakers.contains(&2));
    }

    #[test]
    fn threshold_blocks_low_cosine() {
        // Threshold 0.99 — only near-perfect matches allowed.
        let r = OverlapResegmenter::new(0.99, 0.0);
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        // Overlap embedding along axis 0 (matches primary); cosine to centroid 1 = 0.
        let regions = vec![region(0.0, 1.0, 0, 3, 0)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out, primary, "no secondary should be appended");
    }

    #[test]
    fn min_duration_blocks_short_region() {
        // Region duration 0.05s < default 0.1s → skipped.
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        let regions = vec![region(0.10, 0.15, 0, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out, primary);
    }

    #[test]
    fn output_is_sorted_by_start() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(2.0, 3.0, 0), turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        let regions = vec![region(2.0, 3.0, 0, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let out = r.resegment(inputs).unwrap();
        for w in out.windows(2) {
            assert!(w[0].time.start <= w[1].time.start);
        }
    }

    #[test]
    fn missing_primary_centroid_errors() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(1, 3, 1), centroid(2, 3, 2)];
        let regions = vec![region(0.0, 1.0, 0, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let err = r.resegment(inputs).expect_err("missing primary must error");
        assert!(matches!(
            err,
            ResegmentError::MissingPrimaryCentroid { primary: SpeakerId(0), .. }
        ));
    }

    #[test]
    fn centroid_dim_mismatch_errors() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![
            centroid(0, 3, 0),
            SpeakerCentroid {
                speaker: SpeakerId(1),
                embedding: vec![1.0, 0.0], // dim 2, not 3
            },
        ];
        let regions = vec![region(0.0, 1.0, 0, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let err = r.resegment(inputs).expect_err("dim mismatch must error");
        assert!(matches!(err, ResegmentError::CentroidDimMismatch { .. }));
    }

    #[test]
    fn overlap_dim_mismatch_errors() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        let regions = vec![OverlapRegionInput {
            time: TimeRange { start: 0.0, end: 1.0 },
            primary_speaker: SpeakerId(0),
            embedding: vec![1.0, 0.0], // dim 2, not 3
        }];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let err = r.resegment(inputs).expect_err("dim mismatch must error");
        assert!(matches!(err, ResegmentError::OverlapDimMismatch { .. }));
    }

    #[test]
    fn empty_centroids_passes_through() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &[],
            overlap_regions: &[],
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out, primary);
    }
}
```

- [ ] **Step 4.2: Confirm compile-failure**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --features resegmentation --lib resegmentation::resegmenter_tests 2>&1 | head -10
```

Expected: undefined `OverlapResegmenter`.

- [ ] **Step 4.3: Implement `OverlapResegmenter`**

Add to `src/resegmentation.rs` (above all the test modules, after the helpers):

```rust
/// Default-constructible overlap-aware resegmenter that picks the nearest
/// non-primary cluster centroid (by cosine similarity) for each overlap region
/// above a configurable threshold and minimum duration.
///
/// Typical usage (from `Pipeline` in M6):
///
/// ```rust,ignore
/// let r = OverlapResegmenter::default();
/// let out = r.resegment(ResegmentInputs {
///     primary_turns: &turns,
///     speaker_centroids: &centroids,
///     overlap_regions: &regions,
/// })?;
/// ```
pub struct OverlapResegmenter {
    threshold: f32,
    min_overlap_secs: f32,
}

impl OverlapResegmenter {
    /// `threshold` — minimum cosine similarity required to attach a secondary
    /// speaker to an overlap region. Default `0.0` (always attach the nearest
    /// non-primary cluster).
    /// `min_overlap_secs` — overlap regions shorter than this are skipped.
    /// Default `0.1`.
    pub fn new(threshold: f32, min_overlap_secs: f32) -> Self {
        Self {
            threshold,
            min_overlap_secs: min_overlap_secs.max(0.0),
        }
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn min_overlap_secs(&self) -> f32 {
        self.min_overlap_secs
    }
}

impl Default for OverlapResegmenter {
    fn default() -> Self {
        Self::new(0.0, 0.1)
    }
}

impl Resegmenter for OverlapResegmenter {
    fn resegment(
        &self,
        inputs: ResegmentInputs<'_>,
    ) -> Result<Vec<SpeakerTurn>, ResegmentError> {
        let mut out: Vec<SpeakerTurn> = inputs.primary_turns.to_vec();

        // Fast paths.
        if inputs.speaker_centroids.len() < 2 || inputs.overlap_regions.is_empty() {
            out.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));
            return Ok(out);
        }

        // Validate centroid dimensionality first (single-pass).
        let expected_dim = inputs.speaker_centroids[0].embedding.len();
        for (i, c) in inputs.speaker_centroids.iter().enumerate() {
            if c.embedding.len() != expected_dim {
                return Err(ResegmentError::CentroidDimMismatch {
                    index: i,
                    expected: expected_dim,
                    actual: c.embedding.len(),
                });
            }
        }

        for (i, region) in inputs.overlap_regions.iter().enumerate() {
            // Validate dim.
            if region.embedding.len() != expected_dim {
                return Err(ResegmentError::OverlapDimMismatch {
                    index: i,
                    expected: expected_dim,
                    actual: region.embedding.len(),
                });
            }
            // Validate primary present.
            if !inputs
                .speaker_centroids
                .iter()
                .any(|c| c.speaker == region.primary_speaker)
            {
                return Err(ResegmentError::MissingPrimaryCentroid {
                    index: i,
                    primary: region.primary_speaker,
                });
            }
            // Skip too-short regions.
            if (region.time.duration() as f32) < self.min_overlap_secs {
                continue;
            }
            // Find best non-primary cluster.
            let mut best: Option<(SpeakerId, f32)> = None;
            for c in inputs.speaker_centroids.iter() {
                if c.speaker == region.primary_speaker {
                    continue;
                }
                let s = crate::utils::cosine_similarity(&region.embedding, &c.embedding);
                let take = match best {
                    None => true,
                    Some((_, b)) => s > b,
                };
                if take {
                    best = Some((c.speaker, s));
                }
            }
            if let Some((id, score)) = best {
                if score > self.threshold {
                    out.push(SpeakerTurn {
                        speaker: id,
                        time: region.time,
                        text: None,
                    });
                }
            }
        }

        out.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));
        Ok(out)
    }
}
```

- [ ] **Step 4.4: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --features resegmentation --lib resegmentation::
cargo test --features resegmentation,segmentation --lib resegmentation::
cargo clippy --features resegmentation,segmentation --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib
```

Expected: ~19 tests pass (5 trait + 5 centroid + 4 overlap_extract + 10 resegmenter), all clean.

- [ ] **Step 4.5: Commit**

```bash
git add src/resegmentation.rs
git commit -m "feat(resegmentation): add OverlapResegmenter cosine-matching impl"
```

---

## Task 5: lib.rs re-exports + integration test + miri test + CHANGELOG + tag

**Files:**
- Modify: `src/lib.rs`
- Create: `tests/resegmentation_test.rs`
- Create: `tests/miri_resegmentation.rs`
- Modify: `CHANGELOG.md`

- [ ] **Step 5.1: Add re-exports**

In `/Users/ekhodzitsky/Documents/personal/polyvoice/src/lib.rs`, after the line:

```rust
#[cfg(feature = "resegmentation")]
pub mod resegmentation;
```

append:

```rust

#[cfg(feature = "resegmentation")]
pub use resegmentation::{
    OverlapRegionInput, OverlapResegmenter, ResegmentError, ResegmentInputs, Resegmenter,
    SpeakerCentroid, compute_centroids,
};

#[cfg(all(feature = "resegmentation", feature = "segmentation"))]
pub use resegmentation::extract_overlap_time_ranges;
```

- [ ] **Step 5.2: Create integration test**

Write `/Users/ekhodzitsky/Documents/personal/polyvoice/tests/resegmentation_test.rs`:

```rust
//! Integration test for the M4 OverlapResegmenter on synthetic data.
//! Pure-CPU; runs in normal `cargo test` (no model required).

#![cfg(feature = "resegmentation")]

use polyvoice::resegmentation::{
    OverlapRegionInput, OverlapResegmenter, ResegmentInputs, Resegmenter, SpeakerCentroid,
    compute_centroids,
};
use polyvoice::types::{SpeakerId, SpeakerTurn, TimeRange};

fn unit(dim: usize, axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    v[axis] = 1.0;
    v
}

fn turn(start: f64, end: f64, spk: u32) -> SpeakerTurn {
    SpeakerTurn {
        speaker: SpeakerId(spk),
        time: TimeRange { start, end },
        text: None,
    }
}

#[test]
fn end_to_end_synthetic_two_speakers_overlap() {
    // Two speakers, one overlap region. Embeddings are 8-d unit vectors.
    let dim = 8;
    let embeddings = vec![
        unit(dim, 0),
        unit(dim, 0),
        unit(dim, 0),
        unit(dim, 1),
        unit(dim, 1),
        unit(dim, 1),
    ];
    let labels = vec![0, 0, 0, 1, 1, 1];
    let centroids = compute_centroids(&embeddings, &labels);
    assert_eq!(centroids.len(), 2);

    let primary = vec![turn(0.0, 5.0, 0), turn(5.0, 10.0, 1)];
    // Overlap at 4.5–5.5: primary spk=0, embedding aligned with axis 1 (i.e. spk=1).
    let regions = vec![OverlapRegionInput {
        time: TimeRange { start: 4.5, end: 5.5 },
        primary_speaker: SpeakerId(0),
        embedding: unit(dim, 1),
    }];

    let r = OverlapResegmenter::default();
    let out = r
        .resegment(ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        })
        .unwrap();
    assert_eq!(out.len(), 3, "primary 2 + secondary 1");
    let secondary = out
        .iter()
        .find(|t| (t.time.start - 4.5).abs() < 1e-6 && (t.time.end - 5.5).abs() < 1e-6)
        .expect("secondary turn at 4.5..5.5 missing");
    assert_eq!(secondary.speaker, SpeakerId(1));
}

#[test]
fn end_to_end_three_speakers_two_pairs() {
    let dim = 8;
    let embeddings = vec![
        unit(dim, 0),
        unit(dim, 0),
        unit(dim, 1),
        unit(dim, 1),
        unit(dim, 2),
        unit(dim, 2),
    ];
    let labels = vec![0, 0, 1, 1, 2, 2];
    let centroids = compute_centroids(&embeddings, &labels);
    assert_eq!(centroids.len(), 3);

    let primary = vec![turn(0.0, 2.0, 0), turn(2.0, 4.0, 1), turn(4.0, 6.0, 2)];
    let regions = vec![
        // 1.0..2.0: primary 0, secondary best should be 1.
        OverlapRegionInput {
            time: TimeRange { start: 1.0, end: 2.0 },
            primary_speaker: SpeakerId(0),
            embedding: unit(dim, 1),
        },
        // 4.0..5.0: primary 2, secondary best should be 1.
        OverlapRegionInput {
            time: TimeRange { start: 4.0, end: 5.0 },
            primary_speaker: SpeakerId(2),
            embedding: unit(dim, 1),
        },
    ];

    let r = OverlapResegmenter::default();
    let out = r
        .resegment(ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        })
        .unwrap();
    // Two secondaries appended.
    assert_eq!(out.len(), 5);
    let n_spk1 = out.iter().filter(|t| t.speaker == SpeakerId(1)).count();
    assert!(n_spk1 >= 2, "expected ≥2 turns for speaker 1, got {n_spk1}");
    // Sorted by start.
    for w in out.windows(2) {
        assert!(w[0].time.start <= w[1].time.start);
    }
}

#[test]
fn rttm_round_trip_preserves_overlap_turns() {
    use polyvoice::rttm::write_rttm;

    let dim = 4;
    let centroids = vec![
        SpeakerCentroid {
            speaker: SpeakerId(0),
            embedding: unit(dim, 0),
        },
        SpeakerCentroid {
            speaker: SpeakerId(1),
            embedding: unit(dim, 1),
        },
    ];
    let primary = vec![turn(0.0, 1.0, 0)];
    let regions = vec![OverlapRegionInput {
        time: TimeRange { start: 0.2, end: 0.8 },
        primary_speaker: SpeakerId(0),
        embedding: unit(dim, 1),
    }];
    let r = OverlapResegmenter::default();
    let out = r
        .resegment(ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        })
        .unwrap();
    assert_eq!(out.len(), 2);

    // RTTM writer must accept overlapping spans without error or data loss.
    let mut buf = Vec::new();
    write_rttm(&mut buf, "test", &out).expect("rttm write");
    let s = String::from_utf8(buf).unwrap();
    let n_lines = s.lines().filter(|l| l.starts_with("SPEAKER")).count();
    assert_eq!(n_lines, 2, "expected 2 SPEAKER lines, got {n_lines}: {s}");
    assert!(s.contains("SPEAKER_00"));
    assert!(s.contains("SPEAKER_01"));
}
```

- [ ] **Step 5.3: Verify integration test**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --features resegmentation --test resegmentation_test
```

Expected: 3 tests pass.

If `polyvoice::rttm::write_rttm` does not exist with that exact name, replace the call with whichever function is exported (check `src/rttm.rs` — typical names: `write`, `write_rttm`, `to_rttm`). Adjust the test to match before running.

- [ ] **Step 5.4: Create Miri test**

Write `/Users/ekhodzitsky/Documents/personal/polyvoice/tests/miri_resegmentation.rs`:

```rust
//! Miri-friendly subset of M4 resegmenter tests. Covers no-overlap pass-through,
//! single-overlap cosine matching, and centroid math. ONNX-free, deterministic.

#![cfg(feature = "resegmentation")]

use polyvoice::resegmentation::{
    OverlapRegionInput, OverlapResegmenter, ResegmentInputs, Resegmenter, SpeakerCentroid,
    compute_centroids,
};
use polyvoice::types::{SpeakerId, SpeakerTurn, TimeRange};

fn unit(dim: usize, axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    v[axis] = 1.0;
    v
}

#[test]
fn miri_resegment_no_overlap() {
    let primary = vec![SpeakerTurn {
        speaker: SpeakerId(0),
        time: TimeRange { start: 0.0, end: 1.0 },
        text: None,
    }];
    let centroids = vec![SpeakerCentroid {
        speaker: SpeakerId(0),
        embedding: unit(4, 0),
    }];
    let r = OverlapResegmenter::default();
    let out = r
        .resegment(ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &[],
        })
        .unwrap();
    assert_eq!(out, primary);
}

#[test]
fn miri_resegment_single_overlap() {
    let primary = vec![SpeakerTurn {
        speaker: SpeakerId(0),
        time: TimeRange { start: 0.0, end: 1.0 },
        text: None,
    }];
    let centroids = vec![
        SpeakerCentroid {
            speaker: SpeakerId(0),
            embedding: unit(4, 0),
        },
        SpeakerCentroid {
            speaker: SpeakerId(1),
            embedding: unit(4, 1),
        },
    ];
    let regions = vec![OverlapRegionInput {
        time: TimeRange { start: 0.0, end: 1.0 },
        primary_speaker: SpeakerId(0),
        embedding: unit(4, 1),
    }];
    let r = OverlapResegmenter::default();
    let out = r
        .resegment(ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        })
        .unwrap();
    assert_eq!(out.len(), 2);
}

#[test]
fn miri_compute_centroids() {
    let embeddings = vec![unit(4, 0), unit(4, 0), unit(4, 1), unit(4, 1)];
    let labels = vec![0, 0, 1, 1];
    let centroids = compute_centroids(&embeddings, &labels);
    assert_eq!(centroids.len(), 2);
    for c in &centroids {
        let n: f32 = c.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-3);
    }
}
```

- [ ] **Step 5.5: Verify Miri**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo +nightly miri test --features resegmentation --test miri_resegmentation 2>&1 | tail -15
```

Expected: 3 Miri tests pass. If Miri toolchain is missing locally, this step is permitted to fail with `error: toolchain 'nightly' is not installed` — CI runs Miri separately. Document the skip in the commit message instead of blocking.

- [ ] **Step 5.6: Update CHANGELOG.md**

In `/Users/ekhodzitsky/Documents/personal/polyvoice/CHANGELOG.md`, in the `## [Unreleased]` block, after M3's `### Added (M3 — Clusterer trait + NME-SC)` section, append:

```markdown

### Added (M4 — Overlap resegmenter)
- `polyvoice::resegmentation` module: `Resegmenter` trait, `ResegmentError`,
  `OverlapResegmenter` (pure-Rust post-clustering pass that attaches a second
  speaker to overlap regions via nearest-cosine cluster), `ResegmentInputs`,
  `OverlapRegionInput`, `SpeakerCentroid`, helpers `compute_centroids` and
  `extract_overlap_time_ranges` (gated `segmentation`).
- New Cargo feature `resegmentation` (in default features). Pure-Rust core,
  wasm32-clean, no `onnx` requirement.
- Integration test on synthetic two-speaker / three-speaker data + RTTM
  round-trip — runs in every PR's normal `cargo test`.
- Miri-friendly test target `tests/miri_resegmentation.rs` covering
  no-overlap, single-overlap, and centroid math paths.
```

- [ ] **Step 5.7: Verify full feature matrix + tests + lints**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --lib 2>&1 | tail -3
cargo test --all-features --doc 2>&1 | tail -3
cargo test --all-features --test resegmentation_test 2>&1 | tail -3
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -5
cargo fmt --check
cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib
./scripts/release-gate.sh ; echo "exit=$?"
```

Apply `cargo fmt` if `--check` fails. Apply clippy fixes (struct-update, iter_mut, needless_borrow etc.) if `--all-targets` flags test code.

- [ ] **Step 5.8: Tag**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
git tag -a m4-complete -m "M4 complete: OverlapResegmenter + helpers"
```

(Don't push.)

- [ ] **Step 5.9: Commit**

```bash
git add src/lib.rs tests/resegmentation_test.rs tests/miri_resegmentation.rs CHANGELOG.md
git commit -m "feat(lib): re-export resegmentation surface + integration/miri tests + changelog"
```

- [ ] **Step 5.10: Final git log**

```bash
git log --oneline 53afbf8..HEAD
```

Should show 6 commits (1 per task + the wiring/tests commit).

---

## Self-review checklist

1. **Spec coverage:** all M4 deliverables (Resegmenter trait, OverlapResegmenter cosine-matching impl, compute_centroids, extract_overlap_time_ranges, RTTM round-trip, Miri subset) → Tasks 2–5.
2. **Additive guarantee:** `git diff 53afbf8..HEAD -- src/clusterer.rs src/embedder.rs src/segmentation/ src/types.rs src/overlap.rs src/pipeline.rs src/utils.rs` should show ZERO changes.
3. **Wasm32 cleanness:** `resegmentation` alone (without `segmentation`) compiles to wasm32. With `segmentation` it stays wasm32-clean (segmentation pure-Rust core is wasm32-clean per M1).
4. **No `unwrap`/`expect`/`panic`** in lib non-test code (`src/resegmentation.rs` body uses only `?` and validated error types; in-test code is allowed).
5. **Test coverage:** trait (5) + centroid (5) + overlap_extract (4) + resegmenter (10) + integration (3) + miri (3) ≈ 30 tests.
6. **Atomic commits:** ~6 total — one per task plus the final wiring commit.
7. **No ONNX dependency** introduced into resegmentation.
8. **Threshold/min duration** are tunable via `OverlapResegmenter::new(threshold, min_overlap_secs)` and the defaults match the spec (`0.0` / `0.1`).

---

## Out of scope

- VBx HMM resegmentation (sliding-window posterior smoothing) — sdvинуто в v1.2 per roadmap §2.3.
- Re-running the segmenter on overlap regions — design picked the cosine-only path (spec §"Approach: Variant A").
- Wiring `OverlapResegmenter` into `Pipeline` — M6.
- Producing overlap-region embeddings via `EmbedderPool` + `apply_overlap_mask` — M6.
- Updating Python/FFI bindings to surface secondary turns — M7.
- Closing the DER baseline gate (`tests/der_baseline.json`) on VoxConverse-smoke — M5/M6 (after the Pipeline is rebuilt with new components).
- Removing legacy `src/overlap.rs::detect_overlaps` (interval-only) — M6.
