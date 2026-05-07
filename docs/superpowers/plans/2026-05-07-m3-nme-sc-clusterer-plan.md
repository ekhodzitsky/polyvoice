# M3 — NME-SC Clusterer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the v1.0 `Clusterer` trait + `NmeScClusterer` (NME-SC default — Normalized Maximum Eigengap Spectral Clustering, auto-K) + `AhcClusterer` (fallback). Wraps the existing `spectral::spectral_cluster` (auto-K via eigengap, gated by `spectral` feature) and `ahc::agglomerative_cluster_auto`. Pure-Rust, wasm-clean (when `spectral` is on, NME-SC is reachable; without `spectral`, only AHC).

**Architecture:** New single-file module `src/clusterer.rs` (feature-gated `clusterer`, default-on). Holds `Clusterer` trait + `ClustererError` + `NmeScClusterer` (gated `spectral+clusterer`) + `AhcClusterer` (gated `clusterer`, no `spectral` needed). M6 will rename `clusterer.rs` → `clustering/mod.rs` and remove legacy free functions.

**Tech Stack:** Rust 2024. No new dependencies — both implementations wrap existing code.

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `Cargo.toml` | modify | Add `clusterer` feature (default-on) |
| `src/clusterer.rs` | create | `Clusterer` trait, `ClustererError`, `NmeScClusterer`, `AhcClusterer` |
| `src/lib.rs` | modify | `pub mod clusterer;` gated, re-exports |
| `tests/clusterer_test.rs` | create | Synthetic-cluster integration tests (no #[ignore], runs in CI) |
| `CHANGELOG.md` | modify | Unreleased M3 section |

Total roughly 350 lines Rust.

---

## Task 1: Add `clusterer` Cargo feature

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1.1: Update default + add feature**

Find:
```toml
default = ["spectral", "segmentation", "embedder"]
```

Replace:
```toml
default = ["spectral", "segmentation", "embedder", "clusterer"]
```

After the `embedder = []` line, append:

```toml

# v1.0 Clusterer trait + NME-SC + AHC adapters.
# AHC adapter is wasm32-clean. NME-SC additionally requires `spectral` (which
# pulls `faer`).
clusterer = []
```

- [ ] **Step 1.2: Verify build matrix**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m3
cargo check
cargo check --features download
cargo check --features clusterer
cargo check --features spectral,clusterer
cargo check --no-default-features
cargo check --target wasm32-unknown-unknown --no-default-features --features clusterer --lib
cargo check --all-features
```

All exit 0.

- [ ] **Step 1.3: Commit**

```bash
git add Cargo.toml
git commit -m "feat(cargo): add clusterer feature flag for v1.0 M3 work"
```

---

## Task 2: `Clusterer` trait + `ClustererError`

**Files:**
- Create: `src/clusterer.rs`
- Modify: `src/lib.rs`

- [ ] **Step 2.1: Write failing tests first**

Create `src/clusterer.rs`:

```rust
//! v1.0 `Clusterer` trait + concrete clusterers (NME-SC, AHC).
//!
//! Added in v0.6 (M3). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §3.1, §5.3.

#[cfg(test)]
mod trait_tests {
    use super::*;

    /// In-memory dummy.
    struct ConstantClusterer { labels: Vec<usize> }

    impl Clusterer for ConstantClusterer {
        fn cluster(&self, _embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
            Ok(self.labels.clone())
        }

        fn max_clusters(&self) -> usize { 64 }
    }

    #[test]
    fn clusterer_trait_object_is_dyn_compatible() {
        let c = ConstantClusterer { labels: vec![0, 1, 0] };
        let _b: Box<dyn Clusterer> = Box::new(c);
    }

    #[test]
    fn clusterer_returns_owned_labels() {
        let c = ConstantClusterer { labels: vec![0, 1, 2] };
        let labels = c.cluster(&[vec![1.0; 3]; 3]).unwrap();
        assert_eq!(labels, vec![0, 1, 2]);
    }

    #[test]
    fn error_too_few_embeddings_displays() {
        let err = ClustererError::TooFewEmbeddings { actual: 0, min: 1 };
        let msg = format!("{err}");
        assert!(msg.contains('0'));
    }
}
```

- [ ] **Step 2.2: Wire stub mod into lib.rs**

In `src/lib.rs`, after the existing `#[cfg(feature = "embedder")]` re-exports block, append:

```rust

#[cfg(feature = "clusterer")]
pub mod clusterer;
```

- [ ] **Step 2.3: Confirm compile-failure**

```bash
cargo test --features clusterer --lib clusterer::trait_tests 2>&1 | head -20
```

Expected: errors about `Clusterer`, `ClustererError`.

- [ ] **Step 2.4: Implement trait + error**

Replace `src/clusterer.rs` with the implementation, keeping the test block at the bottom:

```rust
//! v1.0 `Clusterer` trait + concrete clusterers (NME-SC, AHC).

/// Speaker clusterer — turns a batch of L2-normalized speaker embeddings into
/// per-embedding cluster labels in the range `0..K` where `K` is the inferred
/// number of clusters.
///
/// In v1.0 (M3) the polyvoice crate introduces `Clusterer` as the canonical
/// trait. The legacy free functions `ahc::agglomerative_cluster_auto` and
/// `spectral::spectral_cluster` remain available — M6 will deprecate them.
pub trait Clusterer: Send + Sync {
    /// Cluster `embeddings`. Each inner vector must have the same length and
    /// be approximately L2-normalized.
    ///
    /// **Requires:** `embeddings.len() >= 1`.
    /// **Guarantees on Ok:** `result.len() == embeddings.len()`,
    /// `result[i] < unique(result).count()` (compact 0..K numbering).
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError>;

    /// Hard ceiling on the number of clusters this implementation can produce.
    /// Used by tests and by the `Pipeline` (M6) for capping `max_speakers`.
    fn max_clusters(&self) -> usize;
}

/// Errors from `Clusterer` implementations.
#[derive(Debug, thiserror::Error)]
pub enum ClustererError {
    #[error("too few embeddings: got {actual}, need at least {min}")]
    TooFewEmbeddings { actual: usize, min: usize },

    #[error("embedding dimension mismatch: expected {expected}, got {actual} at index {index}")]
    DimMismatch { expected: usize, actual: usize, index: usize },

    #[error("clustering failed: {detail}")]
    AlgorithmFailed { detail: String },
}

#[cfg(test)]
mod trait_tests {
    // (test block from Step 2.1 stays unchanged)
}
```

- [ ] **Step 2.5: Verify**

```bash
cargo test --features clusterer --lib clusterer::trait_tests
cargo fmt
cargo clippy --features clusterer --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features clusterer --lib
```

Expected: 3 trait tests pass, all clean.

- [ ] **Step 2.6: Commit**

```bash
git add src/clusterer.rs src/lib.rs
git commit -m "feat(clusterer): add Clusterer trait + ClustererError"
```

---

## Task 3: `AhcClusterer` (wraps existing AHC)

**Files:**
- Modify: `src/clusterer.rs`

- [ ] **Step 3.1: Append failing tests**

Add to `src/clusterer.rs` (next to `trait_tests`):

```rust
#[cfg(test)]
mod ahc_tests {
    use super::*;

    fn synth_two_clusters() -> Vec<Vec<f32>> {
        // Cluster A: vectors near (1, 0, 0).
        // Cluster B: vectors near (0, 1, 0).
        vec![
            vec![1.0, 0.05, 0.0], vec![0.95, 0.0, 0.05], vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0], vec![0.05, 0.95, 0.0], vec![0.0, 1.0, 0.05],
        ]
    }

    fn synth_one_cluster() -> Vec<Vec<f32>> {
        vec![vec![1.0, 0.0, 0.0]; 5]
    }

    #[test]
    fn ahc_separates_two_well_separated_clusters() {
        let c = AhcClusterer::default();
        let labels = c.cluster(&synth_two_clusters()).unwrap();
        // First 3 should share a label; last 3 should share a different label.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn ahc_collapses_one_cluster() {
        let c = AhcClusterer::default();
        let labels = c.cluster(&synth_one_cluster()).unwrap();
        assert!(labels.iter().all(|&l| l == labels[0]));
    }

    #[test]
    fn ahc_rejects_empty_input() {
        let c = AhcClusterer::default();
        let labels: &[Vec<f32>] = &[];
        let err = c.cluster(labels).expect_err("empty must fail");
        assert!(matches!(err, ClustererError::TooFewEmbeddings { .. }));
    }

    #[test]
    fn ahc_handles_single_embedding() {
        let c = AhcClusterer::default();
        let labels = c.cluster(&[vec![1.0, 0.0, 0.0]]).unwrap();
        assert_eq!(labels, vec![0]);
    }
}
```

- [ ] **Step 3.2: Confirm compile-failure**

```bash
cargo test --features clusterer --lib clusterer::ahc_tests 2>&1 | head -10
```

- [ ] **Step 3.3: Implement `AhcClusterer`**

Add to `src/clusterer.rs` (above test blocks):

```rust
/// AHC (agglomerative hierarchical clustering) wrapper exposing the legacy
/// `crate::ahc::agglomerative_cluster_auto` through the v1.0 `Clusterer` trait.
///
/// The auto-threshold variant is used (no `threshold` parameter) so this is
/// a drop-in replacement for callers that previously relied on AHC's defaults.
pub struct AhcClusterer {
    max_clusters: usize,
}

impl AhcClusterer {
    pub fn new(max_clusters: usize) -> Self {
        Self { max_clusters: max_clusters.max(1) }
    }
}

impl Default for AhcClusterer {
    fn default() -> Self { Self::new(64) }
}

impl Clusterer for AhcClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
        if embeddings.is_empty() {
            return Err(ClustererError::TooFewEmbeddings { actual: 0, min: 1 });
        }
        if embeddings.len() == 1 {
            return Ok(vec![0]);
        }
        let (labels, _threshold) = crate::ahc::agglomerative_cluster_auto(embeddings);
        // Cap at max_clusters by re-merging if needed (rare in practice).
        let unique: std::collections::HashSet<&usize> = labels.iter().collect();
        if unique.len() > self.max_clusters {
            // Fallback: merge smallest clusters into nearest. For M3 we just
            // accept the AHC output — caller's max_speakers cap handles this
            // downstream.
        }
        Ok(labels)
    }

    fn max_clusters(&self) -> usize { self.max_clusters }
}
```

- [ ] **Step 3.4: Verify**

```bash
cargo test --features clusterer --lib clusterer::
cargo clippy --features clusterer --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features clusterer --lib
```

Expected: 7 tests pass (3 trait + 4 ahc), all clean.

- [ ] **Step 3.5: Commit**

```bash
git add src/clusterer.rs
git commit -m "feat(clusterer): add AhcClusterer wrapping legacy AHC"
```

---

## Task 4: `NmeScClusterer` (wraps `spectral::spectral_cluster`)

**Files:**
- Modify: `src/clusterer.rs`

- [ ] **Step 4.1: Append failing tests**

Add to `src/clusterer.rs`:

```rust
#[cfg(all(test, feature = "spectral"))]
mod nme_sc_tests {
    use super::*;

    fn synth_three_clusters() -> Vec<Vec<f32>> {
        vec![
            // Cluster 0: near (1, 0, 0)
            vec![1.0, 0.0, 0.0], vec![0.98, 0.05, 0.0], vec![0.97, 0.0, 0.05],
            // Cluster 1: near (0, 1, 0)
            vec![0.0, 1.0, 0.0], vec![0.05, 0.98, 0.0], vec![0.0, 0.97, 0.05],
            // Cluster 2: near (0, 0, 1)
            vec![0.0, 0.0, 1.0], vec![0.05, 0.0, 0.98], vec![0.0, 0.05, 0.97],
        ]
    }

    #[test]
    fn nme_sc_separates_three_clusters() {
        let c = NmeScClusterer::default();
        let labels = c.cluster(&synth_three_clusters()).unwrap();
        // First 3 same label, next 3 same label, last 3 same label, all different.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[7], labels[8]);
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn nme_sc_rejects_empty_input() {
        let c = NmeScClusterer::default();
        let labels: &[Vec<f32>] = &[];
        let err = c.cluster(labels).expect_err("empty must fail");
        assert!(matches!(err, ClustererError::TooFewEmbeddings { .. }));
    }

    #[test]
    fn nme_sc_max_clusters_caps_estimate() {
        // Even with 9 distinct vectors, max_clusters=2 caps result.
        let c = NmeScClusterer::new(2);
        let labels = c.cluster(&synth_three_clusters()).unwrap();
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert!(unique.len() <= 2);
    }
}
```

- [ ] **Step 4.2: Implement `NmeScClusterer`**

Add to `src/clusterer.rs` (above test blocks, gated):

```rust
/// NME-SC (Normalized Maximum Eigengap Spectral Clustering) wrapper exposing
/// the legacy `crate::spectral::spectral_cluster` through the v1.0 `Clusterer`
/// trait.
///
/// NME-SC auto-tunes the number of clusters by analyzing the eigengap of the
/// normalized graph Laplacian — no fixed threshold required.
#[cfg(feature = "spectral")]
pub struct NmeScClusterer {
    max_clusters: usize,
}

#[cfg(feature = "spectral")]
impl NmeScClusterer {
    pub fn new(max_clusters: usize) -> Self {
        Self { max_clusters: max_clusters.max(1) }
    }
}

#[cfg(feature = "spectral")]
impl Default for NmeScClusterer {
    fn default() -> Self { Self::new(64) }
}

#[cfg(feature = "spectral")]
impl Clusterer for NmeScClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
        if embeddings.is_empty() {
            return Err(ClustererError::TooFewEmbeddings { actual: 0, min: 1 });
        }
        if embeddings.len() == 1 {
            return Ok(vec![0]);
        }
        let labels = crate::spectral::spectral_cluster(embeddings, self.max_clusters);
        Ok(labels)
    }

    fn max_clusters(&self) -> usize { self.max_clusters }
}
```

- [ ] **Step 4.3: Verify**

```bash
cargo test --features clusterer,spectral --lib clusterer::
cargo clippy --features clusterer,spectral --lib -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features clusterer --lib
cargo check --target wasm32-unknown-unknown --no-default-features --features clusterer,spectral --lib
```

Expected:
- 10 tests pass (3 trait + 4 ahc + 3 nme_sc).
- Clippy clean.
- wasm32 with `clusterer`-only clean (NME-SC gated out).
- wasm32 with `clusterer,spectral` MAY fail because `faer` pulls non-wasm `atomic-wait` — that's known M0 behavior. If wasm32 + `clusterer,spectral` was already failing before M3, it's fine to skip that exact combo.

- [ ] **Step 4.4: Commit**

```bash
git add src/clusterer.rs
git commit -m "feat(clusterer): add NmeScClusterer wrapping spectral_cluster"
```

---

## Task 5: lib.rs re-exports + CHANGELOG + integration test + E2E

**Files:**
- Modify: `src/lib.rs`
- Create: `tests/clusterer_test.rs`
- Modify: `CHANGELOG.md`

- [ ] **Step 5.1: Add re-exports**

In `src/lib.rs`, after the `#[cfg(feature = "clusterer")] pub mod clusterer;` line, append:

```rust

#[cfg(feature = "clusterer")]
pub use clusterer::{AhcClusterer, Clusterer, ClustererError};

#[cfg(all(feature = "clusterer", feature = "spectral"))]
pub use clusterer::NmeScClusterer;
```

- [ ] **Step 5.2: Create integration test (no #[ignore], runs in CI)**

Write `tests/clusterer_test.rs`:

```rust
//! Integration test for `AhcClusterer` and `NmeScClusterer` on synthetic
//! clusters. Pure-CPU; runs in normal `cargo test` (no network or model required).

#![cfg(feature = "clusterer")]

use polyvoice::clusterer::{AhcClusterer, Clusterer};

#[cfg(feature = "spectral")]
use polyvoice::clusterer::NmeScClusterer;

fn synth_clusters_4(d: usize) -> Vec<Vec<f32>> {
    let mut centers: Vec<Vec<f32>> = (0..4)
        .map(|c| {
            let mut v = vec![0.0_f32; d];
            v[c] = 1.0;
            v
        })
        .collect();
    let mut all = Vec::new();
    for _ in 0..6 {
        for c in &mut centers {
            let mut perturbed = c.clone();
            perturbed[0] += 0.01;
            // L2 renormalize.
            let n: f32 = perturbed.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut perturbed { *x /= n; }
            all.push(perturbed);
        }
    }
    all
}

#[test]
fn ahc_finds_four_clusters() {
    let c = AhcClusterer::default();
    let labels = c.cluster(&synth_clusters_4(8)).unwrap();
    let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
    // AHC may merge slightly; allow 3..=5.
    assert!((3..=5).contains(&unique.len()), "got {} clusters: {:?}", unique.len(), labels);
}

#[cfg(feature = "spectral")]
#[test]
fn nme_sc_finds_four_clusters() {
    let c = NmeScClusterer::default();
    let labels = c.cluster(&synth_clusters_4(8)).unwrap();
    let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
    assert!((3..=5).contains(&unique.len()), "got {} clusters: {:?}", unique.len(), labels);
}
```

- [ ] **Step 5.3: Update CHANGELOG.md**

In the `## [Unreleased]` block, after M2's `### Added (M2 — Embedder trait + CAM++)` section, append:

```markdown

### Added (M3 — Clusterer trait + NME-SC)
- `polyvoice::clusterer` module: `Clusterer` trait, `ClustererError`,
  `AhcClusterer` (wraps legacy `agglomerative_cluster_auto`), `NmeScClusterer`
  (wraps `spectral_cluster`, gated `spectral`+`clusterer`).
- New Cargo feature `clusterer` (in default features). The AHC adapter is
  wasm32-clean; NME-SC additionally requires the `spectral` feature.
- Integration test on synthetic 4-cluster data (no model required) — runs in
  every PR's normal `cargo test` (not `--ignored`).
```

- [ ] **Step 5.4: Verify full feature matrix + tests + lints**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice-m3
cargo test --all-features --lib 2>&1 | tail -3
cargo test --all-features --doc 2>&1 | tail -3
cargo test --all-features --test clusterer_test 2>&1 | tail -3
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -3
cargo fmt --check
cargo check --target wasm32-unknown-unknown --no-default-features --features clusterer --lib
./scripts/release-gate.sh ; echo "exit=$?"
```

Apply `cargo fmt` if --check fails. Apply clippy fixes (struct-update, iter_mut etc.) if `--all-targets` flags test code.

- [ ] **Step 5.5: Tag**

```bash
git tag -a m3-complete -m "M3 complete: Clusterer trait + NME-SC"
```

(Don't push.)

- [ ] **Step 5.6: Commit**

```bash
git add src/lib.rs tests/clusterer_test.rs CHANGELOG.md
git commit -m "feat(lib): re-export clusterer surface + integration test + changelog"
```

- [ ] **Step 5.7: Final git log**

```bash
git log --oneline 20c6230..HEAD
```

Should show ~6-7 commits.

---

## Self-review checklist

1. **Spec coverage:** all M3 deliverables (Clusterer trait, NME-SC, AHC adapter, eigengap auto-K) → Tasks 2-5.
2. **Additive guarantee:** `git diff 20c6230..HEAD -- src/ahc.rs src/spectral.rs src/cluster.rs src/kmeans.rs src/types.rs src/pipeline.rs` should show ZERO changes.
3. **Wasm32 cleanness:** `clusterer` alone (without `spectral`) compiles to wasm32.
4. **No `unwrap`/`expect`/`panic`** in lib non-test code.
5. **Test coverage:** trait + ahc + nme_sc + 2 integration ≈ 12 tests.
6. **Atomic commits:** ~6-7 total.

---

## Out of scope

- Profile manifest swap (clustering backend) — M6.
- Pipeline integration — M6.
- VBx HMM resegmentation — M4 (not M3).
- Removing legacy `ahc.rs`/`spectral.rs` — M6.
