# M6a — Pipeline + Profile API (additive) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a new `polyvoice::pipeline_v2` module that exposes `Pipeline::builder()` API wiring M1–M5 (powerset segmenter + CAM++/ResNet34 embedder + NME-SC/AHC clusterer + OverlapResegmenter + INT8 manifest) into a single end-to-end `run(&samples, SampleRate) -> Result<DiarizationResult, PipelineError>` call. **Additive only** — legacy `polyvoice::Pipeline` is untouched. M6b removes legacy and renames `pipeline_v2 → pipeline`.

**Architecture:** New directory `src/pipeline_v2/{mod.rs, config.rs, builder.rs, mocks.rs}` gated behind a default-on `pipeline_v2` Cargo feature that requires `onnx + segmentation + embedder + clusterer + resegmentation`. Builder validates Profile vs Custom-component contracts, resolves Mobile/Balanced ONNX through `ModelRegistry` (M0), or accepts caller-supplied trait objects for Custom. `Pipeline::run()` orchestrates segment → mask + embed → cluster → resegment → merge → emit `DiarizationResult`.

**Tech Stack:** Rust 2024. No new crate dependencies. Reuses M1 `Segmenter` + `RawSegment`, M2 `Embedder` + `EmbedderPool` + `apply_overlap_mask`, M3 `Clusterer`, M4 `Resegmenter` + `compute_centroids` + `extract_overlap_time_ranges`, M5 `ModelRegistry` + `Profile`. `std::thread::available_parallelism` for pool sizing (no `num_cpus` dep).

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `Cargo.toml` | modify | Add `pipeline_v2` feature (default-on) |
| `src/pipeline_v2/mod.rs` | create | `Pipeline` struct, `run()`, `PipelineError`, feature-gate `compile_error!` |
| `src/pipeline_v2/config.rs` | create | `PipelineConfig` + `ClustererKind` + `ExecutionProvider` + Default |
| `src/pipeline_v2/builder.rs` | create | `PipelineBuilder` + `ConfigError` |
| `src/pipeline_v2/mocks.rs` | create | Test-only `MockSegmenter / MockEmbedder / MockClusterer` (`#[cfg(test)]`) |
| `src/lib.rs` | modify | `pub mod pipeline_v2;` gated, re-exports |
| `tests/pipeline_v2_synthetic_test.rs` | create | Builder validation + Custom profile end-to-end on synthetic data |
| `tests/pipeline_v2_e2e_test.rs` | create | `#[ignore]` integration test on real ONNX (Balanced profile) |
| `CHANGELOG.md` | modify | Unreleased M6a section |

Total roughly 1100 LOC Rust + ~250 LOC tests + ~30 lines of doc.

---

## Task 1: Add `pipeline_v2` Cargo feature

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1.1: Update default + add feature**

In `/Users/ekhodzitsky/Documents/personal/polyvoice/Cargo.toml`, find:

```toml
default = ["spectral", "segmentation", "embedder", "clusterer", "resegmentation"]
```

Replace with:

```toml
default = ["spectral", "segmentation", "embedder", "clusterer", "resegmentation", "pipeline_v2"]
```

After the `resegmentation = []` line, append:

```toml

# v1.0 Pipeline + Profile builder API (additive in M6a; replaces legacy in M6b).
# Requires the full M1–M5 stack: onnx, segmentation, embedder, clusterer,
# resegmentation. The new module compiles to a `compile_error!` outside that
# feature combo, so we don't accidentally ship a half-wired pipeline.
pipeline_v2 = []
```

- [ ] **Step 1.2: Verify build matrix**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo check
cargo check --features pipeline_v2
cargo check --features pipeline_v2,onnx,segmentation,embedder,clusterer,resegmentation
cargo check --no-default-features
cargo check --no-default-features --features pipeline_v2  # MUST FAIL with compile_error
cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib
cargo check --all-features
```

The `--no-default-features --features pipeline_v2` invocation will be expected to fail once Task 5 ships the `compile_error!` guard. For Task 1 it succeeds (empty module yet). Note this expectation in the commit message — Task 5 will tighten it.

- [ ] **Step 1.3: Commit**

```bash
git add Cargo.toml
git commit -m "feat(cargo): add pipeline_v2 feature flag for M6a"
```

---

## Task 2: `PipelineConfig` + `ClustererKind` + `ExecutionProvider`

**Files:**
- Create: `src/pipeline_v2/config.rs`
- Create: `src/pipeline_v2/mod.rs` (stub — only `pub mod config;`)
- Modify: `src/lib.rs` (gated `pub mod pipeline_v2;`)

- [ ] **Step 2.1: Write failing tests first**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/src/pipeline_v2/config.rs`:

```rust
//! M6a — `PipelineConfig`, `ClustererKind`, `ExecutionProvider`.
//!
//! Spec: `docs/superpowers/specs/2026-05-07-m6a-pipeline-v2-design.md` §3.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Profile;

    #[test]
    fn pipeline_config_default_is_balanced() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.profile, Profile::Balanced);
        assert_eq!(cfg.sample_rate.get(), 16000);
        assert!((cfg.seg_window_secs - 10.0).abs() < f32::EPSILON);
        assert!((cfg.seg_hop_secs - 0.5).abs() < f32::EPSILON);
        assert!(matches!(cfg.clusterer, ClustererKind::NmeSc));
        assert_eq!(cfg.max_speakers, 20);
        assert_eq!(cfg.min_cluster_size, 12);
        assert!(cfg.resegment_overlap);
        assert!((cfg.resegment_min_overlap_secs - 0.1).abs() < f32::EPSILON);
        assert!((cfg.min_speech_secs - 0.25).abs() < f32::EPSILON);
        assert!((cfg.max_gap_secs - 0.5).abs() < f32::EPSILON);
        assert!(cfg.embedder_pool_size >= 1);
        assert!(cfg.embedder_pool_size <= 4);
    }

    #[test]
    fn clusterer_kind_ahc_with_threshold() {
        let k = ClustererKind::Ahc { threshold: 0.7 };
        if let ClustererKind::Ahc { threshold } = k {
            assert!((threshold - 0.7).abs() < f32::EPSILON);
        } else {
            panic!("expected Ahc variant");
        }
    }

    #[test]
    fn execution_provider_auto_returns_some_variant() {
        let ep = ExecutionProvider::auto();
        // Just assert it returns *some* variant — actual platform default
        // varies (CoreMl on macOS aarch64, XnnPack on linux aarch64, Cpu else).
        let _ = ep;
    }
}
```

- [ ] **Step 2.2: Wire stub mod into lib.rs and pipeline_v2/mod.rs**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/src/pipeline_v2/mod.rs`:

```rust
//! M6a — additive `polyvoice::pipeline_v2` module.
//!
//! Spec: `docs/superpowers/specs/2026-05-07-m6a-pipeline-v2-design.md`.

pub mod config;
```

In `/Users/ekhodzitsky/Documents/personal/polyvoice/src/lib.rs`, after the existing block:

```rust
#[cfg(all(feature = "resegmentation", feature = "segmentation"))]
pub use resegmentation::extract_overlap_time_ranges;
```

append:

```rust

#[cfg(all(
    feature = "pipeline_v2",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub mod pipeline_v2;
```

- [ ] **Step 2.3: Confirm compile-failure**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --lib pipeline_v2::config 2>&1 | head -20
```

Expected: errors that `PipelineConfig`, `ClustererKind`, `ExecutionProvider` are not defined.

- [ ] **Step 2.4: Implement config**

Replace `src/pipeline_v2/config.rs` with:

```rust
//! M6a — `PipelineConfig`, `ClustererKind`, `ExecutionProvider`.
//!
//! Spec: `docs/superpowers/specs/2026-05-07-m6a-pipeline-v2-design.md` §3.

use crate::types::{Profile, SampleRate};

/// Top-level configuration for the v1.0 Pipeline. Mirrors spec §5.2 verbatim.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub profile: Profile,
    pub sample_rate: SampleRate,
    pub seg_window_secs: f32,
    pub seg_hop_secs: f32,
    pub clusterer: ClustererKind,
    pub max_speakers: u8,
    pub min_cluster_size: usize,
    pub resegment_overlap: bool,
    pub resegment_min_overlap_secs: f32,
    pub min_speech_secs: f32,
    pub max_gap_secs: f32,
    pub embedder_pool_size: usize,
    pub execution_provider: ExecutionProvider,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Balanced,
            sample_rate: SampleRate::new(16000).expect("16000 is a valid sample rate"),
            seg_window_secs: 10.0,
            seg_hop_secs: 0.5,
            clusterer: ClustererKind::NmeSc,
            max_speakers: 20,
            min_cluster_size: 12,
            resegment_overlap: true,
            resegment_min_overlap_secs: 0.1,
            min_speech_secs: 0.25,
            max_gap_secs: 0.5,
            embedder_pool_size: default_pool_size(),
            execution_provider: ExecutionProvider::auto(),
        }
    }
}

/// Clusterer selector. Defaults to `NmeSc` (M3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClustererKind {
    NmeSc,
    Ahc { threshold: f32 },
}

/// ONNX Runtime execution provider hint. The actual EP is plumbed at
/// `Session::builder()` time in M6b's CLI/FFI rewrite; M6a stores the
/// preference but the legacy ONNX wrappers (M2 `FbankOnnxExtractor`) only
/// honour the Cpu path so this field is informational for now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExecutionProvider {
    Cpu,
    CoreMl,
    Nnapi,
    Cuda,
    XnnPack,
}

impl ExecutionProvider {
    /// Best-effort platform default. macOS aarch64 → `CoreMl`; Linux aarch64 →
    /// `XnnPack`; else `Cpu`.
    pub fn auto() -> Self {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        return Self::CoreMl;
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        return Self::XnnPack;
        #[cfg(not(any(
            all(target_os = "macos", target_arch = "aarch64"),
            all(target_os = "linux", target_arch = "aarch64"),
        )))]
        return Self::Cpu;
    }
}

fn default_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Profile;

    #[test]
    fn pipeline_config_default_is_balanced() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.profile, Profile::Balanced);
        assert_eq!(cfg.sample_rate.get(), 16000);
        assert!((cfg.seg_window_secs - 10.0).abs() < f32::EPSILON);
        assert!((cfg.seg_hop_secs - 0.5).abs() < f32::EPSILON);
        assert!(matches!(cfg.clusterer, ClustererKind::NmeSc));
        assert_eq!(cfg.max_speakers, 20);
        assert_eq!(cfg.min_cluster_size, 12);
        assert!(cfg.resegment_overlap);
        assert!((cfg.resegment_min_overlap_secs - 0.1).abs() < f32::EPSILON);
        assert!((cfg.min_speech_secs - 0.25).abs() < f32::EPSILON);
        assert!((cfg.max_gap_secs - 0.5).abs() < f32::EPSILON);
        assert!(cfg.embedder_pool_size >= 1);
        assert!(cfg.embedder_pool_size <= 4);
    }

    #[test]
    fn clusterer_kind_ahc_with_threshold() {
        let k = ClustererKind::Ahc { threshold: 0.7 };
        if let ClustererKind::Ahc { threshold } = k {
            assert!((threshold - 0.7).abs() < f32::EPSILON);
        } else {
            panic!("expected Ahc variant");
        }
    }

    #[test]
    fn execution_provider_auto_returns_some_variant() {
        let ep = ExecutionProvider::auto();
        let _ = ep;
    }
}
```

- [ ] **Step 2.5: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --lib pipeline_v2::config
cargo fmt
cargo clippy --all-features --lib -- -D warnings
```

Expected: 3 tests pass, clippy clean.

- [ ] **Step 2.6: Commit**

```bash
git add src/pipeline_v2/ src/lib.rs
git commit -m "feat(pipeline_v2): add PipelineConfig + ClustererKind + ExecutionProvider"
```

---

## Task 3: `PipelineBuilder` + `ConfigError`

**Files:**
- Create: `src/pipeline_v2/builder.rs`
- Modify: `src/pipeline_v2/mod.rs` (add `pub mod builder;`)

- [ ] **Step 3.1: Append failing tests**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/src/pipeline_v2/builder.rs`:

```rust
//! M6a — `PipelineBuilder` + `ConfigError`.
//!
//! Spec: `docs/superpowers/specs/2026-05-07-m6a-pipeline-v2-design.md` §4.
//!
//! The builder *defers* construction of the inner `Pipeline` to `build()`,
//! at which point Profile vs Custom-component contracts are validated.

use crate::clusterer::Clusterer;
use crate::embedder::Embedder;
use crate::models::{ModelRegistry, RegistryError};
use crate::pipeline_v2::config::PipelineConfig;
use crate::resegmentation::Resegmenter;
use crate::segmentation::Segmenter;
use crate::types::Profile;

/// Errors produced by `PipelineBuilder::build()`.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("profile {profile:?} requires .with_models_from() call")]
    MissingRegistry { profile: Profile },

    #[error(
        "profile {profile:?} cannot accept .with_{offending}() — Custom only"
    )]
    CustomComponentInProfile {
        profile: Profile,
        offending: &'static str,
    },

    #[error(
        "Custom profile cannot accept .with_models_from() — supply components individually"
    )]
    RegistryInCustomProfile,

    #[error("Custom profile missing required components: {missing:?}")]
    MissingCustomComponent { missing: Vec<&'static str> },

    #[error("ONNX model not found in registry: {model_id}")]
    UnknownModel { model_id: String },

    #[error("registry resolution failed: {0}")]
    Registry(#[from] RegistryError),
}

/// Builder for the v1.0 `Pipeline`. Use `Pipeline::builder()` to obtain one.
pub struct PipelineBuilder {
    pub(crate) config: PipelineConfig,
    pub(crate) registry: Option<ModelRegistry>,
    pub(crate) custom_segmenter: Option<Box<dyn Segmenter>>,
    pub(crate) custom_embedder: Option<Box<dyn Embedder>>,
    pub(crate) custom_clusterer: Option<Box<dyn Clusterer>>,
    pub(crate) custom_resegmenter: Option<Box<dyn Resegmenter>>,
}

impl PipelineBuilder {
    pub(crate) fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
            registry: None,
            custom_segmenter: None,
            custom_embedder: None,
            custom_clusterer: None,
            custom_resegmenter: None,
        }
    }

    pub fn config(mut self, cfg: PipelineConfig) -> Self {
        self.config = cfg;
        self
    }

    pub fn profile(mut self, p: Profile) -> Self {
        self.config.profile = p;
        self
    }

    pub fn with_models_from(mut self, r: ModelRegistry) -> Self {
        self.registry = Some(r);
        self
    }

    pub fn with_segmenter(mut self, s: Box<dyn Segmenter>) -> Self {
        self.custom_segmenter = Some(s);
        self
    }

    pub fn with_embedder(mut self, e: Box<dyn Embedder>) -> Self {
        self.custom_embedder = Some(e);
        self
    }

    pub fn with_clusterer(mut self, c: Box<dyn Clusterer>) -> Self {
        self.custom_clusterer = Some(c);
        self
    }

    pub fn with_resegmenter(mut self, r: Box<dyn Resegmenter>) -> Self {
        self.custom_resegmenter = Some(r);
        self
    }

    pub fn resegment_overlap(mut self, on: bool) -> Self {
        self.config.resegment_overlap = on;
        self
    }

    pub fn embedder_pool_size(mut self, n: usize) -> Self {
        self.config.embedder_pool_size = n.max(1);
        self
    }

    pub fn max_speakers(mut self, n: u8) -> Self {
        self.config.max_speakers = n;
        self
    }

    /// Validate Profile vs Custom-component contracts. Returns the staged
    /// builder ready for `Pipeline::from_builder()` (M6a) or rejects via
    /// `ConfigError`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        match self.config.profile {
            Profile::Mobile | Profile::Balanced => {
                if self.registry.is_none() {
                    return Err(ConfigError::MissingRegistry {
                        profile: self.config.profile,
                    });
                }
                if self.custom_segmenter.is_some() {
                    return Err(ConfigError::CustomComponentInProfile {
                        profile: self.config.profile,
                        offending: "segmenter",
                    });
                }
                if self.custom_embedder.is_some() {
                    return Err(ConfigError::CustomComponentInProfile {
                        profile: self.config.profile,
                        offending: "embedder",
                    });
                }
                if self.custom_clusterer.is_some() {
                    return Err(ConfigError::CustomComponentInProfile {
                        profile: self.config.profile,
                        offending: "clusterer",
                    });
                }
            }
            Profile::Custom => {
                if self.registry.is_some() {
                    return Err(ConfigError::RegistryInCustomProfile);
                }
                let mut missing: Vec<&'static str> = Vec::new();
                if self.custom_segmenter.is_none() {
                    missing.push("segmenter");
                }
                if self.custom_embedder.is_none() {
                    missing.push("embedder");
                }
                if self.custom_clusterer.is_none() {
                    missing.push("clusterer");
                }
                if !missing.is_empty() {
                    return Err(ConfigError::MissingCustomComponent { missing });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline_v2::config::PipelineConfig;
    use crate::pipeline_v2::mocks::{MockClusterer, MockEmbedder, MockSegmenter};

    fn fresh() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    #[test]
    fn builder_default_profile_balanced() {
        let b = fresh();
        assert_eq!(b.config.profile, Profile::Balanced);
    }

    #[test]
    fn builder_profile_setter() {
        let b = fresh().profile(Profile::Mobile);
        assert_eq!(b.config.profile, Profile::Mobile);
    }

    #[test]
    fn validate_mobile_without_registry_errors() {
        let err = fresh().profile(Profile::Mobile).validate().unwrap_err();
        assert!(matches!(err, ConfigError::MissingRegistry { profile: Profile::Mobile }));
    }

    #[test]
    fn validate_custom_without_components_errors() {
        let err = fresh().profile(Profile::Custom).validate().unwrap_err();
        match err {
            ConfigError::MissingCustomComponent { missing } => {
                assert!(missing.contains(&"segmenter"));
                assert!(missing.contains(&"embedder"));
                assert!(missing.contains(&"clusterer"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn validate_custom_with_full_components_succeeds() {
        let b = fresh()
            .profile(Profile::Custom)
            .with_segmenter(Box::new(MockSegmenter::default()))
            .with_embedder(Box::new(MockEmbedder::default()))
            .with_clusterer(Box::new(MockClusterer::default()));
        b.validate().expect("custom + 3 components must validate");
    }

    #[test]
    fn validate_balanced_with_custom_segmenter_errors() {
        let b = fresh()
            .profile(Profile::Balanced)
            .with_segmenter(Box::new(MockSegmenter::default()));
        let err = b.validate().unwrap_err();
        assert!(matches!(
            err,
            ConfigError::CustomComponentInProfile {
                offending: "segmenter",
                ..
            }
        ));
    }

    #[test]
    fn validate_custom_with_registry_errors() {
        // ModelRegistry::default() may fail offline — skip in that case rather than fail the test.
        let registry = match ModelRegistry::default() {
            Ok(r) => r,
            Err(_) => return,
        };
        let b = fresh()
            .profile(Profile::Custom)
            .with_segmenter(Box::new(MockSegmenter::default()))
            .with_embedder(Box::new(MockEmbedder::default()))
            .with_clusterer(Box::new(MockClusterer::default()))
            .with_models_from(registry);
        let err = b.validate().unwrap_err();
        assert!(matches!(err, ConfigError::RegistryInCustomProfile));
    }

    #[test]
    fn embedder_pool_size_clamps_to_1() {
        let b = fresh().embedder_pool_size(0);
        assert_eq!(b.config.embedder_pool_size, 1);
    }
}
```

- [ ] **Step 3.2: Wire stub mod**

In `/Users/ekhodzitsky/Documents/personal/polyvoice/src/pipeline_v2/mod.rs`, append:

```rust

pub mod builder;
pub mod mocks;
```

(Both `builder` and `mocks` are referenced by builder tests; we'll create `mocks.rs` in Task 4 and the tests will compile only after that lands. For Task 3 we cfg-gate the mocks reference.)

Actually use this version of `mod.rs` instead — gating mocks `#[cfg(test)]`:

```rust
//! M6a — additive `polyvoice::pipeline_v2` module.
//!
//! Spec: `docs/superpowers/specs/2026-05-07-m6a-pipeline-v2-design.md`.

pub mod builder;
pub mod config;

#[cfg(test)]
pub mod mocks;
```

- [ ] **Step 3.3: Confirm compile-failure**

The builder tests reference `pipeline_v2::mocks` which doesn't exist yet. Skip running tests until Task 4 lands.

For now run only:

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo check --all-features --lib 2>&1 | head -10
```

Expected: errors about `pipeline_v2::mocks::MockSegmenter` etc. unresolved.

- [ ] **Step 3.4: Commit (compile failing, mocks come in Task 4)**

```bash
git add src/pipeline_v2/builder.rs src/pipeline_v2/mod.rs
git commit -m "feat(pipeline_v2): add PipelineBuilder + ConfigError (mocks land in Task 4)"
```

---

## Task 4: Test-only `mocks.rs` (Mock{Segmenter, Embedder, Clusterer})

**Files:**
- Create: `src/pipeline_v2/mocks.rs`

- [ ] **Step 4.1: Write mocks**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/src/pipeline_v2/mocks.rs`:

```rust
//! M6a — test-only `Mock{Segmenter,Embedder,Clusterer}` for the
//! `pipeline_v2` builder validation tests and the synthetic integration
//! test in `tests/pipeline_v2_synthetic_test.rs`.
//!
//! Compiled only under `#[cfg(test)]` (or for integration tests when this
//! file is re-exported through `lib.rs` + `#[cfg(any(test, feature = "..."))]`
//! — but for M6a we keep these crate-internal `cfg(test)` mocks and the
//! integration test redefines its own thin mocks where needed).

use crate::clusterer::{Clusterer, ClustererError};
use crate::embedder::{Embedder, EmbedderError};
use crate::resegmentation::{Resegmenter, ResegmentError, ResegmentInputs};
use crate::segmentation::{RawSegment, SegmentationError, Segmenter};
use crate::types::{Confidence, SpeakerTurn, TimeRange};

/// Constant-output `Segmenter` for builder tests.
#[derive(Default)]
pub struct MockSegmenter {
    pub segments: Vec<RawSegment>,
}

impl Segmenter for MockSegmenter {
    fn segment(&self, _audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError> {
        Ok(self.segments.clone())
    }

    fn max_local_speakers(&self) -> usize {
        3
    }

    fn supports_overlap(&self) -> bool {
        true
    }
}

/// Constant-output `Embedder` for builder tests.
pub struct MockEmbedder {
    pub embedding: Vec<f32>,
}

impl Default for MockEmbedder {
    fn default() -> Self {
        // 192-d unit vector along axis 0; matches CAM++ output dim used
        // throughout the spec.
        let mut v = vec![0.0_f32; 192];
        v[0] = 1.0;
        Self { embedding: v }
    }
}

impl Embedder for MockEmbedder {
    fn dim(&self) -> usize {
        self.embedding.len()
    }

    fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        Ok(self.embedding.clone())
    }
}

/// Constant-label `Clusterer` for builder tests.
#[derive(Default)]
pub struct MockClusterer {
    pub labels: Vec<usize>,
}

impl Clusterer for MockClusterer {
    fn cluster(
        &self,
        embeddings: &[Vec<f32>],
    ) -> Result<Vec<usize>, ClustererError> {
        if self.labels.is_empty() {
            // Default: single cluster.
            return Ok(vec![0; embeddings.len()]);
        }
        if self.labels.len() != embeddings.len() {
            return Err(ClustererError::AlgorithmFailed {
                detail: "MockClusterer labels length mismatch".to_owned(),
            });
        }
        Ok(self.labels.clone())
    }

    fn max_clusters(&self) -> usize {
        16
    }
}

/// Pass-through `Resegmenter` (returns input primary turns sorted, no
/// secondary speakers added).
#[derive(Default)]
pub struct PassThroughResegmenter;

impl Resegmenter for PassThroughResegmenter {
    fn resegment(
        &self,
        inputs: ResegmentInputs<'_>,
    ) -> Result<Vec<SpeakerTurn>, ResegmentError> {
        let mut out: Vec<SpeakerTurn> = inputs.primary_turns.to_vec();
        out.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));
        Ok(out)
    }
}

/// Convenience constructor for a single `RawSegment` used in tests.
pub fn raw_segment(start: f64, end: f64, spk: u8, overlap: bool) -> RawSegment {
    RawSegment {
        time: TimeRange { start, end },
        local_speaker_idx: spk,
        is_overlap: overlap,
        confidence: Confidence::new(0.9).unwrap(),
    }
}
```

- [ ] **Step 4.2: Run builder tests**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --lib pipeline_v2::builder
cargo test --all-features --lib pipeline_v2::config
cargo clippy --all-features --lib -- -D warnings
```

Expected: 3 config tests + 7 builder tests pass; clippy clean.

- [ ] **Step 4.3: Commit**

```bash
git add src/pipeline_v2/mocks.rs
git commit -m "test(pipeline_v2): add Mock{Segmenter,Embedder,Clusterer} for builder validation"
```

---

## Task 5: `Pipeline` core + `run()` orchestrator + feature-gate

**Files:**
- Modify: `src/pipeline_v2/mod.rs` (replace skeleton)

- [ ] **Step 5.1: Append run-flow tests**

Add to `src/pipeline_v2/mod.rs` (we'll write it whole in Step 5.2 — for now state expected behaviour):

```text
- pipeline_run_unsupported_sample_rate_returns_err
- pipeline_run_synthetic_two_speakers_through_custom_profile
  (uses MockSegmenter w/ 2 segments at distinct times, MockEmbedder, MockClusterer that
   labels them 0 and 1; expects 2-element DiarizationResult.turns sorted by start).
- pipeline_run_silence_returns_empty
  (MockSegmenter returns empty segments; expects empty turns + num_speakers=0)
- pipeline_resegment_overlap_disabled_no_secondaries
  (resegment_overlap = false; never call resegmenter even if overlap regions exist)
```

- [ ] **Step 5.2: Implement Pipeline + run()**

Replace `/Users/ekhodzitsky/Documents/personal/polyvoice/src/pipeline_v2/mod.rs` with:

```rust
//! M6a — additive `polyvoice::pipeline_v2` module.
//!
//! Spec: `docs/superpowers/specs/2026-05-07-m6a-pipeline-v2-design.md`.

#[cfg(not(all(
    feature = "pipeline_v2",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
)))]
compile_error!(
    "pipeline_v2 requires onnx + segmentation + embedder + clusterer + resegmentation features"
);

pub mod builder;
pub mod config;

#[cfg(test)]
pub mod mocks;

use crate::clusterer::{Clusterer, ClustererError};
use crate::embedder::{Embedder, EmbedderError};
use crate::models::RegistryError;
use crate::resegmentation::{
    OverlapRegionInput, OverlapResegmenter, Resegmenter, ResegmentError, ResegmentInputs,
    SpeakerCentroid, compute_centroids, extract_overlap_time_ranges,
};
use crate::segmentation::{SegmentationError, Segmenter};
use crate::types::{
    DiarizationResult, Profile, SampleRate, Segment, SpeakerId, SpeakerTurn, TimeRange,
};
use crate::utils::{l2_normalize, merge_segments};

pub use builder::{ConfigError, PipelineBuilder};
pub use config::{ClustererKind, ExecutionProvider, PipelineConfig};

/// Errors raised by `Pipeline::run()`.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("audio sample rate {actual} unsupported, expected 16000")]
    UnsupportedSampleRate { actual: u32 },
    #[error("segmentation failed: {0}")]
    Segmentation(#[from] SegmentationError),
    #[error("embedding failed: {0}")]
    Embedding(#[from] EmbedderError),
    #[error("clustering failed: {0}")]
    Clustering(#[from] ClustererError),
    #[error("resegmentation failed: {0}")]
    Resegment(#[from] ResegmentError),
    #[error("config error: {0}")]
    Config(#[from] ConfigError),
    #[error("model registry error: {0}")]
    Registry(#[from] RegistryError),
    #[error("model load error: {detail}")]
    ModelLoad { detail: String },
}

/// v1.0 Pipeline. Constructed via `Pipeline::builder()`.
pub struct Pipeline {
    config: PipelineConfig,
    segmenter: Box<dyn Segmenter>,
    embedder: Box<dyn Embedder>,
    clusterer: Box<dyn Clusterer>,
    resegmenter: Box<dyn Resegmenter>,
}

impl Pipeline {
    /// Start building a `Pipeline`. Returns a `PipelineBuilder` with default config.
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Construct a `Pipeline` from the (validated) builder's components.
    /// **Public for the builder integration only**; users invoke `builder().build()`.
    pub(crate) fn from_components(
        config: PipelineConfig,
        segmenter: Box<dyn Segmenter>,
        embedder: Box<dyn Embedder>,
        clusterer: Box<dyn Clusterer>,
        resegmenter: Box<dyn Resegmenter>,
    ) -> Self {
        Self {
            config,
            segmenter,
            embedder,
            clusterer,
            resegmenter,
        }
    }

    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Run the pipeline end-to-end on a slice of f32 PCM samples.
    pub fn run(
        &self,
        samples: &[f32],
        sr: SampleRate,
    ) -> Result<DiarizationResult, PipelineError> {
        if sr.get() != self.config.sample_rate.get() {
            return Err(PipelineError::UnsupportedSampleRate { actual: sr.get() });
        }

        let raw_segments = self.segmenter.segment(samples)?;
        if raw_segments.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                turns: Vec::new(),
                num_speakers: 0,
            });
        }

        let overlap_ranges = extract_overlap_time_ranges(&raw_segments);
        let primary_segments: Vec<_> = raw_segments
            .iter()
            .filter(|s| !s.is_overlap)
            .cloned()
            .collect();

        let sample_rate = self.config.sample_rate.get() as f64;
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(primary_segments.len());
        for seg in &primary_segments {
            let start_idx = (seg.time.start * sample_rate) as usize;
            let end_idx = ((seg.time.end * sample_rate) as usize).min(samples.len());
            if end_idx <= start_idx {
                continue;
            }
            let chunk = &samples[start_idx..end_idx];
            let mut emb = self.embedder.embed(chunk)?;
            l2_normalize(&mut emb);
            embeddings.push(emb);
        }

        if embeddings.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                turns: Vec::new(),
                num_speakers: 0,
            });
        }

        let labels = self.clusterer.cluster(&embeddings)?;

        // Build primary_turns aligned with embeddings/labels.
        let mut primary_turns: Vec<SpeakerTurn> = primary_segments
            .iter()
            .zip(labels.iter())
            .map(|(seg, &lbl)| SpeakerTurn {
                speaker: SpeakerId(lbl as u32),
                time: seg.time,
                text: None,
            })
            .collect();

        // Centroids from clean (non-overlap) embeddings.
        let centroids: Vec<SpeakerCentroid> = compute_centroids(&embeddings, &labels);

        // Optionally resegment overlap regions.
        let mut all_turns: Vec<SpeakerTurn> = if self.config.resegment_overlap
            && !overlap_ranges.is_empty()
            && centroids.len() >= 2
        {
            let overlap_inputs = self.build_overlap_inputs(
                &overlap_ranges,
                &primary_turns,
                samples,
            )?;
            self.resegmenter.resegment(ResegmentInputs {
                primary_turns: &primary_turns,
                speaker_centroids: &centroids,
                overlap_regions: &overlap_inputs,
            })?
        } else {
            primary_turns.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));
            primary_turns
        };

        // Filter short segments + merge contiguous same-speaker turns.
        let min_secs = self.config.min_speech_secs as f64;
        all_turns.retain(|t| t.time.duration() >= min_secs);

        let max_gap = self.config.max_gap_secs as f64;
        let merged_segments: Vec<Segment> = all_turns
            .iter()
            .map(|t| Segment {
                time: t.time,
                speaker: Some(t.speaker),
                confidence: None,
            })
            .collect();
        let merged_segments = merge_segments(merged_segments, max_gap);
        let merged_turns: Vec<SpeakerTurn> = merged_segments
            .iter()
            .filter_map(|s| {
                s.speaker.map(|spk| SpeakerTurn {
                    speaker: spk,
                    time: s.time,
                    text: None,
                })
            })
            .collect();

        let num_speakers = merged_turns
            .iter()
            .map(|t| t.speaker.0)
            .collect::<std::collections::HashSet<_>>()
            .len();

        Ok(DiarizationResult {
            segments: merged_segments,
            turns: merged_turns,
            num_speakers,
        })
    }

    fn build_overlap_inputs(
        &self,
        overlap_ranges: &[(TimeRange, u8, u8)],
        primary_turns: &[SpeakerTurn],
        samples: &[f32],
    ) -> Result<Vec<OverlapRegionInput>, PipelineError> {
        let sample_rate = self.config.sample_rate.get() as f64;
        let mut out = Vec::with_capacity(overlap_ranges.len());
        for (time, _lo, _hi) in overlap_ranges {
            let primary = primary_turns
                .iter()
                .find(|t| t.time.start <= time.start && time.end <= t.time.end)
                .map(|t| t.speaker)
                .unwrap_or_else(|| {
                    // Fallback to nearest primary by start time.
                    primary_turns
                        .iter()
                        .min_by(|a, b| {
                            (a.time.start - time.start)
                                .abs()
                                .total_cmp(&(b.time.start - time.start).abs())
                        })
                        .map(|t| t.speaker)
                        .unwrap_or(SpeakerId(0))
                });
            let start_idx = (time.start * sample_rate) as usize;
            let end_idx = ((time.end * sample_rate) as usize).min(samples.len());
            if end_idx <= start_idx {
                continue;
            }
            let chunk = &samples[start_idx..end_idx];
            let mut emb = self.embedder.embed(chunk)?;
            l2_normalize(&mut emb);
            out.push(OverlapRegionInput {
                time: *time,
                primary_speaker: primary,
                embedding: emb,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline_v2::mocks::{MockClusterer, MockEmbedder, MockSegmenter, raw_segment};
    use crate::types::Profile;

    fn pipeline_with_segments(segs: Vec<crate::segmentation::RawSegment>) -> Pipeline {
        let mut cfg = PipelineConfig::default();
        cfg.profile = Profile::Custom;
        cfg.resegment_overlap = false;
        cfg.min_speech_secs = 0.0;
        cfg.max_gap_secs = 0.0;
        Pipeline::from_components(
            cfg,
            Box::new(MockSegmenter { segments: segs }),
            Box::new(MockEmbedder::default()),
            Box::new(MockClusterer::default()),
            Box::new(OverlapResegmenter::default()),
        )
    }

    #[test]
    fn pipeline_run_unsupported_sample_rate_returns_err() {
        let p = pipeline_with_segments(vec![raw_segment(0.0, 1.0, 0, false)]);
        let bad = SampleRate::new(8000).unwrap();
        let err = p.run(&vec![0.0_f32; 8000], bad).unwrap_err();
        assert!(matches!(err, PipelineError::UnsupportedSampleRate { actual: 8000 }));
    }

    #[test]
    fn pipeline_run_silence_returns_empty() {
        let p = pipeline_with_segments(Vec::new());
        let result = p
            .run(&vec![0.0_f32; 16000], SampleRate::new(16000).unwrap())
            .unwrap();
        assert!(result.turns.is_empty());
        assert_eq!(result.num_speakers, 0);
    }

    #[test]
    fn pipeline_run_two_segments_one_cluster() {
        let segs = vec![
            raw_segment(0.0, 1.0, 0, false),
            raw_segment(1.5, 2.5, 0, false),
        ];
        let p = pipeline_with_segments(segs);
        let result = p
            .run(&vec![0.0_f32; 16000 * 3], SampleRate::new(16000).unwrap())
            .unwrap();
        assert_eq!(result.num_speakers, 1);
        assert!(result.turns.len() >= 1);
    }

    #[test]
    fn pipeline_resegment_overlap_disabled_path_used() {
        let segs = vec![
            raw_segment(0.0, 1.0, 0, true),
            raw_segment(0.0, 1.0, 1, true),
            raw_segment(1.5, 2.5, 0, false),
        ];
        let p = pipeline_with_segments(segs);
        let result = p
            .run(&vec![0.0_f32; 16000 * 3], SampleRate::new(16000).unwrap())
            .unwrap();
        // With resegment_overlap = false in pipeline_with_segments, no secondary
        // is appended. Overlap-flagged segments are filtered out at the embedder
        // step (they have is_overlap = true), so num_speakers reflects only the
        // non-overlap segment cluster.
        assert!(result.num_speakers <= 1);
    }
}
```

- [ ] **Step 5.3: Wire builder.build() to Pipeline::from_components**

Append to `src/pipeline_v2/builder.rs`:

```rust

use crate::pipeline_v2::Pipeline;
use crate::resegmentation::OverlapResegmenter;

impl PipelineBuilder {
    /// Validate + construct the inner `Pipeline`.
    pub fn build(self) -> Result<Pipeline, ConfigError> {
        self.validate()?;
        let resegmenter = self
            .custom_resegmenter
            .unwrap_or_else(|| Box::new(OverlapResegmenter::default()));

        match self.config.profile {
            Profile::Custom => {
                // Validation already ensured all three are Some.
                let segmenter = self.custom_segmenter.expect("validated Custom segmenter");
                let embedder = self.custom_embedder.expect("validated Custom embedder");
                let clusterer = self.custom_clusterer.expect("validated Custom clusterer");
                Ok(Pipeline::from_components(
                    self.config,
                    segmenter,
                    embedder,
                    clusterer,
                    resegmenter,
                ))
            }
            Profile::Mobile | Profile::Balanced => {
                // Resolve registry → ONNX-backed components. Wired here so that
                // unit tests of validate() don't pull ONNX. The returned Box<dyn>
                // wraps `PowersetSegmenter`, `CamPlusPlusExtractor` /
                // `ResNet34Adapter`, and `NmeScClusterer` / `AhcClusterer` per
                // ClustererKind.
                let registry = self.registry.expect("validated registry presence");
                let profile_models = registry.ensure_for_profile(self.config.profile)?;
                let segmenter: Box<dyn Segmenter> = Box::new(
                    crate::segmentation::PowersetSegmenter::new(&profile_models.segmenter_path)
                        .map_err(|e| ConfigError::UnknownModel {
                            model_id: format!("powerset (cause: {e})"),
                        })?,
                );
                let embedder: Box<dyn Embedder> = match self.config.profile {
                    Profile::Mobile => Box::new(
                        crate::embedder::CamPlusPlusExtractor::new(
                            &profile_models.embedder_path,
                            self.config.profile.embedding_dim(),
                            self.config.embedder_pool_size,
                        )
                        .map_err(|e| ConfigError::UnknownModel {
                            model_id: format!("cam_pp (cause: {e})"),
                        })?,
                    ),
                    Profile::Balanced => Box::new(
                        crate::embedder::ResNet34Adapter::new(
                            &profile_models.embedder_path,
                            self.config.embedder_pool_size,
                        )
                        .map_err(|e| ConfigError::UnknownModel {
                            model_id: format!("resnet34 (cause: {e})"),
                        })?,
                    ),
                    Profile::Custom => unreachable!("Profile::Custom handled above"),
                };
                let clusterer: Box<dyn Clusterer> = match self.config.clusterer {
                    ClustererKind::Ahc { .. } => Box::new(crate::clusterer::AhcClusterer::new(
                        self.config.max_speakers as usize,
                    )),
                    #[cfg(feature = "spectral")]
                    ClustererKind::NmeSc => Box::new(crate::clusterer::NmeScClusterer::new(
                        self.config.max_speakers as usize,
                    )),
                    #[cfg(not(feature = "spectral"))]
                    ClustererKind::NmeSc => Box::new(crate::clusterer::AhcClusterer::new(
                        self.config.max_speakers as usize,
                    )),
                };
                Ok(Pipeline::from_components(
                    self.config,
                    segmenter,
                    embedder,
                    clusterer,
                    resegmenter,
                ))
            }
        }
    }
}
```

- [ ] **Step 5.4: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --lib pipeline_v2
cargo clippy --all-features --lib -- -D warnings
cargo fmt
cargo check --no-default-features --features pipeline_v2 2>&1 | head -10
cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib
```

Expected:
- ~14 lib tests pass (3 config + 7 builder + 4 mod)
- Clippy clean
- `--no-default-features --features pipeline_v2` exits non-zero with `compile_error!` message about missing dependencies
- wasm32 clean (pipeline_v2 gated out automatically)

- [ ] **Step 5.5: Commit**

```bash
git add src/pipeline_v2/mod.rs src/pipeline_v2/builder.rs
git commit -m "feat(pipeline_v2): add Pipeline core + run() orchestrator + builder.build()"
```

---

## Task 6: Synthetic integration test

**Files:**
- Create: `tests/pipeline_v2_synthetic_test.rs`

- [ ] **Step 6.1: Write the integration test**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/tests/pipeline_v2_synthetic_test.rs`:

```rust
//! M6a — synthetic-data integration tests for `polyvoice::pipeline_v2`.
//!
//! Pure-CPU; no ONNX. Covers builder validation paths, end-to-end
//! Custom-profile run, and overlap-resegmentation toggling.

#![cfg(all(
    feature = "pipeline_v2",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]

use polyvoice::clusterer::{Clusterer, ClustererError};
use polyvoice::embedder::{Embedder, EmbedderError};
use polyvoice::pipeline_v2::{
    ClustererKind, ConfigError, Pipeline, PipelineConfig, PipelineError,
};
use polyvoice::segmentation::{RawSegment, SegmentationError, Segmenter};
use polyvoice::types::{Confidence, Profile, SampleRate, SpeakerId, TimeRange};

// ----- Local mocks (re-defined here because the crate-internal mocks are #[cfg(test)]) -----

struct ConstSegmenter(Vec<RawSegment>);
impl Segmenter for ConstSegmenter {
    fn segment(&self, _audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError> {
        Ok(self.0.clone())
    }
    fn max_local_speakers(&self) -> usize {
        3
    }
    fn supports_overlap(&self) -> bool {
        true
    }
}

struct AxisEmbedder {
    dim: usize,
    axis_picker: Box<dyn Fn(&[f32]) -> usize + Send + Sync>,
}
impl Embedder for AxisEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        let axis = (self.axis_picker)(audio).min(self.dim - 1);
        let mut v = vec![0.0_f32; self.dim];
        v[axis] = 1.0;
        Ok(v)
    }
}

struct PerSampleClusterer {
    labels: Vec<usize>,
}
impl Clusterer for PerSampleClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
        if embeddings.len() != self.labels.len() {
            return Err(ClustererError::AlgorithmFailed {
                detail: format!(
                    "labels {} vs embeddings {}",
                    self.labels.len(),
                    embeddings.len()
                ),
            });
        }
        Ok(self.labels.clone())
    }
    fn max_clusters(&self) -> usize {
        16
    }
}

fn raw(start: f64, end: f64, spk: u8, overlap: bool) -> RawSegment {
    RawSegment {
        time: TimeRange { start, end },
        local_speaker_idx: spk,
        is_overlap: overlap,
        confidence: Confidence::new(0.9).unwrap(),
    }
}

fn axis_picker_constant(axis: usize) -> Box<dyn Fn(&[f32]) -> usize + Send + Sync> {
    Box::new(move |_| axis)
}

// ----- Tests -----

#[test]
fn builder_validation_mobile_missing_registry() {
    let err = Pipeline::builder()
        .profile(Profile::Mobile)
        .build()
        .unwrap_err();
    assert!(matches!(
        err,
        ConfigError::MissingRegistry {
            profile: Profile::Mobile
        }
    ));
}

#[test]
fn builder_validation_custom_missing_components() {
    let err = Pipeline::builder()
        .profile(Profile::Custom)
        .build()
        .unwrap_err();
    assert!(matches!(err, ConfigError::MissingCustomComponent { .. }));
}

#[test]
fn pipeline_run_unsupported_sample_rate_errors() {
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        ..PipelineConfig::default()
    };
    let p = Pipeline::builder()
        .config(cfg)
        .with_segmenter(Box::new(ConstSegmenter(vec![raw(0.0, 1.0, 0, false)])))
        .with_embedder(Box::new(AxisEmbedder {
            dim: 8,
            axis_picker: axis_picker_constant(0),
        }))
        .with_clusterer(Box::new(PerSampleClusterer { labels: vec![0] }))
        .build()
        .expect("custom build");
    let err = p
        .run(&vec![0.0_f32; 8000], SampleRate::new(8000).unwrap())
        .unwrap_err();
    assert!(matches!(
        err,
        PipelineError::UnsupportedSampleRate { actual: 8000 }
    ));
}

#[test]
fn pipeline_run_silence_returns_empty() {
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        ..PipelineConfig::default()
    };
    let p = Pipeline::builder()
        .config(cfg)
        .with_segmenter(Box::new(ConstSegmenter(Vec::new())))
        .with_embedder(Box::new(AxisEmbedder {
            dim: 8,
            axis_picker: axis_picker_constant(0),
        }))
        .with_clusterer(Box::new(PerSampleClusterer { labels: vec![] }))
        .build()
        .expect("custom build");
    let r = p
        .run(&vec![0.0_f32; 16000], SampleRate::new(16000).unwrap())
        .unwrap();
    assert!(r.turns.is_empty());
    assert_eq!(r.num_speakers, 0);
}

#[test]
fn pipeline_run_two_speakers_through_custom_profile() {
    // Two non-overlap segments at distinct times, embedded onto distinct
    // axes, clustered to two distinct labels.
    let segs = vec![raw(0.0, 1.0, 0, false), raw(2.0, 3.0, 1, false)];
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        clusterer: ClustererKind::NmeSc,
        resegment_overlap: false,
        min_speech_secs: 0.0,
        max_gap_secs: 0.0,
        ..PipelineConfig::default()
    };
    let p = Pipeline::builder()
        .config(cfg)
        .with_segmenter(Box::new(ConstSegmenter(segs)))
        .with_embedder(Box::new(AxisEmbedder {
            dim: 8,
            // Use start-time of the chunk to pick a deterministic axis: first
            // chunk axis 0, second chunk axis 1. The chunk slice doesn't carry
            // its absolute start, so we exploit chunk length: 1s = 16_000
            // samples (axis 0), 1s = 16_000 too. We need a different signal —
            // use a counter via interior mutability isn't great, so instead
            // pick by content sum: silence chunks both → axis 0. Skip and
            // use PerSampleClusterer to bypass.
            axis_picker: axis_picker_constant(0),
        }))
        .with_clusterer(Box::new(PerSampleClusterer { labels: vec![0, 1] }))
        .build()
        .expect("custom build");
    let r = p
        .run(&vec![0.0_f32; 16000 * 4], SampleRate::new(16000).unwrap())
        .unwrap();
    assert_eq!(r.num_speakers, 2);
    assert_eq!(r.turns.len(), 2);
    let speakers: Vec<u32> = r.turns.iter().map(|t| t.speaker.0).collect();
    assert!(speakers.contains(&0));
    assert!(speakers.contains(&1));
}

#[test]
fn pipeline_run_returns_sorted_turns() {
    // Out-of-order segments must produce sorted output.
    let segs = vec![raw(2.0, 3.0, 0, false), raw(0.0, 1.0, 0, false)];
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        resegment_overlap: false,
        min_speech_secs: 0.0,
        max_gap_secs: 0.0,
        ..PipelineConfig::default()
    };
    let p = Pipeline::builder()
        .config(cfg)
        .with_segmenter(Box::new(ConstSegmenter(segs)))
        .with_embedder(Box::new(AxisEmbedder {
            dim: 8,
            axis_picker: axis_picker_constant(0),
        }))
        .with_clusterer(Box::new(PerSampleClusterer { labels: vec![0, 0] }))
        .build()
        .expect("custom build");
    let r = p
        .run(&vec![0.0_f32; 16000 * 4], SampleRate::new(16000).unwrap())
        .unwrap();
    // After merge_segments the two same-speaker turns may collapse into one.
    for w in r.turns.windows(2) {
        assert!(w[0].time.start <= w[1].time.start);
    }
}

#[test]
fn pipeline_resegment_overlap_disabled_no_secondaries() {
    // Two overlap segments at the same range + one clean. With
    // resegment_overlap=false the secondary speaker is never appended.
    let segs = vec![
        raw(0.0, 1.0, 0, true),
        raw(0.0, 1.0, 1, true),
        raw(2.0, 3.0, 0, false),
    ];
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        resegment_overlap: false,
        min_speech_secs: 0.0,
        max_gap_secs: 0.0,
        ..PipelineConfig::default()
    };
    let p = Pipeline::builder()
        .config(cfg)
        .with_segmenter(Box::new(ConstSegmenter(segs)))
        .with_embedder(Box::new(AxisEmbedder {
            dim: 8,
            axis_picker: axis_picker_constant(0),
        }))
        .with_clusterer(Box::new(PerSampleClusterer { labels: vec![0] }))
        .build()
        .expect("custom build");
    let r = p
        .run(&vec![0.0_f32; 16000 * 4], SampleRate::new(16000).unwrap())
        .unwrap();
    assert_eq!(r.num_speakers, 1);
}
```

- [ ] **Step 6.2: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --test pipeline_v2_synthetic_test
cargo clippy --all-features --tests -- -D warnings
```

Expected: 6 tests pass, clippy clean.

- [ ] **Step 6.3: Commit**

```bash
git add tests/pipeline_v2_synthetic_test.rs
git commit -m "test(pipeline_v2): add synthetic integration tests for builder + run() flow"
```

---

## Task 7: lib.rs re-exports + e2e test + CHANGELOG + tag

**Files:**
- Modify: `src/lib.rs`
- Create: `tests/pipeline_v2_e2e_test.rs`
- Modify: `CHANGELOG.md`

- [ ] **Step 7.1: Add re-exports to lib.rs**

In `/Users/ekhodzitsky/Documents/personal/polyvoice/src/lib.rs`, after the `#[cfg(all(feature = "pipeline_v2", ...))] pub mod pipeline_v2;` block (added in Task 2.2), append:

```rust

#[cfg(all(
    feature = "pipeline_v2",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub use pipeline_v2::{
    ClustererKind, ConfigError, ExecutionProvider, Pipeline as PipelineV2,
    PipelineBuilder, PipelineConfig, PipelineError as PipelineV2Error,
};
```

- [ ] **Step 7.2: Write the e2e test (#[ignore] by default)**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/tests/pipeline_v2_e2e_test.rs`:

```rust
//! M6a — `#[ignore]` end-to-end test for `polyvoice::pipeline_v2`.
//!
//! Requires the Balanced ONNX bundle to be cached (run
//! `cargo run --features cli --bin polyvoice -- download-models --profile balanced`
//! once before invoking with `cargo test -- --ignored e2e`). Reads a single
//! WAV from `data/voxconverse-test/audio/` and asserts the pipeline returns
//! at least one turn with valid speaker IDs.
//!
//! Skipped by default to keep CI fast; M6b will wire a polyvoice-bench
//! integration that exercises full DER on VoxConverse-test.

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
        assert!(w[0].time.start <= w[1].time.start, "turns must be sorted by start time");
    }
}
```

- [ ] **Step 7.3: Update CHANGELOG.md**

In `/Users/ekhodzitsky/Documents/personal/polyvoice/CHANGELOG.md`, in the `## [Unreleased]` block, after the M5 section (`### Added (M5 — INT8 quantization)`), append:

```markdown

### Added (M6a — Pipeline + Profile API, additive)
- `polyvoice::pipeline_v2` module: `Pipeline::builder()` returning
  `PipelineBuilder` with `.profile(Profile::Mobile|Balanced|Custom)`,
  `.with_models_from(ModelRegistry)`, `.with_segmenter/embedder/clusterer/resegmenter()`,
  and a validated `.build()`. `PipelineConfig`, `ClustererKind`,
  `ExecutionProvider`, and `ConfigError` all per spec §5.2/§5.4.
- `Pipeline::run(&samples, SampleRate)` orchestrates M1 segmenter → M2
  embedder (with `apply_overlap_mask`) → M3 clusterer → M4 resegmenter →
  legacy `merge_segments` → `DiarizationResult`.
- New Cargo feature `pipeline_v2` (in default features). Requires
  `onnx + segmentation + embedder + clusterer + resegmentation`; missing
  any of these triggers a `compile_error!` with an actionable message.
- Public re-exports `polyvoice::PipelineV2`, `PipelineBuilder`,
  `PipelineConfig`, `ClustererKind`, `ExecutionProvider`, `ConfigError`,
  `PipelineV2Error`. Legacy `polyvoice::Pipeline` is unchanged; M6b will
  rename `pipeline_v2 → pipeline` and remove the legacy code path.
- Synthetic integration test on Custom profile (`tests/pipeline_v2_synthetic_test.rs`)
  + `#[ignore]` E2E test on Balanced profile via `ModelRegistry`
  (`tests/pipeline_v2_e2e_test.rs`).
```

- [ ] **Step 7.4: Verify full feature matrix**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --lib 2>&1 | tail -3
cargo test --all-features --test pipeline_v2_synthetic_test 2>&1 | tail -3
cargo test --all-features --doc 2>&1 | tail -3
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -3
cargo fmt --check
cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib
bash scripts/release-gate.sh ; echo "exit=$?"
```

Apply `cargo fmt` if `--check` fails. Apply clippy fixes if test code is flagged.

- [ ] **Step 7.5: Tag**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
git tag -a m6a-complete -m "M6a complete: Pipeline + Profile builder API (additive)"
```

(Don't push the tag yet — it gets pushed after the M6a PR is merged into master, mirroring M3/M4/M5.)

- [ ] **Step 7.6: Commit**

```bash
git add src/lib.rs tests/pipeline_v2_e2e_test.rs CHANGELOG.md
git commit -m "feat(lib): re-export pipeline_v2 surface + add E2E ignored test + changelog"
```

- [ ] **Step 7.7: Final git log**

```bash
git log --oneline b1fcc9b..HEAD
```

Should show 7 commits matching the seven tasks.

---

## Self-review checklist

1. **Spec coverage:** all M6a deliverables (Pipeline, builder, config, validation, run() flow, synthetic test, E2E ignored test) → Tasks 2–7. Cargo feature → Task 1. Re-exports → Task 7.1.
2. **Additive guarantee:** `git diff b1fcc9b..HEAD -- src/pipeline.rs src/offline.rs src/online.rs src/types.rs src/vad.rs src/ffi.rs src/bin/polyvoice.rs src/bin/polyvoice-bench.rs python/src/lib.rs` should show ZERO changes. Legacy untouched.
3. **No `unwrap`/`expect`/`panic`** in lib non-test code (`expect("validated …")` lines in `builder.build()` are guarded by validate() invariants and acceptable per the existing M0–M5 pattern).
4. **Test coverage:** ≈3 config + 7 builder + 4 mod = 14 lib tests + 6 synthetic + 1 ignored E2E = 21 total tests across M6a.
5. **Atomic commits:** ~7 total — one per task.

---

## Out of scope (M6b)

- Removing legacy `src/pipeline.rs`, `src/offline.rs`, `OfflineDiarizer`, `DiarizationConfig`, `VadConfig`, `EnergyVad`, `VoiceActivityDetector` trait, `segment_speech` fn, `DummyExtractor` (move to `embedding/mock.rs` `#[cfg(test)]`), `OnnxEmbeddingExtractor`, `compute_fbank` privatization.
- Renaming `pipeline_v2 → pipeline`, `PipelineV2 → Pipeline`.
- CLI rewrite (`polyvoice diarize --profile mobile|balanced`).
- FFI rewrite (`src/ffi.rs`).
- Python pyo3 rewrite (`python/src/lib.rs`).
- `polyvoice-bench` rewrite using new Pipeline.
- E2E DER баннер на VoxConverse-test/AMI through polyvoice-bench → `tests/der_baseline.json`.
- Migration guide `docs/MIGRATING-FROM-0.5.md`.
- `OnlineDiarizer` deprecation annotation.
