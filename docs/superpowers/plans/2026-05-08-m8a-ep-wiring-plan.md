# M8a — EP wiring + CI matrix expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Assumes M6b PR #16 is merged into master before M8a starts.

**Goal:** Thread `PipelineConfig::execution_provider` from `Pipeline::builder().build()` through to every `ort::Session::builder()` in M1/M2 components, then add CI jobs that exercise CoreML on macOS aarch64 and XNNPACK on Linux aarch64.

**Architecture:** Single helper `pipeline::ep::apply_execution_provider` matches `ExecutionProvider` to feature-gated `ort` provider construction. Each ONNX-backed component gains a `with_execution_provider(...)` constructor that calls the helper between `Session::builder()` and `commit_from_file`. Builder threads `config.execution_provider` into Mobile/Balanced ONNX construction.

**Tech Stack:** Rust 2024, `ort` crate (existing dep with `coreml`/`nnapi`/`xnnpack`/`cuda` features). No new dependencies.

---

## File structure

| Path | Action |
|---|---|
| `src/pipeline/ep.rs` | create |
| `src/pipeline/mod.rs` | add `pub mod ep;` |
| `src/segmentation/powerset.rs` | add `with_execution_provider` ctor |
| `src/ecapa.rs` | add `with_execution_provider` to `FbankOnnxExtractor` |
| `src/embedder.rs` | add `with_execution_provider` to `CamPlusPlusExtractor` + `ResNet34Adapter` |
| `src/pipeline/builder.rs` | thread `config.execution_provider` into Mobile/Balanced build paths |
| `tests/pipeline_ep_test.rs` | create (3 tests, 2 `#[cfg(target_os)]`-gated, 1 `#[ignore]` E2E) |
| `.github/workflows/ci.yml` | add CoreML + XNNPACK jobs |
| `CHANGELOG.md` | M8a section |

Total ~250 LOC Rust + ~50 LOC YAML.

---

## Task 1: `pipeline::ep` helper module

**Files:**
- Create: `src/pipeline/ep.rs`
- Modify: `src/pipeline/mod.rs`

- [ ] **Step 1.1: Create helper**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/src/pipeline/ep.rs`:

```rust
//! M8a — Execution-provider wiring for `ort::Session::builder()`.

use crate::pipeline::config::ExecutionProvider;
use ort::session::builder::SessionBuilder;

#[derive(Debug, thiserror::Error)]
pub enum EpError {
    #[error("ONNX runtime EP registration failed: {0}")]
    Ort(#[from] ort::Error),
}

/// Register the requested execution provider on a `SessionBuilder`. CPU is a
/// no-op (ort default). Missing-feature variants log a warning and fall through
/// to CPU so a Pipeline built with `ExecutionProvider::Cuda` on a non-CUDA
/// build doesn't panic — it just runs on CPU.
pub fn apply_execution_provider(
    builder: SessionBuilder,
    ep: ExecutionProvider,
) -> Result<SessionBuilder, EpError> {
    match ep {
        ExecutionProvider::Cpu => Ok(builder),

        #[cfg(feature = "coreml")]
        ExecutionProvider::CoreMl => Ok(builder.with_execution_providers([
            ort::execution_providers::CoreMLExecutionProvider::default().build(),
        ])?),
        #[cfg(not(feature = "coreml"))]
        ExecutionProvider::CoreMl => {
            tracing::warn!("CoreML EP requires `coreml` Cargo feature; using CPU");
            Ok(builder)
        }

        #[cfg(feature = "xnnpack")]
        ExecutionProvider::XnnPack => Ok(builder.with_execution_providers([
            ort::execution_providers::XNNPACKExecutionProvider::default().build(),
        ])?),
        #[cfg(not(feature = "xnnpack"))]
        ExecutionProvider::XnnPack => {
            tracing::warn!("XNNPACK EP requires `xnnpack` Cargo feature; using CPU");
            Ok(builder)
        }

        #[cfg(feature = "nnapi")]
        ExecutionProvider::Nnapi => Ok(builder.with_execution_providers([
            ort::execution_providers::NNAPIExecutionProvider::default().build(),
        ])?),
        #[cfg(not(feature = "nnapi"))]
        ExecutionProvider::Nnapi => {
            tracing::warn!("NNAPI EP requires `nnapi` Cargo feature; using CPU");
            Ok(builder)
        }

        #[cfg(feature = "cuda")]
        ExecutionProvider::Cuda => Ok(builder.with_execution_providers([
            ort::execution_providers::CUDAExecutionProvider::default().build(),
        ])?),
        #[cfg(not(feature = "cuda"))]
        ExecutionProvider::Cuda => {
            tracing::warn!("CUDA EP requires `cuda` Cargo feature; using CPU");
            Ok(builder)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_cpu_is_noop() {
        let builder = ort::session::Session::builder().expect("builder");
        let _ = apply_execution_provider(builder, ExecutionProvider::Cpu)
            .expect("CPU EP must always succeed");
    }
}
```

- [ ] **Step 1.2: Wire into `mod.rs`**

In `src/pipeline/mod.rs`, after the existing `pub mod` declarations, append:

```rust
pub mod ep;
pub use ep::{EpError, apply_execution_provider};
```

- [ ] **Step 1.3: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --lib pipeline::ep
cargo clippy --all-features --lib -- -D warnings 2>&1 | tail -3
```

Expected: 1 test passes, clippy clean. Note: `ort` API name may differ — if `CoreMLExecutionProvider` doesn't compile, run `cargo doc --features pipeline,coreml --open` and adjust to actual capitalization (`CoreMl`/`XnnPack`/`NnApi`).

- [ ] **Step 1.4: Commit**

```bash
git add src/pipeline/ep.rs src/pipeline/mod.rs
git commit -m "feat(ep): add pipeline::ep helper module"
```

---

## Task 2: `PowersetSegmenter::with_execution_provider`

**Files:**
- Modify: `src/segmentation/powerset.rs`

- [ ] **Step 2.1: Find existing `new` signature**

```bash
grep -n "pub fn new\|fn commit_from_file" src/segmentation/powerset.rs | head -5
```

The existing `PowersetSegmenter::new(model_path)` calls `Session::builder().commit_from_file(path)`. We add a sibling that injects EP between those two calls.

- [ ] **Step 2.2: Add `with_execution_provider` ctor**

In `src/segmentation/powerset.rs`, find:

```rust
impl PowersetSegmenter {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, SegmentationError> {
        Self::with_config(model_path, PowersetConfig::default())
    }

    pub fn with_config(...) -> Result<Self, SegmentationError> {
        // ... uses Session::builder().commit_from_file(...)
    }
}
```

Add (above `with_config` if it constructs the Session, else inside it):

```rust
impl PowersetSegmenter {
    pub fn with_execution_provider(
        model_path: impl AsRef<Path>,
        ep: crate::pipeline::config::ExecutionProvider,
    ) -> Result<Self, SegmentationError> {
        Self::with_config_and_ep(model_path, PowersetConfig::default(), ep)
    }

    pub fn with_config_and_ep(
        model_path: impl AsRef<Path>,
        config: PowersetConfig,
        ep: crate::pipeline::config::ExecutionProvider,
    ) -> Result<Self, SegmentationError> {
        let path = model_path.as_ref().to_path_buf();
        let builder = Session::builder().map_err(|e| SegmentationError::ModelIo {
            path: path.clone(),
            detail: format!("session builder failed: {e}"),
        })?;
        let builder =
            crate::pipeline::ep::apply_execution_provider(builder, ep).map_err(|e| {
                SegmentationError::ModelIo {
                    path: path.clone(),
                    detail: format!("execution provider failed: {e}"),
                }
            })?;
        let session = builder.commit_from_file(&path).map_err(|e| SegmentationError::ModelIo {
            path: path.clone(),
            detail: format!("commit_from_file failed: {e}"),
        })?;
        // ... rest matches existing `with_config`
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_owned())
            .unwrap_or_else(|| "waveform".to_owned());
        Ok(Self {
            session: std::sync::Mutex::new(session),
            input_name,
            config,
            model_path: path,
        })
    }
}
```

If existing `with_config` already constructs `session` from a `builder`, refactor it to call `with_config_and_ep(.., ExecutionProvider::Cpu)` instead of duplicating logic.

- [ ] **Step 2.3: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo build --features pipeline,onnx,segmentation
cargo clippy --features pipeline,onnx,segmentation --lib -- -D warnings 2>&1 | tail -3
```

- [ ] **Step 2.4: Commit**

```bash
git add src/segmentation/powerset.rs
git commit -m "feat(segmentation): add PowersetSegmenter::with_execution_provider"
```

---

## Task 3: `FbankOnnxExtractor` + adapters `with_execution_provider`

**Files:**
- Modify: `src/ecapa.rs`
- Modify: `src/embedder.rs`

- [ ] **Step 3.1: Add EP-aware ctor to `FbankOnnxExtractor`**

In `src/ecapa.rs`, find the existing `FbankOnnxExtractor::new(path, dim, pool_size)`. Add a sibling:

```rust
impl FbankOnnxExtractor {
    pub fn with_execution_provider(
        model_path: impl AsRef<Path>,
        embedding_dim: usize,
        pool_size: usize,
        ep: crate::pipeline::config::ExecutionProvider,
    ) -> Result<Self, EmbedderError> {
        let path = model_path.as_ref().to_path_buf();
        let mut sessions = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let builder = ort::session::Session::builder().map_err(|e| EmbedderError::ModelIo {
                path: path.clone(),
                detail: format!("session builder: {e}"),
            })?;
            let builder = crate::pipeline::ep::apply_execution_provider(builder, ep).map_err(|e| {
                EmbedderError::ModelIo {
                    path: path.clone(),
                    detail: format!("execution provider: {e}"),
                }
            })?;
            let session = builder.commit_from_file(&path).map_err(|e| EmbedderError::ModelIo {
                path: path.clone(),
                detail: format!("commit_from_file: {e}"),
            })?;
            sessions.push(session);
        }
        Ok(Self::from_sessions(sessions, embedding_dim))
    }
}
```

If `FbankOnnxExtractor::new` returns `Result<Self, _>` and uses an internal helper `from_sessions(sessions, dim)`, just call it. If not, replicate the existing constructor body but inject `apply_execution_provider` per session.

NOTE: The actual `EmbedderError` variant for IO errors might be named differently (`Legacy(String)` etc.). Adjust the error mapping to match what `FbankOnnxExtractor::new` already uses.

- [ ] **Step 3.2: Add EP-aware ctors to `CamPlusPlusExtractor` + `ResNet34Adapter`**

In `src/embedder.rs`, find the existing `CamPlusPlusExtractor::new(path, dim, pool_size)`. Add a sibling that defers to `FbankOnnxExtractor::with_execution_provider`:

```rust
impl CamPlusPlusExtractor {
    pub fn with_execution_provider(
        path: impl AsRef<Path>,
        dim: usize,
        pool_size: usize,
        ep: crate::pipeline::config::ExecutionProvider,
    ) -> Result<Self, EmbedderError> {
        let inner = FbankOnnxExtractor::with_execution_provider(path.as_ref(), dim, pool_size, ep)?;
        Ok(Self { inner, dim })
    }
}

impl ResNet34Adapter {
    pub fn with_execution_provider(
        path: impl AsRef<Path>,
        pool_size: usize,
        ep: crate::pipeline::config::ExecutionProvider,
    ) -> Result<Self, EmbedderError> {
        let inner = FbankOnnxExtractor::with_execution_provider(path.as_ref(), 256, pool_size, ep)?;
        Ok(Self { inner, dim: 256 })
    }
}
```

The struct field names (`inner`, `dim`) come from existing M2 code — verify with `grep -n "struct CamPlusPlusExtractor\|struct ResNet34Adapter" src/embedder.rs` first and adjust to actual layout.

- [ ] **Step 3.3: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo build --features pipeline,onnx,segmentation,embedder
cargo clippy --features pipeline,onnx,segmentation,embedder --lib -- -D warnings 2>&1 | tail -3
```

- [ ] **Step 3.4: Commit**

```bash
git add src/ecapa.rs src/embedder.rs
git commit -m "feat(embedder): add with_execution_provider to FbankOnnxExtractor + adapters"
```

---

## Task 4: `PipelineBuilder::build()` threads `ExecutionProvider`

**Files:**
- Modify: `src/pipeline/builder.rs`

- [ ] **Step 4.1: Find Mobile/Balanced build paths**

```bash
grep -n "PowersetSegmenter::new\|CamPlusPlusExtractor::new\|ResNet34Adapter::new" src/pipeline/builder.rs
```

The Mobile/Balanced arms in `PipelineBuilder::build()` currently call `::new(...)`.

- [ ] **Step 4.2: Replace with `with_execution_provider`**

In `src/pipeline/builder.rs`, find the Mobile/Balanced match arm. Replace each `::new(...)` with the EP-aware sibling:

```rust
let segmenter: Box<dyn Segmenter> = Box::new(
    crate::segmentation::PowersetSegmenter::with_execution_provider(
        &profile_models.segmenter_path,
        self.config.execution_provider,
    )
    .map_err(|e| ConfigError::UnknownModel {
        model_id: format!("powerset (cause: {e})"),
    })?,
);
let embedder: Box<dyn Embedder> = match self.config.profile {
    Profile::Mobile => Box::new(
        crate::embedder::CamPlusPlusExtractor::with_execution_provider(
            &profile_models.embedder_path,
            self.config.profile.embedding_dim(),
            self.config.embedder_pool_size,
            self.config.execution_provider,
        )
        .map_err(|e| ConfigError::UnknownModel {
            model_id: format!("cam_pp (cause: {e})"),
        })?,
    ),
    Profile::Balanced => Box::new(
        crate::embedder::ResNet34Adapter::with_execution_provider(
            &profile_models.embedder_path,
            self.config.embedder_pool_size,
            self.config.execution_provider,
        )
        .map_err(|e| ConfigError::UnknownModel {
            model_id: format!("resnet34 (cause: {e})"),
        })?,
    ),
    Profile::Custom => unreachable!("Profile::Custom handled above"),
};
```

- [ ] **Step 4.3: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --lib pipeline
cargo clippy --all-features --lib -- -D warnings 2>&1 | tail -3
```

Expected: existing 15 pipeline lib tests still pass; clippy clean.

- [ ] **Step 4.4: Commit**

```bash
git add src/pipeline/builder.rs
git commit -m "feat(pipeline): thread ExecutionProvider through builder.build()"
```

---

## Task 5: `tests/pipeline_ep_test.rs`

**Files:**
- Create: `tests/pipeline_ep_test.rs`

- [ ] **Step 5.1: Write the test file**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/tests/pipeline_ep_test.rs`:

```rust
//! M8a — execution-provider wiring tests.

#![cfg(all(
    feature = "pipeline",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]

use polyvoice::pipeline::ep::{EpError, apply_execution_provider};
use polyvoice::pipeline::ExecutionProvider;

#[test]
fn cpu_ep_is_a_noop() {
    let builder = ort::session::Session::builder().expect("builder");
    let result = apply_execution_provider(builder, ExecutionProvider::Cpu);
    assert!(result.is_ok());
}

#[cfg(feature = "coreml")]
#[cfg(target_os = "macos")]
#[test]
fn coreml_ep_registers_on_macos() {
    let builder = ort::session::Session::builder().expect("builder");
    let result = apply_execution_provider(builder, ExecutionProvider::CoreMl);
    assert!(result.is_ok(), "CoreML should register cleanly on macOS");
}

#[cfg(feature = "xnnpack")]
#[test]
fn xnnpack_ep_registers() {
    let builder = ort::session::Session::builder().expect("builder");
    let result = apply_execution_provider(builder, ExecutionProvider::XnnPack);
    assert!(result.is_ok(), "XNNPACK should register");
}

#[ignore = "requires cached Balanced ONNX bundle + CoreML on macOS"]
#[cfg(target_os = "macos")]
#[cfg(feature = "coreml")]
#[test]
fn e2e_balanced_with_coreml() {
    use polyvoice::models::ModelRegistry;
    use polyvoice::pipeline::{Pipeline, PipelineConfig};
    use polyvoice::types::{Profile, SampleRate};

    let registry = ModelRegistry::default().expect("registry");
    let cfg = PipelineConfig {
        profile: Profile::Balanced,
        execution_provider: ExecutionProvider::CoreMl,
        ..PipelineConfig::default()
    };
    let pipeline = Pipeline::builder()
        .config(cfg)
        .with_models_from(registry)
        .build()
        .expect("build with CoreML");
    let samples = vec![0.0_f32; 16_000];
    let sr = SampleRate::new(16_000).unwrap();
    let result = pipeline.run(&samples, sr).expect("run with CoreML");
    let _ = result; // silence assertion isn't meaningful for 1s of zeros
}
```

- [ ] **Step 5.2: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --test pipeline_ep_test
cargo clippy --all-features --tests -- -D warnings 2>&1 | tail -3
```

Expected: at least `cpu_ep_is_a_noop` passes. On macOS aarch64 with `--features pipeline,coreml`, `coreml_ep_registers_on_macos` also passes.

- [ ] **Step 5.3: Commit**

```bash
git add tests/pipeline_ep_test.rs
git commit -m "test(pipeline): add tests/pipeline_ep_test.rs (CPU + CoreML + XNNPACK + #[ignore] E2E)"
```

---

## Task 6: CI matrix + CHANGELOG + tag

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `CHANGELOG.md`

- [ ] **Step 6.1: Add CoreML + XNNPACK matrix entries**

In `.github/workflows/ci.yml`, find the existing `test` job matrix. Add two new entries (or new jobs):

```yaml
  test-coreml:
    name: test (macos-latest, --features pipeline,coreml)
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test --features pipeline,coreml --lib pipeline::ep
      - run: cargo test --features pipeline,coreml --test pipeline_ep_test cpu_ep_is_a_noop coreml_ep_registers_on_macos

  test-xnnpack:
    name: test (ubuntu-latest, --features pipeline,xnnpack)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test --features pipeline,xnnpack --lib pipeline::ep
      - run: cargo test --features pipeline,xnnpack --test pipeline_ep_test cpu_ep_is_a_noop xnnpack_ep_registers
```

If the existing `cross-aarch64-linux` job exists, extend its features list to include `xnnpack`:

```yaml
  cross-aarch64-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: aarch64-unknown-linux-gnu
      - uses: taiki-e/install-action@cargo-cross
      - run: cross check --target aarch64-unknown-linux-gnu --features pipeline,xnnpack --no-default-features
```

Adjust the YAML structure to match the existing `.github/workflows/ci.yml` schema (it may use `strategy.matrix` instead of separate jobs).

- [ ] **Step 6.2: Update CHANGELOG.md**

In `CHANGELOG.md`, after the M6b section, append:

```markdown

### Added (M8a — Execution Provider wiring)
- `polyvoice::pipeline::ep::apply_execution_provider` — single helper that
  registers the requested ONNX runtime EP on a `SessionBuilder`. Unknown
  features fall back to CPU with a `tracing::warn!` instead of panicking.
- `PowersetSegmenter::with_execution_provider`, `FbankOnnxExtractor::with_execution_provider`,
  `CamPlusPlusExtractor::with_execution_provider`, `ResNet34Adapter::with_execution_provider`
  — EP-aware constructors complementing the existing CPU-only `::new()`s.
- `PipelineBuilder::build()` now threads `config.execution_provider` into
  Mobile/Balanced ONNX construction. Custom profiles unchanged (caller
  supplies trait objects).
- `tests/pipeline_ep_test.rs` — CPU EP no-op + `#[cfg(target_os = "macos")]`
  CoreML registration + XNNPACK registration + `#[ignore]` E2E test on
  Balanced profile via CoreML.
- CI matrix expanded with macOS CoreML and Linux XNNPACK jobs.
```

- [ ] **Step 6.3: Verify all-features**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --lib 2>&1 | tail -3
cargo test --all-features --tests 2>&1 | tail -3
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -3
cargo fmt --check
cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib
bash scripts/release-gate.sh ; echo "exit=$?"
```

- [ ] **Step 6.4: Tag**

```bash
git tag -a m8a-complete -m "M8a complete: EP wiring + CI matrix expansion"
```

(Don't push tag — push after PR merge.)

- [ ] **Step 6.5: Commit**

```bash
git add .github/workflows/ci.yml CHANGELOG.md
git commit -m "ci: add macOS CoreML + Linux XNNPACK matrix jobs + CHANGELOG + tag m8a-complete"
```

- [ ] **Step 6.6: Final git log**

```bash
git log --oneline | head -8
```

Should show 6 commits since branching from M6b.

---

## Self-review checklist

1. **Spec coverage:** all 5 deliverables (helper, EP-aware ctors x3, builder hookup, integration test, CI matrix) → Tasks 1–6.
2. **Type consistency:** `apply_execution_provider(SessionBuilder, ExecutionProvider) -> Result<SessionBuilder, EpError>` is identical across all 4 callers (segmentation, ecapa, embedder x2, builder).
3. **Atomic commits:** ~6 total.
4. **No `unwrap`/`expect`/`panic`** in lib non-test code. The helper uses `tracing::warn!` for missing-feature fallthrough; never panics.
5. **Wasm32 unaffected:** `pipeline::ep` is gated behind `pipeline + onnx`; wasm32 builds without `onnx` skip the module.

---

## Out of scope (M8b)

- `cargo-ndk` Android cross-compile job
- NNAPI runtime test (Android emulator/QEMU)
- RT-bench measurements
- `target.aarch64-linux-android` toolchain pinning
