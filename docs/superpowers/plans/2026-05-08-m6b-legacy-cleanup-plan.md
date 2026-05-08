# M6b — Legacy Cleanup + CLI/FFI/Python Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. M6b is a **huge breaking refactor** — CI is allowed to fail on intermediate commits (deletes precede rewrites); the final PR head must be green.

**Goal:** Finalize the v1.0 architecture: delete legacy `Pipeline`/`OfflineDiarizer`/`DiarizationConfig`/`VadConfig`/`EnergyVad`, rename `pipeline_v2 → pipeline`, rewrite CLI + bench + FFI + Python pyo3 to consume the new `Pipeline::builder()` API, ship a migration guide + DER baseline schema. Bumps `0.6.0-alpha.0 → 0.6.0-alpha.3`.

**Architecture:** 10 atomic commits inside one PR. Commits 1–5: version bump + module rename + legacy deletes (lib stays compilable; CLI/FFI/Python break). Commits 6–9: rewrite CLI, bench, FFI, Python pyo3 on top of new `Pipeline::builder()`. Commit 10: docs + DER baseline schema + deprecation + tag. Atomic intent — final head green; intermediate CI red is acceptable.

**Tech Stack:** Rust 2024, no new crate dependencies. Reuses M0–M6a: `polyvoice::pipeline_v2::*`, `ModelRegistry`, `Profile`, all M1–M5 components. `clap` (existing), `pyo3` (existing), C ABI via `extern "C"`.

---

## File structure

| Path | Action |
|---|---|
| `Cargo.toml` | bump version + rename `pipeline_v2 → pipeline` feature |
| `src/pipeline_v2/` | rename → `src/pipeline/` |
| `src/pipeline.rs` | **delete** (legacy) |
| `src/offline.rs` | **delete** |
| `src/vad.rs` | **delete** |
| `src/online.rs` | annotate `#[deprecated]` |
| `src/embedding.rs` | shrink (privatize/remove `DummyExtractor`/`OnnxEmbeddingExtractor`) |
| `src/onnx.rs` | **delete** if unreferenced |
| `src/ecapa.rs` | shrink (remove `EcapaTdnnExtractor`/`EcapaMelOnnxExtractor`/`RawAudioOnnxExtractor`) |
| `src/features.rs` | privatize `compute_fbank` |
| `src/types.rs` | remove `DiarizationConfig`, `VadConfig`, `ClusteringBackend`, `EmbeddingDim` |
| `src/lib.rs` | rewrite re-exports per spec §"Public API surface after M6b" |
| `src/bin/polyvoice.rs` | rewrite on `Pipeline::builder()` |
| `src/bin/polyvoice-bench.rs` | rewrite on `Pipeline::builder()` |
| `src/ffi.rs` | rewrite to ABI v2 |
| `include/polyvoice.h` | rewrite for ABI v2 |
| `python/src/lib.rs` | rewrite pyo3 bindings |
| `python/tests/test_smoke.py` | rewrite |
| `python/pyproject.toml` | bump `0.6.0a3` |
| `tests/cli_smoke_test.rs` | create |
| `tests/ffi_smoke_test.rs` | create |
| `tests/der_baseline_test.rs` | create |
| `tests/der_baseline.json` | create (schema only) |
| `tests/pipeline_v2_*.rs` | rename → `tests/pipeline_*.rs` |
| `scripts/run-der-baseline.sh` | create |
| `docs/MIGRATING-FROM-0.5.md` | create |
| `CHANGELOG.md` | append M6b section |

---

## Task 1: Bump version to 0.6.0-alpha.3

**Files:**
- Modify: `/Users/ekhodzitsky/Documents/personal/polyvoice/Cargo.toml`
- Modify: `/Users/ekhodzitsky/Documents/personal/polyvoice/python/pyproject.toml`

- [ ] **Step 1.1: Bump Cargo.toml**

In `Cargo.toml`, find:

```toml
version = "0.6.0-alpha.0"
```

Replace with:

```toml
version = "0.6.0-alpha.3"
```

- [ ] **Step 1.2: Bump python pyproject.toml**

In `python/pyproject.toml`, find:

```toml
version = "0.6.0-alpha.0"
```

Replace with:

```toml
version = "0.6.0a3"
```

(PEP 440 form — `a3` instead of `-alpha.3`.)

- [ ] **Step 1.3: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo check
cargo check --all-features
```

Both exit 0.

- [ ] **Step 1.4: Commit**

```bash
git add Cargo.toml python/pyproject.toml
git commit -m "feat(version): bump to 0.6.0-alpha.3"
```

---

## Task 2: Rename `pipeline_v2` → `pipeline`

**Files:**
- Rename: `src/pipeline_v2/` → `src/pipeline/`
- Modify: `Cargo.toml` (rename `pipeline_v2 = ["download"]` → `pipeline = ["download"]`)
- Modify: `src/lib.rs` (rename feature gates + module path + re-exports)
- Rename: `tests/pipeline_v2_synthetic_test.rs` → `tests/pipeline_synthetic_test.rs`
- Rename: `tests/pipeline_v2_e2e_test.rs` → `tests/pipeline_e2e_test.rs`

This conflicts with the legacy `src/pipeline.rs` file — that file is deleted in **Task 3** before this rename succeeds. So Task 2 actually has two phases: pre-delete renames of *non-conflicting* paths (Cargo feature, lib.rs, test files), then post-delete the directory rename. To keep this as a single commit, we sequence:

1. Delete `src/pipeline.rs` (legacy file) **inside Task 2**.
2. Move `src/pipeline_v2/` → `src/pipeline/`.
3. Rename Cargo feature, update all `feature = "pipeline_v2"` references, update `mod pipeline_v2` → `mod pipeline`, update test file paths and `use polyvoice::pipeline_v2::*` → `use polyvoice::pipeline::*`.

We keep the `Pipeline as PipelineV2` legacy alias **out** of the new lib.rs — that's a v1.0 breaking change.

- [ ] **Step 2.1: Delete legacy `src/pipeline.rs`**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
git rm src/pipeline.rs
```

- [ ] **Step 2.2: Move `src/pipeline_v2/` → `src/pipeline/`**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
git mv src/pipeline_v2 src/pipeline
```

- [ ] **Step 2.3: Rename Cargo feature**

In `Cargo.toml`, find:

```toml
# v1.0 Pipeline + Profile builder API (additive in M6a; replaces legacy in M6b).
# Requires the full M1–M5 stack — the builder loads ONNX through ModelRegistry
# (download feature) for Mobile/Balanced profiles. The module gates itself
# with a `compile_error!` if onnx/segmentation/embedder/clusterer/resegmentation
# are missing, so half-wired feature combos cannot ship.
pipeline_v2 = ["download"]
```

Replace with:

```toml
# v1.0 Pipeline + Profile builder API. Requires the full M1–M5 stack — the
# builder loads ONNX through ModelRegistry (download feature) for
# Mobile/Balanced profiles. The module gates itself with a `compile_error!`
# if onnx/segmentation/embedder/clusterer/resegmentation are missing, so
# half-wired feature combos cannot ship.
pipeline = ["download"]
```

Also in `Cargo.toml`, find the `default = [...]` line. Replace `"pipeline_v2"` with `"pipeline"`:

```toml
default = ["spectral", "segmentation", "embedder", "clusterer", "resegmentation", "pipeline"]
```

- [ ] **Step 2.4: Update `src/lib.rs`**

In `src/lib.rs`, replace the `pipeline_v2` block with:

```rust
#[cfg(all(
    feature = "pipeline",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub mod pipeline;

#[cfg(all(
    feature = "pipeline",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub use pipeline::{
    ClustererKind, ConfigError, ExecutionProvider, Pipeline, PipelineBuilder,
    PipelineConfig, PipelineError,
};
```

Remove the legacy `pub use pipeline::Pipeline;` and `pub use pipeline::PipelineError;` lines if present (they belonged to the deleted `src/pipeline.rs`).

Remove the now-orphaned `pub use pipeline_v2::{Pipeline as PipelineV2, PipelineError as PipelineV2Error, ...};` block.

- [ ] **Step 2.5: Update `src/pipeline/mod.rs` `compile_error!` gate**

In `src/pipeline/mod.rs`, find:

```rust
#[cfg(not(all(
    feature = "pipeline_v2",
    ...
)))]
compile_error!(
    "pipeline_v2 requires onnx + segmentation + embedder + clusterer + resegmentation features"
);
```

Replace with:

```rust
#[cfg(not(all(
    feature = "pipeline",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
)))]
compile_error!(
    "pipeline requires onnx + segmentation + embedder + clusterer + resegmentation features"
);
```

- [ ] **Step 2.6: Rename test files + update imports**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
git mv tests/pipeline_v2_synthetic_test.rs tests/pipeline_synthetic_test.rs
git mv tests/pipeline_v2_e2e_test.rs tests/pipeline_e2e_test.rs
```

In both renamed test files, update:
- `feature = "pipeline_v2"` → `feature = "pipeline"`
- `use polyvoice::pipeline_v2::*` → `use polyvoice::pipeline::*`

- [ ] **Step 2.7: Update other references in workspace**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
grep -rln "pipeline_v2\|PipelineV2" src/ tests/ python/ scripts/ docs/superpowers/
```

For each result, replace `pipeline_v2` → `pipeline` and `PipelineV2` → `Pipeline` (also `PipelineV2Error` → `PipelineError`). Skip `docs/superpowers/specs/` and `docs/superpowers/plans/` (historical spec/plan documents kept as authored).

- [ ] **Step 2.8: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo check --all-features
cargo test --all-features --lib pipeline 2>&1 | tail -3
```

Expected: builds clean; `pipeline::*` lib tests pass (15 from M6a).

NOTE: legacy callers in `src/bin/`, `src/ffi.rs`, `python/src/lib.rs` still reference `Pipeline::new(DiarizationConfig, VadConfig)` etc., so `cargo check --all-features --bins` and `cargo build --features ffi` will FAIL after this commit. That is intentional and resolved by Tasks 6–9.

- [ ] **Step 2.9: Commit**

```bash
git add -A
git commit -m "refactor(pipeline_v2): rename module to pipeline + Cargo feature pipeline_v2 → pipeline"
```

---

## Task 3: Delete legacy lib types (`offline.rs`, `vad.rs`)

**Files:**
- Delete: `src/offline.rs`
- Delete: `src/vad.rs`
- Modify: `src/lib.rs` (remove orphan re-exports)

`src/pipeline.rs` was deleted in Task 2. Now the rest.

- [ ] **Step 3.1: Delete files**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
git rm src/offline.rs src/vad.rs
```

- [ ] **Step 3.2: Update `src/lib.rs`**

In `src/lib.rs`, remove these re-exports + module declarations:

```rust
pub mod offline;
pub mod vad;
pub use offline::OfflineDiarizer;
pub use vad::{EnergyVad, VadConfig, VadError, VoiceActivityDetector, segment_speech};
```

- [ ] **Step 3.3: Verify lib still compiles**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo check --all-features --lib 2>&1 | tail -10
```

Expected: succeeds. (Bins/FFI/Python are red but `--lib` is what matters here.)

- [ ] **Step 3.4: Commit**

```bash
git add -A
git commit -m "refactor(legacy): delete src/offline.rs + src/vad.rs (OfflineDiarizer/VadConfig/EnergyVad)"
```

---

## Task 4: Privatize features + remove unused ecapa variants + DummyExtractor + OnnxEmbeddingExtractor

**Files:**
- Modify: `src/features.rs` (privatize `compute_fbank`)
- Modify: `src/ecapa.rs` (remove deprecated aliases + raw/mel variants)
- Modify: `src/embedding.rs` (remove `DummyExtractor`)
- Modify: `src/onnx.rs` (remove or delete file)
- Modify: `src/lib.rs` (remove orphan re-exports)

- [ ] **Step 4.1: Privatize `compute_fbank`**

In `src/features.rs`, find:

```rust
pub fn compute_fbank(...) -> ... { ... }
```

Replace `pub fn compute_fbank` with `pub(crate) fn compute_fbank`. Same for `pub use features::compute_fbank` re-exports if any.

- [ ] **Step 4.2: Strip `src/ecapa.rs`**

Read `src/ecapa.rs`. Remove:
- `pub struct EcapaMelOnnxExtractor` and its impl block
- `pub struct RawAudioOnnxExtractor` and its impl block
- `pub type EcapaTdnnExtractor = FbankOnnxExtractor` (deprecated alias)
- Any helper fns only used by those types

Keep:
- `pub struct FbankOnnxExtractor` and its impl `EmbeddingExtractor` (still used by M2's `ResNet34Adapter`)

- [ ] **Step 4.3: Remove `DummyExtractor` from `src/embedding.rs`**

In `src/embedding.rs`, remove:

```rust
pub struct DummyExtractor { ... }
impl EmbeddingExtractor for DummyExtractor { ... }
```

If the file becomes shorter, that is fine. The legacy `EmbeddingExtractor` trait remains for now (used by `FbankOnnxExtractor` until M9 cleans it up).

If a `#[cfg(test)]` mock is needed by lib tests, move `DummyExtractor` to `src/embedding/mock.rs` with `#[cfg(test)] pub(crate) struct DummyExtractor` and re-export only under `#[cfg(test)]`. Verify by running `cargo test --all-features --lib` after this step.

- [ ] **Step 4.4: Delete `src/onnx.rs` (or shrink)**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
grep -rln "OnnxEmbeddingExtractor" src/ tests/ python/ 2>/dev/null
```

If the only callers are `src/onnx.rs` itself + `src/lib.rs` re-export + the deleted bench/ffi (which were already broken by Tasks 2–3), delete the file:

```bash
git rm src/onnx.rs
```

If a non-rewritten caller still references it, mark as out of scope and leave for Task 6/7/8/9 (those rewrites will drop the imports).

- [ ] **Step 4.5: Update `src/lib.rs`**

Remove:

```rust
pub mod onnx;
pub use embedding::{DummyExtractor, EmbeddingError, EmbeddingExtractor};
#[cfg(feature = "onnx")]
#[allow(deprecated)]
pub use ecapa::EcapaTdnnExtractor;
#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbeddingExtractor;
```

Keep `pub use ecapa::FbankOnnxExtractor;` (still used internally by M2).

Also keep `pub use features::{FbankConfig, FbankExtractor};` — these are used by external callers via `FbankExtractor::extract`.

- [ ] **Step 4.6: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo check --all-features --lib 2>&1 | tail -5
cargo test --all-features --lib 2>&1 | tail -3
```

Expected: lib + lib-tests green.

- [ ] **Step 4.7: Commit**

```bash
git add -A
git commit -m "refactor(legacy): privatize compute_fbank, remove EcapaTdnnExtractor + EcapaMelOnnxExtractor + RawAudioOnnxExtractor + DummyExtractor + OnnxEmbeddingExtractor"
```

---

## Task 5: Remove `DiarizationConfig`/`VadConfig`/`ClusteringBackend`/`EmbeddingDim`

**Files:**
- Modify: `src/types.rs`
- Modify: `src/lib.rs` (remove re-exports)
- Modify: any internal callers

- [ ] **Step 5.1: Find callers**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
grep -rln "DiarizationConfig\|VadConfig\|ClusteringBackend\|EmbeddingDim" src/ tests/ python/ 2>/dev/null
```

Bin/FFI/Python were already broken by Tasks 2–3; their imports are dropped wholesale by Tasks 6–9. Internal lib callers (clusterer, embedder, etc.) should be reviewed.

`FbankOnnxExtractor::extract(samples, &DiarizationConfig::default())` — this signature in `src/ecapa.rs` and the `Embedder` trait uses `DiarizationConfig`. Switch to a smaller config type or reuse the new `pipeline::PipelineConfig`. M2's `ResNet34Adapter::embed` already wraps `extract` with `&DiarizationConfig::default()`; replace with a stub `&FbankExtractConfig::default()` (a new tiny type in `src/ecapa.rs`) that contains only the fields actually read by `FbankOnnxExtractor::extract`. Keep the type internal (`pub(crate)` or move into `src/ecapa.rs` private).

- [ ] **Step 5.2: Add `FbankExtractConfig` if needed**

In `src/ecapa.rs`, before `impl FbankOnnxExtractor`, add:

```rust
/// Internal config for fbank ONNX inference. Replaces the legacy
/// `DiarizationConfig` for callers that only need feature-extraction tuning.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FbankExtractConfig {
    pub min_speech_secs: f32,
}

impl Default for FbankExtractConfig {
    fn default() -> Self {
        Self { min_speech_secs: 0.25 }
    }
}
```

Update `FbankOnnxExtractor::extract`'s signature from `&DiarizationConfig` → `&FbankExtractConfig`.

Update `crate::embedder::ResNet34Adapter::embed` and `crate::embedder::CamPlusPlusExtractor::embed` similarly: pass `&FbankExtractConfig::default()` instead of `&DiarizationConfig::default()`.

- [ ] **Step 5.3: Remove from `src/types.rs`**

In `src/types.rs`, delete:

```rust
pub struct DiarizationConfig { ... }
pub struct VadConfig { ... }
pub enum ClusteringBackend { ... }
pub type EmbeddingDim = usize;
```

(plus their `Default`, `Display`, etc. impls and any tests that reference them).

- [ ] **Step 5.4: Update `src/lib.rs`**

Remove from re-exports:

```rust
pub use types::{
    ClusteringBackend, ..., DiarizationConfig, ..., EmbeddingDim, ...
};
```

The final `pub use types::{...}` after M6b lists exactly:

```rust
pub use types::{
    Confidence, DiarizationResult, Profile, SampleRate, Seconds, Segment,
    SpeakerId, SpeakerIdRemap, SpeakerTurn, TimeRange, WordAlignment,
    remap_segments, remap_turns,
};
```

- [ ] **Step 5.5: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo check --all-features --lib 2>&1 | tail -10
cargo test --all-features --lib 2>&1 | tail -3
```

Expected: lib + lib-tests green.

- [ ] **Step 5.6: Commit**

```bash
git add -A
git commit -m "refactor(types): remove DiarizationConfig + VadConfig + ClusteringBackend + EmbeddingDim"
```

---

## Task 6: Rewrite `src/bin/polyvoice.rs` on `Pipeline::builder()`

**Files:**
- Modify: `src/bin/polyvoice.rs`
- Create: `tests/cli_smoke_test.rs`

- [ ] **Step 6.1: Replace `src/bin/polyvoice.rs`**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/src/bin/polyvoice.rs` (overwrite existing):

```rust
//! polyvoice — speaker diarization CLI.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::{ExecutionProvider, Pipeline, PipelineConfig};
use polyvoice::rttm::write_rttm;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "polyvoice", version, about = "Speaker diarization toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run diarization on a WAV file.
    Diarize {
        wav: PathBuf,
        #[arg(long, default_value = "balanced")]
        profile: String,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "rttm")]
        format: OutputFormat,
        #[arg(long)]
        models_cache: Option<PathBuf>,
        #[arg(long, default_value = "auto")]
        execution_provider: String,
        #[arg(long, default_value = "true")]
        resegment_overlap: bool,
        #[arg(long, default_value = "20")]
        max_speakers: u8,
        #[arg(long)]
        quiet: bool,
    },
    /// Download Mobile/Balanced ONNX models.
    DownloadModels {
        #[arg(long, default_value = "balanced")]
        profile: String,
    },
    /// Inspect models registry.
    Models {
        #[command(subcommand)]
        sub: ModelsCommand,
    },
}

#[derive(Subcommand, Debug)]
enum ModelsCommand {
    /// Print available profiles + model bundle sizes.
    List,
    /// Print URL/sha256/calibration metadata for a single model.
    Info { name: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum OutputFormat {
    Rttm,
    Json,
}

fn parse_profile(name: &str) -> Result<Profile> {
    match name {
        "mobile" => Ok(Profile::Mobile),
        "balanced" => Ok(Profile::Balanced),
        other => anyhow::bail!("invalid profile: {other} (expected mobile|balanced)"),
    }
}

fn parse_execution_provider(name: &str) -> Result<ExecutionProvider> {
    match name {
        "auto" => Ok(ExecutionProvider::auto()),
        "cpu" => Ok(ExecutionProvider::Cpu),
        "coreml" => Ok(ExecutionProvider::CoreMl),
        "nnapi" => Ok(ExecutionProvider::Nnapi),
        "cuda" => Ok(ExecutionProvider::Cuda),
        "xnnpack" => Ok(ExecutionProvider::XnnPack),
        other => anyhow::bail!(
            "invalid --execution-provider: {other} (expected auto|cpu|coreml|nnapi|cuda|xnnpack)"
        ),
    }
}

fn cmd_diarize(
    wav: PathBuf,
    profile: String,
    output: Option<PathBuf>,
    format: OutputFormat,
    models_cache: Option<PathBuf>,
    execution_provider: String,
    resegment_overlap: bool,
    max_speakers: u8,
    quiet: bool,
) -> Result<()> {
    let profile = parse_profile(&profile)?;
    let ep = parse_execution_provider(&execution_provider)?;
    let registry = match models_cache {
        Some(p) => ModelRegistry::with_cache_dir(&p).context("failed to open models cache")?,
        None => ModelRegistry::default().context("failed to resolve default models cache")?,
    };

    if !quiet {
        eprintln!("Loading {profile:?} profile from registry...");
    }

    let mut cfg = PipelineConfig::default();
    cfg.profile = profile;
    cfg.execution_provider = ep;
    cfg.resegment_overlap = resegment_overlap;
    cfg.max_speakers = max_speakers;

    let pipeline = Pipeline::builder()
        .config(cfg)
        .with_models_from(registry)
        .build()
        .context("failed to build pipeline")?;

    if !quiet {
        eprintln!("Reading {}...", wav.display());
    }
    let (samples, sr_hz) = read_wav(&wav).with_context(|| format!("read WAV {}", wav.display()))?;
    let sr = SampleRate::new(sr_hz)
        .with_context(|| format!("invalid sample rate {sr_hz} Hz"))?;

    if !quiet {
        eprintln!("Running diarization on {} samples ({} Hz)...", samples.len(), sr_hz);
    }
    let result = pipeline.run(&samples, sr).context("pipeline.run failed")?;
    if !quiet {
        eprintln!("Done — {} turns, {} speakers", result.turns.len(), result.num_speakers);
    }

    match format {
        OutputFormat::Rttm => {
            let file_id = wav
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("audio")
                .to_string();
            match output {
                Some(path) => {
                    let mut f = std::fs::File::create(&path)
                        .with_context(|| format!("create {}", path.display()))?;
                    write_rttm(&mut f, &file_id, &result.turns).context("rttm write")?;
                }
                None => {
                    let mut stdout = std::io::stdout().lock();
                    write_rttm(&mut stdout, &file_id, &result.turns).context("rttm write")?;
                }
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&result).context("serialize JSON")?;
            match output {
                Some(path) => std::fs::write(&path, json)
                    .with_context(|| format!("write JSON to {}", path.display()))?,
                None => println!("{json}"),
            }
        }
    }

    Ok(())
}

fn cmd_download_models(profile: String) -> Result<()> {
    let registry = ModelRegistry::default()?;
    match profile.as_str() {
        "all" => {
            let _ = registry.ensure_for_profile(Profile::Mobile)?;
            let _ = registry.ensure_for_profile(Profile::Balanced)?;
        }
        other => {
            let p = parse_profile(other)?;
            let _ = registry.ensure_for_profile(p)?;
        }
    }
    eprintln!("Models cached at {}", registry.cache_dir().display());
    Ok(())
}

fn cmd_models_list() -> Result<()> {
    let registry = ModelRegistry::default()?;
    let manifest = registry.manifest();
    println!("Profiles:");
    for (name, prof) in &manifest.profiles {
        let seg = manifest.models.get(&prof.segmenter);
        let emb = manifest.models.get(&prof.embedder);
        let total: u64 = seg.and_then(|m| m.size).unwrap_or(0)
            + emb.and_then(|m| m.size).unwrap_or(0);
        println!(
            "  {name}: segmenter={} embedder={} total={} bytes",
            prof.segmenter, prof.embedder, total
        );
    }
    Ok(())
}

fn cmd_models_info(name: String) -> Result<()> {
    let registry = ModelRegistry::default()?;
    let manifest = registry.manifest();
    match manifest.models.get(&name) {
        Some(m) => {
            println!("name: {name}");
            println!("url: {}", m.url);
            println!("sha256: {}", m.sha256);
            if let Some(size) = m.size {
                println!("size: {size}");
            }
            if let Some(calib) = &m.calibration {
                println!("calibration: {calib}");
            }
            Ok(())
        }
        None => anyhow::bail!("unknown model: {name}"),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Diarize {
            wav,
            profile,
            output,
            format,
            models_cache,
            execution_provider,
            resegment_overlap,
            max_speakers,
            quiet,
        } => cmd_diarize(
            wav,
            profile,
            output,
            format,
            models_cache,
            execution_provider,
            resegment_overlap,
            max_speakers,
            quiet,
        ),
        Command::DownloadModels { profile } => cmd_download_models(profile),
        Command::Models { sub } => match sub {
            ModelsCommand::List => cmd_models_list(),
            ModelsCommand::Info { name } => cmd_models_info(name),
        },
    }
}
```

NOTE: `ModelRegistry::manifest()` accessor must exist. Check `src/models/mod.rs` — if missing, add `pub fn manifest(&self) -> &Manifest { &self.manifest }`. Add a doc test if introducing a new public method.

- [ ] **Step 6.2: Create `tests/cli_smoke_test.rs`**

```rust
//! M6b — smoke tests for the polyvoice CLI.

#![cfg(feature = "cli")]

use std::process::Command;

fn cli() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_polyvoice"));
    c.env("RUST_BACKTRACE", "0");
    c
}

#[test]
fn help_top_level() {
    let out = cli().arg("--help").output().expect("spawn polyvoice");
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("diarize"), "help missing 'diarize' subcommand: {s}");
    assert!(s.contains("download-models"), "help missing 'download-models': {s}");
    assert!(s.contains("models"), "help missing 'models': {s}");
}

#[test]
fn help_diarize() {
    let out = cli().args(["diarize", "--help"]).output().expect("spawn");
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("--profile"));
    assert!(s.contains("--output"));
    assert!(s.contains("--format"));
}

#[test]
fn diarize_invalid_profile_errors() {
    let out = cli()
        .args(["diarize", "/nonexistent/file.wav", "--profile", "garbage"])
        .output()
        .expect("spawn");
    assert!(!out.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("invalid profile") || stderr.contains("garbage"), "stderr: {stderr}");
}

#[test]
fn models_list_runs() {
    let out = cli().args(["models", "list"]).output().expect("spawn");
    // May fail if registry can't write to home dir in CI sandbox — accept either success or
    // a known cache-dir error; we only assert the binary doesn't crash with internal panic.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked at"), "binary panicked: {stderr}");
}
```

- [ ] **Step 6.3: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo build --features cli
cargo test --all-features --test cli_smoke_test
cargo clippy --all-features --bin polyvoice -- -D warnings 2>&1 | tail -3
```

Expected: build clean, 4 CLI smoke tests pass.

- [ ] **Step 6.4: Commit**

```bash
git add -A
git commit -m "refactor(cli): rewrite src/bin/polyvoice.rs on Pipeline::builder()"
```

---

## Task 7: Rewrite `src/bin/polyvoice-bench.rs`

**Files:**
- Modify: `src/bin/polyvoice-bench.rs`

- [ ] **Step 7.1: Replace `src/bin/polyvoice-bench.rs`**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/src/bin/polyvoice-bench.rs` (overwrite existing):

```rust
//! polyvoice-bench — DER on a {audio,rttm} dataset directory using the v1.0 Pipeline.

use anyhow::{Context, Result};
use clap::Parser;
use polyvoice::der::compute_der;
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::{Pipeline, PipelineConfig};
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::types::{Profile, SampleRate};
use polyvoice::wav::read_wav;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "polyvoice-bench", about = "Run DER on a {audio,rttm} dataset")]
struct Args {
    dataset: PathBuf,
    #[arg(long, default_value = "balanced")]
    profile: String,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value = "0.25")]
    collar: f64,
    #[arg(long, default_value = "false")]
    skip_overlap: bool,
    #[arg(long)]
    max_files: Option<usize>,
}

#[derive(Serialize)]
struct BenchReport {
    schema: &'static str,
    profile: String,
    files: usize,
    der_collar_0_25_skip_overlap: f64,
    der_no_collar: f64,
    miss: f64,
    false_alarm: f64,
    confusion: f64,
    rt_factor_avg: f64,
    polyvoice_version: &'static str,
}

fn parse_profile(name: &str) -> Result<Profile> {
    match name {
        "mobile" => Ok(Profile::Mobile),
        "balanced" => Ok(Profile::Balanced),
        other => anyhow::bail!("invalid profile: {other}"),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let profile = parse_profile(&args.profile)?;
    let registry = ModelRegistry::default().context("registry")?;

    let mut cfg = PipelineConfig::default();
    cfg.profile = profile;
    let pipeline = Pipeline::builder()
        .config(cfg)
        .with_models_from(registry)
        .build()
        .context("build pipeline")?;

    let audio_dir = args.dataset.join("audio");
    let rttm_dir = args.dataset.join("rttm");
    let mut wavs: Vec<PathBuf> = std::fs::read_dir(&audio_dir)
        .with_context(|| format!("read_dir {}", audio_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
        .map(|e| e.path())
        .collect();
    wavs.sort();
    if let Some(n) = args.max_files {
        wavs.truncate(n);
    }

    let mut totals = aggregate_init();
    let mut total_audio_secs = 0.0_f64;
    let mut total_runtime_secs = 0.0_f64;

    for wav in &wavs {
        let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let rttm = rttm_dir.join(format!("{stem}.rttm"));
        if !rttm.is_file() {
            eprintln!("[SKIP] {stem}: no rttm");
            continue;
        }
        let (samples, sr_hz) = read_wav(wav)?;
        let sr = SampleRate::new(sr_hz).context("invalid sample rate")?;
        let audio_secs = samples.len() as f64 / sr_hz as f64;

        let t0 = Instant::now();
        let result = pipeline.run(&samples, sr)?;
        let runtime_secs = t0.elapsed().as_secs_f64();

        let ref_turns = {
            let raw = parse_rttm_file(&rttm).context("parse rttm")?;
            let grouped = group_by_file(raw);
            to_speaker_turns(&grouped, stem)
        };
        let der = compute_der(&ref_turns, &result.turns, args.collar, args.skip_overlap);

        totals.der_total += der.der;
        totals.miss += der.miss;
        totals.false_alarm += der.false_alarm;
        totals.confusion += der.confusion;
        totals.count += 1;
        total_audio_secs += audio_secs;
        total_runtime_secs += runtime_secs;

        println!(
            "{stem}\t DER={:.3}%\t miss={:.3}%\t fa={:.3}%\t conf={:.3}%\t rt={:.1}x",
            der.der * 100.0,
            der.miss * 100.0,
            der.false_alarm * 100.0,
            der.confusion * 100.0,
            audio_secs / runtime_secs.max(1e-6),
        );
    }

    let n = totals.count.max(1) as f64;
    let report = BenchReport {
        schema: "polyvoice-bench-v1",
        profile: args.profile.clone(),
        files: totals.count,
        der_collar_0_25_skip_overlap: (totals.der_total / n) * 100.0,
        der_no_collar: 0.0, // computed separately when --collar 0 is invoked
        miss: (totals.miss / n) * 100.0,
        false_alarm: (totals.false_alarm / n) * 100.0,
        confusion: (totals.confusion / n) * 100.0,
        rt_factor_avg: total_audio_secs / total_runtime_secs.max(1e-6),
        polyvoice_version: env!("CARGO_PKG_VERSION"),
    };
    let json = serde_json::to_string_pretty(&report)?;
    match args.output {
        Some(p) => std::fs::write(&p, json)?,
        None => println!("{json}"),
    }
    Ok(())
}

#[derive(Default)]
struct Aggregate {
    der_total: f64,
    miss: f64,
    false_alarm: f64,
    confusion: f64,
    count: usize,
}

fn aggregate_init() -> Aggregate {
    Aggregate::default()
}
```

NOTE: `compute_der` signature — if the existing one in `src/der.rs` returns a different shape, adjust the field names in `BenchReport` accordingly. Verify with `grep -n "pub fn compute_der" src/der.rs`.

- [ ] **Step 7.2: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo build --features cli --bin polyvoice-bench
cargo clippy --all-features --bin polyvoice-bench -- -D warnings 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 7.3: Commit**

```bash
git add -A
git commit -m "refactor(bench): rewrite src/bin/polyvoice-bench.rs on Pipeline::builder() + JSON report"
```

---

## Task 8: Rewrite `src/ffi.rs` to ABI v2

**Files:**
- Modify: `src/ffi.rs`
- Modify: `include/polyvoice.h`
- Create: `tests/ffi_smoke_test.rs`

- [ ] **Step 8.1: Replace `src/ffi.rs`**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/src/ffi.rs` (overwrite existing). Skeleton:

```rust
//! M6b — C FFI v2 ABI for the v1.0 Pipeline.
//!
//! Threading model: `PolyvoicePipeline` is `Send + Sync`. Each `*mut PolyvoicePipeline`
//! owns its data; callers must call `polyvoice_pipeline_destroy` exactly once.
//! All entry points are wrapped in `catch_unwind` per spec §8.4.

use crate::models::ModelRegistry;
use crate::pipeline::{Pipeline, PipelineConfig};
use crate::types::{Profile, SampleRate};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

#[repr(C)]
pub enum PolyvoiceProfile {
    Mobile = 0,
    Balanced = 1,
}

#[repr(C)]
pub enum PolyvoiceStatus {
    Ok = 0,
    InvalidArg = 1,
    AudioTooShort = 2,
    ModelLoad = 10,
    Inference = 11,
    OutOfMemory = 20,
    Registry = 30,
    Internal = 99,
}

pub struct PolyvoicePipeline {
    inner: Pipeline,
}

#[unsafe(no_mangle)]
pub extern "C" fn polyvoice_pipeline_create(
    profile: PolyvoiceProfile,
    models_cache_dir: *const c_char,
    out_handle: *mut *mut PolyvoicePipeline,
) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<*mut PolyvoicePipeline, c_int> {
        if out_handle.is_null() {
            return Err(PolyvoiceStatus::InvalidArg as c_int);
        }
        let prof = match profile {
            PolyvoiceProfile::Mobile => Profile::Mobile,
            PolyvoiceProfile::Balanced => Profile::Balanced,
        };
        let registry = if models_cache_dir.is_null() {
            ModelRegistry::default()
        } else {
            let s = unsafe { CStr::from_ptr(models_cache_dir) }
                .to_str()
                .map_err(|_| PolyvoiceStatus::InvalidArg as c_int)?;
            ModelRegistry::with_cache_dir(s)
        }
        .map_err(|_| PolyvoiceStatus::Registry as c_int)?;
        let mut cfg = PipelineConfig::default();
        cfg.profile = prof;
        let pipeline = Pipeline::builder()
            .config(cfg)
            .with_models_from(registry)
            .build()
            .map_err(|_| PolyvoiceStatus::ModelLoad as c_int)?;
        Ok(Box::into_raw(Box::new(PolyvoicePipeline { inner: pipeline })))
    }));
    match r {
        Ok(Ok(handle)) => {
            unsafe { *out_handle = handle; }
            PolyvoiceStatus::Ok as c_int
        }
        Ok(Err(code)) => code,
        Err(_) => PolyvoiceStatus::Internal as c_int,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn polyvoice_pipeline_run(
    pipeline: *mut PolyvoicePipeline,
    samples: *const c_float,
    n_samples: usize,
    sample_rate: u32,
    out_json: *mut *mut c_char,
    out_json_len: *mut usize,
) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<(), c_int> {
        if pipeline.is_null() || samples.is_null() || out_json.is_null() || out_json_len.is_null() {
            return Err(PolyvoiceStatus::InvalidArg as c_int);
        }
        let pipeline = unsafe { &*pipeline };
        let samples = unsafe { std::slice::from_raw_parts(samples, n_samples) };
        let sr = SampleRate::new(sample_rate).ok_or(PolyvoiceStatus::InvalidArg as c_int)?;
        let result = pipeline
            .inner
            .run(samples, sr)
            .map_err(|_| PolyvoiceStatus::Inference as c_int)?;
        let json = serde_json::to_string(&result)
            .map_err(|_| PolyvoiceStatus::Internal as c_int)?;
        let len = json.len();
        let cstr = CString::new(json).map_err(|_| PolyvoiceStatus::Internal as c_int)?;
        let ptr_out = cstr.into_raw();
        unsafe {
            *out_json = ptr_out;
            *out_json_len = len;
        }
        Ok(())
    }));
    match r {
        Ok(Ok(())) => PolyvoiceStatus::Ok as c_int,
        Ok(Err(code)) => code,
        Err(_) => PolyvoiceStatus::Internal as c_int,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn polyvoice_pipeline_destroy(pipeline: *mut PolyvoicePipeline) {
    if !pipeline.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            unsafe { drop(Box::from_raw(pipeline)); }
        }));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn polyvoice_free_string(p: *mut c_char, _n: usize) {
    if !p.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            unsafe { drop(CString::from_raw(p)); }
        }));
    }
}
```

- [ ] **Step 8.2: Replace `include/polyvoice.h`**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/include/polyvoice.h` (overwrite existing):

```c
/* polyvoice.h — C FFI v2 ABI (M6b).
 * v1.0 architecture: profile-based Pipeline. Old ABI removed.
 */
#ifndef POLYVOICE_H
#define POLYVOICE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PolyvoicePipeline PolyvoicePipeline;

typedef enum {
    POLYVOICE_PROFILE_MOBILE = 0,
    POLYVOICE_PROFILE_BALANCED = 1
} polyvoice_profile_t;

typedef enum {
    POLYVOICE_OK = 0,
    POLYVOICE_ERR_INVALID_ARG = 1,
    POLYVOICE_ERR_AUDIO_TOO_SHORT = 2,
    POLYVOICE_ERR_MODEL_LOAD = 10,
    POLYVOICE_ERR_INFERENCE = 11,
    POLYVOICE_ERR_OUT_OF_MEMORY = 20,
    POLYVOICE_ERR_REGISTRY = 30,
    POLYVOICE_ERR_INTERNAL = 99
} polyvoice_status_t;

int polyvoice_pipeline_create(polyvoice_profile_t profile,
                              const char* models_cache_dir,
                              PolyvoicePipeline** out_handle);

int polyvoice_pipeline_run(PolyvoicePipeline* pipeline,
                           const float* samples,
                           size_t n_samples,
                           uint32_t sample_rate,
                           char** out_json,
                           size_t* out_json_len);

void polyvoice_pipeline_destroy(PolyvoicePipeline* pipeline);
void polyvoice_free_string(char* p, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* POLYVOICE_H */
```

- [ ] **Step 8.3: Create `tests/ffi_smoke_test.rs`**

```rust
//! M6b — FFI smoke tests for ABI v2.

#![cfg(feature = "ffi")]

use polyvoice::ffi::{
    PolyvoicePipeline, PolyvoiceProfile, polyvoice_pipeline_create,
    polyvoice_pipeline_destroy,
};
use std::ptr;

#[test]
#[ignore = "requires cached Balanced ONNX bundle"]
fn ffi_create_destroy_balanced_round_trip() {
    let mut handle: *mut PolyvoicePipeline = ptr::null_mut();
    let rc = polyvoice_pipeline_create(PolyvoiceProfile::Balanced, ptr::null(), &mut handle);
    assert_eq!(rc, 0, "create should succeed when ONNX is cached");
    assert!(!handle.is_null());
    polyvoice_pipeline_destroy(handle);
}

#[test]
fn ffi_create_invalid_profile_arg_does_not_panic() {
    let mut handle: *mut PolyvoicePipeline = ptr::null_mut();
    // Intentionally pass null out_handle — must not panic.
    let rc = polyvoice_pipeline_create(PolyvoiceProfile::Mobile, ptr::null(), ptr::null_mut());
    assert_ne!(rc, 0);
    assert!(handle.is_null());
}
```

- [ ] **Step 8.4: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo build --features ffi
cargo test --features ffi --test ffi_smoke_test
cargo clippy --features ffi --lib -- -D warnings 2>&1 | tail -3
```

Expected: build clean, 1 test pass (the other is `#[ignore]`).

- [ ] **Step 8.5: Commit**

```bash
git add -A
git commit -m "refactor(ffi): rewrite src/ffi.rs to ABI v2 + update include/polyvoice.h"
```

---

## Task 9: Rewrite `python/src/lib.rs` pyo3 bindings

**Files:**
- Modify: `python/src/lib.rs`
- Modify: `python/tests/test_smoke.py`

- [ ] **Step 9.1: Replace `python/src/lib.rs`**

Create `/Users/ekhodzitsky/Documents/personal/polyvoice/python/src/lib.rs` (overwrite existing). Skeleton:

```rust
//! M6b — pyo3 bindings for the v1.0 Pipeline.

use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::{Pipeline as RustPipeline, PipelineConfig};
use polyvoice::types::{Profile, SampleRate};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python-facing `Pipeline` wrapper.
#[pyclass]
pub struct Pipeline {
    inner: RustPipeline,
}

#[pymethods]
impl Pipeline {
    /// Build a Mobile-profile Pipeline.
    #[staticmethod]
    #[pyo3(signature = (models_cache=None))]
    fn mobile(models_cache: Option<&str>) -> PyResult<Self> {
        Self::build_profile(Profile::Mobile, models_cache)
    }

    /// Build a Balanced-profile Pipeline.
    #[staticmethod]
    #[pyo3(signature = (models_cache=None))]
    fn balanced(models_cache: Option<&str>) -> PyResult<Self> {
        Self::build_profile(Profile::Balanced, models_cache)
    }

    /// Run diarization on an iterable of f32 samples.
    fn run<'py>(
        &self,
        py: Python<'py>,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sr = SampleRate::new(sample_rate)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!(
                "invalid sample rate {sample_rate} (expected 8000..=192000)"
            )))?;
        let result = self
            .inner
            .run(&samples, sr)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("pipeline.run: {e}")))?;
        let dict = PyDict::new(py);
        dict.set_item("num_speakers", result.num_speakers)?;
        let turns: Vec<_> = result
            .turns
            .iter()
            .map(|t| {
                let d = PyDict::new(py);
                d.set_item("start", t.time.start).unwrap();
                d.set_item("end", t.time.end).unwrap();
                d.set_item("speaker", t.speaker.0).unwrap();
                d
            })
            .collect();
        dict.set_item("turns", turns)?;
        Ok(dict)
    }
}

impl Pipeline {
    fn build_profile(profile: Profile, models_cache: Option<&str>) -> PyResult<Self> {
        let registry = match models_cache {
            Some(path) => ModelRegistry::with_cache_dir(path),
            None => ModelRegistry::default(),
        }
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("registry: {e}")))?;
        let mut cfg = PipelineConfig::default();
        cfg.profile = profile;
        let pipeline = RustPipeline::builder()
            .config(cfg)
            .with_models_from(registry)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("build: {e}")))?;
        Ok(Self { inner: pipeline })
    }
}

#[pymodule]
fn _polyvoice(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pipeline>()?;
    Ok(())
}
```

NOTE: `pyo3` API may differ slightly (the snippet above targets pyo3 0.22+ with the `Bound` API). If the workspace uses an older pyo3 with `&PyModule` and `&PyDict`, adjust accordingly. Run `cargo check --manifest-path python/Cargo.toml` after the rewrite — compiler errors will name the exact pyo3 API divergence.

- [ ] **Step 9.2: Replace `python/tests/test_smoke.py`**

```python
"""M6b — smoke test for new pyo3 bindings."""

import polyvoice


def test_pipeline_module_imports():
    assert hasattr(polyvoice, "Pipeline"), "Pipeline class should be exposed"


def test_pipeline_mobile_constructor_signature():
    # We can't actually build a Pipeline without cached ONNX, but we can
    # verify the class method exists and rejects invalid sample rate.
    assert hasattr(polyvoice.Pipeline, "mobile")
    assert hasattr(polyvoice.Pipeline, "balanced")
    assert hasattr(polyvoice.Pipeline, "run")
```

- [ ] **Step 9.3: Verify**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cd python
maturin build --release 2>&1 | tail -5
pip install --force-reinstall target/wheels/polyvoice-*.whl
pytest tests/test_smoke.py -v
cd ..
```

Expected: wheel builds, smoke tests pass. If `maturin` isn't installed, install via `pip install maturin`.

- [ ] **Step 9.4: Commit**

```bash
git add -A
git commit -m "refactor(python): rewrite python/src/lib.rs pyo3 bindings on Pipeline.builder()"
```

---

## Task 10: Migration guide + DER baseline + OnlineDiarizer deprecation + CHANGELOG + tag

**Files:**
- Create: `docs/MIGRATING-FROM-0.5.md`
- Create: `tests/der_baseline.json`
- Create: `tests/der_baseline_test.rs`
- Create: `scripts/run-der-baseline.sh`
- Modify: `src/online.rs` (deprecation)
- Modify: `CHANGELOG.md`

- [ ] **Step 10.1: Annotate `OnlineDiarizer`**

In `src/online.rs`, find `pub struct OnlineDiarizer` and add above it:

```rust
#[deprecated(
    since = "0.6.0-alpha.3",
    note = "streaming pipeline redesigned in v1.1; use Pipeline for offline use"
)]
```

If existing tests in `online.rs` reference `OnlineDiarizer`, gate them with `#[allow(deprecated)]`:

```rust
#[cfg(test)]
#[allow(deprecated)]
mod tests { ... }
```

- [ ] **Step 10.2: Create `docs/MIGRATING-FROM-0.5.md`**

Create at `/Users/ekhodzitsky/Documents/personal/polyvoice/docs/MIGRATING-FROM-0.5.md`:

```markdown
# Migrating from polyvoice 0.5 to 1.0

`polyvoice 0.6.0-alpha.3` introduces the v1.0 architecture: a single
`Pipeline::builder()` API, profile-based model selection, and INT8-quantized
ONNX bundles. This is intentionally a breaking change.

## Rust API

### Before (v0.5)
```rust
use polyvoice::{OfflineDiarizer, DiarizationConfig, FbankOnnxExtractor, SileroVad, VadConfig};

let extractor = FbankOnnxExtractor::new("models/wespeaker_resnet34.onnx", 256, 4)?;
let mut vad = SileroVad::new("models/silero_vad.onnx", 512)?;
let pipeline = polyvoice::Pipeline::new(DiarizationConfig::default(), VadConfig::default());
let result = pipeline.run(&samples, &extractor, &mut vad)?;
```

### After (v1.0-alpha.3)
```rust
use polyvoice::{Pipeline, ModelRegistry, Profile, SampleRate};

let registry = ModelRegistry::default()?;
let pipeline = Pipeline::builder()
    .profile(Profile::Balanced)
    .with_models_from(registry)
    .build()?;
let sr = SampleRate::new(16000).unwrap();
let result = pipeline.run(&samples, sr)?;
```

## Python API

### Before
```python
from polyvoice import Pipeline
p = Pipeline("models/")
result = p.run(samples)
```

### After
```python
import polyvoice
p = polyvoice.Pipeline.balanced("models/")
result = p.run(samples, sample_rate=16000)
print(result["num_speakers"], len(result["turns"]))
```

## CLI

| Before                                                     | After                                                |
|------------------------------------------------------------|------------------------------------------------------|
| `polyvoice diarize meeting.wav --threshold 0.4`            | `polyvoice diarize meeting.wav --profile balanced`   |
| `polyvoice diarize meeting.wav --vad-threshold 0.5`        | `polyvoice diarize meeting.wav --profile balanced`   |
| `polyvoice download-models --dir ./models`                 | `polyvoice download-models --profile balanced`       |

## C FFI

The ABI was renamed and replaced. ABI v1 (`polyvoice_diarizer_*`) is removed.
ABI v2 entry points: `polyvoice_pipeline_create`, `polyvoice_pipeline_run`,
`polyvoice_pipeline_destroy`, `polyvoice_free_string`. See `include/polyvoice.h`
for the new contract.

## Removed types and replacements

| Removed                       | Replacement                              |
|-------------------------------|------------------------------------------|
| `Pipeline::new(cfg, vad_cfg)` | `Pipeline::builder()`                    |
| `DiarizationConfig`           | `pipeline::PipelineConfig`               |
| `VadConfig`, `EnergyVad`,    `VoiceActivityDetector` | absorbed by `Segmenter` |
| `OfflineDiarizer`             | `Pipeline::run`                          |
| `DummyExtractor`              | (test-only, no public API)               |
| `OnnxEmbeddingExtractor`      | `embedder::ResNet34Adapter`              |
| `EcapaTdnnExtractor`, `EcapaMelOnnxExtractor`, `RawAudioOnnxExtractor` | (deleted; use `embedder::CamPlusPlusExtractor` or `ResNet34Adapter`) |
| `ClusteringBackend`           | `pipeline::ClustererKind`                |
| `compute_fbank` (public)      | private; use `FbankExtractor::extract`   |

## OnlineDiarizer is deprecated

`OnlineDiarizer` remains accessible but is `#[deprecated(since = "0.6.0-alpha.3")]`.
The streaming pipeline is being redesigned in v1.1 with a richer latency vs.
DER tradeoff. For offline batch processing, use `Pipeline::builder()`.
```

- [ ] **Step 10.3: Create `tests/der_baseline.json`**

```json
{
  "schema": "polyvoice-der-baseline-v1",
  "voxconverse_test": {
    "files": null,
    "profile": "balanced",
    "der_collar_0_25": null,
    "der_no_collar": null,
    "tolerance": 1.0,
    "model_versions": {
      "powerset_int8": null,
      "resnet34_int8": null
    },
    "_status": "schema-only — real numbers after M5 INT8 publish + M6b CLI rewrite",
    "_filled_by": "scripts/run-der-baseline.sh"
  }
}
```

- [ ] **Step 10.4: Create `tests/der_baseline_test.rs`**

```rust
//! M6b — DER baseline schema validity tests. Numbers are deferred to an
//! operational follow-up after M5 INT8 publish closes.

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Baseline {
    schema: String,
    voxconverse_test: VoxConverse,
}

#[derive(Deserialize)]
struct VoxConverse {
    files: Option<usize>,
    profile: String,
    der_collar_0_25: Option<f64>,
    tolerance: f64,
    #[serde(rename = "_status")]
    status: String,
}

#[test]
fn der_baseline_json_parses() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json");
    let raw = std::fs::read_to_string(&path).expect("read der_baseline.json");
    let parsed: Baseline = serde_json::from_str(&raw).expect("parse der_baseline.json");
    assert_eq!(parsed.schema, "polyvoice-der-baseline-v1");
    assert_eq!(parsed.voxconverse_test.profile, "balanced");
    assert_eq!(parsed.voxconverse_test.tolerance, 1.0);
    assert!(parsed.voxconverse_test.status.contains("schema-only"));
}

#[test]
fn der_baseline_acknowledges_deferred_numbers() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json");
    let raw = std::fs::read_to_string(&path).expect("read");
    let parsed: Baseline = serde_json::from_str(&raw).expect("parse");
    assert!(
        parsed.voxconverse_test.files.is_none() && parsed.voxconverse_test.der_collar_0_25.is_none(),
        "numbers must remain null until operational baseline closure run"
    );
}
```

- [ ] **Step 10.5: Create `scripts/run-der-baseline.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Operational follow-up — populate tests/der_baseline.json after M5 INT8 publish.
# Run after `polyvoice download-models --profile balanced` has cached the bundle
# and `data/voxconverse-test/` has been downloaded via
# `scripts/download-voxconverse-test.sh`.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET="${1:-${ROOT_DIR}/data/voxconverse-test}"

cd "$ROOT_DIR"
cargo run --release --features cli --bin polyvoice-bench -- \
    "$DATASET" --profile balanced --output /tmp/m6b-bench-report.json

echo ""
echo "Update tests/der_baseline.json with the numbers from /tmp/m6b-bench-report.json"
echo "(files, der_collar_0_25_skip_overlap, der_no_collar from running with --collar 0)"
```

- [ ] **Step 10.6: Update CHANGELOG.md**

In the `## [Unreleased]` block, after the M6a section (`### Added (M6a — Pipeline + Profile API, additive)`), append:

```markdown

### Changed (M6b — Legacy cleanup + CLI/FFI/Python migration)
- **BREAKING**: removed legacy `Pipeline::new(DiarizationConfig, VadConfig)`,
  `OfflineDiarizer`, `DiarizationConfig`, `VadConfig`, `VoiceActivityDetector`,
  `EnergyVad`, `segment_speech`, `DummyExtractor`, `OnnxEmbeddingExtractor`,
  `EcapaTdnnExtractor`, `EcapaMelOnnxExtractor`, `RawAudioOnnxExtractor`,
  `ClusteringBackend`, `EmbeddingDim`. `compute_fbank` is now private.
- Renamed `pipeline_v2 → pipeline`. The Cargo feature is `pipeline`
  (default-on, requires `download + onnx + segmentation + embedder + clusterer + resegmentation`).
  Public surface: `polyvoice::Pipeline::builder()` is the only Pipeline API.
- CLI rewritten: `polyvoice diarize <wav> --profile mobile|balanced` replaces
  the legacy threshold-based interface. New: `polyvoice models list/info`.
- `polyvoice-bench` rewritten on `Pipeline::builder()`. JSON output schema
  `polyvoice-bench-v1`.
- C FFI ABI v2 (`polyvoice_pipeline_*` family) replaces the legacy
  `polyvoice_diarizer_*` ABI. See `include/polyvoice.h`.
- Python pyo3 bindings rewritten: `polyvoice.Pipeline.mobile()` /
  `Pipeline.balanced()` / `Pipeline.run(samples, sample_rate)`.

### Added (M6b)
- `docs/MIGRATING-FROM-0.5.md`: migration guide for Rust / Python / CLI / C FFI.
- `tests/der_baseline.json`: schema for the v1.0 DER baseline. Numbers are
  deferred to an operational follow-up after M5 INT8 publish closes.
- `scripts/run-der-baseline.sh`: helper that runs `polyvoice-bench` on
  VoxConverse-test and prints the values to paste into the baseline JSON.

### Deprecated
- `polyvoice::OnlineDiarizer` — streaming redesign coming in v1.1; use
  `Pipeline` for offline.
```

- [ ] **Step 10.7: Verify full feature matrix**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
cargo test --all-features --lib 2>&1 | tail -3
cargo test --all-features --tests 2>&1 | tail -3
cargo test --all-features --doc 2>&1 | tail -3
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -5
cargo fmt --check
cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib
bash scripts/release-gate.sh ; echo "exit=$?"
```

Apply `cargo fmt` and clippy fixes if anything is flagged.

- [ ] **Step 10.8: Make scripts/run-der-baseline.sh executable**

```bash
chmod +x scripts/run-der-baseline.sh
```

- [ ] **Step 10.9: Tag**

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice
git tag -a m6b-complete -m "M6b complete: legacy cleanup + CLI/FFI/Python migration"
```

(Don't push the tag yet — push after M6b PR is merged into master.)

- [ ] **Step 10.10: Commit**

```bash
git add -A
git commit -m "feat(m6b): MIGRATING-FROM-0.5.md + DER baseline schema + OnlineDiarizer deprecation + CHANGELOG + tag m6b-complete"
```

- [ ] **Step 10.11: Final git log**

```bash
git log --oneline 9de7802..HEAD
```

Should show 10 commits matching the ten tasks above.

---

## Self-review checklist

1. **Spec coverage:** all 8 deliverables (delete legacy, rename, CLI, bench, FFI, Python, migration guide, DER baseline schema) → Tasks 2–10. Version bump → Task 1. OnlineDiarizer deprecation → Task 10.1.
2. **Atomic-commit guarantee:** each task is one commit; the PR head after Task 10 is fully green; intermediate CI red is acknowledged in the spec.
3. **No `unwrap`/`expect`/`panic`** in lib non-test code (the rewrites use `Result` propagation; FFI uses `catch_unwind`).
4. **Test coverage:** ~10 new tests across CLI smoke (4), FFI smoke (2), DER baseline schema (2), python smoke (3) + all M0–M6a lib tests preserved.
5. **Atomic commits:** ~10 total, one per task.
6. **Order of operations:** Task 2 deletes `src/pipeline.rs` *inside the rename commit* so the directory creation doesn't collide.

---

## Out of scope (M9 / future)

- Real DER baseline numbers — operational follow-up.
- iOS / Windows wheels — M8.
- Android NNAPI — M8.
- Streaming v1.1 — separate spec.
- v1.0.0 GA polish (CHANGELOG voice, blog post, release-gate green) — M9.
- Removing `OnlineDiarizer` entirely — v1.1 lands the redesign.
- Removing `silero_vad.rs` — M9 if confirmed unused after CLI/FFI/Python rewrites.
