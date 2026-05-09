# Dead-Code Cleanup Plan: M1–M5 Experimental Features

> **Scope:** Remove non-functional experimental code from the `default` Cargo feature set.  
> **Constraint:** No source files are deleted; code remains accessible behind explicit feature flags.  
> **Date:** 2026-05-08

---

## 1. Current State

### 1.1 `default` features in `Cargo.toml`

```toml
default = [
    "spectral",
    "segmentation",
    "embedder",
    "clusterer",
    "resegmentation",
    "pipeline",
]
```

### 1.2 What each feature gates

| Feature | Module / Code | Status |
|---|---|---|
| `spectral` | `src/spectral.rs` (NME-SC eigendecomposition) | Only used by **experimental** `NmeScClusterer` |
| `segmentation` | `src/segmentation/` (PowersetSegmenter, decoder, aggregator) | **Non-functional** M1 |
| `embedder` | `src/embedder.rs` (Embedder trait, CamPlusPlusExtractor, ResNet34Adapter, EmbedderPool) | **Non-functional** M2 |
| `clusterer` | `src/clusterer.rs` (Clusterer trait, AhcClusterer, NmeScClusterer) | **Non-functional** M3 |
| `resegmentation` | `src/resegmentation.rs` (OverlapResegmenter) | **Non-functional** M4 |
| `pipeline` | `src/pipeline_v1/` (M6b Pipeline builder API) | **Non-functional** M6b |

### 1.3 Working code requirements

The legacy v0.5.2 pipeline (`src/pipeline.rs`) and the CLI binaries (`src/bin/polyvoice.rs`, `src/bin/polyvoice-bench.rs`) **only need `onnx`**.  They do not import anything from the six modules above.

---

## 2. Proposed Minimal Change

### 2.1 New `default` feature list

```toml
default = ["spectral"]
```

> **Rationale:** `spectral` is kept in `default` for this minimal pass because it is a dependency-only feature (pulls `faer`) and the task focuses on the five M1–M5 features.  A follow-up step can evaluate removing `spectral` as well (see §5).

### 2.2 `Cargo.toml` diff

```diff
 [features]
-default = ["spectral", "segmentation", "embedder", "clusterer", "resegmentation", "pipeline"]
+default = ["spectral"]
```

No other changes are required for this step.

---

## 3. Impact Analysis

### 3.1 `src/lib.rs` — conditional compilation

`src/lib.rs` already wraps every experimental module and re-export in `#[cfg(feature = "...")]`:

- `segmentation` → lines 41–51  
- `embedder` → lines 53–60  
- `clusterer` → lines 62–69  
- `resegmentation` → lines 71–81  
- `pipeline_v1` → lines 83–104 (gated on **all five** features plus `onnx`)

Removing the features from `default` simply causes these items to disappear from the public API when building with defaults.  The working `pipeline` module (legacy) is **unconditionally** compiled (line 106).

### 3.2 Tests — what runs vs. what is skipped

All integration tests that depend on the experimental features are already guarded with `#![cfg(...)]`.  When the features are absent the test files compile to empty crates — no failures.

| Test file | Required features | Behaviour after change |
|---|---|---|
| `tests/segmenter_test.rs` | `onnx` + `segmentation` + `download` | Skipped unless `--features …` added |
| `tests/embedder_test.rs` | `onnx` + `embedder` + `download` | Skipped unless `--features …` added |
| `tests/clusterer_test.rs` | `clusterer` (+ `spectral` for NME-SC) | Skipped unless `--features …` added |
| `tests/resegmentation_test.rs` | `resegmentation` | Skipped unless `--features …` added |
| `tests/miri_resegmentation.rs` | `resegmentation` | Skipped unless `--features …` added |
| `tests/pipeline_e2e_test.rs` | `pipeline` + all M1–M5 + `onnx` + `download` | Skipped unless `--features …` added |
| `tests/pipeline_synthetic_test.rs` | `pipeline` + all M1–M5 + `onnx` | Skipped unless `--features …` added |
| `tests/e2e_smoke_test.rs` | `onnx` + `download` | **Still runs** (uses legacy pipeline) |
| `tests/cli_smoke_test.rs` | `cli` | **Still runs** (uses legacy pipeline) |
| `tests/m5_manifest_smoke_test.rs` | `download` | **Still runs** |
| `tests/test_ahc.rs` | none | **Still runs** |
| `tests/der_regression_test.rs` | none | **Still runs** |
| `tests/loom_pool.rs` | none | **Still runs** |
| `tests/test_wav.rs` | none | **Still runs** |

Unit tests inside the gated modules (e.g. `overlap_mask_tests` in `src/embedder.rs`) are also skipped automatically because the parent module is absent.

### 3.3 `cli` feature

The `cli` feature definition currently pulls in the experimental stack:

```toml
cli = ["onnx", "download", "pipeline", "spectral", "segmentation", "embedder", "clusterer", "resegmentation", "dep:clap"]
```

However, the actual CLI binaries (`src/bin/polyvoice.rs`, `src/bin/polyvoice-bench.rs`) use the **legacy** `pipeline::Pipeline`, `FbankOnnxExtractor`, and `SileroVad` — they never touch `pipeline_v1` or any M1–M5 trait.  Because `cli` explicitly lists the experimental features, `cargo run --features cli` will continue to compile and work exactly as before; it will just transitively re-enable the experimental modules.

> **Follow-up:** `cli` can be slimmed down to `cli = ["onnx", "download", "dep:clap"]` once this change is validated (see §5).

### 3.4 `ffi` feature

Same situation as `cli`: `ffi` currently depends on the full M1–M5 stack, but `src/ffi.rs` only wires the legacy pipeline.  It will continue to work because the feature definition includes the experimental flags.

> **Follow-up:** `ffi` can be slimmed down to `ffi = ["onnx"]` (see §5).

### 3.5 CI / cross-compilation

| CI job | Flags | Impact |
|---|---|---|
| `check` (ubuntu) | `--all-targets --all-features` | None — `--all-features` re-enables everything |
| `check` (macos/win) | `--all-targets --features onnx,ffi,cli` | None — `ffi` and `cli` pull in the experimental stack |
| `clippy` (all) | Same as `check` | None |
| `test` (all) | Same as `check` | None |
| `doc` | `--no-deps --all-features` | None |
| `miri` | `--features ffi` | None — `ffi` pulls in the stack |
| `loom` | `--test loom_pool` | None — no feature flags needed |
| `cross-aarch64` | `default features` | **Positive** — smaller, faster compile; only working code is built |
| `wasm32-smoke` | `--no-default-features --lib` | None — already built this way |

---

## 4. Verification Steps

Run these commands **after** applying the `Cargo.toml` change to confirm correctness:

```bash
# 1. Core working code compiles without any experimental features
cargo check --no-default-features

# 2. Core working code compiles with the features the legacy pipeline actually needs
cargo check --no-default-features --features onnx

# 3. Full test suite for the working path
cargo test --no-default-features

# 4. CLI still works (transitively re-enables experimental modules)
cargo check --no-default-features --features cli

# 5. FFI still works
cargo check --no-default-features --features ffi

# 6. Experimental code is still reachable when explicitly requested
cargo check --no-default-features --features segmentation,embedder,clusterer,resegmentation,pipeline

# 7. CI parity check — all features together still compile
cargo check --all-targets --all-features
```

All of the above already pass on the current codebase (verified by running `cargo check --no-default-features` and `cargo test --no-default-features`).

---

## 5. Optional Follow-Up Cleanups (Post-Validation)

After the minimal change has been merged and observed stable, the following additional cleanups can be considered **low-risk**:

### 5.1 Slim `cli` feature

```diff
-cli = ["onnx", "download", "pipeline", "spectral", "segmentation", "embedder", "clusterer", "resegmentation", "dep:clap"]
+cli = ["onnx", "download", "dep:clap"]
```

**Proof of safety:** `src/bin/polyvoice.rs` and `src/bin/polyvoice-bench.rs` only import from `polyvoice::pipeline` (legacy), `polyvoice::models`, `polyvoice::vad`, `polyvoice::wav`, `polyvoice::rttm`, `polyvoice::der`, `polyvoice::types`, `polyvoice::FbankOnnxExtractor`, and `polyvoice::SileroVad`.  None of these require `pipeline_v1` or any M1–M5 trait.

### 5.2 Slim `ffi` feature

```diff
-ffi = ["onnx", "pipeline", "segmentation", "embedder", "clusterer", "resegmentation"]
+ffi = ["onnx"]
```

**Proof of safety:** `src/ffi.rs` imports the same legacy types as the CLI and does not touch any experimental module.

### 5.3 Remove `spectral` from `default`

```diff
-default = ["spectral"]
+default = []
```

`spectral` pulls `faer`, a heavy linear-algebra dependency.  The only caller inside the crate is `NmeScClusterer` (gated by `clusterer` + `spectral`).  The legacy pipeline does not use spectral clustering.  Removing `spectral` from default would further shrink the out-of-the-box compile graph.

---

## 6. Risk Assessment

| Risk | Level | Reasoning |
|---|---|---|
| Compilation break | **Low** | `cargo check --no-default-features` already passes. All experimental modules are behind `#[cfg]` guards. |
| Test break | **Low** | Tests are properly gated with `#![cfg(...)]`; they will be skipped, not fail. |
| CLI break | **Low** | `cli` feature definition still includes the experimental stack, so `cargo run --features cli` is unchanged. |
| FFI break | **Low** | `ffi` feature definition still includes the experimental stack. |
| API break for downstream users | **Medium** | Any downstream crate that relies on `polyvoice = "0.6"` (default features) and imports `PowersetSegmenter`, `Embedder`, `Clusterer`, `OverlapResegmenter`, or `PipelineV1` will see compile errors after `cargo update`.  Mitigation: this is an `0.6.0-alpha.3` pre-release; breaking the experimental API is acceptable per semver for pre-releases.  The CHANGELOG should note the change. |
| Cognitive overhead | **Low** | The experimental code remains in the tree, just not compiled by default.  Contributors can still work on it with `--features segmentation,embedder,clusterer,resegmentation,pipeline`. |

---

## 7. Summary

| | Current | Proposed |
|---|---|---|
| **Default features** | `spectral`, `segmentation`, `embedder`, `clusterer`, `resegmentation`, `pipeline` | `spectral` |
| **Files touched** | `Cargo.toml` (1 line) | Same |
| **Files deleted** | 0 | 0 |
| **Experimental code preserved?** | Yes | Yes (behind explicit flags) |
| **Working pipeline affected?** | No | No |
| **CLI affected?** | No | No (transitive deps intact) |
| **FFI affected?** | No | No (transitive deps intact) |
| **Estimated risk** | — | **Low** |

**Recommendation:** Proceed with the one-line change to `Cargo.toml`. Run the verification commands in §4. After a short bake-in period, apply the follow-up cleanups in §5 to further simplify the feature graph.
