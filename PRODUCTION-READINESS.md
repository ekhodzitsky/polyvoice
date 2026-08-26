# Production Readiness Assessment

> **Version:** 0.18.x | **Date:** 2026-08-26 | **Scope:** Rust library + Python bindings + FFI + CLI
>
> **Last updated:** 2026-08-26 — crate **0.18.0** ships **INT8-only** profiles
> (`powerset_int8` + `resnet34_int8`). Pipeline v2+VBx default since 0.11.
> **Product CLI / FFI / MCP** run hand-written kernels (`pipeline-native`),
> not `libonnxruntime`. ONNX Runtime is opt-in (`cli-ort` / `pipeline-full` /
> Python). Linux/CPU full-split DER gate (ort protocol) remains the non-Apple
> accuracy ceiling; Linux native full-split RTF is still below that band.
> Pure-Rust tract is **opt-in smoke only**. Canonical accuracy protocol:
> [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Zero-deps strategy:
> [`docs/strategy/zero-deps.md`](docs/strategy/zero-deps.md).

## Executive Summary

**Status: NOT GO for public unattended production. OK for controlled internal use.**

As of **0.18.x**, polyvoice is a hardened pre-1.0 engine: model signing is
enforced on release builds for profile-resolved models, CI covers the main
desktop targets, and **full VoxConverse-test + AMI-test DER** keeps
**pipeline v2 + VBx** as the CLI / FFI / Python / MCP default. The **engine**
split is:

- **CLI / FFI / MCP:** hand-written INT8 kernels (`cli` = `pipeline-native`).
  No `libonnxruntime`. Darwin holds the native scoreboard floors
  (`tests/native_scoreboard.json`). Linux native holds AMI DER within the
  ort ceiling; RTF there trails the old ort band.
- **Python / `cli-ort`:** ONNX Runtime INT8 (`ort` 2.0.0-rc.12). This is the
  measured Linux/CPU full-split protocol in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

It is still **not** ready for multi-tenant public APIs or unattended production
services, because:

1. **Pre-1.0 API** — no backward-compatibility commitment until `1.0.0`.
2. **Python still links `ort` RC** (`2.0.0-rc.12`). The product CLI no longer
   does; pip wheels and `cli-ort` still do.
3. **Cross-corpus validation is thin** — solid VoxConverse + AMI coverage;
   NOTSOFAR-1 has a measured micro-gate (3-meeting subset) but CALLHOME /
   DIHARD (and similar) are not release-gated.
4. **Linux native full-split RTF trails ort** — Vox-3 smoke is ~28× vs ~82×
   Linux ort. Native full-split DER ceilings are copied from the ort protocol
   until the first Linux native full split is filled.
5. **Pure-Rust (tract) path is not product-ready** — opt-in only
   (`backend-tract` + signed `powerset_fp32_tract` + FP32 ResNet); ~9× slower
   than ort; no full-split release gate.

**Suitable for:** controlled internal services, desktop apps, and edge pilots
where audio conditions are known and operators can pin versions and re-verify
DER after upgrades. Desktop / CLI deploys can avoid `ort` entirely.

**Not suitable for:** public multi-tenant APIs, unattended SLA-bound services,
or security-critical deployments that require a stable Python runtime +
multi-corpus proof.

---

## Current surface (0.18.x truth)

| Area | State |
|------|--------|
| Crate version | `0.18.0` |
| Production models | **INT8 only** (`powerset_int8` + `resnet34_int8`, ~8.4 MB) |
| CLI / FFI / MCP engine | **kernels** (`pipeline-native`); `--legacy` / `--clusterer ahc` opt out |
| Python engine | **ONNX Runtime** INT8 (same v2 + VBx; pass `clusterer="ahc"` to opt out) |
| Opt-in ONNX CLI | `--features cli-ort` / `pipeline-full` |
| Full-split DER (no-collar micro, INT8, **ort** Linux/CPU) | Vox **14.94%** / AMI **24.19%** — [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| Darwin native full-split (M1 Pro, kernels) | Vox **15.47%** / AMI **25.19%** / ~**130× / 109×** RTFx |
| Darwin native Vox-3 scoreboard | DER₀ **7.11 / 7.39**, ≥**117×**, pair ≤ 8 414 314 B, peak RSS ≤ **556 MiB** |
| Linux native | AMI DER within ort ceiling; Vox-3 RTF ~**28×**; full-split native rows still pending fill |
| Inference (product CLI) | **`polyvoice-kernels`** (Darwin Accelerate/BNNS; Linux `rten-gemm`) |
| Inference (opt-in ONNX) | **`OrtSession` (`ort` 2.0.0-rc.12)** — Python + `cli-ort` |
| Inference (opt-in tract) | `POLYVOICE_INFERENCE_BACKEND=tract` + `backend-tract`: signed `powerset_fp32_tract` + **FP32** ResNet; smoke DER only |
| Models | Profile segmenter/embedder minisign-signed in release; VBx PLDA registry downloads are minisign-signed; opt-in `powerset_fp32_tract` is minisign-signed (release `models-tract-v1`) |
| Native ORT binary | Hash-pinned via ort-sys `dist.txt` **when `onnx` is enabled**; trust model in [`docs/security/ort-native-binary-provenance.md`](docs/security/ort-native-binary-provenance.md) |
| Library features | `pipeline-native` + `vbx` (CLI parity) or `pipeline-full` + `vbx` (ONNX). Crate-root `Pipeline` needs that gate; `PipelineConfig::default()` is still **AHC** |

Honest reading: v2+VBx INT8 is the **measured product pipeline**. The **default
binary** is kernels (no `ort`). Linux/CPU **ort** remains the published
non-Apple full-split accuracy protocol. Legacy remains a supported escape
hatch. Tract is an **opt-in research path**. Public production still needs
multi-corpus gates, a Python engine story that is not an RC dylib, and Linux
native full-split numbers that are measured rather than copied ceilings.

---

## Gap Analysis

### 1. Version & API Stability ❌

| Item | Status | Risk |
|------|--------|------|
| Semantic version | `0.18.0` | Pre-1.0 — API may change between `0.x` minors |
| `semver-checks` | Passes in CI | Only checks public API surface; pre-1.0 still allows breaking changes |
| CHANGELOG | Maintained | Tracks 0.11→0.18; CLI default flip to v2+VBx was 0.11; kernels default was 0.18 |

**Gap:** No commitment to backward compatibility until `1.0.0`. Consumers should
pin a `0.18.x` (or tighter) and read the CHANGELOG before upgrading.

**Remediation:** Freeze the public API, publish a semver policy, then ship
`1.0.0`.

---

### 2. Dependency Supply Chain ⚠️

| Dependency | Version | Risk |
|------------|---------|------|
| `polyvoice-kernels` | workspace | Product CLI. Darwin uses Accelerate/BNNS (C shims); Linux uses `rten-gemm` (pure Rust). MSRV 1.94. |
| `ort` (ONNX Runtime) | `2.0.0-rc.12` | **RC, not stable.** Still linked by **Python** and `cli-ort`. Not on the product CLI. |
| Native ORT binary | pinned via ort-sys | Hash-verified download when `onnx` is on; residual trust in pyke builds + CDN cold-fetch |
| `faer` (spectral clustering) | Optional | Not used in the default pipeline |
| `paste` | Latest | Unmaintained (LOW; no CVE) |

**Gap:** `ort` is no longer the product-CLI backend, but it remains the highest
risk on the **Python** surface: RC track + C++ native runtime. Tract is a
spike/goal, not shipped parity. Kernels replace ort for CLI/FFI/MCP.

**Remediation:**
- Keep the product CLI on kernels; do not pull `ort` back into `cli`.
- Track `ort` 2.0 stable for Python / `cli-ort`; re-verify pins and DER on
  every RC → stable bump.
- Keep the `InferenceRuntime` surface clean so ONNX backends stay swappable.
- Retain provenance docs and CI cache of the verified native binary for the
  opt-in ONNX path.

Evidence: [`docs/security/ort-native-binary-provenance.md`](docs/security/ort-native-binary-provenance.md),
`Cargo.toml` pin, `scripts/check-ort-version.sh`, `scripts/check-zero-deps.sh`.

---

### 3. Security Posture ✅

| Control | Status | Evidence |
|---------|--------|----------|
| Model signing (Minisign) | Implemented | Streaming verify; pubkey baked in; **release builds require signatures** for profile-resolved models |
| ONNX header validation | Implemented | Pre-load DOS guard (ONNX path) |
| ORT native binary provenance | Documented + CI-cached | [`docs/security/ort-native-binary-provenance.md`](docs/security/ort-native-binary-provenance.md) |
| TLS for downloads | Implemented | `ureq` + `rustls` + `webpki-roots` |
| FFI sandbox | Implemented | Path traversal guard, sample limits, panic logging |
| `cargo audit` | In CI | 0 HIGH / 0 MEDIUM expected on green main |
| Fuzzing | Active | libFuzzer targets for fbank, VAD, overlap, cluster assign |

**Gap:** Residual LOW noise (e.g. unmaintained transitive crates). No independent
third-party security audit. RC-track runtime remains a supply-chain residual
on Python / `cli-ort`.

---

### 4. Correctness Verification ✅ / ⚠️

| Tool | Coverage | Note |
|------|----------|------|
| Unit / integration tests | Broad `src/` + `tests/` | Structural coverage good |
| Native scoreboard | `tests/native_scoreboard.rs` | Darwin Vox-3 floors: DER, RTF, model bytes, peak RSS |
| Miri | Focused PR-gate set | `ffi_smoke`, `miri_resegmentation`, `test_ahc` — not a full-lib multi-hour run |
| Loom | `loom_pool.rs` | Session / pool concurrency model |
| Proptest | In CI | DER / k-means / AHC / types property suites |
| DER regression gates | Legacy + v2 + Linux/CPU ort | Headline no-collar metric release-gated; Linux native full-split still pending fill |

**Gap:** Full-lib Miri is intentionally not the PR gate (cost). Linux native
full-split DER/RTF is not yet a filled artifact (ceilings copied from ort).

---

### 5. Dataset Validation ⚠️ / ❌

Canonical figures: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and
`tests/der_baseline.json` (legacy unless noted).

| Dataset | Files | DER (collar 0) | DER (0.25 s) | Used in CI? |
|---------|-------|----------------|--------------|-------------|
| VoxConverse test (legacy) | 232 | **18.54%** | 12.91% | Full split not PR-gated; numbers are release-canonical |
| VoxConverse test (legacy, 10-file) | 10 | 27.08% (micro) | 15.82% (macro gate) | Yes (gated subset) |
| e2e smoke (legacy) | 1 | 14.52% | 6.62% | Yes |
| AMI test Mix-Headset (legacy) | 16 | **32.87%** | 25.20% | Full split tracked; long-form floor via single-meeting gate |
| AMI EN2002a (legacy, single) | 1 | 42.90% | 34.62% | Yes (gated) |
| pipeline v2 + VBx **INT8** (Vox / AMI, **ort** host / CoreML) | 232 / 16 | **15.02%** / **24.50%** | 10.33% / 16.82% | INT8 since 0.17; full-split 2026-08-10 |
| pipeline v2 + VBx **INT8** **Linux/CPU ort** (Vox / AMI) | 232 / 16 | **14.94%** / **24.19%** | 10.27% / 16.60% | Official non-Apple **ort** protocol; gate + CI smoke |
| Darwin native kernels (Vox / AMI, M1 Pro) | 232 / 16 | **15.47%** / **25.19%** | — | 0.18 product CLI; RTFx ~130× / ~109× |
| Linux native kernels | 232 / 16 | ceiling = ort (pending fill) | ceiling = ort | `*_linux_cpu_native` in `der_baseline.json`; Vox-3 RTF ~28× |
| tract pure-Rust (3 short Vox, M1 Pro) | 3 | ~**7.22%** (vs ort ~7.41%) | — | Opt-in; not a release gate |
| tract pure-Rust (10 shortest Vox, ≈560 s) | 10 | **8.86%** (vs ort **9.18%**) | — | RTFx ~11 vs ~99 |
| tract pure-Rust (**AMI-test 16**, M1 Pro) | 16 | **23.42%** (vs ort **24.63%**) | — | RTFx ~19 vs ~154; `scripts/tract-der-gate.sh` |
| CALLHOME | — | — | — | **Not measured / not gated** |
| DIHARD | — | — | — | **Not measured / not gated** |

**Gap:** The default v2+VBx INT8 path has full-split VoxConverse and AMI on
desktop baselines and the **Linux/CPU ort** gate. Darwin native full-split is
measured. **Linux native full-split is not yet filled.** **Multi-corpus DER
beyond Vox/AMI remains absent**: no CALLHOME/DIHARD release gate. Accuracy
still trails pyannote-class systems by roughly ~4 pp no-collar on VoxConverse
(see benchmarks). Tract is **not** release-gated at full-split size.

**Remediation:**
- Keep Linux/CPU **ort** as the non-Apple accuracy protocol until native
  full-split rows are filled (do not substitute CoreML RTF for Linux).
- Fill `voxconverse_test_linux_cpu_native` / `ami_test_linux_cpu_native`
  from `scripts/linux-cpu-native-der-gate.sh`.
- Add at least one additional corpus (CALLHOME and/or DIHARD subset) to the
  release DER matrix.
- Do not pull `ort` back into `cli` to paper over Linux RTF.

---

### 6. Pipeline story (honest dual path) ⚠️

| Path | How to run | Role in 0.18.x |
|------|------------|----------------|
| **v2 + VBx kernels (CLI/FFI/MCP default)** | `cargo install polyvoice --features cli` | Product binary; Darwin scoreboard + Darwin full-split |
| **v2 + VBx ONNX Runtime** | Python wheel; `--features cli-ort` | Measured Linux/CPU full-split protocol |
| **Legacy** | CLI `--legacy` / `--clusterer ahc` | Supported escape hatch; former default (Silero + AHC) |

**Gap:** The pipeline default flipped at 0.11 (v2+VBx) and the engine default
flipped at 0.18 (kernels). Legacy still ships as an escape hatch, so dual
pipelines continue to tax docs, gates, and bindings. Library
`PipelineConfig::default()` is still **AHC** while every front door sets
**VBx**. 1.0 should not ship with two first-class paths; retire or clearly
demote legacy once v2+VBx has broader multi-corpus proof.

---

### 7. Inference runtime independence ⚠️

| Item | Status |
|------|--------|
| Product CLI/FFI/MCP | **`polyvoice-kernels`** (`pipeline-native`) — no `InferenceRuntime` dylib |
| `InferenceRuntime` trait | **Exists** (`src/onnx/runtime.rs`) for ONNX-shaped backends |
| ONNX implementation | **`OrtSession`** (`ort` 2.0.0-rc.12) — Python + `cli-ort` |
| Pure-Rust ONNX backend | **`TractSession`** behind `backend-tract` + `POLYVOICE_INFERENCE_BACKEND=tract` |
| Tract powerset | Shipping graphs fail load; **rewrite** via `scripts/export-powerset-tract.py`; pipeline remaps when present |
| Tract embedder | Builder forces **FP32** `wespeaker_resnet34` (INT8 ResNet under tract collapses speakers) |
| Tract accuracy | 3-file Vox smoke DER ≈ ort; **not** full-split gated; ~9× slower RTFx on smoke host |
| Execution providers | CoreML / XNNPACK (and related) wired as **ort-specific** config, not kernel or tract |

**Gap:** Product CLI **does not lock to ort**. Residual lock: **Python** still
does. Tract is a real optional backend with smoke evidence — but rewrite models
are not the product default, INT8 embedder is unsafe under tract, and
large-corpus DER/RTF is open. See
[`docs/strategy/zero-deps.md`](docs/strategy/zero-deps.md).

---

### 8. CI / Platform Coverage ✅

| Target | CI | Notes |
|--------|-----|-------|
| x86_64 Linux | ✅ | Primary |
| x86_64 / aarch64 macOS | ✅ | Native kernels + CoreML path where configured |
| x86_64 Windows | ✅ | |
| aarch64 Linux | ✅ | Cross job; native INT8 GEMM is the product CLI |
| wasm32 | ✅ | Compile / smoke (not full ONNX diarization) |
| Python wheels | ✅ | Maturin (macOS / Linux / Windows); still `ort` |

Miri is a **focused** PR gate rather than a multi-hour full-suite job. Fuzz and
audit remain active.

---

### 9. Documentation & Onboarding ✅

| Asset | Status |
|-------|--------|
| README | Install, usage, links, honest accuracy framing; kernels default |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Canonical DER / RTF with collar protocol |
| [`docs/PIPELINE-ARCHITECTURE.md`](docs/PIPELINE-ARCHITECTURE.md) | Pipeline who-calls-whom |
| [`docs/DEVELOPMENT-PROCESS.md`](docs/DEVELOPMENT-PROCESS.md) | Development process checklist |
| Security provenance | ORT native binary (opt-in path) + model signing story |
| `CONTRIBUTING.md` | Setup and contribution guidelines |
| FFI | C header + examples / smoke tests (`ffi` = kernels) |

---

## Go/No-Go Matrix

_As of 0.18.x — product CLI is kernels, Python/`cli-ort` still `ort` 2.0.0-rc.12,
INT8 profiles + v2+VBx default, legacy as an escape hatch, and multi-corpus DER
is incomplete. Public unattended stays NO-GO._

| Scenario | Verdict | Rationale |
|----------|---------|-----------|
| Internal microservice (controlled audio, ops on-call) | **GO with caveats** | Pin crate; prefer the kernel CLI to avoid `ort` RC; re-run DER after upgrades; no public SLA |
| Desktop app (local processing) | **GO** | User owns hardware; ~8.4 MB INT8; no `libonnxruntime` on the product CLI; tolerate pre-1.0 API |
| Public cloud API (multi-tenant, unattended) | **NO-GO** | Dual pipeline, thin multi-corpus proof, pre-1.0 API; Python still RC runtime |
| Embedded / edge (aarch64) | **GO with testing** | Cross-compile works; measure DER/RTF on target hardware (Linux native RTF ≠ Darwin) |
| Security-critical (government, finance) | **NO-GO** | Needs broader audit + multi-corpus validation; Python still RC `ort` |

---

## 1.0 GO checklist

All items must be true before declaring production-ready / shipping `1.0.0` as
**GO** for broader deployment. Worded as outcomes — not internal tracker IDs.

- [ ] **Single default pipeline.** One validated CLI/Python/FFI path; no dual
      “legacy vs experimental” default. Experimental flags may remain for R&D
      but must not be required for the shipped claim. Library
      `PipelineConfig::default()` should match front-door VBx or stop looking
      like a safe default.
- [ ] **Public API freeze + semver policy.** Documented stability rules; no
      silent breaking churn on the advertised surface for a freeze window; then
      `1.0.0`.
- [ ] **Runtime story closed.** Product CLI is kernels (done in 0.18). Remaining:
      Python still `ort` RC; Linux native full-split RTF unpublished (Vox-3
      ~28× vs ort ~82×); tract remains opt-in smoke. Either Python ships
      kernels (or documents ort as the Python-only engine), Linux native
      full-split is filled, and `ort` 2.x stable is re-verified for the opt-in
      path.
- [ ] **Multi-corpus DER gate.** Release-blocking DER on VoxConverse **and** AMI
      **and** at least one additional corpus (CALLHOME and/or DIHARD subset),
      with collar and overlap policy published next to the numbers.
- [ ] **Accuracy target path.** VoxConverse-test no-collar success metric on the
      default path at **≤13–14%** (stretch ≤12%), with AMI not stagnating in the
      high-20s/30s without a documented plan — see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
- [ ] **This document says GO.** `PRODUCTION-READINESS.md` re-reviewed and
      signed off for the intended deployment class (internal vs public).

Until every box is checked, the honest status remains:

> **NOT GO for public unattended production; OK for controlled internal use.**

---

## Recommended blockers (summary)

| Blocker | Why it blocks 1.0 / public GO |
|---------|-------------------------------|
| Dual pipeline families (BYO vs v2) | Intentional; still doubles docs/gates if not documented |
| Python / `cli-ort` still `ort` RC | Supply-chain and upgrade risk on those surfaces |
| Linux native full-split unfilled | Product CLI on Linux is not the published full-split protocol yet |
| Thin multi-corpus DER | Outside Vox/AMI only NOTSOFAR micro-gate; no CALLHOME/DIHARD |
| Pre-1.0 API | Breaking changes without major bump |
| Accuracy gap vs leaders | ~4 pp no-collar on VoxConverse; speaker counting still dominant error |

---

## Metrics (snapshot, 0.18.x)

| Metric | Value |
|--------|-------|
| Crate version | 0.18.0 |
| Deployable footprint | **~8.4 MB** INT8 production pair (FP32 ids optional / not profile-default) |
| Product CLI engine | kernels (`pipeline-native`); no `libonnxruntime` |
| Speed (kernels, Darwin Vox-3 scoreboard) | ≥**117×** realtime; peak RSS ≤ **556 MiB** |
| Speed (kernels, Darwin full-split M1 Pro) | Vox ~**130×**; AMI ~**109×** |
| Speed (kernels, Linux Vox-3) | ~**28×** RTFx |
| Speed (INT8, Linux/CPU **ort** full-split) | Vox ~**82×** RTFx; AMI ~**95×** RTFx |
| VoxConverse-test DER (v2+VBx INT8, 232, collar 0, **ort** host) | **15.02%** |
| VoxConverse-test DER (v2+VBx INT8 Linux/CPU **ort**, 232, collar 0) | **14.94%** |
| VoxConverse-test DER (v2+VBx INT8 Darwin **kernels**, 232, collar 0) | **15.47%** |
| VoxConverse-test DER (legacy, 232, collar 0) | 18.54% |
| AMI-test DER (v2+VBx INT8, 16, collar 0, **ort** host / Linux) | **24.50%** / **24.19%** |
| AMI-test DER (v2+VBx INT8 Darwin **kernels**, 16, collar 0) | **25.19%** |
| AMI-test DER (legacy, 16, collar 0) | 32.87% |
| Default pipeline | v2 + VBx |
| Default CLI engine | kernels (0.18+) |
| Escape hatch | legacy (`--legacy` / `--clusterer ahc`); ONNX CLI (`cli-ort`) |
| Inference backends | **Product CLI:** kernels. **Python / `cli-ort`:** ort. **Opt-in:** tract |
| Model authenticity | Minisign; required on release profile resolution |
| Security audit (cargo audit on green main) | 0 HIGH, 0 MEDIUM expected |

For competitor context, collar protocol, and reproduction commands, use
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — do not treat this readiness file as
the accuracy source of truth.
