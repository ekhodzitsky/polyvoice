# Production Readiness Assessment

> **Version:** 0.20.x | **Date:** 2026-09-14 | **Scope:** Rust library + Python bindings + FFI + CLI
>
> **Last updated:** 2026-09-14 — Linux kernel full-split filled (VoxConverse-test
> DER₀ **13.34 %** / AMI-test **24.19 %**, VBx AHC seed 0.6). Product CLI /
> FFI / MCP / Python / `polyvoice-transcribe` diarization run hand-written
> kernels (`pipeline-native`), not `libonnxruntime`. ONNX Runtime is opt-in
> (`cli-ort` / `pipeline-full`). Pure-Rust tract is **opt-in smoke only**.
> Canonical accuracy protocol: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
> Zero-deps strategy: [`docs/strategy/zero-deps.md`](docs/strategy/zero-deps.md).

## Executive Summary

**Status: NOT GO for public unattended production. OK for controlled internal use.**

As of **0.20.x**, polyvoice is a hardened pre-1.0 engine: model signing is
enforced on release builds for profile-resolved models, CI covers the main
desktop targets, and **full VoxConverse-test + AMI-test DER** keeps
**pipeline v2 + VBx** as the CLI / FFI / Python / MCP default. The **engine**
split is:

- **CLI / FFI / MCP / Python / transcribe diarization:** hand-written INT8
  kernels (`cli` = `pipeline-native`). No `libonnxruntime`. Darwin holds the
  native scoreboard floors (`tests/native_scoreboard.json`). Linux kernels
  hold the published non-Apple product numbers.
- **`cli-ort`:** ONNX Runtime INT8 (`ort` 2.0.0-rc.12). Comparison protocol
  in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Parakeet TDT (ASR companion)
  still uses `ort`.

It is still **not** ready for multi-tenant public APIs or unattended production
services, because:

1. **Pre-1.0 API** — no backward-compatibility commitment until `1.0.0`.
2. **`cli-ort` still links `ort` RC** (`2.0.0-rc.12`). Product diarization
   surfaces no longer do.
3. **Cross-corpus validation is thin** — solid VoxConverse + AMI coverage;
   NOTSOFAR-1 has a measured micro-gate (3-meeting subset) but CALLHOME /
   DIHARD (and similar) are not release-gated.
4. **Darwin full-split was not re-run** after the VBx AHC seed 0.6 retune
   (still 15.47 % / 25.19 % from 0.18). Linux was: Vox **13.34 %** /
   AMI **24.19 %**.
5. **Pure-Rust (tract) path is not product-ready** — opt-in only
   (`backend-tract` + signed `powerset_fp32_tract` + FP32 ResNet); ~9× slower
   than ort; no full-split release gate.

**Suitable for:** controlled internal services, desktop apps, and edge pilots
where audio conditions are known and operators can pin versions and re-verify
DER after upgrades. Desktop / CLI / Python deploys can avoid `ort` entirely.

**Not suitable for:** public multi-tenant APIs, unattended SLA-bound services,
or security-critical deployments that require a frozen public API and
multi-corpus proof.

---

## Current surface (0.20.x truth)

| Area | State |
|------|--------|
| Crate version | `0.20.0` |
| WAVE ingest | **`ryf`** (WAVE family → mono f32); `audio-io` still `symphonia` + `rubato` for non-WAV |
| Production models | **INT8 only** (`powerset_int8` + `resnet34_int8`, ~8.4 MB) |
| CLI / FFI / MCP engine | **kernels** (`pipeline-native`); `--legacy` / `--clusterer ahc` opt out |
| Python engine | **kernels** (same v2 + VBx as the CLI; pass `clusterer="ahc"` to opt out) |
| Opt-in ONNX CLI | `--features cli-ort` (**deprecated**) / `pipeline-full` |
| Full-split DER (no-collar micro, INT8, **Linux kernels**) | Vox **13.34%** / AMI **24.19%** — [`linux-cpu-native-der-2026-09-13-vbx-ahc/`](benchmarks/results/linux-cpu-native-der-2026-09-13-vbx-ahc/) |
| Full-split DER (no-collar micro, INT8, **ort** Linux/CPU, AHC seed 0.5 protocol) | Vox **14.94%** / AMI **24.19%** — [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| Darwin native full-split (M1 Pro, kernels) | Vox **15.47%** / AMI **25.19%** / ~**130× / 109×** RTFx (0.18; not re-run after AHC seed 0.6) |
| Darwin native Vox-3 scoreboard | DER₀ **7.11 / 7.39**, ≥**117×**, pair ≤ 8 414 314 B, peak RSS ≤ **556 MiB** |
| Linux native RTF (Ryzen AI 9 HX 370) | Vox ~**162×**; AMI ~**193×**; Vox-3 ~**111×** jobs=1 / ~**158×** wall at `--jobs 3` |
| Inference (product CLI) | **`polyvoice-kernels`** (Darwin Accelerate/BNNS; Linux `rten-gemm`) |
| Inference (opt-in ONNX) | **`OrtSession` (`ort` 2.0.0-rc.12)** — `cli-ort` only |
| Inference (opt-in tract) | `POLYVOICE_INFERENCE_BACKEND=tract` + `backend-tract`: signed `powerset_fp32_tract` + **FP32** ResNet; smoke DER only |
| Models | Profile segmenter/embedder minisign-signed in release; VBx PLDA registry downloads are minisign-signed; opt-in `powerset_fp32_tract` is minisign-signed (release `models-tract-v1`) |
| Native ORT binary | Hash-pinned via ort-sys `dist.txt` **when `onnx` is enabled**; trust model in [`docs/security/ort-native-binary-provenance.md`](docs/security/ort-native-binary-provenance.md) |
| Library features | `pipeline-native` + `vbx` (CLI parity) or `pipeline-full` + `vbx` (ONNX). Crate-root `Pipeline` needs that gate; `PipelineConfig::default()` is **VBx** when `vbx` is on |

Honest reading: v2+VBx INT8 kernels are the **measured product pipeline** on
Linux (Vox 13.34 % / AMI 24.19 %) and Darwin Vox-3 (scoreboard floors).
Linux/CPU **ort** is a comparison protocol, not the product CLI. Legacy remains
a supported escape hatch. Tract is an **opt-in research path**. Public
production still needs multi-corpus gates, an API freeze, and a Darwin
full-split re-run after the AHC seed 0.6 retune.

---

## Gap Analysis

### 1. Version & API Stability ❌

| Item | Status | Risk |
|------|--------|------|
| Semantic version | `0.20.0` | Pre-1.0 — API may change between `0.x` minors |
| `semver-checks` | Passes in CI | Only checks public API surface; pre-1.0 still allows breaking changes |
| CHANGELOG | Maintained | Tracks 0.11→0.20; CLI default flip to v2+VBx was 0.11; kernels default was 0.18; WAVE `ryf` was 0.19 |

**Gap:** No commitment to backward compatibility until `1.0.0`. Consumers should
pin a `0.19.x` (or tighter) and read the CHANGELOG before upgrading.

**Remediation:** Freeze the public API, publish a semver policy, then ship
`1.0.0`.

---

### 2. Dependency Supply Chain ⚠️

| Dependency | Version | Risk |
|------------|---------|------|
| `polyvoice-kernels` | workspace | Product CLI. Darwin uses Accelerate/BNNS (C shims); Linux uses `rten-gemm` (pure Rust). MSRV 1.94. |
| `ort` (ONNX Runtime) | `2.0.0-rc.12` | **RC, not stable.** Still linked by **`cli-ort`**. Not on the product CLI or Python wheel. |
| Native ORT binary | pinned via ort-sys | Hash-verified download when `onnx` is on; residual trust in pyke builds + CDN cold-fetch |
| `faer` (spectral clustering) | Optional | Not used in the default pipeline |
| `paste` | Latest | Unmaintained (LOW; no CVE) |

**Gap:** `ort` is no longer the product-CLI or Python-wheel backend. Residual
risk is the opt-in `cli-ort` RC track. Tract is a spike/goal, not shipped
parity. Kernels replace ort for CLI/FFI/MCP/Python.

**Remediation:**
- Keep the product CLI and Python wheel on kernels; do not pull `ort` back
  into `cli` or the wheel.
- Track `ort` 2.0 stable for `cli-ort`; re-verify pins and DER on every
  RC → stable bump.
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
on `cli-ort`.

---

### 4. Correctness Verification ✅ / ⚠️

| Tool | Coverage | Note |
|------|----------|------|
| Unit / integration tests | Broad `src/` + `tests/` | Structural coverage good |
| Native scoreboard | `tests/native_scoreboard.rs` | Darwin Vox-3 floors: DER, RTF, model bytes, peak RSS |
| Miri | Focused PR-gate set | `ffi_smoke`, `miri_resegmentation`, `test_ahc` — not a full-lib multi-hour run |
| Loom | `loom_pool.rs` | Session / pool concurrency model |
| Proptest | In CI | DER / k-means / AHC / types property suites |
| DER regression gates | Legacy + v2 + Linux/CPU ort + Linux native | Headline no-collar metric release-gated; Linux native full-split filled 2026-09-13 |

**Gap:** Full-lib Miri is intentionally not the PR gate (cost). Darwin
full-split has not been re-run since the VBx AHC seed 0.6 retune.

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
| pipeline v2 + VBx **INT8** **Linux/CPU ort** (Vox / AMI) | 232 / 16 | **14.94%** / **24.19%** | 10.27% / 16.60% | Comparison protocol (`cli-ort`); gate + CI smoke |
| Darwin native kernels (Vox / AMI, M1 Pro) | 232 / 16 | **15.47%** / **25.19%** | — | 0.18 product CLI; RTFx ~130× / ~109×; not re-run after AHC seed 0.6 |
| Linux native kernels | 232 / 16 | **13.34%** / **24.19%** | — | 2026-09-13, AHC seed 0.6; RTFx ~162× / ~193× |
| tract pure-Rust (3 short Vox, M1 Pro) | 3 | ~**7.22%** (vs ort ~7.41%) | — | Opt-in; not a release gate |
| tract pure-Rust (10 shortest Vox, ≈560 s) | 10 | **8.86%** (vs ort **9.18%**) | — | RTFx ~11 vs ~99 |
| tract pure-Rust (**AMI-test 16**, M1 Pro) | 16 | **23.42%** (vs ort **24.63%**) | — | RTFx ~19 vs ~154; `scripts/tract-der-gate.sh` |
| CALLHOME | — | — | — | **Not measured / not gated** |
| DIHARD | — | — | — | **Not measured / not gated** |

**Gap:** The default v2+VBx INT8 path has full-split VoxConverse and AMI on
desktop baselines, the **Linux/CPU ort** comparison protocol, and **Linux
native kernels** (13.34 % / 24.19 %). Darwin native full-split is measured
but predates AHC seed 0.6. **Multi-corpus DER beyond Vox/AMI remains
absent**: no CALLHOME/DIHARD release gate. Linux kernels trail pyannote 3.1
published 11.3 % by about **2 pp** no-collar on VoxConverse. Tract is **not**
release-gated at full-split size.

**Remediation:**
- Cite Linux kernels as the non-Apple product protocol
  ([`linux-cpu-native-der-2026-09-13-vbx-ahc/`](benchmarks/results/linux-cpu-native-der-2026-09-13-vbx-ahc/)).
  Ort remains a comparison row (`cli-ort`).
- Re-run Darwin full-split after AHC seed 0.6 before treating 15.47 % / 25.19 %
  as current: `bash scripts/darwin-native-der-gate.sh` on macOS.
- Add at least one additional corpus (CALLHOME and/or DIHARD subset) to the
  release DER matrix.
- Do not pull `ort` back into `cli`.

---

### 6. Pipeline story (honest dual path) ⚠️

| Path | How to run | Role in 0.20.x |
|------|------------|----------------|
| **v2 + VBx kernels (CLI/FFI/MCP/Python/transcribe default)** | `cargo install polyvoice --features cli` / `pip install polyvoice` | Product; Darwin scoreboard + Linux full-split |
| **v2 + VBx ONNX Runtime** | `--features cli-ort` (deprecated) | Comparison protocol; not the product CLI |
| **Legacy** | CLI `--legacy` / `--clusterer ahc` | Supported escape hatch; former default (Silero + AHC) |

**Gap:** The pipeline default flipped at 0.11 (v2+VBx) and the engine default
flipped at 0.18 (kernels). Legacy still ships as an escape hatch, so dual
pipelines continue to tax docs, gates, and bindings. Library
`PipelineConfig::default()` matches the front doors (**VBx** when the `vbx`
feature is on). 1.0 should not ship with two first-class paths; retire or
clearly demote legacy once v2+VBx has broader multi-corpus proof.

---

### 7. Inference runtime independence ⚠️

| Item | Status |
|------|--------|
| Product CLI/FFI/MCP/Python | **`polyvoice-kernels`** (`pipeline-native`) — no `InferenceRuntime` dylib |
| `InferenceRuntime` trait | **Exists** (`src/onnx/runtime.rs`) for ONNX-shaped backends |
| ONNX implementation | **`OrtSession`** (`ort` 2.0.0-rc.12) — `cli-ort` |
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
| Python wheels | ✅ | Maturin (macOS / Linux / Windows); kernels, no `ort` |

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

_As of 0.20.x — product CLI, Python wheel, and transcribe diarization are
kernels, `cli-ort` still `ort` 2.0.0-rc.12, INT8 profiles + v2+VBx default,
legacy as an escape hatch, and multi-corpus DER is incomplete. Public
unattended stays NO-GO._

| Scenario | Verdict | Rationale |
|----------|---------|-----------|
| Internal microservice (controlled audio, ops on-call) | **GO with caveats** | Pin crate; prefer the kernel CLI to avoid `ort` RC; re-run DER after upgrades; no public SLA |
| Desktop app (local processing) | **GO** | User owns hardware; ~8.4 MB INT8; no `libonnxruntime` on the product CLI; tolerate pre-1.0 API |
| Public cloud API (multi-tenant, unattended) | **NO-GO** | Dual pipeline, thin multi-corpus proof, pre-1.0 API |
| Embedded / edge (aarch64) | **GO with testing** | Cross-compile works; measure DER/RTF on target hardware (Linux native RTF ≠ Darwin) |
| Security-critical (government, finance) | **NO-GO** | Needs broader audit + multi-corpus validation |

---

## 1.0 GO checklist

All items must be true before declaring production-ready / shipping `1.0.0` as
**GO** for broader deployment. Worded as outcomes — not internal tracker IDs.

- [ ] **Single default pipeline.** One validated CLI/Python/FFI path; no dual
      “legacy vs experimental” default. Experimental flags may remain for R&D
      but must not be required for the shipped claim. Library
      `PipelineConfig::default()` matches front-door VBx when `vbx` is on.
- [ ] **Public API freeze + semver policy.** Documented stability rules; no
      silent breaking churn on the advertised surface for a freeze window; then
      `1.0.0`.
- [ ] **Runtime story closed.** Product CLI, Python wheel, and transcribe
      diarization are kernels. Remaining: `cli-ort` still `ort` RC; Parakeet
      TDT still uses `ort`; tract remains opt-in smoke; `ort` 2.x stable
      should be re-verified for the opt-in path.
- [ ] **Multi-corpus DER gate.** Release-blocking DER on VoxConverse **and** AMI
      **and** at least one additional corpus (CALLHOME and/or DIHARD subset),
      with collar and overlap policy published next to the numbers.
- [ ] **Accuracy target path.** VoxConverse-test no-collar success metric on the
      default path at **≤13–14%** (stretch ≤12%), with AMI not stagnating in the
      high-20s/30s without a documented plan — see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
      Linux kernels are **13.34 %** / AMI **24.19 %**. Darwin full-split is
      still **15.47 %** / **25.19 %** (0.18, pre AHC seed 0.6).
- [ ] **This document says GO.** `PRODUCTION-READINESS.md` re-reviewed and
      signed off for the intended deployment class (internal vs public).

Until every box is checked, the honest status remains:

> **NOT GO for public unattended production; OK for controlled internal use.**

---

## Recommended blockers (summary)

| Blocker | Why it blocks 1.0 / public GO |
|---------|-------------------------------|
| Dual pipeline families (BYO vs v2) | Intentional; still doubles docs/gates if not documented |
| `cli-ort` still `ort` RC | Supply-chain risk on the opt-in ONNX CLI |
| Darwin full-split stale vs AHC seed 0.6 | Linux remeasured; Darwin 15.47 % / 25.19 % is 0.18 |
| Thin multi-corpus DER | Outside Vox/AMI only NOTSOFAR micro-gate; no CALLHOME/DIHARD |
| Pre-1.0 API | Breaking changes without major bump |
| Accuracy gap vs leaders | ~2 pp no-collar on VoxConverse vs pyannote 3.1 (11.3 %); speaker counting still dominant error |

---

## Metrics (snapshot, 0.20.x)

| Metric | Value |
|--------|-------|
| Crate version | 0.20.0 |
| Deployable footprint | **~8.4 MB** INT8 production pair (FP32 ids optional / not profile-default) |
| Product CLI engine | kernels (`pipeline-native`); no `libonnxruntime` |
| Speed (kernels, Darwin Vox-3 scoreboard) | ≥**117×** realtime; peak RSS ≤ **556 MiB** |
| Speed (kernels, Darwin full-split M1 Pro) | Vox ~**130×**; AMI ~**109×** |
| Speed (kernels, Linux Vox-3, Ryzen AI 9 HX 370) | ~**111×** jobs=1; ~**158×** wall at `--jobs 3` |
| Speed (kernels, Linux full-split) | Vox ~**162×**; AMI ~**193×** |
| Speed (INT8, Linux/CPU **ort** full-split, same host) | Vox ~**150×**; AMI ~**171×** |
| VoxConverse-test DER (v2+VBx INT8 Linux **kernels**, 232, collar 0) | **13.34%** |
| VoxConverse-test DER (v2+VBx INT8, 232, collar 0, **ort** host) | **15.02%** |
| VoxConverse-test DER (v2+VBx INT8 Linux/CPU **ort**, 232, collar 0) | **14.94%** |
| VoxConverse-test DER (v2+VBx INT8 Darwin **kernels**, 232, collar 0) | **15.47%** |
| VoxConverse-test DER (legacy, 232, collar 0) | 18.54% |
| AMI-test DER (v2+VBx INT8 Linux **kernels**, 16, collar 0) | **24.19%** |
| AMI-test DER (v2+VBx INT8, 16, collar 0, **ort** host / Linux) | **24.50%** / **24.19%** |
| AMI-test DER (v2+VBx INT8 Darwin **kernels**, 16, collar 0) | **25.19%** |
| AMI-test DER (legacy, 16, collar 0) | 32.87% |
| Default pipeline | v2 + VBx |
| Default CLI engine | kernels (0.18+) |
| Escape hatch | legacy (`--legacy` / `--clusterer ahc`); ONNX CLI (`cli-ort`) |
| Inference backends | **Product CLI / Python:** kernels. **Opt-in ONNX CLI:** `cli-ort`. **Opt-in:** tract |
| Model authenticity | Minisign; required on release profile resolution |
| Security audit (cargo audit on green main) | 0 HIGH, 0 MEDIUM expected |

For competitor context, collar protocol, and reproduction commands, use
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — do not treat this readiness file as
the accuracy source of truth.
