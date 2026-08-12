# Production Readiness Assessment

> **Version:** 0.17.x | **Date:** 2026-08-12 | **Scope:** Rust library + Python bindings + FFI + CLI
>
> **Last updated:** 2026-08-12 — crate **0.17.0** ships **INT8-only** profiles
> (`powerset_int8` + `resnet34_int8`). Pipeline v2+VBx default since 0.11.
> Linux/CPU full-split DER gate landed; pure-Rust tract path is **opt-in smoke
> only**, not product default. Canonical accuracy protocol:
> [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Zero-deps strategy:
> [`docs/strategy/zero-deps.md`](docs/strategy/zero-deps.md).

## Executive Summary

**Status: NOT GO for public unattended production. OK for controlled internal use.**

As of **0.17.x**, polyvoice is a hardened pre-1.0 engine: model signing is
enforced on release builds for profile-resolved models, the ONNX Runtime native
binary is hash-pinned and documented, CI covers the main desktop targets, and
**full VoxConverse-test + AMI-test DER** (including an official **Linux/CPU**
protocol) keeps **pipeline v2 + VBx** as the CLI / FFI / Python / MCP default.
It is still **not** ready for multi-tenant public APIs or unattended production
services, because:

1. **Pre-1.0 API** — no backward-compatibility commitment until `1.0.0`.
2. **`ort` is still a release candidate** (`2.0.0-rc.12`) and remains the only
   **product-default** inference backend.
3. ~~**VBx PLDA signatures**~~ — registry PLDA entries are minisign-signed
   (local `fixtures/vbx-plda/` override still works without network).
4. **Cross-corpus validation is thin** — solid VoxConverse + AMI coverage;
   NOTSOFAR-1 has a measured micro-gate (3-meeting subset) but CALLHOME /
   DIHARD (and similar) are not release-gated.
5. **Pure-Rust (tract) path is not product-ready** — opt-in only
   (`backend-tract` + signed `powerset_fp32_tract` + FP32 ResNet); ~9× slower
   than ort; no full-split release gate.

**Suitable for:** controlled internal services, desktop apps, and edge pilots
where audio conditions are known and operators can pin versions and re-verify
DER after upgrades.

**Not suitable for:** public multi-tenant APIs, unattended SLA-bound services,
or security-critical deployments that require a stable runtime + multi-corpus
proof.

---

## Current surface (0.17.x truth)

| Area | State |
|------|--------|
| Crate version | `0.17.0` (0.17.x line; HEAD may be 0.17.0+commits) |
| Production models | **INT8 only** (`powerset_int8` + `resnet34_int8`, ~8.4 MB) |
| CLI default | **v2 + VBx** (powerset → ResNet34 → VB-HMM/PLDA); `--legacy` / `--clusterer ahc` opt out |
| Python default | Pipeline v2 + **VBx** (same as CLI); pass `clusterer="ahc"` to opt out |
| Full-split DER (no-collar micro, INT8) | Host baseline Vox **15.02%** / AMI **24.50%** — [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| **Linux/CPU** full-split (product protocol) | Vox **14.94%** / **10.27%** @0.25 / ~**82×** RTFx; AMI **24.19%** / **16.60%** / ~**95×** — `tests/der_baseline.json` `*_linux_cpu`, `scripts/linux-cpu-der-gate.sh` |
| Inference (product) | **`OrtSession` (ort)** only |
| Inference (opt-in pure-Rust) | `POLYVOICE_INFERENCE_BACKEND=tract` + feature `backend-tract`: signed `powerset_fp32_tract` + **FP32** ResNet; smoke DER only — **not** product default |
| Models | Profile segmenter/embedder minisign-signed in release; VBx PLDA registry downloads are minisign-signed; opt-in `powerset_fp32_tract` is minisign-signed (release `models-tract-v1`) |
| Native ORT binary | Hash-pinned via ort-sys `dist.txt`; trust model in [`docs/security/ort-native-binary-provenance.md`](docs/security/ort-native-binary-provenance.md) |
| Library ONNX features | `pipeline-full` (+ optional `vbx`) exports crate-root `Pipeline` |

Honest reading: v2+VBx + ort INT8 is the **measured product default** on full
VoxConverse-test and AMI-test (desktop + Linux/CPU gate). Legacy remains a
supported escape hatch. Pure-Rust tract is an **opt-in research path**. Public
production still needs multi-corpus gates, a stable ort (or measured pure-Rust
default), and signed PLDA distribution.

---

## Gap Analysis

### 1. Version & API Stability ❌

| Item | Status | Risk |
|------|--------|------|
| Semantic version | `0.17.0` | Pre-1.0 — API may change between `0.x` minors |
| `semver-checks` | Passes in CI | Only checks public API surface; pre-1.0 still allows breaking changes |
| CHANGELOG | Maintained | Tracks 0.11→0.17; CLI default flip to v2+VBx was 0.11 |

**Gap:** No commitment to backward compatibility until `1.0.0`. Consumers should
pin a `0.17.x` (or tighter) and read the CHANGELOG before upgrading.

**Remediation:** Freeze the public API, publish a semver policy, then ship
`1.0.0`.

---

### 2. Dependency Supply Chain ⚠️

| Dependency | Version | Risk |
|------------|---------|------|
| `ort` (ONNX Runtime) | `2.0.0-rc.12` | **RC, not stable.** Single production inference backend. EP / API churn possible. |
| Native ORT binary | pinned via ort-sys | Hash-verified download; residual trust in pyke builds + CDN cold-fetch |
| `faer` (spectral clustering) | Optional | Not used in the default pipeline |
| `paste` | Latest | Unmaintained (LOW; no CVE) |

**Gap:** `ort` is still the highest-risk dependency: RC track + C++ native
runtime + only production `InferenceRuntime` implementation. A pure-Rust
optional backend (e.g. tract) is a spike/goal, not shipped parity.

**Remediation:**
- Track `ort` 2.0 stable; re-verify pins and DER on every RC → stable bump.
- Keep the `InferenceRuntime` surface clean so a second backend can land without
  rewriting stages.
- Retain provenance docs and CI cache of the verified native binary.

Evidence: [`docs/security/ort-native-binary-provenance.md`](docs/security/ort-native-binary-provenance.md),
`Cargo.toml` pin, `scripts/check-ort-version.sh`.

---

### 3. Security Posture ✅

| Control | Status | Evidence |
|---------|--------|----------|
| Model signing (Minisign) | Implemented | Streaming verify; pubkey baked in; **release builds require signatures** for profile-resolved models |
| ONNX header validation | Implemented | Pre-load DOS guard |
| ORT native binary provenance | Documented + CI-cached | [`docs/security/ort-native-binary-provenance.md`](docs/security/ort-native-binary-provenance.md) |
| TLS for downloads | Implemented | `ureq` + `rustls` + `webpki-roots` |
| FFI sandbox | Implemented | Path traversal guard, sample limits, panic logging |
| `cargo audit` | In CI | 0 HIGH / 0 MEDIUM expected on green main |
| Fuzzing | Active | libFuzzer targets for fbank, VAD, overlap, cluster assign |

**Gap:** Residual LOW noise (e.g. unmaintained transitive crates). No independent
third-party security audit. RC-track runtime is itself a supply-chain residual.

---

### 4. Correctness Verification ✅ / ⚠️

| Tool | Coverage | Note |
|------|----------|------|
| Unit / integration tests | Broad `src/` + `tests/` | Structural coverage good |
| Miri | Focused PR-gate set | `ffi_smoke`, `miri_resegmentation`, `test_ahc` — not a full-lib multi-hour run |
| Loom | `loom_pool.rs` | Session / pool concurrency model |
| Proptest | In CI | DER / k-means / AHC / types property suites |
| DER regression gates | Legacy + selected v2 smoke | Headline no-collar metric release-gated for legacy subsets |

**Gap:** Full-lib Miri is intentionally not the PR gate (cost). Experimental
pipeline paths have thinner automated DER coverage than legacy.

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
| pipeline v2 + VBx **INT8** (Vox / AMI, **default**) | 232 / 16 | **15.02%** / **24.50%** | 10.33% / 16.82% | Default models since 0.17; full-split 2026-08-10 |
| pipeline v2 + VBx **INT8** **Linux/CPU** (Vox / AMI) | 232 / 16 | **14.94%** / **24.19%** | 10.27% / 16.60% | Official non-Apple protocol; gate + CI smoke |
| tract pure-Rust (3 short Vox, M1 Pro) | 3 | ~**7.22%** (vs ort ~7.41%) | — | Opt-in; not a release gate |
| tract pure-Rust (10 shortest Vox, ≈560 s) | 10 | **8.86%** (vs ort **9.18%**) | — | RTFx ~11 vs ~99; still not full-split |
| CALLHOME | — | — | — | **Not measured / not gated** |
| DIHARD | — | — | — | **Not measured / not gated** |

**Gap:** The default v2+VBx INT8 path has full-split VoxConverse and AMI on both
desktop baselines and the **Linux/CPU** gate, but **multi-corpus DER beyond
Vox/AMI remains absent**: no CALLHOME/DIHARD release gate. Accuracy still trails
pyannote-class systems by roughly ~4 pp no-collar on VoxConverse (see benchmarks).
Tract pure-Rust is **not** release-gated at full-split size.

**Remediation:**
- Keep legacy published as the honesty baseline alongside the v2+VBx default
  (the 0.11 full-split gate already flipped the default).
- Keep Linux/CPU as the non-Apple product truth (do not substitute CoreML RTF).
- Add at least one additional corpus (CALLHOME and/or DIHARD subset) to the
  release DER matrix.
- Before any pure-Rust product default: larger tract DER/RTF + shipped rewrite
  artifact + MSRV policy for tract.

---

### 6. Pipeline story (honest dual path) ⚠️

| Path | How to run | Role in 0.17.x |
|------|------------|----------------|
| **v2 + VBx (default)** | CLI default (`--v2` is a hidden no-op kept for old scripts); PLDA dir/env/registry | Production accuracy path; won the full-split Vox + AMI DER gate |
| **Legacy** | CLI `--legacy` / `--clusterer ahc` | Supported escape hatch; former default (Silero + AHC) |

**Gap:** The default flip landed at 0.11, but legacy still ships as an escape
hatch, so dual pipelines continue to tax docs, gates, and bindings. 1.0 should
not ship with two first-class paths; retire or clearly demote legacy once
v2+VBx has broader multi-corpus proof.

---

### 7. Inference runtime independence ⚠️

| Item | Status |
|------|--------|
| `InferenceRuntime` trait | **Exists** (`src/onnx/runtime.rs`) |
| Production implementation | **`OrtSession` only** (`ort` 2.0.0-rc.12) — product default |
| Pure-Rust optional backend | **`TractSession`** behind `backend-tract` + `POLYVOICE_INFERENCE_BACKEND=tract` |
| Tract powerset | Shipping graphs fail load; **rewrite** via `scripts/export-powerset-tract.py`; pipeline remaps when present |
| Tract embedder | Builder forces **FP32** `wespeaker_resnet34` (INT8 ResNet under tract collapses speakers) |
| Tract accuracy | 3-file Vox smoke DER ≈ ort; **not** full-split gated; ~9× slower RTFx on smoke host |
| Execution providers | CoreML / XNNPACK (and related) wired as **ort-specific** config, not alternate runtimes |

**Gap:** Product still **locks to ort**. Tract is a real optional backend with
smoke evidence, not trait-only scaffolding — but rewrite models are not registry
shipped, INT8 embedder is unsafe under tract, and large-corpus DER/RTF is open.
See [`docs/strategy/zero-deps.md`](docs/strategy/zero-deps.md) and
`benchmarks/results/powerset-tract-rtf-der-2026-08-12/`.

---

### 8. CI / Platform Coverage ✅

| Target | CI | Notes |
|--------|-----|-------|
| x86_64 Linux | ✅ | Primary |
| x86_64 / aarch64 macOS | ✅ | CoreML path exercised where configured |
| x86_64 Windows | ✅ | |
| aarch64 Linux | ✅ | Cross job |
| wasm32 | ✅ | Compile / smoke (not full ONNX diarization) |
| Python wheels | ✅ | Maturin (macOS / Linux / Windows) |

Miri is a **focused** PR gate rather than a multi-hour full-suite job. Fuzz and
audit remain active.

---

### 9. Documentation & Onboarding ✅

| Asset | Status |
|-------|--------|
| README | Install, usage, links, honest accuracy framing |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Canonical DER / RTF with collar protocol |
| [`docs/PIPELINE-ARCHITECTURE.md`](docs/PIPELINE-ARCHITECTURE.md) | Pipeline who-calls-whom |
| [`docs/DEVELOPMENT-PROCESS.md`](docs/DEVELOPMENT-PROCESS.md) | Development process checklist |
| Security provenance | ORT native binary + model signing story |
| `CONTRIBUTING.md` | Setup and contribution guidelines |
| FFI | C header + examples / smoke tests |

---

## Go/No-Go Matrix

_As of 0.17.x — `ort` remains `2.0.0-rc.12`, INT8 profiles + v2+VBx default, legacy
as an escape hatch, and multi-corpus DER is incomplete. Public unattended stays
NO-GO._

| Scenario | Verdict | Rationale |
|----------|---------|-----------|
| Internal microservice (controlled audio, ops on-call) | **GO with caveats** | Pin crate + `ort` RC, monitor memory, re-run DER after upgrades, no public SLA |
| Desktop app (local processing) | **GO** | User owns hardware; ~8.4 MB INT8 footprint; tolerate pre-1.0 API |
| Public cloud API (multi-tenant, unattended) | **NO-GO** | RC runtime, dual pipeline, thin multi-corpus proof, pre-1.0 API |
| Embedded / edge (aarch64) | **GO with testing** | Cross-compile works; measure DER/RTF on target hardware |
| Security-critical (government, finance) | **NO-GO** | Needs stable runtime story + broader audit + multi-corpus validation |

---

## 1.0 GO checklist

All items must be true before declaring production-ready / shipping `1.0.0` as
**GO** for broader deployment. Worded as outcomes — not internal tracker IDs.

- [ ] **Single default pipeline.** One validated CLI/Python/FFI path; no dual
      “legacy vs experimental” default. Experimental flags may remain for R&D
      but must not be required for the shipped claim.
- [ ] **Public API freeze + semver policy.** Documented stability rules; no
      silent breaking churn on the advertised surface for a freeze window; then
      `1.0.0`.
- [ ] **Runtime story closed.** Either `ort` 2.x **stable** is integrated and
      re-verified, **or** the pure-Rust (tract) path is product-grade: shipped
      rewrite + embedder assets, full-split DER/RTF gate, MSRV policy, and no
      silent INT8 collapse (today: opt-in smoke only).
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
| Dual pipeline families (BYO vs ONNX) | Intentional; still doubles docs/gates if not documented |
| `ort` RC-only + single backend | Supply-chain and upgrade risk |
| Thin multi-corpus DER | Outside Vox/AMI only NOTSOFAR micro-gate; no CALLHOME/DIHARD |
| Pre-1.0 API | Breaking changes without major bump |
| Accuracy gap vs leaders | ~4 pp no-collar on VoxConverse; speaker counting still dominant error |

---

## Metrics (snapshot, 0.17.x)

| Metric | Value |
|--------|-------|
| Crate version | 0.17.0 |
| Deployable footprint | **~8.4 MB** INT8 production pair (FP32 ids optional / not profile-default) |
| Speed (INT8 default, CoreML, M1 Pro) | ~111–130× realtime (RTF ~0.008–0.009) |
| Speed (INT8, Linux/CPU full-split) | Vox ~**82×** RTFx; AMI ~**95×** RTFx |
| VoxConverse-test DER (v2+VBx INT8, 232, collar 0) | **15.02%** (host baseline) |
| VoxConverse-test DER (v2+VBx INT8 Linux/CPU, 232, collar 0) | **14.94%** |
| VoxConverse-test DER (v2+VBx INT8, 232, collar 0.25) | 10.33% host / **10.27%** Linux/CPU |
| VoxConverse-test DER (legacy, 232, collar 0) | 18.54% |
| AMI-test DER (v2+VBx INT8, 16, collar 0) | **24.50%** host / **24.19%** Linux/CPU |
| AMI-test DER (v2+VBx INT8, 16, collar 0.25) | 16.82% host / **16.60%** Linux/CPU |
| AMI-test DER (legacy, 16, collar 0) | 32.87% |
| Default pipeline | v2 + VBx |
| Escape hatch | legacy (`--legacy` / `--clusterer ahc`) |
| Inference backends | **Product:** ort (`OrtSession`). **Opt-in:** tract (`TractSession`, pure-Rust smoke) |
| Model authenticity | Minisign; required on release profile resolution |
| Security audit (cargo audit on green main) | 0 HIGH, 0 MEDIUM expected |

For competitor context, collar protocol, and reproduction commands, use
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — do not treat this readiness file as
the accuracy source of truth.
