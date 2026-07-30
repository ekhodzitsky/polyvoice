# Production Readiness Assessment

> **Version:** 0.11.x | **Date:** 2026-07-25 | **Scope:** Rust library + Python bindings + FFI + CLI
>
> **Last updated:** 2026-07-25 — default pipeline flipped to v2+VBx after the
> full-split DER gate. Canonical accuracy numbers live in
> [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md); this file is the deployment
> GO / NO-GO judgment, not a second leaderboard.

## Executive Summary

**Status: NOT GO for public unattended production. OK for controlled internal use.**

As of **0.11.x**, polyvoice is a hardened pre-1.0 engine: model signing is
enforced on release builds, the ONNX Runtime native binary is hash-pinned and
documented, CI covers the main desktop targets, and a **full VoxConverse-test +
AMI-test DER gate** promotes **pipeline v2 + VBx** as the CLI default. It is
still **not** ready for multi-tenant public APIs or unattended production
services, because:

1. **Pre-1.0 API** — no backward-compatibility commitment until `1.0.0`.
2. **`ort` is still a release candidate** (`2.0.0-rc.12`) and remains the only
   production inference backend.
3. **VBx PLDA signatures deferred** — default clustering auto-downloads the
   six PLDA `.npy` files via the model registry with SHA-256 verification;
   minisign signatures are still pending a release-key sign step (optional
   override: `POLYVOICE_VBX_PLDA_DIR` / `--vbx-plda-dir`).
4. **Cross-corpus validation is thin** — solid VoxConverse + AMI coverage;
   CALLHOME / DIHARD (and similar) not gated for release.

**Suitable for:** controlled internal services, desktop apps, and edge pilots
where audio conditions are known and operators can pin versions and re-verify
DER after upgrades.

**Not suitable for:** public multi-tenant APIs, unattended SLA-bound services,
or security-critical deployments that require a stable runtime + multi-corpus
proof.

---

## Current surface (0.11.x truth)

| Area | State |
|------|--------|
| Crate version | `0.11.0` (0.11.x line) |
| CLI default | **v2 + VBx** (powerset → ResNet34 → VB-HMM/PLDA); `--legacy` / `--clusterer ahc` opt out |
| Python default | Pipeline v2; VBx when PLDA env/arg is set, else AHC |
| Full-split DER (no-collar micro) | Vox **15.37%**, AMI **25.17%** (≤ legacy; gate PASS 2026-07-25) |
| Inference | `InferenceRuntime` trait exists; **only `OrtSession` (ort) is a production impl** |
| Models | Bundled models minisign-signed; release builds refuse unsigned profile-resolved models |
| Native ORT binary | Hash-pinned via ort-sys `dist.txt`; trust model in [`docs/security/ort-native-binary-provenance.md`](docs/security/ort-native-binary-provenance.md) |

Honest reading: v2+VBx is the **measured default** on full VoxConverse-test and
AMI-test. Legacy remains a supported escape hatch. Public production still
needs multi-corpus gates, a stable ort, and frictionless PLDA distribution.

---

## Gap Analysis

### 1. Version & API Stability ❌

| Item | Status | Risk |
|------|--------|------|
| Semantic version | `0.11.0` | Pre-1.0 — API may change between `0.x` minors |
| `semver-checks` | Passes in CI | Only checks public API surface; pre-1.0 still allows breaking changes |
| CHANGELOG | Maintained | 0.11.0 documents CLI default flip to v2+VBx |

**Gap:** No commitment to backward compatibility until `1.0.0`. Consumers should
pin a `0.11.x` (or tighter) and read the CHANGELOG before upgrading.

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
| pipeline v2 + VBx (Vox / AMI, **default**) | 232 / 16 | **15.37%** / **25.17%** | 11.12% / 17.66% | Default since 0.11; full-split numbers release-canonical |
| CALLHOME | — | — | — | **Not measured / not gated** |
| DIHARD | — | — | — | **Not measured / not gated** |

**Gap:** The default v2+VBx path now has full-split VoxConverse (15.37%) and
AMI (25.17%) numbers next to the legacy baselines above, but **multi-corpus DER
beyond Vox/AMI remains absent**: no CALLHOME/DIHARD release gate. Accuracy
still trails pyannote-class systems by roughly ~4 pp no-collar on VoxConverse
(see benchmarks).

**Remediation:**
- Keep legacy published as the honesty baseline alongside the v2+VBx default
  (the 0.11 full-split gate already flipped the default).
- Add at least one additional corpus (CALLHOME and/or DIHARD subset) to the
  release DER matrix.
- Re-measure full Vox + AMI after clustering upgrades land on the default path.

---

### 6. Pipeline story (honest dual path) ⚠️

| Path | How to run | Role in 0.12.x |
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
| Production implementation | **`OrtSession` only** (`ort` 2.0.0-rc.12) |
| Pure-Rust optional backend | Not shipped (spike/goal; full-pipeline pure-Rust parity not proven) |
| Execution providers | CoreML / XNNPACK (and related) wired as **ort-specific** config, not alternate runtimes |

**Gap:** Runtime lock-in to ort remains a production risk even after the trait
extraction. Trait existence is necessary but not sufficient for independence.

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
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | Development process checklist |
| Security provenance | ORT native binary + model signing story |
| `CONTRIBUTING.md` | Setup and contribution guidelines |
| FFI | C header + examples / smoke tests |

---

## Go/No-Go Matrix

_As of 0.11.x — `ort` remains `2.0.0-rc.12`, v2+VBx is the default with legacy
as an escape hatch, and multi-corpus DER is incomplete. Public unattended stays
NO-GO._

| Scenario | Verdict | Rationale |
|----------|---------|-----------|
| Internal microservice (controlled audio, ops on-call) | **GO with caveats** | Pin crate + `ort` RC, monitor memory, re-run DER after upgrades, no public SLA |
| Desktop app (local processing) | **GO** | User owns hardware; ~30 MB class footprint; tolerate pre-1.0 API |
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
      re-verified, **or** a documented pure-Rust optional backend exists with
      measured parity on the default pipeline (not trait-only scaffolding).
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
| Dual default pipeline | Split gates, split docs, split user trust |
| `ort` RC-only + single backend | Supply-chain and upgrade risk |
| Thin multi-corpus DER | Unknown behavior outside Vox/AMI |
| Pre-1.0 API | Breaking changes without major bump |
| Accuracy gap vs leaders | ~4 pp no-collar on VoxConverse; speaker counting still dominant error |

---

## Metrics (snapshot, 0.12.x)

| Metric | Value |
|--------|-------|
| Crate version | 0.12.0 |
| Deployable footprint | ~30 MB class |
| Speed (CPU, legacy steady-state) | ~33× realtime (RTF ~0.03, multi-core intra-op) |
| VoxConverse-test DER (v2+VBx default, 232, collar 0) | **15.37%** |
| VoxConverse-test DER (v2+VBx default, 232, collar 0.25) | 11.12% |
| VoxConverse-test DER (legacy, 232, collar 0) | 18.54% |
| AMI-test DER (v2+VBx default, 16, collar 0) | **25.17%** |
| AMI-test DER (v2+VBx default, 16, collar 0.25) | 17.66% |
| AMI-test DER (legacy, 16, collar 0) | 32.87% |
| Default pipeline | v2 + VBx |
| Escape hatch | legacy (`--legacy` / `--clusterer ahc`) |
| Inference backends | ort only (via `InferenceRuntime` → `OrtSession`) |
| Model authenticity | Minisign; required on release profile resolution |
| Security audit (cargo audit on green main) | 0 HIGH, 0 MEDIUM expected |

For competitor context, collar protocol, and reproduction commands, use
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — do not treat this readiness file as
the accuracy source of truth.
