# Production Readiness Assessment

> **Version:** 0.7.0 | **Date:** 2026-06-14 | **Scope:** Rust library + Python bindings + FFI
>
> **Last updated:** 2026-06-14 — refreshed for 0.7.0. The legacy v0.5 pipeline is the validated CLI/Python default; pipeline v2 was reverted from the CLI default after the 0.6.1 long-form DER regression and remains opt-in (`--v2`). No longer an alpha — this is a 0.7.x pre-1.0 release.

## Executive Summary

**Status: NOT production-ready.**

**(As of 0.7.0.)** The project is hardened against common attack vectors and passes an extensive CI matrix, but it is a pre-1.0 (`0.7.x`) release with no backward-compatibility commitment until `1.0.0`, a key dependency (`ort`) is still a release candidate (`2.0.0-rc.12`), and cross-dataset validation is thin. It is suitable for **controlled internal deployments** where the audio pipeline and environment are known. It is not yet suitable for **public APIs** or **unattended production services**.

---

## Gap Analysis

### 1. Version & API Stability ❌

| Item | Status | Risk |
|------|--------|------|
| Semantic version | `0.7.0` | Pre-1.0 — API may change between `0.x` minors |
| `semver-checks` | Passes in CI | Only checks public API surface; pre-1.0 allows breaking changes |
| CHANGELOG | Exists | Good; the `0.6.x` patch cadence is rapid |

**Gap:** No commitment to backward compatibility until `1.0.0`. Consumers should pin to a `0.7.x` minor and review the CHANGELOG before upgrading.

**Remediation:** Stabilize the public API toward `1.0.0`, then adhere to semver.

---

### 2. Dependency Supply Chain ⚠️

| Dependency | Version | Risk |
|------------|---------|------|
| `ort` (ONNX Runtime) | `2.0.0-rc.12` | **RC, not stable.** Runtime behavior may change. No security advisory channel for RCs. |
| `faer` (spectral clustering) | Latest | Optional (`spectral` feature). Not used in default pipeline. |
| `paste` | Latest | Unmaintained (LOW severity, no CVE). |

**Gap:** `ort` is the single highest-risk dependency. It bridges Rust to a large C++ runtime (ONNX Runtime). An `ort` 2.0 stable release could introduce breaking changes or require model re-export.

**Remediation:**
- Track `ort` 2.0 stable release. Test immediately on RC → stable transition.
- Pin `ort` to exact RC version with a comment linking to the stable tracking issue.
- Consider vendoring ONNX Runtime or providing a static-link option for supply-chain isolation.

---

### 3. Security Posture ✅

| Control | Status | Evidence |
|---------|--------|----------|
| Model signing (Minisign) | Implemented | Streaming verification in 64 KB chunks, pubkey baked into binary |
| ONNX header validation | Implemented | Pre-load DOS guard (`ONNX_MIN_HEADER_BYTES`) |
| TLS pinning | Implemented | `ureq` → `rustls` with `webpki-roots` |
| FFI sandbox | Implemented | Path traversal guard, `MAX_SAMPLES` limit, panic logging |
| `cargo audit` | Passes | 0 HIGH, 0 MEDIUM, 0 CVEs |
| Fuzzing | Active | `fuzz_cluster_assign` (libFuzzer) |

**Gap:** Only LOW findings remain (`paste` unmaintained, JSON null-byte graceful failure). No critical gaps.

---

### 4. Correctness Verification ✅ / ⚠️

| Tool | Coverage | Note |
|------|----------|------|
| Unit tests | 175+ tests in `src/` | Good structural coverage |
| Miri | Runs on `--lib`, `--test test_ahc` | **Takes ~2 hours on CI** (see §6) |
| Loom | `loom_pool.rs` | Concurrency model checking for session pool |
| Proptest | In CI | `property_der_test`, `property_kmeans_test` run in the `test` job |

**Gap:** Miri runtime is a CI bottleneck. If it is skipped or times out, UB coverage is lost.

---

### 5. Dataset Validation ❌

| Dataset | Files | DER (0.25 s collar) | Used in CI? |
|---------|-------|---------------------|-------------|
| VoxConverse test (legacy) | 232 | 13.83% (no-collar not measured) | No (232-file not gated) |
| VoxConverse test (legacy) | 10 | 17.43% collar / 25.99% no-collar | Yes (10-file gated) |
| e2e smoke (legacy) | 1 | 6.62% | Yes (gated) |
| AMI EN2002a (legacy, 1 meeting) | 1 | 36.30% collar / 44.73% no-collar | Yes (gated) |
| CALLHOME | — | — | Not measured |
| CHiME | — | — | Not measured |
| VoxCeleb1 | Subset only | — | Speaker ID, not diarization |

All DER figures are sourced from `tests/der_baseline.json`.

**Gap:** DER numbers exist for only VoxConverse + a single AMI meeting; the AMI figure is **one** meeting (EN2002a, ~79% overlap), **not** a 16-meeting average (the older "~23%" claim had no committed backing and is withdrawn). No cross-corpora validation. The default pipeline is the legacy v0.5.2 (DER 13.83%, 232-file collar=0.25) — the experimental M6b `pipeline_v1` was demoted due to DER 52–64%.

**Remediation:**
- Run AMI full-test DER in CI (currently `perf-regression` is schedule-only).
- Add CALLHOME and CHiME evaluation scripts.
- Document expected DER variance per dataset in README.

---

### 6. CI / DX Performance ⚠️

| Job | Runtime | Issue |
|-----|---------|-------|
| `miri` | **~2 hours** | Blocks merge feedback loop |
| `e2e-smoke` | ~1.5 min (after fix) | Was ~25 min before bundled test clip |
| `test (windows-latest)` | ~4 min | Acceptable |
| `build (windows-latest, py)` | ~3 min | Acceptable |

**Gap:** Miri is the single longest job. It dominates wall-clock CI time. If it fails, the failure surface is large (all lib tests + integration tests in one sequential run).

**Root cause analysis (Miri):**
- `cargo miri test --features ffi --lib` compiles and runs 175 unit tests.
- `cargo miri test --features ffi --test integration` references a **deleted** target (`tests/integration.rs` removed in `75d92ca`). This command exits with code 101, but the CI job historically passed because the file existed on the commit that produced the 2h9m run.
- The old `tests/integration.rs` contained `OfflineDiarizer` / `OnlineDiarizer` tests over 10 s of synthetic audio. Miri interprets every instruction; even lightweight float loops are ~100× slower.
- **Compounding factor:** `ort` crate pulls in heavy build-time logic. Miri must interpret any `unsafe` init paths in `ort`.

**Remediation:**
1. **Fix the CI command:** Remove `--test integration` (target no longer exists).
2. **Split Miri into parallel jobs:** `--lib`, `--test miri_resegmentation`, `--test test_ahc`. Reduces single-job runtime and provides granular failure signals.
3. **Tag heavy tests with `#[cfg_attr(miri, ignore)]`:** Already done for `test_fbank_shape`. Audit remaining 174 tests for any that exercise large loops or `ort` init.
4. **Consider `cargo miri nextest`:** Parallel test runner reduces wall-clock time.
5. **Schedule Miri nightly, not per-PR:** Move Miri from PR gates to a scheduled run. Keep a fast Miri smoke test (e.g., `miri_resegmentation.rs`) in PR.

---

### 7. Platform Coverage ✅

| Target | CI | Notes |
|--------|-----|-------|
| x86_64 Linux | ✅ | Primary |
| x86_64 macOS | ✅ | |
| x86_64 Windows | ✅ | |
| aarch64 Linux | ✅ | `cross-aarch64-linux` |
| wasm32 | ✅ | `wasm32-smoke` (compilation only) |
| Python (macOS/Linux/Windows) | ✅ | Maturin wheels |

---

### 8. Documentation & Onboarding ✅

| Asset | Status |
|-------|--------|
| README | Badges, multi-language quick starts, canonical benchmark table |
| `docs/` | Formalism, glossary, pipeline, severity, migrating guides |
| `AGENTS.md` | Coding guidelines for contributors |
| FFI examples | C header + Python tests |

---

## Go/No-Go Matrix

_As of 0.7.0 — `ort` is still `2.0.0-rc.12` and cross-corpora validation is still thin, so the NO-GO verdicts below stand._

| Scenario | Verdict | Rationale |
|----------|---------|-----------|
| Internal microservice (controlled audio, ops on-call) | **GO with caveats** | Pin `ort` RC, monitor memory, no public SLA |
| Desktop app (local processing) | **GO** | User owns the hardware, can tolerate ~30 MB RAM |
| Public cloud API (multi-tenant) | **NO-GO** | `ort` RC risk, no CHiME/CALLHOME validation, API may break |
| Embedded / edge (aarch64) | **GO with testing** | Cross-compilation works, but measure DER on target hardware |
| Security-critical (government, finance) | **NO-GO** | Needs `ort` stable + independent security audit |

---

## Recommended Blockers for `1.0.0` Stable

1. [ ] `ort` 2.0 stable released and integrated
2. [ ] Miri CI split into parallel jobs or moved to nightly
3. [ ] CALLHOME + CHiME DER benchmarks documented
4. [ ] API frozen (no `#[doc(hidden)]` churn for 2 weeks)
5. [ ] Python wheel tested on a clean VM (no Rust toolchain)

---

## Metrics

| Metric | Value |
|--------|-------|
| Crate size (crates.io) | 1.5 MiB |
| Runtime memory (Balanced profile) | ~30 MB |
| Speed (CPU) | 10× RT |
| VoxConverse DER (legacy, 232, 0.25 s collar) | 13.83% |
| AMI DER (legacy, EN2002a single, 0.25 s collar) | 36.30% (44.73% no-collar) |
| CI checks | 30 |
| Security audit | 0 HIGH, 0 MEDIUM, 0 CVE |
