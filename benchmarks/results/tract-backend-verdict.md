# tract backend spike — verdict

**Date:** 2026-07-24  
**Branch:** `feat/tract-backend`  
**tract:** `tract-onnx` 0.23.4 (Sonos, MIT/Apache-2.0)  
**Default backend:** unchanged — **ort** (`ort` 2.0.0-rc.12)  
**Opt-in feature:** `backend-tract`  
**Selection:** env `POLYVOICE_INFERENCE_BACKEND=tract` or `InferenceBackend::force(Some(Tract))`

## Recommendation

**Keep tract optional only.** Do **not** switch the default backend.

- Feed-forward WeSpeaker-style embedders (CAM++, ResNet34) load, run, and match ort numerically within fixed tolerances.
- Core pipeline models **Silero VAD** and **powerset segmentation** fail tract’s ONNX analyse / `into_optimized` path (nested `If`, `InstanceNormalization` shape rules). Without those, a full diarization pipeline cannot run on tract today.
- Release RTF is mixed: ResNet34 roughly competitive with ort on this machine; CAM++ still ~6× slower. Debug builds are much worse (10–70×) — never use debug for RTF claims.
- MSRV: tract-onnx declares **1.91**; this crate declares **1.88**. Enabling `backend-tract` requires a newer toolchain. **Do not bump crate MSRV for this spike** (coordinate separately if productized).

## Per-model status

| Model | Load (tract) | Run | Ort parity | Notes |
|-------|--------------|-----|------------|-------|
| `cam_pp_fp32.onnx` | OK | OK | **PASS** (abs≤1e-3 or rel≤1e-2) | Input `[1,T,80]` fbank |
| `wespeaker_resnet34.onnx` | OK | OK | **PASS** (same tol) | Input `[1,T,80]` fbank |
| `silero_vad.onnx` | **FAIL** | — | n/a | Nested `If` / `Squeeze` analysis failure even with concrete LSTM I/O facts |
| `powerset_fp32.onnx` | **FAIL** | — | n/a | `If_117` and/or `InstanceNormalization` analyse failure |
| `ecapa_tdnn_mel.onnx` | **FAIL** | — | n/a | External weight blob `ecapa_tdnn_mel.onnx.data` missing in this tree |

Tests live in `src/onnx/parity.rs` (feature `backend-tract`). Missing `models/*.onnx` → tests skip cleanly. Known load failures for silero/powerset are asserted as *documented incompatibilities* (suite stays green).

## RTF (single forward, release profile)

**Hardware:** Apple Silicon (host of 2026-07-24 measurement)  
**Profile:** `cargo test --release` with `CARGO_PROFILE_RELEASE_LTO=false` and `codegen-units=16` (crate default LTO=true is too slow to rebuild tract for a spike).  
**Method:** one cold load per backend, one timed `run_ordered` on fixed synthetic input; times from test harness stderr (`parity … ort=… tract=…`).  
**Threads:** ort `intra_threads=1`; tract default CPU executor.  
**Not** end-to-end DER / vox-30 (blocked on silero + powerset).

| Model | Input | ort (ms) | tract (ms) | tract/ort |
|-------|-------|----------|------------|-----------|
| cam_pp_fp32 | `[1,200,80]` | ~50 | ~315 | **~6.3×** |
| wespeaker_resnet34 | `[1,200,80]` | ~414\* | ~305 | **~0.74×** |

\*ResNet ort sample looked cold/noisy; treat as order-of-magnitude only. Re-run with warmup + multi-iter benches before product decisions.

Debug-profile spot checks (unoptimized): cam++ ~35–70× slower, resnet ~11× — **discard for RTF**.

## Incompatibilities (tract 0.23.4)

1. **Silero VAD** — `into_optimized` fails analysing nested `If` branches (`If_0` → decoder `Squeeze` with non-unit axis). Concrete facts for `input [1,576]`, `state [2,1,128]`, `sr []` do not unblock. LSTM state *as named tensors* is fine in our trait design; the blocker is graph load, not the trait.
2. **Powerset** — `If_117` and `InstanceNormalization` (`/sincnet/wav_norm1d`) fail shape/fact unification under both direct optimize and symbol-bound free dims (`N,1,T`).
3. **ECAPA mel** — ONNX external data path; not a tract op issue.
4. **Execution providers** — tract path is pure-Rust CPU only; CoreML/XNNPACK/etc. remain ort-only.

Research note: source-level op registration ≠ end-to-end analyse success. This spike confirms that gap for silero + powerset.

## Factory / wiring

- `InferenceRuntime` unchanged in spirit; stages store `RuntimeSession` (enum: ort | tract).
- `build_session_with_ep` → `RuntimeSession::from_path` (header validation first).
- Default resolve order: `InferenceBackend::force` override → env `POLYVOICE_INFERENCE_BACKEND` → **ort**.
- `cargo tree -e normal` without `backend-tract`: **no** `tract*` crates.
- Candle: **not implemented** (NO-GO: missing biLSTM/InstanceNorm for powerset; not revisited here).

## rten fallback (written only — no impl)

**rten** (~0.24, MIT/Apache) is the secondary pure-Rust candidate if tract stays blocked on If/InstanceNorm for our audio zoo:

- Author claims CPU competitiveness with ORT on some models (including Apple Silicon reports).
- **Solo-developer bus factor** — must never be the sole production runtime.
- Smaller ecosystem / op coverage less battle-tested than tract (Sonos production footprint).
- Recommendation: re-evaluate rten **only** if product needs a pure-Rust path *and* tract still cannot load silero+powerset after an upstream fix or model export tweak. Do not land rten in-tree until a second spike mirrors this parity harness.

## MSRV

| Component | Declared rust-version |
|-----------|------------------------|
| polyvoice | **1.88.0** |
| tract-onnx 0.23.4 | **1.91** |

Enabling `backend-tract` needs rustc ≥ 1.91. This spike does **not** change `Cargo.toml` `rust-version`. CI default features remain free of tract. Productizing tract implies either a coordinated MSRV bump or a documented dual-toolchain story.

## How to re-run

```bash
# Default suite (no tract in the graph)
cargo test --lib --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
cargo tree -e normal --features "onnx,download,segmentation,embedder,clusterer,resegmentation" | rg tract   # empty

# Spike suite
cargo test --lib --features "onnx,download,segmentation,embedder,clusterer,resegmentation,backend-tract" onnx::parity

# Force tract for stage construction (when feature enabled)
POLYVOICE_INFERENCE_BACKEND=tract cargo test --lib --features "...,backend-tract" …
```

Models are gitignored (`*.onnx`). Place or symlink blobs under `models/` for parity; without them tests skip.

## Bottom line

| Question | Answer |
|----------|--------|
| Ship as default? | **No** |
| Keep as opt-in hedge? | **Yes** — valuable for embedder-only / pure-Rust experiments |
| Full pipeline on tract today? | **No** (silero + powerset blocked) |
| Next unblockers | Upstream tract fixes or re-export models without fragile If/InstanceNorm patterns; optional rten spike later |
