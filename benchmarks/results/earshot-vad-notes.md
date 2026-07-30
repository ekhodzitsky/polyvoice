# earshot VAD backend — notes & parity gate

> **Historical note (2026-07-30):** these measurement notes describe the
> original adapter contract, where `EarshotVad` buffered arbitrary chunk
> sizes into 256-sample windows. The adapter now uses a reject contract:
> a partial (non-multiple-of-256) chunk fails with
> `VadError::InvalidChunkSize` and no hidden buffering occurs. The
> measurements and verdict below were taken under the old contract and are
> kept as recorded; re-run the parity gate before citing them against the
> current adapter.

**Date:** 2026-07-24  
**Branch:** `feat/earshot-vad`  
**earshot:** `1.2.x` (pykeio, MIT OR Apache-2.0)  
**Default VAD:** unchanged — **Silero** (ONNX, feature `onnx`)  
**Opt-in feature:** `vad-earshot`  
**Adapter type id:** `earshot` (`AdapterStage::Vad`)

## Recommendation

**Keep earshot optional only.** Do **not** switch the default VAD.

- Adapter implements `VoiceActivityDetector` with 256-sample frame buffering @ 16 kHz.
- Continuous scores in `[0, 1]` map directly to our speech-probability contract (threshold at the caller / `VadConfig`).
- Vendor RTF / accuracy claims are **unverified** on this harness until the DER parity gate below is run and recorded.
- Silero remains the production path and the parity reference.

## What landed

| Item | Status |
|------|--------|
| Feature `vad-earshot` + optional dep `earshot` | done |
| `EarshotVad` behind feature | done |
| Frame buffer for arbitrary chunk sizes → 256-sample windows | done |
| Unit tests (silence / speech-ish relative scores / reset / odd chunks) | done |
| `AdapterRegistry` name marker `"earshot"` when feature on; aliases stay `silero` | done |
| Default build free of `earshot` (`cargo tree`) | required verify |
| Full DER parity numbers on vox-30 | **not run here** (optional / heavy; methodology frozen below) |

## Frame / API contract

| Property | earshot | Silero (reference) |
|----------|---------|--------------------|
| Sample rate | 16 kHz mono | 16 kHz mono |
| Native frame | **256** samples (16 ms) | typically **512** samples |
| Score | continuous `[0, 1]` via `predict_f32` | continuous `[0, 1]` ONNX out |
| State | ~8 KiB on detector; reset between streams | LSTM state + context; reset between streams |
| Runtime | pure Rust, weights in-crate (~40 KiB) | ONNX Runtime + ~2.3 MB model file |
| Chunk sizes | adapter **buffers** odd lengths | `InvalidChunkSize` if not multiple of chunk |

When using `segment_speech`, set `VadConfig.frame_size = 256` so sample indices align with model frames.

## Vendor claims (unverified)

From the earshot README / crate page — **do not cite as product facts**:

| Claim | Status |
|-------|--------|
| RTF ≈ 0.0003 (~3600× realtime) | **unverified** on polyvoice hardware |
| “40× faster than Silero VAD v6 and TEN-VAD” | **unverified** |
| “and more accurate” (PR curve figure) | **unverified** on our labels / corpora |

Any public claim must come from the parity gate measurements below (or a later re-run with the same protocol).

## Watch list

### TEN-VAD — forbidden

Do **not** integrate TEN-VAD. Research notes flag an Apache-looking license with a **non-compete** rider (Agora). That is incompatible with this project’s license hygiene. Document only as a negative: never a dependency, never a benchmark dependency that would require linking it.

### FireRedVAD — watch until ONNX

FireRedVAD (Xiaohongshu): Apache-2.0 code/weights reported, vendor F1 97.57 vs Silero 95.95 on FLEURS-VAD-102 (vendor-reported, not re-run here). Streaming variant exists; footprint ~2.2 MB. **Candidate for the model registry only when an ONNX export is available** and can go through the same signed-download path as other ONNX stages. PyTorch/NCNN-only distributions stay out of tree.

## Parity gate methodology (frozen)

Run **before** any proposal to change the default VAD. Full DER is optional in lightweight CI; when run, use this protocol.

### Tolerance (frozen proposal)

- **DER absolute delta** earshot − Silero ≤ **0.3** percentage points on the same collar, same audio, same non-VAD stages  
  (i.e. if Silero DER = 12.9%, earshot must be ≤ 13.2%).  
- If delta > 0.3 → **regression** verdict; keep Silero default.  
- If delta ≤ 0.3 on both collars → **parity** verdict; default change still requires a separate decision (not automatic).

### Datasets / collars

1. **vox-30** (VoxConverse test subset used by existing DER harness) — **required** for a parity verdict.  
2. **ami-test-single** — optional if compute budget allows (far-field stress).  
3. Collars: **0.0 s** and **0.25 s** (both).

### Pipeline controls

- Identical non-VAD stack for both arms (same embedder, clusterer, config).  
- earshot arm: `VadConfig { frame_size: 256, threshold: 0.5, min_silence_ms: … }` (match Silero threshold policy unless a sensitivity sweep is documented).  
- Silero arm: existing default chunk (512) / threshold.  
- Report wall-clock VAD-only RTF on the same machine, release profile, after warmup.

### Metrics to record

| Metric | Notes |
|--------|-------|
| DER @ collar 0 / 0.25 | primary gate |
| Frame-level P / R / F1 | speech labels derived from RTTM (any speech vs none); document frame hop |
| VAD RTF | processed_audio_seconds / wall_time; compare to vendor 0.0003 |
| Mean speech probability on silence / speech regions | sanity |

### Suggested commands

```bash
# Unit / adapter (no models)
cargo test --lib --features vad-earshot earshot_vad

# Default graph must not include earshot
cargo tree -e normal | rg earshot   # empty

# With download + registry marker
cargo test --lib --features "vad-earshot,download" earshot

# DER parity (heavy; needs models + dataset env)
POLYVOICE_DER_EVAL=1 cargo run --features "cli,vad-earshot" --bin polyvoice-bench -- \
  --dataset voxconverse-test --collar 0
# repeat --collar 0.25; compare Silero vs earshot arms when CLI --vad exists
```

If the CLI has no `--vad earshot` switch yet, drive the legacy `Pipeline` from a small harness that constructs `EarshotVad` vs `SileroVad` and reuses the same DER evaluation path as `polyvoice-bench`.

### Measured verdict (2026-07-24)

```
Dataset: VoxConverse-test first 10 files (sorted), legacy pipeline, ResNet34 embedder
Hardware: Apple M1 Pro, 10 cores, release build
Collar 0:   Silero 23.89%  earshot 26.54%  Δ +2.65 pp  → FAIL (|Δ|≤0.3)
Collar 0.25: Silero 15.82%  earshot 18.68%  Δ +2.86 pp  → FAIL
RTF: Silero 0.102  earshot 0.099  (nearly tied; not 40×)
Verdict: regression — keep optional only; do not switch default
Vendor RTF claim 0.0003: not measured as pure VAD microbench (pipeline RTF only)
Artifact: benchmarks/results/vad-parity-earshot-silero.json
```

## Dependency hygiene

```bash
cargo tree -e normal | rg earshot                 # must be empty
cargo tree -e normal --features vad-earshot | rg earshot   # earshot present
```

Licenses: earshot is MIT OR Apache-2.0 — already on the `deny.toml` allow-list.

## MSRV / no_std note

- earshot declares rust-version **1.87**; polyvoice is **1.88** — no MSRV bump.  
- earshot supports `#![no_std]` via `default-features = false, features = ["libm"]`. This adapter uses the default `std` feature set. A no_std consumer path is not wired in polyvoice; document only if productizing embedded builds later.

## Bottom line

Optional pure-Rust VAD backend is available behind `vad-earshot`. **Silero stays default.** Vendor speed/accuracy numbers are labeled unverified. TEN-VAD is forbidden; FireRedVAD waits on ONNX. Full DER numbers fill in the verdict template above when the heavy gate is run.
