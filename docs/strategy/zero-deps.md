# Zero-dependency aspiration (pure Rust / no native dylib)

**Status:** active engineering constraint (Claude.md / AGENTS.md).  
**Updated:** 2026-08-12  

## Goal

Ship speaker diarization without a C/C++ toolchain or prebuilt native
runtimes (`libonnxruntime`, CoreML bindings, etc.) for the **core library
path**, and progressively shrink the production path toward the same.

## Current matrix

| Surface | Pure Rust? | Quality / notes |
|---------|------------|-----------------|
| `default = []` + BYO `Embedder` + `EnergyVad` + AHC/VBx | **Yes** | Library mode; no models bundled |
| `vad-earshot` (`EarshotVad`) | **Yes** (weights in crate) | Legacy VAD only; **+2.65 pp DER** vs Silero on measured subset — **opt-in only** |
| `backend-tract` embedders (ResNet34 FP32/INT8, CAM++) | **Yes** (tract is pure Rust) | Numerical parity with ort within tol |
| `backend-tract` + Silero ONNX | **No (load fail)** | Nested `If` / analyse |
| `backend-tract` + powerset **shipping** ONNX | **No (load fail)** | nested `If` + `InstanceNormalization` |
| `backend-tract` + **tract-friendly rewrite** (`export-powerset-tract.py`) | **Yes (load+run)** | concrete T (10 s / 1 s); 1 s tight ort/tract parity; **not** pipeline-wired yet |
| Production CLI / pipeline v2 (`onnx` + powerset + ResNet34) | **No** | Downloads ORT dylib via `ort` |

CI freezes the pure-Rust **invariants** via:

```bash
bash scripts/check-zero-deps.sh   # includes check-ort-free.sh
```

## What “done” looks like for production

A shipping profile where:

1. Powerset segmentation runs without ort (tract or successor, or re-exported graph).
2. Embedder runs without ort (tract already can for ResNet34-class).
3. VAD is either unused (v2 powerset) or pure-Rust with DER gate ≤ ε vs Silero.
4. Clustering stays Rust-only (already: AHC / VBx / optional faer spectral).

## Unblockers (ordered)

1. ~~**Powerset ONNX re-export**~~ — **spike done** (`scripts/export-powerset-tract.py`):
   inline identical `If`, expand InstanceNorm; tract loads with concrete
   `[1,1,160000]`. See
   [`benchmarks/results/powerset-tract-export-2026-08-12/NOTES.md`](../../benchmarks/results/powerset-tract-export-2026-08-12/NOTES.md).
2. **Wire pipeline** to optional tract powerset (fixed window pad to 10 s) +
   tract ResNet34; measure release RTF + DER smoke.
3. **Silero** only if legacy remains product-relevant; else drop from pure-Rust target.
4. **Earshot** re-tune if legacy pure path is needed (current Δ fails 0.3 pp gate).
5. **Optional rten spike** only if fixed-T tract powerset is too slow/limited.
6. Do **not** bump crate MSRV solely for tract until productizing (tract MSRV 1.91 vs crate 1.88).

## Commands

```bash
# Invariants (CI)
bash scripts/check-zero-deps.sh

# Tract load / parity (needs models under models/ or models/int8/)
cargo test --lib --features "onnx,backend-tract" onnx::parity -- --nocapture

# Build tract-friendly powerset (needs models/powerset_fp32.onnx)
python3 scripts/export-powerset-tract.py --verify
cargo test --lib --features "onnx,backend-tract" powerset_fp32_tract_friendly -- --nocapture

# Earshot unit tests
cargo test --lib --features vad-earshot earshot_vad

# Earshot vs Silero DER (legacy arm)
cargo run --release --features "cli,vad-earshot" --bin polyvoice-measure -- vad-parity \
  --dataset data/ami-test-single --output /tmp/vad-parity.json
```

## Non-goals (for now)

- Making CoreML / XNNPACK pure-Rust (EP is ort-native).
- Training our own models.
- Silent quality regressions to claim “zero deps”.

## Related artifacts

- [`benchmarks/results/tract-backend-verdict.md`](../../benchmarks/results/tract-backend-verdict.md)
- [`benchmarks/results/earshot-vad-notes.md`](../../benchmarks/results/earshot-vad-notes.md)
- [`docs/library-mode.md`](../library-mode.md)
