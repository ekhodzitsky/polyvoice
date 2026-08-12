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
| `backend-tract` + rewrite + **FP32** ResNet | **Yes (smoke DER)** | remaps powerset; builder forces `wespeaker_resnet34` (INT8 ResNet unsafe under tract); ~9× slower RTFx; 3-file Vox DER ≈ ort |
| `backend-tract` + INT8 ResNet | **No (accuracy)** | ort↔tract cosine ~0; speakers collapse — **not used** when tract is selected |
| Production CLI / pipeline v2 default | **No** (ort + INT8) | Opt-in: `POLYVOICE_INFERENCE_BACKEND=tract` + `backend-tract` + rewrite ONNX |

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

1. ~~**Powerset ONNX re-export**~~ — **done** (`scripts/export-powerset-tract.py`):
   inline identical `If`, expand InstanceNorm; tract loads with concrete
   `[1,1,160000]`. See
   [`benchmarks/results/powerset-tract-export-2026-08-12/NOTES.md`](../../benchmarks/results/powerset-tract-export-2026-08-12/NOTES.md).
2. ~~**Wire pipeline**~~ — **done** (opt-in remap + N=1).
3. ~~**Release RTF + DER smoke**~~ — **measured + root-caused** (2026-08-12):
   rewrite OK; INT8 ResNet under tract collapses; builder uses FP32 ResNet.
4. ~~**Larger DER + RTF**~~ — **Vox-10 shortest** (≈560 s, M1 Pro): tract
   DER₀ **8.86%** vs ort **9.18%** (Δ **−0.33 pp**); RTFx **11.2** vs **98.7**
   (~**8.8×** slower). Notes:
   [`powerset-tract-rtf-der-2026-08-12/NOTES.md`](../../benchmarks/results/powerset-tract-rtf-der-2026-08-12/NOTES.md).
   Still **not** full-split / not product default.
5. ~~**Ship rewrite via registry**~~ — signed `powerset_fp32_tract` on release
   `models-tract-v1`; builder `ensure`s it under tract. Local
   `install-tract-models.sh` remains a dev shortcut.
6. **Silero** only if legacy remains product-relevant; else drop from pure-Rust target.
7. **Earshot** re-tune if legacy pure path is needed (current Δ fails 0.3 pp gate).
8. **Optional rten spike** only if fixed-T tract remains too slow/limited after accuracy work.
9. Do **not** bump crate MSRV solely for tract until productizing (tract MSRV 1.91 vs crate 1.88).

## Install tract assets

**Preferred (registry):** with `download` + network, the pipeline builder calls
`ModelRegistry::ensure("powerset_fp32_tract")` and `ensure("wespeaker_resnet34")`
when `POLYVOICE_INFERENCE_BACKEND=tract`. Artifact release:
`models-tract-v1` on GitHub.

**Local dev shortcut** (offline / regenerate rewrite):

```bash
bash scripts/install-tract-models.sh
# or: bash scripts/install-tract-models.sh --skip-export
```

## Commands

```bash
# Invariants (CI)
bash scripts/check-zero-deps.sh

# Tract load / parity (needs models under models/ or models/int8/)
cargo test --lib --features "onnx,backend-tract" onnx::parity -- --nocapture

# Build tract-friendly powerset (needs models/powerset_fp32.onnx)
python3 scripts/export-powerset-tract.py --verify
cargo test --lib --features "onnx,segmentation,backend-tract" powerset_fp32_tract_friendly -- --nocapture
cargo test --lib --features "onnx,segmentation,backend-tract" tract_backend_segments -- --nocapture

# Optional pure-Rust pipeline (not product default):
# bash scripts/install-tract-models.sh
# POLYVOICE_INFERENCE_BACKEND=tract cargo run --release --features "cli,backend-tract" -- …


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
