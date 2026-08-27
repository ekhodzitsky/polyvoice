# Zero-dependency aspiration (pure Rust / no native dylib)

**Status:** active engineering constraint (Claude.md / AGENTS.md).  
**Updated:** 2026-08-26  

## Goal

Ship speaker diarization without a C/C++ toolchain or prebuilt native
runtimes (`libonnxruntime`, CoreML bindings, etc.) for the **core library
path**, and progressively shrink the production path toward the same.

**How:** only the ops polyvoice actually runs, in small workspace crates.
Do **not** clone ONNX Runtime, tract, or a general ONNX executor.

## Incremental path (narrow crates, not clones)

| Step | What | Not |
|------|------|-----|
| **0** | Honest feature graph: `infer` / `onnx` / `backend-tract`. `pipeline-tract` has **no** `ort`. | (done) |
| **1** | Optional `cli-tract` (same CLI bins, tract engine, no dylib). `--legacy` rejected. | Rewriting the CLI |
| **2** | `polyvoice-kernels`: **WeSpeaker ResNet34 only** (fused-BN Conv2d, ReLU, residual, stats-pool, GEMM). Initializers from shipping ONNX. Feature `embedder-native`. | Candle/Burn/tract clone |
| **3** | Same crate: **powerset LSTM** (SincNet + 4× biLSTM). Feature `segmenter-native`. N>1 works. | Generic Scan / all ONNX ops |
| **4 (now)** | `cli` / `ffi` / `mcp` = `pipeline-native` (no ort, no tract). Darwin Vox-3 holds the scoreboard floors. Linux DER holds the AMI ceiling; RTF is still below the old ort band. `cli-ort` keeps ONNX Runtime opt-in. | Pulling `ort` back into `cli` |
| **skip** | Silero (v2 powerset already does VAD). Full protobuf ONNX parser. Sortformer stays opt-in/`onnx` until someone needs it. | — |

Cross-platform is the default of this path: no `libonnxruntime`, no glibc pin, no CoreML/XNNPACK. Linux / macOS / Windows; clustering is already wasm32-clean.

Keep `ort` as an optional `onnx` feature so INT8 + EP + Sortformer + Python keep working. Do not add another general ML framework. Do not pull `ort` back into `cli`.

## Current matrix

| Surface | Pure Rust? | Quality / notes |
|---------|------------|-----------------|
| `default = []` + BYO `Embedder` + `EnergyVad` + AHC/VBx | **Yes** | Library mode; no models bundled |
| `vad-earshot` (`EarshotVad`) | **Yes** (weights in crate) | Legacy VAD only; **+2.65 pp DER** vs Silero on measured subset — **opt-in only** |
| `backend-tract` / `pipeline-tract` (no `onnx`) | **Yes** (tract-onnx; **no ort**) | Feature graph no longer pulls `libonnxruntime` |
| `backend-tract` embedders (ResNet34 FP32/INT8, CAM++) | **Yes** (tract is pure Rust) | Numerical parity with ort within tol |
| `backend-tract` + Silero ONNX | **No (load fail)** | Nested `If` / analyse |
| `backend-tract` + powerset **shipping** ONNX | **No (load fail)** | nested `If` + `InstanceNormalization` |
| `backend-tract` + rewrite + **FP32** ResNet | **Yes (smoke DER)** | remaps powerset; builder forces `wespeaker_resnet34` (INT8 ResNet unsafe under tract); ~9× slower RTFx; 3-file Vox DER ≈ ort |
| `backend-tract` + INT8 ResNet | **No (accuracy)** | ort↔tract cosine ~0; speakers collapse — **not used** when tract is selected |
| `cli-tract` | **Yes** (tract-onnx; **no ort**) | Same `polyvoice` / `polyvoice-bench` / `polyvoice-measure` bins; `--legacy` rejected |
| `embedder-native` (`ResNet34Native`) | **Yes** | Hand-written ResNet34; ort cosine 1.0 on 1 s fixture; no dylib |
| `segmenter-native` (`PowersetNative`) | **Yes** | SincNet + 4× biLSTM; N>1; 1 s vs ort cosine 1.0 |
| `pipeline-native` / `cli` / `ffi` | **Kernels** (Darwin: C shims + Accelerate/BNNS; Linux: `rten-gemm`) | **Product default.** No `ort`. Vox-3 DER₀ **7.11%**, **≥117×** on Apple. Linux AMI DER within ort ceiling; RTF ~28× (Vox-3). BYO `default = []` stays pure Rust. |
| `cli-ort` / `pipeline-full` | **No** (ort + INT8) | Opt-in previous product |

CI freezes the pure-Rust **invariants** via:

```bash
bash scripts/check-zero-deps.sh   # includes check-ort-free.sh
```

## What “done” looks like for production

The product CLI (`cli` / `ffi` / `mcp`) already meets this bar via
`polyvoice-kernels` (step 4): powerset + ResNet34 INT8 without `ort`, VAD
folded into powerset, clustering already Rust-only. Residual: Linux native
RTF still trails the old ort band; Python still links `ort`; tract remains
the slower ONNX-shaped opt-in.

A shipping *tract* profile would still need:

1. Powerset segmentation without ort (rewrite graph — done, not product).
2. Embedder without ort (tract can for FP32 ResNet34; INT8 collapses).
3. VAD unused (v2 powerset) or pure-Rust with DER gate ≤ ε vs Silero.
4. Clustering Rust-only (already: AHC / VBx / optional faer spectral).

## Unblockers (ordered)

1. ~~**Powerset ONNX re-export**~~ — **done** (`scripts/export-powerset-tract.py`):
   inline identical `If`, expand InstanceNorm; tract loads with concrete
   `[1,1,160000]`. See
   [`benchmarks/results/powerset-tract-export-2026-08-12/NOTES.md`](../../benchmarks/results/powerset-tract-export-2026-08-12/NOTES.md).
2. ~~**Wire pipeline**~~ — **done** (opt-in remap + N=1).
3. ~~**Release RTF + DER smoke**~~ — **measured + root-caused** (2026-08-12):
   rewrite OK; INT8 ResNet under tract collapses; builder uses FP32 ResNet.
4. ~~**Larger DER + RTF**~~ — Vox-10 short subset + **AMI-test full 16**
   (M1 Pro): AMI DER₀ tract **23.42%** vs ort **24.63%** (Δ **−1.21 pp**);
   RTFx **18.8** vs **153.5** (~**8.2×** slower). Helper:
   `scripts/tract-der-gate.sh`. Notes:
   [`tract-der-ami-2026-08-13`](../../benchmarks/results/tract-der-ami-2026-08-13/NOTES.md),
   [`powerset-tract-rtf-der-2026-08-12`](../../benchmarks/results/powerset-tract-rtf-der-2026-08-12/NOTES.md).
   Still **not** product default / not Linux full-split tract gate.
   Tract **cannot** micro-batch N>1 (LSTM `Scan` fails at eval). Parallelism:
   - session **pool>1** (windows in one file): Vox-3 pool=4 vs 1 → **19.3× vs 8.1×** RTFx
   - `polyvoice-bench --jobs N` (files): cuts corpus **wall** (3 files: 5.5 s @
     jobs=3 vs 6.9 s @ jobs=1/pool=4). `jobs × pool ≤ cores`. Same DER.
5. ~~**Ship rewrite via registry**~~ — signed `powerset_fp32_tract` on release
   `models-tract-v1`; builder `ensure`s it under tract. Local
   `install-tract-models.sh` remains a dev shortcut.
6. **Silero** only if legacy remains product-relevant; else drop from pure-Rust target.
7. **Earshot** re-tune if legacy pure path is needed (current Δ fails 0.3 pp gate).
8. **Optional rten spike** only if fixed-T tract remains too slow/limited after accuracy work.
9. Do **not** bump crate MSRV solely for tract (tract MSRV 1.91; crate is already 1.94 for `rten-simd`).

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

# Optional tract CLI (not product default):
# bash scripts/install-tract-models.sh
# cargo run --release --features cli-tract -- …

# Product CLI (kernels, no ort/tract):
# cargo run --release --features cli -- …


# Native ResNet34 + powerset (no ONNX runtime)
cargo test -p polyvoice-kernels
cargo test --lib --features "onnx,embedder,embedder-native" native_matches_onnx_resnet34
cargo test --lib --features "onnx,segmentation,segmenter-native" native_matches_ort_one_second
# Full v2, no ort/tract:
cargo test --lib --features "pipeline-native,vbx" native_pipeline_runs_short_sine
cargo run --release --features cli-native -- meeting.wav

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
- [`benchmarks/results/native-kernels-rtf-der-2026-08-13/NOTES.md`](../../benchmarks/results/native-kernels-rtf-der-2026-08-13/NOTES.md)
- [`docs/library-mode.md`](../library-mode.md)
