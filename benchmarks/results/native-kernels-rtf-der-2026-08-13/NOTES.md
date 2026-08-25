# Native kernel RTF + DER (2026-08-13)

## Setup

| Item | Value |
|------|--------|
| Host | Apple M1 Pro, Darwin, arm64 (10 threads) |
| Binary | `polyvoice-bench` release, features `cli-native` |
| Pipeline | v2 + VBx, profile balanced, collar 0.0 |
| Models | FP32 `powerset_fp32` + `wespeaker_resnet34` (kernels read initializers only) |
| Ort product (reference) | `powerset_int8` + `resnet34_int8` |

Kernels: LSTM precomputes `X @ Wᵀ` then per-step `H @ Rᵀ`; SincNet first
conv (k=251) and 5-tap convs go through im2col + GEMM; ResNet34 is im2col
+ GEMM. On Apple, GEMM is Accelerate `cblas_sgemm` (AMX). Elsewhere the
in-crate 4×8 NEON / blocked kernel. `VECLIB_MAXIMUM_THREADS=1` so AMX
does not fight the window/embed pools.

## Results

### e2e-smoke (`fuzfh`, 26.05 s)

| Run | RTFx | DER₀ % | wall (s) | seg (s) | emb (s) |
|-----|------|--------|----------|---------|---------|
| native before kernel rewrite | **0.23** | **11.65** | 112.8 | 12.00 | 100.8 |
| native after GEMM rewrite | **3.54** | **11.65** | **7.36** | **0.56** | **6.80** |
| native + parallel embed | **5.84** | **11.65** | **4.46** | **0.56** | **3.91** |
| native + NEON 4×8 GEMM | **6.92** | **11.65** | **3.77** | **0.77** | **2.99** |
| **native + Accelerate AMX** | **50.9** | **11.65** | **0.51** | **0.11** | **0.40** |

DER bits (miss / fa / conf) are unchanged.

### 3-file smoke (same files as tract notes)

| Run | Backend | Embedder | RTFx | DER₀ micro % | DER₀ macro % |
|-----|---------|----------|------|--------------|--------------|
| `ort-smoke` (2026-08-12) | ort | INT8 | **107.7** | **7.41** | 7.72 |
| **`ort-vox3-live` (same host/files)** | ort | INT8 | **80.2** | **7.41** | 7.72 |
| `tract-fp32fix-smoke` (2026-08-12) | tract | FP32 | **12.0** | **7.22** | 7.51 |
| `native-vox3` (GEMM rewrite) | kernels | FP32 | **4.26** | **7.22** | **7.51** |
| `native-vox3-batch` | kernels | FP32 | **6.98** | **7.22** | **7.51** |
| `native-vox3-neon` | kernels | FP32 | **8.00** | **7.22** | **7.51** |
| **`native-vox3-accelerate`** | kernels | FP32 | **61–65** | **7.22** | **7.51** |
| **`native-vox3-bnns`** (same-host rerun) | kernels | FP32 | **111–119** | **7.22** | **7.51** |
| **`ort-vox3-live-rerun`** (same-host) | ort | INT8 | **108–110** | **7.41** | 7.72 |

- Native DER₀ matches tract FP32 (same weights). vs live ort INT8: **−0.19 pp**.
- vs **live** ort back-to-back: native **~113×** vs ort **~108×**. vs tract
  **12×**: native is **~9× faster**.
- Stage split (BNNS, back-to-back): seg **0.43 s** (faster than live ort
  **0.56 s**), emb **0.51 s** (ort **0.41 s**). Segmentation still carries
  the net win; ResNet is close after BNNS + fused ReLU + uninit outputs.
- Layer-1 GEMM: saxpy **4.4** → NEON **28** → Accelerate **318** GFLOP/s.
- Tried / rejected here: N-only tiles, N-outer mem accumulators, 4×8 on LSTM
  steps, capping embed threads to 1–2 (AMX contention hypothesis — wall got
  worse). SincNet k=251 as scalar dots was the powerset hole; im2col+AMX fixed it.

JSON: `native-e2e.json` / `native-vox3.json` (early GEMM);
`native-e2e-neon.json` / `native-vox3-neon.json` (4×8);
`native-e2e-accelerate.json` / `native-vox3-accelerate.json` (AMX);
`native-vox3-bnns.json` / `ort-vox3-live-rerun.json` (BNNS vs live ort);
`ort-vox3-live.json` (earlier same-host ort INT8).

## Product default

Unchanged: `cli` (ort + INT8). Native loads `powerset_int8` + `resnet34_int8`
(~8.1 MB). The powerset **head** is UINT8 `DynamicQuantizeLinear` +
`MatMulInteger` (ORT-matching); LSTM stays dequantized FP32 (hidden-state
cosine 0.9997 vs ORT). Vox-3: DER₀ **7.11%**, **~117–121×** RTFx. Product
default is not flipped.

## Memory (process peak RSS, same host / files / v2+VBx / collar 0)

Back-to-back, CPU EP, `/usr/bin/time -l`:

| Run | Backend | DER₀ micro / macro | RTFx | peak RSS |
|-----|---------|--------------------|------|----------|
| `ort-vox3-int8-cpu-rss` | ort INT8 CPU | 7.41 / 7.72 | 126 | **584.7 MiB** |
| `native-vox3-int8-rss` | kernels INT8 | **7.11 / 7.39** | 121 | **551.9 MiB** |
| `ort-vox3-int8-cpu-rss-rerun` | ort INT8 CPU | 7.41 / 7.72 | 137 | **580.1 MiB** |
| `native-vox3-int8-rss-rerun` | kernels INT8 | **7.11 / 7.39** | 119 | **540.1 MiB** |

Native stays under live ort RSS without giving back DER or the 117× RTF
floor. Peak cut is in-place QDQ on the residual-free conv and in-place
SincNet InstanceNorm (no extra activation clone). Scoreboard floors:
`tests/native_scoreboard.json` (DER 7.11 / 7.39, RTFx ≥ 117, pair ≤
8 414 314 bytes, RSS ≤ 556 MiB).

## Embedder vs ort INT8 CPU

Vox-3 files are a handful of long speaker runs (euqef: 3 turns / 41 s),
not a crowd of short windows. BNNS create is cheap (~30 ms / corpus,
~0 cache hits because every T is unique). Skipping QDQ saves ~4 ms.
The remaining gap is the conv itself: native FP32 BNNS apply vs ort's
fused integer conv. Isolated LTO: native emb **0.40 s** / RTFx **~120–128**
vs ort emb **0.32 s** / **~130–139**. Tried and rejected for speed:
time-tiling (DER held, RTF collapsed), BNNS `n_threads>1`, a naive
im2col+INT8 GEMM (im2col dominated).

Implicit INT8 GEMM is in `polyvoice-kernels/src/conv_i8.rs` (`sdot` 4×4
tiles, pad encoded as activation zp). It matches QDQ float
(`implicit_i8_3x3_tracks_fakequant_float`) and holds Vox-3 DER 7.11, but
row-pack is still ~2 s emb vs BNNS ~0.40 s. Default stays BNNS; opt in
with `POLYVOICE_I8_CONV=1`. Beating ort still needs a fatter implicit
tile or INT8 Winograd, not more threads.
