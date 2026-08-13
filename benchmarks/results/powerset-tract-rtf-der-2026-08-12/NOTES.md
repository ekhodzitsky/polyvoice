# Tract pipeline RTF + DER (2026-08-12)

## Setup

| Item | Value |
|------|--------|
| Host | Apple M1 Pro, Darwin 25.1, arm64 |
| Binary | `polyvoice-bench` release, features `cli,backend-tract` |
| Pipeline | v2 + VBx, profile balanced, EP=cpu, collar 0.0 |
| Tract powerset | remapped `powerset_fp32_tract.onnx` (beside model cache) |
| Tract embedder | builder forces FP32 `wespeaker_resnet34` |
| Ort product | `powerset_int8` + `resnet34_int8` |

Local dataset copies (wav gitignored): `smoke-vox3/`, `smoke-vox10/` (shortest 10 VoxConverse-test files, ≈ 560 s audio).

## Results

### 3-file smoke (shortest Vox)

| Run | Backend | Embedder | RTFx | DER₀ micro % | wall (s) |
|-----|---------|----------|------|--------------|----------|
| `ort-smoke` | ort | INT8 | **107.7** | **7.41** | ~2.7 |
| `tract-smoke` (INT8 emb) | tract | INT8 | 11.4 | **42.85** | ~10 |
| **`tract-fp32fix-smoke`** | tract | **FP32** | **12.0** | **7.22** | ~10 |

### 10-file subset (shortest 10 Vox, ≈ 560 s)

| Run | Backend | Embedder | RTFx | DER₀ micro % | wall (s) | JSON |
|-----|---------|----------|------|--------------|----------|------|
| `ort-vox10` | ort | INT8 | **98.74** | **9.18** | **7.4** | `ort-vox10.json` |
| **`tract-vox10`** | tract | **FP32** | **11.17** | **8.86** | **51.1** | `tract-vox10.json` |

- **ΔDER₀ = −0.33 pp** (tract slightly better on this short-biased subset; not a product win claim).
- **RTFx ratio tract/ort ≈ 0.113** (~**8.8×** slower).
- Stage times: ort seg 2.86 s + emb 2.81 s; tract seg 22.97 s + emb 27.20 s.

Files (duration order): `fuzfh`, `msbyq`, `euqef`, `dohag`, `neiye`, `xkmqx`, `wdvva`, `gyomp`, `lubpm`, `eucfa`.

## Root cause (INT8 collapse — fixed for product path)

| Check | Result |
|-------|--------|
| Powerset rewrite 10 s ort↔tract argmax | **100%** |
| `PowersetSegmenter` fuzfh ort vs tract | **identical** segments |
| ResNet **FP32** cosine ort↔tract | **1.000** |
| ResNet **INT8** cosine ort↔tract | **~0.02–0.06** (speakers collapse) |

Builder: `registry.ensure("wespeaker_resnet34")` when tract is selected.

## Rewrite distribution (ops)

The rewrite graph is **not** a signed registry profile model yet. To install locally:

```bash
# Preferred helper (export + copy into user model cache)
bash scripts/install-tract-models.sh

# Or manual:
python3 scripts/export-powerset-tract.py --verify
cp models/powerset_fp32_tract.onnx \
  ~/Library/Caches/polyvoice/models/powerset_fp32_tract.onnx   # macOS
# Linux: ~/.cache/polyvoice/models/
```

Pipeline remaps any shipping powerset path to sibling `powerset_fp32_tract.onnx`
when the active backend is tract. FP32 ResNet is ensured via the registry.

**Product default remains ort + INT8.** Tract is opt-in only.

## Reproduce Vox-10

```bash
bash scripts/install-tract-models.sh --skip-export   # if rewrite already built
cargo build --release --features "cli,backend-tract" --bin polyvoice-bench
export POLYVOICE_VBX_PLDA_DIR=$PWD/fixtures/vbx-plda
# build smoke-vox10 from the 10 shortest VoxConverse-test files (see NOTES)
COMMON=(--profile balanced --pipeline v2 --clusterer vbx --collar 0.0 --execution-provider cpu)
./target/release/polyvoice-bench smoke-vox10 "${COMMON[@]}" --output ort-vox10.json
POLYVOICE_INFERENCE_BACKEND=tract \
  ./target/release/polyvoice-bench smoke-vox10 "${COMMON[@]}" --output tract-vox10.json
```

## Parallelism (pure-Rust product path)

Tract **cannot** run powerset micro-batch N>1: LSTM `Scan` fails at eval
(`powerset_fp32_tract_batch8_lstm_scan_documented`). Speedup is **session pool**
(N=1 per `run`, several windows in parallel).

| Tract pool | Vox-3 RTFx | DER₀ | wall (s) |
|------------|------------|------|----------|
| 1 | 8.1 | 7.22 | 14.9 |
| **4** (default cap) | **19.3** | 7.22 | 7.6 |

`POLYVOICE_SESSION_POOL_SIZE` overrides. Embedder pool already used the same knob.

## Interpretation

1. **Accuracy:** with rewrite + FP32 ResNet, tract matches ort within ~0.3 pp on
   3- and 10-file short-biased Vox subsets.
2. **Speed:** ~**9×** slower RTFx than ort; still multi-realtime (~11×) on M1 Pro.
3. **Not a full-split gate:** 10 shortest files ≠ Vox-232 product protocol.
4. **Still open for pure-Rust product:** signed rewrite in registry, AMI/full Vox
   tract gate, MSRV policy, INT8 tract embedder (if ever).

## Follow-ups

1. Full-split or AMI-16 tract DER/RTF (optional; expensive).
2. Registry-signed `powerset_fp32_tract` artifact for `ModelRegistry::ensure`.
3. INT8 ResNet under tract remains **out of scope** until cosine parity exists.
