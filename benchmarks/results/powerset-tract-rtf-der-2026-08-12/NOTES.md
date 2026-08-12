# Tract pipeline RTF + DER smoke (2026-08-12)

## Setup

| Item | Value |
|------|--------|
| Host | Apple M1 Pro, Darwin 25.1, arm64 |
| Binary | `polyvoice-bench` release, features `cli,backend-tract` |
| Pipeline | v2 + VBx, profile balanced, EP=cpu, collar 0.0 |
| Smoke set | 3 shortest VoxConverse-test files (`fuzfh`, `msbyq`, `euqef`) ≈ 105.5 s audio |
| Tract powerset | remapped `powerset_fp32_tract.onnx` (rewrite beside cache) |
| Ort product | `powerset_int8` + `resnet34_int8` |

Dataset copy: `smoke-vox3/` (wav + rttm only; not a full corpus).

## Results

| Run | Backend | Powerset graph | Embedder | RTFx | DER₀ micro % | seg_s | emb_s |
|-----|---------|----------------|----------|------|--------------|-------|-------|
| `ort-smoke` | ort | product INT8 | INT8 | **107.7** | **7.41** | 0.55 | 0.43 |
| `ort-rewrite-smoke` | ort | FP32 tract rewrite (INT8 filename slot) | INT8 | 111.3 | **7.41** | 0.54 | 0.41 |
| `ort-fp32-smoke` | ort | shipping FP32 (INT8 slot) | INT8 | 111.0 | **7.41** | 0.54 | 0.41 |
| `tract-smoke` | tract | rewrite | INT8 (profile) | **11.4** | **42.85** | 4.37 | 4.88 |
| `tract-fp32emb-smoke` | tract | rewrite | “FP32” slot (SHA re-download) | 11.4 | **42.85** | 5.00 | 4.28 |
| **`tract-fp32fix-smoke`** | tract | rewrite | **FP32 `wespeaker_resnet34` (builder fix)** | **12.0** | **7.22** | 4.75 | 4.01 |

Wall (`time -p`): ort ≈ 2.7 s real; tract ≈ 10 s real on the same 105 s audio.

### Per-file (after FP32 embedder fix)

| File | ref spk | ort DER / hyp | tract+FP32 DER / hyp |
|------|---------|---------------|----------------------|
| euqef | 1 | 5.7% / 1 | 5.5% / 1 |
| fuzfh | 3 | 12.0% / 3 | 11.7% / 3 |
| msbyq | 4 | 5.4% / 3 | 5.3% / 3 |

## Root cause (isolated)

| Check | Result |
|-------|--------|
| Powerset rewrite 10 s ort↔tract argmax | **100%** agree (sine + real first window) |
| `PowersetSegmenter` on fuzfh ort vs tract | **identical** segments (n=3, same times/locals) |
| ResNet **FP32** real-segment cosine ort↔tract | **1.000** |
| ResNet **INT8** real-segment cosine ort↔tract | **~0.02–0.06** (garbage) |
| INT8 tract pairwise 0↔1 / 0↔2 | **~0.94 / 0.94** (speakers collapse) |
| INT8 ort pairwise | **0.32 / −0.04** (separable) |

Earlier “FP32 embedder” smoke still hit 42% DER because the registry **re-verified SHA** and restored `resnet34_int8.onnx` after a file swap. The builder now calls `registry.ensure("wespeaker_resnet34")` when `POLYVOICE_INFERENCE_BACKEND=tract`.

## Interpretation

1. **Speed:** tract ~**9×** slower RTFx than ort (~12 vs ~108); still real-time on M1 Pro.
2. **Accuracy (fixed path):** rewrite powerset + **FP32** ResNet under tract ≈ ort product DER on this smoke (7.22% vs 7.41%).
3. **INT8 ResNet under tract is unsafe** — do not use for product pure-Rust path.
4. **Product status:** still **opt-in** (not default). Needs larger DER + RTF before any release claim.

## Reproduce

```bash
# rewrite next to registry cache (macOS example)
cp models/powerset_fp32_tract.onnx \
  ~/Library/Caches/polyvoice/models/powerset_fp32_tract.onnx

cargo build --release --features "cli,backend-tract" --bin polyvoice-bench

export POLYVOICE_VBX_PLDA_DIR=$PWD/fixtures/vbx-plda
DATA=benchmarks/results/powerset-tract-rtf-der-2026-08-12/smoke-vox3
COMMON=(--profile balanced --pipeline v2 --clusterer vbx --collar 0.0 --execution-provider cpu)

# ort
./target/release/polyvoice-bench "$DATA" "${COMMON[@]}" \
  --output benchmarks/results/powerset-tract-rtf-der-2026-08-12/ort-smoke.json

# tract
POLYVOICE_INFERENCE_BACKEND=tract ./target/release/polyvoice-bench "$DATA" "${COMMON[@]}" \
  --output benchmarks/results/powerset-tract-rtf-der-2026-08-12/tract-smoke.json
```

## Follow-ups

1. **10 s ort/tract max-abs + argmax agreement** on powerset rewrite (extend 1 s parity).
2. If 10 s logits disagree: tighten rewrite / tract optim, or pad-and-chunk differently.
3. If logits agree: isolate embedding cosine under tract vs ort on the same segments.
4. Only then re-run larger DER; not a release gate until micro DER is within noise of ort on a fixed smoke set.
