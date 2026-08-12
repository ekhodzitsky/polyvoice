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
| `tract-smoke` | tract | rewrite | INT8 | **11.4** | **42.85** | 4.37 | 4.88 |
| `tract-fp32emb-smoke` | tract | rewrite | FP32 ResNet | 11.4 | **42.85** | 5.00 | 4.28 |

Wall (`time -p`): ort ≈ 2.7 s real; tract ≈ 10.0 s real on the same 105 s audio.

### Per-file (tract vs ort product)

| File | ref spk | ort DER / hyp | tract DER / hyp |
|------|---------|---------------|-----------------|
| euqef | 1 | 5.7% / 1 | 5.5% / 1 |
| fuzfh | 3 | 12.0% / 3 | 52.7% / **1** |
| msbyq | 4 | 5.4% / 3 | 61.3% / **1** |

Multi-speaker files collapse to a single hypothesis speaker under tract (confusion-dominated DER).

## Interpretation

1. **Speed:** pure-Rust tract path is ~**9.4×** slower RTFx than ort on this host (stage times both ~8–11×: powerset and ResNet). Still faster than real-time (~11× RTFx) on M1 Pro for this tiny set.
2. **Rewrite quality (ort):** the tract-friendly graph is **DER-identical** to product INT8 on this smoke when run under ort. Graph rewrite is not the accuracy bug.
3. **Tract runtime accuracy:** full pipeline under tract is **not** DER-safe (+35 pp on 3 files). Same collapse with FP32 or INT8 embedder → not explained by INT8 embedder alone. Likely cumulative numerical drift on product **10 s** windows (existing tight parity is **1 s** only) and/or interaction with VBx under altered posteriors/embeddings.
4. **Product status:** still **opt-in / not default**. Wiring remains useful for pure-Rust load+run; **do not** claim production DER.

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
