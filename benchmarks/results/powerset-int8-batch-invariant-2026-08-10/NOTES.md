# Batch-invariant powerset INT8 re-quant research (2026-08-10)

## Goal

Produce a **batch-invariant** `powerset_int8` so ONNX micro-batch N=8
(`[N,1,T]`) matches N×1 logits bit-for-bit (or DER-identical), enabling the
CPU speed win from batching without DER regression.

## Why production fails batch invariance

Shipping `powerset_int8` (models-int8-v2, ~1.65 MB, sha `175896d2…`) uses
**weights-only dynamic quant** with `DynamicQuantizeLinear` on activations
(+ `ConvInteger` / `MatMulInteger` / `DynamicQuantizeLSTM`). Activation
scales are computed from the **current tensor min/max**, so N=1 vs N=8
change scales → different logits.

| Model | Size | Batch max‖Δ‖ | Argmax mismatches (8×10s AMI) | vs FP32 KL / arg% |
|-------|-----:|-------------:|------------------------------:|------------------:|
| FP32 | 5.99 MB | 0 | 0 | — |
| **prod dynamic INT8** | **1.65 MB** | **1.6** | **286** | **0.0095 / 4.9%** |
| old local static QDQ | 5.74 MB | 0 | 0 | 0.32 / 28% |
| static QDQ percentile (LSTM+IN fp32) | 5.73 MB | 0 | 0 | 0.042 / 13% |
| freeze DQL→QL scales | 1.65 MB | broken | — | ~0.45 / 30% |

Manifest already documents: static QDQ on the full recurrent path “destroys
posteriors”; that is why production chose dynamic quant.

## Best static candidate (percentile QUInt8, LSTM+InstanceNorm FP32)

- Artifact: `models/int8/candidates/qdq_u8_percentile_exLstmIn.onnx` (local only)
- **Batch-invariant**: AMI-16 DER₀ N=1 == N=8 = **25.04%** (identical)
- **Speed**: N=8 RTFx 93× vs N=1 79× (**+18%**); seg 284 s → 215 s
- **Quality vs prod 24.50%**: **+0.54 pp DER** (worse) — **fails no-regression**
- Also **slower** than prod dynamic (prod N=1 ~102× on same host)

## Conclusion

| Requirement | prod dynamic | best static |
|-------------|--------------|-------------|
| DER ≤ prod | ✅ 24.50 | ❌ 25.04 |
| Batch-invariant | ❌ | ✅ |
| Size ~1.6 MB | ✅ | ❌ ~5.7 MB |
| CPU RTFx with N=8 | ~129× but DER drifts | ~93×, DER fixed |

**Do not ship a re-quant as the default model** under the multi-gate policy.
Keep:

- production dynamic INT8 as default accuracy/size tradeoff
- micro-batch path with **default N=1**
- Product later shipped dynamic-INT8 N=8 on CPU (see
  `../int8-batch8-default-2026-08-10/`); this static re-quant remains unshipped

## Next research (not done)

1. **Hybrid export**: keep dynamic weight packing, replace only non-recurrent
   `DynamicQuantizeLinear` with calibrated `QuantizeLinear` (freeze-scales
   prototype broke graph consumers; needs careful type/scale wiring).
2. **ORT weight-only** (`MatMulNBits` / RTN) leaving activations float.
3. **Re-export from PyTorch** with QAT that is batch-aware.
4. Accept N=8 on prod dynamic + re-baseline full Vox+AMI (Vox improved,
   AMI +0.14 pp on prod — policy call).

## Reproduce quant (static candidate)

```bash
uv venv /tmp/polyvoice-quant-venv --python 3.12
uv pip install --python /tmp/polyvoice-quant-venv/bin/python \
  onnxruntime onnx librosa numpy
EXCLUDE=$(python -c "import onnx; m=onnx.load('models/powerset_fp32.onnx'); print(','.join(
  n.name for n in m.graph.node if n.op_type in ('LSTM','InstanceNormalization')))")
python scripts/quantize_models.py \
  --fp32 models/powerset_fp32.onnx \
  --int8 models/int8/candidates/qdq_u8_percentile_exLstmIn.onnx \
  --calib data/voxconverse-dev/audio \
  --input-shape 1,1,160000 --num-samples 100 --seed 42 \
  --exclude-nodes "$EXCLUDE"
# then point quantize_static activation_type at QUInt8 + Percentile (see experiment script notes)
```

## Artifacts in this dir

- `ami-16-batch1.json` / `ami-16-batch8.json` — static percentile candidate, CPU EP
