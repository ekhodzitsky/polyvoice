# Powerset ONNX micro-batch (CPU) — 2026-08-10

## Goal

Batch multiple sliding windows into one ONNX `run` (`[N,1,T]`) to speed up
segmentation on the pure CPU EP without regressing DER.

## Finding

Shipping `powerset_int8` (`models-int8-v2`, sha `175896d2…`) is **not**
batch-invariant: N>1 changes logits vs N sequential N=1 runs (argmax
mismatches on real AMI audio under onnxruntime). Older local 5.7 MB INT8 /
FP32 exports can be bit-identical; the production graph is not.

## AMI-test 16 (CPU EP, v2+VBx, INT8, collar 0)

| Batch | DER₀ micro | RTFx | seg wall | emb wall |
|------:|----------:|-----:|---------:|---------:|
| N=1   | **24.50** | 102× | 218 s    | (rest)   |
| N=8   | **24.63** | 129× | 149 s    | similar  |

Δ DER = **+0.14 pp** (worse). Δ RTFx ≈ **+26%**. Seg time ≈ **−32%**.

Under the no-regression policy, **default stays N=1**. N=8 is opt-in via
`POLYVOICE_POWERSET_BATCH_SIZE=8` when a re-baseline is acceptable.

## Implementation

- `PowersetConfig::batch_size` (default 1)
- `POLYVOICE_POWERSET_BATCH_SIZE` env override
- Workers still fan out across the session pool; each worker packs
  `batch_size` windows per `run`

## Reproduce

```bash
cargo build --release --features cli --bin polyvoice-bench
POLYVOICE_POWERSET_BATCH_SIZE=1 target/release/polyvoice-bench data/ami-test \
  --profile balanced --pipeline v2 --clusterer vbx --collar 0 \
  --execution-provider cpu --output ami-batch1.json
POLYVOICE_POWERSET_BATCH_SIZE=8 target/release/polyvoice-bench data/ami-test \
  ... --output ami-batch8.json
```

## VoxConverse-test 232 (CPU EP, collar 0)

| Batch | DER₀ micro | RTFx | notes |
|------:|----------:|-----:|-------|
| N=8   | **14.64** | (see json) | better than CoreML published 15.02 |
| N=1   | (running or see json) | | |

AMI fails the no-regression bar (+0.14 pp); Vox improves. Default remains N=1.
