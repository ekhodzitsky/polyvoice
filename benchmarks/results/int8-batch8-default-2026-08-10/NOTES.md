# INT8 micro-batch N=8 product default (2026-08-10)

## Decision

Ship **powerset micro-batch N=8** as the default on non-CoreML EPs
(CPU / XNNPACK / …). **CoreML clamps to N=1** and a single-session pool:
long VoxConverse runs fail mid-corpus with N=8 on the embedder EP
("dynamically resizing for sequence length").

Rationale: CPU N=8 improves Vox DER and RTFx; AMI within ~0.15 pp of N=1.
Static re-quant was rejected (+0.54 pp AMI). CoreML reliability > batch on
that EP. Published Mac CoreML headline DER (15.02 / 24.50) is unchanged
because the clamp restores sequential powerset.

## Numbers (CPU EP, v2+VBx, INT8, M1 Pro)

| Split | DER₀ micro | RTFx | vs prior CoreML N=1 headline |
|-------|----------:|-----:|------------------------------|
| Vox-232 | **14.64%** | **120.8×** | was 15.02% CoreML |
| AMI-16 | **24.63%** | **136.9×** | was 24.50% CoreML |

Artifacts: `voxconverse-test-232-cpu-batch8.json`, `ami-test-16-cpu-batch8.json`.

CoreML AMI N=8 (pre-clamp experiment in this folder: `ami-test-16-collar0.json`
/ `…collar025.json`) reached **23.68% / 16.00%** collar 0 / 0.25 but is **not**
the Mac product path once the clamp is in place.

## Knobs

- Default `PowersetConfig.batch_size = 8`
- `POLYVOICE_POWERSET_BATCH_SIZE=1` sequential ablation
- CoreML: forced N=1 + single session pool (segmenter + embedder)
