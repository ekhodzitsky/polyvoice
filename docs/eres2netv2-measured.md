# ERes2NetV2 vs default embedder — measured notes

**Date:** 2026-07-24  
**Artifact:** `benchmarks/results/embedder-short-eres2net.json`

## Short-segment EER (VoxConverse RTTM pairs, not VoxCeleb1)

| Duration | ResNet34 EER | ERes2NetV2 (zh-cn) EER |
|----------|--------------|-------------------------|
| 0.5 s | 18.86% | 27.46% |
| 1.0 s | 7.21% | 20.09% |
| 2.0 s | 4.75% | 13.03% |
| 3.0 s | 3.84% | 10.74% |

## Legacy pipeline DER (Vox-test 10 files, Silero fixed)

| Embedder | collar 0 | collar 0.25 |
|----------|----------|-------------|
| ResNet34 default | 23.89% | 15.82% |
| ERes2NetV2 zh-cn | 53.84% | 49.18% |

## Verdict

Optional **zh-cn** ERes2NetV2 is **not** competitive on English VoxConverse under
the shared fbank front-end. Keep the adapter for CJK experiments; do not default.
