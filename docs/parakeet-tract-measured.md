# Parakeet TDT on tract — measured notes

**Date:** 2026-09-21  
**Host:** AMD Ryzen AI 9 HX 370, Linux x86_64  
**Artifact:** [`benchmarks/results/parakeet-tract-2026-09-21/`](../benchmarks/results/parakeet-tract-2026-09-21/)

`polyvoice-asr` is the only workspace crate that still links ONNX Runtime
(Parakeet TDT 0.6B v3). sonos/tract ships `examples/nemo-parakeet-asr`, which
suggests a pure-Rust path. This spike loaded the **shipping istupakov ONNX**
graphs with tract-onnx (same line as `backend-tract`) and compared a 7.435 s
LibriSpeech clip to `polyvoice-transcribe` / ort.

## LSTM

Not unfolded. The prediction network is a 2-layer LSTM with **state ports**
`[2, 1, 640]` (hidden and cell), stepped when a non-blank token is emitted.
The joint (or fused `decoder_joint`) sees one encoder frame `[1, 1024, 1]`.
TDT duration logits skip frames. parakeet-rs and the sonos NNEF example use
the same machine; the example splits decoder/joint, the ONNX export fuses
them.

## Load

| Graph | tract-onnx 0.23 |
|-------|-----------------|
| FP32 `decoder_joint-model.onnx` with greedy-step shapes | runnable |
| FP32 `encoder-model.onnx` with concrete `[1,128,T]` | runnable |
| Weights-only INT8 encoder (`MatMulNBits`) | optimize fails (dynamic axes) |

## RTF and RSS (7.435 s clip)

| Backend | Infer | Process wall | Peak RSS | Text |
|---------|------:|-------------:|---------:|------|
| tract FP32 | 0.95 s (**7.8×**) | 3.08 s (includes ~2 s plan) | 3970 MiB | Errors (`wish` dropped, `don't`/`Phoebe` broken) |
| ort FP32 `polyvoice-transcribe` | (CLI one-shot) | 2.31 s | 2641 MiB | Correct |
| ort INT8 `polyvoice-transcribe` | (CLI one-shot) | 8.8 s cold | 931 MiB | Correct |

Published long-file companion figures remain ort FP32 **11.94×** / INT8
**11.66×**. Tract does not load the INT8 encoder and is not faster on FP32.

## Word times

Frame clock matches parakeet-rs (`frame × 8 × hop 160 / 16 kHz`). Shared words
(`Well`, `I`, `to`, `see`, `it`, `any`) start at the same instant (e.g. Well
0.320). The official NNEF example does not emit word timestamps.

## Verdict

**Do not switch Parakeet off ONNX Runtime.** Keep `ort` in `polyvoice-asr`
only. A tract port would need INT8/MatMulNBits (or NNEF + t2n), a cached
symbolic plan, and feature/greedy parity — without a speed or memory win on
this evidence.
