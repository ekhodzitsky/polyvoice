# Parakeet TDT on tract — spike notes

**Date:** 2026-09-21  
**Host:** AMD Ryzen AI 9 HX 370, Linux x86_64  
**Clip:** LibriSpeech `2086-149220-0033.wav` (7.435 s, 16 kHz) — the same file the
sonos/tract `nemo-parakeet-asr` example asserts on.  
**Models:** `istupakov/parakeet-tdt-0.6b-v3-onnx` FP32 encoder + decoder_joint;
weights-only INT8 encoder from the shipping companion export.

Throwaway loader/decoder lived in `/tmp/parakeet-tract-spike` (not in this
crate). tract-onnx **0.23.8** (polyvoice pins 0.23.4).

## Load (tract-onnx)

| Graph | Parse | Optimize + runnable |
|-------|-------|---------------------|
| `decoder_joint-model.onnx` (70 MB), LSTM states bound to `[2,1,640]`, B=1 | 57 ms | **OK** (24 ms) |
| FP32 `encoder-model.onnx` (~2.3 GB data), `audio_signal` `[1,128,T]` | 2.3 s | **OK** for concrete T (T=100: 1.1 s; T=744: ~2 s with decoder) |
| Weights-only INT8 encoder (`MatMulNBits`) | 0.2–0.8 s | **FAIL** (PermuteAxes / dynamic-axis unify) |

Unbound dynamic axes fail LSTM analysis (`input_states_*` batch vs `targets`
batch). Binding the greedy-step shapes that parakeet-rs already uses is enough
for the decoder. The INT8 encoder does not get there.

## LSTM / state (not unfolding)

Both the sonos example and parakeet-rs treat the prediction network as a
**2-layer LSTM with explicit state ports**, stepped once per emitted token:

- `input_states_1` / `output_states_1`: hidden `[2, 1, 640]`
- `input_states_2` / `output_states_2`: cell `[2, 1, 640]`
- Joint (or fused `decoder_joint`) sees one encoder frame `[1, 1024, 1]` and
  the last token; duration logits skip frames (TDT).

The official example splits decoder.nnef / joint.nnef via
`t2n_export_nemo --split-joint-decoder`. Shipping ONNX fuses them. Same state
machine. **No sequence unfold.**

## RTF (this clip, process-cold except tract infer split)

| Backend | Infer-only | Process wall | Peak RSS | Transcript |
|---------|------------|--------------|----------|------------|
| tract FP32 encoder + decoder_joint | **0.95 s (7.8×)** enc 0.90 / dec 0.06 | 3.08 s (includes ~2 s optimize) | **3970 MiB** | Garbled (see below) |
| ort FP32 via `polyvoice-transcribe` (diarize + ASR) | n/a (one-shot CLI) | 2.31 s | 2641 MiB | Correct Libri sentence |
| ort weights-only INT8 via same CLI | n/a | 8.8 s (cold session; not the long-file 11.7× figure) | 931 MiB | Correct |

Published companion protocol (warmup, longer audio): ort FP32 **11.94×**, INT8
**11.66×**, INT8 peak **5.5 GiB** on that protocol. Tract on this 7 s clip is
already slower than that FP32 figure at the encoder, and cannot load the INT8
encoder at all.

Encoder is ~94 % of tract infer time. LSTM decode is cheap.

## Timestamp parity

Same hop (160) × encoder stride (8) formula as parakeet-rs. Words that both
stacks emit line up on the frame clock:

| Word | ort start–end | tract token span |
|------|---------------|------------------|
| Well | 0.320–0.560 | 0.320–0.560 (` W`+`ell`) |
| , | 0.560–0.640 | 0.560–0.640 |
| I | 0.640–0.800 | 0.640–0.800 |
| to | 1.280–1.360 | 1.280–1.360 |
| see | 1.360–1.520 | 1.360–1.520 |
| it | 1.520–1.680 | 1.520–1.680 |
| any | 1.680–1.920 | 1.680–1.920 |

tract dropped `wish` and broke `don't` / `Phoebe` / `eyes` — feature or greedy
mismatch, not a different time base. The official NNEF example prints a string
only (no word times); word times are a parakeet-rs post-process.

## Official example

`sonos/tract/examples/nemo-parakeet-asr` exports **NNEF** with
`torch_to_nnef` / `t2n_export_nemo`, not the istupakov ONNX pair. This spike
did not install NeMo + t2n. Running the shipping ONNX graphs on tract-onnx is
the product-relevant experiment.

## Decision

**Do not replace `polyvoice-asr`'s ONNX Runtime with tract.** Keep `ort`
isolated in that companion. Effort to productize tract here is weeks of NNEF
or symbolic-T work plus an INT8 gap that may never close; it would not beat
the INT8 RSS or the published RTFx.

Follow-up: none for a tract ASR port. Compression work (int4 / sub-GB) stays
on the ort encoder if pursued.
