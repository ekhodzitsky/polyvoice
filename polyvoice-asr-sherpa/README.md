# polyvoice-asr-sherpa

Opt-in **CJK/multilingual** ASR companion for [polyvoice]: SenseVoice and
Paraformer via [sherpa-onnx], implementing the same core `Asr` trait as the
Parakeet companion — so the who-said-what cascade (diarize → transcribe →
word→speaker join, all in the core crate) works unchanged.

## The trade-off, up front

sherpa-onnx ships a **second C++ ONNX Runtime** that does not share the core
crate's `ort`. That is why this crate:

- is strictly **opt-in** and deliberately **outside the polyvoice workspace**
  (own lockfile, own build) — it can never leak into the core dependency
  graph, the ~30 MB default footprint, or the wasm target;
- should only be chosen when you actually need CJK.

## Choosing a companion

| | `polyvoice-asr` (Parakeet TDT) | `polyvoice-asr-sherpa` (this crate) |
|---|---|---|
| Languages | ~25 European (en, de, fr, …) | zh, en, ja, ko, yue (SenseVoice); zh (Paraformer) |
| Runtime | shared core `ort` (one ONNX runtime) | second C++ runtime (sherpa-onnx) |
| Timestamps | native word-level | token starts → words (see below) |
| Confidence | per-word | not exposed by sherpa (`1.0` reported) |

Rule of thumb: European languages → Parakeet; Chinese/Japanese/Korean/
Cantonese → this crate (SenseVoice `language = "auto"` handles mixed input).

## Timestamp granularity

Sherpa's offline recognizers emit **token start times** only. This crate
merges BPE continuation tokens into words (CJK characters stay one word
each — the natural unit for zh/ja), sets each word's end to the next word's
start (clip end for the last word), and reports `confidence = 1.0`.

## Models (provenance)

Same discipline as the Parakeet companion: you point the constructor at a
model directory; nothing downloads implicitly.

**SenseVoice-Small** (zh/en/ja/ko/yue, Apache-2.0), sherpa-onnx export
`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`:
<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>
`model.int8.onnx` sha256 `c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51` (measured 2026-07-13) — verify after download
(`shasum -a 256`); if upstream re-uploads the bundle the value changes, so a
mismatch with this readme means "check the release page", not tampering.

**Paraformer** (zh, Apache-2.0), sherpa-onnx export
`sherpa-onnx-paraformer-zh-2023-09-14` from the same release page.

```rust,no_run
use polyvoice::{Asr, types::SampleRate};
use polyvoice_asr_sherpa::SenseVoiceAsr;

let dir = std::path::Path::new("data/sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17");
let asr = SenseVoiceAsr::from_files(dir.join("model.int8.onnx"), dir.join("tokens.txt"), "auto")?;
let words = asr.transcribe(&samples, SampleRate::new(16_000).expect("rate"))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Verification

```bash
cargo test                        # unit: token→word merging (model-free)
cargo test -- --ignored           # integration: needs the SenseVoice bundle
cargo clippy --all-targets -- -D warnings
# isolation: from the workspace root, sherpa must NOT be in the core tree
cargo tree -e no-dev | grep -i sherpa && echo LEAK || echo CLEAN
```

[polyvoice]: https://github.com/ekhodzitsky/polyvoice
[sherpa-onnx]: https://github.com/k2-fsa/sherpa-onnx
