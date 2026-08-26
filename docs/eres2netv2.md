# Optional ERes2NetV2 (and CAM++-zh) embedders

These models are **not** part of the default ~8.4 MB INT8 profile. They are optional
downloads for accuracy experiments on short segments and CJK audio.

## Enable

```rust
// After ModelRegistry::ensure("eres2netv2")
use polyvoice::embedder::{ERes2NetV2Extractor, Embedder};
use polyvoice::onnx::ExecutionProvider;

let emb = ERes2NetV2Extractor::new(path, /* pool_size */ 2, ExecutionProvider::Cpu)?;
assert_eq!(emb.dim(), 192);
let vec = emb.embed(&audio_16k_mono)?;
```

Adapter registry names (built-ins):

| Config string | Model key | Dim | Notes |
|---------------|-----------|-----|--------|
| `eres2netv2` / `eres2net-v2` | `eres2netv2` | 192 | Short-utterance ERes2NetV2 |
| `cam++-zh` / `campplus-zh` | `cam_pp_zh` | 192 | CAM++ zh-cn common |

Default embedder remains WeSpeaker ResNet34 / CAM++ advanced as already configured
in profiles — these keys are never profile-resolved automatically.

## License

Apache-2.0 weights from 3D-Speaker / ModelScope, ONNX mirrors at
[csukuangfj/speaker-embedding-models](https://huggingface.co/csukuangfj/speaker-embedding-models)
(ungated). Integrity is SHA-256 gated; minisign signatures can be added later
with the project release key.

## Preprocessing

Same 80-bin log-mel fbank path as the in-tree CAM++ adapter (`FbankOnnxExtractor`).
If upstream training used a different front-end, DER/EER may not match the paper
exactly — treat this as an engineering adapter, not a bit-identical reimplementation.

## Non-goals

- Not bundled in crates.io package or Docker default image
- Not the CLI default embedder
- Full short-segment EER leaderboard is a follow-up measurement, not a gate for this adapter
