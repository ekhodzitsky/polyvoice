# Optional Sortformer adapter

NVIDIA **Streaming Sortformer v2** as an opt-in end-to-end diarizer, proving
that polyvoice’s adapter registry is pluggable: a third-party model plugs in
via feature flag + manifest entry without changing the default pipeline.

## Enablement

```toml
# Cargo.toml
polyvoice = { version = "...", features = ["sortformer", "onnx", "download"] }
```

```bash
# Explicit download only — never pulled by default profiles or CI
cargo run --features "cli,sortformer" -- ensure-model sortformer_v2   # if CLI supports it
# or in library code:
# ModelRegistry::default()?.ensure("sortformer_v2")?
```

```rust,ignore
use polyvoice::sortformer::{SortformerConfig, SortformerDiarizer};

let path = registry.ensure("sortformer_v2")?;
let mut diarizer = SortformerDiarizer::from_path(path)?;
let turns = diarizer.diarize(&audio_16k_mono)?;

// Streaming: state (FIFO + speaker cache) is kept between calls
let chunk_turns = diarizer.diarize_chunk(&chunk)?;
```

Adapter registry (when `download` is enabled):

```rust,ignore
use polyvoice::models::{AdapterRegistry, AdapterStage};
use polyvoice::sortformer::{register_with, ADAPTER_TYPE};

let mut reg = AdapterRegistry::with_builtins(); // already includes sortformer-v2 when feature is on
// or: register_with(&mut reg)?;
assert!(reg.contains(AdapterStage::Diarizer, ADAPTER_TYPE));
```

## Hard speaker cap

The architecture uses **four sigmoid heads**. Config with `max_speakers > 4`
is a **config error** (not silent degradation). On DIHARD subsets with ≥5
speakers, published DER is ~41–44% — out of scope for this model.

## When to prefer VBx (default polyvoice path)

| Situation | Prefer |
|-----------|--------|
| ≤4 speakers, low-latency streaming, single ONNX graph | **Sortformer** |
| Unknown / large speaker count, offline batch, accuracy on meetings | **VBx** (+ powerset segmenter + ResNet34) |
| Default product / MIT-only default delivery | **VBx path** (Sortformer is never default) |

VBx remains the path for arbitrary speaker counts. Sortformer is an optional
specialist for the ≤4-speaker streaming niche.

## License and attribution (v2, CC-BY-4.0)

- **Weights used:** community ONNX of
  [`nvidia/diar_streaming_sortformer_4spk-v2`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)
  mirrored at
  [`cgus/diar_streaming_sortformer_4spk-v2-onnx`](https://huggingface.co/cgus/diar_streaming_sortformer_4spk-v2-onnx).
- **License:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
- **Attribution:** NVIDIA Corporation (Sortformer v2). ONNX conversion by
  Altunenes for [parakeet-rs](https://github.com/altunenes/parakeet-rs) (MIT).
- **polyvoice source** stays MIT; weights are **not** redistributed with the
  crate. Redistributors who ship the ONNX file must comply with CC-BY-4.0
  (see `NOTICE`).

### v2.1 (NVIDIA Open Model License) — not wired

`diar_streaming_sortformer_4spk-v2.1` is under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/):
revocable, requires attribution notice, Trustworthy-AI terms, indemnity, and
patent-litigation termination. **Do not enable v2.1 in polyvoice without a
written legal review.** This adapter defaults to **v2 (CC-BY-4.0)** only.

## Model size and download path

- FP32 ONNX ≈ **470–492 MB**. Not edge-friendly; optional download only.
- Manifest id: `sortformer_v2` (SHA-256 verified). Not signed with the
  polyvoice minisign key until a release artifact is published; integrity is
  SHA-256 for this optional path.
- **Never** included in `cargo package` artifacts or default CI jobs.

## Streaming notes

State between calls follows the NeMo / parakeet-rs pattern: `spkcache`,
`fifo`, and length tensors are inputs; new embeddings and predictions update
FIFO and (with smart compression) the speaker cache. Frame hop is **80 ms**;
default latency ≈ `(chunk_len + right_context) * 80 ms` ≈ **10 s** for the
stock chunk geometry (overridable via ONNX metadata).

## Provenance of metrics

Do not republish NVIDIA GPU RTF / DER figures as polyvoice results. Any
published polyvoice numbers must be measured on this adapter (CPU or
explicitly labelled EP) against the project’s RTTM references (AMI
forced-alignment refs from NVIDIA cards are **not** directly comparable to
polyvoice’s AMI evaluation).
