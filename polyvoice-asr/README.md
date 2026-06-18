# polyvoice-asr

Opt-in ASR companion for [polyvoice](https://github.com/ekhodzitsky/polyvoice).
Wraps [parakeet-rs](https://crates.io/crates/parakeet-rs) (NVIDIA Parakeet TDT)
behind polyvoice's core `Asr` trait and emits **native word-level timestamps**
for the who-said-what cascade.

## Why a separate crate

The Parakeet TDT 0.6B v3 model is ~600 MB — incompatible with polyvoice's core
footprint (default build ~30 MB, wasm-clean). So ASR lives here, **never** as a
core default feature. The two crates **share one ONNX runtime**: this crate pins
the exact same `ort = 2.0.0-rc.12` as the core, enforced by
`scripts/check-ort-version.sh` in CI (two `ort` versions = two runtimes = crashes).

## Usage

```rust,no_run
use polyvoice_asr::ParakeetAsr;
use polyvoice::{Asr, types::SampleRate};

let asr = ParakeetAsr::from_dir("./models/parakeet-tdt")?;
let sr = SampleRate::new(16_000).expect("valid rate");
let words = asr.transcribe(&audio_16k_mono, sr)?; // Vec<Word> with global timestamps
# Ok::<(), Box<dyn std::error::Error>>(())
```

Long audio is handled automatically: TDT has a ~8-10 min sequence limit, so input
longer than the chunk window (default 240 s, 5 s overlap) is split into
overlapping chunks whose word timestamps are stitched at the overlap midpoint —
no duplicated or dropped words at the seams. Tune with `.with_chunking(secs, overlap)`.

## Model files

Download the TDT ONNX export into one directory and point `from_dir` at it.
Source: [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)

- `encoder-model.onnx` + `encoder-model.onnx.data`
- `decoder_joint-model.onnx`
- `vocab.txt`

## Execution providers

CPU by default. Forward an execution provider via `from_dir_with_config` using the
re-exported `ExecutionProvider` / `ExecutionConfig`, and enable the matching
feature (`coreml` / `xnnpack` / `nnapi`).

> Note: upstream reports CoreML is **unstable** with the TDT model — prefer CPU or
> XNNPACK on Apple Silicon.

## Verification

```bash
cargo test -p polyvoice-asr                 # unit (chunk-stitch) + gated smoke
cargo clippy -p polyvoice-asr --all-targets -- -D warnings
bash scripts/check-ort-version.sh           # single shared ort across the workspace
POLYVOICE_ASR_MODEL_DIR=./models/parakeet-tdt cargo test -p polyvoice-asr  # real inference
```

## License

MIT (matching polyvoice). parakeet-rs is MIT OR Apache-2.0.
