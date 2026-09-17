# polyvoice-asr

Opt-in ASR companion for [polyvoice](https://github.com/ekhodzitsky/polyvoice).
Wraps [parakeet-rs](https://crates.io/crates/parakeet-rs) (NVIDIA Parakeet TDT)
behind polyvoice's core `Asr` trait and emits **native word-level timestamps**
for the who-said-what cascade.

## Why a separate crate

The Parakeet TDT 0.6B v3 model is a 2 549 805 858-byte (≈2.55 GB) FP32 ONNX
export (≈669 MB as the recommended weights-only INT8 encoder — see
*Model files*) — incompatible with polyvoice's core
footprint (INT8 production pair ~8.4 MB, wasm-clean clustering). So ASR lives
here, **never** as a core default feature. The `polyvoice-transcribe` CLI
diarizes with the same INT8 kernels as the product `polyvoice` CLI (`pipeline-native`,
no `libonnxruntime`). Parakeet still needs ONNX Runtime: this crate pins
`ort = 2.0.0-rc.12` (same version as core's optional `onnx` feature), enforced
by `scripts/check-ort-version.sh` in CI (two `ort` versions = two runtimes =
crashes).

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

## CLI: who-said-what

The `polyvoice-transcribe` binary (behind the `cli` feature) runs the full
cascade — diarize (v2 + VBx kernels, same as the product CLI) → one ASR pass →
join — and emits who-said-what. It lives here rather than in the core
`polyvoice` CLI because the core crate cannot depend on this companion
(package cycle). Pass `--clusterer ahc` for the cosine-AHC backend.

```bash
cargo run -p polyvoice-asr --features cli --bin polyvoice-transcribe -- \
    meeting.wav --asr-model ./models/parakeet-tdt --format json
```

- `--format json` — turns with `text` + a per-word array (matches
  `schema/diarization-result-v1.json`); `srt` / `vtt` / `txt` render
  `SPEAKER_NN: text`.
- stdout carries only the result; progress goes to stderr.

## Model files

### Recommended: weights-only INT8 (≈741 MB download)

Prebuilt export — MatMul weights int8 `MatMulNBits` (128-wide blocks), Conv
weights per-output-channel int8, activations and decoder FP32. Word-loss
neutral vs the FP32 export (2 / 13 312 boundary words; protocol and numbers
in [docs/BENCHMARKS.md](https://github.com/ekhodzitsky/polyvoice/blob/main/docs/BENCHMARKS.md)):

```bash
mkdir -p models/parakeet-tdt-w8 && cd models/parakeet-tdt-w8
for f in encoder-model.int8.onnx encoder-model.int8.onnx.data \
         decoder_joint-model.onnx vocab.txt SHA256SUMS; do
  curl -fSLO "https://huggingface.co/ekhodzitsky/parakeet-tdt-0.6b-v3-onnx-weights-only-int8/resolve/main/$f"
done
sha256sum -c SHA256SUMS && cd -
```

Point `from_dir` / `--asr-model` at that directory — the loader picks the
`*.int8.onnx` encoder automatically (do not put the FP32 `encoder-model.onnx`
in the same directory: it silently wins). Requires ONNX Runtime ≥ 1.22 CPU
(`MatMulNBits` int8 kernels) — the `ort` version pinned by this crate already
includes them. Measured against FP32 on the same fixtures: 11.66× vs 11.94×
RTFx, 5.50 vs 7.07 GiB peak RSS.

### FP32 reference (≈2.55 GB)

Source: [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)

- `encoder-model.onnx` (41 770 866 B) + `encoder-model.onnx.data` (2 435 420 160 B)
- `decoder_joint-model.onnx` (72 520 893 B)
- `vocab.txt` (93 939 B)

Total download: 2 549 805 858 B (≈2.55 GB).

### Build the INT8 encoder yourself

Deterministic rebuild of the hosted export from the FP32 download
(~1 min, ~3.2 GiB peak RAM; the rebuild is byte-identical):

```bash
pip install "onnxruntime>=1.30" onnx onnx-ir numpy
python3 scripts/quantize-parakeet-encoder.py ./models/parakeet-tdt ./models/parakeet-tdt-w8
```

Keep the decoder FP32 in any custom build: the INT8 decoder regresses word
parity (21 / 13 312 FP32-only words).

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
