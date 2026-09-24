# polyvoice

WAV in, speaker turns out.

[![Crates.io](https://img.shields.io/crates/v/polyvoice)](https://crates.io/crates/polyvoice)
[![Downloads](https://img.shields.io/crates/d/polyvoice)](https://crates.io/crates/polyvoice)
[![Docs.rs](https://docs.rs/polyvoice/badge.svg)](https://docs.rs/polyvoice)
[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ekhodzitsky/polyvoice)](https://github.com/ekhodzitsky/polyvoice/releases)
[![Codecov](https://codecov.io/gh/ekhodzitsky/polyvoice/branch/master/graph/badge.svg)](https://codecov.io/gh/ekhodzitsky/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A speaker diarization crate. Powerset neural segmentation, WeSpeaker
ResNet34 embeddings, VBx clustering with automatic speaker count. One
`Pipeline` call from 16 kHz mono to timestamped turns. The native product uses
**no ONNX Runtime**: hand-written INT8 kernels, ~8.4 MB production
model pair, MIT, ungated. No ONNX Runtime in the core crate.
Python, C FFI and a CLI ship from the same crate.

## Examples

Library (kernels, models auto-download):

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::{Pipeline, PipelineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // PipelineConfig::default() is VBx when the `vbx` feature is on.
    let pipeline = Pipeline::builder()
        .config(PipelineConfig {
            profile: Profile::Balanced,
            ..PipelineConfig::default()
        })
        .with_models_from(ModelRegistry::default()?)
        .build()?;
    // 16 kHz WAV via ryf. Other rates/formats: `--features audio-io`.
    let (samples, sr) = polyvoice::wav::load_audio(std::path::Path::new("meeting.wav"))?;
    let result = pipeline.run(&samples, SampleRate::new(sr).ok_or("bad sample rate")?)?;
    for turn in &result.turns {
        println!("{}: {:.1}s - {:.1}s", turn.speaker, turn.time.start, turn.time.end);
    }
    Ok(())
}
```

CLI:

```bash
polyvoice download-models --profile balanced   # ~8.4 MB, MIT, no token
polyvoice diarize meeting.wav --output meeting.rttm
```

```
SPEAKER meeting 1   0.000  12.784  <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER meeting 1  13.005   2.530  <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER meeting 1  15.688  10.323  <NA> <NA> SPEAKER_02 <NA> <NA>
```

A 1-hour meeting diarizes in about a minute on a laptop. Python:
`pip install polyvoice` ([python/README.md](python/README.md)). C FFI:
[docs/FFI.md](docs/FFI.md).

## Surfaces

| Surface | Engine | Links ort |
|---|---|---|
| CLI, `--features cli` | INT8 kernels | no |
| Rust library, `pipeline-native,vbx` | INT8 kernels | no |
| C FFI, `--features ffi` | INT8 kernels | no |
| BYO embedder, `--no-default-features` | yours | no |
| Python wheel, `pip install polyvoice` | INT8 kernels | no |

## Compared to pyannote

Like-for-like, strict collar 0, VoxConverse-test (232 files). Full matrix
(incl. diart, whisperx, speakrs): [compare](docs/COMPETITORS.md).

| | polyvoice | pyannote 3.1 |
|---|---|---|
| Job | diarization crate | research diarization |
| Runtime | Rust, CPU-only | PyTorch, GPU recommended |
| Weights | MIT, ungated | HF token required |
| External ML runtime | none (native kernels) | PyTorch stack |
| DER₀ | 13.3 % | **11.3 %** |
| Speed | **~162× realtime** (Ryzen AI 9 HX 370) | GPU-bound |

The trade is explicit: ~2 DER points for a CPU-only, MIT, ungated deploy
with no Python. Not the accuracy leader — the deployability leader.

## Speed

Kernels (product default) vs same-host ONNX Runtime, EP=cpu, INT8.
Linux x86_64: Ryzen AI 9 HX 370, 2026-09-08. Darwin: Apple Silicon.
DER₀ is strict collar 0. Protocol: [benchmarks](docs/BENCHMARKS.md).

| Corpus | DER₀ | kernels Linux | ort Linux | kernels Darwin |
|---|---:|---:|---:|---:|
| VoxConverse-test (232) | 13.3 % | **~162×** | ~150× | ~130× |
| AMI-test (16) | 24.2 % | **~193×** | ~171× | ~109× |
| Vox-3 smoke | 7.0 % | ~111×, **~158×** wall at `--jobs 3` | ~129×, ~151× at `--jobs 3` | ≥117× |

Peak RSS on the Vox-3 smoke: **~300 MiB** kernels vs ~620 MiB ort at
jobs=1; ~470 MiB vs ~740 MiB at `--jobs 3` (one shared pipeline, DER
bit-identical to jobs=1). On-disk INT8 pair: **8,414,314 bytes** — a
locked scoreboard floor, as are DER and RSS (`tests/native_scoreboard.json`).

## How it works

```
audio (f32 PCM)
  → powerset neural segmentation (overlap-aware)
  → WeSpeaker ResNet34 embeddings
  → VBx clustering (AHC / K-means / NME-SC alternatives, automatic speaker count)
  → overlap resegmentation → speaker turns
```

## Install

| Platform | Get it |
|---|---|
| Linux x86_64 / ARM64, macOS ARM64, Windows x86_64 | [Pre-built binaries](https://github.com/ekhodzitsky/polyvoice/releases/latest) |
| Rust library (kernels, no ort) | `cargo add polyvoice --features "pipeline-native,vbx"` |
| From source | `cargo install polyvoice --features cli` · `"cli,audio-io"` · `cli-tract` · `ffi` |

```toml
[dependencies]
polyvoice = { version = "0.21", features = ["pipeline-native", "vbx"] }
```

rustc **1.94**. Default features are empty: the published crate is the
ort-free BYO core; models and engines are opt-in features
([library mode](docs/library-mode.md)). Frozen surfaces and bump rules:
[semver](docs/semver.md).

[benchmarks](docs/BENCHMARKS.md) | [api](docs/API.md) |
[architecture](docs/PIPELINE-ARCHITECTURE.md) |
[library mode](docs/library-mode.md) | [ffi](docs/FFI.md) |
[python](python/README.md) |
[production readiness](PRODUCTION-READINESS.md) |
[CHANGELOG](CHANGELOG.md)

The native product provides batch diarization: no ASR or speaker identification.
BYO Rust streaming is a separate API. Empty default features still depend on
Rust crates; native Darwin uses C shims and Accelerate, and download-enabled
builds include native crypto. See the [dependency contract](docs/strategy/zero-deps.md)
and [1.0 scope and release gates](PRODUCTION-READINESS.md).
Beta (0.x): the public API may break between minor versions — pin an exact
version in production. MIT.

---

> **Name:** this project is **polyvoice — speaker diarization for Rust**,
> unrelated to ByteDance's "PolyVoice" speech-translation research.
