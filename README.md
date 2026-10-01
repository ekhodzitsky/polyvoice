# polyvoice

Stable CPU speaker diarization for Rust, Python, C and the command line.

[![Crates.io](https://img.shields.io/crates/v/polyvoice)](https://crates.io/crates/polyvoice)
[![PyPI](https://img.shields.io/pypi/v/polyvoice)](https://pypi.org/project/polyvoice/)
[![Downloads](https://img.shields.io/crates/d/polyvoice)](https://crates.io/crates/polyvoice)
[![Docs.rs](https://docs.rs/polyvoice/badge.svg)](https://docs.rs/polyvoice/1.0.0/polyvoice/)
[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ekhodzitsky/polyvoice)](https://github.com/ekhodzitsky/polyvoice/releases)
[![Codecov](https://codecov.io/gh/ekhodzitsky/polyvoice/branch/master/graph/badge.svg)](https://codecov.io/gh/ekhodzitsky/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A speaker diarization crate. Powerset neural segmentation, WeSpeaker
ResNet34 embeddings, VBx clustering with automatic speaker count. One
`Pipeline` call from 16 kHz mono to timestamped turns. The native product uses
**no ONNX Runtime**: hand-written INT8 kernels, ~8.4 MB production
neural model pair, with no gated model access. Code and neural weights are MIT;
VBx PLDA parameters are CC-BY-4.0 (see [NOTICE](NOTICE)).
Python, C FFI and a CLI ship from the same crate.

## Examples

Library (kernels, models auto-download):

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::{Pipeline, PipelineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // PipelineConfig::default() is VBx when the `vbx` feature is on. The
    // struct is non-exhaustive: start from the default and assign fields.
    let mut config = PipelineConfig::default();
    config.profile = Profile::Balanced;
    let pipeline = Pipeline::builder()
        .config(config)
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
polyvoice download-models --profile balanced   # ~8.4 MB neural pair; no token
polyvoice diarize meeting.wav --output meeting.rttm
```

```
SPEAKER meeting 1   0.000  12.784  <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER meeting 1  13.005   2.530  <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER meeting 1  15.688  10.323  <NA> <NA> SPEAKER_02 <NA> <NA>
```

Processing speed depends on the host and recording; measured results are below. Python:
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
| Weights | MIT neural pair; CC-BY-4.0 PLDA, ungated | HF token required |
| External ML runtime | none (native kernels) | PyTorch stack |
| DER₀ | 13.3 % | **11.3 %** |
| Speed | See the measured release results below | Hardware-dependent |

These are collar-matched published results, not a new same-host head-to-head.
Polyvoice favors CPU deployment and native integration; it does not claim the
lowest DER. Rust and CLI use do not require Python.

## Speed

The **1.0.0** release was measured at commit `80bf4b1` on Linux x86_64
(Ryzen AI 9 HX 370) and macOS ARM64 (Apple M1 Pro). DER₀ below is micro,
collar 0, with overlap scored. RTFx is audio duration / processing time;
higher is faster. These are recorded runs, not a universal speed guarantee.

| Corpus | Linux DER₀ | Linux RTFx | Mac DER₀ | Mac RTFx |
|---|---:|---:|---:|---:|
| VoxConverse-test (232 files) | 13.34% | 85.7× | 13.33% | 192.8× |
| AMI-test (16 files) | 24.19% | 103.1× | 23.61% | 229.2× |

The locked Mac Vox-3 smoke passed at **7.11% micro / 7.39% macro DER₀**,
**163.3× RTFx** and **454.9 MiB peak RSS**. The neural model pair is
**8,414,314 bytes**, excluding PLDA parameters and application memory.
Linux ARM64 and Windows have packaged-consumer qualification; these rows do
not claim full-corpus timing on those targets.

[Release evidence](https://github.com/ekhodzitsky/polyvoice/releases/tag/v1.0.0)
includes the complete platform reports. [Benchmark history](docs/BENCHMARKS.md)
retains earlier measurements, including faster Linux runs; their dates,
builds and timings must not be substituted for this release run.

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
| CLI from source | `cargo install polyvoice --version 1.0.0 --locked --features cli` |
| Python 3.12 wheels | `python -m pip install polyvoice==1.0.0` |

```toml
[dependencies]
polyvoice = { version = "1", features = ["pipeline-native", "vbx"] }
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
The advertised 1.x API is stable under [SemVer](docs/semver.md). Experimental
tract/MCP surfaces and companion ASR have separate stability boundaries.
Rust source builds require rustc 1.94 or newer; published Python wheels target
CPython 3.12. See the [platform and ABI matrix](docs/release-artifacts.md).

---

> **Name:** this project is **polyvoice — speaker diarization for Rust**,
> unrelated to ByteDance's "PolyVoice" speech-translation research.
