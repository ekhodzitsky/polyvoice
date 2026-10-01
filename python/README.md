# polyvoice

[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/polyvoice)](https://pypi.org/project/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ekhodzitsky/polyvoice/blob/master/LICENSE)

**Speaker diarization for Python — who spoke when.**

CPU speaker diarization implemented in Rust, with a neural INT8 model pair of
~8.4 MB (plus VBx PLDA parameters). Python remains the host interpreter. The wheel uses the same hand-written
INT8 kernels as the Rust CLI — no ONNX Runtime. Pipeline v2 with VBx
clustering and overlap detection.

## Install

```bash
python -m pip install polyvoice==1.0.0
```

Published **1.0.0 wheels require CPython 3.12**. Linux x86_64/ARM64 wheels
use the `manylinux_2_34` ABI (glibc 2.34+); macOS ARM64 wheels target macOS
11+; Windows wheels target x86_64. The source metadata allows Python 3.9+,
but that does not provide a wheel for every interpreter. Build other
combinations from source with Rust and maturin; see the repository
[contributor guide](https://github.com/ekhodzitsky/polyvoice/blob/master/CONTRIBUTING.md#python-bindings).

## Quick start

```python
import wave
from array import array
import sys
import polyvoice

# Example input: uncompressed mono PCM16 WAV at 16 kHz.
with wave.open("meeting.wav", "rb") as wav:
    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
    pcm = array("h", wav.readframes(wav.getnframes()))
if sys.byteorder != "little":
    pcm.byteswap()
samples = [sample / 32768.0 for sample in pcm]

# Models auto-download on first run (~8.4 MB INT8 pair plus PLDA)
pipeline = polyvoice.Pipeline.balanced()

result = pipeline.run(samples, sample_rate=16000)

print(f"Speakers: {result['num_speakers']}")
for turn in result["turns"]:
    print(f"Speaker {turn['speaker']}: {turn['start']:.1f}s - {turn['end']:.1f}s")
```

## API

- `polyvoice.Pipeline.balanced(models_cache=None, clusterer=None, vbx_plda_dir=None)` — balanced accuracy / speed.
- `polyvoice.Pipeline.mobile(models_cache=None, clusterer=None, vbx_plda_dir=None)` — mobile profile; currently resolves the same INT8 model pair.
  `clusterer` is `"vbx"` (default, matching the CLI) or `"ahc"`. VBx resolves
  its PLDA params via `vbx_plda_dir`, then the `POLYVOICE_VBX_PLDA_DIR` env
  var, then a registry download.
- `pipeline.run(samples, sample_rate)` → `dict` with `num_speakers` and `turns`.
- `pipeline.run_result(samples, sample_rate)` → typed `DiarizationResult` with
  `.to_json()` / `.to_rttm()` / `.to_srt()` / `.to_vtt()` / `.to_txt()` projections.
- `polyvoice.DiarizationResult.from_json(json)` — re-hydrate a saved result.

## Performance

| Pipeline | VoxConverse-test DER (collar 0, overlap-scored) | Model size |
|----------|-------------------------------------------------|------------|
| default (v2+VBx, INT8 kernels) | **13.3%** | ~8.4 MB |

Full protocol, collar/averaging disclosure, and competitor numbers:
[docs/BENCHMARKS.md](https://github.com/ekhodzitsky/polyvoice/blob/master/docs/BENCHMARKS.md).

See the [full repository](https://github.com/ekhodzitsky/polyvoice) for Rust / C / CLI APIs, benchmarks, and development docs.

Release wheels are qualified on CPython 3.12 for Linux x86_64/ARM64, macOS
ARM64 and Windows x86_64. Other source-build interpreters are not an implied
wheel support promise. See the [artifact matrix and checks](https://github.com/ekhodzitsky/polyvoice/blob/master/docs/release-artifacts.md).
