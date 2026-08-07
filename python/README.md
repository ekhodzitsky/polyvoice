# polyvoice

[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/polyvoice)](https://pypi.org/project/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ekhodzitsky/polyvoice/blob/master/LICENSE)

**Speaker diarization for Python — who spoke when.**

Rust-powered, ONNX-based speaker diarization that runs on CPU, fits in 30 MB,
and requires zero Python runtime overhead. Pipeline v2 with VBx clustering
and overlap detection.

## Install

```bash
pip install polyvoice
```

Requires Python 3.9+.

## Quick start

```python
import polyvoice

# Models auto-download on first run (~30 MB)
pipeline = polyvoice.Pipeline.balanced()

result = pipeline.run(samples, sample_rate=16000)

print(f"Speakers: {result['num_speakers']}")
for turn in result["turns"]:
    print(f"Speaker {turn['speaker']}: {turn['start']:.1f}s - {turn['end']:.1f}s")
```

## API

- `polyvoice.Pipeline.balanced(models_cache=None, clusterer=None, vbx_plda_dir=None)` — balanced accuracy / speed.
- `polyvoice.Pipeline.mobile(models_cache=None, clusterer=None, vbx_plda_dir=None)` — smaller, faster model.
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
| default (v2+VBx) | **15.2%** | ~30 MB |

Full protocol, collar/averaging disclosure, and competitor numbers:
[docs/BENCHMARKS.md](https://github.com/ekhodzitsky/polyvoice/blob/master/docs/BENCHMARKS.md).

See the [full repository](https://github.com/ekhodzitsky/polyvoice) for Rust / C / CLI APIs, benchmarks, and development docs.
