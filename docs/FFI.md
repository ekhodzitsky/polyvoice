# C FFI (ABI v3)

polyvoice exposes a small C API for embedding the **production pipeline v2**
(powerset → ResNet34 → **VBx** by default) without going through Rust.

| Artifact | Path |
|----------|------|
| Header | [`include/polyvoice.h`](../include/polyvoice.h) |
| Example | [`examples/ffi_usage.c`](../examples/ffi_usage.c) |
| Cargo feature | `ffi` (= `pipeline-native` + `vbx`; no `ort`) |

## Build

```bash
# Shared library + install header on the search path you choose
cargo build --release --features ffi

# Example (debug)
cc -I include examples/ffi_usage.c \
  -L target/debug -lpolyvoice \
  -o ffi_usage
# macOS: DYLD_LIBRARY_PATH=target/debug ./ffi_usage
# Linux:  LD_LIBRARY_PATH=target/debug ./ffi_usage
```

## Lifecycle

1. `polyvoice_pipeline_create(profile, models_cache_dir, &handle)`
2. One or more `polyvoice_pipeline_run` / `polyvoice_pipeline_run_format`
3. `polyvoice_free_string` for every string returned by run\*
4. `polyvoice_pipeline_destroy(handle)` **exactly once**

`PolyvoicePipeline` is Send: different handles may run on different threads.
**Do not** call `run` and `destroy` concurrently on the **same** handle.

## Profiles

| Enum | Meaning |
|------|---------|
| `POLYVOICE_PROFILE_MOBILE` | Smaller / mobile-oriented model pair |
| `POLYVOICE_PROFILE_BALANCED` | Default production pair (CLI “balanced”) |

`models_cache_dir` may be `NULL` (default cache). Paths containing `..`
components are rejected (`POLYVOICE_ERR_INVALID_ARG`).

## Audio contract

| Constraint | Value |
|------------|--------|
| Sample format | interleaved `float` PCM mono |
| Sample rate | **16000** Hz (other rates → invalid / unsupported) |
| Max length | **1 hour** @ 16 kHz (`POLYVOICE_ERR_AUDIO_TOO_LONG`) |

## Output formats

`polyvoice_pipeline_run` always returns **JSON** (canonical diarization-result
v1 shape; see [`schema/diarization-result-v1.json`](../schema/diarization-result-v1.json)).

`polyvoice_pipeline_run_format` accepts:

| Enum | Format |
|------|--------|
| `POLYVOICE_FORMAT_JSON` | Same as `run` |
| `POLYVOICE_FORMAT_RTTM` | RTTM (`file_id` fixed to `"audio"`) |
| `POLYVOICE_FORMAT_SRT` | SubRip |
| `POLYVOICE_FORMAT_VTT` | WebVTT |
| `POLYVOICE_FORMAT_TXT` | Plain speaker turns |

Free with `polyvoice_free_string(ptr, len)`.

## Status codes (subset)

| Code | Meaning |
|------|---------|
| `POLYVOICE_OK` | Success |
| `POLYVOICE_ERR_INVALID_ARG` | Null pointer, bad rate, path traversal, bad format |
| `POLYVOICE_ERR_AUDIO_TOO_LONG` | Over max samples |
| `POLYVOICE_ERR_MODEL_LOAD` / `REGISTRY` | Model missing or failed verify |
| `POLYVOICE_ERR_INFERENCE` | Runtime / pipeline stage failure |
| `POLYVOICE_ERR_OUT_OF_MEMORY` | Allocation failure |
| `POLYVOICE_ERR_INTERNAL` | Panic isolation or unexpected error |

`POLYVOICE_ERR_AUDIO_TOO_SHORT` is reserved for ABI stability and is **not**
returned by the current v2 implementation.

## Clustering default

FFI builds **VBx** the same way as the CLI default (PLDA via registry or
`POLYVOICE_VBX_PLDA_DIR`). There is no C flag for AHC/AS-norm; use the Rust or
Python API for those knobs.

## Related

- Runtime architecture: [PIPELINE-ARCHITECTURE.md](PIPELINE-ARCHITECTURE.md)
- Security / readiness: [../PRODUCTION-READINESS.md](../PRODUCTION-READINESS.md)
- Agents / schema: [../examples/agent_quickstart.md](../examples/agent_quickstart.md)
