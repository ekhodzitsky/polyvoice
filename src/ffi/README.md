# src/ffi

## Purpose

C FFI bindings (ABI v3) exposing polyvoice pipeline to C callers.

## Surfaces

- `polyvoice_pipeline_create(profile, models_cache_dir, out_handle)`
- `polyvoice_pipeline_run(handle, samples, n_samples, sample_rate, out_json, out_json_len)` — JSON output
- `polyvoice_pipeline_run_format(handle, samples, n_samples, sample_rate, format, out_str, out_str_len)` — JSON/RTTM/SRT/VTT/TXT
- `polyvoice_pipeline_destroy(handle)`
- `polyvoice_free_string(ptr, len)`
- `PolyvoicePipeline` — opaque C handle

Both run functions share one internal `run_impl` (input validation, pipeline
run, format rendering); the format selector is the only difference.

## Dependencies

- `pipeline` — Pipeline orchestration
- `models` — ModelRegistry
- `types` — DiarizationConfig, Profile
- `vad` — VadConfig

## Invariants

- Every create has exactly one free (memory safety).
- C-visible structs use `#[repr(C)]`.
- `models_cache_dir` rejects paths with parent-dir (`..`) components
  (`InvalidArg`); absolute and relative paths are both accepted.

## Verification

```bash
cargo test --test ffi_smoke_test --features ffi
```

## Notes

- Header file: `include/polyvoice.h`
- Example usage: `examples/ffi_usage.c`
