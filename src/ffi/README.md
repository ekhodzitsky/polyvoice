# src/ffi

## Purpose

C FFI bindings (ABI v2) exposing polyvoice pipeline to C callers.

## Surfaces

- `polyvoice_pipeline_create(profile)`
- `polyvoice_pipeline_run(handle, samples, sample_rate)`
- `polyvoice_pipeline_free(handle)`
- `PolyvoicePipeline` — opaque C handle

## Dependencies

- `pipeline` — Pipeline orchestration
- `models` — ModelRegistry
- `types` — DiarizationConfig, Profile
- `vad` — VadConfig

## Invariants

- Every create has exactly one free (memory safety).
- C-visible structs use `#[repr(C)]`.

## Verification

```bash
cargo test --test ffi_smoke_test --features ffi
```

## Notes

- Header file: `include/polyvoice.h`
- Example usage: `examples/ffi_usage.c`
