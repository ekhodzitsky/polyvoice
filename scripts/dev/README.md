# Dev / lab scripts

One-off calibration, quantization probes, and feature dumps. **Not**
supported product examples — those live in `examples/` (`byo_embedder`,
`build_asnorm_cohort`, `ffi_usage.c`).

These files are plain Rust sources, not Cargo `[[example]]` targets; run them
with an ad-hoc `cargo script` / `rustc` / temporary `[[bin]]` only when
developing models. Do not add them under `examples/` (Cargo auto-discovers
`examples/*.rs` and may pull them into `--all-targets` builds).
