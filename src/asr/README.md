# src/asr

## Purpose

Trait-only ASR (speech-to-text) interface: `Asr` + `AsrError`. The stable target
for the opt-in `polyvoice-asr` companion crate and the word→speaker join. No
backend, no `ort`/model dependency — pure-Rust, wasm-clean.

## Surfaces

- `Asr` trait — `transcribe(&[f32], SampleRate) -> Result<Vec<Word>, AsrError>` (object-safe)
- `AsrError`

## Dependencies

- `types` — `Word`, `SampleRate`

## Invariants

- No backend / `ort` / model dependency here (default build stays ~30 MB, wasm-clean).
- `Asr` is object-safe (`Box<dyn Asr>` compiles).

## Verification

```bash
cargo test --lib asr
cargo build --no-default-features
cargo build --target wasm32-unknown-unknown --no-default-features --lib
```

## Notes

- The Parakeet backend (`polyvoice-asr`) and the word→speaker join are separate tasks.
