# src/earshot_vad

Optional **pure-Rust** VAD via [earshot](https://crates.io/crates/earshot).

## Feature

```bash
cargo test --lib earshot_vad --features vad-earshot
```

Default builds never link earshot.

## Role

**Silero remains the production default** and DER reference. Earshot is for
embedded / no-ort experiments. See
`benchmarks/results/earshot-vad-notes.md` before any default switch.

## Contract

- Mono PCM **16 kHz**
- Analysis frame **256 samples** (16 ms)
- Scores continuous in `[0, 1]`; thresholding left to `VadConfig`
