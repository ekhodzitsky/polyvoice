# speakrs-rttm

Thin CLI that turns one 16 kHz WAV into RTTM via [speakrs](https://github.com/avencera/speakrs).
Used by `benchmarks/runners/speakrs_runner.py`.

## Build

Point the path dependency at a local speakrs checkout (default assumes
`~/src/speakrs` next to `~/src/polyvoice-h2h`):

```bash
# edit Cargo.toml path if needed
cargo build --release --manifest-path benchmarks/tools/speakrs-rttm/Cargo.toml

# CoreML modes on macOS:
cargo build --release --manifest-path benchmarks/tools/speakrs-rttm/Cargo.toml --features coreml
```

## Run

```bash
./benchmarks/tools/speakrs-rttm/target/release/speakrs-rttm --mode cpu file.wav
SPEAKRS_MODELS_DIR=/path/to/models ./... --mode coreml file.wav -o out.rttm
```
