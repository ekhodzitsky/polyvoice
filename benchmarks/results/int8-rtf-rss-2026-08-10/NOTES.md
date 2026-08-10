# INT8 RTF + peak RSS (2026-08-10)

## Config

| Knob | Value |
|------|--------|
| Crate | **0.17.0** (`polyvoice-bench` report) |
| Profile | `balanced` → `powerset_int8` + `resnet34_int8` |
| Pipeline | v2 + **VBx** |
| EP | **CoreML** (`auto` on Apple M1 Pro) |
| DER collar | **0.0** (headline micro) |
| Host | Apple M1 Pro, macOS Darwin 25.1 arm64 |

Reproduce:

```bash
cargo build --release --features cli --bin polyvoice-bench --bin polyvoice
MAX_VOX=10 MAX_AMI=16 DATE=2026-08-10 bash scripts/measure-rtf-rss.sh
```

## RTF (batch)

`polyvoice-bench` field `rt_factor_avg` is **RTFx** (= audio_secs / wall_secs).  
**RTF** = 1 / RTFx.

| Run | Files | Audio | Wall | **RTFx** | **RTF** | DER₀ micro |
|-----|------:|------:|-----:|--------:|--------:|-----------:|
| AMI-test | 16 | 9.06 h | 242 s | **134.8×** | **0.0074** | 24.5 % |
| AMI EN2002a only | 1 | 35.7 min | 14.7 s | **145.9×** | **0.0069** | 42.0 % |
| VoxConverse-test slice | 10 | 1.86 h | 60.9 s | **110.0×** | **0.0091** | 22.7 % |

Stage mix (AMI-16): segmentation ~67 %, embedding ~33 %, clustering negligible.

Compared to published **FP32** full-split tables (M1 Pro, ~53–68× / RTF 0.015–0.019), this INT8+CoreML run is **~2× faster**.

## Peak RSS (`/usr/bin/time -l`)

| File | Duration class | Peak RSS |
|------|----------------|----------|
| AMI EN2002a Mix-Headset | ~36 min | **749 MiB** |
| VoxConverse `aepyx.wav` | short | **489 MiB** |

RSS is **process peak** including ORT/CoreML, model weights, and working buffers — not on-disk model size (~8.4 MB).

## Caveats

1. DER collar 0 only in this pass (not 0.25).
2. Single-file AMI DER (42 %) is not the full-split AMI figure (24.5 %).
3. Full VoxConverse-test 232 not re-run here (slice of 10).
4. EP is CoreML, not CPU — for CPU-only RTF set `--execution-provider cpu`.
