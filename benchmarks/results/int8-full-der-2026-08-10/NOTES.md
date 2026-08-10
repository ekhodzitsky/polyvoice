# INT8 full-split DER gate (2026-08-10)

Full VoxConverse-test (232) + AMI-test Mix-Headset (16) remeasure on the
**shipping INT8 pair** after 0.17.0 made every profile resolve
`powerset_int8` + `resnet34_int8`.

## Config

| | |
|---|---|
| Crate | 0.17.0 |
| Profile | `balanced` (same models as `mobile` / `fast`) |
| Pipeline | v2 + VBx |
| Models | `powerset_int8`, `resnet34_int8` |
| EP | CoreML (auto) |
| Host | Apple M1 Pro (10 cores) |
| Date | 2026-08-10 |

## Headline micro DER

| Split | collar 0 | collar 0.25 | RTFx (collar 0 / 0.25) |
|---|---:|---:|---:|
| VoxConverse-test (232) | **15.02 %** | **10.33 %** | 110.6× / 122.1× |
| AMI-test (16) | **24.50 %** | **16.82 %** | 130.5× / 108.9× |

Collar-0 decomposition (micro):

| Split | miss | FA | conf | spk exact / ±1 / off-2+ |
|---|---:|---:|---:|---|
| Vox | 3.54 | 3.95 | 7.62 | 83 / 62 / 87 |
| AMI | 11.17 | 3.62 | 8.80 | 1 / 3 / 12 |

## vs previous published (FP32 hop-2.0, 2026-07-30/31)

| Split | metric | FP32 published | INT8 (this run) | Δ |
|---|---|---:|---:|---:|
| Vox collar 0 | micro | 15.24 | 15.02 | −0.22 pp |
| Vox collar 0.25 | micro | 10.52 | 10.33 | −0.19 pp |
| AMI collar 0 | micro | 23.42 | 24.50 | **+1.08 pp** |
| AMI collar 0.25 | micro | 15.71 | 16.82 | **+1.11 pp** |

Vox is at parity (slightly better). AMI pays ~+1 pp for the INT8 default —
within the previously stated “~+2 pp on AMI-style” caveat, not a surprise.

## Artifacts

- `summary.json` — rollup
- `voxconverse-test-232-collar{0,025}.json` (+ `.log`)
- `ami-test-16-collar{0,025}.json` (+ `.log`)

## Reproduce

```bash
cargo build --release --features cli --bin polyvoice-bench
OUT=benchmarks/results/int8-full-der-2026-08-10
target/release/polyvoice-bench data/voxconverse-test \
  --profile balanced --pipeline v2 --clusterer vbx --collar 0.0 \
  --output "$OUT/voxconverse-test-232-collar0.json"
# … collar 0.25, then AMI Mix-Headset 16 the same way
```

Published tables and `tests/der_baseline.json` were updated from this run.
