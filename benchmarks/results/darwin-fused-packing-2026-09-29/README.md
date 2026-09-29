# Direct Apple INT8 input packing

Direct row-to-zip packing removes the intermediate K-by-N copy for stride-one
INT8 convolution on the experimental Apple ARM64 Rust path. Four input tap
vectors are loaded from the padded row ring and interleaved directly into the
unchanged SDOT panel layout. Dot-product arithmetic, quantization, threads and
product defaults stay unchanged. Other backends retain the reference packer.
KN scratch remains available for fallbacks: this reduces copying, not allocation.

Candidate source: `d9852269857e701da0f98333f7cac1506c861193` (clean).
Baseline/product control: `0ed55f75df9fe2ba504c5e4c5ab1877de4e90f39` (clean),
using the previously verified release binaries. Intervening commits only add
research documentation/results. All builds use Rust 1.98.0; the host is a
10-core M1 Pro with 16 GiB RAM and Darwin 25.1.0.

## Numerical verification

The packing test preceded implementation and first failed the Apple cross-check
because the direct packer was absent. On M1 Pro it now matches both the original
packer and an independent scalar layout reference exactly. Cases cover channel
crossings, K padding, every row-ring position, four spatial borders, 16/32-pixel
panels, overlapping tails and zero points -128, -17, 0 and 127. Output guards
remain untouched.

A separate convolution test compares float output bits with the test-only
reference path, including odd channels, partial output-channel groups, two-image
batches and widths around panel boundaries. These are exact comparisons, not
floating-point tolerances. Existing numerical suites also pass on Apple and Linux.

## Representative convolution timings

The ignored `bench_direct_packing_shapes` test compares both packers in the same
optimized binary. Six alternating paired measurements per shape each include a
warm-up and five full convolutions. These synthetic shapes measure layer cost,
not DER or full-pipeline qualification.

| IC / OC / H / W | Reference median ms | Direct median ms | Time reduction |
|---|---:|---:|---:|
| 32 / 32 / 80 / 400 | 3.277200 | 2.617642 | 20.1% |
| 64 / 64 / 40 / 200 | 2.481425 | 2.138363 | 13.8% |
| 128 / 128 / 20 / 100 | 2.142571 | 1.965675 | 8.3% |
| 256 / 256 / 10 / 50 | 2.179984 | 2.062121 | 5.4% |

Every direct measurement beats its paired reference. `shapes.log` retains all
values; `provenance.json` records the command and repetition protocol.

## Full-pipeline comparison

Protocol: hash-verified committed Vox-3, balanced, v2 + VBx, CPU, collar zero,
overlap scored, jobs=1; identical INT8 models and six PLDA files. One retained
warm-up per variant precedes six rounds using every permutation of product /
Rust-before / Rust-after once. Five seconds separate processes. Each binary runs
from its own clean checkout so the reported source matches its build.

| Round | Product RTFx | Rust before RTFx | Rust after RTFx | After RSS MiB |
|---|---:|---:|---:|---:|
| Warm-up | 166.287 | 114.033 | 121.048 | 442.656 |
| 1 | 162.119 | 120.424 | 129.525 | 456.578 |
| 2 | 153.801 | 113.162 | 130.902 | 444.719 |
| 3 | 161.643 | 112.072 | 124.381 | 444.188 |
| 4 | 164.411 | 118.352 | 120.687 | 444.922 |
| 5 | 158.243 | 114.072 | 128.413 | 437.766 |
| 6 | 161.241 | 116.294 | 119.718 | 441.094 |

Rust median RTFx increases from 115.183 to 126.397 (9.7%); median embedding time
falls from 0.640448 s to 0.554276 s. The candidate beats the baseline in all six
paired rounds. Each run is independently checked against all locked floors;
medians summarize the series and do not substitute for individual verdicts.

- Candidate and baseline DER micro/macro match in every run:
  7.006802721% / 7.329480110% (limits 7.11% / 7.39%). Per-file quality,
  speaker counts and turn counts also match.
- Candidate RTFx is 119.718–130.902, above 117 in every measured run; its
  warm-up also passes. The old Rust path misses in four rounds and warm-up.
  Product controls pass every floor throughout.
- Candidate peak RSS is at most 456.578125 MiB, below 556 MiB.
- Every variant uses the same 8,414,314-byte INT8 pair.

The user authorized measurement with the current background load. Initial CPU
idle samples were 77.88% and 70.96%; host state is retained. All runs use AC power,
low-power mode is off, and no thermal warning was reported. No compilation,
profiling or artifact transfer overlapped the series. Background variation limits
conclusions near the speed floor. This supports retaining the optimization in
the experimental backend, not quiet-host certification or replacing Accelerate.
Held-out full-corpus release qualification remains separate.

## Reproduction and evidence

`provenance.json` records source/binary/model hashes, compiler, build commands,
run order, exact commands, host state, metrics and individual verdicts. All 21
raw reports and matching resource logs are retained. RSS is maximum resident set
size from `/usr/bin/time -l`, converted from bytes to MiB. Existing quality helpers
validate coverage, protocol, source/model identity and metrics.

From the candidate source on Apple ARM64:

```bash
cargo test --locked --release -p polyvoice-kernels \
  --features experimental-darwin-rust --lib fused_pack_tests -- --test-threads=1
cargo test --locked --release -p polyvoice-kernels \
  --features experimental-darwin-rust --lib bench_direct_packing_shapes \
  -- --ignored --nocapture --test-threads=1
```

`collect.py` uses Python 3.11+ and the existing quality helpers. From the clean
candidate checkout, prepare the three release binaries at its recorded paths,
verify the default model cache, then run `caffeinate -i python3 /path/to/collect.py`.
It refuses to overwrite prior evidence. Original checkouts/builds remain on the
Mac. This change adds no project dependency and does not promote the backend.
