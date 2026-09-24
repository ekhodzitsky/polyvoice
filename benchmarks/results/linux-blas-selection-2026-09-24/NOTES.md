# Explicit Linux OpenBLAS selection: before/after evidence

Host: AMD Ryzen AI 9 HX 370, Linux x86_64, 24 logical CPUs, performance
CPU governor. Date: 2026-09-24. Release profile, balanced INT8, v2 + VBx,
CPU, powerset batch 8, collar 0, overlap included, checked-in PLDA fixtures.

## Revisions and artifacts

- `baseline`: master `a8461d42f7793f2f21409b41f7619e881b4c5a43`, `--features cli`.
  Its auto-detection found no BLAS; the saved binary has no BLAS imports.
- `rust`: `64f13c8433c684b6ff73f8cc5e5f5597baa278ec`, `--features cli`.
  Default Rust backend even with OpenBLAS available to pkg-config.
- `openblas`: same implementation revision, `--features cli,system-openblas`.
  Ubuntu LP64 pthread OpenBLAS 0.3.32, extracted locally with its Fortran
  runtime dependencies; no system installation changes. pkg-config points
  to that prefix and the runtime loader path points to its shared libraries.

[Measurements](measurements.json) include binary SHA-256, source revision,
DER, wall RTFx and process peak RSS for every completed run. Source revision
is recorded explicitly: the bench program itself queries the current checkout
at runtime, which cannot identify a saved baseline executable.

Model pair unchanged, **8,414,314 bytes**:

- powerset_int8: `175896d26f639933cd86906d2dd3e6796eddb23c1f719925a3949052da76183b`
- resnet34_int8: `24b58559fefb2af624a5d371c43ebae891a9a8ca363b2f9e7c31fd8e440a36b3`

## Scope and results

Vox-3 is euqef/fuzfh/msbyq. Vox-10 is the first ten filenames in sorted
VoxConverse-test order; AMI-1 is the first sorted AMI-test recording.
The exact filenames and all per-file quality fields are in
[quality.json](quality.json). These are fixed diagnostic subsets, not the
published full-split Vox/AMI reference scores.

Vox-3 runs alternate baseline/rust/openblas five times for each jobs setting.
Vox-10 has one run per backend; AMI-1 repeats baseline/rust three times and
OpenBLAS once. Table timing/RSS values are medians, **not release floors**.

| Dataset | Jobs | Backend | Runs | DER₀ micro % | Wall RTFx | Peak RSS MiB |
|---------|------|---------|------|--------------|-----------|--------------|
| vox3 | 1 | baseline | 5 | 7.0295 | 91.7 | 316.5 |
| vox3 | 1 | rust | 5 | 7.0295 | 89.4 | 314.8 |
| vox3 | 1 | openblas | 5 | 7.0295 | 93.0 | 321.2 |
| vox3 | 3 | baseline | 5 | 7.0295 | 117.0 | 469.5 |
| vox3 | 3 | rust | 5 | 7.0295 | 127.7 | 480.4 |
| vox3 | 3 | openblas | 5 | 7.0295 | 132.3 | 480.5 |
| vox10 | 3 | baseline | 1 | 16.4243 | 134.8 | 2283.0 |
| vox10 | 3 | rust | 1 | 16.4243 | 95.6 | 2224.9 |
| vox10 | 3 | openblas | 1 | 15.6938 | 165.1 | 2242.2 |
| ami1 | 3 | baseline | 3 | 32.5584 | 144.7 | 1263.9 |
| ami1 | 3 | rust | 3 | 32.5584 | 150.0 | 1292.6 |
| ami1 | 3 | openblas | 1 | 31.7272 | 163.5 | 1299.3 |

Default Rust matches the baseline on every recorded per-file quality field,
including speaker counts and turn counts. OpenBLAS matches Vox-3 quality but
changes clustering on Vox-10/AMI-1; its lower DER here is subset evidence,
not a universal improvement or numerical-equivalence claim.

**Performance limitation:** unrelated builds and CPU-heavy tests were active
on the host. Throughput fluctuated substantially, including between repeated
runs of the same binary. Full-corpus comparisons were stopped after detecting
that contention; no incomplete full-corpus result is included. These runs
measure before/after behavior but do not certify isolated performance parity
or establish new speed/RSS baselines. Default median AMI RSS is slightly
higher than baseline, with overlapping observed ranges; no memory improvement
is claimed. Re-run on an idle reference host for performance qualification.

Darwin implementation and all five locked scoreboard limits are unchanged.
Darwin scoreboard CI passed; its shared-runner result does not substitute for
an isolated Darwin RTF/RSS measurement. Linux subset RSS is not the Darwin
Vox-3 memory protocol.

## Reproduction and verification

Save a release `polyvoice-bench` from each revision/feature selection. Prepare
subset directories with `audio/` and `rttm/` symlinks for the recorded filenames.
For each binary and dataset:

```bash
POLYVOICE_VBX_PLDA_DIR="$PWD/fixtures/vbx-plda" POLYVOICE_POWERSET_BATCH_SIZE=8 \
  /usr/bin/time -v ./saved-polyvoice-bench DATASET \
  --profile balanced --pipeline v2 --clusterer vbx --execution-provider cpu \
  --collar 0 --jobs 3 --output result.json
```

For the OpenBLAS binary, install the runtime or set `LD_LIBRARY_PATH` to its
local prefix. Read process RSS from `/usr/bin/time -v`, not summed workers.

Validation:

- `scripts/check-linux-blas.sh`: passes for both artifacts with OpenBLAS
  available, including missing-prerequisite failure and `readelf`/`ldd` checks.
- Kernel tests: 58 Rust / 59 OpenBLAS passed; three manual tests ignored each.
- Product CLI/FFI/OpenBLAS library tests: 943 passed; local-only library: 875.
- Clippy: product CLI/FFI and all-features with `-D warnings` passed.
- Dependency invariants, standalone lockfiles, formatting and doc links passed.
