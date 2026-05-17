# Benchmarks

This directory contains reproducible benchmark artifacts for polyvoice.

## Dataset Layout

Every dataset must follow this directory structure:

```
data/<dataset-name>/
├── audio/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── rttm/
    ├── file1.rttm
    ├── file2.rttm
    └── ...
```

- `audio/`: 16 kHz mono WAV files. File stem must match the corresponding RTTM file.
- `rttm`: Reference speaker turns in [RTTM format](https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format.txt).

## Running Benchmarks

### Prerequisites

Download models for the profile you want to benchmark:

```bash
cargo run --release --features cli --bin polyvoice -- download-models --profile balanced
```

### Full Dataset

```bash
cargo run --release --features cli --bin polyvoice-bench -- \
  data/voxconverse-test \
  --profile balanced \
  --collar 0.25 \
  --output benchmarks/results/voxconverse-test-$(date +%Y%m%d).json
```

### Subset (smoke test)

```bash
cargo run --release --features cli --bin polyvoice-bench -- \
  data/ami-test-single \
  --profile balanced \
  --max-files 1 \
  --output benchmarks/results/ami-smoke.json
```

## Result Schema

Benchmark artifacts are JSON files produced by `polyvoice-bench`. The current schema is `polyvoice-bench-v0.6`.

Key fields:

| Field | Description |
|---|---|
| `schema` | Schema version (`polyvoice-bench-v0.6`) |
| `crate_version` | polyvoice crate version |
| `git_sha` | Git commit SHA of the code that produced the result |
| `host_arch` / `host_os` | Target platform |
| `command_line` | Exact command used |
| `dataset_name` | Name of the dataset directory |
| `profile` | Model profile (`mobile` or `balanced`) |
| `files_processed` | Number of files successfully evaluated |
| `files_skipped` | Files without matching RTTM |
| `der_collar` | Average DER with forgiveness collar (%) |
| `der_no_collar` | Average DER without collar (%) |
| `miss` / `false_alarm` / `confusion` | Average decomposition (%) |
| `rt_factor_avg` | Average real-time factor |
| `speaker_count` | Exact / ±1 / off-by-2+ counts |
| `model_hashes` | SHA-256 of models used |
| `per_file` | Array of per-file metrics |

## Baselines

Baseline results are committed to `tests/der_baseline.json`. These numbers are used by regression tests to detect accuracy drift.

To update baselines after a deliberate algorithmic change:

1. Run the full benchmark suite on the target dataset.
2. Copy the relevant fields into `tests/der_baseline.json`.
3. Update `_status` and `_filled_by` metadata.
4. Commit the JSON with the code change.

## Available Datasets

| Dataset | Files | Purpose | Download Command |
|---|---|---|---|
| `ami-test-single` | 1 | Smoke / regression | `bash scripts/download-ami-test-single.sh` |
| `ami-test` | ~170 | Full AMI eval | `bash scripts/download-ami-test.sh` |
| `voxconverse-test` | 232 | Primary benchmark | `bash scripts/download-voxconverse-test.sh` |
| `voxconverse-dev` | 216 | Development / threshold tuning | `bash scripts/download-voxconverse-dev.sh` |
| `voxceleb1-subset` | 40 | Embedding sanity check | `bash scripts/download-voxceleb1-subset.sh` |

## Quality Gates

- Any algorithmic change must include a before/after benchmark artifact.
- Any default change must improve at least one target dataset without unacceptable regression on another.
- DER regression > 1.0% absolute on VoxConverse-test requires justification in the PR.
