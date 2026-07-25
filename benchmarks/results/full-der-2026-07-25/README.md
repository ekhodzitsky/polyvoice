# Full DER gate: legacy vs v2+VBx (2026-07-25)

Gate for promoting pipeline v2 + VBx to the CLI/Python default.

## Protocol

| Item | Value |
|------|--------|
| Collar | 0.25 (CLI flag) — report also includes **no-collar** micro/macro |
| Overlap | scored |
| Mapping | Hungarian |
| Profile | balanced |
| EP | CPU |
| Hardware | Apple M1 Pro (record in JSON via wall RTF) |
| PLDA | `data/vbx-plda` via `POLYVOICE_VBX_PLDA_DIR` |
| Legacy knobs | threshold 0.45, `min_cluster_size=2` (matches `tests/der_baseline.json`) |
| V2 knobs | `--pipeline v2 --clusterer vbx` (cVBx defaults in code) |

## Commands

```bash
export POLYVOICE_VBX_PLDA_DIR="$PWD/data/vbx-plda"
BIN=./target/release/polyvoice-bench   # features: cli,vbx,spectral

$BIN data/voxconverse-test --pipeline legacy --profile balanced --collar 0.25 \
  --min-cluster-size 2 --execution-provider cpu \
  --output benchmarks/results/full-der-2026-07-25/legacy-voxconverse-test-232.json

$BIN data/voxconverse-test --pipeline v2 --clusterer vbx --profile balanced --collar 0.25 \
  --execution-provider cpu \
  --output benchmarks/results/full-der-2026-07-25/v2-vbx-voxconverse-test-232.json

$BIN data/ami-test --pipeline legacy --profile balanced --collar 0.25 \
  --min-cluster-size 2 --execution-provider cpu \
  --output benchmarks/results/full-der-2026-07-25/legacy-ami-test-16.json

$BIN data/ami-test --pipeline v2 --clusterer vbx --profile balanced --collar 0.25 \
  --execution-provider cpu \
  --output benchmarks/results/full-der-2026-07-25/v2-vbx-ami-test-16.json
```

## Gate (hard)

Flip v2 to default **only if**:

1. VoxConverse-test **no-collar micro DER** (v2+VBx) ≤ legacy  
2. AMI-test **no-collar micro DER** (v2+VBx) ≤ legacy  

Otherwise: document gaps, no flip.

## Status / speed protocol

Legacy Vox 232 finished on **CPU** (`full-run.log`, `legacy-voxconverse-test-232.json`).

Remainder restarted as a **fast suite** (`run_fast.sh`, `fast-run.log`):

- **EP:** CoreML (DER matches CPU smoke; RTF only differs)
- **Parallelism:** 4 file shards × `polyvoice-bench`, then `merge_shard_reports.py`
- Then AMI legacy + AMI v2+VBx (CoreML, serial)

See `VERDICT.md` when `write_verdict.sh` finishes.
