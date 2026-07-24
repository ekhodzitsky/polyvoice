# cVBx clustering upgrades — ablation notes

Date: 2026-07-24  
Branch work: clustering-layer port of arXiv:2510.19572 ideas (no DiariZen weights).

## Environment

- Full VoxConverse-test / AMI DER re-measurement **not run** in this worktree:
  ONNX model weights are present only as `.minisig` signatures under `models/`;
  PLDA params for VBx live outside the worktree (`~/Documents/personal/polyvoice/data/vbx-plda/`)
  and were not wired into a DER bench run here.
- Unit / property coverage is green for AHC, clusterer (incl. VBx math), resegmentation,
  aggregator permutation, and the global-hyperparam CI guard.

## Items landed (algorithm + config)

| # | Item | Default | How to enable / disable |
|---|------|---------|-------------------------|
| 1 | Short-segment embedding filter | **1.6 s** on `VbxClusterer::from_dir` | `POLYVOICE_VBX_MIN_EMB_SECS` (set `0` to off); pipeline passes durations via `cluster_with_durations` |
| 2 | GMM-VBx (`loop_prob = 0`) | HMM `0.9` for contiguous; **auto GMM** when `embed_window_secs` is set | `POLYVOICE_VBX_LOOP_PROB`; `VbxConfig::gmm()` / `auto_gmm_for_windowed` |
| 3 | Hungarian local→global + inactive-speaker audit | **on** in pipeline_v2 | `POLYVOICE_V2_MAJORITY_LOCAL_MAP` → majority ablation; aggregator uses active-only cost matrix |
| 4 | cAHC-ASC stop | **off** (seed AHC); API ready | `POLYVOICE_VBX_AHC_ASC_MEMBERS` or `AscStop::{MinMembers,MinSecs}` |
| 5 | Gap-fill Δ=0.5 s | **on** (`PipelineConfig.max_gap_secs = 0.5`) | already shipped via `merge_segments` |
| 6 | One global hyperparam set | CI guard test | `tests/global_hyperparam_guard.rs` |

## Expected DER impact (from paper; not re-measured here)

Paper (fixed EEND-VC front-end): ~−1.5 macro-DER; VoxConverse no-collar 9.1 → 8.8; MSCE drop from short-segment filter is the largest speaker-counting lever.

Polyvoice front-end differs (powerset ONNX + ResNet34/CAM++), so absolute numbers will not match; re-run:

```bash
# After models + PLDA are available:
export POLYVOICE_VBX_PLDA_DIR=/path/to/vbx-plda
POLYVOICE_DER_EVAL=1 cargo run --features cli --bin polyvoice-bench -- \
  --dataset voxconverse-test --collar 0 --clusterer vbx
POLYVOICE_DER_EVAL=1 cargo run --features cli --bin polyvoice-bench -- \
  --dataset ami-test-single --collar 0 --clusterer vbx
```

Ablate per item by toggling the env vars above (one change at a time) and recording DER / miss / FA / conf / speaker-count error.

## Deferred

- Full-corpus DER baseline update (`tests/der_baseline.json`) — needs models + data.
- Random-init VBx for >30 min recordings.
- OSD closest-in-time second-speaker in overlap (ASoBO).
- CLI default clusterer flip (explicitly out of scope).
