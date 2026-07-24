# Worktree status — cVBx clustering upgrades

**Branch:** `feat/task-401-cvbx`  
**Base:** master @ 8049537  
**Date:** 2026-07-24

## Landed

1. **Short-segment embedding filter** (`src/clusterer/short_filter.rs`)
   - Partition by duration; reassign shorts by cosine / PLDA features.
   - `VbxClusterer::from_dir` default **1.6 s** (`POLYVOICE_VBX_MIN_EMB_SECS`).
   - Pipeline/hybrid pass durations via `Clusterer::cluster_with_durations`.

2. **GMM-VBx** (`loop_prob = 0`)
   - Explicit `VbxConfig::gmm()` / `hmm()` / `is_gmm()`.
   - Auto GMM when dense `embed_window_secs` is set (`auto_gmm_for_windowed`).
   - Env override still wins: `POLYVOICE_VBX_LOOP_PROB`.

3. **Hungarian local→global + inactive-speaker audit**
   - `src/clusterer/assign.rs`: active-only co-occurrence Hungarian + cannot-link.
   - Pipeline `map_local_to_global` uses it (majority ablation via `POLYVOICE_V2_MAJORITY_LOCAL_MAP`).
   - Aggregator window permutation: cost matrix over **active** speakers only.
   - Regression tests: inventing inactive locals fails the good path; buggy full-matrix helper documents the old behaviour.

4. **cAHC-ASC** (`src/ahc/mod.rs`)
   - `AscStop::{Off, MinMembers, MinSecs}` + `agglomerative_cluster_asc`.
   - Wired into VBx AHC seed via `POLYVOICE_VBX_AHC_ASC_MEMBERS` / `with_ahc_established_min_members`.

5. **Gap-fill Δ=0.5 s** — already default `PipelineConfig.max_gap_secs = 0.5`; documented as cVBx gap-fill.

6. **Global hyperparam CI guard** — `tests/global_hyperparam_guard.rs` scans clustering config surfaces for per-dataset branches.

## Deferred

- Full VoxConverse / AMI DER numbers and `tests/der_baseline.json` update (models/data not runnable end-to-end in this worktree; only `.minisig` under `models/`).
- Random-init VBx for >30 min.
- OSD closest-in-time second-speaker.
- CLI default clusterer flip (out of scope).

See `benchmarks/results/cvbx-ablation-notes.md` for enable/disable knobs and intended bench commands.

## How to verify

```bash
cargo test --lib --features "onnx,segmentation,embedder,clusterer,resegmentation,vbx,spectral,download"
cargo test --test global_hyperparam_guard
cargo clippy --lib --features "onnx,segmentation,embedder,clusterer,resegmentation,vbx,spectral,download" -- -D warnings
```

Targeted filters: `ahc::tests::cahc_asc`, `clusterer::short_filter`, `clusterer::assign`, `clusterer::vbx`, `window_perm_ignores_inactive`.

## Remaining work for gate 117

- Run DER on VoxConverse-test (collar 0 and 0.25) and AMI-4 with VBx + cVBx defaults.
- Update baselines only if both no-collar and collar improve (or AMI no worse).
- Only then consider pipeline_v2 / VBx as default (117).
