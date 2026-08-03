# Speakrs H2H — smoke complete (10-file VoxConverse-test)

**Status:** complete  
**Collected:** 2026-08-03T15:08Z  
**Host:** Apple M1 Pro, Darwin arm64  
**polyvoice:** `f2bf57f` base + harness branch (0.14.0), CLI default **v2 + VBx**, balanced fp32, ORT CPU  
**speakrs:** local clone 0.5.0, modes `cpu` and `coreml`  
**Scorer:** `benchmarks/der.py` (NIST 10 ms, Hungarian, overlap scored)  
**Artifact:** `smoke-10file.json`

> **Subset caveat:** first 10 alphabetical VoxConverse-test files (includes hard
> `aorju`). This is **not** the full-232 release baseline. Numbers are for
> peer comparison under one scorer, not README headline claims.

## Primary table (10 files, same audio + same scorer)

| Engine | DER collar 0 micro | DER 0.25 micro | miss | FA | conf | spk exact / ±1 / ≥2 | RTF (cold CLI) | RTFx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **speakrs-coreml** | **13.74** | **7.83** | 3.29 | 2.28 | **2.25** | 5 / 0 / 5 | 0.044 | 22.7× |
| **speakrs-cpu** | **14.06** | **8.25** | 3.36 | 2.34 | **2.54** | 5 / 1 / 4 | 0.925 | 1.1× |
| polyvoice (v2+VBx) | 17.72 | 11.02 | 3.04 | 2.21 | **5.77** | 3 / 2 / 5 | **0.019** | **52.5×** |

95% bootstrap CI (no-collar micro): polyvoice [10.6–27.1], speakrs-cpu [9.5–19.7], speakrs-coreml [8.9–19.5].

## Interpretation

1. **Accuracy gap is real on our scorer.** speakrs wins by **~3.7–4.0 pp** no-collar
   micro on this 10-file smoke (17.72 → 13.74/14.06).
2. **Almost all of the gap is confusion** (+3.2–3.5 pp). Miss and FA are
   comparable. Same residual class as full-split analysis (speaker count /
   clustering), not segmentation miss.
3. **Speed story flips by backend.**
   - polyvoice cold CLI: **~53×** realtime (already strong).
   - speakrs CoreML: **~23×** here (cold per-file; their published 500×+ is
     warm M4 Pro, different protocol).
   - speakrs CPU (this openblas-static build): **~1.1×** — not a fair production
     CPU path for them on this machine; treat as accuracy-only row.
4. CoreML vs CPU for speakrs: **accuracy nearly tied** (13.74 vs 14.06); use
   CoreML for speed, CPU only for EP-fairness experiments.

## Earlier 4-file matched slice (aepyx, aggyz, aiqwk, aorju)

| Engine | DER₀ micro | conf |
|---|---:|---:|
| speakrs-cpu | 10.15 | 4.06 |
| polyvoice | 13.30 | 7.31 |

Same pattern: ~3 pp gap, confusion-dominated.

## Decision (smoke-level)

| Question | Answer |
|---|---|
| Is speakrs README accuracy marketing? | **No** — they win on our harness |
| Should polyvoice chase RTF vs speakrs CoreML? | **No** — already faster cold CLI on M1 Pro |
| What to fix next? | **Confusion / speaker count** (VBx, PLDA, prune, embeddings) |
| Full-232 needed? | **Yes** before README competitor row as “measured” — smoke is directional |

## Next steps

1. Full VoxConverse-test 232 with `polyvoice` + `speakrs-coreml` (CPU optional).
2. AMI-test 16 matched run.
3. Accuracy sprint targeting confusion, not miss.
4. After full-232: update `docs/BENCHMARKS.md` + `docs/COMPETITORS.md` with
   **measured** speakrs row (collar 0, this scorer, this host).

## Reproduce

```bash
cd ~/src/polyvoice-h2h   # or cherry-pick harness into Documents worktree
export SPEAKRS_RTTM_BIN=$PWD/benchmarks/tools/speakrs-rttm/target/release/speakrs-rttm
# build helper once: cargo build --release --manifest-path benchmarks/tools/speakrs-rttm/Cargo.toml --features coreml
./benchmarks/run_speakrs_h2h.sh 10
```

## Working tree

Agent could not write `~/Documents/personal/polyvoice` (macOS TCC). Branch:

```text
~/src/polyvoice-h2h  feat/speakrs-h2h-harness
```

Merge:

```bash
cd ~/Documents/personal/polyvoice
git fetch /Users/ekhodzitsky/src/polyvoice-h2h feat/speakrs-h2h-harness
git merge FETCH_HEAD   # or cherry-pick
```
