# Speakrs H2H — interim summary

**Status:** smoke in progress (10-file VoxConverse-test subset)  
**Host:** Apple M1 Pro
Darwin 25.1.0 arm64  
**polyvoice:** `f2bf57f` (0.14.0)  
**speakrs:** `b0756b1` (0.5.0)  
**Scorer:** `benchmarks/der.py` (collar 0 primary; overlap scored)

## Matched 4-file slice (aepyx, aggyz, aiqwk, aorju)

| Engine | DER collar 0 micro | collar 0.25 | miss | FA | conf |
|---|---:|---:|---:|---:|---:|
| **speakrs-cpu** | **10.15** | **6.40** | 1.80 | 4.29 | **4.06** |
| polyvoice (v2+VBx default) | 13.30 | 9.31 | 1.70 | 4.29 | **7.31** |

**Gap:** ~**3.2 pp** no-collar micro on this 4-file set. Dominant delta is **confusion**
(+3.25 pp), not miss/FA — consistent with speaker-count / clustering residual on
polyvoice.

## polyvoice alone — full 10-file smoke

| Metric | Value |
|---|---:|
| DER collar 0 micro / macro | 17.72 / 17.94 |
| DER collar 0.25 micro / macro | 11.02 / 11.90 |
| spk exact / ±1 / off≥2 | 3 / 2 / 5 |

(10-file subset is harder than full-test average; not a release baseline.)

## Timing notes (harness = cold process per file)

- speakrs CPU on long files (aorju ~20 min audio) is slow on M1 Pro in this build
  (openblas-static path) — tens of minutes wall per long file. CoreML not yet
  scored in this interim dump.
- polyvoice 10-file cold CLI completed first; RTF not yet locked in SUMMARY.

## Artifacts

- RTTMs: `benchmarks/results_full/voxconverse_test/{polyvoice,speakrs-cpu}/`
- Protocol: `PROTOCOL.md`
- Final JSON (when smoke finishes): `smoke-10file.json`

## Provisional verdict

speakrs **accuracy claim is directionally real** on a matched scorer/protocol:
lower confusion than polyvoice default on the same hard files. Not yet full-232.
Next: finish 10-file + coreml; then full test when budget allows.

## Working tree note

Agent could not write to `~/Documents/personal/polyvoice` (macOS TCC). All work
is on:

```text
~/src/polyvoice-h2h          # branch feat/speakrs-h2h-harness @ 902e150
~/src/speakrs                # speakrs 0.5 path dependency
```

Merge into the Documents worktree:

```bash
cd ~/Documents/personal/polyvoice
git fetch /Users/ekhodzitsky/src/polyvoice-h2h feat/speakrs-h2h-harness
git cherry-pick 902e150
# or: git merge --ff-only FETCH_HEAD after fetch
```

Resume / finish smoke (after build):

```bash
cd ~/src/polyvoice-h2h
export SPEAKRS_RTTM_BIN=$PWD/benchmarks/tools/speakrs-rttm/target/release/speakrs-rttm
# continue with cache (already-done RTTMs reused unless --no-cache)
./benchmarks/run_speakrs_h2h.sh 10
```
