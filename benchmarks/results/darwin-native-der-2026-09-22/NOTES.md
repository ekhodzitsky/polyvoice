# Darwin kernel full-split DER — v2+VBx INT8, product CLI

**Date:** 2026-09-22
**Host:** Apple M1 Pro (10 cores, 16 GB), Darwin arm64 (xnu 25.1.0)
**Build:** `--features cli` (kernels, no `libonnxruntime`), polyvoice 0.21.0,
git `5509435`.
**Protocol:** INT8 balanced (`powerset_int8` + `resnet34_int8`), pipeline
v2 + VBx, EP=`cpu`, powerset micro-batch N=8, collar request 0.25 s (the JSON
also carries no-collar micro/macro; the headline below is collar 0).
VBx AHC seed 0.6.
**Command:** `bash scripts/darwin-native-der-gate.sh`

First Darwin full-split re-run after the VBx AHC seed 0.6 retune; supersedes
the 0.18 Darwin numbers (Vox 15.47 % / AMI 25.19 %, ~130× / ~109× RTFx).

| Split | files | DER₀ micro | DER₀ macro | DER @0.25 micro | RTFx |
|---|---:|---:|---:|---:|---:|
| VoxConverse-test | 232 | **13.33 %** | 14.40 % | 8.55 % | **169×** |
| AMI-test | 16 | **23.61 %** | 22.69 % | 15.71 % | **200×** |

Delta vs the 0.18 Darwin row: Vox **−2.14 pp**, AMI **−1.58 pp**; RTFx
~130→~169× (Vox), ~109→~200× (AMI). Same protocol on Linux x86_64
(Ryzen AI 9 HX 370, 2026-09-13): Vox 13.34 % / AMI 24.19 %, ~162× / ~193×.

Vox-3 scoreboard re-checked on this host right after the run
(`POLYVOICE_SCOREBOARD_PERF=1 cargo test --release --features cli --test native_scoreboard`):
DER₀ micro 7.1088 / macro 7.3889 (floors 7.11 / 7.39), RTFx 124× (floor 117×),
peak RSS 454 MiB (floor 556 MiB), INT8 pair size floor — all hold.

Model SHA-256 hashes: `summary.json` → `model_hashes`.
Gate verdict: `gate-result.json` `ok=true` (files 232/16, EP resolved `Cpu`).
Auto-generated context: `NOTES.auto.md` — its title says "Linux CPU" because
the shared gate script writes it verbatim; this run is Darwin.
