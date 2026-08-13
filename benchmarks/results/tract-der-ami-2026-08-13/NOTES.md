# Tract vs ort DER — AMI-test full 16 (2026-08-13)

## Protocol

| Item | Value |
|------|--------|
| Host | Apple M1 Pro, Darwin 25.1, arm64 |
| Script | `scripts/tract-der-gate.sh` |
| Dataset | `data/ami-test` (16 Mix-Headset, ≈ 9.06 h audio) |
| Pipeline | v2 + VBx, profile balanced, EP=cpu, collar 0.0 |
| Ort | product INT8 (`powerset_int8` + `resnet34_int8`) |
| Tract | `POLYVOICE_INFERENCE_BACKEND=tract` → signed `powerset_fp32_tract` + FP32 `wespeaker_resnet34` |
| Binary | `polyvoice-bench` release, features `cli,backend-tract` |

## Results

| Backend | files | DER₀ micro % | RTFx | seg_s | emb_s | wall (approx) |
|---------|-------|--------------|------|-------|-------|---------------|
| **ort** | 16 | **24.63** | **153.5** | 130 | 83 | ~4 min |
| **tract** | 16 | **23.42** | **18.8** | 1091 | 644 | ~29 min |

| Metric | Value |
|--------|--------|
| **ΔDER₀** | **−1.21 pp** (tract lower DER on this host/split) |
| **RTFx ratio** tract/ort | **0.123** (~**8.2×** slower) |

JSON: `ort.json`, `tract.json`, `summary.json`. Logs: `ort.log`, `tract.log`.

Reference product Linux/CPU AMI-16 (ort INT8): DER₀ **24.19%** (`ami_test_linux_cpu` baseline) — this host’s ort run is within ~0.4 pp.

## Reproduce

```bash
cargo build --release --features "cli,backend-tract" --bin polyvoice-bench
OUT=benchmarks/results/tract-der-ami-2026-08-13 \
  DATASET=data/ami-test \
  bash scripts/tract-der-gate.sh
```

## Interpretation

1. On **full AMI-test**, opt-in pure-Rust tract matches or slightly beats ort INT8 DER
   (−1.2 pp). Not claimed as a systematic accuracy win (FP32 powerset/embedder
   vs product INT8 pair; host noise).
2. Speed remains ~**8×** below ort but multi-realtime (~19× RTFx) on M1 Pro CPU.
3. Still **not** product default: larger download (FP32 models), MSRV for tract,
   no Linux/CPU full-split tract gate yet.
4. Vox-232 full-split under tract remains optional (~4 h wall at this ratio).

## Prior smokes

- Vox-3 / Vox-10 short subsets: `../powerset-tract-rtf-der-2026-08-12/`
