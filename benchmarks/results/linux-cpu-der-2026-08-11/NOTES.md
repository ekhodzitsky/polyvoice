# Linux CPU DER gate (2026-08-11)

## Protocol

| Knob | Value |
|------|--------|
| Host | Linux aarch64, Ubuntu 24.04 (Docker on Apple Silicon / linuxkit), 10 CPUs |
| Profile | `balanced` → `powerset_int8` + `resnet34_int8` |
| Pipeline | v2 + VBx |
| EP | **cpu** (explicit; not auto/CoreML) |
| Powerset micro-batch | **N=8** (`POLYVOICE_POWERSET_BATCH_SIZE=8`) |
| Collar | request 0.25 (JSON also carries no-collar) |
| Crate | 0.17.0 @ `8013949` |
| Command | `DOCKER=1 bash scripts/linux-cpu-der-gate.sh` |

## Headline numbers

| Split | files | DER₀ micro | DER₀.₂₅ micro | RTFx |
|-------|------:|----------:|-------------:|-----:|
| VoxConverse-test | 232 | **14.94 %** | **10.27 %** | **81.5×** |
| AMI-test Mix-Headset | 16 | **24.19 %** | **16.60 %** | **94.9×** |

Miss / FA / conf in the per-split JSON are for the **0.25 s collar** pass.

## vs other product paths (same models)

| Path | Vox DER₀ | AMI DER₀ | Notes |
|------|--------:|--------:|-------|
| **Linux CPU N=8 (this gate)** | **14.94** | **24.19** | non-Apple product truth |
| Mac CoreML N=1 (published gate) | 15.02 | 24.50 | `int8-full-der-2026-08-10/` |
| Mac CPU N=8 | 14.64 | 24.63 | `int8-batch8-default-2026-08-10/` |

Linux CPU is **within noise of Mac** on DER and is the gate to cite for server/Linux deploys.

## Reproduce

```bash
# full splits under data/voxconverse-test (232) and data/ami-test (16)
DOCKER=1 bash scripts/linux-cpu-der-gate.sh
# or native Linux host:
bash scripts/linux-cpu-der-gate.sh
```

Smoke (subset):

```bash
MAX_VOX=10 MAX_AMI=4 DOCKER=1 bash scripts/linux-cpu-der-gate.sh
```

CI: `.github/workflows/linux-cpu-der.yml` (workflow_dispatch + weekly).
