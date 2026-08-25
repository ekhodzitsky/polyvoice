# Linux CPU DER gate (auto)

**Date:** 2026-08-17
**Protocol:** INT8 balanced, pipeline v2 + VBx, EP=`cpu`, powerset micro-batch N=`8`.
**Assert baseline:** 0 (`/work/tests/der_baseline.json`)
**Command:** `bash scripts/linux-cpu-der-gate.sh`

## Headline (no-collar micro)

See `summary.json` / per-split JSON. **miss/FA/conf** in reports are for the
**requested collar (0.25 s)**, not collar 0.

## Reproduce

```bash
DOCKER=1 bash scripts/linux-cpu-der-gate.sh
# smoke:
MAX_VOX=10 MAX_AMI=16 bash scripts/linux-cpu-der-gate.sh
```

Hand-written context (if present): `NOTES.md`.
