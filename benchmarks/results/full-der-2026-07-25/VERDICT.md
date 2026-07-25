# Full DER gate verdict — legacy vs v2+VBx

**Date:** 2026-07-25  
**Git:** `143d47fb46a4201c8d93a5721824ddeb17d5e988`  
**Host:** macos aarch64 cpus=10  
**Protocol:** collar 0.25 scored + no-collar; overlap scored; Hungarian; profile balanced; legacy Vox on CPU, remainder CoreML (DER EP-stable); PLDA `data/vbx-plda`  
**Legacy knobs:** min_cluster_size=2  
**V2 knobs:** `--pipeline v2 --clusterer vbx`

## Decision

**GO — promote pipeline v2 + VBx to default**

Gate rule (hard): flip only if v2+VBx **no-collar micro DER ≤ legacy** on both VoxConverse-test and AMI-test.

| Dataset | legacy no-collar micro | v2+VBx no-collar micro | Δ (v2 − legacy) | Pass |
|---------|------------------------:|------------------------:|----------------:|:----:|
| VoxConverse-test (232) | 18.538% | 15.366% | -3.173 pp | PASS |
| AMI-test (16) | 32.868% | 25.167% | -7.701 pp | PASS |

**Flip default?** `yes`

## Full metrics

### VoxConverse-test (232 files)

| Metric | legacy | v2+VBx |
|--------|-------:|-------:|
| DER no-collar micro | 18.538% | 15.366% |
| DER collar 0.25 micro | 12.909% | 11.123% |
| Miss | 4.486% | 2.291% |
| FA | 3.186% | 1.862% |
| Confusion | 4.986% | 6.970% |
| RT factor (avg) | 9.862× | 8.136× |

### AMI-test (16 files)

| Metric | legacy | v2+VBx |
|--------|-------:|-------:|
| DER no-collar micro | 32.868% | 25.167% |
| DER collar 0.25 micro | 25.197% | 17.656% |
| Miss | 17.092% | 7.690% |
| FA | 2.443% | 1.701% |
| Confusion | 5.215% | 7.467% |
| RT factor (avg) | 10.185× | 8.926× |

## Artifacts

- `legacy-voxconverse-test-232.json`
- `v2-vbx-voxconverse-test-232.json`
- `legacy-ami-test-16.json`
- `v2-vbx-ami-test-16.json`
- `full-run.log`
- `smoke-legacy.json` / `smoke-v2-vbx.json` (1-file smoke)

## Next steps

1. Flip CLI/Python defaults to pipeline v2 + VBx (separate PR).
2. Refresh `tests/der_baseline.json` from these aggregates.
3. Update `docs/BENCHMARKS.md` and production-readiness notes.
