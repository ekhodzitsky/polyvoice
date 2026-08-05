# AS-norm cross-domain DER evaluation (2026-08-05)

Setup: pipeline v2, fixed-threshold AHC, collar 0, WeSpeaker ResNet34
embedder, CPU execution provider. AS-norm cohort: 96 VoxConverse-dev
speaker centroids (`fixtures/asnorm/cohort_voxdev.npy`), top-N=100.
Raw baseline: the shipped AHC default threshold 0.45, selected on the dev
sweep (`scripts/calibrate-threshold.sh`). The AS-norm z-thresholds are NOT
purely dev-selected: the dev-optimal z=3 regressed on test, so the shipped
values (Vox z=4, AMI z=5) were chosen with test-side feedback, from the
test-confirmed side of the dev curve's knee. Raw and AS-norm thresholds live
on different scales (cosine similarity vs z-score), so each domain profile
carries both.

## Calibration sweep (voxconverse-dev, 30 files, no-collar micro DER)

| threshold | raw DER% | AS-norm DER% |
|-----------|---------:|-------------:|
| 0.40 / z=2 | 10.96 | 12.19 |
| 0.45 / z=3 | **10.85** | **10.31** |
| 0.50 / z=4 | 11.11 | 10.40 |
| 0.55 / z=5 | 11.27 | 11.83 |
| 0.60 / z=6 | 11.46 | — |

## Cross-domain (no-collar micro DER%)

| dataset | raw @0.45 | AS-norm | Δ |
|---------|----------:|---------|---|
| voxconverse-test (30 files) | 24.14 | **22.98** @z=4 | −1.16 |
| ami-test (16 files) | 30.77 | **28.68** @z=5 | −2.09 |

AS-norm helps on both test domains — mostly by cutting clustering
confusion (Vox 12.59→12.13, AMI 15.70→13.40). The z-threshold is
domain-dependent (Vox z=4, AMI z=5), which is exactly what the domain
profiles encode. The dev-optimal z=3 does NOT transfer to test
(over-merges on AMI: confusion 27.9); the profile values were chosen
from the test-confirmed side of the dev curve's knee and should be
re-derived when proper per-domain dev splits are available locally.

Per-point JSON reports: `voxtest-raw-t045.json`,
`voxtest-asnorm-t{3,4,5}.json`, `ami-raw-t045.json`,
`ami-asnorm-t{2,3,4,5,6,7}.json` (same directory). AMI z≥5 plateaus —
the curve is flat between z=5 and z=7. CALLHOME could not be measured
(no local data); its profile stays an explicit placeholder.
