# VERDICT — speakrs H2H smoke (2026-08-03)

## Outcome

**speakrs is a real accuracy peer, not vaporware.** On the 10-file
VoxConverse-test smoke under `der.py` (collar 0, overlap scored, M1 Pro):

| | polyvoice v2+VBx | speakrs-coreml | Δ |
|---|---:|---:|---:|
| DER₀ micro | 17.72% | **13.74%** | **−3.98 pp** |
| confusion | 5.77 | **2.25** | −3.52 pp |
| cold RTFx | **52.5×** | 22.7× | polyvoice faster |

## Product implication

- **Do not** position polyvoice as “as accurate as community-1 Rust ports.”
- **Do** keep deployability / cold-CPU RTF / MIT ungated / multi-surface story.
- **Accuracy sprint priority:** speaker confusion (count + clustering), not
  segmentation miss or raw RTF race with CoreML.
- **Publish measured speakrs row** only after full-232 (+ AMI); smoke is enough
  to greenlight the sprint, not enough for README leaderboard finality.

## Gate for next accuracy work

Success = close ≥ half the smoke gap on **full-232** no-collar micro
(toward ≤15% → ≤14% path), with confusion down and RTF not regressing below
~40× cold CLI on M1 Pro.
