# Speakrs H2H — status board

See **VERDICT.md** for the full write-up.

## Headline numbers (VoxConverse-test 232, collar 0, our scorer)

| Engine | DER₀ micro | conf | RTFx | Status |
|---|---:|---:|---:|---|
| **speakrs-coreml (warm)** | **11.08%** | 3.63 | **144×** M1 Pro | **DONE** |
| polyvoice v2+VBx | 15.24% (baseline) / re-measure WIP | ~7 | ~53× cold | re-measure running |
| Gap | **~4.2 pp** | mostly confusion | | |

## Reproduce speakrs full-232

```bash
cd ~/src/polyvoice-h2h
./benchmarks/tools/speakrs-rttm/target/release/speakrs-rttm \
  --mode coreml \
  --hyp-dir benchmarks/results_full/voxconverse_test/speakrs-coreml-warm \
  data/voxconverse-test/audio
python3 benchmarks/score_h2h.py \
  --ref data/voxconverse-test/rttm \
  --hyp benchmarks/results_full/voxconverse_test/speakrs-coreml-warm \
  --output benchmarks/results/speakrs-h2h-2026-08-03/speakrs-coreml-232-score.json
```

## Branch

`feat/speakrs-h2h-harness` @ `~/src/polyvoice-h2h`
