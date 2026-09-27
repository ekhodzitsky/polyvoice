# Native held-out NOTSOFAR baseline

Measured on Linux x86_64, AMD Ryzen AI 9 HX 370, using the default native
balanced INT8 v2 + VBx pipeline. All 129 meetings (13.3366 hours), one
far-field channel per meeting; collar 0, overlap scored.

| Metric | Baseline | Regression maximum |
|---|---:|---:|
| DER micro | 38.338620% | 40.338620% |
| DER macro | 36.704850% | 38.704850% |
| Miss (micro) | 22.286656% | — |
| False alarm (micro) | 1.746166% | — |
| Confusion (micro) | 14.305798% | — |

The 2 percentage point allowance was frozen before inference, matching the
historical NOTSOFAR tolerance. Both aggregate limits must pass; the precise
unrounded values in the baseline and manifest govern the check. Model and
scoring protocol changes require explicit baseline review. No per-file tuning
or exclusions were made.

Speaker count: exact 37/129, off by one 46/129, off by at least two 46/129.
Over-counting occurs in 82 meetings, under-counting in 10. The largest errors
combine missed speech with fragmentation of reference speakers.

| Meeting | DER % | Reference speakers | Predicted speakers |
|---|---:|---:|---:|
| MTG_32055 | 77.4476 | 5 | 10 |
| MTG_32188 | 67.7444 | 6 | 9 |
| MTG_32041 | 64.4469 | 5 | 9 |
| MTG_32095 | 63.1253 | 5 | 11 |
| MTG_32110 | 60.1782 | 7 | 9 |

These scores are a regression reference, not evidence of uniformly good
far-field diarization. Reference speech comes from transcript utterance
intervals, which can include within-utterance pauses; this is not the official
challenge's ASR/tcpWER evaluation. The single selected microphone does not
represent every device in the multi-device challenge. No telephone, multilingual
or cross-platform quality claim follows from this Linux measurement.

[report.json](report.json) contains the code revision, manifest/scorer/binary/model hashes,
host and compiler, exact commands, and every meeting's raw error components.
[hypotheses.zip](hypotheses.zip) retains the exact 129 CLI JSON outputs; their hashes are checked
against the report by the protocol test. Raw audio and GT transcript text are
not redistributed.

Protocol, corpus licensing and reproduction: [held-out evaluation](../../../docs/notsofar-eval.md).
Baseline: `notsofar_eval_native` in [der_baseline.json](../../../tests/der_baseline.json).

A second clean-checkout run ([verification.json](verification.json), revision `b8392a6ff84ce78104f17010fe3422dc294ffc7a`)
passed both limits. All 129 hypothesis files are byte-identical to the initial
run; both DER aggregates and their decomposition match exactly.
