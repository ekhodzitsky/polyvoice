# Held-out NOTSOFAR evaluation

This protocol uses the full NOTSOFAR-1 evaluation split
`benchmark-datasets/eval_set/240825.1_eval_full_with_GT`: **129 meetings**, one
single-channel far-field device per meeting. The first lexicographically sorted
`sc_*` device's `ch0.wav` is selected before inference. No meeting is selected or
excluded based on its score. This is distinct from the historical 36-meeting
dev-set-1 measurement; the meeting ID sets are disjoint.

## Access and licensing

Microsoft's [official dataset mirror](https://huggingface.co/datasets/microsoft/NOTSOFAR)
provides audio and ground truth without a purchase or access token under
[CC BY 4.0](https://huggingface.co/datasets/microsoft/NOTSOFAR/blob/ba8fd0f034ce185fe4d24f47e53b4b8194795f07/LICENSE.txt).
Attribute the NOTSOFAR-1 dataset and its creators when publishing results; see
[the challenge repository](https://github.com/microsoft/NOTSOFAR1-Challenge).
The checked-in [manifest](../benchmarks/manifests/notsofar-eval.json) pins the
repository revision, all source paths, byte sizes and SHA-256 checksums,
including the license and reference annotations. Audio totals 1,536,380,832
bytes. Audio and ground-truth text are downloaded locally, not redistributed.

## Frozen protocol

- Default native CLI, balanced INT8 powerset + ResNet34, v2 + VBx; no speaker
  count hint, domain profile, threshold override, or evaluation-set calibration.
- Collar **0 seconds**, overlap scored, 10 ms frame scoring using the existing
  `benchmarks/der.py` implementation and optimal speaker mapping.
- Reference utterance start/end times are used, preserving overlap. Segments
  shorter than 1 ms are excluded, matching the historical NOTSOFAR converter.
  Word-level timestamps are not substituted for speech-activity annotation.
- DER micro and macro, micro miss/false-alarm/confusion, speaker counts and
  per-meeting components are reported. Empty hypotheses count as missed speech.
- The regression allowance is **2.0 percentage points** above the frozen native
  baseline for **each** of micro and macro DER. This allowance is declared before
  measurement and carries over the historical NOTSOFAR baseline's tolerance.
  It is a regression budget, not a claim that the baseline quality is sufficient
  for every application or a substitute for the locked Darwin scoreboard floors.

The evaluation split is held out from project calibration. Historical development
results do not qualify it; do not tune configuration or models against these
scores. This does not establish absence of overlap with every upstream pretrained
model's training data. The domain is distant-microphone office meetings, not
telephone calls or arbitrary languages; AMI's close-talk mix is a different
recording condition, but both corpora contain meetings.

## Reproduction and required gate

Use Python 3.11+ and Cargo; the harness adds no dependency. Run the bundled
smoke clip once to populate a dedicated cache with both INT8 models and all six
PLDA arrays. The downloader-only CLI command fetches the profile pair but does
not fetch PLDA; the normal VBx pipeline does. This bootstrap is not calibration.

```bash
cargo run --locked --release --features cli --bin polyvoice -- diarize \
  tests/data/e2e-smoke/audio/fuzfh.wav --models-cache "$PWD/.cache/notsofar-models" --json > /dev/null
python3 scripts/notsofar-eval.py download
python3 scripts/notsofar-eval.py run --models .cache/notsofar-models
```

Use a clean committed checkout. The harness builds the current CLI with
`cargo build --locked --release --features cli --bin polyvoice` and records the
revision, binary/scorer/manifest/model hashes, host, compiler, build flags, exact
commands and individual hypotheses. Output goes to the ignored
`bench-results/notsofar-eval/`; use a fresh `--output` directory for another run.

The required run fails on missing/corrupt inputs, partial or duplicate coverage,
inference errors, protocol changes or either exceeded regression limit. Download
failures never become successful skips. Existing files are rehashed rather than
trusted by filename. All 129 meetings are required; there is no max-files option.
The separate `--record-only` mode establishes the initial baseline and explicitly
**does not** report a passing release gate.

## Native reference measurement

The [full report and failure analysis](../benchmarks/results/notsofar-eval-native-2026-09-27/README.md)
record Linux x86_64 DER micro **38.338620%**, macro **36.704850%**. Miss,
false alarm and confusion contribute **22.286656% / 1.746166% / 14.305798%**
respectively to micro DER. All 129 hypotheses were byte-identical in a second
full run, which passed the required gate.

The baseline is `notsofar_eval_native` in [der_baseline.json](../tests/der_baseline.json).
The precise baseline plus the predeclared 2 pp allowance gives limits of
**40.338620% micro / 38.704850% macro** (display rounded; the gate uses full
precision). Speaker count is exact in 37 meetings; 82 are over-counted and 10
under-counted. These results expose a substantial far-field quality limitation.
Reference utterance intervals may include within-utterance pauses; this protocol
is not the challenge's official ASR/tcpWER evaluation.

This establishes a Linux reference, not cross-platform or release-candidate
qualification. Release integration and revision-bound multi-corpus evidence
remain required.

```bash
python3 scripts/test-notsofar-eval.py
```
