# Release quality evidence

A release needs successful measurements of its **exact clean commit**, not a
passing smoke test or results from an ancestor. `scripts/release-check.sh` first
runs the stdlib validator in `scripts/release-quality.py`. Missing, expired
artifacts, partial coverage, changed checksums, mismatched revision/platform/
models/protocol, nonfinite metrics and regressions stop publication.

## Required measurements

| Host | Full quality reports | Resource report |
|---|---|---|
| Linux x86_64, native `cli` | VoxConverse-test 232, AMI Mix-Headset 16, NOTSOFAR evaluation 129 | Informational full-corpus timing |
| Darwin ARM64, isolated Apple M1 Pro | VoxConverse-test 232, AMI Mix-Headset 16 | Vox-3: DER micro ≤7.11%, macro ≤7.39%, RTFx ≥117, INT8 pair ≤8,414,314 bytes, peak RSS ≤556 MiB |

Full corpus runs use balanced INT8, v2 + VBx, CPU, collar zero, overlap scored,
one file worker and default configuration. The collector builds with
`cargo build --locked --release --features cli`; it clears `POLYVOICE_*` tuning
variables and rejects custom Rust build flags. Model and all six PLDA files are
hashed from the actual registry cache, before and after measurement.

VoxConverse/AMI file identities, audio and RTTM SHA-256/size are frozen in
[`release-corpora.json`](../benchmarks/manifests/release-corpora.json), matching
all files in the existing full native reports. No raw corpus data is distributed
with evidence. Obtain datasets under their original terms; see
[`DATA_LICENSE`](../benchmarks/DATA_LICENSE) and the existing download scripts.
The Vox download script may provide only the ten-file smoke subset; provision
the complete 232-file split before collecting release evidence. The
[NOTSOFAR protocol](notsofar-eval.md) retains its separately frozen manifest,
license checks, existing scorer and raw hypotheses.

Linux VoxConverse/AMI DER micro **and macro** must stay within the existing
`*_linux_cpu_native` baselines plus their 1 / 1.5 percentage point tolerances in
[`der_baseline.json`](../tests/der_baseline.json). Darwin uses the measured
[2026-09-22 full-split reports](../benchmarks/results/darwin-native-der-2026-09-22/)
with the same respective tolerances. NOTSOFAR keeps its predeclared 2 pp budget
for each aggregate. The five independent
[Darwin scoreboard limits](../tests/native_scoreboard.json) have no added slack.
The resource collector uses the existing `polyvoice-bench` under
`/usr/bin/time -l`; RSS is converted from bytes to MiB, and RTF uses the existing
single-worker `rt_factor_avg`. It does not substitute shared CI timing or select
the best metric from different runs.

## Collect on the release commit

Use Python 3.11+, the project's Rust toolchain, full datasets and an idle host.
Bootstrap the default registry cache including PLDA (do not use a custom cache
for this step):

```bash
cargo run --locked --release --features cli --bin polyvoice -- diarize \
  tests/data/e2e-smoke/audio/fuzfh.wav --json > /dev/null
```

On Linux, also download the licensed NOTSOFAR corpus using its documented
command. On each host, from the same clean committed source tree:

```bash
# Linux (data contains voxconverse-test, ami-test and notsofar-eval)
python3 scripts/release-quality.py collect \
  --data-root /path/to/data --output bench-results/release-quality/Linux-x86_64

# Isolated Apple M1 Pro: attest that no other workload shares the measurement.
python3 scripts/release-quality.py collect --isolated-host \
  --data-root /path/to/data --output bench-results/release-quality/Darwin-arm64
```

Output directories must be new. A failed collection never leaves a passing
`evidence.json`. Retain raw bench reports, logs and NOTSOFAR hypotheses together
with that file. It includes UTC measurement time, source SHA, build/measurement
commands, host/toolchain details, model/PLDA hashes, binary hash and report
hashes. Copy both platform directories into one ignored evidence directory:

```bash
python3 scripts/release-quality.py verify --evidence bench-results/release-quality
POLYVOICE_RELEASE_EVIDENCE=bench-results/release-quality bash scripts/release-check.sh
```

The verifier deliberately rejects earlier source revisions even if runtime
source appears unchanged. Run measurements after the final source commit;
do not commit the reports afterwards and then tag a different commit.

## Tag workflow and runner provisioning

`Release quality evidence` is manually dispatched **on master before tagging**.
It requires self-hosted runners with these labels:

- Linux: `self-hosted`, `Linux`, `X64`, `polyvoice-quality`.
- Darwin: `self-hosted`, `macOS`, `ARM64`, `polyvoice-scoreboard` on the isolated
  Apple M1 Pro. Reserve it exclusively; the flag records operator attestation,
  not an automatic check that every other process is idle.

Preinstall Rust and Python and bootstrap each account's default model cache.
Set repository variables `QUALITY_LINUX_DATA` and `QUALITY_DARWIN_DATA` to the
provisioned dataset directories outside the disposable checkout. The workflow
serializes measurement runs; do not assign other work to the scoreboard runner.
It is never triggered by pull requests or untrusted fork code.

The tag gate locates a successful dispatch of this specific workflow on master
with the exact tag commit SHA. It downloads both platform artifacts, validates
them before compilation, and retains the bundle as `release-quality-evidence`.
The existing GitHub release job attaches that bundle along with the release
artifacts. Artifacts are retained for 90 days; if unavailable, rerun measurement
on the same commit before retrying release. No fallback to committed historical
reports exists. Local bundles trust the operator; tag bundles additionally rely
on trusted workflow execution and runner administration, not self-signed JSON.

## Historical Darwin evidence

The September 22 full-split rerun already superseded the old 0.18 measurements:
VoxConverse 13.33% and AMI 23.61% DER micro. Its notes also record a passing Vox-3
run. This reconciles the earlier pending rerun with evidence that is already
present; it does **not** certify any newer release SHA. Provisioning runners and
obtaining fresh passing measurements remain prerequisites for an actual release.
Linux ARM64 and Windows artifact smoke qualification remains separate; these
reports do not claim full-corpus measurements on those platforms.
