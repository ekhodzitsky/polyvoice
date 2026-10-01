# Pre-candidate readiness review

**Historical review: 2026-09-29. Verdict at that revision: NO-GO.**

**Superseded for the published 1.0.0:** see the [final readiness evidence](../PRODUCTION-READINESS.md).
The table and measurements below describe the earlier revision, not current release status.

The reviewed development revision is
`391deca1961221fba531ac89507fcdd5c919904f`, version **0.22.0**, with
`polyvoice-kernels` **0.1.3**. This is not a published release candidate.
The [product scope](../PRODUCTION-READINESS.md) remains batch native CPU
diarization through Rust, CLI, Python and C ABI v3. No release, tag or package
publication is authorized by this review.

Subsequent preparation adds the [packaged consumer scenario suite](release-artifacts.md#consumer-scenarios),
including duration boundaries and OS-enforced offline execution on Linux and
Darwin, plus application-scoped Windows network filters. The table below remains
the dated audit snapshot, not a claim about later candidates. Exact-candidate
qualification and published RC qualification were open at review time.

**Policy update, 2026-10-01:** neither a mandatory 14-day waiting period nor
a second release candidate is required.
The historical stability-window row below is superseded by the current
[RC policy](semver.md#release-candidate-window). One published, qualified
candidate and all technical gates remain required; elapsed time is not a gate.

## Evidence and gaps

| Requirement | Observed evidence | Remaining release blocker |
|---|---|---|
| API boundary | The advertised surface, extensibility rules and migration notes are documented in [semver](semver.md) and [CHANGELOG](../CHANGELOG.md). The preceding integration passed semver checks. | No qualifying RC has been published; compatibility must be checked for the actual candidate. |
| Dependencies | Local-only inference, explicit Linux BLAS and the experimental Apple Rust path are implemented. Core and Python metadata agree on 0.22.0. | The public sparse registry index lists kernels 0.1.0, 0.1.1 and 0.1.2, but not required 0.1.3. Staged kernel packages do not establish registry consumability. |
| Packaged artifacts | [Four-platform staged consumer checks](https://github.com/ekhodzitsky/polyvoice/actions/runs/36580156201) passed for `dc5e117ed49287fe1226c491ae0310afe1979fe3`. They exercise actual CLI, wheel, C and Rust archives. | These reports identify an earlier revision and use staged kernels. The exact candidate must pass with `require_registry=true`. |
| Consumer scenarios | The artifact harness runs real speech with local assets and HTTP proxies pointing at an unavailable localhost port, BYO silence and a C invalid-sample-rate call. It compares deterministic native results across front doors. | There is no retained complete candidate scenario matrix for invalid input, silence, short clips, overlap and maximum-duration operation across the advertised surfaces. Proxy settings alone are not proof of network isolation. |
| Corpus and resources | Historical full Linux/Darwin reports and recent experimental Apple comparisons exist. | No complete quality bundle for the reviewed revision. The verifier fails because current Linux evidence is absent; historical reports cannot substitute. Darwin requires an isolated M1 Pro, full Vox/AMI and all five scoreboard floors. |
| Trusted release workflow | The revision-bound collector and tag gate are implemented. | GitHub reports zero registered self-hosted runners and no runs of the Release quality evidence workflow. Provisioned Linux and isolated Darwin runners are prerequisites for the tag gate. |
| Repository checks | [Python on reviewed master](https://github.com/ekhodzitsky/polyvoice/actions/runs/36580867464) passed. The previous integration has successful checks including native scoreboard and Apple Rust kernels. | [Master CI](https://github.com/ekhodzitsky/polyvoice/actions/runs/36580867440) was still running at review time. Neither pending checks nor earlier source revisions certify a candidate. |
| Stability window | The policy requires two published candidates and 14 consecutive days after the final advertised-surface break. | Window not started. Its start requires the first qualifying publication and evidence for non-window gates. Elapsed development time does not count. |

Registry observation is from the
[public sparse index](https://index.crates.io/po/ly/polyvoice-kernels).
The crates.io API request returned HTTP 403; absence was established from the
successfully retrieved index instead. Registry and CI observations are a
dated snapshot and must be refreshed before publication.

## Required consumer evidence

Retain artifact hashes, exact source revision, platform/interpreter, models,
inputs, commands, expected outcomes and actual results for each scenario:

- External Rust BYO, native and local-only consumers; CLI; installed Python
  wheel; and a C caller using the shipped library and header.
- Real speech from local models with network access blocked during inference;
  missing/corrupt assets and invalid inputs must fail with the documented
  errors. Verify the local Rust dependency graph separately.
- Empty/short input and silence with valid finite bounded output or the
  documented rejection, plus overlapping speech under the scored protocol.
- The default one-hour boundary at 16 kHz, and rejection beyond the limit.
  A small configurable Rust limit test or a short speech fixture does not
  establish successful operation at the advertised maximum. BYO has its own
  caller-supplied configuration contract.

Source tests are complementary evidence; inspect their executed results and
skips. For example, model-dependent Python source tests can skip when
`POLYVOICE_MODEL_DIR` is absent. Their existence is not a completed consumer
scenario. The current C artifact fixture deliberately accepts at most one
minute, so it cannot establish the one-hour case.

## Order of completion

1. Finish the missing consumer scenario coverage and resolve any failures.
   Review and explicitly authorize the separate kernel publication; confirm
   registry resolution without a staged patch afterward.
2. Prepare candidate version/changelog metadata and select its final clean
   commit. Provision the two trusted quality runners with pinned datasets,
   models and toolchains, reserving the reference Mac exclusively.
3. On that exact commit, pass all repository checks, registry-backed artifact
   checks on the four platforms, and the complete release-quality workflow.
   Retain downloadable reports, including the wheel interpreter/ABI matrix.
   A local Linux collection alone does not satisfy the trusted tag gate.
4. Review the concrete artifacts and evidence before requesting publication
   authorization. Publish the first qualifying RC only after authorization;
   record its revision and publication date.
5. A further RC is required when candidate fixes need requalification, not
   merely to advance a counter. A compatibility break requires migration
   notes and a new candidate. No calendar waiting period is required.
6. Reassess the exact artifact being promoted, link the passing evidence in
   the readiness checklist and make an explicit release decision.

Every new source commit, including documentation/version changes, needs fresh
revision-bound qualification. Keep final measurement bundles as retained
external artifacts rather than committing them after measurement and tagging
a different SHA. This review does not mark any quality or stability gate done.

## Limitations and regression response

Keep the existing native product default on Darwin. The faster experimental
Rust packing results were measured with shared background load and do not
qualify the isolated release scoreboard. Pure Rust and literal zero external
dependencies remain separate goals, not reasons to delay an otherwise
qualified supported product.

The held-out NOTSOFAR baseline is about 38.34% micro DER, with documented
speaker over-counting; platform speed and accuracy claims remain tied to
their measured protocol. Wheel support is the tested platform/interpreter/ABI
matrix, not every Python version accepted by package metadata.

Any failed gate stops promotion. Reproduce against the recorded artifact and
inputs, fix or revert the regression, publish a replacement candidate only
with authorization, and retain both failing and passing evidence. Never lower
the five Darwin limits to accept a speed/memory trade-off. Compatibility
regressions require migration notes and a replacement candidate; compatible
fixes still require a new candidate and fresh applicable evidence.
