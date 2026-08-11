# Benchmarks

Honest, reproducible Diarization Error Rate (DER) for polyvoice, with a
collar-disclosed, protocol-annotated comparison to open-source diarizers.

**DER is meaningless without a stated collar and overlap policy.** Every number
below is labelled with its collar (0.25 s "forgiveness" vs **0** "strict") and
whether overlapping speech was scored. polyvoice's numbers are produced by
[`polyvoice-bench`](../src/bin/polyvoice-bench.rs) and reproduced by the
cross-engine [`benchmarks/`](../benchmarks/) harness, whose single DER scorer
([`benchmarks/der.py`](../benchmarks/der.py)) agrees with `polyvoice-bench` to
~0.02 pp. **Competitor numbers are their own published figures** (sourced below),
measured under their own protocols — they are NOT yet re-run through our scorer.
The `benchmarks/` runners are wired to re-measure them like-for-like once their
(often gated) stacks are installed; until then, compare only within a matched
collar.

### Source of truth (VoxConverse-test, v2+VBx INT8 default, collar 0 micro)

| Figure | Role | Artifact |
|------|------|----------|
| **14.94 %** | **Linux / CPU product truth** (server deploys) | `scripts/linux-cpu-der-gate.sh` → [`linux-cpu-der-2026-08-11/`](../benchmarks/results/linux-cpu-der-2026-08-11/) |
| **15.02 %** | Mac CoreML headline + historical CI gate | INT8 full-split 2026-08-10 — `benchmarks/results/int8-full-der-2026-08-10/` |
| **15.24 %** | Historical FP32 hop-2.0 published (pre-0.17) | `benchmarks/results/powerset-hop2-2026-07-30/`, `voxconverse-test-232-2026-07-31.json` |
| **15.22 %** | Same-scorer H2H vs speakrs (FP32-era CLI, 2026-08-03) | `benchmarks/results/speakrs-h2h-2026-08-03/` |

**Default since 0.17.0** is the INT8 pair (`powerset_int8` + `resnet34_int8`)
on every profile. **Cite 14.94 % for Linux/CPU deploys** (powerset micro-batch
N=8, EP=cpu). Cite **15.02 %** for Mac CoreML (N=1 clamp). Reproduce Linux:

```bash
DOCKER=1 bash scripts/linux-cpu-der-gate.sh   # or native Linux host
```

## At a glance

| | polyvoice (v2+VBx, INT8 default) | pyannote 3.1 | WhisperX | NeMo Sortformer |
|--|--|--|--|--|
| **VoxConverse-test DER** | **15.0 %** ¹ | **11.3 %** ¹ | 11.3 % (= pyannote) | not published |
| **Model size** | **~8.4 MB** (+ PLDA for VBx) | ~32.5 MB | ~32.5 MB + Whisper | 123 M params |
| **Runtime** | **CPU ~80–95× (Linux aarch64) / CoreML ~111–130× (M1)** | CPU/GPU (PyTorch) | GPU recommended | GPU |
| **Weights** | **MIT, ungated** | MIT code, **gated** (HF token) | gated (pyannote) | **CC-BY-NC** (non-commercial) |
| **Dependencies** | **Rust API, optional ONNX Runtime; no PyTorch** | PyTorch | PyTorch + Whisper | PyTorch / NeMo |
| **Bindings** | **Rust / Python / C / CLI** | Python | Python | Python |
| **Streaming** | **Yes** | No | No | No |

¹ VoxConverse-test, **no forgiveness collar (collar 0), overlap scored** — the
strict protocol pyannote 3.1 reports against, so these two are collar-matched.
polyvoice trails the accuracy leader by ~4 DER points and trades that for
deployability: a Rust-native, CPU, MIT, **ungated** engine (ONNX Runtime for
the production path) with four bindings and streaming. It is **not** the
accuracy leader.

## The collar caveat (read this first)

A 0.25 s forgiveness collar removes ~3–8 DER points versus collar 0 on the same
audio, because errors near every speaker boundary are forgiven. **pyannote 3.1
and diart publish at collar 0; NeMo's CALLHOME and most toolkit recipes use 0.25.**
Treat collar-0 and collar-0.25 as two different leaderboards. Where we compare to
pyannote we use polyvoice's **no-collar** number.

## Accuracy — VoxConverse test (232 files)

**Collar 0, overlap scored, Hungarian mapping** (the pyannote 3.1 protocol).

| System | DER % | miss | FA | conf | collar | source |
|---|---|---|---|---|---|---|
| pyannote.audio 3.1 | **11.3** | 3.4 | 4.1 | 3.8 | 0 | model card ² |
| pyannote community-1 | 11.2 | — | — | — | 0 | pyannote benchmark ⁷ |
| **speakrs CoreML (warm)** | **11.08** | 3.35 | 4.10 | 3.63 | 0 | this repo H2H ⁸ |
| VBx (offline baseline) | 11.1 | 4.6 | 3.1 | 3.4 | 0 | diart paper ³ |
| 3D-Speaker toolkit | 11.75 | — | — | — | unstated | repo ⁴ |
| **polyvoice (v2+VBx, INT8 default)** | **15.02** | 3.54 | 3.95 | 7.62 | 0 | this repo ⁵ |
| polyvoice (v2+VBx, FP32 historical) | 15.24 | 2.14 | 1.73 | 7.16 | 0 | hop-2.0 ⁵ |
| diart (online, 5 s latency) | 16.8 | 4.9 | 3.8 | 8.2 | 0 | diart paper ³ |
| polyvoice (legacy, `--legacy`) | 18.54 | 4.49 | 3.19 | 4.99 | 0 | this repo ⁵ |
| diart (online, 1 s latency) | 20.1 | 3.3 | 5.1 | 11.7 | 0 | diart paper ³ |

### Head-to-head: polyvoice vs speakrs (measured, same scorer)

VoxConverse-test 232, collar 0, overlap scored, `benchmarks/der.py`, Apple M1 Pro
(2026-08-03/04). speakrs = warm CoreML batch; polyvoice = CLI default v2+VBx
(cold process per file for RTF).

| Engine | DER₀ micro | DER₀.₂₅ | conf | spk exact | RTFx |
|---|---:|---:|---:|---:|---:|
| **speakrs-coreml (warm)** | **11.08** | 6.70 | **3.63** | 115/232 | ~144× |
| **polyvoice v2+VBx** | **15.22** | 10.47 | **8.04** | 84/232 | ~40× cold CLI |

**Gap 4.14 pp** is almost entirely **confusion** (speaker assignment / count),
not miss/FA. Full protocol and RTTMs:
[`benchmarks/results/speakrs-h2h-2026-08-03/`](../benchmarks/results/speakrs-h2h-2026-08-03/).

For reference, polyvoice INT8 default at the **0.25 s collar** is **10.33 %**
micro (macro 10.67). FP32-era hop-2.0 was **10.52 %** micro. Legacy at collar
0.25 is **12.91 %** micro. No collar-0.25 pyannote number is published, so do
not compare that figure across systems.

## Accuracy — VoxConverse dev (216 files)

| polyvoice (v2+VBx, default) | DER micro % | macro % |
|---|---|---|
| collar 0 | **11.36** | 11.54 |
| collar 0.25 | **7.70** | 8.09 |

Measured 2026-07-30 on the hop-2.0 pipeline; artifact
[`benchmarks/results/powerset-hop2-2026-07-30/`](../benchmarks/results/powerset-hop2-2026-07-30/).

## Accuracy — AMI test (16 meetings, Mix-Headset)

**Collar 0, overlap scored.** AMI is long-form with heavy overlap; miss
dominates, and speaker mis-counting drives confusion.

| System | DER % | miss | FA | conf | setup | source |
|---|---|---|---|---|---|---|
| DiariZen (WavLM, CC-BY-NC) | 15.4 | — | — | — | SDM | 3D-Speaker eval ⁴ |
| pyannote community-1 | 17.0 | — | — | — | IHM | pyannote benchmark ⁷ |
| pyannote.audio 3.1 | **18.8** | 9.5 | 3.6 | 5.7 | Mix-Headset | model card ² |
| pyannote community-1 (SDM) | 19.9 | — | — | — | SDM | pyannote benchmark ⁷ |
| 3D-Speaker toolkit | 21.76 | — | — | — | SDM | repo ⁴ |
| pyannote 3.1 (SDM) | 22.4 | 11.2 | 3.8 | 7.5 | array1-ch1 | model card ² |
| polyvoice (v2+VBx, FP32 historical) | 23.42 | 7.44 | 1.75 | 6.05 | Mix-Headset | hop-2.0 ⁵ |
| VBx (offline baseline) | 24.1 | 17.2 | 3.1 | 3.8 | — | diart paper ³ |
| **polyvoice (v2+VBx, INT8 default)** | **24.50** | 11.17 | 3.62 | 8.80 | Mix-Headset | this repo ⁵ |
| diart (online, 5 s) | 27.5 | 10.0 | 5.0 | 12.4 | headset | diart paper ³ |
| polyvoice (legacy, `--legacy`) | 32.87 | 17.09 | 2.44 | 5.21 | Mix-Headset | this repo ⁵ |

### Head-to-head on AMI-test (measured, same scorer)

AMI-test 16 Mix-Headset, collar 0, `benchmarks/der.py`, Apple M1 Pro (2026-08-04).

| Engine | DER₀ micro | DER₀.₂₅ | conf | spk exact | RTFx |
|---|---:|---:|---:|---:|---:|
| **speakrs-coreml (warm)** | **17.43** | 10.96 | **4.33** | **11/16** | ~215× |
| **polyvoice v2+VBx** | **23.40** | 15.65 | **8.47** | 2/16 | ~56× cold CLI |

**Gap 5.97 pp**; speaker-count collapse is the polyvoice failure mode on AMI
(exact 2/16 vs speakrs 11/16). Artifact:
[`benchmarks/results/speakrs-h2h-2026-08-03/ami-16-matched-score.json`](../benchmarks/results/speakrs-h2h-2026-08-03/ami-16-matched-score.json).

polyvoice INT8 default at the 0.25 s collar: **16.82 %** micro (macro 16.26).
FP32-era hop-2.0: **15.71 %** micro (macro 15.24). Legacy at collar 0.25:
**25.20 %** micro (macro 24.75).

## Accuracy — NOTSOFAR-1 dev-set-1 (36 meetings, single far-field channel)

**Collar 0, overlap scored** (collar 0.25 in parentheses). Distant-microphone
office meetings are a much harder domain than VoxConverse/AMI; the gate exists
to track regressions on a third corpus, not to compare with array-based
systems (multi-channel beamforming is out of scope for the default pipeline).

Protocol: `benchmark-datasets/dev_set/240825.1_dev1`, the first sorted
single-channel `sc_*` device per meeting, GT from `gt_transcription.json`
converted to RTTM by `scripts/notsofar-to-rttm.py`. Download:
`scripts/download-notsofar.sh` (CC BY 4.0, HF mirror).

| System | DER₀ micro (macro) | DER₀.₂₅ micro (macro) | miss | FA | conf | source |
|---|---|---|---|---|---|---|
| **polyvoice (v2+VBx, default)** | **47.08 (45.11)** | **29.77 (30.45)** | 24.59 | 1.86 | 18.66 | this repo ⁵ |

Speaker count on the 36 meetings: exact 8, ±1 18, off-by-2+ 10 — the pipeline
under-counts on 4–8-speaker far-field meetings, the same failure mode as AMI.
Artifacts: [`benchmarks/results/notsofar-dev/`](../benchmarks/results/notsofar-dev/).
A fixed 3-meeting subset (`MTG_30860/30861/30862`) is gated in
`tests/der_v2_baseline_test.rs`.

## polyvoice speaker-count & error decomposition

A low DER can hide bad speaker counting; we report it explicitly.

| Split | DER (collar 0) | miss | FA | conf | spk exact | spk ±1 | spk off-by-2+ |
|---|---|---|---|---|---|---|---|
| VoxConverse-test (232, v2+VBx **INT8**) | **15.02 %** | 3.54 | 3.95 | 7.62 | 83 | 62 | 87 |
| AMI-test (16, v2+VBx **INT8**) | **24.50 %** | 11.17 | 3.62 | 8.80 | 1 | 3 | 12 |
| VoxConverse-test (232, v2+VBx FP32 hist.) | 15.24 % | 2.14 | 1.73 | 7.16 | 84 | 67 | 81 |
| VoxConverse-dev (216, v2+VBx FP32 hist.) | 11.36 % | 1.55 | 1.07 | 5.47 | 125 | 52 | 39 |
| AMI-test (16, v2+VBx FP32 hist.) | 23.42 % | 7.44 | 1.75 | 6.05 | 2 | 3 | 11 |
| VoxConverse-test (232, legacy) | 18.54 % | 4.49 | 3.19 | 4.99 | 57 | 46 | 129 |
| AMI-test (16, legacy) | 32.87 % | 17.09 | 2.44 | 5.21 | 1 | 0 | 15 |

The dominant residual error is still **speaker mis-counting** (confusion + the
off-by-2+ tail), especially on long meetings — better than legacy but still the
honest weak spot versus pyannote's end-to-end segmentation.

## Default pipeline: v2 + VBx (since 0.11)

As of **0.11**, the CLI default is **pipeline v2 + VBx** after a hard full-split
gate (2026-07-25): no-collar micro DER ≤ legacy on **both** VoxConverse-test
(232) and AMI-test (16). Artifacts:
[`benchmarks/results/full-der-2026-07-25/`](../benchmarks/results/full-der-2026-07-25/)
(`VERDICT.md`).

| Path | CLI | Vox no-collar micro | AMI no-collar micro |
|---|---|---:|---:|
| **v2 + VBx INT8 (default, 0.17+)** | `polyvoice file.wav` (+ PLDA) | **15.02** | **24.50** |
| v2 + VBx FP32 (historical, pre-0.17) | model ids `powerset_fp32` / `wespeaker_resnet34` | 15.24 | 23.42 |
| legacy | `polyvoice --legacy file.wav` | 18.54 | 32.87 |
| v2 + AHC | `polyvoice --clusterer ahc file.wav` | (subset; see archive) | (subset) |

VBx PLDA weights auto-download via the model registry; optional override
`--vbx-plda-dir <dir>` / `POLYVOICE_VBX_PLDA_DIR` (see
[`docs/vbx-plda-release.md`](vbx-plda-release.md)). Use `--clusterer ahc` or
`--legacy` for non-VBx paths.

Earlier **subset** bootstrap numbers (60-file Vox / full AMI, pre-gate) are
retained below for history only — prefer the full-split rows above.

**No-collar macro DER % (95 % CI) — historical subsets:**

| Pipeline | VoxConverse-test | VoxConverse-dev | AMI |
|---|---|---|---|
| legacy (full split) | 18.54 | — | 32.87 |
| v2 + AHC (subset) | 20.4 [16.4–24.9] | 13.2 [10.5–16.1] | 35.8 [29.8–41.9] |
| v2 + VBx (subset) | 17.0 [13.5–20.8] | 13.6 [10.6–16.9] | 37.0 [31.0–43.1] |
| **v2 + VBx (full split, default)** | **15.83 macro / 15.37 micro** | — | **24.12 macro / 25.17 micro** |

**Honest reading (historical 60-file subset era, pre–full-split gate):**

- These subset numbers motivated shipping **v2 + VBx** as the product default
  (since 0.11). They are **not** the current leaderboard — use the full-split
  tables above (Vox 232 / AMI 16) and the at-a-glance section.
- On that older subset, v2 + AHC over-clustered (high confusion); VBx cut
  confusion and became competitive with legacy on conversational audio.
- **Superseded on AMI:** full-split AMI-test favors **v2 + VBx** (INT8
  **24.50 %** / FP32 historical 23.42 % no-collar) over legacy (32.87 %), not
  the reverse. Do not treat the “legacy remains the robust default” sentence
  from early subset notes as current product policy.
- Reproduce subset-style runs with
  [`benchmarks/bench_subset.py`](../benchmarks/bench_subset.py) only for
  historical comparison.

## Speed — real-time factor (RTF; lower = faster)

Measured end-to-end with `polyvoice-bench` on **Apple M1 Pro (10 cores)**,
release build, single file stream at a time.

### Current default — INT8 (2026-08-10/11, v0.17+)

| Configuration | Corpus | RTFx (× realtime) | RTF | DER₀ micro |
|---|---|---:|---:|---:|
| **v2 + VBx INT8 Linux CPU (N=8)** | VoxConverse-test (232) | **~82×** | ~0.012 | **14.94 %** |
| **v2 + VBx INT8 Linux CPU (N=8)** | AMI-test (16) | **~95×** | ~0.011 | **24.19 %** |
| **v2 + VBx INT8 CoreML (N=1)** | VoxConverse-test (232) | **~111–122×** | ~0.008–0.009 | 15.02 % |
| **v2 + VBx INT8 CoreML (N=1)** | AMI-test (16) | **~109–130×** | ~0.008–0.009 | 24.50 % |
| **v2 + VBx INT8 Mac CPU (N=8)** | VoxConverse-test (232) | **~121×** | ~0.008 | 14.64 % |
| **v2 + VBx INT8 Mac CPU (N=8)** | AMI-test (16) | **~137×** | ~0.007 | 24.63 % |

Linux CPU (product path for servers):
[`benchmarks/results/linux-cpu-der-2026-08-11/`](../benchmarks/results/linux-cpu-der-2026-08-11/)
via [`scripts/linux-cpu-der-gate.sh`](../scripts/linux-cpu-der-gate.sh).
CoreML: [`int8-full-der-2026-08-10/`](../benchmarks/results/int8-full-der-2026-08-10/).
Mac CPU N=8:
[`int8-batch8-default-2026-08-10/`](../benchmarks/results/int8-batch8-default-2026-08-10/).
CoreML forces powerset micro-batch **N=1**; **N=8 is the default on CPU**.

### Historical — FP32, CPU EP (2026-07-30/31, v0.14.0)

| Configuration | Corpus | RTFx (× realtime) | RTF |
|---|---|---:|---:|
| v2 + VBx (fp32) | AMI-test (16) | 68.1× | 0.015 |
| v2 + VBx (fp32) | VoxConverse-dev (216) | 56.3× | 0.018 |
| v2 + VBx (fp32) | VoxConverse-test (232) | 53.5× | 0.019 |
| v2 + VBx INT8 single-file probe | AMI EN2002a | 83.3× | 0.012 |
| legacy (`--legacy`, steady state) | VoxConverse subset | ~33× | ~0.03 |

| Engine | RTF | Notes |
|---|---|---|
| **polyvoice (INT8 default, CoreML N=1)** | **~0.008–0.009** (~111–130×) | Rust + ONNX Runtime; full-split 2026-08-10 |
| polyvoice (INT8 default, CPU N=8) | **~0.007–0.008** (~121–137×) | powerset micro-batch; 2026-08-10 |
| polyvoice (FP32 historical, CPU) | 0.015–0.019 (53–68×) | pre-0.17 default models |
| pyannote 3.1 | not published | PyTorch; GPU recommended for throughput |
| WhisperX | > 1 on CPU | Whisper + pyannote; GPU recommended |
| NeMo Sortformer | GPU-only | ~48 GB GPU for ~12-min recordings |

**INT8 accuracy note:** full-split INT8 is **parity on VoxConverse** (−0.2 pp
vs FP32 published) and about **+1.1 pp on AMI** vs FP32 hop-2.0. That is the
shipping tradeoff as of 0.17.0 — not an opt-in `fast` caveat anymore.

The cross-engine harness measures polyvoice end-to-end through its CLI (which
cold-loads the model per file), a conservative lower bound; the figures above
are steady-state in-process numbers.

## Footprint, license & gating

| Engine | Deployable size | License | Gated weights? | Runtime |
|---|---|---|---|---|
| **polyvoice** | **~8.4 MB** INT8 production pair (+ PLDA for VBx) | **MIT** | **No** | Rust + ONNX Runtime, CPU/CoreML |
| pyannote 3.1 | ~32.5 MB (seg 5.9 + embed 26.6) | MIT code | **Yes** (HF token + accept) | PyTorch, CPU/GPU |
| WhisperX | ~32.5 MB + Whisper model | BSD code | Yes (pyannote) | PyTorch, GPU |
| sherpa-onnx | seg ~5.9 MB + embed (int8 avail.) | Apache-2.0 | No | ONNX, CPU |
| NeMo Sortformer | 123 M params | **CC-BY-NC** | No token, **non-commercial** | PyTorch, GPU |
| diart | pyannote@2021 models | MIT code | Yes (pyannote) | streaming, CPU/GPU |

polyvoice's production footprint is **~8.4 MB** (INT8 default since 0.17) —
smaller than pyannote's ~32.5 MB stack — and uniquely **MIT + ungated +
Rust-native + CPU**. sherpa-onnx is the closest architectural peer (ONNX, CPU)
but publishes **no DER**.

## Datasets

| Dataset | Files | Source | License |
|---|---|---|---|
| VoxConverse dev / test | 216 / 232 | [voxconverse](https://github.com/joonson/voxconverse) | annotations CC-BY-4.0; audio from YouTube (not redistributed) |
| AMI test (Mix-Headset) | 16 | [AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | CC-BY-4.0 |
| NOTSOFAR-1 dev-set-1 | 36 | [microsoft/NOTSOFAR](https://huggingface.co/datasets/microsoft/NOTSOFAR) | CC-BY-4.0 |

Audio is downloaded by `scripts/download-*.sh`; this repo redistributes none of
it. See [`benchmarks/DATA_LICENSE`](../benchmarks/DATA_LICENSE).

## Reproduce

```bash
# polyvoice, full decomposition (Rust harness, the source of our numbers):
scripts/run-der-sweep.sh                       # dev/test/AMI, collar 0.25 + no-collar in one run

# cross-engine, single scorer (like-for-like when competitors are installed):
cd benchmarks && python make_manifests.py
python benchmark.py --dataset voxconverse_test --runners all
```

## Caveats & protocol notes (why cross-source numbers are not head-to-head)

1. **Collar** is the biggest confound: pyannote/diart at 0; NeMo CALLHOME and
   most recipes at 0.25; a 0.25 s collar is ~3–8 DER points more forgiving.
2. **Overlap**: pyannote 3.1, diart, NeMo, and our scorer all score overlapped
   speech (`skip_overlap=False`). Older literature that excludes overlap reports
   deflated DER.
3. **VAD / speaker-count oracle**: pyannote 3.1 and polyvoice use automatic VAD
   and automatic speaker counting (no oracle). Some published numbers use oracle
   VAD or oracle speaker count, which is far easier.
4. **Dataset/split drift**: VoxConverse v0.0.2 vs v0.3; AMI Mix-Headset (easier)
   vs SDM/array; DIHARD "Full" vs ≤4-speaker subsets; CALLHOME split by speaker
   count. Match all of these before comparing.
5. **Scoring tool**: most use `pyannote.metrics` (Hungarian mapping); wespeaker's
   recipe uses NIST `md-eval.pl`. Our scorer follows the `md-eval` frame model
   with Hungarian mapping and is cross-checked against `polyvoice-bench`.
6. **Cross-repo numbers are unreliable**: e.g. DiariZen on VoxConverse was
   reported as 28.39 % by one harness and ~5.2 % by another — a version/config
   mismatch. Reproduce before trusting any third-party number.

## Sources

- ² pyannote 3.1 (VoxConverse, AMI, DIHARD; protocol; MIT; gated): https://huggingface.co/pyannote/speaker-diarization-3.1 — footprint: [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 5.91 MB + [wespeaker-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) 26.6 MB
- ³ diart (Coria et al., ASRU 2021; collar 0, overlap scored): https://arxiv.org/abs/2109.06483
- ⁴ 3D-Speaker toolkit (VoxConverse 11.75, AMI_SDM 21.76; collar unstated): https://github.com/modelscope/3D-Speaker
- ⁵ polyvoice **published** tables (0.17+) use the INT8 full-split remeasure
  (**15.02 %** Vox / **24.50 %** AMI no-collar micro; CoreML N=1, M1 Pro,
  2026-08-10). Gate + README match
  [`tests/der_baseline.json`](../tests/der_baseline.json). Artifact:
  [`benchmarks/results/int8-full-der-2026-08-10/`](../benchmarks/results/int8-full-der-2026-08-10/).
  CPU powerset micro-batch N=8 (product default off CoreML): Vox **14.64 %** /
  AMI **24.63 %** — `int8-batch8-default-2026-08-10/`. Historical FP32 hop-2.0
  (15.24 / 23.42) remains under `powerset-hop2-2026-07-30/` and early full-split
  under `full-der-2026-07-25/`. Default (0.11+ pipeline): v2 + VBx; models INT8
  since 0.17; legacy via `--legacy`
- ⁶ RTF artifact: [`benchmarks/results/voxconverse-test-10files-20260516.json`](../benchmarks/results/voxconverse-test-10files-20260516.json)
- ⁷ pyannote official benchmark (updated 2025-09; collar 0, overlap scored; community-1 weights CC-BY-4.0 but still HF-gated): https://www.pyannote.ai/benchmark + https://huggingface.co/pyannote/speaker-diarization-community-1 — on VoxConverse community-1 ties 3.1 (11.2 vs the 11.3 model-card figure; annotation-version drift), so the README headline comparison vs 3.1 stands
- ⁸ speakrs CoreML warm on Apple M1 Pro, full VoxConverse-test 232, scored with `benchmarks/der.py` (collar 0, overlap scored): DER 11.08% micro (miss 3.35 / FA 4.10 / conf 3.63), RTFx ~144×. Artifact `benchmarks/results/speakrs-h2h-2026-08-03/full-232-matched-score.json` (2026-08-03/04). speakrs code Apache-2.0: https://github.com/avencera/speakrs. polyvoice on the **same** scorer/split: 15.22% no-collar micro (conf 8.04) — gap **4.14 pp**, confusion-dominated. speakrs' own README quotes 11.1% / 631× on M4 Pro; do not mix hardware for RTFx.
- NeMo Sortformer (CC-BY-NC; DIHARD 16.28, CALLHOME): https://huggingface.co/nvidia/diar_sortformer_4spk-v1
- WhisperX (bundles pyannote 3.1): https://github.com/m-bain/whisperX
- sherpa-onnx (no DER published): https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html
- "Benchmarking Diarization Models" (DiariZen ~5.2 VoxConverse @0.25 collar): https://arxiv.org/abs/2509.26177

## Streaming latency presets (measured)

Named presets for `polyvoice::streaming::LatencyPreset` / CLI `--latency-preset`.

**Protocol (2026-07-24):** VoxConverse-test **first 10 files** (sorted names),
overlap scored, Hungarian DER, collar 0 and 0.25. `StreamingPipeline` + Silero
VAD + WeSpeaker ResNet34, feed chunks of 3200 samples (~200 ms) @ 16 kHz.
Release build on **Apple M1 Pro (10 cores)**. Full artifact:
[`benchmarks/results/streaming-latency-measured.json`](../benchmarks/results/streaming-latency-measured.json).

| Preset     | window | hop  | right ctx | cache | input-buffer latency¹ | RTF   | DER collar 0 | DER collar 0.25 |
|------------|--------|------|-----------|-------|----------------------|-------|--------------|-----------------|
| `realtime` | 1.0 s  | 0.5  | 0.0       | 16    | **1.032 s**          | **0.117** | **42.15%** | **32.74%** |
| `balanced` | 1.5 s  | 0.75 | 0.0       | 32    | **1.532 s**          | **0.109** | **29.99%** | **20.15%** |
| `accurate` | 2.0 s  | 1.0  | 0.25      | 64    | **2.282 s**          | **0.111** | **30.10%** | **20.85%** |

¹ **Input-buffer latency** is configuration:
`window_secs + right_context_secs + vad_frame_secs` with frame 512 @ 16 kHz
(`≈ 0.032 s`). It is **not** wall-clock RTF. RTF = total wall / total audio
(~6701 s audio × 3 presets on this run).

**Notes:** online DER is expected to trail the offline legacy baseline
(~18.5% no-collar full test). On this 10-file slice `balanced` beats
`realtime` by ~12 pp (collar 0); `accurate` does **not** improve further
(right-context currently contributes to the latency budget but does not yet
delay emission — see streaming module docs). Reproduce:

```bash
cargo run --release --features cli --bin polyvoice-measure -- streaming \
  --dataset data/voxconverse-test --max-files 10 \
  --output benchmarks/results/streaming-latency-measured.json
```

See also `benchmarks/results/streaming-latency-methodology.md`.

## VAD parity: Silero vs earshot (measured)

**Protocol:** same VoxConverse-test **10-file** slice, legacy offline pipeline
(WeSpeaker ResNet34 + AHC threshold 0.45), only the VAD backend swapped.
Release build, Apple M1 Pro. Artifact:
[`benchmarks/results/vad-parity-earshot-silero.json`](../benchmarks/results/vad-parity-earshot-silero.json).

| VAD | frame | DER collar 0 | DER collar 0.25 | RTF |
|-----|-------|--------------|-----------------|-----|
| **Silero** (default) | 512 | **23.89%** | **15.82%** | 0.102 |
| **earshot** (`vad-earshot`) | 256 | **26.54%** | **18.68%** | 0.099 |
| **Δ (earshot − Silero)** | — | **+2.65 pp** | **+2.86 pp** | −0.0025 |

**Parity gate** (notes file): |Δ DER| ≤ **0.3 pp** absolute. **Failed** on both
collars → keep earshot **optional only**; do **not** switch the default VAD.
Vendor “40× faster / more accurate” claims remain **unverified / not supported**
by this DER gate (RTF nearly tied; accuracy worse).

```bash
cargo run --release --features "cli,vad-earshot" --bin polyvoice-measure -- vad-parity \
  --dataset data/voxconverse-test --max-files 10 \
  --output benchmarks/results/vad-parity-earshot-silero.json
```

## Embedder short-segment: ResNet34 vs ERes2NetV2 (measured)

**EER protocol:** 400 same/different-speaker pairs built from **VoxConverse-test
RTTM segments** (VoxCeleb1 audio not present in this tree). Center-crop to
0.5 / 1 / 2 / 3 s, cosine scoring, equal-error rate. **Not** the official
VoxCeleb1 `veri_test` protocol — domain is in-the-wild multi-party English.

**DER protocol:** same 10-file Vox slice, legacy pipeline, Silero VAD fixed;
only the embedder ONNX swapped (ResNet34 256-d vs ERes2NetV2 192-d zh-cn
optional download). Artifact:
[`benchmarks/results/embedder-short-eres2net.json`](../benchmarks/results/embedder-short-eres2net.json).

| Duration | ResNet34 EER % | ERes2NetV2 EER % |
|----------|----------------|------------------|
| 0.5 s | **18.86** | 27.46 |
| 1.0 s | **7.21** | 20.09 |
| 2.0 s | **4.75** | 13.03 |
| 3.0 s | **3.84** | 10.74 |

| Embedder | DER collar 0 | DER collar 0.25 |
|----------|--------------|-----------------|
| WeSpeaker ResNet34 (default) | **23.89%** | **15.82%** |
| ERes2NetV2 (zh-cn common ONNX) | **53.84%** | **49.18%** |

**Verdict:** the shipped **zh-cn** ERes2NetV2 optional weights are **not** a
drop-in upgrade on English VoxConverse under our fbank front-end: both short-seg
EER and full-file DER regress hard. Keep the adapter for CJK / experiment
paths; do **not** make it default. A VoxCeleb-English ERes2Net export (if
ungated Apache) would need a separate measurement before any accuracy claim.

```bash
cargo run --release --features cli --bin polyvoice-measure -- embedder-short \
  --veri-list data/voxceleb1-subset/lists/veri_test.txt \
  --wav-root data/voxceleb1-subset \
  --der-dataset data/voxconverse-test --der-max-files 10 \
  --output benchmarks/results/embedder-short-eres2net.json
```
