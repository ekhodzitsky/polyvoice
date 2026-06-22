# WavLM + Conformer-EEND max-accuracy backend — spike & go/no-go

**Date:** 2026-06-20 · **Verdict: NO-GO** on the DiariZen-style WavLM+Conformer
EEND backend, on **licensing** grounds. Technical export/runtime is green; the
accuracy prize (the EEND head) is non-commercial and unusable in MIT polyvoice.

## Why we looked

The shipped clustering stack (powerset segmentation → WeSpeaker ResNet34 → AHC /
VBx) plateaus around legacy parity (~12.9% no-collar VoxConverse-test; VBx beats
AHC on dev but not legacy on held-out test). The modern SOTA (~5–8% DER) is the
DiariZen recipe: WavLM-large features (structurally pruned) → Conformer EEND head
→ VBx stitching. This spike de-risked adopting it as an opt-in "accuracy" profile
(explicitly heavy, not wasm-clean, default unchanged).

## Findings (measured)

**1. ONNX export — RESOLVED (green).** WavLM-large exports to ONNX cleanly; this
is not a research risk:
- `onnx-community/wavlm-large-ONNX`: `model.onnx` 1263 MB (fp32), `model_fp16`
  632 MB, `model_q4` / `model_bnb4` 327 MB, `model_q4f16` 234 MB.
- `yunfengwang/wavlm-large-onnx` (Apache-2.0): `wavlm_large.onnx` 1297 MB.
- HuggingFace Optimum officially supports the export
  (`ORTModelForAudioFrameClassification` lists `wavlm`).

**2. Runtime / op-coverage — RESOLVED (green).** `ort` 2.0.0-rc.12 is a Rust
binding to Microsoft **ONNX Runtime** (our `download-binaries` feature pulls the
ONNX Runtime prebuilts). WavLM is a standard transformer (LayerNorm, GELU,
relative-position attention, grouped conv) fully supported by ONNX Runtime, and
the onnx-community exports are ONNX-Runtime-targeted — so `ort` runs them. The
roadmap's "ort inference unverified" risk is moot once the runtime is ONNX
Runtime itself. (Local env: `onnxruntime` 1.24.4 + `onnx` 1.21 present; `torch`
and `transformers` absent, so no in-house PyTorch-parity re-export was done — but
upstream exports already exist, so that is not on the critical path.)

**3. License — BLOCKER (the no-go).** The efficient SOTA model,
`BUT-FIT/diarizen-wavlm-large-s80-md` (pruned WavLM-s80 + Conformer EEND;
316.6M → 63.3M params, 17.8G → 3.8G MACs/s), is **CC-BY-NC-4.0** — non-commercial
weights. polyvoice ships MIT and must stay commercial-friendly, so the DiariZen
EEND head (which delivers the ~5–8% DER) cannot be shipped or relied upon.

**4. Footprint.** The default build is ~30 MB. The *permissive* option is the
**full** WavLM-large (234 MB q4f16 … 1.3 GB fp32) — the small one (pruned-s80,
~63M) is the non-commercial one. A heavy opt-in "accuracy" profile is allowed by
the roadmap, but the favourable size/accuracy point (pruned + EEND) is exactly
the license-blocked artifact.

## Verdict

**NO-GO** on the DiariZen-style WavLM + Conformer EEND backend. Export and `ort`
runtime are not blockers; the SOTA accuracy lives in a **non-commercial
(CC-BY-NC) EEND head**, incompatible with MIT polyvoice.

A permissive fallback exists but is weak: full `microsoft/wavlm-large` (MIT-class;
Apache-2.0 ONNX at `yunfengwang/wavlm-large-onnx`) could replace WeSpeaker as a
heavy opt-in **embedding extractor** feeding the existing AHC/VBx clustering. But
(a) it is heavy (≥234 MB), and (b) the SOTA gain comes from the (non-commercial)
EEND head, not the embeddings alone — so a 5× footprint for an unproven
embedding-only delta is a poor trade. Not pursued now; revisit only if a
permissive EEND head appears or an embedding-only A/B vs WeSpeaker shows a large,
durable DER win.

## Explicit non-goals (max-accuracy backends ruled out)

- **DiariZen WavLM+Conformer EEND** — CC-BY-NC-4.0 (non-commercial weights).
- **Sortformer** — ONNX export broken; hard 4-speaker cap.
- **ReDimNet** — no official ONNX export.

## What still moves accuracy (commercial-clean)

Structural clustering levers were comprehensively explored and largely exhausted
(post-hoc prune, threshold/auto-threshold, NME-SC, VBx GMM/HMM, within-file
AS-norm). The remaining commercial-clean candidates are external-cohort AS-norm
(needs a shipped imposter-embedding artifact; roadmap risk: "barely moves
cosine") and a permissive embedding upgrade if one with a clear DER win emerges.
