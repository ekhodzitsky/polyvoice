# Open-source speaker diarization competitors

> Research snapshot: 2026-06-23. Stars are approximate and change daily.

Polyvoice positions itself as a **pure-Rust, CPU-only, MIT-licensed, ungated** speaker diarization engine with first-class streaming and a small (~30 MB) footprint. The list below compares the main open-source alternatives and what polyvoice can learn from them.

## Polyvoice baseline

| Property | Value |
|---|---|
| Repository | https://github.com/ekhodzitsky/polyvoice |
| Language | Rust (pure-Rust core, ONNX via `ort`) |
| License | MIT (code and shipped models) |
| Deployment | CPU-first, ~30 MB, no Python runtime |
| Bindings | Rust library, Python (maturin), C FFI, CLI, MCP server |
| Streaming | First-class `OnlineDiarizer` |
| DER benchmark | 18.5% on VoxConverse-test (collar 0, overlap-scored) |

## Competitor comparison

| Competitor | Stars | Stack | Code license | Model license | Main advantage over polyvoice | Where polyvoice wins |
|---|---|---|---|---|---|---|
| **pyannote.audio** | 10 160 | Python / PyTorch | MIT | Gated (HuggingFace token + terms) | SOTA accuracy (DER ~11.2% on VoxConverse), mature pipelines, fine-tuning | CPU-only, no Python, no HF token, streaming, ~30 MB |
| **WhisperX** | 22 620 | Python / faster-whisper / PyTorch | BSD-2-Clause | pyannote gated + Whisper MIT | ASR + diarization + word-level timestamps in one CLI, 99 languages | Pure diarization, Rust-native, CPU, MIT without gated models |
| **NVIDIA NeMo** | 17 459 | Python / PyTorch / CUDA | Apache-2.0 | NGC / Riva terms (not pure OSS) | GPU SOTA (MSDD, Sortformer), enterprise ecosystem, ASR/TTS/LLM | Pure-Rust, CPU, MIT, simple deploy |
| **FunASR** | ~18 500 | Python / PyTorch (+ C++ runtime) | MIT | Custom FunASR Model License | Industrial ASR+VAD+diarization, 50+ languages, Docker/API | Size, license simplicity, Rust integration |
| **Kaldi** | 15 418 | C++ / Shell / Python | Apache-2.0 | Apache-2.0, some models trained on LDC data | Research standard, x-vector + PLDA recipes, ASR+speaker | Ready pipeline, streaming, no C++ build |
| **sherpa-onnx** | 13 132 | C++ / ONNX | Apache-2.0 | pyannote gated / 3D-Speaker Apache / Silero MIT | Offline SDK, 12 language bindings, ASR+TTS+VAD+diarization, edge platforms | Pure-Rust core, streaming diarization, fewer dependencies |
| **SpeechBrain** | 11 644 | Python / PyTorch | Apache-2.0 | Apache-2.0 | Universal speech toolkit, 200+ recipes, embedding training | CPU-only deploy, Rust, size, streaming |
| **diart** | 1 988 | Python / PyTorch / RxPY | MIT | pyannote gated | Real-time streaming diarization, DER 16.8% on VoxConverse | CPU-first, no Python/HF token, Rust-native |
| **Resemblyzer** | 3 274 | Python / PyTorch | Apache-2.0 | Apache-2.0 / MIT | Popular embeddings, multi-task use cases | Ready diarization pipeline, streaming, Rust |
| **VBx** | 287 | Python / PyTorch / Kaldi | Apache-2.0 | Possible CN-Celeb restrictions | Classic SOTA baseline (CALLHOME 4.42%), flexibility | Easier to use, streaming, Rust, no Kaldi |
| **simple-diarizer** | 157 | Python / PyTorch | **GPL-3.0** | MIT / Apache | Minimal Python pipeline | GPL copyleft, Rust, CPU-only, streaming |

## What polyvoice can borrow

1. **Accuracy improvements from pyannote / NeMo / EEND**
   - Overlap-aware segmentation quality and automatic speaker-count estimation.
   - End-to-end neural diarization back-ends (Sortformer, EEND) as optional ONNX profiles.
   - Better embedding extractors (WavLM, ECAPA-TDNN v3, CAM++ improvements).
   - These are already on the roadmap, but the gap to pyannote 11.2% is the biggest risk.

2. **Integrated "who said what" product shape from WhisperX / FunASR**
   - Word-level timestamps + speaker labels are the user-facing end state.
   - Polyvoice already has attribution (`attribution` module) and `polyvoice-transcribe`; double down on this instead of competing as a raw diarization library alone.

3. **Packaging and platform reach from sherpa-onnx**
   - Pre-built binaries, Docker images, mobile examples, and broader language bindings lower friction.
   - Sherpa-onnx proves there is demand for local, offline, multi-platform speech stacks.

4. **Streaming latency/accuracy trade-offs from diart**
   - Diart shows real-time diarization is possible with tunable latency.
   - Polyvoice can publish latency vs. DER curves and recommend profiles for real-time use.

5. **Training recipes and data augmentation from SpeechBrain**
   - Fine-tuning recipes for domain adaptation (meetings, call centers, podcasts).
   - Augmentation (reverb, noise, speed) could improve robustness without growing runtime.

6. **CJK/industrial coverage from FunASR**
   - Chinese, Japanese, Korean ASR + diarization demand is large.
   - Consider a SenseVoice / Paraformer companion ASR backend, not just Parakeet.

7. **Simplicity from simple-diarizer**
   - A one-liner Python API (`pipe = Diarizer(); pipe("file.wav")`) helps adoption.
   - Polyvoice Python bindings can emulate this without sacrificing the Rust core.

## Strategic assessment

**Should polyvoice continue? Yes.**

Polyvoice owns a defensible niche that the Python/C++ incumbents do not serve well: a **pure-Rust, CPU-only, MIT-licensed, ungated, streaming diarization engine** that embeds into Rust applications without a Python runtime. Competitors are stronger on accuracy, ecosystem, and bindings, but they are heavier, GPU-oriented, and often gated or commercially restricted.

The main threat is the **accuracy gap**. If polyvoice stays at 18.5% DER while pyannote sits at 11.2%, accuracy-sensitive users will accept the heavier stack. The roadmap already targets this with VBx, PLDA, better segmentation, and EEND/Sortformer spikes. The priority should be:

1. **Close the DER gap** with optional ONNX accuracy profiles (EEND/Sortformer, WavLM/CAM++ embeddings) while keeping the CPU-first default.
2. **Own the deployability story** — pre-built binaries, Docker, WASM, mobile, and clear "no Python, no GPU, no token" messaging.
3. **Productize who-said-what** — diarization + ASR attribution as the primary user-facing surface, not just raw speaker turns.
4. **Stay license-clean** — do not introduce gated or copyleft models; use permissive weights or ship conversion/provenance docs.

If the project cannot narrow the accuracy gap to within ~3–5 DER points of pyannote on standard benchmarks within the next major releases, it risks remaining a curiosity rather than a production option. With the current trajectory, it is worth continuing.
