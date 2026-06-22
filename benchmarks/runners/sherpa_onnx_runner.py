"""sherpa-onnx offline speaker-diarization runner.

The closest pure-ONNX competitor: pyannote segmentation + a speaker embedder +
clustering, run through sherpa-onnx. Skips unless `sherpa_onnx` imports and the
model paths are provided via env:

    SHERPA_SEGMENTATION_MODEL  — pyannote segmentation .onnx
    SHERPA_EMBEDDING_MODEL     — speaker-embedding .onnx (e.g. 3D-Speaker / WeSpeaker)

Set SHERPA_NUM_SPEAKERS to a positive int to fix the count; otherwise the
clustering threshold (SHERPA_CLUSTER_THRESHOLD, default 0.5) drives auto-count.
"""

from __future__ import annotations

import os
import time

from .base import Turn


class SherpaOnnxRunner:
    name = "sherpa-onnx"
    license = "Apache-2.0 (models vary)"

    def __init__(self):
        self._sd = None

    def is_available(self) -> bool:
        try:
            import sherpa_onnx  # noqa: F401
        except Exception:
            return False
        seg = os.environ.get("SHERPA_SEGMENTATION_MODEL")
        emb = os.environ.get("SHERPA_EMBEDDING_MODEL")
        if not (seg and emb and os.path.isfile(seg) and os.path.isfile(emb)):
            print("[sherpa-onnx] set SHERPA_SEGMENTATION_MODEL + SHERPA_EMBEDDING_MODEL to enable")
            return False
        try:
            import sherpa_onnx
            cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg),
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=int(os.environ.get("SHERPA_NUM_SPEAKERS", "-1")),
                    threshold=float(os.environ.get("SHERPA_CLUSTER_THRESHOLD", "0.5")),
                ),
            )
            self._sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
            return True
        except Exception as e:
            print(f"[sherpa-onnx] unavailable: {e}")
            return False

    def diarize(self, wav_path: str) -> tuple[list[Turn], float]:
        import soundfile as sf
        samples, sr = sf.read(wav_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        start = time.perf_counter()
        result = self._sd.process(samples).sort_by_start_time()
        elapsed = time.perf_counter() - start
        turns = [(float(s.start), float(s.end), f"spk{s.speaker}") for s in result]
        return turns, elapsed
