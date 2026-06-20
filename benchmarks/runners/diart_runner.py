"""diart runner (online/streaming diarization, scored offline over the file).

diart is the streaming-diarization comparison point; here it is run over a whole
file and scored like the offline systems. Skips unless `diart` + `torch` import
and the gated pyannote models load (HF token).
"""

from __future__ import annotations

import time

from .base import Turn


class DiartRunner:
    name = "diart"
    license = "MIT code; uses gated pyannote weights (HF token)"

    def __init__(self):
        self._pipeline_cls = None
        self._config = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            from diart import SpeakerDiarization, SpeakerDiarizationConfig
        except Exception:
            return False
        try:
            self._pipeline_cls = SpeakerDiarization
            self._config = SpeakerDiarizationConfig()
            return True
        except Exception as e:
            print(f"[diart] unavailable: {e}")
            return False

    def diarize(self, wav_path: str) -> tuple[list[Turn], float]:
        from diart.inference import Benchmark  # noqa: F401
        from diart.sources import FileAudioSource
        from diart import SpeakerDiarization
        pipeline = SpeakerDiarization(self._config)
        source = FileAudioSource(wav_path, sample_rate=16000)
        start = time.perf_counter()
        prediction = pipeline(list(source.stream))  # simplistic offline drive
        elapsed = time.perf_counter() - start
        annotation = prediction[0] if isinstance(prediction, tuple) else prediction
        turns = [(float(seg.start), float(seg.end), str(label))
                 for seg, _, label in annotation.itertracks(yield_label=True)]
        return turns, elapsed
