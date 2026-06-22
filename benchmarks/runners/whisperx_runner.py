"""WhisperX diarization runner.

WhisperX diarizes via pyannote under the hood, so its diarization quality tracks
pyannote's; this runner exists to measure that path as users actually invoke it.
Skips unless `whisperx` + `torch` import and the gated pyannote model loads.
"""

from __future__ import annotations

import os
import time

from .base import Turn


class WhisperXRunner:
    name = "whisperx"
    license = "BSD-4-Clause code; uses gated pyannote weights (HF token)"

    def __init__(self):
        self._pipe = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import whisperx  # noqa: F401
        except Exception:
            return False
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        try:
            from whisperx.diarize import DiarizationPipeline
            self._pipe = DiarizationPipeline(use_auth_token=token, device="cpu")
            return True
        except Exception as e:
            print(f"[whisperx] unavailable: {e}")
            return False

    def diarize(self, wav_path: str) -> tuple[list[Turn], float]:
        start = time.perf_counter()
        df = self._pipe(wav_path)  # pandas DataFrame: start,end,speaker
        elapsed = time.perf_counter() - start
        turns = [(float(r["start"]), float(r["end"]), str(r["speaker"]))
                 for _, r in df.iterrows()]
        return turns, elapsed
