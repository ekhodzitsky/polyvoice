"""pyannote.audio runner (the de-facto diarization baseline).

Skips unless `pyannote.audio` + `torch` are importable AND the gated
`speaker-diarization-3.1` pipeline loads (needs an HF access token in
``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` and prior license acceptance on the model
page). pyannote 3.1's published numbers are scored WITHOUT a forgiveness collar
and WITH overlap; run this runner through der.py at ``--collar 0`` to match.
"""

from __future__ import annotations

import os
import time

from .base import Turn


class PyannoteRunner:
    name = "pyannote-3.1"
    license = "MIT code; model weights gated (HF token + license acceptance)"

    def __init__(self, model: str = "pyannote/speaker-diarization-3.1"):
        self.model = model
        self._pipe = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            from pyannote.audio import Pipeline
        except Exception:
            return False
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        try:
            self._pipe = Pipeline.from_pretrained(self.model, use_auth_token=token)
            return self._pipe is not None
        except Exception as e:
            print(f"[pyannote] unavailable: {e}")
            return False

    def diarize(self, wav_path: str) -> tuple[list[Turn], float]:
        start = time.perf_counter()
        diar = self._pipe(wav_path)
        elapsed = time.perf_counter() - start
        turns = [(float(seg.start), float(seg.end), str(label))
                 for seg, _, label in diar.itertracks(yield_label=True)]
        return turns, elapsed
