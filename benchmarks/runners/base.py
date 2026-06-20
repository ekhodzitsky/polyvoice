"""Diarization runner interface + shared helpers.

A runner turns a WAV file into a speaker diarization (a list of
``(start, end, speaker)`` turns) and reports how long that took. Every runner is
*optional*: ``is_available()`` returns ``False`` — and the engine is silently
skipped — when its stack, binary, or model is absent, exactly like the gigastt
ASR harness. The suite therefore runs end-to-end with whatever engines the host
has installed, and competitor rows simply do not appear until their stack is
present (the published numbers are cited in the docs until then).
"""

from __future__ import annotations

import os
import shutil
from typing import Protocol, runtime_checkable

Turn = tuple[float, float, str]  # (start, end, speaker_label)


@runtime_checkable
class DiarizationRunner(Protocol):
    name: str
    license: str  # SPDX-ish; note "(gated)" / "(non-commercial)" where it applies

    def is_available(self) -> bool:
        """Cheap probe; may build/download/load. False ⇒ skip this engine."""
        ...

    def diarize(self, wav_path: str) -> tuple[list[Turn], float]:
        """Return (turns, processing_seconds) for one WAV."""
        ...


def write_rttm(turns: list[Turn], file_id: str, path: str) -> None:
    """Write turns as a standard RTTM (one SPEAKER line per turn)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for start, end, spk in turns:
            dur = max(0.0, end - start)
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n")


def parse_rttm_text(text: str) -> list[Turn]:
    """Parse RTTM text (e.g. an engine's stdout) into turns."""
    turns: list[Turn] = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur, spk = float(p[3]), float(p[4]), p[7]
            if dur > 0:
                turns.append((start, start + dur, spk))
    return turns


def find_executable(*candidates: str) -> str | None:
    """Return the first candidate that exists on PATH or as a file path."""
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
        found = shutil.which(c)
        if found:
            return found
    return None
