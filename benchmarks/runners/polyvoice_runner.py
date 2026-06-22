"""polyvoice runner — drives the shipped `polyvoice` CLI to emit RTTM.

Uses only the public CLI surface (`polyvoice diarize <wav> --format rttm`), so it
measures exactly what a user gets. Two variants are exposed:

- ``polyvoice`` — the shipped legacy default (Silero VAD + WeSpeaker + AHC).
- ``polyvoice-v2`` — the powerset-segmentation pipeline (CLI ``--v2``).

The opt-in VBx clusterer is a library feature not exposed on the CLI; its numbers
are produced by `polyvoice-bench` and reported separately in the docs.
"""

from __future__ import annotations

import os
import subprocess
import time

from .base import Turn, find_executable, parse_rttm_text


class PolyvoiceRunner:
    def __init__(self, variant: str = "legacy", profile: str = "balanced",
                 threshold: float | None = None, binary: str | None = None):
        self.variant = variant
        self.profile = profile
        self.threshold = threshold
        self.name = "polyvoice" if variant == "legacy" else f"polyvoice-{variant}"
        self.license = "MIT"
        self._binary = binary

    def is_available(self) -> bool:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._binary = self._binary or find_executable(
            os.path.join(repo, "target", "release", "polyvoice"),
            os.path.join(repo, "target", "debug", "polyvoice"),
            "polyvoice",
        )
        return self._binary is not None

    def diarize(self, wav_path: str) -> tuple[list[Turn], float]:
        if not self._binary:
            raise RuntimeError("polyvoice binary not located; call is_available() first")
        cmd = [self._binary, "diarize", wav_path, "--format", "rttm",
               "--quiet", "--profile", self.profile]
        if self.variant == "v2":
            cmd.append("--v2")
        if self.threshold is not None:
            cmd.append(f"--threshold={self.threshold}")
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            raise RuntimeError(f"polyvoice diarize failed ({proc.returncode}): {proc.stderr[-400:]}")
        return parse_rttm_text(proc.stdout), elapsed
