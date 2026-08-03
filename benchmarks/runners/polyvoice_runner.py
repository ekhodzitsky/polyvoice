"""polyvoice runner — drives the shipped `polyvoice` CLI to emit RTTM.

Uses only the public CLI surface (`polyvoice diarize <wav> --format rttm`).

Variants:
- ``default`` → name ``polyvoice`` — CLI default since 0.11 (**v2 + VBx**)
- ``v2`` → name ``polyvoice-v2`` — same path (``--v2`` no-op for old scripts)
- ``legacy`` → name ``polyvoice-legacy`` — ``--legacy``
- ``ahc`` → name ``polyvoice-ahc`` — v2 + ``--clusterer ahc``

Harness RTF includes cold CLI load; warm RTF is `polyvoice-bench` separately.
"""

from __future__ import annotations

import os
import subprocess
import time

from .base import Turn, find_executable, parse_rttm_text

_NAME = {
    "default": "polyvoice",
    "v2": "polyvoice-v2",
    "legacy": "polyvoice-legacy",
    "ahc": "polyvoice-ahc",
    "vbx": "polyvoice-vbx",
}


class PolyvoiceRunner:
    def __init__(
        self,
        variant: str = "default",
        profile: str = "balanced",
        threshold: float | None = None,
        binary: str | None = None,
    ):
        self.variant = variant
        self.profile = profile
        self.threshold = threshold
        self.name = _NAME.get(variant, f"polyvoice-{variant}")
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
        cmd = [
            self._binary,
            "diarize",
            wav_path,
            "--format",
            "rttm",
            "--quiet",
            "--profile",
            self.profile,
        ]
        if self.variant == "legacy":
            cmd.append("--legacy")
        elif self.variant == "ahc":
            cmd.extend(["--clusterer", "ahc"])
        elif self.variant == "v2":
            cmd.append("--v2")
        elif self.variant in ("default", "vbx"):
            cmd.extend(["--clusterer", "vbx"])
        if self.threshold is not None:
            cmd.append(f"--threshold={self.threshold}")

        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            raise RuntimeError(
                f"polyvoice diarize failed ({proc.returncode}): {proc.stderr[-400:]}"
            )
        return parse_rttm_text(proc.stdout), elapsed
