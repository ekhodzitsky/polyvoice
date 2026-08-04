"""speakrs (Rust) runner for the cross-engine harness.

speakrs is the closest peer: pure-Rust, ONNX/CoreML, community-1 style VBx+PLDA.
Optional — skipped unless a ``speakrs-rttm`` binary (or SPEAKRS_RTTM_BIN) exists.

Build the in-tree helper (path dep on a local speakrs checkout):

    cargo build --release --manifest-path benchmarks/tools/speakrs-rttm/Cargo.toml
    # macOS CoreML:
    cargo build --release --manifest-path benchmarks/tools/speakrs-rttm/Cargo.toml --features coreml

Env:
    SPEAKRS_RTTM_BIN    — path to speakrs-rttm
    SPEAKRS_MODE        — default mode if not set on the runner (cpu/coreml/…)
    SPEAKRS_MODELS_DIR  — optional local model bundle
"""

from __future__ import annotations

import os
import subprocess
import time

from .base import Turn, find_executable, parse_rttm_text


class SpeakrsRunner:
    """Drive the speakrs-rttm helper for one execution mode."""

    def __init__(self, mode: str = "cpu", binary: str | None = None):
        self.mode = mode
        self.name = f"speakrs-{mode}"
        self.license = "Apache-2.0"
        self._binary = binary

    def is_available(self) -> bool:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_bin = os.environ.get("SPEAKRS_RTTM_BIN")
        candidates = []
        if env_bin:
            candidates.append(env_bin)
        candidates.extend(
            [
                os.path.join(
                    repo,
                    "benchmarks",
                    "tools",
                    "speakrs-rttm",
                    "target",
                    "release",
                    "speakrs-rttm",
                ),
                os.path.join(repo, "target", "release", "speakrs-rttm"),
                "speakrs-rttm",
            ]
        )
        self._binary = self._binary or find_executable(*candidates)
        return self._binary is not None

    def diarize(self, wav_path: str) -> tuple[list[Turn], float]:
        if not self._binary:
            raise RuntimeError("speakrs-rttm not located; call is_available() first")
        cmd = [self._binary, "--mode", self.mode]
        models = os.environ.get("SPEAKRS_MODELS_DIR")
        if models:
            cmd.extend(["--models-dir", models])
        cmd.append(wav_path)

        env = os.environ.copy()
        env["SPEAKRS_MODE"] = self.mode

        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-600:]
            raise RuntimeError(f"speakrs-rttm failed ({proc.returncode}): {err}")
        return parse_rttm_text(proc.stdout), elapsed
