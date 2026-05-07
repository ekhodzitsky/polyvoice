"""Smoke test: quantize_models.py on a trivial synthetic ONNX.

Builds a 1-conv 1x1x16 model in-memory, runs quantize_static via our reader,
asserts the output file exists and is at most equal in size (per-channel
quantization on a 1-element weight tensor may not actually shrink the file —
the test just guarantees the script runs end-to-end without crashing).
"""

from __future__ import annotations

import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

# These M5 smoke tests need the dev tooling — onnxruntime, numpy, librosa.
# In the python wheel CI workflow only the polyvoice wheel is installed, so
# skip cleanly when the deps are absent.
np = pytest.importorskip("numpy")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

ROOT = Path(__file__).resolve().parents[2]


def _build_synthetic_onnx(out_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 16])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 16])
    weight = helper.make_tensor(
        "W", TensorProto.FLOAT, [1, 1, 1], np.array([0.5], dtype=np.float32).tobytes(), raw=True
    )
    conv = helper.make_node("Conv", ["input", "W"], ["output"], pads=[0, 0])
    graph = helper.make_graph([conv], "smoke", [x], [y], initializer=[weight])
    model = helper.make_model(
        graph,
        producer_name="m5-smoke",
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=7,
    )
    onnx.save(model, str(out_path))


def _write_silence_wav(path: Path, n_samples: int = 16) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))


def test_quantize_smoke(tmp_path: Path) -> None:
    fp32 = tmp_path / "synth.onnx"
    int8 = tmp_path / "synth_int8.onnx"
    calib = tmp_path / "calib"
    calib.mkdir()
    _build_synthetic_onnx(fp32)
    _write_silence_wav(calib / "silence_a.wav")
    _write_silence_wav(calib / "silence_b.wav")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "quantize_models.py"),
            "--fp32", str(fp32),
            "--int8", str(int8),
            "--calib", str(calib),
            "--input-shape", "1,1,16",
            "--input-name", "input",
            "--num-samples", "2",
            "--seed", "1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Tiny synthetic models may not actually shrink under per-channel
    # quantization (one weight, no activation pruning to do). The smoke test
    # accepts both PASS (rc=0) and the trivial rc=2 "INT8 not smaller" exit
    # path — the goal is to verify the pipeline is wired end-to-end.
    if result.returncode != 0:
        assert result.returncode == 2, (
            f"unexpected exit {result.returncode}\nstdout=\n{result.stdout}\nstderr=\n{result.stderr}"
        )
        assert "not smaller" in result.stderr, result.stderr
    assert int8.exists(), "INT8 file not produced"
