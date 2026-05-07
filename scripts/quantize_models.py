"""M5 — INT8 static quantization for polyvoice ONNX models.

Usage:
    python quantize_models.py \
        --fp32 models/powerset_fp32.onnx \
        --int8 models/int8/powerset_int8.onnx \
        --calib data/voxconverse-dev/audio \
        --num-samples 500 \
        --seed 42 \
        --input-shape 1,1,160000 \
        --sample-rate 16000

Static quantization with per-channel weights, asymmetric activations, and a
MinMax calibration method. The CalibrationDataReader streams 10-second chunks
from random WAV files in the calibration directory.

Acceptance gates (see scripts/validate_int8.py and spec §9.4) are checked by
the validation script, not here. This script only produces the artifact.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import librosa
except ImportError as exc:
    sys.exit(f"librosa missing — `pip install -r python/requirements-dev.txt` ({exc})")

try:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
except ImportError as exc:
    sys.exit(f"onnxruntime.quantization missing — install onnxruntime>=1.20 ({exc})")


def _list_wav_files(calib_dir: Path) -> list[Path]:
    files = sorted([p for p in calib_dir.rglob("*.wav") if p.is_file()])
    if not files:
        raise SystemExit(f"No .wav files found under {calib_dir}")
    return files


def _load_chunk(path: Path, sample_rate: int, num_samples: int) -> np.ndarray:
    """Load the first `num_samples` samples of audio at `sample_rate`.

    Pads with zeros if shorter; truncates if longer. Returns float32 PCM in [-1, 1].
    """
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    if audio.shape[0] >= num_samples:
        return audio[:num_samples]
    pad = np.zeros(num_samples - audio.shape[0], dtype=np.float32)
    return np.concatenate([audio, pad])


class VoxConverseChunkReader(CalibrationDataReader):
    """Streams `(1, 1, T)` or `(1, mels, T)` tensors from VoxConverse-dev WAVs.

    The output tensor matches whatever shape the caller passes via `input_shape`.
    For raw-audio models (powerset) we feed `(1, 1, T)`; for fbank-input
    embedders (CAM++/ResNet34) the validation script computes mel features
    inline, but here calibration just feeds raw audio chunks shaped to the
    model's first dim — embedder ONNX graphs that ingest fbank will need
    their `--input-shape` set to e.g. `1,80,300` and audio will be reshaped
    accordingly (with zero-padding when the shape doesn't match T).
    """

    def __init__(
        self,
        calib_dir: Path,
        input_name: str,
        sample_rate: int,
        chunk_samples: int,
        num_samples: int,
        seed: int,
        target_shape: tuple[int, ...],
    ) -> None:
        wavs = _list_wav_files(calib_dir)
        rng = random.Random(seed)
        if len(wavs) > num_samples:
            wavs = rng.sample(wavs, num_samples)
        self._wavs = wavs
        self._iter = iter(wavs)
        self._input_name = input_name
        self._sample_rate = sample_rate
        self._chunk_samples = chunk_samples
        self._target_shape = target_shape
        self._index = 0
        self._total = len(wavs)

    def get_next(self) -> dict[str, Any] | None:  # type: ignore[override]
        try:
            path = next(self._iter)
        except StopIteration:
            return None
        chunk = _load_chunk(path, self._sample_rate, self._chunk_samples)
        # Reshape to target shape; pad / truncate the time-axis as needed.
        flat = chunk.flatten()
        target_len = int(np.prod(self._target_shape))
        if flat.shape[0] < target_len:
            flat = np.concatenate([flat, np.zeros(target_len - flat.shape[0], dtype=np.float32)])
        flat = flat[:target_len]
        tensor = flat.reshape(self._target_shape).astype(np.float32)
        self._index += 1
        if self._index % 50 == 0 or self._index == self._total:
            print(f"  calibrated {self._index}/{self._total} files", file=sys.stderr)
        return {self._input_name: tensor}

    def rewind(self) -> None:
        self._iter = iter(self._wavs)
        self._index = 0


def _parse_shape(spec: str) -> tuple[int, ...]:
    return tuple(int(x) for x in spec.split(","))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", required=True, type=Path)
    p.add_argument("--int8", required=True, type=Path)
    p.add_argument("--calib", required=True, type=Path, help="Dir with .wav calibration files")
    p.add_argument("--input-shape", required=True, help="comma-separated, e.g. 1,1,160000")
    p.add_argument("--input-name", default=None, help="ONNX input tensor name (default: first input)")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--exclude-nodes",
        default="",
        help="Comma-separated list of node names to skip; pass when a layer crashes the quantizer",
    )
    args = p.parse_args(argv)

    if not args.fp32.exists():
        return _die(f"FP32 model missing: {args.fp32}")
    if not args.calib.exists():
        return _die(f"Calibration dir missing: {args.calib}")
    args.int8.parent.mkdir(parents=True, exist_ok=True)

    shape = _parse_shape(args.input_shape)
    if len(shape) < 2:
        return _die(f"--input-shape must have ≥ 2 dims, got {shape}")
    chunk_samples = max(int(np.prod(shape)), 1)

    if args.input_name is None:
        import onnx

        m = onnx.load(str(args.fp32), load_external_data=False)
        args.input_name = m.graph.input[0].name
        print(f"  resolved input name: {args.input_name}", file=sys.stderr)

    reader = VoxConverseChunkReader(
        calib_dir=args.calib,
        input_name=args.input_name,
        sample_rate=args.sample_rate,
        chunk_samples=chunk_samples,
        num_samples=args.num_samples,
        seed=args.seed,
        target_shape=shape,
    )

    nodes_to_exclude = [n for n in args.exclude_nodes.split(",") if n]

    print(f"Quantizing {args.fp32} -> {args.int8}", file=sys.stderr)
    print(f"  calibration: {len(reader._wavs)} files (seed={args.seed})", file=sys.stderr)
    print(f"  exclude_nodes: {nodes_to_exclude or '<none>'}", file=sys.stderr)

    t0 = time.time()
    quantize_static(
        model_input=str(args.fp32),
        model_output=str(args.int8),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=nodes_to_exclude,
    )
    elapsed = time.time() - t0

    fp32_bytes = args.fp32.stat().st_size
    int8_bytes = args.int8.stat().st_size
    ratio = fp32_bytes / int8_bytes if int8_bytes else 0
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)
    print(f"  FP32 size : {fp32_bytes:_} bytes", file=sys.stderr)
    print(f"  INT8 size : {int8_bytes:_} bytes (compression {ratio:.2f}x)", file=sys.stderr)

    if int8_bytes >= fp32_bytes:
        return _die(
            f"INT8 size {int8_bytes:_} not smaller than FP32 size {fp32_bytes:_} — "
            "quantization likely had no effect"
        )
    return 0


def _die(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
