"""Basic tests for polyvoice Python bindings."""

import os
import pytest

MODEL_DIR = os.environ.get("POLYVOICE_MODEL_DIR")


@pytest.fixture
def pipeline():
    if not MODEL_DIR:
        pytest.skip("POLYVOICE_MODEL_DIR not set")
    import polyvoice

    return polyvoice.Pipeline(MODEL_DIR)


def test_version():
    import polyvoice

    assert polyvoice.__version__ == "0.6.0-alpha.0"


def test_pipeline_repr(pipeline):
    assert "Pipeline" in repr(pipeline)


def test_diarize_wav(pipeline, tmp_path):
    """Diarize a synthetic WAV file with two tones."""
    import struct
    import wave

    sr = 16000
    samples = []
    # 3 seconds of 200 Hz tone
    for i in range(sr * 3):
        t = i / sr
        samples.append(int(32000 * __import__("math").sin(2 * 3.14159 * 200 * t)))
    # 0.5s silence
    samples.extend([0] * int(sr * 0.5))
    # 3 seconds of 800 Hz tone
    for i in range(sr * 3):
        t = i / sr
        samples.append(int(32000 * __import__("math").sin(2 * 3.14159 * 800 * t)))

    wav_path = str(tmp_path / "test.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    turns = pipeline(wav_path)
    assert isinstance(turns, list)
    for turn in turns:
        assert hasattr(turn, "speaker")
        assert hasattr(turn, "start")
        assert hasattr(turn, "end")
        assert turn.end > turn.start


def test_diarize_samples(pipeline):
    """Diarize from a list of f32 samples."""
    import math

    sr = 16000
    samples = [
        0.5 * math.sin(2 * math.pi * 300 * i / sr) for i in range(sr * 3)
    ]
    turns = pipeline(samples)
    assert isinstance(turns, list)


def test_missing_models():
    import polyvoice

    with pytest.raises(FileNotFoundError):
        polyvoice.Pipeline("/nonexistent/path")
