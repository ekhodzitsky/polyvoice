"""Basic tests for polyvoice Python bindings (legacy v0.5 API)."""

import os
import pytest

MODEL_DIR = os.environ.get("POLYVOICE_MODEL_DIR")


@pytest.fixture
def pipeline():
    if not MODEL_DIR:
        pytest.skip("POLYVOICE_MODEL_DIR not set")
    import polyvoice

    return polyvoice.Pipeline.balanced(MODEL_DIR)


def test_version():
    import polyvoice

    assert polyvoice.__version__ == "0.6.0-alpha.0"


def test_pipeline_repr(pipeline):
    assert "Pipeline" in repr(pipeline)


def test_diarize_samples(pipeline):
    """Diarize from a list of f32 samples."""
    import math
    import struct
    import wave

    sr = 16000
    samples = [
        0.5 * math.sin(2 * math.pi * 300 * i / sr) for i in range(sr * 3)
    ]

    result = pipeline.run(samples, sr)
    assert isinstance(result, dict)
    assert "num_speakers" in result
    assert "turns" in result
    assert isinstance(result["turns"], list)
    for turn in result["turns"]:
        assert "speaker" in turn
        assert "start" in turn
        assert "end" in turn
        assert turn["end"] > turn["start"]


def test_diarize_wav(pipeline, tmp_path):
    """Diarize a synthetic WAV file with two tones."""
    import struct
    import wave
    import math

    sr = 16000
    samples = []
    # 3 seconds of 200 Hz tone
    for i in range(sr * 3):
        t = i / sr
        samples.append(int(32000 * math.sin(2 * math.pi * 200 * t)))
    # 0.5s silence
    samples.extend([0] * int(sr * 0.5))
    # 3 seconds of 800 Hz tone
    for i in range(sr * 3):
        t = i / sr
        samples.append(int(32000 * math.sin(2 * math.pi * 800 * t)))

    wav_path = str(tmp_path / "test.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    # Read WAV back as f32 samples
    with wave.open(wav_path, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == sr
        raw = w.readframes(w.getnframes())
        import array
        pcm = array.array("h", raw)
        f32_samples = [s / 32768.0 for s in pcm]

    result = pipeline.run(f32_samples, sr)
    assert isinstance(result, dict)
    assert "num_speakers" in result
    assert "turns" in result
    assert isinstance(result["turns"], list)
    for turn in result["turns"]:
        assert "speaker" in turn
        assert "start" in turn
        assert "end" in turn
        assert turn["end"] > turn["start"]


def test_missing_models(tmp_path):
    import polyvoice

    # Create a file — passing it as cache_dir must fail because
    # create_dir_all on a file path returns "Not a directory".
    fake_dir = tmp_path / "not_a_dir"
    fake_dir.write_text("i am a file")
    with pytest.raises(RuntimeError):
        polyvoice.Pipeline.balanced(str(fake_dir))
