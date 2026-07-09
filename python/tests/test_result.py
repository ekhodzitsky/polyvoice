"""Model-free tests for the typed DiarizationResult and its projections.

These construct a result from canonical JSON (no ONNX models needed), so they
run everywhere — including CI without a model cache.
"""

import json

import pytest

from polyvoice import DiarizationResult

FIXTURE = {
    "schema_version": "diarization-result-v1",
    "segments": [],
    "turns": [
        {"speaker": 0, "time": {"start": 0.5, "end": 2.8}},
        {"speaker": 1, "time": {"start": 3.0, "end": 4.5}, "text": "hello"},
    ],
    "num_speakers": 2,
    "audio": {"duration_secs": 5.0, "sample_rate": 16000},
    "provenance": {
        "version": "0.9.0",
        "profile": "balanced",
        "segmenter": "",
        "embedder": "",
        "clusterer": "",
    },
    "speakers": [
        {"label": "SPEAKER_00", "id": 0, "total_speech_s": 2.3, "turn_count": 1},
        {"label": "SPEAKER_01", "id": 1, "total_speech_s": 1.5, "turn_count": 1},
    ],
}


@pytest.fixture
def result():
    return DiarizationResult.from_json(json.dumps(FIXTURE))


def test_from_json_invalid_raises_value_error():
    with pytest.raises(ValueError):
        DiarizationResult.from_json("not json")


def test_properties(result):
    assert result.num_speakers == 2
    assert result.schema_version == "diarization-result-v1"
    assert "num_speakers=2" in repr(result)

    turns = result.turns
    assert turns[0] == {"speaker": 0, "start": 0.5, "end": 2.8, "text": None}
    assert turns[1] == {"speaker": 1, "start": 3.0, "end": 4.5, "text": "hello"}

    speakers = result.speakers
    assert speakers[0]["label"] == "SPEAKER_00"
    assert speakers[1]["turn_count"] == 1


def test_json_round_trip_and_dict_parity(result):
    doc = json.loads(result.to_json())
    assert doc["schema_version"] == "diarization-result-v1"
    assert doc["num_speakers"] == 2
    # text is skip_serializing_if=None: absent on the untranscribed turn
    assert "text" not in doc["turns"][0]
    assert doc["turns"][1]["text"] == "hello"
    assert result.to_dict() == doc

    # round-trip: from_json(to_json) is lossless
    again = DiarizationResult.from_json(result.to_json())
    assert again.to_json() == result.to_json()


def test_to_rttm_golden(result):
    assert result.to_rttm() == (
        "SPEAKER audio 1 0.500 2.300 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER audio 1 3.000 1.500 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
    )
    assert result.to_rttm(file_id="meeting1").startswith("SPEAKER meeting1 1")


def test_to_rttm_rejects_bad_file_id(result):
    """RTTM is whitespace-delimited: empty or whitespace ids would corrupt columns."""
    for bad in ["", "my meeting", "a\tb", "a\nb"]:
        with pytest.raises(ValueError):
            result.to_rttm(file_id=bad)


def test_to_srt_golden(result):
    assert result.to_srt() == (
        "1\n"
        "00:00:00,500 --> 00:00:02,800\n"
        "SPEAKER_00\n"
        "\n"
        "2\n"
        "00:00:03,000 --> 00:00:04,500\n"
        "SPEAKER_01: hello\n"
        "\n"
    )


def test_to_vtt_golden(result):
    assert result.to_vtt() == (
        "WEBVTT\n"
        "\n"
        "00:00:00.500 --> 00:00:02.800\n"
        "SPEAKER_00\n"
        "\n"
        "00:00:03.000 --> 00:00:04.500\n"
        "<v SPEAKER_01>hello</v>\n"
        "\n"
    )


def test_to_txt_golden(result):
    assert result.to_txt() == (
        "[0.50 - 2.80] SPEAKER_00\n"
        "[3.00 - 4.50] SPEAKER_01: hello\n"
    )


def test_v1_consumer_json_without_optional_fields_parses():
    """A minimal pre-rollup v1 document (no audio/provenance/speakers) still loads."""
    minimal = {
        "segments": [],
        "turns": [{"speaker": 0, "time": {"start": 0.0, "end": 1.0}}],
        "num_speakers": 1,
    }
    r = DiarizationResult.from_json(json.dumps(minimal))
    assert r.num_speakers == 1
    assert r.schema_version == "diarization-result-v1"
