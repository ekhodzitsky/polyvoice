"""Diarization benchmark runners.

polyvoice always runs when its binary is present; every competitor is optional
(``is_available()`` skips it when its stack/model is absent), so the suite runs
end-to-end with whatever the host has installed.
"""

from .diart_runner import DiartRunner
from .polyvoice_runner import PolyvoiceRunner
from .pyannote_runner import PyannoteRunner
from .sherpa_onnx_runner import SherpaOnnxRunner
from .speakrs_runner import SpeakrsRunner
from .whisperx_runner import WhisperXRunner

__all__ = [
    "PolyvoiceRunner",
    "SpeakrsRunner",
    "PyannoteRunner",
    "WhisperXRunner",
    "SherpaOnnxRunner",
    "DiartRunner",
]
