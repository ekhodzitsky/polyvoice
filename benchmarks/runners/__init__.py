"""Diarization benchmark runners.

polyvoice always runs; every competitor is optional (``is_available()`` skips it
when its stack/model is absent), so the suite runs end-to-end with whatever the
host has installed.
"""

from .polyvoice_runner import PolyvoiceRunner
from .pyannote_runner import PyannoteRunner
from .whisperx_runner import WhisperXRunner
from .sherpa_onnx_runner import SherpaOnnxRunner
from .diart_runner import DiartRunner

__all__ = [
    "PolyvoiceRunner",
    "PyannoteRunner",
    "WhisperXRunner",
    "SherpaOnnxRunner",
    "DiartRunner",
]
