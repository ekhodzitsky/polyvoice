"""polyvoice — speaker diarization, powered by Rust + ONNX."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from polyvoice._polyvoice import DiarizationResult, Pipeline

__all__ = ["DiarizationResult", "Pipeline"]

try:
    __version__ = _version("polyvoice")
except PackageNotFoundError:  # editable/source checkout without installed metadata
    __version__ = "0.0.0"
