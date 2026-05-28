"""Camera capture module (FLIR cameras via PySpin)."""

from .base import BaseCapture, CaptureFrame

__all__ = ["BaseCapture", "CaptureFrame"]

try:
    from .flir import FlirCapture  # noqa: F401

    __all__.append("FlirCapture")
except ImportError:
    pass
