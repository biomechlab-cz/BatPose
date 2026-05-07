"""Camera capture module (FLIR or video simulator)."""

from .base import BaseCapture, CaptureFrame
from .simulator import VideoSimulator

__all__ = ["BaseCapture", "CaptureFrame", "VideoSimulator"]

try:
    from .flir import FlirCapture  # noqa: F401

    __all__.append("FlirCapture")
except ImportError:
    pass
