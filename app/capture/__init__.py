"""Camera capture module (FLIR cameras via PySpin, plus simulator fallback)."""

from .base import BaseCapture, CaptureFrame
from .simulator import SimulatorCapture

__all__ = ["BaseCapture", "CaptureFrame", "SimulatorCapture"]

try:
    from .flir import FlirCapture  # noqa: F401

    __all__.append("FlirCapture")
except ImportError:
    pass
