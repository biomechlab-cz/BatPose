"""Abstract base class for stereo capture sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class CaptureFrame:
    """A synchronized stereo frame pair."""

    frame_left: np.ndarray  # BGR uint8 [H, W, 3]
    frame_right: np.ndarray  # BGR uint8 [H, W, 3]
    timestamp: float  # seconds from capture start
    frame_index: int
    # Hardware timestamps from the camera's internal clock (nanoseconds).
    # Set by FlirCapture via img.GetTimeStamp(); None for VideoSimulator.
    # The delta between the two values measures hardware sync quality.
    hw_timestamp_left_ns: int | None = None
    hw_timestamp_right_ns: int | None = None

    @property
    def hw_delta_us(self) -> float | None:
        """Absolute difference between hardware timestamps in microseconds, or None."""
        if self.hw_timestamp_left_ns is None or self.hw_timestamp_right_ns is None:
            return None
        return abs(self.hw_timestamp_left_ns - self.hw_timestamp_right_ns) / 1_000.0


class BaseCapture(ABC):
    """
    Abstract interface for a synchronized stereo capture source.

    Implementors:
    - VideoSimulator: replay from two pre-recorded video files
    - FlirCapture:    live capture from two FLIR cameras via PySpin
    """

    @abstractmethod
    def start(self) -> None:
        """Initialize hardware / open files."""

    @abstractmethod
    def stop(self) -> None:
        """Release all resources."""

    @abstractmethod
    def read(self) -> CaptureFrame | None:
        """
        Return the next synchronized frame pair, or None if exhausted / stopped.
        Must be non-blocking in terms of indefinite waits (may block briefly for sync).
        """

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frame rate."""

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of each frame."""

    def __enter__(self) -> BaseCapture:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
