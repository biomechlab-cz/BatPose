"""Synthetic stereo capture source used when FLIR/PySpin is unavailable."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .base import BaseCapture, CaptureFrame


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration for :class:`SimulatorCapture`."""

    width: int = 1280
    height: int = 720
    fps: float = 30.0
    max_frames: int | None = None


class SimulatorCapture(BaseCapture):
    """A deterministic CPU-only stereo source for development and demos.

    The source generates a moving synthetic scene with a small left/right
    disparity and monotonic hardware-like timestamps.  It is intentionally not
    a pose or calibration benchmark; its purpose is to keep Live Capture,
    recording, and UI plumbing usable on machines without the FLIR SDK.
    """

    def __init__(
        self,
        fps: float = 30.0,
        frame_size: tuple[int, int] = (1280, 720),
        max_frames: int | None = None,
    ):
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError(f"frame_size must be positive, got {frame_size!r}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self._cfg = SimulatorConfig(
            width=int(width), height=int(height), fps=float(fps), max_frames=max_frames
        )
        self._started = False
        self._frame_index = 0
        self._start_perf = 0.0
        self._next_perf = 0.0

    def start(self) -> None:
        self._started = True
        self._frame_index = 0
        self._start_perf = time.perf_counter()
        self._next_perf = self._start_perf

    def stop(self) -> None:
        self._started = False

    def read(self) -> CaptureFrame | None:
        if not self._started:
            return None
        if self._cfg.max_frames is not None and self._frame_index >= self._cfg.max_frames:
            return None

        now = time.perf_counter()
        if now < self._next_perf:
            time.sleep(self._next_perf - now)

        idx = self._frame_index
        timestamp = idx / self._cfg.fps
        left = self._render(idx, disparity_px=0)
        right = self._render(idx, disparity_px=-18)
        hw_left = int(timestamp * 1_000_000_000)
        hw_right = hw_left + 250_000  # fixed 250 us clock offset, zero jitter.

        self._frame_index += 1
        self._next_perf = self._start_perf + self._frame_index / self._cfg.fps
        return CaptureFrame(
            frame_left=left,
            frame_right=right,
            timestamp=timestamp,
            frame_index=idx,
            hw_timestamp_left_ns=hw_left,
            hw_timestamp_right_ns=hw_right,
            dropped_frames=0,
        )

    @property
    def fps(self) -> float:
        return self._cfg.fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._cfg.width, self._cfg.height)

    def _render(self, idx: int, disparity_px: int) -> np.ndarray:
        w, h = self._cfg.width, self._cfg.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (24, 24, 28)

        # Subtle grid gives the preview a camera-like sense of motion/scale.
        step = 80
        for x in range(0, w, step):
            cv2.line(frame, (x, 0), (x, h), (42, 42, 48), 1, cv2.LINE_AA)
        for y in range(0, h, step):
            cv2.line(frame, (0, y), (w, y), (42, 42, 48), 1, cv2.LINE_AA)

        phase = idx * 0.08
        cx = int(w * 0.5 + np.sin(phase) * w * 0.18) + disparity_px
        cy = int(h * 0.48 + np.cos(phase * 0.7) * h * 0.10)
        radius = max(28, min(w, h) // 14)

        cv2.circle(frame, (cx, cy), radius, (70, 150, 240), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius, (220, 235, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 90, cy + 120), (cx + 90, cy + 120), (80, 220, 120), 3, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + radius), (cx, cy + 120), (210, 210, 210), 3, cv2.LINE_AA)
        cv2.putText(
            frame,
            "SIMULATED CAPTURE",
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"frame {idx:06d}",
            (24, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (160, 160, 170),
            2,
            cv2.LINE_AA,
        )
        return frame
