"""Video-file based stereo capture simulator (no FLIR hardware required)."""

from __future__ import annotations

import math
import time

import cv2

from .base import BaseCapture, CaptureFrame


class VideoSimulator(BaseCapture):
    """
    Simulates a synchronized stereo camera by reading from two video files.

    If realtime=True, rate-limits reads to match the video's native FPS.
    If realtime=False (default), reads as fast as possible (useful for batch processing).
    """

    def __init__(
        self,
        video_left: str,
        video_right: str,
        realtime: bool = False,
        loop: bool = False,
    ):
        """
        Args:
            video_left:  path to left camera video
            video_right: path to right camera video
            realtime:    if True, sleep between frames to match FPS
            loop:        if True, loop videos when exhausted
        """
        self._path_left = str(video_left)
        self._path_right = str(video_right)
        self._realtime = realtime
        self._loop = loop

        self._cap_left: cv2.VideoCapture | None = None
        self._cap_right: cv2.VideoCapture | None = None
        self._frame_idx: int = 0
        self._t_start: float = 0.0
        self._fps_val: float = 30.0
        self._frame_size_val: tuple[int, int] = (640, 480)

    def start(self) -> None:
        self._cap_left = cv2.VideoCapture(self._path_left)
        self._cap_right = cv2.VideoCapture(self._path_right)

        if not self._cap_left.isOpened():
            self._cap_left.release()
            self._cap_left = None
            raise FileNotFoundError(f"Cannot open left video: {self._path_left!r}")
        if not self._cap_right.isOpened():
            self._cap_left.release()
            self._cap_left = None
            self._cap_right.release()
            self._cap_right = None
            raise FileNotFoundError(f"Cannot open right video: {self._path_right!r}")

        fps_raw = self._cap_left.get(cv2.CAP_PROP_FPS)
        self._fps_val = fps_raw if math.isfinite(fps_raw) and fps_raw > 0 else 30.0
        w = int(self._cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_size_val = (w, h)
        self._frame_idx = 0
        self._t_start = time.monotonic()

    def stop(self) -> None:
        if self._cap_left is not None:
            self._cap_left.release()
            self._cap_left = None
        if self._cap_right is not None:
            self._cap_right.release()
            self._cap_right = None

    def read(self) -> CaptureFrame | None:
        if self._cap_left is None or self._cap_right is None:
            raise RuntimeError("Call start() before read()")

        if self._realtime:
            # Rate-limit to nominal FPS
            target_t = self._t_start + self._frame_idx / self._fps_val
            now = time.monotonic()
            if target_t > now:
                time.sleep(target_t - now)

        ret_l, frame_l = self._cap_left.read()
        ret_r, frame_r = self._cap_right.read()

        if not ret_l or not ret_r:
            if self._loop:
                self._cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_idx = 0
                self._t_start = time.monotonic()
                ret_l, frame_l = self._cap_left.read()
                ret_r, frame_r = self._cap_right.read()
                if not ret_l or not ret_r:
                    return None
            else:
                return None

        ts = self._frame_idx / self._fps_val
        result = CaptureFrame(
            frame_left=frame_l,
            frame_right=frame_r,
            timestamp=ts,
            frame_index=self._frame_idx,
        )
        self._frame_idx += 1
        return result

    @property
    def fps(self) -> float:
        return self._fps_val

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size_val
