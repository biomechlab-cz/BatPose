"""Abstract base class for pose backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PoseBackend(ABC):
    """
    Abstract interface for 2D pose detection backends.

    All backends must:
    - Accept a BGR frame (H×W×3 uint8)
    - Return keypoints in COCO-17 format (pixel coordinates)
    - Return per-joint confidence values in [0, 1]
    """

    name: str = "unknown"

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect poses in a single frame.

        Args:
            frame_bgr: uint8 BGR image [H, W, 3]

        Returns:
            keypoints: float32 [P, 17, 2]  pixel (x, y) per joint
            conf:      float32 [P, 17]     confidence in [0, 1]

        P is the number of detected persons (≥ 1; may be 1 with all-zero kps if none found).
        """

    def set_fps(self, fps: float) -> None:  # noqa: B027
        """Optional: notify the backend of video frame rate (used by timestamp-based APIs)."""

    def close(self) -> None:  # noqa: B027
        """Release any resources held by the backend."""
