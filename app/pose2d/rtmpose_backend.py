"""
RTMPose-m backend via rtmlib (optional upgrade path — ADR-002).

Install extras: pip install rtmlib onnxruntime
Model files (~30-80 MB) are auto-downloaded by rtmlib to ~/.cache/rtmlib/ on
first use.  Available modes (rtmlib Body presets):

    lightweight  — YOLOX-tiny + RTMPose-s  (~30 MB total, fastest)
    balanced     — YOLOX-m   + RTMPose-m  (~60 MB, recommended)
    performance  — YOLOX-x   + RTMPose-x  (~80 MB, most accurate)
"""

from __future__ import annotations

import numpy as np

from .base import PoseBackend


class RTMPoseBackend(PoseBackend):
    """
    RTMPose backend via rtmlib.  Natively outputs COCO-17 — no remapping
    needed (ADR-002).

    Requires: pip install rtmlib onnxruntime   (or: pip install -e ".[rtmpose]")

    Args:
        mode:   rtmlib Body preset — 'lightweight', 'balanced' (default),
                or 'performance'.
        device: 'cpu' (default) or 'cuda' if CUDA-capable onnxruntime is
                installed.
    """

    name = "rtmpose_m"

    def __init__(
        self,
        mode: str = "balanced",
        device: str = "cpu",
    ):
        try:
            from rtmlib import Body
        except ImportError as e:
            raise ImportError(
                "rtmlib is not installed.  Run:  pip install rtmlib onnxruntime"
            ) from e

        self._body = Body(
            mode=mode,
            device=device,
            backend="onnxruntime",
        )

    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            frame_bgr: uint8 BGR [H, W, 3]

        Returns:
            keypoints: float32 [P, 17, 2]
            conf:      float32 [P, 17]
        """
        keypoints, scores = self._body(frame_bgr)

        if keypoints is None or len(keypoints) == 0:
            return (
                np.zeros((1, 17, 2), dtype=np.float32),
                np.zeros((1, 17), dtype=np.float32),
            )

        kps = np.array(keypoints, dtype=np.float32)  # [P, 17, 2]
        conf = np.array(scores, dtype=np.float32)  # [P, 17]

        # Guard: rtmlib may return [17, 2] for a single person
        if kps.ndim == 2:
            kps = kps[np.newaxis]
            conf = conf[np.newaxis]

        return kps, conf

    def close(self) -> None:
        pass  # rtmlib handles cleanup internally
