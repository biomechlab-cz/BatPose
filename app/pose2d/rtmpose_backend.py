"""
RTMPose-m backend via rtmlib (optional upgrade path — ADR-002).

Install extras: pip install rtmlib onnxruntime
Model files (~30 MB) are auto-downloaded by rtmlib to ~/.rtmlib/ on first use.
"""

from __future__ import annotations

import numpy as np

from .base import PoseBackend


class RTMPoseBackend(PoseBackend):
    """
    RTMPose-m backend.  Natively outputs COCO-17 — no remapping needed (ADR-002).

    Requires: pip install rtmlib onnxruntime
    """

    name = "rtmpose_m"

    def __init__(
        self,
        det_model: str = "human",
        pose_model: str = "body",
        mode: str = "performance",  # 'performance' or 'balanced'
        device: str = "cpu",
    ):
        try:
            from rtmlib import Body
        except ImportError as e:
            raise ImportError("rtmlib is not installed. Run: pip install rtmlib onnxruntime") from e

        self._body = Body(
            det=det_model,
            pose=pose_model,
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

        # Ensure at least 1 person dimension
        if kps.ndim == 2:
            kps = kps[np.newaxis]
            conf = conf[np.newaxis]

        return kps, conf

    def close(self) -> None:
        pass  # rtmlib handles cleanup internally
