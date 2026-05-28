"""MediaPipe Pose Landmarker backend (Tasks API, CPU-first)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .base import PoseBackend

# MediaPipe-33 index → COCO-17 index mapping (from docs/skeleton_mapping.md)
MP_TO_COCO17: list[int] = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Pose Landmarker variants.  "heavy" is the most accurate (best for wide stereo
# vergence where landmark precision matters most) at higher CPU cost; "full" is
# the balanced default; "lite" is fastest/least accurate.
_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
_MODELS: dict[str, str] = {
    "lite": f"{_MODEL_BASE}/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": f"{_MODEL_BASE}/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": f"{_MODEL_BASE}/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}
_CACHE_DIR = Path.home() / ".cache" / "BatPose"


def _ensure_model(complexity: str = "full") -> str:
    """Return path to the cached model for *complexity*, downloading if needed."""
    complexity = complexity if complexity in _MODELS else "full"
    url = _MODELS[complexity]
    cache_path = _CACHE_DIR / f"pose_landmarker_{complexity}.task"
    if cache_path.exists():
        return str(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe '{complexity}' model -> {cache_path}", flush=True)

    def _reporthook(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, int(count * block_size * 100 / total_size))
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, cache_path, reporthook=_reporthook)
    print(" done", flush=True)
    return str(cache_path)


class MediaPipeBackend(PoseBackend):
    """
    CPU-first pose backend using MediaPipe Pose Landmarker (Tasks API).

    Outputs COCO-17 keypoints via the MP_TO_COCO17 mapping defined in
    docs/skeleton_mapping.md.

    Multi-person support: up to num_poses persons per frame (default 2).
    """

    name = "mediapipe_full"

    def __init__(
        self,
        num_poses: int = 2,
        model_complexity: str = "full",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | None = None,
        running_mode: str = "video",
    ):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._num_poses = num_poses
        self._fps: float = 30.0
        self._frame_idx: int = 0
        # "video" → detect_for_video (temporal tracking; correct for a single
        #           continuous stream, e.g. the offline pipeline).
        # "image" → detect (stateless per-frame).  Use this for LIVE stereo:
        #           two camera streams can't share a VIDEO-mode tracker, and
        #           multiple VIDEO-mode landmarkers interfere in one process.
        self._image_mode = running_mode == "image"

        if model_path is None:
            model_path = _ensure_model(model_complexity)

        rm = vision.RunningMode.IMAGE if self._image_mode else vision.RunningMode.VIDEO
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=rm,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def set_fps(self, fps: float) -> None:
        self._fps = fps
        self._frame_idx = 0

    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        if self._image_mode:
            result = self._landmarker.detect(mp_image)
        else:
            timestamp_ms = int(self._frame_idx * 1000.0 / self._fps)
            self._frame_idx += 1
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        pose_lms_list = result.pose_landmarks or []
        n_persons = max(1, len(pose_lms_list))

        kps = np.zeros((n_persons, 17, 2), dtype=np.float32)
        conf = np.zeros((n_persons, 17), dtype=np.float32)

        for p_idx, pose_lms in enumerate(pose_lms_list):
            for coco_idx, mp_idx in enumerate(MP_TO_COCO17):
                if mp_idx >= len(pose_lms):
                    continue
                lm = pose_lms[mp_idx]
                kps[p_idx, coco_idx] = [lm.x * w, lm.y * h]

                vis = float(lm.visibility) if lm.visibility is not None else 0.0
                pres = float(lm.presence) if lm.presence is not None else 1.0
                conf[p_idx, coco_idx] = vis if pres >= 0.5 else 0.0

        return kps, conf

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None  # type: ignore[assignment]


def make_mediapipe_backend(**kwargs: object) -> MediaPipeBackend:
    """Factory function; raises ImportError with a clear message if mediapipe is missing."""
    try:
        return MediaPipeBackend(**kwargs)  # type: ignore[arg-type]
    except ImportError as e:
        raise ImportError("mediapipe is not installed. Run: pip install mediapipe") from e
