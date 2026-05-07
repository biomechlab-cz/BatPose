"""Video-level 2D pose estimation pipeline."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import PoseBackend


def process_video(
    video_path: str,
    backend: PoseBackend,
    progress_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Run 2D pose detection on every frame of *video_path*.

    Args:
        video_path: path to input video
        backend: PoseBackend instance (already constructed)
        progress_cb: (percent, message) callback
        cancel_check: returns True to abort

    Returns:
        keypoints: float32 [T, P, 17, 2]
        conf:      float32 [T, P, 17]
        meta:      dict (fps, timestamps, model_name, skeleton, image_size)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path!r}")

    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_raw if math.isfinite(fps_raw) and fps_raw > 0 else 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    backend.set_fps(fps)

    all_kps: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    timestamps: list[float] = []

    frame_idx = 0
    try:
        while True:
            if cancel_check and cancel_check():
                return _empty_result(fps, width, height, backend.name)

            ret, frame_bgr = cap.read()
            if not ret:
                break

            kps, conf = backend.detect(frame_bgr)  # [P, 17, 2], [P, 17]
            all_kps.append(kps)
            all_conf.append(conf)
            timestamps.append(frame_idx / fps)

            if progress_cb and n_frames > 0:
                pct = int(frame_idx * 100 / n_frames)
                progress_cb(pct, f"Frame {frame_idx}/{n_frames}")

            frame_idx += 1
    finally:
        cap.release()

    if not all_kps:
        return _empty_result(fps, width, height, backend.name)

    # Pad to uniform shape (max persons across frames)
    max_p = max(k.shape[0] for k in all_kps)
    T = len(all_kps)
    J = 17

    keypoints = np.zeros((T, max_p, J, 2), dtype=np.float32)
    confs = np.zeros((T, max_p, J), dtype=np.float32)

    for t, (k, c) in enumerate(zip(all_kps, all_conf)):
        p = k.shape[0]
        keypoints[t, :p] = k
        confs[t, :p] = c

    meta: dict[str, Any] = {
        "fps": float(fps),
        "timestamps": np.array(timestamps, dtype=np.float32),
        "model_name": backend.name,
        "skeleton": "coco17",
        "image_size": [width, height],
    }

    if progress_cb:
        progress_cb(100, f"Processed {T} frames")

    return keypoints, confs, meta


def _empty_result(
    fps: float, width: int, height: int, model_name: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = {
        "fps": float(fps),
        "timestamps": np.array([], dtype=np.float32),
        "model_name": model_name,
        "skeleton": "coco17",
        "image_size": [width, height],
    }
    return (
        np.zeros((0, 1, 17, 2), dtype=np.float32),
        np.zeros((0, 1, 17), dtype=np.float32),
        meta,
    )


def save_pose2d(
    keypoints: np.ndarray,
    conf: np.ndarray,
    meta: dict[str, Any],
    output_path: str,
) -> None:
    """Save pose2d results to NPZ (spec: docs/skeleton_mapping.md §4)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        keypoints=keypoints,
        conf=conf,
        meta=np.array([meta], dtype=object),
    )


def load_pose2d(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a pose2d NPZ file. Returns (keypoints, conf, meta)."""
    d = np.load(path, allow_pickle=True)
    required = {"keypoints", "conf", "meta"}
    missing = required - set(d.files)
    if missing:
        raise ValueError(
            f"Pose2D file {path!r} is missing required keys: "
            f"{sorted(missing)}. Required keys are: {sorted(required)}"
        )
    meta = d["meta"].item()
    return d["keypoints"], d["conf"], meta
