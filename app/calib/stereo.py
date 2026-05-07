"""Stereo calibration pipeline and calibration.yml I/O."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .board import make_detector
from .frame_select import FrameSelection, extract_calibration_frames


def calibrate_stereo(
    selections: list[FrameSelection],
    img_size: tuple[int, int],
    fix_intrinsics: bool = False,
    progress_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    all_det_l: list | None = None,
    all_det_r: list | None = None,
) -> dict[str, Any]:
    """
    Run stereo calibration from a list of FrameSelection objects.

    When *all_det_l* and *all_det_r* are supplied and each has more detections
    than the stereo pairs alone, two-phase calibration is used automatically:

      Phase 1 — per-camera intrinsics from ALL single-camera detections
                 (``cv2.calibrateCamera`` on each set independently).
      Phase 2 — extrinsics-only via ``cv2.stereoCalibrate`` with
                 ``CALIB_FIX_INTRINSIC`` on the stereo pairs.

    This significantly improves intrinsics estimation when simultaneous board
    visibility is rare (e.g. board shown to each camera separately).

    Args:
        selections: list produced by extract_calibration_frames
        img_size: (width, height) of source frames
        fix_intrinsics: if True force CALIB_FIX_INTRINSIC even without all_det lists
        progress_cb: optional progress callback
        cancel_check: optional cancel callback
        all_det_l: all left-camera DetectionResult objects (including non-paired)
        all_det_r: all right-camera DetectionResult objects (including non-paired)

    Returns:
        Dict with K1, D1, K2, D2, R, T, E, F, rms, n_frames, image_size
    """
    w, h = img_size
    if w <= 0 or h <= 0:
        raise ValueError(f"img_size must have positive width and height, got img_size={img_size!r}")

    MIN_RECOMMENDED_FRAMES = 20
    if len(selections) < 3:
        raise ValueError(
            f"Need at least 3 paired frames, got {len(selections)}. "
            "Check that the board is visible in both cameras."
        )
    if len(selections) < MIN_RECOMMENDED_FRAMES:
        import warnings

        warnings.warn(
            f"Only {len(selections)} calibration frames available; "
            f"at least {MIN_RECOMMENDED_FRAMES} are recommended for reliable results. "
            "RMS error may be unreliable.",
            stacklevel=2,
        )
        if progress_cb:
            progress_cb(
                5,
                f"⚠ Warning: only {len(selections)} frames — recommend ≥"
                f"{MIN_RECOMMENDED_FRAMES} for reliable calibration.",
            )

    if progress_cb:
        progress_cb(5, f"Calibrating from {len(selections)} frames…")

    obj_pts_all = []
    img_pts_l_all = []
    img_pts_r_all = []

    for i, sel in enumerate(selections):
        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")
        n_l = len(sel.det_left.obj_pts)
        n_r = len(sel.det_right.img_pts)
        if n_l != n_r:
            raise ValueError(
                f"Frame {i}: left camera detected {n_l} corners but right camera detected "
                f"{n_r}. cv2.stereoCalibrate requires the same point set for each view. "
                "This can occur with ChArUco boards when each camera detects a different "
                "subset of corners. Ensure both cameras have an unobstructed view of the "
                "full board, or use a detector that returns intersected corner IDs."
            )
        obj_pts_all.append(sel.det_left.obj_pts.reshape(-1, 1, 3).astype(np.float32))
        img_pts_l_all.append(sel.det_left.img_pts.reshape(-1, 1, 2).astype(np.float32))
        img_pts_r_all.append(sel.det_right.img_pts.reshape(-1, 1, 2).astype(np.float32))

    # Decide between two-phase and single-phase calibration.
    # Two-phase is used when per-camera lists with more data than stereo pairs are available.
    use_two_phase = (
        all_det_l is not None
        and all_det_r is not None
        and len(all_det_l) > len(selections)
        and len(all_det_r) > len(selections)
    )

    if use_two_phase:
        # Phase 1 — intrinsics from all per-camera detections
        obj_l = [d.obj_pts.reshape(-1, 1, 3).astype(np.float32) for d in all_det_l]
        ipts_l = [d.img_pts.reshape(-1, 1, 2).astype(np.float32) for d in all_det_l]
        obj_r = [d.obj_pts.reshape(-1, 1, 3).astype(np.float32) for d in all_det_r]
        ipts_r = [d.img_pts.reshape(-1, 1, 2).astype(np.float32) for d in all_det_r]

        if progress_cb:
            progress_cb(
                15,
                f"Two-phase calibration: Phase 1 — left intrinsics from {len(obj_l)} frames…",
            )
        _, K1, D1, _, _ = cv2.calibrateCamera(obj_l, ipts_l, img_size, None, None)

        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        if progress_cb:
            progress_cb(
                35,
                f"Two-phase calibration: Phase 1 — right intrinsics from {len(obj_r)} frames…",
            )
        _, K2, D2, _, _ = cv2.calibrateCamera(obj_r, ipts_r, img_size, None, None)

        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        if progress_cb:
            progress_cb(
                55,
                f"Two-phase calibration: Phase 2 — extrinsics from {len(selections)} stereo pairs…",
            )
        stereo_flags = cv2.CALIB_FIX_INTRINSIC
    else:
        if progress_cb:
            progress_cb(20, "Calibrating left camera…")

        _, K1, D1, _, _ = cv2.calibrateCamera(obj_pts_all, img_pts_l_all, img_size, None, None)

        if progress_cb:
            progress_cb(40, "Calibrating right camera…")

        _, K2, D2, _, _ = cv2.calibrateCamera(obj_pts_all, img_pts_r_all, img_size, None, None)

        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        if progress_cb:
            progress_cb(60, "Running stereo calibration…")

        stereo_flags = cv2.CALIB_USE_INTRINSIC_GUESS
        if fix_intrinsics:
            stereo_flags = cv2.CALIB_FIX_INTRINSIC

    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts_all,
        img_pts_l_all,
        img_pts_r_all,
        K1,
        D1,
        K2,
        D2,
        img_size,
        flags=stereo_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    )

    if progress_cb:
        progress_cb(90, f"Done. RMS={rms:.3f}px")

    return {
        "K1": K1,
        "D1": D1,
        "K2": K2,
        "D2": D2,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "rms": float(rms),
        "n_frames": len(selections),
        "image_size": list(img_size),
    }


def run_calibration_pipeline(
    video_left: str,
    video_right: str,
    board_cfg: dict,
    output_path: str,
    max_frames: int = 60,
    sample_every: int = 5,
    min_coverage: float = 0.05,
    temporal_window: int = 4,
    progress_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    frame_cb: Callable | None = None,
    scan_cb: Callable | None = None,
) -> str | None:
    """
    Full calibration pipeline: frame extraction → stereo calibration → save.

    Args:
        video_left: path to left camera video
        video_right: path to right camera video
        board_cfg: dict describing the board (see board.py)
        output_path: where to write calibration.yml
        max_frames: max calibration frames to use
        sample_every: subsample factor for video frames
        min_coverage: minimum board coverage fraction to accept a frame
        temporal_window: ±sampled-frame search window for temporal pair relaxation (0=off)
        progress_cb: (percent, message) callback
        cancel_check: return True to abort
        frame_cb: called with (ann_left, ann_right) for each accepted frame pair
        scan_cb: called on every sampled frame (throttled) with
                 (frame_l, frame_r, det_l_ok, det_r_ok)

    Returns:
        output_path on success, None if cancelled
    """

    detector = make_detector(board_cfg)

    def _progress_extract(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(int(pct * 0.5), msg)

    def _progress_calib(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(50 + int(pct * 0.5), msg)

    if progress_cb:
        progress_cb(0, "Extracting calibration frames…")

    # Collect per-camera detections for two-phase calibration (NEW-7)
    all_det_l: list = []
    all_det_r: list = []

    selections = extract_calibration_frames(
        video_left,
        video_right,
        detector,
        max_frames=max_frames,
        sample_every=sample_every,
        min_coverage=min_coverage,
        temporal_window=temporal_window,
        progress_cb=_progress_extract,
        cancel_check=cancel_check,
        frame_cb=frame_cb,
        scan_cb=scan_cb,
        all_det_l_out=all_det_l,
        all_det_r_out=all_det_r,
    )

    if cancel_check and cancel_check():
        return None

    if len(selections) == 0:
        raise RuntimeError(
            "No valid paired frames found. "
            "Verify the board is visible and the board config is correct."
        )

    # Infer image size from first frame
    h, w = selections[0].frame_left.shape[:2]
    img_size = (w, h)

    calib_data = calibrate_stereo(
        selections,
        img_size,
        progress_cb=_progress_calib,
        cancel_check=cancel_check,
        all_det_l=all_det_l,
        all_det_r=all_det_r,
    )

    if cancel_check and cancel_check():
        return None

    save_calibration(calib_data, board_cfg, output_path)
    return output_path


def save_calibration(calib_data: dict, board_cfg: dict, output_path: str) -> None:
    """Save calibration results to a YAML file (OpenCV-compatible structure)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _to_list(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    out = {
        "image_size": _to_list(calib_data["image_size"]),
        "board_cfg": board_cfg,
        "K1": _to_list(calib_data["K1"]),
        "D1": _to_list(calib_data["D1"]),
        "K2": _to_list(calib_data["K2"]),
        "D2": _to_list(calib_data["D2"]),
        "R": _to_list(calib_data["R"]),
        "T": _to_list(calib_data["T"]),
        "E": _to_list(calib_data["E"]),
        "F": _to_list(calib_data["F"]),
        "quality": {
            "rms": float(calib_data["rms"]),
            "n_frames_used": int(calib_data["n_frames"]),
        },
    }

    with open(output_path, "w") as fh:
        yaml.dump(out, fh, default_flow_style=False, sort_keys=False)


def load_calibration(path: str) -> dict[str, Any]:
    """
    Load calibration.yml and return a dict with numpy arrays.

    Keys: K1, D1, K2, D2, R, T, E, F, image_size (tuple), board_cfg, quality
    """
    with open(path) as fh:
        data = yaml.safe_load(fh)

    required = {"K1", "D1", "K2", "D2", "R", "T", "image_size"}
    missing = required - data.keys()
    if missing:
        raise ValueError(
            f"Calibration file {path!r} is missing required keys: "
            f"{sorted(missing)}. Required keys are: {sorted(required)}"
        )

    return {
        "K1": np.array(data["K1"], dtype=np.float64),
        "D1": np.array(data["D1"], dtype=np.float64),
        "K2": np.array(data["K2"], dtype=np.float64),
        "D2": np.array(data["D2"], dtype=np.float64),
        "R": np.array(data["R"], dtype=np.float64),
        "T": np.array(data["T"], dtype=np.float64),
        "E": np.array(data.get("E", np.zeros((3, 3))), dtype=np.float64),
        "F": np.array(data.get("F", np.zeros((3, 3))), dtype=np.float64),
        "image_size": tuple(int(x) for x in data["image_size"]),
        "board_cfg": data.get("board_cfg", {}),
        "quality": data.get("quality", {}),
    }
