"""Full 3D reconstruction pipeline (ADR-004)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..calib.stereo import load_calibration
from ..pose2d.pipeline import load_pose2d
from .smooth import smooth_trajectory
from .triangulate import reprojection_error, triangulate_points_dlt, undistort_points


def reconstruct3d(
    calib_path: str,
    pose2d_left_path: str,
    pose2d_right_path: str,
    output_path: str,
    min_conf: float = 0.3,
    max_reproj_err: float = 20.0,
    min_cutoff: float = 1.0,
    beta: float = 0.5,
    d_cutoff: float = 1.0,
    no_smooth: bool = False,
    progress_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> str | None:
    """
    Full 3D reconstruction: load calibration + pose2d → triangulate → smooth → save.

    Pipeline (per ADR-004):
    1. Undistort 2D keypoints (cv2.undistortPoints with P=None)
    2. Build P1 = [I|0], P2 = [R|T] in normalized coordinates
    3. Triangulate via DLT
    4. Compute reprojection error in pixel space
    5. Outlier rejection (low conf or high reproj error → conf3d = 0)
    6. OneEuro temporal smoothing (skipped when no_smooth=True)

    Args:
        calib_path:         path to calibration.yml
        pose2d_left_path:   path to pose2d_left.npz
        pose2d_right_path:  path to pose2d_right.npz
        output_path:        where to write pose3d.npz
        min_conf:           minimum 2D confidence to accept joint
        max_reproj_err:     maximum reprojection error (pixels)
        min_cutoff:         OneEuro min_cutoff (Hz)
        beta:               OneEuro beta
        d_cutoff:           OneEuro d_cutoff (Hz)
        no_smooth:          skip temporal smoothing entirely when True
        progress_cb:        (percent, message) callback
        cancel_check:       returns True to abort

    Returns:
        output_path on success, None if cancelled
    """
    if not 0.0 <= min_conf <= 1.0:
        raise ValueError(f"min_conf must be in [0.0, 1.0], got {min_conf}")
    if not max_reproj_err >= 0.0:
        raise ValueError(f"max_reproj_err must be >= 0, got {max_reproj_err}")

    if progress_cb:
        progress_cb(2, "Loading calibration…")

    calib = load_calibration(calib_path)
    K1, D1 = calib["K1"], calib["D1"]
    K2, D2 = calib["K2"], calib["D2"]
    R, T = calib["R"], calib["T"]
    fisheye = calib.get("lens_model", "standard") == "fisheye"

    if progress_cb:
        progress_cb(5, "Loading 2D poses…")

    # Load pose2d
    kps_left, conf_left, meta_left = load_pose2d(pose2d_left_path)
    kps_right, conf_right, _ = load_pose2d(pose2d_right_path)

    T_frames, P, J, _ = kps_left.shape
    fps = float(meta_left.get("fps", 30.0))

    # Align frame counts (take minimum) — warn if mismatch
    T_r = kps_right.shape[0]
    if T_frames != T_r:
        import warnings

        dropped = abs(T_frames - T_r)
        _msg = (
            f"Frame count mismatch: left={T_frames}, right={T_r}. "
            f"{dropped} frame(s) will be dropped. "
            "Ensure both videos are synchronised."
        )
        warnings.warn(_msg, UserWarning, stacklevel=2)
        if progress_cb:
            progress_cb(5, f"⚠ {_msg}")
    T_frames = min(T_frames, T_r)
    kps_left = kps_left[:T_frames]
    kps_right = kps_right[:T_frames]
    conf_left = conf_left[:T_frames]
    conf_right = conf_right[:T_frames]

    # Align person counts (take minimum) — warn if mismatch
    P_r = kps_right.shape[1]
    if P != P_r:
        import warnings as _warnings

        _pmsg = (
            f"Person count mismatch: left={P}, right={P_r}. "
            f"Truncating to {min(P, P_r)} person(s). "
            "Ensure both videos were processed with the same pose backend."
        )
        _warnings.warn(_pmsg, UserWarning, stacklevel=2)
        if progress_cb:
            progress_cb(5, f"⚠ {_pmsg}")
        P = min(P, P_r)
        kps_left = kps_left[:, :P]
        kps_right = kps_right[:, :P]
        conf_left = conf_left[:, :P]
        conf_right = conf_right[:, :P]

    # Projection matrices in normalized image coordinates
    # Camera 1: P1 = [I | 0]
    P1_norm = np.hstack([np.eye(3), np.zeros((3, 1))])  # [3, 4]
    # Camera 2: P2 = [R | T]
    P2_norm = np.hstack([R, T.reshape(3, 1)])  # [3, 4]

    # Identity rotation for camera 1 (for reprojection)
    R1 = np.eye(3)
    t1 = np.zeros(3)
    t2 = T.flatten()

    joints3d = np.zeros((T_frames, P, J, 3), dtype=np.float32)
    conf3d = np.zeros((T_frames, P, J), dtype=np.float32)
    repro_arr = np.full((T_frames, P, J), np.inf, dtype=np.float32)

    for t in range(T_frames):
        if cancel_check and cancel_check():
            return None

        if progress_cb:
            pct = 10 + int(t * 75 / T_frames)
            if t % max(1, T_frames // 20) == 0:
                progress_cb(pct, f"Triangulating frame {t}/{T_frames}…")

        for p in range(P):
            pts_l = kps_left[t, p]  # [17, 2] pixels
            pts_r = kps_right[t, p]  # [17, 2] pixels
            c_l = conf_left[t, p]  # [17]
            c_r = conf_right[t, p]  # [17]

            # Undistort → normalized camera coordinates
            pts_l_norm = undistort_points(pts_l, K1, D1, fisheye=fisheye)  # [17, 2]
            pts_r_norm = undistort_points(pts_r, K2, D2, fisheye=fisheye)  # [17, 2]

            # Triangulate
            pts3d = triangulate_points_dlt(
                P1_norm.astype(np.float64),
                P2_norm.astype(np.float64),
                pts_l_norm,
                pts_r_norm,
            )  # [17, 3]

            # Reprojection error in original pixel coordinates
            err_l = reprojection_error(pts3d, K1, D1, R1, t1, pts_l, fisheye=fisheye)
            err_r = reprojection_error(pts3d, K2, D2, R, t2, pts_r, fisheye=fisheye)
            err_mean = (err_l + err_r) * 0.5  # [17]

            # Outlier rejection mask
            valid = (c_l >= min_conf) & (c_r >= min_conf) & (err_mean <= max_reproj_err)

            joints3d[t, p] = pts3d
            conf3d[t, p] = np.minimum(c_l, c_r) * valid.astype(np.float32)
            repro_arr[t, p] = err_mean

            # Zero out outlier positions
            joints3d[t, p, ~valid] = 0.0

    if cancel_check and cancel_check():
        return None

    if no_smooth:
        if progress_cb:
            progress_cb(85, "Temporal smoothing skipped (disabled).")
    else:
        if progress_cb:
            progress_cb(85, "Applying temporal smoothing…")
        # Temporal smoothing per person × joint × coordinate
        for p in range(P):
            for j in range(J):
                traj = joints3d[:, p, j, :]  # [T, 3]
                joints3d[:, p, j, :] = smooth_trajectory(traj, fps, min_cutoff, beta, d_cutoff)

    # Express joints in the floor-board world frame if calibration defines one.
    # Otherwise leave them in the camera-1 OpenCV frame (consumers then apply the
    # legacy Z-up swap).  The frame is recorded in meta so consumers know which.
    world_frame = calib.get("world_frame")
    coordinate_frame = "opencv"
    if world_frame is not None:
        from ..calib.coordinate_system import apply_world_frame

        if progress_cb:
            progress_cb(93, "Transforming to floor world frame…")
        joints3d = apply_world_frame(joints3d, world_frame).astype(np.float32)
        coordinate_frame = "world"

    if progress_cb:
        progress_cb(95, "Saving results…")

    # Build meta
    timestamps = meta_left.get("timestamps", np.full(T_frames, np.nan, dtype=np.float32))
    if hasattr(timestamps, "__len__") and len(timestamps) > T_frames:
        timestamps = np.asarray(timestamps[:T_frames])

    meta_out: dict[str, Any] = {
        "fps": fps,
        "timestamps": np.asarray(timestamps, dtype=np.float32),
        "model_name": meta_left.get("model_name", "unknown"),
        "skeleton": "coco17",
        "image_size": meta_left.get("image_size", [0, 0]),
        "calibration_file": str(calib_path),
        "smoothing": "none" if no_smooth else "oneeuro",
        "smooth_params": {} if no_smooth else {"min_cutoff": min_cutoff, "beta": beta, "d_cutoff": d_cutoff},
        # "world" → joints3d already in the Z-up floor frame (no swap downstream);
        # "opencv" → camera-1 frame (consumers apply the X,Z,-Y swap, as before).
        "coordinate_frame": coordinate_frame,
        "world_frame": world_frame,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        joints3d=joints3d,
        conf3d=conf3d,
        repro_err=repro_arr,
        meta=np.array([meta_out], dtype=object),
    )

    if progress_cb:
        progress_cb(100, f"Saved {output_path}")

    return output_path
