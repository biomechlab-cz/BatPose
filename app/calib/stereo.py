"""Stereo calibration pipeline and calibration.yml I/O."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .board import make_detector
from .frame_select import FrameSelection, extract_calibration_frames

# Per-camera frame cap for Phase 1 intrinsics. Optimizer cost grows with the
# number of point sets, but accuracy plateaus around 30-60 well-distributed
# frames. We subsample by spatial coverage of the board centroid so the kept
# frames span the image plane.
PHASE1_MAX_FRAMES = 80

# Termination criteria for cv2.calibrateCamera. OpenCV's defaults can run
# unbounded on poorly-conditioned wide-angle data, which is what manifested as
# the GUI "freezing" during Phase 1 in real-world recordings.
_CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-5)

# Log the full cv2.fisheye.stereoCalibrate assertion only once per process —
# it fires on every fisheye calibration on OpenCV 4.11 (known binding bug) and
# would otherwise spam the console with the same multi-line traceback.
_FISHEYE_STEREO_DETAIL_LOGGED = False


def _spatial_subsample(
    detections: list,
    img_size: tuple[int, int],
    n_max: int,
    grid: int = 6,
) -> list:
    """Subsample *detections* to <= n_max while preserving spatial spread.

    Bins each detection by the centroid of its corner points into a grid×grid
    grid over the image, then picks round-robin from non-empty cells. Frames
    whose boards land in unique cells are kept first; this keeps the optimizer's
    geometric coverage while shrinking the data volume.
    """
    if len(detections) <= n_max:
        return list(detections)

    w, h = img_size
    cell_w = w / grid
    cell_h = h / grid

    bins: dict[tuple[int, int], list] = {}
    for det in detections:
        ctr = det.img_pts.reshape(-1, 2).mean(axis=0)
        ci = min(max(int(ctr[0] / cell_w), 0), grid - 1)
        cj = min(max(int(ctr[1] / cell_h), 0), grid - 1)
        bins.setdefault((ci, cj), []).append(det)

    chosen: list = []
    cell_keys = list(bins.keys())
    while len(chosen) < n_max:
        any_picked = False
        for k in cell_keys:
            if bins[k]:
                chosen.append(bins[k].pop(0))
                any_picked = True
                if len(chosen) >= n_max:
                    break
        if not any_picked:
            break
    return chosen


def _calibrate_camera_cancellable(
    obj_pts: list,
    img_pts: list,
    img_size: tuple[int, int],
    flags: int,
    criteria: tuple,
    cancel_check: Callable[[], bool] | None,
) -> tuple:
    """Run cv2.calibrateCamera in a worker thread, polling cancel_check while it runs.

    Returns the same 5-tuple as cv2.calibrateCamera: (rms, K, D, rvecs, tvecs).
    If cancel_check returns True the call raises RuntimeError("Cancelled");
    the underlying C++ call cannot be hard-killed but the Python caller bails
    immediately.
    """
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        future = ex.submit(
            cv2.calibrateCamera,
            obj_pts,
            img_pts,
            img_size,
            None,
            None,
            flags=flags,
            criteria=criteria,
        )
        while True:
            try:
                return future.result(timeout=0.25)
            except FutureTimeoutError:
                if cancel_check and cancel_check():
                    raise RuntimeError("Cancelled")
    finally:
        # Don't block on a runaway optimizer if we're bailing.
        ex.shutdown(wait=False)


def _avg_rotations(Rs: list[np.ndarray]) -> np.ndarray:
    """Average a list of rotation matrices via quaternion mean → orthonormalised."""
    # Project each R to a quaternion (Hamilton convention), average, renormalise.
    qs = []
    for R in Rs:
        # Robust quaternion-from-matrix (Shepperd's method).
        m00, m01, m02 = R[0]
        m10, m11, m12 = R[1]
        m20, m21, m22 = R[2]
        tr = m00 + m11 + m22
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = np.sqrt(1.0 + m00 - m11 - m22) * 2
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = np.sqrt(1.0 + m11 - m00 - m22) * 2
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = np.sqrt(1.0 + m22 - m00 - m11) * 2
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float64)
        if qs and np.dot(q, qs[0]) < 0:
            q = -q                           # fix antipodal hemisphere
        qs.append(q)
    q_avg = np.mean(qs, axis=0)
    q_avg /= np.linalg.norm(q_avg)
    w, x, y, z = q_avg
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),    2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),    1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _fisheye_reproj_rms(
    obj_pts: list[np.ndarray],
    ipts_l:  list[np.ndarray],
    ipts_r:  list[np.ndarray],
    K1: np.ndarray, D1: np.ndarray,
    K2: np.ndarray, D2: np.ndarray,
    R: np.ndarray, T: np.ndarray,
) -> float:
    """Proper joint reprojection RMS for a fisheye stereo calibration.

    For each stereo pair, solve PnP in the left view to recover the board
    pose (R1, T1), derive the right-view pose from the stereo R/T, then
    project the board's 3D corners through BOTH fisheye cameras and
    compare to the actual image points.  Returns the joint RMS in pixels.
    """
    sq_err_sum = 0.0
    n_pts = 0
    D_zero = np.zeros((4, 1), dtype=np.float64)
    for o, il, ir in zip(obj_pts, ipts_l, ipts_r, strict=False):
        if o.shape[0] < 6:
            continue
        try:
            und_l = cv2.fisheye.undistortPoints(il.reshape(-1, 1, 2), K1, D1, P=K1)
            ok, rvec1, tvec1 = cv2.solvePnP(
                o.reshape(-1, 1, 3), und_l.reshape(-1, 1, 2), K1, D_zero,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            continue
        if not ok:
            continue
        R1, _ = cv2.Rodrigues(rvec1)
        T1 = tvec1.reshape(3, 1)

        # Right-camera pose derived from the stereo R/T applied to the left pose.
        R2 = R @ R1
        T2 = (R @ T1 + T.reshape(3, 1))
        rvec2, _ = cv2.Rodrigues(R2)

        # Reproject through BOTH fisheye cameras.
        proj_l, _ = cv2.fisheye.projectPoints(
            o.reshape(-1, 1, 3).astype(np.float64),
            rvec1.reshape(1, 3), tvec1.reshape(1, 3), K1, D1,
        )
        proj_r, _ = cv2.fisheye.projectPoints(
            o.reshape(-1, 1, 3).astype(np.float64),
            rvec2.reshape(1, 3), T2.reshape(1, 3), K2, D2,
        )

        err_l = (proj_l.reshape(-1, 2) - il.reshape(-1, 2)) ** 2
        err_r = (proj_r.reshape(-1, 2) - ir.reshape(-1, 2)) ** 2
        sq_err_sum += float(err_l.sum() + err_r.sum())
        n_pts += err_l.shape[0] + err_r.shape[0]

    return float(np.sqrt(sq_err_sum / n_pts)) if n_pts else float("inf")


def _fisheye_per_pair_rms(
    obj_pts: list[np.ndarray],
    ipts_l:  list[np.ndarray],
    ipts_r:  list[np.ndarray],
    K1: np.ndarray, D1: np.ndarray,
    K2: np.ndarray, D2: np.ndarray,
    R: np.ndarray, T: np.ndarray,
) -> list[float]:
    """Per-pair joint reprojection RMS (px), one value per stereo pair.

    Same geometry as _fisheye_reproj_rms but reports each pair separately so
    outlier pairs (motion blur, a mis-interpolated ChArUco corner) can be
    identified and dropped before a final stereo refinement.  Pairs that can't
    be evaluated (too few corners / PnP failure) get inf so they sort as the
    worst candidates for removal.
    """
    D_zero = np.zeros((4, 1), dtype=np.float64)
    out: list[float] = []
    for o, il, ir in zip(obj_pts, ipts_l, ipts_r, strict=False):
        if o.shape[0] < 6:
            out.append(float("inf"))
            continue
        try:
            und_l = cv2.fisheye.undistortPoints(il.reshape(-1, 1, 2), K1, D1, P=K1)
            ok, rvec1, tvec1 = cv2.solvePnP(
                o.reshape(-1, 1, 3), und_l.reshape(-1, 1, 2), K1, D_zero,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            out.append(float("inf"))
            continue
        if not ok:
            out.append(float("inf"))
            continue
        R1, _ = cv2.Rodrigues(rvec1)
        T1 = tvec1.reshape(3, 1)
        R2 = R @ R1
        T2 = (R @ T1 + T.reshape(3, 1))
        rvec2, _ = cv2.Rodrigues(R2)
        proj_l, _ = cv2.fisheye.projectPoints(
            o.reshape(-1, 1, 3).astype(np.float64),
            rvec1.reshape(1, 3), tvec1.reshape(1, 3), K1, D1,
        )
        proj_r, _ = cv2.fisheye.projectPoints(
            o.reshape(-1, 1, 3).astype(np.float64),
            rvec2.reshape(1, 3), T2.reshape(1, 3), K2, D2,
        )
        err_l = (proj_l.reshape(-1, 2) - il.reshape(-1, 2)) ** 2
        err_r = (proj_r.reshape(-1, 2) - ir.reshape(-1, 2)) ** 2
        n = err_l.shape[0] + err_r.shape[0]
        out.append(float(np.sqrt((err_l.sum() + err_r.sum()) / n)) if n else float("inf"))
    return out


def _fisheye_extrinsics_from_solvepnp(
    obj_pts: list[np.ndarray],
    ipts_l:  list[np.ndarray],
    ipts_r:  list[np.ndarray],
    K1: np.ndarray, D1: np.ndarray,
    K2: np.ndarray, D2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute stereo extrinsics by averaging per-pair solvePnP solutions.

    Fallback used when cv2.fisheye.stereoCalibrate raises its OutputArray
    assertion.  For each pair (board_3d, img_l, img_r):
        • Undistort both views, solvePnP to recover board poses R1/T1 and R2/T2
        • Pair-wise stereo: R_pair = R2 @ R1ᵀ ,  T_pair = T2 − R_pair @ T1
    Then average R via quaternion mean and T via plain mean.  The final RMS
    is the genuine joint reprojection error computed via _fisheye_reproj_rms.
    """
    Rs: list[np.ndarray] = []
    Ts: list[np.ndarray] = []
    D_zero = np.zeros((4, 1), dtype=np.float64)

    skipped = 0
    for o, il, ir in zip(obj_pts, ipts_l, ipts_r, strict=False):
        # cv2.solvePnP's default DLT init requires ≥ 6 correspondences.
        # Any pair below that → silently skip (the joint of all pairs is what
        # matters, and one short pair would crash the whole calibration).
        if o.shape[0] < 6:
            skipped += 1
            continue
        try:
            und_l = cv2.fisheye.undistortPoints(il.reshape(-1, 1, 2), K1, D1, P=K1)
            und_r = cv2.fisheye.undistortPoints(ir.reshape(-1, 1, 2), K2, D2, P=K2)
            ok_l, rvec_l, tvec_l = cv2.solvePnP(
                o.reshape(-1, 1, 3), und_l.reshape(-1, 1, 2), K1, D_zero,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            ok_r, rvec_r, tvec_r = cv2.solvePnP(
                o.reshape(-1, 1, 3), und_r.reshape(-1, 1, 2), K2, D_zero,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            skipped += 1
            continue
        if not (ok_l and ok_r):
            skipped += 1
            continue
        R1, _ = cv2.Rodrigues(rvec_l)
        R2, _ = cv2.Rodrigues(rvec_r)
        R_pair = R2 @ R1.T
        T_pair = (tvec_r - R_pair @ tvec_l).reshape(3, 1)
        Rs.append(R_pair)
        Ts.append(T_pair)

    if skipped:
        print(
            f"[fisheye-2phase fallback] Skipped {skipped} of {len(obj_pts)} "
            f"stereo pair(s) (insufficient corresponding ChArUco corners or PnP failure)."
        )
    if not Rs:
        raise RuntimeError(
            "Fisheye solvePnP fallback found no usable pair — every stereo "
            "frame had < 6 matched ChArUco corners or solvePnP failed. "
            "Recapture with the board fully visible in BOTH cameras so each "
            "pair shares more interior corners."
        )

    R = _avg_rotations(Rs)
    T = np.mean(np.stack(Ts, axis=0), axis=0).reshape(3, 1)

    # Genuine joint reprojection RMS — replaces the bogus distortion-magnitude
    # metric the earlier draft was returning.
    rms = _fisheye_reproj_rms(obj_pts, ipts_l, ipts_r, K1, D1, K2, D2, R, T)
    return R, T, rms


def _intersect_stereo_points(
    selections: list,
    dtype: type,
    cancel_check: Callable[[], bool] | None = None,
    min_per_frame: int = 6,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Build matched (obj, imgL, imgR) point arrays from FrameSelection list.

    For ChArUco boards each camera may interpolate a different subset of corner
    IDs — cv2.stereoCalibrate requires identical 3D↔2D correspondences in both
    views, so we intersect by id.  Frames that share fewer than *min_per_frame*
    corners after intersection are silently dropped (they would destabilise the
    optimiser).  Default 6 matches the lower bound for cv2.solvePnP (used by
    the fisheye fallback path) and is the practical minimum for a stable pose
    even in cv2.stereoCalibrate.
    """
    obj_pts: list[np.ndarray] = []
    img_l: list[np.ndarray] = []
    img_r: list[np.ndarray] = []

    for i, sel in enumerate(selections):
        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        det_l = sel.det_left
        det_r = sel.det_right
        ids_l = det_l.ids
        ids_r = det_r.ids

        # No ids → assume both views see the same N points in order (chessboard).
        # We still verify shape compatibility before trusting that.
        if ids_l is None or ids_r is None:
            if len(det_l.obj_pts) != len(det_r.img_pts):
                raise ValueError(
                    f"Frame {i}: left={len(det_l.obj_pts)} corners, "
                    f"right={len(det_r.img_pts)} corners; detector did not record "
                    "per-point ids so intersection is impossible. Use a detector "
                    "that returns ids (CharucoDetector/ChessboardDetector do)."
                )
            o = det_l.obj_pts
            il = det_l.img_pts
            ir = det_r.img_pts
        else:
            # Intersect ids and re-index both views.
            common, idx_l, idx_r = np.intersect1d(
                ids_l, ids_r, return_indices=True, assume_unique=True,
            )
            if len(common) < min_per_frame:
                # Silently skip — not enough overlap to constrain extrinsics from this pair
                continue
            o = det_l.obj_pts[idx_l]
            il = det_l.img_pts[idx_l]
            ir = det_r.img_pts[idx_r]

        obj_pts.append(o.reshape(-1, 1, 3).astype(dtype))
        img_l.append(il.reshape(-1, 1, 2).astype(dtype))
        img_r.append(ir.reshape(-1, 1, 2).astype(dtype))

    if len(obj_pts) < 3:
        raise RuntimeError(
            f"Only {len(obj_pts)} stereo pair(s) have ≥ {min_per_frame} matched corners "
            "after intersecting by ChArUco id. Recapture with both cameras seeing more "
            "of the board (move it into the overlap zone of the two FOVs)."
        )
    return obj_pts, img_l, img_r


def calibrate_stereo(
    selections: list[FrameSelection],
    img_size: tuple[int, int],
    fix_intrinsics: bool = False,
    progress_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    all_det_l: list | None = None,
    all_det_r: list | None = None,
    intrinsics_flags: int = 0,
) -> dict[str, Any]:
    """
    Run stereo calibration from a list of FrameSelection objects.

    When *all_det_l* and *all_det_r* are supplied and each has more detections
    than the stereo pairs alone, two-phase calibration is used automatically:

      Phase 1 — per-camera intrinsics from a spatially-balanced subsample of
                 single-camera detections (capped at PHASE1_MAX_FRAMES).
      Phase 2 — extrinsics-only via cv2.stereoCalibrate with CALIB_FIX_INTRINSIC
                 on the stereo pairs.

    Args:
        selections: list produced by extract_calibration_frames
        img_size: (width, height) of source frames
        fix_intrinsics: if True force CALIB_FIX_INTRINSIC even without all_det lists
        progress_cb: optional progress callback
        cancel_check: optional cancel callback (polled every ~250 ms during
                      cv2.calibrateCamera)
        all_det_l: all left-camera DetectionResult objects (including non-paired)
        all_det_r: all right-camera DetectionResult objects (including non-paired)
        intrinsics_flags: extra cv2.calibrateCamera flags (e.g.
                          cv2.CALIB_RATIONAL_MODEL for severely wide-angle lenses)

    Returns:
        Dict with K1, D1, K2, D2, R, T, E, F, rms, n_frames, image_size
    """
    w, h = img_size
    if w <= 0 or h <= 0:
        raise ValueError(
            f"img_size must have positive width and height, got img_size={img_size!r}"
        )

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

    obj_pts_all, img_pts_l_all, img_pts_r_all = _intersect_stereo_points(
        selections, dtype=np.float32, cancel_check=cancel_check,
    )

    # Decide between two-phase and single-phase calibration.
    use_two_phase = (
        all_det_l is not None
        and all_det_r is not None
        and len(all_det_l) > len(selections)
        and len(all_det_r) > len(selections)
    )

    if use_two_phase:
        # Subsample per-camera detections by spatial coverage to a manageable
        # size. With wide-angle lenses, hundreds of frames make the LM optimizer
        # crawl; ~80 well-distributed frames give equivalent intrinsics in seconds.
        sub_l = _spatial_subsample(all_det_l, img_size, PHASE1_MAX_FRAMES)
        sub_r = _spatial_subsample(all_det_r, img_size, PHASE1_MAX_FRAMES)

        obj_l = [d.obj_pts.reshape(-1, 1, 3).astype(np.float32) for d in sub_l]
        ipts_l = [d.img_pts.reshape(-1, 1, 2).astype(np.float32) for d in sub_l]
        obj_r = [d.obj_pts.reshape(-1, 1, 3).astype(np.float32) for d in sub_r]
        ipts_r = [d.img_pts.reshape(-1, 1, 2).astype(np.float32) for d in sub_r]

        if progress_cb:
            progress_cb(
                15,
                f"Two-phase calibration: Phase 1 — left intrinsics from "
                f"{len(obj_l)} of {len(all_det_l)} frames…",
            )
        _, K1, D1, _, _ = _calibrate_camera_cancellable(
            obj_l, ipts_l, img_size,
            flags=intrinsics_flags,
            criteria=_CALIB_CRITERIA,
            cancel_check=cancel_check,
        )

        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        if progress_cb:
            progress_cb(
                35,
                f"Two-phase calibration: Phase 1 — right intrinsics from "
                f"{len(obj_r)} of {len(all_det_r)} frames…",
            )
        _, K2, D2, _, _ = _calibrate_camera_cancellable(
            obj_r, ipts_r, img_size,
            flags=intrinsics_flags,
            criteria=_CALIB_CRITERIA,
            cancel_check=cancel_check,
        )

        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        if progress_cb:
            progress_cb(
                55,
                f"Two-phase calibration: Phase 2 — extrinsics from "
                f"{len(selections)} stereo pairs…",
            )
        stereo_flags = cv2.CALIB_FIX_INTRINSIC
    else:
        if progress_cb:
            progress_cb(20, "Calibrating left camera…")

        _, K1, D1, _, _ = _calibrate_camera_cancellable(
            obj_pts_all, img_pts_l_all, img_size,
            flags=intrinsics_flags,
            criteria=_CALIB_CRITERIA,
            cancel_check=cancel_check,
        )

        if progress_cb:
            progress_cb(40, "Calibrating right camera…")

        _, K2, D2, _, _ = _calibrate_camera_cancellable(
            obj_pts_all, img_pts_r_all, img_size,
            flags=intrinsics_flags,
            criteria=_CALIB_CRITERIA,
            cancel_check=cancel_check,
        )

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


def calibrate_stereo_fisheye(
    selections: list[FrameSelection],
    img_size: tuple[int, int],
    progress_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    all_det_l: list | None = None,
    all_det_r: list | None = None,
) -> dict[str, Any]:
    """
    Run stereo calibration using OpenCV's fisheye camera model.

    Use this when lenses have a FOV ≥ 150°.  The fisheye model uses a
    θ-based projection (equidistant / stereographic variants) that can
    represent extreme barrel distortion accurately with only 4 coefficients
    (k1-k4), unlike the polynomial model which breaks down for wide FOVs.

    When *all_det_l* / *all_det_r* are larger than the stereo pair set,
    intrinsics are calibrated from the per-camera pools (independent of
    whether the other camera saw the board) and only extrinsics use the
    stereo pairs.  This is the recommended workflow for fisheye stereo
    because the per-camera FOV overlap is small.

    Returns the same dict schema as calibrate_stereo so callers are
    interchangeable; D1/D2 will be (4,) fisheye coefficients.
    """
    w, h = img_size
    if w <= 0 or h <= 0:
        raise ValueError(f"img_size must be positive, got {img_size!r}")
    if len(selections) < 3:
        raise ValueError(
            f"Need at least 3 paired frames for fisheye calibration, got {len(selections)}."
        )

    if progress_cb:
        progress_cb(5, f"Fisheye calibration from {len(selections)} frames…")

    # Build point arrays — intersect by corner ID so each pair has identical sets.
    obj_pts_all, img_pts_l_all, img_pts_r_all = _intersect_stereo_points(
        selections, dtype=np.float64, cancel_check=cancel_check,
    )

    # ------------------------------------------------------------------ #
    # Initial K: focal length ≈ width/2 is a stable starting estimate   #
    # for a fisheye lens.  cx/cy at image centre is exact for most       #
    # digital sensors.  D starts at zero; the optimizer refines freely.  #
    # ------------------------------------------------------------------ #
    _w, _h = img_size
    _K_init = np.array(
        [[_w / 2.0, 0.0, _w / 2.0],
         [0.0, _w / 2.0, _h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    _D_init = np.zeros((4, 1), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Pre-screen frames: cv2.fisheye.calibrate calls InitExtrinsics on   #
    # every frame before the LM loop.  InitExtrinsics asserts            #
    #   fabs(norm_u1) > 0                                                #
    # where norm_u1 is the undistorted norm of the FIRST image corner.   #
    # If that corner undistorts to (0, 0) — i.e. it lands exactly on     #
    # the principal point — the assertion fires as a hard crash.         #
    #                                                                    #
    # We screen each frame with undistortPoints before calling           #
    # calibrate and silently drop any frame that would trigger this.     #
    # ------------------------------------------------------------------ #
    def _valid_for_fisheye(img_pts_list: list, K: np.ndarray, D: np.ndarray) -> list[int]:
        """Return indices of frames whose first corner normalises to non-zero."""
        good = []
        for idx, ipts in enumerate(img_pts_list):
            # Check only the first corner (what InitExtrinsics checks).
            first = ipts[0].reshape(1, 1, 2)
            undist = cv2.fisheye.undistortPoints(first, K, D)
            if float(np.linalg.norm(undist)) > 1e-9:
                good.append(idx)
        return good

    valid_l = set(_valid_for_fisheye(img_pts_l_all, _K_init, _D_init))
    valid_r = set(_valid_for_fisheye(img_pts_r_all, _K_init, _D_init))
    valid_both = sorted(valid_l & valid_r)

    n_dropped = len(obj_pts_all) - len(valid_both)
    if n_dropped:
        if progress_cb:
            progress_cb(
                10,
                f"Fisheye: dropped {n_dropped} degenerate frame(s) "
                f"(corner at principal point) — {len(valid_both)} remain.",
            )

    if len(valid_both) < 3:
        raise RuntimeError(
            f"Only {len(valid_both)} frame(s) pass the fisheye sanity check "
            f"({n_dropped} dropped). "
            "Recapture with the board well away from the image centre, "
            "or switch to 'Wide-angle (rational model)'."
        )

    obj_pts  = [obj_pts_all[i]   for i in valid_both]
    ipts_l   = [img_pts_l_all[i] for i in valid_both]
    ipts_r   = [img_pts_r_all[i] for i in valid_both]

    # CALIB_RECOMPUTE_EXTRINSIC refines R/T each LM iteration.
    # CALIB_FIX_SKEW: skew is 0 on every digital sensor.
    # CALIB_USE_INTRINSIC_GUESS: start from _K_init/_D_init.
    # CALIB_CHECK_COND is intentionally omitted (raises on large Jacobian
    # condition numbers which are common before a good K is established).
    fisheye_flags_single = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    def _fisheye_intrinsics_from_dets(
        all_det: list, side_label: str, init_progress: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Calibrate fisheye intrinsics from a per-camera detection pool.

        Uses retry-with-frame-dropping when OpenCV's `InitExtrinsics`
        asserts.  That assertion fires per-frame *during* the optimizer
        loop when one frame's board pose recovery becomes rank-deficient;
        the C++ doesn't say which frame, so we drop a random one and retry
        until a working subset is found.
        """
        sub = _spatial_subsample(all_det, img_size, PHASE1_MAX_FRAMES)
        obj_p = [d.obj_pts.reshape(-1, 1, 3).astype(np.float64) for d in sub]
        ipt_p = [d.img_pts.reshape(-1, 1, 2).astype(np.float64) for d in sub]
        # Drop frames whose first corner is degenerate under the initial K/D.
        kept = [i for i in range(len(ipt_p))
                if float(np.linalg.norm(
                    cv2.fisheye.undistortPoints(ipt_p[i][0].reshape(1, 1, 2),
                                                _K_init, _D_init))) > 1e-9]
        if len(kept) < 3:
            raise RuntimeError(
                f"Fisheye {side_label}: only {len(kept)} of {len(sub)} frames "
                "survive the principal-point sanity check. "
                "Recapture with the board well off-axis or switch to wide-angle."
            )
        obj_p = [obj_p[i] for i in kept]
        ipt_p = [ipt_p[i] for i in kept]
        n_total = len(obj_p)

        if progress_cb:
            progress_cb(
                init_progress,
                f"Fisheye: {side_label} intrinsics from {n_total} of "
                f"{len(all_det)} frames…",
            )

        # Retry loop: progressively drop random frames on InitExtrinsics failure.
        rng = np.random.default_rng(seed=42)
        max_attempts = min(15, n_total - 3)
        last_exc: cv2.error | None = None
        indices = list(range(n_total))
        dropped: set[int] = set()

        for attempt in range(max_attempts + 1):
            use = [i for i in indices if i not in dropped]
            if len(use) < 4:
                break
            o_use = [obj_p[i] for i in use]
            ip_use = [ipt_p[i] for i in use]
            try:
                rms_, K_, D_, _, _ = cv2.fisheye.calibrate(
                    o_use, ip_use, img_size,
                    _K_init.copy(), _D_init.copy(),
                    flags=fisheye_flags_single,
                    criteria=criteria,
                )
                if attempt > 0 and progress_cb:
                    progress_cb(
                        init_progress + 5,
                        f"Fisheye: {side_label} succeeded after dropping "
                        f"{attempt} problem frame(s) ({len(use)} used).",
                    )
                return K_, D_, float(rms_)
            except cv2.error as exc:
                last_exc = exc
                # Drop one more random frame and retry. Only do this for the
                # well-known degeneracy assertions — propagate everything else.
                msg = str(exc)
                if not ("InitExtrinsics" in msg or "norm_u1" in msg
                        or "CalibrateExtrinsics" in msg or "InitIntrinsics" in msg):
                    raise
                candidates = [i for i in indices if i not in dropped]
                if not candidates:
                    break
                dropped.add(int(rng.choice(candidates)))

        # Exhausted retries — surface a clean message that includes the OpenCV detail.
        raise RuntimeError(
            f"Fisheye {side_label}-camera calibration failed even after dropping "
            f"{len(dropped)} of {n_total} candidate frames.\n\n"
            f"Underlying OpenCV error: {last_exc}\n\n"
            "Suggestions:\n"
            "• Switch to 'Wide-angle (rational model)' in the Lens selector — its\n"
            "  polynomial model handles your lens without the fisheye degeneracies.\n"
            "• Recapture frames with the board well off the optical axis at various tilts;\n"
            "  the fisheye optimiser fails on boards held parallel to the image plane.\n"
            "• Make sure at least one ChArUco corner is far from the principal point\n"
            "  (image centre) in every captured frame."
        ) from last_exc

    # Decide between two-phase (preferred for fisheye) and single-phase.
    use_two_phase = (
        all_det_l is not None and all_det_r is not None
        and len(all_det_l) > len(valid_both)
        and len(all_det_r) > len(valid_both)
    )

    if use_two_phase:
        K1, D1, rms_l = _fisheye_intrinsics_from_dets(all_det_l, "left",  20)
        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")
        K2, D2, rms_r = _fisheye_intrinsics_from_dets(all_det_r, "right", 40)
        print(
            f"[fisheye-2phase] Phase-1 intrinsics RMS: "
            f"left={rms_l:.3f} px (from {len(all_det_l)} dets), "
            f"right={rms_r:.3f} px (from {len(all_det_r)} dets)"
        )
    else:
        if progress_cb:
            progress_cb(20, f"Fisheye: calibrating left camera ({len(obj_pts)} frames)…")
        try:
            rms_l, K1, D1, _, _ = cv2.fisheye.calibrate(
                obj_pts, ipts_l, img_size,
                _K_init.copy(), _D_init.copy(),
                flags=fisheye_flags_single,
                criteria=criteria,
            )
        except cv2.error as exc:
            raise RuntimeError(
                f"Fisheye left-camera calibration failed: {exc}\n\n"
                "Suggestions:\n"
                "• Switch to 'Wide-angle (rational model)' in the Lens selector.\n"
                "• Recapture frames with board clearly visible, not centred on axis.\n"
                "• Use ChArUco board instead of chessboard."
            ) from exc

        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        if progress_cb:
            progress_cb(40, f"Fisheye: left RMS={rms_l:.3f}px — calibrating right camera…")
        try:
            rms_r, K2, D2, _, _ = cv2.fisheye.calibrate(
                obj_pts, ipts_r, img_size,
                _K_init.copy(), _D_init.copy(),
                flags=fisheye_flags_single,
                criteria=criteria,
            )
        except cv2.error as exc:
            raise RuntimeError(
                f"Fisheye right-camera calibration failed: {exc}\n\n"
                "Suggestions:\n"
                "• Switch to 'Wide-angle (rational model)' in the Lens selector.\n"
                "• Recapture frames keeping board away from image centre.\n"
                "• Use ChArUco board instead of chessboard."
            ) from exc

    if cancel_check and cancel_check():
        raise RuntimeError("Cancelled")

    if progress_cb:
        progress_cb(60, f"Fisheye: right RMS={rms_r:.3f}px — stereo calibration…")

    # ── cv2.fisheye.stereoCalibrate defensive prep ──────────────────────────
    # The OpenCV Python binding for fisheye.stereoCalibrate is notoriously
    # picky about input shapes / array provenance.  The K/D arrays returned
    # by fisheye.calibrate carry "fixedSize" metadata that triggers the
    # !fixedSize() assertion in matrix_wrap.cpp when stereoCalibrate's
    # output-array allocator tries to (re-)create them.  Workarounds applied:
    #   1. Re-create K/D as fresh, contiguous, exact-shape float64 arrays.
    #   2. Pre-allocate R and T as positional output buffers.
    #   3. Ensure obj/img point arrays are float64 lists of (N_i, 1, k).
    # ─────────────────────────────────────────────────────────────────────────
    K1 = np.ascontiguousarray(np.asarray(K1, dtype=np.float64)).reshape(3, 3)
    K2 = np.ascontiguousarray(np.asarray(K2, dtype=np.float64)).reshape(3, 3)
    D1 = np.ascontiguousarray(np.asarray(D1, dtype=np.float64)).reshape(4, 1)
    D2 = np.ascontiguousarray(np.asarray(D2, dtype=np.float64)).reshape(4, 1)
    obj_pts = [np.ascontiguousarray(a, dtype=np.float64) for a in obj_pts]
    ipts_l  = [np.ascontiguousarray(a, dtype=np.float64) for a in ipts_l]
    ipts_r  = [np.ascontiguousarray(a, dtype=np.float64) for a in ipts_r]

    # cv2.fisheye.stereoCalibrate returns (rms, K1, D1, K2, D2, R, T) in
    # OpenCV ≤4.6 but inserts (rvecs, tvecs) before (flags, criteria) in the
    # PARAMETER list of OpenCV ≥4.x as well as returning them.  CRITICAL:
    # flags and criteria MUST be passed as KEYWORD arguments.  If passed
    # positionally (the historical 4.6 layout: ...R, T, flags, criteria),
    # then on 4.11 the integer flags value binds to the new `rvecs` output
    # parameter — OpenCV tries to use the int as an OutputArray and raises
    # the "!fixedSize() ... in cv::_OutputArray::create" assertion, while
    # `flags` silently defaults to 0 (so CALIB_FIX_INTRINSIC is lost).
    # Keyword binding is correct on every version regardless of the inserted
    # rvecs/tvecs positional parameters; we also let OpenCV allocate R and T.
    def _run_stereo(o_list, l_list, r_list):
        """Compute fisheye stereo extrinsics for the given pairs.

        We try cv2.fisheye.stereoCalibrate first, but on OpenCV 4.11 its Python
        binding raises a `!fixedSize()` assertion in cv::_OutputArray::create
        for the ragged (variable-corner-count) ChArUco point lists this app
        produces — a known upstream binding bug, not a data problem.  The
        per-pair solvePnP-averaging path is therefore the normal, reliable
        route here and, combined with outlier rejection, yields sub-pixel RMS.
        We attempt stereoCalibrate anyway in case a future OpenCV fixes it.
        """
        try:
            _res = cv2.fisheye.stereoCalibrate(
                o_list, l_list, r_list,
                K1, D1, K2, D2,
                img_size,
                flags=cv2.fisheye.CALIB_FIX_INTRINSIC,
                criteria=criteria,
            )
            _rms, _, _, _, _, _R, _T = _res[:7]
            return _R, _T, float(_rms), "stereoCalibrate"
        except cv2.error as _exc:
            # Calm, single-line note — this is the expected path on OpenCV 4.11,
            # not an error the user needs to act on.  Log the full assertion
            # only once per process for diagnostics.
            global _FISHEYE_STEREO_DETAIL_LOGGED
            if not _FISHEYE_STEREO_DETAIL_LOGGED:
                print(
                    "[fisheye-2phase] note: cv2.fisheye.stereoCalibrate is "
                    "unavailable on this OpenCV build (known binding bug); using "
                    "the per-pair solvePnP extrinsics path. Detail: "
                    f"{str(_exc).splitlines()[-1].strip()}"
                )
                _FISHEYE_STEREO_DETAIL_LOGGED = True
            _R, _T, _rms = _fisheye_extrinsics_from_solvepnp(
                o_list, l_list, r_list, K1, D1, K2, D2,
            )
            return _R, _T, float(_rms), "solvePnP-averaging"

    # ── Initial stereo solve on all pairs ───────────────────────────────────
    R, T, rms, stereo_path = _run_stereo(obj_pts, ipts_l, ipts_r)

    # ── Outlier rejection: drop pairs whose joint reproj error is far above
    #    the median, then refine on the inliers.  ChArUco auto-capture at high
    #    frame rate almost always yields a few motion-blurred or mis-interpolated
    #    pairs that dominate the RMS; removing them typically halves it. ────────
    per_pair = _fisheye_per_pair_rms(obj_pts, ipts_l, ipts_r, K1, D1, K2, D2, R, T)
    finite = [e for e in per_pair if np.isfinite(e)]
    if len(finite) >= 6:
        median = float(np.median(finite))
        # A pair is an outlier if its error exceeds 2.5× the median AND is over
        # 1.5 px (so we never reject when everything is already excellent).
        thresh = max(1.5, 2.5 * median)
        keep = [i for i, e in enumerate(per_pair) if np.isfinite(e) and e <= thresh]
        n_drop = len(obj_pts) - len(keep)
        if n_drop > 0 and len(keep) >= 6:
            print(
                f"[fisheye-2phase] Outlier rejection: dropping {n_drop} of "
                f"{len(obj_pts)} stereo pair(s) with per-pair RMS > {thresh:.2f} px "
                f"(median {median:.2f} px); refining on {len(keep)} inliers."
            )
            if progress_cb:
                progress_cb(80, f"Refining stereo on {len(keep)} inlier pairs…")
            o_in = [obj_pts[i] for i in keep]
            l_in = [ipts_l[i]  for i in keep]
            r_in = [ipts_r[i]  for i in keep]
            R2_, T2_, rms2_, path2_ = _run_stereo(o_in, l_in, r_in)
            rms2_joint = _fisheye_reproj_rms(o_in, l_in, r_in, K1, D1, K2, D2, R2_, T2_)
            rms_full_joint = _fisheye_reproj_rms(obj_pts, ipts_l, ipts_r, K1, D1, K2, D2, R, T)
            # Accept the refinement only if it actually improved the inlier fit.
            if rms2_joint < rms_full_joint:
                R, T, rms, stereo_path = R2_, T2_, rms2_, f"{path2_} (+outlier reject)"
                # Use the inlier pairs as the working set for the reported metric.
                obj_pts, ipts_l, ipts_r = o_in, l_in, r_in

    # Always compute the genuine joint reprojection RMS so we have an
    # apples-to-apples quality number regardless of which path produced R/T.
    rms_joint = _fisheye_reproj_rms(obj_pts, ipts_l, ipts_r, K1, D1, K2, D2, R, T)
    print(
        f"[fisheye-2phase] Phase-2 extrinsics path: {stereo_path}\n"
        f"[fisheye-2phase] Reported RMS: {rms:.3f} px | "
        f"Joint reproj RMS: {rms_joint:.3f} px (true quality metric)"
    )
    # Surface the joint RMS as the headline number — it's the meaningful one.
    rms = rms_joint

    if progress_cb:
        progress_cb(90, f"Fisheye done. RMS={rms:.3f}px ({stereo_path})")

    # E and F are not produced by fisheye.stereoCalibrate; fill with zeros so
    # save_calibration / load_calibration remain schema-compatible.
    E = np.zeros((3, 3), dtype=np.float64)
    F = np.zeros((3, 3), dtype=np.float64)

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
        "n_frames": len(obj_pts),   # actual frames used (after degenerate-frame filter)
        "image_size": list(img_size),
        "lens_model": "fisheye",
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
    intrinsics_flags: int = 0,
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
        intrinsics_flags: extra cv2.calibrateCamera flags
                          (e.g. cv2.CALIB_RATIONAL_MODEL for wide-angle lenses)

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
        intrinsics_flags=intrinsics_flags,
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
        # Distortion model: "standard" (pinhole radial-tangential) or "fisheye"
        # (θ-based).  Triangulation MUST use the matching model — without this
        # key a fisheye calibration would be undistorted as pinhole and produce
        # garbage 3D (reprojection errors in the thousands of pixels).
        "lens_model": calib_data.get("lens_model", "standard"),
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
        # Default "standard" so calibrations saved before this key existed
        # still load (they were all pinhole).
        "lens_model": data.get("lens_model", "standard"),
    }
