"""Triangulation and reprojection utilities (ADR-004)."""

from __future__ import annotations

import cv2
import numpy as np


def undistort_points(
    pts: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    fisheye: bool = False,
) -> np.ndarray:
    """
    Undistort 2D pixel points and normalize by the camera matrix.

    Returns points in normalized camera coordinates (not pixels).

    CRITICAL: the distortion model must match the one used at calibration.
    A fisheye calibration stores 4 θ-based coefficients; feeding those to the
    pinhole cv2.undistortPoints (which reads them as [k1,k2,p1,p2]) produces
    wildly wrong rays and triangulated points with reprojection errors in the
    thousands of pixels.  Pass fisheye=True for fisheye calibrations.

    Args:
        pts:     [N, 2] float32/float64 pixel coordinates
        K:       [3, 3] camera intrinsic matrix
        D:       distortion coefficients (5/8 for pinhole, 4 for fisheye)
        fisheye: use cv2.fisheye.undistortPoints instead of the pinhole model

    Returns:
        [N, 2] float64 normalized image coordinates
    """
    pts_in = pts.reshape(-1, 1, 2).astype(np.float64)
    Kf = K.astype(np.float64)
    Df = D.astype(np.float64)
    if fisheye:
        # fisheye D must be exactly 4 coefficients, shape (4,1) or (1,4).
        pts_out = cv2.fisheye.undistortPoints(pts_in, Kf, Df.reshape(4, 1))
    else:
        pts_out = cv2.undistortPoints(pts_in, Kf, Df, P=None)
    return pts_out.reshape(-1, 2)


def triangulate_points_dlt(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1_norm: np.ndarray,
    pts2_norm: np.ndarray,
) -> np.ndarray:
    """
    Triangulate 3D points from two views using the DLT method.

    Args:
        P1:        [3, 4] projection matrix for camera 1
        P2:        [3, 4] projection matrix for camera 2
        pts1_norm: [N, 2] normalized image coordinates from camera 1
                   (output of undistort_points with P=None)
        pts2_norm: [N, 2] normalized image coordinates from camera 2

    Returns:
        pts3d: [N, 3] float32 3D points in camera-1 coordinate frame (metres)
    """
    pts4d = cv2.triangulatePoints(
        P1.astype(np.float64),
        P2.astype(np.float64),
        pts1_norm.T.astype(np.float64),
        pts2_norm.T.astype(np.float64),
    )  # [4, N]

    # Convert homogeneous to 3D
    w = pts4d[3:4, :]  # avoid divide-by-zero
    w = np.where(np.abs(w) < 1e-10, 1e-10, w)
    pts3d = (pts4d[:3, :] / w).T  # [N, 3]
    return pts3d.astype(np.float32)


def triangulate_frame_pair(
    kp2d_l: np.ndarray,
    kp2d_r: np.ndarray,
    conf_l: np.ndarray,
    conf_r: np.ndarray,
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    min_conf: float = 0.3,
    max_reproj_err: float = 20.0,
    lens_model: str = "standard",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate one stereo frame of 2D keypoints into 3D.

    Single-frame analogue of recon3d.pipeline.reconstruct3d — same DLT
    triangulation, reprojection-error gate, and confidence rule, but without
    the time loop and OneEuro smoothing.  Designed for live use.

    Args:
        kp2d_l, kp2d_r:  [P, J, 2] or [J, 2] pixel keypoints
        conf_l, conf_r:  [P, J] or [J] confidences
        K1, D1, K2, D2, R, T:  stereo calibration
        min_conf:        joints with either-view conf below this → conf3d=0
        max_reproj_err:  joints with reprojection err above this → conf3d=0
        lens_model:      "fisheye" to undistort/reproject with the fisheye model;
                         anything else uses the standard pinhole model.  MUST
                         match the model used during calibration or the 3D
                         output is garbage (reproj error in the thousands).

    Returns:
        (joints3d, conf3d, repro_err) — shapes [P, J, 3], [P, J], [P, J].
        joints3d entries for rejected joints are zeroed.
    """
    fisheye = lens_model == "fisheye"

    # Promote single-person to [1, J, …]
    if kp2d_l.ndim == 2:
        kp2d_l = kp2d_l[None]
        kp2d_r = kp2d_r[None]
        conf_l = conf_l[None]
        conf_r = conf_r[None]
    P, J, _ = kp2d_l.shape

    P1_norm = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2_norm = np.hstack([R, T.reshape(3, 1)])
    R1 = np.eye(3)
    t1 = np.zeros(3)
    t2 = T.flatten()

    out_3d = np.zeros((P, J, 3), dtype=np.float32)
    out_conf = np.zeros((P, J), dtype=np.float32)
    out_err = np.full((P, J), np.inf, dtype=np.float32)

    for p in range(P):
        pts_l = kp2d_l[p]
        pts_r = kp2d_r[p]
        c_l = conf_l[p]
        c_r = conf_r[p]

        pts_l_norm = undistort_points(pts_l, K1, D1, fisheye=fisheye)
        pts_r_norm = undistort_points(pts_r, K2, D2, fisheye=fisheye)
        pts3d = triangulate_points_dlt(P1_norm, P2_norm, pts_l_norm, pts_r_norm)

        err_l = reprojection_error(pts3d, K1, D1, R1, t1, pts_l, fisheye=fisheye)
        err_r = reprojection_error(pts3d, K2, D2, R, t2, pts_r, fisheye=fisheye)
        err_mean = (err_l + err_r) * 0.5
        valid = (c_l >= min_conf) & (c_r >= min_conf) & (err_mean <= max_reproj_err)

        out_3d[p] = pts3d
        out_conf[p] = np.minimum(c_l, c_r) * valid.astype(np.float32)
        out_err[p] = err_mean
        out_3d[p, ~valid] = 0.0

    return out_3d, out_conf, out_err


def reprojection_error(
    pts3d: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    pts2d_observed: np.ndarray,
    fisheye: bool = False,
) -> np.ndarray:
    """
    Compute per-point reprojection error in pixels.

    Args:
        pts3d:         [N, 3] 3D points
        K:             [3, 3] camera intrinsic matrix
        D:             distortion coefficients
        R:             [3, 3] rotation matrix (world→camera)
        t:             [3] or [3, 1] translation vector
        pts2d_observed: [N, 2] observed 2D keypoints (original pixel coords)
        fisheye:       use cv2.fisheye.projectPoints (must match calibration)

    Returns:
        err: [N] float32 per-point reprojection error (pixels)
    """
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)

    if fisheye:
        # fisheye.projectPoints is picky: object points must be (N,1,3) float64
        # contiguous, distortion exactly 4 coeffs.
        obj = np.ascontiguousarray(pts3d.reshape(-1, 1, 3), dtype=np.float64)
        proj, _ = cv2.fisheye.projectPoints(
            obj,
            rvec,
            tvec,
            K.astype(np.float64),
            D.astype(np.float64).reshape(4, 1),
        )
    else:
        proj, _ = cv2.projectPoints(
            pts3d.astype(np.float64),
            rvec,
            tvec,
            K.astype(np.float64),
            D.astype(np.float64),
        )  # [N, 1, 2]
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - pts2d_observed.astype(np.float64), axis=1)
    return err.astype(np.float32)
