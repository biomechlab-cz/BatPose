"""Triangulation and reprojection utilities (ADR-004)."""

from __future__ import annotations

import cv2
import numpy as np


def undistort_points(
    pts: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    Undistort 2D pixel points and normalize by the camera matrix.

    Equivalent to: cv2.undistortPoints(pts, K, D, P=None)
    Returns points in normalized camera coordinates (not pixels).

    Args:
        pts: [N, 2] float32/float64 pixel coordinates
        K:   [3, 3] camera intrinsic matrix
        D:   distortion coefficients

    Returns:
        [N, 2] float64 normalized image coordinates
    """
    pts_in = pts.reshape(-1, 1, 2).astype(np.float64)
    pts_out = cv2.undistortPoints(pts_in, K.astype(np.float64), D.astype(np.float64), P=None)
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


def reprojection_error(
    pts3d: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    pts2d_observed: np.ndarray,
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

    Returns:
        err: [N] float32 per-point reprojection error (pixels)
    """
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)

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
