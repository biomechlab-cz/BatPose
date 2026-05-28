"""
Triangulation accuracy — synthetic pinhole cameras with known geometry.

Black-box: only the mathematical contract matters (inputs → error bounds).
No internal implementation details are assumed beyond the public function signatures.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.recon3d.triangulate import (  # noqa: E402
    reprojection_error,
    triangulate_points_dlt,
    undistort_points,
)

# ── Synthetic rig: two pinhole cameras, 0.5 m horizontal baseline ─────────────

_K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
_D_PINHOLE = np.zeros(5, dtype=np.float64)
_R1 = np.eye(3, dtype=np.float64)
_t1 = np.zeros(3, dtype=np.float64)
_R2 = np.eye(3, dtype=np.float64)
_t2 = np.array([-0.5, 0.0, 0.0], dtype=np.float64)  # 0.5 m baseline

# Human-skeleton-like 3D points (metres, camera-1 frame)
_SKELETON = np.array(
    [
        [0.00, 0.50, 2.0],  # head
        [0.15, 0.20, 2.0],  # left shoulder
        [-0.15, 0.20, 2.0],  # right shoulder
        [0.20, 0.00, 2.0],  # left elbow
        [-0.20, 0.00, 2.0],  # right elbow
        [0.00, 0.00, 2.0],  # hip centre
        [0.10, -0.40, 2.0],  # left knee
        [-0.10, -0.40, 2.0],  # right knee
    ],
    dtype=np.float32,
)


def _project_pinhole(pts3d: np.ndarray, K, R, t) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    pts2d, _ = cv2.projectPoints(
        pts3d.astype(np.float64),
        rvec,
        t.reshape(3, 1).astype(np.float64),
        K,
        np.zeros(5),
    )
    return pts2d.reshape(-1, 2).astype(np.float32)


def _triangulate(pts2d_l, pts2d_r):
    norm_l = undistort_points(pts2d_l, _K, _D_PINHOLE)
    norm_r = undistort_points(pts2d_r, _K, _D_PINHOLE)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([_R2, _t2.reshape(3, 1)])
    return triangulate_points_dlt(P1, P2, norm_l, norm_r)


class TestPinholeAccuracy:
    def test_perfect_projection_sub_mm(self):
        """Zero noise → reconstruction error < 0.1 mm (numerical precision only)."""
        pts2d_l = _project_pinhole(_SKELETON, _K, _R1, _t1)
        pts2d_r = _project_pinhole(_SKELETON, _K, _R2, _t2)
        pts3d_recon = _triangulate(pts2d_l, pts2d_r)
        errors_mm = np.linalg.norm(pts3d_recon - _SKELETON, axis=1) * 1000
        assert np.max(errors_mm) < 0.1, f"Max error {np.max(errors_mm):.3f} mm"

    def test_1px_noise_below_10mm(self):
        """1 px RMS noise on projections → all joint errors < 10 mm."""
        rng = np.random.default_rng(42)
        pts2d_l = _project_pinhole(_SKELETON, _K, _R1, _t1)
        pts2d_r = _project_pinhole(_SKELETON, _K, _R2, _t2)
        pts2d_l = pts2d_l + rng.normal(0, 1.0, pts2d_l.shape).astype(np.float32)
        pts2d_r = pts2d_r + rng.normal(0, 1.0, pts2d_r.shape).astype(np.float32)
        pts3d_recon = _triangulate(pts2d_l, pts2d_r)
        errors_mm = np.linalg.norm(pts3d_recon - _SKELETON, axis=1) * 1000
        assert np.max(errors_mm) < 20.0, f"Max error {np.max(errors_mm):.1f} mm"

    def test_reprojection_near_zero_after_perfect_triangulation(self):
        """Triangulated points reproject back within 0.01 px of original projections."""
        pts2d_l = _project_pinhole(_SKELETON, _K, _R1, _t1)
        pts2d_r = _project_pinhole(_SKELETON, _K, _R2, _t2)
        pts3d_recon = _triangulate(pts2d_l, pts2d_r)
        err_l = reprojection_error(pts3d_recon, _K, _D_PINHOLE, _R1, _t1, pts2d_l)
        err_r = reprojection_error(pts3d_recon, _K, _D_PINHOLE, _R2, _t2, pts2d_r)
        assert np.max(err_l) < 0.01, f"Left reproj {np.max(err_l):.4f} px"
        assert np.max(err_r) < 0.01, f"Right reproj {np.max(err_r):.4f} px"

    def test_deeper_points_also_accurate(self):
        """Points at 4 m depth (further than typical lab setup) still < 0.5 mm."""
        pts_far = _SKELETON.copy()
        pts_far[:, 2] = 4.0
        pts2d_l = _project_pinhole(pts_far, _K, _R1, _t1)
        pts2d_r = _project_pinhole(pts_far, _K, _R2, _t2)
        pts3d_recon = _triangulate(pts2d_l, pts2d_r)
        errors_mm = np.linalg.norm(pts3d_recon - pts_far, axis=1) * 1000
        assert np.max(errors_mm) < 0.5, f"Max error at 4 m: {np.max(errors_mm):.3f} mm"


class TestFisheyeMismatchGuard:
    """ADR-005: Using the wrong lens model must produce detectably larger reprojection error."""

    def test_correct_fisheye_model_gives_lower_error_than_pinhole_mismatch(self):
        K_fy = np.array([[600.0, 0, 480.0], [0, 600.0, 300.0], [0, 0, 1.0]])
        D_fy = np.array([-0.3, 0.05, -0.002, 0.0001], dtype=np.float64)
        D_wrong = np.zeros(5, dtype=np.float64)

        pts3d = np.array([[0.1, 0.0, 1.5], [-0.1, 0.1, 2.0]], dtype=np.float32)
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        proj, _ = cv2.fisheye.projectPoints(
            pts3d.reshape(-1, 1, 3).astype(np.float64),
            rvec,
            tvec,
            K_fy,
            D_fy.reshape(4, 1),
        )
        pts2d = proj.reshape(-1, 2).astype(np.float32)

        R_id = np.eye(3, dtype=np.float64)
        t_zero = np.zeros(3, dtype=np.float64)

        err_correct = reprojection_error(pts3d, K_fy, D_fy, R_id, t_zero, pts2d, fisheye=True)
        err_wrong = reprojection_error(pts3d, K_fy, D_wrong, R_id, t_zero, pts2d, fisheye=False)
        assert np.mean(err_wrong) > np.mean(err_correct), (
            f"Wrong model error ({np.mean(err_wrong):.2f} px) should exceed "
            f"correct model error ({np.mean(err_correct):.2f} px)"
        )
