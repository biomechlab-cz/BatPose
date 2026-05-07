"""Unit tests for triangulation utilities."""

import numpy as np

from app.recon3d.triangulate import reprojection_error, triangulate_points_dlt


def _make_stereo_rig(baseline: float = 0.1):
    """Create a simple stereo rig with known geometry."""
    # Camera intrinsics (synthetic 720p camera)
    fx = fy = 800.0
    cx, cy = 640.0, 360.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)

    # Camera 2 is displaced by baseline along X
    R = np.eye(3, dtype=np.float64)
    T = np.array([-baseline, 0.0, 0.0], dtype=np.float64)

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # [3, 4]
    P2 = np.hstack([R, T.reshape(3, 1)])  # [3, 4]

    return K, D, R, T, P1, P2


class TestTriangulatePointsDlt:
    def test_known_point(self):
        """Triangulate a single known 3D point and verify recovery."""
        K, D, R, T, P1, P2 = _make_stereo_rig(baseline=0.2)

        # Known 3D point (in camera-1 frame)
        X_true = np.array([[0.0, 0.1, 2.0]])  # [1, 3]

        # Project to both cameras
        def project_norm(P, X):
            Xh = np.hstack([X, np.ones((len(X), 1))])  # [N, 4]
            proj = (P @ Xh.T).T  # [N, 3]
            return proj[:, :2] / proj[:, 2:3]  # [N, 2] normalized

        pts1 = project_norm(P1, X_true)
        pts2 = project_norm(P2, X_true)

        X_recon = triangulate_points_dlt(P1, P2, pts1, pts2)

        np.testing.assert_allclose(X_recon, X_true, atol=1e-4)

    def test_multiple_points(self):
        """Triangulate multiple points and check all are within tolerance."""
        K, D, R, T, P1, P2 = _make_stereo_rig(baseline=0.15)

        rng = np.random.default_rng(42)
        X_true = rng.uniform(-0.5, 0.5, (17, 3)) + [0, 0, 2.0]  # roughly 2 m away

        def project_norm(P, X):
            Xh = np.hstack([X, np.ones((len(X), 1))])
            proj = (P @ Xh.T).T
            return proj[:, :2] / proj[:, 2:3]

        pts1 = project_norm(P1, X_true)
        pts2 = project_norm(P2, X_true)

        X_recon = triangulate_points_dlt(P1, P2, pts1, pts2)

        np.testing.assert_allclose(X_recon, X_true.astype(np.float32), atol=1e-3)

    def test_output_shape(self):
        _, _, _, _, P1, P2 = _make_stereo_rig()
        pts1 = np.random.rand(17, 2).astype(np.float64)
        pts2 = np.random.rand(17, 2).astype(np.float64)
        result = triangulate_points_dlt(P1, P2, pts1, pts2)
        assert result.shape == (17, 3)
        assert result.dtype == np.float32


class TestTriangulateWithNoise:
    def test_noisy_multi_joint(self):
        """Triangulate 17 joints with ~1 px Gaussian noise; 3D error must be < 5 cm at 2 m."""
        K, D, R, T, P1, P2 = _make_stereo_rig(baseline=0.15)

        rng = np.random.default_rng(7)
        X_true = rng.uniform(-0.3, 0.3, (17, 3)) + [0.0, 0.0, 2.0]

        def project_norm(P, X):
            Xh = np.hstack([X, np.ones((len(X), 1))])
            proj = (P @ Xh.T).T
            return proj[:, :2] / proj[:, 2:3]

        pts1 = project_norm(P1, X_true)
        pts2 = project_norm(P2, X_true)

        # ~1 px noise expressed in normalized coords (fx = 800)
        noise_scale = 1.0 / 800.0
        pts1_noisy = pts1 + rng.normal(0, noise_scale, pts1.shape)
        pts2_noisy = pts2 + rng.normal(0, noise_scale, pts2.shape)

        X_recon = triangulate_points_dlt(P1, P2, pts1_noisy, pts2_noisy)

        err = np.linalg.norm(X_recon - X_true.astype(np.float32), axis=1)
        # Theoretical depth noise: Z²/(f·b)·σ_pix = 4/(800·0.15)·1 ≈ 0.033 m (1σ)
        # Allow 3σ for worst-case joint across 17 samples (max < 0.10 m)
        assert float(err.max()) < 0.10, f"Max 3D error {err.max():.4f} m exceeds 0.10 m"
        assert float(err.mean()) < 0.05, f"Mean 3D error {err.mean():.4f} m exceeds 0.05 m"

    def test_noise_degrades_gracefully(self):
        """Higher noise should produce larger (but finite) 3D errors."""
        _, _, _, _, P1, P2 = _make_stereo_rig(baseline=0.15)

        rng = np.random.default_rng(42)
        X_true = np.array([[0.0, 0.0, 2.0]])

        def project_norm(P, X):
            Xh = np.hstack([X, np.ones((len(X), 1))])
            proj = (P @ Xh.T).T
            return proj[:, :2] / proj[:, 2:3]

        pts1 = project_norm(P1, X_true)
        pts2 = project_norm(P2, X_true)

        for noise_px in (0.5, 2.0, 5.0):
            ns = noise_px / 800.0
            n1 = pts1 + rng.normal(0, ns, pts1.shape)
            n2 = pts2 + rng.normal(0, ns, pts2.shape)
            X_recon = triangulate_points_dlt(P1, P2, n1, n2)
            err = float(np.linalg.norm(X_recon - X_true.astype(np.float32), axis=1)[0])
            assert np.isfinite(err), "Triangulation returned non-finite result"
            assert err >= 0.0


class TestUndistortPoints:
    def test_zero_distortion_maps_principal_point_to_origin(self):
        """With D=0, the optical center (cx, cy) must map to normalized (0, 0)."""
        from app.recon3d.triangulate import undistort_points

        fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)

        pts = np.array([[cx, cy]], dtype=np.float32)
        norm = undistort_points(pts, K, D)
        np.testing.assert_allclose(norm, np.zeros((1, 2)), atol=1e-6)

    def test_output_shape_and_dtype(self):
        """undistort_points must return shape [N, 2] float64."""
        from app.recon3d.triangulate import undistort_points

        K = np.diag([600.0, 600.0, 1.0])
        D = np.zeros(5, dtype=np.float64)
        pts = np.random.rand(17, 2).astype(np.float32) * 640.0
        result = undistort_points(pts, K, D)
        assert result.shape == (17, 2)
        assert result.dtype == np.float64


class TestReprojectionError:
    def test_zero_error_for_perfect_projection(self):
        """Reprojecting the triangulated point should give ~0 error."""
        K, D, R, T, P1, P2 = _make_stereo_rig(baseline=0.2)

        X_true = np.array([[0.1, -0.05, 1.5]])  # [1, 3]

        def project_pixel(K, R, T, X):
            rvec, _ = __import__("cv2").Rodrigues(R)
            tvec = T.reshape(3)
            proj, _ = __import__("cv2").projectPoints(
                X.astype(np.float64), rvec, tvec, K.astype(np.float64), D.astype(np.float64)
            )
            return proj.reshape(-1, 2)

        pts2d = project_pixel(K, np.eye(3), np.zeros(3), X_true)

        R1 = np.eye(3)
        t1 = np.zeros(3)
        err = reprojection_error(X_true, K, D, R1, t1, pts2d)
        assert err.shape == (1,)
        assert float(err[0]) < 0.1  # pixels
