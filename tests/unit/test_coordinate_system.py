"""Unit tests for the floor-board world coordinate system (app/calib/coordinate_system)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.calib.coordinate_system import (
    apply_world_frame,
    compute_world_frame,
    world_frame_from_corners,
)

# Board: 8x6 squares, 4 cm each → 0.32 m (X, long) x 0.24 m (Y, short).
SX, SY, SS = 8, 6, 0.04
LX, LY = SX * SS, SY * SS


def _board_obj_pts() -> np.ndarray:
    """Inner-corner grid in board frame (Z=0), cv2 ChArUco convention."""
    pts = []
    for j in range(1, SY):
        for i in range(1, SX):
            pts.append((i * SS, j * SS, 0.0))
    return np.asarray(pts, dtype=np.float64)


def _board_to_cam1():
    """A realistic board-on-floor pose: board X→camX, board Y→cam +Z (receding),
    board normal→cam -Y (up, since camera Y points down).  Camera sits above."""
    R_bc = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # cols ex,ey,ez
    # ex=[1,0,0], ey=[0,0,1], ez=[0,-1,0]; det=+1
    t_bc = np.array([-0.16, 0.5, 1.5], dtype=np.float64)
    return R_bc, t_bc


@pytest.fixture
def synthetic():
    obj = _board_obj_pts()
    R_bc, t_bc = _board_to_cam1()
    pts_cam1 = obj @ R_bc.T + t_bc
    wf = world_frame_from_corners(obj, pts_cam1, SX, SY, SS)
    return obj, pts_cam1, wf


class TestWorldFrameGeometry:
    def test_side_lengths(self, synthetic):
        _, _, wf = synthetic
        assert wf["board_long_m"] == pytest.approx(LX, abs=1e-9)
        assert wf["board_short_m"] == pytest.approx(LY, abs=1e-9)

    def test_exact_fit_rms_zero(self, synthetic):
        _, _, wf = synthetic
        assert wf["fit_rms_m"] == pytest.approx(0.0, abs=1e-6)

    def test_origin_maps_to_world_origin(self, synthetic):
        _, _, wf = synthetic
        origin_cam1 = np.asarray(wf["origin"])
        w = apply_world_frame(origin_cam1, wf)
        assert np.allclose(w, [0, 0, 0], atol=1e-5)

    def test_basis_is_right_handed(self, synthetic):
        _, _, wf = synthetic
        B = np.column_stack([wf["x_axis"], wf["y_axis"], wf["z_axis"]])
        assert np.linalg.det(B) == pytest.approx(1.0, abs=1e-6)

    def test_axes_orthonormal(self, synthetic):
        _, _, wf = synthetic
        B = np.column_stack([wf["x_axis"], wf["y_axis"], wf["z_axis"]])
        assert np.allclose(B.T @ B, np.eye(3), atol=1e-6)

    def test_z_points_up_toward_camera(self, synthetic):
        """Z must point from the floor toward the camera (z · origin < 0)."""
        _, _, wf = synthetic
        z = np.asarray(wf["z_axis"])
        origin = np.asarray(wf["origin"])
        assert float(z @ origin) < 0.0

    def test_all_corners_lie_in_world_floor_plane(self, synthetic):
        """Every board corner maps to world Z ≈ 0 (the board IS the floor)."""
        _, pts_cam1, wf = synthetic
        w = apply_world_frame(pts_cam1, wf)
        assert np.max(np.abs(w[:, 2])) < 1e-4

    def test_long_axis_spans_x_short_axis_spans_y(self, synthetic):
        """In world coords the corner spread along X ≈ long side, along Y ≈ short."""
        _, pts_cam1, wf = synthetic
        w = apply_world_frame(pts_cam1, wf)
        span_x = w[:, 0].max() - w[:, 0].min()
        span_y = w[:, 1].max() - w[:, 1].min()
        assert span_x > span_y                      # X is the longer side
        # inner corners span (SX-2)*SS in X, (SY-2)*SS in Y
        assert span_x == pytest.approx((SX - 2) * SS, abs=1e-4)
        assert span_y == pytest.approx((SY - 2) * SS, abs=1e-4)

    def test_too_few_corners_raises(self):
        obj = _board_obj_pts()[:4]
        with pytest.raises(ValueError):
            world_frame_from_corners(obj, obj.copy(), SX, SY, SS)


class TestApplyWorldFrame:
    def test_preserves_shape_4d(self, synthetic):
        _, _, wf = synthetic
        pts = np.random.RandomState(0).standard_normal((5, 2, 17, 3)).astype(np.float32)
        out = apply_world_frame(pts, wf)
        assert out.shape == pts.shape

    def test_rigid_preserves_distances(self, synthetic):
        _, _, wf = synthetic
        rng = np.random.RandomState(1)
        a = rng.standard_normal((10, 3)).astype(np.float32)
        b = rng.standard_normal((10, 3)).astype(np.float32)
        da = np.linalg.norm(a - b, axis=1)
        wa, wb = apply_world_frame(a, wf), apply_world_frame(b, wf)
        dw = np.linalg.norm(wa - wb, axis=1)
        assert np.allclose(da, dw, atol=1e-4)


class TestComputeWorldFrameStereo:
    """End-to-end: project corners into a synthetic stereo pair, then recover."""

    def _project(self, pts_cam, K):
        z = pts_cam[:, 2:3]
        xy = pts_cam[:, :2] / z
        px = xy[:, 0] * K[0, 0] + K[0, 2]
        py = xy[:, 1] * K[1, 1] + K[1, 2]
        return np.column_stack([px, py]).astype(np.float32)

    def test_recovers_same_frame_as_direct(self):
        from app.calib.board import DetectionResult

        obj = _board_obj_pts()
        R_bc, t_bc = _board_to_cam1()
        pts_cam1 = obj @ R_bc.T + t_bc

        K = np.array([[800.0, 0, 512.0], [0, 800.0, 512.0], [0, 0, 1.0]])
        R = np.eye(3)
        T = np.array([-0.2, 0.0, 0.0])           # 20 cm stereo baseline
        pts_cam2 = pts_cam1 @ R.T + T

        det_l = DetectionResult(
            obj_pts=obj.astype(np.float32),
            img_pts=self._project(pts_cam1, K),
            ids=np.arange(len(obj), dtype=np.int32),
        )
        det_r = DetectionResult(
            obj_pts=obj.astype(np.float32),
            img_pts=self._project(pts_cam2, K),
            ids=np.arange(len(obj), dtype=np.int32),
        )
        calib = {
            "K1": K, "D1": np.zeros(5), "K2": K, "D2": np.zeros(5),
            "R": R, "T": T, "lens_model": "standard",
            "board_cfg": {"squares_x": SX, "squares_y": SY, "square_size": SS},
        }
        wf = compute_world_frame(det_l, det_r, calib)
        ref = world_frame_from_corners(obj, pts_cam1, SX, SY, SS)
        assert np.allclose(wf["R"], ref["R"], atol=1e-3)
        assert np.allclose(wf["t"], ref["t"], atol=1e-3)
        assert wf["fit_rms_m"] < 1e-3


def _minimal_calib_data() -> dict:
    return {
        "image_size": [640, 480],
        "lens_model": "standard",
        "K1": np.eye(3), "D1": np.zeros(5),
        "K2": np.eye(3), "D2": np.zeros(5),
        "R": np.eye(3), "T": np.array([-0.2, 0.0, 0.0]),
        "E": np.zeros((3, 3)), "F": np.zeros((3, 3)),
        "rms": 0.4, "n_frames": 12,
    }


class TestPersistence:
    def test_world_frame_round_trips(self, tmp_path, synthetic):
        from app.calib.stereo import load_calibration, save_calibration

        _, _, wf = synthetic
        data = _minimal_calib_data()
        data["world_frame"] = wf
        path = str(tmp_path / "calibration.yml")
        save_calibration(data, {"type": "charuco", "squares_x": SX, "squares_y": SY}, path)

        loaded = load_calibration(path)
        assert loaded["world_frame"] is not None
        assert np.allclose(loaded["world_frame"]["R"], wf["R"])
        assert np.allclose(loaded["world_frame"]["t"], wf["t"])

    def test_absent_world_frame_loads_as_none(self, tmp_path):
        from app.calib.stereo import load_calibration, save_calibration

        path = str(tmp_path / "calibration.yml")
        save_calibration(_minimal_calib_data(), {"type": "charuco"}, path)
        assert load_calibration(path)["world_frame"] is None

    def test_update_world_frame_patches_existing(self, tmp_path, synthetic):
        from app.calib.stereo import load_calibration, save_calibration, update_world_frame

        _, _, wf = synthetic
        path = str(tmp_path / "calibration.yml")
        save_calibration(_minimal_calib_data(), {"type": "charuco"}, path)
        assert load_calibration(path)["world_frame"] is None

        update_world_frame(path, wf)
        loaded = load_calibration(path)
        assert loaded["world_frame"] is not None
        # Other fields survive the patch.
        assert np.allclose(loaded["K1"], np.eye(3))
        assert loaded["lens_model"] == "standard"

        update_world_frame(path, None)  # removal
        assert load_calibration(path)["world_frame"] is None


_PROJECT = Path(__file__).parents[2] / "data" / "Test project" / "test_fixture"
_HAS_FIXTURE = (
    (_PROJECT / "calibration.yml").exists()
    and (_PROJECT / "pose2d_left.npz").exists()
    and (_PROJECT / "pose2d_right.npz").exists()
)


@pytest.mark.skipif(not _HAS_FIXTURE, reason="Test fixture not available")
class TestPipelineAppliesWorldFrame:
    def test_pipeline_transforms_and_flags(self, tmp_path):
        """reconstruct3d applies a world frame and tags meta coordinate_frame."""
        import shutil

        from app.recon3d.pipeline import reconstruct3d
        from app.calib.stereo import update_world_frame

        # Copy the fixture calib and inject a pure-translation world frame
        # (R=I, t=[1,0,0]) so the effect is trivially checkable.
        calib_path = str(tmp_path / "calibration.yml")
        shutil.copy(_PROJECT / "calibration.yml", calib_path)
        wf = {"R": np.eye(3).tolist(), "t": [1.0, 0.0, 0.0],
              "origin": [0, 0, 0], "x_axis": [1, 0, 0], "y_axis": [0, 1, 0],
              "z_axis": [0, 0, 1], "board_long_m": 0.32, "board_short_m": 0.24,
              "n_corners": 20, "fit_rms_m": 0.0}
        update_world_frame(calib_path, wf)

        out = str(tmp_path / "pose3d.npz")
        reconstruct3d(
            calib_path=calib_path,
            pose2d_left_path=str(_PROJECT / "pose2d_left.npz"),
            pose2d_right_path=str(_PROJECT / "pose2d_right.npz"),
            output_path=out,
        )
        d = np.load(out, allow_pickle=True)
        meta = d["meta"].item()
        assert meta["coordinate_frame"] == "world"
        assert meta["world_frame"] is not None

        # Compare to the un-transformed (opencv) run: detected joints shift by +1 in X.
        calib_plain = str(tmp_path / "calib_plain.yml")
        shutil.copy(_PROJECT / "calibration.yml", calib_plain)
        out_plain = str(tmp_path / "pose3d_plain.npz")
        reconstruct3d(
            calib_path=calib_plain,
            pose2d_left_path=str(_PROJECT / "pose2d_left.npz"),
            pose2d_right_path=str(_PROJECT / "pose2d_right.npz"),
            output_path=out_plain,
        )
        dp = np.load(out_plain, allow_pickle=True)
        assert dp["meta"].item()["coordinate_frame"] == "opencv"

        c = d["conf3d"] > 0
        jw, jp = d["joints3d"], dp["joints3d"]
        # X shifted by +1, Y/Z unchanged for detected joints.
        assert np.allclose(jw[c][:, 0] - jp[c][:, 0], 1.0, atol=1e-3)
        assert np.allclose(jw[c][:, 1], jp[c][:, 1], atol=1e-3)
        assert np.allclose(jw[c][:, 2], jp[c][:, 2], atol=1e-3)
