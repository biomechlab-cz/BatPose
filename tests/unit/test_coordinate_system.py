"""Unit tests for the floor-board world coordinate system (app/calib/coordinate_system)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.calib.coordinate_system import (
    apply_world_frame,
    board_squares,
    centered_board_objpoints,
    compute_world_frame,
    compute_world_frame_monocular,
    detect_chessboard_corners,
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
        assert span_x > span_y  # X is the longer side
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
        T = np.array([-0.2, 0.0, 0.0])  # 20 cm stereo baseline
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
            "K1": K,
            "D1": np.zeros(5),
            "K2": K,
            "D2": np.zeros(5),
            "R": R,
            "T": T,
            "lens_model": "standard",
            "board_cfg": {"squares_x": SX, "squares_y": SY, "square_size": SS},
        }
        wf = compute_world_frame(det_l, det_r, calib)
        ref = world_frame_from_corners(obj, pts_cam1, SX, SY, SS)
        assert np.allclose(wf["R"], ref["R"], atol=1e-3)
        assert np.allclose(wf["t"], ref["t"], atol=1e-3)
        assert wf["fit_rms_m"] < 1e-3


class TestBoardSquares:
    """board_squares() maps both persisted board-config conventions."""

    def test_charuco_cfg_passthrough(self):
        cfg = {"type": "charuco", "squares_x": 5, "squares_y": 7, "square_size": 0.04}
        assert board_squares(cfg) == (5, 7, 0.04)

    def test_checkerboard_corners_map_to_squares(self):
        # cols/rows are INNER-CORNER counts → squares are one more each.
        cfg = {"type": "checkerboard", "cols": 9, "rows": 6, "square_size": 0.025}
        assert board_squares(cfg) == (10, 7, 0.025)

    def test_unusable_cfgs_return_zeros(self):
        assert board_squares(None) == (0, 0, 0.0)
        assert board_squares({}) == (0, 0, 0.0)
        assert board_squares({"type": "charuco"}) == (0, 0, 0.0)
        # Counts without a square size are useless for geometry.
        assert board_squares({"cols": 9, "rows": 6}) == (0, 0, 0.0)
        assert board_squares({"cols": 1, "rows": 6, "square_size": 0.025}) == (0, 0, 0.0)


class TestBoardCfgConventions:
    """compute_world_frame() board geometry for checkerboard / missing configs.

    Checkerboards persist cols/rows (inner corners, object points from (0,0));
    reading them through the charuco keys used to drop into a fallback that
    mixed metres with square counts (a 9×6 / 25 mm board became ~1.2 m wide).
    """

    K = np.array([[800.0, 0, 512.0], [0, 800.0, 512.0], [0, 0, 1.0]])

    def _project(self, pts_cam):
        z = pts_cam[:, 2:3]
        xy = pts_cam[:, :2] / z
        return np.column_stack(
            [xy[:, 0] * self.K[0, 0] + self.K[0, 2], xy[:, 1] * self.K[1, 1] + self.K[1, 2]]
        ).astype(np.float32)

    def _stereo_frame(self, obj, board_cfg):
        from app.calib.board import DetectionResult

        R_bc, t_bc = _board_to_cam1()
        pts_cam1 = obj @ R_bc.T + t_bc
        R = np.eye(3)
        T = np.array([-0.2, 0.0, 0.0])
        pts_cam2 = pts_cam1 @ R.T + T
        det_l = DetectionResult(
            obj_pts=obj.astype(np.float32), img_pts=self._project(pts_cam1), ids=None
        )
        det_r = DetectionResult(
            obj_pts=obj.astype(np.float32), img_pts=self._project(pts_cam2), ids=None
        )
        calib = {
            "K1": self.K,
            "D1": np.zeros(5),
            "K2": self.K,
            "D2": np.zeros(5),
            "R": R,
            "T": T,
            "lens_model": "standard",
            "board_cfg": board_cfg,
        }
        return compute_world_frame(det_l, det_r, calib), pts_cam1

    def test_checkerboard_cfg_size_and_centre(self):
        cols, rows, ss = 9, 6, 0.025
        # Checkerboard object-point convention: first inner corner at (0, 0).
        obj = np.array(
            [(i * ss, j * ss, 0.0) for j in range(rows) for i in range(cols)], dtype=np.float64
        )
        wf, pts_cam1 = self._stereo_frame(
            obj, {"type": "checkerboard", "cols": cols, "rows": rows, "square_size": ss}
        )
        assert wf["method"] == "checkerboard_stereo"
        # Physical board adds one square per side beyond the inner corners.
        assert wf["board_long_m"] == pytest.approx((cols + 1) * ss, abs=1e-9)  # 0.25 m
        assert wf["board_short_m"] == pytest.approx((rows + 1) * ss, abs=1e-9)  # 0.175 m
        # Board centre = inner-corner-grid centroid → centroid of the 3D points.
        assert np.allclose(wf["origin"], pts_cam1.mean(axis=0), atol=1e-3)
        assert np.allclose(apply_world_frame(np.asarray(wf["origin"]), wf), 0, atol=1e-4)

    def test_missing_cfg_fallback_stays_metric(self):
        obj = _board_obj_pts()  # charuco-style 8×6 grid, no usable cfg given
        wf, pts_cam1 = self._stereo_frame(obj, {})
        assert wf["method"] == "corner_span_stereo"
        # Side lengths = detected corner spans in METRES (the old fallback
        # produced span+1 "squares" of 1 m each → ~1.2 m phantom boards).
        assert wf["board_long_m"] == pytest.approx((SX - 2) * SS, abs=1e-6)
        assert wf["board_short_m"] == pytest.approx((SY - 2) * SS, abs=1e-6)
        assert wf["board_long_m"] < 1.0
        assert np.allclose(wf["origin"], pts_cam1.mean(axis=0), atol=1e-3)


class TestMonocularChessboardFrame:
    """Chessboard + monocular solvePnP path (for a far floor board where ArUco
    markers can't be decoded)."""

    K = np.array([[800.0, 0, 512.0], [0, 800.0, 512.0], [0, 0, 1.0]])

    def _project_pinhole(self, pts_cam):
        z = pts_cam[:, 2:3]
        xy = pts_cam[:, :2] / z
        return np.column_stack(
            [xy[:, 0] * self.K[0, 0] + self.K[0, 2], xy[:, 1] * self.K[1, 1] + self.K[1, 2]]
        ).astype(np.float64)

    def test_centered_objpoints_centroid_is_origin(self):
        objp = centered_board_objpoints(SX, SY, SS)
        assert objp.shape == ((SX - 1) * (SY - 1), 3)
        assert np.allclose(objp.mean(axis=0), [0, 0, 0], atol=1e-9)

    def test_recovers_known_board_pose(self):
        R_bc = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        t_bc = np.array([0.10, 0.30, 2.00])  # board centre, 2 m in front
        objp = centered_board_objpoints(SX, SY, SS)
        img = self._project_pinhole(objp @ R_bc.T + t_bc)

        wf = compute_world_frame_monocular(img, self.K, np.zeros(5), SX, SY, SS, fisheye=False)

        assert wf["method"] == "chessboard_pnp"
        assert wf["n_corners"] == (SX - 1) * (SY - 1)
        assert wf["fit_rms_px"] < 0.5  # synthetic → near-perfect
        # origin recovered at the board centre
        assert np.allclose(wf["origin"], t_bc, atol=1e-3)
        assert np.allclose(apply_world_frame(np.array(wf["origin"]), wf), [0, 0, 0], atol=1e-4)
        # right-handed, orthonormal, Z up toward the camera, correct side lengths
        B = np.column_stack([wf["x_axis"], wf["y_axis"], wf["z_axis"]])
        assert np.linalg.det(B) == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(B.T @ B, np.eye(3), atol=1e-6)
        assert float(np.array(wf["z_axis"]) @ np.array(wf["origin"])) < 0
        # This fixture board is 8×6 squares → long side = X (8), short = Y (6).
        assert wf["board_long_m"] == pytest.approx(LX, abs=1e-9)
        assert wf["board_short_m"] == pytest.approx(LY, abs=1e-9)

    def test_recovers_same_frame_from_right_camera(self):
        """When only the RIGHT camera sees the board, detecting on it and passing
        the stereo extrinsics as ``cam_to_ref`` must recover the SAME camera-1
        world frame as detecting on the left."""
        R_bc = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        t_bc = np.array([0.10, 0.30, 2.00])
        objp = centered_board_objpoints(SX, SY, SS)
        pts_cam1 = objp @ R_bc.T + t_bc

        # A non-trivial stereo pose (toe-in + baseline), p_cam2 = R p_cam1 + T.
        ang = np.deg2rad(15.0)
        R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
        T = np.array([-0.20, 0.0, 0.0])
        pts_cam2 = pts_cam1 @ R.T + T
        img_r = self._project_pinhole(pts_cam2)

        wf_ref = compute_world_frame_monocular(
            self._project_pinhole(pts_cam1), self.K, np.zeros(5), SX, SY, SS, fisheye=False
        )
        wf = compute_world_frame_monocular(
            img_r, self.K, np.zeros(5), SX, SY, SS, fisheye=False, cam_to_ref=(R, T)
        )

        assert wf["method"] == "chessboard_pnp_ref"
        assert np.allclose(wf["origin"], wf_ref["origin"], atol=1e-3)
        assert np.allclose(wf["R"], wf_ref["R"], atol=1e-3)
        assert np.allclose(wf["t"], wf_ref["t"], atol=1e-3)
        # Frame still lands at the board centre in cam-1 coords.
        assert np.allclose(wf["origin"], t_bc, atol=1e-3)

    def test_wrong_corner_count_raises(self):
        with pytest.raises(ValueError):
            compute_world_frame_monocular(
                np.zeros((5, 2)), self.K, np.zeros(5), SX, SY, SS, fisheye=False
            )


class TestPlanarBranchDisambiguation:
    """The two-fold planar-PnP ambiguity must be resolved by the floor-up prior.

    Which branch the ITERATIVE optimiser converges to is placement-dependent —
    at some board placements the WRONG branch wins the per-frame majority, so
    voting locked in a flipped frame (hardware: world Z 114° from camera-up →
    live skeleton lay flat).  compute_world_frame_monocular now evaluates BOTH
    IPPE branches and picks the up-pointing one, so the recovered normal must
    match the ground truth at ANY oblique placement.
    """

    K = np.array([[800.0, 0, 512.0], [0, 800.0, 512.0], [0, 0, 1.0]])

    @staticmethod
    def _rot_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def _rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _project(self, pts_cam):
        z = pts_cam[:, 2:3]
        xy = pts_cam[:, :2] / z
        return np.column_stack(
            [xy[:, 0] * self.K[0, 0] + self.K[0, 2], xy[:, 1] * self.K[1, 1] + self.K[1, 2]]
        ).astype(np.float64)

    def test_normal_correct_across_oblique_placements(self):
        """Sweep pitch/yaw obliquities — recovered Z must track the true normal."""
        base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # flat on floor
        objp = centered_board_objpoints(SX, SY, SS)
        for pitch_deg in (-30, -15, 0, 15, 30):
            for yaw_deg in (-40, -20, 0, 20, 40):
                R_cam = self._rot_y(np.radians(yaw_deg)) @ self._rot_x(np.radians(pitch_deg))
                R_bc = R_cam @ base
                if R_bc[1, 2] > -0.2:
                    continue  # normal no longer clearly 'up' — not a floor-board pose
                t_bc = R_cam @ np.array([0.1, 0.5, 2.2])
                img = self._project(objp @ R_bc.T + t_bc)
                wf = compute_world_frame_monocular(
                    img, self.K, np.zeros(5), SX, SY, SS, fisheye=False
                )
                true_z = -R_bc[:, 2] if R_bc[:, 2] @ t_bc > 0 else R_bc[:, 2]
                ang = np.degrees(np.arccos(np.clip(np.dot(wf["z_axis"], true_z), -1, 1)))
                assert ang < 2.0, (
                    f"normal off by {ang:.1f} deg at pitch={pitch_deg} yaw={yaw_deg} "
                    "(flipped PnP branch chosen?)"
                )

    def test_both_branches_exist_and_flipped_one_is_rejected(self):
        """At an oblique pose IPPE genuinely returns two branches; the function
        must return the up-pointing one."""
        import cv2

        base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        R_bc = self._rot_x(np.radians(25)) @ base
        t_bc = np.array([0.3, 0.6, 2.0])
        objp = centered_board_objpoints(SX, SY, SS)
        img = self._project(objp @ R_bc.T + t_bc)

        n_sol, rvecs, _tvecs, _err = cv2.solvePnPGeneric(
            objp, img.reshape(-1, 1, 2), self.K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE
        )
        assert n_sol == 2  # the ambiguity is real at this obliquity
        normals = [cv2.Rodrigues(rv)[0][:, 2] for rv in rvecs]
        spread = np.degrees(np.arccos(np.clip(abs(float(normals[0] @ normals[1])), -1, 1)))
        assert spread > 20  # and the branches are far apart

        wf = compute_world_frame_monocular(img, self.K, np.zeros(5), SX, SY, SS, fisheye=False)
        # The chosen Z points up (towards camera -Y), never into the floor.
        assert float(np.dot(wf["z_axis"], [0, -1, 0])) > 0.7

    def test_detect_chessboard_returns_none_on_blank(self):
        blank = np.zeros((480, 640), dtype=np.uint8)
        assert detect_chessboard_corners(blank, SX, SY) is None


def _minimal_calib_data() -> dict:
    return {
        "image_size": [640, 480],
        "lens_model": "standard",
        "K1": np.eye(3),
        "D1": np.zeros(5),
        "K2": np.eye(3),
        "D2": np.zeros(5),
        "R": np.eye(3),
        "T": np.array([-0.2, 0.0, 0.0]),
        "E": np.zeros((3, 3)),
        "F": np.zeros((3, 3)),
        "rms": 0.4,
        "n_frames": 12,
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

        from app.calib.stereo import update_world_frame
        from app.recon3d.pipeline import reconstruct3d

        # Copy the fixture calib and inject a pure-translation world frame
        # (R=I, t=[1,0,0]) so the effect is trivially checkable.
        calib_path = str(tmp_path / "calibration.yml")
        shutil.copy(_PROJECT / "calibration.yml", calib_path)
        wf = {
            "R": np.eye(3).tolist(),
            "t": [1.0, 0.0, 0.0],
            "origin": [0, 0, 0],
            "x_axis": [1, 0, 0],
            "y_axis": [0, 1, 0],
            "z_axis": [0, 0, 1],
            "board_long_m": 0.32,
            "board_short_m": 0.24,
            "n_corners": 20,
            "fit_rms_m": 0.0,
        }
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
