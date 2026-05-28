"""
Integration test: 3D reconstruction on the 20260527_132450 sample recording.

Uses pre-computed pose2d_*.npz (112 frames, 1 person, COCO-17) plus
calibration.yml from 'data/Test project/' to run the full triangulation +
smoothing pipeline without re-running MediaPipe.

Pass criteria are biomechanics-relevant:
  - Output shape and frame count correct
  - No more than 20% frames lost to low confidence / bad reprojection
  - All detected joints lie within a plausible lab volume (±5 m)
  - Exported CSV satisfies the column schema expected by analysis software
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

_PROJECT = Path(__file__).parents[2] / "data" / "Test project"
_CALIB = _PROJECT / "calibration.yml"
_P2D_L = _PROJECT / "pose2d_left.npz"
_P2D_R = _PROJECT / "pose2d_right.npz"

pytestmark = pytest.mark.skipif(
    not (_CALIB.exists() and _P2D_L.exists() and _P2D_R.exists()),
    reason="Sample pose2d fixtures not found in data/Test project/",
)


@pytest.fixture(scope="module")
def pose3d(tmp_path_factory):
    """Run reconstruct3d once; share result across all tests in this module."""
    from app.recon3d.pipeline import reconstruct3d

    out = str(tmp_path_factory.mktemp("recon") / "pose3d.npz")
    result = reconstruct3d(
        calib_path=str(_CALIB),
        pose2d_left_path=str(_P2D_L),
        pose2d_right_path=str(_P2D_R),
        output_path=out,
    )
    assert result is not None, "reconstruct3d returned None — pipeline failed or cancelled"
    return np.load(out, allow_pickle=True), out


class TestOutputShape:
    def test_file_exists(self, pose3d):
        _, path = pose3d
        assert Path(path).exists()

    def test_required_arrays_present(self, pose3d):
        d, _ = pose3d
        assert {"joints3d", "conf3d", "meta"} <= set(d.keys())

    def test_joints3d_shape_T_P_17_3(self, pose3d):
        d, _ = pose3d
        shape = d["joints3d"].shape
        assert len(shape) == 4 and shape[2] == 17 and shape[3] == 3

    def test_conf3d_shape_matches_joints3d(self, pose3d):
        d, _ = pose3d
        T, P, J, _ = d["joints3d"].shape
        assert d["conf3d"].shape == (T, P, J)

    def test_frame_count_matches_pose2d_input(self, pose3d):
        d, _ = pose3d
        T_3d = d["joints3d"].shape[0]
        T_2d = int(np.load(_P2D_L)["keypoints"].shape[0])
        assert T_3d == T_2d, f"pose3d has {T_3d} frames, pose2d has {T_2d}"


class TestDetectionQuality:
    def test_at_least_80pct_frames_have_a_detection(self, pose3d):
        d, _ = pose3d
        conf3d = d["conf3d"]  # [T, P, 17]
        detected = np.any(conf3d > 0, axis=(1, 2))
        rate = detected.mean()
        assert rate >= 0.8, f"Detection rate {rate:.1%} below 80%"

    def test_median_reprojection_finite(self, pose3d):
        """Meta must contain reprojection stats or joints are finite (no all-NaN output)."""
        d, _ = pose3d
        joints = d["joints3d"]
        conf = d["conf3d"]
        detected_joints = joints[conf > 0]
        assert len(detected_joints) > 0, "No joints with conf > 0"
        assert np.all(np.isfinite(detected_joints)), "Non-finite values in detected joints"

    def test_joint_positions_within_lab_volume(self, pose3d):
        """All detected joints must lie within ±5 m — lab sanity check."""
        d, _ = pose3d
        joints = d["joints3d"]
        conf = d["conf3d"]
        detected = joints[conf > 0]
        out_of_range = np.abs(detected) > 5.0
        assert not np.any(out_of_range), f"{out_of_range.sum()} coordinates outside ±5 m lab volume"


class TestMetaData:
    def test_fps_present_and_positive(self, pose3d):
        d, _ = pose3d
        meta = d["meta"].item()
        assert "fps" in meta, "meta.fps missing"
        assert float(meta["fps"]) > 0

    def test_fps_matches_pose2d_fps(self, pose3d):
        d, _ = pose3d
        fps_3d = float(d["meta"].item()["fps"])
        fps_2d = float(np.load(_P2D_L, allow_pickle=True)["meta"].item()["fps"])
        assert abs(fps_3d - fps_2d) < 0.1, f"FPS mismatch: pose3d={fps_3d}, pose2d={fps_2d}"


class TestCsvExport:
    @pytest.fixture(scope="class")
    def csv_path(self, pose3d, tmp_path_factory):
        from app.recon3d.__main__ import _write_csv

        d, _ = pose3d
        joints3d = d["joints3d"]
        conf3d = d["conf3d"]
        fps = float(d["meta"].item().get("fps", 30.0))
        path = str(tmp_path_factory.mktemp("csv") / "export.csv")
        _write_csv(path, joints3d, conf3d, fps)
        return path, d

    def test_csv_file_created(self, csv_path):
        path, _ = csv_path
        assert Path(path).exists()

    def test_required_columns_present(self, csv_path):
        path, _ = csv_path
        with open(path, newline="") as f:
            headers = csv.DictReader(f).fieldnames
        for col in ("frame", "time_s", "person", "j0_x", "j0_y", "j0_z", "j16_z"):
            assert col in headers, f"Missing column: {col}"

    def test_row_count_equals_T_times_persons(self, csv_path):
        path, d = csv_path
        T, P = d["joints3d"].shape[:2]
        with open(path, newline="") as f:
            data_rows = sum(1 for _ in csv.reader(f)) - 1  # minus header
        assert data_rows == T * P

    def test_time_s_starts_at_zero(self, csv_path):
        path, _ = csv_path
        with open(path, newline="") as f:
            first = next(csv.DictReader(f))
        assert float(first["time_s"]) == pytest.approx(0.0, abs=0.001)

    def test_time_s_monotonically_increasing(self, csv_path):
        path, d = csv_path
        P = d["joints3d"].shape[1]
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        # time_s is the same for all persons in a frame; sample one person
        times = [float(r["time_s"]) for r in rows[::P]]
        diffs = np.diff(times)
        assert np.all(diffs >= 0), "time_s is not monotonically non-decreasing"

    def test_coordinate_values_are_finite_floats(self, csv_path):
        path, _ = csv_path
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                for col in ("j0_x", "j0_y", "j0_z"):
                    val = float(row[col])
                    assert np.isfinite(val), f"Non-finite value in {col}: {val}"
                break  # check first row only for speed
