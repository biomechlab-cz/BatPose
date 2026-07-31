"""Unit tests for biomech joint-angle computation."""

import csv

import numpy as np
import pytest

from app.biomech import (
    ANGLE_DEFINITIONS,
    AngleStats,
    angles_to_csv,
    compute_extended_stats,
    compute_joint_angles,
    compute_stats,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_pose(positions: dict[int, tuple[float, float, float]]) -> np.ndarray:
    """Build a [1, 1, 17, 3] pose array with the given joints set; rest = 0."""
    pose = np.zeros((1, 1, 17, 3), dtype=np.float32)
    for idx, (x, y, z) in positions.items():
        pose[0, 0, idx] = (x, y, z)
    return pose


def _full_conf() -> np.ndarray:
    """Return an all-confident [1, 1, 17] confidence array."""
    return np.ones((1, 1, 17), dtype=np.float32)


# ----------------------------------------------------------------------
# 3-point angle tests — use the L Knee Flex angle (indices 11, 13, 15)
# ----------------------------------------------------------------------


class TestAngleComputation:
    def test_known_90_degree_angle(self):
        """Right angle at the vertex must produce 90°."""
        pose = _make_pose(
            {
                11: (1.0, 0.0, 0.0),  # A = LHip at +x
                13: (0.0, 0.0, 0.0),  # B = LKnee at origin (vertex)
                15: (0.0, 1.0, 0.0),  # C = LAnkle at +y
            }
        )
        out = compute_joint_angles(pose, _full_conf())
        assert out[0, 0, 0] == pytest.approx(90.0, abs=1e-3)

    def test_known_180_degree_angle(self):
        """Three collinear points (vertex in the middle) give 180°."""
        pose = _make_pose(
            {
                11: (-1.0, 0.0, 0.0),
                13: (0.0, 0.0, 0.0),
                15: (1.0, 0.0, 0.0),
            }
        )
        out = compute_joint_angles(pose, _full_conf())
        assert out[0, 0, 0] == pytest.approx(180.0, abs=1e-3)

    def test_known_60_degree_angle(self):
        """Equilateral-triangle vertex angle is 60°."""
        # A, B, C as 3 vertices of an equilateral triangle in the XY plane;
        # interior angle at any vertex is 60°.
        pose = _make_pose(
            {
                11: (1.0, 0.0, 0.0),
                13: (0.0, 0.0, 0.0),
                15: (0.5, np.sqrt(3) / 2, 0.0),
            }
        )
        out = compute_joint_angles(pose, _full_conf())
        assert out[0, 0, 0] == pytest.approx(60.0, abs=1e-3)

    def test_zero_conf_on_flanking_joint_yields_nan(self):
        """Conf=0 on the proximal joint must produce NaN."""
        pose = _make_pose(
            {
                11: (1.0, 0.0, 0.0),
                13: (0.0, 0.0, 0.0),
                15: (0.0, 1.0, 0.0),
            }
        )
        conf = _full_conf()
        conf[0, 0, 11] = 0.0  # zero out LHip
        out = compute_joint_angles(pose, conf)
        assert np.isnan(out[0, 0, 0])

    def test_zero_conf_on_vertex_yields_nan(self):
        """Conf=0 on the vertex joint must produce NaN."""
        pose = _make_pose(
            {
                11: (1.0, 0.0, 0.0),
                13: (0.0, 0.0, 0.0),
                15: (0.0, 1.0, 0.0),
            }
        )
        conf = _full_conf()
        conf[0, 0, 13] = 0.0  # zero out LKnee (the vertex)
        out = compute_joint_angles(pose, conf)
        assert np.isnan(out[0, 0, 0])

    def test_zero_conf_on_distal_yields_nan(self):
        """Conf=0 on the distal joint must produce NaN."""
        pose = _make_pose(
            {
                11: (1.0, 0.0, 0.0),
                13: (0.0, 0.0, 0.0),
                15: (0.0, 1.0, 0.0),
            }
        )
        conf = _full_conf()
        conf[0, 0, 15] = 0.0  # zero out LAnkle
        out = compute_joint_angles(pose, conf)
        assert np.isnan(out[0, 0, 0])

    def test_all_positive_conf_no_nan(self):
        """If all flanking joints have positive conf, output is finite."""
        pose = _make_pose(
            {
                11: (1.0, 0.0, 0.0),
                13: (0.0, 0.0, 0.0),
                15: (0.0, 1.0, 0.0),
            }
        )
        out = compute_joint_angles(pose, _full_conf())
        assert np.isfinite(out[0, 0, 0])


class TestConfidenceThreshold:
    """min_conf parameter filters keypoints below the threshold."""

    def _knee_pose(self):
        return _make_pose(
            {
                11: (1.0, 0.0, 0.0),
                13: (0.0, 0.0, 0.0),
                15: (0.0, 1.0, 0.0),
            }
        )

    def test_default_min_conf_keeps_low_confidence(self):
        """Default min_conf=0 accepts any non-zero confidence (90° still computed)."""
        conf = _full_conf()
        conf[0, 0, 13] = 0.1  # low but non-zero
        out = compute_joint_angles(self._knee_pose(), conf)
        assert out[0, 0, 0] == pytest.approx(90.0, abs=1e-3)

    def test_threshold_discards_below_min_conf(self):
        """A flanking joint below min_conf turns the angle into NaN."""
        conf = _full_conf()
        conf[0, 0, 13] = 0.2  # vertex below threshold
        out = compute_joint_angles(self._knee_pose(), conf, min_conf=0.5)
        assert np.isnan(out[0, 0, 0])

    def test_threshold_keeps_at_or_above_min_conf(self):
        """A flanking joint exactly at min_conf is kept (>= comparison)."""
        conf = _full_conf()
        conf[0, 0, 13] = 0.5
        out = compute_joint_angles(self._knee_pose(), conf, min_conf=0.5)
        assert out[0, 0, 0] == pytest.approx(90.0, abs=1e-3)

    def test_raising_threshold_increases_nan_count(self):
        """Monotonic: a higher threshold never produces fewer NaNs."""
        rng = np.random.default_rng(1)
        pose = rng.standard_normal((20, 1, 17, 3)).astype(np.float32)
        conf = rng.uniform(0.0, 1.0, (20, 1, 17)).astype(np.float32)
        nan_low = np.isnan(compute_joint_angles(pose, conf, min_conf=0.1)).sum()
        nan_high = np.isnan(compute_joint_angles(pose, conf, min_conf=0.8)).sum()
        assert nan_high >= nan_low


# ----------------------------------------------------------------------
# Shape / dtype contracts
# ----------------------------------------------------------------------


class TestOutputContract:
    def test_output_shape(self):
        """Output last axis equals len(ANGLE_DEFINITIONS)."""
        rng = np.random.default_rng(42)
        pose = rng.standard_normal((10, 2, 17, 3)).astype(np.float32)
        conf = np.ones((10, 2, 17), dtype=np.float32)
        out = compute_joint_angles(pose, conf)
        assert out.shape == (10, 2, len(ANGLE_DEFINITIONS))

    def test_output_dtype_float32(self):
        """Output dtype is float32 to keep memory low."""
        pose = np.zeros((3, 1, 17, 3), dtype=np.float32)
        conf = np.ones((3, 1, 17), dtype=np.float32)
        out = compute_joint_angles(pose, conf)
        assert out.dtype == np.float32

    def test_mismatched_conf_shape_raises(self):
        """Wrong-shape conf array raises ValueError."""
        pose = np.zeros((3, 1, 17, 3), dtype=np.float32)
        conf = np.ones((3, 1, 16), dtype=np.float32)  # wrong J
        with pytest.raises(ValueError):
            compute_joint_angles(pose, conf)

    def test_bad_pose_rank_raises(self):
        """Non-4D pose array raises ValueError."""
        pose = np.zeros((17, 3), dtype=np.float32)
        conf = np.ones((1, 1, 17), dtype=np.float32)
        with pytest.raises(ValueError):
            compute_joint_angles(pose, conf)


# ----------------------------------------------------------------------
# Synthetic standing person — sanity check the full angle set
# ----------------------------------------------------------------------


def _synthetic_standing_pose() -> np.ndarray:
    """A crude, anatomically plausible Z-up standing pose for sanity tests.

    Person faces +Y, head at z≈1.7m, hips at z≈0.9m, feet at z=0.
    All limbs straight (extended), so every flexion angle is ~180°.
    """
    positions = {
        0: (0.0, 0.05, 1.70),  # nose
        1: (-0.03, 0.05, 1.72),  # L eye
        2: (0.03, 0.05, 1.72),  # R eye
        3: (-0.08, 0.0, 1.70),  # L ear
        4: (0.08, 0.0, 1.70),  # R ear
        5: (-0.18, 0.0, 1.45),  # L shoulder
        6: (0.18, 0.0, 1.45),  # R shoulder
        7: (-0.18, 0.0, 1.15),  # L elbow
        8: (0.18, 0.0, 1.15),  # R elbow
        9: (-0.18, 0.0, 0.85),  # L wrist
        10: (0.18, 0.0, 0.85),  # R wrist
        11: (-0.10, 0.0, 0.90),  # L hip
        12: (0.10, 0.0, 0.90),  # R hip
        13: (-0.10, 0.0, 0.50),  # L knee
        14: (0.10, 0.0, 0.50),  # R knee
        15: (-0.10, 0.0, 0.0),  # L ankle
        16: (0.10, 0.0, 0.0),  # R ankle
    }
    return _make_pose(positions)


class TestSyntheticPose:
    def test_all_angles_in_valid_range(self):
        """For a plausible standing pose every non-NaN angle is in [0, 180]."""
        pose = _synthetic_standing_pose()
        out = compute_joint_angles(pose, _full_conf())[0, 0]
        finite = out[~np.isnan(out)]
        assert np.all(finite >= 0.0)
        assert np.all(finite <= 180.0 + 1e-3)

    def test_straight_limbs_near_180(self):
        """Straight (extended) knees and elbows should read ~180°."""
        pose = _synthetic_standing_pose()
        out = compute_joint_angles(pose, _full_conf())[0, 0]
        # L Knee Flex (index 0) and R Knee Flex (index 1) → straight limbs.
        assert out[0] == pytest.approx(180.0, abs=1.0)
        assert out[1] == pytest.approx(180.0, abs=1.0)
        # L Elbow (4) and R Elbow (5) — also straight in this pose.
        assert out[4] == pytest.approx(180.0, abs=1.0)
        assert out[5] == pytest.approx(180.0, abs=1.0)


# ----------------------------------------------------------------------
# Trunk inclination
# ----------------------------------------------------------------------


def _trunk_index() -> int:
    return next(i for i, a in enumerate(ANGLE_DEFINITIONS) if a.name == "Trunk Inclination")


class TestTrunkInclination:
    def test_upright_trunk_is_zero(self):
        """A perfectly vertical trunk gives ~0°."""
        pose = _make_pose(
            {
                5: (-0.18, 0.0, 1.5),  # L shoulder above L hip
                6: (0.18, 0.0, 1.5),
                11: (-0.10, 0.0, 0.9),
                12: (0.10, 0.0, 0.9),
            }
        )
        out = compute_joint_angles(pose, _full_conf())
        assert out[0, 0, _trunk_index()] == pytest.approx(0.0, abs=1e-3)

    def test_horizontal_trunk_is_ninety(self):
        """A trunk lying flat along +Y gives ~90°."""
        pose = _make_pose(
            {
                5: (-0.18, 0.6, 0.0),
                6: (0.18, 0.6, 0.0),
                11: (-0.10, 0.0, 0.0),
                12: (0.10, 0.0, 0.0),
            }
        )
        out = compute_joint_angles(pose, _full_conf())
        assert out[0, 0, _trunk_index()] == pytest.approx(90.0, abs=1e-3)

    def test_inverted_trunk_is_one_eighty(self):
        """An upside-down trunk (handstand) gives ~180°."""
        pose = _make_pose(
            {
                5: (-0.18, 0.0, 0.0),
                6: (0.18, 0.0, 0.0),
                11: (-0.10, 0.0, 1.5),
                12: (0.10, 0.0, 1.5),
            }
        )
        out = compute_joint_angles(pose, _full_conf())
        assert out[0, 0, _trunk_index()] == pytest.approx(180.0, abs=1e-3)

    def test_nan_when_hip_missing(self):
        """Any hip or shoulder with conf=0 must NaN-out the trunk angle."""
        pose = _make_pose(
            {
                5: (-0.18, 0.0, 1.5),
                6: (0.18, 0.0, 1.5),
                11: (-0.10, 0.0, 0.9),
                12: (0.10, 0.0, 0.9),
            }
        )
        conf = _full_conf()
        conf[0, 0, 12] = 0.0  # zero one of the hips
        out = compute_joint_angles(pose, conf)
        assert np.isnan(out[0, 0, _trunk_index()])


# ----------------------------------------------------------------------
# Definitions table sanity
# ----------------------------------------------------------------------


class TestAngleDefinitions:
    def test_all_indices_in_coco17_range(self):
        for adef in ANGLE_DEFINITIONS:
            if adef.indices is None:
                continue
            for idx in adef.indices:
                assert 0 <= idx <= 16, f"angle {adef.name!r} has out-of-range index {idx}"

    def test_names_are_unique(self):
        names = [a.name for a in ANGLE_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_expected_count(self):
        """Plan calls for exactly nine angles."""
        assert len(ANGLE_DEFINITIONS) == 9


# ----------------------------------------------------------------------
# compute_stats
# ----------------------------------------------------------------------


class TestComputeStats:
    def test_rom_equals_max_minus_min(self):
        # T=5, P=1, N=1 — controlled values 10, 20, 30, 40, 50
        angles = np.array([10, 20, 30, 40, 50], dtype=np.float32).reshape(5, 1, 1)
        stats = compute_stats(angles, angle_idx=0, person_idx=0)
        assert isinstance(stats, AngleStats)
        assert stats.min_deg == 10.0
        assert stats.max_deg == 50.0
        assert stats.mean_deg == pytest.approx(30.0)
        assert stats.rom_deg == 40.0

    def test_ignores_nan_frames(self):
        angles = np.array([10.0, np.nan, 30.0, np.nan, 50.0], dtype=np.float32).reshape(5, 1, 1)
        stats = compute_stats(angles, 0, 0)
        assert stats is not None
        assert stats.min_deg == 10.0
        assert stats.max_deg == 50.0
        assert stats.mean_deg == pytest.approx(30.0)
        assert stats.rom_deg == 40.0

    def test_all_nan_returns_none(self):
        angles = np.full((5, 1, 1), np.nan, dtype=np.float32)
        assert compute_stats(angles, 0, 0) is None

    def test_is_finite_property(self):
        angles = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1, 1)
        stats = compute_stats(angles, 0, 0)
        assert stats is not None
        assert stats.is_finite is True


# ----------------------------------------------------------------------
# CSV export
# ----------------------------------------------------------------------


class TestAnglesToCsv:
    def test_writes_header_and_rows(self, tmp_path):
        # T=3, P=2, N=2 — values clearly distinguishable.
        angles = np.array(
            [
                [[10.0, 100.0], [11.0, 101.0]],
                [[20.0, 200.0], [21.0, 201.0]],
                [[30.0, 300.0], [31.0, 301.0]],
            ],
            dtype=np.float32,
        )
        out = tmp_path / "angles.csv"
        angles_to_csv(str(out), angles, fps=30.0, angle_names=["A", "B"])

        with open(out, newline="") as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == ["frame", "time_s", "person", "A", "B"]
        # 3 frames × 2 persons → 6 data rows + header
        assert len(rows) == 7
        # Frame 0 / person 0 row
        assert rows[1][0] == "0"
        assert rows[1][2] == "0"
        assert float(rows[1][3]) == pytest.approx(10.0)

    def test_nan_written_as_empty_string(self, tmp_path):
        angles = np.array([[[np.nan, 1.0]]], dtype=np.float32)  # T=1, P=1, N=2
        out = tmp_path / "nan.csv"
        angles_to_csv(str(out), angles, fps=30.0, angle_names=["X", "Y"])
        with open(out, newline="") as fh:
            rows = list(csv.reader(fh))
        # First angle cell (column 3) is empty; second (column 4) is "1.0000"
        assert rows[1][3] == ""
        assert float(rows[1][4]) == pytest.approx(1.0)

    def test_invalid_fps_raises(self, tmp_path):
        angles = np.zeros((1, 1, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            angles_to_csv(str(tmp_path / "x.csv"), angles, fps=0.0, angle_names=["A", "B"])


class TestPeakVelocityGapHandling:
    """Peak angular velocity must not be inflated by NaN tracking gaps.

    Regression: compute_extended_stats used to differentiate the NaN-COMPACTED
    series with a uniform dt, so two samples straddling an N-frame gap were
    treated as one frame apart — inflating peak velocity by up to Nx.
    """

    @staticmethod
    def _stats(series, fps=30.0):
        arr = np.asarray(series, dtype=np.float64).reshape(-1, 1, 1)
        return compute_extended_stats(arr, 0, 0, fps=fps)

    def test_gap_does_not_inflate_peak_velocity(self):
        series = np.full(20, np.nan)
        series[0:5] = [0.0, 1.0, 2.0, 3.0, 4.0]  # run 1: 1 deg/frame -> 30 deg/s
        series[15:20] = [54.0, 55.0, 56.0, 57.0, 58.0]  # run 2, +50 deg over a 10-frame gap
        st = self._stats(series)
        # Genuine within-run velocity is 1 deg/frame = 30 deg/s.  The 50 deg
        # jump across the gap must NOT be counted (old bug -> ~765 deg/s).
        assert st.peak_vel_deg_s == pytest.approx(30.0, abs=1.0)
        assert st.peak_vel_deg_s < 100.0

    def test_no_gap_matches_plain_gradient(self):
        fps = 30.0
        series = np.array([0.0, 2.0, 4.0, 9.0, 12.0, 13.0])
        expected = float(np.max(np.abs(np.gradient(series, 1.0 / fps))))
        assert self._stats(series, fps).peak_vel_deg_s == pytest.approx(expected)

    def test_fast_within_run_movement_is_captured(self):
        # Sustained 20 deg/frame ramp at 30 fps = 600 deg/s (real, not a gap).
        st = self._stats([0.0, 20.0, 40.0, 60.0])
        assert st.peak_vel_deg_s == pytest.approx(600.0, abs=1.0)

    def test_isolated_samples_between_gaps_give_zero_velocity(self):
        series = np.full(7, np.nan)
        series[0], series[3], series[6] = 5.0, 40.0, 5.0  # each isolated by gaps
        assert self._stats(series).peak_vel_deg_s == 0.0
