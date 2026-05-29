"""Integration test: joint angles on the real 20260528 sample recording.

Runs the full 3D reconstruction once via the existing pipeline, then computes
biomech angles from the resulting joints3d/conf3d arrays.  Validates that the
output has the expected shape, that NaN/finite frames line up with the
confidence mask, and that the per-angle statistics make physical sense.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.biomech import (
    ANGLE_DEFINITIONS,
    compute_joint_angles,
    compute_stats,
)


_PROJECT = Path(__file__).parents[2] / "data" / "Test project"
_CALIB = _PROJECT / "calibration.yml"
_P2D_L = _PROJECT / "pose2d_left.npz"
_P2D_R = _PROJECT / "pose2d_right.npz"

pytestmark = pytest.mark.skipif(
    not (_CALIB.exists() and _P2D_L.exists() and _P2D_R.exists()),
    reason="Sample pose2d fixtures not found in data/Test project/",
)


@pytest.fixture(scope="module")
def angles_and_conf(tmp_path_factory):
    """Run reconstruct3d once, apply the OpenCV→Z-up swap, compute angles."""
    from app.recon3d.pipeline import reconstruct3d

    out = str(tmp_path_factory.mktemp("recon_for_angles") / "pose3d.npz")
    result = reconstruct3d(
        calib_path=str(_CALIB),
        pose2d_left_path=str(_P2D_L),
        pose2d_right_path=str(_P2D_R),
        output_path=out,
    )
    assert result is not None, "reconstruct3d failed"

    d = np.load(out, allow_pickle=True)
    joints_cv = d["joints3d"]  # OpenCV camera frame
    conf = d["conf3d"]

    # Apply the same axis swap that SkeletonViewer3D.set_data() applies on
    # ingest so the angles are computed in the Z-up world frame.
    joints_zup = np.stack(
        [joints_cv[..., 0], joints_cv[..., 2], -joints_cv[..., 1]], axis=-1
    ).astype(np.float32)

    angles = compute_joint_angles(joints_zup, conf)
    return angles, conf


class TestAnglesOnRealData:
    def test_output_shape(self, angles_and_conf):
        """Angles must be [T, P, N_ANGLES]."""
        angles, conf = angles_and_conf
        T, P = conf.shape[:2]
        assert angles.shape == (T, P, len(ANGLE_DEFINITIONS))

    def test_dtype_float32(self, angles_and_conf):
        angles, _ = angles_and_conf
        assert angles.dtype == np.float32

    def test_nan_where_conf_zero(self, angles_and_conf):
        """A frame with conf=0 on the L Knee joint (vertex of angle 0) must NaN-out angle 0."""
        angles, conf = angles_and_conf
        # Find frames where LKnee (idx 13) confidence is zero (post-outlier-rejection)
        zero_knee = conf[:, :, 13] == 0
        # All such frames should have NaN in angle 0 ("L Knee Flex")
        knee_angles = angles[..., 0]
        assert np.all(np.isnan(knee_angles[zero_knee])), (
            "Some frames with conf3d[LKnee]==0 still produced finite L Knee Flex angle"
        )

    def test_finite_where_all_conf_positive(self, angles_and_conf):
        """For a frame with all relevant joints confident, the angle is finite."""
        angles, conf = angles_and_conf
        # All flanking joints of L Knee Flex (11, 13, 15) confident
        mask = (conf[..., 11] > 0) & (conf[..., 13] > 0) & (conf[..., 15] > 0)
        assert np.all(np.isfinite(angles[..., 0][mask]))

    def test_stats_finite_for_each_angle(self, angles_and_conf):
        """Per-angle statistics are finite for every angle in the single-person fixture."""
        angles, _ = angles_and_conf
        for k, adef in enumerate(ANGLE_DEFINITIONS):
            stats = compute_stats(angles, angle_idx=k, person_idx=0)
            assert stats is not None, f"{adef.name}: stats came back None"
            assert stats.is_finite, f"{adef.name}: non-finite stats {stats}"

    def test_angles_in_zero_to_180(self, angles_and_conf):
        """Every non-NaN angle lies in [0°, 180°] (with a tiny epsilon)."""
        angles, _ = angles_and_conf
        finite = angles[~np.isnan(angles)]
        assert np.all(finite >= -1e-3)
        assert np.all(finite <= 180.0 + 1e-3)
