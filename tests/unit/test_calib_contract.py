"""
Calibration YAML contract tests.

Validates the real calibration.yml produced by BatPose against the
file-format specification and biomechanics-relevant invariants.
Independent of implementation — tests what the file *must contain*,
not how it was produced.

Fixture: data/Test project/calibration.yml  (RMS ~4.6 px, fisheye, 10 frames)
Note: the test rig uses the solvePnP-averaging fallback (cv2.fisheye.stereoCalibrate
is broken on OpenCV 4.11).  Sub-pixel RMS requires ≥20 well-distributed frames;
the sample fixture was captured with only 10 frames, so ~4–5 px is expected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

_CALIB_PATH = (
    Path(__file__).parents[2] / "data" / "Test project" / "test_fixture" / "calibration.yml"
)
pytestmark = pytest.mark.skipif(
    not _CALIB_PATH.exists(), reason="calibration.yml fixture not found"
)


@pytest.fixture(scope="module")
def calib() -> dict:
    with open(_CALIB_PATH) as f:
        return yaml.safe_load(f)


class TestRequiredKeys:
    def test_intrinsic_and_extrinsic_keys_present(self, calib):
        required = {"K1", "D1", "K2", "D2", "R", "T", "lens_model", "image_size"}
        missing = required - calib.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_lens_model_is_valid_string(self, calib):
        assert calib["lens_model"] in ("fisheye", "standard")

    def test_image_size_is_two_positive_ints(self, calib):
        sz = calib["image_size"]
        assert len(sz) == 2, "image_size must have exactly 2 elements"
        w, h = sz
        assert isinstance(w, int) and w > 0
        assert isinstance(h, int) and h > 0


class TestRotationMatrix:
    def test_R_is_orthogonal(self, calib):
        R = np.array(calib["R"], dtype=np.float64)
        err = np.max(np.abs(R @ R.T - np.eye(3)))
        assert err < 1e-5, f"R @ R^T - I max error: {err:.2e}"

    def test_R_is_proper_rotation_det_1(self, calib):
        R = np.array(calib["R"], dtype=np.float64)
        assert abs(np.linalg.det(R) - 1.0) < 1e-5

    def test_R_is_3x3(self, calib):
        R = np.array(calib["R"])
        assert R.shape == (3, 3)


class TestBaseline:
    def test_baseline_within_lab_range(self, calib):
        """Camera baseline must be between 0.1 m and 5 m for a biomechanics rig."""
        T = np.array(calib["T"]).flatten()
        baseline_m = float(np.linalg.norm(T))
        assert 0.1 <= baseline_m <= 5.0, f"Baseline {baseline_m:.3f} m outside 0.1–5 m"


class TestIntrinsics:
    @pytest.mark.parametrize("key", ["K1", "K2"])
    def test_camera_matrix_positive_focal_lengths(self, calib, key):
        K = np.array(calib[key])
        assert K.shape == (3, 3), f"{key} must be 3×3"
        assert K[0, 0] > 0, f"{key} fx must be positive"
        assert K[1, 1] > 0, f"{key} fy must be positive"

    @pytest.mark.parametrize("key", ["K1", "K2"])
    def test_principal_point_within_image(self, calib, key):
        K = np.array(calib[key])
        w, h = calib["image_size"]
        cx, cy = K[0, 2], K[1, 2]
        assert 0 < cx < w, f"{key} cx={cx:.1f} outside [0, {w}]"
        assert 0 < cy < h, f"{key} cy={cy:.1f} outside [0, {h}]"


class TestQuality:
    def test_quality_key_present(self, calib):
        assert "quality" in calib, "quality section missing from calibration.yml"

    def test_rms_present_and_numeric(self, calib):
        rms = calib.get("quality", {}).get("rms")
        assert rms is not None, "quality.rms missing"
        assert isinstance(rms, float) and rms > 0

    def test_rms_below_acceptance_threshold(self, calib):
        """RMS < 5.0 px sanity check — catches completely failed calibrations.

        A well-calibrated production rig should reach < 2.0 px (ideally sub-pixel)
        with 20+ frames.  The sample test fixture was captured with only 10 frames
        using the solvePnP-averaging fallback (cv2.fisheye.stereoCalibrate is broken
        on OpenCV 4.11), so ~4–5 px is the expected range for this fixture.
        Values > 5 px indicate a calibration failure, not just limited data.
        """
        rms = calib["quality"]["rms"]
        assert rms < 5.0, f"Calibration RMS {rms:.3f} px exceeds 5.0 px threshold"

    def test_n_frames_used_reasonable(self, calib):
        n = calib.get("quality", {}).get("n_frames_used", 0)
        assert n >= 5, f"Only {n} frames used — calibration likely unreliable"
