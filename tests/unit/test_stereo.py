"""Unit tests for stereo calibration helpers (calibrate_stereo, load_calibration)."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import numpy as np
import pytest
import yaml


def _make_frame_selection(n_pts: int = 9):
    """Return a minimal synthetic FrameSelection (no real image needed)."""
    from app.calib.board import DetectionResult
    from app.calib.frame_select import FrameSelection

    # Simple planar grid: 3×3 = 9 corners
    grid_x, grid_y = 3, 3
    obj = np.zeros((n_pts, 3), dtype=np.float32)
    obj[:, 0] = np.tile(np.arange(grid_x), grid_y) * 0.04
    obj[:, 1] = np.repeat(np.arange(grid_y), grid_x) * 0.04

    rng = np.random.default_rng(0)
    img = rng.uniform(100, 500, (n_pts, 2)).astype(np.float32)

    det = DetectionResult(obj_pts=obj, img_pts=img)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return FrameSelection(frame, frame, det, det, score=0.5)


def _fake_cv2_returns():
    """Return plausible fake outputs for mocked cv2 calibration calls."""
    fake_K = np.eye(3, dtype=np.float64)
    fake_D = np.zeros(5, dtype=np.float64)
    fake_R = np.eye(3, dtype=np.float64)
    fake_T = np.zeros((3, 1), dtype=np.float64)
    fake_E = np.zeros((3, 3), dtype=np.float64)
    fake_F = np.zeros((3, 3), dtype=np.float64)
    return fake_K, fake_D, fake_R, fake_T, fake_E, fake_F


class TestCalibrateStereo:
    """Minimum-frame checks in calibrate_stereo (CODE-BUG-002 / BUG-005)."""

    def test_raises_for_zero_frames(self):
        """Empty selection list must raise ValueError."""
        from app.calib.stereo import calibrate_stereo

        with pytest.raises(ValueError, match="at least 3"):
            calibrate_stereo([], img_size=(640, 480))

    def test_raises_for_two_frames(self):
        """Two frames must still raise ValueError."""
        from app.calib.stereo import calibrate_stereo

        sel = _make_frame_selection()
        with pytest.raises(ValueError, match="at least 3"):
            calibrate_stereo([sel, sel], img_size=(640, 480))

    def test_warns_below_20_frames(self):
        """3–19 frames must emit a UserWarning about the recommended minimum."""
        from app.calib.stereo import calibrate_stereo

        selections = [_make_frame_selection()] * 5
        fake_K, fake_D, fake_R, fake_T, fake_E, fake_F = _fake_cv2_returns()

        with (
            patch("cv2.calibrateCamera", return_value=(0.5, fake_K, fake_D, [], [])),
            patch(
                "cv2.stereoCalibrate",
                return_value=(
                    0.5,
                    fake_K,
                    fake_D,
                    fake_K,
                    fake_D,
                    fake_R,
                    fake_T,
                    fake_E,
                    fake_F,
                ),
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            calibrate_stereo(selections, img_size=(640, 480))

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warns, "Expected UserWarning for < 20 calibration frames"
        assert any("recommended" in str(w.message).lower() for w in user_warns)

    def test_no_warning_at_exactly_20_frames(self):
        """Exactly 20 frames must not emit a UserWarning."""
        from app.calib.stereo import calibrate_stereo

        selections = [_make_frame_selection()] * 20
        fake_K, fake_D, fake_R, fake_T, fake_E, fake_F = _fake_cv2_returns()

        with (
            patch("cv2.calibrateCamera", return_value=(0.5, fake_K, fake_D, [], [])),
            patch(
                "cv2.stereoCalibrate",
                return_value=(
                    0.5,
                    fake_K,
                    fake_D,
                    fake_K,
                    fake_D,
                    fake_R,
                    fake_T,
                    fake_E,
                    fake_F,
                ),
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            calibrate_stereo(selections, img_size=(640, 480))

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert not user_warns, f"Unexpected UserWarning with 20 frames: {user_warns}"


class TestCalibrateStereoPointMismatch:
    """BUG-H: calibrate_stereo must raise ValueError on left/right point-count mismatch."""

    def _make_mismatched_selection(self, n_left: int, n_right: int):
        """FrameSelection where left has n_left corners and right has n_right corners."""
        from app.calib.board import DetectionResult
        from app.calib.frame_select import FrameSelection

        rng = np.random.default_rng(7)
        obj_l = np.zeros((n_left, 3), dtype=np.float32)
        img_l = rng.uniform(50, 590, (n_left, 2)).astype(np.float32)
        obj_r = np.zeros((n_right, 3), dtype=np.float32)
        img_r = rng.uniform(50, 590, (n_right, 2)).astype(np.float32)

        det_l = DetectionResult(obj_pts=obj_l, img_pts=img_l)
        det_r = DetectionResult(obj_pts=obj_r, img_pts=img_r)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return FrameSelection(frame, frame, det_l, det_r, score=0.4)

    def test_raises_on_first_frame_mismatch(self):
        """If any frame has mismatched corner counts, ValueError must be raised."""
        from app.calib.stereo import calibrate_stereo

        # 9 corners in left, 6 in right (simulates ChArUco partial detection)
        sel = self._make_mismatched_selection(9, 6)
        selections = [sel] * 5  # > 3 to pass the minimum-frame guard

        with pytest.raises(ValueError, match="left camera detected"):
            calibrate_stereo(selections, img_size=(640, 480))

    def test_error_message_includes_frame_index(self):
        """The error must report which frame caused the mismatch."""
        from app.calib.stereo import calibrate_stereo

        good = _make_frame_selection(9)  # equal counts
        bad = self._make_mismatched_selection(9, 4)
        selections = [good, good, good, bad, good]  # frame index 3 is bad

        with pytest.raises(ValueError, match="Frame 3"):
            calibrate_stereo(selections, img_size=(640, 480))

    def test_matching_counts_do_not_raise(self):
        """Consistent point counts (both cameras detect same N) must not raise."""
        from unittest.mock import patch

        from app.calib.stereo import calibrate_stereo

        selections = [_make_frame_selection(9)] * 5
        fake_K, fake_D, fake_R, fake_T, fake_E, fake_F = _fake_cv2_returns()

        with (
            patch("cv2.calibrateCamera", return_value=(0.5, fake_K, fake_D, [], [])),
            patch(
                "cv2.stereoCalibrate",
                return_value=(
                    0.5,
                    fake_K,
                    fake_D,
                    fake_K,
                    fake_D,
                    fake_R,
                    fake_T,
                    fake_E,
                    fake_F,
                ),
            ),
            pytest.warns(UserWarning),
        ):
            result = calibrate_stereo(selections, img_size=(640, 480))
        assert "rms" in result


class TestCalibrateStereoImgSizeValidation:
    """BUG-T: calibrate_stereo must reject img_size with zero or negative dimensions.

    cv2.calibrateCamera raises a cryptic cv2.error when given (0, 0) or negative
    dimensions. The check should surface as a clear ValueError before any cv2 call.
    """

    def test_zero_width_raises(self):
        """img_size=(0, 480) must raise ValueError with a mention of img_size."""
        from app.calib.stereo import calibrate_stereo

        sel = _make_frame_selection()
        with pytest.raises(ValueError, match="img_size"):
            calibrate_stereo([sel] * 3, img_size=(0, 480))

    def test_zero_height_raises(self):
        """img_size=(640, 0) must raise ValueError."""
        from app.calib.stereo import calibrate_stereo

        sel = _make_frame_selection()
        with pytest.raises(ValueError, match="img_size"):
            calibrate_stereo([sel] * 3, img_size=(640, 0))

    def test_negative_dimension_raises(self):
        """Negative dimensions are unambiguously invalid."""
        from app.calib.stereo import calibrate_stereo

        sel = _make_frame_selection()
        with pytest.raises(ValueError, match="img_size"):
            calibrate_stereo([sel] * 3, img_size=(-1, 480))


class TestCalibrateStereoTwoPhase:
    """NEW-7: two-phase calibration triggers when per-camera lists exceed stereo pairs."""

    def _run_calibrate(self, selections, all_det_l=None, all_det_r=None):
        """Helper: run calibrate_stereo with mocked cv2, return call counts."""
        from unittest.mock import patch

        from app.calib.stereo import calibrate_stereo

        fake_K, fake_D, fake_R, fake_T, fake_E, fake_F = _fake_cv2_returns()

        with (
            patch("cv2.calibrateCamera", return_value=(0.5, fake_K, fake_D, [], [])) as mock_cc,
            patch(
                "cv2.stereoCalibrate",
                return_value=(0.5, fake_K, fake_D, fake_K, fake_D, fake_R, fake_T, fake_E, fake_F),
            ) as mock_sc,
            pytest.warns(UserWarning),
        ):
            calibrate_stereo(
                selections,
                img_size=(640, 480),
                all_det_l=all_det_l,
                all_det_r=all_det_r,
            )
        return mock_cc, mock_sc

    def test_two_phase_triggers_when_more_single_cam_detections(self):
        """calibrateCamera called twice (per-camera) before stereoCalibrate when all_det lists
        have more entries than stereo pairs."""
        import cv2

        selections = [_make_frame_selection(9)] * 5
        # 10 single-camera detections > 5 stereo pairs → two-phase
        all_det_l = [_make_frame_selection(9).det_left] * 10
        all_det_r = [_make_frame_selection(9).det_right] * 10

        mock_cc, mock_sc = self._run_calibrate(selections, all_det_l, all_det_r)

        assert mock_cc.call_count == 2, "Expected two calibrateCamera calls in two-phase mode"
        # stereoCalibrate must use CALIB_FIX_INTRINSIC in two-phase mode
        stereo_flags = mock_sc.call_args.kwargs.get("flags") or mock_sc.call_args[1].get("flags")
        assert stereo_flags is not None
        assert stereo_flags & cv2.CALIB_FIX_INTRINSIC

    def test_single_phase_when_all_det_none(self):
        """When all_det_l/r are None, single-phase runs (calibrateCamera called twice on stereo
        pairs only, then stereoCalibrate with CALIB_USE_INTRINSIC_GUESS)."""
        import cv2

        selections = [_make_frame_selection(9)] * 5
        mock_cc, mock_sc = self._run_calibrate(selections, None, None)

        assert mock_cc.call_count == 2
        stereo_flags = mock_sc.call_args.kwargs.get("flags") or mock_sc.call_args[1].get("flags")
        assert stereo_flags is not None
        assert stereo_flags & cv2.CALIB_USE_INTRINSIC_GUESS

    def test_single_phase_when_all_det_not_larger(self):
        """When all_det lists are same size as selections, single-phase is used."""
        import cv2

        selections = [_make_frame_selection(9)] * 5
        # Same number of per-camera detections as stereo pairs → no benefit, stay single-phase
        all_det_l = [_make_frame_selection(9).det_left] * 5
        all_det_r = [_make_frame_selection(9).det_right] * 5

        mock_cc, mock_sc = self._run_calibrate(selections, all_det_l, all_det_r)

        stereo_flags = mock_sc.call_args.kwargs.get("flags") or mock_sc.call_args[1].get("flags")
        assert stereo_flags & cv2.CALIB_USE_INTRINSIC_GUESS

    def test_two_phase_result_has_expected_keys(self):
        """Result dict must contain all expected keys in two-phase mode."""
        from unittest.mock import patch

        from app.calib.stereo import calibrate_stereo

        fake_K, fake_D, fake_R, fake_T, fake_E, fake_F = _fake_cv2_returns()
        selections = [_make_frame_selection(9)] * 5
        all_det_l = [_make_frame_selection(9).det_left] * 10
        all_det_r = [_make_frame_selection(9).det_right] * 10

        with (
            patch("cv2.calibrateCamera", return_value=(0.5, fake_K, fake_D, [], [])),
            patch(
                "cv2.stereoCalibrate",
                return_value=(0.5, fake_K, fake_D, fake_K, fake_D, fake_R, fake_T, fake_E, fake_F),
            ),
            pytest.warns(UserWarning),
        ):
            result = calibrate_stereo(
                selections,
                img_size=(640, 480),
                all_det_l=all_det_l,
                all_det_r=all_det_r,
            )

        for key in ("K1", "D1", "K2", "D2", "R", "T", "E", "F", "rms", "n_frames", "image_size"):
            assert key in result, f"Missing key {key!r} in two-phase result"


class TestLoadCalibration:
    """Backward-compatibility tests for load_calibration."""

    def test_missing_required_key_raises_valueerror(self, tmp_path):
        """load_calibration must raise ValueError (not bare KeyError) if K1 is absent."""
        from app.calib.stereo import load_calibration

        calib_dict = {
            "image_size": [640, 480],
            # K1 intentionally absent — used to raise a bare KeyError: 'K1'
            "D1": np.zeros(5).tolist(),
            "K2": np.eye(3).tolist(),
            "D2": np.zeros(5).tolist(),
            "R": np.eye(3).tolist(),
            "T": np.zeros(3).tolist(),
        }
        out = str(tmp_path / "calib_missing_k1.yml")
        with open(out, "w") as f:
            yaml.dump(calib_dict, f)

        with pytest.raises(ValueError, match="K1"):
            load_calibration(out)

    def test_missing_image_size_raises_valueerror(self, tmp_path):
        """load_calibration must raise ValueError (not bare KeyError) if image_size absent."""
        from app.calib.stereo import load_calibration

        calib_dict = {
            # image_size intentionally absent
            "K1": np.eye(3).tolist(),
            "D1": np.zeros(5).tolist(),
            "K2": np.eye(3).tolist(),
            "D2": np.zeros(5).tolist(),
            "R": np.eye(3).tolist(),
            "T": np.zeros(3).tolist(),
        }
        out = str(tmp_path / "calib_missing_size.yml")
        with open(out, "w") as f:
            yaml.dump(calib_dict, f)

        with pytest.raises(ValueError, match="image_size"):
            load_calibration(out)

    def test_missing_e_and_f_keys(self, tmp_path):
        """load_calibration must default E and F to zero matrices if absent."""
        from app.calib.stereo import load_calibration

        calib_dict = {
            "image_size": [640, 480],
            "K1": np.eye(3).tolist(),
            "D1": np.zeros(5).tolist(),
            "K2": np.eye(3).tolist(),
            "D2": np.zeros(5).tolist(),
            "R": np.eye(3).tolist(),
            "T": np.zeros(3).tolist(),
            # E, F, quality, board_cfg intentionally absent
        }
        out = str(tmp_path / "calib_no_ef.yml")
        with open(out, "w") as f:
            yaml.dump(calib_dict, f)

        loaded = load_calibration(out)
        assert loaded["E"].shape == (3, 3)
        np.testing.assert_array_equal(loaded["E"], np.zeros((3, 3)))
        assert loaded["F"].shape == (3, 3)
        assert loaded["board_cfg"] == {}
        assert loaded["quality"] == {}

    def test_image_size_returned_as_int_tuple(self, tmp_path):
        """image_size must be a tuple of ints (not floats or lists)."""
        from app.calib.stereo import load_calibration, save_calibration

        calib = {
            "K1": np.eye(3),
            "D1": np.zeros(5),
            "K2": np.eye(3),
            "D2": np.zeros(5),
            "R": np.eye(3),
            "T": np.zeros(3),
            "E": np.zeros((3, 3)),
            "F": np.zeros((3, 3)),
            "rms": 0.42,
            "n_frames": 30,
            "image_size": [1280, 720],
        }
        out = str(tmp_path / "calib.yml")
        save_calibration(calib, {}, out)
        loaded = load_calibration(out)
        assert isinstance(loaded["image_size"], tuple)
        assert all(isinstance(x, int) for x in loaded["image_size"])
        assert loaded["image_size"] == (1280, 720)
