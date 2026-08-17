"""Unit tests for frame selection utilities (coverage score + video open guard)."""

import numpy as np
import pytest

from app.calib.frame_select import _coverage_score, _match_temporal_pairs


def _det(n: int = 4) -> "DetectionResult":  # noqa: F821
    """Return a minimal DetectionResult with *n* points."""
    from app.calib.board import DetectionResult

    obj = np.zeros((n, 3), dtype=np.float32)
    img = np.zeros((n, 2), dtype=np.float32)
    return DetectionResult(obj_pts=obj, img_pts=img)


class TestCoverageScore:
    def test_full_coverage(self):
        """Points spread across all grid cells → score == 1.0."""
        # 4×4 grid, place one point in the centre of each cell
        grid = 4
        w, h = 640, 480
        cell_w = w / grid
        cell_h = h / grid
        pts = []
        for row in range(grid):
            for col in range(grid):
                x = (col + 0.5) * cell_w
                y = (row + 0.5) * cell_h
                pts.append([x, y])
        img_pts = np.array(pts, dtype=np.float32)
        score = _coverage_score(img_pts, (w, h), grid=grid)
        assert score == pytest.approx(1.0)

    def test_single_cell_coverage(self):
        """All points in top-left cell → score == 1/16."""
        grid = 4
        w, h = 640, 480
        # All points well inside cell (0, 0)
        img_pts = np.array([[10.0, 10.0], [20.0, 15.0], [15.0, 20.0]], dtype=np.float32)
        score = _coverage_score(img_pts, (w, h), grid=grid)
        expected = 1.0 / (grid * grid)
        assert score == pytest.approx(expected)

    def test_half_coverage(self):
        """Points only in top two rows of 4×4 grid → score == 0.5."""
        grid = 4
        w, h = 640, 480
        cell_w = w / grid
        cell_h = h / grid
        pts = []
        for row in range(2):  # only top half
            for col in range(grid):
                x = (col + 0.5) * cell_w
                y = (row + 0.5) * cell_h
                pts.append([x, y])
        img_pts = np.array(pts, dtype=np.float32)
        score = _coverage_score(img_pts, (w, h), grid=grid)
        assert score == pytest.approx(0.5)

    def test_edge_points_clamped(self):
        """Points at image boundary should be clamped to the last cell, not error."""
        grid = 4
        w, h = 640, 480
        # Exactly at the right/bottom edge
        img_pts = np.array([[float(w), float(h)], [0.0, 0.0]], dtype=np.float32)
        score = _coverage_score(img_pts, (w, h), grid=grid)
        assert 0.0 < score <= 1.0

    def test_score_range(self):
        """Coverage score must always be in [0, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            pts = rng.uniform([0, 0], [640, 480], (rng.integers(1, 50), 2)).astype(np.float32)
            score = _coverage_score(pts, (640, 480))
            assert 0.0 <= score <= 1.0

    def test_negative_coords_clamped(self):
        """Negative pixel coordinates must be clamped to cell 0; score must not exceed 1.0."""
        grid = 4
        w, h = 640, 480
        cell_w = w / grid
        cell_h = h / grid
        # Place one point in each of the 16 cells (score = 1.0)
        pts = []
        for row in range(grid):
            for col in range(grid):
                pts.append([(col + 0.5) * cell_w, (row + 0.5) * cell_h])
        # Add a point with clearly negative x — should be clamped to cell (0, 0)
        pts.append([-50.0, 10.0])
        img_pts = np.array(pts, dtype=np.float32)
        score = _coverage_score(img_pts, (w, h), grid=grid)
        # Without clamping, the negative-coord point creates a ghost cell (-1, 0)
        # making len(cells) == 17 and score == 17/16 > 1.0
        assert score <= 1.0, (
            f"Score {score:.4f} > 1.0 — negative coord leaked an out-of-bounds cell"
        )
        assert score == pytest.approx(1.0)  # all 16 in-bounds cells remain covered


class TestExtractCalibrationFramesMaxFramesGuard:
    """BUG-L: extract_calibration_frames must reject max_frames < 1 immediately."""

    def test_zero_max_frames_raises(self):
        """max_frames=0 silently returns [] via candidates[:0]; must raise ValueError instead."""
        from app.calib.board import ChessboardDetector
        from app.calib.frame_select import extract_calibration_frames

        det = ChessboardDetector(cols=9, rows=6, square_size=0.025)
        with pytest.raises(ValueError, match="max_frames must be >= 1"):
            extract_calibration_frames("left.mp4", "right.mp4", det, max_frames=0)

    def test_negative_max_frames_raises(self):
        """max_frames=-5 would silently strip frames from the end; must raise ValueError."""
        from app.calib.board import ChessboardDetector
        from app.calib.frame_select import extract_calibration_frames

        det = ChessboardDetector(cols=9, rows=6, square_size=0.025)
        with pytest.raises(ValueError, match="max_frames must be >= 1"):
            extract_calibration_frames("left.mp4", "right.mp4", det, max_frames=-5)


class TestExtractCalibrationFramesSampleEveryGuard:
    """BUG-K: extract_calibration_frames must reject sample_every < 1 immediately."""

    def test_zero_sample_every_raises(self):
        """sample_every=0 must raise ValueError, not ZeroDivisionError deep in the loop."""
        from app.calib.board import ChessboardDetector
        from app.calib.frame_select import extract_calibration_frames

        det = ChessboardDetector(cols=9, rows=6, square_size=0.025)
        with pytest.raises(ValueError, match="sample_every must be >= 1"):
            extract_calibration_frames(
                "left.mp4",
                "right.mp4",
                det,
                sample_every=0,
            )

    def test_negative_sample_every_raises(self):
        """sample_every=-1 must raise ValueError."""
        from app.calib.board import ChessboardDetector
        from app.calib.frame_select import extract_calibration_frames

        det = ChessboardDetector(cols=9, rows=6, square_size=0.025)
        with pytest.raises(ValueError, match="sample_every must be >= 1"):
            extract_calibration_frames(
                "left.mp4",
                "right.mp4",
                det,
                sample_every=-1,
            )


class TestExtractCalibrationFramesVideoGuard:
    """BUG-I: extract_calibration_frames must raise FileNotFoundError for invalid video paths."""

    def test_raises_for_missing_left_video(self):
        from app.calib.board import ChessboardDetector
        from app.calib.frame_select import extract_calibration_frames

        det = ChessboardDetector(cols=9, rows=6, square_size=0.025)
        with pytest.raises(FileNotFoundError, match="Cannot open left video"):
            extract_calibration_frames(
                "/nonexistent/left.mp4",
                "/nonexistent/right.mp4",
                det,
            )

    def test_raises_for_missing_right_video(self, tmp_path):
        """Right video missing while left is valid (mocked) raises FileNotFoundError."""
        from unittest.mock import MagicMock, patch

        from app.calib.board import ChessboardDetector
        from app.calib.frame_select import extract_calibration_frames

        det = ChessboardDetector(cols=9, rows=6, square_size=0.025)

        # Patch VideoCapture so left "opens" but right does not
        mock_cap_left = MagicMock()
        mock_cap_left.isOpened.return_value = True
        mock_cap_right = MagicMock()
        mock_cap_right.isOpened.return_value = False

        def _fake_vc(path, *args, **kwargs):
            if "left" in path:
                return mock_cap_left
            return mock_cap_right

        with patch("cv2.VideoCapture", side_effect=_fake_vc):
            with pytest.raises(FileNotFoundError, match="Cannot open right video"):
                extract_calibration_frames("left.mp4", "right.mp4", det)


class TestExtractCalibrationFramesMinCoverageGuard:
    """BUG-P: min_coverage outside [0, 1] must raise immediately (not return confusing empty list)."""

    def test_min_coverage_above_one_raises(self):
        """min_coverage > 1.0 is impossible to satisfy and must raise ValueError."""
        from app.calib.board import ChessboardDetector
        from app.calib.frame_select import extract_calibration_frames

        det = ChessboardDetector(cols=9, rows=6, square_size=0.025)
        with pytest.raises(ValueError, match="min_coverage must be in"):
            extract_calibration_frames("left.mp4", "right.mp4", det, min_coverage=1.5)

    def test_min_coverage_negative_raises(self):
        """min_coverage < 0.0 would accept every frame with no coverage filter and must raise."""
        from app.calib.board import ChessboardDetector
        from app.calib.frame_select import extract_calibration_frames

        det = ChessboardDetector(cols=9, rows=6, square_size=0.025)
        with pytest.raises(ValueError, match="min_coverage must be in"):
            extract_calibration_frames("left.mp4", "right.mp4", det, min_coverage=-0.1)


class TestMatchTemporalPairs:
    """NEW-6: _match_temporal_pairs greedy matching logic."""

    def test_empty_solo_l_returns_empty(self):
        """No left detections → no matches possible."""
        result = _match_temporal_pairs({}, {10: _det()}, actual_window=5)
        assert result == []

    def test_empty_solo_r_returns_empty(self):
        """No right detections → no matches possible."""
        result = _match_temporal_pairs({10: _det()}, {}, actual_window=5)
        assert result == []

    def test_exact_match_within_window(self):
        """Frame indices equal → distance 0, within any window."""
        dl = _det()
        dr = _det()
        result = _match_temporal_pairs({5: dl}, {5: dr}, actual_window=0)
        assert len(result) == 1
        fi_l, fi_r, out_dl, out_dr = result[0]
        assert fi_l == 5
        assert fi_r == 5
        assert out_dl is dl
        assert out_dr is dr

    def test_within_window_matched(self):
        """Distance == actual_window boundary is included (<=)."""
        dl = _det()
        dr = _det()
        result = _match_temporal_pairs({0: dl}, {4: dr}, actual_window=4)
        assert len(result) == 1

    def test_outside_window_not_matched(self):
        """Distance just beyond window is excluded."""
        result = _match_temporal_pairs({0: _det()}, {5: _det()}, actual_window=4)
        assert result == []

    def test_each_detection_used_at_most_once(self):
        """Greedy: one right detection cannot be paired to two left detections."""
        # Both L frames are close to R frame 10; only one should be matched.
        solo_l = {9: _det(), 11: _det()}
        solo_r = {10: _det()}
        result = _match_temporal_pairs(solo_l, solo_r, actual_window=5)
        assert len(result) == 1

    def test_closest_pair_wins(self):
        """Closest temporal pair is assigned first (greedy by distance)."""
        # L=0 is closer to R=1 (dist=1) than L=5 is (dist=4).
        # After L=0 takes R=1, L=5 may still match R=5 (dist=0) if present.
        solo_l = {0: _det(), 5: _det()}
        solo_r = {1: _det(), 5: _det()}
        result = _match_temporal_pairs(solo_l, solo_r, actual_window=6)
        paired_l = {r[0] for r in result}
        paired_r = {r[1] for r in result}
        # Both L frames should be matched (two distinct R frames available)
        assert len(result) == 2
        assert paired_l == {0, 5}
        assert paired_r == {1, 5}

    def test_multiple_independent_pairs(self):
        """When pairs are well separated in time, all are matched."""
        solo_l = {0: _det(), 100: _det(), 200: _det()}
        solo_r = {2: _det(), 102: _det(), 202: _det()}
        result = _match_temporal_pairs(solo_l, solo_r, actual_window=5)
        assert len(result) == 3


class TestPartialDetectionsExcluded:
    """Regression: a corner-less "partial" detection is not calibration data.

    CharucoDetector returns DetectionResult(partial=True) with EMPTY point
    arrays when ArUco markers are visible but the board layout does not match —
    a UI diagnostic only.  Treating it as a detection put empty point sets into
    all_det_*_out, and the fisheye phase-1 _spatial_subsample then took the
    centroid of zero points → NaN → "cannot convert float NaN to integer",
    aborting the whole calibration run.
    """

    @staticmethod
    def _partial():
        from app.calib.board import DetectionResult

        return DetectionResult(
            obj_pts=np.empty((0, 3), np.float32),
            img_pts=np.empty((0, 2), np.float32),
            n_markers=3,
            partial=True,
        )

    @staticmethod
    def _write_video(path, n_frames=6, size=(64, 48)):
        cv2 = pytest.importorskip("cv2")
        w, h = size
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h))
        for _ in range(n_frames):
            vw.write(np.full((h, w, 3), 128, np.uint8))
        vw.release()
        return str(path)

    class _StubDetector:
        """Minimal BoardDetector returning a fixed result for every frame."""

        def __init__(self, result):
            self._result = result

        def detect(self, gray):
            return self._result

        @property
        def board_cfg(self):
            return {"type": "charuco"}

        @property
        def min_corners(self):
            return 4

    def test_partial_not_collected_and_no_pairs(self, tmp_path):
        from app.calib.frame_select import extract_calibration_frames

        left = self._write_video(tmp_path / "l.avi")
        right = self._write_video(tmp_path / "r.avi")
        all_l, all_r = [], []
        sel = extract_calibration_frames(
            left,
            right,
            self._StubDetector(self._partial()),
            sample_every=1,
            all_det_l_out=all_l,
            all_det_r_out=all_r,
        )
        assert all_l == [] and all_r == []
        assert sel == []

    def test_real_detections_still_collected(self, tmp_path):
        """The fix must not stop genuine detections from being collected."""
        from app.calib.frame_select import extract_calibration_frames

        left = self._write_video(tmp_path / "l.avi")
        right = self._write_video(tmp_path / "r.avi")
        all_l, all_r = [], []
        extract_calibration_frames(
            left,
            right,
            self._StubDetector(_det(8)),
            sample_every=1,
            all_det_l_out=all_l,
            all_det_r_out=all_r,
        )
        assert len(all_l) > 0 and len(all_r) > 0


class TestSpatialSubsampleEmptyGuard:
    """Defence in depth for the same failure inside _spatial_subsample."""

    def test_corner_less_detection_does_not_crash(self):
        from app.calib.board import DetectionResult
        from app.calib.stereo import _spatial_subsample

        empty = DetectionResult(
            obj_pts=np.empty((0, 3), np.float32),
            img_pts=np.empty((0, 2), np.float32),
            partial=True,
        )
        good = [_det(8) for _ in range(5)]
        out = _spatial_subsample([empty, *good, empty], (640, 480), n_max=3)
        assert len(out) == 3
        assert all(len(d.img_pts) > 0 for d in out)
