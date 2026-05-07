"""Unit tests for VideoSimulator (app/capture/simulator.py)."""

from __future__ import annotations

import pytest

from app.capture.simulator import VideoSimulator


class TestVideoSimulatorStartCleanup:
    """BUG-F: start() must release already-opened captures when a later one fails."""

    def test_left_missing_raises_and_cleans_up(self, tmp_path):
        """FileNotFoundError for left video; _cap_left must be None afterwards."""
        sim = VideoSimulator(
            str(tmp_path / "nonexistent_left.mp4"),
            str(tmp_path / "nonexistent_right.mp4"),
        )
        with pytest.raises(FileNotFoundError, match="left video"):
            sim.start()
        # After failure, internal state must be clean (no leaked handles)
        assert sim._cap_left is None

    def test_right_missing_raises_and_releases_left(self, tmp_path, monkeypatch):
        """When left opens OK but right fails, _cap_left must be released (set to None)."""
        import cv2

        # Patch VideoCapture so left "opens" but right does not
        call_count = {"n": 0}

        class _FakeCap:
            def __init__(self, path):
                call_count["n"] += 1
                self._opened = call_count["n"] == 1  # first call = left = OK

            def isOpened(self):
                return self._opened

            def release(self):
                self._opened = False

            def get(self, prop):
                return 30.0

        monkeypatch.setattr(cv2, "VideoCapture", _FakeCap)

        sim = VideoSimulator("left.mp4", "right.mp4")
        with pytest.raises(FileNotFoundError, match="right video"):
            sim.start()

        # Both handles must be cleaned up after failure
        assert sim._cap_left is None
        assert sim._cap_right is None

    def test_read_before_start_raises(self, tmp_path):
        """read() before start() must raise RuntimeError."""
        sim = VideoSimulator(
            str(tmp_path / "a.mp4"),
            str(tmp_path / "b.mp4"),
        )
        with pytest.raises(RuntimeError, match="start()"):
            sim.read()


class TestVideoSimulatorFpsGuard:
    """BUG-X: start() must fall back to 30.0 fps when video metadata is NaN or invalid."""

    def _make_fake_cap_class(self, monkeypatch, fps_value: float):
        """Patch cv2.VideoCapture to report a specific FPS value."""
        import cv2

        call_count = {"n": 0}

        class _FakeCap:
            def __init__(self, path):
                call_count["n"] += 1
                self._opened = True

            def isOpened(self):
                return self._opened

            def release(self):
                self._opened = False

            def get(self, prop):
                import cv2 as _cv2

                if prop == _cv2.CAP_PROP_FPS:
                    return fps_value
                return 0.0

        monkeypatch.setattr(cv2, "VideoCapture", _FakeCap)

    def test_nan_fps_falls_back_to_30(self, monkeypatch):
        """NaN fps from video metadata must fall back to 30.0 (NaN is truthy in Python)."""
        import math

        self._make_fake_cap_class(monkeypatch, float("nan"))
        sim = VideoSimulator("left.mp4", "right.mp4")
        sim.start()
        assert sim.fps == 30.0, f"Expected 30.0 fallback, got {sim.fps}"
        assert math.isfinite(sim.fps)
        sim.stop()

    def test_negative_fps_falls_back_to_30(self, monkeypatch):
        """Negative fps from video metadata must fall back to 30.0."""
        self._make_fake_cap_class(monkeypatch, -25.0)
        sim = VideoSimulator("left.mp4", "right.mp4")
        sim.start()
        assert sim.fps == 30.0, f"Expected 30.0 fallback, got {sim.fps}"
        sim.stop()

    def test_valid_fps_is_preserved(self, monkeypatch):
        """A valid fps (e.g. 60.0) must not be replaced by the fallback."""
        self._make_fake_cap_class(monkeypatch, 60.0)
        sim = VideoSimulator("left.mp4", "right.mp4")
        sim.start()
        assert sim.fps == 60.0
        sim.stop()
