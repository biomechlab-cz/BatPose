"""
Tests for CaptureWorker's recording path (two writer threads, one per camera).

Uses a fake capture source + tiny synthetic frames so the write pipeline can be
exercised without FLIR hardware.  Verifies that every captured frame is written
(no time-compressing drops) and the timestamp sidecar matches.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from app.gui.workers import CaptureWorker


class _FakeFrame:
    def __init__(self, idx: int, fps: float, h: int, w: int):
        # Distinct-ish content per frame (not strictly needed, but realistic).
        self.frame_left = np.full((h, w, 3), (idx * 3) % 255, dtype=np.uint8)
        self.frame_right = np.full((h, w, 3), (idx * 7) % 255, dtype=np.uint8)
        self.frame_index = idx
        self.timestamp = idx / fps
        # Hardware clocks at a steady nominal period (ns).
        period_ns = int(1e9 / fps)
        self.hw_timestamp_left_ns = 1_000_000_000 + idx * period_ns
        self.hw_timestamp_right_ns = 1_000_000_000 + idx * period_ns + 80_000_000
        self.hw_delta_us = 80_000.0


class _FakeSource:
    """Yields *n* frames then None, mimicking BaseCapture."""

    def __init__(self, n: int, fps: float = 50.0, h: int = 48, w: int = 64):
        self.fps = fps
        self._n = n
        self._i = 0
        self._h, self._w = h, w
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def read(self):
        if self._i >= self._n:
            return None
        f = _FakeFrame(self._i, self.fps, self._h, self._w)
        self._i += 1
        return f

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


def _run_recording(tmp_path: Path, n: int, fps: float = 50.0):
    out_l = str(tmp_path / "rec_left.avi")
    out_r = str(tmp_path / "rec_right.avi")
    worker = CaptureWorker(_FakeSource(n, fps=fps))
    worker.begin_recording(out_l, out_r)  # record from the first frame
    worker._run()                         # synchronous — no QThread needed
    return out_l, out_r


def _frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


class TestCaptureWriter:
    def test_both_videos_created(self, tmp_path):
        out_l, out_r = _run_recording(tmp_path, n=30)
        assert Path(out_l).exists()
        assert Path(out_r).exists()

    def test_all_frames_written_left(self, tmp_path):
        """No frames dropped — written count equals captured count (no time compression)."""
        out_l, _ = _run_recording(tmp_path, n=30)
        assert _frame_count(out_l) == 30

    def test_all_frames_written_right(self, tmp_path):
        _, out_r = _run_recording(tmp_path, n=30)
        assert _frame_count(out_r) == 30

    def test_left_and_right_have_equal_frame_counts(self, tmp_path):
        out_l, out_r = _run_recording(tmp_path, n=45)
        assert _frame_count(out_l) == _frame_count(out_r) == 45

    def test_timestamp_sidecar_row_per_frame(self, tmp_path):
        out_l, out_r = _run_recording(tmp_path, n=30)
        ts = CaptureWorker._timestamps_path(out_l, out_r)
        assert Path(ts).exists()
        with open(ts, newline="") as f:
            rows = list(csv.reader(f))
        # header + one row per written frame
        assert len(rows) == 30 + 1
        assert rows[0] == ["frame", "hw_left_ns", "hw_right_ns", "abs_delta_us"]
        # frame indices are sequential 0..29
        assert [int(r[0]) for r in rows[1:]] == list(range(30))

    def test_video_fps_tag_matches_nominal(self, tmp_path):
        """With no drops the file's fps tag is correct, so playback duration is right."""
        out_l, _ = _run_recording(tmp_path, n=50, fps=50.0)
        cap = cv2.VideoCapture(out_l)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert abs(fps - 50.0) < 0.5

    def test_source_started_and_stopped(self, tmp_path):
        src = _FakeSource(10)
        worker = CaptureWorker(src)
        worker.begin_recording(
            str(tmp_path / "l.avi"), str(tmp_path / "r.avi")
        )
        worker._run()
        assert src.started and src.stopped
