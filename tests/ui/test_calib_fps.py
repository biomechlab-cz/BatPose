"""
Tests for the calibration frame-rate override in the Live Capture tab.

When the user enters Calibration Mode while streaming faster than the
calibration-optimal rate, the tab offers to drop to ~20 fps (board detection
can't keep up at 50 fps → dropped frames) and restores the original rate on
exit.  These tests cover the decision/state logic; the camera-touching stream
restart is stubbed (no hardware in CI).
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from app.gui.capture_tab import CaptureTab


@pytest.fixture
def tab(qtbot):
    w = CaptureTab()
    qtbot.addWidget(w)
    w.show()
    return w


def _stub_restart(tab) -> list:
    """Replace the (camera-touching) restart with a recorder of requested fps."""
    calls: list[float] = []

    def _fake(fps: float) -> bool:
        calls.append(fps)
        tab._fps_spin.setValue(fps)
        return True

    tab._restart_stream_at_fps = _fake
    return calls


def _patch_question(monkeypatch, answer) -> dict:
    seen = {"shown": 0}

    def _q(*a, **k):
        seen["shown"] += 1
        return answer

    monkeypatch.setattr("app.gui.capture_tab.QMessageBox.question", _q)
    return seen


class TestCalibFpsOverride:
    def test_accept_lowers_fps_and_flags_restore(self, tab, monkeypatch):
        tab._fps_spin.setValue(50.0)
        tab._worker = object()  # pretend a stream is running
        calls = _stub_restart(tab)
        _patch_question(monkeypatch, QMessageBox.StandardButton.Yes)

        tab._maybe_lower_fps_for_calibration()

        assert tab._calib_fps_active is True
        assert tab._precalib_fps == 50.0
        assert calls == [CaptureTab._CALIB_FPS]

    def test_decline_keeps_fps(self, tab, monkeypatch):
        tab._fps_spin.setValue(50.0)
        tab._worker = object()
        calls = _stub_restart(tab)
        _patch_question(monkeypatch, QMessageBox.StandardButton.No)

        tab._maybe_lower_fps_for_calibration()

        assert tab._calib_fps_active is False
        assert tab._precalib_fps is None
        assert calls == []

    def test_no_prompt_when_already_optimal(self, tab, monkeypatch):
        tab._fps_spin.setValue(CaptureTab._CALIB_FPS)
        tab._worker = object()
        seen = _patch_question(monkeypatch, QMessageBox.StandardButton.Yes)

        tab._maybe_lower_fps_for_calibration()

        assert seen["shown"] == 0
        assert tab._calib_fps_active is False

    def test_no_prompt_when_not_streaming(self, tab, monkeypatch):
        tab._fps_spin.setValue(50.0)
        tab._worker = None
        seen = _patch_question(monkeypatch, QMessageBox.StandardButton.Yes)

        tab._maybe_lower_fps_for_calibration()

        assert seen["shown"] == 0

    def test_close_restores_original_fps(self, tab):
        # Simulate an active override, then close calibration mode.
        tab._fps_spin.setValue(CaptureTab._CALIB_FPS)
        tab._calib_fps_active = True
        tab._precalib_fps = 50.0
        tab._worker = object()
        calls = _stub_restart(tab)

        tab._on_calib_close_clicked()  # no captures → no confirm dialog

        assert calls == [50.0]
        assert tab._calib_fps_active is False
        assert tab._precalib_fps is None

    def test_session_state_reports_original_fps_during_override(self, tab):
        tab._fps_spin.setValue(CaptureTab._CALIB_FPS)
        tab._calib_fps_active = True
        tab._precalib_fps = 50.0
        assert tab.session_state()["fps"] == 50.0

    def test_stream_stop_during_override_restores_spinbox(self, tab):
        tab._fps_spin.setValue(CaptureTab._CALIB_FPS)
        tab._calib_fps_active = True
        tab._precalib_fps = 50.0
        tab._worker = object()

        tab._on_worker_finished(None)  # stream ended (e.g. user pressed Stop)

        assert tab._fps_spin.value() == 50.0
        assert tab._calib_fps_active is False
        assert tab._precalib_fps is None

    def test_entering_live_pose_restores_fps(self, tab):
        """Leaving calibration mode via the Live Pose button must restore the
        fps too — it used to bypass the restore, leaving the stream stuck at
        the ~20 fps calibration rate ('live pose is slower')."""
        tab._fps_spin.setValue(CaptureTab._CALIB_FPS)
        tab._calib_fps_active = True
        tab._precalib_fps = 50.0
        tab._worker = object()
        tab._calib_mode_btn.setChecked(True)
        calls = _stub_restart(tab)

        tab._on_pose_mode_clicked(True)  # open Live Pose (closes calib mode)

        assert calls == [50.0]
        assert tab._calib_fps_active is False
        assert tab._precalib_fps is None
        assert not tab._calib_mode_btn.isChecked()


class TestCalibCloseConfirmation:
    """Closing Calibration Mode only warns about captures NOT yet consumed by a
    successful 'Run Calibration' — after a run (e.g. run → set coordinate
    system → close) it just closes."""

    def _add_captures(self, tab, n: int = 4) -> None:
        tab._calib_left_dets.extend(object() for _ in range(n))
        tab._calib_right_dets.extend(object() for _ in range(n))
        tab._calib_pairs.extend(object() for _ in range(n))

    def test_no_confirm_after_successful_run(self, tab, monkeypatch):
        self._add_captures(tab)
        tab._calib_run_done = True  # set by _on_calib_finished
        seen = _patch_question(monkeypatch, QMessageBox.StandardButton.No)

        tab._on_calib_close_clicked()

        assert seen["shown"] == 0  # no nag
        assert not tab._calib_mode_btn.isChecked()  # closed

    def test_confirm_still_shown_for_unused_captures(self, tab, monkeypatch):
        self._add_captures(tab)
        tab._calib_run_done = False
        seen = _patch_question(monkeypatch, QMessageBox.StandardButton.Yes)

        tab._on_calib_close_clicked()

        assert seen["shown"] == 1

    def test_clear_all_resets_run_done(self, tab, monkeypatch):
        """Clearing the captures invalidates the 'already calibrated' state."""
        self._add_captures(tab)
        tab._calib_run_done = True
        _patch_question(monkeypatch, QMessageBox.StandardButton.Yes)

        tab._on_calib_clear()

        assert tab._calib_run_done is False
        assert len(tab._calib_pairs) == 0
