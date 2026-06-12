"""The offline Calibration tab exposes a lens selector and passes it through.

Without this, offline calibration always ran the pinhole path — fisheye videos
silently produced a calibration tagged "standard" that poisons reconstruction.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QObject, Signal

from app.gui.calib_tab import CalibTab


@pytest.fixture
def tab(qtbot):
    w = CalibTab()
    qtbot.addWidget(w)
    return w


class _FakeWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)
    frame_ready = Signal(object)
    captured: dict = {}

    def __init__(self, **kwargs):
        super().__init__()
        type(self).captured = kwargs

    def start(self):  # don't actually spawn a thread in tests
        pass


def test_lens_combo_defaults_to_fisheye(tab):
    # The deployed rig is fisheye — same default as the live calibration panel.
    assert tab._lens_combo.currentIndex() == 2
    assert "Fisheye" in tab._lens_combo.currentText()


def test_run_passes_lens_model_to_worker(tab, monkeypatch):
    monkeypatch.setattr("app.gui.calib_tab.CalibWorker", _FakeWorker)
    tab._left_edit.setText("left.mp4")
    tab._right_edit.setText("right.mp4")
    tab._lens_combo.setCurrentIndex(2)
    tab._on_run()
    assert _FakeWorker.captured.get("lens_model") == 2

    tab._lens_combo.setCurrentIndex(0)
    tab._on_run()
    assert _FakeWorker.captured.get("lens_model") == 0
