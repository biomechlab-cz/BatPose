from __future__ import annotations

from app.capture import SimulatorCapture
from app.gui import capture_tab as capture_mod
from app.gui.capture_tab import CaptureTab


def test_start_preview_enabled_without_pyspin(monkeypatch, qtbot):
    monkeypatch.setattr(capture_mod, "_PYSPIN_AVAILABLE", False)
    tab = CaptureTab()
    qtbot.addWidget(tab)

    assert tab._start_btn.isEnabled()
    assert "simulator" in tab._sdk_label.text().lower()
    assert "simulated stereo pair" in tab._cam_label.text().lower()


def test_build_source_without_pyspin_returns_simulator(monkeypatch, qtbot):
    monkeypatch.setattr(capture_mod, "_PYSPIN_AVAILABLE", False)
    tab = CaptureTab()
    qtbot.addWidget(tab)
    tab._fps_spin.setValue(12.0)

    source = tab._build_source()

    assert isinstance(source, SimulatorCapture)
    assert source.fps == 12.0
