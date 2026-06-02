"""
Top-level pytest configuration.

Sets QT_QPA_PLATFORM=offscreen so Qt widgets render without a display,
which is required for headless CI and for the UI tests to run in parallel
with other tests without popping windows.

Also patches blocking modal dialogs so tests never hang waiting for a
human to click OK.  (show_worker_error, QMessageBox.warning, etc.)
"""

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _suppress_modal_dialogs():
    """Prevent any blocking modal dialog from hanging headless tests.

    - show_worker_error: shown by CalibTab/_ReconTab on worker errors
    - QMessageBox.warning / .question / .information: board-detection alerts,
      frame-count mismatch confirmations, welcome dialog, etc.

    Tests that specifically need to inspect dialog content should override
    this fixture locally with their own monkeypatch.
    """
    # Configure WelcomeDialog mock so it does not show a window and does not
    # trigger any sample-project or new-project action.
    from unittest.mock import MagicMock
    mock_dlg = MagicMock()
    mock_dlg.open_sample = False
    mock_dlg.new_project = False
    mock_dlg.exec.return_value = 0

    with (
        patch("app.gui.calib_tab.show_worker_error"),
        patch("app.gui.recon_tab.show_worker_error"),
        patch("app.gui.recon_tab.QMessageBox.question", return_value=0x4000),  # Yes
        patch("app.gui.calib_tab.QMessageBox.warning"),
        patch("app.gui.main_window.WelcomeDialog", return_value=mock_dlg),
    ):
        yield
