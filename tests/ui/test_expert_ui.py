"""
UI Expert Test Suite — automated + manual protocols.

Written from the perspective of a UX / usability engineer who cares about:
  - Every button has a defined, predictable state machine
  - Users can always recover from errors without restarting
  - Feedback is immediate and unambiguous (✓/✗, progress bar, log)
  - Session state is restored correctly on restart
  - No silent failures (all error paths show a message)
  - Keyboard accessibility

Run automated only:
    uv run pytest tests/ui/test_expert_ui.py -m "not manual" -v

Run with manual (requires operator):
    uv run pytest tests/ui/test_expert_ui.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

from app.gui.calib_tab import CalibTab
from app.gui.recon_tab import ReconTab

_PROJECT = Path(__file__).parents[2] / "data" / "Test project" / "test_fixture"
_CALIB = _PROJECT / "calibration.yml"
_P2D_L = _PROJECT / "pose2d_left.npz"
_P2D_R = _PROJECT / "pose2d_right.npz"
_HAS_FIXTURES = _CALIB.exists() and _P2D_L.exists() and _P2D_R.exists()


@pytest.fixture
def calib_tab(qtbot):
    w = CalibTab()
    qtbot.addWidget(w)
    w.show()
    return w


@pytest.fixture
def recon_tab(qtbot):
    w = ReconTab()
    qtbot.addWidget(w)
    w.show()
    return w


def _make_pose3d(tmp_path: Path, T: int = 60, fps: float = 30.0) -> str:
    rng = np.random.default_rng(0)
    joints3d = rng.uniform(-1.0, 1.0, (T, 1, 17, 3)).astype(np.float32)
    conf3d = np.ones((T, 1, 17), dtype=np.float32)
    meta = {"fps": fps, "model_name": "synthetic"}
    path = str(tmp_path / "pose3d.npz")
    np.savez(path, joints3d=joints3d, conf3d=conf3d, meta=np.array(meta))
    return path


# ── Input validation feedback ────────────────────────────────────────────────

class TestInputValidationFeedback:
    """Every invalid input must produce immediate, visible feedback."""

    def test_empty_left_video_shows_red_x(self, recon_tab):
        recon_tab._left_edit.setText("")
        assert recon_tab._left_status.text() == "✗"
        assert "red" in recon_tab._left_status.styleSheet()

    def test_nonexistent_left_video_shows_red_x(self, recon_tab, tmp_path):
        recon_tab._left_edit.setText(str(tmp_path / "ghost.avi"))
        assert recon_tab._left_status.text() == "✗"

    def test_existing_left_video_shows_green_check(self, recon_tab, tmp_path):
        f = tmp_path / "left.avi"
        f.write_bytes(b"x")
        recon_tab._left_edit.setText(str(f))
        assert recon_tab._left_status.text() == "✓"
        assert "green" in recon_tab._left_status.styleSheet()

    def test_calib_tab_left_status_indicator_works(self, calib_tab):
        calib_tab._left_edit.setText("")
        assert calib_tab._left_status.text() == "✗"

    def test_run_tooltip_identifies_missing_inputs(self, recon_tab):
        recon_tab._left_edit.setText("")
        recon_tab._right_edit.setText("")
        recon_tab._calib_edit.setText("")
        tip = recon_tab._run_btn.toolTip()
        assert "left video" in tip
        assert "right video" in tip
        assert "calibration" in tip.lower()


# ── Error recovery ───────────────────────────────────────────────────────────

class TestErrorRecovery:
    """After any error, the user must be able to retry without restarting."""

    def test_calib_error_re_enables_run_button(self, calib_tab, tmp_path):
        """After _on_error, Run button must re-enable if files are still valid."""
        left = tmp_path / "left.avi"
        right = tmp_path / "right.avi"
        left.write_bytes(b"x")
        right.write_bytes(b"x")
        calib_tab._left_edit.setText(str(left))
        calib_tab._right_edit.setText(str(right))
        calib_tab._out_edit.setText(str(tmp_path / "calibration.yml"))
        # Simulate an error from a running worker
        calib_tab._run_btn.setEnabled(False)
        calib_tab._on_error("IO error reading video")
        assert calib_tab._run_btn.isEnabled(), "Run button must re-enable after error"

    def test_calib_error_disables_cancel_button(self, calib_tab):
        calib_tab._cancel_btn.setEnabled(True)
        calib_tab._on_error("some error")
        assert not calib_tab._cancel_btn.isEnabled()

    def test_calib_error_resets_progress_bar(self, calib_tab):
        """Progress bar must reset to 0 after error (Bug 4 fix)."""
        calib_tab._progress_bar.setValue(55)
        calib_tab._on_error("some error")
        assert calib_tab._progress_bar.value() == 0

    def test_calib_cancel_allows_immediate_rerun(self, calib_tab, tmp_path):
        """After cancel, if files are valid, Run button must be re-enabled."""
        left = tmp_path / "left.avi"
        right = tmp_path / "right.avi"
        left.write_bytes(b"x")
        right.write_bytes(b"x")
        calib_tab._left_edit.setText(str(left))
        calib_tab._right_edit.setText(str(right))
        calib_tab._out_edit.setText(str(tmp_path / "calibration.yml"))
        calib_tab._run_btn.setEnabled(False)  # simulate running state
        # Simulate the cancelled-run finish signal
        calib_tab._on_finished(None)
        assert calib_tab._run_btn.isEnabled()

    def test_recon_error_logged_not_silent(self, recon_tab):
        """_on_error must append something to the log (not silently discard)."""
        log_before = recon_tab._log.toPlainText()
        recon_tab._on_error("triangulation failed")
        log_after = recon_tab._log.toPlainText()
        assert len(log_after) > len(log_before)


# ── Progress bar state machine ───────────────────────────────────────────────

class TestProgressBarStates:
    def test_progress_bar_starts_at_zero(self, calib_tab):
        assert calib_tab._progress_bar.value() == 0

    def test_progress_bar_recon_starts_at_zero(self, recon_tab):
        assert recon_tab._progress_bar.value() == 0

    def test_progress_callback_updates_bar(self, recon_tab):
        recon_tab._on_progress(42, "processing")
        assert recon_tab._progress_bar.value() == 42

    def test_progress_message_appears_in_log(self, recon_tab):
        recon_tab._on_progress(50, "halfway there")
        assert "halfway there" in recon_tab._log.toPlainText()


# ── Playback controls state machine ─────────────────────────────────────────

class TestPlaybackStateMachine:
    @pytest.fixture
    def loaded(self, recon_tab, tmp_path):
        path = _make_pose3d(tmp_path, T=90)
        recon_tab._load_pose3d(path)
        return recon_tab

    def test_play_btn_starts_unchecked(self, loaded):
        assert not loaded._play_btn.isChecked()

    def test_play_btn_toggleable(self, loaded, qtbot):
        qtbot.mouseClick(loaded._play_btn, Qt.MouseButton.LeftButton)
        assert loaded._play_btn.isChecked()
        qtbot.mouseClick(loaded._play_btn, Qt.MouseButton.LeftButton)
        assert not loaded._play_btn.isChecked()

    def test_seek_fwd_btn_advances_one_second(self, loaded, qtbot):
        loaded._slider.setValue(0)
        qtbot.mouseClick(loaded._seek_fwd_btn, Qt.MouseButton.LeftButton)
        assert loaded._slider.value() == 30  # 30 fps × 1 s

    def test_seek_back_btn_retreats_one_second(self, loaded, qtbot):
        loaded._slider.setValue(60)
        qtbot.mouseClick(loaded._seek_back_btn, Qt.MouseButton.LeftButton)
        assert loaded._slider.value() == 30

    def test_next_prev_clamp_at_boundaries(self, loaded, qtbot):
        loaded._slider.setValue(0)
        qtbot.mouseClick(loaded._prev_btn, Qt.MouseButton.LeftButton)
        assert loaded._slider.value() == 0  # can't go below 0

        loaded._slider.setValue(89)
        qtbot.mouseClick(loaded._next_btn, Qt.MouseButton.LeftButton)
        assert loaded._slider.value() == 89  # can't exceed T-1

    def test_speed_combo_has_four_options(self, loaded):
        assert loaded._speed_combo.count() == 4

    def test_frame_label_updates_on_scrub(self, loaded):
        loaded._slider.setValue(30)
        text = loaded._frame_label.text()
        assert "30" in text and "00:01" in text


# ── Session-state completeness ───────────────────────────────────────────────

class TestSessionStateCompleteness:
    @pytest.mark.skipif(not _HAS_FIXTURES, reason="Sample fixtures required")
    def test_calib_path_set_via_set_calibration(self, recon_tab):
        recon_tab.set_calibration(str(_CALIB))
        assert recon_tab._calib_edit.text() == str(_CALIB)

    def test_out_path_default_updated_by_set_project_dir(self, recon_tab, tmp_path):
        recon_tab._out_edit.clear()
        recon_tab.set_project_dir(str(tmp_path))
        assert recon_tab._out_edit.text() == str(tmp_path / "pose3d.npz")

    def test_backend_combo_default_is_mediapipe(self, recon_tab):
        assert "mediapipe" in recon_tab._backend_combo.currentText().lower()

    def test_max_persons_default_is_1(self, recon_tab):
        assert recon_tab._num_poses.value() == 1


# ── Accessibility: keyboard tab order ────────────────────────────────────────

class TestKeyboardAccessibility:
    def test_run_button_is_focusable(self, recon_tab):
        """Run button must be in the keyboard tab order."""
        from PySide6.QtCore import Qt
        assert recon_tab._run_btn.focusPolicy() != Qt.FocusPolicy.NoFocus

    def test_calib_run_button_is_focusable(self, calib_tab):
        from PySide6.QtCore import Qt
        assert calib_tab._run_btn.focusPolicy() != Qt.FocusPolicy.NoFocus

    def test_slider_responds_to_keyboard(self, recon_tab, tmp_path, qtbot):
        path = _make_pose3d(tmp_path)
        recon_tab._load_pose3d(path)
        recon_tab._slider.setFocus()
        recon_tab._slider.setValue(10)
        qtbot.keyClick(recon_tab._slider, Qt.Key.Key_Right)
        assert recon_tab._slider.value() == 11


# ── Manual expert protocols ──────────────────────────────────────────────────

@pytest.mark.manual
class TestManualUIExpert:
    """
    MANUAL PROTOCOL — UI / UX Expert Walkthrough
    ==============================================
    Purpose: Verify that the BatPose UI is consistent, predictable, and
             recoverable across realistic lab usage scenarios.

    Operator requirements: One person to drive the UI, one to observe.
    Estimated time: 45 minutes for the full protocol.

    Each test ID maps to a defect report if it fails.

    Steps:
    -------
    BP-UI-001  Full workflow — calibration to 3D view (happy path)
        1. Launch BatPose.  Welcome dialog appears.
        2. Click "Open Sample Project" — verify Reconstruction tab activates
           and 3D skeleton is visible.
        3. Navigate to Calibration tab — verify calibration.yml path is filled.
        4. Navigate back to Reconstruction tab — verify 3D skeleton still shows.
        PASS if: no blank viewer or lost state on tab switch.

    BP-UI-002  Calibration cancel mid-run → retry
        1. Set valid left/right calib videos.
        2. Click "Run Calibration".  Within 3 seconds click "Cancel".
        3. Verify: Cancel button disables immediately (no double-click possible).
        4. Verify: Run button re-enables after cancellation completes.
        5. Click "Run Calibration" again — must start a fresh run.
        PASS if: second run starts cleanly with progress bar at 0.

    BP-UI-003  Live → Reconstruction → second run with different backend
        1. Open sample project.  Reconstruction tab shows 3D skeleton from
           pre-computed pose3d.npz.
        2. Switch backend combo to RTMPose.
        3. Click "Run Pipeline".
        4. After completion: verify viewer shows NEW skeleton (new model_name in log).
        5. Switch back to MediaPipe, run again.
        6. Verify viewer updates again.
        PASS if: viewer always reflects the last completed pipeline run.

    BP-UI-004  Partial path entry does not block recovery
        1. Type a partial/non-existent path into "Left video" field.
        2. Verify: ✗ indicator appears immediately.
        3. Verify: Run button stays disabled (not enabled on partial input).
        4. Complete the path to a valid file.
        5. Verify: ✓ appears and Run button enables.
        PASS if: transitions are immediate (< 100 ms) and no crash.

    BP-UI-005  Session restore after restart
        1. Set a custom output folder for capture.
        2. Close BatPose normally (X button or Ctrl+Q).
        3. Reopen BatPose.
        4. Verify: the same project folder is restored; calib path populated;
           pose3d.npz loaded if present.
        PASS if: Reconstruction tab is in the same state as before close.

    BP-UI-006  Error path: corrupt pose3d.npz
        1. Create a text file named "pose3d.npz" in the project folder.
        2. Trigger auto-load (open project or type output path).
        3. Verify: an error message appears in the log — not a crash.
        4. Verify: Run button remains usable.
        PASS if: graceful error, no unhandled exception dialog.

    BP-UI-007  FPS mismatch warning
        1. Set left video (30 fps) and right video (25 fps).
        2. Verify: a ⚠ warning appears on the right video metadata line.
        3. Verify: Run button is disabled (FPS mismatch > 1% blocks run).
        PASS if: warning is visible and run is blocked.

    BP-UI-008  Export CSV with metadata
        1. Load a pose3d.npz.
        2. Click "Export CSV…" → save to a temp folder.
        3. Verify: both export.csv and export_metadata.json are created.
        4. Open metadata.json — verify fps, coordinate_units=meters, n_joints=17.
        PASS if: both files present with correct content.
    """

    def test_placeholder(self):
        pytest.skip("Manual protocol — run by human operator; see docstring for steps")
