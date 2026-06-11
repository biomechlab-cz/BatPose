"""
UI tests for the Reconstruction tab (ReconTab).

Uses pytest-qt to drive the widget state machine without a real display
(QT_QPA_PLATFORM=offscreen is set in tests/conftest.py).

Tests verify widget behaviour — button states, label text, slider values —
not visual rendering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from app.gui.recon_tab import ReconTab

_PROJECT = Path(__file__).parents[2] / "data" / "Test project" / "test_fixture"
_CALIB = _PROJECT / "calibration.yml"
_P2D_L = _PROJECT / "pose2d_left.npz"
_P2D_R = _PROJECT / "pose2d_right.npz"
_HAS_FIXTURES = _CALIB.exists() and _P2D_L.exists() and _P2D_R.exists()


@pytest.fixture
def tab(qtbot):
    widget = ReconTab()
    qtbot.addWidget(widget)
    widget.show()
    return widget


def _make_pose3d_npz(
    tmp_path: Path, T: int = 60, fps: float = 30.0, coordinate_frame: str | None = None
) -> str:
    """Create a minimal synthetic pose3d.npz for playback tests."""
    rng = np.random.default_rng(0)
    joints3d = rng.uniform(-1.0, 1.0, (T, 1, 17, 3)).astype(np.float32)
    conf3d = np.ones((T, 1, 17), dtype=np.float32)
    meta = {"fps": fps, "model_name": "synthetic"}
    if coordinate_frame is not None:
        meta["coordinate_frame"] = coordinate_frame
    path = str(tmp_path / "pose3d.npz")
    np.savez(path, joints3d=joints3d, conf3d=conf3d, meta=np.array(meta))
    return path


class TestRunButtonGating:
    def test_run_disabled_on_empty_fields(self, tab):
        assert not tab._run_btn.isEnabled()

    def test_run_disabled_with_only_left_video(self, tab, tmp_path):
        f = tmp_path / "left.avi"
        f.write_bytes(b"fake")
        tab._left_edit.setText(str(f))
        assert not tab._run_btn.isEnabled()

    def test_run_disabled_with_only_calib(self, tab):
        if _CALIB.exists():
            tab._calib_edit.setText(str(_CALIB))
        assert not tab._run_btn.isEnabled()

    @pytest.mark.skipif(not _HAS_FIXTURES, reason="Sample fixtures required")
    def test_run_enabled_with_all_real_files(self, tab, tmp_path):
        f_l = tmp_path / "left.avi"
        f_r = tmp_path / "right.avi"
        f_l.write_bytes(b"x")
        f_r.write_bytes(b"x")
        tab._left_edit.setText(str(f_l))
        tab._right_edit.setText(str(f_r))
        tab._calib_edit.setText(str(_CALIB))
        # validate_inputs checks file existence, not content
        assert tab._run_btn.isEnabled()


class TestPlaybackTransport:
    @pytest.fixture
    def loaded_tab(self, tab, tmp_path):
        path = _make_pose3d_npz(tmp_path, T=60, fps=30.0)
        tab._load_pose3d(path)
        return tab

    def test_slider_range_set_after_load(self, loaded_tab):
        assert loaded_tab._slider.maximum() == 59  # T-1

    def test_slider_starts_at_zero(self, loaded_tab):
        assert loaded_tab._slider.value() == 0

    def test_frame_label_shows_time(self, loaded_tab):
        text = loaded_tab._frame_label.text()
        assert "Frame:" in text
        assert "00:00" in text  # current time
        assert "00:01" in text  # duration (60 frames @ 30 fps = 2s... wait 60/30=2s = 00:02)

    def test_next_frame_advances_slider(self, loaded_tab, qtbot):
        loaded_tab._slider.setValue(0)
        qtbot.mouseClick(loaded_tab._next_btn, Qt.MouseButton.LeftButton)
        assert loaded_tab._slider.value() == 1

    def test_prev_frame_decrements_slider(self, loaded_tab, qtbot):
        loaded_tab._slider.setValue(10)
        qtbot.mouseClick(loaded_tab._prev_btn, Qt.MouseButton.LeftButton)
        assert loaded_tab._slider.value() == 9

    def test_prev_frame_at_zero_stays_zero(self, loaded_tab, qtbot):
        loaded_tab._slider.setValue(0)
        qtbot.mouseClick(loaded_tab._prev_btn, Qt.MouseButton.LeftButton)
        assert loaded_tab._slider.value() == 0

    def test_next_frame_at_end_stays_at_end(self, loaded_tab, qtbot):
        loaded_tab._slider.setValue(59)
        qtbot.mouseClick(loaded_tab._next_btn, Qt.MouseButton.LeftButton)
        assert loaded_tab._slider.value() == 59

    def test_seek_forward_1s_jumps_fps_frames(self, loaded_tab, qtbot):
        loaded_tab._slider.setValue(0)
        qtbot.mouseClick(loaded_tab._seek_fwd_btn, Qt.MouseButton.LeftButton)
        assert loaded_tab._slider.value() == 30  # 30 fps = 30 frames per second

    def test_seek_back_1s_moves_fps_frames(self, loaded_tab, qtbot):
        loaded_tab._slider.setValue(30)
        qtbot.mouseClick(loaded_tab._seek_back_btn, Qt.MouseButton.LeftButton)
        assert loaded_tab._slider.value() == 0

    def test_seek_back_clamps_to_zero(self, loaded_tab, qtbot):
        loaded_tab._slider.setValue(5)
        qtbot.mouseClick(loaded_tab._seek_back_btn, Qt.MouseButton.LeftButton)
        assert loaded_tab._slider.value() == 0

    def test_seek_forward_clamps_to_end(self, loaded_tab, qtbot):
        loaded_tab._slider.setValue(50)
        qtbot.mouseClick(loaded_tab._seek_fwd_btn, Qt.MouseButton.LeftButton)
        assert loaded_tab._slider.value() == 59


class TestPlayPauseIcon:
    @pytest.fixture
    def loaded_tab(self, tab, tmp_path):
        path = _make_pose3d_npz(tmp_path, T=60)
        tab._load_pose3d(path)
        return tab

    def test_play_button_starts_unchecked(self, loaded_tab):
        assert not loaded_tab._play_btn.isChecked()

    def test_play_icon_changes_to_pause_on_toggle(self, loaded_tab):
        # Record the icon before toggling; it must change (cacheKey is not
        # stable across QIcon instances in offscreen mode, so compare keys).
        key_before = loaded_tab._play_btn.icon().cacheKey()
        loaded_tab._play_btn.setChecked(True)
        assert loaded_tab._play_btn.icon().cacheKey() != key_before

    def test_pause_icon_reverts_to_play_on_untoggle(self, loaded_tab):
        loaded_tab._play_btn.setChecked(True)
        key_paused = loaded_tab._play_btn.icon().cacheKey()
        loaded_tab._play_btn.setChecked(False)
        assert loaded_tab._play_btn.icon().cacheKey() != key_paused


class TestFrameLabel:
    def test_frame_label_updates_on_slider_change(self, tab, tmp_path):
        path = _make_pose3d_npz(tmp_path, T=90, fps=30.0)
        tab._load_pose3d(path)
        tab._slider.setValue(30)
        text = tab._frame_label.text()
        assert "Frame: 30" in text
        assert "00:01" in text  # 30 frames / 30 fps = 1 second

    def test_frame_label_shows_total_duration(self, tab, tmp_path):
        path = _make_pose3d_npz(tmp_path, T=150, fps=30.0)
        tab._load_pose3d(path)
        text = tab._frame_label.text()
        assert "00:04" in text  # 150/30 = 5 s → duration label shows 00:04 (T-1 = 149 frames)


class TestSessionState:
    @pytest.mark.skipif(not _HAS_FIXTURES, reason="Sample fixtures required")
    def test_calibration_path_persists_in_session(self, tab):
        tab._calib_edit.setText(str(_CALIB))
        state = tab.session_state() if hasattr(tab, "session_state") else {}
        # ReconTab doesn't yet have session_state — skip gracefully
        if not state:
            pytest.skip("ReconTab.session_state not implemented")
        assert state.get("calib") == str(_CALIB)


class TestLoadLastRecording:
    """The 'Load Last Recording' button scans <project>/capture/ for the
    newest *_left.avi / *_right.avi pair (ADR-007 naming)."""

    def _make_recording(
        self, capture_dir: Path, stamp: str, mtime: float | None = None
    ) -> tuple[str, str]:
        capture_dir.mkdir(parents=True, exist_ok=True)
        left = capture_dir / f"{stamp}_left.avi"
        right = capture_dir / f"{stamp}_right.avi"
        left.write_bytes(b"fake-avi")
        right.write_bytes(b"fake-avi")
        if mtime is not None:
            import os

            os.utime(left, (mtime, mtime))
            os.utime(right, (mtime, mtime))
        return str(left), str(right)

    def test_button_disabled_without_project(self, tab):
        assert not tab._load_last_btn.isEnabled()

    def test_button_disabled_when_no_capture_folder(self, tab, tmp_path):
        tab.set_project_dir(str(tmp_path))
        assert not tab._load_last_btn.isEnabled()

    def test_button_enabled_when_recording_present(self, tab, tmp_path):
        self._make_recording(tmp_path / "capture", "20260101_120000")
        tab.set_project_dir(str(tmp_path))
        assert tab._load_last_btn.isEnabled()

    def test_find_returns_newest_pair(self, tab, tmp_path):
        cap = tmp_path / "capture"
        self._make_recording(cap, "20260101_120000", mtime=1_000_000)
        newer_l, newer_r = self._make_recording(cap, "20260102_130000", mtime=2_000_000)
        tab.set_project_dir(str(tmp_path))
        found = tab._find_last_recording()
        assert found == (newer_l, newer_r)

    def test_find_skips_left_without_matching_right(self, tab, tmp_path):
        cap = tmp_path / "capture"
        cap.mkdir(parents=True)
        # Orphan left (newest) with no matching right — must be skipped.
        orphan = cap / "20260103_140000_left.avi"
        orphan.write_bytes(b"fake")
        import os

        os.utime(orphan, (3_000_000, 3_000_000))
        good_l, good_r = self._make_recording(cap, "20260101_120000", mtime=1_000_000)
        tab.set_project_dir(str(tmp_path))
        found = tab._find_last_recording()
        assert found == (good_l, good_r)

    def test_load_populates_both_edit_boxes(self, tab, tmp_path):
        left, right = self._make_recording(tmp_path / "capture", "20260101_120000")
        tab.set_project_dir(str(tmp_path))
        tab._on_load_last_recording()
        assert tab._left_edit.text() == left
        assert tab._right_edit.text() == right

    def test_find_returns_none_without_project(self, tab):
        assert tab._find_last_recording() is None


class TestOutputNaming:
    """Output filename tracks the source videos (20260602_121151.npz)."""

    def _make_recording(self, capture_dir: Path, stamp: str) -> tuple[str, str]:
        capture_dir.mkdir(parents=True, exist_ok=True)
        left = capture_dir / f"{stamp}_left.avi"
        right = capture_dir / f"{stamp}_right.avi"
        left.write_bytes(b"fake")
        right.write_bytes(b"fake")
        return str(left), str(right)

    def test_stem_strips_left_right_suffix(self, tab, tmp_path):
        tab._left_edit.setText(str(tmp_path / "20260602_121151_left.avi"))
        tab._right_edit.setText(str(tmp_path / "20260602_121151_right.avi"))
        assert tab._derive_output_stem() == "20260602_121151"

    def test_stem_falls_back_to_pose3d_when_empty(self, tab):
        assert tab._derive_output_stem() == "pose3d"

    def test_output_named_after_recording_on_load(self, tab, tmp_path):
        self._make_recording(tmp_path / "capture", "20260602_121151")
        tab.set_project_dir(str(tmp_path))
        tab._on_load_last_recording()
        out = Path(tab._out_edit.text())
        assert out.name == "20260602_121151.npz"
        assert out.parent == tmp_path  # anchored in the project folder

    def test_output_defaults_to_pose3d_before_videos(self, tab, tmp_path):
        # Project set, no videos yet → falls back to <project>/pose3d.npz
        tab.set_project_dir(str(tmp_path))
        assert Path(tab._out_edit.text()).name == "pose3d.npz"

    def test_manual_output_edit_not_clobbered(self, tab, tmp_path):
        self._make_recording(tmp_path / "capture", "20260602_121151")
        tab.set_project_dir(str(tmp_path))
        # Simulate a user hand-editing the output field (textEdited signal).
        tab._out_edit.setText(str(tmp_path / "my_custom_name.npz"))
        tab._out_edit.textEdited.emit(str(tmp_path / "my_custom_name.npz"))
        # Loading videos afterwards must NOT overwrite the custom name.
        tab._on_load_last_recording()
        assert Path(tab._out_edit.text()).name == "my_custom_name.npz"

    def test_browse_output_disables_auto_naming(self, tab, tmp_path, monkeypatch):
        self._make_recording(tmp_path / "capture", "20260602_121151")
        tab.set_project_dir(str(tmp_path))
        custom = str(tmp_path / "chosen.npz")
        monkeypatch.setattr(
            "app.gui.recon_tab.QFileDialog.getSaveFileName",
            lambda *a, **k: (custom, "NumPy (*.npz)"),
        )
        tab._browse_output()
        assert tab._out_auto is False
        # A later video load keeps the browsed name.
        tab._on_load_last_recording()
        assert Path(tab._out_edit.text()).name == "chosen.npz"


class TestCsvExportMetadata:
    """The CSV export's sidecar metadata records the coordinate frame (ADR-010)."""

    def _export(self, tab, tmp_path, monkeypatch, coordinate_frame):
        import json

        tab._pose3d_path = _make_pose3d_npz(tmp_path, T=3, coordinate_frame=coordinate_frame)
        out = tmp_path / "export.csv"
        monkeypatch.setattr(
            "app.gui.recon_tab.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(out), "CSV (*.csv)"),
        )
        tab._on_export()
        assert out.exists()
        return json.loads((tmp_path / "export_metadata.json").read_text())

    def test_world_frame_tag(self, tab, tmp_path, monkeypatch):
        meta = self._export(tab, tmp_path, monkeypatch, "world")
        assert meta["coordinate_frame"] == "world"

    def test_legacy_npz_defaults_to_opencv(self, tab, tmp_path, monkeypatch):
        meta = self._export(tab, tmp_path, monkeypatch, None)
        assert meta["coordinate_frame"] == "opencv"


class TestStaleWorldFrameCacheUpgrade:
    """A pose3d.npz cached before 'Set coordinate system' is in the camera frame;
    on load it must be auto-upgraded to the floor world frame using the world
    frame from its OWN recorded calibration (so startup matches Run pipeline)."""

    def _calib(self, tmp_path, with_world: bool):
        from app.calib.stereo import save_calibration, update_world_frame

        data = {
            "image_size": [1024, 1024],
            "lens_model": "standard",
            "K1": np.eye(3),
            "D1": np.zeros(5),
            "K2": np.eye(3),
            "D2": np.zeros(5),
            "R": np.eye(3),
            "T": np.array([-0.2, 0.0, 0.0]),
            "E": np.zeros((3, 3)),
            "F": np.zeros((3, 3)),
            "rms": 0.5,
            "n_frames": 10,
        }
        path = str(tmp_path / "calibration.yml")
        save_calibration(data, {"type": "charuco"}, path)
        if with_world:
            # Pure +1 m X translation so the transform is trivially checkable.
            update_world_frame(
                path,
                {
                    "R": np.eye(3).tolist(),
                    "t": [1.0, 0.0, 0.0],
                    "origin": [0, 0, 0],
                    "x_axis": [1, 0, 0],
                    "y_axis": [0, 1, 0],
                    "z_axis": [0, 0, 1],
                    "board_long_m": 0.28,
                    "board_short_m": 0.2,
                },
            )
        return path

    def _pose3d(self, tmp_path, calib_file, coordinate_frame=None):
        joints = np.zeros((3, 1, 17, 3), np.float32)
        joints[..., 0] = 2.0  # x=2 → a +1 X world frame makes it 3
        conf = np.ones((3, 1, 17), np.float32)
        meta = {"fps": 30.0, "calibration_file": calib_file}
        if coordinate_frame:
            meta["coordinate_frame"] = coordinate_frame
        path = str(tmp_path / "pose3d.npz")
        np.savez(
            path,
            joints3d=joints,
            conf3d=conf,
            repro_err=np.zeros((3, 1, 17), np.float32),
            meta=np.array([meta], dtype=object),
        )
        return path

    def test_stale_cache_upgraded_to_world(self, tab, tmp_path):
        calib = self._calib(tmp_path, with_world=True)
        path = self._pose3d(tmp_path, calib)  # no coordinate_frame → camera frame
        tab._load_pose3d(path)
        d = np.load(path, allow_pickle=True)
        assert d["meta"].item()["coordinate_frame"] == "world"
        assert np.allclose(d["joints3d"][..., 0], 3.0)  # 2 + 1 (world t_x)
        assert np.allclose(d["joints3d"][..., 1], 0.0)  # Y/Z untouched

    def test_camera_frame_cache_without_world_frame_untouched(self, tab, tmp_path):
        calib = self._calib(tmp_path, with_world=False)
        path = self._pose3d(tmp_path, calib)
        tab._load_pose3d(path)
        d = np.load(path, allow_pickle=True)
        assert d["meta"].item().get("coordinate_frame") is None  # unchanged
        assert np.allclose(d["joints3d"][..., 0], 2.0)  # not transformed

    def test_already_world_cache_not_transformed_again(self, tab, tmp_path):
        calib = self._calib(tmp_path, with_world=True)
        path = self._pose3d(tmp_path, calib, coordinate_frame="world")
        tab._load_pose3d(path)
        d = np.load(path, allow_pickle=True)
        assert d["meta"].item()["coordinate_frame"] == "world"
        assert np.allclose(d["joints3d"][..., 0], 2.0)  # no double transform

    def test_missing_source_calib_leaves_cache_alone(self, tab, tmp_path):
        # calibration_file recorded in the NPZ no longer exists → don't guess.
        path = self._pose3d(tmp_path, str(tmp_path / "gone.yml"))
        tab._load_pose3d(path)
        d = np.load(path, allow_pickle=True)
        assert d["meta"].item().get("coordinate_frame") is None
        assert np.allclose(d["joints3d"][..., 0], 2.0)


class TestMatchingPose3dAutoload:
    """On startup / video change the 3D auto-load follows the loaded videos (the
    output path), never a generic pose3d.npz from a DIFFERENT recording — that
    mismatch showed an unrelated skeleton over the previewed video."""

    def _npz(self, path):
        np.savez(
            path,
            joints3d=np.zeros((5, 1, 17, 3), np.float32),
            conf3d=np.ones((5, 1, 17), np.float32),
            repro_err=np.zeros((5, 1, 17), np.float32),
            meta=np.array([{"fps": 30.0, "coordinate_frame": "opencv"}], dtype=object),
        )

    def test_loads_reconstruction_matching_output(self, tab, tmp_path):
        match = tmp_path / "20260611_123852.npz"
        self._npz(match)
        tab._out_edit.setText(str(match))
        tab._autoload_matching_pose3d(clear_if_missing=True)
        assert tab._pose3d_path == str(match)

    def test_ignores_generic_pose3d_when_match_absent(self, tab, tmp_path):
        # A generic pose3d.npz (different recording) exists, but the output that
        # matches the loaded videos does not → nothing is loaded.
        self._npz(tmp_path / "pose3d.npz")
        tab._out_edit.setText(str(tmp_path / "20260611_123852.npz"))  # no such file
        tab._autoload_matching_pose3d(clear_if_missing=True)
        assert tab._pose3d_path is None

    def test_clear_drops_stale_skeleton_on_recording_switch(self, tab, tmp_path):
        first = tmp_path / "rec_a.npz"
        self._npz(first)
        tab._load_pose3d(str(first))
        assert tab._pose3d_path == str(first)
        tab._out_edit.setText(str(tmp_path / "rec_b.npz"))  # no reconstruction yet
        tab._autoload_matching_pose3d(clear_if_missing=True)
        assert tab._pose3d_path is None  # stale skeleton cleared

    def test_missing_without_clear_preserves_current(self, tab, tmp_path):
        first = tmp_path / "rec_a.npz"
        self._npz(first)
        tab._load_pose3d(str(first))
        tab._out_edit.setText(str(tmp_path / "rec_b.npz"))
        tab._autoload_matching_pose3d(clear_if_missing=False)
        assert tab._pose3d_path == str(first)  # not cleared (load-only context)


class TestSiblingAutoSelect:
    """Picking one stereo video auto-proposes the matching pair."""

    def _make_pair(self, d: Path, stamp: str) -> tuple[str, str]:
        d.mkdir(parents=True, exist_ok=True)
        left = d / f"{stamp}_left.avi"
        right = d / f"{stamp}_right.avi"
        left.write_bytes(b"fake")
        right.write_bytes(b"fake")
        return str(left), str(right)

    def test_sibling_path_left_to_right(self, tab, tmp_path):
        left, right = self._make_pair(tmp_path, "20260602_113443")
        assert tab._sibling_video_path(left, "left", "right") == right

    def test_sibling_path_right_to_left(self, tab, tmp_path):
        left, right = self._make_pair(tmp_path, "20260602_113443")
        assert tab._sibling_video_path(right, "right", "left") == left

    def test_sibling_none_when_missing(self, tab, tmp_path):
        left = tmp_path / "20260602_113443_left.avi"
        left.write_bytes(b"fake")  # no matching _right
        assert tab._sibling_video_path(str(left), "left", "right") is None

    def _patch_prompt(self, monkeypatch, answer):
        monkeypatch.setattr("app.gui.recon_tab.QMessageBox.question", lambda *a, **k: answer)

    def test_confirm_fills_empty_right(self, tab, tmp_path, monkeypatch):
        """Picking left → prompt → Yes → right is filled with the sibling."""
        self._patch_prompt(monkeypatch, QMessageBox.StandardButton.Yes)
        left, right = self._make_pair(tmp_path, "20260602_113443")
        tab._left_edit.setText(left)
        tab._auto_select_sibling(tab._left_edit, left)
        assert tab._right_edit.text() == right

    def test_decline_leaves_right_empty(self, tab, tmp_path, monkeypatch):
        """Picking left → prompt → No → right stays empty."""
        self._patch_prompt(monkeypatch, QMessageBox.StandardButton.No)
        left, right = self._make_pair(tmp_path, "20260602_113443")
        tab._left_edit.setText(left)
        tab._auto_select_sibling(tab._left_edit, left)
        assert tab._right_edit.text() == ""

    def test_prompts_to_replace_a_different_right(self, tab, tmp_path, monkeypatch):
        """When the right field holds a *different* video, the prompt offers to
        replace it (this is the common case after a session restore)."""
        prompted = {"shown": False}

        def _spy(*a, **k):
            prompted["shown"] = True
            return QMessageBox.StandardButton.Yes

        monkeypatch.setattr("app.gui.recon_tab.QMessageBox.question", _spy)
        left, right = self._make_pair(tmp_path, "20260602_113443")
        tab._right_edit.setText(str(tmp_path / "some_other_clip.avi"))
        tab._auto_select_sibling(tab._left_edit, left)
        assert prompted["shown"] is True
        assert tab._right_edit.text() == right  # replaced after Yes

    def test_replace_declined_keeps_existing_right(self, tab, tmp_path, monkeypatch):
        """Declining the replace prompt keeps the previously-set right video."""
        self._patch_prompt(monkeypatch, QMessageBox.StandardButton.No)
        left, _ = self._make_pair(tmp_path, "20260602_113443")
        other = str(tmp_path / "some_other_clip.avi")
        tab._right_edit.setText(other)
        tab._auto_select_sibling(tab._left_edit, left)
        assert tab._right_edit.text() == other

    def test_no_prompt_when_right_already_is_sibling(self, tab, tmp_path, monkeypatch):
        """No dialog when the other field already holds the matching sibling."""
        prompted = {"shown": False}

        def _spy(*a, **k):
            prompted["shown"] = True
            return QMessageBox.StandardButton.Yes

        monkeypatch.setattr("app.gui.recon_tab.QMessageBox.question", _spy)
        left, right = self._make_pair(tmp_path, "20260602_113443")
        tab._right_edit.setText(right)  # already the sibling
        tab._auto_select_sibling(tab._left_edit, left)
        assert prompted["shown"] is False

    def test_no_prompt_when_no_sibling_exists(self, tab, tmp_path, monkeypatch):
        """No dialog when there is no matching sibling file on disk."""
        prompted = {"shown": False}

        def _spy(*a, **k):
            prompted["shown"] = True
            return QMessageBox.StandardButton.Yes

        monkeypatch.setattr("app.gui.recon_tab.QMessageBox.question", _spy)
        left = tmp_path / "20260602_113443_left.avi"
        left.write_bytes(b"fake")  # no _right sibling
        tab._auto_select_sibling(tab._left_edit, str(left))
        assert prompted["shown"] is False

    def test_confirm_fills_empty_left_from_right(self, tab, tmp_path, monkeypatch):
        """Picking right → prompt → Yes → left is filled (reverse direction)."""
        self._patch_prompt(monkeypatch, QMessageBox.StandardButton.Yes)
        left, right = self._make_pair(tmp_path, "20260602_113443")
        tab._right_edit.setText(right)
        tab._auto_select_sibling(tab._right_edit, right)
        assert tab._left_edit.text() == left
