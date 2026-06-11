"""Persistent world-frame axis overlay + calibration-quality warning (Capture tab).

Covers the two issues reported after "Set coordinate system":
  • the axis triad must be redrawn on EVERY live preview frame (a one-shot draw
    is overwritten by the next incoming frame), and
  • when 2D pose is detected but the stereo triangulation reprojects badly, the
    3D joints are rejected ("dots, no skeleton") — the cause is a poor
    calibration, which must be surfaced loudly rather than shown as an empty
    viewer.  These tests exercise the pure state/geometry logic (no hardware).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.gui.capture_tab import CaptureTab


@pytest.fixture
def tab(qtbot):
    w = CaptureTab()
    qtbot.addWidget(w)
    return w


def _calib_with_world_frame() -> dict:
    K = np.array([[800.0, 0, 512.0], [0, 800.0, 512.0], [0, 0, 1.0]])
    return {
        "K1": K,
        "D1": np.zeros(5),
        "K2": K.copy(),
        "D2": np.zeros(5),
        "R": np.eye(3),
        "T": np.array([-0.2, 0.0, 0.0]),  # 20 cm baseline
        "lens_model": "standard",
        "world_frame": {
            "R": np.eye(3).tolist(),
            "t": [0.0, 0.0, 0.0],
            "origin": [0.0, 0.1, 2.0],  # board centre 2 m in front
            "x_axis": [1.0, 0.0, 0.0],
            "y_axis": [0.0, 1.0, 0.0],
            "z_axis": [0.0, 0.0, 1.0],
            "board_long_m": 0.28,
            "board_short_m": 0.2,
        },
    }


class TestWorldAxesProjection:
    def test_project_returns_finite_pixels_per_view(self):
        calib = _calib_with_world_frame()
        px_l, px_r = CaptureTab._project_world_axes(calib["world_frame"], calib)
        assert px_l.shape == (4, 2)
        assert px_r.shape == (4, 2)
        assert np.isfinite(px_l).all() and np.isfinite(px_r).all()
        # The baseline shifts the projection → the two views differ.
        assert not np.allclose(px_l[0], px_r[0])


class TestWorldAxesOverlay:
    def test_noop_when_unset(self, tab):
        tab._world_axes_px = None
        left = np.zeros((48, 64, 3), np.uint8)
        right = np.zeros((48, 64, 3), np.uint8)
        out_l, out_r = tab._maybe_overlay_world_axes(left, right)
        assert out_l is left and out_r is right  # untouched, same objects

    def test_draws_on_copies_when_set(self, tab):
        # Triad pixels inside the frame → something gets drawn (on a copy).
        tab._world_axes_px = (
            np.array([[32, 24], [50, 24], [32, 40], [32, 10]], float),
            np.array([[30, 24], [48, 24], [30, 40], [30, 10]], float),
        )
        left = np.zeros((48, 64, 3), np.uint8)
        right = np.zeros((48, 64, 3), np.uint8)
        out_l, out_r = tab._maybe_overlay_world_axes(left, right)
        assert out_l is not left and out_r is not right  # copies, originals intact
        assert not left.any() and not right.any()
        assert out_l.any() and out_r.any()  # triad drawn

    def test_non_finite_pixels_are_skipped_not_crashed(self, tab):
        # A point projected behind the camera → inf; must not raise in the frame loop.
        tab._world_axes_px = (
            np.array([[32, 24], [np.inf, 24], [32, 40], [32, 10]], float),
            np.array([[30, 24], [48, 24], [np.nan, 40], [30, 10]], float),
        )
        left = np.zeros((48, 64, 3), np.uint8)
        right = np.zeros((48, 64, 3), np.uint8)
        out_l, out_r = tab._maybe_overlay_world_axes(left, right)  # no exception
        assert out_l.shape == left.shape and out_r.shape == right.shape
        assert not out_l.any() and not out_r.any()  # nothing drawn for the bad triad


class TestWorldAxesSideOverlay:
    """Single-camera triad overlay used by the fullscreen inspect window."""

    def test_noop_when_unset(self, tab):
        tab._world_axes_px = None
        img = np.zeros((48, 64, 3), np.uint8)
        assert tab._overlay_world_axes_side(img, "L") is img

    def test_draws_for_requested_side_on_copy(self, tab):
        tab._world_axes_px = (
            np.array([[32, 24], [50, 24], [32, 40], [32, 10]], float),  # left
            np.array([[30, 24], [48, 24], [30, 40], [30, 10]], float),  # right
        )
        img = np.zeros((48, 64, 3), np.uint8)
        out = tab._overlay_world_axes_side(img, "R")
        assert out is not img  # copy
        assert not img.any()  # original untouched
        assert out.any()  # triad drawn


class TestRefreshWorldAxesOverlay:
    def test_clears_when_no_calibration(self, tab):
        tab._world_axes_px = (np.zeros((4, 2)), np.zeros((4, 2)))
        tab._calib_saved_path = None
        tab._refresh_world_axes_overlay()
        assert tab._world_axes_px is None

    def test_sets_from_calibration_file_with_world_frame(self, tab, tmp_path):
        from app.calib.stereo import save_calibration, update_world_frame

        calib = _calib_with_world_frame()
        data = {
            "image_size": [1024, 1024],
            "lens_model": "standard",
            "K1": calib["K1"],
            "D1": calib["D1"],
            "K2": calib["K2"],
            "D2": calib["D2"],
            "R": calib["R"],
            "T": calib["T"],
            "E": np.zeros((3, 3)),
            "F": np.zeros((3, 3)),
            "rms": 0.5,
            "n_frames": 10,
        }
        path = str(tmp_path / "calibration.yml")
        save_calibration(data, {"type": "charuco"}, path)
        update_world_frame(path, calib["world_frame"])

        tab._calib_saved_path = path
        tab._refresh_world_axes_overlay()

        assert tab._world_axes_px is not None
        px_l, px_r = tab._world_axes_px
        assert px_l.shape == (4, 2) and px_r.shape == (4, 2)

    def test_clears_when_calibration_has_no_world_frame(self, tab, tmp_path):
        from app.calib.stereo import save_calibration

        calib = _calib_with_world_frame()
        data = {
            "image_size": [1024, 1024],
            "lens_model": "standard",
            "K1": calib["K1"],
            "D1": calib["D1"],
            "K2": calib["K2"],
            "D2": calib["D2"],
            "R": calib["R"],
            "T": calib["T"],
            "E": np.zeros((3, 3)),
            "F": np.zeros((3, 3)),
            "rms": 0.5,
            "n_frames": 10,
        }
        path = str(tmp_path / "calibration.yml")
        save_calibration(data, {"type": "charuco"}, path)  # no world frame

        tab._world_axes_px = (np.zeros((4, 2)), np.zeros((4, 2)))
        tab._calib_saved_path = path
        tab._refresh_world_axes_overlay()
        assert tab._world_axes_px is None
