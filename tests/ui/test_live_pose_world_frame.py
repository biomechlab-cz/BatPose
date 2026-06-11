"""Live Pose expresses joints in the floor world frame (ADR-010).

Before this, the live 3D viewer always showed camera-frame joints with the
legacy OpenCV→Z-up swap, so after "Set coordinate system" the origin triad sat
at the camera-based origin NEXT to the subject — even when the subject stood
physically on the board.  Now the display copy is re-expressed in the world
frame (transform LAST, after smoothing — mirroring the offline pipeline) and
the viewer origin IS the board.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.gui.capture_tab import CaptureTab
from app.gui.viewer3d import SkeletonViewer3D

K = np.array([[800.0, 0, 512.0], [0, 800.0, 512.0], [0, 0, 1.0]])


@pytest.fixture
def tab(qtbot):
    w = CaptureTab()
    qtbot.addWidget(w)
    return w


def _project(p: np.ndarray) -> np.ndarray:
    return np.array([p[0] / p[2] * 800 + 512, p[1] / p[2] * 800 + 512], dtype=np.float32)


def _calib(world: bool) -> dict:
    c = {
        "K1": K,
        "D1": np.zeros(5),
        "K2": K,
        "D2": np.zeros(5),
        "R": np.eye(3),
        "T": np.array([-0.2, 0.0, 0.0]),
        "lens_model": "standard",
        "world_frame": None,
    }
    if world:
        # Pure +1 m X translation → trivially checkable transform.
        c["world_frame"] = {
            "R": np.eye(3).tolist(),
            "t": [1.0, 0.0, 0.0],
            "origin": [0, 0, 0],
            "x_axis": [1, 0, 0],
            "y_axis": [0, 1, 0],
            "z_axis": [0, 0, 1],
            "board_long_m": 0.28,
            "board_short_m": 0.2,
        }
    return c


class TestViewerSetFrameWorldFlag:
    def test_world_frame_skips_swap(self, qtbot):
        v = SkeletonViewer3D()
        qtbot.addWidget(v)
        j = np.zeros((1, 17, 3), np.float32)
        j[0, 0] = [1.0, 2.0, 3.0]
        v.set_frame(j, np.ones((1, 17), np.float32), world_frame=True)
        assert np.allclose(v._joints3d[0, 0, 0], [1.0, 2.0, 3.0])  # as-is

    def test_legacy_applies_swap(self, qtbot):
        v = SkeletonViewer3D()
        qtbot.addWidget(v)
        j = np.zeros((1, 17, 3), np.float32)
        j[0, 0] = [1.0, 2.0, 3.0]
        v.set_frame(j, np.ones((1, 17), np.float32))
        assert np.allclose(v._joints3d[0, 0, 0], [1.0, 3.0, -2.0])  # [X, Z, -Y]


class TestLivePoseWorldFrame:
    def _run(self, tab, world: bool):
        tab._pose_calib = _calib(world)
        tab._pose_smooth_check.setChecked(False)
        point = np.array([0.1, 0.2, 2.0])  # cam-1 frame, 2 m in front
        pl = _project(point)
        pr = _project(point + np.array([-0.2, 0.0, 0.0]))  # p_cam2 = R p + T
        kps_l = np.tile(pl, (1, 17, 1)).astype(np.float32)
        kps_r = np.tile(pr, (1, 17, 1)).astype(np.float32)
        conf = np.ones((1, 17), np.float32)
        tab._process_pose_result(kps_l, conf.copy(), kps_r, conf.copy())
        return point, tab._pose_last_view_3d[0, 0], tab._pose_last_3d[0, 0]

    def test_world_frame_applied_to_display_copy(self, tab):
        point, view, cam = self._run(tab, world=True)
        # Smoothing/hold-last stash stays CAMERA-frame…
        assert np.allclose(cam, point, atol=1e-3)
        # …while the viewer copy is in the world frame (+1 m X here).
        assert np.allclose(view, point + np.array([1.0, 0.0, 0.0]), atol=1e-3)
        assert tab._pose_world is True

    def test_legacy_calibration_unchanged(self, tab):
        point, view, cam = self._run(tab, world=False)
        assert np.allclose(view, point, atol=1e-3)
        assert np.allclose(cam, point, atol=1e-3)
        assert tab._pose_world is False
