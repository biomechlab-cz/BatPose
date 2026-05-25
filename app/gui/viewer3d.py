"""
3D skeleton viewer widget using pyqtgraph.opengl.

Renders COCO-17 skeleton joints and edges in an interactive 3D view
with mouse orbit/pan/zoom and a frame-stepped timeline.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

# COCO-17 edge list (from docs/skeleton_mapping.md §5)
COCO17_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

# Per-edge colour: left-side = blue, right-side = red, centre = green
_LEFT_JOINTS = {1, 3, 5, 7, 9, 11, 13, 15}
_RIGHT_JOINTS = {2, 4, 6, 8, 10, 12, 14, 16}

_EDGE_COLORS = []
for _i, _j in COCO17_EDGES:
    if _i in _LEFT_JOINTS or _j in _LEFT_JOINTS:
        _EDGE_COLORS.append((0.2, 0.5, 1.0, 1.0))  # blue — left
    elif _i in _RIGHT_JOINTS or _j in _RIGHT_JOINTS:
        _EDGE_COLORS.append((1.0, 0.3, 0.3, 1.0))  # red — right
    else:
        _EDGE_COLORS.append((0.3, 1.0, 0.3, 1.0))  # green — centre


def _try_import_gl():
    """Return pyqtgraph.opengl, or None if unavailable."""
    try:
        import pyqtgraph.opengl as gl

        return gl
    except Exception:
        return None


class SkeletonViewer3D(QWidget):
    """
    Interactive 3D viewer for a COCO-17 skeleton sequence.

    Usage:
        viewer = SkeletonViewer3D(parent)
        viewer.set_data(joints3d, conf3d)  # [T, P, 17, 3], [T, P, 17]
        viewer.show_frame(t)

    Emits *clicked* on a clean left click (press + release without dragging) —
    a drag is reserved for orbit, so this lets a plain click toggle fullscreen
    just like the camera previews.  *double_clicked* is also emitted for the
    double-click gesture (kept for backward compatibility).
    """

    clicked = Signal()
    double_clicked = Signal()

    # Max pointer travel (px) between press and release still counted as a
    # "click" rather than a drag/orbit.
    _CLICK_DRAG_TOLERANCE = 4

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._joints3d: np.ndarray | None = None  # [T, P, 17, 3]
        self._conf3d: np.ndarray | None = None  # [T, P, 17]
        self._T = 0
        self._P = 0
        self._current_frame = 0
        # Live-tracking helpers — set by setup_live_view().
        self._live_mode: bool = False
        self._live_floor_item = None  # second GLGridItem at the floor plane
        # Click-vs-drag discrimination for the *clicked* signal.
        self._press_pos = None

        self._gl = _try_import_gl()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Expand to fill whatever space the parent layout grants — otherwise the
        # GL canvas sits at its minimum size and the scene occupies only part of
        # the panel.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        if self._gl is None:
            label = QLabel(
                "3D viewer unavailable.\n"
                "Install: pip install pyqtgraph PyOpenGL PyOpenGL-accelerate",
                self,
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            return

        gl = self._gl
        self._glview = gl.GLViewWidget()
        self._glview.setBackgroundColor((30, 30, 30, 255))
        self._glview.setCameraPosition(distance=3.0, elevation=20, azimuth=45)
        # Forward double-clicks on the GL canvas to a signal — single-click
        # is reserved by pyqtgraph for orbit, so double-click is the natural
        # gesture for "pop out / fullscreen".
        self._glview.installEventFilter(self)
        self._glview.setToolTip("Click to view fullscreen · drag to orbit")

        # Grid
        grid = gl.GLGridItem()
        grid.setSize(x=4, y=4, z=1)
        grid.setSpacing(x=0.5, y=0.5, z=0.5)
        grid.setColor((80, 80, 80, 100))
        self._glview.addItem(grid)

        # Axes helper (RGB = XYZ)
        axes_pts = np.array(
            [
                [0, 0, 0],
                [0.3, 0, 0],
                [0, 0, 0],
                [0, 0.3, 0],
                [0, 0, 0],
                [0, 0, 0.3],
            ],
            dtype=np.float32,
        )
        axes_colors = np.array(
            [
                [1, 0, 0, 0.8],
                [1, 0, 0, 0.8],
                [0, 1, 0, 0.8],
                [0, 1, 0, 0.8],
                [0, 0, 1, 0.8],
                [0, 0, 1, 0.8],
            ],
            dtype=np.float32,
        )
        axes_item = gl.GLLinePlotItem(pos=axes_pts, color=axes_colors, mode="lines", width=2)
        self._glview.addItem(axes_item)

        # Placeholder skeleton items (will be updated per frame)
        self._edge_items: list = []
        self._dot_items: list = []

        # The GL canvas takes all the vertical space; the legend row below sits
        # at its natural height.  stretch=1 ensures the canvas — not empty space —
        # absorbs any extra height in the panel.
        self._glview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._glview, 1)

        # Coordinate system legend
        legend = QHBoxLayout()
        legend.addWidget(QLabel("<span style='color:#ff4444;'>■ X lateral</span>"))
        legend.addWidget(QLabel("<span style='color:#44ff44;'>■ Y anterior</span>"))
        legend.addWidget(QLabel("<span style='color:#4488ff;'>■ Z vertical (up)</span>"))
        legend.addWidget(
            QLabel(
                "<span style='color:#888;'>  Units: metres · Z-up right-handed · "
                "origin: stereo baseline midpoint</span>"
            )
        )
        legend.addStretch()
        # Wrap in a label-row widget
        legend_widget = QWidget()
        legend_widget.setLayout(legend)
        layout.addWidget(legend_widget)

    def set_data(self, joints3d: np.ndarray, conf3d: np.ndarray) -> None:
        """
        Load a new skeleton sequence.

        Args:
            joints3d: float32 [T, P, 17, 3] — 3D positions in metres
            conf3d:   float32 [T, P, 17]    — confidence [0, 1]
        """
        self._joints3d = joints3d
        self._conf3d = conf3d
        if joints3d.ndim == 4:
            self._T, self._P = joints3d.shape[:2]
        else:
            self._T = self._P = 0

        self._current_frame = 0
        self._rebuild_items()
        self.show_frame(0)

    def eventFilter(self, obj, event) -> bool:  # noqa: D401
        """Emit *clicked* on a no-drag left click and *double_clicked* on dbl-click.

        We must not consume the press / move / release events — pyqtgraph needs
        them to orbit the camera.  We only observe them: record the press
        position, and on release emit *clicked* if the pointer barely moved
        (a genuine click, not an orbit drag).
        """
        if obj is self._glview:
            et = event.type()
            if (
                et == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self.double_clicked.emit()
                return True
            if (
                et == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._press_pos = event.position().toPoint()
            elif (
                et == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
                and self._press_pos is not None
            ):
                delta = event.position().toPoint() - self._press_pos
                self._press_pos = None
                if (
                    abs(delta.x()) <= self._CLICK_DRAG_TOLERANCE
                    and abs(delta.y()) <= self._CLICK_DRAG_TOLERANCE
                ):
                    self.clicked.emit()
                # fall through (return False) so orbit release is handled too
        return super().eventFilter(obj, event)

    def set_frame(self, joints: np.ndarray, conf: np.ndarray) -> None:
        """Live-update helper: render a single frame without a time loop.

        Args:
            joints: [P, 17, 3] float32 — current pose joints in metres
            conf:   [P, 17]    float32 — current confidence
        """
        if joints.ndim == 3:
            joints = joints[None]  # add T dim
            conf = conf[None]
        # Guard against NaN/inf coordinates — a bad triangulation (e.g. a
        # lens-model mismatch) yields non-finite points, and pyqtgraph's GL
        # items raise "Error while drawing item" for every such frame.  Replace
        # non-finite values with 0 so the canvas stays drawable.
        joints = np.nan_to_num(joints, nan=0.0, posinf=0.0, neginf=0.0)
        # In live mode, the triangulator hands us points in the left-camera
        # OpenCV frame (X right, Y DOWN, Z forward).  Swap to a Z-up world
        # frame so the person appears standing upright in the viewer:
        #     viewer.x = cv.x  (right)
        #     viewer.y = cv.z  (forward / depth)
        #     viewer.z = -cv.y (up — flip the downward axis)
        if self._live_mode:
            j = joints
            joints = np.stack([j[..., 0], j[..., 2], -j[..., 1]], axis=-1)
        # Only rebuild GL items if the person count changed; otherwise just
        # update in-place — rebuilding allocates new line/scatter items per
        # call and is too expensive for live tracking at 5-10 Hz.
        P_new = int(joints.shape[1])
        if P_new != self._P or self._joints3d is None:
            self._joints3d = joints
            self._conf3d = conf
            self._T = 1
            self._P = P_new
            self._rebuild_items()
        else:
            self._joints3d = joints
            self._conf3d = conf
            self._T = 1
        self.show_frame(0)

    def setup_live_view(
        self,
        person_depth: float = 2.0,
        person_height: float = 1.0,
        camera_height_above_floor: float = 1.5,
    ) -> None:
        """Frame the viewer for typical live-tracking volume.

        Enables OpenCV-camera → Z-up world axis swap inside set_frame() (so
        the skeleton stands upright), zooms to the volume where a standing
        person at *person_depth* metres in front of the left camera would
        appear, and drops a translucent floor grid for orientation.

        Args:
            person_depth:  expected metres from the left camera to the subject's
                            torso (along the camera's optical axis).
            person_height: subject's expected vertical centre (~0.5–1.0 m above
                            the floor, for a standing-pose midpoint).
            camera_height_above_floor: where the left camera sits above the
                            floor — used to place the floor grid in the viewer.
        """
        self._live_mode = True
        if self._gl is None:
            return
        gl = self._gl
        # Drop an existing live floor before redoing.
        if self._live_floor_item is not None:
            try:
                self._glview.removeItem(self._live_floor_item)
            except Exception:
                pass
            self._live_floor_item = None
        # Floor grid in viewer coords (Z up).  Camera is at viewer-Z = 0;
        # the floor is therefore at viewer-Z = -camera_height_above_floor.
        floor = gl.GLGridItem()
        floor.setSize(x=6, y=6, z=1)
        floor.setSpacing(x=0.5, y=0.5, z=0.5)
        floor.setColor((120, 120, 120, 140))
        floor.translate(0.0, person_depth, -camera_height_above_floor)
        self._glview.addItem(floor)
        self._live_floor_item = floor

        # Aim the camera at the person's expected position and pull back
        # enough to see ~2 m vertical and ~2 m lateral comfortably.
        try:
            from pyqtgraph import Vector
            self._glview.setCameraPosition(
                pos=Vector(0.0, person_depth, person_height - camera_height_above_floor),
                distance=4.0,
                elevation=10,
                azimuth=-60,
            )
        except Exception:
            # Older pyqtgraph without Vector kwarg — best-effort fall back.
            self._glview.setCameraPosition(distance=4.0, elevation=10, azimuth=-60)

    def clear(self) -> None:
        """Remove all skeleton data and any live-mode helpers (floor grid)."""
        self._joints3d = None
        self._conf3d = None
        self._T = self._P = 0
        self._rebuild_items()
        if self._gl is not None and self._live_floor_item is not None:
            try:
                self._glview.removeItem(self._live_floor_item)
            except Exception:
                pass
            self._live_floor_item = None
        self._live_mode = False

    def show_frame(self, t: int) -> None:
        """Update display for frame index *t*."""
        if self._gl is None or self._joints3d is None:
            return
        t = max(0, min(t, self._T - 1))
        self._current_frame = t
        self._update_items(t)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_items(self) -> None:
        """Remove old GL items and create fresh ones for current P."""
        if self._gl is None:
            return
        gl = self._gl

        for item in self._edge_items + self._dot_items:
            try:
                self._glview.removeItem(item)
            except Exception:
                pass
        self._edge_items.clear()
        self._dot_items.clear()

        P = max(1, self._P)
        for _p in range(P):
            # Edge lines (one GLLinePlotItem in 'lines' mode per skeleton)
            n_edges = len(COCO17_EDGES)
            dummy_pos = np.zeros((n_edges * 2, 3), dtype=np.float32)
            dummy_col = np.zeros((n_edges * 2, 4), dtype=np.float32)
            edge_item = gl.GLLinePlotItem(
                pos=dummy_pos, color=dummy_col, mode="lines", width=2, antialias=True
            )
            self._glview.addItem(edge_item)
            self._edge_items.append(edge_item)

            # Joint dots
            dot_item = gl.GLScatterPlotItem(
                pos=np.zeros((17, 3), dtype=np.float32),
                color=np.ones((17, 4), dtype=np.float32),
                size=6,
                pxMode=True,
            )
            self._glview.addItem(dot_item)
            self._dot_items.append(dot_item)

    def _update_items(self, t: int) -> None:
        """Push frame *t* geometry to GL items."""
        if self._joints3d is None or self._P == 0:
            return

        for p in range(self._P):
            joints = self._joints3d[t, p]  # [17, 3]
            conf = self._conf3d[t, p]  # [17]

            # ── Edge lines ──────────────────────────────────────────
            n_edges = len(COCO17_EDGES)
            edge_pos = np.zeros((n_edges * 2, 3), dtype=np.float32)
            edge_col = np.zeros((n_edges * 2, 4), dtype=np.float32)

            for idx, (i, j) in enumerate(COCO17_EDGES):
                v0 = joints[i]
                v1 = joints[j]
                edge_pos[idx * 2] = v0
                edge_pos[idx * 2 + 1] = v1

                vis = min(conf[i], conf[j])
                base_col = _EDGE_COLORS[idx]
                alpha = float(vis) * base_col[3]
                edge_col[idx * 2] = (*base_col[:3], alpha)
                edge_col[idx * 2 + 1] = (*base_col[:3], alpha)

            if p < len(self._edge_items):
                self._edge_items[p].setData(pos=edge_pos, color=edge_col)

            # ── Joint dots ───────────────────────────────────────────
            dot_col = np.zeros((17, 4), dtype=np.float32)
            for j_idx in range(17):
                vis = float(conf[j_idx])
                dot_col[j_idx] = (1.0, 1.0, 0.0, vis)  # yellow

            if p < len(self._dot_items):
                self._dot_items[p].setData(pos=joints.astype(np.float32), color=dot_col)

    @property
    def frame_count(self) -> int:
        return self._T

    @property
    def current_frame(self) -> int:
        return self._current_frame
