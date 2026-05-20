"""
3D skeleton viewer widget using pyqtgraph.opengl.

Renders COCO-17 skeleton joints and edges in an interactive 3D view
with mouse orbit/pan/zoom and a frame-stepped timeline.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

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
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._joints3d: np.ndarray | None = None  # [T, P, 17, 3]
        self._conf3d: np.ndarray | None = None  # [T, P, 17]
        self._T = 0
        self._P = 0
        self._current_frame = 0

        self._gl = _try_import_gl()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

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

        layout.addWidget(self._glview)

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

    def set_frame(self, joints: np.ndarray, conf: np.ndarray) -> None:
        """Live-update helper: render a single frame without a time loop.

        Args:
            joints: [P, 17, 3] float32 — current pose joints in metres
            conf:   [P, 17]    float32 — current confidence
        """
        if joints.ndim == 3:
            joints = joints[None]  # add T dim
            conf = conf[None]
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

    def clear(self) -> None:
        """Remove all skeleton data."""
        self._joints3d = None
        self._conf3d = None
        self._T = self._P = 0
        self._rebuild_items()

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
