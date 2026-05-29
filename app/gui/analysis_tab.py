"""
Analysis tab — biomechanical joint-angle inspection and export.

The tab loads a pose3d.npz file (either on demand via `Load pose3d.npz…`
or automatically when the Reconstruction tab finishes and emits its
``pose3d_ready`` signal).  It computes nine clinically relevant joint
angles (see :mod:`app.biomech.angles`) and displays them as time-series
in a pyqtgraph plot.  A vertical cursor on the plot is kept in lockstep
with the Reconstruction tab's frame slider — moving the slider moves the
cursor, and clicking on the plot scrubs the 3D viewer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.biomech import (
    ANGLE_DEFINITIONS,
    angles_to_csv,
    compute_joint_angles,
    compute_stats,
)


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

# Curve colours indexed identically to ANGLE_DEFINITIONS.  Chosen for
# colour-blind accessibility — blue/cyan for left limbs, red/orange for
# right limbs, green/purple/teal/gold for the upper-body angles, grey
# for the centred trunk inclination.
_CURVE_COLORS: list[tuple[int, int, int]] = [
    (100, 149, 237),  # L Knee Flex          — cornflower blue
    (220, 80, 80),    # R Knee Flex          — coral red
    (64, 196, 255),   # L Hip Flex           — cyan
    (255, 128, 64),   # R Hip Flex           — orange
    (100, 220, 100),  # L Elbow Flex         — green
    (180, 100, 220),  # R Elbow Flex         — purple
    (40, 180, 160),   # L Shoulder Flex      — teal
    (220, 180, 40),   # R Shoulder Flex      — gold
    (200, 200, 200),  # Trunk Inclination    — light grey
]


def _opencv_to_zup(joints_cv: np.ndarray) -> np.ndarray:
    """Convert pose3d.npz coordinates (OpenCV camera frame) to viewer Z-up frame.

    Matches the transform inside :meth:`SkeletonViewer3D.set_data`.
    """
    return np.stack(
        [joints_cv[..., 0], joints_cv[..., 2], -joints_cv[..., 1]], axis=-1
    ).astype(np.float32)


class AnalysisTab(QWidget):
    """Biomechanical analysis tab — joint angles, statistics, CSV export.

    Signals:
        frame_seek(int): emitted when the user clicks on the plot;
            the receiving Reconstruction tab moves its slider to this frame.
    """

    frame_seek = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pose3d_path: str | None = None
        self._angles: np.ndarray | None = None  # [T, P, N_ANGLES]
        self._fps: float = 30.0
        self._n_frames: int = 0
        self._n_persons: int = 0
        # Avoid the cursor-update → emit → slider → cursor-update loop.
        self._suppress_seek_signal: bool = False

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)
        col = QVBoxLayout(panel)

        # ── Angle selector ─────────────────────────────────────────
        sel_group = QGroupBox("Angles")
        sel_layout = QVBoxLayout(sel_group)
        self._angle_checks: list[QCheckBox] = []
        for k, adef in enumerate(ANGLE_DEFINITIONS):
            cb = QCheckBox(adef.name)
            cb.setChecked(True)
            cb.toggled.connect(self._on_angle_toggled)
            # Coloured swatch in the checkbox text to match the curve.
            r, g, b = _CURVE_COLORS[k]
            cb.setStyleSheet(f"QCheckBox {{ color: rgb({r},{g},{b}); }}")
            sel_layout.addWidget(cb)
            self._angle_checks.append(cb)
        col.addWidget(sel_group)

        # ── Person selector ────────────────────────────────────────
        person_row = QHBoxLayout()
        person_row.addWidget(QLabel("Person:"))
        self._person_combo = QComboBox()
        self._person_combo.addItem("0")
        self._person_combo.currentIndexChanged.connect(self._on_person_changed)
        person_row.addWidget(self._person_combo, 1)
        col.addLayout(person_row)

        # ── Statistics table ───────────────────────────────────────
        stats_group = QGroupBox("Statistics (degrees)")
        stats_layout = QVBoxLayout(stats_group)
        self._stats_table = QTableWidget(len(ANGLE_DEFINITIONS), 4)
        self._stats_table.setHorizontalHeaderLabels(["Min", "Max", "Mean", "ROM"])
        self._stats_table.setVerticalHeaderLabels(
            [a.name for a in ANGLE_DEFINITIONS]
        )
        self._stats_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        stats_layout.addWidget(self._stats_table)
        col.addWidget(stats_group, 1)

        # ── Buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load pose3d.npz…")
        load_btn.clicked.connect(self._on_browse_pose3d)
        export_btn = QPushButton("Export CSV…")
        export_btn.clicked.connect(self._on_export_csv)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(export_btn)
        col.addLayout(btn_row)

        # ── Status line ────────────────────────────────────────────
        self._status_label = QLabel("(no pose3d loaded)")
        self._status_label.setStyleSheet("color: #888; font-size: 10px;")
        col.addWidget(self._status_label)

        return panel

    def _build_right_panel(self) -> QWidget:
        # Dark-background plot to match the rest of the app theme.
        pg.setConfigOption("background", "#1e1e1e")
        pg.setConfigOption("foreground", "#cccccc")

        self._plot = pg.PlotWidget()
        self._plot.setLabel("left", "Angle (degrees)")
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        # Pin the Y range to a useful default — autoscale would otherwise
        # zoom in on whatever the first loaded sequence happens to span.
        self._plot.setYRange(0, 200)
        legend = self._plot.addLegend(offset=(-10, 10))
        legend.setBrush((30, 30, 30, 200))

        # Pre-allocate one curve per angle.  Curves are hidden when their
        # checkbox is unchecked rather than removed — keeps the legend
        # stable and avoids re-allocating GL items on every toggle.
        self._curves: list[pg.PlotDataItem] = []
        for k, adef in enumerate(ANGLE_DEFINITIONS):
            r, g, b = _CURVE_COLORS[k]
            pen = pg.mkPen(color=QColor(r, g, b), width=1.5)
            curve = self._plot.plot(name=adef.name, pen=pen)
            self._curves.append(curve)

        # Vertical cursor — synchronised with the Reconstruction slider.
        self._cursor = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color="w", width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot.addItem(self._cursor)

        # Click-to-seek.  pyqtgraph's MouseClickEvent gives us the scene
        # position; we map it back to data coordinates via the view box.
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        return self._plot

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    def load_pose3d(self, path: str) -> None:
        """Load a pose3d.npz file and refresh the plot + stats.

        Safe to call from a signal — gracefully handles missing files and
        malformed NPZ contents by setting the status label rather than
        raising.
        """
        if not path or not Path(path).is_file():
            self._status_label.setText(f"(missing file: {path})")
            return

        try:
            d = np.load(path, allow_pickle=True)
            joints_cv = d["joints3d"]  # [T, P, 17, 3] in OpenCV frame
            conf3d = d["conf3d"]       # [T, P, 17]
            meta = d["meta"].item()
        except Exception as exc:  # noqa: BLE001 — any load failure is a UI error
            self._status_label.setText(f"Failed to load {Path(path).name}: {exc}")
            return

        self._pose3d_path = path
        self._fps = float(meta.get("fps", 30.0))
        joints_zup = _opencv_to_zup(joints_cv)
        self._angles = compute_joint_angles(joints_zup, conf3d)
        self._n_frames, self._n_persons = self._angles.shape[:2]

        # Refresh the person combo without firing currentIndexChanged.
        self._person_combo.blockSignals(True)
        self._person_combo.clear()
        for p in range(self._n_persons):
            self._person_combo.addItem(str(p))
        self._person_combo.blockSignals(False)

        self._redraw_curves()
        self._refresh_stats_table()

        self._status_label.setText(
            f"{Path(path).name} — {self._n_frames} frames, "
            f"{self._n_persons} person(s), {self._fps:.1f} fps"
        )

    def seek_to_frame(self, frame: int) -> None:
        """Move the cursor line to *frame*.  Does NOT re-emit frame_seek."""
        if self._angles is None or self._fps <= 0:
            return
        frame = max(0, min(int(frame), self._n_frames - 1))
        # Suppress the click-handler side effects since this is a programmatic
        # move triggered by the Reconstruction slider.
        self._suppress_seek_signal = True
        try:
            self._cursor.setValue(frame / self._fps)
        finally:
            self._suppress_seek_signal = False

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_angle_toggled(self, _checked: bool) -> None:
        self._redraw_curves()
        self._refresh_stats_table()

    def _on_person_changed(self, _idx: int) -> None:
        self._redraw_curves()
        self._refresh_stats_table()

    def _on_browse_pose3d(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load pose3d.npz", "", "NumPy (*.npz)"
        )
        if path:
            self.load_pose3d(path)

    def _on_export_csv(self) -> None:
        if self._angles is None:
            self._status_label.setText("(no data to export)")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export angles CSV", "angles.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            angles_to_csv(path, self._angles, fps=self._fps)
        except Exception as exc:  # noqa: BLE001 — surfaced via the status line
            self._status_label.setText(f"Export failed: {exc}")
            return
        self._status_label.setText(f"Exported {Path(path).name}")

    def _on_plot_clicked(self, event) -> None:
        """Translate a left-click in the plot scene into a frame_seek emit."""
        if (
            self._angles is None
            or self._suppress_seek_signal
            or event.button() != Qt.MouseButton.LeftButton
        ):
            return
        vb = self._plot.getViewBox()
        if vb is None:
            return
        # Map scene click to data coords.
        scene_pos = event.scenePos()
        view_pos = vb.mapSceneToView(scene_pos)
        t_sec = float(view_pos.x())
        if t_sec < 0:
            return
        frame = int(round(t_sec * self._fps))
        frame = max(0, min(frame, self._n_frames - 1))
        self.frame_seek.emit(frame)

    # ------------------------------------------------------------------
    # Plot / stats refresh helpers
    # ------------------------------------------------------------------

    def _selected_person(self) -> int:
        idx = self._person_combo.currentIndex()
        return max(0, idx)

    def _redraw_curves(self) -> None:
        if self._angles is None or self._n_frames == 0:
            for c in self._curves:
                c.setData([], [])
            return
        p = self._selected_person()
        t_axis = np.arange(self._n_frames, dtype=np.float32) / self._fps
        for k, curve in enumerate(self._curves):
            visible = self._angle_checks[k].isChecked()
            if not visible:
                curve.setData([], [])
                continue
            series = self._angles[:, p, k]
            # Mask NaN: pyqtgraph supports `connect="finite"` to skip them
            # in the rendered line (otherwise the line crosses gaps).
            curve.setData(
                t_axis,
                series,
                connect="finite",
            )

    def _refresh_stats_table(self) -> None:
        if self._angles is None:
            for r in range(len(ANGLE_DEFINITIONS)):
                for c in range(4):
                    self._stats_table.setItem(r, c, QTableWidgetItem("—"))
            return
        p = self._selected_person()
        for k in range(len(ANGLE_DEFINITIONS)):
            stats = compute_stats(self._angles, angle_idx=k, person_idx=p)
            if stats is None:
                cells = ["—"] * 4
            else:
                cells = [
                    f"{stats.min_deg:.1f}",
                    f"{stats.max_deg:.1f}",
                    f"{stats.mean_deg:.1f}",
                    f"{stats.rom_deg:.1f}",
                ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                # Grey out rows whose checkbox is off (so the table mirrors
                # the visible curves at a glance).
                if not self._angle_checks[k].isChecked():
                    item.setForeground(QColor(120, 120, 120))
                self._stats_table.setItem(k, c, item)
