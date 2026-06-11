"""
Analysis tab — biomechanical joint-angle inspection, segmentation, and export.

Layout (right panel)
    ┌──────────────────────────────────────────────────────────────┐
    │  Left 2D preview  │  Right 2D preview  │  3D viewer         │  ← top row
    ├──────────────────────────────────────────────────────────────┤
    │                    Angle time-series plot                    │  ← middle (stretch)
    ├──────────────────────────────────────────────────────────────┤
    │   frame label  ←  ‖ play controls ‖  →   speed   fps        │  ← playback (centred)
    └──────────────────────────────────────────────────────────────┘

Left panel
    ├─ Angles checkboxes
    ├─ Person selector
    ├─ Stats tabs: Recording | Segment | Asymmetry  (stretch to fill)
    └─ Region of Interest group

Mouse interaction on plot
    • Left drag on background  → define/resize the selection region
    • Ctrl + left drag         → pan the view (classic pyqtgraph behaviour)
    • Scroll wheel             → zoom (unchanged)
    • Hover on region border   → ↔ cursor; hover on region body → ✥ cursor

Research-backed metrics (ADR-008)
    Recording: Min, Max, Mean, SD, CV%, ROM, NaN%
    Segment  : Min, Max, Mean, SD, CV%, ROM, Excursion(°)
    Asymmetry: SI for Mean, ROM, Peak-Velocity (Robinson 1987) per L/R pair
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QElapsedTimer, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.biomech import (
    ANGLE_DEFINITIONS,
    ANGLE_PAIRS,
    angles_to_csv,
    compute_extended_stats,
    compute_joint_angles,
    compute_symmetry_index,
)
from app.gui.pose2d_preview import Pose2DPreview, derive_pose2d_paths
from app.gui.viewer3d import SkeletonViewer3D

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

_CURVE_COLORS: list[tuple[int, int, int]] = [
    (100, 149, 237),  # L Knee Flex   — cornflower blue
    (220, 80, 80),  # R Knee Flex   — coral red
    (64, 196, 255),  # L Hip Flex    — cyan
    (255, 128, 64),  # R Hip Flex    — orange
    (100, 220, 100),  # L Elbow Flex  — green
    (180, 100, 220),  # R Elbow Flex  — purple
    (40, 180, 160),  # L Shoulder    — teal
    (220, 180, 40),  # R Shoulder    — gold
    (200, 200, 200),  # Trunk         — grey
]

_PLAYBACK_SPEEDS = [0.25, 0.5, 1.0, 2.0]

# Max frames a single playback tick may advance — caps catch-up surges after a
# transient render lag (e.g. GL warm-up) so playback never visibly sprints.
_MAX_PLAY_STEP = 4

_TABLE_STYLE = (
    "QTableWidget { font-size: 10px; }QHeaderView::section { font-size: 10px; padding: 2px; }"
)

# Tooltip text for each column header — shown on mouse hover.
# Shared across all three stats tables; keys match the header strings exactly.
_HEADER_TOOLTIPS: dict[str, str] = {
    "Min": "Minimum angle (°)\nLowest value recorded in the selected range.",
    "Max": "Maximum angle (°)\nHighest value recorded in the selected range.",
    "Mean": "Mean angle (°)\nAverage over all detected (non-NaN) frames.",
    "SD": "Standard Deviation (°)\n"
    "Spread around the mean.\n"
    "Higher SD = more variable / less consistent movement.",
    "CV%": "Coefficient of Variation (%)\n"
    "= SD / |Mean| × 100\n"
    "Normalises variability so joints with different baseline\n"
    "angles (e.g. 5° trunk vs 90° knee) are comparable.",
    "ROM": "Range of Motion (°)\n= Max − Min\nTotal arc covered during the recording or segment.",
    "NaN%": "Missing data (%)\n"
    "Percentage of frames where one or more flanking joints\n"
    "were not detected. Values > 20 % should be treated\n"
    "with caution.",
    "Exc(°)": "Total Angular Excursion (°)\n"
    "= Σ |θ[t+1] − θ[t]| over all consecutive frame pairs\n"
    "Total path length traveled by the joint angle.\n"
    "Larger than ROM when the joint oscillates back and forth.",
    "Mean SI%": "Symmetry Index for Mean angle  (Robinson 1987)\n"
    "= 100 × (Left − Right) / (0.5 × (|Left| + |Right|))\n"
    "Positive = left-dominant, negative = right-dominant.\n"
    "Clinical threshold: |SI| > 10 % is considered asymmetric.",
    "ROM SI%": "Symmetry Index for Range of Motion\n"
    "Asymmetry in how much each joint moves.\n"
    "Return-to-sport criterion: Limb Symmetry Index ≥ 90 %\n"
    "(equivalent to |SI| ≤ ~11 %).",
    "PkVel SI%": "Symmetry Index for Peak Angular Velocity  (deg/s)\n"
    "Asymmetry in maximum movement speed.\n"
    "Relevant for power-based tasks such as jumping, kicking,\n"
    "or throwing.",
}


# ---------------------------------------------------------------------------
# Custom ViewBox: left-drag = selection; Ctrl+left-drag = pan
# ---------------------------------------------------------------------------


class _SelectionViewBox(pg.ViewBox):
    """ViewBox that replaces left-drag panning with region selection.

    Default: left-drag on empty plot area creates / resizes the attached
    :class:`pg.LinearRegionItem`.

    CTRL + left-drag: pan the view exactly as pyqtgraph's default PanMode.
    Right-drag / scroll-wheel: zoom (unchanged).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._roi: pg.LinearRegionItem | None = None
        self._drag_anchor_x: float | None = None

    def attach_roi(self, roi: pg.LinearRegionItem) -> None:
        self._roi = roi

    def mouseDragEvent(self, ev, axis=None):  # noqa: N802
        ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl or ev.button() != Qt.MouseButton.LeftButton or self._roi is None:
            # Fall back to normal pyqtgraph pan behaviour.
            super().mouseDragEvent(ev, axis=axis)
            return

        ev.accept()
        x = float(self.mapToView(ev.pos()).x())

        if ev.isStart():
            self._drag_anchor_x = x
            self._roi.setRegion([x, x])
            self._roi.setVisible(True)
        else:
            anchor = self._drag_anchor_x if self._drag_anchor_x is not None else x
            lo, hi = sorted([anchor, x])
            if hi > lo:
                self._roi.setRegion([lo, hi])

        if ev.isFinish():
            self._drag_anchor_x = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opencv_to_zup(joints_cv: np.ndarray) -> np.ndarray:
    return np.stack([joints_cv[..., 0], joints_cv[..., 2], -joints_cv[..., 1]], axis=-1).astype(
        np.float32
    )


def _fmt(value: float | None, decimals: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------


class AnalysisTab(QWidget):
    """
    Biomechanical analysis tab — joint angles, statistics, segmentation, export.

    Public signals
        frame_seek(int): emitted when the user scrubs locally (plot drag or
            slider). MainWindow forwards this to the Reconstruction tab so
            both views stay in sync.

    Public slots
        load_pose3d(path)       — load a pose3d.npz
        set_video_paths(l, r)   — assign source videos for the 2D preview
        seek_to_frame(frame)    — called by the Reconstruction slider; moves
                                  the local slider WITHOUT re-emitting frame_seek
    """

    frame_seek = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Data ────────────────────────────────────────────────────────────────
        self._pose3d_path: str | None = None
        self._joints_zup: np.ndarray | None = None
        self._conf3d: np.ndarray | None = None
        self._angles: np.ndarray | None = None
        self._fps: float = 30.0
        self._n_frames: int = 0
        self._n_persons: int = 0

        self._video_left: str = ""
        self._video_right: str = ""

        # Playback ────────────────────────────────────────────────────────────
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_timer_tick)
        # Wall-clock playback: fixed timer rate, each tick advances by the
        # number of frames real elapsed time covers (frames dropped when a
        # tick runs long), so a 50 fps clip plays at true speed instead of
        # slowing to the render rate.
        self._play_timer.setInterval(16)
        self._play_clock = QElapsedTimer()
        self._play_frac: float = 0.0
        self._suppress_seek_signal: bool = False

        # Auto-evaluate: fires 150 ms after the ROI region stops changing.
        # Using a single-shot timer so rapid drag events collapse into one
        # computation rather than re-filling the table on every pixel moved.
        self._eval_timer = QTimer(self)
        self._eval_timer.setSingleShot(True)
        self._eval_timer.setInterval(150)
        self._eval_timer.timeout.connect(self._on_evaluate_segment)

        # Previous ROI boundaries — used in _on_roi_region_changed to detect
        # which endpoint moved so the preview can seek to the right frame.
        self._prev_roi_t0: float = 0.0
        self._prev_roi_t1: float = 0.0

        self._setup_ui()

    # =========================================================================
    # Public API
    # =========================================================================

    def load_pose3d(self, path: str) -> None:
        if not path or not Path(path).is_file():
            self._set_status(f"(file not found: {path})")
            return
        try:
            d = np.load(path, allow_pickle=True)
            joints_cv = d["joints3d"]
            conf3d = d["conf3d"]
            meta = d["meta"].item()
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")
            return

        self._pose3d_path = path
        self._fps = float(meta.get("fps", 30.0))
        # When the pipeline already expressed joints in a Z-up floor world frame
        # ("Set coordinate system"), use them as-is; otherwise apply the
        # OpenCV→Z-up swap (angles are always computed in a Z-up frame).
        if meta.get("coordinate_frame") == "world":
            self._joints_zup = np.asarray(joints_cv, dtype=np.float32)
        else:
            self._joints_zup = _opencv_to_zup(joints_cv)
        self._conf3d = conf3d
        self._n_frames, self._n_persons = self._joints_zup.shape[:2]
        self._angles = compute_joint_angles(
            self._joints_zup, conf3d, min_conf=self._conf_spin.value()
        )

        # Person combo
        self._person_combo.blockSignals(True)
        self._person_combo.clear()
        for p in range(self._n_persons):
            self._person_combo.addItem(str(p))
        self._person_combo.blockSignals(False)

        # 3D viewer — world-frame data is already Z-up; the viewer must skip its
        # OpenCV→Z-up swap or the skeleton is rotated a second time.
        self._viewer.set_data(
            joints_cv, conf3d, world_frame=meta.get("coordinate_frame") == "world"
        )

        # Playback slider
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(0, self._n_frames - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._frame_label.setText(self._frame_label_text(0))
        self._fps_label.setText(f"{self._fps:.1f} fps")
        self._on_speed_changed()

        # 2D preview
        p2d_l, p2d_r = derive_pose2d_paths(path)
        self._refresh_2d_preview(p2d_l, p2d_r)

        self._redraw_curves()
        self._refresh_full_stats()
        self._refresh_asymmetry_tab()
        self._clear_segment_stats()
        self._roi_region.setVisible(False)

        self._set_status(
            f"{Path(path).name} — {self._n_frames} frames, "
            f"{self._n_persons} person(s), {self._fps:.1f} fps"
        )

    def set_video_paths(self, left: str, right: str) -> None:
        self._video_left = left
        self._video_right = right
        if self._pose3d_path:
            p2d_l, p2d_r = derive_pose2d_paths(self._pose3d_path)
            self._refresh_2d_preview(p2d_l, p2d_r)

    def seek_to_frame(self, frame: int) -> None:
        if self._n_frames == 0:
            return
        frame = max(0, min(int(frame), self._n_frames - 1))
        self._suppress_seek_signal = True
        try:
            self._slider.setValue(frame)
        finally:
            self._suppress_seek_signal = False

    # =========================================================================
    # UI construction
    # =========================================================================

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(370)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ── Angles checkboxes ─────────────────────────────────────────
        from PySide6.QtWidgets import QCheckBox

        sel_group = QGroupBox("Angles")
        sel_layout = QVBoxLayout(sel_group)
        sel_layout.setSpacing(2)
        self._angle_checks: list[QCheckBox] = []
        for k, adef in enumerate(ANGLE_DEFINITIONS):
            cb = QCheckBox(adef.name)
            cb.setChecked(True)
            cb.toggled.connect(self._on_angle_toggled)
            r, g, b = _CURVE_COLORS[k]
            cb.setStyleSheet(f"QCheckBox {{ color: rgb({r},{g},{b}); font-size: 11px; }}")
            sel_layout.addWidget(cb)
            self._angle_checks.append(cb)
        outer.addWidget(sel_group)

        # ── Person selector ───────────────────────────────────────────
        p_row = QHBoxLayout()
        p_row.addWidget(QLabel("Person:"))
        self._person_combo = QComboBox()
        self._person_combo.addItem("0")
        self._person_combo.currentIndexChanged.connect(self._on_person_changed)
        p_row.addWidget(self._person_combo, 1)
        outer.addLayout(p_row)

        # ── Confidence threshold ──────────────────────────────────────
        # Keypoints below this confidence are treated as not-detected, so any
        # angle that relies on one becomes NaN (a gap in the plot).  Lets the
        # user trade coverage for reliability without re-running the pipeline.
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Min confidence:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.0, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.0)
        self._conf_spin.setToolTip(
            "Discard keypoints with stored confidence below this value.\n"
            "Angles using a discarded joint become NaN (a gap in the curve)\n"
            "and are excluded from the statistics. 0.00 keeps every detected\n"
            "keypoint."
        )
        self._conf_spin.valueChanged.connect(self._on_conf_threshold_changed)
        conf_row.addWidget(self._conf_spin, 1)
        outer.addLayout(conf_row)

        # ── Statistics tabs (stretch=1 so they fill available height) ─
        self._stats_tabs = QTabWidget()
        self._stats_tabs.setDocumentMode(True)
        self._stats_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer.addWidget(self._stats_tabs, 1)

        # Tab 1: Full recording
        self._full_table = self._make_stats_table(
            ["Min", "Max", "Mean", "SD", "CV%", "ROM", "NaN%"],
            len(ANGLE_DEFINITIONS),
            [a.name for a in ANGLE_DEFINITIONS],
        )
        tab1 = QWidget()
        t1l = QVBoxLayout(tab1)
        t1l.setContentsMargins(0, 0, 0, 0)
        t1l.addWidget(self._full_table)
        self._stats_tabs.addTab(tab1, "Recording")

        # Tab 2: Segment
        self._seg_table = self._make_stats_table(
            ["Min", "Max", "Mean", "SD", "CV%", "ROM", "Exc(°)"],
            len(ANGLE_DEFINITIONS),
            [a.name for a in ANGLE_DEFINITIONS],
        )
        tab2 = QWidget()
        t2l = QVBoxLayout(tab2)
        t2l.setContentsMargins(0, 0, 0, 0)
        t2l.addWidget(self._seg_table)
        self._stats_tabs.addTab(tab2, "Segment")

        # Tab 3: Asymmetry
        asym_row_names = [name for *_, name in ANGLE_PAIRS]
        self._asym_table = self._make_stats_table(
            ["Mean SI%", "ROM SI%", "PkVel SI%"],
            len(ANGLE_PAIRS),
            asym_row_names,
        )
        tab3 = QWidget()
        t3l = QVBoxLayout(tab3)
        t3l.setContentsMargins(0, 0, 0, 0)
        t3l.addWidget(self._asym_table)
        self._stats_tabs.addTab(tab3, "Asymmetry")

        # ── Region of Interest ────────────────────────────────────────
        roi_group = QGroupBox("Region of Interest")
        roi_layout = QVBoxLayout(roi_group)
        roi_layout.setSpacing(4)

        hint = QLabel("Drag on plot to select · CTRL+drag to pan · drag edge to resize")
        hint.setStyleSheet("color: #888; font-size: 9px;")
        hint.setWordWrap(True)
        roi_layout.addWidget(hint)

        info_row = QHBoxLayout()
        self._roi_start_lbl = QLabel("Start: —")
        self._roi_end_lbl = QLabel("End: —")
        self._roi_dur_lbl = QLabel("Dur: —")
        for lbl in (self._roi_start_lbl, self._roi_end_lbl, self._roi_dur_lbl):
            lbl.setStyleSheet("font-size: 10px;")
        info_row.addWidget(self._roi_start_lbl)
        info_row.addWidget(self._roi_end_lbl)
        info_row.addWidget(self._roi_dur_lbl)
        roi_layout.addLayout(info_row)

        lbl_row = QHBoxLayout()
        lbl_row.addWidget(QLabel("Label:"))
        self._roi_label_edit = QLineEdit()
        self._roi_label_edit.setPlaceholderText("Annotation…")
        lbl_row.addWidget(self._roi_label_edit, 1)
        roi_layout.addLayout(lbl_row)

        roi_btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Remove the selection region")
        clear_btn.clicked.connect(self._on_clear_roi)
        save_btn = QPushButton("Save CSV…")
        save_btn.setToolTip("Export cropped angle data for the selected region")
        save_btn.clicked.connect(self._on_save_segment_csv)
        roi_btn_row.addWidget(clear_btn)
        roi_btn_row.addWidget(save_btn)
        roi_layout.addLayout(roi_btn_row)

        outer.addWidget(roi_group)

        # ── I/O buttons + status ──────────────────────────────────────
        io_row = QHBoxLayout()
        load_btn = QPushButton("Load pose3d.npz…")
        load_btn.clicked.connect(self._on_browse_pose3d)
        export_btn = QPushButton("Export CSV…")
        export_btn.setToolTip("Export full angle time series to CSV")
        export_btn.clicked.connect(self._on_export_full_csv)
        io_row.addWidget(load_btn)
        io_row.addWidget(export_btn)
        outer.addLayout(io_row)

        self._status_lbl = QLabel("(no pose3d loaded)")
        self._status_lbl.setStyleSheet("color: #888; font-size: 10px;")
        self._status_lbl.setWordWrap(True)
        outer.addWidget(self._status_lbl)

        return panel

    @staticmethod
    def _make_stats_table(
        col_headers: list[str],
        n_rows: int,
        row_headers: list[str],
    ) -> QTableWidget:
        t = QTableWidget(n_rows, len(col_headers))
        # Set header items individually so we can attach per-column tooltips.
        for c, label in enumerate(col_headers):
            item = QTableWidgetItem(label)
            tip = _HEADER_TOOLTIPS.get(label)
            if tip:
                item.setToolTip(tip)
            t.setHorizontalHeaderItem(c, item)
        t.setVerticalHeaderLabels(row_headers)
        # Distribute all columns equally across the available width.
        # setMinimumSectionSize prevents any column from collapsing below
        # the width of a 5-digit value at the table's font size.
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hh.setMinimumSectionSize(36)
        t.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setStyleSheet(_TABLE_STYLE)
        t.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        for r in range(n_rows):
            for c in range(len(col_headers)):
                item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, c, item)
        return t

    # ------------------------------------------------------------------
    # Right panel
    # ------------------------------------------------------------------

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # ── Top row: 2D previews + 3D viewer ─────────────────────────
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._preview_2d = Pose2DPreview()
        top_splitter.addWidget(self._preview_2d)

        self._viewer = SkeletonViewer3D()
        top_splitter.addWidget(self._viewer)
        # Give the 3D viewer equal share with the 2 camera panels (640+640 vs viewer)
        top_splitter.setStretchFactor(0, 2)
        top_splitter.setStretchFactor(1, 1)

        layout.addWidget(top_splitter)

        # ── Angle plot (fills remaining vertical space) ───────────────
        layout.addWidget(self._build_plot(), 1)

        # ── Playback controls ─────────────────────────────────────────
        layout.addWidget(self._build_playback_group())

        return panel

    def _build_plot(self) -> pg.PlotWidget:
        pg.setConfigOption("background", "#1e1e1e")
        pg.setConfigOption("foreground", "#cccccc")

        vb = _SelectionViewBox()
        pw = pg.PlotWidget(viewBox=vb)
        self._view_box = vb

        pw.setLabel("left", "Angle (°)")
        pw.setLabel("bottom", "Time (s)")
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setYRange(0, 200)
        legend = pw.addLegend(offset=(-10, 10))
        legend.setBrush((30, 30, 30, 200))

        # One curve per angle
        self._curves: list[pg.PlotDataItem] = []
        for k, adef in enumerate(ANGLE_DEFINITIONS):
            r, g, b = _CURVE_COLORS[k]
            pen = pg.mkPen(color=QColor(r, g, b), width=1.5)
            curve = pw.plot(name=adef.name, pen=pen)
            self._curves.append(curve)

        # Frame cursor
        self._cursor = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color="w", width=1, style=Qt.PenStyle.DashLine),
        )
        pw.addItem(self._cursor)

        # Selection region (hidden until the user drags)
        self._roi_region = pg.LinearRegionItem(
            [0, 1],
            brush=pg.mkBrush(255, 255, 100, 40),
            pen=pg.mkPen("y", width=1),
            movable=True,
        )
        self._roi_region.setVisible(False)
        self._roi_region.sigRegionChanged.connect(self._on_roi_region_changed)
        pw.addItem(self._roi_region)

        # Cursor shapes on the region's edge lines (↔) and body (✥)
        self._roi_region.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        for line in self._roi_region.lines:
            line.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))

        # Attach region to the ViewBox so left-drag creates/resizes it
        vb.attach_roi(self._roi_region)

        self._plot_widget = pw
        return pw

    def _build_playback_group(self) -> QWidget:
        group = QGroupBox("Playback")
        layout = QVBoxLayout(group)
        layout.setSpacing(2)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)

        row = QHBoxLayout()
        _si = QApplication.style().standardIcon
        SP = QStyle.StandardPixmap

        # Frame label on the left
        self._frame_label = QLabel("Frame: 0 / 0")
        row.addWidget(self._frame_label)

        # Centred transport controls (stretch on both sides)
        row.addStretch()

        def _btn(icon_enum, tip, callback, w=32) -> QPushButton:
            b = QPushButton()
            b.setIcon(_si(icon_enum))
            b.setToolTip(tip)
            b.setFixedWidth(w)
            b.clicked.connect(callback)
            return b

        row.addWidget(_btn(SP.SP_MediaSeekBackward, "−1 s", self._on_seek_back))
        row.addWidget(_btn(SP.SP_MediaSkipBackward, "Prev frame", self._on_prev_frame))

        self._play_btn = QPushButton()
        self._play_btn.setIcon(_si(SP.SP_MediaPlay))
        self._play_btn.setToolTip("Play / Pause")
        self._play_btn.setFixedWidth(32)
        self._play_btn.setCheckable(True)
        self._play_btn.toggled.connect(self._on_play_toggle)
        row.addWidget(self._play_btn)

        row.addWidget(_btn(SP.SP_MediaSkipForward, "Next frame", self._on_next_frame))
        row.addWidget(_btn(SP.SP_MediaSeekForward, "+1 s", self._on_seek_fwd))

        row.addStretch()

        # Speed + fps on the right
        row.addWidget(QLabel("Speed:"))
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25×", "0.5×", "1×", "2×"])
        self._speed_combo.setCurrentIndex(2)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        row.addWidget(self._speed_combo)

        self._fps_label = QLabel("")
        row.addWidget(self._fps_label)

        layout.addLayout(row)
        return group

    # =========================================================================
    # Playback slots
    # =========================================================================

    def _on_slider_changed(self, value: int) -> None:
        frame = value
        # Render only when this tab is visible (the Reconstruction slider drives
        # this one via seek_to_frame while it plays; the guard stops both tabs
        # decoding video per tick).  showEvent() catches up on tab switch.
        if self.isVisible():
            self._viewer.show_frame(frame)  # cheap GL update — every frame
            # Non-blocking: the preview decodes on a background thread and
            # coalesces to the latest requested frame, so driving it every
            # frame during playback does not stall the 3D view.
            self._preview_2d.show_frame(frame)
        if self._fps > 0:
            self._cursor.setValue(frame / self._fps)
        self._frame_label.setText(self._frame_label_text(frame))
        if not self._suppress_seek_signal:
            self.frame_seek.emit(frame)

    def showEvent(self, event) -> None:  # noqa: N802
        """Sync the viewer and preview the moment this tab becomes visible."""
        super().showEvent(event)
        if self._n_frames > 0:
            frame = self._slider.value()
            self._viewer.show_frame(frame)
            self._preview_2d.show_frame(frame)

    def _on_play_toggle(self, checked: bool) -> None:
        _si = QApplication.style().standardIcon
        SP = QStyle.StandardPixmap
        if checked:
            self._play_btn.setIcon(_si(SP.SP_MediaPause))
            self._play_frac = 0.0
            self._play_clock.start()  # anchor the wall clock
            self._play_timer.start()
        else:
            self._play_btn.setIcon(_si(SP.SP_MediaPlay))
            self._play_timer.stop()

    def sync_play_state(self, playing: bool) -> None:
        """Mirror the play/pause state from the other tab.

        Updates the button icon without starting this tab's own timer —
        only one tab's timer runs at a time.  Stops this timer when
        *playing* is False so stale timers never survive a tab switch.
        """
        if not playing:
            self._play_timer.stop()
        _si = QApplication.style().standardIcon
        SP = QStyle.StandardPixmap
        self._play_btn.blockSignals(True)
        self._play_btn.setChecked(playing)
        self._play_btn.setIcon(_si(SP.SP_MediaPause) if playing else _si(SP.SP_MediaPlay))
        self._play_btn.blockSignals(False)

    def _on_timer_tick(self) -> None:
        if self._n_frames == 0:
            self._play_timer.stop()
            self._play_btn.setChecked(False)
            return
        self._advance_by_elapsed(self._play_clock.restart())

    def _advance_by_elapsed(self, elapsed_ms: float) -> None:
        """Advance the slider by the whole frames *elapsed_ms* of real time covers.

        Pure of any wall-clock reading (the caller supplies elapsed_ms), so it
        is deterministic and unit-testable.  Fractional frames accumulate in
        ``_play_frac``; when a tick runs long the larger elapsed value advances
        multiple frames, dropping the intermediate ones to hold real-time speed.
        """
        if self._n_frames == 0:
            return
        speed = _PLAYBACK_SPEEDS[self._speed_combo.currentIndex()]
        fps = self._fps if self._fps > 0 else 30.0
        self._play_frac += elapsed_ms / 1000.0 * fps * speed
        step = int(self._play_frac)
        if step <= 0:
            return
        self._play_frac -= step
        # Cap the jump so a transient lag (e.g. GL warm-up on the first frames)
        # doesn't trigger a visible catch-up surge; drop the backlog instead of
        # sprinting through frames.
        if step > _MAX_PLAY_STEP:
            step = _MAX_PLAY_STEP
            self._play_frac = 0.0
        self._slider.setValue((self._slider.value() + step) % self._n_frames)

    def _on_prev_frame(self) -> None:
        self._slider.setValue(max(0, self._slider.value() - 1))

    def _on_next_frame(self) -> None:
        self._slider.setValue(min(self._slider.maximum(), self._slider.value() + 1))

    def _on_seek_back(self) -> None:
        step = max(1, int(self._fps))
        self._slider.setValue(max(0, self._slider.value() - step))

    def _on_seek_fwd(self) -> None:
        step = max(1, int(self._fps))
        self._slider.setValue(min(self._slider.maximum(), self._slider.value() + step))

    def _on_speed_changed(self, _idx: int = 0) -> None:
        """Re-anchor the wall clock so a mid-playback speed change is seamless.

        The timer interval is fixed; speed is applied per-tick from elapsed
        wall-clock time, so changing speed only resets the fractional
        accumulator from the current position.
        """
        self._play_frac = 0.0
        if self._play_timer.isActive():
            self._play_clock.restart()

    def _frame_label_text(self, frame: int) -> str:
        fps = self._fps if self._fps > 0 else 30.0
        t = frame / fps
        m, s = divmod(int(t), 60)
        total = max(0, self._n_frames - 1) / fps
        dm, ds = divmod(int(total), 60)
        return (
            f"Frame: {frame} / {max(0, self._n_frames - 1)}   {m:02d}:{s:02d} / {dm:02d}:{ds:02d}"
        )

    # =========================================================================
    # Angle display slots
    # =========================================================================

    def _on_angle_toggled(self, _checked: bool) -> None:
        self._redraw_curves()
        self._refresh_full_stats()

    def _on_person_changed(self, _idx: int) -> None:
        self._redraw_curves()
        self._refresh_full_stats()
        self._refresh_asymmetry_tab()
        self._clear_segment_stats()

    def _on_conf_threshold_changed(self, _value: float) -> None:
        """Recompute angles at the new confidence threshold and refresh all views."""
        if self._joints_zup is None or self._conf3d is None:
            return
        self._angles = compute_joint_angles(
            self._joints_zup, self._conf3d, min_conf=self._conf_spin.value()
        )
        self._redraw_curves()
        self._refresh_full_stats()
        self._refresh_asymmetry_tab()
        # Re-evaluate the active segment so its stats reflect the new threshold.
        if self._roi_region.isVisible():
            self._on_evaluate_segment()
        else:
            self._clear_segment_stats()

    # =========================================================================
    # ROI / segment slots
    # =========================================================================

    def _on_roi_region_changed(self) -> None:
        t0, t1 = self._roi_region.getRegion()
        if self._fps <= 0:
            return
        f0 = max(0, int(round(t0 * self._fps)))
        f1 = min(self._n_frames - 1, int(round(t1 * self._fps)))
        dur = (f1 - f0) / self._fps
        self._roi_start_lbl.setText(f"Start: {f0}")
        self._roi_end_lbl.setText(f"End: {f1}")
        self._roi_dur_lbl.setText(f"Dur: {dur:.2f}s")

        # Seek the preview to whichever boundary just moved.
        # When both move equally (whole-region drag) we prefer the start.
        if self._n_frames > 0:
            delta0 = abs(t0 - self._prev_roi_t0)
            delta1 = abs(t1 - self._prev_roi_t1)
            seek_t = t0 if delta0 >= delta1 else t1
            seek_frame = max(0, min(int(round(seek_t * self._fps)), self._n_frames - 1))
            self._suppress_seek_signal = True
            try:
                self._slider.setValue(seek_frame)
            finally:
                self._suppress_seek_signal = False
        self._prev_roi_t0 = t0
        self._prev_roi_t1 = t1

        # Switch to Segment tab immediately and schedule auto-evaluation.
        if self._angles is not None:
            self._stats_tabs.setCurrentIndex(1)
            self._eval_timer.start(150)

    def _on_clear_roi(self) -> None:
        self._eval_timer.stop()
        self._roi_region.setVisible(False)
        self._roi_start_lbl.setText("Start: —")
        self._roi_end_lbl.setText("End: —")
        self._roi_dur_lbl.setText("Dur: —")
        self._clear_segment_stats()
        # Return focus to the full-recording statistics.
        self._stats_tabs.setCurrentIndex(0)

    def _roi_frame_range(self) -> tuple[int, int] | None:
        if not self._roi_region.isVisible() or self._n_frames == 0:
            return None
        t0, t1 = self._roi_region.getRegion()
        f0 = max(0, int(round(t0 * self._fps)))
        f1 = min(self._n_frames - 1, int(round(t1 * self._fps)))
        if f1 <= f0:
            return None
        return f0, f1

    def _on_evaluate_segment(self) -> None:
        rng = self._roi_frame_range()
        if rng is None:
            self._set_status("Drag on the plot to define a selection first.")
            return
        if self._angles is None:
            return
        f0, f1 = rng
        seg = self._angles[f0 : f1 + 1]
        p = self._selected_person()
        for k in range(len(ANGLE_DEFINITIONS)):
            stats = compute_extended_stats(seg, k, p, fps=self._fps)
            if stats is None:
                cells = ["—"] * 7
            else:
                cells = [
                    _fmt(stats.min_deg),
                    _fmt(stats.max_deg),
                    _fmt(stats.mean_deg),
                    _fmt(stats.std_deg),
                    _fmt(stats.cv_pct),
                    _fmt(stats.rom_deg),
                    _fmt(stats.excursion_deg, 0),
                ]
            self._fill_table_row(self._seg_table, k, cells)
        self._stats_tabs.setCurrentIndex(1)
        label = self._roi_label_edit.text().strip() or "segment"
        self._set_status(f"Segment '{label}': frames {f0}–{f1} ({(f1 - f0) / self._fps:.2f} s)")

    def _on_save_segment_csv(self) -> None:
        rng = self._roi_frame_range()
        if rng is None or self._angles is None:
            self._set_status("Drag to define a selection before saving.")
            return
        f0, f1 = rng
        label = self._roi_label_edit.text().strip() or "segment"
        default_name = f"angles_{label}_{f0}-{f1}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save segment CSV", default_name, "CSV (*.csv)")
        if not path:
            return
        seg = self._angles[f0 : f1 + 1]
        try:
            angles_to_csv(path, seg, fps=self._fps)
        except Exception as exc:
            self._set_status(f"Save failed: {exc}")
            return
        self._set_status(f"Saved segment to {Path(path).name}")

    # =========================================================================
    # File I/O slots
    # =========================================================================

    def _on_browse_pose3d(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load pose3d.npz", "", "NumPy (*.npz)")
        if path:
            self.load_pose3d(path)

    def _on_export_full_csv(self) -> None:
        if self._angles is None:
            self._set_status("No data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export full angles CSV", "angles.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            angles_to_csv(path, self._angles, fps=self._fps)
        except Exception as exc:
            self._set_status(f"Export failed: {exc}")
            return
        self._set_status(f"Exported to {Path(path).name}")

    # =========================================================================
    # Rendering helpers
    # =========================================================================

    def _selected_person(self) -> int:
        return max(0, self._person_combo.currentIndex())

    def _redraw_curves(self) -> None:
        if self._angles is None or self._n_frames == 0:
            for c in self._curves:
                c.setData([], [])
            return
        p = self._selected_person()
        t_axis = np.arange(self._n_frames, dtype=np.float32) / self._fps
        for k, curve in enumerate(self._curves):
            if not self._angle_checks[k].isChecked():
                curve.setData([], [])
            else:
                curve.setData(t_axis, self._angles[:, p, k], connect="finite")

    def _refresh_2d_preview(
        self, p2d_left: str | None = None, p2d_right: str | None = None
    ) -> None:
        self._preview_2d.set_data(
            self._video_left or None,
            self._video_right or None,
            p2d_left,
            p2d_right,
        )

    # =========================================================================
    # Statistics table helpers
    # =========================================================================

    def _refresh_full_stats(self) -> None:
        if self._angles is None:
            for k in range(len(ANGLE_DEFINITIONS)):
                self._fill_table_row(self._full_table, k, ["—"] * 7)
            return
        p = self._selected_person()
        for k in range(len(ANGLE_DEFINITIONS)):
            stats = compute_extended_stats(self._angles, k, p, fps=self._fps)
            if stats is None:
                cells = ["—"] * 7
            else:
                cells = [
                    _fmt(stats.min_deg),
                    _fmt(stats.max_deg),
                    _fmt(stats.mean_deg),
                    _fmt(stats.std_deg),
                    _fmt(stats.cv_pct),
                    _fmt(stats.rom_deg),
                    _fmt(stats.nan_pct),
                ]
            self._fill_table_row(self._full_table, k, cells)
            self._set_row_enabled(self._full_table, k, self._angle_checks[k].isChecked())

    def _refresh_asymmetry_tab(self) -> None:
        if self._angles is None:
            for r in range(len(ANGLE_PAIRS)):
                self._fill_table_row(self._asym_table, r, ["—"] * 3)
            return
        p = self._selected_person()
        for row_idx, (l_idx, r_idx, _name) in enumerate(ANGLE_PAIRS):
            l_stats = compute_extended_stats(self._angles, l_idx, p, fps=self._fps)
            r_stats = compute_extended_stats(self._angles, r_idx, p, fps=self._fps)
            if l_stats is None or r_stats is None:
                cells = ["—"] * 3
            else:
                si_mean = compute_symmetry_index(l_stats.mean_deg, r_stats.mean_deg)
                si_rom = compute_symmetry_index(l_stats.rom_deg, r_stats.rom_deg)
                si_vel = compute_symmetry_index(l_stats.peak_vel_deg_s, r_stats.peak_vel_deg_s)
                cells = [_fmt(si_mean), _fmt(si_rom), _fmt(si_vel)]
            self._fill_table_row(self._asym_table, row_idx, cells)

    def _clear_segment_stats(self) -> None:
        for k in range(len(ANGLE_DEFINITIONS)):
            self._fill_table_row(self._seg_table, k, ["—"] * 7)

    @staticmethod
    def _fill_table_row(table: QTableWidget, row: int, cells: list[str]) -> None:
        for c, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, c, item)

    @staticmethod
    def _set_row_enabled(table: QTableWidget, row: int, enabled: bool) -> None:
        colour = QColor(180, 180, 180) if enabled else QColor(90, 90, 90)
        for c in range(table.columnCount()):
            item = table.item(row, c)
            if item is not None:
                item.setForeground(colour)

    def _set_status(self, msg: str) -> None:
        self._status_lbl.setText(msg)
