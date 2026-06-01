"""
Reconstruction tab: import exercise videos, run full pipeline, view 3D skeleton.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .error_dialog import show_worker_error
from .pose2d_preview import Pose2DPreview, derive_pose2d_paths
from .viewer3d import SkeletonViewer3D
from .workers import Pose2DWorker, Recon3DWorker


def _video_meta(path: str) -> dict:
    """Return dict with fps, width, height, duration_s. Empty dict on failure."""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {}
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        dur = n / fps if fps > 0 else 0
        return {"fps": fps, "width": w, "height": h, "duration_s": dur}
    except Exception:
        return {}


class ReconTab(QWidget):
    """
    Reconstruction + visualization tab.

    Signals:
        pose3d_ready(str): emitted when pose3d.npz has been created
    """

    pose3d_ready = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._worker: QWidget | None = None
        self._project_dir: str | None = None
        self._pose3d_path: str | None = None
        self._pose2d_left_path: str | None = None
        self._pose2d_right_path: str | None = None
        # Deferred pose3d path: set when set_project_dir() is called before the
        # widget is first shown (GL context not yet created).  Processed in
        # showEvent() once the GL context is ready.
        self._pending_pose3d_path: str | None = None
        self._playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_timer_tick)
        self._left_meta_cache: dict = {}
        self._right_meta_cache: dict = {}
        self._base_fps: float = 30.0
        self._setup_ui()

    def showEvent(self, event) -> None:  # noqa: N802
        """Sync viewer and 2D preview to the current slider position on tab switch."""
        super().showEvent(event)
        if self._pose3d_path is not None:
            self._viewer.show_frame(self._slider.value())
            self._preview_2d.show_frame(self._slider.value())

    def set_project_dir(self, path: str) -> None:
        self._project_dir = path
        # Update default paths
        d = Path(path)
        if not self._calib_edit.text():
            calib = d / "calibration.yml"
            if calib.exists():
                self._calib_edit.setText(str(calib))
        pose3d = d / "pose3d.npz"
        current_out = self._out_edit.text().strip()
        # Set project-folder default whenever output path is empty or still
        # shows the bare relative placeholder "pose3d.npz".
        if not current_out or current_out == "pose3d.npz":
            self._out_edit.setText(str(pose3d))

        # Auto-detect pose2d NPZ files so the 2D preview is populated even when
        # no pose3d.npz exists yet (e.g. fresh project where only the 2D
        # extraction has been run, or after replacing a recording).
        p2d_l = d / "pose2d_left.npz"
        p2d_r = d / "pose2d_right.npz"
        if p2d_l.exists() and not self._pose2d_left_path:
            self._pose2d_left_path = str(p2d_l)
        if p2d_r.exists() and not self._pose2d_right_path:
            self._pose2d_right_path = str(p2d_r)

        # Auto-load any existing reconstruction so the user can view results
        # immediately without re-running the pipeline.
        # Guard: pyqtgraph's GLViewWidget.addItem() calls makeCurrent() which
        # fails before the widget is first shown (GL context not yet created).
        # If not visible yet, store the path and defer until showEvent().
        if pose3d.exists():
            if self.isVisible():
                self._try_auto_load_pose3d(str(pose3d))
            else:
                self._pending_pose3d_path = str(pose3d)

    def showEvent(self, event) -> None:
        """Process any deferred pose3d load the first time the tab is shown.

        set_project_dir() is called during session restore, before the main
        window is shown, so the GL context does not yet exist.  Rather than
        calling _load_pose3d() immediately (which would try to add GL items
        before makeCurrent() succeeds), we stash the path and process it here
        via a zero-timeout singleShot so we run after the first paintGL().
        """
        super().showEvent(event)
        if self._pending_pose3d_path is not None:
            path = self._pending_pose3d_path
            self._pending_pose3d_path = None
            QTimer.singleShot(0, lambda: self._try_auto_load_pose3d(path))

    def set_calibration(self, calib_path: str) -> None:
        """Auto-fill calibration path (called from calibration tab signal)."""
        self._calib_edit.setText(calib_path)

    def _try_auto_load_pose3d(self, explicit_path: str | None = None) -> None:
        """Load pose3d.npz automatically if it exists and is not already loaded.

        Call with an *explicit_path* to load a known location (e.g. from
        set_project_dir).  Without an argument the path is derived from the
        current output-path field or the video directory.
        """
        if explicit_path:
            candidate = explicit_path
        else:
            candidate = self._out_edit.text().strip()
            if not candidate:
                left = self._left_edit.text().strip()
                if left:
                    candidate = str(Path(left).parent / "pose3d.npz")
        if candidate and Path(candidate).is_file() and candidate != self._pose3d_path:
            self._log_msg("Found existing pose3d.npz — loading automatically…")
            self._load_pose3d(candidate)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: controls ──────────────────────────────────────
        ctrl_panel = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_panel.setFixedWidth(340)

        # Videos
        vid_group = QGroupBox("Exercise Videos")
        vform = QFormLayout(vid_group)
        self._left_edit = QLineEdit()
        self._left_edit.setPlaceholderText("left camera video…")
        self._left_edit.textChanged.connect(self._validate_inputs)
        btn_l = QPushButton("Browse…")
        btn_l.clicked.connect(lambda: self._browse_video(self._left_edit))
        self._left_status = QLabel("✗")
        self._left_status.setFixedWidth(16)
        self._left_meta_lbl = QLabel("")
        self._left_meta_lbl.setStyleSheet("color: #888; font-size: 10px;")
        row_l = QHBoxLayout()
        row_l.addWidget(self._left_edit)
        row_l.addWidget(btn_l)
        row_l.addWidget(self._left_status)
        vform.addRow("Left video:", row_l)
        vform.addRow("", self._left_meta_lbl)

        self._right_edit = QLineEdit()
        self._right_edit.setPlaceholderText("right camera video…")
        self._right_edit.textChanged.connect(self._validate_inputs)
        btn_r = QPushButton("Browse…")
        btn_r.clicked.connect(lambda: self._browse_video(self._right_edit))
        self._right_status = QLabel("✗")
        self._right_status.setFixedWidth(16)
        self._right_meta_lbl = QLabel("")
        self._right_meta_lbl.setStyleSheet("color: #888; font-size: 10px;")
        row_r = QHBoxLayout()
        row_r.addWidget(self._right_edit)
        row_r.addWidget(btn_r)
        row_r.addWidget(self._right_status)
        vform.addRow("Right video:", row_r)
        vform.addRow("", self._right_meta_lbl)
        ctrl_layout.addWidget(vid_group)

        # Calibration
        cal_group = QGroupBox("Calibration")
        cform = QFormLayout(cal_group)
        self._calib_edit = QLineEdit()
        self._calib_edit.setPlaceholderText("calibration.yml…")
        self._calib_edit.textChanged.connect(self._validate_inputs)
        btn_calib = QPushButton("Browse…")
        btn_calib.clicked.connect(self._browse_calib)
        self._calib_status = QLabel("✗")
        self._calib_status.setFixedWidth(16)
        row_c = QHBoxLayout()
        row_c.addWidget(self._calib_edit)
        row_c.addWidget(btn_calib)
        row_c.addWidget(self._calib_status)
        cform.addRow("Calib file:", row_c)
        ctrl_layout.addWidget(cal_group)

        # Pipeline options
        opt_group = QGroupBox("Options")
        oform = QFormLayout(opt_group)

        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["MediaPipe (default)", "RTMPose (optional)"])
        oform.addRow("Pose backend:", self._backend_combo)

        self._num_poses = QSpinBox()
        self._num_poses.setRange(1, 6)
        self._num_poses.setValue(1)
        oform.addRow("Max persons:", self._num_poses)

        self._min_conf = QDoubleSpinBox()
        self._min_conf.setRange(0.0, 1.0)
        self._min_conf.setValue(0.3)
        self._min_conf.setSingleStep(0.05)
        self._min_conf.setDecimals(2)
        oform.addRow("Min confidence:", self._min_conf)

        self._max_reproj = QDoubleSpinBox()
        self._max_reproj.setRange(1.0, 200.0)
        self._max_reproj.setValue(20.0)
        self._max_reproj.setSuffix(" px")
        self._max_reproj.setDecimals(1)
        oform.addRow("Max reproj err:", self._max_reproj)

        self._smooth_check = QCheckBox("Apply temporal smoothing (1€ filter)")
        self._smooth_check.setChecked(True)
        self._smooth_check.setToolTip(
            "When checked, the OneEuro filter smooths the 3D trajectories after "
            "triangulation.  Uncheck to get raw, frame-by-frame positions — "
            "useful for fast movements where smoothing introduces lag."
        )
        self._smooth_check.toggled.connect(self._on_smooth_toggled)
        oform.addRow("Smoothing:", self._smooth_check)

        self._min_cutoff = QDoubleSpinBox()
        self._min_cutoff.setRange(0.01, 10.0)
        self._min_cutoff.setValue(1.0)   # less aggressive: was 0.5 Hz
        self._min_cutoff.setSuffix(" Hz")
        self._min_cutoff.setDecimals(2)
        oform.addRow("1€ min_cutoff:", self._min_cutoff)

        self._beta = QDoubleSpinBox()
        self._beta.setRange(0.0, 10.0)
        self._beta.setValue(0.5)         # less aggressive: was 0.05
        self._beta.setDecimals(3)
        oform.addRow("1€ beta:", self._beta)

        self._d_cutoff = QDoubleSpinBox()
        self._d_cutoff.setRange(0.01, 10.0)
        self._d_cutoff.setValue(1.0)
        self._d_cutoff.setSuffix(" Hz")
        self._d_cutoff.setDecimals(2)
        oform.addRow("1€ d_cutoff:", self._d_cutoff)

        ctrl_layout.addWidget(opt_group)

        # Output — label removed so the edit box fills the full group width
        out_group = QGroupBox("Output")
        out_inner = QHBoxLayout(out_group)
        out_inner.setContentsMargins(6, 4, 6, 4)
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("pose3d.npz (set when project is loaded)")
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(30)
        btn_out.clicked.connect(self._browse_output)
        out_inner.addWidget(self._out_edit)
        out_inner.addWidget(btn_out)
        ctrl_layout.addWidget(out_group)

        ctrl_layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Pipeline")
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip("Select left video, right video, and calibration file to enable.")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        ctrl_layout.addLayout(btn_row)

        # Load cached
        load_row = QHBoxLayout()
        load_btn = QPushButton("Load pose3d.npz…")
        load_btn.clicked.connect(self._on_load)
        export_btn = QPushButton("Export CSV…")
        export_btn.clicked.connect(self._on_export)
        load_row.addWidget(load_btn)
        load_row.addWidget(export_btn)
        ctrl_layout.addLayout(load_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        ctrl_layout.addWidget(self._progress_bar)

        # Status log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFontFamily("monospace")
        self._log.setMaximumHeight(100)
        ctrl_layout.addWidget(QLabel("Log:"))
        ctrl_layout.addWidget(self._log)

        splitter.addWidget(ctrl_panel)

        # ── Right panel: 2D previews + 3D viewer + timeline ──────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Small 2D previews to sanity-check pose detection
        self._preview_2d = Pose2DPreview()
        right_layout.addWidget(self._preview_2d)

        self._viewer = SkeletonViewer3D()
        right_layout.addWidget(self._viewer, 1)

        # Timeline + playback
        tl_group = QGroupBox("Playback")
        tl_layout = QVBoxLayout(tl_group)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        tl_layout.addWidget(self._slider)

        play_row = QHBoxLayout()

        _si = QApplication.style().standardIcon

        self._seek_back_btn = QPushButton()
        self._seek_back_btn.setIcon(_si(QStyle.StandardPixmap.SP_MediaSeekBackward))
        self._seek_back_btn.setToolTip("-1 second")
        self._seek_back_btn.setFixedWidth(34)
        self._seek_back_btn.clicked.connect(self._on_seek_back)

        self._prev_btn = QPushButton()
        self._prev_btn.setIcon(_si(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self._prev_btn.setToolTip("Previous frame")
        self._prev_btn.setFixedWidth(34)
        self._prev_btn.clicked.connect(self._on_prev_frame)

        self._play_btn = QPushButton()
        self._play_btn.setIcon(_si(QStyle.StandardPixmap.SP_MediaPlay))
        self._play_btn.setToolTip("Play / Pause")
        self._play_btn.setFixedWidth(34)
        self._play_btn.setCheckable(True)
        self._play_btn.toggled.connect(self._on_play_toggle)

        self._next_btn = QPushButton()
        self._next_btn.setIcon(_si(QStyle.StandardPixmap.SP_MediaSkipForward))
        self._next_btn.setToolTip("Next frame")
        self._next_btn.setFixedWidth(34)
        self._next_btn.clicked.connect(self._on_next_frame)

        self._seek_fwd_btn = QPushButton()
        self._seek_fwd_btn.setIcon(_si(QStyle.StandardPixmap.SP_MediaSeekForward))
        self._seek_fwd_btn.setToolTip("+1 second")
        self._seek_fwd_btn.setFixedWidth(34)
        self._seek_fwd_btn.clicked.connect(self._on_seek_fwd)

        self._frame_label = QLabel("Frame: 0 / 0")
        self._fps_label = QLabel("")
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25×", "0.5×", "1×", "2×"])
        self._speed_combo.setCurrentIndex(2)  # default 1×
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)

        play_row.addWidget(self._frame_label)
        play_row.addStretch()
        play_row.addWidget(self._seek_back_btn)
        play_row.addWidget(self._prev_btn)
        play_row.addWidget(self._play_btn)
        play_row.addWidget(self._next_btn)
        play_row.addWidget(self._seek_fwd_btn)
        play_row.addStretch()
        play_row.addWidget(QLabel("Speed:"))
        play_row.addWidget(self._speed_combo)
        play_row.addWidget(self._fps_label)
        tl_layout.addLayout(play_row)

        right_layout.addWidget(tl_group)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter)

        # Initialise status indicators so they show ✗/red on an empty widget
        # rather than appearing blank until the user first types something.
        self._validate_inputs()

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def _on_smooth_toggled(self, checked: bool) -> None:
        """Enable / disable the 1€ filter parameter spinboxes."""
        self._min_cutoff.setEnabled(checked)
        self._beta.setEnabled(checked)
        self._d_cutoff.setEnabled(checked)

    def _validate_inputs(self) -> None:
        """Enable Run button when all required paths are non-empty and exist."""
        left = self._left_edit.text().strip()
        right = self._right_edit.text().strip()
        calib = self._calib_edit.text().strip()

        left_ok = bool(left) and Path(left).is_file()
        right_ok = bool(right) and Path(right).is_file()
        calib_ok = bool(calib) and Path(calib).is_file()

        self._left_status.setText("✓" if left_ok else "✗")
        self._left_status.setStyleSheet("color: green;" if left_ok else "color: red;")
        self._right_status.setText("✓" if right_ok else "✗")
        self._right_status.setStyleSheet("color: green;" if right_ok else "color: red;")
        self._calib_status.setText("✓" if calib_ok else "✗")
        self._calib_status.setStyleSheet("color: green;" if calib_ok else "color: red;")

        all_ok = left_ok and right_ok and calib_ok
        self._run_btn.setEnabled(all_ok)
        if all_ok:
            self._run_btn.setToolTip("")
            # Auto-load a pre-existing reconstruction so the user can skip re-running.
            self._try_auto_load_pose3d()
            # Refresh the 2D preview now that valid video paths are confirmed.
            # Fire whenever pose2d *or* pose3d data is already loaded — this
            # covers two cases:
            #   1. Session restore: set_project_dir() detected pose2d / pose3d
            #      before _left_edit/_right_edit were populated; the first
            #      _refresh_2d_preview() call had empty video paths.
            #   2. Fresh project with pose2d but no pose3d yet: user wants to
            #      inspect 2D detections before running 3D reconstruction.
            if self._pose3d_path is not None or self._pose2d_left_path is not None:
                self._refresh_2d_preview()
        else:
            missing = []
            if not left_ok:
                missing.append("left video")
            if not right_ok:
                missing.append("right video")
            if not calib_ok:
                missing.append("calibration file")
            self._run_btn.setToolTip(f"Still needed: {', '.join(missing)}")

        # Check for FPS / resolution mismatch between L and R
        if left_ok and right_ok and self._left_meta_cache and self._right_meta_cache:
            lm, rm = self._left_meta_cache, self._right_meta_cache
            fps_ok = abs(lm["fps"] - rm["fps"]) / max(lm["fps"], 1) < 0.01
            res_ok = lm["width"] == rm["width"] and lm["height"] == rm["height"]
            if not fps_ok:
                self._right_meta_lbl.setStyleSheet("color: #e67e22; font-size: 10px;")
                self._right_meta_lbl.setText(
                    f"{rm['width']}×{rm['height']}  {rm['fps']:.1f} fps  "
                    f"{rm['duration_s']:.1f} s  ⚠ FPS mismatch"
                )
            elif not res_ok:
                self._right_meta_lbl.setStyleSheet("color: #e67e22; font-size: 10px;")
                self._right_meta_lbl.setText(
                    f"{rm['width']}×{rm['height']}  {rm['fps']:.1f} fps  "
                    f"{rm['duration_s']:.1f} s  ⚠ resolution mismatch"
                )

    def _browse_video(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov *.mkv);;All (*)"
        )
        if path:
            edit.setText(path)
            meta = _video_meta(path)
            meta_text = ""
            if meta:
                meta_text = (
                    f"{meta['width']}×{meta['height']}  "
                    f"{meta['fps']:.1f} fps  {meta['duration_s']:.1f} s"
                )
            if edit is self._left_edit:
                self._left_meta_cache = meta
                self._left_meta_lbl.setStyleSheet("color: #888; font-size: 10px;")
                self._left_meta_lbl.setText(meta_text)
                if meta:
                    self._base_fps = meta["fps"]
            else:
                self._right_meta_cache = meta
                self._right_meta_lbl.setStyleSheet("color: #888; font-size: 10px;")
                self._right_meta_lbl.setText(meta_text)

    def _browse_calib(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Calibration", "", "YAML (*.yml *.yaml)")
        if path:
            self._calib_edit.setText(path)

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save pose3d", self._out_edit.text(), "NumPy (*.npz)"
        )
        if path:
            self._out_edit.setText(path)

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------

    def _preflight_check(self, left: str, right: str, calib: str) -> bool:
        """
        Show a pre-flight checklist in the log and return True if safe to proceed.

        Returns False (and blocks run) only when a blocking error is present
        (e.g. FPS mismatch > 1%, file not found). Warnings are shown but do not block.
        """
        self._log.clear()
        self._log_msg("<b>Pre-flight check</b>")
        self._log_msg("─" * 50)

        ok = True

        # Left video
        lm = _video_meta(left)
        if lm:
            self._log_msg(
                f"✓  Left video found     {Path(left).name}  "
                f"{lm['width']}×{lm['height']}  {lm['fps']:.1f} fps  {lm['duration_s']:.1f} s"
            )
        else:
            self._log_msg(f"✗  Left video cannot be opened: {left}")
            ok = False

        # Right video
        rm = _video_meta(right)
        if rm:
            self._log_msg(
                f"✓  Right video found    {Path(right).name}  "
                f"{rm['width']}×{rm['height']}  {rm['fps']:.1f} fps  {rm['duration_s']:.1f} s"
            )
        else:
            self._log_msg(f"✗  Right video cannot be opened: {right}")
            ok = False

        # FPS / resolution mismatch (only when both videos opened)
        if lm and rm:
            fps_diff_pct = abs(lm["fps"] - rm["fps"]) / max(lm["fps"], 1) * 100
            if fps_diff_pct > 1.0:
                self._log_msg(
                    f"✗  FPS mismatch  left={lm['fps']:.1f}  right={rm['fps']:.1f}  "
                    f"({fps_diff_pct:.1f}% — exceeds 1% tolerance)"
                )
                ok = False
            dur_diff = abs(lm["duration_s"] - rm["duration_s"])
            if dur_diff > 1.0:
                self._log_msg(
                    f"⚠  Duration mismatch  {dur_diff:.1f} s difference  "
                    "(acceptable, will truncate to shorter clip)"
                )

        # Calibration file
        if Path(calib).is_file():
            try:
                from app.calib.stereo import load_calibration

                cal = load_calibration(calib)
                q = cal.get("quality", {})
                rms = q.get("rms")
                rms_str = f"  RMS {rms:.2f} px" if rms is not None else ""
                self._log_msg(f"✓  Calibration loaded   {Path(calib).name}{rms_str}")
                if rms is not None and rms > 1.5:
                    self._log_msg("⚠  Calibration RMS > 1.5 px — quality is marginal")
            except Exception as e:
                self._log_msg(f"✗  Calibration file invalid: {e}")
                ok = False
        else:
            self._log_msg(f"✗  Calibration file not found: {calib}")
            ok = False

        self._log_msg("─" * 50)
        if ok:
            self._log_msg("All checks passed — starting pipeline…")
        else:
            self._log_msg("✗  Pre-flight failed — fix the errors above before running.")
        return ok

    def _on_run(self) -> None:
        left = self._left_edit.text().strip()
        right = self._right_edit.text().strip()
        calib = self._calib_edit.text().strip()

        if not left or not right or not calib:
            self._log_msg("Error: fill in left video, right video, and calibration file.")
            return

        if not self._preflight_check(left, right, calib):
            return

        if self._project_dir:
            proj = Path(self._project_dir)
            out_left = str(proj / "pose2d_left.npz")
            out_right = str(proj / "pose2d_right.npz")
            out_3d = self._out_edit.text().strip() or str(proj / "pose3d.npz")
        else:
            out_dir = Path(left).parent
            out_left = str(out_dir / "pose2d_left.npz")
            out_right = str(out_dir / "pose2d_right.npz")
            _raw_out = self._out_edit.text().strip()
            out_3d = _raw_out if _raw_out else str(out_dir / "pose3d.npz")

        self._out_3d = out_3d
        # Remember pose2d locations so the 2D preview can pick them up after the
        # pipeline completes (or even mid-pipeline if pose2d finishes first).
        self._pose2d_left_path = out_left
        self._pose2d_right_path = out_right
        backend_name = "rtmpose" if "rtmpose" in self._backend_combo.currentText().lower() else "mediapipe"

        self._log_msg(f"\nStep 1/2: Extracting 2D poses (backend={backend_name})…")
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress_bar.setValue(0)

        self._pose2d_worker = Pose2DWorker(
            video_left=left,
            video_right=right,
            out_left=out_left,
            out_right=out_right,
            backend_name=backend_name,
            num_poses=self._num_poses.value(),
        )
        self._pose2d_worker.progress.connect(self._on_progress)
        self._pose2d_worker.finished.connect(
            lambda result: self._on_pose2d_done(result, calib, out_left, out_right, out_3d)
        )
        self._pose2d_worker.error.connect(self._on_error)
        self._pose2d_worker.start()
        self._worker = self._pose2d_worker

    def _on_pose2d_done(
        self, result: object, calib: str, out_left: str, out_right: str, out_3d: str
    ) -> None:
        if result is None:
            self._on_pipeline_done(None)
            return

        # Refresh the 2D preview as soon as pose2d files exist — gives the user a
        # quick sanity check on detection quality before the slower 3D step finishes.
        self._refresh_2d_preview()

        # Check for large NPZ frame-count mismatch before starting recon3d
        try:
            left_d = np.load(out_left, allow_pickle=True)
            right_d = np.load(out_right, allow_pickle=True)
            T_l = int(left_d["keypoints"].shape[0])
            T_r = int(right_d["keypoints"].shape[0])
            if T_l != T_r:
                mismatch_pct = abs(T_l - T_r) / max(T_l, T_r) * 100
                if mismatch_pct > 5.0:
                    dropped = abs(T_l - T_r)
                    reply = QMessageBox.question(
                        self,
                        "Frame count mismatch",
                        f"Left pose2d has <b>{T_l}</b> frames, right has <b>{T_r}</b> frames "
                        f"({mismatch_pct:.1f}% difference).\n\n"
                        f"Processing will truncate to the shorter length "
                        f"(<b>{min(T_l, T_r)}</b> frames). "
                        f"<b>{dropped}</b> frame(s) will be silently dropped.\n\n"
                        "Ensure both videos were recorded simultaneously and cover the same "
                        "time range. Continue anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        self._on_pipeline_done(None)
                        return
        except Exception:
            pass  # if NPZ can't be inspected, let recon3d handle it

        self._log_msg("Step 2/2: 3D reconstruction…")
        self._progress_bar.setValue(0)

        self._recon_worker = Recon3DWorker(
            calib_path=calib,
            pose2d_left=out_left,
            pose2d_right=out_right,
            output_path=out_3d,
            min_conf=self._min_conf.value(),
            max_reproj_err=self._max_reproj.value(),
            min_cutoff=self._min_cutoff.value(),
            beta=self._beta.value(),
            d_cutoff=self._d_cutoff.value(),
            no_smooth=not self._smooth_check.isChecked(),
        )
        self._recon_worker.progress.connect(self._on_progress)
        self._recon_worker.finished.connect(self._on_pipeline_done)
        self._recon_worker.error.connect(self._on_error)
        self._recon_worker.start()
        self._worker = self._recon_worker

    def _on_cancel(self) -> None:
        if self._worker and hasattr(self._worker, "cancel"):
            self._worker.cancel()
        self._log_msg("Cancellation requested…")

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress_bar.setValue(pct)
        if msg:
            self._log_msg(msg)

    def _on_pipeline_done(self, result: object) -> None:
        self._validate_inputs()  # re-enable Run only if fields are still valid
        self._cancel_btn.setEnabled(False)

        if result is None:
            self._log_msg("Pipeline cancelled or failed.")
            return

        self._progress_bar.setValue(100)
        self._log_msg(f"\n✓ Done. Output: {result}")
        # _load_pose3d() emits pose3d_ready on success, so downstream
        # consumers (Analysis tab) get notified for every successful load —
        # not only when the pipeline runs end-to-end here.
        self._load_pose3d(str(result))

    def _on_error(self, tb: str) -> None:
        self._validate_inputs()
        self._cancel_btn.setEnabled(False)
        self._log_msg(f"\nError:\n{tb}")
        show_worker_error(self, tb, context="Reconstruction")

    # ------------------------------------------------------------------
    # Load / export
    # ------------------------------------------------------------------

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load pose3d", "", "NumPy (*.npz)")
        if path:
            self._load_pose3d(path)

    def _load_pose3d(self, path: str) -> None:
        try:
            d = np.load(path, allow_pickle=True)
            joints3d = d["joints3d"]  # [T, P, 17, 3]
            conf3d = d["conf3d"]  # [T, P, 17]
            meta = d["meta"].item()

            self._viewer.set_data(joints3d, conf3d)
            T = joints3d.shape[0]
            fps = float(meta.get("fps", 30.0))

            self._slider.setRange(0, max(0, T - 1))
            self._slider.setValue(0)
            self._frame_label.setText(self._frame_label_text(0, T))
            self._fps_label.setText(f"{fps:.1f} fps")
            self._pose3d_path = path
            self._base_fps = fps

            # Update timer interval (respects current speed setting)
            self._on_speed_changed(0)

            # Try to derive matching pose2d paths next to pose3d.npz, then
            # populate the small 2D previews. Falls back gracefully if files
            # aren't present.
            inferred_l, inferred_r = derive_pose2d_paths(path)
            if inferred_l:
                self._pose2d_left_path = inferred_l
            if inferred_r:
                self._pose2d_right_path = inferred_r
            self._refresh_2d_preview()

            self._log_msg(f"Loaded {path} — {T} frames, {joints3d.shape[1]} person(s)")
            # Notify downstream consumers (Analysis tab) that fresh data is
            # available — applies to auto-load on startup, manual file open
            # via the "Load pose3d.npz…" button, *and* pipeline completion.
            self.pose3d_ready.emit(path)
        except Exception as e:
            self._log_msg(f"Failed to load {path}: {e}")

    def _refresh_2d_preview(self) -> None:
        """Push current video + pose2d paths into the preview widget."""
        left_video = self._left_edit.text().strip() or None
        right_video = self._right_edit.text().strip() or None
        self._preview_2d.set_data(
            left_video,
            right_video,
            self._pose2d_left_path,
            self._pose2d_right_path,
        )

    def _on_export(self) -> None:
        if self._pose3d_path is None:
            self._log_msg("No pose3d data loaded.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV (*.csv)")
        if not path:
            return

        try:
            import csv
            import datetime
            import json

            d = np.load(self._pose3d_path, allow_pickle=True)
            joints3d = d["joints3d"]  # [T, P, 17, 3]
            conf3d = d["conf3d"]  # [T, P, 17]
            meta = d["meta"].item()
            fps = float(meta.get("fps", 30.0))

            T, P, J, _ = joints3d.shape
            headers = ["frame", "time_s", "person"]
            for j in range(J):
                headers += [f"j{j}_x", f"j{j}_y", f"j{j}_z", f"j{j}_conf"]

            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for t in range(T):
                    for p in range(P):
                        row = [t, f"{t / fps:.4f}", p]
                        for j in range(J):
                            x, y, z = joints3d[t, p, j]
                            c = conf3d[t, p, j]
                            row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{c:.4f}"]
                        writer.writerow(row)

            self._log_msg(f"Exported CSV: {path}")

            # Write companion metadata.json alongside the CSV
            from pathlib import Path as _Path

            meta_path = str(_Path(path).parent / (_Path(path).stem + "_metadata.json"))
            metadata: dict = {
                "fps": fps,
                "coordinate_units": "meters",
                "n_frames": T,
                "n_persons": P,
                "n_joints": J,
                "software_version": "BatPose 0.1.0",
                "export_date": datetime.datetime.now().isoformat(),
            }
            # Include calibration RMS if available
            try:
                calib_path = self._calib_edit.text().strip()
                if calib_path:
                    from app.calib.stereo import load_calibration

                    cal = load_calibration(calib_path)
                    rms = cal.get("quality", {}).get("rms")
                    if rms is not None:
                        metadata["calibration_rms_px"] = float(rms)
            except Exception:
                pass

            with open(meta_path, "w") as mf:
                json.dump(metadata, mf, indent=2)
            self._log_msg(f"Metadata: {meta_path}")
        except Exception as e:
            self._log_msg(f"Export failed: {e}")

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _frame_label_text(self, frame: int, total: int) -> str:
        fps = self._base_fps if self._base_fps > 0 else 30.0
        secs = frame / fps
        m, s = divmod(int(secs), 60)
        dur_secs = max(0, total - 1) / fps
        dm, ds = divmod(int(dur_secs), 60)
        return f"Frame: {frame} / {max(0, total - 1)}   {m:02d}:{s:02d} / {dm:02d}:{ds:02d}"

    def _on_slider_changed(self, value: int) -> None:
        # Skip the expensive render when this tab is hidden — the Analysis tab
        # drives the slider via frame_seek while its own playback is running,
        # which would otherwise cause a double video-read per tick.
        if self.isVisible():
            self._viewer.show_frame(value)
            self._preview_2d.show_frame(value)
        T = self._viewer.frame_count
        self._frame_label.setText(self._frame_label_text(value, T))

    def _on_play_toggle(self, checked: bool) -> None:
        _si = QApplication.style().standardIcon
        if checked:
            self._play_btn.setIcon(_si(QStyle.StandardPixmap.SP_MediaPause))
            self._play_timer.start()
        else:
            self._play_btn.setIcon(_si(QStyle.StandardPixmap.SP_MediaPlay))
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
        self._play_btn.setIcon(
            _si(SP.SP_MediaPause) if playing else _si(SP.SP_MediaPlay)
        )
        self._play_btn.blockSignals(False)

    def _on_prev_frame(self) -> None:
        self._slider.setValue(max(0, self._slider.value() - 1))

    def _on_next_frame(self) -> None:
        self._slider.setValue(min(self._slider.maximum(), self._slider.value() + 1))

    def _on_seek_back(self) -> None:
        step = max(1, int(self._base_fps if self._base_fps > 0 else 30.0))
        self._slider.setValue(max(0, self._slider.value() - step))

    def _on_seek_fwd(self) -> None:
        step = max(1, int(self._base_fps if self._base_fps > 0 else 30.0))
        self._slider.setValue(min(self._slider.maximum(), self._slider.value() + step))

    def _on_speed_changed(self, _idx: int) -> None:
        """Update the play-timer interval when speed combo changes."""
        speeds = [0.25, 0.5, 1.0, 2.0]
        speed = speeds[self._speed_combo.currentIndex()]
        fps = self._base_fps if self._base_fps > 0 else 30.0
        interval_ms = max(1, int(1000.0 / (fps * speed)))
        self._play_timer.setInterval(interval_ms)

    def _on_timer_tick(self) -> None:
        T = self._viewer.frame_count
        if T == 0:
            self._play_timer.stop()
            self._play_btn.setChecked(False)
            return
        next_frame = (self._slider.value() + 1) % T
        self._slider.setValue(next_frame)

    def _log_msg(self, msg: str) -> None:
        self._log.append(msg)
