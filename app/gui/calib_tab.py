"""Calibration tab: import stereo videos, configure board, run calibration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
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
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .error_dialog import show_worker_error
from .workers import CalibWorker


def _bgr_to_pixmap(bgr: np.ndarray, max_w: int = 320) -> QPixmap:
    """Convert a BGR numpy frame to a scaled QPixmap (preserving aspect ratio)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    # Use rgb.tobytes() so QImage owns a stable copy independent of the numpy array lifetime.
    qimg = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if w > max_w:
        pix = pix.scaledToWidth(max_w, Qt.TransformationMode.SmoothTransformation)
    return pix


def _video_meta_str(path: str) -> str:
    """Return a short string with video metadata, or an error hint."""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return "Cannot open file"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        dur = n / fps if fps > 0 else 0
        return f"{w}×{h}  {fps:.1f} fps  {dur:.1f} s"
    except Exception:
        return "?"


class CalibTab(QWidget):
    """
    Calibration tab widget.

    Signals:
        calibration_saved(str): emitted with path when calibration.yml is saved
    """

    calibration_saved = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._worker: CalibWorker | None = None
        self._project_dir: str | None = None
        self._setup_ui()

    def set_project_dir(self, path: str) -> None:
        self._project_dir = path
        self._update_out_path()

    def _setup_ui(self) -> None:
        root = QHBoxLayout(self)

        # ── Left: controls ───────────────────────────────────────────
        ctrl_panel = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_panel.setFixedWidth(340)

        # Video inputs
        vid_group = QGroupBox("Source Videos")
        vid_form = QFormLayout(vid_group)

        self._left_edit = QLineEdit()
        self._left_edit.setPlaceholderText("left camera video…")
        self._left_edit.textChanged.connect(self._validate_inputs)
        btn_left = QPushButton("Browse…")
        btn_left.clicked.connect(lambda: self._browse_video(self._left_edit))
        self._left_status = QLabel("✗")
        self._left_status.setFixedWidth(16)
        self._left_meta = QLabel("")
        self._left_meta.setStyleSheet("color: #888; font-size: 10px;")
        row_l = QHBoxLayout()
        row_l.addWidget(self._left_edit)
        row_l.addWidget(btn_left)
        row_l.addWidget(self._left_status)
        vid_form.addRow("Left video:", row_l)
        vid_form.addRow("", self._left_meta)

        self._right_edit = QLineEdit()
        self._right_edit.setPlaceholderText("right camera video…")
        self._right_edit.textChanged.connect(self._validate_inputs)
        btn_right = QPushButton("Browse…")
        btn_right.clicked.connect(lambda: self._browse_video(self._right_edit))
        self._right_status = QLabel("✗")
        self._right_status.setFixedWidth(16)
        self._right_meta = QLabel("")
        self._right_meta.setStyleSheet("color: #888; font-size: 10px;")
        row_r = QHBoxLayout()
        row_r.addWidget(self._right_edit)
        row_r.addWidget(btn_right)
        row_r.addWidget(self._right_status)
        vid_form.addRow("Right video:", row_r)
        vid_form.addRow("", self._right_meta)

        ctrl_layout.addWidget(vid_group)

        # Board configuration
        board_group = QGroupBox("Board Configuration")
        board_layout = QVBoxLayout(board_group)

        board_type_row = QHBoxLayout()
        board_type_row.addWidget(QLabel("Board type:"))
        self._board_combo = QComboBox()
        self._board_combo.addItems(["ChArUco (recommended)", "Chessboard"])
        self._board_combo.currentIndexChanged.connect(self._on_board_type_changed)
        board_type_row.addWidget(self._board_combo)
        board_layout.addLayout(board_type_row)

        # Stacked params
        self._board_stack = QStackedWidget()

        # ChArUco params
        charuco_widget = QWidget()
        cform = QFormLayout(charuco_widget)
        self._sq_x = QSpinBox()
        self._sq_x.setRange(3, 20)
        self._sq_x.setValue(7)
        self._sq_y = QSpinBox()
        self._sq_y.setRange(3, 20)
        self._sq_y.setValue(5)
        self._sq_size = QDoubleSpinBox()
        self._sq_size.setRange(0.001, 1.0)
        self._sq_size.setValue(0.04)
        self._sq_size.setSuffix(" m")
        self._sq_size.setDecimals(4)
        self._mk_size = QDoubleSpinBox()
        self._mk_size.setRange(0.001, 1.0)
        self._mk_size.setValue(0.03)
        self._mk_size.setSuffix(" m")
        self._mk_size.setDecimals(4)
        self._aruco_dict = QComboBox()
        self._aruco_dict.addItems(["DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_6X6_250"])
        cform.addRow("Squares X:", self._sq_x)
        cform.addRow("Squares Y:", self._sq_y)
        cform.addRow("Square size:", self._sq_size)
        cform.addRow("Marker size:", self._mk_size)
        cform.addRow("ArUco dict:", self._aruco_dict)
        self._board_stack.addWidget(charuco_widget)

        # Chessboard params
        chess_widget = QWidget()
        chform = QFormLayout(chess_widget)
        self._chess_cols = QSpinBox()
        self._chess_cols.setRange(3, 20)
        self._chess_cols.setValue(9)
        self._chess_rows = QSpinBox()
        self._chess_rows.setRange(3, 20)
        self._chess_rows.setValue(6)
        self._chess_sq_size = QDoubleSpinBox()
        self._chess_sq_size.setRange(0.001, 1.0)
        self._chess_sq_size.setValue(0.025)
        self._chess_sq_size.setSuffix(" m")
        self._chess_sq_size.setDecimals(4)
        self._chess_cols.setToolTip(
            "Number of inner (non-border) corner columns.\n"
            "A 9-column printed board has 8 inner corners."
        )
        self._chess_rows.setToolTip(
            "Number of inner (non-border) corner rows.\nA 7-row printed board has 6 inner corners."
        )
        chform.addRow("Columns (inner):", self._chess_cols)
        chform.addRow("Rows (inner):", self._chess_rows)
        chform.addRow("Square size:", self._chess_sq_size)
        self._board_stack.addWidget(chess_widget)

        board_layout.addWidget(self._board_stack)
        ctrl_layout.addWidget(board_group)

        # Frame selection options
        frame_group = QGroupBox("Frame Selection")
        fform = QFormLayout(frame_group)
        self._max_frames = QSpinBox()
        self._max_frames.setRange(10, 200)
        self._max_frames.setValue(60)
        self._sample_every = QSpinBox()
        self._sample_every.setRange(1, 30)
        self._sample_every.setValue(5)
        fform.addRow("Max frames:", self._max_frames)
        fform.addRow("Sample every:", self._sample_every)
        ctrl_layout.addWidget(frame_group)

        # Output
        out_group = QGroupBox("Output")
        oform = QFormLayout(out_group)
        self._out_edit = QLineEdit("calibration.yml")
        self._out_edit.textChanged.connect(self._validate_inputs)
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(30)
        btn_out.clicked.connect(self._browse_output)
        row_out = QHBoxLayout()
        row_out.addWidget(self._out_edit)
        row_out.addWidget(btn_out)
        oform.addRow("Save to:", row_out)
        ctrl_layout.addWidget(out_group)

        ctrl_layout.addStretch()

        # Run / Cancel
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Calibration")
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip("Select left video, right video, and output path to enable.")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        ctrl_layout.addLayout(btn_row)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        ctrl_layout.addWidget(self._progress_bar)

        root.addWidget(ctrl_panel)

        # ── Right: results log + frame preview ───────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Frame preview (shows left/right side-by-side when board is detected)
        preview_group = QGroupBox("Live Frame Preview")
        preview_layout = QHBoxLayout(preview_group)
        self._preview_left = QLabel("No frame yet")
        self._preview_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_left.setMinimumWidth(240)
        self._preview_left.setStyleSheet("background:#1a1a1a; color:#888;")
        self._preview_right = QLabel("No frame yet")
        self._preview_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_right.setMinimumWidth(240)
        self._preview_right.setStyleSheet("background:#1a1a1a; color:#888;")
        preview_layout.addWidget(self._preview_left)
        preview_layout.addWidget(self._preview_right)
        right_layout.addWidget(preview_group)

        log_group = QGroupBox("Results")
        log_layout = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFontFamily("monospace")
        log_layout.addWidget(self._log)
        right_layout.addWidget(log_group, 1)

        root.addWidget(right_panel, 1)

    def _on_board_type_changed(self, idx: int) -> None:
        self._board_stack.setCurrentIndex(idx)
        if idx == 1:
            # Reset to sensible chessboard defaults when switching to Chessboard mode
            self._chess_cols.setValue(8)
            self._chess_rows.setValue(6)
            self._chess_sq_size.setValue(0.05)

    def _validate_inputs(self) -> None:
        """Enable Run button only when all required paths are non-empty and exist."""
        left = self._left_edit.text().strip()
        right = self._right_edit.text().strip()
        out = self._out_edit.text().strip()

        left_ok = bool(left) and Path(left).is_file()
        right_ok = bool(right) and Path(right).is_file()
        out_ok = bool(out)

        self._left_status.setText("✓" if left_ok else "✗")
        self._left_status.setStyleSheet("color: green;" if left_ok else "color: red;")
        self._right_status.setText("✓" if right_ok else "✗")
        self._right_status.setStyleSheet("color: green;" if right_ok else "color: red;")

        all_ok = left_ok and right_ok and out_ok
        self._run_btn.setEnabled(all_ok)
        if all_ok:
            self._run_btn.setToolTip("")
        else:
            missing = []
            if not left_ok:
                missing.append("left video")
            if not right_ok:
                missing.append("right video")
            if not out_ok:
                missing.append("output path")
            self._run_btn.setToolTip(f"Still needed: {', '.join(missing)}")

    def _browse_video(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov *.mkv *.wmv);;All (*)"
        )
        if path:
            edit.setText(path)
            # Show metadata for the selected video
            meta_label = self._left_meta if edit is self._left_edit else self._right_meta
            meta_label.setText(_video_meta_str(path))

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", self._out_edit.text(), "YAML (*.yml *.yaml)"
        )
        if path:
            # Strip doubled extension that Qt adds on some platforms (e.g. foo.yml.yml)
            p = Path(path)
            if p.suffix in (".yml", ".yaml") and Path(p.stem).suffix in (".yml", ".yaml"):
                path = str(p.with_suffix(""))
            self._out_edit.setText(path)

    def _update_out_path(self) -> None:
        if self._project_dir:
            default = str(Path(self._project_dir) / "calibration.yml")
            self._out_edit.setText(default)

    def _board_cfg(self) -> dict:
        if self._board_combo.currentIndex() == 0:
            return {
                "type": "charuco",
                "squares_x": self._sq_x.value(),
                "squares_y": self._sq_y.value(),
                "square_size": self._sq_size.value(),
                "marker_size": self._mk_size.value(),
                "dictionary": self._aruco_dict.currentText(),
            }
        else:
            return {
                "type": "checkerboard",
                "cols": self._chess_cols.value(),
                "rows": self._chess_rows.value(),
                "square_size": self._chess_sq_size.value(),
            }

    def _on_run(self) -> None:
        left = self._left_edit.text().strip()
        right = self._right_edit.text().strip()
        out = self._out_edit.text().strip()

        if not left or not right:
            self._log_msg("Error: select both left and right videos first.")
            return

        self._log.clear()
        self._log_msg(f"Starting calibration…\nLeft:  {left}\nRight: {right}\nOutput: {out}\n")
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress_bar.setValue(0)

        self._worker = CalibWorker(
            video_left=left,
            video_right=right,
            board_cfg=self._board_cfg(),
            output_path=out,
            max_frames=self._max_frames.value(),
            sample_every=self._sample_every.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
        self._log_msg("\nCancellation requested…")

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress_bar.setValue(pct)
        if msg:
            self._log_msg(msg)

    def _on_finished(self, result: object) -> None:
        self._validate_inputs()  # re-enable Run only if fields are still valid
        self._cancel_btn.setEnabled(False)

        if result is None:
            self._log_msg("\nCancelled.")
            return

        out_path = str(result)
        self._progress_bar.setValue(100)

        # Show quality metrics with color coding
        try:
            from app.calib.stereo import load_calibration

            calib = load_calibration(out_path)
            q = calib.get("quality", {})
            rms = q.get("rms", None)
            n_frames = q.get("n_frames_used", "?")

            # Compute stereo baseline from T vector
            import numpy as np

            T = calib.get("T", None)
            baseline_str = "?"
            if T is not None:
                baseline_m = float(np.linalg.norm(T))
                baseline_str = f"{baseline_m * 100:.1f} cm"

            # Color-code RMS
            if rms is not None:
                if rms < 0.5:
                    rms_rating = "Excellent"
                    rms_color = "#2ecc71"
                elif rms <= 1.5:
                    rms_rating = "Acceptable"
                    rms_color = "#3498db"
                else:
                    rms_rating = "Poor — consider recalibrating"
                    rms_color = "#e74c3c"
                rms_str = f"{rms:.4f} px"
            else:
                rms_str, rms_rating, rms_color = "?", "", "#888888"

            html = (
                f"<br><b>✓ Calibration saved:</b> {out_path}<br>"
                f"<table style='margin-top:6px;'>"
                f"<tr><td>RMS reprojection error</td>"
                f"<td><b>{rms_str}</b></td>"
                f"<td><span style='background:{rms_color};color:white;padding:1px 6px;"
                f"border-radius:3px;'>{rms_rating}</span></td></tr>"
                f"<tr><td>Frames used</td><td><b>{n_frames}</b></td><td></td></tr>"
                f"<tr><td>Stereo baseline</td><td><b>{baseline_str}</b></td><td></td></tr>"
                f"</table>"
            )
            if rms is not None and rms > 1.5:
                html += (
                    "<br><i>Tip: recalibrate with more board positions and better "
                    "coverage of image corners.</i>"
                )
            self._log.append(html)

            # Show blocking warning dialog if frame count is below recommended minimum
            if isinstance(n_frames, int) and n_frames < 20:
                QMessageBox.warning(
                    self,
                    "Low frame count — calibration may be unreliable",
                    f"Calibration was computed from only <b>{n_frames}</b> frame(s).\n\n"
                    "At least 20 frames with a detected board pattern are recommended "
                    "for a reliable result.\n\n"
                    "The output has been saved, but the calibration may be degenerate. "
                    "Consider recording a longer calibration video with more varied "
                    "board positions covering all corners of the image.",
                )
        except Exception as e:
            self._log_msg(f"\nSaved: {out_path} (could not load metrics: {e})")

        self.calibration_saved.emit(out_path)

    def _on_error(self, msg: str) -> None:
        self._validate_inputs()
        self._cancel_btn.setEnabled(False)
        self._log_msg(f"\nError: {msg}")

        if "no valid paired frames" in msg.lower():
            board_type = "ChArUco" if self._board_combo.currentIndex() == 0 else "Chessboard"
            QMessageBox.warning(
                self,
                "Calibration failed \u2014 no patterns detected",
                f"No {board_type} pattern was detected in any frame pair.\n\n"
                "Check the following:\n"
                "  \u2022 Board type matches your printed target (ChArUco vs Chessboard)\n"
                "  \u2022 Columns and Rows are inner corner counts, not total squares\n"
                "  \u2022 The board fills at least 30% of the frame in both cameras\n"
                "  \u2022 Lighting is even and there is no motion blur\n"
                "  \u2022 The board is visible simultaneously in BOTH cameras\n\n"
                "Tip: Try lowering 'Min coverage' in advanced settings, or "
                "move the board closer to both cameras.",
            )
            self._log_msg(
                "\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u2500\u2500\u2500\u2500\u2500\u2500\n"
                "Calibration failed: 0 frames detected.\n\n"
                "Troubleshooting:\n"
                "  \u2022 Board type must match your printed target\n"
                "  \u2022 Columns/Rows are INNER corner counts\n"
                "  \u2022 Board must fill \u226530% of the frame\n"
                "  \u2022 Even lighting, no motion blur\n"
                "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u2500\u2500\u2500\u2500\u2500\u2500"
            )
        else:
            show_worker_error(self, msg, context="Calibration")

    def _on_frame_ready(self, frames: tuple) -> None:
        """Display annotated left/right frame pair in the preview labels."""
        left_bgr, right_bgr = frames
        self._preview_left.setPixmap(_bgr_to_pixmap(left_bgr, 240))
        self._preview_right.setPixmap(_bgr_to_pixmap(right_bgr, 240))

    def _log_msg(self, msg: str) -> None:
        self._log.append(msg)
