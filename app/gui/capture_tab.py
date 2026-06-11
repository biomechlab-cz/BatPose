"""
Live capture tab — FLIR cameras.

The tab guides the user through setup:
  Step 1 (SDK missing)    → installation instructions
  Step 2 (SDK found)      → connect cameras + wiring diagram + Detect button
  Step 3 (2+ cams found)  → assign left/right serials, pick primary, configure sync
  Streaming               → Start Preview, Start/Stop Recording
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from concurrent.futures import Future as _Future
from concurrent.futures import ThreadPoolExecutor as _ThreadPool
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .viewer3d import COCO17_EDGES, SkeletonViewer3D
from .workers import CaptureWorker


class _ClickableLabel(QLabel):
    """QLabel that emits a `clicked` signal on left mouse-button release."""

    clicked = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(ev)


class _Fullscreen3DSkeleton(QDialog):
    """Frameless fullscreen dialog wrapping a SkeletonViewer3D.

    The caller pushes live frames via `set_frame()`.  A clean click (or
    double-click) anywhere on the GL canvas closes the dialog; Esc and F11 do
    the same.  Dragging still orbits the view.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("3D Skeleton — full screen")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setStyleSheet("background:#0a0a0a;")
        self.setModal(False)

        # Local import to avoid a circular: viewer3d imports nothing of ours.
        from .viewer3d import SkeletonViewer3D as _SkV  # noqa: PLC0415

        self.viewer = _SkV(self)
        self.viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Same gesture as the camera fullscreen preview: a clean click closes
        # (drag still orbits).  Double-click also closes, for backward compat.
        self.viewer.clicked.connect(self.close)
        self.viewer.double_clicked.connect(self.close)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.viewer)

        hint = QLabel(
            "3D Skeleton — full screen   —   click or press Esc to close",
            self.viewer,
        )
        hint.setStyleSheet(
            "background: rgba(0,0,0,160); color:#ddd; padding:6px 12px;"
            "border-radius:4px; font-size:13px;"
        )
        hint.adjustSize()
        hint.move(16, 16)
        hint.raise_()

        QShortcut(QKeySequence(Qt.Key.Key_F11), self, activated=self.close)

    def set_frame(self, joints: np.ndarray, conf: np.ndarray, world_frame: bool = False) -> None:
        self.viewer.set_frame(joints, conf, world_frame=world_frame)


class _FullscreenPreview(QDialog):
    """Frameless fullscreen dialog showing a single camera preview.

    Used to inspect focus by stretching the live frame to the whole screen.
    Click anywhere on the image, press Esc, or press F11 to close it.
    """

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setStyleSheet("background:#000;")
        self.setModal(False)

        self._label = _ClickableLabel("Waiting for next frame…")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color:#888; font-size:18px;")
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.clicked.connect(self.close)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label)

        # Tiny hint overlay (esc / click to close)
        self._hint = QLabel(f"{title}   —   click or press Esc to close", self._label)
        self._hint.setStyleSheet(
            "background: rgba(0,0,0,160); color:#ddd; padding:6px 12px;"
            "border-radius:4px; font-size:13px;"
        )
        self._hint.adjustSize()
        self._hint.move(16, 16)
        self._hint.raise_()

        # Esc closes (default for QDialog), bind F11 as an alt-shortcut.
        QShortcut(QKeySequence(Qt.Key.Key_F11), self, activated=self.close)

    def set_frame(self, bgr: np.ndarray) -> None:
        """Update the preview to a fresh BGR frame, scaled to the dialog size."""
        if bgr is None:
            return
        # Format_BGR888 consumes OpenCV's native byte order directly — no
        # per-frame channel-swap copy (bgr[:, :, ::-1] is strided + materialised).
        h, w = bgr.shape[:2]
        img = QImage(bgr.tobytes(), w, h, w * 3, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(img)
        target = self._label.size()
        if target.width() > 0 and target.height() > 0:
            # Fast (nearest) up-scale: smooth bilinear to fullscreen every frame
            # was the dominant cost here, and for a focus check nearest preserves
            # the true pixels (smoothing only blurs sharpness away).
            pix = pix.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        self._label.setPixmap(pix)


_COCO17_NAMES = [
    "nose",
    "L-eye",
    "R-eye",
    "L-ear",
    "R-ear",
    "L-shoulder",
    "R-shoulder",
    "L-elbow",
    "R-elbow",
    "L-wrist",
    "R-wrist",
    "L-hip",
    "R-hip",
    "L-knee",
    "R-knee",
    "L-ankle",
    "R-ankle",
]

try:
    import PySpin as _PySpin  # noqa: F401

    _PYSPIN_AVAILABLE = True
except ImportError:
    _PYSPIN_AVAILABLE = False


def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _set_preview(label: QLabel, bgr: np.ndarray) -> None:
    """Convert a BGR frame and display it scaled to fill the label (aspect-correct)."""
    pix = _bgr_to_pixmap(bgr)
    target = label.size()
    if target.width() > 0 and target.height() > 0:
        pix = pix.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    label.setPixmap(pix)


def _status_html(text: str, color: str) -> str:
    return f"<span style='color:{color};font-weight:bold;'>{text}</span>"


class CaptureTab(QWidget):
    """Always-visible live capture tab with step-by-step camera setup guidance."""

    recording_saved = Signal(str, str)  # left_avi_path, right_avi_path
    calibration_saved = Signal(str)  # calibration.yml path

    #: Suggested capture rate during calibration — board detection is CPU-heavy
    #: and can't keep up with 50 fps, so the camera buffer drops frames.  ~20 fps
    #: leaves ample headroom while still giving smooth board feedback.
    _CALIB_FPS = 20.0

    # Pose-diversity grid constants
    _CALIB_GRID = 4  # divide each image axis into this many cells
    _CALIB_MAX_PER_CELL = 3  # max captures whose board centroid falls in one cell
    _CALIB_MIN_COV = 0.04  # board must cover ≥ 4 % of image area

    # Live-pose triangulation health: median reprojection error above this (px)
    # means the stereo calibration can't place this working volume — the 2D pose
    # is fine but the 3D joints get rejected ("dots, no skeleton").  ~15-20 px is
    # the geometry-limited floor on a high-vergence rig (ADR-006); 40 px is well
    # past anything a usable calibration produces.
    _POSE_REPROJ_WARN_PX = 40.0

    # "Set coordinate system" votes the board pose over this many live frames.
    # Single-view planar PnP flips the recovered normal ~110° on ~10 % of frames
    # with NO reprojection-error signature (measured on hardware — the flips are
    # sporadic, max 2 consecutive), so a majority vote over a short burst removes
    # them where a single-frame estimate would silently store a garbage frame.
    _COORDSYS_FRAMES = 12

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._worker: CaptureWorker | None = None
        self._project_dir: str | None = None
        self._is_recording = False
        self._detected_serials: list[str] = []
        self._sync_history: deque = deque(maxlen=30)  # signed deltas µs

        # Live calibration state — three-pile design for fisheye-friendly capture:
        #   _calib_left_dets  : every frame where LEFT detected the board (intrinsics K1,D1)
        #   _calib_right_dets : every frame where RIGHT detected the board (intrinsics K2,D2)
        #   _calib_pairs      : frames where BOTH detected simultaneously (extrinsics R,T)
        # A single capture event may add to 1, 2, or all 3 piles depending on what detected.
        self._calib_left_dets: list = []  # list[DetectionResult]
        self._calib_right_dets: list = []  # list[DetectionResult]
        self._calib_pairs: list = []  # list[FrameSelection]
        # Separate pose-diversity grids per pile so each is independently well-spread.
        self._calib_left_grid: dict[tuple[int, int], int] = {}
        self._calib_right_grid: dict[tuple[int, int], int] = {}
        self._det_left = None  # DetectionResult | None
        self._det_right = None
        # Frames whose pixel data MATCHES the current _det_left/_det_right.
        # The thread pool detects frame N; by the time the result arrives
        # self._last_frame has advanced to N+k.  Using the wrong frame would
        # pair correct corner coords with a different image (conceptual mismatch).
        self._detect_submitted_fl: np.ndarray | None = None
        self._detect_submitted_fr: np.ndarray | None = None
        self._detect_frame_fl: np.ndarray | None = None  # matched to current detection
        self._detect_frame_fr: np.ndarray | None = None
        self._last_frame = None  # most recent CaptureFrame (for re-render)
        self._last_auto_capture_time: float = 0.0
        # Rate-limit live board detection.  ChArUco detection at full 1920×1200
        # with the fisheye-tuned params is very CPU-heavy; running it back-to-back
        # saturates the cores and starves the capture-read thread, so the camera's
        # NewestOnly buffer drops frames (shown as "Drops").  ~5 detections/sec is
        # ample for live feedback / auto-capture and leaves cores for the reader.
        self._last_detect_submit: float = 0.0
        self._detect_min_interval_s: float = 0.2
        # Calibration fps override: calibration runs best at a lower fps (heavy
        # board detection can't keep up with 50 fps → dropped frames).  When the
        # user accepts the suggestion we drop to _CALIB_FPS for the duration of
        # calibration and restore their fps on exit.  _precalib_fps holds the
        # original; _calib_fps_active flags that a restore is pending.
        self._calib_fps_active: bool = False
        self._precalib_fps: float | None = None
        self._detector = None  # BoardDetector | None
        # Path of the calibration.yml saved this session (or found on open) —
        # gates the "Set coordinate system" button and is the file patched with
        # the floor world frame.
        self._calib_saved_path: str | None = None
        # Projected world-frame axis triad [(px_left[4,2], px_right[4,2])] or None.
        # The board (hence the world frame) is fixed relative to the stationary
        # cameras, so the projected pixels are constant — compute once, redraw the
        # triad on every live preview frame (a one-shot draw is overwritten by the
        # next incoming frame).  Set whenever a calibration with a world_frame is
        # loaded or "Set coordinate system" succeeds.
        self._world_axes_px: tuple[np.ndarray, np.ndarray] | None = None
        # "Set coordinate system" async state: while collecting, a list of
        # (frame_left, frame_right) copies fed by _on_frame_ready; the pooled
        # vote job runs off the GUI thread and is polled by a timer (frames may
        # stop flowing — e.g. user stops the stream — so polling can't rely on
        # frame ticks).
        self._coordsys_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None
        self._coordsys_future: _Future | None = None
        self._coordsys_ctx: dict | None = None  # calib + board numbers for the job
        self._coordsys_poll = QTimer(self)
        self._coordsys_poll.setInterval(100)
        self._coordsys_poll.timeout.connect(self._poll_coordsys)
        # The outer pool only ever has one in-flight orchestration task at a time;
        # max_workers=1 is fine here.  The actual left/right parallelism happens
        # inside _detect_pair via _detect_lr_pool below.
        self._detect_pool = _ThreadPool(max_workers=1, thread_name_prefix="board_det")
        self._detect_future: _Future | None = None
        self._pool_shutdown: bool = False
        # Fullscreen preview dialog ("L" or "R" side, or None when closed).
        self._fs_preview: _FullscreenPreview | None = None
        self._fs_side: str | None = None
        # Fullscreen 3D-skeleton dialog (None when closed).
        self._fs_3d: _Fullscreen3DSkeleton | None = None

        # ── Live Pose Tracking state ───────────────────────────────────────
        # Mutually exclusive with calibration mode (toggling one closes the other).
        self._pose_active: bool = False
        # SEPARATE VIDEO-mode landmarker per camera.  VIDEO mode keeps a temporal
        # track that sustains detection through frames the one-shot detector
        # would miss (it detects far more reliably than stateless IMAGE mode).
        # A single VIDEO tracker can't be shared across the two views (alternating
        # frames corrupt its track), so each camera gets its own instance — these
        # do NOT interfere (verified).  The harder of the two views needs a lower
        # detection-confidence threshold to get its initial lock (see _POSE_CONF).
        self._pose_backend = None  # left-camera PoseBackend
        self._pose_backend_r = None  # right-camera PoseBackend
        self._pose_calib: dict | None = None  # loaded calibration.yml as dict
        # Single in-flight pose-detection future, mirrors the ChArUco pattern.
        self._pose_future: _Future | None = None
        self._pose_submitted_fl: np.ndarray | None = None
        self._pose_submitted_fr: np.ndarray | None = None
        # OneEuro filters per (person, joint, axis), lazily allocated.
        self._pose_filters: list | None = None
        # Last computed 2D/3D pose, used to repaint the overlay between detections.
        self._pose_last_kp_l: np.ndarray | None = None  # [P, 17, 2]
        self._pose_last_kp_r: np.ndarray | None = None
        self._pose_last_conf_l: np.ndarray | None = None  # [P, 17]
        self._pose_last_conf_r: np.ndarray | None = None
        self._pose_last_3d: np.ndarray | None = None  # [P, 17, 3] CAMERA frame
        self._pose_last_conf_3d: np.ndarray | None = None
        # Display copy of the last 3D in the floor world frame (when the
        # calibration defines one) — _pose_last_3d must stay camera-frame
        # because the smoothing hold-last fallback feeds back into the
        # camera-frame stream.
        self._pose_last_view_3d: np.ndarray | None = None
        self._pose_world: bool = False  # live joints shown in the world frame?
        # Track outcome of last 20 detection cycles: 'both' / 'left' / 'right' / 'none'.
        # Used to render the stability indicator under the L/R status labels.
        self._detect_history: deque = deque(maxlen=20)
        self._calib_worker = None  # LiveCalibWorker | None
        # 4×4 pose-diversity grid: tracks how many captures fall in each cell.
        # Capped at _CALIB_MAX_PER_CELL so the user must move the board.
        # Used only for the stereo-pair pile.  Per-camera intrinsics use
        # _calib_left_grid / _calib_right_grid declared above.
        self._calib_grid: dict[tuple[int, int], int] = {}

        self._setup_ui()
        self._refresh_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Stop any running capture thread synchronously.

        Must be called before the application window is destroyed (e.g. from
        MainWindow.closeEvent).  Without this the QThread C++ object is torn
        down while the thread is still blocked inside GetNextImage(), producing
        "QThread: Destroyed while thread '' is still running".
        """
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            # Wait up to 3 s — FlirCapture.stop() needs time to call
            # EndAcquisition / DeInit / ReleaseInstance.
            if not self._worker.wait(3000):
                self._worker.terminate()  # last resort; avoids the Qt warning
                self._worker.wait(1000)
        self._worker = None
        # Close fullscreen preview / 3D pop-out if they're open — they would
        # otherwise outlive their parent.
        if self._fs_preview is not None:
            self._fs_preview.close()
        if self._fs_3d is not None:
            self._fs_3d.close()
        # Stop live pose tracking and release the backend's C++ resources.
        if self._pose_active:
            self._stop_pose_tracking()
        # Shut down the board-detection thread pools gracefully.
        self._detect_pool.shutdown(wait=False)
        CaptureTab._detect_lr_pool.shutdown(wait=False)
        self._pool_shutdown = True
        # Drop any in-flight "Set coordinate system" work.
        self._coordsys_poll.stop()
        self._coordsys_pairs = None
        self._coordsys_future = None

    def set_project_dir(self, path: str) -> None:
        self._project_dir = path
        self._out_edit.setText(str(Path(path) / "capture"))
        self._calib_out_edit.setText(str(Path(path) / "calibration.yml"))

    def session_state(self) -> dict:
        return {
            "serials": self._detected_serials,
            "left_serial": self._left_combo.currentText(),
            "right_serial": self._right_combo.currentText(),
            "primary": "left" if self._primary_left.isChecked() else "right",
            "sync": self._sync_check.isChecked(),
            # Persist the user's real fps, not a transient calibration override.
            "fps": self._precalib_fps if self._calib_fps_active else self._fps_spin.value(),
            "exposure_us": self._exposure_spin.value(),
            "gain_db": self._gain_spin.value(),
            "out_folder": self._out_edit.text(),
            "prefix": self._prefix_edit.text(),
        }

    def restore_session(self, state: dict) -> None:
        serials = state.get("serials", [])
        if serials:
            self._detected_serials = serials
            self._populate_serial_combos(serials)

        left = state.get("left_serial", "")
        right = state.get("right_serial", "")
        if left:
            idx = self._left_combo.findText(left)
            if idx >= 0:
                self._left_combo.setCurrentIndex(idx)
        if right:
            idx = self._right_combo.findText(right)
            if idx >= 0:
                self._right_combo.setCurrentIndex(idx)

        if state.get("primary") == "right":
            self._primary_right.setChecked(True)
        else:
            self._primary_left.setChecked(True)

        self._sync_check.setChecked(bool(state.get("sync", True)))
        if fps := state.get("fps"):
            self._fps_spin.setValue(fps)
        if exp := state.get("exposure_us"):
            self._exposure_spin.setValue(exp)
        if gain := state.get("gain_db"):
            self._gain_spin.setValue(gain)
        if out := state.get("out_folder"):
            self._out_edit.setText(out)
        # Restore a custom prefix, but treat the legacy "session" default as
        # empty so older sessions adopt the new auto-timestamp naming.
        if (prefix := state.get("prefix")) and prefix != "session":
            self._prefix_edit.setText(prefix)

        self._refresh_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        # Root is vertical so the calibration panel can sit at the bottom
        # spanning the full width of the tab.
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Top area: controls (left) + preview (right) ──────────────
        main_area = QWidget()
        main_layout = QHBoxLayout(main_area)
        main_layout.setContentsMargins(6, 6, 6, 4)

        # ── Left panel (controls + setup wizard) ─────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(400)
        ctrl_outer = QVBoxLayout(ctrl)
        ctrl_outer.setContentsMargins(0, 0, 0, 0)

        # Scrollable inner area (so the wizard steps don't clip)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        ctrl_layout = QVBoxLayout(inner)
        ctrl_layout.setContentsMargins(0, 4, 0, 4)
        scroll.setWidget(inner)
        ctrl_outer.addWidget(scroll, 1)

        # ── FLIR area ────────────────────────────────────────────────
        self._flir_area = QWidget()
        flir_layout = QVBoxLayout(self._flir_area)
        flir_layout.setContentsMargins(0, 0, 0, 0)

        # Status row
        status_group = QGroupBox("Status")
        status_form = QFormLayout(status_group)
        self._sdk_label = QLabel()
        self._cam_label = QLabel()
        self._sync_label = QLabel()
        status_form.addRow("PySpin SDK:", self._sdk_label)
        status_form.addRow("Cameras:", self._cam_label)
        status_form.addRow("Sync:", self._sync_label)
        flir_layout.addWidget(status_group)

        # ── Step 1: SDK not installed ────────────────────────────────
        self._step1 = QGroupBox("Step 1 — Install Spinnaker SDK")
        step1_layout = QVBoxLayout(self._step1)
        step1_text = QLabel(
            "<b>The PySpin (Spinnaker) SDK is not installed.</b><br><br>"
            "<b>⚠ Do NOT use <tt>pip install spinnaker-python</tt></b> — "
            "that package on PyPI is outdated and will fail.<br><br>"
            "<b>Correct install steps:</b><br><br>"
            "1. Create a free account and log in at:<br>"
            "&nbsp;&nbsp;<tt>teledynevisionsolutions.com</tt><br><br>"
            "2. Go to <i>Support → Software &amp; Firmware Downloads</i><br><br>"
            "3. Download <b>two</b> items with matching version numbers:<br>"
            "&nbsp;&nbsp;• <b>Spinnaker SDK</b> — C++ runtime and USB drivers<br>"
            "&nbsp;&nbsp;• <b>PySpin</b> — the Python wheel "
            "(listed separately; look for 'PySpin', not 'spinnaker_python')<br><br>"
            "4. <b>Python version note:</b> PySpin for Windows currently supports "
            "only up to <b>Python 3.10</b>. Run <tt>python --version</tt> to check.<br>"
            "&nbsp;&nbsp;If you are on Python 3.11/3.12, create a 3.10 environment:<br>"
            "&nbsp;&nbsp;<tt>uv venv .venv --python 3.10</tt><br><br>"
            "5. Install the SDK runtime first, then the wheel:<br>"
            "&nbsp;&nbsp;<tt>pip install spinnaker_python-4.x.x.x-cp310-...-win_amd64.whl</tt><br><br>"
            "6. Restart BatPose<br><br>"
            "<i>Until then, live capture is unavailable.</i>"
        )
        step1_text.setWordWrap(True)
        step1_text.setTextFormat(Qt.TextFormat.RichText)
        step1_layout.addWidget(step1_text)
        flir_layout.addWidget(self._step1)

        # ── Step 2: SDK found — connect & detect ─────────────────────
        self._step2 = QGroupBox("Step 2 — Connect cameras")
        step2_layout = QVBoxLayout(self._step2)
        step2_text = QLabel(
            "Connect both BlackflyS cameras via <b>USB 3.0</b>, then click "
            "<i>Detect cameras</i>.<br><br>"
            "<b>Hardware sync wiring (6-pin GPIO):</b><br><br>"
            "<b>Signal wire:</b><br>"
            "&nbsp;• Primary &nbsp;<b>pin 4</b> (white, Line 1 OPTOOUT)<br>"
            "&nbsp;&nbsp;&nbsp;→ Secondary <b>pin 1</b> (green, Line 3 GPI)<br><br>"
            "<b>Ground wire:</b><br>"
            "&nbsp;• Primary &nbsp;<b>pin 5</b> (blue, opto-GND)<br>"
            "&nbsp;&nbsp;&nbsp;→ Secondary <b>pin 6</b> (brown, GND)<br><br>"
            "<b>Pull-up resistor (10 kΩ, required):</b><br>"
            "&nbsp;• Primary <b>pin 3</b> (red, 3.3 V) → resistor → primary pin 4<br>"
            "&nbsp;&nbsp;&nbsp;and → secondary pin 1<br><br>"
            "<i>BatPose enables the 3.3 V rail and configures Line 1 as strobe "
            "output automatically when you click Start Preview.</i>"
        )
        step2_text.setWordWrap(True)
        step2_text.setTextFormat(Qt.TextFormat.RichText)
        step2_layout.addWidget(step2_text)

        detect_row = QHBoxLayout()
        self._detect_btn = QPushButton("Detect cameras")
        self._detect_btn.clicked.connect(self._on_detect)
        detect_row.addWidget(self._detect_btn)
        detect_row.addStretch()
        step2_layout.addLayout(detect_row)

        self._detect_msg = QLabel("")
        self._detect_msg.setWordWrap(True)
        step2_layout.addWidget(self._detect_msg)
        flir_layout.addWidget(self._step2)

        # ── Step 3: Assign roles + configure ─────────────────────────
        self._step3 = QGroupBox("Step 3 — Assign roles and configure")
        step3_form = QFormLayout(self._step3)

        self._left_combo = QComboBox()
        self._right_combo = QComboBox()
        self._left_combo.currentTextChanged.connect(self._on_left_camera_changed)
        self._right_combo.currentTextChanged.connect(self._on_right_camera_changed)
        step3_form.addRow("Left camera serial:", self._left_combo)
        step3_form.addRow("Right camera serial:", self._right_combo)

        primary_row = QHBoxLayout()
        self._primary_left = QRadioButton("Left")
        self._primary_right = QRadioButton("Right")
        self._primary_left.setChecked(True)
        self._primary_left.toggled.connect(self._refresh_state)
        primary_row.addWidget(self._primary_left)
        primary_row.addWidget(self._primary_right)
        primary_row.addStretch()
        step3_form.addRow("Primary camera:", primary_row)

        self._sync_check = QCheckBox("Use hardware sync (Line 2 → Line 3)")
        self._sync_check.setChecked(True)
        self._sync_check.toggled.connect(self._refresh_state)
        step3_form.addRow("", self._sync_check)

        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 200.0)
        self._fps_spin.setValue(30.0)
        self._fps_spin.setSuffix(" fps")
        step3_form.addRow("Frame rate:", self._fps_spin)

        self._exposure_spin = QDoubleSpinBox()
        self._exposure_spin.setRange(10.0, 500_000.0)
        self._exposure_spin.setValue(5000.0)
        self._exposure_spin.setSuffix(" µs")
        step3_form.addRow("Exposure:", self._exposure_spin)

        self._gain_spin = QDoubleSpinBox()
        self._gain_spin.setRange(0.0, 47.9)
        self._gain_spin.setValue(10.0)
        self._gain_spin.setSuffix(" dB")
        self._gain_spin.setSingleStep(1.0)
        self._gain_spin.setDecimals(1)
        self._gain_spin.setToolTip(
            "Analogue gain applied to both cameras equally.\n"
            "0 dB = sensor floor (may be too dark indoors).\n"
            "10–15 dB is typical for indoor fluorescent lighting."
        )
        step3_form.addRow("Gain:", self._gain_spin)

        flir_layout.addWidget(self._step3)
        ctrl_layout.addWidget(self._flir_area)

        # ── Output ───────────────────────────────────────────────────
        out_group = QGroupBox("Recording output")
        out_form = QFormLayout(out_group)
        self._out_edit = QLineEdit("capture")
        btn_out = QPushButton("Browse…")
        btn_out.setFixedWidth(70)
        btn_out.clicked.connect(self._browse_output)
        row_out = QHBoxLayout()
        row_out.addWidget(self._out_edit)
        row_out.addWidget(btn_out)
        self._prefix_edit = QLineEdit()
        self._prefix_edit.setPlaceholderText("auto: YYYYMMDD_HHMMSS (leave empty)")
        self._prefix_edit.setToolTip(
            "Filename prefix for the recorded pair and timestamp sidecar.\n"
            "Leave empty to auto-name each recording with the start time, e.g.\n"
            "20260126_143512_left.avi / _right.avi / _timestamps.csv."
        )
        out_form.addRow("Output folder:", row_out)
        out_form.addRow("File prefix:", self._prefix_edit)
        ctrl_layout.addWidget(out_group)

        ctrl_layout.addStretch()

        # ── Stream controls (below scroll area) ──────────────────────
        stream_widget = QWidget()
        stream_layout = QVBoxLayout(stream_widget)
        stream_layout.setContentsMargins(0, 4, 0, 0)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Preview")
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        stream_layout.addLayout(btn_row)

        self._record_btn = QPushButton("Start Recording")
        self._record_btn.setEnabled(False)
        self._record_btn.setCheckable(True)
        self._record_btn.clicked.connect(self._on_record_toggle)
        stream_layout.addWidget(self._record_btn)

        self._calib_mode_btn = QPushButton("🎯  Calibration Mode")
        self._calib_mode_btn.setCheckable(True)
        self._calib_mode_btn.setEnabled(False)
        self._calib_mode_btn.setToolTip(
            "Open the live calibration panel — capture board positions "
            "from both cameras, then run stereo calibration."
        )
        # Use clicked (not toggled) so we can intercept and show a discard warning.
        self._calib_mode_btn.clicked.connect(self._on_calib_mode_clicked)
        stream_layout.addWidget(self._calib_mode_btn)

        self._pose_mode_btn = QPushButton("🧍  Live Pose")
        self._pose_mode_btn.setCheckable(True)
        self._pose_mode_btn.setEnabled(False)
        self._pose_mode_btn.setToolTip(
            "Run 2D pose detection on both cameras and triangulate to 3D in real "
            "time. Requires a calibration.yml."
        )
        self._pose_mode_btn.clicked.connect(self._on_pose_mode_clicked)
        stream_layout.addWidget(self._pose_mode_btn)

        self._status_label = QLabel("Idle")
        self._status_label.setStyleSheet("color: #888;")
        stream_layout.addWidget(self._status_label)
        ctrl_outer.addWidget(stream_widget)

        main_layout.addWidget(ctrl)

        # ── Right panel: live preview ─────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_group = QGroupBox("Live Preview")
        preview_layout = QHBoxLayout(preview_group)

        self._preview_left = _ClickableLabel("No frame")
        self._preview_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_left.setMinimumSize(320, 240)
        self._preview_left.setStyleSheet("background:#1a1a1a; color:#888;")
        self._preview_left.setToolTip("Click to inspect at full screen (Esc to close)")
        self._preview_left.clicked.connect(lambda: self._open_fullscreen_preview("L"))

        self._preview_right = _ClickableLabel("No frame")
        self._preview_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_right.setMinimumSize(320, 240)
        self._preview_right.setStyleSheet("background:#1a1a1a; color:#888;")
        self._preview_right.setToolTip("Click to inspect at full screen (Esc to close)")
        self._preview_right.clicked.connect(lambda: self._open_fullscreen_preview("R"))

        # 3D skeleton preview, sized to match the camera previews.  Visible
        # only while Live Pose tracking is running; hidden otherwise so it
        # doesn't steal layout space from the camera images.
        self._preview_3d = SkeletonViewer3D()
        self._preview_3d.setMinimumSize(320, 240)
        self._preview_3d.setVisible(False)
        # Single clean click opens fullscreen — same gesture as the camera
        # previews; drag still orbits.  (Not double-click: a double-click would
        # fire `clicked` on its first release and then toggle straight back.)
        self._preview_3d.clicked.connect(self._open_fullscreen_3d)

        # Equal stretch so each panel gets a fair share of the width.  The 3D
        # viewer (Expanding) then fills its third instead of sitting at its
        # 320 px minimum while the camera labels hog the row.  When the 3D
        # viewer is hidden (not tracking) the two cameras split the full width.
        preview_layout.addWidget(self._preview_left, 1)
        preview_layout.addWidget(self._preview_right, 1)
        preview_layout.addWidget(self._preview_3d, 1)
        right_layout.addWidget(preview_group, 1)

        # Sync-quality bar — shows hardware timestamp delta per frame pair
        sync_row = QHBoxLayout()
        sync_row.addWidget(QLabel("Sync Δt:"))
        self._sync_delta_label = QLabel("—")
        self._sync_delta_label.setMinimumWidth(120)
        sync_row.addWidget(self._sync_delta_label)
        self._sync_quality_label = QLabel("")
        self._sync_quality_label.setMinimumWidth(160)
        sync_row.addWidget(self._sync_quality_label)
        sync_row.addWidget(QLabel("  Drops:"))
        self._drop_label = QLabel("0")
        self._drop_label.setMinimumWidth(40)
        sync_row.addWidget(self._drop_label)
        sync_row.addStretch()
        self._rec_status = QLabel("")
        self._rec_status.setStyleSheet("color:#e74c3c; font-weight:bold;")
        sync_row.addWidget(self._rec_status)
        right_layout.addLayout(sync_row)

        main_layout.addWidget(right, 1)
        root.addWidget(main_area, 1)

        # ── Calibration panel (full width, bottom, hidden by default) ─────
        self._calib_panel = QFrame()
        self._calib_panel.setObjectName("calibPanel")
        self._calib_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self._calib_panel.setStyleSheet(
            "QFrame#calibPanel {  background-color: #1c2b3a;  border-top: 2px solid #2980b9;}"
        )
        self._calib_panel.setVisible(False)

        panel_v = QVBoxLayout(self._calib_panel)
        panel_v.setSpacing(6)
        panel_v.setContentsMargins(10, 8, 10, 10)

        # Panel header
        hdr_row = QHBoxLayout()
        panel_title = QLabel("📐  Live Calibration")
        panel_title.setStyleSheet("font-size:14px; font-weight:bold; color:#3498db;")
        hdr_row.addWidget(panel_title)

        panel_tip = QLabel(
            "<small>Move the board to <b>all areas</b> of the frame "
            "(corners, edges, center) and tilt it at various angles. "
            "The 4×4 grid enforces coverage — each cell accepts up to 3 captures.</small>"
        )
        panel_tip.setTextFormat(Qt.TextFormat.RichText)
        panel_tip.setWordWrap(True)
        hdr_row.addWidget(panel_tip, 2)

        self._calib_close_btn = QPushButton("✕  Close Calibration")
        self._calib_close_btn.setFixedWidth(140)
        self._calib_close_btn.setToolTip(
            "Close the calibration panel. Captured frames are kept until you "
            "click 'Clear All' or re-open and confirm discard."
        )
        self._calib_close_btn.clicked.connect(self._on_calib_close_clicked)
        hdr_row.addWidget(self._calib_close_btn)
        panel_v.addLayout(hdr_row)

        # ── 3-column body ─────────────────────────────────────────────────
        cols = QHBoxLayout()
        cols.setSpacing(10)

        # ---- Column 1: Board configuration ----
        board_col = QGroupBox("Board Configuration")
        board_col.setMinimumWidth(220)
        board_v = QVBoxLayout(board_col)
        board_v.setSpacing(4)

        board_type_row = QHBoxLayout()
        board_type_row.addWidget(QLabel("Type:"))
        self._calib_board_combo = QComboBox()
        self._calib_board_combo.addItems(["ChArUco (recommended)", "Chessboard"])
        self._calib_board_combo.currentIndexChanged.connect(self._on_calib_board_type)
        board_type_row.addWidget(self._calib_board_combo)
        board_v.addLayout(board_type_row)

        self._calib_board_stack = QStackedWidget()

        charuco_w = QWidget()
        cf = QFormLayout(charuco_w)
        cf.setContentsMargins(0, 0, 0, 0)
        cf.setSpacing(3)
        self._calib_sq_x = QSpinBox()
        self._calib_sq_x.setRange(3, 20)
        self._calib_sq_x.setValue(5)
        self._calib_sq_y = QSpinBox()
        self._calib_sq_y.setRange(3, 20)
        self._calib_sq_y.setValue(7)
        self._calib_sq_size = QDoubleSpinBox()
        self._calib_sq_size.setRange(0.001, 1.0)
        self._calib_sq_size.setDecimals(4)  # before setValue → no rounding to 2 dp
        self._calib_sq_size.setValue(0.04)
        self._calib_sq_size.setSuffix(" m")
        self._calib_mk_size = QDoubleSpinBox()
        self._calib_mk_size.setRange(0.001, 1.0)
        self._calib_mk_size.setDecimals(4)  # before setValue → keeps 0.024 (not 0.02)
        self._calib_mk_size.setValue(0.024)
        self._calib_mk_size.setSuffix(" m")
        self._calib_aruco_dict = QComboBox()
        self._calib_aruco_dict.addItems(
            ["DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_6X6_250"]
        )
        self._calib_aruco_dict.setCurrentText("DICT_6X6_250")
        cf.addRow("Squares X:", self._calib_sq_x)
        cf.addRow("Squares Y:", self._calib_sq_y)
        cf.addRow("Square size:", self._calib_sq_size)
        cf.addRow("Marker size:", self._calib_mk_size)
        cf.addRow("ArUco dict:", self._calib_aruco_dict)
        self._calib_board_stack.addWidget(charuco_w)

        chess_w = QWidget()
        chf = QFormLayout(chess_w)
        chf.setContentsMargins(0, 0, 0, 0)
        chf.setSpacing(3)
        self._calib_chess_cols = QSpinBox()
        self._calib_chess_cols.setRange(3, 20)
        self._calib_chess_cols.setValue(9)
        self._calib_chess_cols.setToolTip("Inner corner columns (total squares − 1)")
        self._calib_chess_rows = QSpinBox()
        self._calib_chess_rows.setRange(3, 20)
        self._calib_chess_rows.setValue(6)
        self._calib_chess_rows.setToolTip("Inner corner rows (total squares − 1)")
        self._calib_chess_sq_size = QDoubleSpinBox()
        self._calib_chess_sq_size.setRange(0.001, 1.0)
        self._calib_chess_sq_size.setDecimals(4)  # before setValue → keeps 0.025
        self._calib_chess_sq_size.setValue(0.025)
        self._calib_chess_sq_size.setSuffix(" m")
        chf.addRow("Cols (inner):", self._calib_chess_cols)
        chf.addRow("Rows (inner):", self._calib_chess_rows)
        chf.addRow("Square size:", self._calib_chess_sq_size)
        self._calib_board_stack.addWidget(chess_w)
        board_v.addWidget(self._calib_board_stack)

        # Recreate the board detector whenever any parameter changes.
        # This makes editing live — no need to stop/restart the preview.
        for _sig in (
            self._calib_sq_x.valueChanged,
            self._calib_sq_y.valueChanged,
            self._calib_sq_size.valueChanged,
            self._calib_mk_size.valueChanged,
            self._calib_aruco_dict.currentIndexChanged,
            self._calib_chess_cols.valueChanged,
            self._calib_chess_rows.valueChanged,
            self._calib_chess_sq_size.valueChanged,
        ):
            _sig.connect(self._on_board_params_changed)

        # Lens / distortion model
        board_v.addSpacing(6)
        lens_row = QHBoxLayout()
        lens_row.addWidget(QLabel("Lens:"))
        self._calib_lens_combo = QComboBox()
        self._calib_lens_combo.addItems(
            [
                "Standard  (normal lenses)",
                "Wide-angle  (rational model)",
                "Fisheye  (≥ 150° FOV)",
            ]
        )
        self._calib_lens_combo.setToolTip(
            "Standard — 5 distortion coefficients, best for lenses with ≤ 90° FOV.\n"
            "Wide-angle — 8 coefficients (rational model), use for 90–150° FOV.\n"
            "Fisheye — OpenCV fisheye model (θ-based), use for > 150° FOV.\n\n"
            "If RMS > 1.5 px your lens is probably wider than the selected model allows."
        )
        self._calib_lens_combo.setCurrentIndex(2)  # default Fisheye (the deployed rig)
        lens_row.addWidget(self._calib_lens_combo)
        board_v.addLayout(lens_row)

        # Output path
        board_v.addSpacing(6)
        board_v.addWidget(QLabel("Save calibration to:"))
        out_calib_row = QHBoxLayout()
        self._calib_out_edit = QLineEdit("calibration.yml")
        btn_calib_out = QPushButton("…")
        btn_calib_out.setFixedWidth(28)
        btn_calib_out.clicked.connect(self._browse_calib_output)
        out_calib_row.addWidget(self._calib_out_edit)
        out_calib_row.addWidget(btn_calib_out)
        board_v.addLayout(out_calib_row)
        board_v.addStretch()

        cols.addWidget(board_col, 1)

        # ---- Column 2: Detection status ----
        det_col = QGroupBox("Detection Status")
        det_v = QVBoxLayout(det_col)
        det_v.setSpacing(10)

        self._calib_left_status = QLabel("LEFT: —")
        self._calib_left_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._calib_left_status.setStyleSheet(
            "font-weight: bold; font-size: 18px; color: #888;"
            "background: #111; border-radius: 6px; padding: 14px;"
        )
        self._calib_right_status = QLabel("RIGHT: —")
        self._calib_right_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._calib_right_status.setStyleSheet(
            "font-weight: bold; font-size: 18px; color: #888;"
            "background: #111; border-radius: 6px; padding: 14px;"
        )
        det_v.addWidget(self._calib_left_status)
        det_v.addWidget(self._calib_right_status)
        det_v.addStretch()

        cols.addWidget(det_col, 1)

        # ---- Column 3: Capture & Calibrate ----
        cap_col = QGroupBox("Capture && Calibrate")
        cap_v = QVBoxLayout(cap_col)
        cap_v.setSpacing(6)

        # Three independent counters — two-stage stereo calibration:
        #   - Left intrinsics:  capture any frame where LEFT cam detects (target 20+)
        #   - Right intrinsics: capture any frame where RIGHT cam detects (target 20+)
        #   - Stereo extrinsics: only when both cameras detect simultaneously (target 6+)
        def _mk_bar(target: int, fmt: str) -> QProgressBar:
            b = QProgressBar()
            b.setRange(0, target)
            b.setValue(0)
            b.setFormat(fmt)
            b.setTextVisible(True)
            return b

        self._calib_left_bar = _mk_bar(20, "Left intrinsics: 0 / 20 frames")
        self._calib_right_bar = _mk_bar(20, "Right intrinsics: 0 / 20 frames")
        self._calib_stereo_bar = _mk_bar(8, "Stereo pairs: 0 / 8 (need ≥ 6)")
        cap_v.addWidget(self._calib_left_bar)
        cap_v.addWidget(self._calib_right_bar)
        cap_v.addWidget(self._calib_stereo_bar)

        # Auto-capture controls
        auto_row = QHBoxLayout()
        self._calib_auto_check = QCheckBox("Auto-capture")
        self._calib_auto_check.setChecked(True)
        self._calib_auto_check.setToolTip(
            "Automatically capture whenever EITHER camera detects the board "
            "(and the cooldown has elapsed). Per-camera detections feed the "
            "intrinsics pools; simultaneous detections also feed the stereo pool."
        )
        auto_row.addWidget(self._calib_auto_check)
        auto_row.addWidget(QLabel("cooldown:"))
        self._calib_cooldown_spin = QDoubleSpinBox()
        self._calib_cooldown_spin.setRange(0.5, 10.0)
        self._calib_cooldown_spin.setValue(1.5)
        self._calib_cooldown_spin.setSuffix(" s")
        self._calib_cooldown_spin.setFixedWidth(72)
        auto_row.addWidget(self._calib_cooldown_spin)
        auto_row.addStretch()
        cap_v.addLayout(auto_row)

        # Manual capture + clear row
        cap_btns = QHBoxLayout()
        self._calib_capture_btn = QPushButton("📷  Capture Frame")
        self._calib_capture_btn.setEnabled(False)
        self._calib_capture_btn.setMinimumHeight(36)
        self._calib_capture_btn.setToolTip(
            "Capture this frame — at least one camera must detect the board.\n"
            "Adds to per-camera intrinsics pools; if both detect, also to the\n"
            "stereo extrinsics pool."
        )
        self._calib_capture_btn.clicked.connect(self._do_capture_frame)
        self._calib_clear_btn = QPushButton("Clear All")
        self._calib_clear_btn.setToolTip("Discard all captured frame pairs and start over")
        self._calib_clear_btn.clicked.connect(self._on_calib_clear)
        cap_btns.addWidget(self._calib_capture_btn, 2)
        cap_btns.addWidget(self._calib_clear_btn, 1)
        cap_v.addLayout(cap_btns)

        # Run calibration
        self._calib_run_btn = QPushButton("▶  Run Calibration  (0 frames)")
        self._calib_run_btn.setEnabled(False)
        self._calib_run_btn.setMinimumHeight(32)
        self._calib_run_btn.clicked.connect(self._on_run_calib)
        cap_v.addWidget(self._calib_run_btn)

        # Define the floor world coordinate system from a board lying on the
        # floor (visible to both cameras).  Enabled only once a calibration
        # exists — it needs the intrinsics/extrinsics to triangulate the board.
        self._set_coord_btn = QPushButton("Set coordinate system")
        self._set_coord_btn.setEnabled(False)
        self._set_coord_btn.setMinimumHeight(28)
        self._set_coord_btn.setToolTip(
            "Place the calibration board flat on the floor where both cameras "
            "see it, then click.\nDefines the world origin at the board centre, "
            "X along its long side, Y along its short side, Z up.\n"
            "Stored in calibration.yml and used for all further analysis."
        )
        self._set_coord_btn.clicked.connect(self._on_set_coordinate_system)
        cap_v.addWidget(self._set_coord_btn)

        self._calib_progress_bar = QProgressBar()
        self._calib_progress_bar.setRange(0, 100)
        self._calib_progress_bar.setValue(0)
        self._calib_progress_bar.setVisible(False)
        cap_v.addWidget(self._calib_progress_bar)

        self._calib_status_label = QLabel("")
        self._calib_status_label.setWordWrap(True)
        self._calib_status_label.setTextFormat(Qt.TextFormat.RichText)
        cap_v.addWidget(self._calib_status_label)
        cap_v.addStretch()

        cols.addWidget(cap_col, 1)
        panel_v.addLayout(cols)

        root.addWidget(self._calib_panel)

        # ── Live Pose Tracking panel (parallel to calibration panel) ────────
        self._setup_pose_panel(root)

    # ------------------------------------------------------------------
    # Live Pose Tracking — UI construction
    # ------------------------------------------------------------------

    def _setup_pose_panel(self, root: QVBoxLayout) -> None:
        """Bottom-of-tab panel for live 2D→3D pose tracking."""
        self._pose_panel = QFrame()
        self._pose_panel.setObjectName("posePanel")
        self._pose_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self._pose_panel.setStyleSheet(
            "QFrame#posePanel { background:#162026; border:1px solid #2a3940; border-radius:6px; }"
        )
        self._pose_panel.setVisible(False)

        panel_v = QVBoxLayout(self._pose_panel)
        panel_v.setContentsMargins(12, 10, 12, 10)

        # Header row — title + close button + hint
        hdr = QHBoxLayout()
        title = QLabel("<b style='color:#2ecc71;'>🧍 Live Pose Tracking</b>")
        hdr.addWidget(title)
        hint = QLabel(
            "<span style='color:#aaa;'>Loads <i>calibration.yml</i>, runs MediaPipe on "
            "each camera and triangulates joints into 3D in real time.</span>"
        )
        hint.setWordWrap(True)
        hdr.addWidget(hint, 1)
        self._pose_close_btn = QPushButton("✕  Close Live Pose")
        self._pose_close_btn.clicked.connect(self._on_pose_close_clicked)
        hdr.addWidget(self._pose_close_btn)
        panel_v.addLayout(hdr)

        # Body: 3 columns — Settings | Status | 3D viewer
        cols = QHBoxLayout()

        # Column 1 — Settings
        set_col = QGroupBox("Settings")
        set_form = QFormLayout(set_col)

        calib_row = QHBoxLayout()
        self._pose_calib_edit = QLineEdit()
        self._pose_calib_edit.setPlaceholderText("Path to calibration.yml")
        calib_browse = QPushButton("…")
        calib_browse.setFixedWidth(32)
        calib_browse.clicked.connect(self._on_pose_browse_calib)
        calib_row.addWidget(self._pose_calib_edit)
        calib_row.addWidget(calib_browse)
        calib_row_w = QWidget()
        calib_row_w.setLayout(calib_row)
        set_form.addRow("Calibration:", calib_row_w)

        self._pose_backend_combo = QComboBox()
        self._pose_backend_combo.addItems(["MediaPipe (CPU)"])
        set_form.addRow("Backend:", self._pose_backend_combo)

        self._pose_num_spin = QSpinBox()
        self._pose_num_spin.setRange(1, 4)
        self._pose_num_spin.setValue(1)
        set_form.addRow("Persons:", self._pose_num_spin)

        self._pose_min_conf_spin = QDoubleSpinBox()
        self._pose_min_conf_spin.setRange(0.0, 1.0)
        self._pose_min_conf_spin.setSingleStep(0.05)
        self._pose_min_conf_spin.setValue(0.3)
        set_form.addRow("Min conf:", self._pose_min_conf_spin)

        self._pose_max_reproj_spin = QDoubleSpinBox()
        self._pose_max_reproj_spin.setRange(1.0, 200.0)
        self._pose_max_reproj_spin.setSingleStep(1.0)
        self._pose_max_reproj_spin.setValue(20.0)
        self._pose_max_reproj_spin.setSuffix(" px")
        set_form.addRow("Max reproj err:", self._pose_max_reproj_spin)

        self._pose_overlay_check = QCheckBox("Overlay 2D skeleton on previews")
        self._pose_overlay_check.setChecked(True)
        set_form.addRow(self._pose_overlay_check)

        self._pose_smooth_check = QCheckBox("Temporal smoothing (One-Euro)")
        self._pose_smooth_check.setChecked(True)
        set_form.addRow(self._pose_smooth_check)

        self._pose_start_btn = QPushButton("▶  Start Tracking")
        self._pose_start_btn.setCheckable(True)
        self._pose_start_btn.setMinimumHeight(34)
        self._pose_start_btn.clicked.connect(self._on_pose_start_clicked)
        set_form.addRow(self._pose_start_btn)
        cols.addWidget(set_col, 1)

        # Column 2 — Status
        stat_col = QGroupBox("Status")
        stat_v = QVBoxLayout(stat_col)
        stat_v.setSpacing(8)
        self._pose_status_label = QLabel("Idle")
        self._pose_status_label.setStyleSheet(
            "font-size: 14px; color:#aaa; background:#1a1a1a; padding:8px; border-radius:4px;"
        )
        self._pose_status_label.setWordWrap(True)
        stat_v.addWidget(self._pose_status_label)
        self._pose_metrics_label = QLabel("")
        self._pose_metrics_label.setStyleSheet("font-size: 12px; color:#888;")
        self._pose_metrics_label.setWordWrap(True)
        stat_v.addWidget(self._pose_metrics_label)
        stat_v.addStretch()
        cols.addWidget(stat_col, 1)

        # (3D viewer lives in the Live Preview row, not in this bottom panel —
        # see self._preview_3d.  Showing it twice would be wasteful.)

        panel_v.addLayout(cols)
        root.addWidget(self._pose_panel)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _refresh_state(self) -> None:
        # SDK status
        if _PYSPIN_AVAILABLE:
            self._sdk_label.setText(_status_html("✓ installed", "#2ecc71"))
        else:
            self._sdk_label.setText(_status_html("✗ not installed", "#e74c3c"))

        # Camera count
        n = len(self._detected_serials)
        if n == 0:
            self._cam_label.setText(_status_html("none detected — click Detect", "#e67e22"))
        elif n == 1:
            self._cam_label.setText(_status_html("1 found (need at least 2)", "#e67e22"))
        else:
            self._cam_label.setText(_status_html(f"{n} detected", "#2ecc71"))

        # Sync status
        if n < 2:
            self._sync_label.setText(_status_html("— need 2 cameras", "#888888"))
        elif self._sync_check.isChecked():
            primary = "left" if self._primary_left.isChecked() else "right"
            self._sync_label.setText(
                _status_html(f"✓ hardware sync — {primary} is primary", "#2ecc71")
            )
        else:
            self._sync_label.setText(_status_html("software sync only", "#e67e22"))

        # Show/hide steps
        self._step1.setVisible(not _PYSPIN_AVAILABLE)
        self._step2.setVisible(_PYSPIN_AVAILABLE)
        self._step3.setVisible(_PYSPIN_AVAILABLE and n >= 2)

        # Enable Start Preview only when ready
        can_start = self._worker is None and _PYSPIN_AVAILABLE and n >= 2
        self._start_btn.setEnabled(can_start)
        if not can_start and self._worker is None:
            if not _PYSPIN_AVAILABLE:
                self._status_label.setText("Install PySpin to enable FLIR cameras.")
            elif n < 2:
                self._status_label.setText("Detect at least 2 cameras to start.")

    def _populate_serial_combos(self, serials: list[str]) -> None:
        self._left_combo.blockSignals(True)
        self._right_combo.blockSignals(True)
        self._left_combo.clear()
        self._right_combo.clear()
        for s in serials:
            self._left_combo.addItem(s)
            self._right_combo.addItem(s)
        if len(serials) >= 2:
            self._left_combo.setCurrentIndex(0)
            self._right_combo.setCurrentIndex(1)
        self._left_combo.blockSignals(False)
        self._right_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Camera combo mutual-exclusion helpers
    # ------------------------------------------------------------------

    def _on_left_camera_changed(self, text: str) -> None:
        """If left camera = right camera, auto-switch right to a different serial.

        Prevents the user from accidentally selecting the same physical camera
        for both roles, which would produce a degenerate (identical) stereo pair
        and crash the FLIR acquisition with a duplicate-serial error.
        """
        if text and text == self._right_combo.currentText():
            self._right_combo.blockSignals(True)
            for i in range(self._right_combo.count()):
                if self._right_combo.itemText(i) != text:
                    self._right_combo.setCurrentIndex(i)
                    break
            self._right_combo.blockSignals(False)
        self._refresh_state()

    def _on_right_camera_changed(self, text: str) -> None:
        """If right camera = left camera, auto-switch left to a different serial."""
        if text and text == self._left_combo.currentText():
            self._left_combo.blockSignals(True)
            for i in range(self._left_combo.count()):
                if self._left_combo.itemText(i) != text:
                    self._left_combo.setCurrentIndex(i)
                    break
            self._left_combo.blockSignals(False)
        self._refresh_state()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_detect(self) -> None:
        self._detect_btn.setEnabled(False)
        self._detect_msg.setText("Detecting…")
        try:
            from app.capture.flir import list_camera_serials

            serials = list_camera_serials()
            self._detected_serials = serials
            self._populate_serial_combos(serials)

            if not serials:
                self._detect_msg.setText(
                    _status_html("No cameras found.", "#e74c3c")
                    + " Check USB connections and power."
                )
            elif len(serials) == 1:
                self._detect_msg.setText(
                    _status_html(f"1 camera found: {serials[0]}", "#e67e22")
                    + "<br>Connect a second camera for stereo capture."
                )
            else:
                self._detect_msg.setText(
                    _status_html(f"{len(serials)} cameras found: " + ", ".join(serials), "#2ecc71")
                )
        except Exception as exc:
            self._detect_msg.setText(_status_html(f"Detection failed: {exc}", "#e74c3c"))
        finally:
            self._detect_btn.setEnabled(True)
            self._refresh_state()

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self._out_edit.setText(path)

    def _on_start(self) -> None:
        # Warn only when exposure exceeds 95% of the frame period.
        # The camera firmware enforces the hard limit itself; this warning
        # catches configurations that leave almost no headroom for the
        # hardware trigger handshake on the secondary camera.
        if self._sync_check.isChecked():
            fps = self._fps_spin.value()
            exposure_us = self._exposure_spin.value()
            frame_period_us = 1_000_000.0 / fps
            usage_pct = exposure_us / frame_period_us * 100.0
            if usage_pct > 95.0:
                ans = QMessageBox.question(
                    self,
                    "Exposure near frame-period limit",
                    f"At {fps:.0f} fps the frame period is {frame_period_us / 1000:.1f} ms. "
                    f"A {exposure_us / 1000:.1f} ms exposure uses <b>{usage_pct:.0f}%</b> of "
                    f"that, leaving under 5% for the hardware trigger handshake.<br><br>"
                    "The secondary camera may miss triggers at this setting.<br><br>"
                    "Continue anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if ans == QMessageBox.StandardButton.No:
                    return

        source = self._build_source()
        if source is None:
            return

        self._make_and_start_worker(source)

        self._sync_history.clear()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._record_btn.setEnabled(True)
        self._calib_mode_btn.setEnabled(True)
        self._pose_mode_btn.setEnabled(True)
        # Lock camera assignment while streaming — changing serials mid-stream
        # would cause a mismatched pair without restarting the capture source.
        self._left_combo.setEnabled(False)
        self._right_combo.setEnabled(False)
        self._status_label.setText("Streaming…")

    def _make_and_start_worker(self, source) -> None:
        """Create, wire and start a CaptureWorker around *source*."""
        self._worker = CaptureWorker(source)
        # deleteLater must be connected BEFORE our finished slot so Qt has
        # already scheduled C++ cleanup by the time we drop the Python reference.
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.recording_finished.connect(self._on_recording_finished)
        self._worker.progress.connect(lambda _pct, msg: self._status_label.setText(msg))
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _restart_stream_at_fps(self, new_fps: float) -> bool:
        """Stop the current stream and restart it at *new_fps*.

        A full restart (vs a live frame-rate change) re-runs the sync/photometric
        configuration, so the cameras come back up correctly synchronised at the
        new rate.  Returns True if a stream is running afterwards.
        """
        if self._worker is None:
            self._fps_spin.setValue(new_fps)
            return False

        old = self._worker
        self._worker = None
        # Detach the old worker so its finished/error signals don't run our
        # teardown slot (which would hide the calibration panel mid-restart).
        for sig, slot in (
            (old.frame_ready, self._on_frame_ready),
            (old.recording_finished, self._on_recording_finished),
            (old.finished, self._on_worker_finished),
            (old.error, self._on_worker_error),
        ):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

        if self._is_recording:
            old.end_recording()
            self._is_recording = False
        old.cancel()
        if not old.wait(5000):  # let FlirCapture.stop() release the cameras
            old.terminate()
            old.wait(1000)
        old.deleteLater()

        self._sync_history.clear()
        self._fps_spin.setValue(new_fps)
        source = self._build_source()
        if source is None:
            self._status_label.setText("Restart failed — could not rebuild the capture source.")
            return False
        self._make_and_start_worker(source)
        self._status_label.setText(f"Streaming… ({new_fps:.0f} fps)")
        return True

    def _on_stop(self) -> None:
        if self._worker is not None:
            if self._is_recording:
                self._worker.end_recording()
                self._is_recording = False
            self._worker.cancel()

    def _on_record_toggle(self, checked: bool) -> None:
        if self._worker is None:
            return
        if checked:
            out_folder = self._out_edit.text().strip() or "capture"
            # Default to a start-time stamp so successive recordings don't
            # overwrite each other: YYYYMMDD_HHMMSS_{left,right,timestamps}.
            prefix = self._prefix_edit.text().strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
            Path(out_folder).mkdir(parents=True, exist_ok=True)
            out_l = str(Path(out_folder) / f"{prefix}_left.avi")
            out_r = str(Path(out_folder) / f"{prefix}_right.avi")
            self._worker.begin_recording(out_l, out_r)
            self._is_recording = True
            self._record_btn.setText("Stop Recording")
            self._rec_status.setText("● REC")
        else:
            self._worker.end_recording()
            self._is_recording = False
            self._record_btn.setText("Start Recording")
            self._rec_status.setText("")

    def _on_frame_ready(self, frame) -> None:
        self._last_frame = frame

        # Feed a pending "Set coordinate system" collection (mode-independent;
        # no-op unless the button was just clicked).
        self._coordsys_feed(frame.frame_left, frame.frame_right)

        drops = frame.dropped_frames
        if drops > 0:
            color = "#e74c3c" if self._is_recording else "#e67e22"
            self._drop_label.setText(f"<b style='color:{color}'>{drops}</b>")
            self._drop_label.setTextFormat(Qt.TextFormat.RichText)
        else:
            self._drop_label.setText("0")
            self._drop_label.setStyleSheet("color: inherit;")

        if self._calib_mode_btn.isChecked():
            # Collect completed detection result and pair it with the submitted frames
            if self._detect_future is not None and self._detect_future.done():
                try:
                    self._det_left, self._det_right = self._detect_future.result()
                    # Commit the matched frames — these are pixel-aligned with the results
                    self._detect_frame_fl = self._detect_submitted_fl
                    self._detect_frame_fr = self._detect_submitted_fr
                except Exception as _exc:
                    import traceback as _tb

                    print(f"[board detect] exception: {_exc}\n{_tb.format_exc()}")
                    self._det_left = self._det_right = None
                finally:
                    self._detect_future = None
                self._on_detection_ready()

            # Submit next detection if idle and detector is configured.
            # Skip submission once cleanup() has shut the pool down — otherwise
            # any in-flight CaptureWorker frame would crash with
            # "cannot schedule new futures after shutdown".
            now = time.monotonic()
            if (
                self._detect_future is None
                and self._detector is not None
                and not self._pool_shutdown
                # Rate-limit: don't start another detection until the minimum
                # interval has elapsed, so detection can't monopolise the CPU
                # and starve the capture-read thread (→ dropped frames).
                and (now - self._last_detect_submit) >= self._detect_min_interval_s
            ):
                self._last_detect_submit = now
                fl = frame.frame_left.copy()
                fr = frame.frame_right.copy()
                self._detect_submitted_fl = fl  # remember what we submitted
                self._detect_submitted_fr = fr
                try:
                    self._detect_future = self._detect_pool.submit(
                        CaptureTab._detect_pair, self._detector, fl, fr
                    )
                except RuntimeError:
                    # Race: cleanup() ran between the flag check and submit.
                    self._pool_shutdown = True

            # Display with overlay using last known detection result
            left = CaptureTab._draw_overlay(frame.frame_left, self._det_left, "L")
            right = CaptureTab._draw_overlay(frame.frame_right, self._det_right, "R")
            left, right = self._maybe_overlay_world_axes(left, right)
            _set_preview(self._preview_left, left)
            _set_preview(self._preview_right, right)
        elif self._pose_active:
            # ── Live Pose Tracking branch ─────────────────────────────────
            # Collect any completed detection.
            if self._pose_future is not None and self._pose_future.done():
                try:
                    kps_l, conf_l, kps_r, conf_r = self._pose_future.result()
                    self._process_pose_result(kps_l, conf_l, kps_r, conf_r)
                except Exception as _exc:
                    import traceback as _tb

                    print(f"[live pose] exception: {_exc}\n{_tb.format_exc()}")
                    self._pose_status_label.setText(
                        f"<span style='color:#e74c3c'>Error: {_exc}</span>"
                    )
                finally:
                    self._pose_future = None

            # Submit next detection if idle (reuse the same shared thread pool).
            if (
                self._pose_future is None
                and self._pose_backend is not None
                and self._pose_backend_r is not None
                and not self._pool_shutdown
            ):
                fl = frame.frame_left.copy()
                fr = frame.frame_right.copy()
                self._pose_submitted_fl = fl
                self._pose_submitted_fr = fr
                try:
                    self._pose_future = self._detect_pool.submit(
                        CaptureTab._pose_detect_pair,
                        self._pose_backend,
                        self._pose_backend_r,
                        fl,
                        fr,
                    )
                except RuntimeError:
                    self._pool_shutdown = True

            # Display with 2D skeleton overlay drawn from the last completed result.
            if self._pose_overlay_check.isChecked() and self._pose_last_kp_l is not None:
                left = self._draw_pose_overlay(
                    frame.frame_left,
                    self._pose_last_kp_l,
                    self._pose_last_conf_l,
                )
                right = self._draw_pose_overlay(
                    frame.frame_right,
                    self._pose_last_kp_r,
                    self._pose_last_conf_r,
                )
            else:
                left, right = frame.frame_left, frame.frame_right
            left, right = self._maybe_overlay_world_axes(left, right)
            _set_preview(self._preview_left, left)
            _set_preview(self._preview_right, right)
        else:
            left, right = self._maybe_overlay_world_axes(frame.frame_left, frame.frame_right)
            _set_preview(self._preview_left, left)
            _set_preview(self._preview_right, right)

        # Push the latest frame into the fullscreen inspect window if open,
        # with the world-frame triad overlaid (matches the inline previews).
        if self._fs_preview is not None and self._fs_side is not None:
            src = frame.frame_left if self._fs_side == "L" else frame.frame_right
            self._fs_preview.set_frame(self._overlay_world_axes_side(src, self._fs_side))

        self._update_sync_indicator(frame)

    def _open_fullscreen_preview(self, side: str) -> None:
        """Open a fullscreen, click-to-close inspect view of one camera.

        Used to verify focus by stretching the live frame across the whole screen.
        """
        # Toggle: clicking the same side again (while open) closes it.
        if self._fs_preview is not None:
            self._fs_preview.close()
            return
        title = "LEFT camera — full screen" if side == "L" else "RIGHT camera — full screen"
        dlg = _FullscreenPreview(title, parent=self)
        dlg.finished.connect(self._on_fullscreen_closed)
        self._fs_preview = dlg
        self._fs_side = side
        # Seed with the most recent frame so we don't show "Waiting…" for one tick.
        if self._last_frame is not None:
            src = self._last_frame.frame_left if side == "L" else self._last_frame.frame_right
            dlg.showFullScreen()
            dlg.set_frame(self._overlay_world_axes_side(src, side))
        else:
            dlg.showFullScreen()

    def _on_fullscreen_closed(self, _result: int = 0) -> None:
        self._fs_preview = None
        self._fs_side = None

    def _open_fullscreen_3d(self) -> None:
        """Open the 3D skeleton in a frameless fullscreen window.

        Re-opens (toggles closed) if it's already up — same gesture as the
        camera previews.  The dialog's own viewer is fed live frames from
        the same source data already going to the inline preview.
        """
        if self._fs_3d is not None:
            self._fs_3d.close()
            return
        dlg = _Fullscreen3DSkeleton(parent=self)
        dlg.finished.connect(self._on_fullscreen_3d_closed)
        self._fs_3d = dlg
        # Show FIRST so the GLViewWidget's OpenGL context is created.  Building
        # or updating GL items (setup_live_view / set_frame) before the context
        # exists yields undrawable items → "Error while drawing item" + a blank
        # canvas.  Defer configuration to the next event-loop tick so the show
        # (and its initializeGL) has fully completed.
        dlg.showFullScreen()

        def _configure(d=dlg):
            if self._fs_3d is not d:
                return  # closed again before the tick fired
            d.viewer.setup_live_view(world_frame=self._pose_world)
            if self._pose_last_view_3d is not None and self._pose_last_conf_3d is not None:
                d.set_frame(
                    self._pose_last_view_3d, self._pose_last_conf_3d, world_frame=self._pose_world
                )

        QTimer.singleShot(0, _configure)

    def _on_fullscreen_3d_closed(self, _result: int = 0) -> None:
        self._fs_3d = None

    def _update_sync_indicator(self, frame) -> None:
        if frame.hw_timestamp_left_ns is None or frame.hw_timestamp_right_ns is None:
            self._sync_delta_label.setText("— (no hw timestamps)")
            self._sync_delta_label.setStyleSheet("color: #888;")
            self._sync_quality_label.setText("")
            return

        # Signed delta (µs): left minus right — positive means left fires later.
        # We track the rolling history for two metrics:
        #   jitter (stdev) — timing noise; immune to steady crystal-clock drift
        #   mean offset    — non-zero means cameras are temporally misaligned
        #                    e.g. |mean| ≈ frame_period → secondary is one frame behind
        signed_us = (frame.hw_timestamp_left_ns - frame.hw_timestamp_right_ns) / 1_000.0
        self._sync_history.append(signed_us)

        if len(self._sync_history) < 2:
            self._sync_delta_label.setText("measuring…")
            self._sync_delta_label.setStyleSheet("color: #888;")
            self._sync_quality_label.setText("")
            return

        jitter_us = statistics.stdev(self._sync_history)

        # Jitter (stdev of signed deltas) is the only reliable hardware-sync metric.
        # The mean signed delta reflects the difference in camera clock origins at
        # power-on — it is a fixed constant unrelated to trigger alignment and must
        # NOT be used for quality assessment.
        if jitter_us < 50:
            color, rating = "#2ecc71", "✓ Excellent"
        elif jitter_us < 200:
            color, rating = "#27ae60", "✓ Good"
        elif jitter_us < 1_000:
            color, rating = "#e67e22", "⚠ Marginal"
        else:
            color, rating = "#e74c3c", "✗ Poor — check wiring"

        jitter_str = f"{jitter_us / 1_000:.2f} ms" if jitter_us >= 1_000 else f"{jitter_us:.1f} µs"

        self._sync_delta_label.setText(f"jitter {jitter_str}")
        self._sync_delta_label.setStyleSheet(f"color:{color}; font-weight:bold;")
        self._sync_quality_label.setText(f"<span style='color:{color}'>{rating}</span>")
        self._sync_quality_label.setTextFormat(Qt.TextFormat.RichText)

    def _on_recording_finished(self, left_path: str, right_path: str) -> None:
        self._rec_status.setText(f"Saved: {Path(left_path).name}, {Path(right_path).name}")
        self.recording_saved.emit(left_path, right_path)

    def _on_worker_finished(self, _result: object) -> None:
        self._worker = None
        # If the stream stopped while a calibration fps override was active,
        # there is nothing to restart — just restore the displayed/persisted
        # rate to the user's original so it isn't left stuck at the calib value.
        if self._calib_fps_active and self._precalib_fps is not None:
            self._fps_spin.setValue(self._precalib_fps)
            self._calib_fps_active = False
            self._precalib_fps = None
        self._is_recording = False
        self._record_btn.setEnabled(False)
        self._record_btn.setChecked(False)
        self._record_btn.setText("Start Recording")
        self._stop_btn.setEnabled(False)
        self._rec_status.setText("")
        self._status_label.setText("Idle")
        self._sync_history.clear()
        self._sync_delta_label.setText("—")
        self._sync_delta_label.setStyleSheet("color: #888;")
        self._sync_quality_label.setText("")
        self._drop_label.setText("0")
        self._drop_label.setStyleSheet("")
        # Disable calibration mode and clean up detection state
        self._calib_mode_btn.setEnabled(False)
        self._calib_mode_btn.setChecked(False)
        self._calib_panel.setVisible(False)
        self._det_left = None
        self._det_right = None
        self._detect_future = None
        self._detect_submitted_fl = None
        self._detect_submitted_fr = None
        # Frames stopped flowing mid "Set coordinate system" collection — cancel
        # it (an already-submitted job keeps running and completes via the poll).
        if self._coordsys_pairs is not None:
            self._coordsys_pairs = None
            self._set_coord_btn.setEnabled(True)
            self._calib_status_label.setText(
                "Stream stopped before the floor board could be analysed."
            )
        self._detect_frame_fl = None
        self._detect_frame_fr = None
        # Also disable Live Pose mode — no stream means no input.
        if self._pose_active:
            self._stop_pose_tracking()
            self._pose_start_btn.setChecked(False)
            self._pose_start_btn.setText("▶  Start Tracking")
        self._pose_mode_btn.setEnabled(False)
        self._pose_mode_btn.setChecked(False)
        self._pose_panel.setVisible(False)
        self._last_frame = None
        # Re-enable camera assignment now that streaming has stopped.
        self._left_combo.setEnabled(True)
        self._right_combo.setEnabled(True)
        self._refresh_state()

    def _on_worker_error(self, msg: str) -> None:
        self._on_worker_finished(None)
        QMessageBox.critical(self, "Capture error", msg)

    # ------------------------------------------------------------------
    # Live calibration — panel open/close
    # ------------------------------------------------------------------

    def _on_calib_mode_clicked(self, checked: bool) -> None:
        """Intercept the toggle so we can warn before discarding captured frames.

        When the user clicks the button to OPEN calibration mode, we proceed
        immediately.  When they click again to CLOSE (checked → False), and there
        are unsaved captures, we ask for confirmation first.
        """
        if not checked:
            # User is trying to close calibration mode.
            self._on_calib_close_clicked()
            return

        # Opening calibration mode — set up the detector and show the panel.
        self._apply_calib_open()

    def _on_calib_close_clicked(self) -> None:
        """Close the calibration panel, with a discard warning if frames exist."""
        total = len(self._calib_left_dets) + len(self._calib_right_dets) + len(self._calib_pairs)
        if total:
            ans = QMessageBox.question(
                self,
                "Close Calibration Mode?",
                f"You have <b>{len(self._calib_left_dets)}</b> left, "
                f"<b>{len(self._calib_right_dets)}</b> right, and "
                f"<b>{len(self._calib_pairs)}</b> stereo captures.<br><br>"
                "Closing Calibration Mode will <b>hide</b> the panel but your captures are "
                "preserved — they will be discarded only if you click <i>Clear All</i>.<br><br>"
                "Close the panel now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.No:
                # Revert the button state — keep calibration mode active.
                self._calib_mode_btn.setChecked(True)
                return

        # Proceed with closing.
        self._calib_mode_btn.setChecked(False)
        self._calib_panel.setVisible(False)
        # Reset transient detection state; keep _calib_pairs for later reuse.
        self._det_left = None
        self._det_right = None
        self._detect_future = None
        self._reset_detection_labels()
        self._calib_capture_btn.setEnabled(False)
        self._calib_capture_btn.setStyleSheet("")

        # Restore the user's original frame rate if we lowered it for calibration.
        if self._calib_fps_active and self._precalib_fps is not None:
            original = self._precalib_fps
            self._calib_fps_active = False
            self._precalib_fps = None
            if self._worker is not None:
                self._status_label.setText(f"Restoring {original:.0f} fps…")
                self._restart_stream_at_fps(original)
            else:
                self._fps_spin.setValue(original)

    def _apply_calib_open(self) -> None:
        """Open the calibration panel and initialise the board detector."""
        try:
            from app.calib.board import make_detector

            self._detector = make_detector(self._live_board_cfg())
        except Exception as exc:
            QMessageBox.warning(self, "Board config error", str(exc))
            self._calib_mode_btn.setChecked(False)
            return

        # Mutually exclusive with Live Pose — close that panel if open so the
        # user only ever sees one bottom panel at a time.
        if self._pose_mode_btn.isChecked() or self._pose_panel.isVisible():
            if self._pose_active:
                self._stop_pose_tracking()
                self._pose_start_btn.setChecked(False)
                self._pose_start_btn.setText("▶  Start Tracking")
            self._pose_mode_btn.setChecked(False)
            self._pose_panel.setVisible(False)

        self._calib_panel.setVisible(True)
        self._update_calib_counter()
        self._reset_detection_labels()

        # If a calibration already exists at the output path, allow defining the
        # floor world frame straight away (no need to re-run calibration).
        existing = self._calib_out_edit.text().strip()
        if existing and Path(existing).is_file():
            self._calib_saved_path = existing
            self._set_coord_btn.setEnabled(True)
            # Show the floor triad immediately if this calib already has a world frame.
            self._refresh_world_axes_overlay()

        # Suggest dropping to the calibration-optimal fps if the live stream is
        # running faster (board detection can't keep up → dropped frames).
        self._maybe_lower_fps_for_calibration()

    def _maybe_lower_fps_for_calibration(self) -> None:
        """Offer to restart the stream at the calibration-optimal fps."""
        if self._worker is None:
            return  # not streaming — nothing to change
        current = float(self._fps_spin.value())
        if current <= self._CALIB_FPS:
            return  # already at/below optimal
        ans = QMessageBox.question(
            self,
            "Lower frame rate for calibration?",
            f"Calibration runs best at about <b>{self._CALIB_FPS:.0f} fps</b>.<br><br>"
            f"Board detection is CPU-intensive; at the current "
            f"<b>{current:.0f} fps</b> the cameras can deliver frames faster than "
            f"they can be processed, which shows up as dropped frames "
            f"(and a misleading sync warning).<br><br>"
            f"Drop to {self._CALIB_FPS:.0f} fps for calibration? The stream will "
            f"restart and your original rate is restored when you leave "
            f"Calibration Mode.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        self._precalib_fps = current
        self._calib_fps_active = True
        self._status_label.setText(f"Restarting at {self._CALIB_FPS:.0f} fps for calibration…")
        self._restart_stream_at_fps(self._CALIB_FPS)

    # ------------------------------------------------------------------
    # Live Pose Tracking — handlers
    # ------------------------------------------------------------------

    def _on_pose_mode_clicked(self, checked: bool) -> None:
        """Toggle the Live Pose panel.  Mutually exclusive with Calibration Mode."""
        if not checked:
            self._on_pose_close_clicked()
            return
        # Close calibration mode first if it's open (mutually exclusive).
        if self._calib_mode_btn.isChecked():
            self._calib_mode_btn.setChecked(False)
            self._calib_panel.setVisible(False)
            self._detector = None
        # Pre-fill the calibration path from the project if not set.
        if not self._pose_calib_edit.text().strip():
            if self._project_dir:
                guess = Path(self._project_dir) / "calibration.yml"
                if guess.exists():
                    self._pose_calib_edit.setText(str(guess))
        self._pose_panel.setVisible(True)

    def _on_pose_close_clicked(self) -> None:
        """Stop live tracking (if running) and hide the panel."""
        if self._pose_active:
            self._stop_pose_tracking()
        self._pose_mode_btn.setChecked(False)
        self._pose_panel.setVisible(False)

    def _on_pose_browse_calib(self) -> None:
        start = self._pose_calib_edit.text().strip() or (self._project_dir or "")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select calibration.yml",
            start,
            "YAML (*.yml *.yaml)",
        )
        if path:
            self._pose_calib_edit.setText(path)

    def _on_pose_start_clicked(self, checked: bool) -> None:
        if checked:
            try:
                self._start_pose_tracking()
                self._pose_start_btn.setText("⏹  Stop Tracking")
            except Exception as exc:
                QMessageBox.warning(self, "Live Pose error", str(exc))
                self._pose_start_btn.setChecked(False)
                self._pose_start_btn.setText("▶  Start Tracking")
        else:
            self._stop_pose_tracking()
            self._pose_start_btn.setText("▶  Start Tracking")

    def _start_pose_tracking(self) -> None:
        """Load calibration, build the backend, and switch into live mode."""
        calib_path = self._pose_calib_edit.text().strip()
        if not calib_path or not Path(calib_path).exists():
            raise RuntimeError(
                "Set a valid path to calibration.yml first "
                "(use the Calibration tab or live calibration to produce one)."
            )

        from app.calib.stereo import load_calibration  # noqa: PLC0415

        self._pose_calib = load_calibration(calib_path)

        # Build the backend on the main thread (it owns C++ resources / model load).
        # Using the same instance for both cameras means detect() is serialised
        # within the worker thread.  MediaPipe is fast enough on CPU for live use.
        from app.pose2d.mediapipe_backend import MediaPipeBackend  # noqa: PLC0415

        n_poses = int(self._pose_num_spin.value())
        # One VIDEO-mode landmarker per camera.  Confidence is lowered to 0.3:
        # at the default 0.5 the harder camera view often fails to get its
        # initial detection lock and shows no skeleton, while 0.3 lets both
        # cameras lock reliably (measured) without spurious detections.
        conf = 0.3
        self._pose_backend = MediaPipeBackend(
            num_poses=n_poses,
            running_mode="video",
            min_detection_confidence=conf,
            min_tracking_confidence=conf,
        )
        self._pose_backend_r = MediaPipeBackend(
            num_poses=n_poses,
            running_mode="video",
            min_detection_confidence=conf,
            min_tracking_confidence=conf,
        )

        # Drop any prior smoothing state so a fresh stream starts clean.
        self._pose_filters = None
        self._pose_last_kp_l = None
        self._pose_last_kp_r = None
        self._pose_last_conf_l = None
        self._pose_last_conf_r = None
        self._pose_last_3d = None
        self._pose_last_conf_3d = None
        self._pose_last_view_3d = None
        # When the calibration defines a floor world frame, live joints are
        # re-expressed in it (ADR-010) — the viewer origin is then the board on
        # the floor (subject stands at/near it), not the camera.
        self._pose_world = self._pose_calib.get("world_frame") is not None

        self._pose_active = True
        # Reveal the inline 3D viewer next to the camera previews and frame
        # it on the volume where a standing person typically appears (~2 m
        # in front of left camera).  User can still orbit / zoom with the mouse.
        self._preview_3d.setVisible(True)
        self._preview_3d.setup_live_view(world_frame=self._pose_world)
        self._pose_status_label.setText("Tracking — waiting for first frame…")

    def _stop_pose_tracking(self) -> None:
        """Stop the live pose loop and release backend resources."""
        self._pose_active = False
        for _bk in (self._pose_backend, self._pose_backend_r):
            if _bk is not None:
                try:
                    _bk.close()
                except Exception:
                    pass
        self._pose_backend = None
        self._pose_backend_r = None
        # Cancel any in-flight future result by discarding it on next tick.
        self._pose_future = None
        self._pose_status_label.setText("Idle")
        # Clear overlays so the preview stops showing stale 2D landmarks.
        self._pose_last_kp_l = None
        self._pose_last_kp_r = None
        # Hide the inline 3D viewer so it gives space back to the camera previews.
        self._preview_3d.clear()
        self._preview_3d.setVisible(False)
        # Close the fullscreen 3D pop-out if it's open — it would otherwise
        # keep showing the last (now-stale) frame.
        if self._fs_3d is not None:
            self._fs_3d.close()

    @staticmethod
    def _pose_detect_pair(backend_l, backend_r, frame_l: np.ndarray, frame_r: np.ndarray):
        """Run pose detection on both frames with per-camera VIDEO landmarkers.

        Each camera has its OWN backend so its VIDEO-mode temporal track isn't
        corrupted by the other view (see __init__ note).  Runs serially in one
        worker thread — MediaPipe isn't thread-safe.
        Returns (kps_l, conf_l, kps_r, conf_r) with shapes [P,17,2] / [P,17].
        """
        kps_l, conf_l = backend_l.detect(frame_l)
        kps_r, conf_r = backend_r.detect(frame_r)
        return kps_l, conf_l, kps_r, conf_r

    def _process_pose_result(
        self,
        kps_l: np.ndarray,
        conf_l: np.ndarray,
        kps_r: np.ndarray,
        conf_r: np.ndarray,
    ) -> None:
        """Triangulate the freshly-detected stereo pose, smooth, update the viewer."""
        assert self._pose_calib is not None
        from app.recon3d.triangulate import triangulate_frame_pair  # noqa: PLC0415

        # Align person counts across views (MediaPipe may detect a different number
        # per view); take the minimum so the [P, ...] shapes match for triangulation.
        P = min(kps_l.shape[0], kps_r.shape[0])
        if P == 0:
            self._pose_status_label.setText(
                "<span style='color:#e67e22'>No person detected in either view.</span>"
            )
            return
        kps_l = kps_l[:P]
        conf_l = conf_l[:P]
        kps_r = kps_r[:P]
        conf_r = conf_r[:P]

        calib = self._pose_calib
        joints3d, conf3d, repro = triangulate_frame_pair(
            kps_l,
            kps_r,
            conf_l,
            conf_r,
            calib["K1"],
            calib["D1"],
            calib["K2"],
            calib["D2"],
            calib["R"],
            calib["T"],
            min_conf=float(self._pose_min_conf_spin.value()),
            max_reproj_err=float(self._pose_max_reproj_spin.value()),
            lens_model=calib.get("lens_model", "standard"),
        )

        # OneEuro smoothing — lazy-init filters on first frame (sized to the
        # actual P × J × 3 we got).  Falls back to identity when the smoothing
        # toggle is off.
        if self._pose_smooth_check.isChecked():
            joints3d = self._apply_pose_smoothing(joints3d, conf3d)

        # Stash for overlay/repaint between detections.  _pose_last_3d stays in
        # the CAMERA frame — the smoothing hold-last fallback feeds it back into
        # the camera-frame stream.
        self._pose_last_kp_l = kps_l
        self._pose_last_kp_r = kps_r
        self._pose_last_conf_l = conf_l
        self._pose_last_conf_r = conf_r
        self._pose_last_3d = joints3d
        self._pose_last_conf_3d = conf3d

        # Express the DISPLAY copy in the floor world frame when the calibration
        # defines one — mirrors the offline pipeline, where the world transform
        # is the LAST step (after smoothing).  The viewer then skips its legacy
        # OpenCV→Z-up swap and its origin IS the board on the floor.
        wf = calib.get("world_frame")
        self._pose_world = wf is not None
        view3d = joints3d
        if wf is not None:
            from app.calib.coordinate_system import apply_world_frame  # noqa: PLC0415

            view3d = apply_world_frame(joints3d, wf).astype(np.float32)
        self._pose_last_view_3d = view3d

        # Push to the inline 3D viewer (next to the camera previews) and to
        # the fullscreen pop-out if it's currently open.
        self._preview_3d.set_frame(view3d, conf3d, world_frame=self._pose_world)
        if self._fs_3d is not None:
            self._fs_3d.set_frame(view3d, conf3d, world_frame=self._pose_world)

        # Metrics readout.
        n_valid = int((conf3d > 0).sum())
        n_total = conf3d.size
        finite = repro[np.isfinite(repro)]
        mean_err = float(np.mean(finite)) if finite.size else float("nan")
        med_err = float(np.median(finite)) if finite.size else float("nan")

        # A person IS detected in 2D, but if the triangulation reprojects far off
        # the joints get rejected (conf3d=0) → "dots but no skeleton".  That's a
        # stereo-calibration problem, not a detection one — surface it loudly
        # instead of silently showing an empty 3D viewer.
        calib_bad = (np.isfinite(med_err) and med_err > self._POSE_REPROJ_WARN_PX) or n_valid == 0
        if calib_bad:
            self._pose_status_label.setText(
                f"<span style='color:#e74c3c'>⚠ Pose detected, but 3D triangulation is failing</span> "
                f"<span style='color:#aaa'>· joints {n_valid}/{n_total} valid · "
                f"median reproj {med_err:.0f} px</span><br>"
                f"<span style='color:#e67e22'>The stereo calibration looks unreliable for this "
                f"volume (good is &lt; 20 px). Re-run Calibration — widen the camera baseline / "
                f"reduce toe-in and cover more of the view with the board.</span>"
            )
        else:
            self._pose_status_label.setText(
                f"<span style='color:#2ecc71'>● Tracking</span> "
                f"<span style='color:#aaa'>· persons {P} · joints {n_valid}/{n_total} valid</span>"
            )
        self._pose_metrics_label.setText(
            f"mean reproj err: {mean_err:.2f} px" if np.isfinite(mean_err) else "mean reproj err: —"
        )

    def _apply_pose_smoothing(
        self,
        joints3d: np.ndarray,
        conf3d: np.ndarray,
    ) -> np.ndarray:
        """Apply per-axis OneEuro smoothing.  Uses ~camera-fps as the sampling rate."""
        from app.recon3d.smooth import OneEuroFilter  # noqa: PLC0415

        P, J, _ = joints3d.shape
        # Allocate one filter per (p, j, axis).  Re-allocate if P changed.
        if self._pose_filters is None or len(self._pose_filters) != P * J * 3:
            fps_hint = (
                float(self._last_frame.fps)
                if self._last_frame is not None and getattr(self._last_frame, "fps", 0)
                else 20.0
            )
            self._pose_filters = [
                OneEuroFilter(fps=fps_hint, min_cutoff=0.5, beta=0.05, d_cutoff=1.0)
                for _ in range(P * J * 3)
            ]

        smoothed = joints3d.copy()
        for p in range(P):
            for j in range(J):
                if conf3d[p, j] <= 0:
                    # No valid measurement — repeat the last smoothed value if any,
                    # else leave the zeroed sample.
                    if self._pose_last_3d is not None and p < self._pose_last_3d.shape[0]:
                        smoothed[p, j] = self._pose_last_3d[p, j]
                    continue
                for ax in range(3):
                    idx = (p * J + j) * 3 + ax
                    smoothed[p, j, ax] = float(self._pose_filters[idx](float(joints3d[p, j, ax])))
        return smoothed

    def _draw_pose_overlay(self, bgr: np.ndarray, kps: np.ndarray, conf: np.ndarray) -> np.ndarray:
        """Draw the COCO-17 skeleton over a BGR frame.  Returns a new image."""
        if kps is None or kps.size == 0:
            return bgr
        out = bgr.copy()
        for p in range(kps.shape[0]):
            pts = kps[p]
            c = conf[p]
            for i, j in COCO17_EDGES:
                if c[i] < 0.1 or c[j] < 0.1:
                    continue
                a = (int(pts[i, 0]), int(pts[i, 1]))
                b = (int(pts[j, 0]), int(pts[j, 1]))
                cv2.line(out, a, b, (0, 220, 80), 2, cv2.LINE_AA)
            for k in range(pts.shape[0]):
                if c[k] < 0.1:
                    continue
                cv2.circle(
                    out,
                    (int(pts[k, 0]), int(pts[k, 1])),
                    4,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )
        return out

    def _reset_detection_labels(self) -> None:
        """Reset the detection status labels to the idle (grey) state."""
        idle_style = (
            "font-weight: bold; font-size: 18px; color: #888;"
            "background: #111; border-radius: 6px; padding: 14px;"
        )
        self._calib_left_status.setText("LEFT: —")
        self._calib_left_status.setStyleSheet(idle_style)
        self._calib_right_status.setText("RIGHT: —")
        self._calib_right_status.setStyleSheet(idle_style)
        self._detect_history.clear()

    def _on_board_params_changed(self) -> None:
        """Recreate the board detector immediately when any parameter widget changes.

        This means the user can tweak cols/rows/square-size while the preview
        is running and detection responds instantly — no stop/restart needed.
        The in-flight future (if any) is left to complete; the new detector is
        picked up on the very next submitted frame.
        """
        if not self._calib_mode_btn.isChecked():
            return  # panel not open — detector will be built on next open
        try:
            from app.calib.board import make_detector

            self._detector = make_detector(self._live_board_cfg())
            # Clear any previous config-error message
            if self._calib_status_label.text().startswith("<span style='color:#e67e22'>⚠ Board"):
                self._calib_status_label.setText("")
        except Exception as exc:
            self._detector = None  # detection pauses until fixed
            self._calib_status_label.setText(
                f"<span style='color:#e67e22'>⚠ Board config error: {exc}</span>"
            )

    def _on_calib_board_type(self, idx: int) -> None:
        self._calib_board_stack.setCurrentIndex(idx)
        # Board-type change also counts as a parameter change.
        self._on_board_params_changed()

    def _live_board_cfg(self) -> dict:
        if self._calib_board_combo.currentIndex() == 0:
            return {
                "type": "charuco",
                "squares_x": self._calib_sq_x.value(),
                "squares_y": self._calib_sq_y.value(),
                "square_size": self._calib_sq_size.value(),
                "marker_size": self._calib_mk_size.value(),
                "dictionary": self._calib_aruco_dict.currentText(),
            }
        return {
            "type": "checkerboard",
            "cols": self._calib_chess_cols.value(),
            "rows": self._calib_chess_rows.value(),
            "square_size": self._calib_chess_sq_size.value(),
        }

    # Shared inner pool — left and right detections run in parallel here.
    # OpenCV's ArUco/ChArUco detection releases the GIL during the C++ work,
    # so 2 threads actually halve end-to-end latency on multi-core machines.
    _detect_lr_pool = _ThreadPool(max_workers=2, thread_name_prefix="board_det_lr")

    @staticmethod
    def _detect_pair(detector, frame_l: np.ndarray, frame_r: np.ndarray):
        """Run board detection on both frames concurrently."""
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        fut_l = CaptureTab._detect_lr_pool.submit(detector.detect, gray_l)
        fut_r = CaptureTab._detect_lr_pool.submit(detector.detect, gray_r)
        return fut_l.result(), fut_r.result()

    @staticmethod
    def _draw_overlay(bgr: np.ndarray, det, label: str) -> np.ndarray:
        """Return a copy of *bgr* with a colored border and corner dots.

        Colour coding:
          green  — full board detected (obj_pts + img_pts populated)
          orange — ArUco markers found but ChArUco board layout didn't match
                   (wrong squares_x/y or dictionary)
          red    — nothing detected at all
        """
        out = bgr.copy()
        h, w = out.shape[:2]
        if det is not None and not det.partial:
            # Full detection — green
            color = (0, 220, 0)
            for pt in det.img_pts.reshape(-1, 2):
                cv2.circle(out, (int(pt[0]), int(pt[1])), 5, color, -1)
            cv2.rectangle(out, (4, 4), (w - 4, h - 4), color, 4)
            cv2.putText(
                out,
                f"{label}  OK  {len(det.img_pts)} pts",
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
                cv2.LINE_AA,
            )
        elif det is not None and det.partial:
            # Partial detection — orange: markers seen but board didn't fit
            color = (0, 165, 255)  # BGR orange
            cv2.rectangle(out, (4, 4), (w - 4, h - 4), color, 4)
            cv2.putText(
                out,
                f"{label}  {det.n_markers} markers — check board config",
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )
        else:
            # Nothing at all — red
            color = (30, 30, 210)
            cv2.rectangle(out, (4, 4), (w - 4, h - 4), color, 4)
            cv2.putText(
                out,
                f"{label}  not detected",
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
                cv2.LINE_AA,
            )
        return out

    def _update_stability_label(self) -> None:
        pass

    def _on_detection_ready(self) -> None:
        """Called in the main thread when a detection result just arrived."""
        det_l, det_r = self._det_left, self._det_right
        l_ok = det_l is not None and not det_l.partial
        r_ok = det_r is not None and not det_r.partial
        # Only count as "both detected" when both have full (non-partial) results
        both = l_ok and r_ok

        # Record this detection cycle for the stability indicator.
        if l_ok and r_ok:
            self._detect_history.append("both")
        elif l_ok:
            self._detect_history.append("left")
        elif r_ok:
            self._detect_history.append("right")
        else:
            self._detect_history.append("none")
        self._update_stability_label()

        # Detection status labels — large, color-coded
        ok_style = (
            "font-weight: bold; font-size: 18px; color: #2ecc71;"
            "background: #0d1f0d; border-radius: 6px; padding: 14px;"
        )
        partial_style = (
            "font-weight: bold; font-size: 16px; color: #e67e22;"
            "background: #1f1500; border-radius: 6px; padding: 14px;"
        )
        fail_style = (
            "font-weight: bold; font-size: 18px; color: #e74c3c;"
            "background: #1f0d0d; border-radius: 6px; padding: 14px;"
        )

        if det_l is not None and not det_l.partial:
            self._calib_left_status.setText(f"✓ LEFT  {len(det_l.img_pts)} pts")
            self._calib_left_status.setStyleSheet(ok_style)
        elif det_l is not None and det_l.partial:
            self._calib_left_status.setText(
                f"⚠ LEFT  {det_l.n_markers} markers — check squares/dict"
            )
            self._calib_left_status.setStyleSheet(partial_style)
        else:
            self._calib_left_status.setText("✗ LEFT  not detected")
            self._calib_left_status.setStyleSheet(fail_style)

        if det_r is not None and not det_r.partial:
            self._calib_right_status.setText(f"✓ RIGHT  {len(det_r.img_pts)} pts")
            self._calib_right_status.setStyleSheet(ok_style)
        elif det_r is not None and det_r.partial:
            self._calib_right_status.setText(
                f"⚠ RIGHT  {det_r.n_markers} markers — check squares/dict"
            )
            self._calib_right_status.setStyleSheet(partial_style)
        else:
            self._calib_right_status.setText("✗ RIGHT  not detected")
            self._calib_right_status.setStyleSheet(fail_style)

        # Highlight the Capture button green when both cameras see the board
        if both and self._calib_worker is None:
            self._calib_capture_btn.setEnabled(True)
            self._calib_capture_btn.setStyleSheet(
                "QPushButton {"
                "  background-color: #27ae60; color: white;"
                "  font-weight: bold; border-radius: 4px;"
                "}"
                "QPushButton:hover { background-color: #2ecc71; }"
                "QPushButton:pressed { background-color: #1e8449; }"
            )
        else:
            self._calib_capture_btn.setEnabled(False)
            self._calib_capture_btn.setStyleSheet("")

        # Auto-capture: fire when AT LEAST ONE camera detects AND cooldown elapsed.
        # The capture function decides which piles (left intrinsics / right intrinsics /
        # stereo pairs) to add to based on which cameras have full detection.
        any_detect = (det_l is not None and not det_l.partial) or (
            det_r is not None and not det_r.partial
        )
        if any_detect and self._calib_auto_check.isChecked() and self._calib_worker is None:
            elapsed = time.monotonic() - self._last_auto_capture_time
            if elapsed >= self._calib_cooldown_spin.value():
                self._do_capture_frame()

    def _do_capture_frame(self) -> None:
        """Capture this detection into whichever pile(s) currently apply.

        Two-stage stereo calibration: intrinsics come from per-camera pools,
        extrinsics from stereo pairs.  One capture can populate up to three piles:
            • LEFT detects (regardless of right)  → left-intrinsics pile
            • RIGHT detects (regardless of left)  → right-intrinsics pile
            • BOTH detect simultaneously          → stereo-pairs pile

        Each pile has its own pose-diversity grid so the user has to move the
        board to add to it, regardless of the other piles.
        """
        if self._detect_frame_fl is None or self._detect_frame_fr is None:
            return

        det_l = self._det_left
        det_r = self._det_right
        l_full = det_l is not None and not det_l.partial
        r_full = det_r is not None and not det_r.partial
        if not (l_full or r_full):
            return

        from app.calib.frame_select import FrameSelection, _coverage_score  # noqa: PLC0415

        fl = self._detect_frame_fl
        fr = self._detect_frame_fr
        h, w = fl.shape[:2]
        img_size = (w, h)
        g = self._CALIB_GRID
        max_per = self._CALIB_MAX_PER_CELL
        min_cov = self._CALIB_MIN_COV

        added_to: list[str] = []
        skipped_msg: list[str] = []

        def _cell_for(img_pts: np.ndarray) -> tuple[int, int]:
            ctr = img_pts.reshape(-1, 2).mean(axis=0)
            return (
                min(int(ctr[0] / w * g), g - 1),
                min(int(ctr[1] / h * g), g - 1),
            )

        cov_l = _coverage_score(det_l.img_pts, img_size) if l_full else 0.0
        cov_r = _coverage_score(det_r.img_pts, img_size) if r_full else 0.0
        cov_pair = min(cov_l, cov_r) if (l_full and r_full) else 0.0

        stereo_added = False

        # --- Stereo pile first (highest priority): if it accepts, also
        # unconditionally feed both intrinsics piles. This guarantees
        #     n_stereo  ≤  n_left_dets   and   n_stereo  ≤  n_right_dets
        # at all times — a frame in the stereo pile inherently belongs in both
        # per-camera intrinsics pools too.
        if l_full and r_full and cov_pair >= min_cov:
            scell = _cell_for(det_l.img_pts)
            if self._calib_grid.get(scell, 0) < max_per:
                self._calib_grid[scell] = self._calib_grid.get(scell, 0) + 1
                self._calib_pairs.append(
                    FrameSelection(
                        frame_left=fl,
                        frame_right=fr,
                        det_left=det_l,
                        det_right=det_r,
                        score=cov_pair,
                    )
                )
                # Feed the per-camera piles too, bypassing their grid checks —
                # the stereo grid already enforces diversity for this capture.
                self._calib_left_dets.append(det_l)
                self._calib_right_dets.append(det_r)
                cell_l = _cell_for(det_l.img_pts)
                cell_r = _cell_for(det_r.img_pts)
                self._calib_left_grid[cell_l] = self._calib_left_grid.get(cell_l, 0) + 1
                self._calib_right_grid[cell_r] = self._calib_right_grid.get(cell_r, 0) + 1
                added_to.append("STEREO+L+R")
                stereo_added = True

        # --- Per-camera intrinsics piles (only when not already covered by stereo)
        # If only ONE camera detected, or stereo's cell was full, capture the
        # detected camera(s) into the corresponding intrinsics pile(s).
        if not stereo_added:
            if l_full:
                if cov_l < min_cov:
                    skipped_msg.append(f"L too small ({cov_l * 100:.0f}%)")
                else:
                    cell = _cell_for(det_l.img_pts)
                    if self._calib_left_grid.get(cell, 0) >= max_per:
                        skipped_msg.append("L cell full")
                    else:
                        self._calib_left_grid[cell] = self._calib_left_grid.get(cell, 0) + 1
                        self._calib_left_dets.append(det_l)
                        added_to.append("L")
            if r_full:
                if cov_r < min_cov:
                    skipped_msg.append(f"R too small ({cov_r * 100:.0f}%)")
                else:
                    cell = _cell_for(det_r.img_pts)
                    if self._calib_right_grid.get(cell, 0) >= max_per:
                        skipped_msg.append("R cell full")
                    else:
                        self._calib_right_grid[cell] = self._calib_right_grid.get(cell, 0) + 1
                        self._calib_right_dets.append(det_r)
                        added_to.append("R")

        if added_to:
            self._last_auto_capture_time = time.monotonic()
            self._calib_status_label.setText(
                f"<span style='color:#2ecc71'>✓ Captured: {' + '.join(added_to)}</span>"
            )
        elif skipped_msg:
            self._calib_status_label.setText(
                f"<span style='color:#e67e22'>⚠ Skipped — {', '.join(skipped_msg)} "
                "— move board to a new position</span>"
            )
        self._update_calib_counter()

    def _update_calib_counter(self) -> None:
        n_l = len(self._calib_left_dets)
        n_r = len(self._calib_right_dets)
        n_s = len(self._calib_pairs)

        self._calib_left_bar.setValue(min(n_l, 20))
        self._calib_right_bar.setValue(min(n_r, 20))
        self._calib_stereo_bar.setValue(min(n_s, 8))

        # Status formatting: green tick when each pile meets its target
        l_tag = " ✓" if n_l >= 20 else ""
        r_tag = " ✓" if n_r >= 20 else ""
        s_tag = " ✓" if n_s >= 6 else ""
        self._calib_left_bar.setFormat(f"Left intrinsics: {n_l} / 20 frames{l_tag}")
        self._calib_right_bar.setFormat(f"Right intrinsics: {n_r} / 20 frames{r_tag}")
        self._calib_stereo_bar.setFormat(f"Stereo pairs: {n_s} / 8 (need ≥ 6){s_tag}")

        # Calibration is runnable when:
        #   • Stereo pile has ≥ 6 (extrinsics)
        #   • Each intrinsics pile has ≥ 6 (otherwise per-camera K is unreliable)
        ready = n_s >= 6 and n_l >= 6 and n_r >= 6 and self._calib_worker is None
        self._calib_run_btn.setEnabled(ready)
        self._calib_run_btn.setText(f"▶  Run Calibration  (L:{n_l} R:{n_r} pairs:{n_s})")

    def _on_calib_clear(self) -> None:
        n_l = len(self._calib_left_dets)
        n_r = len(self._calib_right_dets)
        n_s = len(self._calib_pairs)
        if n_l + n_r + n_s == 0:
            return
        ans = QMessageBox.question(
            self,
            "Clear captured frames",
            "Discard all captures and start over?<br><br>"
            f"• Left intrinsics: <b>{n_l}</b><br>"
            f"• Right intrinsics: <b>{n_r}</b><br>"
            f"• Stereo pairs: <b>{n_s}</b>",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            self._calib_left_dets.clear()
            self._calib_right_dets.clear()
            self._calib_pairs.clear()
            self._calib_grid.clear()
            self._calib_left_grid.clear()
            self._calib_right_grid.clear()
            self._calib_status_label.setText("")
            self._update_calib_counter()

    def _browse_calib_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration",
            self._calib_out_edit.text(),
            "YAML (*.yml *.yaml)",
        )
        if path:
            self._calib_out_edit.setText(path)

    def _on_run_calib(self) -> None:
        n_l = len(self._calib_left_dets)
        n_r = len(self._calib_right_dets)
        n_s = len(self._calib_pairs)

        # Gate: every pile needs a usable count.
        if n_s < 6 or n_l < 6 or n_r < 6:
            QMessageBox.warning(
                self,
                "Not enough frames",
                "Need ≥ 6 captures in every pile:<br><br>"
                f"• Left intrinsics: <b>{n_l}</b> (need 6, recommend 20+)<br>"
                f"• Right intrinsics: <b>{n_r}</b> (need 6, recommend 20+)<br>"
                f"• Stereo pairs: <b>{n_s}</b> (need 6, recommend 8+)",
            )
            return
        if n_l < 20 or n_r < 20 or n_s < 8:
            ans = QMessageBox.question(
                self,
                "Low frame count",
                "Some piles are below the recommended count — calibration may be "
                "less reliable.<br><br>"
                f"• Left intrinsics: {n_l} / 20<br>"
                f"• Right intrinsics: {n_r} / 20<br>"
                f"• Stereo pairs: {n_s} / 8<br><br>"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.No:
                return

        out_path = self._calib_out_edit.text().strip() or "calibration.yml"
        h, w = self._calib_pairs[0].frame_left.shape[:2]

        from .workers import LiveCalibWorker

        self._calib_worker = LiveCalibWorker(
            selections=list(self._calib_pairs),
            img_size=(w, h),
            board_cfg=self._live_board_cfg(),
            output_path=out_path,
            lens_model=self._calib_lens_combo.currentIndex(),
            all_det_l=list(self._calib_left_dets),
            all_det_r=list(self._calib_right_dets),
        )
        self._calib_worker.finished.connect(self._calib_worker.deleteLater)
        self._calib_worker.progress.connect(self._on_calib_progress)
        self._calib_worker.finished.connect(self._on_calib_finished)
        self._calib_worker.error.connect(self._on_calib_error)
        self._calib_worker.start()

        self._calib_run_btn.setEnabled(False)
        self._calib_capture_btn.setEnabled(False)
        self._calib_capture_btn.setStyleSheet("")
        self._calib_progress_bar.setValue(0)
        self._calib_progress_bar.setVisible(True)
        self._calib_status_label.setText("Running calibration…")

    def _on_calib_progress(self, pct: int, msg: str) -> None:
        self._calib_progress_bar.setValue(pct)
        if msg:
            self._calib_status_label.setText(msg)

    def _on_calib_finished(self, result: object) -> None:
        self._calib_worker = None
        self._calib_progress_bar.setVisible(False)
        self._update_calib_counter()

        if result is None:
            self._calib_status_label.setText("Calibration cancelled.")
            return

        out_path = str(result)
        try:
            from app.calib.stereo import load_calibration

            calib = load_calibration(out_path)
            q = calib.get("quality", {})
            rms = q.get("rms")
            n_frames = q.get("n_frames_used", "?")
            T = calib.get("T")
            baseline_str = f"{float(np.linalg.norm(T)) * 100:.1f} cm" if T is not None else "?"
            if rms is not None:
                if rms < 0.5:
                    color, rating = "#2ecc71", "Excellent"
                    hint = ""
                elif rms <= 1.5:
                    color, rating = "#3498db", "Acceptable"
                    hint = ""
                else:
                    color, rating = "#e74c3c", "Poor"
                    # Give the user a targeted next step based on current lens model
                    lens_idx = self._calib_lens_combo.currentIndex()
                    if lens_idx == 0:
                        hint = (
                            "<br><span style='color:#e67e22'>⚠ RMS &gt; 1.5 px — your lens "
                            "is probably wider than the Standard model can represent.<br>"
                            "→ Try <b>Wide-angle</b> or <b>Fisheye</b> in the Lens selector "
                            "and recalibrate.</span>"
                        )
                    elif lens_idx == 1:
                        hint = (
                            "<br><span style='color:#e67e22'>⚠ RMS &gt; 1.5 px — the rational "
                            "model still can't fit the lens. If FOV ≥ 150° try <b>Fisheye</b>. "
                            "Also verify the square size with a ruler.</span>"
                        )
                    else:
                        hint = (
                            "<br><span style='color:#e67e22'>⚠ RMS &gt; 1.5 px even with the "
                            "fisheye model. Check: (1) physical square size matches the spinner, "
                            "(2) board corners are sharp and fully visible, "
                            "(3) ≥ 20 well-distributed frames.</span>"
                        )
                msg = (
                    f"<b>✓ Saved:</b> {out_path}<br>"
                    f"RMS: <b>{rms:.3f} px</b> "
                    f"<span style='color:{color}'>[{rating}]</span>"
                    f"&nbsp;&nbsp;Frames: <b>{n_frames}</b>"
                    f"&nbsp;&nbsp;Baseline: <b>{baseline_str}</b>"
                    f"{hint}"
                )
            else:
                msg = f"<b>✓ Saved:</b> {out_path}"
            self._calib_status_label.setText(msg)
        except Exception as exc:
            self._calib_status_label.setText(f"Saved (metrics unavailable: {exc})")

        # Calibration exists now → the floor coordinate system can be defined.
        self._calib_saved_path = out_path
        self._set_coord_btn.setEnabled(True)
        # A freshly-run calibration has no world frame yet → clears any stale triad.
        self._refresh_world_axes_overlay()

        self.calibration_saved.emit(out_path)

    def _on_calib_error(self, msg: str) -> None:
        self._calib_worker = None
        self._calib_progress_bar.setVisible(False)
        self._update_calib_counter()
        self._calib_status_label.setText(f"<span style='color:#e74c3c'>Error: {msg}</span>")
        QMessageBox.critical(self, "Calibration error", msg)

    # ------------------------------------------------------------------
    # Floor world coordinate system
    # ------------------------------------------------------------------

    def _on_set_coordinate_system(self) -> None:
        """Define the world frame from a board lying flat on the floor.

        Collects ~`_COORDSYS_FRAMES` live frames, then a pooled background job
        (no GUI freeze — a single failed full-res chessboard search costs
        ~450 ms) tries two detection strategies:
          1. **ChArUco stereo** — when the markers decode (board reasonably
             close): corners are triangulated and a board-centred frame is fit.
             Corner IDs anchor the orientation, so one frame suffices.
          2. **Chessboard monocular** — for a board far across the floor where
             the ArUco markers are too small to decode, the plain checker
             corners are still found; the pose is recovered by solvePnP from
             whichever camera sees the board (left preferred; if only the right
             sees it, its pose is transformed into camera-1 coords via the
             stereo extrinsics).  The pose is computed on EVERY collected frame
             and **majority-voted** — single-view planar PnP sporadically flips
             the normal ~110° with no reprojection-error signature.
        The frame (origin = board centre, X = long side, Y = short side, Z = up)
        is stored in calibration.yml and the axis triad is drawn on both previews.
        """
        from pathlib import Path as _Path

        if self._coordsys_pairs is not None or self._coordsys_future is not None:
            return  # a click is already being processed
        if self._last_frame is None:
            QMessageBox.warning(
                self, "No frame", "Start the cameras first — no live frame to analyse."
            )
            return
        if not self._calib_saved_path or not _Path(self._calib_saved_path).is_file():
            QMessageBox.warning(self, "Not calibrated", "Run or load a calibration first.")
            return

        if self._detector is None:
            try:
                from app.calib.board import make_detector

                self._detector = make_detector(self._live_board_cfg())
            except Exception as exc:
                QMessageBox.warning(self, "Board config error", str(exc))
                return

        from app.calib.coordinate_system import board_squares
        from app.calib.stereo import load_calibration

        calib = load_calibration(self._calib_saved_path)
        board = calib.get("board_cfg") or self._live_board_cfg()
        # board_squares maps BOTH cfg conventions (charuco squares_x/squares_y,
        # plain-checkerboard cols/rows) — reading squares_x directly yielded 0
        # for checkerboard calibrations and silently disabled the chessboard
        # detection strategy.
        sx, sy, ss = board_squares(board)
        self._coordsys_ctx = {
            "calib": calib,
            "sx": sx,
            "sy": sy,
            "ss": ss,
            "fisheye": calib.get("lens_model", "standard") == "fisheye",
        }
        self._set_coord_btn.setEnabled(False)

        if self._worker is not None:
            # Stream running → collect a short burst; _on_frame_ready feeds the
            # frames and the job is submitted once enough are gathered.
            self._coordsys_pairs = []
            self._calib_status_label.setText(
                f"Analysing the floor board over the next {self._COORDSYS_FRAMES} frames…"
            )
        else:
            # No live stream → single (last) frame through the same pipeline.
            fl, fr = self._last_frame.frame_left, self._last_frame.frame_right
            self._calib_status_label.setText("Analysing the floor board…")
            self._submit_coordsys_job([(fl.copy(), fr.copy())])

    def _coordsys_feed(self, fl: np.ndarray, fr: np.ndarray) -> None:
        """Accumulate live frames for a pending 'Set coordinate system' click."""
        if self._coordsys_pairs is None:
            return
        self._coordsys_pairs.append((fl.copy(), fr.copy()))
        if len(self._coordsys_pairs) >= self._COORDSYS_FRAMES:
            pairs, self._coordsys_pairs = self._coordsys_pairs, None
            self._submit_coordsys_job(pairs)

    def _submit_coordsys_job(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
        ctx = self._coordsys_ctx or {}
        try:
            self._coordsys_future = self._detect_pool.submit(
                CaptureTab._coordsys_job,
                self._detector,
                pairs,
                ctx.get("calib"),
                ctx.get("sx", 0),
                ctx.get("sy", 0),
                ctx.get("ss", 0.0),
                ctx.get("fisheye", False),
            )
        except RuntimeError:  # pool already shut down (app closing)
            self._pool_shutdown = True
            self._coordsys_future = None
            self._set_coord_btn.setEnabled(True)
            return
        self._coordsys_poll.start()

    def _poll_coordsys(self) -> None:
        fut = self._coordsys_future
        if fut is None:
            self._coordsys_poll.stop()
            return
        if not fut.done():
            return
        self._coordsys_poll.stop()
        self._coordsys_future = None
        try:
            res = fut.result()
        except Exception as exc:  # job crashed — report, don't swallow
            res = {"wf": None, "detail": "", "nodetect": False, "error": str(exc)}
        self._on_coordsys_done(res)

    @staticmethod
    def _coordsys_job(
        detector,
        pairs: list[tuple[np.ndarray, np.ndarray]],
        calib: dict,
        sx: int,
        sy: int,
        ss: float,
        fisheye: bool,
    ) -> dict:
        """Pooled compute for 'Set coordinate system' (no Qt — runs off-thread).

        Returns {"wf": dict|None, "detail": str, "nodetect": bool, "error": str|None}.
        """
        from app.calib.coordinate_system import (
            compute_world_frame,
            compute_world_frame_monocular,
            detect_chessboard_corners,
            vote_world_frames,
        )

        def _ok(d) -> bool:
            return d is not None and not getattr(d, "partial", False) and len(d.img_pts) >= 6

        # Strategy 1 — ChArUco stereo on the first pair (IDs anchor orientation,
        # stereo Kabsch has no planar ambiguity → one frame suffices).
        if detector is not None:
            try:
                det_l, det_r = CaptureTab._detect_pair(detector, pairs[0][0], pairs[0][1])
                if _ok(det_l) and _ok(det_r):
                    wf = compute_world_frame(det_l, det_r, calib)
                    detail = (
                        f"ChArUco stereo — fit RMS {wf['fit_rms_m'] * 1000:.1f} mm "
                        f"over {wf['n_corners']} corners"
                    )
                    return {"wf": wf, "detail": detail, "nodetect": False, "error": None}
            except Exception:
                pass  # fall through to chessboard

        if not (sx >= 2 and sy >= 2 and ss > 0):
            return {"wf": None, "detail": "", "nodetect": True, "error": None}

        # Strategy 2 — chessboard.  Decide which camera sees the board from the
        # first few pairs (left preferred — it IS the reference frame), then run
        # monocular solvePnP on every frame and majority-vote the pose.
        side = None
        for fl, fr in pairs[:3]:
            if detect_chessboard_corners(cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), sx, sy) is not None:
                side = "left"
                break
            if detect_chessboard_corners(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), sx, sy) is not None:
                side = "right"
                break
        if side is None:
            return {"wf": None, "detail": "", "nodetect": True, "error": None}

        cands = []
        for fl, fr in pairs:
            gray = cv2.cvtColor(fl if side == "left" else fr, cv2.COLOR_BGR2GRAY)
            ip = detect_chessboard_corners(gray, sx, sy)
            if ip is None:
                continue
            try:
                if side == "left":
                    cands.append(
                        compute_world_frame_monocular(
                            ip, calib["K1"], calib["D1"], sx, sy, ss, fisheye
                        )
                    )
                else:
                    cands.append(
                        compute_world_frame_monocular(
                            ip,
                            calib["K2"],
                            calib["D2"],
                            sx,
                            sy,
                            ss,
                            fisheye,
                            cam_to_ref=(calib["R"], calib["T"]),
                        )
                    )
            except ValueError:
                continue  # bad frame (e.g. partial detection) — skip
        if not cands:
            return {"wf": None, "detail": "", "nodetect": True, "error": None}

        try:
            wf = vote_world_frames(cands)
        except ValueError as exc:
            return {"wf": None, "detail": "", "nodetect": False, "error": str(exc)}
        detail = (
            f"chessboard ({side} camera) — {wf['n_agree']}/{wf['n_votes']} frames agree, "
            f"reproj RMS {wf['fit_rms_px']:.1f} px"
            if "fit_rms_px" in wf
            else f"chessboard ({side} camera) — {wf['n_agree']}/{wf['n_votes']} frames agree"
        )
        return {"wf": wf, "detail": detail, "nodetect": False, "error": None}

    def _on_coordsys_done(self, res: dict) -> None:
        """Apply the result of a 'Set coordinate system' job (GUI thread)."""
        from pathlib import Path as _Path

        from app.calib.stereo import update_world_frame

        self._set_coord_btn.setEnabled(True)
        wf = res.get("wf")
        if wf is None:
            self._calib_status_label.setText("")
            if res.get("error"):
                QMessageBox.warning(
                    self, "Coordinate system", f"Pose estimation failed:\n{res['error']}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Board not detected",
                    "Could not find the calibration board in EITHER camera.\n\n"
                    "• Lay it flat on the floor where at least one camera sees the whole "
                    "board clearly (it need not be both).\n"
                    "• If it's far away, the ArUco markers can't be decoded — the plain "
                    "checker pattern is used instead, so keep the whole board visible "
                    "and unblurred.\n"
                    "• A larger board, or moving it a little closer, helps.",
                )
            return
        if not self._calib_saved_path:
            return  # calibration vanished mid-flight — nothing to patch

        update_world_frame(self._calib_saved_path, wf)

        # Draw the axis triad on both previews (X red, Y green, Z blue); the
        # cached pixels keep it on every subsequent live frame.
        calib = (self._coordsys_ctx or {}).get("calib") or {}
        try:
            if self._last_frame is not None:
                self._draw_world_axes(
                    wf, calib, self._last_frame.frame_left, self._last_frame.frame_right
                )
            else:
                self._world_axes_px = CaptureTab._project_world_axes(wf, calib)
        except Exception:
            pass  # visualization is best-effort; the frame is already saved

        self._calib_status_label.setText(
            f"<b>✓ Coordinate system set</b> — origin at board centre; "
            f"X = {wf['board_long_m'] * 100:.0f} cm (long side), "
            f"Y = {wf['board_short_m'] * 100:.0f} cm (short side), Z up.<br>"
            f"<span style='color:#888'>{res.get('detail', '')}. Saved to "
            f"{_Path(self._calib_saved_path).name}.</span>"
        )

        # If Live Pose is currently tracking off this same calibration file, pick
        # the new frame up immediately: the very next pose frame is re-expressed
        # in the floor world frame and the viewer floor/origin re-anchor to the
        # board (no restart needed).
        if self._pose_active and self._pose_calib is not None:
            try:
                same = (
                    _Path(self._pose_calib_edit.text().strip()).resolve()
                    == _Path(self._calib_saved_path).resolve()
                )
            except Exception:
                same = False
            if same:
                self._pose_calib["world_frame"] = wf
                self._pose_world = True
                self._preview_3d.setup_live_view(world_frame=True)

        # Re-emit so the Reconstruction tab reloads the (now world-aware) calib.
        self.calibration_saved.emit(self._calib_saved_path)

    @staticmethod
    def _project_world_axes(world_frame: dict, calib: dict) -> tuple[np.ndarray, np.ndarray]:
        """Project the world-frame triad into both views → (px_left[4,2], px_right[4,2]).

        The endpoints are fixed in camera-1 coords, so the result is constant for
        a given (world_frame, calib) — compute once and reuse every frame.
        """
        from app.calib.coordinate_system import axis_endpoints_cam1
        from app.recon3d.triangulate import project_points

        fisheye = calib.get("lens_model", "standard") == "fisheye"
        pts3d = axis_endpoints_cam1(world_frame)  # [4,3] in camera-1 coords
        # Left camera: points already in camera-1 frame (R=I, t=0).
        px_l = project_points(pts3d, calib["K1"], calib["D1"], np.eye(3), np.zeros(3), fisheye)
        # Right camera: transform camera-1 → camera-2 via the stereo extrinsics.
        px_r = project_points(
            pts3d, calib["K2"], calib["D2"], calib["R"], calib["T"].reshape(3), fisheye
        )
        return px_l, px_r

    def _refresh_world_axes_overlay(self) -> None:
        """Load the current calibration's world frame (if any) and cache the
        projected axis triad so it's drawn on every live preview frame.

        Clears the overlay when there's no calibration or no world frame.
        """
        self._world_axes_px = None
        path = self._calib_saved_path
        if not path or not Path(path).is_file():
            return
        try:
            from app.calib.stereo import load_calibration

            calib = load_calibration(path)
            wf = calib.get("world_frame")
            if wf:
                self._world_axes_px = CaptureTab._project_world_axes(wf, calib)
        except Exception:
            self._world_axes_px = None  # best-effort; never break the stream

    def _maybe_overlay_world_axes(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw the cached world-frame triad onto preview copies, if set."""
        if self._world_axes_px is None:
            return left, right
        px_l, px_r = self._world_axes_px
        return (
            CaptureTab._draw_axes(left.copy(), px_l),
            CaptureTab._draw_axes(right.copy(), px_r),
        )

    def _overlay_world_axes_side(self, src: np.ndarray, side: str) -> np.ndarray:
        """Return *src* with the world-frame triad for one camera ("L"/"R") drawn
        on a copy, if a frame is set — used by the fullscreen inspect window so the
        triad shows there too (e.g. while checking the coordinate system during
        calibration)."""
        if self._world_axes_px is None:
            return src
        px = self._world_axes_px[0] if side == "L" else self._world_axes_px[1]
        return CaptureTab._draw_axes(src.copy(), px)

    def _draw_world_axes(
        self, world_frame: dict, calib: dict, fl: np.ndarray, fr: np.ndarray
    ) -> None:
        """Project the world-frame axis triad into both views and show it once
        (instant feedback; subsequent frames redraw it from the cached pixels)."""
        px_l, px_r = CaptureTab._project_world_axes(world_frame, calib)
        self._world_axes_px = (px_l, px_r)
        _set_preview(self._preview_left, CaptureTab._draw_axes(fl.copy(), px_l))
        _set_preview(self._preview_right, CaptureTab._draw_axes(fr.copy(), px_r))

    @staticmethod
    def _draw_axes(bgr: np.ndarray, px: np.ndarray) -> np.ndarray:
        """Draw the [origin, +X, +Y, +Z] triad as thin, semi-transparent arrows.

        Mutates *bgr* in place (every caller passes a private copy).  The alpha
        blend is restricted to the triad's bounding box — blending the whole
        1920×1200 frame every tick was a real cost, especially behind the
        fullscreen up-scale.  Guards non-finite pixels (a point projected behind
        the camera → inf/NaN) so it never raises inside the frame callback.
        """
        px = np.asarray(px, dtype=np.float64)
        if not np.isfinite(px).all():
            return bgr  # endpoint projects to infinity — skip the overlay this call
        h, w = bgr.shape[:2]
        m = 50  # margin for arrowheads, labels, line width
        x0 = max(0, int(np.floor(px[:, 0].min())) - m)
        y0 = max(0, int(np.floor(px[:, 1].min())) - m)
        x1 = min(w, int(np.ceil(px[:, 0].max())) + m)
        y1 = min(h, int(np.ceil(px[:, 1].max())) + m)
        if x1 <= x0 or y1 <= y0:
            return bgr  # triad entirely off-frame
        roi = bgr[y0:y1, x0:x1]
        # Draw on a copy of just the ROI, then alpha-blend it back: untouched
        # pixels are identical so only the arrows/labels appear semi-transparent.
        overlay = roi.copy()
        o = (int(round(px[0, 0])) - x0, int(round(px[0, 1])) - y0)
        # BGR: X=red, Y=green, Z=blue (matches the 3D viewer legend).
        for i, (col, lab) in enumerate(
            [((0, 0, 255), "X"), ((0, 255, 0), "Y"), ((255, 0, 0), "Z")], start=1
        ):
            p = (int(round(px[i, 0])) - x0, int(round(px[i, 1])) - y0)
            cv2.arrowedLine(overlay, o, p, col, 2, cv2.LINE_AA, tipLength=0.2)
            cv2.putText(overlay, lab, p, cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        cv2.circle(overlay, o, 4, (255, 255, 255), -1, cv2.LINE_AA)
        alpha = 0.7  # a bit transparent
        roi[:] = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0)
        return bgr

    # ------------------------------------------------------------------
    # Source factory
    # ------------------------------------------------------------------

    def _build_source(self):
        serial_l = self._left_combo.currentText().strip() or None
        serial_r = self._right_combo.currentText().strip() or None
        if serial_l and serial_r and serial_l == serial_r:
            QMessageBox.warning(
                self,
                "Same serial for both cameras",
                "Left and right cameras have the same serial number. "
                "Select different cameras in Step 3.",
            )
            return None
        from app.capture.flir import FlirCapture

        return FlirCapture(
            serial_left=serial_l,
            serial_right=serial_r,
            fps=self._fps_spin.value(),
            exposure_us=self._exposure_spin.value(),
            gain_db=self._gain_spin.value(),
            sync=self._sync_check.isChecked(),
            primary="left" if self._primary_left.isChecked() else "right",
        )
