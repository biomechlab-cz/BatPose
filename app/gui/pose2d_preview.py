"""
Side-by-side small previews of left/right videos with 2D pose-estimation overlay.

Used in the Reconstruction and Analysis tabs so the user can sanity-check that
the 2D detections align with the actual person.

Decoding runs on a background thread (`_PreviewDecodeWorker`).  The two source
clips are 1920×1200 and decode at ~10 ms/frame each, which is far too heavy to
do synchronously on the GUI thread during 50 fps playback — doing so stalls the
3D viewer.  Instead `show_frame(t)` just records the desired frame; the worker
decodes the latest requested frame best-effort and posts a finished QImage back
to the GUI thread.  Requests coalesce (only the most recent target matters), so
the preview naturally drops frames to keep up while the 3D view stays smooth.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, Qt, QThread, QWaitCondition, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

# COCO-17 edges (kept in sync with viewer3d.COCO17_EDGES).
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
_LEFT_JOINTS = {1, 3, 5, 7, 9, 11, 13, 15}
_RIGHT_JOINTS = {2, 4, 6, 8, 10, 12, 14, 16}

# BGR colours mirroring the 3D viewer's left=blue / right=red / centre=green.
_BGR_LEFT = (255, 128, 51)
_BGR_RIGHT = (76, 76, 255)
_BGR_CENTER = (76, 255, 76)
_BGR_JOINT = (0, 255, 255)  # yellow dots

# Confidence threshold below which an edge/joint is hidden.
_MIN_DRAW_CONF = 0.1

PREVIEW_W = 320
PREVIEW_H = 180


def _edge_color_bgr(i: int, j: int) -> tuple[int, int, int]:
    if i in _LEFT_JOINTS or j in _LEFT_JOINTS:
        return _BGR_LEFT
    if i in _RIGHT_JOINTS or j in _RIGHT_JOINTS:
        return _BGR_RIGHT
    return _BGR_CENTER


def derive_pose2d_paths(pose3d_path: str) -> tuple[str | None, str | None]:
    """
    Given a pose3d.npz path, infer the conventional pose2d_{left,right}.npz
    paths next to it. Returns (left, right) where each is the path if it
    exists on disk, else None.
    """
    p = Path(pose3d_path).parent
    left = p / "pose2d_left.npz"
    right = p / "pose2d_right.npz"
    return (
        str(left) if left.is_file() else None,
        str(right) if right.is_file() else None,
    )


def _draw_overlay(frame: np.ndarray, kps_t: np.ndarray, conf_t: np.ndarray) -> None:
    """Draw COCO-17 skeleton on *frame* in place. kps_t: [P,17,2], conf_t: [P,17]."""
    if kps_t.ndim != 3 or kps_t.shape[-1] != 2:
        return
    P = kps_t.shape[0]
    for p_idx in range(P):
        kp = kps_t[p_idx]
        cv = conf_t[p_idx]
        for i, j in COCO17_EDGES:
            if cv[i] < _MIN_DRAW_CONF or cv[j] < _MIN_DRAW_CONF:
                continue
            pt0 = (int(kp[i, 0]), int(kp[i, 1]))
            pt1 = (int(kp[j, 0]), int(kp[j, 1]))
            cv2.line(frame, pt0, pt1, _edge_color_bgr(i, j), 2, cv2.LINE_AA)
        for j_idx in range(17):
            if cv[j_idx] < _MIN_DRAW_CONF:
                continue
            pt = (int(kp[j_idx, 0]), int(kp[j_idx, 1]))
            cv2.circle(frame, pt, 3, _BGR_JOINT, -1, cv2.LINE_AA)


def _frame_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert a BGR frame to a scaled, self-owned QImage (safe to cross threads)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    # .copy() detaches from the local numpy buffer (freed when this returns).
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return qimg.scaled(
        PREVIEW_W,
        PREVIEW_H,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class _PreviewDecodeWorker(QThread):
    """Background video decoder for the side previews.

    Owns the two VideoCaptures (opened inside run(), so they live entirely on
    this thread).  ``request(frame)`` sets the latest desired frame and wakes
    the loop; the loop decodes that frame on both clips, draws the overlay, and
    emits ``ready`` with scaled QImages for the GUI thread to display.
    """

    ready = Signal(int, object, object)  # frame_index, qimg_left|None, qimg_right|None

    def __init__(
        self,
        video_left: str | None,
        video_right: str | None,
        kps_l: np.ndarray | None,
        conf_l: np.ndarray | None,
        kps_r: np.ndarray | None,
        conf_r: np.ndarray | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._vl = video_left
        self._vr = video_right
        self._kps_l, self._conf_l = kps_l, conf_l
        self._kps_r, self._conf_r = kps_r, conf_r
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._target: int | None = None
        self._abort = False
        self._last_l = -2
        self._last_r = -2

    def request(self, frame: int) -> None:
        with QMutexLocker(self._mutex):
            self._target = frame
            self._cond.wakeAll()

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._abort = True
            self._cond.wakeAll()
        self.wait(3000)

    # -- worker thread ------------------------------------------------------

    def run(self) -> None:
        cap_l = self._open(self._vl)
        cap_r = self._open(self._vr)
        n_l = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_l else 0
        n_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_r else 0
        try:
            while True:
                with QMutexLocker(self._mutex):
                    while self._target is None and not self._abort:
                        self._cond.wait(self._mutex)
                    if self._abort:
                        break
                    t = self._target
                    self._target = None
                qil = self._decode(cap_l, self._kps_l, self._conf_l, n_l, t, "l")
                qir = self._decode(cap_r, self._kps_r, self._conf_r, n_r, t, "r")
                if not self._abort:
                    self.ready.emit(t, qil, qir)
        finally:
            if cap_l is not None:
                cap_l.release()
            if cap_r is not None:
                cap_r.release()

    @staticmethod
    def _open(path: str | None):
        if not path:
            return None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _decode(self, cap, kps, conf, n_frames, t, side):
        if cap is None:
            return None
        if n_frames > 0:
            t = max(0, min(t, n_frames - 1))
        else:
            t = max(0, t)
        last = self._last_l if side == "l" else self._last_r
        delta = t - last
        # Reaching frame t: sequential read is ~10 ms, a random seek ~70 ms.
        #   delta == 1 → plain read; 2..7 ahead → grab()-skip; else → seek.
        if last >= 0 and delta == 1:
            pass
        elif last >= 0 and 2 <= delta <= 7:
            for _ in range(delta - 1):
                if not cap.grab():
                    break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(t))
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        if side == "l":
            self._last_l = t
        else:
            self._last_r = t
        if kps is not None and conf is not None and t < kps.shape[0]:
            _draw_overlay(frame, kps[t], conf[t])
        return _frame_to_qimage(frame)


class Pose2DPreview(QWidget):
    """
    Two small video frames side-by-side (left / right camera), each with the
    detected COCO-17 skeleton drawn on top.  show_frame(t) requests both videos
    be shown at frame t; decoding happens on a background thread.

    set_data() takes optional video paths and pose2d NPZ paths; either side is
    independent — if e.g. pose2d_left.npz is missing, the left preview shows the
    raw frame.
    """

    PREVIEW_W = PREVIEW_W
    PREVIEW_H = PREVIEW_H

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: _PreviewDecodeWorker | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        title_row = QHBoxLayout()
        lbl_l = QLabel("Left 2D")
        lbl_r = QLabel("Right 2D")
        for lbl in (lbl_l, lbl_r):
            lbl.setStyleSheet("color: #aaa; font-size: 10px; font-weight: bold;")
            lbl.setFixedWidth(self.PREVIEW_W)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_row.addWidget(lbl_l)
        title_row.addWidget(lbl_r)
        title_row.addStretch()
        outer.addLayout(title_row)

        img_row = QHBoxLayout()
        img_row.setSpacing(4)
        self._left_lbl = self._make_image_label("(no left preview yet)")
        self._right_lbl = self._make_image_label("(no right preview yet)")
        img_row.addWidget(self._left_lbl)
        img_row.addWidget(self._right_lbl)
        img_row.addStretch()
        outer.addLayout(img_row)

    def _make_image_label(self, placeholder: str) -> QLabel:
        lbl = QLabel(placeholder)
        lbl.setFixedSize(self.PREVIEW_W, self.PREVIEW_H)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            "background-color: #1e1e1e; color: #888; font-size: 11px; border: 1px solid #333;"
        )
        return lbl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(
        self,
        video_left: str | None,
        video_right: str | None,
        pose2d_left: str | None,
        pose2d_right: str | None,
    ) -> None:
        """Load videos and pose2d arrays. Any of the four can be None."""
        self.clear()

        has_left = bool(video_left and Path(video_left).is_file())
        has_right = bool(video_right and Path(video_right).is_file())

        kps_l = conf_l = kps_r = conf_r = None
        if pose2d_left and Path(pose2d_left).is_file():
            kps_l, conf_l = self._load_pose2d(pose2d_left)
        if pose2d_right and Path(pose2d_right).is_file():
            kps_r, conf_r = self._load_pose2d(pose2d_right)

        # Placeholder text for sides without video / pose.
        if not has_left:
            self._left_lbl.setText("(no left video)")
        elif kps_l is None:
            self._left_lbl.setText("(left video, no pose2d)")
        if not has_right:
            self._right_lbl.setText("(no right video)")
        elif kps_r is None:
            self._right_lbl.setText("(right video, no pose2d)")

        # Only spin up a decode thread when there is at least one real video.
        if has_left or has_right:
            self._worker = _PreviewDecodeWorker(
                video_left if has_left else None,
                video_right if has_right else None,
                kps_l,
                conf_l,
                kps_r,
                conf_r,
                parent=self,
            )
            self._worker.ready.connect(self._on_frame_ready)
            self._worker.start()
            self.show_frame(0)

    @staticmethod
    def _load_pose2d(path: str):
        try:
            from app.pose2d.pipeline import load_pose2d

            kps, conf, _meta = load_pose2d(path)
            return kps, conf
        except Exception:
            return None, None

    def show_frame(self, t: int) -> None:
        """Request frame *t* be shown.  Non-blocking — the worker decodes it."""
        if self._worker is not None:
            self._worker.request(int(t))

    def clear(self) -> None:
        if self._worker is not None:
            try:
                self._worker.ready.disconnect(self._on_frame_ready)
            except (RuntimeError, TypeError):
                pass
            self._worker.stop()
            self._worker = None
        self._left_lbl.setPixmap(QPixmap())
        self._right_lbl.setPixmap(QPixmap())
        self._left_lbl.setText("(no left preview yet)")
        self._right_lbl.setText("(no right preview yet)")

    # ------------------------------------------------------------------
    # Worker callback (GUI thread)
    # ------------------------------------------------------------------

    def _on_frame_ready(self, _frame: int, qimg_l, qimg_r) -> None:
        if qimg_l is not None and not qimg_l.isNull():
            self._left_lbl.setPixmap(QPixmap.fromImage(qimg_l))
        if qimg_r is not None and not qimg_r.isNull():
            self._right_lbl.setPixmap(QPixmap.fromImage(qimg_r))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 (Qt API)
        self.clear()
        super().closeEvent(event)
