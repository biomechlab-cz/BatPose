"""
Side-by-side small previews of left/right videos with 2D pose-estimation overlay.

Used in the Reconstruction tab so the user can sanity-check that the 2D
detections align with the actual person before trusting the 3D output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

if TYPE_CHECKING:
    pass

# COCO-17 edges (kept in sync with viewer3d.COCO17_EDGES).
COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
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


class Pose2DPreview(QWidget):
    """
    Two small video frames side-by-side (left / right camera), each with the
    detected COCO-17 skeleton drawn on top. show_frame(t) seeks both videos
    to frame t and refreshes the overlays.

    set_data() takes optional video paths and pose2d NPZ paths; either side
    is independent — if e.g. the pose2d_left.npz is missing, the left preview
    just shows the raw frame.
    """

    PREVIEW_W = 320
    PREVIEW_H = 180

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cap_l: cv2.VideoCapture | None = None
        self._cap_r: cv2.VideoCapture | None = None
        self._kps_l: np.ndarray | None = None
        self._kps_r: np.ndarray | None = None
        self._conf_l: np.ndarray | None = None
        self._conf_r: np.ndarray | None = None
        self._n_frames_l = 0
        self._n_frames_r = 0
        # Track last frame read per side for the seek-or-read-forward shortcut.
        self._last_t_l = -2
        self._last_t_r = -2
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
            "background-color: #1e1e1e; color: #888; font-size: 11px; "
            "border: 1px solid #333;"
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

        if video_left and Path(video_left).is_file():
            cap = cv2.VideoCapture(video_left)
            if cap.isOpened():
                self._cap_l = cap
                self._n_frames_l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                cap.release()

        if video_right and Path(video_right).is_file():
            cap = cv2.VideoCapture(video_right)
            if cap.isOpened():
                self._cap_r = cap
                self._n_frames_r = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                cap.release()

        if pose2d_left and Path(pose2d_left).is_file():
            try:
                from app.pose2d.pipeline import load_pose2d

                kps, conf, _meta = load_pose2d(pose2d_left)
                self._kps_l = kps
                self._conf_l = conf
            except Exception:
                self._kps_l = None
                self._conf_l = None

        if pose2d_right and Path(pose2d_right).is_file():
            try:
                from app.pose2d.pipeline import load_pose2d

                kps, conf, _meta = load_pose2d(pose2d_right)
                self._kps_r = kps
                self._conf_r = conf
            except Exception:
                self._kps_r = None
                self._conf_r = None

        self._update_placeholders()
        # Render frame 0 if anything loaded.
        if self._cap_l is not None or self._cap_r is not None:
            self.show_frame(0)

    def clear(self) -> None:
        for cap in (self._cap_l, self._cap_r):
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        self._cap_l = None
        self._cap_r = None
        self._kps_l = None
        self._kps_r = None
        self._conf_l = None
        self._conf_r = None
        self._n_frames_l = 0
        self._n_frames_r = 0
        self._last_t_l = -2
        self._last_t_r = -2
        self._left_lbl.setPixmap(QPixmap())
        self._right_lbl.setPixmap(QPixmap())
        self._left_lbl.setText("(no left preview yet)")
        self._right_lbl.setText("(no right preview yet)")

    def show_frame(self, t: int) -> None:
        """Seek (or step) both sides to frame *t* and refresh the labels."""
        self._render_side(
            self._cap_l, self._kps_l, self._conf_l, self._n_frames_l,
            t, self._left_lbl, "_last_t_l",
        )
        self._render_side(
            self._cap_r, self._kps_r, self._conf_r, self._n_frames_r,
            t, self._right_lbl, "_last_t_r",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_placeholders(self) -> None:
        if self._cap_l is None:
            self._left_lbl.setText("(no left video)")
        elif self._kps_l is None:
            self._left_lbl.setText("(left video, no pose2d)")
        if self._cap_r is None:
            self._right_lbl.setText("(no right video)")
        elif self._kps_r is None:
            self._right_lbl.setText("(right video, no pose2d)")

    def _render_side(
        self,
        cap: cv2.VideoCapture | None,
        kps: np.ndarray | None,
        conf: np.ndarray | None,
        n_frames: int,
        t: int,
        label: QLabel,
        last_attr: str,
    ) -> None:
        if cap is None:
            return
        if n_frames > 0:
            t = max(0, min(t, n_frames - 1))
        else:
            t = max(0, t)

        last = getattr(self, last_attr)
        if t != last + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(t))

        ret, frame = cap.read()
        if not ret or frame is None:
            return
        setattr(self, last_attr, t)

        if kps is not None and conf is not None and t < kps.shape[0]:
            self._draw_overlay(frame, kps[t], conf[t])

        pixmap = self._frame_to_pixmap(frame, label.size())
        label.setPixmap(pixmap)

    @staticmethod
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

    @staticmethod
    def _frame_to_pixmap(frame_bgr: np.ndarray, target_size) -> QPixmap:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Make sure the buffer is contiguous; QImage needs a stable backing array.
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        return pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 (Qt API)
        self.clear()
        super().closeEvent(event)
