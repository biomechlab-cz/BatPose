"""
QThread workers for long-running pipeline stages.

Each worker:
- Accepts task parameters in __init__
- Runs the pipeline in a background thread (non-blocking UI)
- Emits: progress(int, str), finished(object), error(str)
- Is cancellable via cancel()
"""

from __future__ import annotations

import traceback
from typing import Any

import cv2 as _cv2
from PySide6.QtCore import QThread, Signal


class _BaseWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _check_cancelled(self) -> bool:
        return self._cancelled

    def _progress(self, pct: int, msg: str = "") -> None:
        self.progress.emit(pct, msg)

    # Exceptions that carry a user-readable message — show cleanly, no traceback.
    _USER_EXCEPTIONS = (RuntimeError, ValueError, FileNotFoundError, OSError)

    def run(self) -> None:
        try:
            result = self._run()
            # Always emit finished (result=None when cancelled) so UI re-enables buttons.
            self.finished.emit(result if not self._cancelled else None)
        except self._USER_EXCEPTIONS as exc:
            self.error.emit(str(exc))
        except Exception:
            self.error.emit(traceback.format_exc())

    def _run(self) -> Any:
        raise NotImplementedError


class CalibWorker(_BaseWorker):
    """Run the full stereo calibration pipeline in a background thread."""

    # Emits (left_bgr, right_bgr) numpy arrays when a good frame pair is detected.
    frame_ready = Signal(object)

    def __init__(
        self,
        video_left: str,
        video_right: str,
        board_cfg: dict,
        output_path: str,
        max_frames: int = 60,
        sample_every: int = 5,
        parent=None,
    ):
        super().__init__(parent)
        self._video_left = video_left
        self._video_right = video_right
        self._board_cfg = board_cfg
        self._output_path = output_path
        self._max_frames = max_frames
        self._sample_every = sample_every

    def _run(self) -> Any:
        from app.calib.stereo import run_calibration_pipeline

        def _scan_cb(frame_l, frame_r, det_l_ok, det_r_ok) -> None:
            ann_l = frame_l.copy()
            ann_r = frame_r.copy()
            color_l = (0, 255, 0) if det_l_ok else (0, 0, 200)
            _cv2.putText(
                ann_l,
                "OK" if det_l_ok else "\u2014",
                (10, 36),
                _cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color_l,
                2,
            )
            color_r = (0, 255, 0) if det_r_ok else (0, 0, 200)
            _cv2.putText(
                ann_r,
                "OK" if det_r_ok else "\u2014",
                (10, 36),
                _cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color_r,
                2,
            )
            self.frame_ready.emit((ann_l, ann_r))

        result = run_calibration_pipeline(
            self._video_left,
            self._video_right,
            self._board_cfg,
            self._output_path,
            max_frames=self._max_frames,
            sample_every=self._sample_every,
            progress_cb=self._progress,
            cancel_check=self._check_cancelled,
            frame_cb=None,
            scan_cb=_scan_cb,
        )
        return result


class Pose2DWorker(_BaseWorker):
    """Run 2D pose extraction for both cameras in sequence."""

    def __init__(
        self,
        video_left: str,
        video_right: str,
        out_left: str,
        out_right: str,
        backend_name: str = "mediapipe",
        num_poses: int = 2,
        parent=None,
    ):
        super().__init__(parent)
        self._video_left = video_left
        self._video_right = video_right
        self._out_left = out_left
        self._out_right = out_right
        self._backend_name = backend_name
        self._num_poses = num_poses

    def _make_backend(self):
        if self._backend_name == "mediapipe":
            from app.pose2d.mediapipe_backend import MediaPipeBackend

            return MediaPipeBackend(num_poses=self._num_poses)
        elif self._backend_name == "rtmpose":
            from app.pose2d.rtmpose_backend import RTMPoseBackend

            return RTMPoseBackend()
        else:
            raise ValueError(f"Unknown backend: {self._backend_name!r}")

    def _run(self) -> Any:
        from app.pose2d.pipeline import process_video, save_pose2d

        results = {}
        for label, vid, out in [
            ("left", self._video_left, self._out_left),
            ("right", self._video_right, self._out_right),
        ]:
            if self._cancelled:
                return None

            backend = self._make_backend()
            try:

                def _cb(pct: int, msg: str, lbl: str = label) -> None:
                    # Map 0-100 to 0-50 for left, 50-100 for right
                    offset = 0 if lbl == "left" else 50
                    self._progress(offset + pct // 2, f"[{lbl.upper()}] {msg}")

                kps, conf, meta = process_video(
                    vid,
                    backend,
                    progress_cb=_cb,
                    cancel_check=self._check_cancelled,
                )
                if not self._cancelled:
                    save_pose2d(kps, conf, meta, out)
                    results[label] = out
            finally:
                backend.close()

        return results


class Recon3DWorker(_BaseWorker):
    """Run 3D reconstruction in a background thread."""

    def __init__(
        self,
        calib_path: str,
        pose2d_left: str,
        pose2d_right: str,
        output_path: str,
        min_conf: float = 0.3,
        max_reproj_err: float = 20.0,
        min_cutoff: float = 0.5,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
        parent=None,
    ):
        super().__init__(parent)
        self._calib = calib_path
        self._left = pose2d_left
        self._right = pose2d_right
        self._out = output_path
        self._min_conf = min_conf
        self._max_reproj_err = max_reproj_err
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff

    def _run(self) -> Any:
        from app.recon3d.pipeline import reconstruct3d

        return reconstruct3d(
            calib_path=self._calib,
            pose2d_left_path=self._left,
            pose2d_right_path=self._right,
            output_path=self._out,
            min_conf=self._min_conf,
            max_reproj_err=self._max_reproj_err,
            min_cutoff=self._min_cutoff,
            beta=self._beta,
            d_cutoff=self._d_cutoff,
            progress_cb=self._progress,
            cancel_check=self._check_cancelled,
        )
