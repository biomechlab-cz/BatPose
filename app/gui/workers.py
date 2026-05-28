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
    """Run 2D pose extraction for both cameras in parallel.

    Uses IMAGE (stateless, per-frame) mode for the offline pipeline so that
    fast-movement frames are analysed independently without MediaPipe's
    built-in temporal smoothing, which introduces lag on quick gestures.
    Temporal smoothing is handled explicitly by the OneEuro filter in the
    3D reconstruction step.

    Both camera videos are processed concurrently in separate threads —
    IMAGE-mode backends have no cross-frame state so they are fully
    independent and safe to parallelize.
    """

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

            # IMAGE (stateless) mode for offline reconstruction: each frame is
            # analysed independently, which avoids MediaPipe's internal temporal
            # smoothing that blurs fast-movement keypoints.  Temporal coherence
            # is restored afterwards by the OneEuro filter in the 3D pipeline.
            return MediaPipeBackend(num_poses=self._num_poses, running_mode="image")
        elif self._backend_name == "rtmpose":
            from app.pose2d.rtmpose_backend import RTMPoseBackend

            return RTMPoseBackend()
        else:
            raise ValueError(f"Unknown backend: {self._backend_name!r}")

    def _run(self) -> Any:
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from app.pose2d.pipeline import process_video, save_pose2d

        results: dict = {}
        first_exc: list = []

        # Shared progress state — protected by _lock so the combined value
        # is always consistent regardless of which thread updates it.
        _lock = threading.Lock()
        _pct: dict[str, int] = {"left": 0, "right": 0}

        def _process_one(label: str, vid: str, out: str) -> None:
            """Process one camera video in its own thread with its own backend."""
            backend = self._make_backend()
            try:
                _last_milestone: list[int] = [-1]  # per-camera milestone tracker

                def _cb(pct: int, msg: str) -> None:
                    # Update this camera's progress and derive the combined
                    # average so the bar advances smoothly without jumps.
                    with _lock:
                        _pct[label] = pct
                        combined = (_pct["left"] + _pct["right"]) // 2
                    # Log only at 0 / 25 / 50 / 75 / 100% milestones to
                    # avoid flooding the log with per-frame messages.
                    log_msg = ""
                    milestone = (pct // 25) * 25
                    if milestone != _last_milestone[0]:
                        _last_milestone[0] = milestone
                        log_msg = f"[{label.upper()}] {pct}%  {msg}"
                    self.progress.emit(combined, log_msg)

                kps, conf, meta = process_video(
                    vid,
                    backend,
                    progress_cb=_cb,
                    cancel_check=self._check_cancelled,
                )
                if not self._cancelled:
                    save_pose2d(kps, conf, meta, out)
                    results[label] = out
                    with _lock:
                        _pct[label] = 100
                        combined = (_pct["left"] + _pct["right"]) // 2
                    self.progress.emit(combined, f"[{label.upper()}] done")
            except Exception as exc:
                first_exc.append(exc)
                raise
            finally:
                backend.close()

        # Both cameras run concurrently.  Each reports its own 0-100% progress;
        # the combined average is emitted so the progress bar advances smoothly
        # and the log only shows milestone lines (not every frame).
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pose2d") as pool:
            fut_l = pool.submit(_process_one, "left", self._video_left, self._out_left)
            fut_r = pool.submit(_process_one, "right", self._video_right, self._out_right)
            # Collect results / re-raise the first exception from either thread.
            for fut in (fut_l, fut_r):
                fut.result()

        if self._cancelled:
            return None
        return results


class CaptureWorker(_BaseWorker):
    """
    Live stereo capture worker.

    Streams frames from any BaseCapture source.
    Emits frame_ready for UI preview at a throttled rate (~15 fps).
    Recording to AVI can be toggled at runtime via begin_recording / end_recording.
    When recording stops (or the worker is cancelled while recording), recording_finished
    is emitted with the output file paths.
    """

    frame_ready = Signal(object)  # CaptureFrame
    recording_finished = Signal(str, str)  # left_path, right_path

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self._source = source
        # These are written by the main thread and read by the worker thread.
        # Plain bool/str assignments are atomic under the GIL.
        self._do_record: bool = False
        self._out_left: str = ""
        self._out_right: str = ""

    def begin_recording(self, out_left: str, out_right: str) -> None:
        self._out_left = out_left
        self._out_right = out_right
        self._do_record = True

    def end_recording(self) -> None:
        self._do_record = False

    @staticmethod
    def _timestamps_path(out_left: str, out_right: str) -> str:
        """Derive the sidecar CSV path (e.g. '…/session_timestamps.csv')."""
        import os

        base = os.path.commonprefix([out_left, out_right]).rstrip("_")
        return f"{base}_timestamps.csv"

    def _run(self) -> Any:
        import cv2 as cv

        self._source.start()
        writer_l: Any = None
        writer_r: Any = None
        ts_file: Any = None  # sidecar CSV handle, open only while recording
        was_recording = False
        frame_idx = 0
        rec_idx = 0  # frames written to the current recording
        preview_stride = max(1, round(self._source.fps / 15))

        def _close_writers() -> None:
            nonlocal writer_l, writer_r, ts_file
            if writer_l is not None:
                writer_l.release()
            if writer_r is not None:
                writer_r.release()
            if ts_file is not None:
                ts_file.close()
            writer_l = writer_r = ts_file = None

        try:
            while not self._cancelled:
                frame = self._source.read()
                if frame is None:
                    break

                if frame_idx % preview_stride == 0:
                    self.frame_ready.emit(frame)

                # Transition: idle → recording
                if self._do_record and not was_recording:
                    h, w = frame.frame_left.shape[:2]
                    fourcc = cv.VideoWriter_fourcc(*"MJPG")
                    writer_l = cv.VideoWriter(self._out_left, fourcc, self._source.fps, (w, h))
                    writer_r = cv.VideoWriter(self._out_right, fourcc, self._source.fps, (w, h))
                    # Per-frame hardware-timestamp log so the pair can be
                    # sync-verified after the fact.  Columns: the written frame
                    # index, the left/right camera hardware timestamps (ns, from
                    # each camera's clock), and their absolute difference (µs).
                    ts_file = open(
                        self._timestamps_path(self._out_left, self._out_right),
                        "w",
                        encoding="utf-8",
                        newline="",
                    )
                    ts_file.write("frame,hw_left_ns,hw_right_ns,abs_delta_us\n")
                    rec_idx = 0
                    was_recording = True

                # Transition: recording → idle
                if not self._do_record and was_recording:
                    _close_writers()
                    was_recording = False
                    self.recording_finished.emit(self._out_left, self._out_right)

                if was_recording:
                    writer_l.write(frame.frame_left)
                    writer_r.write(frame.frame_right)
                    tl = frame.hw_timestamp_left_ns
                    tr = frame.hw_timestamp_right_ns
                    delta = frame.hw_delta_us
                    ts_file.write(
                        f"{rec_idx},"
                        f"{'' if tl is None else tl},"
                        f"{'' if tr is None else tr},"
                        f"{'' if delta is None else f'{delta:.3f}'}\n"
                    )
                    rec_idx += 1

                frame_idx += 1
                self._progress(0, f"Frame {frame.frame_index}  {frame.timestamp:.1f} s")

        finally:
            already = was_recording
            _close_writers()
            if already:
                self.recording_finished.emit(self._out_left, self._out_right)
            self._source.stop()

        return None


class LiveCalibWorker(_BaseWorker):
    """
    Run stereo calibration from frame pairs captured live from the cameras.

    Accepts the list of FrameSelection objects already collected in the UI,
    calls calibrate_stereo() directly (no video I/O), and saves calibration.yml.

    lens_model controls which distortion model is used:
        0 — Standard   (5 coefficients: k1 k2 p1 p2 k3)
        1 — Wide-angle  (8 coefficients: rational model, k1-k6)
        2 — Fisheye    (OpenCV fisheye θ-based model)
    """

    #: Lens model index → (label, intrinsics_flags, use_fisheye)
    _LENS_MODELS = [
        ("standard", 0, False),
        ("wide-angle", _cv2.CALIB_RATIONAL_MODEL, False),
        ("fisheye", 0, True),
    ]

    def __init__(
        self,
        selections: list,  # list[FrameSelection] — stereo pairs (extrinsics)
        img_size: tuple[int, int],
        board_cfg: dict,
        output_path: str,
        lens_model: int = 0,  # 0=standard, 1=wide-angle, 2=fisheye
        all_det_l: list | None = None,  # DetectionResult list — left intrinsics pool
        all_det_r: list | None = None,  # DetectionResult list — right intrinsics pool
        parent=None,
    ):
        super().__init__(parent)
        self._selections = selections
        self._img_size = img_size
        self._board_cfg = board_cfg
        self._output_path = output_path
        self._lens_model = lens_model
        self._all_det_l = all_det_l
        self._all_det_r = all_det_r

    def _run(self) -> Any:
        from app.calib.stereo import calibrate_stereo, calibrate_stereo_fisheye, save_calibration

        _label, intrinsics_flags, use_fisheye = self._LENS_MODELS[self._lens_model]

        if use_fisheye:
            calib = calibrate_stereo_fisheye(
                self._selections,
                self._img_size,
                progress_cb=self._progress,
                cancel_check=self._check_cancelled,
                all_det_l=self._all_det_l,
                all_det_r=self._all_det_r,
            )
        else:
            calib = calibrate_stereo(
                self._selections,
                self._img_size,
                intrinsics_flags=intrinsics_flags,
                progress_cb=self._progress,
                cancel_check=self._check_cancelled,
                all_det_l=self._all_det_l,
                all_det_r=self._all_det_r,
            )
        if self._cancelled:
            return None

        save_calibration(calib, self._board_cfg, self._output_path)
        return self._output_path


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
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
        no_smooth: bool = False,
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
        self._no_smooth = no_smooth

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
            no_smooth=self._no_smooth,
            progress_cb=self._progress,
            cancel_check=self._check_cancelled,
        )
