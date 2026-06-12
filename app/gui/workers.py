"""
QThread workers for long-running pipeline stages.

Each worker:
- Accepts task parameters in __init__
- Runs the pipeline in a background thread (non-blocking UI)
- Emits: progress(int, str), finished(object), error(str)
- Is cancellable via cancel()
"""

from __future__ import annotations

import queue
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


#: Lens model index → (lens_model string, intrinsics_flags, use_fisheye).
#: Shared by the offline (CalibWorker) and live (LiveCalibWorker) calibration
#: paths so both produce identically-tagged calibration.yml files.
LENS_MODELS = [
    ("standard", 0, False),
    ("wide-angle", _cv2.CALIB_RATIONAL_MODEL, False),
    ("fisheye", 0, True),
]


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
        lens_model: int = 0,  # index into LENS_MODELS (0=standard, 1=wide, 2=fisheye)
        parent=None,
    ):
        super().__init__(parent)
        self._video_left = video_left
        self._video_right = video_right
        self._board_cfg = board_cfg
        self._output_path = output_path
        self._max_frames = max_frames
        self._sample_every = sample_every
        self._lens_model = lens_model

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

        lens_name, intrinsics_flags, _use_fisheye = LENS_MODELS[self._lens_model]
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
            intrinsics_flags=intrinsics_flags,
            lens_model="fisheye" if _use_fisheye else "standard",
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

    Write architecture (one writer thread PER CAMERA)
    -------------------------------------------------
    Synchronous VideoWriter.write() blocks the camera-read thread long enough
    for the FLIR's NewestOnly buffer to drop frames during encoding / disk I/O.
    Disk writes are therefore delegated to background writer threads; the read
    thread only copies the pixel data into a queue and continues.

    Crucially there is **one writer thread per camera**, each with its own
    queue.  Encoding a single 1920×1200 MJPEG frame takes ~15 ms, so a single
    thread encoding *both* cameras sequentially tops out near ~32 fps — below a
    50 fps capture rate.  The queue then fills and frames are dropped, yet the
    file is still tagged at the nominal fps, which **compresses time and makes
    the clip play back too fast** (and corrupts downstream velocity/angle
    timing).  Two parallel writers roughly double throughput (~60+ fps), so the
    queues stay drained and no frames are dropped.

    The per-frame timestamp sidecar is written by the capture (producer) thread
    itself — it is tiny (one CSV line) and keeps the recorded timestamps exactly
    in step with the frames actually enqueued for writing.

    On stop, the achieved fps is computed from the hardware timestamps and a
    loud warning is emitted if it falls short of nominal or any frame was
    dropped, so a time-compressed clip can never pass silently.
    """

    frame_ready = Signal(object)  # CaptureFrame
    recording_finished = Signal(str, str)  # left_path, right_path

    #: Per-camera write-queue depth.  At 50 fps this is ~1.8 s of headroom for a
    #: writer to absorb transient I/O spikes without dropping frames.
    _WRITE_QUEUE_FRAMES = 90

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

    # ------------------------------------------------------------------
    # Per-camera writer thread
    # ------------------------------------------------------------------

    @staticmethod
    def _open_video_writer(out_path: str, fps: float, frame_w: int, frame_h: int):
        """Create an MJPG VideoWriter and FAIL LOUDLY if it could not open.

        cv2.VideoWriter never raises on a bad path / unavailable codec — it just
        returns a writer whose write() calls are silently discarded, so a broken
        recording would otherwise look like a saved one.
        """
        writer = _cv2.VideoWriter(
            out_path, _cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame_w, frame_h)
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"Could not open video writer for {out_path!r} "
                "(check the output folder exists and is writable, and that the "
                "MJPG codec is available)."
            )
        return writer

    @staticmethod
    def _camera_writer(
        q: queue.Queue[Any],
        writer,
        errors: list,
    ) -> None:
        """Drain *q* and write frames for ONE camera via an already-open writer.

        Each item is an independent numpy frame copy; a ``None`` sentinel
        signals the thread to flush and exit.  After creation the VideoWriter is
        used exclusively by this thread.  On a write error the queue keeps being
        drained (discarding frames) — a dead consumer would deadlock the
        producer's blocking ``q.put(None)`` sentinel at stop time.
        """
        try:
            while True:
                frame = q.get()
                if frame is None:  # sentinel — flush and exit
                    break
                writer.write(frame)
        except Exception as exc:
            errors.append(exc)
            while True:  # keep draining so stop can't deadlock
                if q.get() is None:
                    break
        finally:
            writer.release()

    # ------------------------------------------------------------------
    # Main capture loop
    # ------------------------------------------------------------------

    def _run(self) -> Any:
        import queue
        import threading

        self._source.start()

        was_recording = False
        frame_idx = 0
        rec_idx = 0
        dropped_frames = 0
        preview_stride = max(1, round(self._source.fps / 15))

        # Per-camera write state.
        q_l: queue.Queue[Any] | None = None
        q_r: queue.Queue[Any] | None = None
        thread_l: threading.Thread | None = None
        thread_r: threading.Thread | None = None
        errors_l: list = []
        errors_r: list = []
        ts_fh: Any = None  # timestamp sidecar, owned by THIS (producer) thread
        first_hw_ns: int | None = None
        last_hw_ns: int | None = None

        def _start_writers(h: int, w: int) -> None:
            nonlocal q_l, q_r, thread_l, thread_r, ts_fh
            fps = self._source.fps
            # Open both writers HERE (producer thread) so a bad path / missing
            # codec aborts recording with a visible error instead of silently
            # producing an unplayable file (cv2 write() on an unopened writer
            # is a no-op).  The opened writers are handed to the threads.
            writer_l = self._open_video_writer(self._out_left, fps, w, h)
            try:
                writer_r = self._open_video_writer(self._out_right, fps, w, h)
            except Exception:
                writer_l.release()
                raise
            q_l = queue.Queue(maxsize=self._WRITE_QUEUE_FRAMES)
            q_r = queue.Queue(maxsize=self._WRITE_QUEUE_FRAMES)
            errors_l.clear()
            errors_r.clear()
            thread_l = threading.Thread(
                target=self._camera_writer,
                args=(q_l, writer_l, errors_l),
                name="capture-writer-left",
                daemon=True,
            )
            thread_r = threading.Thread(
                target=self._camera_writer,
                args=(q_r, writer_r, errors_r),
                name="capture-writer-right",
                daemon=True,
            )
            thread_l.start()
            thread_r.start()
            ts_fh = open(
                self._timestamps_path(self._out_left, self._out_right),
                "w",
                encoding="utf-8",
                newline="",
            )
            ts_fh.write("frame,hw_left_ns,hw_right_ns,abs_delta_us\n")

        def _stop_writers() -> None:
            nonlocal q_l, q_r, thread_l, thread_r, ts_fh
            for q in (q_l, q_r):
                if q is not None:
                    q.put(None)  # sentinel
            for th in (thread_l, thread_r):
                if th is not None:
                    th.join(timeout=30.0)
                    if th.is_alive():
                        self._progress(0, "WARNING: writer thread did not finish in 30 s")
            if ts_fh is not None:
                ts_fh.close()
            q_l = q_r = thread_l = thread_r = ts_fh = None

        def _report_recording_quality() -> None:
            """Warn loudly if frames were dropped or the achieved fps fell short.

            A clip recorded below its nominal fps (because frames were dropped)
            is time-compressed: it plays too fast and every derived velocity /
            joint-angle rate is scaled wrong.  Surface that explicitly.
            """
            nominal = float(self._source.fps)
            achieved = 0.0
            if (
                first_hw_ns is not None
                and last_hw_ns is not None
                and last_hw_ns > first_hw_ns
                and rec_idx > 1
            ):
                achieved = (rec_idx - 1) / ((last_hw_ns - first_hw_ns) / 1e9)
            if dropped_frames or (achieved and nominal and achieved < 0.95 * nominal):
                msg = f"⚠ Recording quality: wrote {rec_idx} frames; {dropped_frames} dropped"
                if achieved:
                    msg += (
                        f"; achieved ~{achieved:.1f} fps vs {nominal:.0f} fps nominal. "
                        f"The clip is time-compressed — it will play too fast and "
                        f"its timing is unreliable for biomechanics."
                    )
                self._progress(0, msg)

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
                    dropped_frames = 0
                    rec_idx = 0
                    first_hw_ns = last_hw_ns = None
                    _start_writers(h, w)
                    was_recording = True

                # Transition: recording → idle
                if not self._do_record and was_recording:
                    _stop_writers()
                    was_recording = False
                    for errs in (errors_l, errors_r):
                        if errs:
                            self._progress(0, f"Write error: {errs[0]}")
                    _report_recording_quality()
                    self.recording_finished.emit(self._out_left, self._out_right)

                if was_recording and q_l is not None and q_r is not None:
                    # Encoding now runs on two parallel writer threads (one per
                    # camera).  The capture thread only copies pixel data (so it
                    # outlives the camera DMA buffer) and writes the tiny CSV row.
                    # Both queues are gated together so left/right frame indices
                    # stay aligned with the timestamp sidecar.
                    if (
                        q_l.qsize() < self._WRITE_QUEUE_FRAMES
                        and q_r.qsize() < self._WRITE_QUEUE_FRAMES
                    ):
                        q_l.put_nowait(frame.frame_left.copy())
                        q_r.put_nowait(frame.frame_right.copy())
                        tl = frame.hw_timestamp_left_ns
                        tr = frame.hw_timestamp_right_ns
                        delta = frame.hw_delta_us
                        ts_fh.write(
                            f"{rec_idx},"
                            f"{'' if tl is None else tl},"
                            f"{'' if tr is None else tr},"
                            f"{'' if delta is None else f'{delta:.3f}'}\n"
                        )
                        if tl is not None:
                            if first_hw_ns is None:
                                first_hw_ns = tl
                            last_hw_ns = tl
                        rec_idx += 1
                    else:
                        # Both writers fell behind — skip rather than stall the
                        # camera read (which would drop at the hardware buffer).
                        dropped_frames += 1
                        self._progress(0, f"Frame {frame.frame_index} skipped (write queue full)")

                frame_idx += 1
                self._progress(0, f"Frame {frame.frame_index}  {frame.timestamp:.1f} s")

        finally:
            already = was_recording
            _stop_writers()
            for errs in (errors_l, errors_r):
                if errs:
                    self._progress(0, f"Write error: {errs[0]}")
            if already:
                _report_recording_quality()
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

    #: Lens model index → (label, intrinsics_flags, use_fisheye) — shared with
    #: the offline CalibWorker so both paths stay in lockstep.
    _LENS_MODELS = LENS_MODELS

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
