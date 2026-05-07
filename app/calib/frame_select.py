"""Intelligent frame selection for stereo calibration."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .board import BoardDetector, DetectionResult


def _annotate_frame(frame: np.ndarray, img_pts: np.ndarray) -> np.ndarray:
    """Draw detected corner points on a copy of *frame* (green circles)."""
    out = frame.copy()
    for pt in img_pts.reshape(-1, 2):
        cv2.circle(out, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)
    return out


def _coverage_score(img_pts: np.ndarray, img_size: tuple[int, int], grid: int = 4) -> float:
    """
    Score a detected frame by how well its corners cover the image.

    Divides the image into grid×grid cells and counts how many cells
    contain at least one corner.  Returns fraction of cells covered [0, 1].
    """
    w, h = img_size
    cell_w = w / grid
    cell_h = h / grid
    cells: set[tuple[int, int]] = set()
    for x, y in img_pts:
        ci = min(max(int(x / cell_w), 0), grid - 1)
        cj = min(max(int(y / cell_h), 0), grid - 1)
        cells.add((ci, cj))
    return len(cells) / (grid * grid)


class FrameSelection:
    """Selected frame pair (BGR) with associated detection results."""

    __slots__ = ("frame_left", "frame_right", "det_left", "det_right", "score")

    def __init__(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        det_left: DetectionResult,
        det_right: DetectionResult,
        score: float,
    ):
        self.frame_left = frame_left
        self.frame_right = frame_right
        self.det_left = det_left
        self.det_right = det_right
        self.score = score


def _board_centroid(img_pts: np.ndarray) -> np.ndarray:
    """Return mean (x, y) of detected corner points."""
    return img_pts.reshape(-1, 2).mean(axis=0)


def _match_temporal_pairs(
    solo_l: dict[int, DetectionResult],
    solo_r: dict[int, DetectionResult],
    actual_window: int,
) -> list[tuple[int, int, DetectionResult, DetectionResult]]:
    """
    Match unmatched left detections to unmatched right detections within ±actual_window
    actual video frames.  Each detection is used at most once.

    Returns list of (frame_idx_l, frame_idx_r, det_l, det_r) sorted by temporal distance.
    """
    if not solo_l or not solo_r:
        return []

    r_frames = sorted(solo_r.keys())
    used_l: set[int] = set()
    used_r: set[int] = set()

    # Build candidate matches sorted by temporal distance (closest first)
    candidates: list[tuple[int, int, int]] = []  # (dist, fi_l, fi_r)
    for fi_l in sorted(solo_l.keys()):
        for fi_r in r_frames:
            dist = abs(fi_r - fi_l)
            if dist <= actual_window:
                candidates.append((dist, fi_l, fi_r))

    candidates.sort()  # closest temporal distance first

    matched: list[tuple[int, int, DetectionResult, DetectionResult]] = []
    for dist, fi_l, fi_r in candidates:
        if fi_l in used_l or fi_r in used_r:
            continue
        matched.append((fi_l, fi_r, solo_l[fi_l], solo_r[fi_r]))
        used_l.add(fi_l)
        used_r.add(fi_r)

    return matched


def extract_calibration_frames(
    video_left: str,
    video_right: str,
    detector: BoardDetector,
    max_frames: int = 60,
    sample_every: int = 5,
    min_coverage: float = 0.05,
    progress_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    frame_cb: Callable[[np.ndarray, np.ndarray], None] | None = None,
    scan_cb: Callable[[np.ndarray, np.ndarray, bool, bool], None] | None = None,
    all_det_l_out: list | None = None,
    all_det_r_out: list | None = None,
    temporal_window: int = 4,
) -> list[FrameSelection]:
    """
    Extract up to *max_frames* paired calibration frames from two synchronized videos.

    Strategy:
    1. Sample every *sample_every* frames.
    2. Detect board in both left and right in parallel (OpenCV releases GIL).
    3. Accept simultaneous pairs when both cameras detect AND coverage >= min_coverage.
    4. Temporal relaxation: for frames where only one camera detects the board, search
       ±temporal_window sampled frames for a matching detection in the other camera.
       Valid when the board is held still (common in research recordings where the board
       is shown to each camera separately).  Creates additional candidate pairs.
    5. Greedily keep frames with the highest coverage score up to max_frames.

    Args:
        video_left: path to left camera video
        video_right: path to right camera video
        detector: BoardDetector instance
        max_frames: maximum number of frames to select
        sample_every: skip frames (1 = process every frame, 5 = every 5th)
        min_coverage: minimum corner coverage fraction to accept frame
        progress_cb: called with (percent, message)
        cancel_check: called each iteration; return True to abort
        frame_cb: called with (annotated_left_bgr, annotated_right_bgr) for each accepted
                  frame pair so callers can display live previews
        scan_cb: called on every sampled frame (throttled to every 3rd) with
                 (frame_l, frame_r, det_l_ok, det_r_ok); use for live preview of all frames
        all_det_l_out: if provided, DetectionResult objects for every frame where the left
                       camera detected the board are appended (used for two-phase calibration)
        all_det_r_out: same for the right camera
        temporal_window: search ±this many sampled-frame intervals for temporal relaxation;
                         set to 0 to disable.  Default 4 ≈ ±(4 × sample_every) actual frames.

    Returns:
        List of FrameSelection objects (sorted by score descending, capped at max_frames)
    """
    if max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got {max_frames}")
    if sample_every < 1:
        raise ValueError(f"sample_every must be >= 1, got {sample_every}")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(
            f"min_coverage must be in [0.0, 1.0], got {min_coverage}. "
            "Values > 1.0 are impossible to satisfy (score is at most 1.0) and "
            "values < 0.0 accept every detected frame regardless of coverage quality."
        )

    cap_l = cv2.VideoCapture(video_left, cv2.CAP_FFMPEG)
    cap_r = cv2.VideoCapture(video_right, cv2.CAP_FFMPEG)

    try:
        if not cap_l.isOpened():
            raise FileNotFoundError(f"Cannot open left video: {video_left!r}")
        if not cap_r.isOpened():
            raise FileNotFoundError(f"Cannot open right video: {video_right!r}")

        n_frames_l = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(n_frames_l, n_frames_r) if n_frames_l > 0 and n_frames_r > 0 else 0

        candidates: list[FrameSelection] = []
        frame_idx = 0
        n_det_l = 0
        n_det_r = 0
        scan_cb_tick = 0
        t_start = time.monotonic()

        # Buffers for temporal relaxation: frame_idx → DetectionResult for solo detections
        _solo_l: dict[int, DetectionResult] = {}  # left detected, right did not
        _solo_r: dict[int, DetectionResult] = {}  # right detected, left did not

        while True:
            if cancel_check and cancel_check():
                break

            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r:
                break

            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue

            # Build progress message with per-camera counts and ETA
            if total > 0 and progress_cb:
                pct = int(frame_idx * 100 / total)
                elapsed = time.monotonic() - t_start
                eta_str = ""
                if elapsed > 10 and frame_idx > 0:
                    frames_per_sec = frame_idx / elapsed
                    remaining_frames = total - frame_idx
                    eta_secs = remaining_frames / frames_per_sec if frames_per_sec > 0 else 0
                    if eta_secs >= 60:
                        eta_str = f" — ~{int(eta_secs / 60)} min remaining"
                    else:
                        eta_str = f" — ~{int(eta_secs)} s remaining"
                msg = (
                    f"Frame {frame_idx}/{total} — L:{n_det_l} R:{n_det_r} "
                    f"pairs:{len(candidates)}{eta_str}"
                )
                progress_cb(pct, msg)

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            # Run both detections in parallel — OpenCV releases the GIL during
            # findChessboardCorners so two threads run on separate CPU cores.
            with ThreadPoolExecutor(max_workers=2) as _ex:
                _fut_l = _ex.submit(detector.detect, gray_l)
                _fut_r = _ex.submit(detector.detect, gray_r)
                det_l = _fut_l.result()
                det_r = _fut_r.result()

            det_l_ok = det_l is not None
            det_r_ok = det_r is not None

            if det_l_ok:
                n_det_l += 1
                if all_det_l_out is not None:
                    all_det_l_out.append(det_l)
            if det_r_ok:
                n_det_r += 1
                if all_det_r_out is not None:
                    all_det_r_out.append(det_r)

            # scan_cb: called every 3 sampled frames with raw frames (caller annotates)
            if scan_cb is not None:
                scan_cb_tick += 1
                if scan_cb_tick >= 3:
                    scan_cb_tick = 0
                    scan_cb(frame_l.copy(), frame_r.copy(), det_l_ok, det_r_ok)

            if det_l_ok and det_r_ok:
                h, w = frame_l.shape[:2]
                img_size = (w, h)
                cov_l = _coverage_score(det_l.img_pts, img_size)
                cov_r = _coverage_score(det_r.img_pts, img_size)
                score = min(cov_l, cov_r)

                if score >= min_coverage:
                    fl_copy = frame_l.copy()
                    fr_copy = frame_r.copy()
                    candidates.append(
                        FrameSelection(
                            frame_left=fl_copy,
                            frame_right=fr_copy,
                            det_left=det_l,
                            det_right=det_r,
                            score=score,
                        )
                    )
                    if frame_cb is not None:
                        ann_l = _annotate_frame(fl_copy, det_l.img_pts)
                        ann_r = _annotate_frame(fr_copy, det_r.img_pts)
                        frame_cb(ann_l, ann_r)
            elif det_l_ok and temporal_window > 0:
                _solo_l[frame_idx] = det_l
            elif det_r_ok and temporal_window > 0:
                _solo_r[frame_idx] = det_r

            frame_idx += 1

        # ── Temporal relaxation pass ──────────────────────────────────────────
        # Match unmatched single-camera detections across cameras within ±window.
        if temporal_window > 0 and _solo_l and _solo_r:
            actual_window = temporal_window * sample_every
            temporal_matches = _match_temporal_pairs(_solo_l, _solo_r, actual_window)

            if temporal_matches and progress_cb:
                progress_cb(
                    98,
                    f"Temporal relaxation: {len(temporal_matches)} candidate pairs found, "
                    "reading frames…",
                )

            for fi_l, fi_r, t_det_l, t_det_r in temporal_matches:
                if cancel_check and cancel_check():
                    break

                # Seek and read the frames at the matched indices
                cap_l.set(cv2.CAP_PROP_POS_FRAMES, fi_l)
                cap_r.set(cv2.CAP_PROP_POS_FRAMES, fi_r)
                ret_l, t_frame_l = cap_l.read()
                ret_r, t_frame_r = cap_r.read()
                if not ret_l or not ret_r:
                    continue

                h, w = t_frame_l.shape[:2]
                img_size_t = (w, h)
                cov_l = _coverage_score(t_det_l.img_pts, img_size_t)
                cov_r = _coverage_score(t_det_r.img_pts, img_size_t)
                score = min(cov_l, cov_r)

                if score >= min_coverage:
                    candidates.append(
                        FrameSelection(
                            frame_left=t_frame_l,
                            frame_right=t_frame_r,
                            det_left=t_det_l,
                            det_right=t_det_r,
                            score=score,
                        )
                    )

    finally:
        cap_l.release()
        cap_r.release()

    # Sort by score descending, take up to max_frames
    candidates.sort(key=lambda x: x.score, reverse=True)
    selected = candidates[:max_frames]

    if progress_cb:
        progress_cb(100, f"Selected {len(selected)} calibration frames")

    return selected
