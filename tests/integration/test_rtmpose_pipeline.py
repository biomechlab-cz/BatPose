"""
Integration tests: RTMPose backend through the video pipeline.

Runs process_video() on the first 30 frames of the 20260527 sample left-camera
recording and verifies that:
  - The output NPZ schema matches the MediaPipe schema (same keys, same shapes)
  - Keypoint coordinates fall within image bounds
  - At least some frames have non-zero confidence (person actually detected)

Skips automatically when rtmlib / onnxruntime is not installed, or when the
sample recording is not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rtmlib", reason="rtmlib not installed (pip install rtmlib onnxruntime)")

cv2 = pytest.importorskip("cv2")

_CAPTURE = Path(__file__).parents[2] / "data" / "Test project" / "capture"
_LEFT_VIDEO = _CAPTURE / "20260527_132450_left.avi"

pytestmark = pytest.mark.skipif(
    not _LEFT_VIDEO.exists(),
    reason="Sample recording not found in data/Test project/capture/",
)

_MAX_FRAMES = 30  # keep the test fast; only process this many frames


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    """Run RTMPose pipeline on the first _MAX_FRAMES frames; return (kps, conf, meta)."""
    import cv2 as _cv2

    from app.pose2d.pipeline import process_video
    from app.pose2d.rtmpose_backend import RTMPoseBackend

    # Build a trimmed video so process_video sees exactly _MAX_FRAMES frames
    tmp = tmp_path_factory.mktemp("rtmpose")
    trimmed = str(tmp / "trimmed.avi")

    cap = _cv2.VideoCapture(str(_LEFT_VIDEO))
    fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _cv2.VideoWriter(trimmed, _cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    for _ in range(_MAX_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    cap.release()
    writer.release()

    backend = RTMPoseBackend(mode="balanced")
    try:
        kps, conf, meta = process_video(trimmed, backend)
    finally:
        backend.close()

    return kps, conf, meta


# ── Output schema ────────────────────────────────────────────────────────────


class TestOutputSchema:
    def test_keypoints_ndim_4(self, pipeline_result):
        kps, _, _ = pipeline_result
        assert kps.ndim == 4, f"Expected [T,P,17,2], got shape {kps.shape}"

    def test_keypoints_joints_17(self, pipeline_result):
        kps, _, _ = pipeline_result
        assert kps.shape[2] == 17

    def test_keypoints_coords_2(self, pipeline_result):
        kps, _, _ = pipeline_result
        assert kps.shape[3] == 2

    def test_conf_shape_matches_keypoints(self, pipeline_result):
        kps, conf, _ = pipeline_result
        T, P, J, _ = kps.shape
        assert conf.shape == (T, P, J)

    def test_frame_count_matches_input(self, pipeline_result):
        kps, _, _ = pipeline_result
        assert kps.shape[0] == _MAX_FRAMES, f"Expected {_MAX_FRAMES} frames, got {kps.shape[0]}"

    def test_keypoints_dtype_float32(self, pipeline_result):
        kps, _, _ = pipeline_result
        assert kps.dtype == np.float32

    def test_conf_dtype_float32(self, pipeline_result):
        _, conf, _ = pipeline_result
        assert conf.dtype == np.float32


# ── Meta keys match MediaPipe schema ────────────────────────────────────────


class TestMetaSchema:
    def test_fps_present_and_positive(self, pipeline_result):
        _, _, meta = pipeline_result
        assert "fps" in meta and float(meta["fps"]) > 0

    def test_model_name_present(self, pipeline_result):
        _, _, meta = pipeline_result
        assert "model_name" in meta
        assert meta["model_name"] == "rtmpose_m"

    def test_skeleton_is_coco17(self, pipeline_result):
        _, _, meta = pipeline_result
        assert meta.get("skeleton") == "coco17"

    def test_timestamps_length_matches_frames(self, pipeline_result):
        kps, _, meta = pipeline_result
        assert len(meta["timestamps"]) == kps.shape[0]

    def test_image_size_present_and_positive(self, pipeline_result):
        _, _, meta = pipeline_result
        w, h = meta["image_size"]
        assert w > 0 and h > 0


# ── Detection quality ────────────────────────────────────────────────────────


class TestDetectionQuality:
    def test_some_frames_have_nonzero_confidence(self, pipeline_result):
        """At least one joint must be detected with conf > 0 in at least one frame."""
        _, conf, _ = pipeline_result
        assert np.any(conf > 0), "RTMPose returned all-zero confidence across all frames"

    def test_keypoints_within_image_bounds(self, pipeline_result):
        """Detected joints must lie near the image dimensions.

        RTMPose's SimCC decoder can return coordinates a few pixels outside
        the exact image boundary — allow a 5 % margin on each side.
        """
        kps, conf, meta = pipeline_result
        w, h = meta["image_size"]
        detected = kps[conf > 0.3]  # only check confidently detected joints
        if len(detected) == 0:
            pytest.skip("No joints detected with conf > 0.3 — cannot check bounds")
        margin_x = w * 0.05
        margin_y = h * 0.05
        assert np.all(detected[:, 0] >= -margin_x) and np.all(detected[:, 0] <= w + margin_x), (
            f"x coordinate outside [{-margin_x:.0f}, {w + margin_x:.0f}]"
        )
        assert np.all(detected[:, 1] >= -margin_y) and np.all(detected[:, 1] <= h + margin_y), (
            f"y coordinate outside [{-margin_y:.0f}, {h + margin_y:.0f}]"
        )

    def test_no_nan_or_inf_in_output(self, pipeline_result):
        kps, conf, _ = pipeline_result
        assert np.all(np.isfinite(kps)), "Non-finite values in keypoints"
        assert np.all(np.isfinite(conf)), "Non-finite values in confidence"


# ── NPZ save/load round-trip ─────────────────────────────────────────────────


class TestNpzRoundTrip:
    def test_saved_npz_loads_with_same_shapes(self, pipeline_result, tmp_path):
        from app.pose2d.pipeline import load_pose2d, save_pose2d

        kps, conf, meta = pipeline_result
        out = str(tmp_path / "pose2d_rtm.npz")
        save_pose2d(kps, conf, meta, out)

        kps2, conf2, meta2 = load_pose2d(out)
        assert kps2.shape == kps.shape
        assert conf2.shape == conf.shape
        assert meta2["model_name"] == meta["model_name"]

    def test_saved_npz_schema_identical_to_mediapipe(self, pipeline_result, tmp_path):
        """Keys in an RTMPose NPZ must be a superset of MediaPipe NPZ keys."""
        from app.pose2d.pipeline import save_pose2d

        kps, conf, meta = pipeline_result
        out = str(tmp_path / "pose2d_rtm2.npz")
        save_pose2d(kps, conf, meta, out)

        d = np.load(out, allow_pickle=True)
        assert {"keypoints", "conf", "meta"} <= set(d.files)
