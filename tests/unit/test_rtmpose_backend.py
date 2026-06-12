"""
Unit tests for RTMPoseBackend.

All tests skip automatically when rtmlib / onnxruntime is not installed.
They verify the PoseBackend contract (shapes, dtypes, value ranges) against
synthetic frames — no real video is required.

Models are downloaded once into ~/.cache/rtmlib/ (module-scoped fixture).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rtmlib", reason="rtmlib not installed (pip install rtmlib onnxruntime)")

from app.pose2d.rtmpose_backend import RTMPoseBackend  # noqa: E402

_BLANK = np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def backend():
    """One RTMPoseBackend instance shared across all tests (models load once)."""
    b = RTMPoseBackend(mode="balanced")
    yield b
    b.close()


# ── Attribute contract ──────────────────────────────────────────────────────


class TestAttributes:
    def test_name_is_rtmpose_m(self, backend):
        assert backend.name == "rtmpose_m"

    def test_implements_pose_backend_interface(self, backend):
        from app.pose2d.base import PoseBackend

        assert isinstance(backend, PoseBackend)

    def test_close_is_idempotent(self, backend):
        """close() must not raise even if called multiple times."""
        backend.close()
        backend.close()


# ── Output contract: blank frame ────────────────────────────────────────────


class TestContractBlankFrame:
    """Blank (all-zero) frames — no person present."""

    @pytest.fixture(scope="class")
    def result(self, backend):
        return backend.detect(_BLANK)

    def test_returns_two_arrays(self, result):
        assert len(result) == 2

    def test_keypoints_ndim_3(self, result):
        kps, _ = result
        assert kps.ndim == 3, f"Expected 3-D keypoints, got {kps.ndim}"

    def test_keypoints_joints_axis_is_17(self, result):
        kps, _ = result
        assert kps.shape[1] == 17, f"Expected 17 joints, got {kps.shape[1]}"

    def test_keypoints_coords_axis_is_2(self, result):
        kps, _ = result
        assert kps.shape[2] == 2, f"Expected 2 coords (x,y), got {kps.shape[2]}"

    def test_conf_ndim_2(self, result):
        _, conf = result
        assert conf.ndim == 2, f"Expected 2-D confidence, got {conf.ndim}"

    def test_conf_joints_axis_is_17(self, result):
        _, conf = result
        assert conf.shape[1] == 17

    def test_person_dims_match(self, result):
        kps, conf = result
        assert kps.shape[0] == conf.shape[0], "Person dim mismatch between kps and conf"

    def test_at_least_one_person_returned(self, result):
        kps, _ = result
        assert kps.shape[0] >= 1

    def test_keypoints_dtype_float32(self, result):
        kps, _ = result
        assert kps.dtype == np.float32, f"Expected float32, got {kps.dtype}"

    def test_conf_dtype_float32(self, result):
        _, conf = result
        assert conf.dtype == np.float32, f"Expected float32, got {conf.dtype}"

    def test_conf_non_negative(self, result):
        _, conf = result
        assert np.all(conf >= 0.0), "Confidence contains negative values"


# ── Output contract: non-trivial frames ─────────────────────────────────────


class TestContractVariousFrames:
    @pytest.mark.parametrize("h,w", [(240, 320), (480, 640), (720, 1280)])
    def test_handles_various_resolutions(self, backend, h, w):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        kps, conf = backend.detect(frame)
        assert kps.shape[1:] == (17, 2), f"Bad shape for {h}x{w}: {kps.shape}"
        assert conf.shape[1] == 17

    def test_handles_random_noise_frame(self, backend):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        kps, conf = backend.detect(frame)
        assert kps.shape[1:] == (17, 2)
        assert conf.shape[1] == 17
        assert np.all(conf >= 0.0)

    def test_keypoints_within_image_bounds_when_person_detected(self, backend):
        """Detected joints must lie near the image (RTMPose SimCC may go a few px outside)."""
        h, w = 480, 640
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        kps, conf = backend.detect(frame)
        detected = kps[conf > 0.3]
        if len(detected):
            margin_x, margin_y = w * 0.05, h * 0.05
            assert np.all(detected[:, 0] >= -margin_x) and np.all(detected[:, 0] <= w + margin_x)
            assert np.all(detected[:, 1] >= -margin_y) and np.all(detected[:, 1] <= h + margin_y)

    def test_multiple_calls_consistent_shapes(self, backend):
        """Repeated detect() calls on the same frame must return the same shape."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r1 = backend.detect(frame)
        r2 = backend.detect(frame)
        assert r1[0].shape == r2[0].shape
        assert r1[1].shape == r2[1].shape


# ── Import / availability guard ─────────────────────────────────────────────


class TestImportGuard:
    def test_import_error_without_rtmlib(self, monkeypatch):
        """RTMPoseBackend must raise ImportError if rtmlib is unavailable."""
        import importlib
        import sys

        # Remove rtmlib from sys.modules so the import inside __init__ fails
        saved = {k: v for k, v in sys.modules.items() if "rtmlib" in k}
        for k in saved:
            monkeypatch.delitem(sys.modules, k)
        monkeypatch.setitem(sys.modules, "rtmlib", None)

        import app.pose2d.rtmpose_backend as mod

        importlib.reload(mod)

        with pytest.raises(ImportError, match="rtmlib"):
            mod.RTMPoseBackend()

        # Restore so subsequent tests still have rtmlib
        for k, v in saved.items():
            sys.modules[k] = v
        importlib.reload(mod)
