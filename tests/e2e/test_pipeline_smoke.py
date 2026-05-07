"""
End-to-end smoke test: synthetic stereo rig → full pipeline.

Uses no real camera footage — generates synthetic data programmatically.
Validates output shapes and file formats as specified in CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: synthetic data generators
# ──────────────────────────────────────────────────────────────────────────────


def _make_synthetic_calibration(img_size=(640, 480), baseline=0.12):
    """Return a synthetic calibration dict (no real cameras needed)."""
    w, h = img_size
    fx = fy = 600.0
    cx, cy = w / 2, h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    T = np.array([-baseline, 0.0, 0.0], dtype=np.float64)
    return {
        "K1": K,
        "D1": D,
        "K2": K.copy(),
        "D2": D.copy(),
        "R": R,
        "T": T,
        "E": np.zeros((3, 3)),
        "F": np.zeros((3, 3)),
        "rms": 0.1,
        "n_frames": 30,
        "image_size": list(img_size),
        "board_cfg": {},
        "quality": {"rms": 0.1, "n_frames_used": 30},
    }


def _write_calibration(calib: dict, path: str) -> None:
    from app.calib.stereo import save_calibration

    save_calibration(calib, calib.get("board_cfg", {}), path)


def _make_synthetic_pose2d(
    T: int = 30,
    P: int = 1,
    J: int = 17,
    img_size=(640, 480),
    fps=30.0,
    model_name="test_model",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return fake pose2d arrays in the correct format."""
    rng = np.random.default_rng(0)
    w, h = img_size
    kps = rng.uniform(0, 1, (T, P, J, 2)).astype(np.float32)
    kps[..., 0] *= w
    kps[..., 1] *= h
    conf = rng.uniform(0.5, 1.0, (T, P, J)).astype(np.float32)
    meta = {
        "fps": float(fps),
        "timestamps": np.linspace(0, T / fps, T, dtype=np.float32),
        "model_name": model_name,
        "skeleton": "coco17",
        "image_size": list(img_size),
    }
    return kps, conf, meta


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestSmokeCalibrationIO:
    def test_save_load_roundtrip(self, tmp_path):
        """calibration.yml round-trip preserves values."""
        from app.calib.stereo import load_calibration, save_calibration

        calib = _make_synthetic_calibration()
        out = str(tmp_path / "calibration.yml")
        save_calibration(calib, {}, out)

        loaded = load_calibration(out)
        np.testing.assert_allclose(loaded["K1"], calib["K1"], atol=1e-9)
        np.testing.assert_allclose(loaded["R"], calib["R"], atol=1e-9)
        np.testing.assert_allclose(loaded["T"], calib["T"], atol=1e-9)

    def test_yml_is_valid_yaml(self, tmp_path):
        from app.calib.stereo import save_calibration

        calib = _make_synthetic_calibration()
        out = str(tmp_path / "calib.yml")
        save_calibration(calib, {"type": "charuco"}, out)

        with open(out) as f:
            data = yaml.safe_load(f)
        assert "K1" in data
        assert "quality" in data


class TestSmokePose2DIO:
    def test_save_load_roundtrip(self, tmp_path):
        from app.pose2d.pipeline import load_pose2d, save_pose2d

        kps, conf, meta = _make_synthetic_pose2d(T=10, P=2)
        out = str(tmp_path / "pose2d_left.npz")
        save_pose2d(kps, conf, meta, out)

        kps2, conf2, meta2 = load_pose2d(out)
        assert kps2.shape == kps.shape
        np.testing.assert_array_equal(kps2, kps)
        assert meta2["skeleton"] == "coco17"
        assert meta2["fps"] == meta["fps"]

    def test_output_shapes(self, tmp_path):
        kps, conf, _ = _make_synthetic_pose2d(T=50, P=1, J=17)
        assert kps.shape == (50, 1, 17, 2)
        assert conf.shape == (50, 1, 17)
        assert kps.dtype == np.float32
        assert conf.dtype == np.float32


class TestLoadPose2DMissingKeys:
    """BUG-AA: load_pose2d must raise ValueError when NPZ is missing required keys."""

    def test_missing_keypoints_raises(self, tmp_path):
        """NPZ without 'keypoints' key must raise ValueError naming the missing key."""
        from app.pose2d.pipeline import load_pose2d

        path = str(tmp_path / "bad.npz")
        np.savez(path, conf=np.zeros((5, 1, 17), dtype=np.float32))
        with pytest.raises(ValueError, match="missing required keys"):
            load_pose2d(path)

    def test_missing_conf_raises(self, tmp_path):
        """NPZ without 'conf' key must raise ValueError naming the missing key."""
        from app.pose2d.pipeline import load_pose2d

        path = str(tmp_path / "bad.npz")
        np.savez(path, keypoints=np.zeros((5, 1, 17, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="missing required keys"):
            load_pose2d(path)


class TestSmokeRecon3D:
    def test_full_recon_pipeline(self, tmp_path):
        """Run reconstruct3d on synthetic data and validate output shapes."""
        from app.calib.stereo import save_calibration
        from app.pose2d.pipeline import save_pose2d
        from app.recon3d.pipeline import reconstruct3d

        T, P, J = 20, 1, 17
        img_size = (640, 480)

        # Write calibration
        calib = _make_synthetic_calibration(img_size)
        calib_path = str(tmp_path / "calibration.yml")
        save_calibration(calib, {}, calib_path)

        # Write pose2d
        kps, conf, meta = _make_synthetic_pose2d(T=T, P=P, J=J, img_size=img_size)
        left_path = str(tmp_path / "pose2d_left.npz")
        right_path = str(tmp_path / "pose2d_right.npz")
        save_pose2d(kps, conf, meta, left_path)
        save_pose2d(kps, conf, meta, right_path)

        # Run reconstruction
        out_path = str(tmp_path / "pose3d.npz")
        result = reconstruct3d(
            calib_path=calib_path,
            pose2d_left_path=left_path,
            pose2d_right_path=right_path,
            output_path=out_path,
        )

        assert result == out_path
        assert Path(out_path).exists()

        # Validate output
        d = np.load(out_path, allow_pickle=True)
        joints3d = d["joints3d"]
        conf3d = d["conf3d"]
        repro_err = d["repro_err"]
        meta_out = d["meta"].item()

        assert joints3d.shape == (T, P, J, 3), f"Expected ({T},{P},{J},3) got {joints3d.shape}"
        assert conf3d.shape == (T, P, J)
        assert repro_err.shape == (T, P, J)
        assert joints3d.dtype == np.float32
        assert conf3d.dtype == np.float32
        assert meta_out["skeleton"] == "coco17"
        assert meta_out["smoothing"] == "oneeuro"

    def test_recon_with_low_confidence_joints(self, tmp_path):
        """Joints with conf < min_conf should have conf3d = 0."""
        from app.calib.stereo import save_calibration
        from app.pose2d.pipeline import save_pose2d
        from app.recon3d.pipeline import reconstruct3d

        T, P, J = 5, 1, 17
        img_size = (640, 480)

        calib = _make_synthetic_calibration(img_size)
        calib_path = str(tmp_path / "calibration.yml")
        save_calibration(calib, {}, calib_path)

        kps, _, meta = _make_synthetic_pose2d(T=T, P=P, J=J, img_size=img_size)
        # All confidence = 0
        conf_zero = np.zeros((T, P, J), dtype=np.float32)

        left_path = str(tmp_path / "pose2d_left.npz")
        right_path = str(tmp_path / "pose2d_right.npz")
        save_pose2d(kps, conf_zero, meta, left_path)
        save_pose2d(kps, conf_zero, meta, right_path)

        out_path = str(tmp_path / "pose3d.npz")
        reconstruct3d(calib_path, left_path, right_path, out_path, min_conf=0.3)

        d = np.load(out_path, allow_pickle=True)
        conf3d = d["conf3d"]
        # All joints should be masked out
        assert np.all(conf3d == 0.0), "All low-confidence joints should be zeroed"

    def test_mismatched_frame_counts_truncated(self, tmp_path):
        """BUG-006: when L/R frame counts differ, pipeline warns and truncates to the shorter side."""
        from app.calib.stereo import save_calibration
        from app.pose2d.pipeline import save_pose2d
        from app.recon3d.pipeline import reconstruct3d

        T_long, T_short, P, J = 20, 8, 1, 17
        img_size = (640, 480)

        calib = _make_synthetic_calibration(img_size)
        calib_path = str(tmp_path / "calibration.yml")
        save_calibration(calib, {}, calib_path)

        kps_long, conf_long, meta_long = _make_synthetic_pose2d(
            T=T_long, P=P, J=J, img_size=img_size
        )
        kps_short, conf_short, meta_short = _make_synthetic_pose2d(
            T=T_short, P=P, J=J, img_size=img_size
        )

        left_path = str(tmp_path / "pose2d_left.npz")
        right_path = str(tmp_path / "pose2d_right.npz")
        save_pose2d(kps_long, conf_long, meta_long, left_path)
        save_pose2d(kps_short, conf_short, meta_short, right_path)

        out_path = str(tmp_path / "pose3d.npz")
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = reconstruct3d(calib_path, left_path, right_path, out_path)

        assert result == out_path, "Pipeline should return output path on success"
        assert any("Frame count mismatch" in str(w.message) for w in caught), (
            "Expected a UserWarning about frame count mismatch"
        )

        d = np.load(out_path, allow_pickle=True)
        joints3d = d["joints3d"]
        assert joints3d.shape[0] == T_short, (
            f"Expected {T_short} frames after truncation, got {joints3d.shape[0]}."
        )

    def test_reconstruct3d_cancel_immediately(self, tmp_path):
        """cancel_check returning True on first call must make reconstruct3d return None."""
        from app.calib.stereo import save_calibration
        from app.pose2d.pipeline import save_pose2d
        from app.recon3d.pipeline import reconstruct3d

        T, P, J = 10, 1, 17
        img_size = (640, 480)

        calib = _make_synthetic_calibration(img_size)
        calib_path = str(tmp_path / "calibration.yml")
        save_calibration(calib, {}, calib_path)

        kps, conf, meta = _make_synthetic_pose2d(T=T, P=P, J=J, img_size=img_size)
        left_path = str(tmp_path / "pose2d_left.npz")
        right_path = str(tmp_path / "pose2d_right.npz")
        save_pose2d(kps, conf, meta, left_path)
        save_pose2d(kps, conf, meta, right_path)

        out_path = str(tmp_path / "pose3d.npz")
        result = reconstruct3d(
            calib_path,
            left_path,
            right_path,
            out_path,
            cancel_check=lambda: True,
        )
        assert result is None, "Immediate cancellation must return None"

    def test_reconstruct3d_progress_cb_called(self, tmp_path):
        """progress_cb must be called with strictly increasing percentage values."""
        from app.calib.stereo import save_calibration
        from app.pose2d.pipeline import save_pose2d
        from app.recon3d.pipeline import reconstruct3d

        T, P, J = 10, 1, 17
        img_size = (640, 480)

        calib = _make_synthetic_calibration(img_size)
        calib_path = str(tmp_path / "calibration.yml")
        save_calibration(calib, {}, calib_path)

        kps, conf, meta = _make_synthetic_pose2d(T=T, P=P, J=J, img_size=img_size)
        left_path = str(tmp_path / "pose2d_left.npz")
        right_path = str(tmp_path / "pose2d_right.npz")
        save_pose2d(kps, conf, meta, left_path)
        save_pose2d(kps, conf, meta, right_path)

        out_path = str(tmp_path / "pose3d.npz")

        progress_calls: list[tuple[int, str]] = []

        def progress_cb(pct: int, msg: str) -> None:
            progress_calls.append((pct, msg))

        reconstruct3d(calib_path, left_path, right_path, out_path, progress_cb=progress_cb)

        assert len(progress_calls) > 0, "progress_cb was never called"
        pcts = [p for p, _ in progress_calls]
        assert pcts[0] < pcts[-1], "Progress percentage must increase over the run"
        assert pcts[-1] == 100, "Final progress callback must report 100%"

    def test_mismatched_person_counts_truncated(self, tmp_path):
        """BUG-O: when L/R person counts differ, pipeline warns and truncates to the smaller side."""
        from app.calib.stereo import save_calibration
        from app.pose2d.pipeline import save_pose2d
        from app.recon3d.pipeline import reconstruct3d

        T, J = 10, 17
        img_size = (640, 480)

        calib = _make_synthetic_calibration(img_size)
        calib_path = str(tmp_path / "calibration.yml")
        save_calibration(calib, {}, calib_path)

        # Left has 2 persons, right has 1 person
        kps_left, conf_left, meta_left = _make_synthetic_pose2d(T=T, P=2, J=J, img_size=img_size)
        kps_right, conf_right, meta_right = _make_synthetic_pose2d(T=T, P=1, J=J, img_size=img_size)

        left_path = str(tmp_path / "pose2d_left.npz")
        right_path = str(tmp_path / "pose2d_right.npz")
        save_pose2d(kps_left, conf_left, meta_left, left_path)
        save_pose2d(kps_right, conf_right, meta_right, right_path)

        out_path = str(tmp_path / "pose3d.npz")
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = reconstruct3d(calib_path, left_path, right_path, out_path)

        assert result == out_path, "Pipeline should succeed despite person count mismatch"
        assert any("Person count mismatch" in str(w.message) for w in caught), (
            "Expected a UserWarning about person count mismatch"
        )

        d = np.load(out_path, allow_pickle=True)
        joints3d = d["joints3d"]
        assert joints3d.shape == (T, 1, J, 3), (
            f"Expected ({T}, 1, {J}, 3) after truncation, got {joints3d.shape}"
        )


class TestRecon3DParamValidation:
    """BUG-R: reconstruct3d must validate min_conf and max_reproj_err before file I/O."""

    def test_negative_max_reproj_err_raises(self):
        """max_reproj_err < 0 silently rejects all joints; must raise ValueError early."""
        from app.recon3d.pipeline import reconstruct3d

        with pytest.raises(ValueError, match="max_reproj_err"):
            reconstruct3d(
                "dummy_calib.yml",
                "dummy_left.npz",
                "dummy_right.npz",
                "dummy_out.npz",
                max_reproj_err=-1.0,
            )

    def test_min_conf_above_one_raises(self):
        """min_conf > 1.0 is out-of-range (confidence is in [0,1]); must raise ValueError."""
        from app.recon3d.pipeline import reconstruct3d

        with pytest.raises(ValueError, match="min_conf"):
            reconstruct3d(
                "dummy_calib.yml",
                "dummy_left.npz",
                "dummy_right.npz",
                "dummy_out.npz",
                min_conf=1.5,
            )

    def test_min_conf_negative_raises(self):
        """min_conf < 0 disables confidence filtering silently; must raise ValueError."""
        from app.recon3d.pipeline import reconstruct3d

        with pytest.raises(ValueError, match="min_conf"):
            reconstruct3d(
                "dummy_calib.yml",
                "dummy_left.npz",
                "dummy_right.npz",
                "dummy_out.npz",
                min_conf=-0.1,
            )


class TestSmokeOneEuro:
    def test_cli_imports(self):
        """Ensure all CLI __main__ modules import without error."""
        import importlib

        for mod in ["app.calib.__main__", "app.pose2d.__main__", "app.recon3d.__main__"]:
            m = importlib.import_module(mod)
            assert hasattr(m, "main")
