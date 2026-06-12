"""Multi-person reconstruction warns about slot-order pairing (no ID matching)."""

from __future__ import annotations

import numpy as np
import pytest

from app.calib.stereo import save_calibration
from app.recon3d.pipeline import reconstruct3d

K = np.array([[800.0, 0, 512.0], [0, 800.0, 512.0], [0, 0, 1.0]])


def _project(p: np.ndarray) -> np.ndarray:
    return np.array([p[0] / p[2] * 800 + 512, p[1] / p[2] * 800 + 512], dtype=np.float32)


def _write_inputs(tmp_path, persons: int):
    calib = str(tmp_path / "calibration.yml")
    save_calibration(
        {
            "image_size": [1024, 1024],
            "K1": K,
            "D1": np.zeros(5),
            "K2": K,
            "D2": np.zeros(5),
            "R": np.eye(3),
            "T": np.array([-0.2, 0.0, 0.0]),
            "E": np.zeros((3, 3)),
            "F": np.zeros((3, 3)),
            "rms": 0.4,
            "n_frames": 10,
        },
        {"type": "charuco"},
        calib,
    )
    T_frames, J = 3, 17
    point = np.array([0.1, 0.2, 2.0])
    kl = np.tile(_project(point), (T_frames, persons, J, 1)).astype(np.float32)
    kr = np.tile(_project(point + np.array([-0.2, 0, 0])), (T_frames, persons, J, 1)).astype(
        np.float32
    )
    conf = np.ones((T_frames, persons, J), np.float32)
    left = str(tmp_path / "pose2d_left.npz")
    right = str(tmp_path / "pose2d_right.npz")
    np.savez(left, keypoints=kl, conf=conf, meta=np.array({"fps": 30.0}))
    np.savez(right, keypoints=kr, conf=conf, meta=np.array({"fps": 30.0}))
    return calib, left, right


def test_multi_person_emits_pairing_warning(tmp_path):
    calib, left, right = _write_inputs(tmp_path, persons=2)
    with pytest.warns(UserWarning, match="paired by detection ORDER"):
        reconstruct3d(calib, left, right, str(tmp_path / "pose3d.npz"))


def test_single_person_stays_silent(tmp_path):
    import warnings

    calib, left, right = _write_inputs(tmp_path, persons=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning would fail the test
        reconstruct3d(calib, left, right, str(tmp_path / "pose3d.npz"))
