"""Offline calibration must honour the lens model (fisheye vs pinhole).

run_calibration_pipeline() used to ALWAYS call calibrate_stereo(); fisheye
videos silently produced a pinhole calibration tagged lens_model="standard",
poisoning every downstream undistort/triangulation.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import app.calib.stereo as stereo


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Stub the heavy stages so only the dispatch logic runs."""
    calls: list = []

    monkeypatch.setattr(stereo, "make_detector", lambda cfg: object())
    monkeypatch.setattr(
        stereo,
        "extract_calibration_frames",
        lambda *a, **k: [SimpleNamespace(frame_left=np.zeros((8, 10, 3), np.uint8))],
    )
    monkeypatch.setattr(
        stereo,
        "calibrate_stereo",
        lambda *a, **k: calls.append("standard") or {"marker": "std"},
    )
    monkeypatch.setattr(
        stereo,
        "calibrate_stereo_fisheye",
        lambda *a, **k: calls.append("fisheye") or {"marker": "fish", "lens_model": "fisheye"},
    )
    monkeypatch.setattr(
        stereo,
        "save_calibration",
        lambda data, cfg, path: calls.append(("saved", data.get("lens_model", "standard"))),
    )
    return calls, str(tmp_path / "calibration.yml")


class TestLensDispatch:
    def test_fisheye_routes_to_fisheye_calibration(self, patched):
        calls, out = patched
        stereo.run_calibration_pipeline("l.mp4", "r.mp4", {}, out, lens_model="fisheye")
        assert "fisheye" in calls
        assert "standard" not in calls
        assert ("saved", "fisheye") in calls  # yml tagged correctly

    def test_default_routes_to_pinhole(self, patched):
        calls, out = patched
        stereo.run_calibration_pipeline("l.mp4", "r.mp4", {}, out)
        assert "standard" in calls
        assert "fisheye" not in calls
        assert ("saved", "standard") in calls


class TestCalibCliLensFlag:
    def test_lens_flag_accepted(self):
        from app.calib.__main__ import build_parser

        p = build_parser()
        args = p.parse_args(["--left", "l.mp4", "--right", "r.mp4", "--lens", "fisheye"])
        assert args.lens == "fisheye"

    def test_lens_default_standard(self):
        from app.calib.__main__ import build_parser

        args = build_parser().parse_args(["--left", "l.mp4", "--right", "r.mp4"])
        assert args.lens == "standard"

    def test_invalid_lens_rejected(self):
        from app.calib.__main__ import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["--left", "l.mp4", "--right", "r.mp4", "--lens", "pancake"])
