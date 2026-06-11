"""The pose3d CSV sidecar metadata must record the coordinate frame (ADR-010).

The NPZ stores meta["coordinate_frame"] ("opencv" camera frame vs "world"
floor-board frame); without it in the CSV export, downstream consumers cannot
tell which frame the x/y/z columns are in.
"""

from __future__ import annotations

import json

import numpy as np

from app.recon3d.__main__ import _write_csv


def _arrays(T: int = 2, P: int = 1, J: int = 17):
    return np.zeros((T, P, J, 3), np.float32), np.ones((T, P, J), np.float32)


def test_world_frame_tag_written(tmp_path):
    joints3d, conf3d = _arrays()
    _write_csv(str(tmp_path / "out.csv"), joints3d, conf3d, 30.0, coordinate_frame="world")
    meta = json.loads((tmp_path / "out_metadata.json").read_text())
    assert meta["coordinate_frame"] == "world"
    assert meta["coordinate_units"] == "meters"


def test_legacy_default_is_opencv(tmp_path):
    joints3d, conf3d = _arrays()
    _write_csv(str(tmp_path / "out.csv"), joints3d, conf3d, 30.0)
    meta = json.loads((tmp_path / "out_metadata.json").read_text())
    assert meta["coordinate_frame"] == "opencv"


def test_csv_columns_unchanged(tmp_path):
    """The frame tag lives in the sidecar JSON — the CSV layout must not change."""
    joints3d, conf3d = _arrays()
    _write_csv(str(tmp_path / "out.csv"), joints3d, conf3d, 30.0, coordinate_frame="world")
    header = (tmp_path / "out.csv").read_text().splitlines()[0]
    assert header.startswith("frame,time_s,person,j0_x,j0_y,j0_z,j0_conf")
    assert "coordinate" not in header
