"""Unit tests for app.calib.__main__ CLI parser (BUG-E regression)."""

from __future__ import annotations

import pytest

from app.calib.__main__ import build_parser
from app.calib.board import ARUCO_DICTS


class TestCalibBuildParser:
    def test_default_aruco_dict(self):
        """Default --aruco-dict should be DICT_4X4_50."""
        p = build_parser()
        args = p.parse_args(["--left", "l.mp4", "--right", "r.mp4"])
        assert args.aruco_dict == "DICT_4X4_50"

    def test_valid_aruco_dict_accepted(self):
        """Every key in ARUCO_DICTS must be accepted by the parser."""
        p = build_parser()
        for name in ARUCO_DICTS:
            args = p.parse_args(["--left", "l.mp4", "--right", "r.mp4", "--aruco-dict", name])
            assert args.aruco_dict == name

    def test_invalid_aruco_dict_rejected(self):
        """An unknown dictionary name must cause a SystemExit (argparse error)."""
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--left", "l.mp4", "--right", "r.mp4", "--aruco-dict", "INVALID_DICT"])

    def test_board_type_choices(self):
        """--board must only accept charuco or checkerboard."""
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--left", "l.mp4", "--right", "r.mp4", "--board", "invalid"])
