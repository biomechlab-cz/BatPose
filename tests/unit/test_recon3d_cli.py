"""Unit tests for recon3d CLI helpers (_write_csv, build_parser)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from app.recon3d.__main__ import _write_csv, build_parser


class TestWriteCsv:
    """Tests for the _write_csv helper (BUG-004 / CODE-BUG-003)."""

    def _make_data(self, T=3, P=1, J=4):
        joints3d = np.arange(T * P * J * 3, dtype=np.float32).reshape(T, P, J, 3)
        conf3d = np.linspace(0.0, 1.0, T * P * J, dtype=np.float32).reshape(T, P, J)
        return joints3d, conf3d

    def test_file_is_created(self, tmp_path):
        """CSV file should be written to disk."""
        out = str(tmp_path / "out.csv")
        j, c = self._make_data()
        _write_csv(out, j, c, fps=30.0)
        assert Path(out).exists()

    def test_header_columns_present(self, tmp_path):
        """Header must contain frame, time_s, person, j*_x/y/z/conf columns."""
        out = str(tmp_path / "out.csv")
        j, c = self._make_data(T=2, P=1, J=3)
        _write_csv(out, j, c, fps=30.0)
        with open(out) as f:
            header = next(csv.reader(f))
        assert header[:3] == ["frame", "time_s", "person"]
        # 3 joints → j0_x, j0_y, j0_z, j0_conf, j1_x, ...
        for joint in range(3):
            assert f"j{joint}_x" in header
            assert f"j{joint}_y" in header
            assert f"j{joint}_z" in header
            assert f"j{joint}_conf" in header

    def test_row_count(self, tmp_path):
        """Total rows = 1 header + T*P data rows."""
        T, P, J = 5, 2, 3
        out = str(tmp_path / "out.csv")
        j, c = self._make_data(T=T, P=P, J=J)
        _write_csv(out, j, c, fps=30.0)
        with open(out) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1 + T * P

    def test_time_s_values(self, tmp_path):
        """time_s column must equal frame_index / fps."""
        out = str(tmp_path / "out.csv")
        fps = 25.0
        j, c = self._make_data(T=4, P=1, J=2)
        _write_csv(out, j, c, fps=fps)
        with open(out) as f:
            rows = list(csv.reader(f))
        for data_row in rows[1:]:
            frame = int(data_row[0])
            expected_time = frame / fps
            actual_time = float(data_row[1])
            assert abs(actual_time - expected_time) < 1e-5, (
                f"frame {frame}: time_s={actual_time}, expected {expected_time}"
            )

    def test_zero_fps_produces_nan_time_s(self, tmp_path):
        """fps=0 must write 'nan' (not raise) for time_s."""
        out = str(tmp_path / "out.csv")
        j, c = self._make_data(T=2, P=1, J=2)
        _write_csv(out, j, c, fps=0.0)
        with open(out) as f:
            rows = list(csv.reader(f))
        # All data rows should have "nan" in the time_s column
        for row in rows[1:]:
            assert row[1] == "nan", f"Expected 'nan', got {row[1]!r}"

    def test_person_column_values(self, tmp_path):
        """person column must enumerate persons 0..P-1 within each frame."""
        T, P = 2, 3
        out = str(tmp_path / "out.csv")
        j, c = self._make_data(T=T, P=P, J=2)
        _write_csv(out, j, c, fps=30.0)
        with open(out) as f:
            rows = list(csv.reader(f))
        persons_in_frame_0 = [int(r[2]) for r in rows[1 : 1 + P]]
        assert persons_in_frame_0 == list(range(P))

    def test_joint_coordinate_values(self, tmp_path):
        """Joint x/y/z and conf values must round-trip through CSV accurately."""
        out = str(tmp_path / "out.csv")
        joints3d = np.array([[[[1.5, -2.3, 0.7]]]], dtype=np.float32)  # T=1, P=1, J=1
        conf3d = np.array([[[0.85]]], dtype=np.float32)
        _write_csv(out, joints3d, conf3d, fps=10.0)
        with open(out) as f:
            rows = list(csv.reader(f))
        data_row = rows[1]
        assert int(data_row[0]) == 0  # frame 0
        assert abs(float(data_row[1]) - 0.0) < 1e-5  # time_s = 0/10 = 0.0
        assert int(data_row[2]) == 0  # person 0
        assert abs(float(data_row[3]) - 1.5) < 1e-4  # j0_x
        assert abs(float(data_row[4]) - (-2.3)) < 1e-4  # j0_y
        assert abs(float(data_row[5]) - 0.7) < 1e-4  # j0_z
        assert abs(float(data_row[6]) - 0.85) < 1e-4  # j0_conf


class TestBuildParser:
    """Tests for the recon3d CLI argument parser."""

    def test_export_csv_flag_exists(self):
        """--export-csv flag must be present (CODE-BUG-003)."""
        args = build_parser().parse_args(
            ["--calib", "c.yml", "--pose2d", "p/", "--export-csv", "out.csv"]
        )
        assert args.export_csv == "out.csv"

    def test_export_csv_default_is_none(self):
        """--export-csv must default to None when not supplied."""
        args = build_parser().parse_args(["--calib", "c.yml", "--pose2d", "p/"])
        assert args.export_csv is None
