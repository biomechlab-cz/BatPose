"""Unit tests for app.calib.validate (calibration repeatability comparison)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.calib.validate import (
    SEGMENTS,
    MetricRow,
    build_parser,
    collect_parameter_rows,
    compare_calibrations,
    format_report,
    pairwise_rotation_matrix,
    report_to_dict,
    rotation_angle_deg,
    rotation_diff_deg,
    write_csv,
)

# ── Fixtures: minimal loaded-calibration dicts (as load_calibration returns) ───

_K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 600.0], [0.0, 0.0, 1.0]])
_D4 = np.zeros((4, 1))


def _calib(
    fx: float = 1000.0,
    baseline_m: float = 1.0,
    yaw_deg: float = 0.0,
    rms: float = 0.5,
    n_frames: int = 40,
    lens_model: str = "fisheye",
    image_size: tuple[int, int] = (1920, 1200),
    board_cfg: dict | None = None,
    n_coeffs: int = 4,
) -> dict:
    K1 = _K.copy()
    K1[0, 0] = fx
    yaw = np.radians(yaw_deg)
    R = np.array(
        [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]]
    )
    return {
        "K1": K1,
        "D1": np.zeros((n_coeffs, 1)),
        "K2": _K.copy(),
        "D2": np.zeros((n_coeffs, 1)),
        "R": R,
        "T": np.array([[-baseline_m], [0.0], [0.0]]),
        "image_size": image_size,
        "board_cfg": board_cfg if board_cfg is not None else {"type": "charuco", "squares_x": 5},
        "quality": {"rms": rms, "n_frames_used": n_frames},
        "lens_model": lens_model,
        "world_frame": None,
    }


def _row(name: str, rows: list[MetricRow]) -> MetricRow:
    matches = [r for r in rows if r.name == name]
    assert matches, f"no row named {name!r} in {[r.name for r in rows]}"
    return matches[0]


# ── Statistics ────────────────────────────────────────────────────────────────


class TestMetricRow:
    def test_mean_sd_range_cov(self):
        r = MetricRow("x", "px", [10.0, 12.0, 14.0])
        assert r.mean == pytest.approx(12.0)
        assert r.sd == pytest.approx(2.0)  # sample SD (ddof=1)
        assert r.rng == pytest.approx(4.0)
        assert r.cov_pct == pytest.approx(100.0 * 2.0 / 12.0)

    def test_identical_values_give_zero_spread(self):
        r = MetricRow("x", "px", [7.0, 7.0, 7.0])
        assert r.sd == pytest.approx(0.0)
        assert r.cov_pct == pytest.approx(0.0)

    def test_cov_suppressed_for_signed_quantities(self):
        r = MetricRow("Tx", "mm", [-1000.0, -1002.0], cov_meaningful=False)
        assert np.isnan(r.cov_pct)
        assert np.isfinite(r.sd)

    def test_nan_values_ignored(self):
        r = MetricRow("x", "px", [10.0, float("nan"), 12.0])
        assert r.mean == pytest.approx(11.0)
        assert np.isfinite(r.sd)

    def test_all_nan_is_not_a_crash(self):
        r = MetricRow("x", "px", [float("nan"), float("nan")])
        assert np.isnan(r.mean) and np.isnan(r.sd) and np.isnan(r.cov_pct)

    def test_single_value_has_undefined_sd(self):
        assert np.isnan(MetricRow("x", "px", [3.0]).sd)

    def test_range_of_one_finite_value_is_not_zero(self):
        """0.0 would read as perfect agreement; only one run produced a number."""
        assert np.isnan(MetricRow("x", "mm", [401.8, float("nan")]).rng)


# ── Rotation helpers ──────────────────────────────────────────────────────────


class TestRotation:
    def test_identity_is_zero_degrees(self):
        assert rotation_angle_deg(np.eye(3)) == pytest.approx(0.0)

    @pytest.mark.parametrize("deg", [0.5, 18.0, 45.0, 90.0, 179.0])
    def test_known_yaw_recovered(self, deg):
        yaw = np.radians(deg)
        R = np.array(
            [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]]
        )
        assert rotation_angle_deg(R) == pytest.approx(deg, abs=1e-6)

    def test_diff_of_two_rotations(self):
        a, b = _calib(yaw_deg=18.0)["R"], _calib(yaw_deg=20.5)["R"]
        assert rotation_diff_deg(a, b) == pytest.approx(2.5, abs=1e-6)

    def test_trace_clamp_survives_non_orthonormal_input(self):
        """Float drift must not produce a NaN out of arccos."""
        R = np.eye(3) * (1.0 + 1e-9)
        assert np.isfinite(rotation_angle_deg(R))

    def test_pairwise_matrix_is_symmetric_with_zero_diagonal(self):
        calibs = [_calib(yaw_deg=d) for d in (0.0, 2.0, 5.0)]
        m = np.array(pairwise_rotation_matrix(calibs))
        assert np.allclose(np.diag(m), 0.0)
        assert np.allclose(m, m.T)
        assert m[0, 2] == pytest.approx(5.0, abs=1e-6)


# ── Parameter rows ────────────────────────────────────────────────────────────


class TestParameterRows:
    def test_expected_rows_present(self):
        rows = collect_parameter_rows([_calib(), _calib()])
        names = {r.name for r in rows}
        for expected in ("L fx", "R cy", "|T| baseline", "Tx", "R total angle", "reported RMS"):
            assert expected in names

    def test_baseline_reported_in_mm(self):
        rows = collect_parameter_rows([_calib(baseline_m=1.13), _calib(baseline_m=1.13)])
        assert _row("|T| baseline", rows).mean == pytest.approx(1130.0)

    def test_baseline_spread_detected(self):
        rows = collect_parameter_rows([_calib(baseline_m=1.10), _calib(baseline_m=1.12)])
        r = _row("|T| baseline", rows)
        assert r.rng == pytest.approx(20.0, abs=1e-6)  # 20 mm apart
        assert r.cov_pct == pytest.approx(100.0 * r.sd / 1110.0)

    def test_focal_spread_detected(self):
        rows = collect_parameter_rows([_calib(fx=1000.0), _calib(fx=1004.0)])
        assert _row("L fx", rows).rng == pytest.approx(4.0)

    def test_rotation_angle_row(self):
        rows = collect_parameter_rows([_calib(yaw_deg=18.0), _calib(yaw_deg=18.4)])
        assert _row("R total angle", rows).rng == pytest.approx(0.4, abs=1e-6)

    def test_distortion_coefficient_count_follows_model(self):
        rows5 = collect_parameter_rows([_calib(n_coeffs=5), _calib(n_coeffs=5)])
        assert _row("L k5", rows5)
        rows4 = collect_parameter_rows([_calib(), _calib()])
        assert not [r for r in rows4 if r.name == "L k5"]

    def test_missing_quality_becomes_nan_not_crash(self):
        a, b = _calib(), _calib()
        a["quality"] = {}
        rows = collect_parameter_rows([a, b])
        assert np.isnan(_row("reported RMS", rows).values[0])


# ── Input validation ──────────────────────────────────────────────────────────


class TestCompareValidation:
    def test_single_calibration_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            compare_calibrations([_calib()], ["A"])

    def test_label_count_mismatch_rejected(self):
        with pytest.raises(ValueError, match="labels"):
            compare_calibrations([_calib(), _calib()], ["A"])

    def test_mixed_lens_models_rejected(self):
        with pytest.raises(ValueError, match="lens model"):
            compare_calibrations(
                [_calib(lens_model="fisheye"), _calib(lens_model="standard")], ["A", "B"]
            )

    def test_mixed_image_sizes_rejected(self):
        with pytest.raises(ValueError, match="image sizes"):
            compare_calibrations([_calib(), _calib(image_size=(1280, 800))], ["A", "B"])

    def test_mixed_coefficient_counts_rejected(self):
        with pytest.raises(ValueError, match="distortion-coefficient counts"):
            compare_calibrations([_calib(n_coeffs=4), _calib(n_coeffs=5)], ["A", "B"])

    def test_different_boards_warns_but_proceeds(self):
        rep = compare_calibrations(
            [_calib(), _calib(board_cfg={"type": "checkerboard", "cols": 9})], ["A", "B"]
        )
        assert rep.warnings
        assert "board" in rep.warnings[0].lower()

    def test_identical_calibrations_report_zero_spread(self):
        rep = compare_calibrations([_calib(), _calib(), _calib()], ["A", "B", "C"])
        for r in rep.parameter_rows:
            assert r.sd == pytest.approx(0.0), r.name


# ── Rendering / export ────────────────────────────────────────────────────────


class TestRendering:
    def test_report_is_ascii_only(self):
        """Windows consoles here are cp1250 — non-ASCII output crashes print."""
        rep = compare_calibrations([_calib(), _calib(fx=1003.0)], ["A", "B"])
        text = format_report(rep)
        assert text.isascii(), [c for c in text if not c.isascii()]

    def test_report_contains_labels_and_values(self):
        rep = compare_calibrations(
            [_calib(baseline_m=1.10), _calib(baseline_m=1.12)],
            ["runA", "runB"],
            paths=["a.yml", "b.yml"],
        )
        text = format_report(rep)
        assert "runA" in text and "runB" in text
        assert "a.yml" in text
        assert "|T| baseline" in text
        assert "1100.00" in text and "1120.00" in text

    def test_long_title_does_not_widen_name_column(self):
        from app.calib.validate import format_table

        rows = [MetricRow("L fx", "px", [1.0, 2.0])]
        short = format_table(rows, ["A", "B"], "T")
        long = format_table(rows, ["A", "B"], "A very long table title indeed")
        assert short.splitlines()[2] == long.splitlines()[2]  # header row identical

    def test_nan_rendered_as_na(self):
        a, b = _calib(), _calib()
        a["quality"] = {}
        b["quality"] = {}
        text = format_report(compare_calibrations([a, b], ["A", "B"]))
        assert "n/a" in text

    def test_csv_roundtrip(self, tmp_path):
        rep = compare_calibrations([_calib(), _calib(fx=1002.0)], ["A", "B"])
        path = tmp_path / "p.csv"
        write_csv(path, rep.parameter_rows, rep.labels)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0].split(",") == [
            "parameter",
            "unit",
            "A",
            "B",
            "mean",
            "sd",
            "range",
            "cov_pct",
        ]
        assert len(lines) == len(rep.parameter_rows) + 1

    def test_json_is_serializable_and_keyed_by_label(self, tmp_path):
        rep = compare_calibrations([_calib(), _calib(fx=1002.0)], ["A", "B"])
        blob = json.loads(json.dumps(report_to_dict(rep)))
        assert blob["test"] == "repeat"
        fx = [p for p in blob["parameters"] if p["parameter"] == "L fx"][0]
        assert set(fx["values"]) == {"A", "B"}
        assert fx["values"]["B"] == pytest.approx(1002.0)


# ── Segment-length cross-check (needs cv2) ────────────────────────────────────


class TestSegmentCrossCheck:
    """A real triangulation round-trip on synthetic pinhole projections."""

    @staticmethod
    def _synthetic_pose2d(n_frames: int = 4):
        cv2 = pytest.importorskip("cv2")

        K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        R = np.eye(3)
        T = np.array([[-0.5], [0.0], [0.0]])

        # 17 COCO joints on a plausible standing body, 2.5 m out.
        rng = np.random.default_rng(0)
        base = np.zeros((17, 3), dtype=np.float64)
        base[:, 2] = 2.5
        base[5] = [0.20, 0.55, 2.5]  # L shoulder
        base[6] = [-0.20, 0.55, 2.5]  # R shoulder
        base[7] = [0.22, 0.25, 2.5]  # L elbow
        base[8] = [-0.22, 0.25, 2.5]  # R elbow
        base[9] = [0.24, -0.02, 2.5]  # L wrist
        base[10] = [-0.24, -0.02, 2.5]  # R wrist
        base[11] = [0.10, 0.00, 2.5]  # L hip
        base[12] = [-0.10, 0.00, 2.5]  # R hip
        base[13] = [0.11, -0.45, 2.5]  # L knee
        base[14] = [-0.11, -0.45, 2.5]  # R knee
        base[15] = [0.12, -0.88, 2.5]  # L ankle
        base[16] = [-0.12, -0.88, 2.5]  # R ankle

        def project(pts, Rc, tc):
            rvec, _ = cv2.Rodrigues(Rc)
            out, _ = cv2.projectPoints(pts, rvec, tc.reshape(3, 1), K, np.zeros(5))
            return out.reshape(-1, 2)

        kps_l = np.zeros((n_frames, 1, 17, 2), dtype=np.float32)
        kps_r = np.zeros((n_frames, 1, 17, 2), dtype=np.float32)
        for t in range(n_frames):
            pts = base + rng.normal(0.0, 0.002, base.shape)  # tiny sway
            kps_l[t, 0] = project(pts, np.eye(3), np.zeros(3))
            kps_r[t, 0] = project(pts, R, T)
        conf = np.ones((n_frames, 1, 17), dtype=np.float32)

        calib = {
            "K1": K,
            "D1": np.zeros(5),
            "K2": K,
            "D2": np.zeros(5),
            "R": R,
            "T": T,
            "image_size": (640, 480),
            "board_cfg": {"type": "charuco"},
            "quality": {"rms": 0.4, "n_frames_used": 30},
            "lens_model": "standard",
            "world_frame": None,
        }
        return calib, (kps_l, conf, kps_r, conf), base

    def test_segment_lengths_recover_truth(self):
        calib, pose2d, base = self._synthetic_pose2d()
        from app.calib.validate import segment_lengths_mm

        medians, counts, med_err = segment_lengths_mm(
            calib, pose2d[0], pose2d[1], pose2d[2], pose2d[3]
        )
        truth_thigh = np.linalg.norm(base[11] - base[13]) * 1000.0
        assert medians["L thigh"] == pytest.approx(truth_thigh, abs=5.0)  # < 5 mm
        assert counts["L thigh"] == pose2d[0].shape[0]
        assert med_err < 1.0  # noiseless projections reproject sub-pixel

    def test_identical_calibrations_give_zero_segment_spread(self):
        calib, pose2d, _ = self._synthetic_pose2d()
        rep = compare_calibrations([calib, dict(calib)], ["A", "B"], pose2d=pose2d)
        assert rep.segment_rows
        for r in rep.segment_rows:
            assert r.sd == pytest.approx(0.0, abs=1e-9), r.name

    def test_perturbed_baseline_scales_segments(self):
        """A 1 % baseline error must show up as ~1 % on every segment length."""
        calib, pose2d, _ = self._synthetic_pose2d()
        bad = dict(calib)
        bad["T"] = calib["T"] * 1.01
        rep = compare_calibrations([calib, bad], ["good", "bad"], pose2d=pose2d)
        thigh = _row("L thigh", rep.segment_rows)
        ratio = thigh.values[1] / thigh.values[0]
        assert ratio == pytest.approx(1.01, abs=2e-3)

    def test_segment_row_names_match_definitions(self):
        calib, pose2d, _ = self._synthetic_pose2d()
        rep = compare_calibrations([calib, dict(calib)], ["A", "B"], pose2d=pose2d)
        names = [r.name for r in rep.segment_rows]
        for seg_name, _, _ in SEGMENTS:
            assert seg_name in names
        assert "median reproj err" in names

    def test_report_includes_worst_case_line(self):
        calib, pose2d, _ = self._synthetic_pose2d()
        bad = dict(calib)
        bad["T"] = calib["T"] * 1.01
        text = format_report(compare_calibrations([calib, bad], ["A", "B"], pose2d=pose2d))
        assert "Worst-case segment-length agreement" in text
        assert text.isascii()


# ── CLI parser ────────────────────────────────────────────────────────────────


class TestCli:
    def test_repeat_accepts_multiple_calibs(self):
        args = build_parser().parse_args(["repeat", "--calib", "a.yml", "b.yml", "c.yml"])
        assert args.command == "repeat"
        assert args.calib == ["a.yml", "b.yml", "c.yml"]

    def test_defaults(self):
        args = build_parser().parse_args(["repeat", "--calib", "a.yml", "b.yml"])
        assert args.min_conf == 0.3
        assert args.max_reproj_err == 20.0
        assert args.person == 0
        assert args.label is None
        assert args.out is None

    def test_subcommand_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_calib_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["repeat"])

    def test_labels_and_out(self):
        args = build_parser().parse_args(
            ["repeat", "--calib", "a.yml", "b.yml", "--label", "A", "B", "--out", "o"]
        )
        assert args.label == ["A", "B"]
        assert args.out == "o"


class TestCliEndToEnd:
    """Exercise main() through real files."""

    @staticmethod
    def _write_calib(path, **kw):
        pytest.importorskip("cv2")
        import yaml

        c = _calib(**kw)
        data = {
            "image_size": list(c["image_size"]),
            "board_cfg": c["board_cfg"],
            "lens_model": c["lens_model"],
            "K1": c["K1"].tolist(),
            "D1": c["D1"].tolist(),
            "K2": c["K2"].tolist(),
            "D2": c["D2"].tolist(),
            "R": c["R"].tolist(),
            "T": c["T"].tolist(),
            "quality": c["quality"],
        }
        with open(path, "w") as fh:
            yaml.dump(data, fh)
        return str(path)

    def test_main_prints_table(self, tmp_path, capsys):
        from app.calib.validate import main

        a = self._write_calib(tmp_path / "a.yml", baseline_m=1.10)
        b = self._write_calib(tmp_path / "b.yml", baseline_m=1.12)
        rc = main(["repeat", "--calib", a, b])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Calibration repeatability: 2 runs" in out
        assert "|T| baseline" in out

    def test_main_writes_outputs(self, tmp_path):
        from app.calib.validate import main

        a = self._write_calib(tmp_path / "a.yml")
        b = self._write_calib(tmp_path / "b.yml", fx=1005.0)
        out_dir = tmp_path / "res"
        assert main(["repeat", "--calib", a, b, "--out", str(out_dir)]) == 0
        assert (out_dir / "repeat_parameters.csv").exists()
        blob = json.loads((out_dir / "repeat_metrics.json").read_text(encoding="utf-8"))
        assert blob["lens_model"] == "fisheye"
        assert len(blob["rotation_pairwise_deg"]) == 2

    def test_main_rejects_single_calib(self, tmp_path, capsys):
        from app.calib.validate import main

        a = self._write_calib(tmp_path / "a.yml")
        assert main(["repeat", "--calib", a]) == 1
        assert "at least two" in capsys.readouterr().err

    def test_main_rejects_half_a_pose2d_pair(self, tmp_path, capsys):
        from app.calib.validate import main

        a = self._write_calib(tmp_path / "a.yml")
        b = self._write_calib(tmp_path / "b.yml")
        rc = main(["repeat", "--calib", a, b, "--pose2d-left", "l.npz"])
        assert rc == 1
        assert "together" in capsys.readouterr().err

    def test_main_reports_mixed_lens_models_as_error(self, tmp_path, capsys):
        from app.calib.validate import main

        a = self._write_calib(tmp_path / "a.yml", lens_model="fisheye")
        b = self._write_calib(tmp_path / "b.yml", lens_model="standard")
        assert main(["repeat", "--calib", a, b]) == 1
        assert "lens model" in capsys.readouterr().err
