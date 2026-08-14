"""Calibration validation tools.

Currently one subcommand:

``repeat`` — compare N independently produced ``calibration.yml`` files and
report the spread of every calibration parameter (mean / SD / CoV / range).
The point is that ``quality.rms`` is a residual of the fit that produced it:
it says nothing about how reproducible the calibration is.  Calibrating the
same rig several times and quantifying the disagreement does.

Optionally (``--pose2d-left/--pose2d-right``) the same 2D pose recording is
triangulated with each calibration and the resulting body-segment lengths are
compared.  Parameter SDs are hard to interpret on their own — "the three
calibrations agree to +/-4 mm on the same recording" is not.

Pure library functions take already-loaded calibration dicts (as returned by
:func:`app.calib.stereo.load_calibration`), so they are testable without file
I/O; only :func:`cmd_repeat` touches the filesystem.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.console import console_safe

# COCO-17 segments with a physically stable length (see docs/skeleton_mapping.md).
# Used only for the optional cross-check: the absolute values depend on
# MediaPipe's joint placement, but the *spread across calibrations* does not.
SEGMENTS: list[tuple[str, int, int]] = [
    ("L thigh", 11, 13),
    ("R thigh", 12, 14),
    ("L shank", 13, 15),
    ("R shank", 14, 16),
    ("L upper arm", 5, 7),
    ("R upper arm", 6, 8),
    ("L forearm", 7, 9),
    ("R forearm", 8, 10),
    ("Shoulder width", 5, 6),
    ("Hip width", 11, 12),
]


# ── Metric rows ───────────────────────────────────────────────────────────────


@dataclass
class MetricRow:
    """One compared quantity across N calibration runs."""

    name: str
    unit: str
    values: list[float]
    decimals: int = 3
    #: Coefficient of variation needs a meaningful zero and a single sign.
    #: It is nonsense for signed components (Tx, rvec, distortion coeffs), so
    #: those rows report SD only.
    cov_meaningful: bool = True

    @property
    def finite(self) -> np.ndarray:
        v = np.asarray(self.values, dtype=np.float64)
        return v[np.isfinite(v)]

    @property
    def mean(self) -> float:
        f = self.finite
        return float(np.mean(f)) if f.size else float("nan")

    @property
    def sd(self) -> float:
        f = self.finite
        # Sample SD: we are estimating the spread of the process, not
        # describing a fixed population of runs.
        return float(np.std(f, ddof=1)) if f.size > 1 else float("nan")

    @property
    def rng(self) -> float:
        f = self.finite
        # A range over a single value is 0.0, which reads as "perfect
        # agreement" when it actually means "only one run produced a number".
        return float(np.max(f) - np.min(f)) if f.size > 1 else float("nan")

    @property
    def cov_pct(self) -> float:
        if not self.cov_meaningful:
            return float("nan")
        m, s = self.mean, self.sd
        if not np.isfinite(m) or not np.isfinite(s) or abs(m) < 1e-12:
            return float("nan")
        return 100.0 * s / abs(m)


@dataclass
class RepeatReport:
    """Everything :func:`compare_calibrations` produced."""

    labels: list[str]
    paths: list[str]
    lens_model: str
    image_size: tuple[int, int]
    parameter_rows: list[MetricRow]
    rotation_pairwise_deg: list[list[float]]
    segment_rows: list[MetricRow] = field(default_factory=list)
    segment_frames: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ── Rotation helpers ──────────────────────────────────────────────────────────


def rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic magnitude of rotation *R* in degrees.

    For stereo extrinsics this is the relative orientation of the two cameras
    (the "toe-in"), the single number most sensitive to a bad calibration.
    """
    R = np.asarray(R, dtype=np.float64)
    # Clamp guards against |trace| drifting outside the valid range through
    # floating-point error in a nominally orthonormal matrix.
    cos_theta = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def rotation_diff_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    Ra = np.asarray(Ra, dtype=np.float64)
    Rb = np.asarray(Rb, dtype=np.float64)
    return rotation_angle_deg(Ra.T @ Rb)


def rvec_deg(R: np.ndarray) -> np.ndarray:
    """Rotation vector of *R* in degrees (axis * angle).

    Convention-free alternative to Euler angles: no gimbal ambiguity and no
    XYZ-order footgun when comparing runs.
    """
    import cv2

    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return np.degrees(rvec.reshape(3))


# ── Parameter comparison ──────────────────────────────────────────────────────


def _dist_coeffs(D: np.ndarray) -> np.ndarray:
    return np.asarray(D, dtype=np.float64).reshape(-1)


def _check_comparable(calibs: list[dict[str, Any]]) -> list[str]:
    """Reject sets that are not a repeatability comparison; warn on the rest."""
    warnings: list[str] = []

    lens = {c.get("lens_model", "standard") for c in calibs}
    if len(lens) > 1:
        raise ValueError(
            f"Calibrations use different lens models {sorted(lens)} — their distortion "
            "coefficients are not the same quantity, so comparing them is meaningless. "
            "Re-run the odd one out with the correct --lens."
        )

    sizes = {tuple(c["image_size"]) for c in calibs}
    if len(sizes) > 1:
        raise ValueError(
            f"Calibrations have different image sizes {sorted(sizes)} — intrinsics are in "
            "pixels and cannot be compared across resolutions."
        )

    n_coeffs = {_dist_coeffs(c["D1"]).size for c in calibs} | {
        _dist_coeffs(c["D2"]).size for c in calibs
    }
    if len(n_coeffs) > 1:
        raise ValueError(
            f"Calibrations have different distortion-coefficient counts {sorted(n_coeffs)}."
        )

    boards = {json.dumps(c.get("board_cfg") or {}, sort_keys=True) for c in calibs}
    if len(boards) > 1:
        warnings.append(
            "Runs used different board configurations — the spread below mixes board "
            "differences with calibration repeatability."
        )

    return warnings


def collect_parameter_rows(calibs: list[dict[str, Any]]) -> list[MetricRow]:
    """Build the intrinsics/extrinsics comparison rows."""
    rows: list[MetricRow] = []

    for cam, (k_key, d_key) in enumerate([("K1", "D1"), ("K2", "D2")], start=1):
        side = "L" if cam == 1 else "R"
        Ks = [np.asarray(c[k_key], dtype=np.float64) for c in calibs]
        for label, (i, j) in [("fx", (0, 0)), ("fy", (1, 1)), ("cx", (0, 2)), ("cy", (1, 2))]:
            rows.append(
                MetricRow(f"{side} {label}", "px", [float(K[i, j]) for K in Ks], decimals=2)
            )
        Ds = [_dist_coeffs(c[d_key]) for c in calibs]
        for i in range(Ds[0].size):
            rows.append(
                MetricRow(
                    f"{side} k{i + 1}",
                    "-",
                    [float(D[i]) for D in Ds],
                    decimals=5,
                    cov_meaningful=False,
                )
            )

    Ts = [np.asarray(c["T"], dtype=np.float64).reshape(3) * 1000.0 for c in calibs]
    rows.append(MetricRow("|T| baseline", "mm", [float(np.linalg.norm(t)) for t in Ts], decimals=2))
    for i, axis in enumerate("xyz"):
        rows.append(
            MetricRow(f"T{axis}", "mm", [float(t[i]) for t in Ts], decimals=2, cov_meaningful=False)
        )

    Rs = [np.asarray(c["R"], dtype=np.float64) for c in calibs]
    rows.append(MetricRow("R total angle", "deg", [rotation_angle_deg(R) for R in Rs], decimals=3))
    rvecs = [rvec_deg(R) for R in Rs]
    for i, axis in enumerate("xyz"):
        rows.append(
            MetricRow(
                f"R rvec {axis}",
                "deg",
                [float(v[i]) for v in rvecs],
                decimals=3,
                cov_meaningful=False,
            )
        )

    rows.append(
        MetricRow(
            "reported RMS",
            "px",
            [float(c.get("quality", {}).get("rms", float("nan"))) for c in calibs],
            decimals=3,
        )
    )
    rows.append(
        MetricRow(
            "frames used",
            "-",
            [float(c.get("quality", {}).get("n_frames_used", float("nan"))) for c in calibs],
            decimals=0,
        )
    )
    return rows


def pairwise_rotation_matrix(calibs: list[dict[str, Any]]) -> list[list[float]]:
    """Symmetric matrix of geodesic angles (deg) between every pair of runs."""
    n = len(calibs)
    out = [[0.0] * n for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            d = rotation_diff_deg(calibs[a]["R"], calibs[b]["R"])
            out[a][b] = out[b][a] = d
    return out


# ── Optional segment-length cross-check ───────────────────────────────────────


def segment_lengths_mm(
    calib: dict[str, Any],
    kps_l: np.ndarray,
    conf_l: np.ndarray,
    kps_r: np.ndarray,
    conf_r: np.ndarray,
    person: int = 0,
    min_conf: float = 0.3,
    max_reproj_err: float = 20.0,
) -> tuple[dict[str, float], dict[str, int], float]:
    """Triangulate a whole 2D recording with *calib* and measure segments.

    Returns ``(median_length_mm, n_frames, median_reproj_px)``.  Medians are
    used because a handful of badly triangulated frames must not move the
    number being compared across calibrations.
    """
    from app.recon3d.triangulate import triangulate_frame_pair

    fisheye = calib.get("lens_model", "standard") == "fisheye"
    n_frames = kps_l.shape[0]
    per_seg: dict[str, list[float]] = {name: [] for name, _, _ in SEGMENTS}
    errs: list[float] = []

    for t in range(n_frames):
        j3d, c3d, err = triangulate_frame_pair(
            kps_l[t, person],
            kps_r[t, person],
            conf_l[t, person],
            conf_r[t, person],
            calib["K1"],
            calib["D1"],
            calib["K2"],
            calib["D2"],
            calib["R"],
            calib["T"],
            min_conf=min_conf,
            max_reproj_err=max_reproj_err,
            lens_model="fisheye" if fisheye else "standard",
        )
        pts, conf, e = j3d[0], c3d[0], err[0]
        good = conf > 0
        errs.extend(float(x) for x in e[good & np.isfinite(e)])
        for name, a, b in SEGMENTS:
            if good[a] and good[b]:
                per_seg[name].append(float(np.linalg.norm(pts[a] - pts[b]) * 1000.0))

    medians = {name: (float(np.median(v)) if v else float("nan")) for name, v in per_seg.items()}
    counts = {name: len(v) for name, v in per_seg.items()}
    med_err = float(np.median(errs)) if errs else float("nan")
    return medians, counts, med_err


def collect_segment_rows(
    calibs: list[dict[str, Any]],
    kps_l: np.ndarray,
    conf_l: np.ndarray,
    kps_r: np.ndarray,
    conf_r: np.ndarray,
    person: int = 0,
    min_conf: float = 0.3,
    max_reproj_err: float = 20.0,
) -> tuple[list[MetricRow], dict[str, list[int]]]:
    """Segment-length rows plus the per-segment usable-frame counts."""
    results = [
        segment_lengths_mm(c, kps_l, conf_l, kps_r, conf_r, person, min_conf, max_reproj_err)
        for c in calibs
    ]
    rows = [
        MetricRow(name, "mm", [r[0][name] for r in results], decimals=2) for name, _, _ in SEGMENTS
    ]
    rows.append(MetricRow("median reproj err", "px", [r[2] for r in results], decimals=2))
    frames = {name: [r[1][name] for r in results] for name, _, _ in SEGMENTS}
    return rows, frames


# ── Top-level comparison ──────────────────────────────────────────────────────


def compare_calibrations(
    calibs: list[dict[str, Any]],
    labels: list[str],
    paths: list[str] | None = None,
    pose2d: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    person: int = 0,
    min_conf: float = 0.3,
    max_reproj_err: float = 20.0,
) -> RepeatReport:
    """Compare N loaded calibrations; optionally cross-check on one recording."""
    if len(calibs) < 2:
        raise ValueError(f"Repeatability needs at least 2 calibrations, got {len(calibs)}.")
    if len(labels) != len(calibs):
        raise ValueError(f"Got {len(labels)} labels for {len(calibs)} calibrations.")

    warnings = _check_comparable(calibs)
    report = RepeatReport(
        labels=labels,
        paths=list(paths or [""] * len(calibs)),
        lens_model=calibs[0].get("lens_model", "standard"),
        image_size=tuple(int(x) for x in calibs[0]["image_size"]),  # type: ignore[arg-type]
        parameter_rows=collect_parameter_rows(calibs),
        rotation_pairwise_deg=pairwise_rotation_matrix(calibs),
        warnings=warnings,
    )

    if pose2d is not None:
        kps_l, conf_l, kps_r, conf_r = pose2d
        rows, frames = collect_segment_rows(
            calibs, kps_l, conf_l, kps_r, conf_r, person, min_conf, max_reproj_err
        )
        report.segment_rows = rows
        report.segment_frames = frames

    return report


# ── Rendering ─────────────────────────────────────────────────────────────────


def _fmt(value: float, decimals: int) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def format_table(rows: list[MetricRow], labels: list[str], title: str) -> str:
    """ASCII table — Windows consoles here are cp1250, so no box characters."""
    # Driven by the row names only — a long title must not stretch the column.
    name_w = max([len(r.name) for r in rows] + [len("Parameter"), 18])
    val_w = max([11] + [len(x) + 2 for x in labels])
    head = (
        f"{'Parameter':<{name_w}}  {'Unit':<5}"
        + "".join(f"{lab:>{val_w}}" for lab in labels)
        + f"{'Mean':>{val_w}}{'SD':>{val_w}}{'Range':>{val_w}}{'CoV%':>9}"
    )
    lines = [title, "-" * len(head), head, "-" * len(head)]
    for r in rows:
        cells = "".join(f"{_fmt(v, r.decimals):>{val_w}}" for v in r.values)
        lines.append(
            f"{r.name:<{name_w}}  {r.unit:<5}{cells}"
            f"{_fmt(r.mean, r.decimals):>{val_w}}"
            f"{_fmt(r.sd, r.decimals):>{val_w}}"
            f"{_fmt(r.rng, r.decimals):>{val_w}}"
            f"{_fmt(r.cov_pct, 3):>9}"
        )
    return "\n".join(lines)


def format_report(report: RepeatReport) -> str:
    n = len(report.labels)
    w, h = report.image_size
    out = [
        f"=== Calibration repeatability: {n} runs ===",
        "",
        "Runs:",
    ]
    for lab, path in zip(report.labels, report.paths):
        out.append(f"  {lab} = {path}")
    out += [
        f"  lens_model={report.lens_model}  image_size={w}x{h}",
        "",
        format_table(report.parameter_rows, report.labels, "Intrinsics / extrinsics"),
        "",
        "Pairwise relative-rotation difference (deg):",
    ]
    lab_w = max(len(x) for x in report.labels) + 2
    out.append("  " + " " * lab_w + "".join(f"{lab:>9}" for lab in report.labels))
    for i, lab in enumerate(report.labels):
        cells = "".join(f"{report.rotation_pairwise_deg[i][j]:>9.3f}" for j in range(n))
        out.append(f"  {lab:<{lab_w}}{cells}")

    if report.segment_rows:
        out += [
            "",
            format_table(
                report.segment_rows,
                report.labels,
                "Same recording triangulated with each calibration",
            ),
        ]
        seg_rows = [r for r in report.segment_rows if r.unit == "mm"]
        usable = [r for r in seg_rows if np.isfinite(r.sd)]
        if usable:
            worst = max(usable, key=lambda r: r.sd)
            out += [
                "",
                f"Worst-case segment-length agreement: SD = {worst.sd:.2f} mm "
                f"({worst.name}), over {len(usable)} of {len(seg_rows)} segments, "
                f"{n} calibrations, one recording.",
            ]
        dropped = [r.name for r in seg_rows if not np.isfinite(r.sd)]
        if dropped:
            out += [
                "",
                "NOTE: no usable frames in at least one run for: "
                + ", ".join(dropped)
                + " (see segment_frames in the JSON for per-run counts).",
            ]

    if report.warnings:
        out.append("")
        for msg in report.warnings:
            out.append(f"WARNING: {msg}")
    return "\n".join(out)


def write_csv(path: Path, rows: list[MetricRow], labels: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["parameter", "unit", *labels, "mean", "sd", "range", "cov_pct"])
        for r in rows:
            wr.writerow(
                [r.name, r.unit, *[f"{v:.10g}" for v in r.values]]
                + [f"{x:.10g}" for x in (r.mean, r.sd, r.rng, r.cov_pct)]
            )


def report_to_dict(report: RepeatReport) -> dict[str, Any]:
    def rows_to_dict(rows: list[MetricRow]) -> list[dict[str, Any]]:
        return [
            {
                "parameter": r.name,
                "unit": r.unit,
                "values": dict(zip(report.labels, r.values)),
                "mean": r.mean,
                "sd": r.sd,
                "range": r.rng,
                "cov_pct": r.cov_pct,
            }
            for r in rows
        ]

    return {
        "test": "repeat",
        "runs": dict(zip(report.labels, report.paths)),
        "lens_model": report.lens_model,
        "image_size": list(report.image_size),
        "parameters": rows_to_dict(report.parameter_rows),
        "rotation_pairwise_deg": report.rotation_pairwise_deg,
        "segments": rows_to_dict(report.segment_rows),
        "segment_frames": report.segment_frames,
        "warnings": report.warnings,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.calib.validate",
        description="Validate stereo calibrations (currently: repeatability).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    rp = sub.add_parser(
        "repeat",
        help="Compare N calibration.yml files and report parameter spread.",
        description=(
            "Compare N independently produced calibrations of the same rig. "
            "quality.rms is an in-sample residual; the spread across repeated "
            "calibrations is what tells you how reproducible the result is."
        ),
    )
    rp.add_argument(
        "--calib",
        required=True,
        nargs="+",
        metavar="YML",
        help="Two or more calibration.yml files to compare",
    )
    rp.add_argument(
        "--label",
        nargs="+",
        metavar="NAME",
        help="Short column labels (default: A, B, C, …)",
    )
    rp.add_argument(
        "--pose2d-left",
        help="Optional pose2d_left.npz — triangulate this recording with every "
        "calibration and compare the resulting segment lengths",
    )
    rp.add_argument("--pose2d-right", help="Matching pose2d_right.npz")
    rp.add_argument("--person", type=int, default=0, help="Person index (default: 0)")
    rp.add_argument("--min-conf", type=float, default=0.3, help="2D confidence gate (default: 0.3)")
    rp.add_argument(
        "--max-reproj-err",
        type=float,
        default=20.0,
        help="Reprojection gate in px (default: 20.0)",
    )
    rp.add_argument("--out", help="Directory for repeat_*.csv and repeat_metrics.json")
    return p


def cmd_repeat(args: argparse.Namespace) -> int:
    from app.calib.stereo import load_calibration

    paths = list(args.calib)
    if len(paths) < 2:
        print("Error: --calib needs at least two files.", file=sys.stderr)
        return 1

    if args.label:
        if len(args.label) != len(paths):
            print(
                f"Error: got {len(args.label)} --label values for {len(paths)} calibrations.",
                file=sys.stderr,
            )
            return 1
        labels = list(args.label)
    else:
        # A, B, C, … then A2, B2, … past 26 runs (never expected, but no crash).
        labels = [
            chr(ord("A") + i % 26) + ("" if i < 26 else str(i // 26 + 1)) for i in range(len(paths))
        ]

    calibs = [load_calibration(p) for p in paths]

    pose2d = None
    if bool(args.pose2d_left) != bool(args.pose2d_right):
        print(
            "Error: --pose2d-left and --pose2d-right must be given together.",
            file=sys.stderr,
        )
        return 1
    if args.pose2d_left:
        from app.pose2d.pipeline import load_pose2d

        kps_l, conf_l, _ = load_pose2d(args.pose2d_left)
        kps_r, conf_r, _ = load_pose2d(args.pose2d_right)
        if kps_l.shape[0] != kps_r.shape[0]:
            print(
                f"Error: pose2d frame counts differ ({kps_l.shape[0]} vs "
                f"{kps_r.shape[0]}) — these are not the same recording.",
                file=sys.stderr,
            )
            return 1
        if args.person >= kps_l.shape[1]:
            print(
                f"Error: --person {args.person} out of range (recording has {kps_l.shape[1]}).",
                file=sys.stderr,
            )
            return 1
        pose2d = (kps_l, conf_l, kps_r, conf_r)

    report = compare_calibrations(
        calibs,
        labels,
        paths=paths,
        pose2d=pose2d,
        person=args.person,
        min_conf=args.min_conf,
        max_reproj_err=args.max_reproj_err,
    )

    print(console_safe(format_report(report), sys.stdout))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "repeat_parameters.csv", report.parameter_rows, labels)
        if report.segment_rows:
            write_csv(out_dir / "repeat_segments.csv", report.segment_rows, labels)
        with open(out_dir / "repeat_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(report_to_dict(report), fh, indent=2)
        print(f"\nWrote results to {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "repeat":
        try:
            return cmd_repeat(args)
        except (OSError, ValueError) as e:
            print(f"\nError: {console_safe(e, sys.stderr)}", file=sys.stderr)
            return 1
    print(f"Unknown command: {args.command}", file=sys.stderr)  # pragma: no cover
    return 1  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
