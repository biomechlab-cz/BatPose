"""Figures for the calibration-repeatability report (docs/calibration_repeatability.md).

Reads the repeat_metrics.json written by `python -m app.calib.validate repeat --out ...`
and renders three publication-ready panels.  Generic in the number of runs.

Both SVG and PNG are written for every figure: the SVG is what the report links (vector,
small enough to version-control, scales for print), the PNG is a convenience raster.

Usage:
    python scripts/calib_repeatability_figures.py <repeat_metrics.json> <figure_dir>

Requires matplotlib (dev extra); it is not a runtime dependency of the application.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 300
#: Keeps text as text in the SVG rather than converting glyphs to paths — much
#: smaller files, and the labels stay selectable/searchable.
matplotlib.rcParams["svg.fonttype"] = "none"


def _save(fig, out: Path, stem: str, **kw) -> None:
    """Write both the vector and raster form of one figure."""
    fig.savefig(out / f"{stem}.svg", **kw)
    fig.savefig(out / f"{stem}.png", dpi=DPI, **kw)
    plt.close(fig)


def _by_name(rows: list[dict]) -> dict[str, dict]:
    return {r["parameter"]: r for r in rows}


def fig1_parameter_cov(params: dict[str, dict], labels: list[str], out: Path) -> None:
    """Relative spread of the scale-type parameters + absolute rotation spread."""
    scale_keys = [
        "L fx",
        "L fy",
        "L cx",
        "L cy",
        "R fx",
        "R fy",
        "R cx",
        "R cy",
        "|T| baseline",
    ]
    scale_keys = [k for k in scale_keys if k in params]
    cov = [params[k]["cov_pct"] for k in scale_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1]})

    colors = ["#4C72B0" if not k.startswith("|T|") else "#C44E52" for k in scale_keys]
    y = np.arange(len(scale_keys))
    ax1.barh(y, cov, color=colors)
    ax1.set_yticks(y, scale_keys)
    ax1.invert_yaxis()
    ax1.set_xlabel("Coefficient of variation across runs (%)")
    ax1.set_title(f"(a) Parameter repeatability, n = {len(labels)} calibrations")
    ax1.grid(axis="x", alpha=0.3)
    for i, v in enumerate(cov):
        ax1.text(v, i, f" {v:.2f}", va="center", fontsize=8)

    rot_keys = [k for k in ("R total angle", "R rvec x", "R rvec y", "R rvec z") if k in params]
    sd = [params[k]["sd"] for k in rot_keys]
    y2 = np.arange(len(rot_keys))
    ax2.barh(y2, sd, color="#55A868")
    ax2.set_yticks(y2, rot_keys)
    ax2.invert_yaxis()
    ax2.set_xlabel("SD across runs (deg)")
    ax2.set_title("(b) Relative orientation")
    ax2.grid(axis="x", alpha=0.3)
    for i, v in enumerate(sd):
        ax2.text(v, i, f" {v:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    _save(fig, out, "fig1_parameter_repeatability")


def fig2_per_run(params: dict[str, dict], labels: list[str], out: Path) -> None:
    """Run-by-run baseline, relative rotation and reported RMS with mean +/- SD."""
    panels = [
        ("|T| baseline", "Baseline |T| (mm)", "#C44E52"),
        ("R total angle", "Relative rotation (deg)", "#55A868"),
        ("reported RMS", "Reported board RMS (px)", "#8172B2"),
    ]
    panels = [p for p in panels if p[0] in params]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.6))
    if len(panels) == 1:
        axes = [axes]

    x = np.arange(len(labels))
    for ax, (key, ylabel, color) in zip(axes, panels):
        row = params[key]
        vals = [row["values"][lab] for lab in labels]
        m, sd = row["mean"], row["sd"]
        ax.axhspan(m - sd, m + sd, color=color, alpha=0.15, label="mean $\\pm$ SD")
        ax.axhline(m, color=color, ls="--", lw=1)
        ax.plot(x, vals, "o-", color=color, ms=7)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_title(f"SD = {sd:.3g}", fontsize=10)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Per-run calibration parameters", y=1.02)
    fig.tight_layout()
    _save(fig, out, "fig2_per_run_parameters", bbox_inches="tight")


def fig3_segments(segments: dict[str, dict], labels: list[str], out: Path) -> None:
    """Segment lengths from the same recording, triangulated with each calibration."""
    keys = [k for k, r in segments.items() if r["unit"] == "mm" and np.isfinite(r["sd"])]
    if not keys:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2, 1]})

    x = np.arange(len(keys))
    width = 0.8 / len(labels)
    cmap = plt.get_cmap("viridis")
    for i, lab in enumerate(labels):
        vals = [segments[k]["values"][lab] for k in keys]
        ax1.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            label=lab,
            color=cmap(i / max(1, len(labels) - 1)),
        )
    ax1.set_xticks(x, keys, rotation=35, ha="right")
    ax1.set_ylabel("Segment length (mm)")
    ax1.set_title("(a) Same recording triangulated with each calibration")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(axis="y", alpha=0.3)

    sds = [segments[k]["sd"] for k in keys]
    ax2.barh(np.arange(len(keys)), sds, color="#C44E52")
    ax2.set_yticks(np.arange(len(keys)), keys)
    ax2.invert_yaxis()
    ax2.set_xlabel("SD across calibrations (mm)")
    ax2.set_title("(b) Disagreement in mm")
    ax2.grid(axis="x", alpha=0.3)
    for i, v in enumerate(sds):
        ax2.text(v, i, f" {v:.1f}", va="center", fontsize=8)

    fig.tight_layout()
    _save(fig, out, "fig3_segment_agreement")


def main() -> int:
    blob = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    labels = list(blob["runs"].keys())
    params = _by_name(blob["parameters"])
    segments = _by_name(blob.get("segments") or [])

    fig1_parameter_cov(params, labels, out)
    fig2_per_run(params, labels, out)
    if segments:
        fig3_segments(segments, labels, out)

    for p in sorted(list(out.glob("*.svg")) + list(out.glob("*.png"))):
        print(f"wrote {p} ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
