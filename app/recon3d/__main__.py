"""CLI entry point: python -m app.recon3d"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.recon3d",
        description="Reconstruct 3D joint positions from stereo pose2d detections.",
    )
    p.add_argument("--calib", required=True, help="Path to calibration.yml")
    p.add_argument(
        "--pose2d",
        required=True,
        help="Directory containing pose2d_left.npz and pose2d_right.npz "
        "(or prefix path, will append _left.npz / _right.npz)",
    )
    p.add_argument("--out", default="pose3d.npz", help="Output NPZ file path")
    p.add_argument("--min-conf", type=float, default=0.3, help="Min 2D confidence (default 0.3)")
    p.add_argument(
        "--max-reproj-err",
        type=float,
        default=20.0,
        help="Max reprojection error in pixels (default 20)",
    )
    p.add_argument("--min-cutoff", type=float, default=0.5, help="OneEuro min_cutoff Hz")
    p.add_argument("--beta", type=float, default=0.05, help="OneEuro beta")
    p.add_argument("--d-cutoff", type=float, default=1.0, help="OneEuro d_cutoff Hz")
    p.add_argument(
        "--export-csv",
        metavar="PATH",
        default=None,
        help="Export 3D joints to CSV with columns: frame,time_s,person,j*_x,j*_y,j*_z,j*_conf",
    )
    return p


def _write_csv(
    csv_path: str, joints3d, conf3d, fps: float, coordinate_frame: str = "opencv"
) -> None:
    """Write joints3d / conf3d arrays to a flat CSV file plus a sidecar metadata JSON.

    CSV columns: frame, time_s, person, j0_x, j0_y, j0_z, j0_conf, j1_x, ...
    Metadata JSON: <csv_stem>_metadata.json with fps, coordinate_units,
    coordinate_frame ("opencv" camera frame vs "world" floor frame), n_frames, n_joints.
    joints3d: [T, P, J, 3]   conf3d: [T, P, J]
    """

    T, P, J, _ = joints3d.shape
    header = ["frame", "time_s", "person"]
    for j in range(J):
        header += [f"j{j}_x", f"j{j}_y", f"j{j}_z", f"j{j}_conf"]

    csv_p = Path(csv_path)
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for t in range(T):
            time_s = t / fps if fps > 0 else float("nan")
            for p in range(P):
                row: list = [t, f"{time_s:.6f}", p]
                for j in range(J):
                    x, y, z = joints3d[t, p, j]
                    c = conf3d[t, p, j]
                    row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{c:.6f}"]
                writer.writerow(row)

    # Write sidecar metadata JSON alongside the CSV
    meta_path = csv_p.with_name(csv_p.stem + "_metadata.json")
    metadata = {
        "fps": fps,
        "coordinate_units": "meters",
        # "opencv" = camera-1 frame (X right, Y down, Z forward);
        # "world"  = floor-board frame (origin on the floor, Z up) — ADR-010.
        "coordinate_frame": coordinate_frame,
        "n_frames": T,
        "n_persons": P,
        "n_joints": J,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))


def _progress(pct: int, msg: str) -> None:
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r[{bar}] {pct:3d}%  {msg:<50}", end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    pose2d_dir = Path(args.pose2d)
    if pose2d_dir.is_dir():
        left_path = str(pose2d_dir / "pose2d_left.npz")
        right_path = str(pose2d_dir / "pose2d_right.npz")
    else:
        # Treat as prefix
        left_path = str(args.pose2d) + "_left.npz"
        right_path = str(args.pose2d) + "_right.npz"

    for path in (left_path, right_path, args.calib):
        if not Path(path).exists():
            print(f"Error: file not found: {path!r}", file=sys.stderr)
            return 1

    print(f"Calibration:  {args.calib}")
    print(f"Pose2D left:  {left_path}")
    print(f"Pose2D right: {right_path}")
    print(f"Output:       {args.out}")
    print()

    from .pipeline import reconstruct3d

    try:
        result = reconstruct3d(
            calib_path=args.calib,
            pose2d_left_path=left_path,
            pose2d_right_path=right_path,
            output_path=args.out,
            min_conf=args.min_conf,
            max_reproj_err=args.max_reproj_err,
            min_cutoff=args.min_cutoff,
            beta=args.beta,
            d_cutoff=args.d_cutoff,
            progress_cb=_progress,
        )
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        return 1

    if result is None:
        print("\nCancelled.", file=sys.stderr)
        return 1

    import numpy as np

    d = np.load(result, allow_pickle=True)
    meta = d["meta"].item()
    joints3d = d["joints3d"]
    conf3d = d["conf3d"]
    fps = float(meta.get("fps", 30.0))
    print(f"\n\nDone.  Output: {result}")
    print(f"  Shape: joints3d={joints3d.shape}  (T×P×J×3)")
    print(f"  FPS:   {fps:.1f}")

    if args.export_csv:
        _write_csv(
            args.export_csv,
            joints3d,
            conf3d,
            fps,
            coordinate_frame=str(meta.get("coordinate_frame", "opencv")),
        )
        print(f"  CSV:   {args.export_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
