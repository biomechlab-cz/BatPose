"""CLI entry point: python -m app.calib"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.calib",
        description="Stereo camera calibration from synchronized videos.",
    )
    p.add_argument("--left", required=True, help="Path to left camera video")
    p.add_argument("--right", required=True, help="Path to right camera video")
    p.add_argument(
        "--board",
        choices=["charuco", "checkerboard"],
        default="charuco",
        help="Board type (default: charuco)",
    )
    # ChArUco params
    p.add_argument("--squares-x", type=int, default=7, help="ChArUco squares in X")
    p.add_argument("--squares-y", type=int, default=5, help="ChArUco squares in Y")
    p.add_argument("--square-size", type=float, default=0.04, help="Square size in metres")
    p.add_argument(
        "--marker-size", type=float, default=0.03, help="Marker size in metres (ChArUco only)"
    )
    from .board import ARUCO_DICTS

    p.add_argument(
        "--aruco-dict",
        choices=list(ARUCO_DICTS.keys()),
        default="DICT_4X4_50",
        help="ArUco dictionary (ChArUco only, default: DICT_4X4_50)",
    )
    # Chessboard params (also re-use --squares-x/y and --square-size)
    # Frame selection
    p.add_argument("--max-frames", type=int, default=60, help="Max calibration frames to use")
    p.add_argument("--sample-every", type=int, default=5, help="Sample every N frames")
    p.add_argument(
        "--min-coverage",
        type=float,
        default=0.05,
        help="Minimum board coverage fraction to accept a frame (default: 0.05)",
    )
    p.add_argument("--out", default="calibration.yml", help="Output calibration YAML file")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.board == "charuco":
        board_cfg = {
            "type": "charuco",
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_size": args.square_size,
            "marker_size": args.marker_size,
            "dictionary": args.aruco_dict,
        }
    else:
        board_cfg = {
            "type": "checkerboard",
            "cols": args.squares_x,
            "rows": args.squares_y,
            "square_size": args.square_size,
        }

    def progress(pct: int, msg: str) -> None:
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {pct:3d}%  {msg:<50}", end="", flush=True)

    print(f"Calibrating stereo pair: {args.left!r} + {args.right!r}")
    print(f"Board: {board_cfg}")
    print()

    from .stereo import run_calibration_pipeline

    try:
        out = run_calibration_pipeline(
            args.left,
            args.right,
            board_cfg,
            args.out,
            max_frames=args.max_frames,
            sample_every=args.sample_every,
            min_coverage=args.min_coverage,
            progress_cb=progress,
        )
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        return 1

    if out is None:
        print("\nCancelled.", file=sys.stderr)
        return 1

    # Load and display quality metrics
    from .stereo import load_calibration

    calib = load_calibration(out)
    q = calib.get("quality", {})

    rms = q.get("rms")
    rms_str = f"{rms:.4f}" if rms is not None else "?"
    print(f"\n\nCalibration saved to: {out}")
    print(f"  RMS reprojection error: {rms_str} px")
    print(f"  Frames used:            {q.get('n_frames_used', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
