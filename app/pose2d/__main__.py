"""CLI entry point: python -m app.pose2d"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.console import console_safe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.pose2d",
        description="Extract 2D pose keypoints from stereo exercise videos.",
    )
    p.add_argument("--left", required=True, help="Path to left camera video")
    p.add_argument("--right", required=True, help="Path to right camera video")
    p.add_argument("--outdir", default=".", help="Output directory (default: current dir)")
    p.add_argument(
        "--backend",
        choices=["mediapipe", "rtmpose"],
        default="mediapipe",
        help="Pose backend (default: mediapipe)",
    )
    p.add_argument("--num-poses", type=int, default=2, help="Max persons per frame")
    return p


def _make_backend(args: argparse.Namespace):
    if args.backend == "mediapipe":
        from .mediapipe_backend import MediaPipeBackend

        # IMAGE (stateless) mode for offline extraction — same as the GUI
        # pipeline (Pose2DWorker): each frame is analysed independently, which
        # avoids MediaPipe's internal temporal smoothing blurring fast-movement
        # keypoints; temporal coherence is restored by the OneEuro filter in the
        # 3D step.  VIDEO mode is reserved for live tracking (ADR-006).
        return MediaPipeBackend(num_poses=args.num_poses, running_mode="image")
    elif args.backend == "rtmpose":
        try:
            from .rtmpose_backend import RTMPoseBackend

            return RTMPoseBackend()
        except ImportError:
            print(
                "RTMPose backend requires: pip install rtmlib onnxruntime",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        raise ValueError(f"Unknown backend: {args.backend!r}")


def _progress(pct: int, msg: str) -> None:
    # ASCII-only: Windows consoles are often cp1250 and block characters
    # crash plain print there.
    msg = console_safe(msg, sys.stdout)
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {pct:3d}%  {msg:<50}", end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_left = str(outdir / "pose2d_left.npz")
    out_right = str(outdir / "pose2d_right.npz")

    from .pipeline import process_video, save_pose2d

    print(f"Backend: {args.backend}")

    for label, video_path, out_path in [
        ("LEFT", args.left, out_left),
        ("RIGHT", args.right, out_right),
    ]:
        print(f"\nProcessing {label}: {video_path!r}")
        backend = _make_backend(args)
        try:
            kps, conf, meta = process_video(video_path, backend, progress_cb=_progress)
            save_pose2d(kps, conf, meta, out_path)
            print(f"\n  -> saved {out_path}  shape={kps.shape}  fps={meta['fps']:.1f}")
        except Exception as e:
            print(f"\n\nError: {console_safe(e, sys.stderr)}", file=sys.stderr)
            return 1
        finally:
            backend.close()

    print(f"\nDone. Files written to: {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
