"""
Verify a recorded stereo pair: equal frame counts and (if available) sync quality.

Usage:
    python -m app.capture.verify_recording LEFT.avi RIGHT.avi [--max-jitter-us 1000]

Frame-count check works on any pair.  Sync verification additionally requires
the per-frame timestamp sidecar written during recording — a CSV named
'<prefix>_timestamps.csv' next to the videos (columns: frame, hw_left_ns,
hw_right_ns, abs_delta_us).  Older recordings made before timestamp logging
have no sidecar, so only the frame count can be verified for those.

Sync metric — JITTER, not absolute delta:
    The two cameras have INDEPENDENT hardware clocks, each relative to its own
    power-on.  So the absolute left-vs-right timestamp difference is dominated
    by a large CONSTANT offset (the clock-start difference) that has nothing to
    do with sync.  Hardware-triggered sync quality is measured by the *jitter*
    (standard deviation) of the per-frame delta, which cancels that constant
    offset — the same metric the live "Sync Δt" indicator uses.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys


def _frame_count(path: str) -> int:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _timestamps_path(left: str, right: str) -> str:
    base = os.path.commonprefix([left, right]).rstrip("_")
    return f"{base}_timestamps.csv"


def _rate_jitter(jitter_us: float) -> str:
    """Same rating bands as the live Sync Δt indicator."""
    if jitter_us < 50:
        return "✓ Excellent"
    if jitter_us < 200:
        return "✓ Good"
    if jitter_us < 1000:
        return "⚠ Marginal"
    return "✗ Poor — check trigger wiring"


def _check_drops(label: str, ts_ns: list[int]) -> int:
    """Flag dropped frames within one camera stream via timestamp gaps.

    A frame was dropped before recording if the gap between consecutive
    recorded timestamps exceeds 1.5× the median inter-frame interval.
    Returns the estimated number of dropped frames.
    """
    if len(ts_ns) < 3:
        return 0
    gaps = [b - a for a, b in zip(ts_ns, ts_ns[1:]) if b > a]
    if not gaps:
        return 0
    period = statistics.median(gaps)
    dropped = sum(max(0, round(g / period) - 1) for g in gaps if g > 1.5 * period)
    if dropped:
        print(
            f"              ⚠ {label}: ~{dropped} dropped frame(s) "
            f"(timestamp gaps > 1.5× the {period / 1e6:.1f} ms frame interval)"
        )
    return dropped


def verify(left: str, right: str, max_jitter_us: float = 1000.0) -> bool:
    """Print a sync/frame-count report. Return True if the pair passes."""
    ok = True

    # ── Frame counts ────────────────────────────────────────────────────────
    nl, nr = _frame_count(left), _frame_count(right)
    print(f"Frame count:  left={nl}  right={nr}", end="  ")
    if nl == nr:
        print("✓ equal")
    else:
        print(f"✗ MISMATCH (differ by {abs(nl - nr)})")
        ok = False

    # ── Sync (needs the timestamp sidecar) ──────────────────────────────────
    ts_path = _timestamps_path(left, right)
    if not os.path.exists(ts_path):
        print(f"Sync:         no timestamp sidecar found ({os.path.basename(ts_path)}).")
        print("              Frame count verified; sync cannot be checked for this")
        print("              recording (made before per-frame timestamp logging).")
        return ok

    lefts: list[int] = []
    rights: list[int] = []
    missing = 0
    with open(ts_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ls = (row.get("hw_left_ns") or "").strip()
            rs = (row.get("hw_right_ns") or "").strip()
            if ls == "" or rs == "":
                missing += 1
                continue
            lefts.append(int(ls))
            rights.append(int(rs))

    if len(lefts) < 2:
        print(
            f"Sync:         sidecar present but has too few hardware timestamps "
            f"({missing} rows without them)."
        )
        return ok

    # Signed per-frame delta (left − right), in microseconds.  Jitter (stdev)
    # is the offset-invariant sync metric; the mean is just the constant
    # difference between the two cameras' independent clocks.
    deltas = [(lt - rt) / 1000.0 for lt, rt in zip(lefts, rights)]
    n = len(deltas)
    mean_off = statistics.mean(deltas)
    jitter = statistics.stdev(deltas)
    rating = _rate_jitter(jitter)

    print(f"Sync (hw Δt): {n} frame-pairs")
    print(f"              jitter (stdev) = {jitter:.1f}µs   {rating}")
    print(
        f"              constant clock offset = {mean_off / 1000:.1f} ms "
        f"(independent camera clocks — not a sync error)"
    )
    if missing:
        print(f"              ({missing} frame-pair(s) had a dropped/placeholder view)")

    # Per-camera dropped-frame detection.
    _check_drops("left", lefts)
    _check_drops("right", rights)

    if jitter > max_jitter_us:
        ok = False

    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify stereo recording sync + frame counts.")
    ap.add_argument("left", help="left .avi path")
    ap.add_argument("right", help="right .avi path")
    ap.add_argument(
        "--max-jitter-us",
        type=float,
        default=1000.0,
        help="max acceptable sync jitter (stdev of per-frame Δt) in microseconds (default 1000)",
    )
    args = ap.parse_args(argv)

    passed = verify(args.left, args.right, args.max_jitter_us)
    print("\nRESULT:", "PASS ✓" if passed else "FAIL ✗")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
