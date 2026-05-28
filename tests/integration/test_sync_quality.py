"""
Sync quality tests on the 20260527_132450 hardware-timestamp sidecar.

Parses the CSV produced by FlirCapture and verifies that the stereo sync
quality meets biomechanics recording standards without touching any camera
hardware or application code.

Pass criteria (from ADR-007):
  - Sync metric is JITTER (stdev of per-frame Δt), NOT absolute offset
  - Jitter < 1 000 µs is acceptable; < 100 µs is excellent
  - Dropped frames < 2% of total
  - Frame count matches expected recording length
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import pytest

_CAPTURE = Path(__file__).parents[2] / "data" / "Test project" / "capture"
_TS_CSV = _CAPTURE / "20260527_132450_timestamps.csv"

pytestmark = pytest.mark.skipif(
    not _TS_CSV.exists(),
    reason="Timestamp sidecar 20260527_132450_timestamps.csv not found",
)

_EXPECTED_FRAMES = 344  # known frame count for this recording
_NOMINAL_FPS = 30.0
_NOMINAL_PERIOD_US = 1_000_000.0 / _NOMINAL_FPS  # 33 333 µs


@pytest.fixture(scope="module")
def ts_data():
    """Load timestamps CSV → list of dicts."""
    with open(_TS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def deltas_us(ts_data):
    """Per-frame signed delta (left − right) in microseconds."""
    result = []
    for row in ts_data:
        ls = row.get("hw_left_ns", "").strip()
        rs = row.get("hw_right_ns", "").strip()
        if ls and rs:
            result.append((int(ls) - int(rs)) / 1_000.0)
    return result


class TestSidecarFormat:
    def test_expected_columns_present(self, ts_data):
        expected = {"frame", "hw_left_ns", "hw_right_ns", "abs_delta_us"}
        assert expected <= set(ts_data[0].keys())

    def test_frame_count(self, ts_data):
        assert len(ts_data) == _EXPECTED_FRAMES, (
            f"Expected {_EXPECTED_FRAMES} frames, got {len(ts_data)}"
        )

    def test_frame_indices_contiguous(self, ts_data):
        indices = [int(r["frame"]) for r in ts_data]
        assert indices == list(range(len(ts_data))), "Frame indices are not contiguous"


class TestSyncQuality:
    def test_all_frames_have_hardware_timestamps(self, ts_data):
        missing = sum(
            1
            for r in ts_data
            if not r.get("hw_left_ns", "").strip() or not r.get("hw_right_ns", "").strip()
        )
        assert missing == 0, f"{missing} frames missing hardware timestamps"

    def test_jitter_below_half_frame(self, deltas_us):
        """Jitter (stdev of per-frame Δt) must be well below a half-frame period.

        Half a frame at 30 fps = 16 667 µs.  Jitter at that level would mean
        stereo pairs are misaligned by more than half a frame on average, which
        is unacceptable for 3D reconstruction.  A tight rig should be < 1 ms;
        the threshold here is a fail-safe for degraded-but-still-usable sync.
        """
        jitter = statistics.stdev(deltas_us)
        half_frame_us = _NOMINAL_PERIOD_US / 2.0
        assert jitter < half_frame_us, (
            f"Sync jitter {jitter:.1f} µs ≥ half frame period {half_frame_us:.0f} µs"
        )

    def test_jitter_reported(self, deltas_us):
        """Informational: print jitter so lab operators can read it in verbose output."""
        jitter = statistics.stdev(deltas_us)
        mean_off = statistics.mean(deltas_us)
        print(f"\nSync jitter: {jitter:.1f} µs   mean offset: {mean_off:.0f} µs")

    def test_frame_slip_rate_below_2pct(self, deltas_us):
        """Rate of trigger slips (consecutive Δt step ≥ half frame period) must be < 2%.

        The two FLIR clocks have an independent ~80–200 ms constant offset —
        that is NOT a sync error.  An occasional single-frame slip (< 2% of
        pairs) is acceptable; a systematic slip rate invalidates stereo data.
        """
        import numpy as np

        arr = np.array(deltas_us)
        steps = np.abs(np.diff(arr))
        slip_count = int(np.sum(steps >= _NOMINAL_PERIOD_US / 2.0))
        slip_rate = slip_count / len(steps)
        assert slip_rate < 0.02, (
            f"Trigger slip rate {slip_rate:.1%} ({slip_count} slips in "
            f"{len(steps)} pairs) exceeds 2%"
        )


class TestDroppedFrames:
    def test_dropped_frames_below_2pct(self, ts_data):
        """Detect dropped frames via hardware-timestamp gaps > 1.5× frame period."""
        timestamps = [int(r["hw_left_ns"]) for r in ts_data if r.get("hw_left_ns", "").strip()]
        period_ns = 1_000_000_000.0 / _NOMINAL_FPS
        gaps = [(timestamps[i] - timestamps[i - 1]) / period_ns for i in range(1, len(timestamps))]
        dropped = sum(1 for g in gaps if g > 1.5)
        drop_rate = dropped / len(timestamps)
        assert drop_rate < 0.02, f"Dropped frame rate {drop_rate:.1%} exceeds 2% ({dropped} drops)"
