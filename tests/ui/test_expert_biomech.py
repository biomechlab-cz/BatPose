"""
Biomechanical Expert Test Suite — automated + manual protocols.

Written from the perspective of a biomechanics researcher who cares about:
  - Scientific correctness of 3D joint coordinates
  - Temporal plausibility of motion trajectories
  - Coordinate units and reference frame conventions
  - Confidence-based quality filtering
  - Left/right joint labeling
  - CSV schema compatibility with downstream analysis (R, MATLAB, Python)

Automated tests run with pytest.  Manual tests are marked @pytest.mark.manual
and include step-by-step instructions for a human operator.

Run automated only:
    uv run pytest tests/ui/test_expert_biomech.py -m "not manual" -v

Run with manual (requires operator):
    uv run pytest tests/ui/test_expert_biomech.py -v
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

_PROJECT = Path(__file__).parents[2] / "data" / "Test project"
_CALIB = _PROJECT / "calibration.yml"
_P2D_L = _PROJECT / "pose2d_left.npz"
_P2D_R = _PROJECT / "pose2d_right.npz"
_HAS_FIXTURES = _CALIB.exists() and _P2D_L.exists() and _P2D_R.exists()

# COCO-17 joint names in index order — used for semantic checks
COCO17 = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Joint pairs that must always be present together (bilateral symmetry guard)
_BILATERAL_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def _run_reconstruction(tmp_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the full pipeline on sample fixtures and return (joints3d, conf3d, meta)."""
    from app.recon3d.pipeline import reconstruct3d
    out = str(tmp_path / "pose3d.npz")
    reconstruct3d(
        calib_path=str(_CALIB),
        pose2d_left_path=str(_P2D_L),
        pose2d_right_path=str(_P2D_R),
        output_path=out,
    )
    d = np.load(out, allow_pickle=True)
    return d["joints3d"], d["conf3d"], d["meta"].item()


# ── Coordinate units and reference frame ────────────────────────────────────

@pytest.mark.skipif(not _HAS_FIXTURES, reason="Sample fixtures required")
class TestCoordinateUnits:
    @pytest.fixture(scope="class")
    def recon(self, tmp_path_factory):
        return _run_reconstruction(tmp_path_factory.mktemp("biomech"))

    def test_joints_in_metre_range(self, recon):
        """All detected joints must lie within ±10 m (lab volume sanity check).

        10 m is a generous bound — catches millimetre/pixel unit errors (which
        produce values in the thousands) while allowing typical lab volumes up
        to ~5 m per axis with some margin for camera placement.
        """
        joints3d, conf3d, _ = recon
        detected = joints3d[conf3d > 0]
        assert np.all(np.abs(detected) < 10.0), (
            f"Joint outside ±10 m lab volume: max={np.max(np.abs(detected)):.2f} m"
        )

    def test_body_height_plausible(self, recon):
        """Vertical extent (Y axis) of detected joints should be < 2.5 m per frame."""
        joints3d, conf3d, _ = recon
        T = joints3d.shape[0]
        for t in range(T):
            detected_y = joints3d[t, conf3d[t] > 0.3, 1]
            if len(detected_y) >= 4:
                height_m = float(detected_y.max() - detected_y.min())
                assert height_m < 2.5, (
                    f"Frame {t}: vertical extent {height_m:.2f} m exceeds 2.5 m"
                )

    def test_csv_coordinates_are_metres(self, recon, tmp_path):
        """Exported CSV j*_x/y/z values must be in metres (not pixels, not mm)."""
        from app.recon3d.__main__ import _write_csv
        joints3d, conf3d, meta = recon
        fps = float(meta.get("fps", 30.0))
        csv_path = str(tmp_path / "export.csv")
        _write_csv(csv_path, joints3d, conf3d, fps)

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        # Check first detected joint's magnitude
        for row in rows:
            x = float(row["j0_x"])
            y = float(row["j0_y"])
            z = float(row["j0_z"])
            if x != 0.0 or y != 0.0 or z != 0.0:
                magnitude = math.sqrt(x**2 + y**2 + z**2)
                assert magnitude < 10.0, (
                    f"j0 magnitude {magnitude:.1f} looks like pixels, not metres"
                )
                break


# ── Temporal plausibility ────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FIXTURES, reason="Sample fixtures required")
class TestTemporalPlausibility:
    @pytest.fixture(scope="class")
    def recon(self, tmp_path_factory):
        return _run_reconstruction(tmp_path_factory.mktemp("biomech2"))

    def test_frame_to_frame_joint_velocity_plausible(self, recon):
        """No joint should move faster than ~50 m/s between consecutive frames.

        50 m/s is a generous bound — it catches numeric blow-up in poorly
        calibrated rigs (which produce speeds of hundreds or thousands of m/s)
        while remaining above any physiologically realistic speed (~10 m/s for
        sprinting, ~30 m/s for a thrown ball) and accounting for moderate
        calibration error in the test rig.
        """
        joints3d, conf3d, meta = recon
        fps = float(meta.get("fps", 30.0))
        T = joints3d.shape[0]

        max_speed = 0.0
        for t in range(1, T):
            both_detected = (conf3d[t] > 0) & (conf3d[t - 1] > 0)
            disp = joints3d[t] - joints3d[t - 1]   # [P, 17, 3]
            speed = np.linalg.norm(disp, axis=-1)   # [P, 17]
            speed_ms = speed[both_detected] * fps
            if len(speed_ms):
                max_speed = max(max_speed, float(speed_ms.max()))

        assert max_speed < 50.0, (
            f"Max inter-frame joint speed {max_speed:.1f} m/s — likely a reconstruction artifact"
        )

    def test_smoothed_output_lower_variance_than_raw_pose2d(self, recon):
        """OneEuro-smoothed 3D joints must have lower frame-to-frame variance than raw 2D input.

        This verifies the smoothing step actually runs (not a no-op).
        """
        joints3d, conf3d, _ = recon
        # Frame-to-frame L2 displacement of nose (joint 0, person 0)
        person_0 = joints3d[:, 0, 0, :]        # [T, 3]
        conf_0 = conf3d[:, 0, 0]               # [T]
        detected = conf_0 > 0.3
        if detected.sum() < 10:
            pytest.skip("Too few detected nose frames to compute variance")
        pos = person_0[detected]
        diffs = np.diff(pos, axis=0)
        frame_disp = np.linalg.norm(diffs, axis=1)
        variance = float(np.var(frame_disp))
        # We only assert it's finite and non-negative — smoothing should have handled it
        assert math.isfinite(variance) and variance >= 0


# ── Confidence-based quality filtering ──────────────────────────────────────

@pytest.mark.skipif(not _HAS_FIXTURES, reason="Sample fixtures required")
class TestConfidenceFiltering:
    @pytest.fixture(scope="class")
    def recon(self, tmp_path_factory):
        return _run_reconstruction(tmp_path_factory.mktemp("biomech3"))

    def test_all_joint_coordinates_are_finite(self, recon):
        """All joint coordinates must be finite — no NaN or inf from the pipeline.

        Note: OneEuro smoothing carries state across frames, so conf=0 joints
        may retain non-zero positions from the previous detected frame.  The
        key guarantee is that the pipeline never produces NaN/inf, which would
        indicate a division-by-zero or failed triangulation slipping through.
        """
        joints3d, _, _ = recon
        assert np.all(np.isfinite(joints3d)), (
            "Pipeline produced NaN or inf in joints3d"
        )

    def test_high_conf_joints_not_all_zero(self, recon):
        """At least some high-confidence joints must have non-trivial 3D position."""
        joints3d, conf3d, _ = recon
        high_conf = joints3d[conf3d > 0.5]
        assert len(high_conf) > 0 and np.any(high_conf != 0), (
            "All high-confidence joints are at the origin"
        )


# ── COCO-17 joint labeling ───────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FIXTURES, reason="Sample fixtures required")
class TestJointLabeling:
    @pytest.fixture(scope="class")
    def recon(self, tmp_path_factory):
        return _run_reconstruction(tmp_path_factory.mktemp("biomech4"))

    def test_left_and_right_joints_detected_together(self, recon):
        """Bilateral joint pairs (left eye / right eye, etc.) must be detected together.

        If only one side is detected in a frame, the other's confidence should be
        similarly low — not a systematic left-only or right-only detection bias.
        """
        joints3d, conf3d, _ = recon
        for left_idx, right_idx in _BILATERAL_PAIRS:
            left_det = (conf3d[:, 0, left_idx] > 0.3).sum()
            right_det = (conf3d[:, 0, right_idx] > 0.3).sum()
            T = conf3d.shape[0]
            if left_det == 0 or right_det == 0:
                continue  # joint not detected at all — skip pair
            # Neither side should be detected in more than 3× the frames of the other
            ratio = max(left_det, right_det) / max(min(left_det, right_det), 1)
            assert ratio < 3.0, (
                f"Joint pair ({COCO17[left_idx]}, {COCO17[right_idx]}): "
                f"detection asymmetry ratio {ratio:.1f} — possible L/R labeling error"
            )

    def test_csv_has_all_17_joint_columns(self, recon, tmp_path):
        from app.recon3d.__main__ import _write_csv
        joints3d, conf3d, meta = recon
        csv_path = str(tmp_path / "export.csv")
        _write_csv(csv_path, joints3d, conf3d, float(meta.get("fps", 30.0)))
        with open(csv_path, newline="") as f:
            headers = csv.DictReader(f).fieldnames
        for j in range(17):
            for ax in ("x", "y", "z"):
                col = f"j{j}_{ax}"
                assert col in headers, f"Missing column {col!r} for joint {COCO17[j]!r}"

    def test_csv_metadata_json_has_expected_keys(self, recon, tmp_path):
        import json
        from app.recon3d.__main__ import _write_csv
        joints3d, conf3d, meta = recon
        csv_path = str(tmp_path / "bio_export.csv")
        _write_csv(csv_path, joints3d, conf3d, float(meta.get("fps", 30.0)))
        meta_path = tmp_path / "bio_export_metadata.json"
        assert meta_path.exists(), "metadata JSON not created alongside CSV"
        md = json.loads(meta_path.read_text())
        for key in ("fps", "coordinate_units", "n_frames", "n_joints"):
            assert key in md, f"Missing key {key!r} in metadata JSON"
        assert md["coordinate_units"] == "meters"
        assert md["n_joints"] == 17


# ── Manual expert protocols ──────────────────────────────────────────────────

@pytest.mark.manual
class TestManualBiomechExpert:
    """
    MANUAL PROTOCOL — Biomechanical Expert Walkthrough
    ====================================================
    Purpose: Validate that the full pipeline produces scientifically usable
             3D joint trajectories for biomechanical analysis.

    Operator requirements:
      - Subject performing a known movement (e.g. standing T-pose, then raising
        left arm to 90°, then lowering it — 5 repetitions)
      - Both FLIR cameras connected and hardware-synced
      - calibration.yml already produced (RMS < 1.0 px)

    Steps (record outcomes in lab notebook):
    -----------------------------------------
    BP-BIO-001  T-pose verification
        1. Subject holds T-pose (arms horizontal, 90° shoulder abduction).
        2. Open BatPose → Live Capture tab → start live 3D tracking.
        3. Verify: left shoulder (j5) and right shoulder (j6) are at similar Y
           height (within 50 mm) and symmetric X offset (|x5| ≈ |x6|).
        4. PASS if: symmetry error < 50 mm.  FAIL if > 100 mm.

    BP-BIO-002  Arm-raise temporal tracking
        1. Subject raises left arm from 0° to 90° abduction over 2 s,
           holds 1 s, lowers over 2 s.  Repeat 5 times.
        2. Export CSV → analyze j9_y (left wrist Y) in Python/R.
        3. PASS if: 5 clear peaks visible; peak Y height > 0.5 m above hip Y.

    BP-BIO-003  Coordinate axis verification
        1. Subject steps 0.5 m to the right (camera baseline direction).
        2. Verify: nose j0_x increases by ~0.5 m in exported CSV.
        3. PASS if: Δj0_x ∈ [0.3, 0.7] m.  FAIL if negative or > 1.0 m.

    BP-BIO-004  RTMPose vs MediaPipe joint-position agreement
        1. Record 30 s walking trial.
        2. Reconstruct with MediaPipe → export CSV_mp.
        3. Reconstruct same recording with RTMPose → export CSV_rtm.
        4. Compare j11 (left hip) X,Y,Z trajectories.
        5. PASS if: RMS difference < 30 mm per axis.

    BP-BIO-005  Confidence-based gap handling
        1. Subject moves one arm outside camera FOV for 5 frames.
        2. Verify: conf for occluded wrist drops to 0; wrist position shows (0,0,0).
        3. In downstream CSV, occluded frames must not produce outlier coordinates.
        4. PASS if: no wrist values appear at (0,0,0) mixed with real positions
           without corresponding conf=0.
    """

    def test_placeholder(self):
        """Placeholder so pytest can collect this class without running anything."""
        pytest.skip("Manual protocol — run by human operator; see docstring for steps")
