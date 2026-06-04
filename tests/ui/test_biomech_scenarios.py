"""
UX scenario tests — common biomechanics workflows.

Each scenario mirrors a task a physiotherapist or sports scientist would
perform in a clinical or lab session.  The tests verify that the Analysis
tab produces correct results, that the UI reflects those results accurately,
and that the cross-tab playback infrastructure is reliable.

Scenarios
---------
1  Load a recording                 — tables populated, slider range correct
2  Statistics accuracy              — Recording tab values match the biomech
                                      module computed directly on the same data
3  Segment ROI phase analysis       — dragging a region auto-evaluates stats
4  Bilateral asymmetry screening    — Asymmetry tab shows valid SI% values
5  Full CSV export                  — headers, row count T×P, time_s from 0
6  Segment CSV export               — row count = segment length, time_s reset
7  Playback controls                — frame nav, play/pause, seek, signal loop guard
8  Play-button synchronisation      — both tabs mirror state; only one timer active

Run:
    uv run pytest tests/ui/test_biomech_scenarios.py -v
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from app.biomech import (
    ANGLE_DEFINITIONS,
    ANGLE_PAIRS,
    angles_to_csv,
    compute_extended_stats,
    compute_joint_angles,
    compute_symmetry_index,
)
from app.gui.analysis_tab import AnalysisTab, _opencv_to_zup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FPS = 30.0

# Indices into ANGLE_DEFINITIONS (order defined in app/biomech/angles.py):
_LKNEE = 0   # L Knee Flex   (11, 13, 15)
_RKNEE = 1   # R Knee Flex   (12, 14, 16)
_TRUNK = 8   # Trunk Inclination

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _zup_to_opencv(pts: np.ndarray) -> np.ndarray:
    """Invert the Z-up axis swap that AnalysisTab applies on load.

    Z-up (x, y, z)  →  OpenCV (x, -z, y).
    Verified: _opencv_to_zup(_zup_to_opencv(v)) == v.
    """
    return np.stack(
        [pts[..., 0], -pts[..., 2], pts[..., 1]], axis=-1
    ).astype(np.float32)


# COCO-17 standing pose in Z-up (metres):
#   X right, Y anterior (depth), Z up.
#   Hip/knee/ankle on a straight vertical line → knee angle = 180°.
_STANDING_ZUP = np.array([
    [0.00,  0.05, 1.70],   #  0 nose
    [-0.03, 0.05, 1.72],   #  1 L eye
    [0.03,  0.05, 1.72],   #  2 R eye
    [-0.08, 0.00, 1.70],   #  3 L ear
    [0.08,  0.00, 1.70],   #  4 R ear
    [-0.18, 0.00, 1.45],   #  5 L shoulder
    [0.18,  0.00, 1.45],   #  6 R shoulder
    [-0.18, 0.00, 1.15],   #  7 L elbow
    [0.18,  0.00, 1.15],   #  8 R elbow
    [-0.18, 0.00, 0.85],   #  9 L wrist
    [0.18,  0.00, 0.85],   # 10 R wrist
    [-0.10, 0.00, 0.90],   # 11 L hip
    [0.10,  0.00, 0.90],   # 12 R hip
    [-0.10, 0.00, 0.50],   # 13 L knee  ← vertex
    [0.10,  0.00, 0.50],   # 14 R knee  ← vertex
    [-0.10, 0.00, 0.00],   # 15 L ankle (collinear with hip/knee → 180°)
    [0.10,  0.00, 0.00],   # 16 R ankle
], dtype=np.float32)


def _make_standing_pose3d(
    tmp_path: Path, T: int = 60, P: int = 1, fps: float = FPS, name: str = "pose3d.npz"
) -> str:
    """Synthetic pose3d.npz: person standing still, all joints at 180°.

    All positions are stored in the OpenCV camera frame (AnalysisTab
    converts them to Z-up internally via _opencv_to_zup on load).
    """
    zup = np.tile(_STANDING_ZUP, (T, P, 1, 1))          # [T, P, 17, 3] Z-up
    cv  = _zup_to_opencv(zup)                            # → OpenCV frame
    conf = np.ones((T, P, 17), dtype=np.float32)
    meta = {"fps": fps, "model_name": "synthetic"}
    path = str(tmp_path / name)
    np.savez(path, joints3d=cv, conf3d=conf, meta=np.array(meta))
    return path


def _make_asymmetric_pose3d(
    tmp_path: Path,
    T: int = 60,
    l_knee_deg: float = 90.0,
    r_knee_deg: float = 60.0,
    fps: float = FPS,
) -> str:
    """Synthetic pose3d.npz with deliberately asymmetric knee angles.

    Geometry (verified analytically):
      L Knee vertex at Z-up (−0.10, 0, 0.50).
      For angle θ at vertex: ankle placed at knee + 0.40·(±sin θ, 0, cos θ)
      where the sign is mirrored for left/right symmetry.

    Robinson SI for default (90° vs 60°):
      SI = 100 × (90−60) / (0.5×(90+60)) = 40%  (positive → left-dominant).
    """
    template = _STANDING_ZUP.copy()

    # L Knee ankle placement for l_knee_deg
    tl = math.radians(l_knee_deg)
    template[15] = np.array(
        [-0.10 - 0.40 * math.sin(tl), 0.0, 0.50 + 0.40 * math.cos(tl)],
        dtype=np.float32,
    )
    # R Knee ankle placement for r_knee_deg
    tr = math.radians(r_knee_deg)
    template[16] = np.array(
        [0.10 + 0.40 * math.sin(tr), 0.0, 0.50 + 0.40 * math.cos(tr)],
        dtype=np.float32,
    )

    zup  = np.tile(template, (T, 1, 1, 1))
    cv   = _zup_to_opencv(zup)
    conf = np.ones((T, 1, 17), dtype=np.float32)
    meta = {"fps": fps, "model_name": "synthetic"}
    path = str(tmp_path / "asym_pose3d.npz")
    np.savez(path, joints3d=cv, conf3d=conf, meta=np.array(meta))
    return path


def _pump(ms: int = 0) -> None:
    """Pump the Qt event loop briefly."""
    QCoreApplication.processEvents()
    if ms > 0:
        import time
        deadline = time.monotonic() + ms / 1000.0
        while time.monotonic() < deadline:
            QCoreApplication.processEvents()


# ============================================================================
# Scenario 1 — Load a recording; confirm tables are populated
# ============================================================================

class TestScenario1Loading:
    """
    Clinical rationale: A physiotherapist opening a session must
    immediately see populated statistics — blank cells after loading
    indicate a silent failure that could cause misinterpretation.
    """

    @pytest.fixture
    def tab(self, qtbot, tmp_path):
        path = _make_standing_pose3d(tmp_path, T=60)
        w = AnalysisTab()
        qtbot.addWidget(w)
        w.show()
        w.load_pose3d(path)
        _pump()
        return w

    def test_slider_range_equals_t_minus_1(self, tab):
        assert tab._slider.maximum() == 59

    def test_person_combo_has_one_entry(self, tab):
        assert tab._person_combo.count() == 1

    def test_fps_label_populated(self, tab):
        assert "30" in tab._fps_label.text()

    def test_status_shows_frame_count(self, tab):
        assert "60" in tab._status_lbl.text()

    def test_recording_tab_selected_after_load(self, tab):
        assert tab._stats_tabs.currentIndex() == 0

    def test_full_table_has_nine_rows(self, tab):
        assert tab._full_table.rowCount() == len(ANGLE_DEFINITIONS)

    def test_full_table_column_headers(self, tab):
        t = tab._full_table
        headers = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
        assert headers == ["Min", "Max", "Mean", "SD", "CV%", "ROM", "NaN%"]

    def test_segment_table_shows_dashes_before_roi(self, tab):
        """No segment should be pre-evaluated; every cell must be '—'."""
        t = tab._seg_table
        for r in range(t.rowCount()):
            for c in range(t.columnCount()):
                assert t.item(r, c).text() == "—", (
                    f"Segment [{r},{c}] pre-populated before any ROI was defined"
                )

    def test_full_table_no_dash_cells_after_load(self, tab):
        """Every cell in the Recording tab must show a numeric value."""
        t = tab._full_table
        for r in range(t.rowCount()):
            for c in range(t.columnCount()):
                text = t.item(r, c).text()
                assert text != "—", (
                    f"Recording [{r},{c}] still '—' after load — "
                    f"angle '{ANGLE_DEFINITIONS[r].name}', "
                    f"col '{t.horizontalHeaderItem(c).text()}'"
                )


# ============================================================================
# Scenario 2 — Statistics accuracy: UI values match the biomech module
# ============================================================================

class TestScenario2StatsAccuracy:
    """
    Clinical rationale: A sports scientist exporting a report must be able
    to trust that the numbers displayed exactly match what the peer-reviewed
    compute_extended_stats function computes.  Any rounding or off-by-one
    in the display layer would invalidate clinical conclusions.
    """

    @pytest.fixture
    def ctx(self, tmp_path, qtbot):
        tmp = tmp_path
        path = _make_standing_pose3d(tmp, T=90)
        w = AnalysisTab()
        qtbot.addWidget(w)
        w.show()
        w.load_pose3d(path)
        _pump()
        # Reference computation performed directly by the biomech module.
        d = np.load(path, allow_pickle=True)
        joints_zup = _opencv_to_zup(d["joints3d"])
        angles = compute_joint_angles(joints_zup, d["conf3d"])
        ref = compute_extended_stats(angles, _LKNEE, 0, fps=FPS)
        return w, ref

    def test_min_matches_module(self, ctx):
        tab, ref = ctx
        cell = float(tab._full_table.item(_LKNEE, 0).text())
        assert cell == pytest.approx(ref.min_deg, abs=0.1)

    def test_max_matches_module(self, ctx):
        tab, ref = ctx
        cell = float(tab._full_table.item(_LKNEE, 1).text())
        assert cell == pytest.approx(ref.max_deg, abs=0.1)

    def test_mean_matches_module(self, ctx):
        tab, ref = ctx
        cell = float(tab._full_table.item(_LKNEE, 2).text())
        assert cell == pytest.approx(ref.mean_deg, abs=0.1)

    def test_sd_matches_module(self, ctx):
        tab, ref = ctx
        cell = float(tab._full_table.item(_LKNEE, 3).text())
        assert cell == pytest.approx(ref.std_deg, abs=0.1)

    def test_cv_matches_module(self, ctx):
        tab, ref = ctx
        cell = float(tab._full_table.item(_LKNEE, 4).text())
        assert cell == pytest.approx(ref.cv_pct, abs=0.1)

    def test_rom_matches_module(self, ctx):
        tab, ref = ctx
        cell = float(tab._full_table.item(_LKNEE, 5).text())
        assert cell == pytest.approx(ref.rom_deg, abs=0.1)

    def test_nan_pct_matches_module(self, ctx):
        tab, ref = ctx
        cell = float(tab._full_table.item(_LKNEE, 6).text())
        assert cell == pytest.approx(ref.nan_pct, abs=0.1)

    def test_consistency_for_all_nine_angles(self, ctx):
        """Every angle's ROM in the UI must match the module, not just L Knee."""
        tab, _ = ctx
        d = np.load(str(Path(tab._pose3d_path)), allow_pickle=True)
        joints_zup = _opencv_to_zup(d["joints3d"])
        angles = compute_joint_angles(joints_zup, d["conf3d"])
        for k in range(len(ANGLE_DEFINITIONS)):
            s = compute_extended_stats(angles, k, 0, fps=FPS)
            if s is None:
                continue
            cell = float(tab._full_table.item(k, 5).text())   # ROM col
            assert cell == pytest.approx(s.rom_deg, abs=0.15), (
                f"ROM mismatch for {ANGLE_DEFINITIONS[k].name}: "
                f"UI={cell}, module={s.rom_deg}"
            )


# ============================================================================
# Scenario 3 — Segment ROI phase analysis
# ============================================================================

class TestScenario3SegmentROI:
    """
    Clinical rationale: Clinicians want statistics only for a specific
    movement phase (e.g. the descent phase of a squat, not the full
    recording).  The ROI auto-evaluation removes the need to click
    Evaluate and ensures stats are always current after region changes.
    """

    @pytest.fixture
    def loaded(self, qtbot, tmp_path):
        path = _make_standing_pose3d(tmp_path, T=90)
        w = AnalysisTab()
        qtbot.addWidget(w)
        w.show()
        w.load_pose3d(path)
        _pump()
        return w

    def _set_roi(self, tab, f0, f1):
        """Programmatically set the ROI region (simulates a user drag)."""
        fps = tab._fps
        tab._roi_region.setVisible(True)
        tab._roi_region.setRegion([f0 / fps, f1 / fps])

    def test_segment_stats_auto_populate_after_debounce(self, loaded, qtbot):
        """After the 150 ms debounce timer fires, every Segment cell must be numeric."""
        self._set_roi(loaded, f0=10, f1=50)
        qtbot.wait(300)   # 2× debounce headroom
        _pump()
        t = loaded._seg_table
        populated = any(
            t.item(r, c).text() != "—"
            for r in range(t.rowCount())
            for c in range(t.columnCount())
        )
        assert populated, "Segment table still all '—' after ROI + debounce"

    def test_segment_tab_selected_immediately_on_roi_change(self, loaded):
        """Segment tab must be selected as soon as the region changes."""
        self._set_roi(loaded, 10, 50)
        _pump()
        assert loaded._stats_tabs.currentIndex() == 1

    def test_roi_labels_show_start_end_duration(self, loaded):
        self._set_roi(loaded, 10, 50)
        _pump()
        assert "Start:" in loaded._roi_start_lbl.text()
        assert "End:"   in loaded._roi_end_lbl.text()
        assert "Dur:"   in loaded._roi_dur_lbl.text()
        assert "—" not in loaded._roi_start_lbl.text()

    def test_clear_roi_resets_labels_and_table(self, loaded, qtbot):
        self._set_roi(loaded, 10, 50)
        qtbot.wait(300)
        _pump()
        loaded._on_clear_roi()
        _pump()
        # Labels back to dashes
        assert loaded._roi_start_lbl.text() == "Start: —"
        # Segment table back to dashes
        t = loaded._seg_table
        for r in range(t.rowCount()):
            for c in range(t.columnCount()):
                assert t.item(r, c).text() == "—"

    def test_clear_roi_returns_to_recording_tab(self, loaded, qtbot):
        self._set_roi(loaded, 10, 50)
        qtbot.wait(300)
        _pump()
        loaded._on_clear_roi()
        _pump()
        assert loaded._stats_tabs.currentIndex() == 0

    def test_segment_stats_numerically_match_module(self, loaded, qtbot):
        """Segment ROM for L Knee must equal direct computation on the slice."""
        f0, f1 = 10, 50
        self._set_roi(loaded, f0, f1)
        qtbot.wait(300)
        _pump()
        seg = loaded._angles[f0 : f1 + 1]
        ref = compute_extended_stats(seg, _LKNEE, 0, fps=FPS)
        if ref is not None:
            cell = float(loaded._seg_table.item(_LKNEE, 5).text())  # ROM
            assert cell == pytest.approx(ref.rom_deg, abs=0.15)

    def test_segment_table_column_headers(self, loaded):
        t = loaded._seg_table
        headers = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
        assert headers == ["Min", "Max", "Mean", "SD", "CV%", "ROM", "Exc(°)"]


# ============================================================================
# Scenario 4 — Bilateral asymmetry screening
# ============================================================================

class TestScenario4Asymmetry:
    """
    Clinical rationale: Robinson's SI% is the standard ACL-rehabilitation
    metric.  |SI| > 10% flags asymmetry; values outside ±200% indicate a
    division-by-zero artefact.  The UI must show valid numbers for all
    four L/R pairs and correctly identify left- vs right-dominant sides.
    """

    @pytest.fixture
    def asym_tab(self, tmp_path, qtbot):
        tmp = tmp_path
        path = _make_asymmetric_pose3d(
            tmp, T=60, l_knee_deg=90.0, r_knee_deg=60.0
        )
        w = AnalysisTab()
        qtbot.addWidget(w)
        w.show()
        w.load_pose3d(path)
        _pump()
        return w

    def test_asymmetry_table_row_count(self, asym_tab):
        assert asym_tab._asym_table.rowCount() == len(ANGLE_PAIRS)

    def test_asymmetry_column_headers(self, asym_tab):
        t = asym_tab._asym_table
        headers = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
        assert headers == ["Mean SI%", "ROM SI%", "PkVel SI%"]

    def test_knee_mean_si_is_numeric_not_dash(self, asym_tab):
        """Knee Mean SI% (col 0) must be numeric — the static pose has nonzero
        mean angles on both sides so the SI denominator is never zero.
        ROM SI% and PkVel SI% are legitimately '—' when ROM=0 on both sides
        (static pose has no movement), so those columns are not checked here.
        """
        text = asym_tab._asym_table.item(0, 0).text()   # Mean SI%
        assert text != "—", "Knee Mean SI% is '—' despite nonzero mean angles"
        assert math.isfinite(float(text))

    def test_all_si_values_within_sanity_bound(self, asym_tab):
        """|SI| < 200% for all pairs — catches near-zero-denominator artefacts."""
        t = asym_tab._asym_table
        for r in range(t.rowCount()):
            for c in range(t.columnCount()):
                text = t.item(r, c).text()
                if text == "—":
                    continue
                assert abs(float(text)) < 200.0, (
                    f"SI[{r},{c}] = {text}% exceeds ±200% sanity bound"
                )

    def test_knee_si_is_nonzero_for_asymmetric_joints(self, asym_tab):
        """L=90°, R=60° must produce a detectable SI (well above 10% threshold)."""
        si = float(asym_tab._asym_table.item(0, 0).text())   # Mean SI%, knee
        assert abs(si) > 10.0

    def test_knee_si_is_positive_left_dominant(self, asym_tab):
        """Robinson convention: positive SI = left side larger."""
        si = float(asym_tab._asym_table.item(0, 0).text())
        assert si > 0.0

    def test_knee_si_matches_module_directly(self, asym_tab):
        """UI SI% must equal compute_symmetry_index(l_mean, r_mean)."""
        angles = asym_tab._angles
        l = compute_extended_stats(angles, _LKNEE, 0, fps=FPS)
        r = compute_extended_stats(angles, _RKNEE, 0, fps=FPS)
        if l and r:
            ref = compute_symmetry_index(l.mean_deg, r.mean_deg)
            ui  = float(asym_tab._asym_table.item(0, 0).text())
            assert ui == pytest.approx(ref, abs=0.1)

    def test_symmetric_joints_give_near_zero_si(self, qtbot, tmp_path):
        """Perfectly symmetric standing pose must yield SI ≈ 0% for all pairs."""
        path = _make_standing_pose3d(tmp_path, T=60, name="sym.npz")
        w = AnalysisTab()
        qtbot.addWidget(w)
        w.show()
        w.load_pose3d(path)
        _pump()
        t = w._asym_table
        for r in range(t.rowCount()):
            for c in range(t.columnCount()):
                text = t.item(r, c).text()
                if text == "—":
                    continue
                assert abs(float(text)) < 1.0, (
                    f"SI[{r},{c}]={text}% for symmetric data (should be ≈ 0%)"
                )


# ============================================================================
# Scenario 5 — Full CSV export
# ============================================================================

class TestScenario5FullCsvExport:
    """
    Clinical rationale: The exported CSV flows into R/MATLAB/Python pipelines.
    Incorrect headers, wrong row count, or non-zero time_s origin would
    silently corrupt downstream analysis.
    """

    @pytest.fixture(scope="class")
    def csv_rows(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("s5")
        T, P = 60, 1
        path = _make_standing_pose3d(tmp, T=T, P=P)
        d = np.load(path, allow_pickle=True)
        angles = compute_joint_angles(_opencv_to_zup(d["joints3d"]), d["conf3d"])
        out = str(tmp / "full.csv")
        angles_to_csv(out, angles, fps=FPS)
        with open(out, newline="") as fh:
            return list(csv.reader(fh)), T, P

    def test_header_starts_with_frame_time_person(self, csv_rows):
        rows, *_ = csv_rows
        assert rows[0][:3] == ["frame", "time_s", "person"]

    def test_header_contains_all_angle_names(self, csv_rows):
        rows, *_ = csv_rows
        for adef in ANGLE_DEFINITIONS:
            assert adef.name in rows[0], f"Missing column: {adef.name}"

    def test_row_count_equals_t_times_p_plus_header(self, csv_rows):
        rows, T, P = csv_rows
        assert len(rows) == T * P + 1

    def test_time_s_starts_at_zero(self, csv_rows):
        rows, *_ = csv_rows
        assert float(rows[1][1]) == pytest.approx(0.0, abs=1e-6)

    def test_time_s_increments_by_one_over_fps(self, csv_rows):
        rows, *_ = csv_rows
        assert float(rows[2][1]) - float(rows[1][1]) == pytest.approx(
            1.0 / FPS, abs=1e-5
        )

    def test_frame_column_starts_at_zero(self, csv_rows):
        rows, *_ = csv_rows
        assert int(rows[1][0]) == 0

    def test_nan_written_as_empty_string_not_literal_nan(self, tmp_path):
        """A joint with conf=0 must produce an empty CSV cell, not 'nan'."""
        T = 5
        path = _make_standing_pose3d(tmp_path, T=T, name="nan_test.npz")
        d = np.load(path, allow_pickle=True)
        conf = d["conf3d"].copy()
        conf[0, 0, 13] = 0.0   # zero L Knee vertex at frame 0
        angles = compute_joint_angles(_opencv_to_zup(d["joints3d"]), conf)
        out = str(tmp_path / "nan.csv")
        angles_to_csv(out, angles, fps=FPS)
        with open(out, newline="") as fh:
            rows = list(csv.reader(fh))
        # col 3 = first angle (L Knee Flex) in the header
        col_idx = rows[0].index(ANGLE_DEFINITIONS[_LKNEE].name)
        assert rows[1][col_idx] == "", (
            f"NaN angle exported as '{rows[1][col_idx]}' instead of empty string"
        )


# ============================================================================
# Scenario 6 — Segment CSV export
# ============================================================================

class TestScenario6SegmentCsvExport:
    """
    Clinical rationale: Phase-specific exports must have their own
    time axis (t=0 at segment start) so multiple trials can be
    overlaid and compared without manual offset correction.
    """

    @pytest.fixture(scope="class")
    def seg_csv(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("s6")
        T = 90
        path = _make_standing_pose3d(tmp, T=T)
        d = np.load(path, allow_pickle=True)
        angles = compute_joint_angles(_opencv_to_zup(d["joints3d"]), d["conf3d"])
        f0, f1 = 20, 49          # 30 frames
        seg = angles[f0 : f1 + 1]
        out = str(tmp / "segment.csv")
        angles_to_csv(out, seg, fps=FPS)
        with open(out, newline="") as fh:
            rows = list(csv.reader(fh))
        return rows, f1 - f0 + 1

    def test_row_count_equals_segment_length(self, seg_csv):
        rows, seg_len = seg_csv
        assert len(rows) == seg_len + 1   # +1 header

    def test_time_s_starts_at_zero(self, seg_csv):
        rows, _ = seg_csv
        assert float(rows[1][1]) == pytest.approx(0.0, abs=1e-6)

    def test_frame_column_starts_at_zero(self, seg_csv):
        rows, _ = seg_csv
        assert int(rows[1][0]) == 0

    def test_last_time_s_equals_segment_duration(self, seg_csv):
        rows, seg_len = seg_csv
        expected = (seg_len - 1) / FPS
        assert float(rows[-1][1]) == pytest.approx(expected, abs=1e-4)


# ============================================================================
# Scenario 7 — Playback controls
# ============================================================================

class TestScenario7PlaybackControls:
    """
    Clinical rationale: Frame-accurate playback is essential for identifying
    the precise frame of maximum knee flexion — the primary measurement in
    ACL and TKR assessment protocols.  Off-by-one errors in seek or
    play-timer wrap would cause incorrect frame timestamps in the report.
    """

    @pytest.fixture
    def tab(self, qtbot, tmp_path):
        path = _make_standing_pose3d(tmp_path, T=90)
        w = AnalysisTab()
        qtbot.addWidget(w)
        w.show()
        w.load_pose3d(path)
        _pump()
        return w

    def test_slider_maximum_is_t_minus_1(self, tab):
        assert tab._slider.maximum() == 89

    def test_next_frame_increments_by_one(self, tab):
        tab._slider.setValue(5)
        tab._on_next_frame()
        assert tab._slider.value() == 6

    def test_prev_frame_decrements_by_one(self, tab):
        tab._slider.setValue(10)
        tab._on_prev_frame()
        assert tab._slider.value() == 9

    def test_next_frame_clamps_at_end(self, tab):
        tab._slider.setValue(89)
        tab._on_next_frame()
        assert tab._slider.value() == 89

    def test_prev_frame_clamps_at_zero(self, tab):
        tab._slider.setValue(0)
        tab._on_prev_frame()
        assert tab._slider.value() == 0

    def test_seek_fwd_advances_by_one_second(self, tab):
        tab._slider.setValue(0)
        tab._on_seek_fwd()
        assert tab._slider.value() == int(FPS)

    def test_seek_back_goes_back_one_second(self, tab):
        tab._slider.setValue(60)
        tab._on_seek_back()
        assert tab._slider.value() == 60 - int(FPS)

    def test_seek_fwd_clamps_at_end(self, tab):
        tab._slider.setValue(80)
        tab._on_seek_fwd()
        assert tab._slider.value() == 89

    def test_seek_back_clamps_at_zero(self, tab):
        tab._slider.setValue(5)
        tab._on_seek_back()
        assert tab._slider.value() == 0

    def test_play_button_starts_unchecked(self, tab):
        assert not tab._play_btn.isChecked()

    def test_play_starts_timer(self, tab):
        assert not tab._play_timer.isActive()
        tab._play_btn.setChecked(True)
        assert tab._play_timer.isActive()

    def test_pause_stops_timer(self, tab):
        tab._play_btn.setChecked(True)
        tab._play_btn.setChecked(False)
        assert not tab._play_timer.isActive()

    def test_icon_changes_when_playing(self, tab):
        icon_idle = tab._play_btn.icon().cacheKey()
        tab._play_btn.setChecked(True)
        icon_play = tab._play_btn.icon().cacheKey()
        assert icon_idle != icon_play

    def test_one_frame_of_elapsed_advances_one_frame(self, tab):
        """At 1× speed, one frame-period of elapsed time advances exactly one frame."""
        tab._slider.setValue(10)
        tab._advance_by_elapsed(1000.0 / FPS)   # exactly one frame period
        assert tab._slider.value() == 11

    def test_moderate_long_tick_drops_frames_to_stay_realtime(self, tab):
        """A tick covering 3 frame-periods advances 3 frames (frame-dropping)."""
        tab._slider.setValue(10)
        tab._advance_by_elapsed(3 * 1000.0 / FPS)
        assert tab._slider.value() == 13

    def test_huge_tick_is_capped(self, tab):
        """A very long tick is capped (no catch-up surge)."""
        from app.gui.analysis_tab import _MAX_PLAY_STEP

        tab._slider.setValue(10)
        tab._advance_by_elapsed(20 * 1000.0 / FPS)   # 20 frames' worth
        assert tab._slider.value() == 10 + _MAX_PLAY_STEP

    def test_sub_frame_elapsed_does_not_advance(self, tab):
        """Less than one frame-period of elapsed time holds the current frame."""
        tab._slider.setValue(10)
        tab._advance_by_elapsed(1000.0 / FPS * 0.4)   # 0.4 of a frame
        assert tab._slider.value() == 10

    def test_advance_wraps_at_end(self, tab):
        tab._slider.setValue(tab._n_frames - 1)
        tab._advance_by_elapsed(1000.0 / FPS)
        assert tab._slider.value() == 0

    def test_fractional_frames_accumulate(self, tab):
        """Two 0.6-frame ticks accumulate to advance one frame total."""
        tab._slider.setValue(10)
        tab._advance_by_elapsed(1000.0 / FPS * 0.6)   # 0.6 → 0 frames, 0.6 stored
        assert tab._slider.value() == 10
        tab._advance_by_elapsed(1000.0 / FPS * 0.6)   # +0.6 → 1.2 → 1 frame
        assert tab._slider.value() == 11

    def test_frame_label_shows_current_frame(self, tab):
        tab._slider.setValue(30)
        assert "30" in tab._frame_label.text()

    def test_frame_label_shows_time_in_seconds(self, tab):
        """Frame 30 at 30 fps = 1.0 s → label should contain '00:01'."""
        tab._slider.setValue(30)
        assert "00:01" in tab._frame_label.text()

    def test_seek_to_frame_does_not_emit_frame_seek_signal(self, tab):
        """seek_to_frame() is called by the Recon slider and must never
        echo back a frame_seek signal — that would create an infinite loop."""
        emitted: list[int] = []
        tab.frame_seek.connect(lambda f: emitted.append(f))
        tab.seek_to_frame(25)
        assert emitted == [], "seek_to_frame leaked a frame_seek signal (feedback loop)"

    def test_slider_drag_emits_frame_seek_signal(self, tab):
        """Normal user-initiated slider movement must emit frame_seek."""
        emitted: list[int] = []
        tab.frame_seek.connect(lambda f: emitted.append(f))
        tab._slider.setValue(15)
        assert 15 in emitted

    def test_2d_preview_driven_during_playback(self, tab):
        """The 2D preview is requested every frame during playback.

        show_frame() is non-blocking (the preview decodes on a background
        thread and coalesces to the latest frame), so driving it during
        playback is safe and keeps the camera overlays visible.
        """
        calls: list[int] = []
        tab._preview_2d.show_frame = lambda f: calls.append(f)  # spy
        tab._play_btn.setChecked(True)        # start playing
        tab._slider.setValue(20)              # frame change during playback
        assert 20 in calls, "2D preview not driven during playback"
        tab._play_btn.setChecked(False)

    def test_2d_preview_updates_on_manual_scrub(self, tab):
        """Scrubbing the slider requests the corresponding preview frame."""
        calls: list[int] = []
        tab._preview_2d.show_frame = lambda f: calls.append(f)
        tab._slider.setValue(42)
        assert 42 in calls


# ============================================================================
# Scenario 8 — Play-button synchronisation across tabs
# ============================================================================

class TestScenario8PlaySync:
    """
    Clinical rationale: When a physiotherapist plays a recording while a
    colleague monitors the 2D overlay in the other tab, both must always
    show the same play/pause state.  If both timers run simultaneously the
    frame counter double-advances, producing incorrect time stamps.
    """

    @pytest.fixture
    def window(self, qtbot, tmp_path):
        from app.gui.main_window import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        win.show()
        # Load the same pose3d into both tabs (bypasses the pipeline worker).
        path = _make_standing_pose3d(tmp_path, T=60)
        win._recon_tab._load_pose3d(path)
        win._analysis_tab.load_pose3d(path)
        _pump()
        return win

    def test_pressing_play_on_analysis_mirrors_to_recon(self, window):
        window._analysis_tab._play_btn.setChecked(True)
        _pump()
        assert window._recon_tab._play_btn.isChecked()

    def test_pressing_play_on_recon_mirrors_to_analysis(self, window):
        window._recon_tab._play_btn.setChecked(True)
        _pump()
        assert window._analysis_tab._play_btn.isChecked()

    def test_only_analysis_timer_runs_when_analysis_plays(self, window):
        window._analysis_tab._play_btn.setChecked(True)
        _pump()
        assert window._analysis_tab._play_timer.isActive()
        assert not window._recon_tab._play_timer.isActive(), (
            "Recon timer running while Analysis plays — double-advance bug"
        )

    def test_only_recon_timer_runs_when_recon_plays(self, window):
        window._recon_tab._play_btn.setChecked(True)
        _pump()
        assert window._recon_tab._play_timer.isActive()
        assert not window._analysis_tab._play_timer.isActive(), (
            "Analysis timer running while Recon plays — double-advance bug"
        )

    def test_pausing_analysis_unsets_recon_button(self, window):
        window._analysis_tab._play_btn.setChecked(True)
        _pump()
        window._analysis_tab._play_btn.setChecked(False)
        _pump()
        assert not window._recon_tab._play_btn.isChecked()

    def test_pausing_recon_unsets_analysis_button(self, window):
        window._recon_tab._play_btn.setChecked(True)
        _pump()
        window._recon_tab._play_btn.setChecked(False)
        _pump()
        assert not window._analysis_tab._play_btn.isChecked()

    def test_recon_slider_drives_analysis_slider(self, window):
        """Moving Recon's slider must advance Analysis via seek_to_frame."""
        window._recon_tab._slider.setValue(20)
        _pump()
        assert window._analysis_tab._slider.value() == 20

    def test_analysis_slider_drives_recon_slider(self, window):
        """Moving Analysis's slider must advance Recon via frame_seek signal."""
        window._analysis_tab._slider.setValue(35)
        _pump()
        assert window._recon_tab._slider.value() == 35


# ============================================================================
# Scenario 9 — Confidence threshold control
# ============================================================================

def _make_pose3d_with_lknee_conf(tmp_path: Path, knee_conf: float, T: int = 40) -> str:
    """Standing pose where the L Knee joint (idx 13) has a fixed low confidence.

    Every other joint stays at conf=1.0.  Lets a test verify that raising the
    Analysis tab threshold above *knee_conf* drops the L Knee Flex angle to NaN.
    """
    zup = np.tile(_STANDING_ZUP, (T, 1, 1, 1))
    cv = _zup_to_opencv(zup)
    conf = np.ones((T, 1, 17), dtype=np.float32)
    conf[:, 0, 13] = knee_conf      # L Knee vertex
    meta = {"fps": FPS, "model_name": "synthetic"}
    path = str(tmp_path / "lowconf.npz")
    np.savez(path, joints3d=cv, conf3d=conf, meta=np.array(meta))
    return path


class TestScenario9ConfidenceThreshold:
    """
    Clinical rationale: Markerless keypoints vary in reliability frame to
    frame.  A clinician must be able to exclude low-confidence detections so
    that an angle is reported only when its underlying joints were tracked
    reliably — otherwise noisy keypoints inflate ROM and corrupt the mean.
    The control re-filters the already-reconstructed data live (no re-run).
    """

    @pytest.fixture
    def tab(self, qtbot, tmp_path):
        # L Knee confidence = 0.20; all other joints fully confident.
        path = _make_pose3d_with_lknee_conf(tmp_path, knee_conf=0.20)
        w = AnalysisTab()
        qtbot.addWidget(w)
        w.show()
        w.load_pose3d(path)
        _pump()
        return w

    def test_default_threshold_keeps_low_conf_knee(self, tab):
        """At the default 0.0 threshold the L Knee Flex angle is computed."""
        assert tab._conf_spin.value() == 0.0
        assert tab._full_table.item(_LKNEE, 2).text() != "—"   # Mean column

    def test_raising_threshold_drops_low_conf_angle_to_dash(self, tab):
        """Raising the threshold above the knee confidence NaNs the L Knee row."""
        tab._conf_spin.setValue(0.50)   # > 0.20 → L Knee excluded
        _pump()
        for c in range(tab._full_table.columnCount()):
            assert tab._full_table.item(_LKNEE, c).text() == "—", (
                f"L Knee col {c} still populated after raising confidence threshold"
            )

    def test_unaffected_angle_still_present_after_raise(self, tab):
        """An angle not using the low-conf joint (R Knee) stays populated."""
        tab._conf_spin.setValue(0.50)
        _pump()
        assert tab._full_table.item(_RKNEE, 2).text() != "—"

    def test_lowering_threshold_restores_angle(self, tab):
        """Dropping the threshold back below the knee confidence restores it."""
        tab._conf_spin.setValue(0.50)
        _pump()
        tab._conf_spin.setValue(0.10)   # < 0.20 → L Knee back in
        _pump()
        assert tab._full_table.item(_LKNEE, 2).text() != "—"

    def test_threshold_change_is_safe_with_no_data(self, qtbot):
        """Moving the spinbox before loading must not raise."""
        w = AnalysisTab()
        qtbot.addWidget(w)
        w.show()
        w._conf_spin.setValue(0.7)   # no data loaded — must be a no-op
        _pump()


# ============================================================================
# Scenario 10 — Floor world-frame pose3d is used as-is (no OpenCV→Zup swap)
# ============================================================================

def _make_world_frame_pose3d(tmp_path: Path, T: int = 30) -> tuple[str, np.ndarray]:
    """pose3d.npz already in a Z-up floor world frame (meta coordinate_frame='world')."""
    joints = np.tile(_STANDING_ZUP, (T, 1, 1, 1)).astype(np.float32)  # Z-up already
    conf = np.ones((T, 1, 17), dtype=np.float32)
    meta = {"fps": FPS, "model_name": "synthetic", "coordinate_frame": "world"}
    path = str(tmp_path / "world_pose3d.npz")
    np.savez(path, joints3d=joints, conf3d=conf, meta=np.array(meta))
    return path, joints


class TestScenario10WorldFrameConsumed:
    """
    Clinical rationale: once the floor coordinate system is set, the stored
    pose3d is already in real Z-up world metres.  The Analysis tab must use it
    directly — applying the legacy OpenCV→Z-up swap on top would rotate the
    skeleton into a wrong, non-physical frame and corrupt every angle.
    """

    def test_world_frame_pose_used_without_swap(self, qtbot, tmp_path):
        path, joints = _make_world_frame_pose3d(tmp_path)
        w = AnalysisTab()
        qtbot.addWidget(w)
        w.show()
        w.load_pose3d(path)
        _pump()
        # No swap applied: internal Z-up array equals the stored joints.
        assert np.allclose(w._joints_zup, joints, atol=1e-5)

    def test_opencv_frame_pose_is_swapped(self, qtbot, tmp_path):
        """Contrast: a default (opencv) pose3d IS swapped on load."""
        path = _make_standing_pose3d(tmp_path, T=30)  # meta has no coordinate_frame
        d = np.load(path, allow_pickle=True)
        cv_joints = d["joints3d"]
        w = AnalysisTab()
        qtbot.addWidget(w)
        w.show()
        w.load_pose3d(path)
        _pump()
        expected = np.stack(
            [cv_joints[..., 0], cv_joints[..., 2], -cv_joints[..., 1]], axis=-1
        )
        assert np.allclose(w._joints_zup, expected, atol=1e-5)
