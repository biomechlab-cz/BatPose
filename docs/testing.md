# Testing Guide

## Running tests

```bash
make lint    # ruff check + format (auto-fix)
make test    # unit tests (tests/unit/)
make e2e     # end-to-end smoke tests (tests/e2e/)
```

All three targets must be green before merging.

## Test layout

```
tests/
├── unit/
│   ├── test_board.py         # Board detection (ChessboardDetector, CharucoDetector)
│   ├── test_calib_cli.py     # Calibration CLI argument parsing
│   ├── test_calib_validate.py # Calibration repeatability comparison (validate repeat)
│   ├── test_frame_select.py  # Coverage score for calibration frame selection
│   ├── test_recon3d_cli.py   # Recon3D CLI argument parsing
│   ├── test_smooth.py        # OneEuro filter + smooth_trajectory
│   ├── test_stereo.py        # calibrate_stereo + load_calibration validation
│   └── test_triangulate.py   # DLT triangulation + reprojection error (noiseless + noisy)
└── e2e/
    └── test_pipeline_smoke.py  # Full pipeline: calib I/O → pose2d I/O → recon3d
```

## Unit tests

### test_board.py
Covers `app.calib.board`:
- `make_detector()` factory for `charuco` and `checkerboard` types (and rejects unknowns).
- `make_detector()` raises clear `ValueError` for missing required `board_cfg` keys.
- `ChessboardDetector.detect()` on a synthetically rendered chessboard image.
- Blank-image negative case (should return `None`).
- `board_cfg` property round-trip.
- `ChessboardDetector` rejects `cols/rows < 1` and `square_size <= 0`.
- `CharucoDetector` rejects `squares_x/y < 2`, `marker_size >= square_size`, and unknown `aruco_dict_name`.
- `CharucoDetector.min_corners` never exceeds the board's actual inner corner count.

No real camera or video file is required.

### test_frame_select.py
Covers `app.calib.frame_select`. **Partial-detection regression** (`TestPartialDetectionsExcluded`,
`TestSpatialSubsampleEmptyGuard`): `CharucoDetector` returns a `partial=True` result with
*zero* corners as a UI diagnostic; treating it as a detection put empty point sets into the
fisheye phase-1 list, and `_spatial_subsample` then took the centroid of no points → NaN →
`cannot convert float NaN to integer`, aborting calibration. Tests assert partial results are
neither collected nor paired, that genuine detections still are, and that `_spatial_subsample`
survives a corner-less entry. Also covers `_coverage_score`:
- Full image coverage (all 4×4 cells hit) → score == 1.0.
- Single-cell cluster → score == 1/16.
- Half-grid coverage → score == 0.5.
- Edge/boundary points clamped without error.
- Score always ∈ [0, 1] for random point clouds.

### test_triangulate.py
Covers `app.recon3d.triangulate`:
- Single-point DLT triangulation with exact recovery (atol 1e-4).
- Multi-point (17 joints) at 2 m depth with random spread (atol 1e-3).
- **Noisy triangulation**: 17 joints with ~1 px Gaussian noise; max 3D error < 10 cm,
  mean < 5 cm (theoretical 1σ ≈ 3.3 cm for Z=2 m, b=0.15 m, f=800 px).
- Noise degrades gracefully — output remains finite for 0.5–5 px noise.
- Output shape `(N, 3)` and `float32` dtype.
- Reprojection error returns `(N,)` and is < 0.1 px for perfectly projected points.

Synthetic stereo rig: `fx=fy=800`, `cx=640`, `cy=360`, baseline along X.

### test_smooth.py
Covers `app.recon3d.smooth`:
- `OneEuroFilter`: constant-signal convergence, first-sample passthrough, noise variance
  reduction ≥ 50 %, reset behaviour.
- `smooth_trajectory`: shape preservation for 2-D `(T, 3)` and 4-D `(T, P, J, 3)` arrays,
  constant trajectory stability after warm-up.

### test_calib_validate.py
Covers `app.calib.validate` (the `repeat` repeatability comparison):
- `MetricRow` statistics: sample SD (`ddof=1`), range, CoV; CoV suppressed for signed
  quantities (Tx, rvec, distortion coeffs); NaN values ignored rather than propagated;
  range over a single finite value is NaN, not `0.0` (which would read as perfect agreement).
- Rotation helpers: known yaw recovered by `rotation_angle_deg`, `rotation_diff_deg` on a
  2.5° pair, arccos clamp survives a non-orthonormal matrix, pairwise matrix symmetric with
  a zero diagonal.
- Parameter rows: baseline reported in mm, focal/baseline/rotation spreads detected,
  distortion row count follows the model (4 fisheye vs 5 pinhole), missing `quality` → NaN.
- Input validation: <2 calibrations, label-count mismatch, mixed lens models, mixed image
  sizes and mixed coefficient counts all raise; a differing `board_cfg` warns and proceeds;
  identical calibrations give exactly zero spread.
- Rendering: report is **ASCII-only** (cp1250 consoles), NaN renders as `n/a`, a long table
  title does not widen the name column, CSV header/row count, JSON round-trips and is keyed
  by run label.
- Segment cross-check on synthetic pinhole projections: recovered thigh length within 5 mm
  of truth, identical calibrations → zero segment spread, and a **1 % baseline error scales
  every segment by 1 %** (the property that makes the mm figure interpretable).
- CLI: parser defaults, subcommand/`--calib` required, and `main()` end-to-end through real
  YAML files (table printed, CSV+JSON written, single-calib and half-a-pose2d-pair rejected).

## End-to-end smoke tests

All tests use **synthetic data only** — no network access, no real camera footage.

### TestSmokeCalibrationIO
- `save_calibration` / `load_calibration` round-trip preserves `K1`, `R`, `T` to 1e-9.
- Written YAML is valid and contains mandatory keys `K1`, `quality`.

### TestSmokePose2DIO
- `save_pose2d` / `load_pose2d` round-trip preserves array values and `meta` fields.
- Output shapes: `keypoints (T, P, 17, 2)`, `conf (T, P, 17)`, both `float32`.

### TestLoadPose2DMissingKeys
- `load_pose2d` raises `ValueError` with a clear message when the NPZ is missing required keys
  (`keypoints`, `conf`, or `meta`).

### TestSmokeRecon3D
- Full pipeline `calibration.yml + pose2d_left.npz + pose2d_right.npz → pose3d.npz`.
- Asserts output file exists with shapes `(T, P, 17, 3)` / `(T, P, 17)` and
  `meta["skeleton"] == "coco17"`, `meta["smoothing"] == "oneeuro"`.
- Low-confidence variant: all `conf = 0` → all `conf3d` entries must be `0.0`.
- **Mismatched frame counts**: left=20 frames, right=8 frames → output silently truncated
  to 8 frames (documents BUG-006; update if behaviour changes to raise an error).

### TestSmokeOneEuro
- Imports `app.calib.__main__`, `app.pose2d.__main__`, `app.recon3d.__main__` and
  verifies they each expose a `main` symbol.

## Design constraints

- Tests require **no GPU** and **no network**.
- `pytest-timeout` is set to 120 s globally (see `pyproject.toml`).
- Synthetic data generators live in `tests/e2e/test_pipeline_smoke.py` as module-level
  helpers (`_make_synthetic_calibration`, `_make_synthetic_pose2d`).

## Adding new tests

1. Unit tests go in `tests/unit/test_<module>.py`.
2. Integration/E2E tests go in `tests/e2e/`.
3. Use `tmp_path` (pytest fixture) for any file I/O.
4. Generate data synthetically — never commit real videos or calibration files.
5. Run `make lint` before committing to keep ruff happy.
