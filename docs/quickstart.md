# Quickstart Guide

## Requirements

- Python 3.11+
- `uv` (recommended) or `pip`
- No GPU required — all processing runs on CPU.

## Installation

```bash
# Clone the repo
git clone <repo-url> BatPose
cd BatPose

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Optional extras

```bash
# RTMPose backend (higher accuracy, ~50 fps on i7):
pip install -e ".[rtmpose]"
# Downloads onnxruntime + rtmlib; models auto-download on first run.

# FLIR cameras (requires Spinnaker SDK from FLIR's website):
pip install PySpin  # install the PySpin wheel matching your Python version
```

---

## First Run

### 1. Launch the GUI

```bash
python -m app.gui
# or: make run-gui
```

Create or open a **project folder** via `File → New Project…` or `File → Open Project…`.
All intermediate files (calibration.yml, pose2d_*.npz, pose3d.npz) are written there.

---

## Stereo Calibration

### Via GUI

1. Open the **Calibration** tab.
2. Select left and right camera videos of your calibration board.
3. Configure the board (ChArUco is recommended):
   - `squares_x`, `squares_y`: board dimensions
   - `square_size`: physical size of each square (metres)
   - `marker_size`: ArUco marker size (metres, must be < `square_size`)
4. Click **Run Calibration**.
5. Inspect the RMS reprojection error in the results panel.
   - **< 0.5 px** = excellent
   - **0.5–1.5 px** = acceptable
   - **> 1.5 px** = re-calibrate (check lighting, board flatness, coverage)

### Via CLI

```bash
python -m app.calib \
  --left  /path/to/left_calib.mp4 \
  --right /path/to/right_calib.mp4 \
  --board charuco \
  --squares-x 5 --squares-y 7 \
  --square-size 0.04 --marker-size 0.024 \
  --aruco-dict DICT_6X6_250 \
  --lens fisheye \
  --out calibration.yml
```

`--lens` selects the distortion model (`standard` ≤ 90° FOV, `wide-angle` 90–150°,
`fisheye` > 150°) and **must match the physical lens** — a fisheye rig calibrated
with the standard model produces a calibration that silently breaks 3D reconstruction.

Output: `calibration.yml` containing K1, D1, K2, D2, R, T, the lens model and quality metrics.

---

## 2D Pose Extraction

### Via CLI

```bash
python -m app.pose2d \
  --left  /path/to/left_exercise.mp4 \
  --right /path/to/right_exercise.mp4 \
  --outdir /path/to/project/ \
  --backend mediapipe \
  --num-poses 2
```

**First run note:** MediaPipe will download the `pose_landmarker_full.task` model (~28 MB)
to `~/.cache/BatPose/` on first use.

**Multi-person caveat:** with `--num-poses` > 1 the left/right person slots are paired by
detection ORDER — there is no cross-view identity matching. Reliable for a single person;
when several people overlap or cross between views, identities can swap and the wrong
bodies get triangulated.

Output files:
- `pose2d_left.npz`  — keypoints `[T, P, 17, 2]` + conf `[T, P, 17]`
- `pose2d_right.npz` — same structure

### Via GUI

In the **Reconstruction** tab, click **Run Pipeline** — this runs pose2d extraction
and 3D reconstruction in sequence.

---

## 3D Reconstruction

### Via CLI

```bash
python -m app.recon3d \
  --calib  /path/to/calibration.yml \
  --pose2d /path/to/project/ \
  --out    /path/to/project/pose3d.npz
```

Optional flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--min-conf` | 0.3 | Minimum 2D confidence to accept a joint |
| `--max-reproj-err` | 20.0 | Max reprojection error (px) before marking joint invalid |
| `--min-cutoff` | 1.0 | OneEuro min_cutoff (Hz) — lower = smoother (matches the GUI default) |
| `--beta` | 0.5 | OneEuro beta — higher = less lag on fast motion (matches the GUI default) |

Output: `pose3d.npz` with `joints3d [T, P, 17, 3]`, `conf3d`, `repro_err`, `meta`.

---

## Full Pipeline (one command)

```bash
# 1. Calibrate
python -m app.calib --left left_calib.mp4 --right right_calib.mp4 --out project/calibration.yml

# 2. Extract 2D poses
python -m app.pose2d --left left_exercise.mp4 --right right_exercise.mp4 --outdir project/

# 3. Reconstruct 3D
python -m app.recon3d --calib project/calibration.yml --pose2d project/ --out project/pose3d.npz
```

---

## Export

In the GUI, use the **Export CSV…** button in the Reconstruction tab.
The CSV contains columns: `frame, time_s, person, j0_x, j0_y, j0_z, j0_conf, …, j16_conf`.

---

## Realtime Capture (MVP)

If FLIR cameras are connected and PySpin is installed:

```python
from app.capture.flir import FlirCapture

with FlirCapture(serial_left="12345678", serial_right="87654321") as cap:
    while True:
        frame = cap.read()
        if frame is None:
            break
        # frame.frame_left, frame.frame_right, frame.timestamp
```

If PySpin is not installed, the GUI's Live Capture tab uses the synthetic simulator
automatically. You can also exercise the capture interface directly:

```python
from app.capture import SimulatorCapture

with SimulatorCapture(fps=30.0) as cap:
    frame = cap.read()
    # frame.frame_left, frame.frame_right, frame.timestamp
```

---

## Caching

The app automatically caches intermediate results:

| File | Created by | Re-used by |
|------|-----------|-----------|
| `calibration.yml` | `app.calib` | `app.recon3d`, GUI |
| `pose2d_left.npz` | `app.pose2d` | `app.recon3d` |
| `pose2d_right.npz` | `app.pose2d` | `app.recon3d` |
| `pose3d.npz` | `app.recon3d` | GUI viewer |

On **File → Open Project**, the GUI auto-detects and loads these files.

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| High RMS (>2 px) | Check board flatness, improve lighting, use more frames with better coverage |
| MediaPipe download fails | Set `http_proxy` / `https_proxy` or manually copy `pose_landmarker_full.task` to `~/.cache/BatPose/` |
| 3D joints scattered / noisy | Decrease `--min-cutoff` (more smoothing) or decrease `--beta` |
| GUI freezes | Should not happen; if it does, file a bug — all heavy work runs in QThread |
| `No valid paired frames found` | Board not detected: check board config, ensure board fills >30% of frame |
| `cv2.aruco` not found | Install `opencv-contrib-python` (not plain `opencv-python`) |

---

## Development

```bash
make lint     # ruff check + format
make test     # pytest tests/unit
make e2e      # pytest tests/e2e (end-to-end smoke test)
```

See `docs/decisions.md` for architecture decisions.
See `docs/skeleton_mapping.md` for the COCO-17 joint spec.
