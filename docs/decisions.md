# Architecture Decision Records

This file captures binding technical decisions for the stereo-FLIR 3D pose pipeline.
Format: short ADR with Context / Decision / Consequences.

---

## ADR-001: Default 2D Pose Backend — MediaPipe Pose Landmarker (Full)

**Date:** 2026-02-19
**Status:** Accepted

### Context

We need a CPU-first 2D pose backend that:
- Runs at ≥25 FPS on a typical x86 workstation without GPU.
- Covers full body (especially hips, knees, ankles, and wrists) for exercise biomechanics.
- Has minimal install friction (ideally a single `pip install`).
- Supports Apache-compatible licensing.

Three candidates were evaluated: MediaPipe Pose Landmarker, MoveNet TFLite, and RTMPose via rtmlib.
Full comparison is in `docs/research_pose_backends.md`.

### Decision

**Default backend: MediaPipe Pose Landmarker, Full complexity.**

`pip install mediapipe` is the only required step.
Models are bundled in the wheel; no separate download needed.
The Full model runs at ~27 FPS (CPU, XNNPACK) on a mid-range desktop, meeting the 25 FPS floor for 30 Hz video.
It outputs 33 landmarks mapped to COCO-17 (see `docs/skeleton_mapping.md`) with per-joint visibility scores used as confidence.

### Consequences

- Implementation must include the MediaPipe→COCO-17 index mapping defined in `docs/skeleton_mapping.md`.
- Single-person limitation: for ≥2 subjects, run one inference per bounding box (acceptable for the 2-person exercise use case).
- The `PoseBackend` interface (`detect(frame) → kps[N,17,2], conf[N,17]`) must be abstract so the upgrade path (ADR-002) is drop-in.

---

## ADR-002: Optional Upgrade Pose Backend — RTMPose-m via rtmlib

**Date:** 2026-02-19
**Status:** Accepted (upgrade path, not default)

### Context

Users requiring higher accuracy (e.g., subtle joint-angle analysis, >2 people) need an upgrade path that does not break the NPZ interface.

### Decision

**Upgrade backend: RTMPose-m (COCO-17) deployed via `rtmlib` with ONNXRuntime.**

Activated by passing `--backend rtmpose` to `python -m app.pose2d`.
ONNX model files (~30 MB) are auto-downloaded by rtmlib on first use and cached in `~/.rtmlib/`.
RTMPose-m achieves 75.8 AP on COCO and ~50 FPS on Intel i7 (ONNXRuntime, single thread).
RTMPose natively outputs COCO-17 → no remapping required.

Install: `pip install rtmlib onnxruntime` (add to optional extras in `pyproject.toml`).

### Consequences

- `pose2d` module must expose a `--backend {mediapipe,rtmpose}` CLI flag.
- Both backends must produce identical NPZ format (see `docs/skeleton_mapping.md` §4).
- rtmlib is not included in base `requirements.txt`; it lives in `requirements-extras.txt` or an `[extras]` section.

---

## ADR-003: Canonical Skeleton — COCO-17

**Date:** 2026-02-19
**Status:** Accepted

### Context

MediaPipe outputs 33 landmarks; MoveNet and RTMPose output 17 COCO keypoints.
Downstream code (3D triangulation, visualization, export) must not depend on which backend was used.

### Decision

**All intermediate and output data use COCO-17 as the canonical joint set (J=17).**

Index-to-joint mapping is fixed as specified in `docs/skeleton_mapping.md` §1.
MediaPipe's 33 landmarks are mapped to COCO-17 using the index table in §2 of that document; excess landmarks are discarded (can optionally be stored under key `mp_full`).

### Consequences

- NPZ arrays always have shape `[T, P, 17, 2]` / `[T, P, 17]` regardless of backend.
- Visualization code assumes J=17 and the connectivity defined in `docs/skeleton_mapping.md` §5.
- Future backends (e.g., YOLO-Pose, OpenPifPaf) must be wrapped to output COCO-17.

---

## ADR-004: 3D Reconstruction — Linear Triangulation + OneEuro Smoothing

**Date:** 2026-02-19
**Status:** Accepted

### Context

Given a calibrated stereo pair (intrinsics K1, D1, K2, D2, rotation R, translation T in `calibration.yml`), we need to compute 3D joint positions from synchronized 2D detections.

### Decision

**Pipeline:**

1. **Undistort 2D keypoints** using `cv2.undistortPoints(pts, K, D, P=K)` for each camera independently.
   Do *not* rectify full images; operate on sparse 2D points to save compute.

2. **Build projection matrices** P1 = K1 · [I | 0] and P2 = K2 · [R | T].

3. **Triangulate per joint** using `cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)`.
   Convert from homogeneous to 3D: `X = Xh[:3] / Xh[3]`.

4. **Compute reprojection error** per joint per camera. Reproject the 3D point back to each view and measure pixel distance to original 2D detection.

5. **Outlier rejection:** Set `conf3d[j] = 0` when:
   - Either 2D confidence < 0.3, OR
   - Reprojection error > 20 px (configurable via `--max-reproj-err`)

6. **Temporal smoothing:** Apply a **OneEuro filter** independently per joint per coordinate (X, Y, Z) per person.

**OneEuro filter default parameters:**

| Parameter | Default | Rationale |
|---|---|---|
| `min_cutoff` | 0.5 Hz | Reduces position jitter when subject is stationary |
| `beta` | 0.05 | Allows filter to respond faster to rapid limb movements |
| `d_cutoff` | 1.0 Hz | Low-pass on the derivative estimate (rarely needs tuning) |

Tuning guide: Increase `min_cutoff` if output looks laggy during slow motion. Increase `beta` if fast movements show visible lag.

**Why OneEuro over Kalman:**
- Kalman requires a kinematic model (constant velocity / acceleration), which is incorrect for general exercise movements (sudden direction changes, holds, etc.).
- OneEuro is model-free, has only 2 intuitive parameters, adds negligible compute, and is widely used in interactive motion capture (FreeMoCap, Pose2Sim).
- OneEuro can be applied as a post-processing pass on already-computed trajectories as well as frame-by-frame in realtime.

### Consequences

- Implementation uses `python-one-euro-filter` or the inline 10-line implementation (no heavy dependency).
- `recon3d` module exposes `--min-cutoff` and `--beta` CLI flags.
- Outlier confidence is propagated: downstream code must check `conf3d` before using a 3D joint.
- For future improvement: bundle-adjustment / EKF can be added as an optional post-processing pass without changing the interface.
