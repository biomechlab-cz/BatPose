# Canonical Skeleton Definition and Keypoint Mapping

**Date:** 2026-02-19
**Status:** Final – binding for all pipeline stages

---

## 1. Canonical Joint Set: COCO-17

All internal representations, NPZ files, and downstream 3D reconstruction code use the **COCO-17** joint set as the canonical format. This is the smallest widely-supported set that covers the full body for biomechanical exercise analysis.

### Joint Index Table

| Index | Joint Name | Side | Body Segment |
|-------|------------|------|--------------|
| 0 | nose | center | Head |
| 1 | left_eye | left | Head |
| 2 | right_eye | right | Head |
| 3 | left_ear | left | Head |
| 4 | right_ear | right | Head |
| 5 | left_shoulder | left | Torso |
| 6 | right_shoulder | right | Torso |
| 7 | left_elbow | left | Arm |
| 8 | right_elbow | right | Arm |
| 9 | left_wrist | left | Arm |
| 10 | right_wrist | right | Arm |
| 11 | left_hip | left | Pelvis |
| 12 | right_hip | right | Pelvis |
| 13 | left_knee | left | Leg |
| 14 | right_knee | right | Leg |
| 15 | left_ankle | left | Leg |
| 16 | right_ankle | right | Leg |

**J = 17** (constant; all backends must output exactly this set)

### Derived Virtual Joints (computed, not stored in pose2d)

These virtual joints are computed from the canonical set during 3D reconstruction for biomechanical segment analysis:

| Name | Definition |
|------|------------|
| mid_hip | mean(left_hip, right_hip) |
| mid_shoulder | mean(left_shoulder, right_shoulder) |
| neck | mean(left_shoulder, right_shoulder) shifted toward nose |
| head_top | extrapolated from nose + neck direction |

Virtual joints are stored separately in pose3d output if needed; they do not inflate J.

---

## 2. MediaPipe → COCO-17 Mapping

The default backend (MediaPipe Pose Landmarker Full) produces 33 landmarks. The following direct index mapping selects the COCO-17 subset.

### Mapping Table

| COCO-17 Index | COCO Joint | MediaPipe-33 Index | MediaPipe Landmark Name |
|---|---|---|---|
| 0 | nose | 0 | NOSE |
| 1 | left_eye | 2 | LEFT_EYE |
| 2 | right_eye | 5 | RIGHT_EYE |
| 3 | left_ear | 7 | LEFT_EAR |
| 4 | right_ear | 8 | RIGHT_EAR |
| 5 | left_shoulder | 11 | LEFT_SHOULDER |
| 6 | right_shoulder | 12 | RIGHT_SHOULDER |
| 7 | left_elbow | 13 | LEFT_ELBOW |
| 8 | right_elbow | 14 | RIGHT_ELBOW |
| 9 | left_wrist | 15 | LEFT_WRIST |
| 10 | right_wrist | 16 | RIGHT_WRIST |
| 11 | left_hip | 23 | LEFT_HIP |
| 12 | right_hip | 24 | RIGHT_HIP |
| 13 | left_knee | 25 | LEFT_KNEE |
| 14 | right_knee | 26 | RIGHT_KNEE |
| 15 | left_ankle | 27 | LEFT_ANKLE |
| 16 | right_ankle | 28 | RIGHT_ANKLE |

**Discarded MediaPipe landmarks:** eye inner/outer (1,3,4,6), mouth (9,10), pinky/index/thumb (17–22), heel (29,30), foot_index (31,32).
These are intentionally dropped; they can be stored in an auxiliary key `mp_full` in the NPZ if needed by callers, but must not be used in COCO-17 slots.

### Confidence Mapping

MediaPipe outputs two scores per landmark:
- `visibility`: probability the joint is visible in frame (0–1)
- `presence`: probability the joint exists in the image at all (0–1)

Use `confidence[j] = visibility[mp_idx]` as the COCO-17 confidence value.
If `presence[mp_idx] < 0.5`, override confidence to 0.0 (joint treated as occluded).

### Reference Python Snippet

```python
# Indices of MediaPipe-33 landmarks that map to COCO-17 (in order)
MP_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

def mediapipe_to_coco17(landmarks):
    """
    landmarks: list of 33 NormalizedLandmark objects from MediaPipe
    Returns:
        kps:  np.ndarray [17, 2] float32  (x, y in [0,1] normalized)
        conf: np.ndarray [17]   float32
    """
    import numpy as np
    kps  = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros(17,      dtype=np.float32)
    for coco_idx, mp_idx in enumerate(MP_TO_COCO17):
        lm = landmarks[mp_idx]
        kps[coco_idx]  = [lm.x, lm.y]
        vis  = lm.visibility  if lm.visibility  is not None else 0.0
        pres = lm.presence    if lm.presence     is not None else 0.0
        conf[coco_idx] = float(vis) if float(pres) >= 0.5 else 0.0
    return kps, conf
```

---

## 3. RTMPose → COCO-17 Mapping

RTMPose natively outputs COCO-17 keypoints with indices already matching the canonical table above. **No remapping required.**

Confidence is the per-joint score from the SimCC head (0–1). Use directly as `conf[j]`.

---

## 4. NPZ Storage Format

### pose2d_left.npz / pose2d_right.npz

```
keypoints : float32  shape [T, P, 17, 2]
                T = number of frames
                P = number of people (1 for single-person MediaPipe)
                17 = joint count (COCO-17)
                2  = (x, y) in pixel coordinates of the source frame

conf      : float32  shape [T, P, 17]
                confidence in [0, 1]

meta      : dict stored as numpy array of dtype object, keys:
                fps            : float
                timestamps     : float32 [T]  (seconds from video start; NaN if unavailable)
                model_name     : str  e.g. "mediapipe_full" or "rtmpose_m"
                skeleton       : str  "coco17"
                image_size     : [int, int]  [width, height]
```

**Coordinate system:** pixel coordinates in the *original* (non-rectified) source frame. Undistortion is applied by the reconstruction stage, not the pose stage.

### pose3d.npz

```
joints3d  : float32  shape [T, P, 17, 3]
                (X, Y, Z) in camera-1 (left camera) coordinate frame, metres

conf3d    : float32  shape [T, P, 17]
                min(conf_left, conf_right) for corresponding joint

repro_err : float32  shape [T, P, 17]
                mean reprojection error in pixels across both views

meta      : dict, same keys as pose2d meta plus:
                calibration_file : str  (path relative to project root)
                smoothing        : str  "oneeuro"
                smooth_params    : dict  e.g. {"min_cutoff": 0.5, "beta": 0.05}
```

---

## 5. Skeleton Connectivity (for visualization)

Edge list defining the skeleton graph (for 3D overlay rendering):

```python
COCO17_EDGES = [
    (0, 1), (0, 2),          # nose - eyes
    (1, 3), (2, 4),          # eyes - ears
    (5, 6),                  # shoulders
    (5, 7), (7, 9),          # left arm
    (6, 8), (8, 10),         # right arm
    (5, 11), (6, 12),        # torso sides
    (11, 12),                # hips
    (11, 13), (13, 15),      # left leg
    (12, 14), (14, 16),      # right leg
]
```

---

## 6. Body Side Convention

- **Left/Right** follows the subject's anatomical left/right (not the camera's left/right).
- MediaPipe uses the same convention: `LEFT_SHOULDER` is the subject's left shoulder.
- All backends must confirm this convention. If a backend uses camera-relative coordinates, flip indices before writing to NPZ.
