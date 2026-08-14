# Joint-Angle Definitions

Reference for researchers interpreting BatPose joint angles. This documents
*exactly* what each angle is, as implemented in
[`app/biomech/angles.py`](../app/biomech/angles.py) — not a general convention.

BatPose reports **nine** angles per person per frame, in this fixed order
(the third axis of `compute_joint_angles()` output):

| # | Name | Type | Vertex | Segments compared | COCO-17 landmarks (A, B, C) |
|---|------|------|--------|-------------------|------------------------------|
| 0 | L Knee Flex | 3-point | L knee | thigh vs shank | L hip (11), **L knee (13)**, L ankle (15) |
| 1 | R Knee Flex | 3-point | R knee | thigh vs shank | R hip (12), **R knee (14)**, R ankle (16) |
| 2 | L Hip Flex | 3-point | L hip | trunk-side vs thigh | L shoulder (5), **L hip (11)**, L knee (13) |
| 3 | R Hip Flex | 3-point | R hip | trunk-side vs thigh | R shoulder (6), **R hip (12)**, R knee (14) |
| 4 | L Elbow Flex | 3-point | L elbow | upper arm vs forearm | L shoulder (5), **L elbow (7)**, L wrist (9) |
| 5 | R Elbow Flex | 3-point | R elbow | upper arm vs forearm | R shoulder (6), **R elbow (8)**, R wrist (10) |
| 6 | L Shoulder Flex | 3-point | L shoulder | trunk-side vs upper arm | L hip (11), **L shoulder (5)**, L elbow (7) |
| 7 | R Shoulder Flex | 3-point | R shoulder | trunk-side vs upper arm | R hip (12), **R shoulder (6)**, R elbow (8) |
| 8 | Trunk Inclination | segment-vs-axis | — | trunk vs vertical | mid-hip (11,12) → mid-shoulder (5,6) vs world **+Z** |

## What a "3-point angle" is here

For angles 0–7 the value is the **unsigned interior angle at the vertex B**
between the two 3-D segment vectors **B→A** and **B→C**:

```
θ = arccos( (A−B)·(C−B) / (|A−B| · |C−B|) ),   in degrees, range [0°, 180°]
```

Properties a researcher must keep in mind:

- **It is the true 3-D angle between the two limb segments** — *not* a
  projection onto a sagittal/frontal/transverse plane. Out-of-plane motion is
  included in the value.
- **Unsigned and offset-free.** There is no anatomical-neutral zero and no
  flexion/extension sign. A straight (co-linear) joint reads **180°**, and the
  angle *decreases* as the joint closes. So despite the "Flex" label, e.g.
  *L Knee Flex ≈ 180°* means the knee is **extended (straight)**, and it drops
  toward 0° as the knee flexes. Convert to a clinical flexion angle yourself if
  you need one (commonly `flexion ≈ 180° − θ`), and define your own plane if you
  need planar angles.
- **Rotation-invariant.** Because it depends only on segment directions, the
  result is independent of the camera pose and of whether a floor world frame
  was set. (This is *not* true for Trunk Inclination — see below.)

Interpretation per joint (θ = reported value):

- **Knee / Elbow** — included angle of the limb. ~180° fully extended, smaller
  when flexed.
- **Hip** — angle between the trunk side (hip→shoulder) and the thigh
  (hip→knee). ~180° when standing tall; decreases as the hip flexes (e.g. sit,
  squat, high knee).
- **Shoulder** — angle between the trunk side (shoulder→hip) and the upper arm
  (shoulder→elbow). ~0–20° with the arm hanging by the side; approaches ~180°
  with the arm raised overhead. It mixes flexion and abduction (it is the 3-D
  arm-to-torso angle, not a plane-specific one).

## Trunk Inclination (angle 8)

A special case. The trunk vector is **mid-shoulder − mid-hip**, where
mid-hip = ½(L hip + R hip) and mid-shoulder = ½(L shoulder + R shoulder). The
angle is taken against the **world vertical (+Z)** axis:

```
θ_trunk = arccos( trunk_z / |trunk| ),   in degrees
```

- **0° = perfectly upright**, **90° = horizontal** (bent forward 90° or lying
  down). It measures lean magnitude, not direction.
- **This angle depends on the coordinate frame.** +Z must be true vertical. That
  holds when a **floor world frame** is set (*Set coordinate system*, Z = up out
  of the floor); without it, +Z is the viewer's Z-up-swapped camera axis, which
  only approximates gravity to the extent the cameras were level. For
  gravity-referenced trunk lean, set the floor coordinate system.

## Coordinate frame

Angles are computed in the **Z-up world frame** (X lateral, Y anterior, Z
vertical). `pose3d.npz` is stored in the OpenCV camera frame; the caller (the
Analysis tab) applies the OpenCV→Z-up swap **before** calling
`compute_joint_angles`. As noted, this only affects Trunk Inclination — the
3-point angles are frame-independent. Units are **degrees** throughout.

## Missing data / confidence

A joint is *present* in a frame when its confidence is `> 0` **and** `≥ min_conf`
(the Analysis tab's confidence slider; default 0). An angle is set to **`NaN`**
for any frame in which **any** of its flanking joints is not present:

- 3-point angles need all three landmarks (A, B, C).
- Trunk Inclination needs **both** hips (11, 12) **and both** shoulders (5, 6).

`NaN` frames are excluded from every statistic below and break the plotted
curve, so tracking gaps neither bias the descriptive stats nor connect across
the gap.

## Derived per-angle statistics

Computed over the non-`NaN` frames of each angle/person
(`compute_extended_stats`):

| Metric | Definition |
|--------|------------|
| Min / Max | extremes of θ over valid frames |
| Mean | arithmetic mean of θ |
| Median | 50th percentile of θ |
| SD | population standard deviation (`ddof = 0`) |
| CV% | 100 · SD / \|Mean\| (0 when \|Mean\| ≈ 0) |
| ROM | Max − Min (range of motion) |
| NaN% | percentage of frames lost to low confidence |
| Excursion | Σ \|θ[t+1] − θ[t]\| over **consecutive valid** frame pairs (total arc travelled; ≥ ROM when the joint oscillates) |
| Peak angular velocity | max \|dθ/dt\| in deg/s, via central finite difference **within each gap-free run** at dt = 1/fps (never differenced across a `NaN` gap, so gaps cannot inflate it) |

## Bilateral symmetry (Robinson 1987)

For the four left/right pairs — **Knee, Hip, Elbow, Shoulder** — the Symmetry
Index is:

```
SI = 100 × (X_left − X_right) / (0.5 × (|X_left| + |X_right|))
```

Reported for `X` = Mean, ROM, and Peak angular velocity. **Positive = left-
dominant, negative = right-dominant**; `None` when both sides are missing/zero.
A common clinical flag is |SI| > 10%.

## Skeleton reference

Landmarks are COCO-17 (see [skeleton_mapping.md](skeleton_mapping.md)):

```
0 Nose   1 L Eye   2 R Eye   3 L Ear   4 R Ear
5 L Shoulder   6 R Shoulder   7 L Elbow   8 R Elbow   9 L Wrist   10 R Wrist
11 L Hip   12 R Hip   13 L Knee   14 R Knee   15 L Ankle   16 R Ankle
```
