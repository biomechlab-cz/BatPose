"""
Joint-angle computation for biomechanical analysis (ADR-008).

Public API:
    ANGLE_DEFINITIONS    — list[AngleDef], single source of truth for which
                           angles are computed and in what order
    compute_joint_angles — vectorised angle extraction from a [T, P, 17, 3]
                           pose array, returning [T, P, N_ANGLES] degrees
    compute_stats        — per-angle / per-person min, max, mean, ROM

The 3D pose data is expected to be in the Z-up world frame (X lateral, Y
anterior, Z vertical) — the same frame the SkeletonViewer3D uses.  The
caller is responsible for applying the OpenCV→Z-up axis swap *before*
passing data into compute_joint_angles, since `pose3d.npz` files are
written in the original OpenCV camera frame (see ``viewer3d.set_data``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# COCO-17 joint indices (matching docs/skeleton_mapping.md):
#   0  Nose          5  L Shoulder   10 R Wrist     15 L Ankle
#   1  L Eye         6  R Shoulder   11 L Hip       16 R Ankle
#   2  R Eye         7  L Elbow      12 R Hip
#   3  L Ear         8  R Elbow      13 L Knee
#   4  R Ear         9  L Wrist      14 R Knee


@dataclass(frozen=True)
class AngleDef:
    """Definition of one biomechanical angle.

    For a standard 3-point joint angle, *indices* holds (proximal, vertex,
    distal) — the angle is the interior angle at *vertex* formed by the two
    segments running to *proximal* and *distal*.

    For special-cased virtual angles (currently only Trunk Inclination),
    *indices* is None and the computation is dispatched to a dedicated
    helper inside :func:`compute_joint_angles`.
    """

    name: str  # display name (e.g. "L Knee Flex")
    indices: tuple[int, int, int] | None  # (A, B, C) COCO-17 indices or None
    side: str  # "left" | "right" | "centre" — used for default plot colours


#: The full list of angles BatPose computes — order is significant; the
#: third axis of :func:`compute_joint_angles` output indexes into this list.
ANGLE_DEFINITIONS: list[AngleDef] = [
    AngleDef("L Knee Flex", (11, 13, 15), "left"),
    AngleDef("R Knee Flex", (12, 14, 16), "right"),
    AngleDef("L Hip Flex", (5, 11, 13), "left"),
    AngleDef("R Hip Flex", (6, 12, 14), "right"),
    AngleDef("L Elbow Flex", (5, 7, 9), "left"),
    AngleDef("R Elbow Flex", (6, 8, 10), "right"),
    AngleDef("L Shoulder Flex", (11, 5, 7), "left"),
    AngleDef("R Shoulder Flex", (12, 6, 8), "right"),
    AngleDef("Trunk Inclination", None, "centre"),
]

# Numerical floor used to keep the arccos argument inside [-1, 1] after
# normalisation — without it, sub-eps floating drift produces NaN.
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _three_point_angle(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Angle at vertex *b* between segments b→a and b→c, in degrees.

    Args:
        a, b, c: arrays broadcastable to a common shape ``(..., 3)``.

    Returns:
        Array of degrees in [0, 180] with shape ``a.shape[:-1]``.
    """
    v1 = a - b
    v2 = c - b
    # Norm along the last (xyz) axis.
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    # Dot product along the last axis.
    dot = np.einsum("...i,...i->...", v1, v2)
    cos_a = dot / (n1 * n2 + _EPS)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def _trunk_inclination(
    joints: np.ndarray, conf: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Angle of the trunk segment against the vertical Z-up axis.

    The trunk is the line from the mid-hip to the mid-shoulder.  An upright
    person yields ~0°, a person lying flat yields ~90°.

    Args:
        joints: ``[T, P, 17, 3]`` float array in the Z-up world frame.
        conf:   ``[T, P, 17]`` confidence array.

    Returns:
        Tuple ``(angle_deg, valid_mask)``:
            angle_deg  — ``[T, P]`` float angles in degrees
            valid_mask — ``[T, P]`` bool, True where conf>0 on both hips
                         (11, 12) AND both shoulders (5, 6)
    """
    # Mid-points along the lateral axis — both averaged in the [T, P, 3] frame.
    mid_hip = 0.5 * (joints[..., 11, :] + joints[..., 12, :])
    mid_sh = 0.5 * (joints[..., 5, :] + joints[..., 6, :])
    trunk = mid_sh - mid_hip  # [T, P, 3]
    norm = np.linalg.norm(trunk, axis=-1)
    # Z component over total length → cosine of angle with vertical.
    cos_a = trunk[..., 2] / (norm + _EPS)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_a))

    valid = (
        (conf[..., 5] > 0)
        & (conf[..., 6] > 0)
        & (conf[..., 11] > 0)
        & (conf[..., 12] > 0)
    )
    return angle.astype(np.float32), valid


def compute_joint_angles(
    joints3d: np.ndarray, conf3d: np.ndarray
) -> np.ndarray:
    """Compute all angles in :data:`ANGLE_DEFINITIONS` for every frame/person.

    Args:
        joints3d: ``[T, P, 17, 3]`` float — 3D positions in the **Z-up world
                   frame** (X lateral, Y anterior, Z vertical).  Apply the
                   OpenCV→Z-up swap before calling this function if the data
                   came directly from a `pose3d.npz` file.
        conf3d:   ``[T, P, 17]`` float — per-joint confidence ∈ [0, 1].

    Returns:
        ``[T, P, len(ANGLE_DEFINITIONS)]`` float32 array of degrees.
        Entries are ``np.nan`` for frames in which any flanking joint of the
        corresponding angle has ``conf3d == 0``.
    """
    joints3d = np.asarray(joints3d, dtype=np.float32)
    conf3d = np.asarray(conf3d, dtype=np.float32)

    if joints3d.ndim != 4 or joints3d.shape[-1] != 3:
        raise ValueError(
            f"joints3d must have shape [T, P, J, 3]; got {joints3d.shape}"
        )
    if conf3d.shape != joints3d.shape[:3]:
        raise ValueError(
            f"conf3d shape {conf3d.shape} does not match joints3d {joints3d.shape[:3]}"
        )

    T, P, J = conf3d.shape
    N = len(ANGLE_DEFINITIONS)
    out = np.full((T, P, N), np.nan, dtype=np.float32)

    for k, adef in enumerate(ANGLE_DEFINITIONS):
        if adef.indices is None:
            # Trunk inclination — special case handled separately.
            angle, valid = _trunk_inclination(joints3d, conf3d)
            out[..., k] = np.where(valid, angle, np.nan)
            continue

        a_idx, b_idx, c_idx = adef.indices
        if max(a_idx, b_idx, c_idx) >= J or min(a_idx, b_idx, c_idx) < 0:
            raise IndexError(
                f"Angle {adef.name!r} references joint index outside [0, {J})"
            )

        angle = _three_point_angle(
            joints3d[..., a_idx, :],
            joints3d[..., b_idx, :],
            joints3d[..., c_idx, :],
        )
        valid = (
            (conf3d[..., a_idx] > 0)
            & (conf3d[..., b_idx] > 0)
            & (conf3d[..., c_idx] > 0)
        )
        out[..., k] = np.where(valid, angle, np.nan).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AngleStats:
    """Summary statistics over the non-NaN frames of a single angle/person."""

    min_deg: float
    max_deg: float
    mean_deg: float
    rom_deg: float  # range of motion = max - min

    @property
    def is_finite(self) -> bool:
        """True iff every field is a finite real number."""
        return all(
            math.isfinite(v)
            for v in (self.min_deg, self.max_deg, self.mean_deg, self.rom_deg)
        )


def compute_stats(
    angles: np.ndarray, angle_idx: int, person_idx: int
) -> AngleStats | None:
    """Per-angle / per-person ROM statistics, ignoring NaN frames.

    Args:
        angles:     ``[T, P, N_ANGLES]`` float array (typically the return
                    value of :func:`compute_joint_angles`).
        angle_idx:  index into ``ANGLE_DEFINITIONS``.
        person_idx: which person to summarise.

    Returns:
        :class:`AngleStats` or ``None`` if every frame is NaN for the
        requested angle/person.
    """
    if angles.ndim != 3:
        raise ValueError(f"angles must be [T, P, N]; got {angles.shape}")
    series = angles[:, person_idx, angle_idx]
    finite = series[~np.isnan(series)]
    if finite.size == 0:
        return None
    mn = float(finite.min())
    mx = float(finite.max())
    return AngleStats(
        min_deg=mn,
        max_deg=mx,
        mean_deg=float(finite.mean()),
        rom_deg=mx - mn,
    )
