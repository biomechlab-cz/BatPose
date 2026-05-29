"""Biomechanical analysis utilities (joint angles, range-of-motion statistics, CSV export).

Pure-numpy module; no Qt / GUI dependencies — safe to import from headless
scripts and unit tests.
"""

from .angles import (
    ANGLE_DEFINITIONS,
    AngleDef,
    AngleStats,
    compute_joint_angles,
    compute_stats,
)
from .export import angles_to_csv

__all__ = [
    "ANGLE_DEFINITIONS",
    "AngleDef",
    "AngleStats",
    "compute_joint_angles",
    "compute_stats",
    "angles_to_csv",
]
