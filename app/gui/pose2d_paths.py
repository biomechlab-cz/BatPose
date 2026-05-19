"""Path helpers used by the pose2d preview — kept Qt-free so they're easy to unit-test."""

from __future__ import annotations

from pathlib import Path


def derive_pose2d_paths(pose3d_path: str) -> tuple[str | None, str | None]:
    """
    Given a pose3d.npz path, infer the conventional pose2d_{left,right}.npz
    paths next to it. Returns (left, right) where each is the path if it
    exists on disk, else None.

    Pure Path/IO logic, no Qt dependency, so it can be exercised by tests
    that don't have PySide6 available.
    """
    p = Path(pose3d_path).parent
    left = p / "pose2d_left.npz"
    right = p / "pose2d_right.npz"
    return (
        str(left) if left.is_file() else None,
        str(right) if right.is_file() else None,
    )
