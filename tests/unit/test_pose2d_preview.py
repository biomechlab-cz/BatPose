"""Logic-only tests for pose2d_preview helpers (no Qt instantiation)."""

from __future__ import annotations

import os
from pathlib import Path

# Set Qt platform to offscreen so PySide6 imports without a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_derive_pose2d_paths_both_present(tmp_path: Path):
    """Both pose2d_*.npz files exist next to pose3d.npz → both paths returned."""
    pose3d = tmp_path / "pose3d.npz"
    pose3d.touch()
    pl = tmp_path / "pose2d_left.npz"
    pl.touch()
    pr = tmp_path / "pose2d_right.npz"
    pr.touch()

    try:
        from app.gui.pose2d_preview import derive_pose2d_paths
    except ImportError as e:
        if "PySide6" in str(e) or "shiboken" in str(e).lower():
            import pytest

            pytest.skip(f"PySide6 not available in this environment: {e}")
        raise

    left, right = derive_pose2d_paths(str(pose3d))
    assert left == str(pl)
    assert right == str(pr)


def test_derive_pose2d_paths_only_left(tmp_path: Path):
    """Only pose2d_left.npz exists → right is None."""
    pose3d = tmp_path / "pose3d.npz"
    pose3d.touch()
    pl = tmp_path / "pose2d_left.npz"
    pl.touch()

    try:
        from app.gui.pose2d_preview import derive_pose2d_paths
    except ImportError as e:
        if "PySide6" in str(e) or "shiboken" in str(e).lower():
            import pytest

            pytest.skip(f"PySide6 not available: {e}")
        raise

    left, right = derive_pose2d_paths(str(pose3d))
    assert left == str(pl)
    assert right is None


def test_derive_pose2d_paths_neither_present(tmp_path: Path):
    """Neither pose2d file exists → both None."""
    pose3d = tmp_path / "pose3d.npz"
    pose3d.touch()

    try:
        from app.gui.pose2d_preview import derive_pose2d_paths
    except ImportError as e:
        if "PySide6" in str(e) or "shiboken" in str(e).lower():
            import pytest

            pytest.skip(f"PySide6 not available: {e}")
        raise

    left, right = derive_pose2d_paths(str(pose3d))
    assert left is None
    assert right is None


def test_derive_pose2d_paths_works_for_nonexistent_pose3d(tmp_path: Path):
    """Even if the pose3d path itself doesn't exist, the function should
    just return None for the siblings — never raise."""
    bogus = tmp_path / "subdir" / "pose3d.npz"
    try:
        from app.gui.pose2d_preview import derive_pose2d_paths
    except ImportError as e:
        if "PySide6" in str(e) or "shiboken" in str(e).lower():
            import pytest

            pytest.skip(f"PySide6 not available: {e}")
        raise

    left, right = derive_pose2d_paths(str(bogus))
    assert left is None
    assert right is None
