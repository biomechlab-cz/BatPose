"""
Floor-board world coordinate system for stereo reconstruction.

A ChArUco/chessboard placed flat on the floor (visible to both cameras) defines
a physical world frame:

    • origin  — the centre of the board
    • X axis  — along the board's **longer** side
    • Y axis  — along the board's **shorter** side (right-handed with X, Z)
    • Z axis  — **up** (out of the floor, toward the cameras)

The transform that maps a point from the camera-1 OpenCV frame (the frame the
triangulator works in) into this world frame is stored in ``calibration.yml``
under ``world_frame`` and applied by the reconstruction pipeline so that every
downstream consumer (3D viewer, biomech angles, CSV export) works in real,
physically-meaningful floor coordinates.  If no world frame is set, the pipeline
falls back to the previous behaviour (camera-1 frame + the viewer's Z-up swap).

The board pose is recovered by **triangulating the detected board corners from
both cameras** into camera-1 3D (more robust on fisheye rigs than monocular
solvePnP) and then fitting a rigid board→camera transform (Kabsch).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..recon3d.triangulate import triangulate_points_dlt, undistort_points

# Minimum matched corners required to define a stable frame.
_MIN_CORNERS = 6


# ---------------------------------------------------------------------------
# Rigid alignment
# ---------------------------------------------------------------------------


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-fit rigid transform (R, t) such that ``dst ≈ R @ src + t``.

    Args:
        src: ``[N, 3]`` source points (board frame).
        dst: ``[N, 3]`` destination points (camera-1 frame).

    Returns:
        (R [3,3], t [3]) — a proper rotation (det = +1) and translation.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cd - R @ cs
    return R, t


# ---------------------------------------------------------------------------
# World frame from board corners
# ---------------------------------------------------------------------------


def world_frame_from_corners(
    obj_pts: np.ndarray,
    pts_cam1: np.ndarray,
    squares_x: int,
    squares_y: int,
    square_size: float,
) -> dict[str, Any]:
    """Build the floor-board world frame from corner correspondences.

    Args:
        obj_pts:   ``[N, 3]`` board-frame corner coords (Z=0), metres.
        pts_cam1:  ``[N, 3]`` the same corners triangulated into camera-1 3D.
        squares_x, squares_y: board square counts.
        square_size: square edge length (metres).

    Returns:
        dict with:
            R          — 3×3 rotation, camera-1 → world (list of lists)
            t          — 3 translation, camera-1 → world (list)
            origin     — board centre in camera-1 coords (list)
            x_axis/y_axis/z_axis — world axes expressed in camera-1 coords (lists)
            board_long_m / board_short_m — physical side lengths
            n_corners  — number of corners used
            fit_rms_m  — RMS residual of the rigid board fit (metres)

    Raises:
        ValueError if fewer than _MIN_CORNERS correspondences are given.
    """
    obj_pts = np.asarray(obj_pts, dtype=np.float64)
    pts_cam1 = np.asarray(pts_cam1, dtype=np.float64)
    if obj_pts.shape[0] < _MIN_CORNERS:
        raise ValueError(
            f"Need >= {_MIN_CORNERS} board corners to define a world frame; "
            f"got {obj_pts.shape[0]}"
        )

    # Rigid fit board → camera-1.
    R_bc, t_bc = _kabsch(obj_pts, pts_cam1)
    fit_rms = float(np.sqrt(np.mean(np.sum((pts_cam1 - (obj_pts @ R_bc.T + t_bc)) ** 2, axis=1))))

    ex, ey, ez = R_bc[:, 0], R_bc[:, 1], R_bc[:, 2]  # board axes in cam-1

    Lx = squares_x * square_size  # physical extent along board X
    Ly = squares_y * square_size  # physical extent along board Y

    # X = longer side, the in-plane orthogonal partner becomes the short side.
    if Lx >= Ly:
        x_dir = ex
        long_m, short_m = Lx, Ly
    else:
        x_dir = ey
        long_m, short_m = Ly, Lx

    # Board centre in camera-1 (origin convention: a board corner at (0,0,0),
    # so the physical centre is at (Lx/2, Ly/2, 0) — also the corner centroid).
    origin = R_bc @ np.array([Lx / 2.0, Ly / 2.0, 0.0]) + t_bc

    # Z points "up" = away from the floor, toward the cameras.  The camera-1
    # centre is at the origin, the board centre is at `origin`, so the
    # board→camera direction is -origin.  Orient the board normal to agree
    # with it (robust to any camera tilt; no Y-down/Z-forward assumption).
    z_dir = ez.copy()
    if float(z_dir @ origin) > 0.0:
        z_dir = -z_dir

    # Orthonormalise: X first, then Z ⟂ X, then Y = Z × X (right-handed, along
    # the short side).
    x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-12)
    z_dir = z_dir - (z_dir @ x_dir) * x_dir
    z_dir = z_dir / (np.linalg.norm(z_dir) + 1e-12)
    y_dir = np.cross(z_dir, x_dir)
    y_dir = y_dir / (np.linalg.norm(y_dir) + 1e-12)

    # Basis whose columns are the world axes in camera-1 coords:
    #   p_cam1 = B @ p_world + origin   ⇒   p_world = Bᵀ (p_cam1 − origin)
    B = np.column_stack([x_dir, y_dir, z_dir])
    R_world = B.T
    t_world = -B.T @ origin

    return {
        "R": R_world.tolist(),
        "t": t_world.tolist(),
        "origin": origin.tolist(),
        "x_axis": x_dir.tolist(),
        "y_axis": y_dir.tolist(),
        "z_axis": z_dir.tolist(),
        "board_long_m": float(long_m),
        "board_short_m": float(short_m),
        "n_corners": int(obj_pts.shape[0]),
        "fit_rms_m": fit_rms,
    }


def _match_corners(det_left, det_right):
    """Return (obj_pts, img_l, img_r) for corners seen by BOTH cameras.

    Matches by ChArUco corner id when available; otherwise (chessboard) assumes
    identical ordering and matches by index.
    """
    ol, il = det_left.obj_pts, det_left.img_pts
    orr, ir = det_right.obj_pts, det_right.img_pts
    if getattr(det_left, "ids", None) is not None and getattr(det_right, "ids", None) is not None:
        ids_l = {int(v): k for k, v in enumerate(det_left.ids)}
        ids_r = {int(v): k for k, v in enumerate(det_right.ids)}
        common = sorted(set(ids_l) & set(ids_r))
        li = [ids_l[c] for c in common]
        ri = [ids_r[c] for c in common]
        return ol[li], il[li], ir[ri]
    # Chessboard: same number/order of corners in both views.
    n = min(len(ol), len(orr))
    return ol[:n], il[:n], ir[:n]


def compute_world_frame(det_left, det_right, calib: dict) -> dict[str, Any]:
    """Compute the floor-board world frame from a stereo detection.

    Args:
        det_left, det_right: ``DetectionResult`` from each camera (same board).
        calib: loaded calibration dict (K1,D1,K2,D2,R,T,lens_model,board_cfg).

    Returns:
        world-frame dict (see :func:`world_frame_from_corners`).

    Raises:
        ValueError if too few corners are seen by both cameras.
    """
    obj_common, img_l, img_r = _match_corners(det_left, det_right)
    if len(obj_common) < _MIN_CORNERS:
        raise ValueError(
            f"Only {len(obj_common)} board corner(s) seen by both cameras "
            f"(need >= {_MIN_CORNERS}). Move the board so both cameras see it clearly."
        )

    fisheye = calib.get("lens_model", "standard") == "fisheye"
    K1, D1, K2, D2 = calib["K1"], calib["D1"], calib["K2"], calib["D2"]
    R, T = calib["R"], calib["T"]

    # Triangulate the matched corners into camera-1 3D.
    pl = undistort_points(img_l, K1, D1, fisheye=fisheye)
    pr = undistort_points(img_r, K2, D2, fisheye=fisheye)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, T.reshape(3, 1)])
    pts_cam1 = triangulate_points_dlt(P1.astype(np.float64), P2.astype(np.float64), pl, pr)

    board = calib.get("board_cfg", {})
    sx = int(board.get("squares_x", 0))
    sy = int(board.get("squares_y", 0))
    ss = float(board.get("square_size", 0.0))
    if sx < 2 or sy < 2 or ss <= 0:
        # Fall back to the board's actual corner extent if cfg is incomplete.
        span = obj_common.max(axis=0) - obj_common.min(axis=0)
        ss = 1.0
        sx = float(span[0]) + 1.0
        sy = float(span[1]) + 1.0

    return world_frame_from_corners(obj_common, pts_cam1, sx, sy, ss)


# ---------------------------------------------------------------------------
# Applying the frame
# ---------------------------------------------------------------------------


def apply_world_frame(points_cam1: np.ndarray, world_frame: dict) -> np.ndarray:
    """Map camera-1 points into the world frame.

    Args:
        points_cam1: array with last axis = 3 (any leading shape).
        world_frame: dict with ``R`` (3×3) and ``t`` (3).

    Returns:
        array of the same shape, in world coordinates.
    """
    R = np.asarray(world_frame["R"], dtype=np.float32)
    t = np.asarray(world_frame["t"], dtype=np.float32)
    pts = np.asarray(points_cam1, dtype=np.float32)
    return pts @ R.T + t


def axis_endpoints_cam1(world_frame: dict, length: float | None = None) -> np.ndarray:
    """Return [origin, +X, +Y, +Z] endpoints in camera-1 coords for drawing.

    Args:
        world_frame: world-frame dict.
        length: axis length in metres.  Defaults to a quarter of the board's
                shorter side (so the triad sits neatly on the board).

    Returns:
        ``[4, 3]`` float32: row 0 = origin, rows 1..3 = origin + length·axis.
    """
    origin = np.asarray(world_frame["origin"], dtype=np.float64)
    if length is None:
        length = 0.25 * float(world_frame.get("board_short_m", 0.2)) or 0.05
    pts = [origin]
    for key in ("x_axis", "y_axis", "z_axis"):
        pts.append(origin + length * np.asarray(world_frame[key], dtype=np.float64))
    return np.asarray(pts, dtype=np.float32)
