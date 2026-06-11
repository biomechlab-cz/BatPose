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

import cv2
import numpy as np

from ..recon3d.triangulate import triangulate_points_dlt, undistort_points

# Minimum matched corners required to define a stable frame.
_MIN_CORNERS = 6


def _assemble_world_frame(
    x_dir: np.ndarray,
    normal: np.ndarray,
    origin: np.ndarray,
    long_m: float,
    short_m: float,
    extra: dict | None = None,
) -> dict[str, Any]:
    """Build the world-frame dict from a long-side direction, plane normal and origin.

    Orients Z "up" (toward the camera at the origin: ``z·origin < 0``), then
    orthonormalises X (long side), Z, and Y = Z × X (right-handed).  Returns the
    camera→world rotation/translation plus the axes and metadata.
    """
    x_dir = np.asarray(x_dir, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)

    z_dir = normal.copy()
    if float(z_dir @ origin) > 0.0:  # camera is "above" the floor → up = toward it
        z_dir = -z_dir
    x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-12)
    z_dir = z_dir - (z_dir @ x_dir) * x_dir
    z_dir = z_dir / (np.linalg.norm(z_dir) + 1e-12)
    y_dir = np.cross(z_dir, x_dir)
    y_dir = y_dir / (np.linalg.norm(y_dir) + 1e-12)

    B = np.column_stack([x_dir, y_dir, z_dir])  # world axes in camera coords
    R_world = B.T
    t_world = -B.T @ origin
    out = {
        "R": R_world.tolist(),
        "t": t_world.tolist(),
        "origin": origin.tolist(),
        "x_axis": x_dir.tolist(),
        "y_axis": y_dir.tolist(),
        "z_axis": z_dir.tolist(),
        "board_long_m": float(long_m),
        "board_short_m": float(short_m),
    }
    if extra:
        out.update(extra)
    return out


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


def _world_frame_from_matched(
    obj_pts: np.ndarray,
    pts_cam1: np.ndarray,
    centre_xy: tuple[float, float],
    extent_x_m: float,
    extent_y_m: float,
    method: str,
) -> dict[str, Any]:
    """Kabsch-fit a board pose and build the world frame.

    Args:
        obj_pts:    ``[N, 3]`` board-frame corner coords (Z=0), metres.
        pts_cam1:   ``[N, 3]`` the same corners triangulated into camera-1 3D.
        centre_xy:  physical board centre in *board* coords (depends on the
                    detector's object-point convention — see compute_world_frame).
        extent_x_m / extent_y_m: physical board side lengths along board X / Y.
        method:     tag stored in the result.

    Raises:
        ValueError if fewer than _MIN_CORNERS correspondences are given.
    """
    obj_pts = np.asarray(obj_pts, dtype=np.float64)
    pts_cam1 = np.asarray(pts_cam1, dtype=np.float64)
    if obj_pts.shape[0] < _MIN_CORNERS:
        raise ValueError(
            f"Need >= {_MIN_CORNERS} board corners to define a world frame; got {obj_pts.shape[0]}"
        )

    # Rigid fit board → camera-1.
    R_bc, t_bc = _kabsch(obj_pts, pts_cam1)
    fit_rms = float(np.sqrt(np.mean(np.sum((pts_cam1 - (obj_pts @ R_bc.T + t_bc)) ** 2, axis=1))))

    ex, ey, ez = R_bc[:, 0], R_bc[:, 1], R_bc[:, 2]  # board axes in cam-1

    # X = longer side; its in-plane orthogonal partner is the short side.
    if extent_x_m >= extent_y_m:
        x_dir, long_m, short_m = ex, extent_x_m, extent_y_m
    else:
        x_dir, long_m, short_m = ey, extent_y_m, extent_x_m

    origin = R_bc @ np.array([centre_xy[0], centre_xy[1], 0.0]) + t_bc

    return _assemble_world_frame(
        x_dir,
        ez,
        origin,
        long_m,
        short_m,
        extra={
            "n_corners": int(obj_pts.shape[0]),
            "fit_rms_m": fit_rms,
            "method": method,
        },
    )


def world_frame_from_corners(
    obj_pts: np.ndarray,
    pts_cam1: np.ndarray,
    squares_x: int,
    squares_y: int,
    square_size: float,
) -> dict[str, Any]:
    """Build the floor-board world frame from ChArUco corner correspondences.

    ChArUco convention: squares_x/squares_y are SQUARE counts and the object
    points put a board corner at (0, 0, 0) — inner corners run from (ss, ss),
    so the physical board centre is (squares_x·ss/2, squares_y·ss/2).

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
    Lx = squares_x * square_size
    Ly = squares_y * square_size
    return _world_frame_from_matched(
        obj_pts, pts_cam1, (Lx / 2.0, Ly / 2.0), Lx, Ly, "charuco_stereo"
    )


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

    # Board geometry depends on the detector's object-point convention:
    #   • ChArUco cfg (squares_x/squares_y = SQUARE counts): object points put a
    #     board corner at (0,0) and inner corners start at (ss, ss) → physical
    #     centre = (squares_x·ss/2, squares_y·ss/2).
    #   • Plain checkerboard cfg (cols/rows = INNER-CORNER counts): object
    #     points start at the first inner corner (0,0) → the board is symmetric
    #     about the inner-corner-grid centroid ((cols−1)·ss/2, (rows−1)·ss/2)
    #     and physically spans one extra square per side: (cols+1)·ss × (rows+1)·ss.
    # Reading checkerboard cfgs through the charuco keys used to fall into a
    # fallback that mixed metres with square counts (ss=1.0, sx=span+1 → a
    # 9×6/25 mm board became ~1.2 m wide).
    board = calib.get("board_cfg", {}) or {}
    ss = float(board.get("square_size", 0.0) or 0.0)
    sx = int(board.get("squares_x", 0) or 0)
    sy = int(board.get("squares_y", 0) or 0)
    cols = int(board.get("cols", 0) or 0)
    rows = int(board.get("rows", 0) or 0)

    if sx >= 2 and sy >= 2 and ss > 0:
        return world_frame_from_corners(obj_common, pts_cam1, sx, sy, ss)
    if cols >= 2 and rows >= 2 and ss > 0:
        centre = ((cols - 1) * ss / 2.0, (rows - 1) * ss / 2.0)
        return _world_frame_from_matched(
            obj_common,
            pts_cam1,
            centre,
            (cols + 1) * ss,
            (rows + 1) * ss,
            "checkerboard_stereo",
        )
    # Unknown / incomplete cfg — stay unit-correct: side lengths from the
    # detected corner extents (metres, slightly smaller than the physical
    # board) and the centre from their midpoint.
    mn = obj_common.min(axis=0)
    mx = obj_common.max(axis=0)
    centre = (float(mn[0] + mx[0]) / 2.0, float(mn[1] + mx[1]) / 2.0)
    return _world_frame_from_matched(
        obj_common,
        pts_cam1,
        centre,
        float(mx[0] - mn[0]),
        float(mx[1] - mn[1]),
        "corner_span_stereo",
    )


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
        length: axis length in metres.  Defaults to 0.6× the board's shorter
                side, so the triad spans a clearly-visible portion of the board
                (the board is often ~2 m from the cameras → small in the preview)
                while still fitting on it.

    Returns:
        ``[4, 3]`` float32: row 0 = origin, rows 1..3 = origin + length·axis.
    """
    origin = np.asarray(world_frame["origin"], dtype=np.float64)
    if length is None:
        length = 0.6 * float(world_frame.get("board_short_m", 0.2)) or 0.12
    pts = [origin]
    for key in ("x_axis", "y_axis", "z_axis"):
        pts.append(origin + length * np.asarray(world_frame[key], dtype=np.float64))
    return np.asarray(pts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Chessboard / monocular path (for a board too far for ArUco markers to decode)
# ---------------------------------------------------------------------------
#
# On a board lying on the floor across the room, the ArUco markers are too few
# pixels to detect/decode, so ChArUco yields nothing.  The plain checker corners
# of the SAME board are still found by cv2.findChessboardCorners, but those have
# no per-corner identity, so the stereo left↔right correspondence is ambiguous
# (the two views can order the grid differently → garbage triangulation).  We
# therefore recover the board pose with **monocular solvePnP** from the
# reference camera — no cross-camera correspondence required.


def board_squares(board_cfg: dict | None) -> tuple[int, int, float]:
    """Resolve a board config to ``(squares_x, squares_y, square_size)``.

    Handles both persisted conventions: ChArUco saves SQUARE counts
    (``squares_x``/``squares_y``); plain checkerboards save INNER-CORNER counts
    (``cols``/``rows``), which map to squares as ``cols+1`` / ``rows+1``.
    Returns ``(0, 0, 0.0)`` when the config is unusable — callers treat that as
    "chessboard geometry unknown".
    """
    b = board_cfg or {}
    ss = float(b.get("square_size", 0.0) or 0.0)
    sx = int(b.get("squares_x", 0) or 0)
    sy = int(b.get("squares_y", 0) or 0)
    if sx >= 2 and sy >= 2 and ss > 0:
        return sx, sy, ss
    cols = int(b.get("cols", 0) or 0)
    rows = int(b.get("rows", 0) or 0)
    if cols >= 2 and rows >= 2 and ss > 0:
        return cols + 1, rows + 1, ss
    return 0, 0, 0.0


def centered_board_objpoints(squares_x: int, squares_y: int, square_size: float) -> np.ndarray:
    """Inner-corner grid centred on the board centre (so solvePnP's t = centre).

    Ordered row-major to match ``cv2.findChessboardCorners((squares_x-1,
    squares_y-1))``: (squares_x-1) corners per row, (squares_y-1) rows.
    """
    cols, rows = squares_x - 1, squares_y - 1
    return np.array(
        [
            [(c - (cols - 1) / 2.0) * square_size, (r - (rows - 1) / 2.0) * square_size, 0.0]
            for r in range(rows)
            for c in range(cols)
        ],
        dtype=np.float64,
    )


def detect_chessboard_corners(
    gray: np.ndarray, squares_x: int, squares_y: int
) -> np.ndarray | None:
    """Detect the inner checker corners of the board; sub-pixel refined.

    Returns ``[(squares_x-1)*(squares_y-1), 2]`` float64 pixel corners, or None.
    More robust than ArUco at distance/oblique angle (corner geometry vs marker
    bit-decoding).
    """
    pat = (squares_x - 1, squares_y - 1)
    ok, corners = cv2.findChessboardCorners(
        gray, pat, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ok or corners is None:
        return None
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
    )
    return corners.reshape(-1, 2).astype(np.float64)


def world_frame_from_board_pose(
    R_bc: np.ndarray,
    t_bc: np.ndarray,
    squares_x: int,
    squares_y: int,
    square_size: float,
    extra: dict | None = None,
) -> dict[str, Any]:
    """Build a world frame from a board→camera pose (origin already at centre)."""
    R_bc = np.asarray(R_bc, dtype=np.float64)
    ex, ey = R_bc[:, 0], R_bc[:, 1]
    ez = R_bc[:, 2]
    Lx, Ly = squares_x * square_size, squares_y * square_size
    if Lx >= Ly:
        x_dir, long_m, short_m = ex, Lx, Ly
    else:
        x_dir, long_m, short_m = ey, Ly, Lx
    return _assemble_world_frame(
        x_dir, ez, np.asarray(t_bc, dtype=np.float64), long_m, short_m, extra
    )


def vote_world_frames(
    frames: list[dict],
    angle_tol_deg: float = 15.0,
    min_agreement: float = 0.6,
) -> dict[str, Any]:
    """Majority-vote a single world frame from per-frame candidates.

    Single-view planar PnP has a two-fold pose ambiguity: on ~10 % of frames the
    recovered board normal flips ~110° with NO reprojection-error signature
    (measured on hardware: flipped frames had identical RMS to good ones, and
    the two IPPE branches differed by only 0.1 px — so the error metric cannot
    disambiguate).  The flips are temporally sporadic, so voting over a short
    burst of frames removes them; averaging the winning cluster also reduces
    origin noise by ~1/√N.

    Clustering is by board normal (``z_axis``) around the medoid candidate; the
    chessboard's 180° in-plane ambiguity is folded by aligning each member's X
    to the medoid's before averaging.

    Args:
        frames: world-frame dicts (≥ 1), e.g. from compute_world_frame_monocular.
        angle_tol_deg: cluster radius around the medoid normal.
        min_agreement: required fraction of candidates in the winning cluster.

    Returns:
        Fused world-frame dict with extra keys ``n_votes``, ``n_agree``,
        ``vote_agreement`` and a ``method`` suffixed ``"_voted"``.

    Raises:
        ValueError if no candidates are given, agreement is below
        *min_agreement*, or the in-plane X direction is ambiguous.
    """
    if not frames:
        raise ValueError("no world-frame candidates to vote on")
    if len(frames) == 1:
        out = dict(frames[0])
        out.update({"n_votes": 1, "n_agree": 1, "vote_agreement": 1.0})
        return out

    Z = np.asarray([f["z_axis"] for f in frames], dtype=np.float64)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    # Medoid = the candidate whose normal agrees best with all the others.
    S = Z @ Z.T
    medoid = int(np.argmax(S.sum(axis=1)))
    keep = S[medoid] >= np.cos(np.radians(angle_tol_deg))
    agreement = float(keep.mean())
    if agreement < min_agreement:
        raise ValueError(
            f"board pose is ambiguous: only {int(keep.sum())}/{len(frames)} frames "
            f"agree on the floor orientation. Move the board (or camera) so the "
            f"board is seen less obliquely, then try again."
        )

    sel = [f for f, k in zip(frames, keep) if k]
    med = frames[medoid]
    # Fold the 180° in-plane ambiguity onto the medoid's X before averaging.
    x_ref = np.asarray(med["x_axis"], dtype=np.float64)
    xs = []
    for f in sel:
        x = np.asarray(f["x_axis"], dtype=np.float64)
        xs.append(x if float(x @ x_ref) >= 0.0 else -x)
    x_mean = np.mean(xs, axis=0)
    if float(np.linalg.norm(x_mean)) < 0.5:
        raise ValueError("board X direction is ambiguous across frames — try again")
    z_mean = np.mean([np.asarray(f["z_axis"], dtype=np.float64) for f in sel], axis=0)
    o_mean = np.mean([np.asarray(f["origin"], dtype=np.float64) for f in sel], axis=0)

    extra: dict[str, Any] = {
        "n_votes": len(frames),
        "n_agree": int(keep.sum()),
        "vote_agreement": agreement,
        "method": str(med.get("method", "")) + "_voted",
    }
    rms = [float(f["fit_rms_px"]) for f in sel if "fit_rms_px" in f]
    if rms:
        extra["fit_rms_px"] = float(np.mean(rms))
    if med.get("n_corners") is not None:
        extra["n_corners"] = int(med["n_corners"])

    return _assemble_world_frame(
        x_mean,
        z_mean,
        o_mean,
        float(med["board_long_m"]),
        float(med["board_short_m"]),
        extra,
    )


def compute_world_frame_monocular(
    img_pts: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    squares_x: int,
    squares_y: int,
    square_size: float,
    fisheye: bool,
    cam_to_ref: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """World frame from one camera's chessboard corners via solvePnP.

    The board pose is recovered in the coordinate frame of the camera whose
    (K, D) are given.  The world frame must live in camera-1 (left) coords (the
    triangulation / pose3d frame), so:
      • pass the LEFT camera's (K1, D1) and ``cam_to_ref=None``; or
      • pass the RIGHT camera's (K2, D2) and ``cam_to_ref=(R, T)`` — the stereo
        extrinsics — and the recovered pose is transformed into camera-1
        coords (``p_cam1 = Rᵀ(p_cam2 − T)``) before the frame is built.

    This lets "Set coordinate system" succeed when only one camera sees the
    board clearly (e.g. it sits near the left frame's edge but is central in the
    right).

    Raises ValueError if the corner count doesn't match the board.
    """
    objp = centered_board_objpoints(squares_x, squares_y, square_size)
    img_pts = np.asarray(img_pts, dtype=np.float64)
    if img_pts.shape[0] != objp.shape[0]:
        raise ValueError(f"expected {objp.shape[0]} chessboard corners, got {img_pts.shape[0]}")

    Kf = np.asarray(K, dtype=np.float64)
    Df = np.asarray(D, dtype=np.float64)
    if fisheye:
        # Undistort to normalized rays, then PnP with identity intrinsics.
        und = cv2.fisheye.undistortPoints(img_pts.reshape(-1, 1, 2), Kf, Df.reshape(4, 1))
        ok, rvec, tvec = cv2.solvePnP(
            objp, und.reshape(-1, 1, 2), np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        proj, _ = cv2.fisheye.projectPoints(
            objp.reshape(-1, 1, 3), rvec, tvec, Kf, Df.reshape(4, 1)
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            objp, img_pts.reshape(-1, 1, 2), Kf, Df, flags=cv2.SOLVEPNP_ITERATIVE
        )
        proj, _ = cv2.projectPoints(objp, rvec, tvec, Kf, Df)
    if not ok:
        raise ValueError("solvePnP failed to recover the board pose")

    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts) ** 2, axis=1))))
    R_bc, _ = cv2.Rodrigues(rvec)
    t_bc = tvec.reshape(3)
    method = "chessboard_pnp"

    if cam_to_ref is not None:
        # Transform board→thisCam into board→cam1 using the stereo extrinsics.
        R = np.asarray(cam_to_ref[0], dtype=np.float64)
        T = np.asarray(cam_to_ref[1], dtype=np.float64).reshape(3)
        R_bc = R.T @ R_bc
        t_bc = R.T @ (t_bc - T)
        method = "chessboard_pnp_ref"

    return world_frame_from_board_pose(
        R_bc,
        t_bc,
        squares_x,
        squares_y,
        square_size,
        extra={"n_corners": int(img_pts.shape[0]), "fit_rms_px": rms, "method": method},
    )
