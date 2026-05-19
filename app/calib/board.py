"""Board detection: ChArUco and chessboard."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Supported ArUco dictionaries
ARUCO_DICTS: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
}


@dataclass
class DetectionResult:
    """Result of board detection in a single frame."""

    obj_pts: np.ndarray  # [N, 3] float32
    img_pts: np.ndarray  # [N, 2] float32
    # Unique identifier per detected point (ChArUco corner index for ChArUco,
    # a 0..N-1 range for chessboards).  Used by stereo calibration to intersect
    # the point set when the two cameras don't see the same subset of corners.
    ids: np.ndarray | None = None  # [N] int32
    # Diagnostic fields — filled even on partial / failed detection.
    # n_markers > 0 but len(img_pts)==0 means ArUco markers were found but
    # the ChArUco board layout didn't match (wrong squares_x/y or dictionary).
    n_markers: int = 0    # raw ArUco markers detected
    partial: bool = False # True = markers found but board not matched


class BoardDetector:
    """Abstract base for board detectors."""

    def detect(self, gray: np.ndarray) -> DetectionResult | None:
        raise NotImplementedError

    @property
    def board_cfg(self) -> dict:
        """Return board configuration dict (for calibration.yml)."""
        raise NotImplementedError

    @property
    def min_corners(self) -> int:
        """Minimum acceptable corner count for a valid frame."""
        return 6


class ChessboardDetector(BoardDetector):
    """Detect inner corners of a chessboard pattern."""

    def __init__(self, cols: int, rows: int, square_size: float):
        """
        Args:
            cols: number of inner corner columns (e.g. 9 for a 10-col board)
            rows: number of inner corner rows
            square_size: physical size of each square in metres
        """
        if cols < 1:
            raise ValueError(f"cols must be >= 1, got {cols}")
        if rows < 1:
            raise ValueError(f"rows must be >= 1, got {rows}")
        if square_size <= 0:
            raise ValueError(f"square_size must be positive, got {square_size}")
        self.cols = cols
        self.rows = rows
        self.square_size = square_size
        self.pattern_size = (cols, rows)

        # Build 3D object points in board frame (z=0 plane)
        obj = np.zeros((cols * rows, 3), np.float32)
        obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
        self._obj_pts = obj

        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        self._flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

    def detect(self, gray: np.ndarray) -> DetectionResult | None:
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, self._flags)
        if not ret or corners is None:
            return None
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._criteria)
        n = self._obj_pts.shape[0]
        return DetectionResult(
            obj_pts=self._obj_pts.copy(),
            img_pts=corners.reshape(-1, 2),
            ids=np.arange(n, dtype=np.int32),
        )

    @property
    def board_cfg(self) -> dict:
        return {
            "type": "checkerboard",
            "cols": self.cols,
            "rows": self.rows,
            "square_size": self.square_size,
        }

    @property
    def min_corners(self) -> int:
        return self.cols * self.rows


class CharucoDetector(BoardDetector):
    """Detect corners of a ChArUco board (preferred over plain chessboard)."""

    def __init__(
        self,
        squares_x: int,
        squares_y: int,
        square_size: float,
        marker_size: float,
        aruco_dict_name: str = "DICT_4X4_50",
    ):
        """
        Args:
            squares_x: number of chessboard squares in X
            squares_y: number of chessboard squares in Y
            square_size: chessboard square size in metres
            marker_size: ArUco marker size in metres (< square_size)
            aruco_dict_name: one of ARUCO_DICTS keys
        """
        if squares_x < 2 or squares_y < 2:
            raise ValueError(
                f"squares_x and squares_y must each be >= 2 to produce inner corners; "
                f"got squares_x={squares_x}, squares_y={squares_y}. "
                "A ChArUco board with fewer than 2 squares in any dimension has no inner "
                "corners and cannot be used for calibration."
            )
        if marker_size >= square_size:
            raise ValueError(
                f"marker_size ({marker_size}) must be strictly less than "
                f"square_size ({square_size})"
            )
        if aruco_dict_name not in ARUCO_DICTS:
            raise ValueError(
                f"Unknown ArUco dictionary: {aruco_dict_name!r}. "
                f"Valid choices are: {sorted(ARUCO_DICTS.keys())}"
            )
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_size = square_size
        self.marker_size = marker_size
        self.aruco_dict_name = aruco_dict_name

        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[aruco_dict_name])
        self._board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_size,
            marker_size,
            aruco_dict,
        )

        # Tune ArUco detection parameters for fisheye / wide-angle cameras.
        #
        # Default parameters assume a perspective camera with well-conditioned
        # perspective projections.  Fisheye lenses produce heavy barrel distortion
        # that:
        #   • Makes marker squares appear curved / trapezoid near the image edges
        #   • Reduces the effective pixel size of markers at typical distances
        #
        # Key changes from defaults:
        #   minMarkerPerimeterRate   0.03 → 0.02   slightly smaller markers accepted (fisheye)
        #   adaptiveThreshWinSizeMax 23   → 123    multi-scale sweep covers close-up boards
        #   adaptiveThreshWinSizeStep 10  (kept)   step=20 breaks small-marker detection
        #   polygonalApproxAccuracyRate  0.03 → 0.15  tolerate fisheye-curved marker edges
        #   cornerRefinementMethod   NONE → SUBPIX  sub-pixel corner accuracy
        #   errorCorrectionRate      0.6  (kept)  rate < 0.5 → 0 errors, kills fisheye
        params = cv2.aruco.DetectorParameters()
        # --- Fisheye-friendly geometry relaxations ---
        # minMarkerPerimeterRate: 0.02 instead of default 0.03 so markers near
        # the edges of a fisheye frame (which appear compressed) are still found.
        params.minMarkerPerimeterRate = 0.02        # default 0.03
        params.maxMarkerPerimeterRate = 4.0         # default 4.0 (keep)
        # adaptiveThreshWinSizeMax: must exceed the pixel width of a single checker
        # square so the thresholder bridges across the light→dark boundary.
        # At typical working distance squares are ~30–60 px, close-up ~100–150 px.
        # With step=10, max=123: windows 3,13,23,33,43,53,63,73,83,93,103,113,123.
        # 13 px is the sweet spot for medium-range markers (~6 px cells).
        # 33 px is the sweet spot for close-up markers (~19 px cells).
        params.adaptiveThreshWinSizeMin = 3         # default 3  (keep)
        params.adaptiveThreshWinSizeMax = 123       # default 23 → cover close-up boards too
        params.adaptiveThreshWinSizeStep = 10       # default 10 (keep) — step=20 skipped the
                                                     # 13 px and 33 px windows that are critical
                                                     # for small (medium-range) and close-up
                                                     # markers respectively
        # polygonalApproxAccuracyRate: fisheye distortion bends straight marker edges into
        # curves, so the polygon approximation must allow more deviation from a perfect
        # quadrilateral.  0.15 (vs default 0.03) is needed for ≥150° FOV lenses.
        params.polygonalApproxAccuracyRate = 0.15
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # errorCorrectionRate semantics: max_errors = floor(rate × minHammingDist / 2).
        # For DICT_6X6_250 minHammingDist ≥ 6, so rate=0.6 allows floor(0.6×3)=1 error.
        # Rate < 1/(minHammingDist) → 0 errors (pixel-perfect) — too strict for fisheye.
        # Keep default 0.6 (1-bit tolerance). Background false-positives that pass at
        # 0 errors also pass at 1; the ≥2-marker gate in detect() handles those instead.
        params.errorCorrectionRate = 0.6            # default (keep)

        # Pass via positional args — keyword names differ across OpenCV 4.x Python
        # bindings and using the wrong name causes a silent TypeError that is caught
        # by the caller and leaves self._detector = None with no visible feedback.
        charuco_params = cv2.aruco.CharucoParameters()
        # minMarkers=1: allow interpolating a ChArUco corner from a single adjacent
        # detected marker instead of requiring both neighbours.
        # Default=2 means if 10/17 markers are found (~59%), each corner has only
        # ~35% chance of having both neighbours → ~8 corners expected, below min_corners=12.
        # With minMarkers=1 the same 10 markers yield ~20 corners, well above threshold.
        try:
            charuco_params.minMarkers = 1
        except AttributeError:
            pass  # OpenCV < 4.8 — fall back to default (2)
        self._detector = cv2.aruco.CharucoDetector(self._board, charuco_params, params)

    def detect(self, gray: np.ndarray) -> DetectionResult | None:
        charuco_corners, charuco_ids, _marker_corners, marker_ids = self._detector.detectBoard(
            gray
        )

        n_markers = int(len(marker_ids)) if marker_ids is not None else 0

        # A single-marker "detection" in the background is almost always a false
        # positive — require at least 2 for the partial-detection diagnostic.
        _partial_threshold = 2

        if (
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_corners) < self.min_corners
        ):
            # Return a partial result when multiple ArUco markers were found —
            # this lets the UI distinguish "nothing at all" from "markers seen but
            # board config mismatch (wrong squares_x/y or dictionary)".
            if n_markers >= _partial_threshold:
                return DetectionResult(
                    obj_pts=np.empty((0, 3), np.float32),
                    img_pts=np.empty((0, 2), np.float32),
                    n_markers=n_markers,
                    partial=True,
                )
            return None

        # OpenCV API drift: matchImagePoints returns either (objPoints, imgPoints) in
        # 4.7/4.8 or (ret, objPoints, imgPoints) in some 4.x bindings.  Handle both.
        _ret = self._board.matchImagePoints(charuco_corners, charuco_ids)
        if len(_ret) == 2:
            obj_pts, img_pts = _ret
        else:
            _, obj_pts, img_pts = _ret
        if obj_pts is None or len(obj_pts) < self.min_corners:
            # Board layout didn't fit — markers were found but squares_x/y or
            # dictionary don't match the physical board.
            if n_markers >= _partial_threshold:
                return DetectionResult(
                    obj_pts=np.empty((0, 3), np.float32),
                    img_pts=np.empty((0, 2), np.float32),
                    n_markers=n_markers,
                    partial=True,
                )
            return None

        return DetectionResult(
            obj_pts=obj_pts.reshape(-1, 3).astype(np.float32),
            img_pts=img_pts.reshape(-1, 2).astype(np.float32),
            ids=np.asarray(charuco_ids).flatten().astype(np.int32),
            n_markers=n_markers,
        )

    @property
    def board_cfg(self) -> dict:
        return {
            "type": "charuco",
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_size": self.square_size,
            "marker_size": self.marker_size,
            "dictionary": self.aruco_dict_name,
        }

    @property
    def min_corners(self) -> int:
        # Require at least half the interior corners, capped by the total available.
        # This prevents min_corners from exceeding the board's actual corner count
        # on small boards (e.g. 3×3 only has 4 inner corners, not 6).
        inner = (self.squares_x - 1) * (self.squares_y - 1)
        return min(max(6, inner // 2), inner)


def make_detector(board_cfg: dict) -> BoardDetector:
    """Instantiate a detector from a board_cfg dict (as stored in calibration.yml)."""
    btype = board_cfg.get("type", "charuco")
    if btype == "charuco":
        required = {"squares_x", "squares_y", "square_size", "marker_size"}
        missing = required - board_cfg.keys()
        if missing:
            raise ValueError(
                f"board_cfg for 'charuco' board is missing required keys: "
                f"{sorted(missing)}. Required keys are: {sorted(required)}"
            )
        return CharucoDetector(
            squares_x=board_cfg["squares_x"],
            squares_y=board_cfg["squares_y"],
            square_size=board_cfg["square_size"],
            marker_size=board_cfg["marker_size"],
            aruco_dict_name=board_cfg.get("dictionary", "DICT_4X4_50"),
        )
    elif btype in ("checkerboard", "chessboard"):
        required = {"cols", "rows", "square_size"}
        missing = required - board_cfg.keys()
        if missing:
            raise ValueError(
                f"board_cfg for 'checkerboard' board is missing required keys: "
                f"{sorted(missing)}. Required keys are: {sorted(required)}"
            )
        return ChessboardDetector(
            cols=board_cfg["cols"],
            rows=board_cfg["rows"],
            square_size=board_cfg["square_size"],
        )
    else:
        raise ValueError(f"Unknown board type: {btype!r}")
