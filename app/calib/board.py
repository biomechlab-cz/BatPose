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
        return DetectionResult(
            obj_pts=self._obj_pts.copy(),
            img_pts=corners.reshape(-1, 2),
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
        self._detector = cv2.aruco.CharucoDetector(self._board)

    def detect(self, gray: np.ndarray) -> DetectionResult | None:
        charuco_corners, charuco_ids, _marker_corners, _marker_ids = self._detector.detectBoard(
            gray
        )
        if (
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_corners) < self.min_corners
        ):
            return None

        ret, obj_pts, img_pts = self._board.matchImagePoints(charuco_corners, charuco_ids)
        if not ret or obj_pts is None or len(obj_pts) < self.min_corners:
            return None

        return DetectionResult(
            obj_pts=obj_pts.reshape(-1, 3).astype(np.float32),
            img_pts=img_pts.reshape(-1, 2).astype(np.float32),
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
