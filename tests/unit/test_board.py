"""Unit tests for board detection utilities."""

import numpy as np
import pytest

from app.calib.board import CharucoDetector, ChessboardDetector, make_detector


class TestMakeDetector:
    def test_charuco(self):
        det = make_detector(
            {
                "type": "charuco",
                "squares_x": 7,
                "squares_y": 5,
                "square_size": 0.04,
                "marker_size": 0.03,
                "dictionary": "DICT_4X4_50",
            }
        )
        assert isinstance(det, CharucoDetector)

    def test_checkerboard(self):
        det = make_detector(
            {
                "type": "checkerboard",
                "cols": 9,
                "rows": 6,
                "square_size": 0.025,
            }
        )
        assert isinstance(det, ChessboardDetector)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown board type"):
            make_detector({"type": "hexagon"})

    def test_charuco_missing_key_raises_valueerror(self):
        """BUG-W: missing required key in charuco board_cfg must raise clear ValueError."""
        with pytest.raises(ValueError, match="missing required keys"):
            make_detector(
                {
                    "type": "charuco",
                    "squares_x": 7,
                    "squares_y": 5,
                    # square_size and marker_size intentionally omitted
                }
            )

    def test_checkerboard_missing_key_raises_valueerror(self):
        """BUG-W: missing required key in checkerboard board_cfg must raise clear ValueError."""
        with pytest.raises(ValueError, match="missing required keys"):
            make_detector(
                {
                    "type": "checkerboard",
                    "cols": 9,
                    # rows and square_size intentionally omitted
                }
            )


class TestChessboardDetector:
    def _make_chessboard_image(self, cols: int, rows: int, square_px: int = 60):
        """Render a synthetic chessboard image with exactly cols×rows inner corners."""
        # (cols+1)×(rows+1) squares yield cols×rows inner corners.
        # Add a one-square white border so edge corners are detectable.
        border = square_px
        board_w = (cols + 1) * square_px
        board_h = (rows + 1) * square_px
        img = np.ones((board_h + 2 * border, board_w + 2 * border), dtype=np.uint8) * 255

        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    y0 = border + r * square_px
                    x0 = border + c * square_px
                    img[y0 : y0 + square_px, x0 : x0 + square_px] = 0

        return img

    def test_detect_synthetic(self):
        cols, rows = 9, 6
        det = ChessboardDetector(cols=cols, rows=rows, square_size=0.025)
        img = self._make_chessboard_image(cols, rows)
        result = det.detect(img)
        assert result is not None
        assert result.obj_pts.shape == (cols * rows, 3)
        assert result.img_pts.shape == (cols * rows, 2)

    def test_returns_none_on_blank(self):
        det = ChessboardDetector(cols=9, rows=6, square_size=0.025)
        blank = np.ones((480, 640), dtype=np.uint8) * 128
        assert det.detect(blank) is None

    def test_board_cfg(self):
        det = ChessboardDetector(cols=9, rows=6, square_size=0.025)
        cfg = det.board_cfg
        assert cfg["type"] == "checkerboard"
        assert cfg["cols"] == 9


class TestCharucoDetectorMarkerSizeValidation:
    """BUG-M: CharucoDetector must reject marker_size >= square_size."""

    def test_marker_equal_to_square_raises(self):
        """marker_size == square_size is physically impossible; must raise ValueError."""
        with pytest.raises(ValueError, match="marker_size"):
            CharucoDetector(squares_x=7, squares_y=5, square_size=0.04, marker_size=0.04)

    def test_marker_larger_than_square_raises(self):
        """marker_size > square_size is physically impossible; must raise ValueError."""
        with pytest.raises(ValueError, match="marker_size"):
            CharucoDetector(squares_x=7, squares_y=5, square_size=0.04, marker_size=0.05)

    def test_valid_marker_size_accepted(self):
        """marker_size < square_size is valid and should not raise."""
        det = CharucoDetector(squares_x=7, squares_y=5, square_size=0.04, marker_size=0.03)
        assert det.marker_size == 0.03


class TestCharucoDetectorSquaresValidation:
    """BUG-Q: CharucoDetector must reject squares_x < 2 or squares_y < 2.

    A board with fewer than 2 squares in any dimension has 0 inner corners,
    causing min_corners to evaluate to 0 so the detect filter never rejects
    anything — empty DetectionResult arrays would pass through to calibration.
    """

    def test_squares_x_one_raises(self):
        """squares_x=1 → 0 inner corners in X → must raise ValueError."""
        with pytest.raises(ValueError, match="squares_x"):
            CharucoDetector(squares_x=1, squares_y=5, square_size=0.04, marker_size=0.03)

    def test_squares_y_one_raises(self):
        """squares_y=1 → 0 inner corners in Y → must raise ValueError."""
        with pytest.raises(ValueError, match="squares_y"):
            CharucoDetector(squares_x=5, squares_y=1, square_size=0.04, marker_size=0.03)

    def test_minimum_valid_board_accepted(self):
        """2×2 is the smallest valid board (1 inner corner) and must be accepted."""
        det = CharucoDetector(squares_x=2, squares_y=2, square_size=0.04, marker_size=0.03)
        assert det.squares_x == 2
        assert det.squares_y == 2


class TestChessboardDetectorValidation:
    """BUG-S: ChessboardDetector must reject cols/rows < 1 and square_size <= 0.

    Zero or negative values produce empty object-point arrays or degenerate 3D
    calibration targets, causing cryptic cv2.error failures deep inside calibration.
    """

    def test_zero_cols_raises(self):
        """cols=0 → empty pattern_size → cryptic cv2 error; must raise ValueError."""
        with pytest.raises(ValueError, match="cols"):
            ChessboardDetector(cols=0, rows=7, square_size=0.025)

    def test_negative_cols_raises(self):
        with pytest.raises(ValueError, match="cols"):
            ChessboardDetector(cols=-1, rows=7, square_size=0.025)

    def test_zero_rows_raises(self):
        """rows=0 → empty pattern_size → cryptic cv2 error; must raise ValueError."""
        with pytest.raises(ValueError, match="rows"):
            ChessboardDetector(cols=9, rows=0, square_size=0.025)

    def test_zero_square_size_raises(self):
        """square_size=0 → all object points at origin → degenerate calibration."""
        with pytest.raises(ValueError, match="square_size"):
            ChessboardDetector(cols=9, rows=6, square_size=0.0)

    def test_negative_square_size_raises(self):
        with pytest.raises(ValueError, match="square_size"):
            ChessboardDetector(cols=9, rows=6, square_size=-0.025)

    def test_valid_params_accepted(self):
        """Standard 9×6 board with 25 mm squares must be constructed without error."""
        det = ChessboardDetector(cols=9, rows=6, square_size=0.025)
        assert det.cols == 9
        assert det.rows == 6
        assert det.square_size == 0.025


class TestCharucoDetectorArucoDictValidation:
    """BUG-Z: CharucoDetector must reject unknown aruco_dict_name values."""

    def test_unknown_dict_name_raises(self):
        """An unrecognised dictionary name must raise ValueError with a clear message."""
        with pytest.raises(ValueError, match="Unknown ArUco dictionary"):
            CharucoDetector(
                squares_x=7,
                squares_y=5,
                square_size=0.04,
                marker_size=0.03,
                aruco_dict_name="DICT_INVALID",
            )

    def test_known_dict_name_accepted(self):
        """A known dictionary name must not raise."""
        det = CharucoDetector(
            squares_x=7,
            squares_y=5,
            square_size=0.04,
            marker_size=0.03,
            aruco_dict_name="DICT_4X4_50",
        )
        assert det.aruco_dict_name == "DICT_4X4_50"


class TestCharucoDetectorMinCorners:
    """BUG-G: min_corners must never exceed the board's actual inner corner count."""

    def test_standard_board_min_corners(self):
        """7×5 board: 24 inner corners; min_corners should be 12 (half)."""
        det = CharucoDetector(squares_x=7, squares_y=5, square_size=0.04, marker_size=0.03)
        inner = (7 - 1) * (5 - 1)  # 24
        assert det.min_corners == 12
        assert det.min_corners <= inner

    def test_small_board_min_corners_does_not_exceed_available(self):
        """3×3 board: only 4 inner corners — min_corners must not exceed 4."""
        det = CharucoDetector(squares_x=3, squares_y=3, square_size=0.04, marker_size=0.03)
        inner = (3 - 1) * (3 - 1)  # 4
        assert det.min_corners <= inner, (
            f"min_corners={det.min_corners} exceeds available inner corners={inner}; "
            "no detection would ever be accepted on this board"
        )

    def test_tiny_board_min_corners_does_not_exceed_available(self):
        """2×2 board: only 1 inner corner — min_corners must not exceed 1."""
        det = CharucoDetector(squares_x=2, squares_y=2, square_size=0.04, marker_size=0.03)
        inner = (2 - 1) * (2 - 1)  # 1
        assert det.min_corners <= inner
