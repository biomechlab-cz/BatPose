"""CSV export for joint-angle time series.

Headless — no Qt — so the export logic can be reused by a future CLI tool
without bringing in any GUI dependencies.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .angles import ANGLE_DEFINITIONS


def angles_to_csv(
    path: str,
    angles: np.ndarray,
    fps: float,
    angle_names: list[str] | None = None,
) -> None:
    """Write a per-frame angle time series to a flat CSV.

    Columns:
        ``frame, time_s, person, <angle_1>, <angle_2>, …``

    NaN angle values are emitted as empty strings so spreadsheet tools
    (Excel, LibreOffice, pandas with ``na_values=""``) treat them as missing
    data rather than as the literal string ``"nan"``.

    Args:
        path:        destination CSV file path.  Parent directories are
                     created as needed.
        angles:      ``[T, P, N_ANGLES]`` float array.
        fps:         frame rate in Hz — used to compute the ``time_s`` column.
        angle_names: optional list of column headers.  Defaults to the
                     ``name`` field of each :data:`ANGLE_DEFINITIONS` entry.
    """
    if angles.ndim != 3:
        raise ValueError(f"angles must be [T, P, N]; got {angles.shape}")
    if not (fps > 0 and math.isfinite(fps)):
        raise ValueError(f"fps must be positive and finite; got {fps}")

    T, P, N = angles.shape
    if angle_names is None:
        angle_names = [a.name for a in ANGLE_DEFINITIONS]
    if len(angle_names) != N:
        raise ValueError(
            f"angle_names length {len(angle_names)} != angles' last dim {N}"
        )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["frame", "time_s", "person", *angle_names]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for t in range(T):
            time_s = t / fps
            for p in range(P):
                row: list = [t, f"{time_s:.6f}", p]
                for n in range(N):
                    v = float(angles[t, p, n])
                    # NaN → empty cell so spreadsheets show blank, not "nan"
                    row.append("" if math.isnan(v) else f"{v:.4f}")
                writer.writerow(row)
