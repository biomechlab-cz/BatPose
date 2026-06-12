"""One Euro Filter for temporal smoothing of noisy signals (ADR-004)."""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """
    One Euro Filter — adaptive low-pass filter for noisy input signals.

    Reference:
        Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
        for Noisy Input in Interactive Systems", CHI 2012.

    Usage:
        filt = OneEuroFilter(fps=30.0, min_cutoff=0.5, beta=0.05)
        for sample in signal:
            smoothed = filt(sample)
    """

    def __init__(
        self,
        fps: float,
        min_cutoff: float = 0.5,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        """
        Args:
            fps:        frame rate in Hz (must be > 0)
            min_cutoff: minimum cutoff frequency (Hz) — lower = smoother at rest
            beta:       speed coefficient — higher = less lag during fast motion
            d_cutoff:   cutoff for derivative low-pass (usually leave at 1.0 Hz)
        """
        if not (math.isfinite(fps) and fps > 0):
            raise ValueError(f"fps must be positive and finite, got {fps}")
        if not min_cutoff > 0:
            raise ValueError(f"min_cutoff must be positive, got {min_cutoff}")
        if not d_cutoff > 0:
            raise ValueError(f"d_cutoff must be positive, got {d_cutoff}")
        self.fps = fps
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, fps: float) -> float:
        """Compute smoothing factor alpha from cutoff frequency and sampling rate."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        dt = 1.0 / fps
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float) -> float:
        """Filter a single scalar sample, returning the filtered value."""
        if self._x_prev is None:
            self._x_prev = x
            return x

        # Derivative low-pass
        a_d = self._alpha(self.d_cutoff, self.fps)
        dx = (x - self._x_prev) * self.fps
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.fps)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        """Reset filter state (use between independent sequences)."""
        self._x_prev = None
        self._dx_prev = 0.0


def smooth_trajectory(
    traj: np.ndarray,
    fps: float,
    min_cutoff: float = 0.5,
    beta: float = 0.05,
    d_cutoff: float = 1.0,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply OneEuro filter independently to each scalar dimension of a trajectory.

    Args:
        traj:       [T, …] float array (any shape after first axis)
        fps:        frame rate in Hz
        min_cutoff: OneEuro min_cutoff (Hz)
        beta:       OneEuro beta
        d_cutoff:   OneEuro d_cutoff (Hz)
        valid:      optional [T] bool mask of REAL measurements.  Invalid frames
                    (e.g. rejected joints zeroed by the triangulator) are NOT fed
                    into the filter — they would otherwise act as genuine
                    zero-position samples and drag the filter state toward the
                    origin across tracking gaps.  Instead the last smoothed value
                    is held (matching the live-tracking behaviour); leading
                    invalid frames pass through unchanged.

    Returns:
        smoothed array with the same shape as *traj*
    """
    T = traj.shape[0]
    flat = traj.reshape(T, -1)  # [T, N]
    N = flat.shape[1]

    if valid is not None and valid.shape[0] != T:
        raise ValueError(f"valid mask length {valid.shape[0]} != trajectory length {T}")

    out = np.empty_like(flat)
    for n in range(N):
        filt = OneEuroFilter(fps, min_cutoff, beta, d_cutoff)
        last: float | None = None
        for t in range(T):
            if valid is not None and not valid[t]:
                # Gap: hold the last smoothed value (filter state untouched);
                # before any valid sample just pass the input through.
                out[t, n] = last if last is not None else float(flat[t, n])
                continue
            last = filt(float(flat[t, n]))
            out[t, n] = last

    return out.reshape(traj.shape)
