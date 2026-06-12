"""Unit tests for OneEuro temporal smoothing."""

import numpy as np

from app.recon3d.smooth import OneEuroFilter, smooth_trajectory


class TestOneEuroFilter:
    def test_passthrough_constant_signal(self):
        """Filter should converge to the constant value."""
        filt = OneEuroFilter(fps=30.0)
        # Seed with the constant
        for _ in range(10):
            out = filt(1.0)
        assert abs(out - 1.0) < 0.01

    def test_first_sample_passthrough(self):
        """First sample is always returned as-is."""
        filt = OneEuroFilter(fps=30.0)
        assert filt(42.0) == 42.0

    def test_smoothing_reduces_noise(self):
        """Filter should reduce white-noise variance around a DC signal."""
        rng = np.random.default_rng(0)
        fps = 30.0
        n = 300
        # DC signal: passes unchanged through any low-pass filter, so ALL
        # residual variance comes from filtered noise (no signal distortion).
        noisy = rng.normal(0, 1.0, n)

        filt = OneEuroFilter(fps=fps, min_cutoff=1.0, beta=0.0)
        smoothed = np.array([filt(x) for x in noisy])

        # Skip initial transient (first 30 frames)
        var_noisy = np.var(noisy[30:])
        var_smooth = np.var(smoothed[30:])
        assert var_smooth < 0.5 * var_noisy, (
            f"Filter should cut noise variance by ≥50%: "
            f"var_smooth={var_smooth:.4f}, var_noisy={var_noisy:.4f}"
        )

    def test_reset(self):
        """Reset should allow reuse from fresh state."""
        filt = OneEuroFilter(fps=30.0)
        filt(100.0)
        filt.reset()
        out = filt(5.0)
        assert out == 5.0  # first sample after reset

    def test_zero_fps_raises(self):
        """fps=0 must raise ValueError immediately (not a cryptic ZeroDivisionError)."""
        import pytest

        with pytest.raises(ValueError, match="fps must be positive"):
            OneEuroFilter(fps=0.0)

    def test_negative_fps_raises(self):
        """Negative fps must raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="fps must be positive"):
            OneEuroFilter(fps=-1.0)

    def test_zero_min_cutoff_raises(self):
        """min_cutoff=0 must raise ValueError (not a cryptic ZeroDivisionError)."""
        import pytest

        with pytest.raises(ValueError, match="min_cutoff must be positive"):
            OneEuroFilter(fps=30.0, min_cutoff=0.0)

    def test_negative_min_cutoff_raises(self):
        """Negative min_cutoff must raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="min_cutoff must be positive"):
            OneEuroFilter(fps=30.0, min_cutoff=-0.1)

    def test_zero_d_cutoff_raises(self):
        """d_cutoff=0 must raise ValueError immediately at construction time."""
        import pytest

        with pytest.raises(ValueError, match="d_cutoff must be positive"):
            OneEuroFilter(fps=30.0, d_cutoff=0.0)

    def test_negative_d_cutoff_raises(self):
        """Negative d_cutoff must raise ValueError at construction time."""
        import pytest

        with pytest.raises(ValueError, match="d_cutoff must be positive"):
            OneEuroFilter(fps=30.0, d_cutoff=-0.5)

    def test_nan_fps_raises(self):
        """fps=nan must raise ValueError (nan <= 0 is False, so old guard was insufficient)."""
        import pytest

        with pytest.raises(ValueError, match="fps must be positive"):
            OneEuroFilter(fps=float("nan"))

    def test_nan_min_cutoff_raises(self):
        """min_cutoff=nan must raise ValueError (nan <= 0 is False)."""
        import pytest

        with pytest.raises(ValueError, match="min_cutoff must be positive"):
            OneEuroFilter(fps=30.0, min_cutoff=float("nan"))

    def test_nan_d_cutoff_raises(self):
        """d_cutoff=nan must raise ValueError (nan <= 0 is False)."""
        import pytest

        with pytest.raises(ValueError, match="d_cutoff must be positive"):
            OneEuroFilter(fps=30.0, d_cutoff=float("nan"))

    def test_inf_fps_raises(self):
        """fps=inf must raise ValueError — inf > 0 so the old 'not > 0' guard missed it.

        Without the math.isfinite check, fps=inf passes the guard, then _alpha computes
        dt = 1/inf = 0.0 and raises a cryptic ZeroDivisionError on the second sample.
        """
        import pytest

        with pytest.raises(ValueError, match="fps must be positive and finite"):
            OneEuroFilter(fps=float("inf"))


class TestSmoothTrajectory:
    def test_output_shape_preserved(self):
        traj = np.random.rand(50, 3).astype(np.float32)
        out = smooth_trajectory(traj, fps=30.0)
        assert out.shape == traj.shape

    def test_4d_shape_preserved(self):
        traj = np.random.rand(100, 2, 17, 3).astype(np.float32)
        out = smooth_trajectory(traj, fps=30.0)
        assert out.shape == traj.shape

    def test_constant_signal_unchanged(self):
        """A perfectly constant trajectory should be (nearly) unchanged."""
        traj = np.ones((100, 3), dtype=np.float32) * 2.5
        out = smooth_trajectory(traj, fps=30.0)
        # After convergence (skip first few samples)
        np.testing.assert_allclose(out[20:], traj[20:], atol=1e-4)

    def test_single_frame_input(self):
        """T=1 edge case: the single frame must pass through unchanged."""
        traj = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = smooth_trajectory(traj, fps=30.0)
        assert out.shape == (1, 3)
        np.testing.assert_array_equal(out, traj)

    def test_zero_fps_raises(self):
        """fps=0 passed to smooth_trajectory must raise ValueError."""
        import pytest

        traj = np.ones((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="fps must be positive"):
            smooth_trajectory(traj, fps=0.0)

    def test_zero_min_cutoff_raises(self):
        """min_cutoff=0 must raise ValueError (not a cryptic ZeroDivisionError)."""
        import pytest

        traj = np.ones((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="min_cutoff must be positive"):
            smooth_trajectory(traj, fps=30.0, min_cutoff=0.0)

    def test_negative_min_cutoff_raises(self):
        """min_cutoff < 0 must raise ValueError."""
        import pytest

        traj = np.ones((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="min_cutoff must be positive"):
            smooth_trajectory(traj, fps=30.0, min_cutoff=-1.0)

    def test_zero_d_cutoff_raises(self):
        """d_cutoff=0 must raise ValueError (not a cryptic ZeroDivisionError)."""
        import pytest

        traj = np.ones((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="d_cutoff must be positive"):
            smooth_trajectory(traj, fps=30.0, d_cutoff=0.0)

    def test_negative_d_cutoff_raises(self):
        """d_cutoff < 0 must raise ValueError."""
        import pytest

        traj = np.ones((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="d_cutoff must be positive"):
            smooth_trajectory(traj, fps=30.0, d_cutoff=-1.0)


class TestValidMaskSmoothing:
    """Rejected (zeroed) joints must not be fed into the filter as real samples.

    The offline pipeline zeroes invalid joints; smoothing them unmasked drags
    the filter state toward the origin across tracking gaps.  With valid=…,
    gaps hold the last smoothed value (matching live tracking) and the filter
    state is untouched.
    """

    def _gappy(self, T: int = 60, lo: int = 20, hi: int = 40):
        traj = np.ones((T, 1), dtype=np.float32)
        valid = np.ones(T, dtype=bool)
        traj[lo:hi] = 0.0  # what the pipeline stores for rejected joints
        valid[lo:hi] = False
        return traj, valid

    def test_gap_holds_last_value(self):
        traj, valid = self._gappy()
        out = smooth_trajectory(traj, fps=30.0, min_cutoff=1.0, beta=0.5, valid=valid)
        np.testing.assert_allclose(out[19], 1.0, atol=1e-4)
        np.testing.assert_allclose(out[20:40], float(out[19, 0]))  # held, not dragged to 0
        np.testing.assert_allclose(out[45:], 1.0, atol=1e-3)  # clean recovery

    def test_unmasked_zeros_drag_toward_origin(self):
        """Documents the bug the mask fixes — same data WITHOUT the mask decays."""
        traj, _ = self._gappy()
        out = smooth_trajectory(traj, fps=30.0, min_cutoff=1.0, beta=0.5)
        assert out[35, 0] < 0.5  # filter pulled toward the fake zero samples

    def test_leading_invalid_passthrough(self):
        traj = np.ones((20, 1), dtype=np.float32)
        traj[:5] = 0.0
        valid = np.ones(20, dtype=bool)
        valid[:5] = False
        out = smooth_trajectory(traj, fps=30.0, valid=valid)
        np.testing.assert_allclose(out[:5], 0.0)  # nothing to hold yet
        np.testing.assert_allclose(out[5], 1.0, atol=1e-6)  # first real sample

    def test_no_mask_is_backward_compatible(self):
        traj = np.random.default_rng(0).random((30, 3)).astype(np.float32)
        np.testing.assert_array_equal(
            smooth_trajectory(traj, fps=30.0),
            smooth_trajectory(traj, fps=30.0, valid=np.ones(30, dtype=bool)),
        )

    def test_mask_length_mismatch_raises(self):
        import pytest

        traj = np.ones((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="valid mask length"):
            smooth_trajectory(traj, fps=30.0, valid=np.ones(7, dtype=bool))
