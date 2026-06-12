from __future__ import annotations

import pytest

from app.capture import SimulatorCapture


class TestSimulatorCapture:
    def test_exported_capture_source_produces_stereo_frames(self):
        cap = SimulatorCapture(fps=30.0, frame_size=(320, 180), max_frames=1)
        cap.start()
        try:
            frame = cap.read()
        finally:
            cap.stop()

        assert frame is not None
        assert frame.frame_left.shape == (180, 320, 3)
        assert frame.frame_right.shape == (180, 320, 3)
        assert frame.frame_index == 0
        assert frame.hw_timestamp_left_ns == 0
        assert frame.hw_timestamp_right_ns is not None
        assert frame.hw_delta_us is not None

    def test_stops_at_max_frames(self):
        cap = SimulatorCapture(fps=1000.0, frame_size=(64, 48), max_frames=1)
        cap.start()
        try:
            assert cap.read() is not None
            assert cap.read() is None
        finally:
            cap.stop()

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="fps must be positive"):
            SimulatorCapture(fps=0.0)
        with pytest.raises(ValueError, match="frame_size must be positive"):
            SimulatorCapture(frame_size=(0, 480))
