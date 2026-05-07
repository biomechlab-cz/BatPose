"""
FLIR camera capture via PySpin (Spinnaker SDK).

This module is imported only if PySpin is available.
If the FLIR SDK is not installed, the import will fail gracefully and
VideoSimulator will be used instead (see app/capture/__init__.py).

Install notes:
- Download Spinnaker SDK from FLIR's website.
- Install the Python PySpin wheel matching your Python version.
"""

from __future__ import annotations

import numpy as np

from .base import BaseCapture, CaptureFrame


class FlirCapture(BaseCapture):
    """
    Live stereo capture from two FLIR cameras using the PySpin (Spinnaker) SDK.

    Cameras are identified by serial number or by index (0 and 1 if serials not given).
    Frames are acquired as close in time as possible; hardware trigger sync is
    recommended for sub-millisecond accuracy.
    """

    def __init__(
        self,
        serial_left: str | None = None,
        serial_right: str | None = None,
        fps: float = 30.0,
        exposure_us: float = 5000.0,
        width: int = 1280,
        height: int = 1024,
    ):
        """
        Args:
            serial_left:  FLIR serial number of left camera (or None → index 0)
            serial_right: FLIR serial number of right camera (or None → index 1)
            fps:          desired frame rate
            exposure_us:  exposure time in microseconds
            width, height: frame dimensions
        """
        import PySpin  # noqa: F401 — raises ImportError if SDK not installed

        self._serial_left = serial_left
        self._serial_right = serial_right
        self._fps_val = fps
        self._exposure_us = exposure_us
        self._width = width
        self._height = height

        self._system = None
        self._cam_left = None
        self._cam_right = None
        self._frame_idx = 0

    def start(self) -> None:
        import PySpin

        self._system = PySpin.System.GetInstance()
        cam_list = self._system.GetCameras()

        def _get_cam(serial: str | None, idx: int):
            if serial:
                return cam_list.GetBySerial(serial)
            return cam_list.GetByIndex(idx)

        self._cam_left = _get_cam(self._serial_left, 0)
        self._cam_right = _get_cam(self._serial_right, 1)
        cam_list.Clear()

        for cam in (self._cam_left, self._cam_right):
            cam.Init()
            # Set acquisition mode and rate
            cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            cam.AcquisitionFrameRateEnable.SetValue(True)
            cam.AcquisitionFrameRate.SetValue(self._fps_val)
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            cam.ExposureTime.SetValue(self._exposure_us)
            cam.BeginAcquisition()

        self._frame_idx = 0

    def stop(self) -> None:

        for cam in (self._cam_left, self._cam_right):
            if cam is not None:
                try:
                    cam.EndAcquisition()
                    cam.DeInit()
                    del cam
                except Exception:
                    pass

        self._cam_left = None
        self._cam_right = None

        if self._system is not None:
            self._system.ReleaseInstance()
            self._system = None

    def read(self) -> CaptureFrame | None:
        import PySpin

        def _acquire(cam) -> np.ndarray | None:
            try:
                img = cam.GetNextImage(1000)  # 1-second timeout
                if img.IsIncomplete():
                    img.Release()
                    return None
                arr = img.GetNDArray()
                if arr.ndim == 2:
                    import cv2

                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                img.Release()
                return arr
            except PySpin.SpinnakerException:
                return None

        frame_l = _acquire(self._cam_left)
        frame_r = _acquire(self._cam_right)

        if frame_l is None or frame_r is None:
            return None

        ts = self._frame_idx / self._fps_val
        result = CaptureFrame(
            frame_left=frame_l,
            frame_right=frame_r,
            timestamp=ts,
            frame_index=self._frame_idx,
        )
        self._frame_idx += 1
        return result

    @property
    def fps(self) -> float:
        return self._fps_val

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._width, self._height)
