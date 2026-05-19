"""
FLIR camera capture via PySpin (Spinnaker SDK).

This module is imported only if PySpin is available.
If the FLIR SDK is not installed, the import will fail gracefully and
VideoSimulator will be used instead (see app/capture/__init__.py).

Install notes:
- Download Spinnaker SDK from FLIR's website.
- Install the Python PySpin wheel matching your Python version.

Hardware sync wiring (BlackflyS 6-pin GPIO):
  Signal : Primary pin 4 (white, Line 1 OPTOOUT) → Secondary pin 1 (green, Line 3 GPI)
  Ground : Primary pin 5 (blue,  OPTO_GND)        → Secondary pin 6 (brown, GND)
  Pull-up: 10 kΩ resistor — Primary pin 3 (red, 3.3 V) → Primary pin 4 + Secondary pin 1

  The primary outputs ExposureActive on Line 1; Line 2 supplies the 3.3 V
  pull-up voltage.  The secondary triggers on the rising edge of Line 3.
  Start secondary acquisition before primary so no pulses are missed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import BaseCapture, CaptureFrame

# File-based logger so crashes leave a trace even when the process is killed.
_log_path = Path.home() / ".config" / "wt-app" / "flir.log"
_log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(_log_path),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level sync helpers — call after cam.Init(), before BeginAcquisition
# ---------------------------------------------------------------------------

def _configure_freerun(cam, fps: float, exposure_us: float) -> None:
    """Free-running continuous acquisition; no hardware trigger."""
    import PySpin

    _log.debug("freerun: TriggerMode Off")
    cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    _log.debug("freerun: AcquisitionMode Continuous")
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    _log.debug("freerun: FrameRateEnable")
    cam.AcquisitionFrameRateEnable.SetValue(True)
    _log.debug("freerun: FrameRate %s", fps)
    cam.AcquisitionFrameRate.SetValue(fps)
    _log.debug("freerun: ExposureAuto Off")
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    _log.debug("freerun: ExposureTime %s", exposure_us)
    cam.ExposureTime.SetValue(exposure_us)
    _log.debug("freerun: done")


def _configure_primary(cam, fps: float, exposure_us: float) -> None:
    """
    Primary camera (BlackflyS): free-runs at *fps* and outputs ExposureActive
    on Line 1 (pin 4, white wire — the opto-isolated output) to trigger the
    secondary camera on every frame.

    BlackflyS GPIO wiring:
      Pin 4 white  = Line 1  OPTOOUT  opto-isolated output  ← strobe signal
      Pin 3 red    = Line 2  3.3 V    power rail             ← pull-up supply
      Pin 5 blue             OPTO_GND  opto-isolated ground
      Pin 1 green  = Line 3  GPI      non-isolated input     ← secondary input

    Line 1 IS switchable (LineMode can be set to Output).
    Line 2 is only a 3.3 V power rail (V3_3Enable) — it is NOT a signal line
    and must NOT be used as LineSource.
    """
    import PySpin

    _log.debug("primary: TriggerMode Off")
    cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    _log.debug("primary: AcquisitionMode Continuous")
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    _log.debug("primary: FrameRateEnable")
    cam.AcquisitionFrameRateEnable.SetValue(True)
    _log.debug("primary: FrameRate %s", fps)
    cam.AcquisitionFrameRate.SetValue(fps)
    _log.debug("primary: ExposureAuto Off")
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    _log.debug("primary: ExposureTime %s", exposure_us)
    cam.ExposureTime.SetValue(exposure_us)

    # Line 1 (pin 4, white) — opto-isolated output — set as strobe
    _log.debug("primary: LineSelector Line1")
    cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
    _log.debug("primary: LineMode Output")
    cam.LineMode.SetValue(PySpin.LineMode_Output)
    _log.debug("primary: LineSource ExposureActive")
    cam.LineSource.SetValue(PySpin.LineSource_ExposureActive)

    # Line 2 (pin 3, red) — enable 3.3 V rail for the pull-up resistor circuit
    _log.debug("primary: LineSelector Line2 for V3_3Enable")
    cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
    _log.debug("primary: V3_3Enable True")
    cam.V3_3Enable.SetValue(True)
    _log.debug("primary: done")


def _configure_secondary(cam, exposure_us: float) -> None:
    """
    Secondary camera: waits for a rising-edge hardware trigger on Line 3.
    TriggerOverlap_ReadOut keeps throughput high by accepting a new trigger
    while the sensor is still reading out.

    Important: the internal frame-rate limiter must be DISABLED so that the
    trigger rate — not the camera's own oscillator — controls acquisition speed.
    A previous SpinView session may have left AcquisitionFrameRateEnable=True
    with a low value, causing the secondary to skip triggers and appear one
    frame behind the primary.
    """
    import PySpin

    _log.debug("secondary: TriggerSelector FrameStart")
    cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
    _log.debug("secondary: TriggerSource Line3")
    cam.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
    _log.debug("secondary: TriggerActivation RisingEdge")
    cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
    _log.debug("secondary: TriggerOverlap ReadOut")
    cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
    _log.debug("secondary: TriggerMode On")
    cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
    _log.debug("secondary: AcquisitionMode Continuous")
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    # Disable internal frame-rate limiter — trigger controls the rate.
    _log.debug("secondary: AcquisitionFrameRateEnable False")
    cam.AcquisitionFrameRateEnable.SetValue(False)
    _log.debug("secondary: ExposureAuto Off")
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    _log.debug("secondary: ExposureTime %s", exposure_us)
    cam.ExposureTime.SetValue(exposure_us)
    # Reset any trigger delay left by a previous SpinView session.
    try:
        _log.debug("secondary: TriggerDelay 0")
        cam.TriggerDelay.SetValue(0.0)
    except Exception as exc:
        _log.debug("secondary: TriggerDelay not available (%s)", exc)
    _log.debug("secondary: done")


def list_camera_serials() -> list[str]:
    """Return serial numbers of all connected FLIR cameras. Raises ImportError if PySpin missing."""
    import PySpin

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    serials: list[str] = []
    for i in range(cam_list.GetSize()):
        cam = cam_list.GetByIndex(i)
        cam.Init()
        serials.append(cam.DeviceSerialNumber.GetValue())
        cam.DeInit()
        del cam
    cam_list.Clear()
    system.ReleaseInstance()
    return serials


# ---------------------------------------------------------------------------
# FlirCapture
# ---------------------------------------------------------------------------

class FlirCapture(BaseCapture):
    """
    Live stereo capture from two FLIR cameras using the PySpin (Spinnaker) SDK.

    Cameras are identified by serial number or by index (0 and 1 if serials not given).

    sync=True wires hardware trigger synchronization:
      - Primary camera free-runs and pulses Line 2 on every exposure.
      - Secondary camera triggers on the rising edge of Line 3.
      - Secondary acquisition is started before primary so no frames are missed.
      - Requires the GPIO cable to be connected (Line 2 → Line 3).

    sync=False: both cameras free-run independently (software sync only).
    primary: "left" or "right" — which camera acts as the trigger source.
    """

    def __init__(
        self,
        serial_left: str | None = None,
        serial_right: str | None = None,
        fps: float = 30.0,
        exposure_us: float = 5000.0,
        width: int = 1280,
        height: int = 1024,
        sync: bool = False,
        primary: str = "left",
    ):
        import PySpin  # noqa: F401 — raises ImportError if SDK not installed

        self._serial_left = serial_left
        self._serial_right = serial_right
        self._fps_val = fps
        self._exposure_us = exposure_us
        self._width = width
        self._height = height
        self._sync = sync
        self._primary = primary  # "left" or "right"

        self._system = None
        self._cam_list = None   # kept alive until stop() to satisfy PySpin refcounting
        self._cam_left = None
        self._cam_right = None
        self._frame_idx = 0

    def start(self) -> None:
        import PySpin

        _log.info(
            "start: sync=%s primary=%s fps=%s exposure_us=%s",
            self._sync, self._primary, self._fps_val, self._exposure_us,
        )

        _log.debug("start: GetInstance")
        self._system = PySpin.System.GetInstance()
        _log.debug("start: GetCameras")
        # Keep cam_list as an instance variable — PySpin requires it to stay
        # alive until after all cameras are DeInit'd and cam_list.Clear() is
        # called in stop().  Clearing it early causes the [-1004] error.
        self._cam_list = self._system.GetCameras()

        def _get_cam(serial: str | None, idx: int):
            if serial:
                _log.debug("start: GetBySerial %s", serial)
                return self._cam_list.GetBySerial(serial)
            _log.debug("start: GetByIndex %s", idx)
            return self._cam_list.GetByIndex(idx)

        self._cam_left = _get_cam(self._serial_left, 0)
        self._cam_right = _get_cam(self._serial_right, 1)

        _log.debug("start: Init left")
        self._cam_left.Init()
        _log.debug("start: Init right")
        self._cam_right.Init()

        if self._sync:
            prim = self._cam_left if self._primary == "left" else self._cam_right
            sec = self._cam_right if self._primary == "left" else self._cam_left
            _log.debug("start: configure primary")
            _configure_primary(prim, self._fps_val, self._exposure_us)
            _log.debug("start: configure secondary")
            _configure_secondary(sec, self._exposure_us)
            # Secondary must begin acquisition before primary sends its first trigger
            _log.debug("start: BeginAcquisition secondary")
            sec.BeginAcquisition()
            _log.debug("start: BeginAcquisition primary")
            prim.BeginAcquisition()
        else:
            for label, cam in (("left", self._cam_left), ("right", self._cam_right)):
                _log.debug("start: configure freerun %s", label)
                _configure_freerun(cam, self._fps_val, self._exposure_us)
                _log.debug("start: BeginAcquisition %s", label)
                cam.BeginAcquisition()

        self._frame_idx = 0
        _log.info("start: acquisition running")

    def stop(self) -> None:
        # Official PySpin teardown order (AcquisitionMultipleCamera example):
        #   1. EndAcquisition   2. DeInit   3. del *every* Python name that ever
        #   held a camera ref (including loop variables!)   4. cam_list.Clear()
        #   5. system.ReleaseInstance()
        #
        # Key subtlety: a Python `for` loop leaves the loop variable alive after
        # the loop ends.  If we loop over cameras and never `del cam`, that loop
        # variable keeps the last camera's refcount > 0 and ReleaseInstance()
        # raises [-1004].  We avoid this by moving all refs to local variables,
        # clearing instance attributes immediately, then del-ing every local.

        _log.debug("stop: snapshot refs → locals")
        cam_left  = self._cam_left
        cam_right = self._cam_right
        cam_list  = self._cam_list
        system    = self._system
        self._cam_left  = None
        self._cam_right = None
        self._cam_list  = None
        self._system    = None

        _log.debug("stop: EndAcquisition")
        for cam in filter(None, (cam_left, cam_right)):
            try:
                cam.EndAcquisition()
            except Exception:
                pass

        _log.debug("stop: DeInit")
        for cam in filter(None, (cam_left, cam_right)):
            try:
                cam.DeInit()
            except Exception:
                pass

        # Delete every local name that holds (or held) a camera reference,
        # including the `cam` loop variable left by the last `for` loop.
        _log.debug("stop: del camera refs")
        del cam_left, cam_right
        try:
            del cam  # loop variable from last `for` above
        except NameError:
            pass  # both cameras were None — loop never ran

        _log.debug("stop: cam_list.Clear")
        if cam_list is not None:
            cam_list.Clear()
        del cam_list

        _log.debug("stop: ReleaseInstance")
        if system is not None:
            system.ReleaseInstance()
        del system

        _log.info("stop: done")

    def read(self) -> CaptureFrame | None:
        import PySpin

        def _acquire(cam) -> tuple[np.ndarray | None, int | None]:
            """Return (bgr_array, hardware_timestamp_ns) or (None, None).

            img.Release() is called in a finally block so the camera buffer is
            always returned — even if conversion raises.  An unreleased buffer
            keeps the camera locked and causes ReleaseInstance() to fail with
            [-1004] during stop().

            Bayer demosaicing strategy:
              1. Try PySpin's built-in img.Convert() (available in some versions).
              2. Fall back to OpenCV cvtColor with the appropriate Bayer code,
                 derived from the raw array shape and the camera's PixelFormat.
            """
            import cv2

            img = None
            try:
                img = cam.GetNextImage(1000)  # 1-second timeout
                if img.IsIncomplete():
                    return None, None

                hw_ts = img.GetTimeStamp()  # camera internal clock, nanoseconds

                # Strategy 1: PySpin built-in demosaic (preferred, not in all versions)
                try:
                    converted = img.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
                    arr = converted.GetNDArray()
                    return arr, hw_ts
                except (AttributeError, PySpin.SpinnakerException):
                    pass  # fall through to OpenCV path

                # Strategy 2: OpenCV Bayer demosaicing
                raw = img.GetNDArray()
                if raw.ndim == 3:
                    # Already a colour image (e.g. RGB8 pixel format)
                    arr = raw if raw.shape[2] == 3 else cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                else:
                    # Raw Bayer pattern — pick the right OpenCV code from PixelFormat
                    # PySpin and OpenCV use opposite Bayer naming conventions —
                    # PySpin names the first two pixels of the FIRST row,
                    # OpenCV names the first two pixels of the SECOND row.
                    # Result: every PySpin name maps to the "opposite" OpenCV code.
                    _bayer_map = {
                        "BayerRG8":  cv2.COLOR_BAYER_BG2BGR,  # BlackflyS default
                        "BayerGB8":  cv2.COLOR_BAYER_GR2BGR,
                        "BayerGR8":  cv2.COLOR_BAYER_GB2BGR,
                        "BayerBG8":  cv2.COLOR_BAYER_RG2BGR,
                        "BayerRG16": cv2.COLOR_BAYER_BG2BGR,
                        "BayerGB16": cv2.COLOR_BAYER_GR2BGR,
                        "BayerGR16": cv2.COLOR_BAYER_GB2BGR,
                        "BayerBG16": cv2.COLOR_BAYER_RG2BGR,
                    }
                    try:
                        fmt_name = PySpin.PixelFormatInterfaceClass_GetName(
                            cam.PixelFormat.GetValue()
                        )
                    except Exception:
                        fmt_name = "BayerRG8"  # BlackflyS default
                    code = _bayer_map.get(fmt_name, cv2.COLOR_BAYER_BG2BGR)
                    arr = cv2.cvtColor(raw, code)

                return arr, hw_ts

            except PySpin.SpinnakerException:
                return None, None
            finally:
                # Always release the buffer so the camera doesn't stay locked
                if img is not None:
                    try:
                        img.Release()
                    except Exception:
                        pass

        frame_l, ts_l = _acquire(self._cam_left)
        frame_r, ts_r = _acquire(self._cam_right)

        # Stop only if both cameras fail simultaneously.
        # If only the secondary fails (e.g. sync cable unplugged) keep streaming
        # from the primary with a black placeholder so the user can see the feed.
        if frame_l is None and frame_r is None:
            return None

        if frame_l is None or frame_r is None:
            # Build a same-size black placeholder for the missing camera
            ref = frame_l if frame_l is not None else frame_r
            placeholder = np.zeros_like(ref)
            if frame_l is None:
                frame_l, ts_l = placeholder, None
            else:
                frame_r, ts_r = placeholder, None

        ts = self._frame_idx / self._fps_val
        result = CaptureFrame(
            frame_left=frame_l,
            frame_right=frame_r,
            timestamp=ts,
            frame_index=self._frame_idx,
            hw_timestamp_left_ns=ts_l,
            hw_timestamp_right_ns=ts_r,
        )
        self._frame_idx += 1
        return result

    @property
    def fps(self) -> float:
        return self._fps_val

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._width, self._height)
