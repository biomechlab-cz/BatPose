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
_log_path = Path.home() / ".config" / "BatPose" / "flir.log"
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

def _lock_image_processing(cam, gain_db: float = 0.0, label: str = "?") -> None:
    """Pin photometric settings so both stereo cameras produce comparable images.

    Without this, each FLIR independently auto-adjusts gain, white-balance,
    sharpening, AND retains whatever PixelFormat / Gamma / BlackLevel /
    AdcBitDepth a previous SpinView session left behind → the two cameras
    end up visibly different in brightness, saturation, and edge crispness
    even when ExposureTime and Gain are matched.

    Settings locked, in order of impact on inter-camera consistency:

      • PixelFormat   — force BayerRG8 on both cameras so the demosaicing
                         path in _acquire() is identical.  If the cameras
                         report different formats, one image runs through
                         PySpin's HQ_LINEAR demosaic and the other through
                         OpenCV's, which produce visibly different colour
                         and brightness.
      • AdcBitDepth   — 12-bit on both.  A previous SpinView session can
                         leave one camera at 10-bit and the other at 12-bit,
                         which scales differently into the 8-bit output and
                         shows up as a ~2× brightness mismatch.
      • Gain          — disable GainAuto, set the same dB on both cameras.
      • BlackLevel    — set to 0 on both.  A non-zero pedestal from a
                         previous session adds a digital brightness offset
                         that exposure can't compensate for.
      • Gamma         — enable on both with the same value (0.8 ≈ sRGB).
                         Without this, one camera may have GammaEnable=False
                         (linear, dim mids) and the other GammaEnable=True
                         (curved, bright mids) → looks like an exposure
                         mismatch but isn't.
      • Whitebal      — BalanceWhiteAuto_Continuous on both.  Stereo pair
                         sees the same scene → they converge to similar
                         ratios.  More reliable than _Once and avoids the
                         green cast from forcing 1.0/1.0 ratios.
      • Sharpening    — off on both; in-sensor sharpening differs between
                         units and smears sub-pixel calibration corners.

    Read-back values are logged so a brightness mismatch in the preview can
    be diagnosed by diffing the two cameras' log entries.

    All node operations are wrapped in try/except: node availability varies
    across FLIR model families.
    """
    import PySpin

    def _try(name: str, fn):
        try:
            fn()
            _log.debug("photometric[%s]: %s OK", label, name)
        except Exception as exc:
            _log.debug("photometric[%s]: %s not settable (%s)", label, name, exc)

    # --- PixelFormat: force identical Bayer format on both cameras ----------
    _try("PixelFormat BayerRG8",
         lambda: cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG8))

    # --- AdcBitDepth: same on both (12-bit if supported) --------------------
    _try("AdcBitDepth Bit12",
         lambda: cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit12))

    # --- Gain: disable auto, set fixed value (same on both cameras) ----------
    _try("GainAuto Off", lambda: cam.GainAuto.SetValue(PySpin.GainAuto_Off))
    _try(f"Gain {gain_db} dB", lambda: cam.Gain.SetValue(float(gain_db)))

    # --- BlackLevel: zero pedestal on both ----------------------------------
    _try("BlackLevelSelector All",
         lambda: cam.BlackLevelSelector.SetValue(PySpin.BlackLevelSelector_All))
    _try("BlackLevel 0.0", lambda: cam.BlackLevel.SetValue(0.0))

    # --- Gamma: enable, identical value on both -----------------------------
    _try("GammaEnable True", lambda: cam.GammaEnable.SetValue(True))
    _try("Gamma 0.8", lambda: cam.Gamma.SetValue(0.8))

    # --- Colour correction matrix: try to disable, best-effort -------------
    # ColorTransformationEnable is the prime suspect when "right looks
    # oversaturated/unnatural" but exposure/gain/gamma are matched: a
    # previous SpinView session can leave CCM enabled on one camera and
    # disabled on the other, and it's frequently *not writable* on the
    # already-enabled camera (AccessException -2006).  Belt-and-suspenders:
    # we still attempt to disable it AND we bypass PySpin's img.Convert()
    # in _acquire() so the CCM state cannot affect the displayed image
    # regardless of whether this write succeeds.
    _try("ColorTransformationEnable False",
         lambda: cam.ColorTransformationEnable.SetValue(False))

    # --- Saturation / Hue: identical digital values if exposed -------------
    _try("Saturation 1.0", lambda: cam.Saturation.SetValue(1.0))
    _try("Hue 0.0",        lambda: cam.Hue.SetValue(0.0))

    # --- ISP master enable: keep ON; off would bypass demosaic entirely ----
    _try("IspEnable True", lambda: cam.IspEnable.SetValue(True))

    # --- White balance: per-camera auto-converge, then freeze --------------
    # KEY INSIGHT: forcing IDENTICAL BalanceRatio on both cameras does NOT
    # make them look the same — it makes them look DIFFERENT.  White balance
    # exists to compensate each sensor's individual spectral response; two
    # nominally identical Sony sensors still differ slightly in colour-filter
    # transmission and per-channel gain.  Applying the same multipliers to two
    # different raw responses guarantees a residual colour-tone mismatch.
    #
    # The correct approach is to let EACH camera converge its OWN ratios
    # (individually neutralising its sensor) and then lock them.  We start
    # Continuous here; _warmup_white_balance() (called after BeginAcquisition)
    # lets it settle for a fixed number of frames and then switches it to Off,
    # freezing each camera's converged ratios so they don't drift mid-session.
    _try("BalanceWhiteAuto Continuous",
         lambda: cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Continuous))

    # --- Sharpening: try to disable; node is often read-only on BlackflyS ---
    # SharpeningEnable is frequently locked unless SharpeningAvailable is True
    # AND ISP is enabled.  PySpin builds without SharpeningAuto_Off skip that
    # one entirely.
    _try("SharpeningEnable False", lambda: cam.SharpeningEnable.SetValue(False))
    if hasattr(PySpin, "SharpeningAuto_Off"):
        _try("SharpeningAuto Off",
             lambda: cam.SharpeningAuto.SetValue(PySpin.SharpeningAuto_Off))

    # --- LUT: disable (per-camera luminance LUT from a previous SpinView
    #     session is a classic cause of "one camera permanently darker") -----
    _try("LUTEnable False", lambda: cam.LUTEnable.SetValue(False))

    # --- ExposureMode: Timed (not TriggerWidth) -----------------------------
    _try("ExposureMode Timed",
         lambda: cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed))

    # --- Defect-pixel correction: enabled and same on both ------------------
    _try("DefectCorrectStaticEnable True",
         lambda: cam.DefectCorrectStaticEnable.SetValue(True))

    # --- Binning / decimation: 1× on both (any difference is a 2-4× scale) --
    _try("BinningHorizontal 1", lambda: cam.BinningHorizontal.SetValue(1))
    _try("BinningVertical 1",   lambda: cam.BinningVertical.SetValue(1))
    _try("DecimationHorizontal 1",
         lambda: cam.DecimationHorizontal.SetValue(1))
    _try("DecimationVertical 1",
         lambda: cam.DecimationVertical.SetValue(1))

    # --- Full-sensor ROI on both cameras ------------------------------------
    # A previous SpinView session can leave one camera with a cropped ROI
    # (e.g. Width=960/Height=600 instead of full 1920×1200), which makes its
    # preview look "zoomed in" relative to its stereo partner.  Force both
    # cameras to the sensor's max resolution.
    #
    # Order matters: OffsetX/Y must go to 0 BEFORE Width/Height can expand
    # back to max — otherwise SetValue clips silently against the existing
    # offset+size constraint.
    _try("OffsetX 0", lambda: cam.OffsetX.SetValue(0))
    _try("OffsetY 0", lambda: cam.OffsetY.SetValue(0))
    try:
        max_w = cam.WidthMax.GetValue()
        max_h = cam.HeightMax.GetValue()
        _try(f"Width {max_w}",  lambda: cam.Width.SetValue(max_w))
        _try(f"Height {max_h}", lambda: cam.Height.SetValue(max_h))
    except Exception as exc:
        _log.debug("photometric[%s]: WidthMax/HeightMax unavailable (%s)", label, exc)



def _set_newest_only_buffer(cam, label: str = "?") -> None:
    """Configure the transport-layer stream buffer to deliver only the latest frame.

    Default mode is OldestFirst (FIFO).  In a stereo pair, if one camera's
    host-side reads are momentarily slow, its FIFO buffer accumulates frames
    while the other camera's stays empty — every subsequent GetNextImage()
    on the slow side then returns a stale frame from the back of the queue.
    The result: hardware exposures are synchronized to microseconds, but the
    two delivered frames are visually several hundred milliseconds apart.

    NewestOnly tells the SDK to always hand us the most recent frame and
    discard older queued ones, so the two streams cannot drift out of phase
    at the buffer-delivery level no matter what the host does.

    The buffer handling mode lives on the TLStream (Transport Layer Stream)
    nodemap, NOT the main camera nodemap, so access goes through
    GetTLStreamNodeMap() rather than the cam.* attribute shortcut.
    """
    import PySpin

    try:
        s_nodemap = cam.GetTLStreamNodeMap()
        node = PySpin.CEnumerationPtr(s_nodemap.GetNode("StreamBufferHandlingMode"))
        if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
            _log.debug("buffer[%s]: StreamBufferHandlingMode not writable", label)
            return
        entry = node.GetEntryByName("NewestOnly")
        if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
            _log.debug("buffer[%s]: NewestOnly entry not available", label)
            return
        node.SetIntValue(entry.GetValue())
        _log.info("buffer[%s]: StreamBufferHandlingMode = %s",
                  label, node.GetCurrentEntry().GetSymbolic())
    except Exception as exc:
        _log.debug("buffer[%s]: failed to set NewestOnly (%s)", label, exc)


def _configure_freerun(cam, fps: float, exposure_us: float, gain_db: float = 10.0, label: str = "?") -> None:
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
    _lock_image_processing(cam, gain_db=gain_db, label=label)
    _set_newest_only_buffer(cam, label=label)
    _log.debug("freerun: done")


def _configure_primary(cam, fps: float, exposure_us: float, gain_db: float = 10.0, label: str = "primary") -> None:
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
    _lock_image_processing(cam, gain_db=gain_db, label=label)
    _set_newest_only_buffer(cam, label=label)

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


def _configure_secondary(cam, exposure_us: float, gain_db: float = 10.0, label: str = "secondary") -> None:
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
    _lock_image_processing(cam, gain_db=gain_db, label=label)
    _set_newest_only_buffer(cam, label=label)
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
        gain_db: float = 10.0,
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
        self._gain_db = gain_db
        self._width = width
        self._height = height
        self._sync = sync
        self._primary = primary  # "left" or "right"

        self._system = None
        self._cam_list = None   # kept alive until stop() to satisfy PySpin refcounting
        self._cam_left = None
        self._cam_right = None
        self._frame_idx = 0
        # Drop-detection state — reset in start()
        self._dropped: int = 0
        self._last_ts_left_ns: int | None = None
        self._last_ts_right_ns: int | None = None

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

        # Fail early with a clear message when cameras are powered off /
        # unplugged.  Without this check, GetBySerial/GetByIndex returns an
        # invalid camera pointer and Init() crashes with the cryptic
        # "Spinnaker: NULL pointer dereferenced. [-1015]".
        n_cams = self._cam_list.GetSize()
        _log.debug("start: %d camera(s) enumerated", n_cams)
        if n_cams < 2:
            self._release_system_after_failed_start()
            raise RuntimeError(
                f"Only {n_cams} FLIR camera(s) detected — 2 are required.\n\n"
                "Please check that:\n"
                "• Both cameras are powered on.\n"
                "• Both USB 3.0 cables are securely connected.\n"
                "• No other application (e.g. SpinView) is using the cameras.\n\n"
                "Then click 'Detect cameras' again before starting."
            )

        def _get_cam(serial: str | None, idx: int):
            if serial:
                _log.debug("start: GetBySerial %s", serial)
                return self._cam_list.GetBySerial(serial)
            _log.debug("start: GetByIndex %s", idx)
            return self._cam_list.GetByIndex(idx)

        try:
            self._cam_left = _get_cam(self._serial_left, 0)
            self._cam_right = _get_cam(self._serial_right, 1)

            _log.debug("start: Init left")
            self._cam_left.Init()
            _log.debug("start: Init right")
            self._cam_right.Init()
        except PySpin.SpinnakerException as exc:
            _log.error("start: camera init failed (%s)", exc)
            self._release_system_after_failed_start()
            raise RuntimeError(
                "Failed to initialise the FLIR cameras.\n\n"
                "This usually means a camera was switched off or unplugged, "
                "or another application (e.g. SpinView) is currently using it.\n\n"
                "Power-cycle the cameras, close any other capture software, "
                "then click 'Detect cameras' again before starting.\n\n"
                f"(Spinnaker detail: {exc})"
            ) from exc

        if self._sync:
            prim = self._cam_left if self._primary == "left" else self._cam_right
            sec = self._cam_right if self._primary == "left" else self._cam_left
            prim_label = self._primary               # "left" or "right"
            sec_label = "right" if self._primary == "left" else "left"
            _log.debug("start: configure primary (%s)", prim_label)
            _configure_primary(prim, self._fps_val, self._exposure_us, self._gain_db, label=prim_label)
            _log.debug("start: configure secondary (%s)", sec_label)
            _configure_secondary(sec, self._exposure_us, self._gain_db, label=sec_label)
            # Secondary must begin acquisition before primary sends its first trigger
            _log.debug("start: BeginAcquisition secondary")
            sec.BeginAcquisition()
            _log.debug("start: BeginAcquisition primary")
            prim.BeginAcquisition()
        else:
            for label, cam in (("left", self._cam_left), ("right", self._cam_right)):
                _log.debug("start: configure freerun %s", label)
                _configure_freerun(cam, self._fps_val, self._exposure_us, self._gain_db, label=label)
                _log.debug("start: BeginAcquisition %s", label)
                cam.BeginAcquisition()

        # Let each camera's auto white-balance converge on live frames, then
        # freeze the converged per-camera ratios so colour tone is matched and
        # stable for the rest of the session.
        self._warmup_white_balance()

        self._frame_idx = 0
        self._dropped = 0
        self._last_ts_left_ns = None
        self._last_ts_right_ns = None
        _log.info("start: acquisition running")

    def _warmup_white_balance(self, warmup_frames: int = 30) -> None:
        """Run continuous AWB briefly, then lock each camera's converged ratios.

        Set in _lock_image_processing(), BalanceWhiteAuto is Continuous when we
        reach here.  We grab and discard a fixed number of frames so the camera
        firmware can converge its per-sensor balance ratios on the real scene,
        then switch BalanceWhiteAuto to Off — which freezes the last converged
        BalanceRatio values.  Each camera is thus individually neutralised
        (compensating its own sensor) and won't drift during recording.

        Frames are pulled from both cameras in lock-step so a hardware-synced
        secondary still receives its triggers from the primary during warmup.
        """
        import PySpin

        cams = [c for c in (self._cam_left, self._cam_right) if c is not None]
        if not cams:
            return

        _log.debug("white-balance warmup: %d frames", warmup_frames)
        for _ in range(warmup_frames):
            for cam in cams:
                try:
                    img = cam.GetNextImage(1000)
                    try:
                        # We don't need the pixels — converging AWB just needs
                        # the frames to flow through the ISP.
                        pass
                    finally:
                        img.Release()
                except PySpin.SpinnakerException:
                    pass  # a missed trigger during warmup is harmless

        # Freeze: Auto → Off retains the last auto-computed BalanceRatio.
        for label, cam in (("left", self._cam_left), ("right", self._cam_right)):
            if cam is None:
                continue
            try:
                cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)
                # Log the frozen per-camera ratios so a colour mismatch can be
                # diagnosed from the log if it ever recurs.
                try:
                    cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Red)
                    red = cam.BalanceRatio.GetValue()
                    cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Blue)
                    blue = cam.BalanceRatio.GetValue()
                    _log.info("white-balance[%s] frozen: Red=%.3f Blue=%.3f",
                              label, red, blue)
                except Exception:
                    pass
            except Exception as exc:
                _log.debug("white-balance[%s]: freeze failed (%s)", label, exc)

    def _release_system_after_failed_start(self) -> None:
        """Tear down partially-acquired Spinnaker handles when start() aborts.

        Called before raising a user-facing error so the System instance and
        camera list don't leak — a leaked System keeps the cameras locked and
        makes the *next* start attempt fail too.  Mirrors stop()'s careful
        refcount handling: drop every Python camera reference before clearing
        the list and releasing the system.
        """
        _log.debug("failed-start cleanup: begin")
        cam_left = self._cam_left
        cam_right = self._cam_right
        cam_list = self._cam_list
        system = self._system
        self._cam_left = None
        self._cam_right = None
        self._cam_list = None
        self._system = None

        for cam in filter(None, (cam_left, cam_right)):
            try:
                if cam.IsInitialized():
                    cam.DeInit()
            except Exception:
                pass
        del cam_left, cam_right
        try:
            del cam  # loop variable, if the loop ran
        except NameError:
            pass

        if cam_list is not None:
            try:
                cam_list.Clear()
            except Exception:
                pass
        del cam_list

        if system is not None:
            try:
                system.ReleaseInstance()
            except Exception:
                pass
        del system
        _log.debug("failed-start cleanup: done")

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

            Bayer demosaicing: we deliberately AVOID PySpin's img.Convert().
            Convert() applies the camera's current ISP state (in particular
            ColorTransformationEnable / CCM) during conversion — and that
            node is not always writable, so the two cameras in a stereo pair
            can end up with the CCM in opposite states.  The result is one
            image saturated and bright, the other linear and dim, even when
            every other photometric setting is locked identically.

            Doing the demosaic in OpenCV bypasses the camera's ISP entirely:
            both cameras go through bit-for-bit identical post-processing,
            so any remaining brightness/saturation differences come from the
            scene (parallax, lens vignetting) — not from per-camera state.
            """
            import cv2

            img = None
            try:
                img = cam.GetNextImage(1000)  # 1-second timeout
                if img.IsIncomplete():
                    return None, None

                hw_ts = img.GetTimeStamp()  # camera internal clock, nanoseconds

                raw = img.GetNDArray()
                if raw.ndim == 3:
                    # Already a colour image (e.g. RGB8 pixel format) — we
                    # configured BayerRG8 so this branch is unusual, but
                    # handle it defensively.  .copy() is required: raw shares
                    # memory with the camera buffer and img.Release() (in the
                    # finally block) frees that buffer.  Without the copy,
                    # NumPy tries to decrement the dtype refcount of freed
                    # memory → "Reference count error detected" at runtime.
                    if raw.shape[2] == 3:
                        arr = raw.copy()
                    else:
                        arr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                else:
                    # Raw Bayer pattern — pick the right OpenCV code from PixelFormat.
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
                # Delete raw before releasing the image buffer.  raw is a
                # NumPy array backed by the camera's DMA buffer; img.Release()
                # frees that buffer.  If raw still exists when the GC later
                # collects it, NumPy attempts to decrement the dtype refcount
                # of freed memory → "Reference count error detected (dtype B)".
                try:
                    del raw
                except NameError:
                    pass  # raw was never assigned (e.g. IsIncomplete early-return)
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

        # Detect dropped frames: if the hw timestamp gap between consecutive
        # deliveries exceeds 1.5× the nominal frame period, at least one frame
        # was silently discarded by the NewestOnly stream buffer.
        expected_ns = int(1_000_000_000 / self._fps_val)
        for prev, cur in (
            (self._last_ts_left_ns,  ts_l),
            (self._last_ts_right_ns, ts_r),
        ):
            if prev is not None and cur is not None and cur > prev:
                gap = cur - prev
                if gap > 1.5 * expected_ns:
                    self._dropped += max(1, round(gap / expected_ns) - 1)
                    break  # count once per frame pair, not once per camera
        self._last_ts_left_ns = ts_l if ts_l is not None else self._last_ts_left_ns
        self._last_ts_right_ns = ts_r if ts_r is not None else self._last_ts_right_ns

        ts = self._frame_idx / self._fps_val
        result = CaptureFrame(
            frame_left=frame_l,
            frame_right=frame_r,
            timestamp=ts,
            frame_index=self._frame_idx,
            hw_timestamp_left_ns=ts_l,
            hw_timestamp_right_ns=ts_r,
            dropped_frames=self._dropped,
        )
        self._frame_idx += 1
        return result

    @property
    def fps(self) -> float:
        return self._fps_val

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._width, self._height)
