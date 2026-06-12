"""CLI entry points must match the GUI pipeline (and survive cp1250 consoles).

Covers three review findings:
- pose2d CLI uses IMAGE mode like the GUI Pose2DWorker (VIDEO is live-only);
- recon3d CLI smoothing defaults equal the GUI / reconstruct3d() defaults;
- CLI stdout contains no characters that crash plain print on cp1250 consoles.
"""

from __future__ import annotations

from types import SimpleNamespace


class _Cp1250Stream:
    encoding = "cp1250"

    def __init__(self):
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        text.encode(self.encoding)
        self.parts.append(text)
        return len(text)

    def flush(self) -> None:
        pass


class TestPose2dCliImageMode:
    def test_mediapipe_backend_constructed_in_image_mode(self, monkeypatch):
        import app.pose2d.mediapipe_backend as mb

        captured: dict = {}

        class _Fake:
            def __init__(self, **kw):
                captured.update(kw)

        monkeypatch.setattr(mb, "MediaPipeBackend", _Fake)
        from app.pose2d.__main__ import _make_backend

        _make_backend(SimpleNamespace(backend="mediapipe", num_poses=1))
        assert captured.get("running_mode") == "image"


class TestRecon3dCliDefaults:
    def test_smoothing_defaults_match_gui(self):
        """Same inputs must produce the same pose3d.npz from CLI and GUI."""
        import inspect

        from app.recon3d.__main__ import build_parser
        from app.recon3d.pipeline import reconstruct3d

        args = build_parser().parse_args(["--calib", "c.yml", "--pose2d", "p/"])
        sig = inspect.signature(reconstruct3d)
        assert args.min_cutoff == sig.parameters["min_cutoff"].default == 1.0
        assert args.beta == sig.parameters["beta"].default == 0.5
        assert args.d_cutoff == sig.parameters["d_cutoff"].default == 1.0


class TestCliStdoutAscii:
    """Block characters / arrows crash plain print on cp1250 Windows consoles."""

    def test_console_safe_replaces_unencodable_chars(self):
        from app.console import console_safe

        stream = _Cp1250Stream()
        safe = console_safe("⚠ ≥ → … — ×", stream)
        safe.encode(stream.encoding)

    def test_progress_printers_accept_unicode_messages_on_cp1250(self, monkeypatch):
        """Progress messages from pipelines reach these printers; they must not crash."""
        import sys

        stream = _Cp1250Stream()
        monkeypatch.setattr(sys, "stdout", stream)

        from app.calib.__main__ import _progress as calib_progress
        from app.pose2d.__main__ import _progress as pose_progress
        from app.recon3d.__main__ import _progress as recon_progress

        calib_progress(5, "⚠ Warning: recommend ≥20 frames — retry…")
        pose_progress(10, "Loading → detecting…")
        recon_progress(20, "WARNING: left/right are paired by detection ORDER → unreliable…")
