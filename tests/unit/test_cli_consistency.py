"""CLI entry points must match the GUI pipeline (and survive cp1250 consoles).

Covers three review findings:
- pose2d CLI uses IMAGE mode like the GUI Pose2DWorker (VIDEO is live-only);
- recon3d CLI smoothing defaults equal the GUI / reconstruct3d() defaults;
- CLI stdout contains no characters that crash plain print on cp1250 consoles.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

_APP = Path(__file__).parents[2] / "app"


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

    def test_cli_mains_free_of_console_breaking_chars(self):
        for mod in ("calib", "pose2d", "recon3d"):
            text = (_APP / mod / "__main__.py").read_text(encoding="utf-8")
            for ch in ("█", "░", "→", "⚠"):  # block, shade, arrow, warning
                assert ch not in text, f"app/{mod}/__main__.py contains {ch!r}"

    def test_pipeline_progress_messages_cp1250_safe(self):
        """pipeline.py progress strings reach the CLI printers — must encode."""
        text = (_APP / "recon3d" / "pipeline.py").read_text(encoding="utf-8")
        assert "⚠" not in text  # the old warning sign crashed cp1250 print
