"""Entry point: python -m app.gui"""

from __future__ import annotations

import sys
from pathlib import Path


def _set_windows_app_id() -> None:
    """Give Windows an explicit AppUserModelID so the taskbar uses our icon.

    Without this, a Python process is grouped under ``python.exe`` and the
    taskbar shows Python's generic icon instead of the window icon.  Must run
    before the first window is shown.  No-op on non-Windows platforms.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("biomechlab.BatPose.gui.0.1")
    except Exception:
        pass  # cosmetic only — never block startup on this


def _load_app_icon(icon_path: Path):
    """Build a QIcon from the SVG with several raster sizes baked in.

    The native Windows taskbar needs raster icons at standard sizes; handing it
    a bare SVG often yields a blank/blurry icon.  We render the SVG to a range
    of pixmap sizes so the OS can pick the right one.  Falls back to a plain
    QIcon(svg) if the SVG renderer is unavailable.
    """
    from PySide6.QtGui import QIcon

    if not icon_path.exists():
        return QIcon()
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPainter, QPixmap
        from PySide6.QtSvg import QSvgRenderer

        renderer = QSvgRenderer(str(icon_path))
        icon = QIcon()
        for size in (16, 24, 32, 48, 64, 128, 256):
            pm = QPixmap(size, size)
            pm.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pm)
            renderer.render(painter)
            painter.end()
            icon.addPixmap(pm)
        if not icon.isNull():
            return icon
    except Exception:
        pass
    return QIcon(str(icon_path))


def main(argv: list[str] | None = None) -> int:
    """Launch the BatPose GUI."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    # Set the Windows taskbar grouping identity before the QApplication so the
    # taskbar adopts our window icon rather than python.exe's.
    _set_windows_app_id()

    # Share a single OpenGL context across all widgets.  Without this, a second
    # pyqtgraph GLViewWidget (e.g. the fullscreen 3D pop-out alongside the inline
    # 3D preview) gets an unshared context and its GL items fail to draw —
    # symptom: blank canvas + repeated "Error while drawing item".  MUST be set
    # before the QApplication is constructed.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    app = QApplication(argv or sys.argv)
    app.setApplicationName("BatPose")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("BatPose")

    # Application icon (title bar / taskbar).
    icon_path = Path(__file__).resolve().parent.parent.parent / "assets" / "icon.svg"
    app_icon = _load_app_icon(icon_path)
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    # Use fusion style for consistent look across platforms
    app.setStyle("Fusion")

    from .main_window import MainWindow

    window = MainWindow()
    # Also set the icon on the window itself so it shows in the title bar even
    # if the app-level icon is overridden by a platform theme.
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
