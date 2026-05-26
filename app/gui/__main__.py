"""Entry point: python -m app.gui"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Launch the wt-app GUI."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    # Share a single OpenGL context across all widgets.  Without this, a second
    # pyqtgraph GLViewWidget (e.g. the fullscreen 3D pop-out alongside the inline
    # 3D preview) gets an unshared context and its GL items fail to draw —
    # symptom: blank canvas + repeated "Error while drawing item".  MUST be set
    # before the QApplication is constructed.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    app = QApplication(argv or sys.argv)
    app.setApplicationName("wt-app")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("wt")

    # Use fusion style for consistent look across platforms
    app.setStyle("Fusion")

    from .main_window import MainWindow

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
