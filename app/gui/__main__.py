"""Entry point: python -m app.gui"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Launch the wt-app GUI."""
    from PySide6.QtWidgets import QApplication

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
