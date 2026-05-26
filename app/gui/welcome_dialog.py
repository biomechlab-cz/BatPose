"""Welcome dialog shown on first launch when no project is open."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class WelcomeDialog(QDialog):
    """
    First-run welcome overlay.

    Shown when no session is present.  Guides new users through the
    three-step workflow and provides shortcuts to open a project or
    load the built-in sample data.
    """

    #: Set to True if the user clicked 'Open Sample Project'
    open_sample: bool = False
    #: Set to True if the user clicked 'New Project'
    new_project: bool = False

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to BatPose")
        self.setMinimumWidth(560)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 24, 28, 20)

        # Title
        title = QLabel("<h2>Welcome to Stereo Biomechanics</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Workflow steps
        steps_html = (
            "<ol style='line-height:1.8em;'>"
            "<li><b>Calibrate</b> your stereo camera pair using a ChArUco or chessboard video.</li>"
            "<li><b>Process</b> exercise videos to extract 3D joint positions.</li>"
            "<li><b>Visualise</b>, play back, and export results as CSV.</li>"
            "</ol>"
        )
        steps = QLabel(steps_html)
        steps.setWordWrap(True)
        layout.addWidget(steps)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._sample_btn = QPushButton("Open Sample Project")
        self._sample_btn.setToolTip("Load the bundled sample data for an instant demonstration")
        self._sample_btn.clicked.connect(self._on_sample)

        self._new_btn = QPushButton("New Project…")
        self._new_btn.setToolTip("Choose a folder to use as your project directory")
        self._new_btn.clicked.connect(self._on_new)

        self._guide_btn = QPushButton("User Guide")
        self._guide_btn.setToolTip("Open docs/quickstart.md")
        self._guide_btn.clicked.connect(self._on_guide)

        btn_row.addWidget(self._sample_btn)
        btn_row.addWidget(self._new_btn)
        btn_row.addWidget(self._guide_btn)
        layout.addLayout(btn_row)

        # Close / skip button
        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.accept)
        layout.addWidget(close_box)

    def _on_sample(self) -> None:
        self.open_sample = True
        self.accept()

    def _on_new(self) -> None:
        self.new_project = True
        self.accept()

    def _on_guide(self) -> None:
        from pathlib import Path

        from PySide6.QtWidgets import QMessageBox

        guide = Path(__file__).parent.parent.parent / "docs" / "quickstart.md"
        if guide.exists():
            QMessageBox.information(
                self,
                "User Guide",
                f"<b>User Guide</b> is available at:<br><tt>{guide}</tt><br><br>"
                "Open it in any Markdown viewer or text editor.",
            )
        else:
            QMessageBox.information(
                self,
                "User Guide",
                "See <tt>docs/quickstart.md</tt> in the project directory.",
            )
