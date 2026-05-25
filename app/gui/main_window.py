"""Main application window."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from .calib_tab import CalibTab
from .capture_tab import CaptureTab
from .recon_tab import ReconTab
from .welcome_dialog import WelcomeDialog

_SESSION_FILE = Path.home() / ".config" / "wt-app" / "session.json"


class MainWindow(QMainWindow):
    """Top-level application window with Calibration, Reconstruction, and Capture tabs."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("wt-app  |  Stereo FLIR 3D Pose")
        self.resize(1280, 800)
        self._project_dir: str | None = None
        self._setup_ui()
        self._restore_session()
        self._maybe_show_welcome()

    def _setup_ui(self) -> None:
        # Menu bar
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")

        act_new = file_menu.addAction("&New Project…")
        act_new.setShortcut(QKeySequence("Ctrl+N"))
        act_new.triggered.connect(self._on_new_project)

        act_open = file_menu.addAction("&Open Project…")
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self._on_open_project)

        file_menu.addSeparator()

        act_export = file_menu.addAction("&Export CSV…")
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(self._on_export_csv)

        file_menu.addSeparator()

        act_quit = file_menu.addAction("&Quit")
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)

        help_menu = mb.addMenu("&Help")
        act_guide = help_menu.addAction("User &Guide")
        act_guide.setShortcut(QKeySequence("F1"))
        act_guide.triggered.connect(self._on_user_guide)
        act_coord = help_menu.addAction("&Coordinate Reference…")
        act_coord.triggered.connect(self._on_coord_ref)
        help_menu.addSeparator()
        act_about = help_menu.addAction("&About")
        act_about.triggered.connect(self._on_about)

        # Central: tab widget
        self._tabs = QTabWidget()
        self._calib_tab = CalibTab()
        self._recon_tab = ReconTab()
        self._capture_tab = CaptureTab()

        # Live Capture first — it's the primary workflow (calibrate live, then
        # track live).  The offline Calibration / Reconstruction tabs remain
        # available for processing pre-recorded videos.
        self._tabs.addTab(self._capture_tab, "Live Capture")
        self._tabs.addTab(self._calib_tab, "Calibration")
        self._tabs.addTab(self._recon_tab, "Reconstruction / 3D View")
        self.setCentralWidget(self._tabs)

        # Wire signals between tabs
        self._calib_tab.calibration_saved.connect(self._recon_tab.set_calibration)
        self._calib_tab.calibration_saved.connect(self._on_calib_saved)
        # Live calibration from the Capture tab feeds the same pipeline
        self._capture_tab.calibration_saved.connect(self._recon_tab.set_calibration)
        self._capture_tab.calibration_saved.connect(self._on_calib_saved)
        self._recon_tab.pose3d_ready.connect(self._on_pose3d_ready)
        self._capture_tab.recording_saved.connect(self._on_recording_saved)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._project_label = QLabel("No project open")
        self._status.addPermanentWidget(self._project_label)
        self._status.showMessage("Ready")

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    def _on_new_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select New Project Folder")
        if path:
            self._set_project(path)

    def _on_open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open Project Folder")
        if path:
            self._set_project(path)
            self._load_cached(path)

    def _set_project(self, path: str) -> None:
        self._project_dir = path
        self._project_label.setText(f"Project: {path}")
        self._status.showMessage(f"Project: {path}")
        self._calib_tab.set_project_dir(path)
        self._recon_tab.set_project_dir(path)
        self._capture_tab.set_project_dir(path)

    def _load_cached(self, project_dir: str) -> None:
        """Auto-load cached results if present."""
        proj = Path(project_dir)

        calib = proj / "calibration.yml"
        if calib.exists():
            self._recon_tab.set_calibration(str(calib))
            self._status.showMessage(f"Loaded cached calibration: {calib.name}")

        pose3d = proj / "pose3d.npz"
        if pose3d.exists():
            self._recon_tab._load_pose3d(str(pose3d))
            # Switch to the Reconstruction tab by widget reference (not a hard
            # coded index) so this stays correct if the tab order ever changes.
            self._tabs.setCurrentWidget(self._recon_tab)
            self._status.showMessage(f"Loaded cached pose3d: {pose3d.name}")

    # ------------------------------------------------------------------
    # Cross-tab signals
    # ------------------------------------------------------------------

    def _on_calib_saved(self, path: str) -> None:
        self._status.showMessage(f"Calibration saved: {path}")

    def _on_pose3d_ready(self, path: str) -> None:
        self._status.showMessage(f"3D pose ready: {path}")

    def _on_recording_saved(self, left: str, right: str) -> None:
        self._status.showMessage(f"Recording saved: {Path(left).name}, {Path(right).name}")

    def _on_export_csv(self) -> None:
        """Delegate to recon tab export."""
        self._recon_tab._on_export()

    def _on_user_guide(self) -> None:
        guide = Path(__file__).parent.parent.parent / "docs" / "quickstart.md"
        if guide.exists():
            QMessageBox.information(
                self,
                "User Guide",
                f"<b>User Guide</b> is available at:<br><tt>{guide}</tt><br><br>"
                "Open it in any markdown viewer or text editor.",
            )
        else:
            QMessageBox.information(
                self,
                "User Guide",
                "See <tt>docs/quickstart.md</tt> in the project directory.",
            )

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _save_session(self) -> None:
        try:
            ct = self._calib_tab
            rt = self._recon_tab
            session = {
                "project_dir": self._project_dir,
                "calib": {
                    "left": ct._left_edit.text(),
                    "right": ct._right_edit.text(),
                    "out": ct._out_edit.text(),
                },
                "recon": {
                    "left": rt._left_edit.text(),
                    "right": rt._right_edit.text(),
                    "calib": rt._calib_edit.text(),
                    "out": rt._out_edit.text(),
                },
                "capture": self._capture_tab.session_state(),
            }
            _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
            _SESSION_FILE.write_text(json.dumps(session, indent=2))
        except Exception:
            pass  # session save is best-effort

    def _restore_session(self) -> None:
        try:
            if not _SESSION_FILE.exists():
                return
            session = json.loads(_SESSION_FILE.read_text())
            if proj := session.get("project_dir"):
                if Path(proj).is_dir():
                    self._set_project(proj)

            ct = self._calib_tab
            if calib := session.get("calib", {}):
                if left := calib.get("left"):
                    ct._left_edit.setText(left)
                if right := calib.get("right"):
                    ct._right_edit.setText(right)
                if out := calib.get("out"):
                    ct._out_edit.setText(out)

            rt = self._recon_tab
            if recon := session.get("recon", {}):
                if left := recon.get("left"):
                    rt._left_edit.setText(left)
                if right := recon.get("right"):
                    rt._right_edit.setText(right)
                if calib_path := recon.get("calib"):
                    rt._calib_edit.setText(calib_path)
                if out := recon.get("out"):
                    rt._out_edit.setText(out)

            if cap := session.get("capture"):
                self._capture_tab.restore_session(cap)
        except Exception:
            pass  # restore is best-effort

    def closeEvent(self, event) -> None:
        # Stop the capture thread before the window (and its children) are
        # destroyed; otherwise the QThread C++ destructor runs while the thread
        # is still blocked in GetNextImage, producing a Qt warning / crash.
        self._capture_tab.cleanup()
        self._save_session()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Welcome / first-run dialog
    # ------------------------------------------------------------------

    def _maybe_show_welcome(self) -> None:
        """Show the welcome dialog when no project has been loaded from session."""
        if self._project_dir is not None:
            return  # restored from session — skip
        dlg = WelcomeDialog(self)
        dlg.exec()
        if dlg.open_sample:
            self._load_sample_project()
        elif dlg.new_project:
            self._on_new_project()

    def _load_sample_project(self) -> None:
        """Load bundled sample data if available, otherwise prompt for a folder."""
        from pathlib import Path

        sample_dir = Path(__file__).parent.parent.parent / "data"
        if sample_dir.is_dir():
            self._set_project(str(sample_dir))
            self._load_cached(str(sample_dir))
            self._status.showMessage("Sample project loaded.")
        else:
            QMessageBox.information(
                self,
                "Sample project not found",
                "No bundled sample data was found at <tt>data/</tt>.\n\n"
                "Please use <b>New Project…</b> to select your own project folder.",
            )
            self._on_new_project()

    # ------------------------------------------------------------------
    # About dialog
    # ------------------------------------------------------------------

    def _on_coord_ref(self) -> None:
        QMessageBox.information(
            self,
            "Coordinate Reference",
            "<b>Coordinate system (world frame)</b><br><br>"
            "<table>"
            "<tr><th align='left'>Axis</th><th align='left'>Direction</th></tr>"
            "<tr><td><b>X</b></td><td>Right (from camera 1 towards camera 2 baseline)</td></tr>"
            "<tr><td><b>Y</b></td><td>Up (opposite to gravity)</td></tr>"
            "<tr><td><b>Z</b></td><td>Out of the camera (towards the subject)</td></tr>"
            "</table><br>"
            "<b>Units:</b> metres (m)<br><br>"
            "<b>COCO-17 joint indices</b><br>"
            "<table>"
            "<tr><td>0</td><td>Nose</td>    <td>9</td> <td>L Wrist</td></tr>"
            "<tr><td>1</td><td>L Eye</td>   <td>10</td><td>R Wrist</td></tr>"
            "<tr><td>2</td><td>R Eye</td>   <td>11</td><td>L Hip</td></tr>"
            "<tr><td>3</td><td>L Ear</td>   <td>12</td><td>R Hip</td></tr>"
            "<tr><td>4</td><td>R Ear</td>   <td>13</td><td>L Knee</td></tr>"
            "<tr><td>5</td><td>L Shoulder</td><td>14</td><td>R Knee</td></tr>"
            "<tr><td>6</td><td>R Shoulder</td><td>15</td><td>L Ankle</td></tr>"
            "<tr><td>7</td><td>L Elbow</td><td>16</td><td>R Ankle</td></tr>"
            "<tr><td>8</td><td>R Elbow</td><td></td><td></td></tr>"
            "</table>",
        )

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About wt-app",
            "<b>wt-app</b> v0.1.0<br>"
            "Stereo FLIR Calibration + 3D Human Pose Estimation<br><br>"
            "CPU-first pipeline using:<br>"
            "• MediaPipe Pose Landmarker (COCO-17)<br>"
            "• ChArUco / chessboard stereo calibration<br>"
            "• Linear triangulation + OneEuro smoothing<br><br>"
            "See docs/quickstart.md for usage.",
        )
