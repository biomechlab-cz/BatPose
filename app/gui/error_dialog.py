"""Shared user-facing error dialog with actionable messages and a 'Copy details' button."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class UserErrorDialog(QDialog):
    """
    Modal error dialog designed for non-programmer users.

    Shows a plain-English title + description, an optional actionable hint,
    optional technical detail (collapsible), and a 'Copy error details' button.
    """

    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        message: str,
        detail: str = "",
        action_hint: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Main message
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(msg_label)

        # Actionable hint
        if action_hint:
            hint_label = QLabel(f"<b>What to do next:</b><br>{action_hint}")
            hint_label.setWordWrap(True)
            hint_label.setStyleSheet(
                "color: #1a6b9a; padding: 6px; border-left: 3px solid #1a6b9a;"
            )
            layout.addWidget(hint_label)

        # Technical details (for support)
        if detail:
            self._detail_text = detail
            detail_edit = QTextEdit()
            detail_edit.setReadOnly(True)
            detail_edit.setPlainText(detail)
            detail_edit.setMaximumHeight(140)
            detail_edit.setStyleSheet(
                "font-family: monospace; font-size: 10px; background: #111; color: #ccc;"
            )
            layout.addWidget(detail_edit)
        else:
            self._detail_text = ""

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        if self._detail_text:
            copy_btn = QPushButton("Copy error details")
            copy_btn.clicked.connect(self._copy_detail)
            buttons.addButton(copy_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)

    def _copy_detail(self) -> None:
        QApplication.clipboard().setText(self._detail_text)


def show_worker_error(parent: QWidget | None, raw_error: str, context: str = "") -> None:
    """
    Show a UserErrorDialog from a worker error signal.

    Detects whether raw_error is a user-readable message or a Python traceback,
    and formats the dialog accordingly.

    Args:
        parent:     Parent widget for the dialog.
        raw_error:  String emitted by worker error signal.
        context:    Short description of what was running (e.g. "Calibration").
    """
    is_traceback = raw_error.startswith("Traceback") or "\nTraceback" in raw_error

    if is_traceback:
        # Extract the last line of the traceback as the short description
        lines = raw_error.strip().splitlines()
        short = lines[-1] if lines else raw_error[:120]
        title = f"{context} failed — unexpected error" if context else "Unexpected error"
        message = (
            f"An unexpected error occurred{' during ' + context.lower() if context else ''}.\n\n"
            f"{short}"
        )
        action_hint = (
            "Use <b>Copy error details</b> below and send them to support, "
            "or check the console for more information."
        )
        detail = raw_error
    else:
        title = f"{context} failed" if context else "Error"
        message = raw_error
        action_hint = _actionable_hint(raw_error)
        detail = ""

    dlg = UserErrorDialog(parent, title, message, detail=detail, action_hint=action_hint)
    dlg.exec()


def _actionable_hint(msg: str) -> str:
    """Return a context-specific hint based on keywords in the error message."""
    lower = msg.lower()
    if "no such file" in lower or "cannot open" in lower or "not found" in lower:
        return (
            "Check that the file exists and has not been moved or deleted. "
            "Use <b>Browse…</b> to re-select the file."
        )
    if "calibration" in lower and ("frame" in lower or "board" in lower):
        return (
            "Record a longer calibration video with the board clearly visible in both cameras. "
            "Aim for at least 20 frames with varied board positions."
        )
    if "mismatch" in lower or "count" in lower:
        return (
            "Ensure the left and right videos were recorded simultaneously "
            "and have the same number of frames."
        )
    if "cancelled" in lower:
        return "The operation was cancelled. Click Run to try again."
    return ""
