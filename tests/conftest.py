"""
Top-level pytest configuration.

Sets QT_QPA_PLATFORM=offscreen so Qt widgets render without a display,
which is required for headless CI and for the UI tests to run in parallel
with other tests without popping windows.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
