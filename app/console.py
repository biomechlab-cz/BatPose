"""Console-output helpers for Windows-safe CLI printing."""

from __future__ import annotations

from typing import TextIO


def console_safe(text: object, stream: TextIO) -> str:
    """Return *text* encoded safely for *stream*.

    Windows consoles in this lab are often cp1250.  Plain ``print()`` raises
    ``UnicodeEncodeError`` when a progress callback sends characters such as
    warning signs, arrows, or ellipses.  Replacing only the unencodable
    characters preserves useful messages while preventing CLI crashes.
    """

    encoding = getattr(stream, "encoding", None) or "utf-8"
    return str(text).encode(encoding, errors="replace").decode(encoding, errors="replace")
