"""
gestureify.assets.font_loader
==============================
Registers the bundled Monocraft font with the OS font system so that Tkinter
can reference it by family name.

Strategy
--------
Tkinter's ``font`` module cannot load arbitrary TTF files directly — it can
only reference fonts that are already registered with the underlying OS font
renderer (fontconfig on Linux, GDI on Windows, CoreText on macOS).

We use ``pyglet`` as a lightweight font-loading shim: ``pyglet.font.add_file``
registers the TTF with the OS font system at runtime, after which Tkinter can
address the family by name.  If ``pyglet`` is not installed, we fall back to
``Courier New`` (already bundled in the project's monospace palette) and log a
warning rather than crashing.

Design notes
------------
* ``load()`` is idempotent — calling it multiple times is safe.
* The font family name exposed to Tkinter is ``"Monocraft"``.
* Font files are resolved relative to this module's directory so the loader
  works regardless of the current working directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

_FONT_DIR: Final[Path] = Path(__file__).parent / "fonts"
_REGULAR_TTF: Final[Path] = _FONT_DIR / "Monocraft.ttf"
_BOLD_TTF: Final[Path] = _FONT_DIR / "Monocraft-Bold.ttf"

# Family name Tkinter will use after registration.
MONOCRAFT_FAMILY: Final[str] = "Monocraft"
# Fallback when pyglet is unavailable.
FALLBACK_FAMILY: Final[str] = "Courier New"

_loaded: bool = False


def load() -> str:
    """Register Monocraft with the OS font system and return the family name.

    Returns
    -------
    str
        ``"Monocraft"`` on success, ``"Courier New"`` if pyglet is absent or
        the font files are missing.
    """
    global _loaded  # noqa: PLW0603

    if _loaded:
        return MONOCRAFT_FAMILY

    if not _REGULAR_TTF.exists():
        logger.warning(
            "Monocraft.ttf not found at %s; falling back to %s.",
            _REGULAR_TTF,
            FALLBACK_FAMILY,
        )
        return FALLBACK_FAMILY

    try:
        import pyglet.font as pgfont  # type: ignore[import]

        pgfont.add_file(str(_REGULAR_TTF))
        pgfont.add_file(str(_BOLD_TTF))
        _loaded = True
        logger.info("Monocraft font registered via pyglet.")
        return MONOCRAFT_FAMILY
    except ImportError:
        pass  # pyglet not installed — try tkinter font trick

    # Tkinter-only fallback: copy the font into the user font directory so
    # fontconfig picks it up on Linux.  No-op on Windows/macOS without pyglet.
    try:
        import shutil
        import subprocess

        user_font_dir = Path.home() / ".local" / "share" / "fonts"
        user_font_dir.mkdir(parents=True, exist_ok=True)
        dest_regular = user_font_dir / _REGULAR_TTF.name
        dest_bold = user_font_dir / _BOLD_TTF.name

        if not dest_regular.exists():
            shutil.copy2(_REGULAR_TTF, dest_regular)
        if not dest_bold.exists():
            shutil.copy2(_BOLD_TTF, dest_bold)

        # Refresh fontconfig cache (Linux only; silently ignored elsewhere).
        subprocess.run(  # noqa: S603
            ["fc-cache", "-f", str(user_font_dir)],
            capture_output=True,
            timeout=10,
        )
        _loaded = True
        logger.info("Monocraft font installed to %s via fontconfig.", user_font_dir)
        return MONOCRAFT_FAMILY
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not register Monocraft font (%s); falling back to %s.",
            exc,
            FALLBACK_FAMILY,
        )
        return FALLBACK_FAMILY
