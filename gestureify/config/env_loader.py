"""
gestureify.config.env_loader
============================
Loads user-supplied secrets and optional setting overrides from a ``.env``
file located at the project root.

Only the keys listed in ``REQUIRED_KEYS`` are mandatory; everything else is
optional.  The module raises a descriptive ``EnvironmentError`` on startup if
any required key is absent, so the user gets an actionable message rather than
a cryptic ``KeyError`` deep inside the application.

Design notes
------------
* This module has **no side-effects on import** beyond reading the file.
  Call ``load()`` explicitly from ``main.py``.
* Secrets are never logged.
* All public functions have a cyclomatic complexity ≤ 10 (NASA Rule 4).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

from gestureify.config import settings

logger = logging.getLogger(__name__)

# Keys that MUST be present in the environment (or .env file).
REQUIRED_KEYS: tuple[str, ...] = ("SPOTIFY_CLIENT_ID",)

# Optional keys with defaults sourced from settings (single source of truth).
_OPTIONAL_DEFAULTS: Dict[str, str] = {
    "SPOTIFY_REDIRECT_URI": settings.SPOTIFY_REDIRECT_URI,
    "LOG_LEVEL": settings.LOG_LEVEL,
    "CAMERA_INDEX": str(settings.CAMERA_INDEX),
}


def load(env_path: Path | None = None) -> None:
    """Parse ``.env`` and inject values into ``os.environ``.

    Parameters
    ----------
    env_path:
        Explicit path to the ``.env`` file.  When *None* the function walks
        upward from the current working directory until it finds ``.env`` or
        reaches the filesystem root.

    Raises
    ------
    EnvironmentError
        If any key listed in ``REQUIRED_KEYS`` is missing after loading.
    FileNotFoundError
        If *env_path* is given explicitly but does not exist.
    """
    resolved = _resolve_env_path(env_path)

    if resolved is not None:
        _parse_and_inject(resolved)
        logger.debug("Loaded environment from %s", resolved)
    else:
        logger.debug("No .env file found; relying on existing environment variables.")

    _apply_optional_defaults()
    _validate_required()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_env_path(env_path: Path | None) -> Path | None:
    """Return the resolved .env path or *None* if it cannot be found."""
    if env_path is not None:
        if not env_path.exists():
            raise FileNotFoundError(f".env file not found at: {env_path}")
        return env_path

    # Walk upward from cwd.
    candidate = Path.cwd()
    for _ in range(10):  # bounded loop — NASA Rule 2
        probe = candidate / ".env"
        if probe.exists():
            return probe
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    return None


def _parse_and_inject(path: Path) -> None:
    """Read *path* line-by-line and set environment variables."""
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                logger.warning(".env line %d skipped (no '='): %r", lineno, line)
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _apply_optional_defaults() -> None:
    """Set optional keys to their defaults when absent from the environment."""
    for key, default in _OPTIONAL_DEFAULTS.items():
        os.environ.setdefault(key, default)


def _validate_required() -> None:
    """Raise ``EnvironmentError`` listing all missing required keys."""
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            "The following required environment variables are not set:\n"
            + "\n".join(f"  - {k}" for k in missing)
            + "\n\nCreate a .env file in the project root with these values."
            + "\nSee .env.example for a template."
        )
