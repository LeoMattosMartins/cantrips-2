"""
gestureify.utils.logger
=======================
Application-wide logging configuration.

A single call to ``configure_root_logger()`` from ``main.py`` is sufficient
to initialise the logging hierarchy for the entire process.  All other
modules obtain their logger via ``get_logger(__name__)`` and never call
``logging.basicConfig`` themselves.

Design notes
------------
* Maximum function length: 20 lines (NASA Rule 4 spirit).
* No global mutable state beyond the standard ``logging`` module internals.
"""

from __future__ import annotations

import logging
import os
import sys


def configure_root_logger() -> None:
    """Configure the root logger once at application startup.

    Reads ``LOG_LEVEL`` from the environment (set by
    ``gestureify.config.env_loader``) and attaches a ``StreamHandler``
    writing to *stderr*.  Safe to call multiple times; subsequent calls
    are no-ops if handlers are already attached.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured — do not add duplicate handlers.

    level_name: str = os.environ.get("LOG_LEVEL", "INFO").upper()
    level: int = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.setLevel(level)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A logger scoped to *name*.
    """
    return logging.getLogger(name)
