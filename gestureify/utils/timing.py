"""
gestureify.utils.timing
=======================
Lightweight timing primitives used across the codebase.

``RateGate``  -- enforces a minimum interval between allowed events.
``Stopwatch`` -- measures elapsed time since an arbitrary reference point.

Both classes are thread-safe for the single-threaded use-case of this
application and carry no external dependencies beyond the standard library.
"""

from __future__ import annotations

import time


class RateGate:
    """Allow an event at most once every *min_interval* seconds.

    Usage
    -----
    ::

        gate = RateGate(min_interval=0.3)

        if gate.allow():
            spotify.set_volume(vol)   # called at most ~3 times/second

    Parameters
    ----------
    min_interval:
        Minimum number of seconds that must elapse between two ``allow()``
        calls that return ``True``.
    """

    def __init__(self, min_interval: float) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative.")
        self._min_interval: float = min_interval
        self._last_allowed: float = 0.0

    def allow(self) -> bool:
        """Return ``True`` if enough time has elapsed since the last event.

        Side-effects
        ------------
        Updates the internal timestamp when returning ``True``.
        """
        now = time.monotonic()
        if now - self._last_allowed >= self._min_interval:
            self._last_allowed = now
            return True
        return False

    def reset(self) -> None:
        """Force the gate open immediately (next ``allow()`` will return True)."""
        self._last_allowed = 0.0


class Stopwatch:
    """Measure elapsed wall-clock time from a reference point.

    Usage
    -----
    ::

        sw = Stopwatch()
        sw.start()
        ...
        if sw.elapsed() > 1.5:
            trigger_wake()

    The stopwatch is *not* started on construction; call ``start()``
    explicitly.
    """

    def __init__(self) -> None:
        self._start: float | None = None

    def start(self) -> None:
        """Record the current monotonic time as the reference point."""
        self._start = time.monotonic()

    def elapsed(self) -> float:
        """Return seconds elapsed since ``start()`` was called.

        Returns
        -------
        float
            Elapsed seconds, or 0.0 if ``start()`` has never been called.
        """
        if self._start is None:
            return 0.0
        return time.monotonic() - self._start

    def reset(self) -> None:
        """Clear the reference point (``elapsed()`` returns 0.0 again)."""
        self._start = None

    @property
    def is_running(self) -> bool:
        """``True`` if ``start()`` has been called without a subsequent ``reset()``."""
        return self._start is not None
