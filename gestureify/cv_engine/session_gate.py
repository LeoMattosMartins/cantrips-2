"""
gestureify.cv_engine.session_gate
==================================
Session-level activation gate driven by the open-palm wake gesture.

The gate implements the following state machine::

    ┌──────────────────────────────────────────────────────────────┐
    │                        IDLE                                  │
    │  Camera active, landmarks tracked, but no commands fired.    │
    │  Waiting for OPEN_PALM held ≥ WAKE_HOLD_SECONDS.             │
    └──────────────────────┬───────────────────────────────────────┘
                           │  palm held long enough
                           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                       ACTIVE                                 │
    │  Gestures are dispatched to the controller.                  │
    │  Waiting for OPEN_PALM held ≥ WAKE_HOLD_SECONDS to deactivate│
    └──────────────────────┬───────────────────────────────────────┘
                           │  palm held long enough
                           ▼
                         IDLE  (cycle repeats)

The open-palm gesture is therefore a **toggle**: one hold activates the
session, the next hold deactivates it.  The HUD reflects the current state
in real time.

Design notes
------------
* State transitions are explicit and exhaustive (NASA Rule 5).
* The ``Stopwatch`` is reset whenever the palm is *not* detected, preventing
  accidental activation from brief flashes of an open hand.
"""

from __future__ import annotations

from enum import Enum, auto

from gestureify.config import settings
from gestureify.cv_engine.gesture_classifier import GestureLabel
from gestureify.utils.logger import get_logger
from gestureify.utils.timing import Stopwatch

logger = get_logger(__name__)


class SessionState(Enum):
    """Possible states of the gesture session."""

    IDLE = auto()    # Commands are suppressed.
    ACTIVE = auto()  # Commands are dispatched.


class SessionGate:
    """Toggle gesture-command dispatching via a held open-palm gesture.

    Parameters
    ----------
    hold_seconds:
        How long the open palm must be held continuously to trigger a
        state transition.
    """

    def __init__(
        self,
        hold_seconds: float = settings.WAKE_HOLD_SECONDS,
    ) -> None:
        self._hold_seconds = hold_seconds
        self._state: SessionState = SessionState.IDLE
        self._stopwatch: Stopwatch = Stopwatch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> SessionState:
        """Current session state (read-only)."""
        return self._state

    @property
    def is_active(self) -> bool:
        """``True`` when gesture commands should be dispatched."""
        return self._state is SessionState.ACTIVE

    def update(self, gesture: GestureLabel) -> bool:
        """Feed the latest gesture label and return whether a toggle occurred.

        Call this once per frame with the classified gesture.

        Parameters
        ----------
        gesture:
            The ``GestureLabel`` for the current frame.

        Returns
        -------
        bool
            ``True`` if the session state just changed (useful for HUD
            animations).
        """
        if gesture is GestureLabel.OPEN_PALM:
            return self._handle_palm_present()
        else:
            self._handle_palm_absent()
            return False

    def hold_progress(self) -> float:
        """Return the wake-gesture hold progress as a value in [0.0, 1.0].

        Useful for rendering a progress arc in the HUD.
        """
        if not self._stopwatch.is_running:
            return 0.0
        return min(self._stopwatch.elapsed() / self._hold_seconds, 1.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_palm_present(self) -> bool:
        """Advance the hold timer; toggle state if threshold is reached."""
        if not self._stopwatch.is_running:
            self._stopwatch.start()
            logger.debug("Wake-gesture hold started.")

        if self._stopwatch.elapsed() >= self._hold_seconds:
            self._stopwatch.reset()
            return self._toggle()

        return False

    def _handle_palm_absent(self) -> None:
        """Reset the hold timer when the palm is no longer detected."""
        if self._stopwatch.is_running:
            self._stopwatch.reset()
            logger.debug("Wake-gesture hold interrupted.")

    def _toggle(self) -> bool:
        """Flip the session state and log the transition."""
        if self._state is SessionState.IDLE:
            self._state = SessionState.ACTIVE
            logger.info("Session ACTIVATED via open-palm gesture.")
        else:
            self._state = SessionState.IDLE
            logger.info("Session DEACTIVATED via open-palm gesture.")
        return True
