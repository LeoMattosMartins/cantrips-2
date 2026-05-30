"""
gestureify.cv_engine.swipe_detector
=====================================
Velocity-based horizontal swipe detector with a rearm cooldown.

Algorithm
---------
1. Each frame, the normalised wrist X-coordinate is pushed onto a fixed-size
   ring buffer (length = ``SWIPE_VELOCITY_WINDOW``).
2. Velocity is estimated as the signed difference between the newest and
   oldest sample in the buffer, divided by the window length.
3. If |velocity| > ``SWIPE_VELOCITY_THRESHOLD`` and the detector is not in
   COOLDOWN, a ``SwipeDirection`` is emitted.
4. The detector enters COOLDOWN for ``SWIPE_COOLDOWN_SECONDS``.  During
   cooldown the buffer is cleared, preventing the hand's return motion from
   triggering a reverse swipe.

Design notes
------------
* The ring buffer has a fixed maximum size (NASA Rule 2: bounded data structures).
* State transitions are explicit (NASA Rule 5).
* The detector is completely stateless with respect to the Spotify API;
  it only emits direction signals.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import Deque, Optional

from gestureify.config import settings
from gestureify.utils.logger import get_logger
from gestureify.utils.timing import Stopwatch

logger = get_logger(__name__)


class SwipeDirection(Enum):
    """Direction of a detected swipe gesture."""

    LEFT = auto()   # Hand moved right-to-left → previous track.
    RIGHT = auto()  # Hand moved left-to-right → next track.


class _DetectorState(Enum):
    READY = auto()
    COOLDOWN = auto()


class SwipeDetector:
    """Detect horizontal swipe gestures from a stream of wrist X positions.

    Parameters
    ----------
    velocity_window:
        Number of frames used to estimate velocity.
    velocity_threshold:
        Minimum mean velocity (normalised units per frame) to register a swipe.
    cooldown_seconds:
        Duration of the rearm window after a swipe fires.
    """

    def __init__(
        self,
        velocity_window: int = settings.SWIPE_VELOCITY_WINDOW,
        velocity_threshold: float = settings.SWIPE_VELOCITY_THRESHOLD,
        cooldown_seconds: float = settings.SWIPE_COOLDOWN_SECONDS,
    ) -> None:
        if velocity_window < 2:
            raise ValueError("velocity_window must be at least 2.")
        self._window_size = velocity_window
        self._threshold = velocity_threshold
        self._cooldown_seconds = cooldown_seconds

        self._buffer: Deque[float] = deque(maxlen=velocity_window)
        self._state: _DetectorState = _DetectorState.READY
        self._cooldown_timer: Stopwatch = Stopwatch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, wrist_x: Optional[float]) -> Optional[SwipeDirection]:
        """Feed the current wrist X position and return a swipe if detected.

        Parameters
        ----------
        wrist_x:
            Normalised [0, 1] wrist X coordinate, or *None* if no hand is
            in frame (causes a buffer reset).

        Returns
        -------
        SwipeDirection | None
            A swipe direction if one was just detected, otherwise *None*.
        """
        self._check_cooldown_expiry()

        if wrist_x is None:
            self._reset_buffer()
            return None

        if self._state is _DetectorState.COOLDOWN:
            return None

        self._buffer.append(wrist_x)

        if len(self._buffer) < self._window_size:
            return None  # Not enough samples yet.

        return self._evaluate_velocity()

    def reset(self) -> None:
        """Fully reset the detector (buffer, state, and cooldown timer)."""
        self._reset_buffer()
        self._state = _DetectorState.READY
        self._cooldown_timer.reset()

    @property
    def in_cooldown(self) -> bool:
        """``True`` while the rearm window is active."""
        return self._state is _DetectorState.COOLDOWN

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_velocity(self) -> Optional[SwipeDirection]:
        """Compute velocity and emit a direction if the threshold is exceeded."""
        # Velocity = displacement over the window, normalised by window length.
        oldest = self._buffer[0]
        newest = self._buffer[-1]
        velocity = (newest - oldest) / self._window_size

        if abs(velocity) < self._threshold:
            return None

        direction = SwipeDirection.RIGHT if velocity > 0 else SwipeDirection.LEFT
        logger.info("Swipe detected: %s (velocity=%.4f)", direction.name, velocity)
        self._enter_cooldown()
        return direction

    def _enter_cooldown(self) -> None:
        """Transition to COOLDOWN and clear the buffer."""
        self._state = _DetectorState.COOLDOWN
        self._cooldown_timer.start()
        self._reset_buffer()
        logger.debug("Swipe detector entering cooldown (%.2f s).", self._cooldown_seconds)

    def _check_cooldown_expiry(self) -> None:
        """Exit COOLDOWN if the timer has elapsed."""
        if (
            self._state is _DetectorState.COOLDOWN
            and self._cooldown_timer.elapsed() >= self._cooldown_seconds
        ):
            self._state = _DetectorState.READY
            self._cooldown_timer.reset()
            logger.debug("Swipe detector rearmed.")

    def _reset_buffer(self) -> None:
        """Clear the velocity ring buffer."""
        self._buffer.clear()
