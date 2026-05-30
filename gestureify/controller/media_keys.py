"""
gestureify.controller.media_keys
==================================
OS-level media key fallback using ``pynput``.

When the Spotify Web API is unavailable (rate limited, no Premium, or no
active device), Gestureify falls back to simulating the system media keys
that the Spotify desktop client listens to natively.

Supported keys
--------------
* Play / Pause toggle
* Next Track
* Previous Track
* Volume Up / Down  (repeated presses to approximate a target level)

Design notes
------------
* ``pynput`` is used instead of ``pyautogui`` because it does not require a
  display server on macOS and handles media keys correctly on all platforms.
* Each public method is ≤ 10 lines (NASA Rule 4).
* The class is stateless; instantiate once and call freely.
"""

from __future__ import annotations

from pynput.keyboard import Controller, Key

from gestureify.utils.logger import get_logger

logger = get_logger(__name__)

# Number of volume key presses that represent one "step" in the volume scale.
_VOLUME_STEP_PRESSES: int = 2


class MediaKeyFallback:
    """Simulate OS media keys via ``pynput``.

    Instantiation is lightweight; a single ``pynput.keyboard.Controller``
    is created and reused for all key presses.
    """

    def __init__(self) -> None:
        self._kb = Controller()

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    def play_pause(self) -> None:
        """Simulate the Play/Pause media key."""
        self._tap(Key.media_play_pause)
        logger.debug("Media key: play/pause")

    def next_track(self) -> None:
        """Simulate the Next Track media key."""
        self._tap(Key.media_next)
        logger.debug("Media key: next track")

    def previous_track(self) -> None:
        """Simulate the Previous Track media key."""
        self._tap(Key.media_previous)
        logger.debug("Media key: previous track")

    def volume_up(self, steps: int = 1) -> None:
        """Simulate *steps* Volume Up key presses.

        Parameters
        ----------
        steps:
            Number of key presses (each press ≈ 2% volume on most systems).
        """
        for _ in range(max(1, steps) * _VOLUME_STEP_PRESSES):
            self._tap(Key.media_volume_up)
        logger.debug("Media key: volume up ×%d", steps)

    def volume_down(self, steps: int = 1) -> None:
        """Simulate *steps* Volume Down key presses.

        Parameters
        ----------
        steps:
            Number of key presses.
        """
        for _ in range(max(1, steps) * _VOLUME_STEP_PRESSES):
            self._tap(Key.media_volume_down)
        logger.debug("Media key: volume down ×%d", steps)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tap(self, key: Key) -> None:
        """Press and immediately release *key*."""
        self._kb.press(key)
        self._kb.release(key)
