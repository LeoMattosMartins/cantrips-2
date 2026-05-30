"""
gestureify.controller.playback_controller
==========================================
High-level playback command dispatcher.

``PlaybackController`` is the single entry-point for all playback actions.
It:
  1. Tries the Spotify Web API first (via ``SpotifyClient``).
  2. Falls back to OS media keys (via ``MediaKeyFallback``) if the API is
     unavailable or the user does not have Premium.
  3. Throttles volume API calls to avoid rate-limit errors.
  4. Maintains a cooldown between play/pause toggles to prevent double-fires.

Design notes
------------
* ``_use_api`` is a runtime flag that permanently switches to the fallback
  after a ``SpotifyPremiumRequired`` exception, avoiding repeated 403 calls.
* ``RateGate`` objects enforce all throttle windows (NASA Rule 7).
* Volume delta is computed against a *pre-update* snapshot of ``_last_volume``
  to avoid the off-by-one bug where the delta is always zero.
"""

from __future__ import annotations

from gestureify.auth.spotify_auth import SpotifyAuth
from gestureify.config import settings
from gestureify.controller.media_keys import MediaKeyFallback
from gestureify.controller.spotify_client import (
    SpotifyClient,
    SpotifyPremiumRequired,
    SpotifyRateLimited,
)
from gestureify.utils.logger import get_logger
from gestureify.utils.timing import RateGate

logger = get_logger(__name__)


class PlaybackController:
    """Dispatch playback commands to Spotify API or OS media keys.

    Parameters
    ----------
    auth:
        ``SpotifyAuth`` instance used to obtain fresh access tokens.
    spotify_client:
        ``SpotifyClient`` instance for Web API calls.
    media_keys:
        ``MediaKeyFallback`` instance for OS-level control.
    """

    def __init__(
        self,
        auth: SpotifyAuth,
        spotify_client: SpotifyClient,
        media_keys: MediaKeyFallback,
    ) -> None:
        self._auth = auth
        self._spotify = spotify_client
        self._keys = media_keys

        # True while the Spotify Web API is usable.
        self._use_api: bool = True

        # Throttle gates.
        self._volume_gate = RateGate(settings.VOLUME_API_THROTTLE_SECONDS)
        self._playpause_gate = RateGate(settings.PLAYPAUSE_COOLDOWN_SECONDS)

        # Track last known volume to compute deltas for the key fallback.
        self._last_volume: int = 50

    # ------------------------------------------------------------------
    # Public playback commands
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Pause playback."""
        if not self._playpause_gate.allow():
            return
        logger.info("Command: PAUSE")
        if self._use_api:
            self._try_api(lambda t: self._spotify.pause(t), fallback=self._keys.play_pause)
        else:
            self._keys.play_pause()

    def resume(self) -> None:
        """Resume playback."""
        if not self._playpause_gate.allow():
            return
        logger.info("Command: RESUME")
        if self._use_api:
            self._try_api(lambda t: self._spotify.resume(t), fallback=self._keys.play_pause)
        else:
            self._keys.play_pause()

    def next_track(self) -> None:
        """Skip to the next track."""
        logger.info("Command: NEXT TRACK")
        if self._use_api:
            self._try_api(
                lambda t: self._spotify.next_track(t),
                fallback=self._keys.next_track,
            )
        else:
            self._keys.next_track()

    def previous_track(self) -> None:
        """Go to the previous track."""
        logger.info("Command: PREVIOUS TRACK")
        if self._use_api:
            self._try_api(
                lambda t: self._spotify.previous_track(t),
                fallback=self._keys.previous_track,
            )
        else:
            self._keys.previous_track()

    def set_volume_from_pinch(self, pinch_gap: float) -> None:
        """Map a normalised pinch gap to a volume level and apply it.

        The pinch gap is expected to be in the range [0, ~0.5].  It is
        linearly mapped to [0, 100] percent.

        Parameters
        ----------
        pinch_gap:
            Normalised thumb-index distance from ``GestureClassifier.pinch_gap()``.
        """
        if not self._volume_gate.allow():
            return

        max_gap = 0.5
        target = int(min(pinch_gap / max_gap, 1.0) * 100)
        target = max(0, min(100, target))

        if target == self._last_volume:
            return

        # Capture delta *before* updating _last_volume.
        delta = target - self._last_volume
        self._last_volume = target

        logger.info("Command: SET VOLUME %d%%", target)

        if self._use_api:
            self._try_api(
                lambda t: self._spotify.set_volume(t, target),
                fallback=lambda: self._volume_key_fallback(delta),
            )
        else:
            self._volume_key_fallback(delta)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_api(self, api_call, fallback) -> None:  # type: ignore[type-arg]
        """Attempt an API call; fall back to media keys on known errors."""
        try:
            token = self._auth.ensure_valid_token()
            api_call(token)
        except SpotifyPremiumRequired:
            logger.warning(
                "Spotify Premium not detected. Switching permanently to OS media keys."
            )
            self._use_api = False
            fallback()
        except SpotifyRateLimited:
            logger.warning("Rate limited by Spotify; using media keys this time.")
            fallback()
        except Exception as exc:  # noqa: BLE001
            logger.error("Spotify API call failed: %s; using media keys.", exc)
            fallback()

    @staticmethod
    def _volume_key_fallback(delta: int) -> None:
        """Approximate a volume change using media key presses.

        Parameters
        ----------
        delta:
            Signed volume change in percentage points.  Positive = louder.
        """
        from gestureify.controller.media_keys import MediaKeyFallback  # avoid circular at module level

        keys = MediaKeyFallback()
        steps = abs(delta) // 5  # Each step ≈ 5% volume.
        if steps == 0:
            return
        if delta > 0:
            keys.volume_up(steps)
        else:
            keys.volume_down(steps)
