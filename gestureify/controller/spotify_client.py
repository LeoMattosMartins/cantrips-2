"""
gestureify.controller.spotify_client
======================================
Thin, typed wrapper around the Spotify Web API playback endpoints.

Only the endpoints required by Gestureify are exposed.  Each method:
  * Accepts a fresh ``access_token`` (obtained from ``SpotifyAuth``).
  * Returns ``True`` on success, ``False`` on a recoverable error.
  * Raises ``SpotifyPremiumRequired`` when a 403 is returned, so the caller
    can switch to the OS media-key fallback permanently.
  * Raises ``SpotifyRateLimited`` on a 429 so the caller can back off.

Design notes
------------
* All HTTP calls use the standard library ``urllib`` to avoid adding
  ``requests`` as a dependency (keeps the install footprint small).
* Functions are bounded in complexity (≤ 15 lines of logic each, NASA Rule 4).
* No global mutable state; the token is passed per-call so the client is
  safe to use from any thread.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from gestureify.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.spotify.com/v1/me/player"


class SpotifyPremiumRequired(Exception):
    """Raised when the Spotify API returns 403 (Premium required)."""


class SpotifyRateLimited(Exception):
    """Raised when the Spotify API returns 429 (Too Many Requests).

    Attributes
    ----------
    retry_after:
        Seconds to wait before retrying, as reported by the API.
    """

    def __init__(self, retry_after: int = 1) -> None:
        super().__init__(f"Rate limited; retry after {retry_after}s.")
        self.retry_after = retry_after


class SpotifyClient:
    """Issue playback commands to the Spotify Web API.

    All methods require a valid ``access_token``.  Obtain one via
    ``SpotifyAuth.ensure_valid_token()``.
    """

    # ------------------------------------------------------------------
    # Playback commands
    # ------------------------------------------------------------------

    def pause(self, access_token: str) -> bool:
        """Pause the currently playing track."""
        return self._put(access_token, "/pause")

    def resume(self, access_token: str) -> bool:
        """Resume playback."""
        return self._put(access_token, "/play")

    def next_track(self, access_token: str) -> bool:
        """Skip to the next track in the queue."""
        return self._post(access_token, "/next")

    def previous_track(self, access_token: str) -> bool:
        """Go back to the previous track."""
        return self._post(access_token, "/previous")

    def set_volume(self, access_token: str, volume_percent: int) -> bool:
        """Set playback volume.

        Parameters
        ----------
        access_token:
            Valid Spotify bearer token.
        volume_percent:
            Integer in [0, 100].
        """
        clamped = max(0, min(100, volume_percent))
        params = urllib.parse.urlencode({"volume_percent": clamped})
        return self._put(access_token, f"/volume?{params}")

    def get_playback_state(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Fetch the current playback state.

        Returns
        -------
        dict | None
            Parsed JSON response, or *None* on error / no active device.
        """
        url = _BASE_URL
        req = urllib.request.Request(
            url,
            headers=self._auth_headers(access_token),
            method="GET",
        )
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310
                if resp.status == 204:
                    return None  # No active device.
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self._handle_http_error(exc)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _put(self, token: str, path: str, body: Optional[dict] = None) -> bool:
        """Send a PUT request to the player endpoint."""
        return self._request("PUT", token, path, body)

    def _post(self, token: str, path: str, body: Optional[dict] = None) -> bool:
        """Send a POST request to the player endpoint."""
        return self._request("POST", token, path, body)

    def _request(
        self,
        method: str,
        token: str,
        path: str,
        body: Optional[dict],
    ) -> bool:
        """Execute an HTTP request and return success/failure."""
        url = _BASE_URL + path
        data = json.dumps(body).encode("utf-8") if body else b""
        headers = self._auth_headers(token)
        if data:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310
                logger.debug("%s %s → %d", method, path, resp.status)
                return resp.status in (200, 204)
        except urllib.error.HTTPError as exc:
            self._handle_http_error(exc)
            return False

    @staticmethod
    def _auth_headers(token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _handle_http_error(exc: urllib.error.HTTPError) -> None:
        """Raise typed exceptions for known error codes."""
        if exc.code == 403:
            raise SpotifyPremiumRequired(
                "Spotify Premium is required for playback control via the Web API."
            )
        if exc.code == 429:
            retry_after = int(exc.headers.get("Retry-After", "1"))
            raise SpotifyRateLimited(retry_after=retry_after)
        logger.warning("Spotify API error: HTTP %d — %s", exc.code, exc.reason)
