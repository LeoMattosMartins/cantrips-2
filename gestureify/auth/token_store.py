"""
gestureify.auth.token_store
===========================
Persistent, file-backed storage for Spotify OAuth2 tokens.

Tokens are written as JSON to a local file (path configurable via
``gestureify.config.settings.TOKEN_CACHE_PATH``).  The file is created with
mode ``0o600`` (owner read/write only) to reduce the risk of credential
exposure.

Design notes
------------
* The class raises typed exceptions rather than returning sentinel values so
  callers can handle failure explicitly (NASA Rule 7: check return values).
* Atomic write via a temporary file + rename prevents partial writes from
  corrupting the cache.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from gestureify.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TokenBundle:
    """Container for a Spotify OAuth2 token set.

    Attributes
    ----------
    access_token:
        Short-lived bearer token used in API requests.
    refresh_token:
        Long-lived token used to obtain a new ``access_token``.
    expires_at:
        Unix timestamp (float) at which ``access_token`` expires.
    token_type:
        Always ``"Bearer"`` for Spotify.
    scope:
        Space-separated scopes granted by the user.
    """

    access_token: str
    refresh_token: str
    expires_at: float
    token_type: str = "Bearer"
    scope: str = ""

    @property
    def is_expired(self) -> bool:
        """``True`` if the access token has expired."""
        return time.time() >= self.expires_at

    def expires_in_seconds(self) -> float:
        """Return seconds until expiry (negative if already expired)."""
        return self.expires_at - time.time()


class TokenStore:
    """Read and write ``TokenBundle`` objects to a local JSON file.

    Parameters
    ----------
    cache_path:
        Path to the JSON cache file.  The parent directory must exist.
    """

    def __init__(self, cache_path: str | Path) -> None:
        self._path = Path(cache_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, bundle: TokenBundle) -> None:
        """Persist *bundle* to disk atomically.

        Parameters
        ----------
        bundle:
            The token bundle to write.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        data = asdict(bundle)
        self._atomic_write(data)
        logger.debug("Token bundle saved to %s", self._path)

    def load(self) -> Optional[TokenBundle]:
        """Load and return the cached token bundle.

        Returns
        -------
        TokenBundle | None
            The cached bundle, or *None* if no cache file exists or the
            file is malformed.
        """
        if not self._path.exists():
            logger.debug("No token cache found at %s", self._path)
            return None

        try:
            with self._path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            bundle = TokenBundle(**data)
            logger.debug(
                "Token cache loaded; expires in %.0f s",
                bundle.expires_in_seconds(),
            )
            return bundle
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Token cache is corrupt (%s); ignoring.", exc)
            return None

    def clear(self) -> None:
        """Delete the token cache file if it exists."""
        if self._path.exists():
            self._path.unlink()
            logger.info("Token cache cleared.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _atomic_write(self, data: dict) -> None:
        """Write *data* as JSON to ``self._path`` atomically."""
        parent = self._path.parent
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.chmod(tmp_path, 0o600)
            Path(tmp_path).replace(self._path)  # cross-platform atomic rename
        except Exception:
            # Clean up the temp file on failure.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
