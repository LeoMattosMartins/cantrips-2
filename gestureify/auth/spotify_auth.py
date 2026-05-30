"""
gestureify.auth.spotify_auth
============================
High-level Spotify authentication orchestrator.

``SpotifyAuth`` ties together ``PKCEFlow``, ``TokenStore``, and a temporary
local HTTP callback server to execute the full PKCE authorisation flow.

Typical usage (called once from ``main.py``)::

    auth = SpotifyAuth(
        client_id=os.environ["SPOTIFY_CLIENT_ID"],
        redirect_uri=settings.SPOTIFY_REDIRECT_URI,
        scopes=settings.SPOTIFY_SCOPES,
        token_store=TokenStore(settings.TOKEN_CACHE_PATH),
    )
    token = auth.ensure_valid_token()   # opens browser on first run

Design notes
------------
* The callback server is a minimal ``http.server`` implementation that
  handles exactly one request then shuts down (bounded execution, NASA Rule 2).
* ``ensure_valid_token()`` is the single public entry-point; callers never
  need to manage the flow steps manually.
"""

from __future__ import annotations

import os
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event
from typing import Optional

from gestureify.auth.pkce import PKCEFlow, PKCEPair
from gestureify.auth.token_store import TokenBundle, TokenStore
from gestureify.config import settings
from gestureify.utils.logger import get_logger

logger = get_logger(__name__)


class SpotifyAuth:
    """Manage Spotify OAuth2 tokens using the PKCE flow.

    Parameters
    ----------
    client_id:
        Spotify application client ID (from developer dashboard).
    redirect_uri:
        Local callback URI registered in the Spotify app settings.
    scopes:
        Space-separated permission scopes.
    token_store:
        Persistence backend for the token bundle.
    """

    _TOKEN_ENDPOINT: str = "https://accounts.spotify.com/api/token"

    def __init__(
        self,
        client_id: str,
        redirect_uri: str,
        scopes: str,
        token_store: TokenStore,
    ) -> None:
        self._client_id = client_id
        self._redirect_uri = redirect_uri
        self._scopes = scopes
        self._store = token_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_valid_token(self) -> str:
        """Return a valid access token, refreshing or re-authorising as needed.

        Returns
        -------
        str
            A non-expired Spotify access token.

        Raises
        ------
        RuntimeError
            If the authorisation flow fails or times out.
        """
        bundle = self._store.load()

        if bundle is not None and not bundle.is_expired:
            logger.debug("Using cached access token.")
            return bundle.access_token

        if bundle is not None and bundle.refresh_token:
            logger.info("Access token expired; refreshing.")
            refreshed = self._refresh(bundle.refresh_token)
            if refreshed is not None:
                self._store.save(refreshed)
                return refreshed.access_token

        logger.info("No valid token found; starting PKCE authorisation flow.")
        new_bundle = self._run_pkce_flow()
        self._store.save(new_bundle)
        return new_bundle.access_token

    def revoke(self) -> None:
        """Clear the local token cache (does not call Spotify revoke endpoint)."""
        self._store.clear()
        logger.info("Local token cache cleared.")

    # ------------------------------------------------------------------
    # Private: PKCE flow
    # ------------------------------------------------------------------

    def _run_pkce_flow(self) -> TokenBundle:
        """Execute the full PKCE browser-redirect flow.

        Opens the user's default browser, waits for the callback, then
        exchanges the authorisation code for tokens.

        Returns
        -------
        TokenBundle
            Fresh token bundle.

        Raises
        ------
        RuntimeError
            If the callback is not received within 120 seconds.
        """
        pair: PKCEPair = PKCEFlow.generate_pair()
        auth_url = PKCEFlow.build_auth_url(
            client_id=self._client_id,
            redirect_uri=self._redirect_uri,
            scopes=self._scopes,
            challenge=pair.challenge,
        )

        code_holder: list[str] = []
        received_event = Event()

        server = self._build_callback_server(
            port=settings.SPOTIFY_CALLBACK_PORT,
            code_holder=code_holder,
            received_event=received_event,
        )

        logger.info("Opening browser for Spotify authorisation…")
        webbrowser.open(auth_url)

        server.timeout = 120  # seconds
        while not received_event.is_set():
            server.handle_request()

        server.server_close()

        if not code_holder:
            raise RuntimeError(
                "Spotify authorisation timed out or was denied. "
                "Please re-run the application and complete the login."
            )

        return self._exchange_code(code_holder[0], pair.verifier)

    # ------------------------------------------------------------------
    # Private: token exchange and refresh
    # ------------------------------------------------------------------

    def _exchange_code(self, code: str, verifier: str) -> TokenBundle:
        """Exchange an authorisation code for a token bundle."""
        payload = urllib.parse.urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._redirect_uri,
            "client_id": self._client_id,
            "code_verifier": verifier,
        }).encode("utf-8")

        return self._post_token(payload)

    def _refresh(self, refresh_token: str) -> Optional[TokenBundle]:
        """Attempt a silent token refresh.  Returns *None* on failure."""
        payload = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._client_id,
        }).encode("utf-8")

        try:
            return self._post_token(payload, existing_refresh=refresh_token)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Token refresh failed: %s", exc)
            return None

    def _post_token(
        self,
        payload: bytes,
        existing_refresh: Optional[str] = None,
    ) -> TokenBundle:
        """POST to the Spotify token endpoint and return a ``TokenBundle``."""
        req = urllib.request.Request(
            self._TOKEN_ENDPOINT,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            import json
            data = json.loads(resp.read().decode("utf-8"))

        expires_at = time.time() + int(data.get("expires_in", 3600))
        return TokenBundle(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token") or existing_refresh or "",
            expires_at=expires_at,
            token_type=data.get("token_type", "Bearer"),
            scope=data.get("scope", ""),
        )

    # ------------------------------------------------------------------
    # Private: callback HTTP server
    # ------------------------------------------------------------------

    @staticmethod
    def _build_callback_server(
        port: int,
        code_holder: list[str],
        received_event: Event,
    ) -> HTTPServer:
        """Create a single-use HTTP server that captures the OAuth callback."""

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                if "code" in params:
                    code_holder.append(params["code"][0])
                    body = (
                        b"<html><body><h2>Gestureify: Authorisation successful!"
                        b"</h2><p>You may close this tab.</p></body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(400)
                    self.end_headers()
                received_event.set()

            def log_message(self, *args) -> None:  # noqa: ANN002
                pass  # Suppress default HTTP server logging.

        class _ReuseServer(HTTPServer):
            """HTTPServer subclass with SO_REUSEADDR to survive TIME_WAIT."""

            allow_reuse_address = True

            def server_bind(self) -> None:
                import socket
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                super().server_bind()

        server = _ReuseServer(("localhost", port), _Handler)
        return server
