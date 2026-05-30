"""
gestureify.auth.pkce
====================
PKCE (Proof Key for Code Exchange) helpers for Spotify OAuth2.

Implements the cryptographic primitives required by RFC 7636:
  1. Generate a high-entropy ``code_verifier``.
  2. Derive the ``code_challenge`` via SHA-256 + Base64-URL encoding.

These are pure functions with no I/O or side-effects (NASA Rule 6).

References
----------
* RFC 7636: https://datatracker.ietf.org/doc/html/rfc7636
* Spotify PKCE guide: https://developer.spotify.com/documentation/web-api/tutorials/code-pkce-flow
"""

from __future__ import annotations

import base64
import hashlib
import os
import urllib.parse
from dataclasses import dataclass


@dataclass(frozen=True)
class PKCEPair:
    """Immutable container for a PKCE verifier/challenge pair.

    Attributes
    ----------
    verifier:
        The raw random string kept secret on the client.
    challenge:
        The SHA-256 / Base64-URL encoded value sent to the authorisation
        server.
    """

    verifier: str
    challenge: str


class PKCEFlow:
    """Factory for PKCE verifier/challenge pairs and authorisation URLs.

    All methods are stateless class methods; instantiation is not required
    but is permitted for dependency-injection in tests.
    """

    #: Allowed characters for the code verifier (RFC 7636 §4.1).
    _VERIFIER_CHARS: str = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789-._~"
    )

    #: Verifier length in characters (RFC 7636 recommends 43–128).
    _VERIFIER_LENGTH: int = 96

    @classmethod
    def generate_pair(cls) -> PKCEPair:
        """Generate a fresh PKCE verifier/challenge pair.

        Returns
        -------
        PKCEPair
            Immutable pair ready to use in the authorisation request.
        """
        verifier = cls._generate_verifier()
        challenge = cls._derive_challenge(verifier)
        return PKCEPair(verifier=verifier, challenge=challenge)

    @classmethod
    def build_auth_url(
        cls,
        client_id: str,
        redirect_uri: str,
        scopes: str,
        challenge: str,
        state: str | None = None,
    ) -> str:
        """Construct the Spotify authorisation URL.

        Parameters
        ----------
        client_id:
            The Spotify application client ID.
        redirect_uri:
            The local callback URI (must match the Spotify app dashboard).
        scopes:
            Space-separated list of Spotify permission scopes.
        challenge:
            The ``code_challenge`` from a ``PKCEPair``.
        state:
            Optional opaque CSRF-protection string.

        Returns
        -------
        str
            Full URL to open in the user's browser.
        """
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": client_id,
            "scope": scopes,
            "redirect_uri": redirect_uri,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
        if state is not None:
            params["state"] = state

        base = "https://accounts.spotify.com/authorize"
        return f"{base}?{urllib.parse.urlencode(params)}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _generate_verifier(cls) -> str:
        """Return a cryptographically random code verifier string."""
        raw = os.urandom(cls._VERIFIER_LENGTH)
        # Map each byte to a character in the allowed set.
        return "".join(
            cls._VERIFIER_CHARS[b % len(cls._VERIFIER_CHARS)]
            for b in raw
        )

    @staticmethod
    def _derive_challenge(verifier: str) -> str:
        """Return the Base64-URL-encoded SHA-256 hash of *verifier*."""
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        return (
            base64.urlsafe_b64encode(digest)
            .rstrip(b"=")
            .decode("ascii")
        )
