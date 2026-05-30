"""
gestureify.config.app_config
=============================
Pydantic v2 runtime configuration model.

``AppConfig`` is the single validated, typed source of truth for all runtime
configuration.  It is constructed once in ``main.py`` after the environment
has been loaded, and injected into every sub-system that needs it.

Why Pydantic here?
------------------
``settings.py`` holds compile-time constants (``Final[T]``).  ``AppConfig``
holds *runtime* values that may be overridden by environment variables and
must be validated before the application starts.  Pydantic v2 gives us:

* Automatic type coercion (``CAMERA_INDEX="0"`` → ``int``).
* Field-level validators with clear error messages.
* A single ``model_validate`` call that fails fast with a structured error
  rather than a cryptic ``KeyError`` or ``ValueError`` deep in the app.
* ``model_config = ConfigDict(frozen=True)`` — the config object is immutable
  after construction, enforcing the same "read-only" contract as ``Final``.

Usage::

    from gestureify.config.app_config import AppConfig
    cfg = AppConfig.from_env()   # reads os.environ, validates, raises on error
"""

from __future__ import annotations

import os
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gestureify.config import settings


class AppConfig(BaseModel):
    """Validated runtime configuration for Gestureify.

    All fields have defaults sourced from ``settings.py``.  Override any
    field by setting the corresponding environment variable before calling
    ``AppConfig.from_env()``.

    Environment variable mapping
    ----------------------------
    ``SPOTIFY_CLIENT_ID``     → ``spotify_client_id``  (required, no default)
    ``SPOTIFY_REDIRECT_URI``  → ``spotify_redirect_uri``
    ``CAMERA_INDEX``          → ``camera_index``
    ``LOG_LEVEL``             → ``log_level``
    """

    model_config = ConfigDict(frozen=True)

    # ------------------------------------------------------------------ Spotify
    spotify_client_id: Annotated[str, Field(min_length=1)]
    spotify_redirect_uri: Annotated[str, Field(min_length=10)] = (
        settings.SPOTIFY_REDIRECT_URI
    )
    spotify_scopes: str = settings.SPOTIFY_SCOPES

    # ------------------------------------------------------------------ Camera
    camera_index: Annotated[int, Field(ge=0, le=63)] = settings.CAMERA_INDEX
    camera_width: Annotated[int, Field(ge=160, le=3840)] = settings.CAMERA_WIDTH
    camera_height: Annotated[int, Field(ge=120, le=2160)] = settings.CAMERA_HEIGHT
    camera_fps: Annotated[int, Field(ge=1, le=120)] = settings.CAMERA_FPS

    # ------------------------------------------------------------------ MediaPipe
    mp_max_hands: Annotated[int, Field(ge=1, le=4)] = settings.MP_MAX_HANDS
    mp_detection_confidence: Annotated[float, Field(ge=0.1, le=1.0)] = (
        settings.MP_DETECTION_CONFIDENCE
    )
    mp_tracking_confidence: Annotated[float, Field(ge=0.1, le=1.0)] = (
        settings.MP_TRACKING_CONFIDENCE
    )

    # ------------------------------------------------------------------ Gestures
    wake_palm_ratio_threshold: Annotated[float, Field(gt=0.0)] = (
        settings.WAKE_PALM_RATIO_THRESHOLD
    )
    wake_hold_seconds: Annotated[float, Field(gt=0.0, le=10.0)] = (
        settings.WAKE_HOLD_SECONDS
    )
    fist_ratio_threshold: Annotated[float, Field(gt=0.0)] = (
        settings.FIST_RATIO_THRESHOLD
    )
    pinch_distance_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = (
        settings.PINCH_DISTANCE_THRESHOLD
    )
    swipe_velocity_window: Annotated[int, Field(ge=2, le=30)] = (
        settings.SWIPE_VELOCITY_WINDOW
    )
    swipe_velocity_threshold: Annotated[float, Field(gt=0.0)] = (
        settings.SWIPE_VELOCITY_THRESHOLD
    )

    # ------------------------------------------------------------------ Timings
    swipe_cooldown_seconds: Annotated[float, Field(ge=0.0)] = (
        settings.SWIPE_COOLDOWN_SECONDS
    )
    playpause_cooldown_seconds: Annotated[float, Field(ge=0.0)] = (
        settings.PLAYPAUSE_COOLDOWN_SECONDS
    )
    volume_api_throttle_seconds: Annotated[float, Field(ge=0.0)] = (
        settings.VOLUME_API_THROTTLE_SECONDS
    )

    # ------------------------------------------------------------------ Logging
    log_level: str = settings.LOG_LEVEL

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("spotify_redirect_uri")
    @classmethod
    def _redirect_uri_must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"spotify_redirect_uri must start with http:// or https://, got: {v!r}"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _log_level_must_be_valid(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(
                f"log_level must be one of {sorted(valid)}, got: {v!r}"
            )
        return upper

    @model_validator(mode="after")
    def _fist_below_palm(self) -> "AppConfig":
        """Ensure fist threshold is strictly below palm threshold."""
        if self.fist_ratio_threshold >= self.wake_palm_ratio_threshold:
            raise ValueError(
                f"fist_ratio_threshold ({self.fist_ratio_threshold}) must be "
                f"less than wake_palm_ratio_threshold ({self.wake_palm_ratio_threshold})."
            )
        return self

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Construct and validate ``AppConfig`` from ``os.environ``.

        Reads the following environment variables (all optional except
        ``SPOTIFY_CLIENT_ID``):

        * ``SPOTIFY_CLIENT_ID`` — required.
        * ``SPOTIFY_REDIRECT_URI``
        * ``CAMERA_INDEX``
        * ``LOG_LEVEL``

        Returns
        -------
        AppConfig
            Validated, frozen configuration instance.

        Raises
        ------
        pydantic.ValidationError
            If any field fails validation (missing required key, out-of-range
            value, etc.).  The error message lists all failures at once.
        """
        raw: dict = {
            "spotify_client_id": os.environ.get("SPOTIFY_CLIENT_ID", ""),
            "spotify_redirect_uri": os.environ.get(
                "SPOTIFY_REDIRECT_URI", settings.SPOTIFY_REDIRECT_URI
            ),
            "camera_index": int(
                os.environ.get("CAMERA_INDEX", str(settings.CAMERA_INDEX))
            ),
            "log_level": os.environ.get("LOG_LEVEL", settings.LOG_LEVEL),
        }
        return cls.model_validate(raw)
