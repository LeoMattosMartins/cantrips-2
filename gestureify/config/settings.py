"""
gestureify.config.settings
==========================
Centralised, read-only application configuration.

All tuneable parameters live here.  Every constant is typed with
``typing.Final`` so that static analysers (mypy, Pyright) flag any
accidental mutation elsewhere in the codebase.

Constants are grouped by subsystem and are intentionally ``UPPER_CASE`` to
signal that they must not be mutated at runtime.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Spotify / OAuth
# ---------------------------------------------------------------------------

#: Scopes required for full playback control.
SPOTIFY_SCOPES: Final[str] = (
    "user-modify-playback-state "
    "user-read-playback-state"
)

#: Local redirect URI for the PKCE callback server.
SPOTIFY_REDIRECT_URI: Final[str] = "http://localhost:8080/callback"

#: Port the temporary callback HTTP server listens on.
SPOTIFY_CALLBACK_PORT: Final[int] = 8080

#: Path (relative to project root) where tokens are cached.
TOKEN_CACHE_PATH: Final[str] = ".spotify_token_cache.json"

#: Seconds before expiry at which a proactive token refresh is triggered.
TOKEN_REFRESH_BUFFER_SECONDS: Final[int] = 60

# ---------------------------------------------------------------------------
# Camera / OpenCV
# ---------------------------------------------------------------------------

#: Webcam device index (0 = default built-in camera).
CAMERA_INDEX: Final[int] = 0

#: Desired capture width in pixels.
CAMERA_WIDTH: Final[int] = 640

#: Desired capture height in pixels.
CAMERA_HEIGHT: Final[int] = 480

#: Target frames per second for the capture loop.
CAMERA_FPS: Final[int] = 30

# ---------------------------------------------------------------------------
# MediaPipe Hands
# ---------------------------------------------------------------------------

#: Maximum number of hands to track simultaneously.
MP_MAX_HANDS: Final[int] = 1

#: Minimum detection confidence to accept a new hand detection.
MP_DETECTION_CONFIDENCE: Final[float] = 0.7

#: Minimum tracking confidence before re-running full detection.
MP_TRACKING_CONFIDENCE: Final[float] = 0.6

# ---------------------------------------------------------------------------
# Session Wake-Up Gesture
# ---------------------------------------------------------------------------

#: Landmark distance ratio threshold above which the hand is "open palm".
#: Computed as mean four-fingertip-to-wrist distance / palm width (thumb excluded).
WAKE_PALM_RATIO_THRESHOLD: Final[float] = 1.8

#: Consecutive seconds an open palm must be held to toggle the session.
WAKE_HOLD_SECONDS: Final[float] = 1.5

# ---------------------------------------------------------------------------
# Gesture Thresholds
# ---------------------------------------------------------------------------

#: Landmark distance ratio below which the hand is considered a "closed fist".
FIST_RATIO_THRESHOLD: Final[float] = 0.6

#: Normalised thumb-index distance below which a pinch is active.
PINCH_DISTANCE_THRESHOLD: Final[float] = 0.12

#: Number of past frames used to compute swipe velocity.
SWIPE_VELOCITY_WINDOW: Final[int] = 5

#: Minimum normalised horizontal velocity (units/frame) to register a swipe.
SWIPE_VELOCITY_THRESHOLD: Final[float] = 0.04

# ---------------------------------------------------------------------------
# Cooldown / Debounce Timings  (seconds)
# ---------------------------------------------------------------------------

#: How long to ignore all gestures after a swipe fires (rearm window).
SWIPE_COOLDOWN_SECONDS: Final[float] = 0.8

#: Minimum time between consecutive play/pause toggles.
PLAYPAUSE_COOLDOWN_SECONDS: Final[float] = 0.5

#: Minimum interval between Spotify volume API calls (throttle).
VOLUME_API_THROTTLE_SECONDS: Final[float] = 0.3

# ---------------------------------------------------------------------------
# HUD / Overlay
# ---------------------------------------------------------------------------

#: Width of the floating HUD window in pixels.
HUD_WIDTH: Final[int] = 260

#: Height of the floating HUD window in pixels.
HUD_HEIGHT: Final[int] = 210

#: Opacity of the HUD window (0.0 transparent – 1.0 opaque).
HUD_ALPHA: Final[float] = 0.92

#: How many milliseconds a transient action label stays visible.
HUD_ACTION_LABEL_MS: Final[int] = 1000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

#: Default log level name.  Override via LOG_LEVEL env var.
LOG_LEVEL: Final[str] = "INFO"

#: Log format string.
LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"

#: Date format used in log output.
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
