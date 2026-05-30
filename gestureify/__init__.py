"""
gestureify
==========
Gesture-controlled Spotify player.

Modules
-------
auth        -- Spotify PKCE OAuth2 authentication and token lifecycle.
cv_engine   -- MediaPipe hand-tracking pipeline and gesture state machine.
controller  -- Spotify Web API playback commands with OS media-key fallback.
hud         -- Lightweight Tkinter floating overlay (status + skeleton).
config      -- Centralised, validated application settings.
utils       -- Shared helpers (logging, geometry, timing).
"""

__version__ = "0.1.0"
__author__ = "LeoMattosMartins"
