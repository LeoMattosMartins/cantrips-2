"""
gestureify.hud.overlay
=======================
Minimal monochrome floating overlay — terminal / hacker aesthetic.

Design language
---------------
* Background: pure black (#000000).
* All text: monospace (Courier / Courier New), dim green (#3a3a3a) for
  inactive elements, bright green (#39ff14) for ACTIVE state, grey (#888888)
  for labels.
* No rounded corners, no colour accents beyond the single green status dot.
* Skeleton lines: dim grey (#2a2a2a) connections, slightly brighter (#555555)
  joint dots.
* Volume bar: single-pixel height, grey fill, no border.
* Action flash: uppercase, monospace, white (#e0e0e0), no background.

Architecture
------------
The overlay runs in the **main thread** (Tkinter is not thread-safe).  The
CV pipeline runs in a background thread and communicates with the HUD via a
thread-safe ``queue.Queue``.  The HUD polls the queue on every Tkinter
``after()`` tick (16 ms ≈ 60 Hz) and applies updates atomically.

``HUDMessage`` uses ``slots=True`` (Python 3.10+) to eliminate per-instance
``__dict__`` allocation at 30 FPS.
"""

from __future__ import annotations

import queue
import tkinter as tk
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from gestureify.assets.font_loader import load as _load_font
from gestureify.config import settings
from gestureify.cv_engine.session_gate import SessionState
from gestureify.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_LandmarkList = List[Tuple[float, float]]

# MediaPipe HAND_CONNECTIONS as plain index pairs (no mediapipe import here).
_HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# ---------------------------------------------------------------------------
# Colour palette — monochrome terminal
# ---------------------------------------------------------------------------

_BG          = "#000000"   # window background
_PANEL_BG    = "#0a0a0a"   # canvas background
_BORDER      = "#1c1c1c"   # subtle border
_TEXT_DIM    = "#444444"   # inactive labels
_TEXT_BRIGHT = "#c8c8c8"   # active / flash labels
_GREEN_IDLE  = "#2a4a2a"   # status dot — IDLE
_GREEN_ON    = "#39ff14"   # status dot — ACTIVE (neon green)
_SKEL_LINE   = "#222222"   # skeleton connections
_SKEL_DOT    = "#484848"   # skeleton joints
_SKEL_TIP    = "#787878"   # fingertip joints (slightly brighter)
_VOL_BG      = "#1a1a1a"   # volume bar track
_VOL_FG      = "#3d3d3d"   # volume bar fill
_ARC_IDLE    = "#1e3a1e"   # hold-progress arc — IDLE
_ARC_ACTIVE  = "#39ff14"   # hold-progress arc — ACTIVE

# Fingertip landmark indices (brighter dot colour).
_FINGERTIP_IDX = {4, 8, 12, 16, 20}

# Monocraft font family — resolved at import time (falls back to Courier New).
_MONO_FAMILY: str = _load_font()

_FONT_MONO_SM  = (_MONO_FAMILY, 8)
_FONT_MONO_MED = (_MONO_FAMILY, 10, "bold")
_FONT_MONO_LG  = (_MONO_FAMILY, 11, "bold")


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HUDMessage:
    """Data packet sent from the CV thread to the HUD each frame.

    Attributes
    ----------
    session_state:
        Current ``SessionState``.
    action_label:
        Short uppercase string to flash (e.g. ``"PAUSE"``).  Empty = no flash.
    landmarks:
        21-point normalised landmark list for skeleton drawing, or ``None``.
    hold_progress:
        Wake-gesture hold progress in [0.0, 1.0].
    volume_percent:
        Current volume level in [0, 100], or ``None`` if unknown.
    """

    session_state: SessionState = field(default=SessionState.IDLE)
    action_label: str = ""
    landmarks: Optional[_LandmarkList] = None
    hold_progress: float = 0.0
    volume_percent: Optional[int] = None


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


class HUDOverlay:
    """Minimal Tkinter floating overlay.

    Parameters
    ----------
    message_queue:
        Thread-safe queue through which the CV thread sends ``HUDMessage``
        objects.
    """

    _POLL_MS: int = 16          # ~60 Hz UI refresh.
    _CANVAS_SZ: int = 120       # Square skeleton canvas.
    _MARGIN: int = 12           # Padding inside canvas.

    def __init__(self, message_queue: "queue.Queue[HUDMessage]") -> None:
        self._q = message_queue
        self._root: Optional[tk.Tk] = None
        self._flash_id: Optional[str] = None
        self._cur_state: SessionState = SessionState.IDLE

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Build the window and enter the Tkinter event loop (blocking)."""
        self._root = tk.Tk()
        self._build()
        self._poll()
        self._root.mainloop()

    def stop(self) -> None:
        """Destroy the window and exit the event loop."""
        if self._root is not None:
            self._root.quit()
            self._root.destroy()
            self._root = None
            logger.info("HUD overlay closed.")

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Construct all Tkinter widgets."""
        r = self._root
        r.title("")
        r.geometry(f"{settings.HUD_WIDTH}x{settings.HUD_HEIGHT}+40+40")
        r.attributes("-topmost", True)
        r.attributes("-alpha", settings.HUD_ALPHA)
        r.configure(bg=_BG)
        r.resizable(False, False)
        r.protocol("WM_DELETE_WINDOW", self.stop)

        # ── Top bar: status dot + state text ──────────────────────────
        top = tk.Frame(r, bg=_BG)
        top.pack(fill=tk.X, padx=10, pady=(8, 2))

        self._dot = tk.Label(top, text="●", font=_FONT_MONO_LG,
                             fg=_GREEN_IDLE, bg=_BG)
        self._dot.pack(side=tk.LEFT)

        self._state_lbl = tk.Label(top, text=" IDLE", font=_FONT_MONO_MED,
                                   fg=_TEXT_DIM, bg=_BG)
        self._state_lbl.pack(side=tk.LEFT)

        # ── Skeleton canvas ───────────────────────────────────────────
        self._canvas = tk.Canvas(
            r,
            width=self._CANVAS_SZ,
            height=self._CANVAS_SZ,
            bg=_PANEL_BG,
            highlightthickness=1,
            highlightbackground=_BORDER,
        )
        self._canvas.pack(pady=(2, 2))

        # Hold-progress arc (drawn on top of canvas background).
        sz = self._CANVAS_SZ
        self._arc = self._canvas.create_arc(
            3, 3, sz - 3, sz - 3,
            start=90, extent=0,
            outline=_ARC_IDLE, width=2, style=tk.ARC,
        )

        # ── Action flash label ────────────────────────────────────────
        self._action_lbl = tk.Label(
            r, text="", font=_FONT_MONO_SM,
            fg=_TEXT_BRIGHT, bg=_BG,
        )
        self._action_lbl.pack(pady=(1, 2))

        # ── Volume bar ────────────────────────────────────────────────
        vol_outer = tk.Frame(r, bg=_BG)
        vol_outer.pack(fill=tk.X, padx=10, pady=(0, 6))

        self._vol_track = tk.Frame(vol_outer, bg=_VOL_BG, height=2)
        self._vol_track.pack(fill=tk.X)

        self._vol_fill = tk.Frame(self._vol_track, bg=_VOL_FG, height=2)
        self._vol_fill.place(x=0, y=0, relwidth=0.5, height=2)

    # ------------------------------------------------------------------
    # Polling and rendering
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        """Drain the message queue and schedule the next poll."""
        try:
            while True:
                msg: HUDMessage = self._q.get_nowait()
                self._render(msg)
        except queue.Empty:
            pass
        if self._root is not None:
            self._root.after(self._POLL_MS, self._poll)

    def _render(self, msg: HUDMessage) -> None:
        """Apply a ``HUDMessage`` to all widgets."""
        self._update_status(msg.session_state)
        self._update_skeleton(msg.landmarks)
        self._update_arc(msg.hold_progress, msg.session_state)
        if msg.action_label:
            self._flash(msg.action_label)
        if msg.volume_percent is not None:
            self._update_vol(msg.volume_percent)

    def _update_status(self, state: SessionState) -> None:
        """Update the status dot and label."""
        if state is self._cur_state:
            return
        self._cur_state = state
        if state is SessionState.ACTIVE:
            self._dot.configure(fg=_GREEN_ON)
            self._state_lbl.configure(text=" ACTIVE", fg=_GREEN_ON)
        else:
            self._dot.configure(fg=_GREEN_IDLE)
            self._state_lbl.configure(text=" IDLE", fg=_TEXT_DIM)

    def _update_skeleton(self, landmarks: Optional[_LandmarkList]) -> None:
        """Redraw the hand skeleton on the canvas."""
        self._canvas.delete("sk")
        if not landmarks:
            return

        sz = self._CANVAS_SZ
        m = self._MARGIN
        usable = sz - 2 * m

        def px(lm: Tuple[float, float]) -> Tuple[int, int]:
            return int(lm[0] * usable) + m, int(lm[1] * usable) + m

        pts = [px(lm) for lm in landmarks]

        for a, b in _HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                self._canvas.create_line(
                    pts[a][0], pts[a][1], pts[b][0], pts[b][1],
                    fill=_SKEL_LINE, width=1, tags="sk",
                )

        for i, (x, y) in enumerate(pts):
            r = 3 if i in _FINGERTIP_IDX else 2
            colour = _SKEL_TIP if i in _FINGERTIP_IDX else _SKEL_DOT
            self._canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=colour, outline="", tags="sk",
            )

    def _update_arc(self, progress: float, state: SessionState) -> None:
        """Update the hold-progress arc extent and colour."""
        extent = -int(progress * 360)
        colour = _ARC_ACTIVE if state is SessionState.ACTIVE else _ARC_IDLE
        self._canvas.itemconfigure(self._arc, extent=extent, outline=colour)

    def _flash(self, label: str) -> None:
        """Show *label* briefly then clear it."""
        self._action_lbl.configure(text=label.upper())
        if self._flash_id is not None:
            self._root.after_cancel(self._flash_id)  # type: ignore[union-attr]
        self._flash_id = self._root.after(  # type: ignore[union-attr]
            settings.HUD_ACTION_LABEL_MS,
            lambda: self._action_lbl.configure(text=""),
        )

    def _update_vol(self, pct: int) -> None:
        """Update the volume fill bar."""
        rel = max(0.0, min(1.0, pct / 100))
        self._vol_fill.place(relwidth=rel)
