"""
gestureify.hud.overlay
=======================
Lightweight Tkinter floating overlay that shows the current session state,
a live hand-skeleton preview, and transient action labels.

Architecture
------------
The overlay runs in the **main thread** (Tkinter is not thread-safe).  The
CV pipeline runs in a background thread and communicates with the HUD via a
thread-safe ``queue.Queue``.  The HUD polls the queue on every Tkinter
``after()`` tick (16 ms ≈ 60 Hz) and applies updates atomically.

``HUDMessage`` is a simple dataclass that carries all the information the
HUD needs for one render cycle.  This decouples the CV thread from the UI
thread cleanly.

Design notes
------------
* No shared mutable state between threads other than the queue (NASA Rule 5).
* The overlay is destroyed gracefully when ``stop()`` is called.
* Canvas drawing is bounded: at most 21 landmark circles + 20 connection
  lines per frame (NASA Rule 2).
"""

from __future__ import annotations

import queue
import tkinter as tk
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from gestureify.config import settings
from gestureify.cv_engine.session_gate import SessionState
from gestureify.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for a landmark list as used in the HUD (no mediapipe dependency).
_LandmarkList = List[Tuple[float, float]]

# MediaPipe HAND_CONNECTIONS as plain index pairs (avoids importing mediapipe here).
_HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# State → colour mapping for the border ring.
_STATE_COLOURS = {
    SessionState.IDLE: "#888888",
    SessionState.ACTIVE: "#00cc44",
}


@dataclass
class HUDMessage:
    """Data packet sent from the CV thread to the HUD each frame.

    Attributes
    ----------
    session_state:
        Current ``SessionState``.
    action_label:
        Short string to flash (e.g. ``"⏸ Pause"``).  Empty string = no flash.
    landmarks:
        21-point normalised landmark list for skeleton drawing, or ``None``.
    hold_progress:
        Wake-gesture hold progress in [0.0, 1.0].
    volume_percent:
        Current volume level in [0, 100], or ``None`` if unknown.
    """

    session_state: SessionState = SessionState.IDLE
    action_label: str = ""
    landmarks: Optional[_LandmarkList] = None
    hold_progress: float = 0.0
    volume_percent: Optional[int] = None


class HUDOverlay:
    """Tkinter floating overlay window.

    Parameters
    ----------
    message_queue:
        Thread-safe queue through which the CV thread sends ``HUDMessage``
        objects.
    """

    _POLL_INTERVAL_MS: int = 16   # ~60 Hz UI refresh.
    _CANVAS_SIZE: int = 140       # Square canvas for the skeleton preview.
    _SKELETON_MARGIN: int = 15    # Padding inside the canvas.

    def __init__(self, message_queue: "queue.Queue[HUDMessage]") -> None:
        self._q = message_queue
        self._root: Optional[tk.Tk] = None
        self._action_after_id: Optional[str] = None
        self._current_state: SessionState = SessionState.IDLE

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Build the window and enter the Tkinter event loop (blocking)."""
        self._root = tk.Tk()
        self._build_window()
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

    def _build_window(self) -> None:
        """Create all Tkinter widgets."""
        root = self._root
        root.title("Gestureify")
        root.geometry(f"{settings.HUD_WIDTH}x{settings.HUD_HEIGHT}+50+50")
        root.attributes("-topmost", True)
        root.attributes("-alpha", settings.HUD_ALPHA)
        root.configure(bg="#1a1a2e")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.stop)

        # Status ring label (top).
        self._status_label = tk.Label(
            root,
            text="● IDLE",
            font=("Helvetica", 11, "bold"),
            fg=_STATE_COLOURS[SessionState.IDLE],
            bg="#1a1a2e",
        )
        self._status_label.pack(pady=(8, 2))

        # Skeleton canvas.
        self._canvas = tk.Canvas(
            root,
            width=self._CANVAS_SIZE,
            height=self._CANVAS_SIZE,
            bg="#0f0f1a",
            highlightthickness=1,
            highlightbackground="#333355",
        )
        self._canvas.pack(pady=2)

        # Action flash label (bottom).
        self._action_label = tk.Label(
            root,
            text="",
            font=("Helvetica", 10),
            fg="#ffffff",
            bg="#1a1a2e",
        )
        self._action_label.pack(pady=(2, 4))

        # Volume bar frame.
        self._vol_frame = tk.Frame(root, bg="#1a1a2e")
        self._vol_frame.pack(fill=tk.X, padx=12, pady=(0, 6))
        self._vol_bar_bg = tk.Frame(self._vol_frame, bg="#333355", height=6)
        self._vol_bar_bg.pack(fill=tk.X)
        self._vol_bar_fg = tk.Frame(self._vol_bar_bg, bg="#00cc44", height=6)
        self._vol_bar_fg.place(x=0, y=0, relwidth=0.5, height=6)

        # Hold-progress arc (drawn on canvas).
        self._hold_arc = self._canvas.create_arc(
            4, 4, self._CANVAS_SIZE - 4, self._CANVAS_SIZE - 4,
            start=90, extent=0,
            outline="#ffaa00", width=3, style=tk.ARC,
        )

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
            self._root.after(self._POLL_INTERVAL_MS, self._poll)

    def _render(self, msg: HUDMessage) -> None:
        """Apply a ``HUDMessage`` to the UI widgets."""
        self._update_status(msg.session_state)
        self._update_skeleton(msg.landmarks)
        self._update_hold_arc(msg.hold_progress)
        if msg.action_label:
            self._flash_action(msg.action_label)
        if msg.volume_percent is not None:
            self._update_volume_bar(msg.volume_percent)

    def _update_status(self, state: SessionState) -> None:
        """Update the status label colour and text."""
        if state == self._current_state:
            return
        self._current_state = state
        colour = _STATE_COLOURS[state]
        text = f"● {state.name}"
        self._status_label.configure(text=text, fg=colour)

    def _update_skeleton(self, landmarks: Optional[_LandmarkList]) -> None:
        """Redraw the hand skeleton on the canvas."""
        self._canvas.delete("skeleton")
        if landmarks is None:
            return

        size = self._CANVAS_SIZE
        margin = self._SKELETON_MARGIN
        usable = size - 2 * margin

        def to_px(lm: Tuple[float, float]) -> Tuple[int, int]:
            return (
                int(lm[0] * usable) + margin,
                int(lm[1] * usable) + margin,
            )

        pts = [to_px(lm) for lm in landmarks]

        # Draw connections.
        for a, b in _HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                self._canvas.create_line(
                    pts[a][0], pts[a][1], pts[b][0], pts[b][1],
                    fill="#4466aa", width=1, tags="skeleton",
                )

        # Draw landmark dots.
        r = 3
        for px, py in pts:
            self._canvas.create_oval(
                px - r, py - r, px + r, py + r,
                fill="#88aaff", outline="", tags="skeleton",
            )

    def _update_hold_arc(self, progress: float) -> None:
        """Update the wake-gesture hold progress arc."""
        extent = -int(progress * 360)
        self._canvas.itemconfigure(self._hold_arc, extent=extent)

    def _flash_action(self, label: str) -> None:
        """Show *label* briefly then clear it."""
        self._action_label.configure(text=label)
        if self._action_after_id is not None:
            self._root.after_cancel(self._action_after_id)  # type: ignore[union-attr]
        self._action_after_id = self._root.after(  # type: ignore[union-attr]
            settings.HUD_ACTION_LABEL_MS,
            lambda: self._action_label.configure(text=""),
        )

    def _update_volume_bar(self, volume_percent: int) -> None:
        """Update the volume progress bar."""
        rel = max(0.0, min(1.0, volume_percent / 100))
        self._vol_bar_fg.place(relwidth=rel)
