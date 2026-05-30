"""
gestureify.hud.overlay
=======================
Minimal monochrome floating overlay — terminal / hacker aesthetic.

Design language
---------------
* Background: pure black (#000000).
* All text: monospace (Monocraft / Courier New), dim green (#3a3a3a) for
  inactive elements, bright green (#39ff14) for ACTIVE state, grey (#888888)
  for labels.
* No rounded corners, no colour accents beyond the single green status dot.
* Skeleton lines: dim grey (#2a2a2a) connections, slightly brighter (#555555)
  joint dots.
* Volume bar: single-pixel height, grey fill, no border.
* Action flash: uppercase, monospace, white (#e0e0e0), no background.

Architecture
------------
The overlay runs in the **main thread** (Qt is not thread-safe).  The CV
pipeline runs in a background thread and communicates with the HUD via a
thread-safe ``queue.Queue``.  The HUD polls the queue on every Qt timer tick
(16 ms ≈ 60 Hz) and applies updates atomically.

``HUDMessage`` uses ``slots=True`` (Python 3.10+) to eliminate per-instance
``__dict__`` allocation at 30 FPS.

Migration note
--------------
The original implementation used Tkinter, which crashes on macOS when Python
is built against Tcl/Tk 9.0 (bundled with Python 3.13 via python-build-
standalone / uv).  This version uses PyQt6, which has no such dependency and
works correctly on all supported platforms.
"""

from __future__ import annotations

import queue
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QWidget

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

_BG          = QColor("#000000")
_PANEL_BG    = QColor("#0a0a0a")
_BORDER      = QColor("#1c1c1c")
_TEXT_DIM    = QColor("#444444")
_TEXT_BRIGHT = QColor("#c8c8c8")
_GREEN_IDLE  = QColor("#2a4a2a")
_GREEN_ON    = QColor("#39ff14")
_SKEL_LINE   = QColor("#222222")
_SKEL_DOT    = QColor("#484848")
_SKEL_TIP    = QColor("#787878")
_VOL_BG      = QColor("#1a1a1a")
_VOL_FG      = QColor("#3d3d3d")
_ARC_IDLE    = QColor("#1e3a1e")
_ARC_ACTIVE  = QColor("#39ff14")

# Fingertip landmark indices (brighter dot colour).
_FINGERTIP_IDX = {4, 8, 12, 16, 20}

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
# HUD widget
# ---------------------------------------------------------------------------

_HUD_W = 260
_HUD_H = 210
_CANVAS_SZ = 120
_CANVAS_X = (_HUD_W - _CANVAS_SZ) // 2   # centred horizontally
_CANVAS_Y = 30                             # below the status bar
_MARGIN = 12
_ACTION_Y = _CANVAS_Y + _CANVAS_SZ + 6
_VOL_Y = _ACTION_Y + 18
_VOL_H = 2


class _HUDWidget(QWidget):
    """Internal Qt widget that owns all drawing logic."""

    def __init__(self) -> None:
        super().__init__()
        self._state: SessionState = SessionState.IDLE
        self._landmarks: Optional[_LandmarkList] = None
        self._hold_progress: float = 0.0
        self._action_label: str = ""
        self._volume: float = 0.5
        self._action_timer = QTimer(self)
        self._action_timer.setSingleShot(True)
        self._action_timer.timeout.connect(self._clear_action)

        # Window flags: frameless, always-on-top, tool window (no taskbar entry)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(_HUD_W, _HUD_H)
        self.move(40, 40)

    # ------------------------------------------------------------------
    # State updates (called from the main thread via the poll timer)
    # ------------------------------------------------------------------

    def apply(self, msg: HUDMessage) -> None:
        self._state = msg.session_state
        self._landmarks = msg.landmarks
        self._hold_progress = msg.hold_progress
        if msg.action_label:
            self._action_label = msg.action_label.upper()
            self._action_timer.start(1000)
        if msg.volume_percent is not None:
            self._volume = max(0.0, min(1.0, msg.volume_percent / 100))
        self.update()  # schedule a repaint

    def _clear_action(self) -> None:
        self._action_label = ""
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event: object) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        p.fillRect(self.rect(), _BG)

        self._draw_status(p)
        self._draw_canvas(p)
        self._draw_skeleton(p)
        self._draw_arc(p)
        self._draw_action(p)
        self._draw_volume(p)

        p.end()

    def _draw_status(self, p: QPainter) -> None:
        """Draw the status dot and IDLE/ACTIVE label."""
        active = self._state is SessionState.ACTIVE
        dot_colour = _GREEN_ON if active else _GREEN_IDLE
        text_colour = _GREEN_ON if active else _TEXT_DIM
        label = " ACTIVE" if active else " IDLE"

        p.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        p.setPen(QPen(dot_colour))
        p.drawText(10, 20, "●")

        p.setPen(QPen(text_colour))
        p.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
        p.drawText(24, 20, label)

    def _draw_canvas(self, p: QPainter) -> None:
        """Draw the skeleton canvas background and border."""
        p.fillRect(_CANVAS_X, _CANVAS_Y, _CANVAS_SZ, _CANVAS_SZ, _PANEL_BG)
        p.setPen(QPen(_BORDER, 1))
        p.drawRect(_CANVAS_X, _CANVAS_Y, _CANVAS_SZ - 1, _CANVAS_SZ - 1)

    def _draw_skeleton(self, p: QPainter) -> None:
        """Draw the hand skeleton onto the canvas."""
        if not self._landmarks:
            return

        usable = _CANVAS_SZ - 2 * _MARGIN

        def px(lm: Tuple[float, float]) -> Tuple[int, int]:
            return (
                _CANVAS_X + _MARGIN + int(lm[0] * usable),
                _CANVAS_Y + _MARGIN + int(lm[1] * usable),
            )

        pts = [px(lm) for lm in self._landmarks]

        # Connections
        p.setPen(QPen(_SKEL_LINE, 1))
        for a, b in _HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                p.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        # Joints
        p.setPen(Qt.PenStyle.NoPen)
        for i, (x, y) in enumerate(pts):
            colour = _SKEL_TIP if i in _FINGERTIP_IDX else _SKEL_DOT
            r = 3 if i in _FINGERTIP_IDX else 2
            p.setBrush(colour)
            p.drawEllipse(x - r, y - r, r * 2, r * 2)

    def _draw_arc(self, p: QPainter) -> None:
        """Draw the hold-progress arc around the canvas."""
        colour = _ARC_ACTIVE if self._state is SessionState.ACTIVE else _ARC_IDLE
        pen = QPen(colour, 2)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        extent = int(self._hold_progress * 360 * 16)  # Qt uses 1/16th degrees
        if extent > 0:
            p.drawArc(
                _CANVAS_X + 3,
                _CANVAS_Y + 3,
                _CANVAS_SZ - 6,
                _CANVAS_SZ - 6,
                90 * 16,       # start at 12 o'clock
                -extent,       # counter-clockwise
            )

    def _draw_action(self, p: QPainter) -> None:
        """Draw the action flash label."""
        if not self._action_label:
            return
        p.setPen(QPen(_TEXT_BRIGHT))
        p.setFont(QFont("Courier New", 8))
        p.drawText(0, _ACTION_Y + 12, _HUD_W, 14, Qt.AlignmentFlag.AlignHCenter, self._action_label)

    def _draw_volume(self, p: QPainter) -> None:
        """Draw the volume bar."""
        p.fillRect(10, _VOL_Y, _HUD_W - 20, _VOL_H, _VOL_BG)
        fill_w = int((_HUD_W - 20) * self._volume)
        if fill_w > 0:
            p.fillRect(10, _VOL_Y, fill_w, _VOL_H, _VOL_FG)


# ---------------------------------------------------------------------------
# Public overlay class
# ---------------------------------------------------------------------------


class HUDOverlay:
    """Minimal Qt floating overlay.

    Parameters
    ----------
    message_queue:
        Thread-safe queue through which the CV thread sends ``HUDMessage``
        objects.
    """

    _POLL_MS: int = 16  # ~60 Hz UI refresh

    def __init__(self, message_queue: "queue.Queue[HUDMessage]") -> None:
        self._q = message_queue
        self._app: Optional[QApplication] = None
        self._widget: Optional[_HUDWidget] = None
        self._timer: Optional[QTimer] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Build the window and enter the Qt event loop (blocking)."""
        # QApplication must be created before any QWidget.
        # If one already exists (e.g. in tests) reuse it.
        self._app = QApplication.instance() or QApplication(sys.argv)

        self._widget = _HUDWidget()
        self._widget.show()

        self._timer = QTimer()
        self._timer.setInterval(self._POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

        self._app.exec()

    def stop(self) -> None:
        """Destroy the window and exit the event loop."""
        if self._timer is not None:
            self._timer.stop()
        if self._widget is not None:
            self._widget.close()
            self._widget = None
        if self._app is not None:
            self._app.quit()
            self._app = None
        logger.info("HUD overlay closed.")

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        """Drain the message queue and apply updates to the widget."""
        try:
            while True:
                msg: HUDMessage = self._q.get_nowait()
                if self._widget is not None:
                    self._widget.apply(msg)
        except queue.Empty:
            pass
