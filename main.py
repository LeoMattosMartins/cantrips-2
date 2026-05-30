"""
main.py
=======
Gestureify — Gesture-controlled Spotify player.

Entry point.  Responsibilities:
  1. Load environment / validate secrets.
  2. Configure the root logger.
  3. Authenticate with Spotify (PKCE flow, cached tokens).
  4. Build all sub-system objects and wire them together.
  5. Start the CV loop in a background thread.
  6. Run the HUD overlay in the main thread (Tkinter requirement).
  7. Shut everything down cleanly on exit.

Threading model
---------------
  Main thread  : Tkinter HUD event loop.
  CV thread    : Camera capture + MediaPipe + gesture dispatch.

Communication : ``queue.Queue[HUDMessage]`` (CV → HUD, one-way).

Design notes
------------
* All sub-systems are constructed here and injected into their consumers
  (dependency injection, not global singletons).
* The CV thread is a daemon thread so it is killed automatically if the
  main thread exits unexpectedly.
* Signal handling (SIGINT / KeyboardInterrupt) is caught to ensure the
  camera is always released (NASA Rule 7).
"""

from __future__ import annotations

import os
import queue
import signal
import sys
import threading
from typing import Optional

from gestureify.auth.spotify_auth import SpotifyAuth
from gestureify.auth.token_store import TokenStore
from gestureify.config import load_env, settings
from gestureify.controller.media_keys import MediaKeyFallback
from gestureify.controller.playback_controller import PlaybackController
from gestureify.controller.spotify_client import SpotifyClient
from gestureify.cv_engine.capture import CameraCapture
from gestureify.cv_engine.gesture_classifier import GestureClassifier, GestureLabel
from gestureify.cv_engine.landmark_extractor import LandmarkExtractor
from gestureify.cv_engine.pipeline import CVPipeline, FrameResult
from gestureify.cv_engine.session_gate import SessionGate, SessionState
from gestureify.cv_engine.swipe_detector import SwipeDetector, SwipeDirection
from gestureify.hud.overlay import HUDMessage, HUDOverlay
from gestureify.utils.logger import configure_root_logger, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Action label strings shown in the HUD.
# ---------------------------------------------------------------------------
_LABEL_PAUSE = "⏸  Pause"
_LABEL_RESUME = "▶  Resume"
_LABEL_NEXT = "⏭  Next"
_LABEL_PREV = "⏮  Previous"
_LABEL_ACTIVE = "✋ Session ON"
_LABEL_IDLE = "🤚 Session OFF"


def _cv_loop(
    pipeline: CVPipeline,
    capture: CameraCapture,
    controller: PlaybackController,
    hud_queue: "queue.Queue[HUDMessage]",
    stop_event: threading.Event,
) -> None:
    """Main CV processing loop — runs in the background thread.

    Parameters
    ----------
    pipeline:
        Fully constructed ``CVPipeline``.
    capture:
        Open ``CameraCapture`` instance.
    controller:
        ``PlaybackController`` for dispatching commands.
    hud_queue:
        Queue for sending ``HUDMessage`` objects to the HUD thread.
    stop_event:
        Set this event to request a graceful shutdown.
    """
    # Track playback state locally to know whether to pause or resume.
    is_playing: bool = True

    for frame in capture.frames():
        if stop_event.is_set():
            break

        result: FrameResult = pipeline.process(frame)

        # --- Dispatch commands only when session is ACTIVE ---------------
        action_label: str = ""

        if result.session_toggled:
            action_label = (
                _LABEL_ACTIVE if result.session_state is SessionState.ACTIVE else _LABEL_IDLE
            )

        if result.session_state is SessionState.ACTIVE:
            # Swipe gestures (skip / previous).
            if result.swipe is SwipeDirection.RIGHT:
                controller.next_track()
                action_label = _LABEL_NEXT
            elif result.swipe is SwipeDirection.LEFT:
                controller.previous_track()
                action_label = _LABEL_PREV

            # Static gestures (pause / resume).
            elif result.gesture is GestureLabel.CLOSED_FIST and is_playing:
                controller.pause()
                is_playing = False
                action_label = _LABEL_PAUSE
            elif result.gesture is GestureLabel.OPEN_PALM and not is_playing:
                # Note: OPEN_PALM is suppressed as a command in the pipeline
                # when the session is already ACTIVE, so this branch is only
                # reached during the brief window before the wake-toggle fires.
                pass
            elif result.gesture is GestureLabel.CLOSED_FIST and not is_playing:
                pass  # Already paused; do nothing.

            # Pinch → volume.
            if result.gesture is GestureLabel.PINCH:
                controller.set_volume_from_pinch(result.pinch_gap)

        # --- Build and enqueue HUD message --------------------------------
        msg = HUDMessage(
            session_state=result.session_state,
            action_label=action_label,
            landmarks=result.landmarks,
            hold_progress=result.hold_progress,
        )
        try:
            hud_queue.put_nowait(msg)
        except queue.Full:
            pass  # Drop the frame update rather than blocking the CV loop.

    logger.info("CV loop exited.")


def _build_pipeline() -> CVPipeline:
    """Construct and return a fully wired ``CVPipeline``."""
    extractor = LandmarkExtractor()
    extractor.open()
    classifier = GestureClassifier()
    gate = SessionGate()
    swipe = SwipeDetector()
    return CVPipeline(
        extractor=extractor,
        classifier=classifier,
        session_gate=gate,
        swipe_detector=swipe,
        draw_skeleton=True,
    )


def main() -> int:
    """Application entry point.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    # --- Bootstrap --------------------------------------------------------
    try:
        load_env()
    except (EnvironmentError, FileNotFoundError) as exc:
        # Logger not yet configured; print directly.
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    configure_root_logger()
    logger.info("Gestureify starting up (v%s).", "0.1.0")

    # --- Authentication ---------------------------------------------------
    token_store = TokenStore(settings.TOKEN_CACHE_PATH)
    auth = SpotifyAuth(
        client_id=os.environ["SPOTIFY_CLIENT_ID"],
        redirect_uri=settings.SPOTIFY_REDIRECT_URI,
        scopes=settings.SPOTIFY_SCOPES,
        token_store=token_store,
    )
    try:
        auth.ensure_valid_token()
    except Exception as exc:  # noqa: BLE001
        logger.error("Spotify authentication failed: %s", exc)
        return 1

    # --- Build sub-systems ------------------------------------------------
    spotify_client = SpotifyClient()
    media_keys = MediaKeyFallback()
    controller = PlaybackController(
        auth=auth,
        spotify_client=spotify_client,
        media_keys=media_keys,
    )

    pipeline = _build_pipeline()

    hud_queue: queue.Queue[HUDMessage] = queue.Queue(maxsize=10)
    hud = HUDOverlay(message_queue=hud_queue)

    stop_event = threading.Event()

    # --- Camera -----------------------------------------------------------
    capture = CameraCapture(
        index=int(os.environ.get("CAMERA_INDEX", settings.CAMERA_INDEX)),
        width=settings.CAMERA_WIDTH,
        height=settings.CAMERA_HEIGHT,
        fps=settings.CAMERA_FPS,
    )
    try:
        capture.open()
    except RuntimeError as exc:
        logger.error("Camera error: %s", exc)
        return 1

    # --- Signal handling --------------------------------------------------
    def _shutdown(signum, frame) -> None:  # noqa: ANN001
        logger.info("Shutdown signal received.")
        stop_event.set()
        hud.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # --- Start CV thread --------------------------------------------------
    cv_thread = threading.Thread(
        target=_cv_loop,
        args=(pipeline, capture, controller, hud_queue, stop_event),
        daemon=True,
        name="cv-loop",
    )
    cv_thread.start()
    logger.info("CV thread started.")

    # --- Run HUD (blocks until window is closed) --------------------------
    try:
        hud.run()
    finally:
        stop_event.set()
        cv_thread.join(timeout=3.0)
        capture.close()
        logger.info("Gestureify shut down cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
