"""
main.py
=======
Gestureify — Gesture-controlled Spotify player.

Entry point.  Responsibilities:
  1. Load environment / validate secrets via ``AppConfig``.
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
* ``AppConfig.from_env()`` is the single validated entry-point for all runtime
  configuration.  A ``pydantic.ValidationError`` on startup means the user
  gets a structured, actionable error listing every problem at once.
* All sub-systems are constructed here and injected into their consumers
  (dependency injection, not global singletons).
* The CV thread is a daemon thread so it is killed automatically if the
  main thread exits unexpectedly.
* Signal handling (SIGINT / KeyboardInterrupt) is caught to ensure the
  camera is always released (NASA Rule 7).
"""

from __future__ import annotations

import queue
import signal
import sys
import threading

import pydantic

from gestureify.auth.spotify_auth import SpotifyAuth
from gestureify.auth.token_store import TokenStore
from gestureify.config import load_env, settings
from gestureify.config.app_config import AppConfig
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
_LABEL_PAUSE = "PAUSE"
_LABEL_RESUME = "RESUME"
_LABEL_NEXT = "NEXT >"
_LABEL_PREV = "< PREV"
_LABEL_ACTIVE = "SESSION ON"
_LABEL_IDLE = "SESSION OFF"


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
    is_playing: bool = True

    for frame in capture.frames():
        if stop_event.is_set():
            break

        result: FrameResult = pipeline.process(frame)

        # --- Dispatch commands only when session is ACTIVE ---------------
        action_label: str = ""

        if result.session_toggled:
            action_label = (
                _LABEL_ACTIVE
                if result.session_state is SessionState.ACTIVE
                else _LABEL_IDLE
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


def _build_pipeline(cfg: AppConfig) -> CVPipeline:
    """Construct and return a fully wired ``CVPipeline`` from *cfg*."""
    extractor = LandmarkExtractor(
        max_hands=cfg.mp_max_hands,
        detection_confidence=cfg.mp_detection_confidence,
        tracking_confidence=cfg.mp_tracking_confidence,
    )
    extractor.open()
    classifier = GestureClassifier(
        fist_threshold=cfg.fist_ratio_threshold,
        palm_threshold=cfg.wake_palm_ratio_threshold,
        pinch_threshold=cfg.pinch_distance_threshold,
    )
    gate = SessionGate(hold_seconds=cfg.wake_hold_seconds)
    swipe = SwipeDetector(
        velocity_window=cfg.swipe_velocity_window,
        velocity_threshold=cfg.swipe_velocity_threshold,
        cooldown_seconds=cfg.swipe_cooldown_seconds,
    )
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
    # --- Bootstrap: load .env then validate all config via Pydantic ----------
    try:
        load_env()
    except (EnvironmentError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        cfg = AppConfig.from_env()
    except pydantic.ValidationError as exc:
        print("[ERROR] Configuration is invalid:\n", file=sys.stderr)
        for err in exc.errors():
            loc = " → ".join(str(x) for x in err["loc"])
            print(f"  {loc}: {err['msg']}", file=sys.stderr)
        return 1

    configure_root_logger(cfg.log_level)
    logger.info("Gestureify starting up (v%s).", "0.1.0")

    # --- Authentication ---------------------------------------------------
    token_store = TokenStore(settings.TOKEN_CACHE_PATH)
    auth = SpotifyAuth(
        client_id=cfg.spotify_client_id,
        redirect_uri=cfg.spotify_redirect_uri,
        scopes=cfg.spotify_scopes,
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

    pipeline = _build_pipeline(cfg)

    hud_queue: queue.Queue[HUDMessage] = queue.Queue(maxsize=10)
    hud = HUDOverlay(message_queue=hud_queue)

    stop_event = threading.Event()

    # --- Camera -----------------------------------------------------------
    capture = CameraCapture(
        index=cfg.camera_index,
        width=cfg.camera_width,
        height=cfg.camera_height,
        fps=cfg.camera_fps,
    )
    try:
        capture.open()
    except RuntimeError as exc:
        logger.error("Camera error: %s", exc)
        return 1

    # --- Signal handling --------------------------------------------------
    def _shutdown(signum: int, frame: object) -> None:
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
