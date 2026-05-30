"""
gestureify.cv_engine.pipeline
==============================
Orchestrates the per-frame computer-vision processing pipeline.

``CVPipeline`` wires together:
  * ``LandmarkExtractor``  — MediaPipe hand tracking.
  * ``GestureClassifier``  — static gesture labelling.
  * ``SessionGate``        — open-palm session toggle.
  * ``SwipeDetector``      — velocity-based swipe recognition.

It produces a ``FrameResult`` dataclass each frame, which the main loop
consumes to drive the ``PlaybackController`` and ``HUDOverlay``.

Design notes
------------
* The pipeline does **not** call the Spotify API directly; it only emits
  intent signals.  This separation keeps CV and I/O concerns orthogonal.
* All sub-components are injected via the constructor for testability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gestureify.cv_engine.gesture_classifier import GestureClassifier, GestureLabel
from gestureify.cv_engine.landmark_extractor import LandmarkExtractor, LandmarkList
from gestureify.cv_engine.session_gate import SessionGate, SessionState
from gestureify.cv_engine.swipe_detector import SwipeDetector, SwipeDirection
from gestureify.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameResult:
    """Output produced by ``CVPipeline.process()`` for a single frame.

    Attributes
    ----------
    session_state:
        Whether the session is currently ACTIVE or IDLE.
    session_toggled:
        ``True`` if the session state just changed this frame.
    gesture:
        The static gesture detected this frame (``NONE`` if session is IDLE
        or no hand is present).
    swipe:
        Swipe direction if one was just detected, otherwise ``None``.
    pinch_gap:
        Normalised pinch distance (only meaningful when gesture == PINCH).
    landmarks:
        Raw 21-point landmark list, or ``None`` if no hand detected.
    annotated_frame:
        BGR frame with hand skeleton drawn (for HUD preview).
    hold_progress:
        Wake-gesture hold progress in [0.0, 1.0] for HUD arc rendering.
    """

    session_state: SessionState
    session_toggled: bool
    gesture: GestureLabel
    swipe: Optional[SwipeDirection]
    pinch_gap: float
    landmarks: Optional[LandmarkList]
    annotated_frame: Optional[np.ndarray]
    hold_progress: float = field(default=0.0)


class CVPipeline:
    """Per-frame CV processing pipeline.

    Parameters
    ----------
    extractor:
        Initialised ``LandmarkExtractor`` instance.
    classifier:
        ``GestureClassifier`` instance.
    session_gate:
        ``SessionGate`` instance managing the session toggle.
    swipe_detector:
        ``SwipeDetector`` instance.
    draw_skeleton:
        Whether to annotate frames with the hand skeleton.
    """

    def __init__(
        self,
        extractor: LandmarkExtractor,
        classifier: GestureClassifier,
        session_gate: SessionGate,
        swipe_detector: SwipeDetector,
        draw_skeleton: bool = True,
    ) -> None:
        self._extractor = extractor
        self._classifier = classifier
        self._gate = session_gate
        self._swipe = swipe_detector
        self._draw = draw_skeleton

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, bgr_frame: np.ndarray) -> FrameResult:
        """Run the full pipeline on a single BGR frame.

        Parameters
        ----------
        bgr_frame:
            Raw camera frame.

        Returns
        -------
        FrameResult
            Structured result for the current frame.
        """
        # Step 1: Extract landmarks (and optionally annotate the frame).
        if self._draw:
            landmarks, annotated = self._extractor.extract_with_drawing(bgr_frame)
        else:
            landmarks = self._extractor.extract(bgr_frame)
            annotated = bgr_frame

        # Step 2: Classify the static gesture.
        raw_gesture = self._classifier.classify(landmarks)

        # Step 3: Update the session gate (open-palm toggle).
        #         The gate always sees the raw gesture, even when IDLE,
        #         so the user can activate the session.
        toggled = self._gate.update(raw_gesture)
        hold_progress = self._gate.hold_progress()

        # Step 4: If the session is IDLE, suppress all command gestures.
        if not self._gate.is_active:
            return FrameResult(
                session_state=self._gate.state,
                session_toggled=toggled,
                gesture=GestureLabel.NONE,
                swipe=None,
                pinch_gap=1.0,
                landmarks=landmarks,
                annotated_frame=annotated,
                hold_progress=hold_progress,
            )

        # Step 5: Detect swipes (only when session is ACTIVE).
        from gestureify.utils.geometry import wrist_x  # local import avoids cycle
        wx = wrist_x(landmarks) if landmarks is not None else None
        swipe_direction = self._swipe.update(wx)

        # Step 6: Compute pinch gap for volume mapping.
        pinch_gap = self._classifier.pinch_gap(landmarks)

        # Step 7: Suppress OPEN_PALM as a command gesture (it is the wake
        #         gesture, not a playback command).
        command_gesture = (
            GestureLabel.NONE
            if raw_gesture is GestureLabel.OPEN_PALM
            else raw_gesture
        )

        return FrameResult(
            session_state=self._gate.state,
            session_toggled=toggled,
            gesture=command_gesture,
            swipe=swipe_direction,
            pinch_gap=pinch_gap,
            landmarks=landmarks,
            annotated_frame=annotated,
            hold_progress=hold_progress,
        )
