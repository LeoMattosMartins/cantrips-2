"""
gestureify.cv_engine.pipeline
==============================
Orchestrates the per-frame computer-vision processing pipeline.

``CVPipeline`` wires together:
  * ``LandmarkExtractor``  — MediaPipe hand tracking.
  * ``GestureClassifier``  — static gesture labelling (fist / open palm).
  * ``SessionGate``        — open-palm session toggle.

It produces a ``FrameResult`` dataclass each frame, which the main loop
consumes to drive the ``PlaybackController`` and ``HUDOverlay``.

Design notes
------------
* The pipeline does **not** call the Spotify API directly; it only emits
  intent signals.  This separation keeps CV and I/O concerns orthogonal.
* All sub-components are injected via the constructor for testability.
* ``FrameResult`` uses ``slots=True`` (Python 3.10+) to eliminate per-instance
  ``__dict__`` allocation at 30 FPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gestureify.cv_engine.gesture_classifier import (
    ClassifyResult,
    GestureClassifier,
    GestureLabel,
)
from gestureify.cv_engine.landmark_extractor import LandmarkExtractor, LandmarkList
from gestureify.cv_engine.session_gate import SessionGate, SessionState
from gestureify.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
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
    draw_skeleton:
        Whether to annotate frames with the hand skeleton.
    """

    def __init__(
        self,
        extractor: LandmarkExtractor,
        classifier: GestureClassifier,
        session_gate: SessionGate,
        draw_skeleton: bool = True,
    ) -> None:
        self._extractor = extractor
        self._classifier = classifier
        self._gate = session_gate
        self._draw = draw_skeleton

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, bgr_frame: np.ndarray) -> FrameResult:
        """Run the full pipeline on a single BGR frame."""
        # Step 1: Extract landmarks.
        if self._draw:
            landmarks, annotated = self._extractor.extract_with_drawing(bgr_frame)
        else:
            landmarks = self._extractor.extract(bgr_frame)
            annotated = bgr_frame

        # Step 2: Classify the static gesture.
        clf: ClassifyResult = self._classifier.classify_full(landmarks)

        # Step 3: Update the session gate (open-palm toggle).
        toggled = self._gate.update(clf.label)
        hold_progress = self._gate.hold_progress()

        # Step 4: If the session is IDLE, suppress all command gestures.
        if not self._gate.is_active:
            return FrameResult(
                session_state=self._gate.state,
                session_toggled=toggled,
                gesture=GestureLabel.NONE,
                landmarks=landmarks,
                annotated_frame=annotated,
                hold_progress=hold_progress,
            )

        # Step 5: Suppress OPEN_PALM as a command gesture (it is the wake
        # gesture, not a playback command).
        command_gesture = (
            GestureLabel.NONE
            if clf.label is GestureLabel.OPEN_PALM
            else clf.label
        )

        return FrameResult(
            session_state=self._gate.state,
            session_toggled=toggled,
            gesture=command_gesture,
            landmarks=landmarks,
            annotated_frame=annotated,
            hold_progress=hold_progress,
        )
