"""
gestureify.cv_engine.landmark_extractor
========================================
MediaPipe Hands wrapper that converts raw BGR frames into normalised 2-D
landmark sequences.

This module is the only place in the codebase that imports ``mediapipe``,
keeping the MediaPipe API surface isolated and easy to swap out.

Design notes
------------
* ``LandmarkExtractor`` implements the context-manager protocol so the
  MediaPipe solution is properly closed on exit (resource safety).
* The public ``extract()`` method returns a plain Python list of ``(x, y)``
  tuples rather than MediaPipe's internal types, so downstream modules
  have no dependency on the mediapipe package.
* At most ``MP_MAX_HANDS`` hands are tracked (default 1) to keep CPU usage
  bounded (NASA Rule 2).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from gestureify.config import settings
from gestureify.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias: a list of 21 (x, y) normalised landmark points.
LandmarkList = List[Tuple[float, float]]


class LandmarkExtractor:
    """Extract 21 hand landmarks from a BGR frame using MediaPipe Hands.

    Parameters
    ----------
    max_hands:
        Maximum number of hands to detect simultaneously.
    detection_confidence:
        Minimum confidence for initial palm detection.
    tracking_confidence:
        Minimum confidence for landmark tracking between frames.
    """

    def __init__(
        self,
        max_hands: int = settings.MP_MAX_HANDS,
        detection_confidence: float = settings.MP_DETECTION_CONFIDENCE,
        tracking_confidence: float = settings.MP_TRACKING_CONFIDENCE,
    ) -> None:
        self._max_hands = max_hands
        self._detection_confidence = detection_confidence
        self._tracking_confidence = tracking_confidence
        self._hands: Optional[mp.solutions.hands.Hands] = None  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LandmarkExtractor":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Initialise the MediaPipe Hands solution."""
        self._hands = mp.solutions.hands.Hands(  # type: ignore[attr-defined]
            static_image_mode=False,
            max_num_hands=self._max_hands,
            min_detection_confidence=self._detection_confidence,
            min_tracking_confidence=self._tracking_confidence,
        )
        logger.info(
            "MediaPipe Hands initialised (max_hands=%d, det=%.2f, track=%.2f).",
            self._max_hands,
            self._detection_confidence,
            self._tracking_confidence,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()
            self._hands = None
            logger.info("MediaPipe Hands closed.")

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, bgr_frame: np.ndarray) -> Optional[LandmarkList]:
        """Run MediaPipe on *bgr_frame* and return the first hand's landmarks.

        Parameters
        ----------
        bgr_frame:
            A BGR uint8 frame from OpenCV.

        Returns
        -------
        LandmarkList | None
            A list of 21 ``(x, y)`` normalised coordinate tuples for the
            first detected hand, or *None* if no hand is detected.
        """
        if self._hands is None:
            logger.error("extract() called before open().")
            return None

        # MediaPipe expects RGB; convert in-place (no copy).
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        # Take only the first detected hand.
        hand = result.multi_hand_landmarks[0]
        return [(lm.x, lm.y) for lm in hand.landmark]

    def extract_with_drawing(
        self,
        bgr_frame: np.ndarray,
    ) -> Tuple[Optional[LandmarkList], np.ndarray]:
        """Extract landmarks and draw the hand skeleton onto a copy of the frame.

        Parameters
        ----------
        bgr_frame:
            Input BGR frame.

        Returns
        -------
        tuple[LandmarkList | None, numpy.ndarray]
            Landmark list (or *None*) and an annotated BGR frame copy.
        """
        annotated = bgr_frame.copy()
        landmarks = self.extract(bgr_frame)

        if landmarks is not None:
            # Re-run to get the raw MediaPipe object for drawing utilities.
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self._hands.process(rgb)  # type: ignore[union-attr]
            if result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(  # type: ignore[attr-defined]
                    annotated,
                    result.multi_hand_landmarks[0],
                    mp.solutions.hands.HAND_CONNECTIONS,  # type: ignore[attr-defined]
                )

        return landmarks, annotated
