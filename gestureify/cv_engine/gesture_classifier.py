"""
gestureify.cv_engine.gesture_classifier
========================================
Stateless classifier that maps a 21-point landmark list to a ``GestureLabel``.

Only *static* gestures (fist, open palm, pinch) are classified here.
Dynamic gestures (swipes) are handled by ``SwipeDetector`` because they
require temporal state across frames.

Design notes
------------
* All classification functions are pure (no side-effects, NASA Rule 6).
* Thresholds are read from ``gestureify.config.settings`` so they can be
  tuned without touching this file.
* ``GestureLabel`` is an ``IntEnum`` so it can be used in comparisons and
  stored compactly.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

from gestureify.config import settings
from gestureify.utils.geometry import (
    LandmarkList,
    fingertip_wrist_ratio,
    pinch_distance,
)
from gestureify.utils.logger import get_logger

logger = get_logger(__name__)


class GestureLabel(IntEnum):
    """Enumeration of recognisable static gestures.

    Values are assigned explicitly so they remain stable across refactors.
    """

    NONE = 0        # No confident gesture detected.
    OPEN_PALM = 1   # All fingers extended — used for wake/session toggle.
    CLOSED_FIST = 2 # All fingers folded — pause playback.
    PINCH = 3       # Thumb tip touching index tip — volume control.


class GestureClassifier:
    """Classify a single frame's landmark list into a ``GestureLabel``.

    The classifier applies a priority order:
      1. Pinch (most specific — subset of open-palm geometry).
      2. Closed fist.
      3. Open palm.
      4. None.

    Parameters
    ----------
    fist_threshold:
        Fingertip-wrist ratio below which the hand is classified as a fist.
    palm_threshold:
        Fingertip-wrist ratio above which the hand is classified as open palm.
    pinch_threshold:
        Normalised thumb-index distance below which a pinch is detected.
    """

    def __init__(
        self,
        fist_threshold: float = settings.FIST_RATIO_THRESHOLD,
        palm_threshold: float = settings.WAKE_PALM_RATIO_THRESHOLD,
        pinch_threshold: float = settings.PINCH_DISTANCE_THRESHOLD,
    ) -> None:
        self._fist_threshold = fist_threshold
        self._palm_threshold = palm_threshold
        self._pinch_threshold = pinch_threshold

    def classify(
        self,
        landmarks: Optional[LandmarkList],
    ) -> GestureLabel:
        """Return the gesture label for the given landmark list.

        Parameters
        ----------
        landmarks:
            21-point normalised landmark list, or *None* if no hand detected.

        Returns
        -------
        GestureLabel
            The most specific matching gesture, or ``GestureLabel.NONE``.
        """
        if landmarks is None:
            return GestureLabel.NONE

        if self._is_pinch(landmarks):
            return GestureLabel.PINCH

        ratio = fingertip_wrist_ratio(landmarks)

        if ratio < self._fist_threshold:
            return GestureLabel.CLOSED_FIST

        if ratio > self._palm_threshold:
            return GestureLabel.OPEN_PALM

        return GestureLabel.NONE

    def pinch_gap(self, landmarks: Optional[LandmarkList]) -> float:
        """Return the normalised pinch gap for volume mapping.

        Parameters
        ----------
        landmarks:
            21-point landmark list, or *None*.

        Returns
        -------
        float
            Normalised distance in [0, ∞).  Returns 1.0 when *landmarks*
            is *None* (neutral / no change).
        """
        if landmarks is None:
            return 1.0
        return pinch_distance(landmarks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_pinch(self, landmarks: LandmarkList) -> bool:
        """Return ``True`` if the thumb and index tips are close enough."""
        return pinch_distance(landmarks) < self._pinch_threshold
