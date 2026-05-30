"""
gestureify.cv_engine.gesture_classifier
========================================
Stateless classifier that maps a 21-point landmark list to a ``GestureLabel``.

Only two gestures are recognised:
  * ``CLOSED_FIST``  — all fingers folded  → pause playback.
  * ``OPEN_PALM``    — all fingers extended → resume playback / session toggle.

Design notes
------------
* All classification functions are pure (no side-effects, NASA Rule 6).
* Thresholds are read from ``gestureify.config.settings`` so they can be
  tuned without touching this file.
* ``GestureLabel`` is an ``IntEnum`` so it can be used in comparisons and
  stored compactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from gestureify.config import settings
from gestureify.utils.geometry import (
    LandmarkList,
    fingertip_wrist_ratio,
)
from gestureify.utils.logger import get_logger

logger = get_logger(__name__)


class GestureLabel(IntEnum):
    """Enumeration of recognisable static gestures."""

    NONE = 0         # No confident gesture detected.
    OPEN_PALM = 1    # All four fingers extended — resume playback / session toggle.
    CLOSED_FIST = 2  # All fingers folded — pause playback.


@dataclass(slots=True)
class ClassifyResult:
    """Output of a single ``GestureClassifier.classify_full()`` call."""

    label: GestureLabel


class GestureClassifier:
    """Classify a single frame's landmark list into a ``GestureLabel``.

    Priority order:
      1. Closed fist (ratio below fist threshold).
      2. Open palm (ratio above palm threshold).
      3. None (ambiguous / transitional hand shape).

    Parameters
    ----------
    fist_threshold:
        Fingertip-wrist ratio below which the hand is classified as a fist.
    palm_threshold:
        Fingertip-wrist ratio above which the hand is classified as open palm.
    """

    def __init__(
        self,
        fist_threshold: float = settings.FIST_RATIO_THRESHOLD,
        palm_threshold: float = settings.WAKE_PALM_RATIO_THRESHOLD,
    ) -> None:
        self._fist_threshold = fist_threshold
        self._palm_threshold = palm_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        landmarks: Optional[LandmarkList],
    ) -> GestureLabel:
        """Return the gesture label for the given landmark list."""
        return self.classify_full(landmarks).label

    def classify_full(
        self,
        landmarks: Optional[LandmarkList],
    ) -> ClassifyResult:
        """Classify the gesture and return the result."""
        if landmarks is None:
            return ClassifyResult(label=GestureLabel.NONE)

        ratio = fingertip_wrist_ratio(landmarks)

        if ratio < self._fist_threshold:
            return ClassifyResult(label=GestureLabel.CLOSED_FIST)

        if ratio > self._palm_threshold:
            return ClassifyResult(label=GestureLabel.OPEN_PALM)

        return ClassifyResult(label=GestureLabel.NONE)
