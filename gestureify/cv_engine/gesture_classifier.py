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
* ``classify()`` returns both the label *and* the pre-computed pinch gap so
  that ``pipeline.py`` can call ``pinch_gap()`` without re-running the JAX
  kernel a second time on the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

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

    NONE = 0         # No confident gesture detected.
    OPEN_PALM = 1    # All four fingers extended — used for wake/session toggle.
    CLOSED_FIST = 2  # All fingers folded — pause playback.
    PINCH = 3        # Thumb tip touching index tip — volume control.


@dataclass(slots=True)
class ClassifyResult:
    """Output of a single ``GestureClassifier.classify_full()`` call.

    Attributes
    ----------
    label:
        The detected gesture.
    pinch_gap:
        Pre-computed normalised pinch distance (always available, even when
        the gesture is not PINCH).  Avoids a redundant JAX kernel call in
        ``pipeline.py``.
    """

    label: GestureLabel
    pinch_gap: float


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        landmarks: Optional[LandmarkList],
    ) -> GestureLabel:
        """Return the gesture label for the given landmark list.

        Prefer ``classify_full()`` in the hot path to avoid computing
        ``pinch_distance`` twice per frame.

        Parameters
        ----------
        landmarks:
            21-point normalised landmark list, or *None* if no hand detected.

        Returns
        -------
        GestureLabel
        """
        return self.classify_full(landmarks).label

    def classify_full(
        self,
        landmarks: Optional[LandmarkList],
    ) -> ClassifyResult:
        """Classify the gesture and return both the label and pinch gap.

        ``pinch_distance`` is computed exactly once per call and reused for
        both the PINCH classification decision and the volume-mapping value.

        Parameters
        ----------
        landmarks:
            21-point normalised landmark list, or *None*.

        Returns
        -------
        ClassifyResult
        """
        if landmarks is None:
            return ClassifyResult(label=GestureLabel.NONE, pinch_gap=1.0)

        gap = pinch_distance(landmarks)

        if gap < self._pinch_threshold:
            return ClassifyResult(label=GestureLabel.PINCH, pinch_gap=gap)

        ratio = fingertip_wrist_ratio(landmarks)

        if ratio < self._fist_threshold:
            return ClassifyResult(label=GestureLabel.CLOSED_FIST, pinch_gap=gap)

        if ratio > self._palm_threshold:
            return ClassifyResult(label=GestureLabel.OPEN_PALM, pinch_gap=gap)

        return ClassifyResult(label=GestureLabel.NONE, pinch_gap=gap)

    def pinch_gap(self, landmarks: Optional[LandmarkList]) -> float:
        """Return the normalised pinch gap for volume mapping.

        This is a convenience wrapper for callers that only need the gap.
        In the pipeline hot path, use ``classify_full()`` instead.

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
