"""
gestureify.utils.geometry
=========================
Pure-function geometry helpers for MediaPipe landmark arithmetic.

All functions are stateless and side-effect-free (NASA Rule 6).
Landmark coordinates are expected as ``(x, y)`` tuples of floats in the
[0, 1] normalised image space that MediaPipe returns.

MediaPipe hand landmark indices used throughout the codebase
------------------------------------------------------------
0  -- WRIST
4  -- THUMB_TIP
5  -- INDEX_FINGER_MCP
8  -- INDEX_FINGER_TIP
12 -- MIDDLE_FINGER_TIP
16 -- RING_FINGER_TIP
20 -- PINKY_TIP
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

# Type alias for a 2-D normalised landmark point.
Point2D = Tuple[float, float]

# Type alias for a full 21-point hand landmark sequence.
LandmarkList = List[Point2D]

# Landmark indices (mirrors mediapipe.solutions.hands.HandLandmark).
WRIST: int = 0
THUMB_TIP: int = 4
INDEX_MCP: int = 5
INDEX_TIP: int = 8
MIDDLE_TIP: int = 12
RING_TIP: int = 16
PINKY_TIP: int = 20

# Fingertip landmark indices used for open/closed hand classification.
FINGERTIP_INDICES: Tuple[int, ...] = (
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
)


def euclidean_distance(a: Point2D, b: Point2D) -> float:
    """Return the Euclidean distance between two 2-D points.

    Parameters
    ----------
    a, b:
        Points as ``(x, y)`` float tuples.

    Returns
    -------
    float
        Non-negative distance value.
    """
    return math.hypot(b[0] - a[0], b[1] - a[1])


def palm_width(landmarks: Sequence[Point2D]) -> float:
    """Estimate palm width as the distance from wrist to index MCP.

    This value is used to normalise other distances so that the gesture
    thresholds are invariant to how far the user's hand is from the camera.

    Parameters
    ----------
    landmarks:
        Sequence of at least 6 ``(x, y)`` landmark points (indices 0–5).

    Returns
    -------
    float
        Normalised palm width.  Returns a small epsilon if landmarks are
        degenerate to avoid division-by-zero downstream.
    """
    width = euclidean_distance(landmarks[WRIST], landmarks[INDEX_MCP])
    return max(width, 1e-6)


def normalised_distance(
    a: Point2D,
    b: Point2D,
    reference: float,
) -> float:
    """Return the distance between *a* and *b* divided by *reference*.

    Parameters
    ----------
    a, b:
        Points to measure.
    reference:
        Normalisation factor (e.g. palm width).  Must be > 0.

    Returns
    -------
    float
        Scale-invariant distance ratio.
    """
    return euclidean_distance(a, b) / reference


def fingertip_wrist_ratio(landmarks: Sequence[Point2D]) -> float:
    """Compute the mean normalised fingertip-to-wrist distance.

    A high ratio indicates an open palm; a low ratio indicates a closed fist.

    Parameters
    ----------
    landmarks:
        Full 21-point landmark sequence from MediaPipe.

    Returns
    -------
    float
        Mean ratio of fingertip distances to the palm-width reference.
    """
    ref = palm_width(landmarks)
    wrist = landmarks[WRIST]
    total = sum(
        euclidean_distance(wrist, landmarks[idx])
        for idx in FINGERTIP_INDICES
    )
    return (total / len(FINGERTIP_INDICES)) / ref


def pinch_distance(landmarks: Sequence[Point2D]) -> float:
    """Return the normalised distance between thumb tip and index tip.

    Parameters
    ----------
    landmarks:
        Full 21-point landmark sequence from MediaPipe.

    Returns
    -------
    float
        Scale-invariant pinch gap.  0.0 means fully pinched.
    """
    ref = palm_width(landmarks)
    return normalised_distance(
        landmarks[THUMB_TIP],
        landmarks[INDEX_TIP],
        ref,
    )


def wrist_x(landmarks: Sequence[Point2D]) -> float:
    """Return the normalised x-coordinate of the wrist landmark.

    Parameters
    ----------
    landmarks:
        Full 21-point landmark sequence from MediaPipe.

    Returns
    -------
    float
        Value in [0, 1] where 0 is the left edge of the frame.
    """
    return landmarks[WRIST][0]
