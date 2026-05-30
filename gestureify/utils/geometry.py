"""
gestureify.utils.geometry
==========================
Vectorised landmark geometry using JAX.

All public functions operate on a 21-point landmark array and are JIT-compiled
on first call.  Subsequent calls at 30 FPS hit the compiled cache with zero
Python overhead in the hot path.

Why JAX here?
-------------
MediaPipe returns 21 (x, y) pairs per frame.  Naively computing distances in
a Python loop with ``math.sqrt`` is 21 scalar sqrt calls per frame.  With JAX
we stack all points into a (21, 2) array and compute all distances in a single
``jnp.linalg.norm`` call — one kernel launch, SIMD-vectorised on CPU.

For a hackathon project the performance difference is small, but the pattern
matters: keeping the hot path free of Python-level loops is the correct habit.

Landmark index reference (MediaPipe Hands)
------------------------------------------
 0  WRIST
 1  THUMB_CMC   2  THUMB_MCP   3  THUMB_IP    4  THUMB_TIP
 5  INDEX_MCP   6  INDEX_PIP   7  INDEX_DIP   8  INDEX_TIP
 9  MIDDLE_MCP 10  MIDDLE_PIP 11  MIDDLE_DIP 12  MIDDLE_TIP
13  RING_MCP   14  RING_PIP   15  RING_DIP   16  RING_TIP
17  PINKY_MCP  18  PINKY_PIP  19  PINKY_DIP  20  PINKY_TIP

Design notes
------------
* ``LandmarkArray`` is a (21, 2) JAX float32 array — the canonical internal
  representation.  ``LandmarkList`` (list of tuples) is the MediaPipe-facing
  format; convert with ``to_array()``.
* All JAX functions are pure (no side-effects) and decorated with
  ``@jax.jit`` for ahead-of-time compilation.
* Landmark indices are ``Final[int]`` constants — mutation is a type error.
"""

from __future__ import annotations

from typing import Final, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# MediaPipe-facing format: list of (x, y) normalised float tuples.
LandmarkList = List[Tuple[float, float]]

# Internal JAX format: (21, 2) float32 array.
LandmarkArray = jax.Array

# ---------------------------------------------------------------------------
# Landmark indices  (Final → mutation is a mypy error)
# ---------------------------------------------------------------------------

WRIST: Final[int] = 0
THUMB_TIP: Final[int] = 4
INDEX_MCP: Final[int] = 5
INDEX_TIP: Final[int] = 8
MIDDLE_TIP: Final[int] = 12
RING_TIP: Final[int] = 16
PINKY_TIP: Final[int] = 20

# Fingertip indices — deliberately excludes the thumb (index 4).
# The thumb has fundamentally different extension geometry; including it
# causes a closed fist with an extended thumb to misclassify as OPEN_PALM.
_FINGERTIP_INDICES: Final[Tuple[int, ...]] = (
    INDEX_TIP,   # 8
    MIDDLE_TIP,  # 12
    RING_TIP,    # 16
    PINKY_TIP,   # 20
)

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def to_array(landmarks: LandmarkList) -> LandmarkArray:
    """Convert a MediaPipe landmark list to a (21, 2) JAX float32 array.

    Parameters
    ----------
    landmarks:
        21-element list of ``(x, y)`` normalised coordinate tuples.

    Returns
    -------
    LandmarkArray
        Shape ``(21, 2)``, dtype ``float32``.
    """
    return jnp.array(landmarks, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# JAX-compiled geometry kernels
# ---------------------------------------------------------------------------


@jax.jit
def _fingertip_wrist_ratio_jax(pts: LandmarkArray) -> jax.Array:
    """Compute mean four-fingertip-to-wrist distance normalised by palm width.

    Excludes the thumb.  Returns a scalar float32.
    """
    wrist = pts[WRIST]                          # (2,)
    index_mcp = pts[INDEX_MCP]                  # (2,)
    palm_w = jnp.linalg.norm(index_mcp - wrist)
    palm_w = jnp.where(palm_w < 1e-6, 1e-6, palm_w)  # guard division by zero

    fingertips = pts[jnp.array(_FINGERTIP_INDICES)]   # (4, 2)
    dists = jnp.linalg.norm(fingertips - wrist, axis=1)  # (4,)
    return jnp.mean(dists) / palm_w


@jax.jit
def _pinch_distance_jax(pts: LandmarkArray) -> jax.Array:
    """Normalised Euclidean distance between thumb tip and index tip.

    Normalised by palm width so the value is scale-invariant.
    """
    wrist = pts[WRIST]
    index_mcp = pts[INDEX_MCP]
    palm_w = jnp.linalg.norm(index_mcp - wrist)
    palm_w = jnp.where(palm_w < 1e-6, 1e-6, palm_w)

    gap = jnp.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])
    return gap / palm_w


# ---------------------------------------------------------------------------
# Public API  (accepts LandmarkList for compatibility with MediaPipe callers)
# ---------------------------------------------------------------------------


def fingertip_wrist_ratio(landmarks: LandmarkList) -> float:
    """Return the mean four-fingertip-to-wrist ratio (thumb excluded).

    Higher values → more open hand.  Lower values → closed fist.

    Parameters
    ----------
    landmarks:
        21-element MediaPipe landmark list.

    Returns
    -------
    float
        Dimensionless ratio; typical open-palm ≈ 2.5, closed fist ≈ 0.4.
    """
    pts = to_array(landmarks)
    return float(_fingertip_wrist_ratio_jax(pts))


def pinch_distance(landmarks: LandmarkList) -> float:
    """Return the normalised thumb-tip to index-tip distance.

    Parameters
    ----------
    landmarks:
        21-element MediaPipe landmark list.

    Returns
    -------
    float
        Normalised distance.  Values < ``PINCH_DISTANCE_THRESHOLD`` indicate
        an active pinch.
    """
    pts = to_array(landmarks)
    return float(_pinch_distance_jax(pts))


def wrist_x(landmarks: LandmarkList) -> float:
    """Return the normalised X coordinate of the wrist landmark.

    Parameters
    ----------
    landmarks:
        21-element MediaPipe landmark list.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``.
    """
    return landmarks[WRIST][0]


# ---------------------------------------------------------------------------
# Legacy scalar helpers (kept for tests; not used in the hot path)
# ---------------------------------------------------------------------------


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float(np.sqrt(dx * dx + dy * dy))


def palm_width(landmarks: LandmarkList) -> float:
    """Wrist-to-index-MCP distance (used as a scale reference).

    Returns a small epsilon instead of zero when the two points coincide,
    preventing division-by-zero in callers.
    """
    w = euclidean_distance(landmarks[WRIST], landmarks[INDEX_MCP])
    return w if w > 1e-6 else 1e-6


def normalised_distance(
    a: Tuple[float, float],
    b: Tuple[float, float],
    reference: float,
) -> float:
    """Euclidean distance between *a* and *b*, divided by *reference*."""
    return euclidean_distance(a, b) / reference
