"""
tests/test_geometry.py
======================
Unit tests for gestureify.utils.geometry.

All tests are pure (no I/O, no side-effects) and run without a camera or
MediaPipe model installed.
"""

import math
import pytest

from gestureify.utils.geometry import (
    euclidean_distance,
    fingertip_wrist_ratio,
    normalised_distance,
    palm_width,
    pinch_distance,
    wrist_x,
    WRIST,
    INDEX_MCP,
    THUMB_TIP,
    INDEX_TIP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmarks(n: int = 21) -> list:
    """Return a list of *n* zero-initialised (x, y) landmark tuples."""
    return [(0.0, 0.0)] * n


def _open_hand_landmarks() -> list:
    """Approximate an open hand: fingertips far from wrist."""
    lms = _make_landmarks()
    # Wrist at origin, index MCP at (0.1, 0.0) → palm_width = 0.1
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.1, 0.0)
    # Fingertips at distance ~0.4 from wrist → ratio ≈ 4.0
    for idx in (4, 8, 12, 16, 20):
        lms[idx] = (0.0, 0.4)
    return lms


def _closed_fist_landmarks() -> list:
    """Approximate a closed fist: fingertips close to wrist."""
    lms = _make_landmarks()
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.1, 0.0)
    # Fingertips at distance ~0.05 from wrist → ratio ≈ 0.5
    for idx in (4, 8, 12, 16, 20):
        lms[idx] = (0.0, 0.05)
    return lms


# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------

def test_euclidean_distance_zero():
    assert euclidean_distance((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0)


def test_euclidean_distance_unit():
    assert euclidean_distance((0.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)


def test_euclidean_distance_diagonal():
    assert euclidean_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# palm_width
# ---------------------------------------------------------------------------

def test_palm_width_nonzero():
    lms = _make_landmarks()
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.2, 0.0)
    assert palm_width(lms) == pytest.approx(0.2)


def test_palm_width_degenerate_returns_epsilon():
    lms = _make_landmarks()  # All zeros → wrist == index MCP
    width = palm_width(lms)
    assert width > 0.0  # Must not be zero (avoids division by zero).


# ---------------------------------------------------------------------------
# normalised_distance
# ---------------------------------------------------------------------------

def test_normalised_distance_basic():
    result = normalised_distance((0.0, 0.0), (0.3, 0.0), reference=0.1)
    assert result == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# fingertip_wrist_ratio
# ---------------------------------------------------------------------------

def test_fingertip_wrist_ratio_open_hand_is_high():
    lms = _open_hand_landmarks()
    ratio = fingertip_wrist_ratio(lms)
    assert ratio > 2.0, f"Expected ratio > 2.0 for open hand, got {ratio:.3f}"


def test_fingertip_wrist_ratio_closed_fist_is_low():
    lms = _closed_fist_landmarks()
    ratio = fingertip_wrist_ratio(lms)
    assert ratio < 1.0, f"Expected ratio < 1.0 for closed fist, got {ratio:.3f}"


# ---------------------------------------------------------------------------
# pinch_distance
# ---------------------------------------------------------------------------

def test_pinch_distance_fully_pinched():
    lms = _make_landmarks()
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.1, 0.0)
    lms[THUMB_TIP] = (0.2, 0.2)
    lms[INDEX_TIP] = (0.2, 0.2)  # Same position → pinch gap = 0
    assert pinch_distance(lms) == pytest.approx(0.0)


def test_pinch_distance_open():
    lms = _make_landmarks()
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.1, 0.0)
    lms[THUMB_TIP] = (0.0, 0.0)
    lms[INDEX_TIP] = (0.3, 0.0)  # Far apart
    gap = pinch_distance(lms)
    assert gap > 1.0


# ---------------------------------------------------------------------------
# wrist_x
# ---------------------------------------------------------------------------

def test_wrist_x():
    lms = _make_landmarks()
    lms[WRIST] = (0.42, 0.7)
    assert wrist_x(lms) == pytest.approx(0.42)
