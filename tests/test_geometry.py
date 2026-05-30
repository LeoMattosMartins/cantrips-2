"""
tests/test_geometry.py
======================
Unit tests for gestureify.utils.geometry.

All tests are pure (no I/O, no side-effects) and run without a camera or
MediaPipe model installed.

Note on thumb exclusion
-----------------------
``fingertip_wrist_ratio`` deliberately excludes the thumb (index 4) because
the thumb has fundamentally different extension geometry.  Test fixtures must
not rely on the thumb to produce a high ratio.
"""

from __future__ import annotations

import pytest

from gestureify.utils.geometry import (
    INDEX_MCP,
    INDEX_TIP,
    THUMB_TIP,
    WRIST,
    euclidean_distance,
    fingertip_wrist_ratio,
    normalised_distance,
    palm_width,
    pinch_distance,
    wrist_x,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_landmarks(n: int = 21) -> list:
    """Return a list of *n* zero-initialised (x, y) landmark tuples."""
    return [(0.0, 0.0)] * n


def _open_hand_landmarks() -> list:
    """Approximate an open hand: four fingertips (no thumb) far from wrist."""
    lms = _make_landmarks()
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.1, 0.0)
    # Only the four non-thumb fingertips (8, 12, 16, 20) are used by the ratio.
    for idx in (8, 12, 16, 20):
        lms[idx] = (0.0, 0.4)
    return lms


def _closed_fist_landmarks() -> list:
    """Approximate a closed fist: fingertips close to wrist."""
    lms = _make_landmarks()
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.1, 0.0)
    for idx in (4, 8, 12, 16, 20):
        lms[idx] = (0.0, 0.05)
    return lms


# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------


def test_euclidean_distance_zero() -> None:
    assert euclidean_distance((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0)


def test_euclidean_distance_unit() -> None:
    assert euclidean_distance((0.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)


def test_euclidean_distance_diagonal() -> None:
    assert euclidean_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# palm_width
# ---------------------------------------------------------------------------


def test_palm_width_nonzero() -> None:
    lms = _make_landmarks()
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.2, 0.0)
    assert palm_width(lms) == pytest.approx(0.2)


def test_palm_width_degenerate_returns_epsilon() -> None:
    lms = _make_landmarks()  # All zeros → wrist == index MCP
    width = palm_width(lms)
    assert width > 0.0  # Must not be zero (avoids division by zero).


# ---------------------------------------------------------------------------
# normalised_distance
# ---------------------------------------------------------------------------


def test_normalised_distance_basic() -> None:
    result = normalised_distance((0.0, 0.0), (0.3, 0.0), reference=0.1)
    assert result == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# fingertip_wrist_ratio  (thumb excluded)
# ---------------------------------------------------------------------------


def test_fingertip_wrist_ratio_open_hand_is_high() -> None:
    lms = _open_hand_landmarks()
    ratio = fingertip_wrist_ratio(lms)
    assert ratio > 2.0, f"Expected ratio > 2.0 for open hand, got {ratio:.3f}"


def test_fingertip_wrist_ratio_closed_fist_is_low() -> None:
    lms = _closed_fist_landmarks()
    ratio = fingertip_wrist_ratio(lms)
    assert ratio < 1.0, f"Expected ratio < 1.0 for closed fist, got {ratio:.3f}"


def test_fingertip_wrist_ratio_excludes_thumb() -> None:
    """Moving only the thumb should not affect the ratio."""
    lms = _closed_fist_landmarks()
    ratio_before = fingertip_wrist_ratio(lms)

    lms_thumb_out = list(lms)
    lms_thumb_out[4] = (0.0, 0.9)   # Thumb extended far out
    ratio_after = fingertip_wrist_ratio(lms_thumb_out)

    assert ratio_before == pytest.approx(ratio_after, rel=1e-4), (
        "Thumb extension should not change the ratio (thumb is excluded)."
    )


# ---------------------------------------------------------------------------
# pinch_distance
# ---------------------------------------------------------------------------


def test_pinch_distance_fully_pinched() -> None:
    lms = _make_landmarks()
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.1, 0.0)
    lms[THUMB_TIP] = (0.2, 0.2)
    lms[INDEX_TIP] = (0.2, 0.2)  # Same position → pinch gap = 0
    assert pinch_distance(lms) == pytest.approx(0.0, abs=1e-5)


def test_pinch_distance_open() -> None:
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


def test_wrist_x() -> None:
    lms = _make_landmarks()
    lms[WRIST] = (0.42, 0.7)
    assert wrist_x(lms) == pytest.approx(0.42)
