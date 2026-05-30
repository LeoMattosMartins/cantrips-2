"""
tests/test_classifier.py
========================
Unit tests for ``gestureify.cv_engine.gesture_classifier``.

Covers:
  * ``GestureLabel`` enum stability.
  * ``GestureClassifier.classify_full()`` — the primary hot-path method.
  * ``GestureClassifier.classify()`` — convenience wrapper.
  * ``GestureClassifier.pinch_gap()`` — standalone gap query.
  * ``ClassifyResult`` dataclass contract.

All tests are pure (no I/O, no camera, no MediaPipe model).
"""

from __future__ import annotations

import pytest

from gestureify.cv_engine.gesture_classifier import (
    ClassifyResult,
    GestureClassifier,
    GestureLabel,
)
from gestureify.utils.geometry import INDEX_MCP, INDEX_TIP, THUMB_TIP, WRIST


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_landmarks(n: int = 21) -> list:
    return [(0.0, 0.0)] * n


def _open_palm_landmarks() -> list:
    lms = _make_landmarks()
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.1, 0.0)
    for idx in (8, 12, 16, 20):          # four fingertips (no thumb)
        lms[idx] = (0.0, 0.4)
    lms[THUMB_TIP] = (0.3, 0.3)         # thumb far from index → no pinch
    lms[INDEX_TIP] = (0.0, 0.4)
    return lms


def _closed_fist_landmarks() -> list:
    lms = _make_landmarks()
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.1, 0.0)
    # All fingertips close to wrist — but thumb and index must NOT coincide
    # or the pinch classifier fires first (gap = 0 < threshold).
    lms[THUMB_TIP] = (-0.05, 0.05)   # thumb curled inward, away from index
    lms[INDEX_TIP] = (0.05, 0.05)    # index curled, not touching thumb
    for idx in (12, 16, 20):
        lms[idx] = (0.0, 0.05)
    return lms


def _pinch_landmarks() -> list:
    lms = _make_landmarks()
    lms[WRIST] = (0.0, 0.0)
    lms[INDEX_MCP] = (0.1, 0.0)
    lms[THUMB_TIP] = (0.05, 0.05)
    lms[INDEX_TIP] = (0.05, 0.05)       # coincident → gap = 0
    for idx in (12, 16, 20):
        lms[idx] = (0.0, 0.4)
    return lms


# ---------------------------------------------------------------------------
# GestureLabel enum
# ---------------------------------------------------------------------------


class TestGestureLabel:
    def test_values_are_stable(self):
        assert GestureLabel.NONE == 0
        assert GestureLabel.OPEN_PALM == 1
        assert GestureLabel.CLOSED_FIST == 2
        assert GestureLabel.PINCH == 3

    def test_identity_comparison(self):
        assert GestureLabel.NONE is GestureLabel.NONE
        assert GestureLabel.OPEN_PALM is not GestureLabel.CLOSED_FIST


# ---------------------------------------------------------------------------
# ClassifyResult
# ---------------------------------------------------------------------------


class TestClassifyResult:
    def test_slots_present(self):
        r = ClassifyResult(label=GestureLabel.NONE, pinch_gap=1.0)
        assert not hasattr(r, "__dict__")

    def test_fields_accessible(self):
        r = ClassifyResult(label=GestureLabel.PINCH, pinch_gap=0.05)
        assert r.label is GestureLabel.PINCH
        assert r.pinch_gap == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# GestureClassifier.classify_full()
# ---------------------------------------------------------------------------


class TestClassifyFull:
    def setup_method(self):
        self.clf = GestureClassifier(
            fist_threshold=0.6,
            palm_threshold=1.8,
            pinch_threshold=0.12,
        )

    def test_none_landmarks_returns_none_label(self):
        result = self.clf.classify_full(None)
        assert result.label is GestureLabel.NONE
        assert result.pinch_gap == pytest.approx(1.0)

    def test_open_palm_detected(self):
        result = self.clf.classify_full(_open_palm_landmarks())
        assert result.label is GestureLabel.OPEN_PALM

    def test_closed_fist_detected(self):
        result = self.clf.classify_full(_closed_fist_landmarks())
        assert result.label is GestureLabel.CLOSED_FIST

    def test_pinch_detected(self):
        result = self.clf.classify_full(_pinch_landmarks())
        assert result.label is GestureLabel.PINCH

    def test_pinch_gap_is_precomputed_in_result(self):
        """classify_full() must return pinch_gap even for non-PINCH gestures."""
        result = self.clf.classify_full(_open_palm_landmarks())
        assert isinstance(result.pinch_gap, float)
        assert result.pinch_gap >= 0.0

    def test_pinch_takes_priority_over_palm(self):
        """A pinch should be returned even if fingertip ratio is palm-like."""
        result = self.clf.classify_full(_pinch_landmarks())
        assert result.label is GestureLabel.PINCH


# ---------------------------------------------------------------------------
# GestureClassifier.classify() — convenience wrapper
# ---------------------------------------------------------------------------


class TestClassify:
    def setup_method(self):
        self.clf = GestureClassifier()

    def test_returns_label_only(self):
        label = self.clf.classify(None)
        assert isinstance(label, GestureLabel)

    def test_consistent_with_classify_full(self):
        lms = _open_palm_landmarks()
        assert self.clf.classify(lms) is self.clf.classify_full(lms).label


# ---------------------------------------------------------------------------
# GestureClassifier.pinch_gap()
# ---------------------------------------------------------------------------


class TestPinchGap:
    def setup_method(self):
        self.clf = GestureClassifier()

    def test_none_returns_neutral(self):
        assert self.clf.pinch_gap(None) == pytest.approx(1.0)

    def test_pinched_returns_near_zero(self):
        lms = _pinch_landmarks()
        gap = self.clf.pinch_gap(lms)
        assert gap < 0.1

    def test_open_returns_large(self):
        lms = _open_palm_landmarks()
        gap = self.clf.pinch_gap(lms)
        assert gap > 0.5
