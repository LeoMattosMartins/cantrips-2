"""
tests/test_timing_and_state.py
===============================
Unit tests for:
  * gestureify.utils.timing  (RateGate, Stopwatch)
  * gestureify.cv_engine.session_gate  (SessionGate)
  * gestureify.cv_engine.swipe_detector  (SwipeDetector)
"""

import time
import pytest

from gestureify.utils.timing import RateGate, Stopwatch
from gestureify.cv_engine.gesture_classifier import GestureLabel
from gestureify.cv_engine.session_gate import SessionGate, SessionState
from gestureify.cv_engine.swipe_detector import SwipeDetector, SwipeDirection


# ===========================================================================
# RateGate
# ===========================================================================

class TestRateGate:
    def test_first_call_always_allowed(self):
        gate = RateGate(min_interval=10.0)
        assert gate.allow() is True

    def test_second_call_blocked_within_interval(self):
        gate = RateGate(min_interval=10.0)
        gate.allow()
        assert gate.allow() is False

    def test_call_allowed_after_interval(self):
        gate = RateGate(min_interval=0.01)
        gate.allow()
        time.sleep(0.02)
        assert gate.allow() is True

    def test_reset_opens_gate(self):
        gate = RateGate(min_interval=10.0)
        gate.allow()
        gate.reset()
        assert gate.allow() is True

    def test_negative_interval_raises(self):
        with pytest.raises(ValueError):
            RateGate(min_interval=-1.0)


# ===========================================================================
# Stopwatch
# ===========================================================================

class TestStopwatch:
    def test_elapsed_before_start_is_zero(self):
        sw = Stopwatch()
        assert sw.elapsed() == pytest.approx(0.0)

    def test_is_running_false_before_start(self):
        sw = Stopwatch()
        assert sw.is_running is False

    def test_is_running_true_after_start(self):
        sw = Stopwatch()
        sw.start()
        assert sw.is_running is True

    def test_elapsed_increases_over_time(self):
        sw = Stopwatch()
        sw.start()
        time.sleep(0.05)
        assert sw.elapsed() >= 0.04

    def test_reset_clears_state(self):
        sw = Stopwatch()
        sw.start()
        sw.reset()
        assert sw.is_running is False
        assert sw.elapsed() == pytest.approx(0.0)


# ===========================================================================
# SessionGate
# ===========================================================================

class TestSessionGate:
    def _make_gate(self, hold_seconds: float = 0.05) -> SessionGate:
        return SessionGate(hold_seconds=hold_seconds)

    def test_initial_state_is_idle(self):
        gate = self._make_gate()
        assert gate.state is SessionState.IDLE
        assert gate.is_active is False

    def test_non_palm_gesture_does_not_toggle(self):
        gate = self._make_gate()
        toggled = gate.update(GestureLabel.CLOSED_FIST)
        assert toggled is False
        assert gate.state is SessionState.IDLE

    def test_palm_held_long_enough_activates(self):
        gate = self._make_gate(hold_seconds=0.02)
        # Feed OPEN_PALM repeatedly until toggle fires.
        for _ in range(50):
            toggled = gate.update(GestureLabel.OPEN_PALM)
            time.sleep(0.001)
            if toggled:
                break
        assert gate.state is SessionState.ACTIVE

    def test_interrupted_palm_resets_timer(self):
        gate = self._make_gate(hold_seconds=1.0)
        gate.update(GestureLabel.OPEN_PALM)
        gate.update(GestureLabel.NONE)  # Interrupt
        gate.update(GestureLabel.OPEN_PALM)
        # Progress should have restarted; state must still be IDLE.
        assert gate.state is SessionState.IDLE

    def test_second_palm_hold_deactivates(self):
        gate = self._make_gate(hold_seconds=0.02)
        # Activate.
        for _ in range(50):
            if gate.update(GestureLabel.OPEN_PALM):
                break
            time.sleep(0.001)
        assert gate.state is SessionState.ACTIVE
        # Deactivate.
        for _ in range(50):
            if gate.update(GestureLabel.OPEN_PALM):
                break
            time.sleep(0.001)
        assert gate.state is SessionState.IDLE

    def test_hold_progress_zero_when_not_holding(self):
        gate = self._make_gate()
        assert gate.hold_progress() == pytest.approx(0.0)

    def test_hold_progress_increases(self):
        gate = self._make_gate(hold_seconds=1.0)
        gate.update(GestureLabel.OPEN_PALM)
        time.sleep(0.1)
        gate.update(GestureLabel.OPEN_PALM)
        assert gate.hold_progress() > 0.0


# ===========================================================================
# SwipeDetector
# ===========================================================================

class TestSwipeDetector:
    def _make_detector(self) -> SwipeDetector:
        return SwipeDetector(
            velocity_window=5,
            velocity_threshold=0.04,
            cooldown_seconds=0.05,
        )

    def test_no_swipe_on_static_hand(self):
        det = self._make_detector()
        for _ in range(10):
            result = det.update(0.5)
        assert result is None

    def test_right_swipe_detected(self):
        det = self._make_detector()
        # Simulate hand moving left-to-right quickly.
        positions = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = None
        for x in positions:
            result = det.update(x)
        assert result is SwipeDirection.RIGHT

    def test_left_swipe_detected(self):
        det = self._make_detector()
        positions = [0.9, 0.7, 0.5, 0.3, 0.1]
        result = None
        for x in positions:
            result = det.update(x)
        assert result is SwipeDirection.LEFT

    def test_cooldown_prevents_immediate_re_trigger(self):
        det = self._make_detector()
        positions = [0.1, 0.2, 0.3, 0.4, 0.5]
        for x in positions:
            det.update(x)
        # Immediately try another swipe.
        result = det.update(0.9)
        assert result is None
        assert det.in_cooldown is True

    def test_rearm_after_cooldown(self):
        det = self._make_detector()
        positions = [0.1, 0.2, 0.3, 0.4, 0.5]
        for x in positions:
            det.update(x)
        time.sleep(0.06)  # Wait for cooldown to expire.
        # New swipe should be detectable.
        positions2 = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = None
        for x in positions2:
            result = det.update(x)
        assert result is SwipeDirection.RIGHT

    def test_none_input_resets_buffer(self):
        det = self._make_detector()
        det.update(0.1)
        det.update(0.2)
        det.update(None)  # Hand left frame — buffer reset.
        # After reset, need a full window again.
        result = det.update(0.5)
        assert result is None

    def test_reset_clears_everything(self):
        det = self._make_detector()
        for x in [0.1, 0.2, 0.3, 0.4, 0.5]:
            det.update(x)
        det.reset()
        assert det.in_cooldown is False
        result = det.update(0.5)
        assert result is None

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            SwipeDetector(velocity_window=1)
