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
* ``extract_with_drawing`` caches the raw MediaPipe result from the same
  ``process()`` call — MediaPipe is invoked exactly once per frame regardless
  of whether drawing is requested.
* At most ``MP_MAX_HANDS`` hands are tracked (default 1) to keep CPU usage
  bounded (NASA Rule 2).

Migration note (mediapipe >= 0.10.21)
--------------------------------------
The legacy ``mp.solutions`` API was removed in mediapipe 0.10.21.  This
module now uses the MediaPipe Tasks API (``mp.tasks.vision.HandLandmarker``).
The model bundle is downloaded automatically from Google's CDN on first run
and cached at ``~/.cache/gestureify/hand_landmarker.task``.  Set the
environment variable ``MP_MODEL_PATH`` to override the cache location.

Drawing is implemented manually with OpenCV using the same 21-connection
topology that the old ``mp.solutions.hands.HAND_CONNECTIONS`` constant
provided, so the HUD skeleton looks identical to the previous version.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from gestureify.config import settings
from gestureify.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias: a list of 21 (x, y) normalised landmark points.
LandmarkList = List[Tuple[float, float]]

# ---------------------------------------------------------------------------
# Model asset
# ---------------------------------------------------------------------------

#: Public CDN URL for the hand-landmarker model bundle.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

#: Default local cache path (overridable via MP_MODEL_PATH env var).
_DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "gestureify" / "hand_landmarker.task"


def _resolve_model_path() -> str:
    """Return the local path to the hand-landmarker model, downloading if needed."""
    env_override = os.environ.get("MP_MODEL_PATH", "")
    if env_override:
        path = Path(env_override)
        if not path.is_file():
            raise FileNotFoundError(
                f"MP_MODEL_PATH is set but the file does not exist: {path}"
            )
        logger.debug("Using model from MP_MODEL_PATH: %s", path)
        return str(path)

    cache_path = _DEFAULT_MODEL_CACHE
    if not cache_path.is_file():
        logger.info(
            "Downloading hand-landmarker model to %s (one-time setup)…", cache_path
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, cache_path)  # noqa: S310
        logger.info("Model downloaded successfully.")
    else:
        logger.debug("Using cached model at %s", cache_path)

    return str(cache_path)


# ---------------------------------------------------------------------------
# Hand connection topology (mirrors the old mp.solutions.hands.HAND_CONNECTIONS)
# ---------------------------------------------------------------------------
# Each tuple is a (start_landmark_index, end_landmark_index) pair.
_HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
)

_SKELETON_COLOUR = (0, 255, 0)   # BGR green — same as old drawing_utils default
_LANDMARK_COLOUR = (255, 0, 0)   # BGR blue dot at each joint
_LANDMARK_RADIUS = 4
_CONNECTION_THICKNESS = 2


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
        self._landmarker: Optional[mp.tasks.vision.HandLandmarker] = None  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LandmarkExtractor":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Initialise the MediaPipe HandLandmarker (Tasks API)."""
        model_path = _resolve_model_path()

        BaseOptions = mp.tasks.BaseOptions  # type: ignore[attr-defined]
        HandLandmarker = mp.tasks.vision.HandLandmarker  # type: ignore[attr-defined]
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions  # type: ignore[attr-defined]
        VisionRunningMode = mp.tasks.vision.RunningMode  # type: ignore[attr-defined]

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._detection_confidence,
            min_hand_presence_confidence=self._detection_confidence,
            min_tracking_confidence=self._tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms: int = 0

        logger.info(
            "MediaPipe HandLandmarker initialised (max_hands=%d, det=%.2f, track=%.2f).",
            self._max_hands,
            self._detection_confidence,
            self._tracking_confidence,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("MediaPipe HandLandmarker closed.")

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
        landmarks, _ = self._process(bgr_frame)
        return landmarks

    def extract_with_drawing(
        self,
        bgr_frame: np.ndarray,
    ) -> Tuple[Optional[LandmarkList], np.ndarray]:
        """Extract landmarks and draw the hand skeleton onto a copy of the frame.

        MediaPipe is invoked exactly once — the raw result is reused for
        both landmark extraction and skeleton drawing.

        Parameters
        ----------
        bgr_frame:
            Input BGR frame.

        Returns
        -------
        tuple[LandmarkList | None, numpy.ndarray]
            Landmark list (or *None*) and an annotated BGR frame copy.
        """
        landmarks, raw_landmarks = self._process(bgr_frame)
        annotated = bgr_frame.copy()

        if raw_landmarks is not None:
            h, w = annotated.shape[:2]
            # Draw connections
            for start_idx, end_idx in _HAND_CONNECTIONS:
                x0 = int(raw_landmarks[start_idx].x * w)
                y0 = int(raw_landmarks[start_idx].y * h)
                x1 = int(raw_landmarks[end_idx].x * w)
                y1 = int(raw_landmarks[end_idx].y * h)
                cv2.line(annotated, (x0, y0), (x1, y1), _SKELETON_COLOUR, _CONNECTION_THICKNESS)
            # Draw landmark dots
            for lm in raw_landmarks:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(annotated, (cx, cy), _LANDMARK_RADIUS, _LANDMARK_COLOUR, -1)

        return landmarks, annotated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process(
        self, bgr_frame: np.ndarray
    ) -> Tuple[Optional[LandmarkList], object]:
        """Run MediaPipe once and return both the landmark list and raw landmarks.

        Returns
        -------
        tuple[LandmarkList | None, NormalizedLandmarkList | None]
            Parsed landmark list and the raw MediaPipe landmark object (for
            drawing utilities).  Both are *None* on failure or no detection.
        """
        if self._landmarker is None:
            logger.error("_process() called before open().")
            return None, None

        # Tasks API requires RGB input wrapped in mp.Image.
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(  # type: ignore[attr-defined]
            image_format=mp.ImageFormat.SRGB,  # type: ignore[attr-defined]
            data=rgb,
        )

        # VIDEO mode requires a monotonically increasing timestamp in ms.
        self._frame_timestamp_ms += 1
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.hand_landmarks:
            return None, None

        raw_landmarks = result.hand_landmarks[0]
        landmark_list: LandmarkList = [(lm.x, lm.y) for lm in raw_landmarks]
        return landmark_list, raw_landmarks
