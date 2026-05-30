"""
gestureify.cv_engine.capture
=============================
OpenCV ``VideoCapture`` wrapper with lifecycle management and failure detection.

Design notes
------------
* Implements the context-manager protocol for guaranteed camera release.
* ``frames()`` generator counts consecutive read failures and raises
  ``RuntimeError`` after ``MAX_CONSECUTIVE_FAILURES`` misses, preventing
  a silent infinite CPU-spinning loop on device disconnect (NASA Rule 7).
* Camera properties (width, height, FPS) are set as hints; actual values
  depend on the hardware and driver.
"""

from __future__ import annotations

from typing import Generator, Optional

import cv2
import numpy as np

from gestureify.config import settings
from gestureify.utils.logger import get_logger

logger = get_logger(__name__)

# Hard cap on consecutive frame-read failures before the generator aborts.
_MAX_CONSECUTIVE_FAILURES: int = 30


class CameraCapture:
    """Manage an OpenCV ``VideoCapture`` device.

    Parameters
    ----------
    index:
        Camera device index (0 = default webcam).
    width:
        Requested capture width in pixels.
    height:
        Requested capture height in pixels.
    fps:
        Requested capture frame rate.
    """

    def __init__(
        self,
        index: int = settings.CAMERA_INDEX,
        width: int = settings.CAMERA_WIDTH,
        height: int = settings.CAMERA_HEIGHT,
        fps: int = settings.CAMERA_FPS,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera device.

        Raises
        ------
        RuntimeError
            If the device cannot be opened.
        """
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._index}. "
                "Check that the device is connected and not in use by another application."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap = cap
        logger.info(
            "Camera %d opened (%dx%d @ %d FPS).",
            self._index,
            self._width,
            self._height,
            self._fps,
        )

    def close(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera %d released.", self._index)

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame.

        Returns
        -------
        numpy.ndarray | None
            BGR frame, or *None* if the read failed.
        """
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield frames continuously until the camera is closed or fails.

        Raises
        ------
        RuntimeError
            If the camera has not been opened.
        RuntimeError
            If ``_MAX_CONSECUTIVE_FAILURES`` consecutive reads fail, which
            indicates a device disconnect or driver error.

        Yields
        ------
        numpy.ndarray
            BGR frame array.
        """
        if self._cap is None:
            raise RuntimeError("Camera is not open. Call open() first.")

        consecutive_failures = 0

        while self._cap.isOpened():
            frame = self.read_frame()
            if frame is not None:
                consecutive_failures = 0
                yield frame
            else:
                consecutive_failures += 1
                logger.warning(
                    "Camera read failed (consecutive failures: %d / %d).",
                    consecutive_failures,
                    _MAX_CONSECUTIVE_FAILURES,
                )
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"Camera {self._index} failed to produce a frame "
                        f"{_MAX_CONSECUTIVE_FAILURES} times in a row. "
                        "The device may have been disconnected."
                    )

    @property
    def is_open(self) -> bool:
        """``True`` if the camera device is currently open."""
        return self._cap is not None and self._cap.isOpened()
