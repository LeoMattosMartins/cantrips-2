"""
gestureify.cv_engine.capture
============================
Thin wrapper around ``cv2.VideoCapture`` that enforces resource cleanup and
provides a clean iterator interface for the rest of the pipeline.

Design notes
------------
* Implements the context-manager protocol so the camera is always released,
  even on exceptions (NASA Rule 7: handle all error conditions).
* The ``frames()`` generator is bounded: it yields at most while the camera
  is open and the caller has not requested a stop (NASA Rule 2).
* No frame processing occurs here; this module's sole responsibility is
  reliable frame acquisition.
"""

from __future__ import annotations

from typing import Generator, Optional

import cv2
import numpy as np

from gestureify.utils.logger import get_logger

logger = get_logger(__name__)


class CameraCapture:
    """Manage a webcam capture session.

    Parameters
    ----------
    index:
        OpenCV device index (0 = default camera).
    width:
        Requested capture width in pixels.
    height:
        Requested capture height in pixels.
    fps:
        Requested capture frame rate.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
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

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera device and apply the requested resolution/FPS.

        Raises
        ------
        RuntimeError
            If the camera cannot be opened.
        """
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._index}. "
                "Check that the camera is connected and not in use."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap = cap
        logger.info(
            "Camera opened: index=%d  %dx%d @ %d FPS",
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
            logger.info("Camera released.")

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the camera.

        Returns
        -------
        numpy.ndarray | None
            BGR frame array, or *None* if the read failed.
        """
        if self._cap is None:
            logger.error("read_frame() called before open().")
            return None
        ok, frame = self._cap.read()
        if not ok:
            logger.warning("Failed to read frame from camera.")
            return None
        return frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield frames continuously until the camera is closed.

        Yields
        ------
        numpy.ndarray
            BGR frame array.  Frames that fail to read are skipped silently.
        """
        if self._cap is None:
            raise RuntimeError("Camera is not open. Call open() first.")
        while self._cap.isOpened():
            frame = self.read_frame()
            if frame is not None:
                yield frame

    @property
    def is_open(self) -> bool:
        """``True`` if the camera device is currently open."""
        return self._cap is not None and self._cap.isOpened()
