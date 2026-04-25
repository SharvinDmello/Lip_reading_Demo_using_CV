"""
Video Capture Module
Handles webcam and video file input with robust error handling.
Provides frame buffering and quality checks.
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import Optional, Tuple, Generator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VideoSource:
    """
    Unified video source interface for webcam and video files.
    Provides thread-safe frame access with buffering.
    """

    def __init__(
        self,
        source: str = "webcam",
        device_id: int = 0,
        video_path: Optional[str] = None,
        target_fps: float = 30.0,
        buffer_size: int = 3,
        width: int = 640,
        height: int = 480
    ):
        """
        Initialize video source.

        Args:
            source: "webcam" or "video"
            device_id: Camera device ID (for webcam)
            video_path: Path to video file (for video source)
            target_fps: Target framerate
            buffer_size: Frame buffer size
            width: Capture width
            height: Capture height
        """
        self.source = source
        self.device_id = device_id
        self.video_path = video_path
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._start_time: float = 0.0
        self._actual_fps: float = 0.0
        self._fps_samples = []

    def open(self) -> bool:
        """
        Open video source.

        Returns:
            True if opened successfully
        """
        if self.source == "webcam":
            return self._open_webcam()
        elif self.source == "video":
            return self._open_video()
        else:
            logger.error(f"Unknown source type: {self.source}")
            return False

    def _open_webcam(self) -> bool:
        """Open webcam with multiple backend attempts."""
        backends = [
            cv2.CAP_V4L2,
            cv2.CAP_GSTREAMER,
            cv2.CAP_ANY
        ]

        for backend in backends:
            try:
                cap = cv2.VideoCapture(self.device_id, backend)
                if cap.isOpened():
                    # Configure camera
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

                    # Test read
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get actual resolution
                        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)

                        self._cap = cap
                        logger.info(
                            f"Webcam opened: {self.width}x{self.height} @ {actual_fps:.1f}fps "
                            f"(backend: {backend})"
                        )
                        return True
                    else:
                        cap.release()

            except Exception as e:
                logger.debug(f"Backend {backend} failed: {e}")
                continue

        # Try fallback device IDs
        for dev_id in range(4):
            if dev_id == self.device_id:
                continue
            try:
                cap = cv2.VideoCapture(dev_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.device_id = dev_id
                        self._cap = cap
                        logger.info(f"Webcam opened on fallback device {dev_id}")
                        return True
                    cap.release()
            except Exception:
                continue

        logger.error(
            "Failed to open any webcam. "
            "Please check camera connection and permissions."
        )
        return False

    def _open_video(self) -> bool:
        """Open video file."""
        if self.video_path is None:
            logger.error("No video path provided")
            return False

        path = Path(self.video_path)
        if not path.exists():
            logger.error(f"Video file not found: {self.video_path}")
            return False

        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {self.video_path}")
                return False

            # Get video properties
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.target_fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self._cap = cap
            logger.info(
                f"Video opened: {self.video_path} "
                f"({self.width}x{self.height} @ {self.target_fps:.1f}fps, "
                f"{self.total_frames} frames)"
            )
            return True

        except Exception as e:
            logger.error(f"Video open failed: {e}")
            return False

    def start_capture(self):
        """Start background capture thread."""
        if self._cap is None:
            raise RuntimeError("Video source not opened")

        self._running = True
        self._start_time = time.perf_counter()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CaptureThread"
        )
        self._capture_thread.start()
        logger.info("Capture thread started")

    def _capture_loop(self):
        """Background frame capture loop."""
        frame_interval = 1.0 / self.target_fps
        last_frame_time = 0.0

        while self._running and self._cap is not None:
            current_time = time.perf_counter()

            ret, frame = self._cap.read()

            if not ret:
                if self.source == "video":
                    # Loop video file
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.info("Video looped")
                    continue
                else:
                    logger.warning("Webcam read failed")
                    time.sleep(0.01)
                    continue

            # Calculate FPS
            elapsed = current_time - last_frame_time
            if elapsed > 0 and last_frame_time > 0:
                self._fps_samples.append(1.0 / elapsed)
                if len(self._fps_samples) > 30:
                    self._fps_samples.pop(0)
                self._actual_fps = np.mean(self._fps_samples)

            last_frame_time = current_time
            self._frame_count += 1

            # Add to queue (drop oldest if full)
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.

        Returns:
            Tuple of (success, frame)
        """
        try:
            frame = self._frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            if self._running:
                # Try direct read as fallback
                if self._cap is not None and self._cap.isOpened():
                    ret, frame = self._cap.read()
                    if ret:
                        return True, frame
            return False, None

    def read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Direct read without buffering (lower latency)."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames.

        Usage:
            for frame in source.frames():
                process(frame)
        """
        while self._running:
            ret, frame = self.read()
            if ret and frame is not None:
                yield frame
            elif not self._running:
                break

    def get_fps(self) -> float:
        """Get actual capture FPS."""
        return self._actual_fps if self._actual_fps > 0 else self.target_fps

    def get_frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count

    def get_resolution(self) -> Tuple[int, int]:
        """Get frame resolution (width, height)."""
        return (self.width, self.height)

    def is_opened(self) -> bool:
        """Check if source is open."""
        return self._cap is not None and self._cap.isOpened()

    def stop(self):
        """Stop capture and release resources."""
        self._running = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info("VideoSource stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class VideoWriter:
    """Write processed video frames to output file."""

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        width: int = 640,
        height: int = 480
    ):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self) -> bool:
        """Open video writer."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        if not self._writer.isOpened():
            # Try avi format
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            path = self.output_path.replace(".mp4", ".avi")
            self._writer = cv2.VideoWriter(
                path,
                fourcc,
                self.fps,
                (self.width, self.height)
            )

        return self._writer.isOpened()

    def write(self, frame: np.ndarray) -> bool:
        """Write frame to output."""
        if self._writer is None:
            return False
        self._writer.write(frame)
        return True

    def release(self):
        """Release writer."""
        if self._writer is not None:
            self._writer.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()