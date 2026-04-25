"""
Preprocessing Pipeline Module
Handles frame normalization, sequence buffering, and tensor preparation
for the lip reading model.
"""

import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)


class FrameNormalizer:
    """
    Normalizes lip crop frames for model input.
    Applies ImageNet normalization or grayscale processing.
    """

    # ImageNet normalization constants
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        target_size: Tuple[int, int] = (112, 112),
        normalize: str = "imagenet",  # "imagenet", "zero_one", "neg_one_one"
        use_grayscale: bool = False
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.use_grayscale = use_grayscale

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Normalized float32 array (C, H, W)
        """
        if frame is None or frame.size == 0:
            channels = 1 if self.use_grayscale else 3
            return np.zeros((channels, *self.target_size), dtype=np.float32)

        # Ensure correct size
        if frame.shape[:2] != self.target_size[::-1]:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        if self.use_grayscale:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = gray.astype(np.float32)

            if self.normalize == "imagenet":
                img = img / 255.0
                img = (img - 0.5) / 0.5
            elif self.normalize == "zero_one":
                img = img / 255.0
            elif self.normalize == "neg_one_one":
                img = (img / 127.5) - 1.0

            return img[np.newaxis, :, :]  # (1, H, W)

        else:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = rgb.astype(np.float32) / 255.0

            if self.normalize == "imagenet":
                img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
            elif self.normalize == "neg_one_one":
                img = (img * 2.0) - 1.0
            # zero_one: already done

            # HWC -> CHW
            return img.transpose(2, 0, 1)  # (3, H, W)

    def process_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Process a batch of frames.

        Args:
            frames: List of BGR images

        Returns:
            Float32 array (N, C, H, W)
        """
        processed = [self.process_frame(f) for f in frames]
        return np.stack(processed, axis=0)


class SequenceBuffer:
    """
    Manages a sliding window buffer of lip frames for temporal modeling.

    The buffer maintains a fixed-length sequence of the most recent frames,
    suitable for feeding into LSTM/TCN models.
    """

    def __init__(
        self,
        sequence_length: int = 75,
        stride: int = 1,
        min_frames_for_inference: int = 10
    ):
        """
        Initialize sequence buffer.

        Args:
            sequence_length: Number of frames in each sequence
            stride: Step size for sliding window
            min_frames_for_inference: Minimum frames before running inference
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.min_frames_for_inference = min_frames_for_inference

        self._buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)
        self._frame_count = 0

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a preprocessed frame to the buffer."""
        self._buffer.append(frame.copy())
        self._frame_count += 1

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get current sequence from buffer.

        Returns:
            Array of shape (sequence_length, C, H, W) or None if not ready
        """
        if len(self._buffer) < self.min_frames_for_inference:
            return None

        # Pad with first frame if buffer not full
        frames = list(self._buffer)
        if len(frames) < self.sequence_length:
            pad_count = self.sequence_length - len(frames)
            padding = [frames[0]] * pad_count
            frames = padding + frames

        return np.stack(frames, axis=0)  # (T, C, H, W)

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for inference."""
        return len(self._buffer) >= self.min_frames_for_inference

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self._buffer) == self.sequence_length

    def clear(self):
        """Clear the buffer."""
        self._buffer.clear()
        self._frame_count = 0

    @property
    def current_length(self) -> int:
        """Current number of frames in buffer."""
        return len(self._buffer)

    @property
    def fill_ratio(self) -> float:
        """How full the buffer is (0.0 to 1.0)."""
        return len(self._buffer) / self.sequence_length


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline combining normalization and buffering.

    Provides the main interface for the inference engine.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (112, 112),
        sequence_length: int = 75,
        min_frames: int = 15,
        normalize: str = "imagenet",
        use_grayscale: bool = False,
        stride: int = 1
    ):
        """
        Initialize preprocessing pipeline.

        Args:
            target_size: Lip crop target size (W, H)
            sequence_length: Frames per sequence
            min_frames: Minimum frames before inference
            normalize: Normalization strategy
            use_grayscale: Use grayscale frames
            stride: Sliding window stride
        """
        self.target_size = target_size
        self.sequence_length = sequence_length

        self.normalizer = FrameNormalizer(
            target_size=target_size,
            normalize=normalize,
            use_grayscale=use_grayscale
        )

        self.buffer = SequenceBuffer(
            sequence_length=sequence_length,
            stride=stride,
            min_frames_for_inference=min_frames
        )

        self._channels = 1 if use_grayscale else 3

        logger.info(
            f"PreprocessingPipeline initialized: "
            f"size={target_size}, seq_len={sequence_length}, "
            f"channels={self._channels}"
        )

    def process_and_add(self, lip_frame: np.ndarray) -> bool:
        """
        Process frame and add to sequence buffer.

        Args:
            lip_frame: Raw lip crop (BGR)

        Returns:
            True if buffer is ready for inference
        """
        processed = self.normalizer.process_frame(lip_frame)
        self.buffer.add_frame(processed)
        return self.buffer.is_ready()

    def get_model_input(self, device: torch.device) -> Optional[torch.Tensor]:
        """
        Get current sequence as model-ready tensor.

        Args:
            device: Target device (CPU/GPU)

        Returns:
            Tensor of shape (1, T, C, H, W) or None
        """
        sequence = self.buffer.get_sequence()
        if sequence is None:
            return None

        # sequence: (T, C, H, W) -> add batch dim -> (1, T, C, H, W)
        tensor = torch.from_numpy(sequence).float().unsqueeze(0)
        return tensor.to(device)

    def get_fill_status(self) -> Tuple[int, int, float]:
        """
        Get buffer fill status.

        Returns:
            (current_length, max_length, fill_ratio)
        """
        return (
            self.buffer.current_length,
            self.sequence_length,
            self.buffer.fill_ratio
        )

    def reset(self):
        """Reset the preprocessing pipeline."""
        self.buffer.clear()

    def apply_augmentation(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply mild augmentation for more robust inference.
        Used during sliding window to handle slight variations.
        """
        # Random horizontal flip (for symmetric lip movements)
        # NOTE: Disabled during inference to maintain consistency
        # Random brightness variation
        if np.random.random() > 0.8:
            delta = np.random.uniform(-10, 10)
            frame = np.clip(frame.astype(np.float32) + delta, 0, 255).astype(np.uint8)
        return frame

    @property
    def is_ready(self) -> bool:
        """Check if ready for inference."""
        return self.buffer.is_ready()

    @property
    def channels(self) -> int:
        """Number of input channels."""
        return self._channels


def create_dummy_sequence(
    batch_size: int = 1,
    seq_len: int = 75,
    channels: int = 3,
    h: int = 112,
    w: int = 112,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Create a dummy sequence tensor for testing."""
    return torch.zeros(batch_size, seq_len, channels, h, w, device=device)