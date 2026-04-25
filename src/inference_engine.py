"""
Inference Engine Module
Orchestrates the complete lip reading pipeline:
1. Receives lip crops from extractor
2. Preprocesses and buffers frames
3. Runs model inference
4. Decodes predictions
5. Returns text output
"""

import torch
import numpy as np
import time
import threading
import queue
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, field

from .model import LipReadingModel, ModelLoader
from .preprocessor import PreprocessingPipeline
from .decoder import CTCDecoder, TextPostProcessor, SlidingWindowAggregator, DecodingResult
from .lip_extractor import LipRegion

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Complete result from one inference pass."""
    text: str
    confidence: float
    stable_text: str
    processing_time_ms: float
    buffer_fill: float
    face_detected: bool
    frame_count: int = 0
    raw_result: Optional[DecodingResult] = None


class InferenceEngine:
    """
    Main inference engine for real-time lip reading.

    Manages the complete pipeline from lip crops to text output,
    including frame buffering, batching, and result aggregation.
    """

    def __init__(
        self,
        weights_dir: Path = Path("weights"),
        device: Optional[str] = None,
        sequence_length: int = 75,
        min_frames: int = 15,
        inference_interval: int = 8,    # Run inference every N frames
        use_beam_search: bool = False,
        beam_width: int = 5,
        target_size: tuple = (112, 112)
    ):
        """
        Initialize inference engine.

        Args:
            weights_dir: Directory containing model weights
            device: Compute device ('cpu', 'cuda', 'cuda:0', or None for auto)
            sequence_length: Number of frames per inference window
            min_frames: Minimum frames before starting inference
            inference_interval: Run inference every N frames
            use_beam_search: Use beam search instead of greedy decoding
            beam_width: Beam search width
            target_size: Lip crop target size
        """
        self.weights_dir = weights_dir
        self.sequence_length = sequence_length
        self.min_frames = min_frames
        self.inference_interval = inference_interval
        self.use_beam_search = use_beam_search
        self.target_size = target_size

        # Setup device
        self.device = self._setup_device(device)
        logger.info(f"Using device: {self.device}")

        # Load model and vocabulary
        self._load_model()

        # Setup preprocessing
        self.preprocessor = PreprocessingPipeline(
            target_size=target_size,
            sequence_length=sequence_length,
            min_frames=min_frames,
            normalize="imagenet",
            use_grayscale=False
        )

        # Setup decoder
        self.decoder = CTCDecoder(
            vocabulary=self.vocabulary,
            blank_index=0,
            beam_width=beam_width
        )

        # Setup post-processor
        self.post_processor = TextPostProcessor()

        # Setup aggregator for stable predictions
        self.aggregator = SlidingWindowAggregator(
            window_size=5,
            min_agreement=0.3
        )

        # Inference state
        self._frame_counter = 0
        self._last_result: Optional[InferenceResult] = None
        self._inference_count = 0

        # Performance tracking
        self._inference_times = []
        self._max_time_samples = 100

        logger.info(
            f"InferenceEngine ready: "
            f"seq_len={sequence_length}, "
            f"interval={inference_interval}, "
            f"beam_search={use_beam_search}"
        )

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup compute device with fallback."""
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            # Check VRAM (GTX 1650 has 4GB)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
            return torch.device("cuda:0")
        else:
            logger.info("CUDA not available, using CPU")
            return torch.device("cpu")

    def _load_model(self):
        """Load model and vocabulary."""
        loader = ModelLoader(weights_dir=self.weights_dir)

        try:
            self.model, self.vocabulary = loader.load(device=self.device)
            logger.info("Model loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def process_lip_region(
        self,
        lip_region: Optional[LipRegion]
    ) -> Optional[InferenceResult]:
        """
        Process a lip region and optionally run inference.

        Args:
            lip_region: Extracted lip region from LipExtractor

        Returns:
            InferenceResult if inference was run, else None
        """
        self._frame_counter += 1

        # Handle missing face
        if lip_region is None or not lip_region.face_detected:
            # Add black frame to maintain sequence continuity
            black_frame = np.zeros(
                (*self.target_size[::-1], 3),
                dtype=np.uint8
            )
            self.preprocessor.process_and_add(black_frame)

            if self._last_result is not None:
                # Return last result with face_detected=False
                return InferenceResult(
                    text=self._last_result.text,
                    confidence=self._last_result.confidence,
                    stable_text=self._last_result.stable_text,
                    processing_time_ms=0.0,
                    buffer_fill=self.preprocessor.buffer.fill_ratio,
                    face_detected=False,
                    frame_count=self._frame_counter
                )
            return None

        # Add frame to preprocessing pipeline
        ready = self.preprocessor.process_and_add(lip_region.crop)

        # Run inference at regular intervals
        should_infer = (
            ready and
            self._frame_counter % self.inference_interval == 0
        )

        if not should_infer:
            return self._last_result

        # Run inference
        result = self._run_inference()
        self._last_result = result
        return result

    def _run_inference(self) -> InferenceResult:
        """Run model inference on current buffer."""
        start_time = time.perf_counter()

        # Get model input tensor
        input_tensor = self.preprocessor.get_model_input(self.device)

        if input_tensor is None:
            return InferenceResult(
                text="",
                confidence=0.0,
                stable_text="",
                processing_time_ms=0.0,
                buffer_fill=self.preprocessor.buffer.fill_ratio,
                face_detected=True,
                frame_count=self._frame_counter
            )

        # Forward pass
        with torch.no_grad():
            try:
                outputs = self.model(input_tensor, return_features=False)
                log_probs = outputs["log_probs"]  # (T, B, vocab_size)

                # Decode
                if self.use_beam_search:
                    decode_results = self.decoder.decode_batch(log_probs, method="beam_search")
                else:
                    decode_results = self.decoder.decode_batch(log_probs, method="greedy")

                raw_result = decode_results[0]  # First (and only) in batch

            except Exception as e:
                logger.error(f"Inference error: {e}")
                raw_result = DecodingResult(
                    text="",
                    confidence=0.0,
                    char_probs=[],
                    raw_indices=[]
                )

        # Post-process
        processed = self.post_processor.process(raw_result)

        # Aggregate for stability
        stable_text = self.aggregator.add_result(processed)

        # Track timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(elapsed_ms)
        if len(self._inference_times) > self._max_time_samples:
            self._inference_times.pop(0)

        self._inference_count += 1

        current_len, max_len, fill_ratio = self.preprocessor.get_fill_status()

        return InferenceResult(
            text=processed.text,
            confidence=processed.confidence,
            stable_text=stable_text,
            processing_time_ms=elapsed_ms,
            buffer_fill=fill_ratio,
            face_detected=True,
            frame_count=self._frame_counter,
            raw_result=raw_result
        )

    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics."""
        if not self._inference_times:
            return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "fps": 0.0}

        times = self._inference_times
        avg = np.mean(times)
        return {
            "avg_ms": float(avg),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": float(1000.0 / avg) if avg > 0 else 0.0,
            "inference_count": self._inference_count
        }

    def reset(self):
        """Reset inference engine state."""
        self.preprocessor.reset()
        self.aggregator.reset()
        self._frame_counter = 0
        self._last_result = None

    def save_text_output(self, output_path: Path, text: str, metadata: Dict = None):
        """Save predicted text to file."""
        with open(output_path, "a", encoding="utf-8") as f:
            timestamp = time.strftime("%H:%M:%S")
            line = f"[{timestamp}] {text}"
            if metadata:
                line += f" (conf: {metadata.get('confidence', 0):.2f})"
            f.write(line + "\n")


class AsyncInferenceEngine:
    """
    Asynchronous wrapper for InferenceEngine.
    Runs inference in a separate thread to avoid blocking the main video loop.
    """

    def __init__(self, engine: InferenceEngine, queue_size: int = 2):
        self.engine = engine
        self._input_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._result_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_result: Optional[InferenceResult] = None

    def start(self):
        """Start async inference thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="InferenceThread"
        )
        self._thread.start()
        logger.info("Async inference engine started")

    def stop(self):
        """Stop async inference thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("Async inference engine stopped")

    def submit_lip_region(self, lip_region: Optional[LipRegion]) -> bool:
        """
        Submit lip region for inference.

        Returns:
            True if submitted successfully
        """
        try:
            self._input_queue.put_nowait(lip_region)
            return True
        except queue.Full:
            return False

    def get_latest_result(self) -> Optional[InferenceResult]:
        """Get the most recent inference result."""
        # Drain queue and keep latest
        while True:
            try:
                self._latest_result = self._result_queue.get_nowait()
            except queue.Empty:
                break
        return self._latest_result

    def _inference_loop(self):
        """Main inference loop running in background thread."""
        while self._running:
            try:
                lip_region = self._input_queue.get(timeout=0.1)
                result = self.engine.process_lip_region(lip_region)

                if result is not None:
                    try:
                        self._result_queue.put_nowait(result)
                    except queue.Full:
                        # Clear old result and add new
                        try:
                            self._result_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._result_queue.put_nowait(result)
                        except queue.Full:
                            pass

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Inference thread error: {e}")