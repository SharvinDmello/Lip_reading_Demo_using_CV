"""
Unit tests for the Silent Speech Recognition pipeline.
Run with: python -m pytest tests/ -v
"""

import pytest
import numpy as np
import torch
import cv2
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFaceDetector:
    """Tests for face detection module."""

    def test_import(self):
        from src.face_detector import FaceDetector, FaceDetection
        assert FaceDetector is not None

    def test_init(self):
        from src.face_detector import FaceDetector
        detector = FaceDetector()
        assert detector is not None
        detector.__del__()

    def test_detect_empty_frame(self):
        from src.face_detector import FaceDetector
        detector = FaceDetector()
        result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        # Should return None for empty face frame
        # (may or may not detect depending on black frame)
        assert result is None or hasattr(result, 'bbox')
        detector.__del__()

    def test_detect_none_frame(self):
        from src.face_detector import FaceDetector
        detector = FaceDetector()
        result = detector.detect(None)
        assert result is None
        detector.__del__()


class TestLipExtractor:
    """Tests for lip extraction module."""

    def test_import(self):
        from src.lip_extractor import LipExtractor, LipRegion
        assert LipExtractor is not None

    def test_init(self):
        from src.lip_extractor import LipExtractor
        extractor = LipExtractor(target_size=(112, 112))
        assert extractor is not None
        extractor.__del__()

    def test_extract_black_frame(self):
        from src.lip_extractor import LipExtractor
        extractor = LipExtractor(target_size=(112, 112))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.extract(frame)
        # Should return LipRegion with face_detected=False
        assert result is not None
        assert result.face_detected == False
        extractor.__del__()

    def test_extract_none_frame(self):
        from src.lip_extractor import LipExtractor
        extractor = LipExtractor(target_size=(112, 112))
        result = extractor.extract(None)
        assert result is None
        extractor.__del__()


class TestPreprocessor:
    """Tests for preprocessing pipeline."""

    def test_import(self):
        from src.preprocessor import PreprocessingPipeline, FrameNormalizer, SequenceBuffer
        assert PreprocessingPipeline is not None

    def test_frame_normalizer_rgb(self):
        from src.preprocessor import FrameNormalizer
        normalizer = FrameNormalizer(target_size=(112, 112), normalize="imagenet")
        frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = normalizer.process_frame(frame)
        assert result.shape == (3, 112, 112)
        assert result.dtype == np.float32

    def test_frame_normalizer_grayscale(self):
        from src.preprocessor import FrameNormalizer
        normalizer = FrameNormalizer(target_size=(112, 112), use_grayscale=True)
        frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = normalizer.process_frame(frame)
        assert result.shape == (1, 112, 112)

    def test_sequence_buffer_not_ready(self):
        from src.preprocessor import SequenceBuffer
        buffer = SequenceBuffer(sequence_length=75, min_frames_for_inference=15)
        frame = np.zeros((3, 112, 112), dtype=np.float32)
        for _ in range(5):
            buffer.add_frame(frame)
        assert not buffer.is_ready()
        assert buffer.get_sequence() is None

    def test_sequence_buffer_ready(self):
        from src.preprocessor import SequenceBuffer
        buffer = SequenceBuffer(sequence_length=75, min_frames_for_inference=15)
        frame = np.zeros((3, 112, 112), dtype=np.float32)
        for _ in range(20):
            buffer.add_frame(frame)
        assert buffer.is_ready()
        seq = buffer.get_sequence()
        assert seq is not None
        assert seq.shape == (75, 3, 112, 112)

    def test_pipeline_integration(self):
        from src.preprocessor import PreprocessingPipeline
        pipeline = PreprocessingPipeline(
            sequence_length=30,
            min_frames=10
        )
        frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

        # Add frames until ready
        ready = False
        for i in range(15):
            ready = pipeline.process_and_add(frame)

        assert ready
        tensor = pipeline.get_model_input(torch.device("cpu"))
        assert tensor is not None
        assert tensor.shape == (1, 30, 3, 112, 112)


class TestModel:
    """Tests for lip reading model."""

    def test_import(self):
        from src.model import LipReadingModel, SpatialFeatureExtractor, TemporalEncoder
        assert LipReadingModel is not None

    def test_spatial_encoder_forward(self):
        from src.model import SpatialFeatureExtractor
        encoder = SpatialFeatureExtractor(pretrained=False)
        x = torch.zeros(2, 3, 112, 112)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (2, 576)

    def test_temporal_encoder_forward(self):
        from src.model import TemporalEncoder
        encoder = TemporalEncoder(input_dim=576, hidden_dim=128, num_layers=1)
        x = torch.zeros(2, 30, 576)
        with torch.no_grad():
            out, hidden = encoder(x)
        assert out.shape == (2, 30, 256)  # 128 * 2 bidirectional

    def test_model_forward(self):
        from src.model import LipReadingModel
        model = LipReadingModel(
            vocab_size=68,
            feature_dim=576,
            hidden_dim=128,
            num_lstm_layers=1,
            pretrained_backbone=False,
            use_attention=False
        )
        model.eval()
        x = torch.zeros(1, 30, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        assert "log_probs" in out
        assert out["log_probs"].shape == (30, 1, 68)

    def test_model_parameter_count(self):
        from src.model import LipReadingModel
        model = LipReadingModel(pretrained_backbone=False)
        params = model.count_parameters()
        assert params["total"] > 0
        print(f"\nModel parameters: {params['total']:,}")


class TestDecoder:
    """Tests for CTC decoder."""

    def test_import(self):
        from src.decoder import CTCDecoder, TextPostProcessor, SlidingWindowAggregator
        assert CTCDecoder is not None

    def get_vocab(self):
        return {
            "chars": [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I",
                     "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                     "T", "U", "V", "W", "X", "Y", "Z"]
        }

    def test_greedy_decode(self):
        from src.decoder import CTCDecoder
        vocab = self.get_vocab()
        decoder = CTCDecoder(vocabulary=vocab)

        # Create fake log probs: strong signal at index 5 (char 'D')
        T, V = 20, len(vocab["chars"]) + 1
        log_probs = torch.full((T, V), -10.0)
        log_probs[:, 5] = 0.0  # Strong signal

        result = decoder.greedy_decode(log_probs)
        assert isinstance(result.text, str)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_beam_search_decode(self):
        from src.decoder import CTCDecoder
        vocab = self.get_vocab()
        decoder = CTCDecoder(vocabulary=vocab, beam_width=5)

        T, V = 15, len(vocab["chars"]) + 1
        log_probs = torch.randn(T, V)
        log_probs = torch.log_softmax(log_probs, dim=-1)

        result = decoder.beam_search_decode(log_probs)
        assert isinstance(result.text, str)

    def test_batch_decode(self):
        from src.decoder import CTCDecoder
        vocab = self.get_vocab()
        decoder = CTCDecoder(vocabulary=vocab)

        T, B, V = 20, 3, len(vocab["chars"]) + 1
        log_probs = torch.randn(T, B, V)
        log_probs = torch.log_softmax(log_probs, dim=-1)

        results = decoder.decode_batch(log_probs, method="greedy")
        assert len(results) == B

    def test_post_processor(self):
        from src.decoder import TextPostProcessor, DecodingResult
        processor = TextPostProcessor()
        result = DecodingResult(
            text="  hello  world  ",
            confidence=0.8,
            char_probs=[],
            raw_indices=[]
        )
        processed = processor.process(result)
        assert processed.text == "hello  world"

    def test_aggregator(self):
        from src.decoder import SlidingWindowAggregator, DecodingResult
        agg = SlidingWindowAggregator(window_size=3, min_agreement=0.3)

        result = DecodingResult("hello", 0.9, [], [])
        for _ in range(3):
            stable = agg.add_result(result)

        assert isinstance(stable, str)


class TestEndToEnd:
    """End-to-end integration tests (no weights required)."""

    def test_pipeline_with_random_frames(self):
        """Test complete pipeline with random frames (no face)."""
        from src.preprocessor import PreprocessingPipeline
        from src.model import LipReadingModel
        from src.decoder import CTCDecoder

        # Setup
        pipeline = PreprocessingPipeline(
            sequence_length=30,
            min_frames=10,
            normalize="imagenet"
        )

        model = LipReadingModel(
            vocab_size=28,
            feature_dim=576,
            hidden_dim=64,
            num_lstm_layers=1,
            pretrained_backbone=False,
            use_attention=False
        )
        model.eval()

        vocab = {"chars": list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
        decoder = CTCDecoder(vocabulary=vocab)

        # Feed frames
        for _ in range(15):
            frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            pipeline.process_and_add(frame)

        # Get tensor
        tensor = pipeline.get_model_input(torch.device("cpu"))
        assert tensor is not None

        # Run model
        with torch.no_grad():
            output = model(tensor)

        log_probs = output["log_probs"]
        assert log_probs.shape[1] == 1  # batch size
        assert log_probs.shape[2] == 28  # vocab size

        # Decode
        results = decoder.decode_batch(log_probs, method="greedy")
        assert len(results) == 1
        assert isinstance(results[0].text, str)

        print(f"\nE2E test passed. Decoded: '{results[0].text}'")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])