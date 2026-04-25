"""
CTC Decoder Module
Decodes model output (character probability distributions) into text.
Implements greedy decoding and beam search for lip reading output.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DecodingResult:
    """Container for decoding output."""
    text: str
    confidence: float
    char_probs: List[float]
    raw_indices: List[int]


class CTCDecoder:
    """
    CTC Decoder for lip reading output.

    Supports:
    1. Greedy decoding (fastest, good quality)
    2. Beam search decoding (better quality, slower)
    """

    def __init__(
        self,
        vocabulary: Dict,
        blank_index: int = 0,
        space_index: int = 1,
        beam_width: int = 10
    ):
        """
        Initialize CTC decoder.

        Args:
            vocabulary: Dict with 'chars' list
            blank_index: CTC blank token index
            space_index: Space character index
            beam_width: Beam width for beam search
        """
        self.vocabulary = vocabulary
        self.chars = vocabulary.get("chars", [])
        self.blank_index = blank_index
        self.space_index = space_index
        self.beam_width = beam_width

        # Build index-to-char mapping
        # Index 0 = CTC blank (not in vocab list)
        # Index 1+ = vocabulary characters
        self.idx_to_char = {0: ""}  # blank
        for i, char in enumerate(self.chars):
            self.idx_to_char[i + 1] = char

        self.char_to_idx = {v: k for k, v in self.idx_to_char.items()}
        self.vocab_size = len(self.idx_to_char)

        logger.info(
            f"CTCDecoder initialized: "
            f"vocab_size={self.vocab_size}, "
            f"blank={blank_index}, "
            f"chars_sample={self.chars[:10]}"
        )

    def greedy_decode(
        self,
        log_probs: torch.Tensor,
        return_confidence: bool = True
    ) -> DecodingResult:
        """
        Greedy CTC decoding.

        Args:
            log_probs: (T, vocab_size) log probability tensor

        Returns:
            DecodingResult with decoded text
        """
        # Convert to numpy for processing
        if isinstance(log_probs, torch.Tensor):
            probs = torch.exp(log_probs).cpu().numpy()
        else:
            probs = np.exp(log_probs) if log_probs.dtype != np.float32 else log_probs

        T, V = probs.shape

        # Get most likely token at each timestep
        best_indices = np.argmax(probs, axis=1)
        best_probs = probs[np.arange(T), best_indices]

        # CTC collapsing: remove consecutive duplicates and blanks
        decoded_indices = []
        decoded_probs = []
        prev_idx = -1

        for t, (idx, prob) in enumerate(zip(best_indices, best_probs)):
            if idx != self.blank_index and idx != prev_idx:
                decoded_indices.append(int(idx))
                decoded_probs.append(float(prob))
            prev_idx = idx

        # Convert indices to characters
        text = self._indices_to_text(decoded_indices)

        # Calculate confidence as geometric mean of character probabilities
        if decoded_probs:
            confidence = float(np.exp(np.mean(np.log(np.clip(decoded_probs, 1e-10, 1.0)))))
        else:
            confidence = 0.0

        return DecodingResult(
            text=text,
            confidence=confidence,
            char_probs=decoded_probs,
            raw_indices=decoded_indices
        )

    def beam_search_decode(
        self,
        log_probs: torch.Tensor
    ) -> DecodingResult:
        """
        Beam search CTC decoding for better accuracy.

        Args:
            log_probs: (T, vocab_size) log probability tensor

        Returns:
            DecodingResult with decoded text
        """
        if isinstance(log_probs, torch.Tensor):
            probs = torch.exp(log_probs).cpu().numpy()
        else:
            probs = log_probs

        T, V = probs.shape
        beam_width = min(self.beam_width, V)

        # Beam state: (prefix, last_char, score)
        # prefix: tuple of token indices
        beams = [((), self.blank_index, 0.0)]  # (prefix, last_token, log_score)

        for t in range(T):
            t_probs = probs[t]  # (V,)

            new_beams = {}

            for prefix, last_char, score in beams:
                # Get top-k tokens for this timestep
                top_k_indices = np.argsort(t_probs)[::-1][:beam_width]

                for idx in top_k_indices:
                    idx = int(idx)
                    p = float(t_probs[idx])
                    log_p = np.log(max(p, 1e-30))

                    if idx == self.blank_index:
                        # Blank: keep prefix, update score
                        new_prefix = prefix
                        new_last = self.blank_index
                    elif idx == last_char:
                        # Same as last: only extend if last was blank
                        if last_char == self.blank_index:
                            new_prefix = prefix + (idx,)
                            new_last = idx
                        else:
                            new_prefix = prefix
                            new_last = last_char
                    else:
                        # New character
                        new_prefix = prefix + (idx,)
                        new_last = idx

                    new_score = score + log_p
                    key = (new_prefix, new_last)

                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score

            # Convert back to list and prune
            beams = [
                (prefix, last, score)
                for (prefix, last), score in new_beams.items()
            ]
            beams.sort(key=lambda x: x[2], reverse=True)
            beams = beams[:beam_width]

        if not beams:
            return DecodingResult(text="", confidence=0.0, char_probs=[], raw_indices=[])

        # Best beam
        best_prefix, _, best_score = beams[0]

        # Remove blanks and collapse
        decoded_indices = [idx for idx in best_prefix if idx != self.blank_index]

        # Remove consecutive duplicates (CTC collapsing already done in beam search)
        collapsed = []
        for idx in decoded_indices:
            if not collapsed or collapsed[-1] != idx:
                collapsed.append(idx)

        text = self._indices_to_text(collapsed)

        # Normalize score to confidence
        confidence = min(1.0, max(0.0, np.exp(best_score / max(T, 1))))

        return DecodingResult(
            text=text,
            confidence=float(confidence),
            char_probs=[],
            raw_indices=collapsed
        )

    def _indices_to_text(self, indices: List[int]) -> str:
        """Convert token indices to text string."""
        chars = []
        for idx in indices:
            char = self.idx_to_char.get(idx, "")
            if char:  # Skip blank
                chars.append(char)
        return "".join(chars)

    def decode_batch(
        self,
        log_probs: torch.Tensor,
        method: str = "greedy"
    ) -> List[DecodingResult]:
        """
        Decode a batch of sequences.

        Args:
            log_probs: (T, B, vocab_size) log probabilities
            method: "greedy" or "beam_search"

        Returns:
            List of DecodingResult for each item in batch
        """
        T, B, V = log_probs.shape
        results = []

        for b in range(B):
            seq_log_probs = log_probs[:, b, :]  # (T, V)

            if method == "beam_search":
                result = self.beam_search_decode(seq_log_probs)
            else:
                result = self.greedy_decode(seq_log_probs)

            results.append(result)

        return results


class TextPostProcessor:
    """
    Post-process decoded text for better readability.
    Handles common lip reading artifacts and formatting.
    """

    def __init__(self):
        # Common corrections for lip reading errors
        self.corrections = {
            "  ": " ",    # Double spaces
        }

        # Minimum confidence threshold
        self.min_confidence = 0.05

    def process(self, result: DecodingResult) -> DecodingResult:
        """
        Apply post-processing to decoding result.

        Args:
            result: Raw decoding result

        Returns:
            Post-processed result
        """
        text = result.text

        # Strip whitespace
        text = text.strip()

        # Apply corrections
        for wrong, correct in self.corrections.items():
            text = text.replace(wrong, correct)

        # Filter very low confidence results
        if result.confidence < self.min_confidence and len(text) > 0:
            # Keep text but mark low confidence
            pass

        return DecodingResult(
            text=text,
            confidence=result.confidence,
            char_probs=result.char_probs,
            raw_indices=result.raw_indices
        )

    def format_display(
        self,
        result: DecodingResult,
        max_chars: int = 50
    ) -> Tuple[str, str]:
        """
        Format result for display.

        Args:
            result: Decoding result
            max_chars: Maximum characters to display

        Returns:
            Tuple of (display_text, confidence_str)
        """
        text = result.text

        # Truncate if too long
        if len(text) > max_chars:
            text = "..." + text[-(max_chars-3):]

        confidence_str = f"{result.confidence:.1%}"

        return text, confidence_str


class SlidingWindowAggregator:
    """
    Aggregates predictions from sliding window inference.
    Uses voting and smoothing to produce stable text output.
    """

    def __init__(self, window_size: int = 5, min_agreement: float = 0.4):
        self.window_size = window_size
        self.min_agreement = min_agreement
        self._recent_results: List[DecodingResult] = []
        self._stable_text = ""

    def add_result(self, result: DecodingResult) -> str:
        """
        Add a new decoding result and return stable prediction.

        Args:
            result: New decoding result

        Returns:
            Current stable text prediction
        """
        self._recent_results.append(result)

        # Keep only recent results
        if len(self._recent_results) > self.window_size:
            self._recent_results.pop(0)

        # Find most common prediction
        if not self._recent_results:
            return self._stable_text

        # Weight by confidence
        text_scores: Dict[str, float] = {}
        for r in self._recent_results:
            text = r.text.strip()
            if text:
                text_scores[text] = text_scores.get(text, 0.0) + r.confidence

        if text_scores:
            best_text = max(text_scores, key=text_scores.get)
            max_possible = len(self._recent_results)

            if text_scores[best_text] / max_possible >= self.min_agreement:
                self._stable_text = best_text

        return self._stable_text

    def get_current_text(self) -> str:
        """Get current stable text."""
        return self._stable_text

    def get_latest_confidence(self) -> float:
        """Get confidence of latest result."""
        if self._recent_results:
            return self._recent_results[-1].confidence
        return 0.0

    def reset(self):
        """Reset aggregator state."""
        self._recent_results.clear()
        self._stable_text = ""