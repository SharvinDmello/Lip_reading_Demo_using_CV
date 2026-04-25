"""
Lip Region Extraction Module

Uses MediaPipe Tasks FaceLandmarker (new API, mediapipe >= 0.10)
with an automatic fallback to a geometry-based approach using the
OpenCV DNN face detector when the Task model file is unavailable.

Landmark download is handled automatically via the MediaPipe model
asset downloader.
"""

import os
import cv2
import numpy as np
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe face landmarker model (Tasks API)
# ---------------------------------------------------------------------------
_MP_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MP_LANDMARKER_PATH = os.path.join("weights", "face_landmarker.task")


def _download_landmarker_model() -> bool:
    """Download the MediaPipe FaceLandmarker .task file if missing."""
    if os.path.exists(_MP_LANDMARKER_PATH):
        return True
    os.makedirs(os.path.dirname(_MP_LANDMARKER_PATH), exist_ok=True)
    logger.info(f"Downloading FaceLandmarker model ({_MP_LANDMARKER_URL}) ...")
    try:
        urllib.request.urlretrieve(_MP_LANDMARKER_URL, _MP_LANDMARKER_PATH)
        logger.info(f"FaceLandmarker model saved → {_MP_LANDMARKER_PATH}")
        return True
    except Exception as e:
        logger.warning(f"Could not download FaceLandmarker model: {e}")
        return False


# ---------------------------------------------------------------------------
# Lip landmark indices (MediaPipe 478-point mesh)
# ---------------------------------------------------------------------------
OUTER_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
INNER_LIP_INDICES = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]
ALL_LIP_INDICES = sorted(set(OUTER_LIP_INDICES + INNER_LIP_INDICES))

LEFT_EYE_CENTER  = [33, 133]
RIGHT_EYE_CENTER = [362, 263]


# ---------------------------------------------------------------------------
@dataclass
class LipRegion:
    """Container for extracted lip region data."""
    crop: np.ndarray                        # (H, W, 3) BGR
    landmarks: np.ndarray                   # (N, 2)
    bbox: Tuple[int, int, int, int]         # x, y, w, h
    confidence: float
    face_detected: bool


# ---------------------------------------------------------------------------
class _TasksBackend:
    """
    Backend using the new mediapipe.tasks.python.vision.FaceLandmarker API
    (mediapipe >= 0.10).
    """

    def __init__(self, model_path: str):
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as mp_base
        from mediapipe import tasks as mp_tasks

        base_opts = mp_base.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._ts_ms = 0
        logger.info("MediaPipe Tasks FaceLandmarker backend initialised")

    def process(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns (478, 2) array of (x_px, y_px) landmarks, or None.
        """
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._ts_ms += 33          # ~30 fps synthetic timestamp
        result = self._landmarker.detect_for_video(mp_image, self._ts_ms)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        return pts                 # (478, 2)

    def close(self):
        self._landmarker.close()


class _LegacyBackend:
    """
    Backend using the old mediapipe.solutions.face_mesh API
    (mediapipe < 0.10).  Kept as fallback.
    """

    def __init__(self):
        import mediapipe as mp
        self._fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        logger.info("MediaPipe legacy FaceMesh backend initialised")

    def process(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
        return pts

    def close(self):
        self._fm.close()


class _OpenCVFallbackBackend:
    """
    Pure OpenCV fallback: uses the DNN face detector to estimate a lip
    region heuristically (lower third of the face bounding box).
    No landmarks are returned.
    """

    def __init__(self):
        logger.warning(
            "Using OpenCV geometry fallback for lip extraction. "
            "Lip landmarks will be estimated, not detected."
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns None – caller uses bbox estimation instead."""
        return None

    def close(self):
        pass


# ---------------------------------------------------------------------------
def _build_backend():
    """
    Try backends in order of preference:
      1. MediaPipe Tasks (new API, >= 0.10)
      2. MediaPipe legacy solutions (< 0.10)
      3. OpenCV geometry fallback
    """
    # --- Try Tasks API ---
    try:
        from mediapipe.tasks.python import vision as _v   # noqa: F401
        ok = _download_landmarker_model()
        if ok and os.path.exists(_MP_LANDMARKER_PATH):
            try:
                backend = _TasksBackend(_MP_LANDMARKER_PATH)
                return backend
            except Exception as e:
                logger.warning(f"Tasks backend init failed: {e}")
    except ImportError:
        pass

    # --- Try legacy solutions API ---
    try:
        import mediapipe as mp
        _ = mp.solutions.face_mesh          # will raise on new API
        backend = _LegacyBackend()
        return backend
    except AttributeError:
        pass
    except Exception as e:
        logger.warning(f"Legacy backend failed: {e}")

    # --- Fallback ---
    return _OpenCVFallbackBackend()


# ---------------------------------------------------------------------------
class LipExtractor:
    """
    Extracts lip regions from video frames.

    Automatically selects the best available MediaPipe backend and falls
    back to a geometry-based approach when no landmark model is available.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (112, 112),
        padding_factor: float = 0.4,
        use_face_alignment: bool = True,
        smooth_landmarks: bool = True,
        # kept for API compat
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.target_size      = target_size
        self.padding_factor   = padding_factor
        self.use_face_alignment = use_face_alignment
        self.smooth_landmarks = smooth_landmarks

        self._backend = _build_backend()
        self._is_opencv_fallback = isinstance(self._backend, _OpenCVFallbackBackend)

        # Landmark smoothing state
        self._prev_lip_lm: Optional[np.ndarray] = None
        self._smooth_alpha = 0.6

        # Diagnostics
        self._frame_count     = 0
        self._detection_count = 0

        logger.info(
            f"LipExtractor ready – backend: {type(self._backend).__name__}, "
            f"target_size={target_size}"
        )

    # ------------------------------------------------------------------
    def extract(self, frame: np.ndarray) -> Optional[LipRegion]:
        """
        Extract lip region from *frame* (BGR, H×W×3).

        Returns LipRegion or None on empty / invalid input.
        """
        if frame is None or frame.size == 0:
            return None

        self._frame_count += 1
        h, w = frame.shape[:2]

        # ---- Get full-face landmarks ---------------------------------
        all_lm = self._backend.process(frame)     # (478,2) or None

        if all_lm is None:
            # Return blank region with face_detected=False
            blank = np.zeros(
                (self.target_size[1], self.target_size[0], 3), dtype=np.uint8
            )
            return LipRegion(
                crop=blank,
                landmarks=np.zeros((len(ALL_LIP_INDICES), 2), dtype=np.float32),
                bbox=(0, 0, 0, 0),
                confidence=0.0,
                face_detected=False,
            )

        self._detection_count += 1

        # ---- Lip landmarks -------------------------------------------
        # Guard: some models return 468 pts, others 478
        max_idx = max(ALL_LIP_INDICES)
        if max_idx >= len(all_lm):
            # Remap to available range
            scale = len(all_lm) / 478.0
            safe_idx = [min(int(i * scale), len(all_lm) - 1)
                        for i in ALL_LIP_INDICES]
            lip_lm = all_lm[safe_idx]
        else:
            lip_lm = all_lm[ALL_LIP_INDICES]     # (N, 2)

        # Temporal smoothing
        if self.smooth_landmarks and self._prev_lip_lm is not None:
            if self._prev_lip_lm.shape == lip_lm.shape:
                lip_lm = (self._smooth_alpha * lip_lm +
                          (1 - self._smooth_alpha) * self._prev_lip_lm)
        self._prev_lip_lm = lip_lm.copy()

        # ---- Optional face alignment ---------------------------------
        if self.use_face_alignment and len(all_lm) > max(RIGHT_EYE_CENTER):
            aligned_frame, rot_mat = self._align_face(frame, all_lm, w, h)
            ones = np.ones((len(lip_lm), 1), dtype=np.float32)
            lip_lm_t = (rot_mat @ np.hstack([lip_lm, ones]).T).T[:, :2]
        else:
            aligned_frame = frame
            lip_lm_t = lip_lm

        # ---- Bounding box -------------------------------------------
        bbox = self._get_lip_bbox(lip_lm_t, (h, w), self.padding_factor)
        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            return None

        lip_crop = aligned_frame[y : y + bh, x : x + bw]
        if lip_crop.size == 0:
            return None

        lip_crop = cv2.resize(lip_crop, self.target_size, interpolation=cv2.INTER_LINEAR)

        return LipRegion(
            crop=lip_crop,
            landmarks=lip_lm_t,
            bbox=bbox,
            confidence=1.0,
            face_detected=True,
        )

    # ------------------------------------------------------------------
    def _align_face(
        self,
        frame: np.ndarray,
        all_lm: np.ndarray,
        w: int,
        h: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_eye  = all_lm[LEFT_EYE_CENTER].mean(axis=0)
        right_eye = all_lm[RIGHT_EYE_CENTER].mean(axis=0)
        angle = np.degrees(np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0],
        ))
        eye_center = (
            float((left_eye[0] + right_eye[0]) / 2),
            float((left_eye[1] + right_eye[1]) / 2),
        )
        rot = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned = cv2.warpAffine(frame, rot, (w, h), flags=cv2.INTER_LINEAR)
        return aligned, rot

    def _get_lip_bbox(
        self,
        lip_lm: np.ndarray,
        frame_shape: Tuple[int, int],
        padding: float,
    ) -> Tuple[int, int, int, int]:
        h, w = frame_shape
        xmin, xmax = np.min(lip_lm[:, 0]), np.max(lip_lm[:, 0])
        ymin, ymax = np.min(lip_lm[:, 1]), np.max(lip_lm[:, 1])
        lw, lh = xmax - xmin, ymax - ymin
        max_dim = max(lw, lh)
        pad = max_dim * padding
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        half = (max_dim + 2 * pad) / 2
        x  = int(max(0, cx - half))
        y  = int(max(0, cy - half))
        bw = int(min(w - x, 2 * half))
        bh = int(min(h - y, 2 * half))
        return (x, y, bw, bh)

    # ------------------------------------------------------------------
    def draw_lip_landmarks(
        self,
        frame: np.ndarray,
        lip_region: LipRegion,
        color: Tuple[int, int, int] = (0, 0, 255),
        draw_bbox: bool = True,
    ) -> np.ndarray:
        if lip_region is None or not lip_region.face_detected:
            return frame
        for pt in lip_region.landmarks:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, color, -1)
        if draw_bbox:
            x, y, bw, bh = lip_region.bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
        return frame

    def get_detection_rate(self) -> float:
        if self._frame_count == 0:
            return 0.0
        return self._detection_count / self._frame_count

    def reset(self):
        self._prev_lip_lm    = None
        self._frame_count    = 0
        self._detection_count = 0

    def __del__(self):
        if hasattr(self, "_backend"):
            try:
                self._backend.close()
            except Exception:
                pass