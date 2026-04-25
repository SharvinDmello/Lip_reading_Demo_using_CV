"""
Face Detection Module
Uses OpenCV DNN face detector (ResNet-10 SSD Caffe model).
Always runs on CPU backend to avoid CUDA/DNN assertion errors.
"""

import cv2
import numpy as np
import urllib.request
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
CAFFEMODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)

PROTOTXT_PATH   = os.path.join("weights", "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join("weights", "res10_300x300_ssd.caffemodel")


def _download_file(url: str, dest: str, label: str = "") -> bool:
    """Download *url* → *dest* if not already present."""
    if os.path.exists(dest):
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info(f"Downloading {label} …")
    try:
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Saved {label} → {dest}")
        return True
    except Exception as exc:
        logger.error(f"Download failed ({label}): {exc}")
        return False


def ensure_dnn_weights() -> bool:
    ok_p = _download_file(PROTOTXT_URL,    PROTOTXT_PATH,   "deploy.prototxt")
    ok_m = _download_file(CAFFEMODEL_URL,  CAFFEMODEL_PATH, "SSD caffemodel")
    return ok_p and ok_m


@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    confidence: float
    landmarks: Optional[np.ndarray] = None


class FaceDetector:
    """
    OpenCV DNN ResNet-10 SSD face detector.
    Runs exclusively on CPU to avoid backend/target assertion errors
    in OpenCV builds that lack CUDA-DNN support.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,       # unused, kept for API compat
        max_num_faces: int = 1,
        smoothing_factor: float = 0.7,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.max_num_faces            = max_num_faces
        self._smoothing_factor        = smoothing_factor
        self._prev_bbox: Optional[Tuple[int, int, int, int]] = None

        if not ensure_dnn_weights():
            raise RuntimeError(
                "Could not obtain OpenCV DNN face-detector weights. "
                "Check your internet connection and re-run setup_weights.py."
            )

        self._net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

        # ── Always use plain CPU backend ────────────────────────────────────
        # DNN_BACKEND_CUDA requires an OpenCV build with -DWITH_CUDA=ON
        # *and* opencv-contrib-python; plain opencv-python-headless lacks it.
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("FaceDetector ready (OpenCV DNN, CPU backend)")

    # ── internal ────────────────────────────────────────────────────────────

    def _detect_raw(
        self, frame: np.ndarray
    ) -> List[Tuple[float, Tuple[int, int, int, int]]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        dets = self._net.forward()          # (1, 1, N, 7)

        results: List[Tuple[float, Tuple[int, int, int, int]]] = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < self.min_detection_confidence:
                continue
            x1 = max(0, int(dets[0, 0, i, 3] * w))
            y1 = max(0, int(dets[0, 0, i, 4] * h))
            x2 = min(w, int(dets[0, 0, i, 5] * w))
            y2 = min(h, int(dets[0, 0, i, 6] * h))
            bw, bh = x2 - x1, y2 - y1
            if bw > 0 and bh > 0:
                results.append((conf, (x1, y1, bw, bh)))

        results.sort(key=lambda r: r[0], reverse=True)
        return results

    def _smooth_bbox(
        self, bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        if self._prev_bbox is None:
            self._prev_bbox = bbox
            return bbox
        a = self._smoothing_factor
        s = tuple(int(a * c + (1 - a) * p) for c, p in zip(bbox, self._prev_bbox))
        self._prev_bbox = s
        return s

    # ── public ──────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Optional[FaceDetection]:
        if frame is None or frame.size == 0:
            return None
        raw = self._detect_raw(frame)
        if not raw:
            return None
        conf, bbox = raw[0]
        return FaceDetection(bbox=self._smooth_bbox(bbox), confidence=conf)

    def detect_all(self, frame: np.ndarray) -> List[FaceDetection]:
        if frame is None or frame.size == 0:
            return []
        return [
            FaceDetection(bbox=b, confidence=c)
            for c, b in self._detect_raw(frame)[: self.max_num_faces]
        ]

    def draw_detection(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        if detection is None:
            return frame
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            frame, f"Face {detection.confidence:.0%}",
            (x, max(y - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )
        return frame

    def get_face_crop(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        padding: float = 0.2,
    ) -> Optional[np.ndarray]:
        if detection is None:
            return None
        fh, fw = frame.shape[:2]
        x, y, w, h = detection.bbox
        px, py = int(w * padding), int(h * padding)
        x1 = max(0, x - px);  y1 = max(0, y - py)
        x2 = min(fw, x + w + px); y2 = min(fh, y + h + py)
        return frame[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None

    def reset(self):
        self._prev_bbox = None