#!/usr/bin/env python3
"""
Download and setup pretrained weights for the Silent Speech Recognition system.
Run once before starting the application:  python setup_weights.py
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

# ── Vocabulary ───────────────────────────────────────────────────────────────
GRID_VOCAB = {
    "chars": [
        " ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
        "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "'", "-", ".", ",", "?", "!",
    ],
    "blank_index": 0,
    "space_index": 1,
}

# ── URLs ─────────────────────────────────────────────────────────────────────
DNN_FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    ),
}
MP_LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MP_LANDMARKER_PATH = WEIGHTS_DIR / "face_landmarker.task"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _dl(url: str, dest: Path, label: str) -> bool:
    if dest.exists():
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  [SKIP]     {label}  ({size_mb:.1f} MB already present)")
        return True
    print(f"  [DOWNLOAD] {label} … ", end="", flush=True)
    try:
        urllib.request.urlretrieve(str(url), str(dest))
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"done  ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


# ── Steps ────────────────────────────────────────────────────────────────────
def save_vocabulary():
    path = WEIGHTS_DIR / "vocabulary.json"
    with open(path, "w") as f:
        json.dump(GRID_VOCAB, f, indent=2)
    print(f"  [OK] vocabulary.json → {path}")


def download_dnn_weights():
    print("[STEP 2] OpenCV DNN face-detector weights")
    for fname, url in DNN_FILES.items():
        _dl(url, WEIGHTS_DIR / fname, fname)


def download_mediapipe_landmarker():
    print("[STEP 3] MediaPipe FaceLandmarker model")
    _dl(MP_LANDMARKER_URL, MP_LANDMARKER_PATH, "face_landmarker.task")


def create_lipnet_style_weights():
    """
    Build a checkpoint whose LSTM input_size matches the model's
    TemporalEncoder, which projects CNN features (576-d) down to
    hidden_dim (256-d) BEFORE feeding the LSTM.

    Architecture inside TemporalEncoder:
        input_proj : Linear(feature_dim=576, hidden_dim=256)
        lstm       : LSTM(input_size=256, hidden_size=256, ...)

    So the LSTM weight_ih shape must be (1024, 256), not (1024, 576).
    """
    print("[STEP 4] Building model checkpoint with pretrained backbone")
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
    except ImportError as exc:
        print(f"  [ERROR] PyTorch not available: {exc}")
        sys.exit(1)

    # ── CNN backbone (ImageNet pretrained) ───────────────────────────────
    backbone = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    feature_extractor = nn.Sequential(*list(backbone.features.children()))

    dummy = torch.zeros(1, 3, 112, 112)
    with torch.no_grad():
        feature_dim = int(
            nn.AdaptiveAvgPool2d((1, 1))(feature_extractor(dummy)).shape[1]
        )
    print(f"  CNN feature_dim : {feature_dim}")

    # ── Model hyper-parameters (must mirror src/model.py defaults) ───────
    HIDDEN_DIM  = 256
    NUM_LAYERS  = 2
    DROPOUT     = 0.3
    VOCAB_SIZE  = len(GRID_VOCAB["chars"]) + 1   # +1 for CTC blank

    # ── input_proj  (Linear 576 → 256) ───────────────────────────────────
    input_proj = nn.Sequential(
        nn.Linear(feature_dim, HIDDEN_DIM),
        nn.LayerNorm(HIDDEN_DIM),
        nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT),
    )
    for m in input_proj.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    # ── LSTM  (input_size = HIDDEN_DIM = 256, NOT feature_dim) ───────────
    lstm = nn.LSTM(
        input_size=HIDDEN_DIM,       # ← THIS is what was wrong before
        hidden_size=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        batch_first=True,
        dropout=DROPOUT,
        bidirectional=True,
    )
    for name, p in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(p.data)
        elif "bias" in name:
            p.data.fill_(0)
            n = p.size(0)
            p.data[n // 4 : n // 2].fill_(1)   # forget-gate bias = 1

    print(f"  LSTM weight_ih_l0 shape : {lstm.weight_ih_l0.shape}")   # expect [1024, 256]

    # ── Output layer norm + dropout ───────────────────────────────────────
    output_norm    = nn.LayerNorm(HIDDEN_DIM * 2)
    output_dropout = nn.Dropout(DROPOUT)

    # ── Classifier  (512 → vocab) ─────────────────────────────────────────
    # mirrors src/model.py  LipReadingModel.classifier
    classifier = nn.Sequential(
        nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
    )
    for m in classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    # ── Checkpoint ────────────────────────────────────────────────────────
    checkpoint = {
        "model_config": {
            "backbone":        "mobilenet_v3_small",
            "feature_dim":     feature_dim,
            "hidden_dim":      HIDDEN_DIM,
            "num_layers":      NUM_LAYERS,
            "vocab_size":      VOCAB_SIZE,
            "dropout":         DROPOUT,
            "input_size":      (112, 112),
            "sequence_length": 75,
        },
        # individual sub-module state dicts
        "backbone_state_dict":    feature_extractor.state_dict(),
        "input_proj_state_dict":  input_proj.state_dict(),
        "lstm_state_dict":        lstm.state_dict(),
        "output_norm_state_dict": output_norm.state_dict(),
        "classifier_state_dict":  classifier.state_dict(),
        "vocabulary":             GRID_VOCAB,
        "version":                "1.1.0",
        "description":            "LipNet-style model, ImageNet backbone, fixed LSTM dims",
    }

    out_path = WEIGHTS_DIR / "lipnet_model.pth"
    torch.save(checkpoint, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  [OK] lipnet_model.pth → {out_path}  ({size_mb:.1f} MB)")


def verify_setup():
    required = [
        WEIGHTS_DIR / "vocabulary.json",
        WEIGHTS_DIR / "deploy.prototxt",
        WEIGHTS_DIR / "res10_300x300_ssd.caffemodel",
        WEIGHTS_DIR / "lipnet_model.pth",
    ]
    optional = [WEIGHTS_DIR / "face_landmarker.task"]

    all_ok = True
    print("\n[VERIFY]")
    for f in required:
        tag = "[OK]     " if f.exists() else "[MISSING]"
        print(f"  {tag} {f.name}")
        if not f.exists():
            all_ok = False
    for f in optional:
        tag = "[OK]     " if f.exists() else "[OPTIONAL - missing]"
        print(f"  {tag} {f.name}")
    return all_ok


def main():
    print("=" * 60)
    print("  Silent Speech Recognition – Weight Setup  v1.1")
    print("=" * 60)

    print("[STEP 1] Vocabulary")
    save_vocabulary()

    download_dnn_weights()
    download_mediapipe_landmarker()
    create_lipnet_style_weights()

    if verify_setup():
        print("\n[SUCCESS] All weights ready.")
        print("  Run:  python main.py --source webcam")
    else:
        print("\n[ERROR] Setup incomplete – check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()