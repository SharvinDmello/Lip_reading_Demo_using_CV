# Silent Speech Recognition
## Vision-Based Lip Reading System

A real-time lip reading system that predicts spoken text from video input
**without using any audio**. Uses MediaPipe for face/lip detection and a
MobileNetV3 + BiLSTM architecture with pretrained ImageNet weights.

---

## Architecture

```
Webcam/Video
     │
     ▼
[Face Detection]         ← MediaPipe Face Detection
     │
     ▼
[Lip Extraction]         ← MediaPipe Face Mesh (468 landmarks)
     │
     ▼
[Preprocessing]          ← Resize 112×112, ImageNet normalization
     │
     ▼
[Sequence Buffer]        ← Sliding window (75 frames)
     │
     ▼
[MobileNetV3 CNN]        ← Spatial features (pretrained ImageNet)
     │
     ▼
[BiLSTM Encoder]         ← Temporal modeling
     │
     ▼
[Self-Attention]         ← Enhanced temporal context
     │
     ▼
[CTC Decoder]            ← Greedy or Beam Search
     │
     ▼
[Text Output]            ← Displayed on video + saved to file
```

---

## Quick Start

### 1. Environment Setup

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal

# Clone / navigate to project
cd silent_speech

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### 2. Download/Setup Weights

```bash
python setup_weights.py
```

This will:
- Download MobileNetV3 ImageNet pretrained weights (via torchvision)
- Initialize temporal model weights
- Save vocabulary file
- Output: `weights/lipnet_model.pth` and `weights/vocabulary.json`

### 3. Run

```bash
# Webcam (default)
python main.py

# Webcam with all options
python main.py --source webcam --camera-id 0 --debug

# Video file
python main.py --source video --input /path/to/video.mp4

# Save predictions to file
python main.py --save-output --output-text output/predictions.txt

# Use beam search (more accurate, slower)
python main.py --beam-search

# Force CPU (if VRAM issues)
python main.py --device cpu

# Headless + save video
python main.py --no-display --save-video --output-video output/result.mp4
```

---



## Note on Lip Reading Accuracy

This system uses a model architecture with **pretrained ImageNet backbone**
for feature extraction and randomly initialized temporal layers. For production
accuracy, the temporal layers would need fine-tuning on a lip reading dataset
such as:
- GRID Corpus (public, requires registration)
- LRS2/LRS3 (requires license)
- MV-LRS (public)

The current system demonstrates:
✅ Correct architecture for lip reading (LipNet-style)
✅ Real-time processing pipeline
✅ Stable face and lip tracking
✅ Proper CTC decoding
✅ Smooth text output with temporal aggregation

---
