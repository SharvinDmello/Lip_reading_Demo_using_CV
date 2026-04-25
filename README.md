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

## Controls (during runtime)

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `R` | Reset pipeline |
| `S` | Save current prediction |
| `D` | Toggle debug panel |
| `L` | Toggle lip landmarks |
| `P` | Print performance stats |

---

## Display Layout

```
┌─────────────────────────────────────┬──────────────┐
│  [Lip Preview]                      │  FPS: 28.3   │
│                                     │  Face: YES   │
│  [Controls]                         │  Lips: YES   │
│   Q: Quit                           │  Buffer: 80% │
│   R: Reset                          │  Conf: 45%   │
│   S: Save                           │  Infer: 35ms │
│   D: Debug                          │  Time: 00:15 │
│   L: Landmarks                      │              │
│                     [Video Feed]    │              │
│    ┌──────────────────────┐         │              │
│    │  Face Bbox (green)   │         │              │
│    │  Lip Bbox  (blue)    │         │              │
│    └──────────────────────┘         │              │
│ ▓▓▓▓▓▓▒▒▒░░░  Conf: 45%            │              │
├─────────────────────────────────────┴──────────────┤
│  PREDICTED SPEECH:                                  │
│  place blue at f two now                            │
└────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1650 (4GB) | RTX 3060+ |
| CPU | Intel i5 8th gen | Intel i7+ |
| RAM | 8GB | 16GB |
| Python | 3.10 | 3.11 |

Tested on: Arch Linux, GTX 1650 4GB VRAM

---

## Project Structure

```
silent_speech/
├── README.md
├── pyproject.toml
├── setup_weights.py          # Download/setup pretrained weights
├── main.py                   # Main application entry point
├── src/
│   ├── face_detector.py      # MediaPipe face detection
│   ├── lip_extractor.py      # MediaPipe Face Mesh lip extraction
│   ├── preprocessor.py       # Frame normalization + sequence buffering
│   ├── model.py              # MobileNetV3 + BiLSTM model
│   ├── decoder.py            # CTC greedy + beam search decoder
│   ├── inference_engine.py   # Complete inference pipeline
│   ├── video_capture.py      # Webcam/video source management
│   └── display.py            # OpenCV overlay rendering
├── weights/                  # Model weights (created by setup_weights.py)
├── output/                   # Predictions and video output
└── tests/                    # Unit tests
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

## Troubleshooting

### Camera not found
```bash
# List available cameras
ls /dev/video*
# Try different device ID
python main.py --camera-id 1
```

### CUDA out of memory
```bash
# Force CPU
python main.py --device cpu
# Or reduce sequence length
python main.py --sequence-length 30
```

### Slow performance
```bash
# Reduce inference frequency
python main.py --inference-interval 15
# Use greedy decoding (not beam search)
python main.py  # (greedy is default)
```

### mediapipe installation issues (Arch Linux)
```bash
# Ensure system libs are present
sudo pacman -S mesa libgl
uv pip install mediapipe --no-cache-dir
```