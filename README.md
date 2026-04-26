# LipNet Visual Speech Recognition

## Overview

This project is a **visual speech recognition** demo built with **Streamlit**. It uses a trained lip-reading model to predict spoken words from mouth movement video clips.

The app loads sample videos from `data/s1/`, processes the frames with a convolutional neural network plus bidirectional LSTM model, and displays both the input video and a generated animation of the mouth region. The model outputs a predicted text string based on the lip movements.

## What this app does

- Shows a list of available sample videos in `data/s1/`
- Plays the selected video in the browser
- Extracts and preprocesses the lip region from video frames
- Loads a trained Keras model from `models/checkpoint`
- Predicts the spoken phrase from the selected clip
- Applies a basic text correction step using `textblob` when available
- Maps predictions to a small vocabulary of valid words
- Displays the final cleaned prediction to the user

## Project structure

- `app/streamlitapp.py` - Streamlit application entrypoint
- `app/modelutil.py` - Defines and loads the trained Keras model architecture and weights
- `app/utils.py` - Video loading, preprocessing, and alignment helper functions
- `data/s1/` - Video clips used by the demo
- `data/alignments/s1/` - Alignment files used during prediction preprocessing
- `models/` - Pre-trained model checkpoint files

## How it works

1. The app reads the selected `.mpg` video file.
2. `app/utils.py` converts each frame to grayscale and extracts the mouth region.
3. The video sequence is normalized and provided to the model.
4. `app/modelutil.py` defines a 3D Conv + BiLSTM network and loads saved weights.
5. Predictions are decoded using CTC decoding and converted back to characters.
6. A small post-processing step matches words to a set of valid vocabulary terms.

## Requirements

- Python 3.8 or newer
- `streamlit`
- `tensorflow`
- `opencv-python`
- `imageio`
- `textblob` (optional, for prediction correction)
- `ffmpeg` installed and available on the system path

## Running the app

1. Install dependencies:

```bash
pip install streamlit tensorflow opencv-python imageio textblob
```

2. Make sure `ffmpeg` is installed and accessible from the command line.

3. Run the Streamlit app from the project root:

```bash
streamlit run app/streamlitapp.py
```

4. Open the provided local URL in your browser.

## Notes

- The app currently expects sample videos in `data/s1/` and aligned annotations in `data/alignments/s1/`.
- Model weights are loaded from `models/checkpoint`.
- Prediction quality depends on the provided trained model checkpoint and the sample data.

## Disclaimer

This repository is a demonstration of lip-reading using a neural network. It is intended for educational purposes and may not generalize to arbitrary video input without retraining or further preprocessing.
