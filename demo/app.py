"""
app.py
------
Free Gradio demo for the hybrid PRNU-correlation + CNN pipeline.

Run locally:
    python demo/app.py

Deploy free on Hugging Face Spaces: same steps as v1 (see main README),
just upload saved_models/cnn_branch.keras, meta_classifier.joblib and
fingerprints.npy alongside this app.py and src/.
"""

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import gradio as gr
import joblib
import numpy as np
from tensorflow.keras.models import load_model

from prnu_extraction import (
    extract_patches,
    extract_prnu_residual,
    patches_to_cnn_batch,
    prnu_correlation_features,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
CROP_SIZE, PATCH_SIZE, STRIDE = 512, 96, 64  # must match hybrid_train.py's --crop-size

_cnn = None
_meta = None
_fingerprints = None
_class_names = None


def _lazy_load():
    global _cnn, _meta, _fingerprints, _class_names
    if _cnn is None:
        _cnn = load_model(os.path.join(MODEL_DIR, "cnn_branch.keras"))
        _meta = joblib.load(os.path.join(MODEL_DIR, "meta_classifier.joblib"))
        _fingerprints = list(np.load(os.path.join(MODEL_DIR, "fingerprints.npy")))
        with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
            _class_names = json.load(f)  # whatever cameras this model was trained on
    return _cnn, _meta, _fingerprints, _class_names


def classify(image: np.ndarray):
    if image is None:
        return {}
    cnn, meta, fingerprints, class_names = _lazy_load()

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    residual = extract_prnu_residual(bgr, CROP_SIZE)
    corr = prnu_correlation_features(residual, fingerprints)

    patches = extract_patches(residual, PATCH_SIZE, STRIDE)
    patch_arr = patches_to_cnn_batch(patches)  # same normalization as training
    cnn_probs = cnn.predict(patch_arr, verbose=0).mean(axis=0)

    feats = np.concatenate([corr, cnn_probs]).reshape(1, -1)
    probs = meta.predict_proba(feats)[0]
    return {cls: float(p) for cls, p in zip(class_names, probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(label="Upload a photo"),
    outputs=gr.Label(num_top_classes=10, label="Predicted source camera"),
    title="Source Camera Identification -- Hybrid PRNU + CNN",
    description=(
        "Fuses classical PRNU-fingerprint correlation with a small custom "
        "CNN (trained from scratch) patch classifier. Trained on a fixed "
        "set of specific cameras -- not a general camera-model detector."
    ),
)

if __name__ == "__main__":
    demo.launch()
