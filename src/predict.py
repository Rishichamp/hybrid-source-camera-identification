"""
predict.py
----------
End-to-end inference: load a single image, extract its PRNU residual,
correlate against the saved camera fingerprints, run the CNN branch on its
patches, fuse both via the saved meta-classifier, and print the prediction.

Usage:
    cd src
    python predict.py --image /path/to/photo.jpg
"""

import argparse
import json
import os

import cv2
import joblib
import numpy as np
from tensorflow.keras.models import load_model

from prnu_extraction import (
    extract_patches,
    extract_prnu_residual,
    patches_to_cnn_batch,
    prnu_correlation_features,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")


def predict_source(image_path: str, model_dir: str = MODEL_DIR,
                    crop_size: int = 512, patch_size: int = 96, stride: int = 64):
    cnn = load_model(os.path.join(model_dir, "cnn_branch.keras"))
    meta_clf = joblib.load(os.path.join(model_dir, "meta_classifier.joblib"))
    fingerprints = list(np.load(os.path.join(model_dir, "fingerprints.npy")))
    with open(os.path.join(model_dir, "class_names.json")) as f:
        class_names = json.load(f)  # whatever cameras this model was trained on

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    residual = extract_prnu_residual(img, crop_size)
    corr = prnu_correlation_features(residual, fingerprints)

    # Same normalization used to build the CNN's training data (dataset.py)
    # and the meta-classifier's features (hybrid_train.image_level_features)
    # -- NOT a plain /255.0, since residuals aren't 0-255 pixel intensities.
    patches = extract_patches(residual, patch_size, stride)
    patch_arr = patches_to_cnn_batch(patches)
    cnn_probs = cnn.predict(patch_arr, verbose=0).mean(axis=0)

    feats = np.concatenate([corr, cnn_probs]).reshape(1, -1)
    probs = meta_clf.predict_proba(feats)[0]
    idx = int(np.argmax(probs))

    return class_names[idx], float(probs[idx]), probs, corr, cnn_probs, class_names


def parse_args():
    p = argparse.ArgumentParser(description="Predict source camera (hybrid PRNU+CNN).")
    p.add_argument("--image", required=True)
    p.add_argument("--model-dir", default=MODEL_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label, confidence, probs, corr, cnn_probs, class_names = predict_source(
        args.image, args.model_dir
    )

    print(f"Predicted source camera: {label}  (confidence: {confidence * 100:.2f}%)\n")
    print("Fused (meta-classifier) probabilities:")
    for cls, p in zip(class_names, probs):
        print(f"  {cls:25s}: {p * 100:5.2f}%")

    print("\n[Diagnostic] PRNU correlation scores (higher = more similar):")
    for cls, c in zip(class_names, corr):
        print(f"  {cls:25s}: {c:+.4f}")

    print("\n[Diagnostic] CNN branch avg. patch probabilities:")
    for cls, p in zip(class_names, cnn_probs):
        print(f"  {cls:25s}: {p * 100:5.2f}%")
